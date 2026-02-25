"""
Hybrid ML Predictor for Vulnerability Detection

This module provides inference using the trained HybridVulnerabilityModel (GNN + BiLSTM).
Compatible with hybrid_model_best.pth retrained model (F1=98.81%).
"""

import torch
import torch.nn as nn
import json
import pickle
import re
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import sys
import os

# Bypass transformers safety prompt
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add app directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.ml.hybrid_model import HybridVulnerabilityModel
from app.ml.enhanced_graph_builder import EnhancedFeatureExtractor
from app.ml.code_metrics import CodeMetricsExtractor


class HybridPredictor:
    """
    Predictor for vulnerability detection using trained HybridVulnerabilityModel
    """
    
    def __init__(
        self,
        model_path: str,
        vocab_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint (.pth/.pt)
            vocab_path: Path to vocabulary file (optional – auto-detected from checkpoint)
            device: Device to run on ('cuda' or 'cpu')
        """
        # Device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Load model (also initialises vocab, tokenizer, extractor, codebert)
        self.model, self.vocab, self.tokenizer = self._load_model(model_path, vocab_path)
        self.vocab_size = len(self.vocab)
        
        # Graph feature extractor (EnhancedFeatureExtractor – 832-dim nodes)
        self.extractor = EnhancedFeatureExtractor()
        
        # Code metrics extractor (20-dim)
        self.metrics_ext = CodeMetricsExtractor()
        
        print(f"[HybridPredictor] Loaded model from {model_path}")
        print(f"[HybridPredictor] Vocabulary size: {self.vocab_size}")
        print(f"[HybridPredictor] Device: {self.device}")
    
    # ------------------------------------------------------------------ #
    #  CodeBERT embedder  (768-dim per-node semantic features)            #
    # ------------------------------------------------------------------ #
    class _CodeBERTEmbedder:
        """Wraps CodeBERT to produce 768-dim per-node embeddings."""
        def __init__(self, device):
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.bert = AutoModel.from_pretrained("microsoft/codebert-base").to(device).eval()
            self.device = device

        @torch.no_grad()
        def embed(self, text: str) -> np.ndarray:
            enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=64, padding="max_length").to(self.device)
            out = self.bert(**enc).last_hidden_state[:, 0, :]   # CLS token
            return out.squeeze(0).cpu().numpy()                  # (768,)

    # ------------------------------------------------------------------ #
    #  Simple tokenizer  (mirrors training pipeline)                     #
    # ------------------------------------------------------------------ #
    class _SimpleTokenizer:
        """Re-implementation of the training tokenizer."""
        _SPLIT = re.compile(r'([a-z])([A-Z])|([A-Za-z])([0-9])|([0-9])([A-Za-z])|[_\-./]')

        def __init__(self, vocab: dict, max_len: int = 512):
            self.vocab = vocab
            self.max_len = max_len
            self.pad_id = vocab.get("<PAD>", 0)
            self.unk_id = vocab.get("<UNK>", 1)

        def encode(self, code: str) -> torch.Tensor:
            tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|[^\s]', code)
            ids = [self.vocab.get(t, self.unk_id) for t in tokens][:self.max_len]
            ids += [self.pad_id] * (self.max_len - len(ids))
            return torch.tensor([ids], dtype=torch.long)

    # ------------------------------------------------------------------ #
    #  Vocabulary loader                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_vocabulary(vocab_path: Optional[str], checkpoint: dict) -> dict:
        """Load vocabulary – prefers pickle produced during training."""
        # Get backend root for absolute paths
        BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent.parent
        
        # 1) Try pickle vocab (from training pipeline)
        base = Path(vocab_path).parent if vocab_path else None
        pkl_candidates = []
        if base:
            pkl_candidates.append(base / "vocabulary.pkl")
        pkl_candidates += [
            BACKEND_ROOT / "data" / "processed_graphs" / "vocabulary.pkl",
        ]
        for p in pkl_candidates:
            if p.exists():
                with open(p, "rb") as fh:
                    obj = pickle.load(fh)
                if isinstance(obj, dict) and "vocab" in obj:
                    print(f"[HybridPredictor] Loaded vocab from {p}")
                    return obj["vocab"]

        # 2) Try JSON vocab
        if vocab_path and Path(vocab_path).exists():
            with open(vocab_path, "r", encoding="utf-8") as fh:
                v = json.load(fh)
            if "token_to_id" in v:
                return v["token_to_id"]
            return v

        # 3) Fallback: auto-generate from size detected in checkpoint
        if "lstm_branch.embedding.weight" in (checkpoint.get("model_state_dict") or checkpoint):
            sd = checkpoint.get("model_state_dict", checkpoint)
            sz = sd["lstm_branch.embedding.weight"].shape[0]
            print(f"[HybridPredictor] WARN – generating dummy vocab of size {sz}")
            return {f"tok_{i}": i for i in range(sz)}

        raise RuntimeError("Cannot locate vocabulary for the model")

    # ------------------------------------------------------------------ #
    #  Model loader                                                       #
    # ------------------------------------------------------------------ #
    def _load_model(self, model_path: str, vocab_path: Optional[str] = None):
        """Load trained HybridVulnerabilityModel with correct config."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Extract state dict
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict",
                         checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint

        # Load vocabulary
        vocab = self._load_vocabulary(vocab_path, checkpoint if isinstance(checkpoint, dict) else {})

        # Reconstruct model with config saved in checkpoint (preferred) or known defaults
        saved_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        model = HybridVulnerabilityModel(
            vocab_size=saved_cfg.get("vocab_size", len(vocab)),
            node_feature_dim=saved_cfg.get("node_feature_dim", 832),
            gnn_hidden_dim=saved_cfg.get("gnn_hidden_dim", 128),
            gnn_output_dim=saved_cfg.get("gnn_output_dim", 64),
            lstm_embedding_dim=saved_cfg.get("lstm_embedding_dim", 128),
            lstm_hidden_dim=saved_cfg.get("lstm_hidden_dim", 128),
            lstm_output_dim=saved_cfg.get("lstm_output_dim", 64),
            metrics_input_dim=saved_cfg.get("metrics_input_dim", 20),
            metrics_output_dim=saved_cfg.get("metrics_output_dim", 128),
            fusion_hidden_dim=saved_cfg.get("fusion_hidden_dim", 128),
            dropout=saved_cfg.get("dropout", 0.2),
            use_gat=saved_cfg.get("use_gat", True),
            use_metrics=saved_cfg.get("use_metrics", True),
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()

        # Tokenizer
        tokenizer = self._SimpleTokenizer(vocab, max_len=512)

        # CodeBERT embedder (lazy – created on first use to avoid slow startup)
        self._codebert = None

        return model, vocab, tokenizer
    
    def predict(
        self,
        code: str,
        language: str = 'python',
        return_confidence: bool = True
    ) -> Dict:
        """
        Predict vulnerability in code
        
        Args:
            code: Source code string
            language: Programming language
            return_confidence: Return confidence scores
            
        Returns:
            Dictionary with prediction results:
            {
                'vulnerable': bool,
                'confidence': float,
                'raw_score': float,
                'gnn_features': tensor (optional),
                'lstm_features': tensor (optional)
            }
        """
        try:
            # --- CodeBERT embedder (lazy init) ---
            if self._codebert is None:
                self._codebert = self._CodeBERTEmbedder(self.device)

            # --- Graph features (832-dim per node) ---
            graph = self.extractor.extract_enhanced_graph(
                code, language, embedding_fn=self._codebert.embed
            )
            if graph is None or graph.x is None:
                return {
                    'vulnerable': False,
                    'confidence': 0.0,
                    'raw_score': 0.0,
                    'error': 'Failed to extract graph features'
                }

            # --- Token IDs (LSTM branch) ---
            token_ids = self.tokenizer.encode(code).to(self.device)  # [1, 512]

            # --- Code metrics (20-dim) ---
            metrics_np = self.metrics_ext.extract_all_features(code, language)
            metrics_t = torch.tensor(metrics_np, dtype=torch.float32).unsqueeze(0).to(self.device)

            # --- Batch the graph ---
            from torch_geometric.data import Batch
            graph = graph.to(self.device)
            graph_batch = Batch.from_data_list([graph])

            # --- Forward pass ---
            with torch.no_grad():
                # forward() returns 4 values: predictions, gnn_feats, lstm_feats, met_feats
                predictions, gnn_feats, lstm_feats, met_feats = self.model(
                    graph_batch, token_ids, metrics_t
                )
                score = torch.sigmoid(predictions[0]).item()

            # Interpret results
            is_vulnerable = score > 0.5
            confidence = score if is_vulnerable else (1.0 - score)

            result = {
                'vulnerable': is_vulnerable,
                'confidence': confidence,
                'raw_score': score
            }

            # Add intermediate features if requested
            if return_confidence:
                result['gnn_features'] = gnn_feats.cpu()
                result['lstm_features'] = lstm_feats.cpu()
                result['metrics_features'] = met_feats.cpu()

            return result

        except Exception as e:
            return {
                'vulnerable': False,
                'confidence': 0.0,
                'raw_score': 0.0,
                'error': str(e)
            }
    
    def predict_batch(
        self,
        codes: List[str],
        languages: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Predict vulnerabilities for multiple code samples
        """
        if languages is None:
            languages = ['python'] * len(codes)
        
        results = []
        for code, lang in zip(codes, languages):
            result = self.predict(code, language=lang, return_confidence=False)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': 'CombinedModel',
            'architecture': 'GNN + BiLSTM (Unified)',
            'vocab_size': self.vocab_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device)
        }


if __name__ == "__main__":
    # Test predictor – tries best_model.pt first, then legacy path
    candidates = ["models/best_model.pt", "training/models/hybrid_model_best.pth"]
    model_path = next((p for p in candidates if Path(p).exists()), candidates[0])

    predictor = HybridPredictor(model_path)

    # Print model info
    info = predictor.get_model_info()
    print("\n[Model Info]")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test prediction
    test_code = """
import os
user_input = input("Enter command: ")
os.system(user_input)  # Command injection vulnerability
    """

    print("\n[Test Prediction]")
    result = predictor.predict(test_code, language='python')
    print(f"  Vulnerable: {result['vulnerable']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Raw Score: {result['raw_score']:.4f}")
    if 'error' in result:
        print(f"  Error: {result['error']}")
