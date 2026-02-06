"""
Hybrid ML Predictor for Vulnerability Detection

This module provides inference using the trained HybridVulnerabilityModel (GNN + BiLSTM).
Compatible with hybrid_model_best.pth retrained model (F1=98.81%).
"""

import torch
import torch.nn as nn
import json
from typing import Dict, Optional, List
from pathlib import Path
import sys

# Add app directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.ml.hybrid_model import HybridVulnerabilityModel  # ✅ Updated to use new retrained model
from app.ml.feature_extraction import FeatureExtractor


class HybridPredictor:
    """
    Predictor for vulnerability detection using trained HybridVulnerabilityModel
    """
    
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        device: Optional[str] = None
    ):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint (.pth)
            vocab_path: Path to vocabulary file (.json)
            device: Device to run on ('cuda' or 'cpu')
        """
        # Device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Load vocabulary
        self.vocab = self._load_vocabulary(vocab_path)
        self.vocab_size = len(self.vocab)
        
        # Initialize feature extractor
        self.extractor = FeatureExtractor(max_seq_length=256)
        self.extractor.token_to_id = self.vocab
        self.extractor.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Load model
        self.model = self._load_model(model_path)
        
        print(f"[HybridPredictor] Loaded model from {model_path}")
        print(f"[HybridPredictor] Vocabulary size: {self.vocab_size}")
        print(f"[HybridPredictor] Device: {self.device}")
    
    def _load_vocabulary(self, vocab_path: str) -> Dict[str, int]:
        """Load vocabulary from JSON file"""
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            
            # Handle nested structure if present
            if "token_to_id" in vocab:
                return vocab["token_to_id"]
                
            return vocab
        except Exception as e:
            raise RuntimeError(f"Failed to load vocabulary from {vocab_path}: {e}")
    
    def _load_model(self, model_path: str) -> HybridVulnerabilityModel:
        """Load trained HybridVulnerabilityModel"""
        try:
            # Load checkpoint first to check actual vocab size
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract state dict
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Detect actual vocab size from embedding layer (HybridVulnerabilityModel structure: lstm_branch.embedding)
            if 'lstm_branch.embedding.weight' in state_dict:
                actual_vocab_size = state_dict['lstm_branch.embedding.weight'].shape[0]
            elif 'lstm.embedding.weight' in state_dict:
                 # Legacy format
                actual_vocab_size = state_dict['lstm.embedding.weight'].shape[0]
            else:
                # Fallback to file vocab size
                actual_vocab_size = self.vocab_size
                
            print(f"[HybridPredictor] Detected vocab size from model: {actual_vocab_size}")
            
            # ✅ Use new HybridVulnerabilityModel architecture (matches training)
            model = HybridVulnerabilityModel(
                vocab_size=actual_vocab_size,
                node_feature_dim=64,
                lstm_embedding_dim=256,
                dropout=0.2
            )
            
            # Load trained weights
            model.load_state_dict(state_dict)
            
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
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
            # Extract features
            graph = self.extractor.code_to_graph(code, language=language)
            sequence = self.extractor.code_to_sequence(code, language=language)
            
            if not graph or not sequence or sequence.token_ids is None:
                return {
                    'vulnerable': False,
                    'confidence': 0.0,
                    'raw_score': 0.0,
                    'error': 'Failed to extract features'
                }
            
            # Prepare inputs for HybridVulnerabilityModel
            # Move graph to device
            graph = graph.to(self.device)
            
            # LSTM inputs - add batch dimension
            token_ids = sequence.token_ids.unsqueeze(0).to(self.device)  # [1, seq_len]
            
            # Inference
            with torch.no_grad():
                # Call HybridVulnerabilityModel.forward(graph_data, token_ids)
                # Returns: predictions [batch_size, 1], gnn_features, lstm_features
                predictions, gnn_feats, lstm_feats = self.model(graph, token_ids)
                
                # Get vulnerability score (sigmoid already applied in model)
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
    # Test predictor
    model_path = "training/models/hybrid_model_best.pth"
    vocab_path = "training/models/vocab.json"
    
    predictor = HybridPredictor(model_path, vocab_path)
    
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
