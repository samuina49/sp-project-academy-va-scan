"""
ML Model Predictor for Vulnerability Detection

This module provides inference capabilities for the trained GNN + LSTM model.
Loads the model and makes predictions on new code samples.
"""

import torch
from typing import Dict, List, Tuple, Optional
import os

from ..hybrid_model import HybridVulnerabilityModel  # Updated: Use new model
from ..data.ast_parser import parse_code_to_ast
from ..data.graph_builder import GraphBuilder, GraphData
from ..data.tokenizer import CodeTokenizer
from ..training.config import ModelConfig


class VulnerabilityPredictor:
    """
    Predictor for vulnerability detection using trained model
    """
    
    def __init__(
        self,
        model_path: str,
        vocab_path: Optional[str] = None,
        device: str = None
    ):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to tokenizer vocabulary (optional)
            device: Device to run on ('cuda' or 'cpu')
        """
        # Device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Load model configuration and weights
        self.model, self.config = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Initialize graph builder
        self.graph_builder = GraphBuilder(node_feature_dim=64)
        
        # Initialize tokenizer
        self.tokenizer = CodeTokenizer(
            vocab_size=getattr(self.config, 'vocab_size', 10000),
            max_length=512
        )
        
        # Load vocabulary if provided
        if vocab_path and os.path.exists(vocab_path):
            self.tokenizer.load_vocab(vocab_path)
    
    def _load_model(self, model_path: str) -> Tuple[HybridVulnerabilityModel, Dict]:
        """
        Load trained model from checkpoint
        
        Args:
            model_path: Path to model file
            
        Returns:
            Tuple of (model, config)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint (new format: state_dict only, not full checkpoint)
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Old format with full checkpoint
                    state_dict = checkpoint['model_state_dict']
                    config = checkpoint.get('config', {})
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    config = checkpoint.get('config', {})
                else:
                    # New format: direct state_dict
                    state_dict = checkpoint
                    config = {}
            else:
                state_dict = checkpoint
                config = {}
            
            # Infer vocab_size from embedding layer in state_dict
            vocab_size = 10000  # default
            for key in state_dict.keys():
                if 'embedding.weight' in key:
                    vocab_size = state_dict[key].shape[0]
                    print(f"[VulnerabilityPredictor] Detected vocab size from model: {vocab_size}")
                    break
            
            # Create model with new HybridVulnerabilityModel architecture
            model = HybridVulnerabilityModel(
                vocab_size=vocab_size,
                node_feature_dim=64,
                lstm_embedding_dim=256,
                dropout=0.2
            )
            
            # Load weights
            model.load_state_dict(state_dict)
            
            return model, {'vocab_size': vocab_size, **config}
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
    
    def predict(
        self,
        code: str,
        language: str = "python",
        return_confidence: bool = True
    ) -> Dict:
        """
        Predict vulnerability in code
        
        Args:
            code: Source code string
            language: Programming language
            return_confidence: Return confidence scores
            
        Returns:
            Dictionary with prediction results
        """
        # Parse code to AST
        ast_nodes = parse_code_to_ast(code, language)
        
        # Build graph
        graph_data = self.graph_builder.build_graph(ast_nodes)
        
        # Tokenize code
        token_ids = self.tokenizer.encode(code, add_special_tokens=True)
        
        # Prepare tensors
        node_features = graph_data.node_features.unsqueeze(0).to(self.device)  # Add batch dim
        edge_index = graph_data.edge_index.to(self.device)
        batch = torch.zeros(graph_data.num_nodes, dtype=torch.long).to(self.device)
        
        # Pad/truncate tokens
        if len(token_ids) > 512:
            token_ids = token_ids[:512]
        
        attention_mask = [1] * len(token_ids)
        padding_length = 512 - len(token_ids)
        token_ids = token_ids + [0] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        
        token_ids_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits, intermediates = self.model(
                node_features,
                edge_index,
                batch,
                token_ids_tensor,
                attention_mask_tensor
            )
            
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
        
        # Prepare result
        result = {
            'vulnerable': bool(prediction == 1),
            'prediction': prediction,
            'label': 'VULNERABLE' if prediction == 1 else 'SAFE'
        }
        
        if return_confidence:
            result['confidence'] = float(probabilities[0, prediction].item())
            result['probabilities'] = {
                'safe': float(probabilities[0, 0].item()),
                'vulnerable': float(probabilities[0, 1].item())
            }
        
        return result
    
    def predict_batch(
        self,
        code_samples: List[str],
        language: str = "python"
    ) -> List[Dict]:
        """
        Predict vulnerabilities for multiple code samples
        
        Args:
            code_samples: List of code strings
            language: Programming language
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for code in code_samples:
            result = self.predict(code, language)
            results.append(result)
        
        return results


# Example usage
if __name__ == '__main__':
    # Initialize predictor
    predictor = VulnerabilityPredictor(
        model_path='./models/best_model.pt',
        device='cpu'
    )
    
    # Test code samples
    test_codes = [
        '''import os
def unsafe(user_input):
    os.system("echo " + user_input)
''',
        '''def safe(x):
    return x + 1
'''
    ]
    
    # Predict
    for i, code in enumerate(test_codes):
        result = predictor.predict(code)
        print(f"\nSample {i+1}:")
        print(f"  Prediction: {result['label']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Probabilities: Safe={result['probabilities']['safe']:.2%}, Vulnerable={result['probabilities']['vulnerable']:.2%}")
