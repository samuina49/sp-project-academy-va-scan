"""
Simple ML Predictor for Vulnerability Detection

This module provides inference using the trained SimpleVulnDetector model.
Compatible with simple_model.pth trained model.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import re
import os


class SimpleVulnDetector(nn.Module):
    """Simple feed-forward neural network for vulnerability detection"""
    
    def __init__(self, input_dim=50):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class SimplePredictor:
    """
    Predictor for vulnerability detection using trained SimpleVulnDetector model
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None
    ):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run on ('cuda' or 'cpu')
        """
        # Device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
    
    def _load_model(self, model_path: str) -> SimpleVulnDetector:
        """
        Load trained model from checkpoint
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create model instance
        model = SimpleVulnDetector(input_dim=50)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load model weights (handle both formats)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state_dict format
            model.load_state_dict(checkpoint)
        
        return model
    
    def extract_features(self, code: str) -> torch.Tensor:
        """
        Extract features from code using pattern matching
        
        Args:
            code: Source code string
            
        Returns:
            Feature tensor of shape (50,)
        """
        features = []
        
        # Pattern-based features (45 patterns)
        patterns = [
            r'eval\(', r'exec\(', r'os\.system', r'subprocess\.', 
            r'\.execute\(', r'\.query\(', r'\.filter\(',
            r'password\s*=', r'key\s*=', r'secret\s*=',
            r'md5', r'sha1', r'random\.random',
            r'pickle\.', r'yaml\.load', r'marshal\.',
            r'input\(', r'raw_input\(', r'\.read\(',
            r'open\(', r'file\(', r'\.write\(',
            r'\.format\(', r'%\s*%', r'\+\s*request',
            r'sql\s*=', r'SELECT.*FROM', r'INSERT.*INTO',
            r'DELETE.*FROM', r'UPDATE.*SET', r'DROP.*TABLE',
            r'\.innerHTML', r'\.outerHTML', r'document\.write',
            r'\.cookie', r'localStorage', r'sessionStorage',
            r'http\.get', r'requests\.', r'urllib\.',
            r'__import__', r'compile\(', r'globals\(',
            r'child_process', r'Math\.random', r'new\s+Function',
        ]
        
        # Check each pattern
        for pattern in patterns:
            features.append(1.0 if re.search(pattern, code, re.IGNORECASE) else 0.0)
        
        # Statistical features (5 features)
        features.append(min(len(code) / 1000, 1.0))  # Code length
        features.append(min(code.count('\n') / 100, 1.0))  # Number of lines
        features.append(min(code.count('import') / 10, 1.0))  # Imports
        features.append(min(code.count('def ') / 10, 1.0))  # Functions
        features.append(min(code.count('class ') / 5, 1.0))  # Classes
        
        # Pad to 50 features if needed
        while len(features) < 50:
            features.append(0.0)
        
        return torch.tensor(features[:50], dtype=torch.float32)
    
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
            language: Programming language (for compatibility, not used)
            return_confidence: Return confidence scores
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features = self.extract_features(code).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            score = self.model(features).item()
        
        # Binary classification (threshold = 0.5)
        prediction = 1 if score >= 0.5 else 0
        
        # Prepare result
        result = {
            'vulnerable': bool(prediction == 1),
            'prediction': prediction,
            'label': 'VULNERABLE' if prediction == 1 else 'SAFE',
            'ml_score': float(score)
        }
        
        if return_confidence:
            result['confidence'] = float(score if prediction == 1 else 1 - score)
            result['probabilities'] = {
                'safe': float(1 - score),
                'vulnerable': float(score)
            }
        
        return result


# Example usage and testing
if __name__ == '__main__':
    # Initialize predictor
    predictor = SimplePredictor(
        model_path='./ml/models/simple_model.pth',
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
''',
        '''eval(userInput)
password = "admin123"
'''
    ]
    
    # Predict
    print("\n" + "="*60)
    print("TESTING SIMPLE ML PREDICTOR")
    print("="*60)
    
    for i, code in enumerate(test_codes):
        result = predictor.predict(code)
        print(f"\n📝 Sample {i+1}:")
        print(f"  Prediction: {result['label']}")
        print(f"  ML Score: {result['ml_score']:.4f}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Probabilities: Safe={result['probabilities']['safe']:.2%}, Vulnerable={result['probabilities']['vulnerable']:.2%}")
