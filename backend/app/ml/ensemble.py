"""
Multi-Model Ensemble Support
Combines predictions from multiple models for improved accuracy

Features:
- Model registry for managing multiple models
- Ensemble voting strategies (majority, weighted, stacking)
- Confidence calibration
- Model performance tracking
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class EnsembleStrategy(str, Enum):
    """Ensemble voting strategies"""
    MAJORITY = "majority"           # Simple majority voting
    WEIGHTED = "weighted"           # Weighted by confidence
    AVERAGE = "average"             # Average probabilities
    MAX_CONFIDENCE = "max_confidence"  # Take highest confidence prediction
    STACKING = "stacking"           # Meta-learner combination


@dataclass
class ModelInfo:
    """Information about a registered model"""
    name: str
    path: str
    model_type: str
    weight: float = 1.0
    accuracy: float = 0.0
    is_loaded: bool = False
    model: Optional[nn.Module] = None
    metadata: Dict = None


@dataclass
class EnsemblePrediction:
    """Ensemble prediction result"""
    is_vulnerable: bool
    confidence: float
    individual_predictions: List[Dict]
    strategy_used: str
    agreement_ratio: float


class ModelRegistry:
    """Registry for managing multiple models"""
    
    def __init__(self, models_dir: str = "ml/models"):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, ModelInfo] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def register_model(
        self,
        name: str,
        path: str,
        model_type: str,
        weight: float = 1.0,
        accuracy: float = 0.0
    ) -> bool:
        """Register a model in the registry"""
        full_path = self.models_dir / path if not os.path.isabs(path) else Path(path)
        
        if not full_path.exists():
            logger.warning(f"Model file not found: {full_path}")
            return False
        
        self.models[name] = ModelInfo(
            name=name,
            path=str(full_path),
            model_type=model_type,
            weight=weight,
            accuracy=accuracy,
            metadata={'registered_at': str(Path(full_path).stat().st_mtime)}
        )
        
        logger.info(f"Registered model: {name} ({model_type})")
        return True
    
    def unregister_model(self, name: str) -> bool:
        """Remove a model from the registry"""
        if name in self.models:
            if self.models[name].is_loaded:
                self.unload_model(name)
            del self.models[name]
            return True
        return False
    
    def load_model(self, name: str, model_class: type = None) -> bool:
        """Load a model into memory"""
        if name not in self.models:
            logger.error(f"Model not registered: {name}")
            return False
        
        model_info = self.models[name]
        
        try:
            checkpoint = torch.load(model_info.path, map_location=self.device)
            
            # If model class provided, instantiate it
            if model_class:
                model = model_class()
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                # Assume checkpoint contains the full model
                model = checkpoint.get('model', checkpoint)
            
            if isinstance(model, nn.Module):
                model.to(self.device)
                model.eval()
            
            model_info.model = model
            model_info.is_loaded = True
            
            logger.info(f"Loaded model: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            return False
    
    def unload_model(self, name: str) -> bool:
        """Unload a model from memory"""
        if name in self.models and self.models[name].is_loaded:
            self.models[name].model = None
            self.models[name].is_loaded = False
            torch.cuda.empty_cache()
            return True
        return False
    
    def get_loaded_models(self) -> List[ModelInfo]:
        """Get all currently loaded models"""
        return [m for m in self.models.values() if m.is_loaded]
    
    def list_models(self) -> List[Dict]:
        """List all registered models"""
        return [
            {
                'name': m.name,
                'type': m.model_type,
                'weight': m.weight,
                'accuracy': m.accuracy,
                'is_loaded': m.is_loaded,
                'path': m.path
            }
            for m in self.models.values()
        ]


class EnsemblePredictor:
    """Ensemble prediction using multiple models"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.strategy = EnsembleStrategy.WEIGHTED
    
    def set_strategy(self, strategy: EnsembleStrategy):
        """Set the ensemble strategy"""
        self.strategy = strategy
    
    def predict(
        self,
        embedding: torch.Tensor,
        strategy: Optional[EnsembleStrategy] = None
    ) -> EnsemblePrediction:
        """Make ensemble prediction"""
        strategy = strategy or self.strategy
        loaded_models = self.registry.get_loaded_models()
        
        if not loaded_models:
            raise RuntimeError("No models loaded for prediction")
        
        # Collect individual predictions
        predictions = []
        for model_info in loaded_models:
            try:
                with torch.no_grad():
                    if isinstance(model_info.model, nn.Module):
                        output = model_info.model(embedding)
                        if isinstance(output, tuple):
                            output = output[0]
                        prob = torch.sigmoid(output).item() if output.numel() == 1 else output.mean().item()
                    else:
                        # Assume callable
                        prob = model_info.model(embedding)
                
                predictions.append({
                    'model': model_info.name,
                    'probability': prob,
                    'is_vulnerable': prob > 0.5,
                    'weight': model_info.weight
                })
            except Exception as e:
                logger.error(f"Prediction failed for {model_info.name}: {e}")
        
        if not predictions:
            raise RuntimeError("All model predictions failed")
        
        # Apply ensemble strategy
        result = self._apply_strategy(predictions, strategy)
        
        return result
    
    def _apply_strategy(
        self,
        predictions: List[Dict],
        strategy: EnsembleStrategy
    ) -> EnsemblePrediction:
        """Apply ensemble strategy to combine predictions"""
        
        if strategy == EnsembleStrategy.MAJORITY:
            return self._majority_vote(predictions)
        elif strategy == EnsembleStrategy.WEIGHTED:
            return self._weighted_vote(predictions)
        elif strategy == EnsembleStrategy.AVERAGE:
            return self._average_vote(predictions)
        elif strategy == EnsembleStrategy.MAX_CONFIDENCE:
            return self._max_confidence(predictions)
        else:
            return self._weighted_vote(predictions)
    
    def _majority_vote(self, predictions: List[Dict]) -> EnsemblePrediction:
        """Simple majority voting"""
        vulnerable_votes = sum(1 for p in predictions if p['is_vulnerable'])
        total_votes = len(predictions)
        
        is_vulnerable = vulnerable_votes > total_votes / 2
        confidence = abs(vulnerable_votes - total_votes / 2) / (total_votes / 2)
        agreement = max(vulnerable_votes, total_votes - vulnerable_votes) / total_votes
        
        return EnsemblePrediction(
            is_vulnerable=is_vulnerable,
            confidence=confidence,
            individual_predictions=predictions,
            strategy_used="majority",
            agreement_ratio=agreement
        )
    
    def _weighted_vote(self, predictions: List[Dict]) -> EnsemblePrediction:
        """Weighted voting based on model weights"""
        total_weight = sum(p['weight'] for p in predictions)
        
        weighted_prob = sum(
            p['probability'] * p['weight'] / total_weight
            for p in predictions
        )
        
        is_vulnerable = weighted_prob > 0.5
        confidence = abs(weighted_prob - 0.5) * 2
        
        # Calculate agreement
        vulnerable_weight = sum(
            p['weight'] for p in predictions if p['is_vulnerable']
        )
        agreement = max(vulnerable_weight, total_weight - vulnerable_weight) / total_weight
        
        return EnsemblePrediction(
            is_vulnerable=is_vulnerable,
            confidence=confidence,
            individual_predictions=predictions,
            strategy_used="weighted",
            agreement_ratio=agreement
        )
    
    def _average_vote(self, predictions: List[Dict]) -> EnsemblePrediction:
        """Simple average of probabilities"""
        avg_prob = sum(p['probability'] for p in predictions) / len(predictions)
        
        is_vulnerable = avg_prob > 0.5
        confidence = abs(avg_prob - 0.5) * 2
        
        agreement = sum(
            1 for p in predictions if p['is_vulnerable'] == is_vulnerable
        ) / len(predictions)
        
        return EnsemblePrediction(
            is_vulnerable=is_vulnerable,
            confidence=confidence,
            individual_predictions=predictions,
            strategy_used="average",
            agreement_ratio=agreement
        )
    
    def _max_confidence(self, predictions: List[Dict]) -> EnsemblePrediction:
        """Take prediction with highest confidence"""
        # Find prediction furthest from 0.5
        best_pred = max(
            predictions,
            key=lambda p: abs(p['probability'] - 0.5)
        )
        
        is_vulnerable = best_pred['is_vulnerable']
        confidence = abs(best_pred['probability'] - 0.5) * 2
        
        agreement = sum(
            1 for p in predictions if p['is_vulnerable'] == is_vulnerable
        ) / len(predictions)
        
        return EnsemblePrediction(
            is_vulnerable=is_vulnerable,
            confidence=confidence,
            individual_predictions=predictions,
            strategy_used="max_confidence",
            agreement_ratio=agreement
        )


class ConfidenceCalibrator:
    """Calibrate model confidence scores"""
    
    def __init__(self):
        self.temperature = 1.0
        self.calibration_data = []
    
    def calibrate_temperature(
        self,
        predictions: List[float],
        labels: List[int]
    ) -> float:
        """Learn temperature scaling parameter"""
        # Simple temperature scaling calibration
        predictions_t = torch.tensor(predictions)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        
        best_temp = 1.0
        best_loss = float('inf')
        
        for temp in np.arange(0.5, 3.0, 0.1):
            calibrated = torch.sigmoid(
                torch.logit(predictions_t.clamp(1e-7, 1-1e-7)) / temp
            )
            loss = F.binary_cross_entropy(calibrated, labels_t).item()
            
            if loss < best_loss:
                best_loss = loss
                best_temp = temp
        
        self.temperature = best_temp
        return best_temp
    
    def calibrate(self, probability: float) -> float:
        """Apply calibration to a probability"""
        if self.temperature == 1.0:
            return probability
        
        # Clip to avoid log(0)
        prob = np.clip(probability, 1e-7, 1 - 1e-7)
        logit = np.log(prob / (1 - prob))
        calibrated_logit = logit / self.temperature
        
        return 1 / (1 + np.exp(-calibrated_logit))


# ==================== Factory Function ====================

def create_ensemble_system(models_dir: str = "ml/models") -> Tuple[ModelRegistry, EnsemblePredictor]:
    """Create and configure ensemble system"""
    registry = ModelRegistry(models_dir)
    predictor = EnsemblePredictor(registry)
    
    # Auto-discover and register models
    models_path = Path(models_dir)
    if models_path.exists():
        for model_file in models_path.glob("*.pth"):
            name = model_file.stem
            model_type = "hybrid" if "hybrid" in name else "unknown"
            registry.register_model(name, str(model_file), model_type)
    
    return registry, predictor
