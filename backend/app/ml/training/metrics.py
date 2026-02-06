"""
Metrics for Model Evaluation

This module provides evaluation metrics for vulnerability detection:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- AUC-ROC
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from typing import Dict, Tuple


class MetricsCalculator:
    """Calculate evaluation metrics for model predictions"""
    
    def __init__(self, num_classes: int = 2):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of classes (2 for binary)
        """
        self.num_classes = num_classes
    
    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: torch.Tensor = None
    ) -> Dict[str, float]:
        """
        Calculate all metrics
        
        Args:
            predictions: Predicted classes [batch_size]
            targets: True labels [batch_size]
            probabilities: Class probabilities [batch_size, num_classes] (optional)
        
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Accuracy
        accuracy = accuracy_score(targets_np, preds_np)
        
        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            targets_np, preds_np, average='binary', zero_division=0
        )
        
        # Macro averages (for multi-class)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            targets_np, preds_np, average='macro', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
        }
        
        # AUC-ROC (if probabilities provided)
        if probabilities is not None and self.num_classes == 2:
            probs_np = probabilities.cpu().numpy()
            # Use probability of positive class
            try:
                auc = roc_auc_score(targets_np, probs_np[:, 1])
                metrics['auc_roc'] = auc
            except ValueError:
                # Not enough classes in batch
                metrics['auc_roc'] = 0.0
        
        return metrics
    
    def get_confusion_matrix(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> np.ndarray:
        """
        Calculate confusion matrix
        
        Args:
            predictions: Predicted classes [batch_size]
            targets: True labels [batch_size]
        
        Returns:
            Confusion matrix array
        """
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        cm = confusion_matrix(targets_np, preds_np)
        return cm
    
    def print_classification_report(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        class_names: list = None
    ):
        """
        Print detailed classification report
        
        Args:
            predictions: Predicted classes
            targets: True labels
            class_names: Names of classes
        """
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
        
        report = classification_report(
            targets_np, preds_np,
            target_names=class_names,
            zero_division=0
        )
        
        print(report)


def calculate_batch_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate metrics for a single batch
    
    Args:
        logits: Model output logits [batch_size, num_classes]
        labels: True labels [batch_size]
        num_classes: Number of classes
    
    Returns:
        Tuple of (loss, metrics_dict)
    """
    # Calculate loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    
    # Get predictions
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    
    # Calculate metrics
    calculator = MetricsCalculator(num_classes=num_classes)
    metrics = calculator.calculate_metrics(predictions, labels, probabilities)
    
    return loss.item(), metrics


# Example usage
if __name__ == "__main__":
    # Simulate predictions and targets
    batch_size = 32
    num_classes = 2
    
    # Random logits
    logits = torch.randn(batch_size, num_classes)
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    
    # Random targets
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Calculate metrics
    calculator = MetricsCalculator(num_classes=num_classes)
    metrics = calculator.calculate_metrics(predictions, targets, probabilities)
    
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Confusion matrix
    cm = calculator.get_confusion_matrix(predictions, targets)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Classification report
    print("\nClassification Report:")
    calculator.print_classification_report(
        predictions, targets,
        class_names=['Not Vulnerable', 'Vulnerable']
    )
