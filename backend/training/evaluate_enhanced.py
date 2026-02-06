"""
Enhanced Model Evaluation Script
Evaluate trained model on test set with detailed metrics

Features:
- Load trained checkpoint
- Evaluate on test set
- Detailed metrics (accuracy, precision, recall, F1)
- Confusion matrix
- Per-vulnerability-type analysis
- ROC curve and AUC
- Classification report

Author: AI Vulnerability Scanner
Date: 2026-02-06
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from pathlib import Path
import pickle
import sys
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from typing import List, Tuple, Dict
import json
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.ml.hybrid_model import HybridVulnerabilityModel
from training.train_enhanced import ProcessedSample, load_processed_dataset, samples_to_pyg_data

# =============================================================================
# Evaluation Functions
# =============================================================================

def load_model_checkpoint(checkpoint_path: Path, device: str = "cpu") -> Tuple[nn.Module, dict]:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        (model, checkpoint_dict)
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    
    # Get vocab size from saved model state (embedding layer shape)
    try:
        vocab_size = checkpoint['model_state_dict']['lstm_branch.embedding.weight'].shape[0]
    except (KeyError, AttributeError, IndexError) as e:
        print(f"Could not extract vocab_size from checkpoint, using config: {e}")
        vocab_size = config.get('vocab_size', 1000)
    
    # Recreate model
    model = HybridVulnerabilityModel(
        vocab_size=vocab_size,
        node_feature_dim=config.get('node_feature_dim', 64),
        gnn_hidden_dim=config.get('hidden_dim', 128),
        gnn_output_dim=64,
        lstm_embedding_dim=config.get('hidden_dim', 128),
        lstm_hidden_dim=config.get('lstm_hidden_dim', 128),
        lstm_output_dim=64,
        fusion_hidden_dim=config.get('hidden_dim', 128),
        use_gat=config.get('use_gat', True),
        dropout=config.get('dropout', 0.3)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    
    return model, checkpoint

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on dataset.
    
    Args:
        model: Trained model
        loader: DataLoader with test data
        device: Device
        
    Returns:
        (predictions, labels, probabilities)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)
            token_ids = batch.token_ids
            
            # Forward pass
            predictions, _, _ = model(batch, token_ids)
            
            # Get predictions
            probs = torch.sigmoid(predictions)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.unsqueeze(1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (
        np.array(all_preds).flatten(),
        np.array(all_labels).flatten(),
        np.array(all_probs).flatten()
    )

def compute_metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        labels: Ground truth labels
        predictions: Predicted labels
        probabilities: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0),
        'confusion_matrix': confusion_matrix(labels, predictions).tolist(),
    }
    
    # Add ROC-AUC if we have both classes
    if len(np.unique(labels)) > 1:
        metrics['roc_auc'] = roc_auc_score(labels, probabilities)
    else:
        metrics['roc_auc'] = None
    
    # Add classification report
    report = classification_report(labels, predictions, target_names=['Safe', 'Vulnerable'], output_dict=True)
    metrics['classification_report'] = report
    
    return metrics

def analyze_by_vulnerability_type(
    samples: List[ProcessedSample],
    predictions: np.ndarray,
    probabilities: np.ndarray
) -> Dict[str, dict]:
    """
    Analyze performance by vulnerability type.
    
    Args:
        samples: List of test samples
        predictions: Model predictions
        probabilities: Model probabilities
        
    Returns:
        Dictionary of per-type metrics
    """
    # Group by vulnerability type
    vuln_types = {}
    for i, sample in enumerate(samples):
        if sample.label == 1:  # Only vulnerable samples
            vtype = sample.vulnerability_type
            if vtype not in vuln_types:
                vuln_types[vtype] = {
                    'correct': 0,
                    'total': 0,
                    'avg_confidence': []
                }
            
            vuln_types[vtype]['total'] += 1
            if predictions[i] == 1:
                vuln_types[vtype]['correct'] += 1
            vuln_types[vtype]['avg_confidence'].append(probabilities[i])
    
    # Compute accuracy per type
    results = {}
    for vtype, data in vuln_types.items():
        results[vtype] = {
            'detection_rate': data['correct'] / data['total'] if data['total'] > 0 else 0,
            'num_samples': data['total'],
            'avg_confidence': float(np.mean(data['avg_confidence'])) if data['avg_confidence'] else 0
        }
    
    return results

def plot_confusion_matrix(cm: np.ndarray, save_path: Path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Safe', 'Vulnerable'],
        yticklabels=['Safe', 'Vulnerable']
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_roc_curve(labels: np.ndarray, probabilities: np.ndarray, save_path: Path):
    """Plot and save ROC curve"""
    if len(np.unique(labels)) < 2:
        print("Skipping ROC curve (need both classes)")
        return
    
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    auc = roc_auc_score(labels, probabilities)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path}")

def print_evaluation_report(metrics: dict, vuln_analysis: dict):
    """Print formatted evaluation report"""
    print("\n" + "=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    
    # Overall metrics
    print("\nOverall Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Safe  Vulnerable")
    print(f"  Actual Safe      {cm[0][0]:3d}     {cm[0][1]:3d}")
    print(f"  Actual Vulnerable {cm[1][0]:3d}     {cm[1][1]:3d}")
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    report = metrics['classification_report']
    for class_name in ['Safe', 'Vulnerable']:
        if class_name in report:
            print(f"  {class_name:12s} - Precision: {report[class_name]['precision']:.4f}, "
                  f"Recall: {report[class_name]['recall']:.4f}, "
                  f"F1: {report[class_name]['f1-score']:.4f}, "
                  f"Support: {int(report[class_name]['support'])}")
    
    # Vulnerability type analysis
    if vuln_analysis:
        print("\nDetection Rate by Vulnerability Type:")
        for vtype, data in sorted(vuln_analysis.items()):
            print(f"  {vtype:30s} - {data['detection_rate']:.1%} "
                  f"({data['num_samples']} samples, "
                  f"avg confidence: {data['avg_confidence']:.3f})")
    
    print("=" * 80)

# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate(
    checkpoint_path: Path,
    test_data_path: Path,
    output_dir: Path,
    device: str = "cpu"
):
    """
    Main evaluation function.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        test_data_path: Path to test dataset pickle file
        output_dir: Directory to save evaluation results
        device: Device to run evaluation on
    """
    print("=" * 80)
    print("ENHANCED MODEL EVALUATION")
    print("=" * 80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n[1/5] Loading Model...")
    model, checkpoint = load_model_checkpoint(checkpoint_path, device)
    
    # Load test data
    print("\n[2/5] Loading Test Data...")
    test_samples = load_processed_dataset(test_data_path)
    print(f"Loaded {len(test_samples)} test samples")
    
    # Prepare test graphs
    test_graphs = samples_to_pyg_data(test_samples)
    test_loader = DataLoader(test_graphs, batch_size=8, shuffle=False)
    
    # Evaluate
    print("\n[3/5] Evaluating Model...")
    predictions, labels, probabilities = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    print("\n[4/5] Computing Metrics...")
    metrics = compute_metrics(labels, predictions, probabilities)
    
    # Analyze by vulnerability type
    vuln_analysis = analyze_by_vulnerability_type(test_samples, predictions, probabilities)
    
    # Print report
    print_evaluation_report(metrics, vuln_analysis)
    
    # Save results
    print("\n[5/5] Saving Results...")
    
    # Save metrics as JSON
    results = {
        'checkpoint': str(checkpoint_path),
        'test_samples': len(test_samples),
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'roc_auc': float(metrics['roc_auc']) if metrics['roc_auc'] else None,
            'confusion_matrix': metrics['confusion_matrix'],
            'classification_report': metrics['classification_report']
        },
        'vulnerability_analysis': vuln_analysis
    }
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Save plots
    try:
        plot_confusion_matrix(
            np.array(metrics['confusion_matrix']),
            output_dir / 'confusion_matrix.png'
        )
        plot_roc_curve(
            labels, probabilities,
            output_dir / 'roc_curve.png'
        )
    except Exception as e:
        print(f"Warning: Could not save plots: {e}")
    
    print("\nEvaluation complete!")
    print("=" * 80)

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Enhanced Hybrid Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="training/checkpoints/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed_graphs/test_graphs.pkl",
        help="Path to test dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run evaluation on"
    )
    
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=Path(args.checkpoint),
        test_data_path=Path(args.test_data),
        output_dir=Path(args.output_dir),
        device=args.device
    )
