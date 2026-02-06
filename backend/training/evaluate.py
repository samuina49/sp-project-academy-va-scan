"""
Model Evaluation Script
Evaluates trained model on test dataset
"""
import torch
from torch_geometric.loader import DataLoader
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from app.ml.hybrid_model import HybridVulnerabilityModel
from app.ml.feature_extraction import FeatureExtractor, code_to_graph


def load_test_data(test_path: Path, batch_size=32):
    """Load and prepare test dataset"""
    with open(test_path) as f:
        data = json.load(f)
    
    test_samples = data['samples']
    print(f"Test samples: {len(test_samples)}")
    
    test_graphs = []
    for sample in tqdm(test_samples, desc="Processing test"):
        try:
            graph = code_to_graph(sample['code'])
            if graph and sample['vulnerabilities']:
                graph.y = torch.tensor([1])
                test_graphs.append(graph)
        except Exception:
            continue
    
    return DataLoader(test_graphs, batch_size=batch_size)


def evaluate_model(model, test_loader, device):
    """Evaluate model and return metrics"""
    model.eval()
    
    y_true = []
    y_pred = []
    y_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = batch.to(device)
            out = model(batch)
            
            pred = (out > 0.5).float()
            
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_scores.extend(torch.sigmoid(out).cpu().numpy())
    
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    
    return metrics, y_true, y_pred


def print_evaluation_results(metrics):
    """Print evaluation results in a nice format"""
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    
    print(f"\n🎯 Confusion Matrix:")
    cm = np. array(metrics['confusion_matrix'])
    print(f"  True Negatives:  {cm[0][0] if len(cm) > 0 else 0}")
    print(f"  False Positives: {cm[0][1] if len(cm) > 0 else 0}")
    print(f"  False Negatives: {cm[1][0] if len(cm) > 1 else 0}")
    print(f"  True Positives:  {cm[1][1] if len(cm) > 1 else 0}")
    
    # Performance assessment
    print(f"\n✅ Performance Assessment:")
    if metrics['accuracy'] >= 0.90:
        print("  Excellent! Model exceeds target accuracy (90%+)")
    elif metrics['accuracy'] >= 0.85:
        print("  Good! Model meets target accuracy (85%+)")
    elif metrics['accuracy'] >= 0.80:
        print("  Acceptable. Consider more training or data.")
    else:
        print("  ⚠️  Below target. Needs improvement.")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model_path = Path("backend/ml/models/best_model.pth")
    if not model_path.exists():
        print("ERROR: No trained model found!")
        print("Please run: python backend/ml/train.py")
        sys.exit(1)
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = HybridGNNLSTM(
        node_features=100,
        hidden_dim=256,
        num_classes=1,
        num_lstm_layers=2
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded (trained for {checkpoint['epoch']} epochs)")
    
    # Load test data
    test_path = Path("backend/ml/data/test_dataset.json")
    if not test_path.exists():
        print("ERROR: Test dataset not found!")
        sys.exit(1)
    
    test_loader = load_test_data(test_path)
    
    # Evaluate
    print("\nEvaluating model on test set...")
    metrics, y_true, y_pred = evaluate_model(model, test_loader, device)
    
    # Print results
    print_evaluation_results(metrics)
    
    # Save metrics
    metrics_path = Path("backend/ml/models/evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to {metrics_path}")
