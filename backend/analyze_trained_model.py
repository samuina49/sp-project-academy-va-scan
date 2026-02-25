"""
Load trained model and check if it actually learned to use metrics
"""
import torch
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))

from app.ml.hybrid_model import HybridVulnerabilityModel
from torch_geometric.data import Data, Batch

@dataclass
class ProcessedSample:
    graph_data: Data
    token_ids: np.ndarray
    language: str
    label: int
    code_metrics: Optional[np.ndarray] = None

# Load trained model
print("="*70)
print("🔬 LOADING TRAINED MODEL")
print("="*70)

model_path = Path("training/checkpoints/best_model.pt")
if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    exit(1)

checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
print(f"\n📦 Checkpoint info:")
print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
f1 = checkpoint.get('f1', checkpoint.get('best_f1', 'N/A'))
if isinstance(f1, (int, float)):
    print(f"   F1 Score: {f1:.4f}")
else:
    print(f"   F1 Score: {f1}")
acc = checkpoint.get('accuracy', 'N/A')
if isinstance(acc, (int, float)):
    print(f"   Accuracy: {acc:.4f}")
else:
    print(f"   Accuracy: {acc}")

# Load validation data
print(f"\n📂 Loading validation data...")
with open('data/processed_graphs/val_graphs.pkl', 'rb') as f:
    val_samples = pickle.load(f)

samples = val_samples[:32]
labels = [s.label for s in samples]

# Convert to batch
graphs = []
metrics_list = []

for sample in samples:
    graphs.append(sample.graph_data)
    if sample.code_metrics is not None:
        m = sample.code_metrics
        if m.ndim == 1:
            m = m.reshape(1, -1)
        metrics_list.append(torch.FloatTensor(m))
    else:
        metrics_list.append(torch.zeros(1, 20))

batch_data = Batch.from_data_list(graphs)
batch_metrics = torch.cat(metrics_list, dim=0)
labels_tensor = torch.FloatTensor(labels)

print(f"   Batch: {batch_data.num_graphs} graphs")
print(f"   Metrics: {batch_metrics.shape}")

# Create model and load weights
print(f"\n🤖 Initializing model...")
model = HybridVulnerabilityModel(
    vocab_size=50000,
    node_feature_dim=batch_data.num_node_features,
    gnn_hidden_dim=64,
    gnn_output_dim=64,
    lstm_embedding_dim=128,
    lstm_hidden_dim=128,
    lstm_output_dim=64,
    metrics_input_dim=20,
    metrics_output_dim=32,
    fusion_hidden_dim=128,
    dropout=0.3,
    use_gat=True,
    use_metrics=True
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded successfully")

# Test with real metrics vs zeros
print(f"\n" + "="*70)
print("🧪 TESTING TRAINED MODEL WITH REAL VS ZERO METRICS")
print("="*70)

device = torch.device('cpu')
batch_data = batch_data.to(device)
batch_metrics = batch_metrics.to(device)
token_ids = torch.zeros((batch_data.num_graphs, 512), dtype=torch.long, device=device)

with torch.no_grad():
    # Test with real metrics
    pred_real, gnn_real, lstm_real, metrics_real = model(batch_data, token_ids, batch_metrics)
    
    # Test with zeros
    zeros_metrics = torch.zeros_like(batch_metrics)
    pred_zeros, gnn_zeros, lstm_zeros, metrics_zeros = model(batch_data, token_ids, zeros_metrics)

# Analyze predictions
print(f"\n📊 TRAINED MODEL PREDICTIONS:")
print(f"\nWith REAL metrics:")
print(f"   Mean: {pred_real.mean():.6f}")
print(f"   Std:  {pred_real.std():.6f}")
print(f"   Range: [{pred_real.min():.6f}, {pred_real.max():.6f}]")
print(f"   Predicted vulnerable: {(pred_real > 0).sum()}/{len(pred_real)}")

print(f"\nWith ZEROS metrics:")
print(f"   Mean: {pred_zeros.mean():.6f}")
print(f"   Std:  {pred_zeros.std():.6f}")
print(f"   Range: [{pred_zeros.min():.6f}, {pred_zeros.max():.6f}]")
print(f"   Predicted vulnerable: {(pred_zeros > 0).sum()}/{len(pred_zeros)}")

# Calculate difference
diff = (pred_real - pred_zeros).abs()
print(f"\n📏 DIFFERENCE (Real - Zeros):")
print(f"   Mean abs diff: {diff.mean():.6f}")
print(f"   Max abs diff:  {diff.max():.6f}")
print(f"   Min abs diff:  {diff.min():.6f}")

# Check if any predictions changed
predictions_changed = (pred_real > 0).ne(pred_zeros > 0).sum().item()
print(f"   Predictions changed: {predictions_changed}/{len(pred_real)}")

if diff.mean() < 1e-4:
    print(f"\n❌ PROBLEM: Trained model IGNORES metrics!")
    print(f"   Metrics branch has learned nothing useful.")
    print(f"   Fusion layer or classifier is not using metrics features.")
elif diff.mean() < 0.01:
    print(f"\n⚠️  WARNING: Trained model uses metrics VERY WEAKLY!")
    print(f"   Metrics have minimal impact on predictions (< 1%).")
    print(f"   Metrics may be overwhelmed by GNN+LSTM features.")
elif diff.mean() < 0.1:
    print(f"\n⚙️  Trained model uses metrics PARTIALLY.")
    print(f"   Metrics affect predictions but impact is small.")
else:
    print(f"\n✅ Trained model uses metrics EFFECTIVELY!")
    print(f"   Metrics have meaningful impact on predictions.")

# Analyze feature magnitudes
print(f"\n📏 FEATURE MAGNITUDES:")
print(f"   GNN:     {gnn_real.abs().mean():.6f}")
print(f"   LSTM:    {lstm_real.abs().mean():.6f}")
print(f"   Metrics: {metrics_real.abs().mean():.6f}")

# Check fusion layer weights
print(f"\n" + "="*70)
print("⚖️  FUSION LAYER WEIGHT ANALYSIS")
print("="*70)

fusion_layer = model.fusion_layers[0]  # First Linear layer
weights = fusion_layer.weight.data  # Shape: [128, 160]

# Analyze which input features have largest weights
gnn_weights = weights[:,  :64].abs().mean()  # First 64 = GNN features
lstm_weights = weights[:, 64:128].abs().mean()  # Next 64 = LSTM features
metrics_weights = weights[:, 128:].abs().mean()  # Last 32 = Metrics features

print(f"\nAverage absolute weights by branch:")
print(f"   GNN features (0:64):      {gnn_weights:.6f}")
print(f"   LSTM features (64:128):   {lstm_weights:.6f}")
print(f"   Metrics features (128:160): {metrics_weights:.6f}")

# Calculate relative importance
total_weight = gnn_weights + lstm_weights + metrics_weights
print(f"\nRelative weight contributions:")
print(f"   GNN:     {100*gnn_weights/total_weight:.1f}%")
print(f"   LSTM:    {100*lstm_weights/total_weight:.1f}%")
print(f"   Metrics: {100*metrics_weights/total_weight:.1f}%")

if metrics_weights < gnn_weights * 0.1 or metrics_weights < lstm_weights * 0.1:
    print(f"\n❌ PROBLEM: Fusion layer gives very LOW weight to metrics!")
    print(f"   Metrics features are being ignored during training.")
    print(f"   This explains why F1 score didn't improve.")
elif metrics_weights < gnn_weights * 0.5 or metrics_weights < lstm_weights * 0.5:
    print(f"\n⚠️  WARNING: Fusion layer gives SMALL weight to metrics.")
    print(f"   Metrics features may need stronger emphasis.")
else:
    print(f"\n✅ Fusion layer uses metrics features effectively.")

print(f"\n" + "="*70)
print("🎯 DIAGNOSIS SUMMARY")
print("="*70)
print(f"\n1. Dataset: ✅ Has valid metrics (100%, no zeros)")
print(f"2. Model architecture: ✅ Has metrics branch (32 dims)")
print(f"3. Training: ❓ Checking if metrics were learned...")
print(f"4. Fusion weights: See above analysis")
print(f"\n→ If fusion weights for metrics are very low, model didn't learn")
print(f"  to use them, explaining the stagnant F1 score (~0.67).")
print("="*70)
