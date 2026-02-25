"""
Diagnose if metrics features are being used correctly by the model
"""
import torch
import pickle
import numpy as np
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from app.ml.hybrid_model import HybridVulnerabilityModel
from torch_geometric.data import Data, Batch

@dataclass
class ProcessedSample:
    """Data structure for processed code samples"""
    graph_data: Data
    token_ids: np.ndarray
    language: str
    label: int
    code_metrics: Optional[np.ndarray] = None

def load_sample_data(split='val', n_samples=10):
    """Load sample data from pickled dataset"""
    data_dir = Path("data/processed_graphs")
    pkl_file = data_dir / f"{split}_graphs.pkl"
    
    print(f"📂 Loading from {pkl_file}")
    with open(pkl_file, 'rb') as f:
        samples_list = pickle.load(f)
    
    samples = samples_list[:n_samples]
    labels = [s.label for s in samples]
    
    return samples, labels

def check_metrics_in_data(samples):
    """Check if metrics are present in data"""
    print("\n" + "="*70)
    print("🔍 CHECKING METRICS IN DATA")
    print("="*70)
    
    for i, sample in enumerate(samples[:3]):
        print(f"\n📦 Sample {i+1}:")
        print(f"   Has code_metrics: {sample.code_metrics is not None}")
        if sample.code_metrics is not None:
            print(f"   Metrics shape: {sample.code_metrics.shape}")
            print(f"   Metrics range: [{sample.code_metrics.min():.2f}, {sample.code_metrics.max():.2f}]")
            print(f"   Metrics mean: {sample.code_metrics.mean():.4f}")
            print(f"   Metrics std: {sample.code_metrics.std():.4f}")
            print(f"   Non-zero features: {np.count_nonzero(sample.code_metrics)}/20")
            
            # Show first 5 feature values
            print(f"   First 5 values: {sample.code_metrics[:5]}")

def check_model_uses_metrics(model, batch_data, batch_metrics):
    """Check if model actually uses metrics by comparing outputs"""
    print("\n" + "="*70)
    print("🧪 TESTING IF MODEL USES METRICS")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    batch_data = batch_data.to(device)
    batch_metrics = batch_metrics.to(device) if batch_metrics is not None else None
    token_ids = torch.zeros((batch_data.num_graphs, 512), dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Test 1: With real metrics
        pred_with, gnn_with, lstm_with, metrics_with = model(batch_data, token_ids, batch_metrics)
        
        # Test 2: With zeros instead of real metrics
        zeros_metrics = torch.zeros_like(batch_metrics) if batch_metrics is not None else None
        pred_zeros, gnn_zeros, lstm_zeros, metrics_zeros = model(batch_data, token_ids, zeros_metrics)
    
    print(f"\n🎯 Predictions with real metrics:")
    print(f"   Mean: {pred_with.mean():.4f}, Std: {pred_with.std():.4f}")
    print(f"   Range: [{pred_with.min():.4f}, {pred_with.max():.4f}]")
    print(f"   Vulnerable predictions: {(pred_with > 0.5).sum()}/{len(pred_with)}")
    
    print(f"\n🎯 Predictions with zero metrics:")
    print(f"   Mean: {pred_zeros.mean():.4f}, Std: {pred_zeros.std():.4f}")
    print(f"   Range: [{pred_zeros.min():.4f}, {pred_zeros.max():.4f}]")
    print(f"   Vulnerable predictions: {(pred_zeros > 0.5).sum()}/{len(pred_zeros)}")
    
    # Check differences
    diff_real_vs_zeros = (pred_with - pred_zeros).abs().mean().item()
    
    print(f"\n📊 Difference:")
    print(f"   Real vs Zeros: {diff_real_vs_zeros:.6f}")
    
    if diff_real_vs_zeros < 1e-4:
        print(f"\n⚠️  WARNING: Metrics branch has NO effect on predictions!")
        print(f"   Model is ignoring the metrics features.")
    elif diff_real_vs_zeros < 0.01:
        print(f"\n⚠️  WARNING: Metrics branch has VERY SMALL effect on predictions!")
        print(f"   Metrics features may be overwhelmed by GNN+LSTM.")
    else:
        print(f"\n✅ Metrics branch is affecting predictions.")
    
    # Check metrics features
    if metrics_with is not None:
        print(f"\n🔬 Metrics branch output:")
        print(f"   Shape: {metrics_with.shape}")
        print(f"   Mean: {metrics_with.mean():.4f}, Std: {metrics_with.std():.4f}")
        print(f"   Range: [{metrics_with.min():.4f}, {metrics_with.max():.4f}]")
        
        # Compare feature magnitudes
        print(f"\n📏 Feature magnitude comparison:")
        print(f"   GNN features:     mean={gnn_with.abs().mean():.4f}, std={gnn_with.abs().std():.4f}")
        print(f"   LSTM features:    mean={lstm_with.abs().mean():.4f}, std={lstm_with.abs().std():.4f}")
        print(f"   Metrics features: mean={metrics_with.abs().mean():.4f}, std={metrics_with.abs().std():.4f}")
        
        # Check if metrics are overwhelmed
        gnn_mag = gnn_with.abs().mean().item()
        lstm_mag = lstm_with.abs().mean().item()
        metrics_mag = metrics_with.abs().mean().item()
        
        if metrics_mag < gnn_mag * 0.1 or metrics_mag < lstm_mag * 0.1:
            print(f"\n⚠️  WARNING: Metrics features are much smaller than GNN/LSTM!")
            print(f"   Metrics may be overwhelmed in fusion layer.")

def analyze_batch_metrics(batch_metrics, labels):
    """Analyze metrics values for safe vs vulnerable samples"""
    print("\n" + "="*70)
    print("📊 METRICS ANALYSIS BY CLASS")
    print("="*70)
    
    vuln_mask = labels == 1
    safe_mask = labels == 0
    
    if batch_metrics is not None:
        vuln_metrics = batch_metrics[vuln_mask]
        safe_metrics = batch_metrics[safe_mask]
        
        print(f"\n🔴 Vulnerable samples (n={vuln_mask.sum()}):")
        print(f"   Metrics mean: {vuln_metrics.mean(dim=0)[:5]}")
        print(f"   Metrics std:  {vuln_metrics.std(dim=0)[:5]}")
        
        print(f"\n🟢 Safe samples (n={safe_mask.sum()}):")
        print(f"   Metrics mean: {safe_metrics.mean(dim=0)[:5]}")
        print(f"   Metrics std:  {safe_metrics.std(dim=0)[:5]}")
        
        # Calculate L2 distance
        vuln_mean = vuln_metrics.mean(dim=0)
        safe_mean = safe_metrics.mean(dim=0)
        l2_dist = torch.norm(vuln_mean - safe_mean).item()
        
        print(f"\n📏 L2 distance between classes: {l2_dist:.4f}")
        
        if l2_dist < 0.5:
            print(f"⚠️  WARNING: Classes are NOT well separated in metrics space!")
        else:
            print(f"✅ Classes are separable in metrics space.")

def main():
    print("\n" + "="*70)
    print("🔬 DIAGNOSING METRICS USAGE IN MODEL")
    print("="*70)
    
    # Load data
    samples, labels = load_sample_data('val', n_samples=32)
    
    # Check metrics in data
    check_metrics_in_data(samples)
    
    # Convert to PyG batch
    print("\n📦 Converting to PyG batch...")
    graphs = []
    metrics_list = []
    
    for sample in samples:
        graphs.append(sample.graph_data)
        if sample.code_metrics is not None:
            # Reshape to [1, 20] if needed
            m = sample.code_metrics
            if m.ndim == 1:
                m = m.reshape(1, -1)
            metrics_list.append(torch.FloatTensor(m))
        else:
            metrics_list.append(torch.zeros(1, 20))
    
    batch_data = Batch.from_data_list(graphs)
    batch_metrics = torch.cat(metrics_list, dim=0)  # [batch_size, 20]
    labels_tensor = torch.FloatTensor(labels)
    
    print(f"   Batch data: {batch_data.num_nodes} nodes, {batch_data.num_graphs} graphs")
    print(f"   Batch metrics: {batch_metrics.shape}")
    print(f"   Labels: {labels_tensor.shape}")
    
    # Analyze metrics
    analyze_batch_metrics(batch_metrics, labels_tensor)
    
    # Load model
    print("\n🤖 Loading model...")
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
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Use metrics: {model.use_metrics}")
    print(f"   Has metrics branch: {hasattr(model, 'metrics_branch')}")
    
    # Test model
    check_model_uses_metrics(model, batch_data, batch_metrics)
    
    print("\n" + "="*70)
    print("✅ DIAGNOSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
