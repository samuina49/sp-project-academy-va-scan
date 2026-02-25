"""
Check if training dataset actually has code_metrics
"""
import pickle
from dataclasses import dataclass
from typing import Optional
import numpy as np
from torch_geometric.data import Data

@dataclass
class ProcessedSample:
    graph_data: Data
    token_ids: np.ndarray
    language: str
    label: int
    code_metrics: Optional[np.ndarray] = None

# Load train dataset
with open('data/processed_graphs/train_graphs.pkl', 'rb') as f:
    train_samples = pickle.load(f)

print(f"✅ Loaded {len(train_samples)} training samples\n")

# Check first 10 samples
has_metrics_count = 0
for i, sample in enumerate(train_samples[:10]):
    has_metrics = hasattr(sample, 'code_metrics') and sample.code_metrics is not None
    has_metrics_count += has_metrics
    
    is_zeros = False
    if has_metrics:
        # Check if all zeros
        is_zeros = np.allclose(sample.code_metrics, 0)
    
    print(f"Sample {i+1}:")
    print(f"  Has code_metrics: {has_metrics}")
    print(f"  Label: {'vulnerable' if sample.label == 1 else 'safe'}")
    if has_metrics:
        print(f"  All zeros: {is_zeros}")
        print(f"  Non-zero count: {np.count_nonzero(sample.code_metrics)}/20")
        print(f"  First 5 values: {sample.code_metrics[:5]}")

# Check all samples
print(f"\n{'='*70}")
print("📊 FULL DATASET ANALYSIS")
print(f"{'='*70}")

total_has_metrics = 0
total_is_zeros = 0
vuln_has_metrics = 0
safe_has_metrics = 0

for sample in train_samples:
    has_metrics = hasattr(sample, 'code_metrics') and sample.code_metrics is not None
    if has_metrics:
        total_has_metrics += 1
        if np.allclose(sample.code_metrics, 0):
            total_is_zeros += 1
        
        if sample.label == 1:
            vuln_has_metrics += 1
        else:
            safe_has_metrics += 1

print(f"\nTotal samples: {len(train_samples)}")
print(f"Has code_metrics: {total_has_metrics} ({100*total_has_metrics/len(train_samples):.1f}%)")
print(f"All zeros metrics: {total_is_zeros} ({100*total_is_zeros/total_has_metrics if total_has_metrics > 0 else 0:.1f}%)")
print(f"\nVulnerable with metrics: {vuln_has_metrics}")
print(f"Safe with metrics: {safe_has_metrics}")

if total_has_metrics == 0:
    print(f"\n❌ PROBLEM: No samples have code_metrics!")
    print(f"   Training script cannot use metrics branch.")
elif total_is_zeros == total_has_metrics:
    print(f"\n❌ PROBLEM: All metrics are zeros!")
    print(f"   Metrics branch will learn nothing useful.")
elif total_is_zeros > total_has_metrics * 0.5:
    print(f"\n⚠️  WARNING: {100*total_is_zeros/total_has_metrics:.1f}% of metrics are all zeros!")
    print(f"   Metrics branch may not learn effectively.")
else:
    print(f"\n✅ Dataset has valid metrics!")
    print(f"   Only {100*total_is_zeros/total_has_metrics:.1f}% are zeros.")
