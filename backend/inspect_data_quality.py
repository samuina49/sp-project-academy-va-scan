"""
Data Quality Check: Can we distinguish safe vs vulnerable AT ALL?
==================================================================
Check if graph features have ANY difference between classes
"""

import torch
import pickle
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class ProcessedSample:
    """Match the dataclass from dataset pipeline"""
    code: str
    label: int
    language: str
    graph_data: any
    vulnerability_type: str
    source: str
    metadata: Dict
    token_ids: Optional[torch.Tensor] = None

def check_class_separation():
    """Check if safe and vulnerable samples are actually different"""
    
    print("="*70)
    print("DATA QUALITY CHECK: Are safe and vulnerable actually different?")
    print("="*70)
    
    # Load validation set
    with open("data/processed_graphs/val_graphs.pkl", "rb") as f:
        samples = pickle.load(f)[:500]  # First 500 for speed
    
    print(f"\nLoaded {len(samples)} validation samples")
    
    # Extract features and labels
    features = []
    labels = []
    
    for sample in samples:
        graph = sample.graph_data
        
        # Get node features: mean, std, min, max
        node_feat = graph.x.numpy()
        feat_mean = node_feat.mean(axis=0)
        feat_std = node_feat.std(axis=0)
        feat_min = node_feat.min(axis=0)
        feat_max = node_feat.max(axis=0)
        
        # Combine into single feature vector
        combined = np.concatenate([feat_mean, feat_std, feat_min, feat_max])
        features.append(combined)
        labels.append(sample.label)
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"\nFeature shape: {features.shape}")
    print(f"Safe samples: {(labels == 0).sum()}")
    print(f"Vulnerable samples: {(labels == 1).sum()}")
    
    # Check if features are different
    safe_features = features[labels == 0]
    vuln_features = features[labels == 1]
    
    safe_mean = safe_features.mean(axis=0)
    vuln_mean = vuln_features.mean(axis=0)
    
    # Euclidean distance between class means
    distance = np.linalg.norm(safe_mean - vuln_mean)
    print(f"\n📏 Distance between class means: {distance:.4f}")
    
    # Feature variance
    safe_var = safe_features.var(axis=0).mean()
    vuln_var = vuln_features.var(axis=0).mean()
    print(f"📊 Safe feature variance: {safe_var:.4f}")
    print(f"📊 Vulnerable feature variance: {vuln_var:.4f}")
    
    # Check if they overlap (using simple threshold)
    if distance < 0.01:
        print("\n❌ PROBLEM: Classes are almost identical!")
        print("   → Model CANNOT learn to distinguish them")
        print("   → Need to check:")
        print("     1. Are labels correct?")
        print("     2. Are graph features meaningful?")
        print("     3. Is code actually different between classes?")
    elif distance < 0.1:
        print("\n⚠️  WARNING: Classes have small separation")
        print("   → Model will struggle to learn")
        print("   → May need better features or more data")
    else:
        print("\n✅ Classes are separable")
        print("   → Model SHOULD be able to learn")
        print("   → Problem might be in training strategy")
    
    # Try PCA to check separation
    print("\n\n🔬 Performing PCA for 2D analysis...")
    try:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        # Calculate explained variance
        print(f"✅ PCA complete")
        print(f"   Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
        
        # Check overlap percentage
        safe_2d = features_2d[labels==0]
        vuln_2d = features_2d[labels==1]
        
        # Simple distance-based overlap check
        safe_mean_2d = safe_2d.mean(axis=0)
        vuln_mean_2d = vuln_2d.mean(axis=0)
        separation_2d = np.linalg.norm(safe_mean_2d - vuln_mean_2d)
        
        print(f"   2D separation distance: {separation_2d:.4f}")
        
        if separation_2d < 0.1:
            print("   ❌ Classes are HEAVILY overlapping in 2D!")
        elif separation_2d < 0.5:
            print("   ⚠️  Classes have significant overlap")
        else:
            print("   ✅ Classes show some separation")
            
    except Exception as e:
        print(f"   Could not perform PCA: {e}")
    
    # Sample some actual feature values
    print("\n\n📝 Sample Feature Statistics:")
    print(f"Safe class - First feature: mean={safe_mean[0]:.4f}, std={safe_features[:,0].std():.4f}")
    print(f"Vuln class - First feature: mean={vuln_mean[0]:.4f}, std={vuln_features[:,0].std():.4f}")
    
    # Statistical test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(safe_features[:, 0], vuln_features[:, 0])
    print(f"\nT-test p-value: {p_value:.6f}")
    if p_value < 0.01:
        print("✅ Features are statistically different (p < 0.01)")
    else:
        print("❌ Features are NOT statistically different")
        print("   → This is why model cannot learn!")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    check_class_separation()
