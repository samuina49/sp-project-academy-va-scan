"""
DEEP PROJECT AUDIT - Diagnose Why ROC-AUC is ~0.52
===================================================
This script checks for the REAL root causes of model failure.
"""
import torch
import pickle
import numpy as np
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Dict

# We need the ProcessedSample class to unpickle.
# The pickle file references torch_geometric.data.Data, 
# so we must import it from the correct module path.
sys.path.append(str(Path(__file__).parent.parent))

from torch_geometric.data import Data

@dataclass
class ProcessedSample:
    code: str
    label: int
    language: str
    graph_data: object
    vulnerability_type: str
    source: str
    metadata: dict
    token_ids: Optional[torch.Tensor] = None
    code_metrics: Optional[np.ndarray] = None

def audit():
    data_dir = Path("data/processed_graphs")
    
    # Load train + val
    print("="*80)
    print("DEEP DATA AUDIT")
    print("="*80)
    
    for split in ['train', 'val']:
        path = data_dir / f"{split}_graphs.pkl"
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        
        with open(path, 'rb') as f:
            samples = pickle.load(f)
        
        print(f"\n{'='*80}")
        print(f"SPLIT: {split.upper()} ({len(samples)} samples)")
        print(f"{'='*80}")
        
        # ===== CHECK 1: Label distribution =====
        labels = [s.label for s in samples]
        label_counts = Counter(labels)
        print(f"\n[CHECK 1] Label Distribution:")
        for lbl, cnt in sorted(label_counts.items()):
            print(f"  Label {lbl}: {cnt} ({100*cnt/len(samples):.1f}%)")
        
        # ===== CHECK 2: Duplicate code detection =====
        code_hashes = [hash(s.code) for s in samples]
        unique_codes = len(set(code_hashes))
        print(f"\n[CHECK 2] Duplicate Detection:")
        print(f"  Total: {len(samples)}, Unique code: {unique_codes}")
        print(f"  Duplicates: {len(samples) - unique_codes}")
        
        # Check if same code has different labels
        code_to_labels = {}
        for s in samples:
            h = hash(s.code)
            if h not in code_to_labels:
                code_to_labels[h] = set()
            code_to_labels[h].add(s.label)
        conflicting = sum(1 for v in code_to_labels.values() if len(v) > 1)
        print(f"  Same code, different labels: {conflicting}")
        
        # ===== CHECK 3: Graph feature analysis =====
        print(f"\n[CHECK 3] Graph Feature Analysis (first 200 samples):")
        x_all_zeros = 0
        x_dims = set()
        node_counts = []
        edge_counts = []
        
        for s in samples[:200]:
            g = s.graph_data
            x_dims.add(g.x.shape[1] if g.x.dim() > 1 else g.x.shape[0])
            node_counts.append(g.x.shape[0])
            edge_counts.append(g.edge_index.shape[1])
            if (g.x == 0).all():
                x_all_zeros += 1
        
        print(f"  Feature dimensions: {x_dims}")
        print(f"  All-zero graphs: {x_all_zeros}/200")
        print(f"  Node count: min={min(node_counts)}, max={max(node_counts)}, "
              f"mean={np.mean(node_counts):.1f}, median={np.median(node_counts):.1f}")
        print(f"  Edge count: min={min(edge_counts)}, max={max(edge_counts)}, "
              f"mean={np.mean(edge_counts):.1f}")
        
        # Feature value stats
        all_x = torch.cat([s.graph_data.x for s in samples[:100]], dim=0)
        print(f"  Feature values: min={all_x.min():.4f}, max={all_x.max():.4f}, "
              f"mean={all_x.mean():.4f}, std={all_x.std():.4f}")
        
        # ===== CHECK 4: Are features DIFFERENT between classes? =====
        print(f"\n[CHECK 4] Feature Separability (CRITICAL):")
        vuln_features = []
        safe_features = []
        for s in samples[:500]:
            # Graph-level mean feature
            mean_feat = s.graph_data.x.mean(dim=0)
            if s.label == 1:
                vuln_features.append(mean_feat)
            else:
                safe_features.append(mean_feat)
        
        if vuln_features and safe_features:
            vuln_stack = torch.stack(vuln_features)
            safe_stack = torch.stack(safe_features)
            
            vuln_mean = vuln_stack.mean(dim=0)
            safe_mean = safe_stack.mean(dim=0)
            
            diff = (vuln_mean - safe_mean).abs()
            diff_norm = diff.norm().item()
            
            cosine_sim = torch.nn.functional.cosine_similarity(
                vuln_mean.unsqueeze(0), safe_mean.unsqueeze(0)
            ).item()
            
            print(f"  Vulnerable graph mean norm: {vuln_mean.norm():.4f}")
            print(f"  Safe graph mean norm:       {safe_mean.norm():.4f}")
            print(f"  L2 distance between class centroids: {diff_norm:.6f}")
            print(f"  Cosine similarity:                   {cosine_sim:.6f}")
            
            if cosine_sim > 0.99:
                print(f"  *** CRITICAL: Features are IDENTICAL between classes! ***")
                print(f"  *** The model CANNOT distinguish vulnerable from safe ***")
            elif cosine_sim > 0.95:
                print(f"  *** WARNING: Features are very similar between classes ***")
            else:
                print(f"  ✅ Features show some class separation")
        
        # ===== CHECK 5: Token IDs analysis =====
        print(f"\n[CHECK 5] Token IDs Analysis:")
        has_tokens = sum(1 for s in samples 
                        if hasattr(s.graph_data, 'token_ids') 
                        and s.graph_data.token_ids is not None
                        and s.graph_data.token_ids.sum() > 0)
        zero_tokens = sum(1 for s in samples 
                         if hasattr(s.graph_data, 'token_ids') 
                         and s.graph_data.token_ids is not None
                         and s.graph_data.token_ids.sum() == 0)
        missing_tokens = sum(1 for s in samples 
                            if not hasattr(s.graph_data, 'token_ids') 
                            or s.graph_data.token_ids is None)
        print(f"  Has real tokens: {has_tokens}")
        print(f"  All-zero tokens: {zero_tokens}")
        print(f"  Missing tokens:  {missing_tokens}")
        
        # Check token_ids on sample object vs graph_data
        sample_tokens = sum(1 for s in samples 
                            if hasattr(s, 'token_ids') 
                            and s.token_ids is not None
                            and s.token_ids.sum() > 0)
        print(f"  sample.token_ids (non-zero): {sample_tokens}")
        
        # ===== CHECK 6: Code Metrics analysis =====
        print(f"\n[CHECK 6] Code Metrics Analysis:")
        has_metrics = 0
        zero_metrics = 0
        metrics_list = []
        for s in samples:
            if hasattr(s, 'code_metrics') and s.code_metrics is not None:
                has_metrics += 1
                m = s.code_metrics
                if isinstance(m, np.ndarray):
                    if np.all(m == 0):
                        zero_metrics += 1
                    else:
                        metrics_list.append(m)
        
        print(f"  Has metrics: {has_metrics}/{len(samples)}")
        print(f"  All-zero metrics: {zero_metrics}")
        
        if metrics_list:
            metrics_arr = np.stack(metrics_list[:200])
            print(f"  Metrics shape: {metrics_arr.shape}")
            print(f"  Metrics range: [{metrics_arr.min():.4f}, {metrics_arr.max():.4f}]")
            print(f"  Per-feature std (should not be 0):")
            stds = metrics_arr.std(axis=0)
            zero_std_count = (stds < 1e-6).sum()
            print(f"    Features with zero variance: {zero_std_count}/20")
            if zero_std_count > 0:
                zero_idx = np.where(stds < 1e-6)[0]
                print(f"    Dead features (indices): {zero_idx}")
        
        # ===== CHECK 7: Data leakage between train/val =====
        if split == 'val':
            print(f"\n[CHECK 7] Data Leakage Check:")
            train_path = data_dir / "train_graphs.pkl"
            if train_path.exists():
                with open(train_path, 'rb') as f:
                    train_samples = pickle.load(f)
                
                train_codes = set(hash(s.code) for s in train_samples)
                val_codes = set(hash(s.code) for s in samples)
                overlap = train_codes & val_codes
                print(f"  Train codes: {len(train_codes)}")
                print(f"  Val codes:   {len(val_codes)}")
                print(f"  Overlap:     {len(overlap)}")
                if overlap:
                    print(f"  *** DATA LEAKAGE DETECTED: {len(overlap)} shared samples ***")
        
        # ===== CHECK 8: Source distribution =====
        print(f"\n[CHECK 8] Source Distribution:")
        sources = Counter(s.source for s in samples)
        for src, cnt in sources.most_common():
            vuln_in_src = sum(1 for s in samples if s.source == src and s.label == 1)
            safe_in_src = sum(1 for s in samples if s.source == src and s.label == 0)
            print(f"  {src}: {cnt} (vuln={vuln_in_src}, safe={safe_in_src})")
        
        # ===== CHECK 9: Vulnerability types =====
        print(f"\n[CHECK 9] Vulnerability Types:")
        vtypes = Counter(s.vulnerability_type for s in samples if s.label == 1)
        for vt, cnt in vtypes.most_common(10):
            print(f"  {vt}: {cnt}")
    
    print(f"\n{'='*80}")
    print("AUDIT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    audit()
