"""
FAST AUDIT - avoids slow torch_geometric import
"""
import torch
import pickle
import numpy as np
import sys
from pathlib import Path
from collections import Counter

# Minimal imports - pickle will resolve classes itself via __main__
# We just need a dummy class for unpickling
from dataclasses import dataclass
from typing import Optional

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

def load_pkl(path):
    """Load pickle with proper module resolution"""
    import importlib
    # Pre-import torch_geometric.data so pickle can resolve Data class
    importlib.import_module('torch_geometric.data')
    
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    data_dir = Path("data/processed_graphs")
    
    print("="*80)
    print("DEEP DATA AUDIT")
    print("="*80)
    
    for split in ['train', 'val']:
        path = data_dir / f"{split}_graphs.pkl"
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        
        print(f"\nLoading {split}...")
        samples = load_pkl(path)
        
        print(f"\n{'='*80}")
        print(f"SPLIT: {split.upper()} ({len(samples)} samples)")
        print(f"{'='*80}")
        
        # CHECK 1: Label distribution
        labels = [s.label for s in samples]
        label_counts = Counter(labels)
        print(f"\n[CHECK 1] Label Distribution:")
        for lbl, cnt in sorted(label_counts.items()):
            print(f"  Label {lbl}: {cnt} ({100*cnt/len(samples):.1f}%)")
        
        # CHECK 2: Duplicate detection
        codes = [s.code for s in samples]
        code_hashes = [hash(c) for c in codes]
        unique_codes = len(set(code_hashes))
        print(f"\n[CHECK 2] Duplicate Detection:")
        print(f"  Total: {len(samples)}, Unique: {unique_codes}, Dupes: {len(samples)-unique_codes}")
        
        # Same code different labels
        code_to_labels = {}
        for s in samples:
            h = hash(s.code)
            if h not in code_to_labels:
                code_to_labels[h] = set()
            code_to_labels[h].add(s.label)
        conflicting = sum(1 for v in code_to_labels.values() if len(v) > 1)
        print(f"  Same code, different labels (LABEL NOISE): {conflicting}")
        
        # CHECK 3: Graph features
        print(f"\n[CHECK 3] Graph Feature Analysis:")
        n_check = min(200, len(samples))
        x_all_zeros = 0
        x_dims = set()
        node_counts = []
        edge_counts = []
        
        for s in samples[:n_check]:
            g = s.graph_data
            x_dims.add(tuple(g.x.shape))
            node_counts.append(g.x.shape[0])
            edge_counts.append(g.edge_index.shape[1])
            if (g.x == 0).all():
                x_all_zeros += 1
        
        feat_dim = list(x_dims)[0][1] if len(x_dims) == 1 else x_dims
        print(f"  Feature dim: {feat_dim}")
        print(f"  All-zero graphs: {x_all_zeros}/{n_check}")
        print(f"  Nodes: min={min(node_counts)}, max={max(node_counts)}, mean={np.mean(node_counts):.1f}")
        print(f"  Edges: min={min(edge_counts)}, max={max(edge_counts)}, mean={np.mean(edge_counts):.1f}")
        
        # Feature value stats
        all_x = torch.cat([s.graph_data.x for s in samples[:100]], dim=0)
        print(f"  Feature stats: min={all_x.min():.4f}, max={all_x.max():.4f}, mean={all_x.mean():.4f}, std={all_x.std():.4f}")
        
        # CHECK 4: CRITICAL - Feature separability between classes
        print(f"\n[CHECK 4] ***CRITICAL*** Feature Separability:")
        vuln_feats = []
        safe_feats = []
        for s in samples[:500]:
            mean_feat = s.graph_data.x.mean(dim=0)
            if s.label == 1:
                vuln_feats.append(mean_feat)
            else:
                safe_feats.append(mean_feat)
        
        if vuln_feats and safe_feats:
            v = torch.stack(vuln_feats).mean(dim=0)
            s_f = torch.stack(safe_feats).mean(dim=0)
            
            l2_dist = (v - s_f).norm().item()
            cos_sim = torch.nn.functional.cosine_similarity(v.unsqueeze(0), s_f.unsqueeze(0)).item()
            
            print(f"  Vuln centroid norm:  {v.norm():.4f}")
            print(f"  Safe centroid norm:  {s_f.norm():.4f}")
            print(f"  L2 distance:         {l2_dist:.6f}")
            print(f"  Cosine similarity:   {cos_sim:.6f}")
            
            if cos_sim > 0.999:
                print(f"  *** FATAL: Graph features are IDENTICAL across classes ***")
                print(f"  *** This means GNN branch CANNOT learn to separate them ***")
            elif cos_sim > 0.99:
                print(f"  *** SEVERE: Graph features nearly identical (cos={cos_sim:.6f}) ***")
            elif cos_sim > 0.95:
                print(f"  *** WARNING: Low separability ***")
            else:
                print(f"  OK: Some separability exists")
        
        # CHECK 5: Token IDs
        print(f"\n[CHECK 5] Token IDs:")
        real_tokens = 0
        zero_tokens = 0
        missing_tokens = 0
        for s in samples:
            g = s.graph_data
            if hasattr(g, 'token_ids') and g.token_ids is not None:
                if g.token_ids.sum() > 0:
                    real_tokens += 1
                else:
                    zero_tokens += 1
            else:
                missing_tokens += 1
        
        # Also check sample-level tokens
        sample_real = sum(1 for s in samples if hasattr(s, 'token_ids') and s.token_ids is not None and s.token_ids.sum() > 0)
        
        print(f"  graph.token_ids: real={real_tokens}, zero={zero_tokens}, missing={missing_tokens}")
        print(f"  sample.token_ids (non-zero): {sample_real}")
        
        if real_tokens == 0 and zero_tokens + missing_tokens == len(samples):
            print(f"  *** FATAL: ALL token_ids are zero or missing! ***")
            print(f"  *** LSTM branch receives NO signal - it's dead weight ***")
        
        # CHECK 6: Code Metrics
        print(f"\n[CHECK 6] Code Metrics:")
        has_m = 0
        zero_m = 0
        m_list = []
        for s in samples:
            if hasattr(s, 'code_metrics') and s.code_metrics is not None:
                has_m += 1
                if isinstance(s.code_metrics, np.ndarray):
                    if np.all(s.code_metrics == 0):
                        zero_m += 1
                    else:
                        m_list.append(s.code_metrics)
        
        print(f"  Has metrics: {has_m}/{len(samples)}")
        print(f"  All-zero:    {zero_m}/{has_m}")
        
        if m_list:
            arr = np.stack(m_list[:200])
            stds = arr.std(axis=0)
            dead_feats = (stds < 1e-6).sum()
            print(f"  Dead features (zero variance): {dead_feats}/20")
            
            # Per-class metric comparison
            v_metrics = []
            s_metrics = []
            for s in samples[:500]:
                if hasattr(s, 'code_metrics') and s.code_metrics is not None and not np.all(s.code_metrics == 0):
                    if s.label == 1:
                        v_metrics.append(s.code_metrics)
                    else:
                        s_metrics.append(s.code_metrics)
            
            if v_metrics and s_metrics:
                v_arr = np.stack(v_metrics)
                s_arr = np.stack(s_metrics)
                v_mean = v_arr.mean(axis=0)
                s_mean = s_arr.mean(axis=0)
                diff = np.abs(v_mean - s_mean)
                print(f"  Metric diff between classes (mean abs): {diff.mean():.6f}")
                top_diff_idx = np.argsort(diff)[-5:]
                print(f"  Most discriminative features: {top_diff_idx} (diff={diff[top_diff_idx]})")
        
        # CHECK 7: Source/dataset distribution
        print(f"\n[CHECK 7] Sources:")
        sources = Counter(s.source for s in samples)
        for src, cnt in sources.most_common():
            v = sum(1 for s in samples if s.source == src and s.label == 1)
            print(f"  {src}: {cnt} (vuln={v}, safe={cnt-v})")
        
        # CHECK 8: Code length distribution per class  
        print(f"\n[CHECK 8] Code Length per Class:")
        v_lens = [len(s.code) for s in samples if s.label == 1]
        s_lens = [len(s.code) for s in samples if s.label == 0]
        if v_lens and s_lens:
            print(f"  Vuln: mean={np.mean(v_lens):.0f}, median={np.median(v_lens):.0f}")
            print(f"  Safe: mean={np.mean(s_lens):.0f}, median={np.median(s_lens):.0f}")
    
    # CHECK 9: Data leakage
    train_path = data_dir / "train_graphs.pkl"
    val_path = data_dir / "val_graphs.pkl"
    if train_path.exists() and val_path.exists():
        print(f"\n[CHECK 9] Data Leakage:")
        train_data = load_pkl(train_path)
        val_data = load_pkl(val_path)
        
        train_codes = set(hash(s.code) for s in train_data)
        val_codes = set(hash(s.code) for s in val_data)
        overlap = train_codes & val_codes
        print(f"  Train unique: {len(train_codes)}")
        print(f"  Val unique:   {len(val_codes)}")
        print(f"  Overlap:      {len(overlap)}")
        if overlap:
            pct = 100 * len(overlap) / len(val_codes)
            print(f"  *** DATA LEAKAGE: {len(overlap)} samples ({pct:.1f}% of val) ***")
    
    print(f"\n{'='*80}")
    print("AUDIT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
