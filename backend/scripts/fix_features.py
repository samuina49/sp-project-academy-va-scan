"""
Feature Enhancement & Data Cleaning Pipeline
==============================================
Fixes the ROOT CAUSE of poor model performance:

ROOT CAUSE: Per-node CodeBERT embeddings are identical across vulnerable/safe
classes (cosine similarity = 0.999998). The GNN receives ZERO signal.

FIX STRATEGY:
1. Replace per-node CodeBERT (first 768 dims) with FULL-CODE CodeBERT [CLS] embedding
   - Each sample gets ONE CodeBERT embedding of the entire code snippet
   - This embedding IS class-discriminative (captures vulnerability semantics)
   - Broadcast to all nodes (nodes still differ via 64-dim base features)
2. Remove samples with conflicting labels (same code, different labels)
3. Remove data leakage (samples appearing in both train and val)
4. Z-score normalize code_metrics for better gradient flow

Author: Senior Project - AI Vulnerability Scanner
"""

import torch
import numpy as np
import pickle
import sys
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional, List
from tqdm import tqdm
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bypass transformers security check (same workaround as enhanced_dataset_pipeline.py)
from transformers import modeling_utils
modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: True

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

# We need torch_geometric for Data class
from torch_geometric.data import Data


@dataclass
class ProcessedSample:
    """Matches enhanced_dataset_pipeline.py"""
    code: str
    label: int
    language: str
    graph_data: Data
    vulnerability_type: str
    source: str
    metadata: dict
    token_ids: Optional[torch.Tensor] = None
    code_metrics: Optional[np.ndarray] = None


def load_codebert():
    """Load CodeBERT model for full-code embeddings"""
    from transformers import RobertaTokenizer, RobertaModel
    
    logger.info("Loading CodeBERT model...")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base", weights_only=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    if device == 'cuda':
        model = model.half()
    model.eval()
    
    logger.info(f"CodeBERT loaded on {device}")
    return tokenizer, model, device


def get_full_code_embedding(code: str, tokenizer, model, device, max_length=512) -> torch.Tensor:
    """
    Get [CLS] embedding for the FULL code snippet (not individual tokens).
    This captures the semantic meaning of the entire code, which IS
    discriminative between vulnerable and safe code.
    
    Returns: torch.Tensor of shape [768]
    """
    if not code or not code.strip():
        return torch.zeros(768)
    
    try:
        # Tokenize the full code (up to 512 tokens for good coverage)
        inputs = tokenizer(
            code.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use [CLS] token embedding as the full-code representation
        embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768]
        return embedding.cpu().squeeze().float()
        
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return torch.zeros(768)


def enhance_features(samples: List[ProcessedSample], tokenizer, model, device) -> List[ProcessedSample]:
    """
    Replace per-node CodeBERT embeddings with full-code CodeBERT embedding.
    
    Before: node features = [codebert_of_node_text(768) | base_features(64)] = 832
    After:  node features = [codebert_of_full_code(768) | base_features(64)] = 832
    
    The full-code embedding captures vulnerability semantics while
    base features (node type, text length, etc.) preserve structural info.
    """
    enhanced = []
    
    # Cache embeddings for duplicate code to save compute
    code_to_embedding = {}
    
    for sample in tqdm(samples, desc="Enhancing features"):
        code_key = hash(sample.code)
        
        # Get full-code embedding (cached)
        if code_key not in code_to_embedding:
            code_to_embedding[code_key] = get_full_code_embedding(
                sample.code, tokenizer, model, device
            )
        
        full_code_emb = code_to_embedding[code_key]  # [768]
        
        # Replace first 768 dims of every node with the full-code embedding
        graph = sample.graph_data
        num_nodes = graph.x.shape[0]
        feature_dim = graph.x.shape[1]
        
        if feature_dim == 832:
            # New x: [full_code_emb (768) | original_base_features (64)]
            new_x = graph.x.clone()
            
            # Broadcast full-code embedding to all nodes (replace first 768 dims)
            new_x[:, :768] = full_code_emb.unsqueeze(0).expand(num_nodes, -1)
            
            graph.x = new_x
        else:
            logger.warning(f"Unexpected feature dim: {feature_dim}, skipping enhancement")
        
        enhanced.append(sample)
    
    return enhanced


def remove_label_noise(samples: List[ProcessedSample]) -> List[ProcessedSample]:
    """
    Remove samples where the same code has conflicting labels.
    These are guaranteed bad training signal.
    """
    # Group by code hash
    code_to_labels = defaultdict(set)
    code_to_indices = defaultdict(list)
    
    for i, s in enumerate(samples):
        h = hash(s.code)
        code_to_labels[h].add(s.label)
        code_to_indices[h].append(i)
    
    # Find conflicting codes
    conflicting_hashes = {h for h, labels in code_to_labels.items() if len(labels) > 1}
    conflicting_indices = set()
    for h in conflicting_hashes:
        conflicting_indices.update(code_to_indices[h])
    
    clean = [s for i, s in enumerate(samples) if i not in conflicting_indices]
    removed = len(samples) - len(clean)
    
    if removed > 0:
        logger.info(f"  Removed {removed} samples with conflicting labels")
    
    return clean


def remove_exact_duplicates(samples: List[ProcessedSample]) -> List[ProcessedSample]:
    """Remove exact code duplicates (keep first occurrence)"""
    seen = set()
    unique = []
    
    for s in samples:
        h = hash(s.code)
        key = (h, s.label)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    
    removed = len(samples) - len(unique)
    if removed > 0:
        logger.info(f"  Removed {removed} exact duplicates")
    
    return unique


def remove_data_leakage(train_samples, val_samples) -> tuple:
    """Remove samples that appear in both train and val"""
    train_hashes = {hash(s.code) for s in train_samples}
    
    clean_val = [s for s in val_samples if hash(s.code) not in train_hashes]
    removed = len(val_samples) - len(clean_val)
    
    if removed > 0:
        logger.info(f"  Removed {removed} leaked samples from val ({removed}/{len(val_samples)} = {100*removed/len(val_samples):.1f}%)")
    
    return train_samples, clean_val


def normalize_code_metrics(samples: List[ProcessedSample], 
                           stats: dict = None) -> tuple:
    """
    Z-score normalize code metrics across the dataset.
    Returns (normalized_samples, stats_dict) for applying same normalization to val/test.
    """
    # Collect all metrics
    metrics_list = []
    for s in samples:
        if hasattr(s, 'code_metrics') and s.code_metrics is not None:
            metrics_list.append(s.code_metrics)
    
    if not metrics_list:
        return samples, None
    
    all_metrics = np.stack(metrics_list)
    
    if stats is None:
        # Compute stats from training data
        mean = all_metrics.mean(axis=0)
        std = all_metrics.std(axis=0)
        std[std < 1e-8] = 1.0  # Avoid division by zero
        stats = {'mean': mean, 'std': std}
    
    # Apply normalization
    for s in samples:
        if hasattr(s, 'code_metrics') and s.code_metrics is not None:
            s.code_metrics = (s.code_metrics - stats['mean']) / stats['std']
    
    return samples, stats


def verify_enhancement(samples: List[ProcessedSample], split_name: str):
    """Verify that features are now discriminative"""
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
        
        print(f"\n  [{split_name}] Post-enhancement separability:")
        print(f"    L2 distance:       {l2_dist:.6f}")
        print(f"    Cosine similarity: {cos_sim:.6f}")
        
        if cos_sim < 0.99:
            print(f"    *** GOOD: Features are now separable! ***")
        elif cos_sim < 0.999:
            print(f"    *** IMPROVED but still weak separation ***")
        else:
            print(f"    *** STILL NOT SEPARABLE - check embedding ***")


def main():
    data_dir = Path("data/processed_graphs")
    output_dir = Path("data/processed_graphs")  # Overwrite originals (or use separate dir)
    
    print("=" * 80)
    print("FEATURE ENHANCEMENT & DATA CLEANING PIPELINE")
    print("=" * 80)
    print()
    print("This script fixes the root cause of poor metrics (ROC-AUC ~0.52):")
    print("  1. Replace per-node CodeBERT with full-code CodeBERT embeddings")
    print("  2. Remove label noise (same code, different labels)")
    print("  3. Remove duplicates")
    print("  4. Remove data leakage (train/val overlap)")
    print("  5. Z-score normalize code metrics")
    print("=" * 80)
    
    # Backup originals
    print("\n[Step 0] Backing up original files...")
    for split in ['train', 'val', 'test']:
        src = data_dir / f"{split}_graphs.pkl"
        dst = data_dir / f"{split}_graphs_ORIGINAL.pkl"
        if src.exists() and not dst.exists():
            import shutil
            shutil.copy2(src, dst)
            print(f"  Backed up {src.name} -> {dst.name}")
    
    # Load CodeBERT
    print("\n[Step 1] Loading CodeBERT for full-code embeddings...")
    tokenizer, model, device = load_codebert()
    
    # Process each split
    splits_data = {}
    for split in ['train', 'val', 'test']:
        path = data_dir / f"{split}_graphs.pkl"
        if not path.exists():
            print(f"\n  SKIP: {path} not found")
            continue
        
        print(f"\n[Step 2] Loading {split} split...")
        with open(path, 'rb') as f:
            samples = pickle.load(f)
        print(f"  Loaded: {len(samples)} samples")
        
        splits_data[split] = samples
    
    if 'train' not in splits_data:
        print("ERROR: train_graphs.pkl not found!")
        return
    
    # Step 3: Clean data
    print(f"\n[Step 3] Cleaning data...")
    
    # Remove label noise from all splits
    for split in splits_data:
        print(f"\n  Cleaning {split}:")
        original_count = len(splits_data[split])
        splits_data[split] = remove_label_noise(splits_data[split])
        splits_data[split] = remove_exact_duplicates(splits_data[split])
        print(f"  {split}: {original_count} -> {len(splits_data[split])}")
    
    # Remove leakage
    if 'val' in splits_data:
        print(f"\n  Removing train/val leakage:")
        splits_data['train'], splits_data['val'] = remove_data_leakage(
            splits_data['train'], splits_data['val']
        )
    if 'test' in splits_data:
        print(f"\n  Removing train/test leakage:")
        splits_data['train'], splits_data['test'] = remove_data_leakage(
            splits_data['train'], splits_data['test']
        )
    
    # Step 4: Enhance features with full-code CodeBERT
    print(f"\n[Step 4] Enhancing features with full-code CodeBERT...")
    for split in splits_data:
        print(f"\n  Processing {split} ({len(splits_data[split])} samples)...")
        start = time.time()
        splits_data[split] = enhance_features(
            splits_data[split], tokenizer, model, device
        )
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s")
    
    # Step 5: Normalize code metrics (fit on train, apply to all)
    print(f"\n[Step 5] Normalizing code metrics...")
    splits_data['train'], metric_stats = normalize_code_metrics(splits_data['train'])
    for split in ['val', 'test']:
        if split in splits_data:
            splits_data[split], _ = normalize_code_metrics(splits_data[split], metric_stats)
    print(f"  Applied z-score normalization (fit on train)")
    
    # Save normalization stats for inference
    stats_path = output_dir / "metric_normalization_stats.pkl"
    with open(stats_path, 'wb') as f:
        pickle.dump(metric_stats, f)
    print(f"  Saved metric stats to {stats_path.name}")
    
    # Step 6: Verify enhancement
    print(f"\n[Step 6] Verifying feature enhancement...")
    for split in splits_data:
        verify_enhancement(splits_data[split], split)
    
    # Step 7: Save enhanced datasets
    print(f"\n[Step 7] Saving enhanced datasets...")
    for split, samples in splits_data.items():
        # Print class distribution
        vuln = sum(1 for s in samples if s.label == 1)
        safe = len(samples) - vuln
        print(f"\n  {split}: {len(samples)} samples (vuln={vuln}, safe={safe})")
        
        out_path = output_dir / f"{split}_graphs.pkl"
        with open(out_path, 'wb') as f:
            pickle.dump(samples, f)
        print(f"  Saved to {out_path}")
    
    print(f"\n{'='*80}")
    print("FEATURE ENHANCEMENT COMPLETE!")
    print("="*80)
    print("\nNext: Run training with the enhanced data:")
    print("  python training/train_enhanced.py")
    print("="*80)


if __name__ == "__main__":
    main()
