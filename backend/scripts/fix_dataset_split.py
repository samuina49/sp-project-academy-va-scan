"""
Fix Dataset Split - Re-split existing processed data with stratification
=========================================================================
This script re-splits the existing processed_graphs data using 
stratified random split (by language + label) without rebuilding graphs.

This is much faster than running the full pipeline again.

Author: AI Vulnerability Scanner
Date: February 6, 2026
"""

import pickle
import random
from pathlib import Path
from collections import defaultdict
import logging
import sys
from dataclasses import dataclass
from typing import Dict, Optional
import torch

# Add parent directory to path so we can import from scripts
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Now import enhanced_dataset_pipeline to get ProcessedSample
# Import only what we need to avoid tree-sitter loading
import importlib.util
spec = importlib.util.spec_from_file_location(
    "enhanced_dataset_pipeline",
    Path(__file__).parent / "enhanced_dataset_pipeline.py"
)
edp_module = importlib.util.module_from_spec(spec)
sys.modules['scripts.enhanced_dataset_pipeline'] = edp_module  # Register for pickle

try:
    spec.loader.exec_module(edp_module)
    ProcessedSample = edp_module.ProcessedSample
    logger_msg = "Imported ProcessedSample from enhanced_dataset_pipeline"
except Exception as e:
    # Fallback: define locally
    @dataclass
    class ProcessedSample:
        code: str
        label: int
        language: str
        graph_data: any
        vulnerability_type: str
        source: str
        metadata: Dict
        token_ids: Optional[torch.Tensor] = None
    logger_msg = f"Using local ProcessedSample (import failed: {e})"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(logger_msg)


def load_all_existing_samples():
    """Load all existing processed samples from train/val/test splits"""
    data_dir = Path("data/processed_graphs")
    
    all_samples = []
    
    for split_name in ["train", "val", "test"]:
        file_path = data_dir / f"{split_name}_graphs.pkl"
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            continue
        
        logger.info(f"Loading {file_path.name}...")
        with open(file_path, 'rb') as f:
            samples = pickle.load(f)
        
        logger.info(f"  Loaded {len(samples)} samples from {split_name}")
        all_samples.extend(samples)
    
    return all_samples


def stratified_split(samples, train_ratio=0.7, val_ratio=0.15):
    """
    Stratified random split by (language, label).
    
    Args:
        samples: List of ProcessedSample objects
        train_ratio: Ratio for training set (0.7 = 70%)
        val_ratio: Ratio for validation set (0.15 = 15%)
    
    Returns:
        train_samples, val_samples, test_samples
    """
    # Set seed for reproducibility
    random.seed(42)
    
    # Shuffle all samples first
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    logger.info(f"Total samples: {len(shuffled)}")
    
    # Group by (language, label)
    stratified_groups = defaultdict(list)
    
    for sample in shuffled:
        key = (sample.language, sample.label)
        stratified_groups[key].append(sample)
    
    logger.info(f"\n📊 Stratified groups:")
    for (lang, label), group_samples in stratified_groups.items():
        label_name = "vulnerable" if label == 1 else "safe"
        logger.info(f"  • {lang} + {label_name}: {len(group_samples)} samples")
    
    # Split each group proportionally
    train_samples = []
    val_samples = []
    test_samples = []
    
    for key, group_samples in stratified_groups.items():
        n = len(group_samples)
        train_size = int(train_ratio * n)
        val_size = int(val_ratio * n)
        
        train_samples.extend(group_samples[:train_size])
        val_samples.extend(group_samples[train_size:train_size + val_size])
        test_samples.extend(group_samples[train_size + val_size:])
    
    # Shuffle each split
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    
    return train_samples, val_samples, test_samples


def print_split_stats(split_name, samples):
    """Print statistics for a split"""
    logger.info(f"\n{split_name.upper()} SET:")
    logger.info(f"  Total: {len(samples)}")
    
    # Language distribution
    lang_counts = defaultdict(int)
    for s in samples:
        lang_counts[s.language] += 1
    
    logger.info(f"  Languages:")
    for lang, count in lang_counts.items():
        percentage = (count / len(samples)) * 100
        logger.info(f"    • {lang}: {count} ({percentage:.1f}%)")
    
    # Label distribution
    vuln_count = sum(1 for s in samples if s.label == 1)
    safe_count = len(samples) - vuln_count
    
    logger.info(f"  Labels:")
    logger.info(f"    • Vulnerable: {vuln_count} ({vuln_count/len(samples)*100:.1f}%)")
    logger.info(f"    • Safe: {safe_count} ({safe_count/len(samples)*100:.1f}%)")


def save_splits(train, val, test, output_dir="data/processed_graphs"):
    """Save the re-split datasets"""
    output_dir = Path(output_dir)
    
    splits = {
        'train': train,
        'val': val,
        'test': test
    }
    
    logger.info(f"\n💾 Saving re-split datasets...")
    
    # IMPORTANT: Ensure ProcessedSample has correct module path for pickle
    # This is a workaround to make pickle work correctly
    if hasattr(ProcessedSample, '__module__'):
        original_module = ProcessedSample.__module__
        ProcessedSample.__module__ = 'scripts.enhanced_dataset_pipeline'
    
    try:
        for split_name, samples in splits.items():
            output_file = output_dir / f"{split_name}_graphs.pkl"
            
            with open(output_file, 'wb') as f:
                pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"  ✅ Saved {len(samples)} samples to {output_file.name}")
    finally:
        # Restore original module (good practice)
        if 'original_module' in locals():
            ProcessedSample.__module__ = original_module


def main():
    logger.info("="*70)
    logger.info("🔧 FIXING DATASET SPLIT")
    logger.info("="*70)
    logger.info("\nThis will re-split existing processed data with stratification")
    logger.info("to ensure all splits have balanced language + label distributions.\n")
    
    # Load all existing samples
    logger.info("[1/3] Loading existing processed samples...")
    all_samples = load_all_existing_samples()
    
    if not all_samples:
        logger.error("❌ No samples found! Please run the full pipeline first.")
        return
    
    logger.info(f"\n✅ Loaded {len(all_samples)} total samples")
    
    # Perform stratified split
    logger.info("\n[2/3] Performing stratified random split...")
    train, val, test = stratified_split(all_samples)
    
    # Print statistics
    logger.info("\n[3/3] Split statistics:")
    logger.info("="*70)
    print_split_stats("TRAIN", train)
    print_split_stats("VAL", val)
    print_split_stats("TEST", test)
    logger.info("="*70)
    
    # Save
    save_splits(train, val, test)
    
    logger.info("\n✅ Dataset split fixed!")
    logger.info("\n🎯 NEXT STEP:")
    logger.info("   Run training again: python training/train_enhanced.py")
    logger.info("   OR use the launcher: .\\start_gpu_training.ps1")


if __name__ == "__main__":
    main()
