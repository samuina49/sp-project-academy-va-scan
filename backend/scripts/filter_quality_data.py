"""Filter processed pickle files to keep only quality synthetic samples."""
import pickle
import sys
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class ProcessedSample:
    code: str
    label: int
    language: str
    graph_data: object
    vulnerability_type: str = ""
    source: str = ""
    metadata: dict = field(default_factory=dict)
    token_ids: Optional[object] = None
    code_metrics: Optional[np.ndarray] = None

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed_graphs")
OUT_DIR = DATA_DIR  # overwrite in place (originals already backed up)

def filter_split(split_name):
    path = os.path.join(DATA_DIR, f"{split_name}_graphs.pkl")
    with open(path, 'rb') as f:
        samples = pickle.load(f)
    
    print(f"\n{'='*60}")
    print(f"  {split_name.upper()}: {len(samples)} total samples")
    
    # Check source distribution
    sources = Counter(s.source for s in samples)
    print(f"  Source distribution: {dict(sources)}")
    
    # Filter to quality_synthetic only
    quality = [s for s in samples if s.source == "quality_synthetic"]
    print(f"  Quality synthetic: {len(quality)} samples")
    
    # Check label balance
    labels = Counter(s.label for s in quality)
    print(f"  Labels: {dict(labels)}")
    
    # Save filtered
    backup_path = os.path.join(DATA_DIR, f"{split_name}_graphs_MIXED.pkl")
    os.rename(path, backup_path)
    print(f"  Backed up mixed data to: {split_name}_graphs_MIXED.pkl")
    
    with open(path, 'wb') as f:
        pickle.dump(quality, f)
    print(f"  Saved {len(quality)} quality-only samples to: {split_name}_graphs.pkl")
    
    return len(quality), labels

total = 0
for split in ['train', 'val', 'test']:
    n, labels = filter_split(split)
    total += n

print(f"\n{'='*60}")
print(f"  TOTAL quality-only samples: {total}")
print(f"{'='*60}")
