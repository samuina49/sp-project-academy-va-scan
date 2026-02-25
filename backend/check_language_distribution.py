import pickle
from collections import Counter
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

files = [
    'data/processed_graphs/train_graphs.pkl',
    'data/processed_graphs/val_graphs.pkl',
    'data/processed_graphs/test_graphs.pkl'
]

for f in files:
    print(f"\n📦 {f}")
    with open(f, 'rb') as fp:
        samples = pickle.load(fp)
    langs = [getattr(s, 'language', 'Unknown') for s in samples]
    c = Counter(langs)
    total = len(samples)
    for lang, count in c.items():
        print(f"   {lang:12s}: {count:5d} ({100*count/total:.1f}%)")
    print(f"   Total: {total}")