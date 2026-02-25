from dataclasses import dataclass
from typing import Optional
import numpy as np
from torch_geometric.data import Data
import pickle

@dataclass
class ProcessedSample:
    graph_data: Data
    token_ids: np.ndarray
    language: str
    label: int
    code_metrics: Optional[np.ndarray] = None

with open('data/processed_graphs/val_graphs.pkl', 'rb') as f:
    data = pickle.load(f)
    
print(f"Type: {type(data)}")
if isinstance(data, list):
    print(f"Length: {len(data)}")
    print(f"First item type: {type(data[0])}")
    if len(data) > 0 and hasattr(data[0], '__dict__'):
        print(f"First item attrs: {list(vars(data[0]).keys())}")
        if hasattr(data[0], 'code_metrics'):
            print(f"code_metrics: {data[0].code_metrics}" if data[0].code_metrics is not None else None)
