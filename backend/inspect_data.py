import pickle
import sys
import torch
from pathlib import Path

# Add parent directory to path to import local modules
sys.path.append(str(Path(__file__).parent))

# We need the ProcessedSample class definition for pickle to work
import dataclasses
from dataclasses import dataclass
from torch_geometric.data import Data

# Define ProcessedSample matching train_enhanced.py just in case import fails or causes circular issues
# ideally we import from train_enhanced import ProcessedSample
try:
    from training.train_enhanced import ProcessedSample
except ImportError:
    print("Could not import ProcessedSample from training.train_enhanced. Defining dummy locally.")
    @dataclass
    class ProcessedSample:
        code: str
        label: int
        language: str
        graph_data: Data
        vulnerability_type: str
        source: str
        metadata: dict

def inspect():
    path = Path("data/processed_graphs/train_graphs.pkl")
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return
    
    print(f"Loaded {len(data)} samples.")
    
    total_nodes = 0
    total_edges = 0
    
    # Check first 10 samples
    print("\n--- Sample Inspection ---")
    for i in range(min(10, len(data))):
        sample = data[i]
        graph = sample.graph_data
        
        num_nodes = graph.num_nodes if graph.num_nodes is not None else (graph.x.shape[0] if graph.x is not None else 0)
        num_edges = graph.num_edges if graph.num_edges is not None else (graph.edge_index.shape[1] if graph.edge_index is not None else 0)
        
        total_nodes += num_nodes
        total_edges += num_edges
        
        print(f"Sample {i}: Label={sample.label}, Vuln={sample.vulnerability_type}")
        print(f"  Nodes={num_nodes}, Edges={num_edges}")
        if hasattr(graph, 'x') and graph.x is not None:
             print(f"  Feature shape: {graph.x.shape}")
        
    print(f"\nAverage Nodes (first 10): {total_nodes/10}")
    print(f"Average Edges (first 10): {total_edges/10}")

if __name__ == "__main__":
    inspect()
