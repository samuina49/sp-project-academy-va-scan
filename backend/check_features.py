
import pickle
import sys
import torch
from pathlib import Path
import numpy as np

# Add parent directory to path to import local modules
sys.path.append(str(Path(__file__).parent))

# We need the ProcessedSample class definition for pickle to work
import dataclasses
from dataclasses import dataclass
from torch_geometric.data import Data

# Define ProcessedSample matching train_enhanced.py just in case import fails or causes circular issues
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

def check_features():
    path = Path("data/processed_graphs/train_graphs.pkl")
    print(f"Loading {path}...")
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    # Check first 50 samples
    print("Checking feature variance...")
    
    unique_features = 0
    all_means = []
    
    for i in range(50):
        sample = data[i]
        x = sample.graph_data.x
        
        # Check if contains only zeros
        if torch.all(x == 0):
            print(f"Sample {i}: Label {sample.label} -> ALL ZEROS")
        else:
            # print(f"Sample {i}: Label {sample.label} -> Non-zero stats: Mean={x.mean().item():.4f}, Std={x.std().item():.4f}")
            all_means.append(x.mean().item())
            
        # Check standard deviation of the features
        if x.std() < 1e-6:
             print(f"Sample {i}: Label {sample.label} -> LOW VARIANCE (Std={x.std().item()})")
    
    print(f"\nMean of features across first 50 samples: {np.mean(all_means)}")
    print(f"Std of features across first 50 samples: {np.std(all_means)}")
    
    # Check if safe samples are different from vulnerable samples
    safe_samples = [d for d in data[:100] if d.label == 0]
    vuln_samples = [d for d in data[:100] if d.label == 1]
    
    if not safe_samples or not vuln_samples:
        print("Not enough samples to compare")
        return
        
    s_mean = torch.stack([s.graph_data.x.mean(dim=0) for s in safe_samples]).mean(dim=0)
    v_mean = torch.stack([s.graph_data.x.mean(dim=0) for s in vuln_samples]).mean(dim=0)
    
    diff = torch.norm(s_mean - v_mean)
    print(f"\nEuclidean distance between average Safe vs Vuln feature vectors: {diff.item():.6f}")

    if diff.item() < 1e-5:
        print("CRITICAL: Safe and Vulnerable features are statistically identical!")
    else:
        print("Features appear distinct.")

if __name__ == "__main__":
    check_features()
