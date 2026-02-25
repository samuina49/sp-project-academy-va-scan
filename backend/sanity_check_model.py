"""
Sanity Check: Test if model CAN learn at all
============================================
Train on just 100 samples with high LR to verify model architecture works.
If it can't overfit this tiny set, there's a fundamental problem.
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from pathlib import Path
import pickle
import sys
from dataclasses import dataclass
from typing import Optional, Dict

sys.path.append(str(Path(__file__).parent))
from app.ml.hybrid_model import HybridVulnerabilityModel

@dataclass
class ProcessedSample:
    """Match the dataclass from enhanced_dataset_pipeline.py"""
    code: str
    label: int  # 0 = safe, 1 = vulnerable
    language: str
    graph_data: Data
    vulnerability_type: str
    source: str
    metadata: Dict
    token_ids: Optional[torch.Tensor] = None

def sanity_check():
    print("="*60)
    print("SANITY CHECK: Can the model learn ANYTHING?")
    print("="*60)
    
    # Load tiny subset (100 samples)
    with open("data/processed_graphs/train_graphs.pkl", "rb") as f:
        all_samples = pickle.load(f)
    
    tiny_set = all_samples[:100]
    print(f"\nUsing {len(tiny_set)} samples for overfitting test")
    
    # Check label distribution
    labels = [s.label for s in tiny_set]
    print(f"Labels: {sum(labels)} vulnerable, {len(labels)-sum(labels)} safe")
    
    # Extract graph data from ProcessedSample objects
    graphs = []
    for sample in tiny_set:
        graph = sample.graph_data
        graph.y = torch.tensor([float(sample.label)], dtype=torch.float)
        
        # Ensure token_ids has correct shape [1, seq_len]
        if hasattr(graph, 'token_ids') and graph.token_ids is not None:
            if graph.token_ids.dim() == 1:
                graph.token_ids = graph.token_ids.unsqueeze(0)
        else:
            graph.token_ids = torch.zeros((1, 128), dtype=torch.long)
        
        graphs.append(graph)
    
    # Create dataloader (batch size 16)
    loader = DataLoader(graphs, batch_size=16, shuffle=True)
    
    # Load vocabulary
    with open("data/processed_graphs/vocabulary.pkl", "rb") as f:
        vocab_data = pickle.load(f)
    vocab_size = len(vocab_data['vocab'])
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridVulnerabilityModel(
        vocab_size=vocab_size,
        node_feature_dim=64,
        gnn_hidden_dim=64,
        lstm_hidden_dim=64,
        dropout=0.1
    ).to(device)
    
    print(f"\nDevice: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # High learning rate for fast overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)  # Very high!
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"\n{'='*60}")
    print("Training on 100 samples with LR=0.01 (should overfit fast)")
    print(f"{'='*60}\n")
    
    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            predictions, _, _ = model(batch, batch.token_ids)
            targets = batch.y.unsqueeze(1).float()
            
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            probs = torch.sigmoid(predictions)
            preds = (probs >= 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        
        acc = correct / total
        avg_loss = total_loss / len(loader)
        
        print(f"Epoch {epoch:2d}: Loss={avg_loss:.4f}, Accuracy={acc:.4f} ({acc*100:.1f}%)")
        
        # If we hit 95%+ accuracy on this tiny set, model CAN learn
        if acc >= 0.95:
            print(f"\n✅ SUCCESS! Model reached {acc*100:.1f}% accuracy")
            print("Model architecture is fine. Problem is with hyperparameters or data distribution.")
            return True
    
    if acc < 0.80:
        print(f"\n❌ FAIL! Model only reached {acc*100:.1f}% on 100 samples")
        print("This indicates a fundamental problem with model architecture or data.")
        return False
    else:
        print(f"\n⚠️ PARTIAL: Model reached {acc*100:.1f}%")
        print("Model can learn somewhat but struggles. Check learning rate and architecture.")
        return False

if __name__ == "__main__":
    sanity_check()
