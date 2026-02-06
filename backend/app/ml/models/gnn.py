"""
Graph Neural Network (GNN) Model

This module implements a Graph Convolutional Network for processing
code graphs derived from Abstract Syntax Trees.

Architecture:
- Multiple GCN layers for message passing
- Graph-level pooling for fixed-size representation
- Supports different edge types (AST, control flow, data flow)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from typing import Optional


class GNNModel(nn.Module):
    """
    Graph Neural Network for code vulnerability detection
    
    Takes a graph representation of code and produces a fixed-size
    embedding that captures structural patterns.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = 'mean'
    ):
        """
        Initialize GNN model
        
        Args:
            input_dim: Dimension of node features
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of GCN layers
            dropout: Dropout probability
            pooling: Pooling strategy ('mean', 'max', or 'both')
        """
        super(GNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        
        # Build GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Last layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.batch_norms.append(nn.BatchNorm1d(output_dim))
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Pooling adjustment if using both
        if pooling == 'both':
            self.final_dim = output_dim * 2
        else:
            self.final_dim = output_dim
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GNN
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for each node [num_nodes]
                   (for graph-level pooling in batched graphs)
        
        Returns:
            Graph embedding [batch_size, output_dim] or [batch_size, output_dim*2]
        """
        # Apply GCN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            
            # Apply ReLU except for last layer
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        # Graph-level pooling
        if batch is None:
            # Single graph case
            if self.pooling == 'mean':
                out = x.mean(dim=0, keepdim=True)
            elif self.pooling == 'max':
                out = x.max(dim=0, keepdim=True)[0]
            elif self.pooling == 'both':
                mean_pool = x.mean(dim=0, keepdim=True)
                max_pool = x.max(dim=0, keepdim=True)[0]
                out = torch.cat([mean_pool, max_pool], dim=1)
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")
        else:
            # Batched graphs
            if self.pooling == 'mean':
                out = global_mean_pool(x, batch)
            elif self.pooling == 'max':
                out = global_max_pool(x, batch)
            elif self.pooling == 'both':
                mean_pool = global_mean_pool(x, batch)
                max_pool = global_max_pool(x, batch)
                out = torch.cat([mean_pool, max_pool], dim=1)
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")
        
        return out
    
    def get_output_dim(self) -> int:
        """Get output dimension after pooling"""
        return self.final_dim


# Example usage
if __name__ == "__main__":
    # Create sample graph
    num_nodes = 20
    input_dim = 64
    num_edges = 40
    
    # Random node features
    x = torch.randn(num_nodes, input_dim)
    
    # Random edge index
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Create model
    model = GNNModel(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=64,
        num_layers=3,
        dropout=0.3,
        pooling='mean'
    )
    
    # Forward pass
    output = model(x, edge_index)
    
    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dimension: {model.get_output_dim()}")
    
    # Test with batch
    batch = torch.tensor([0] * 10 + [1] * 10)  # 2 graphs
    output_batched = model(x, edge_index, batch)
    print(f"Batched output shape: {output_batched.shape}")
