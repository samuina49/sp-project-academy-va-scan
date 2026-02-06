"""
Combined GNN + LSTM Model

This module combines Graph Neural Networks and LSTM into a unified
architecture for vulnerability detection.

Architecture:
    Code Input
        │
        ├──> AST ──> Graph ──> GNN ──> Graph Features
        │                                   │
        └──> Tokens ──> Sequence ──> LSTM ──> Sequential Features
                                                │
                                                ▼
                                    Concatenate(GNN, LSTM)
                                                │
                                                ▼
                                        Classifier (MLP)
                                                │
                                                ▼
                                      Vulnerability Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from .gnn import GNNModel
from .lstm import LSTMModel


class CombinedModel(nn.Module):
    """
    Combined GNN + LSTM model for vulnerability detection
    
    This model:
    1. Processes code structure with GNN
    2. Processes code sequence with LSTM
    3. Combines features with an MLP classifier
    """
    
    def __init__(
        self,
        # GNN parameters
        gnn_input_dim: int = 64,
        gnn_hidden_dim: int = 128,
        gnn_output_dim: int = 64,
        gnn_num_layers: int = 3,
        gnn_dropout: float = 0.3,
        gnn_pooling: str = 'mean',
        
        # LSTM parameters
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        lstm_output_dim: int = 128,
        lstm_dropout: float = 0.3,
        lstm_bidirectional: bool = True,
        lstm_use_attention: bool = False,
        
        # Classifier parameters
        combined_hidden_dim: int = 128,
        num_classes: int = 2,
        classifier_dropout: float = 0.3
    ):
        """
        Initialize combined model
        
        Args:
            gnn_*: Parameters for GNN model
            lstm_*: Parameters for LSTM model
            combined_hidden_dim: Hidden dimension for classifier
            num_classes: Number of output classes (2 for binary)
            classifier_dropout: Dropout in classifier
        """
        super(CombinedModel, self).__init__()
        
        # GNN for graph features
        self.gnn = GNNModel(
            input_dim=gnn_input_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_layers=gnn_num_layers,
            dropout=gnn_dropout,
            pooling=gnn_pooling
        )
        
        # LSTM for sequential features
        self.lstm = LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            output_dim=lstm_output_dim,
            dropout=lstm_dropout,
            bidirectional=lstm_bidirectional,
            use_attention=lstm_use_attention
        )
        
        # Calculate combined feature dimension
        self.gnn_dim = self.gnn.get_output_dim()
        self.lstm_dim = self.lstm.get_output_dim()
        self.combined_dim = self.gnn_dim + self.lstm_dim
        
        # Classifier (Multi-Layer Perceptron)
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, combined_hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(combined_hidden_dim, combined_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(combined_hidden_dim // 2, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(
        self,
        # GNN inputs
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        
        # LSTM inputs
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through combined model
        
        Args:
            node_features: Node features for GNN [num_nodes, gnn_input_dim]
            edge_index: Edge indices for GNN [2, num_edges]
            batch: Batch assignment for nodes (for batched graphs)
            input_ids: Token IDs for LSTM [batch_size, seq_length]
            attention_mask: Attention mask for LSTM [batch_size, seq_length]
        
        Returns:
            Tuple of (logits [batch_size, num_classes], 
                     intermediate_outputs dict)
        """
        # GNN forward pass
        gnn_features = self.gnn(node_features, edge_index, batch)
        # gnn_features: [batch_size, gnn_output_dim]
        
        # LSTM forward pass
        lstm_features, attention_weights = self.lstm(input_ids, attention_mask)
        # lstm_features: [batch_size, lstm_output_dim]
        
        # Concatenate features
        combined_features = torch.cat([gnn_features, lstm_features], dim=1)
        # combined_features: [batch_size, gnn_dim + lstm_dim]
        
        # Classifier
        logits = self.classifier(combined_features)
        # logits: [batch_size, num_classes]
        
        # Return logits and intermediate outputs for analysis
        intermediate = {
            'gnn_features': gnn_features,
            'lstm_features': lstm_features,
            'combined_features': combined_features,
            'attention_weights': attention_weights
        }
        
        return logits, intermediate
    
    def predict_proba(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get probability predictions
        
        Returns:
            Probabilities [batch_size, num_classes]
        """
        logits, _ = self.forward(
            node_features, edge_index, batch,
            input_ids, attention_mask
        )
        
        if self.num_classes == 2:
            # Binary classification - use sigmoid
            probs = torch.softmax(logits, dim=1)
        else:
            # Multi-class - use softmax
            probs = torch.softmax(logits, dim=1)
        
        return probs
    
    def predict(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get class predictions
        
        Returns:
            Predicted classes [batch_size]
        """
        probs = self.predict_proba(
            node_features, edge_index, batch,
            input_ids, attention_mask
        )
        
        predictions = torch.argmax(probs, dim=1)
        
        return predictions


# Example usage
if __name__ == "__main__":
    # Sample parameters
    batch_size = 4
    
    # GNN inputs (example: 4 graphs with varying sizes)
    num_nodes = 80  # Total nodes across all graphs
    gnn_input_dim = 64
    num_edges = 160
    
    node_features = torch.randn(num_nodes, gnn_input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.tensor([0]*20 + [1]*20 + [2]*20 + [3]*20)  # 4 graphs
    
    # LSTM inputs
    vocab_size = 10000
    seq_length = 512
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Create model
    model = CombinedModel(
        gnn_input_dim=64,
        gnn_hidden_dim=128,
        gnn_output_dim=64,
        gnn_num_layers=3,
        vocab_size=vocab_size,
        embedding_dim=128,
        lstm_hidden_dim=128,
        lstm_num_layers=2,
        lstm_output_dim=128,
        combined_hidden_dim=128,
        num_classes=2
    )
    
    # Forward pass
    logits, intermediate = model(
        node_features=node_features,
        edge_index=edge_index,
        batch=batch,
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    
    print(f"GNN input shape: {node_features.shape}")
    print(f"LSTM input shape: {input_ids.shape}")
    print(f"\nIntermediate outputs:")
    print(f"  GNN features: {intermediate['gnn_features'].shape}")
    print(f"  LSTM features: {intermediate['lstm_features'].shape}")
    print(f"  Combined features: {intermediate['combined_features'].shape}")
    print(f"\nFinal logits: {logits.shape}")
    
    # Get predictions
    predictions = model.predict(
        node_features, edge_index, batch,
        input_ids, attention_mask
    )
    print(f"Predictions: {predictions}")
    
    # Get probabilities
    probs = model.predict_proba(
        node_features, edge_index, batch,
        input_ids, attention_mask
    )
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample probabilities:\n{probs}")
