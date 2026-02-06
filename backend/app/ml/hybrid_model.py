"""
Script 3: Hybrid Deep Learning Model (GNN + LSTM)
Production-Ready Code for Senior Project: AI-based Vulnerability Scanner

Purpose: Implement hybrid vulnerability detection model combining:
         - Graph Neural Network (GNN) for structural analysis
         - LSTM for sequential analysis
         - Feature Fusion layer for combining both representations

Model Architecture:
    Input Code
        ├─> AST Graph ─> GNN Branch ─┐
        │                             ├─> Feature Fusion ─> Classification
        └─> Token Sequence ─> LSTM ──┘

Addresses OWASP Top 10: A01, A03, A04, A05

Author: Senior Project - AI-based Vulnerability Scanner
Date: 2026-01-25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

# PyTorch Geometric for GNN
try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logging.warning("PyTorch Geometric not installed")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNNBranch(nn.Module):
    """
    Graph Neural Network Branch for STRUCTURAL ANALYSIS.
    
    Extracts structural features from AST/CFG graphs using Graph Convolutional Networks.
    This addresses the structural vulnerability patterns (e.g., dangerous API calls in
    specific contexts).
    
    Architecture:
    - 3 layers of Graph Attention Networks (GAT) or Graph Convolutional Networks (GCN)
    - Global pooling to aggregate node features
    - Fully connected layers for feature transformation
    """
    
    def __init__(
        self,
        node_feature_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_gnn_layers: int = 3,
        use_gat: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize GNN branch.
        
        Args:
            node_feature_dim: Dimension of input node features
            hidden_dim: Hidden dimension for GNN layers
            output_dim: Output dimension (for fusion)
            num_gnn_layers: Number of GNN layers
            use_gat: Use GAT (True) or GCN (False)
            dropout: Dropout rate
        """
        super(GNNBranch, self).__init__()
        
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric required for GNN")
        
        self.use_gat = use_gat
        self.dropout = dropout
        
        # Build GNN layers
        self.conv_layers = nn.ModuleList()
        
        # First layer: node_feature_dim -> hidden_dim
        if use_gat:
            self.conv_layers.append(
                GATConv(node_feature_dim, hidden_dim, heads=4, concat=False)
            )
        else:
            self.conv_layers.append(GCNConv(node_feature_dim, hidden_dim))
        
        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(num_gnn_layers - 2):
            if use_gat:
                self.conv_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
                )
            else:
                self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Last layer: hidden_dim -> hidden_dim
        if num_gnn_layers > 1:
            if use_gat:
                self.conv_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
                )
            else:
                self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization for each layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_gnn_layers)
        ])
        
        # Fully connected layers after pooling
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        logger.info(f"GNN Branch initialized: {'GAT' if use_gat else 'GCN'} with {num_gnn_layers} layers")
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through GNN branch.
        
        Args:
            data: PyTorch Geometric Data object with x, edge_index, batch
            
        Returns:
            Graph-level feature vector [batch_size, output_dim]
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Apply GNN layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling: aggregate all node features
        # This creates a single graph-level representation
        x = global_mean_pool(x, batch)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x


class LSTMBranch(nn.Module):
    """
    LSTM Branch for SEQUENTIAL ANALYSIS.
    
    Extracts sequential patterns from code token sequences using Bidirectional LSTM.
    This captures temporal dependencies and sequential vulnerability patterns
    (e.g., unsafe operations following user input).
    
    Architecture:
    - Embedding layer for tokens
    - 2 layers of Bidirectional LSTM
    - Attention mechanism (optional)
    - Fully connected layers
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM branch.
        
        Args:
            vocab_size: Size of token vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: LSTM hidden dimension
            output_dim: Output dimension (for fusion)
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super(LSTMBranch, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # Token embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layers
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        logger.info(f"LSTM Branch initialized: {'Bi' if bidirectional else ''}LSTM with {num_lstm_layers} layers")
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM branch.
        
        Args:
            token_ids: Token ID tensor [batch_size, seq_length]
            
        Returns:
            Sequence-level feature vector [batch_size, output_dim]
        """
        # Embed tokens
        embedded = self.embedding(token_ids)  # [batch, seq_len, embed_dim]
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: [batch, seq_len, hidden_dim * num_directions]
        # hidden: [num_layers * num_directions, batch, hidden_dim]
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states from last layer
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_combined = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            hidden_combined = hidden[-1, :, :]
        
        # Fully connected layers
        x = F.relu(self.fc1(hidden_combined))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x


class HybridVulnerabilityModel(nn.Module):
    """
    Hybrid Deep Learning Model for Vulnerability Detection.
    
    Combines:
    1. GNN Branch (Structural Analysis) - Extracts AST/CFG patterns
    2. LSTM Branch (Sequential Analysis) - Extracts token sequence patterns
    3. Feature Fusion Layer - Combines both representations
    4. Classification Head - Binary classification (Vulnerable: 0-1)
    
    This hybrid approach addresses the thesis objective of detecting
    OWASP Top 10 vulnerabilities (A01, A03, A04, A05) by leveraging
    both structural and sequential code features.
    """
    
    def __init__(
        self,
        vocab_size: int,
        node_feature_dim: int = 64,
        gnn_hidden_dim: int = 128,
        gnn_output_dim: int = 64,
        lstm_embedding_dim: int = 128,
        lstm_hidden_dim: int = 128,
        lstm_output_dim: int = 64,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.3,
        use_gat: bool = True
    ):
        """
        Initialize Hybrid Model.
        
        Args:
            vocab_size: Token vocabulary size
            node_feature_dim: Graph node feature dimension
            gnn_hidden_dim: GNN hidden dimension
            gnn_output_dim: GNN output dimension
            lstm_embedding_dim: LSTM embedding dimension
            lstm_hidden_dim: LSTM hidden dimension
            lstm_output_dim: LSTM output dimension
            fusion_hidden_dim: Fusion layer hidden dimension
            dropout: Dropout rate
            use_gat: Use GAT instead of GCN
        """
        super(HybridVulnerabilityModel, self).__init__()
        
        # ============================================
        # STRUCTURAL ANALYSIS: GNN Branch
        # ============================================
        self.gnn_branch = GNNBranch(
            node_feature_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            use_gat=use_gat,
            dropout=dropout
        )
        
        # ============================================
        # SEQUENTIAL ANALYSIS: LSTM Branch
        # ============================================
        self.lstm_branch = LSTMBranch(
            vocab_size=vocab_size,
            embedding_dim=lstm_embedding_dim,
            hidden_dim=lstm_hidden_dim,
            output_dim=lstm_output_dim,
            dropout=dropout,
            bidirectional=True
        )
        
        # ============================================
        # FEATURE FUSION: Combine GNN + LSTM features
        # ============================================
        fusion_input_dim = gnn_output_dim + lstm_output_dim
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # ============================================
        # CLASSIFICATION HEAD: Binary classification
        # ============================================
        # NOTE: Output raw logits (no sigmoid) for BCEWithLogitsLoss
        # Sigmoid will be applied in loss function or during inference
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
            # No Sigmoid here - BCEWithLogitsLoss handles it
        )
        
        logger.info("Hybrid Vulnerability Model initialized successfully")
        self._print_model_info()
    
    def forward(
        self,
        graph_data: Data,
        token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through hybrid model.
        
        Args:
            graph_data: PyTorch Geometric Data (AST graph)
            token_ids: Token IDs tensor [batch_size, seq_length]
            
        Returns:
            Tuple of:
            - predictions: Vulnerability probability [batch_size, 1]
            - gnn_features: Features from GNN branch [batch_size, gnn_output_dim]
            - lstm_features: Features from LSTM branch [batch_size, lstm_output_dim]
        """
        # ============================================
        # Branch 1: Extract structural features (GNN)
        # ============================================
        gnn_features = self.gnn_branch(graph_data)
        
        # ============================================
        # Branch 2: Extract sequential features (LSTM)
        # ============================================
        lstm_features = self.lstm_branch(token_ids)
        
        # ============================================
        # FEATURE FUSION: Concatenate both representations
        # ============================================
        # This is the key innovation: combining structural and sequential features
        fused_features = torch.cat([gnn_features, lstm_features], dim=1)
        
        # Pass through fusion layers
        fused_features = self.fusion_layers(fused_features)
        
        # ============================================
        # CLASSIFICATION: Predict vulnerability probability
        # ============================================
        predictions = self.classifier(fused_features)
        
        return predictions, gnn_features, lstm_features
    
    def predict(
        self,
        graph_data: Data,
        token_ids: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores.
        
        Args:
            graph_data: PyTorch Geometric Data
            token_ids: Token IDs
            threshold: Classification threshold
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.eval()
        with torch.no_grad():
            probabilities, _, _ = self.forward(graph_data, token_ids)
            predictions = (probabilities >= threshold).float()
        
        return predictions, probabilities
    
    def _print_model_info(self):
        """Print model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info("=" * 80)
        logger.info("HYBRID VULNERABILITY DETECTION MODEL")
        logger.info("=" * 80)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info("=" * 80)


# ============================================
# Example Usage and Testing
# ============================================
if __name__ == "__main__":
    print("=" * 80)
    print("HYBRID GNN + LSTM VULNERABILITY DETECTION MODEL")
    print("=" * 80)
    
    # Model hyperparameters
    VOCAB_SIZE = 5000
    NODE_FEATURE_DIM = 64
    BATCH_SIZE = 4
    SEQ_LENGTH = 256
    NUM_NODES = 50
    NUM_EDGES = 60
    
    # Initialize model
    model = HybridVulnerabilityModel(
        vocab_size=VOCAB_SIZE,
        node_feature_dim=NODE_FEATURE_DIM,
        gnn_hidden_dim=128,
        gnn_output_dim=64,
        lstm_embedding_dim=128,
        lstm_hidden_dim=128,
        lstm_output_dim=64,
        fusion_hidden_dim=128,
        dropout=0.3,
        use_gat=True
    )
    
    print("\n" + "=" * 80)
    print("Testing model with dummy data...")
    print("=" * 80)
    
    # Create dummy graph data
    x = torch.randn(NUM_NODES, NODE_FEATURE_DIM)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))
    batch = torch.zeros(NUM_NODES, dtype=torch.long)
    
    graph_data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Create dummy token sequence
    token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
    
    # Forward pass
    predictions, gnn_feat, lstm_feat = model(graph_data, token_ids)
    
    print(f"\nInput:")
    print(f"  Graph nodes: {NUM_NODES}")
    print(f"  Graph edges: {NUM_EDGES}")
    print(f"  Token sequence length: {SEQ_LENGTH}")
    print(f"\nOutput:")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  GNN features shape: {gnn_feat.shape}")
    print(f"  LSTM features shape: {lstm_feat.shape}")
    print(f"\nSample predictions (probabilities):")
    print(f"  {predictions.squeeze().tolist()}")
    
    print("\n" + "=" * 80)
    print("Model test completed successfully!")
    print("=" * 80)
