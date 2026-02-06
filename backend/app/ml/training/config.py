"""
Configuration for GNN + LSTM Model

This file contains all hyperparameters and settings for training
and inference of the vulnerability detection model.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    
    # GNN Configuration
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_output_dim: int = 64
    gnn_dropout: float = 0.3
    
    # LSTM Configuration
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True
    lstm_dropout: float = 0.3
    
    # Combined Model
    combined_hidden_dim: int = 128
    num_classes: int = 2  # Binary: vulnerable or not
    
    # Input Configuration
    max_sequence_length: int = 512
    vocab_size: int = 10000
    embedding_dim: int = 128
    max_graph_nodes: int = 1000


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    
    # Training
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Optimization
    optimizer: str = "adam"  # adam, adamw, sgd
    scheduler: str = "reduce_on_plateau"  # reduce_on_plateau, cosine, step
    early_stopping_patience: int = 10
    
    # Loss
    loss_function: str = "cross_entropy"  # cross_entropy, focal_loss
    class_weights: Optional[list] = None  # For imbalanced datasets
    
    # Regularization
    gradient_clip: float = 1.0
    dropout: float = 0.3
    
    # Validation
    val_split: float = 0.2
    test_split: float = 0.1
    
    # Checkpointing
    save_every_n_epochs: int = 5
    checkpoint_dir: str = "./models/checkpoints"
    best_model_path: str = "./models/best_model.pt"
    
    # Logging
    log_every_n_steps: int = 10
    tensorboard_dir: str = "./runs"
    
    # Device
    device: str = "cuda"  # cuda or cpu
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class DataConfig:
    """Data processing configuration"""
    
    # Paths
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    labeled_data_dir: str = "./data/labeled"
    
    # AST Configuration
    max_ast_depth: int = 50
    include_control_flow: bool = True
    include_data_flow: bool = True
    
    # Tokenization
    use_bpe: bool = True  # Byte-Pair Encoding
    vocab_file: str = "./data/vocab.json"
    
    # Graph Configuration
    edge_types: list = None  # Will be set to ["ast", "cfg", "dfg"]
    
    def __post_init__(self):
        if self.edge_types is None:
            self.edge_types = ["ast", "control_flow", "data_flow"]


# Create default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_DATA_CONFIG = DataConfig()
