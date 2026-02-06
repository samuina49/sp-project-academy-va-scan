"""
Enhanced Training Script for Hybrid GNN+LSTM Vulnerability Scanner
Uses pre-built graphs with CFG + DFG from enhanced pipeline

Key Differences from train.py:
- Loads pickle files (not JSON)
- Uses pre-built graphs (no feature extraction during training)
- Supports multi-edge types (AST, DFG, CFG)
- Works with ProcessedSample dataclass

Author: Senior Project - AI-based Vulnerability Scanner
Date: 2026-01-25
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from pathlib import Path
import pickle
import sys
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from dataclasses import dataclass
from typing import List, Tuple
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.ml.hybrid_model import HybridVulnerabilityModel

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters and settings"""
    # Data paths
    data_dir: Path = Path("data/processed_graphs")
    train_file: str = "train_graphs.pkl"
    val_file: str = "val_graphs.pkl"
    test_file: str = "test_graphs.pkl"
    
    # Model architecture
    node_feature_dim: int = 64
    hidden_dim: int = 128
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    num_gnn_layers: int = 3
    use_gat: bool = True  # GAT vs GCN
    
    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 0.001
    warmup_epochs: int = 5  # Warmup for stable training
    num_epochs: int = 50
    weight_decay: float = 0.0001
    dropout: float = 0.2  # Reduced from 0.3 to help learning
    gradient_clip: float = 1.0
    label_smoothing: float = 0.05  # Reduced from 0.1
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output
    checkpoint_dir: Path = Path("training/checkpoints")
    log_dir: Path = Path("training/logs")
    
    # Reproducibility
    seed: int = 42

# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class ProcessedSample:
    """Match the dataclass from enhanced_dataset_pipeline.py"""
    code: str
    label: int  # 0 = safe, 1 = vulnerable
    language: str
    graph_data: Data
    vulnerability_type: str
    source: str
    metadata: dict

def load_processed_dataset(pickle_path: Path) -> List[ProcessedSample]:
    """
    Load ProcessedSample objects from pickle file.
    
    Args:
        pickle_path: Path to pickle file
        
    Returns:
        List of ProcessedSample objects
    """
    if not pickle_path.exists():
        raise FileNotFoundError(f"Dataset not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        samples = pickle.load(f)
    
    print(f"Loaded {len(samples)} samples from {pickle_path.name}")
    return samples

def samples_to_pyg_data(samples: List[ProcessedSample]) -> List[Data]:
    """
    Extract PyG Data objects from ProcessedSamples and prepare for training.
    
    Args:
        samples: List of ProcessedSample objects
        
    Returns:
        List of PyG Data objects ready for DataLoader
    """
    graphs = []
    
    for sample in samples:
        try:
            # Get pre-built graph
            graph = sample.graph_data
            
            # Add label as float tensor [1] for binary classification
            label = float(sample.label)  # 0 or 1
            graph.y = torch.tensor([label], dtype=torch.float)
            
            # Ensure token_ids exists (for LSTM branch)
            # Note: token_ids should be [1, seq_len] for proper batching
            if not hasattr(graph, 'token_ids') or graph.token_ids is None:
                # Fallback: create dummy token sequence if missing
                # In production, this should not happen
                print(f"Warning: Missing token_ids for {sample.language} sample")
                graph.token_ids = torch.zeros((1, 128), dtype=torch.long)
            
            # Ensure token_ids has correct shape [1, seq_len]
            if graph.token_ids.dim() == 1:
                graph.token_ids = graph.token_ids.unsqueeze(0)
            
            graphs.append(graph)
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    return graphs

def print_dataset_stats(samples: List[ProcessedSample], split_name: str):
    """Print statistics about the dataset"""
    total = len(samples)
    vulnerable = sum(1 for s in samples if s.label == 1)
    safe = total - vulnerable
    
    # Count by language
    lang_counts = {}
    for s in samples:
        lang_counts[s.language] = lang_counts.get(s.language, 0) + 1
    
    # Count by vulnerability type
    vuln_type_counts = {}
    for s in samples:
        if s.label == 1:
            # Handle None or empty vulnerability_type
            vuln_type = s.vulnerability_type if s.vulnerability_type else "Unknown"
            vuln_type_counts[vuln_type] = vuln_type_counts.get(vuln_type, 0) + 1
    
    print(f"\n{'='*60}")
    print(f"📊 {split_name.upper()} DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"📦 Total samples:     {total:,}")
    print(f"🔴 Vulnerable:        {vulnerable:,} ({100*vulnerable/total:.1f}%)")
    print(f"🟢 Safe:              {safe:,} ({100*safe/total:.1f}%)")
    print(f"⚖️  Class Balance:     {'Balanced' if abs(vulnerable - safe) / total < 0.1 else 'Imbalanced'}")
    
    print(f"\n💻 Languages:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
        lang_str = str(lang) if lang else "Unknown"
        print(f"   • {lang_str:12s}: {count:4d} ({100*count/total:.1f}%)")
    
    if vuln_type_counts:
        print(f"\n🔐 Vulnerability Types (Top 5):")
        sorted_vulns = sorted(vuln_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for vuln_type, count in sorted_vulns:
            vuln_type_str = str(vuln_type) if vuln_type else "Unknown"
            print(f"   • {vuln_type_str:20s}: {count:3d} ({100*count/vulnerable:.1f}% of vulnerable)")
    
    # Graph statistics
    if samples:
        avg_nodes = np.mean([s.graph_data.x.size(0) for s in samples])
        avg_edges = np.mean([s.graph_data.edge_index.size(1) for s in samples])
        
        print(f"  Avg nodes per graph: {avg_nodes:.1f}")
        print(f"  Avg edges per graph: {avg_edges:.1f}")
        
        # Edge type distribution (if edge_attr exists)
        if hasattr(samples[0].graph_data, 'edge_attr') and samples[0].graph_data.edge_attr is not None:
            edge_types = []
            for s in samples:
                if hasattr(s.graph_data, 'edge_attr') and s.graph_data.edge_attr is not None:
                    edge_types.extend(s.graph_data.edge_attr.tolist())
            
            if edge_types:
                edge_type_counts = {
                    0: edge_types.count(0),  # AST
                    1: edge_types.count(1),  # DFG
                    2: edge_types.count(2)   # CFG
                }
                total_edges = sum(edge_type_counts.values())
                if total_edges > 0:
                    print(f"  Edge types: AST={100*edge_type_counts[0]/total_edges:.1f}%, "
                          f"DFG={100*edge_type_counts[1]/total_edges:.1f}%, "
                          f"CFG={100*edge_type_counts[2]/total_edges:.1f}%")

# =============================================================================
# Training Loop
# =============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    config: TrainingConfig
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        (average_loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Get token_ids for LSTM branch
        # PyG batches custom attributes by concatenation
        # If we added [1, seq_len], batch.token_ids becomes [batch_size, seq_len]
        token_ids = batch.token_ids
        
        # Forward pass
        predictions, _, _ = model(batch, token_ids)
        
        # Apply label smoothing to prevent overconfidence
        targets = batch.y.unsqueeze(1)
        if config.label_smoothing > 0:
            targets = targets * (1 - config.label_smoothing) + 0.5 * config.label_smoothing
        
        # Compute loss
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Monitor gradients (first batch of first epoch only for debugging)
        if total == 0:  # First batch
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            if grad_norm < 1e-7:
                print(f"⚠️  Warning: Very small gradients detected ({grad_norm:.2e})")
            elif grad_norm > 100:
                print(f"⚠️  Warning: Large gradients detected ({grad_norm:.2e})")
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predicted_labels = (torch.sigmoid(predictions) > 0.5).float()
        correct += (predicted_labels == batch.y.unsqueeze(1)).sum().item()
        total += batch.y.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float, dict]:
    """
    Validate the model.
    
    Returns:
        (average_loss, accuracy, detailed_metrics)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            batch = batch.to(device)
            token_ids = batch.token_ids
            
            # Forward pass
            predictions, _, _ = model(batch, token_ids)
            loss = criterion(predictions, batch.y.unsqueeze(1))
            
            total_loss += loss.item()
            
            # Get predictions
            probs = torch.sigmoid(predictions)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.unsqueeze(1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Detailed metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
    }
    
    return avg_loss, accuracy, metrics

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: TrainingConfig,
    filename: str = "best_model.pt",
    verbose: bool = True
):
    """Save model checkpoint with detailed status"""
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.checkpoint_dir / filename
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config.__dict__,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, checkpoint_path)
    
    if verbose:
        # Get file size
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        
        # Print save status
        print(f"\n{'='*60}")
        print(f"💾 MODEL SAVED: {filename}")
        print(f"{'='*60}")
        print(f"📍 Path: {checkpoint_path}")
        print(f"📊 Size: {file_size_mb:.2f} MB")
        print(f"🔢 Epoch: {epoch}")
        print(f"📈 Metrics:")
        print(f"   • Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"   • Precision: {metrics.get('precision', 0):.4f}")
        print(f"   • Recall:    {metrics.get('recall', 0):.4f}")
        print(f"   • F1 Score:  {metrics.get('f1', 0):.4f}")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path
) -> Tuple[int, dict]:
    """Load model checkpoint"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint['epoch'], checkpoint['metrics']

# =============================================================================
# Main Training Function
# =============================================================================

def train_model(config: TrainingConfig):
    """Main training pipeline"""
    
    print("=" * 80)
    print("Enhanced Hybrid GNN+LSTM Training Pipeline")
    print("With CFG + DFG Support")
    print("=" * 80)
    
    # Set seed
    set_seed(config.seed)
    
    # Load datasets
    print("\n[1/6] Loading Datasets...")
    train_samples = load_processed_dataset(config.data_dir / config.train_file)
    val_samples = load_processed_dataset(config.data_dir / config.val_file)
    
    # Print statistics
    print_dataset_stats(train_samples, "Training")
    print_dataset_stats(val_samples, "Validation")
    
    # Convert to PyG Data
    print("\n[2/6] Preparing Data Loaders...")
    train_graphs = samples_to_pyg_data(train_samples)
    val_graphs = samples_to_pyg_data(val_samples)
    
    print(f"Prepared {len(train_graphs)} training graphs")
    print(f"Prepared {len(val_graphs)} validation graphs")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_graphs,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Initialize model
    print("\n[3/6] Initializing Model...")
    
    # Get vocab size from vocabulary file (more accurate than max token_id)
    vocab_path = config.data_dir / 'vocabulary.pkl'
    if vocab_path.exists():
        import pickle
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        vocab_size = len(vocab_data['vocab'])
        print(f"Loaded vocabulary: {vocab_size} tokens")
    else:
        # Fallback: get from token_ids max
        vocab_size = train_graphs[0].token_ids.max().item() + 1 if train_graphs else 1000
        print(f"No vocabulary file, detected vocab size: {vocab_size}")
    
    model = HybridVulnerabilityModel(
        vocab_size=vocab_size,
        node_feature_dim=config.node_feature_dim,
        gnn_hidden_dim=config.hidden_dim,
        gnn_output_dim=64,
        lstm_embedding_dim=config.hidden_dim,
        lstm_hidden_dim=config.lstm_hidden_dim,
        lstm_output_dim=64,
        fusion_hidden_dim=config.hidden_dim,
        use_gat=config.use_gat,
        dropout=config.dropout
    ).to(config.device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Calculate class weights for imbalanced dataset
    print("\n[3.5/6] Calculating Class Weights...")
    train_labels = [sample.label for sample in train_samples]
    num_safe = sum(1 for label in train_labels if label == 0)
    num_vulnerable = sum(1 for label in train_labels if label == 1)
    
    print(f"Training set distribution:")
    print(f"  Safe samples: {num_safe} ({num_safe/len(train_labels)*100:.1f}%)")
    print(f"  Vulnerable samples: {num_vulnerable} ({num_vulnerable/len(train_labels)*100:.1f}%)")
    
    # Set pos_weight to balance classes
    # pos_weight boosts the MINORITY class
    if num_safe > num_vulnerable:
        # Vulnerable is minority - boost it
        pos_weight = torch.tensor([num_safe / num_vulnerable], device=config.device)
        print(f"✓ Using pos_weight={pos_weight.item():.4f} to boost vulnerable class")
    elif num_vulnerable > num_safe:
        # Safe is minority - use inverse weight (or set to 1.0)
        pos_weight = torch.tensor([1.0], device=config.device)
        print(f"✓ Vulnerable is majority, using pos_weight=1.0")
    else:
        pos_weight = None
        print(f"✓ Balanced dataset, no pos_weight needed")
    
    # Loss and optimizer
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        """Warmup for first few epochs, then cosine decay"""
        if epoch < config.warmup_epochs:
            # Linear warmup from 0.1x to 1.0x
            return 0.1 + 0.9 * (epoch / config.warmup_epochs)
        else:
            # Cosine annealing after warmup
            progress = (epoch - config.warmup_epochs) / (config.num_epochs - config.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Also use ReduceLROnPlateau as backup
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("\n[4/6] Training Model...")
    print(f"{'='*80}")
    print(f"🚀 TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"📱 Device:            {config.device}")
    print(f"🔢 Total Epochs:      {config.num_epochs}")
    print(f"📦 Batch Size:        {config.batch_size}")
    print(f"📈 Learning Rate:     {config.learning_rate}")
    print(f"🔥 Warmup Epochs:     {config.warmup_epochs}")
    print(f"⏸️  Early Stop Patience: {config.patience}")
    print(f"{'='*80}\n")
    
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    training_history = []
    
    import time
    training_start_time = time.time()
    
    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()
        
        # Progress bar
        progress = epoch / config.num_epochs
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\n{'='*80}")
        print(f"📅 EPOCH {epoch}/{config.num_epochs} [{bar}] {progress*100:.1f}%")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, config.device, config
        )
        
        # Validate
        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, config.device
        )
        
        # Learning rate scheduling
        scheduler.step()  # Step warmup/cosine scheduler
        plateau_scheduler.step(val_loss)  # Step plateau scheduler
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics with visual separators
        print(f"\n{'─'*80}")
        print(f"📊 EPOCH {epoch}/{config.num_epochs} RESULTS")
        print(f"{'─'*80}")
        
        # Training metrics
        print(f"🏋️  Training:")
        print(f"   Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} ({train_acc*100:.1f}%)")
        
        # Validation metrics
        print(f"\n✅ Validation:")
        print(f"   Loss:      {val_loss:.4f}")
        print(f"   Accuracy:  {val_acc:.4f} ({val_acc*100:.1f}%)")
        print(f"   Precision: {val_metrics['precision']:.4f}")
        print(f"   Recall:    {val_metrics['recall']:.4f}")
        print(f"   F1 Score:  {val_metrics['f1']:.4f}")
        
        # Confusion matrix
        if 'confusion_matrix' in val_metrics:
            cm = val_metrics['confusion_matrix']
            print(f"\n📋 Confusion Matrix:")
            print(f"              Predicted")
            print(f"              Safe  Vuln")
            print(f"   Actual Safe  {cm[0][0]:4d}  {cm[0][1]:4d}")
            print(f"          Vuln  {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        # Learning rate
        print(f"\n⚙️  Learning Rate: {current_lr:.6f}")
        
        # Epoch timing
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - training_start_time
        avg_epoch_time = total_time / epoch
        remaining_epochs = config.num_epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        
        print(f"\n⏱️  Timing:")
        print(f"   This epoch:     {epoch_time:.1f}s")
        print(f"   Total elapsed:  {total_time/60:.1f}m")
        if remaining_epochs > 0:
            print(f"   ETA:            {eta_seconds/60:.1f}m ({remaining_epochs} epochs remaining)")
        
        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_metrics': val_metrics,
            'epoch_time': epoch_time,
            'lr': current_lr
        })
        
        # Save best model (by F1 score)
        improvement = val_metrics['f1'] - best_val_f1
        
        if val_metrics['f1'] > best_val_f1:
            print(f"\n🎉 NEW BEST MODEL!")
            print(f"   Previous best F1: {best_val_f1:.4f}")
            print(f"   Current F1:       {val_metrics['f1']:.4f}")
            print(f"   Improvement:      +{improvement:.4f} ({improvement*100:.2f}%)")
            
            best_val_f1 = val_metrics['f1']
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, val_metrics, config, "best_model.pt", verbose=True
            )
            patience_counter = 0
        else:
            patience_counter += 1
            remaining_patience = config.patience - patience_counter
            
            print(f"\n⚠️  No improvement (Best F1: {best_val_f1:.4f} at epoch {best_epoch})")
            print(f"   Early stopping patience: {patience_counter}/{config.patience}")
            
            if remaining_patience > 0:
                print(f"   🔄 Will continue for {remaining_patience} more epochs...")
            else:
                print(f"   ❌ Patience exhausted!")
        
        # Early stopping check
        if patience_counter >= config.patience:
            total_time = time.time() - training_start_time
            print(f"\n{'='*80}")
            print(f"⏹️  EARLY STOPPING TRIGGERED")
            print(f"{'='*80}")
            print(f"📊 Training Statistics:")
            print(f"   • Reason:           No improvement for {config.patience} consecutive epochs")
            print(f"   • Best F1 Score:    {best_val_f1:.4f} (Epoch {best_epoch})")
            print(f"   • Current F1 Score: {val_metrics['f1']:.4f} (Epoch {epoch})")
            print(f"   • Epochs Wasted:    {patience_counter} epochs without improvement")
            print(f"   • Total Epochs:     {epoch}/{config.num_epochs} ({epoch/config.num_epochs*100:.1f}%)")
            print(f"   • Training Time:    {total_time/60:.1f} minutes")
            print(f"\n💡 Best model was saved at epoch {best_epoch}")
            print(f"   Training stopped early to prevent overfitting.")
            print(f"{'='*80}\n")
            break
    
    # Save final model
    print("\n[5/6] Saving Final Model...")
    save_checkpoint(
        model, optimizer, epoch, val_metrics, config, "final_model.pt", verbose=True
    )
    
    # Save training history
    config.log_dir.mkdir(parents=True, exist_ok=True)
    history_path = config.log_dir / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    
    # Add summary to history
    training_summary = {
        'total_epochs': len(training_history),
        'best_epoch': best_epoch,
        'best_f1': best_val_f1,
        'best_val_loss': best_val_loss,
        'early_stopped': patience_counter >= config.patience,
        'total_time_seconds': total_training_time,
        'total_time_minutes': total_training_time / 60,
        'avg_epoch_time': total_training_time / len(training_history),
        'config': config.__dict__,
        'training_history': training_history
    }
    
    with open(history_path, 'w') as f:
        json.dump(training_summary, f, indent=2, default=str)
    
    print(f"\n📝 Training history saved to {history_path}")
    
    # Print final results
    print("\n[6/6] Training Complete!")
    print("=" * 80)
    print("🏆 FINAL TRAINING RESULTS")
    print("=" * 80)
    print(f"\n📊 Performance Metrics:")
    print(f"   • Best Validation F1:    {best_val_f1:.4f} (Epoch {best_epoch})")
    print(f"   • Best Validation Loss:  {best_val_loss:.4f}")
    print(f"   • Final Validation F1:   {training_history[-1]['val_metrics']['f1']:.4f}")
    
    print(f"\n📈 Training Statistics:")
    print(f"   • Total Epochs:          {len(training_history)}/{config.num_epochs} ({len(training_history)/config.num_epochs*100:.1f}%)")
    print(f"   • Best Epoch:            {best_epoch}")
    print(f"   • Early Stopped:         {'Yes' if patience_counter >= config.patience else 'No'}")
    print(f"   • Total Time:            {total_training_time/60:.1f} minutes")
    print(f"   • Avg Time per Epoch:    {total_training_time/len(training_history):.1f} seconds")
    
    print(f"\n💾 Saved Files:")
    print(f"   • Best Model:   {config.checkpoint_dir / 'best_model.pt'}")
    print(f"   • Final Model:  {config.checkpoint_dir / 'final_model.pt'}")
    print(f"   • Training Log: {history_path}")
    
    print(f"\n{'='*80}")
    print("✅ Training pipeline completed successfully!")
    print("=" * 80)

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Create default config
    config = TrainingConfig()
    
    # Override with command line args if needed
    import argparse
    parser = argparse.ArgumentParser(description="Train Enhanced Hybrid Model")
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--epochs", type=int, default=config.num_epochs)
    parser.add_argument("--lr", type=float, default=config.learning_rate)
    parser.add_argument("--data-dir", type=str, default=str(config.data_dir))
    args = parser.parse_args()
    
    # Update config
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.data_dir = Path(args.data_dir)
    
    # Run training
    train_model(config)
