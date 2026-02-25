"""
Production-Ready Training Pipeline for Hybrid GNN+LSTM Vulnerability Scanner
=================================================================================

Design Philosophy:
- Prioritizes Recall (security-critical: false negatives are dangerous)
- Handles class imbalance via pos_weight + stratified sampling
- Multi-modal fusion optimized with OneCycleLR
- Threshold-independent evaluation (ROC-AUC)
- Production-ready: reproducible, monitored, well-logged

Target Metrics: Accuracy ≥80%, Recall ≥80%, F1 ≥80%, ROC-AUC ≥80%

Author: Senior Project - AI-based Vulnerability Scanner
Date: 2026-02-08 (Production Version)
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
    roc_auc_score, matthews_corrcoef, confusion_matrix
)
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import math

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.ml.hybrid_model import HybridVulnerabilityModel


# =============================================================================
# Focal Loss - focuses learning on hard-to-classify samples
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    
    Reduces the contribution of easy examples and focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    This is critical when the model can easily identify ~80% of samples
    but struggles with ambiguous/hard cases.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.5, 
                 label_smoothing: float = 0.0, pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output [batch, 1] (before sigmoid)
            targets: Ground truth [batch, 1] (0 or 1)
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Compute BCE with logits
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=self.pos_weight
        )
        
        # Compute focal weight
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * bce_loss
        return loss.mean()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Optimized hyperparameters for 80%+ metrics"""
    # Data paths
    data_dir: Path = Path("data/processed_graphs")
    train_file: str = "train_graphs.pkl"
    val_file: str = "val_graphs.pkl"
    test_file: str = "test_graphs.pkl"
    vocab_file: str = "vocabulary.pkl"
    
    # Model architecture (DO NOT CHANGE - matches hybrid_model.py)
    node_feature_dim: int = 832  # CodeBERT feature dimension
    hidden_dim: int = 128
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    use_gat: bool = True
    dropout: float = 0.2
    
    # Training hyperparameters (OPTIMIZED)
    batch_size: int = 32
    learning_rate: float = 3e-4  # Conservative for stable convergence
    num_epochs: int = 120
    weight_decay: float = 1e-3  # Stronger L2 regularization
    gradient_clip: float = 1.0  # Critical for LSTM stability
    
    # Loss configuration
    use_pos_weight: bool = False  # Disabled for balanced 50:50 data
    pos_weight_boost: float = 1.0  # Only used if use_pos_weight=True
    use_focal_loss: bool = True  # Focal loss for hard examples
    focal_gamma: float = 2.0  # Focus on hard-to-classify samples
    label_smoothing: float = 0.05  # Prevent overconfident predictions
    
    # Early stopping
    patience: int = 20  # Stop if val F1 doesn't improve for 20 epochs
    min_delta: float = 0.001  # Minimum improvement to reset patience
    
    # Learning rate schedule
    warmup_pct: float = 0.15  # 15% of training is warm-up (longer warmup)
    div_factor: float = 25.0  # Start LR = max_lr / 25
    final_div_factor: float = 1000.0  # End LR = max_lr / 1000
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output paths
    models_dir: Path = Path("models")
    log_dir: Path = Path("training/logs")
    
    # Data loading
    num_workers: int = 0  # Set to 0 for Windows compatibility
    pin_memory: bool = True
    
    # Reproducibility
    seed: int = 42

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ProcessedSample:
    """Matches enhanced_dataset_pipeline.py structure"""
    code: str
    label: int
    language: str
    graph_data: Data
    vulnerability_type: str
    source: str
    metadata: dict
    token_ids: Optional[torch.Tensor] = None
    code_metrics: Optional[np.ndarray] = None

# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed: int):
    """Ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_processed_dataset(pickle_path: Path) -> List[ProcessedSample]:
    """Load pre-processed dataset"""
    if not pickle_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        samples = pickle.load(f)
    
    print(f"✅ Loaded {len(samples):,} samples from {pickle_path.name}")
    return samples

def linear_probe_diagnostic(train_samples: List[ProcessedSample], val_samples: List[ProcessedSample]):
    """
    Quick linear probe to check if features contain ANY learnable signal.
    Trains a logistic regression on mean-pooled graph features, token stats, and metrics.
    This tells us the theoretical upper bound BEFORE model training.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    print(f"\n{'='*80}")
    print("FEATURE DISCRIMINABILITY DIAGNOSTIC (Linear Probe)")
    print(f"{'='*80}")
    
    # Extract features for linear probe
    def extract_probe_features(samples, max_n=2000):
        features = []
        labels = []
        for s in samples[:max_n]:
            g = s.graph_data
            # Mean-pooled graph features (832-dim)
            graph_mean = g.x.mean(dim=0).numpy()
            # Token ID statistics (basic stats instead of raw tokens)
            if hasattr(g, 'token_ids') and g.token_ids is not None:
                tids = g.token_ids.float()
                token_feats = np.array([
                    tids.mean().item(),
                    tids.std().item(),
                    (tids > 0).float().mean().item(),  # Non-zero ratio
                    tids.max().item() / 10000,  # Normalized max
                    len(set(tids.flatten().tolist())),  # Unique tokens
                ])
            else:
                token_feats = np.zeros(5)
            # Code metrics (20-dim)
            if hasattr(s, 'code_metrics') and s.code_metrics is not None:
                metrics = s.code_metrics.copy()
            else:
                metrics = np.zeros(20)
            
            feat = np.concatenate([graph_mean, token_feats, metrics])
            features.append(feat)
            labels.append(s.label)
        
        return np.array(features), np.array(labels)
    
    X_train, y_train = extract_probe_features(train_samples)
    X_val, y_val = extract_probe_features(val_samples)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    # Test different feature subsets
    tests = [
        ("Full features (832+5+20)", slice(None)),
        ("CodeBERT only (first 768)", slice(0, 768)),
        ("Base graph feats (64)", slice(768, 832)),
        ("Token stats (5)", slice(832, 837)),
        ("Code metrics (20)", slice(837, 857)),
        ("CodeBERT + Metrics", np.r_[0:768, 837:857]),
    ]
    
    for name, idx in tests:
        try:
            clf = LogisticRegression(max_iter=500, C=1.0)
            clf.fit(X_train_s[:, idx], y_train)
            
            train_acc = clf.score(X_train_s[:, idx], y_train)
            val_acc = clf.score(X_val_s[:, idx], y_val)
            val_pred = clf.predict_proba(X_val_s[:, idx])[:, 1]
            val_auc = roc_auc_score(y_val, val_pred) if len(np.unique(y_val)) > 1 else 0
            
            status = "✅" if val_auc >= 0.65 else "⚠️" if val_auc >= 0.55 else "❌"
            print(f"  {status} {name:40s} | Train Acc={train_acc:.3f} | Val Acc={val_acc:.3f} | Val AUC={val_auc:.3f}")
        except Exception as e:
            print(f"  ❌ {name:40s} | Error: {e}")
    
    print(f"{'='*80}\n")

def samples_to_pyg_data(samples: List[ProcessedSample]) -> List[Data]:
    """Convert ProcessedSample objects to PyTorch Geometric Data objects"""
    graphs = []
    
    for sample in samples:
        try:
            graph = sample.graph_data
            
            # Add label as float tensor [1]
            graph.y = torch.tensor([float(sample.label)], dtype=torch.float)
            
            # Ensure token_ids exists [1, seq_len]
            if not hasattr(graph, 'token_ids') or graph.token_ids is None:
                graph.token_ids = torch.zeros((1, 128), dtype=torch.long)
            if graph.token_ids.dim() == 1:
                graph.token_ids = graph.token_ids.unsqueeze(0)
            
            # Add code metrics [1, 20]
            if hasattr(sample, 'code_metrics') and sample.code_metrics is not None:
                graph.code_metrics = torch.from_numpy(sample.code_metrics).float().unsqueeze(0)
            else:
                graph.code_metrics = torch.zeros((1, 20), dtype=torch.float)
            
            graphs.append(graph)
            
        except Exception as e:
            print(f"⚠️  Error processing sample: {e}")
            continue
    
    return graphs

def analyze_class_distribution(samples: List[ProcessedSample], split_name: str) -> dict:
    """Analyze dataset statistics"""
    total = len(samples)
    vulnerable = sum(1 for s in samples if s.label == 1)
    safe = total - vulnerable
    
    stats = {
        'total': total,
        'vulnerable': vulnerable,
        'safe': safe,
        'vuln_pct': 100 * vulnerable / total if total > 0 else 0,
        'safe_pct': 100 * safe / total if total > 0 else 0,
        'imbalance_ratio': safe / vulnerable if vulnerable > 0 else float('inf')
    }
    
    print(f"\n{'='*70}")
    print(f"📊 {split_name.upper()} DATASET ANALYSIS")
    print(f"{'='*70}")
    print(f"📦 Total Samples:     {stats['total']:,}")
    print(f"🔴 Vulnerable:        {stats['vulnerable']:,} ({stats['vuln_pct']:.1f}%)")
    print(f"🟢 Safe:              {stats['safe']:,} ({stats['safe_pct']:.1f}%)")
    print(f"⚖️  Imbalance Ratio:   {stats['imbalance_ratio']:.2f}:1 (safe:vulnerable)")
    
    if abs(stats['imbalance_ratio'] - 1.0) < 0.2:
        print(f"✅ Dataset is BALANCED")
    else:
        print(f"⚠️  Dataset is IMBALANCED - pos_weight will be applied")
    
    return stats

# =============================================================================
# Training & Validation Functions
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: str,
    gradient_clip: float
) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Extract inputs
        token_ids = batch.token_ids
        code_metrics = batch.code_metrics if hasattr(batch, 'code_metrics') else None
        
        # Forward pass
        predictions, _, _, _ = model(batch, token_ids, code_metrics)
        targets = batch.y.unsqueeze(1)
        
        # Compute loss
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Check for NaN gradients (critical for debugging)
        has_nan = False
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan = True
                    break
        
        if has_nan:
            print("⚠️  NaN gradient detected, skipping batch")
            optimizer.zero_grad()
            continue
        
        # Gradient clipping (essential for LSTM)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        # Update weights
        optimizer.step()
        scheduler.step()  # OneCycleLR steps per batch
        
        # Compute metrics
        total_loss += loss.item()
        probs = torch.sigmoid(predictions)
        preds = (probs > 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.3f}'})
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, dict]:
    """Validate and compute comprehensive metrics"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            batch = batch.to(device)
            token_ids = batch.token_ids
            code_metrics = batch.code_metrics if hasattr(batch, 'code_metrics') else None
            
            # Forward pass
            predictions, _, _, _ = model(batch, token_ids, code_metrics)
            targets = batch.y.unsqueeze(1)
            
            # Loss
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            
            # Collect predictions and labels
            probs = torch.sigmoid(predictions)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Convert to numpy
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    
    # Compute metrics
    avg_loss = total_loss / len(loader)
    
    # Handle edge cases (e.g., all one class in batch)
    unique_labels = len(np.unique(all_labels))
    
    # Diagnostic: Check prediction distribution
    pred_vulnerable_count = np.sum(all_preds == 1)
    pred_safe_count = np.sum(all_preds == 0)
    actual_vulnerable_count = np.sum(all_labels == 1)
    actual_safe_count = np.sum(all_labels == 0)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probs) if unique_labels > 1 else 0.0,
        'mcc': matthews_corrcoef(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        # Diagnostic info
        'pred_distribution': {
            'vulnerable': int(pred_vulnerable_count),
            'safe': int(pred_safe_count),
            'actual_vulnerable': int(actual_vulnerable_count),
            'actual_safe': int(actual_safe_count)
        }
    }
    
    return avg_loss, metrics

# =============================================================================
# Model Saving
# =============================================================================

def save_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: TrainingConfig,
    filename: str = "best_model.pt"
):
    """Save model checkpoint"""
    config.models_dir.mkdir(parents=True, exist_ok=True)
    filepath = config.models_dir / filename
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': vars(config),
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, filepath)
    print(f"\n💾 MODEL SAVED: {filepath}")
    print(f"   📊 F1={metrics['f1']:.4f} | Recall={metrics['recall']:.4f} | AUC={metrics['roc_auc']:.4f}")

# =============================================================================
# Main Training Pipeline
# =============================================================================

def train_vulnerability_model(config: TrainingConfig):
    """
    Main training function - optimized for 80%+ metrics
    """
    print("="*80)
    print("🚀 PRODUCTION TRAINING PIPELINE - HYBRID VULNERABILITY DETECTOR")
    print("="*80)
    print(f"Target: Accuracy ≥80%, Recall ≥80%, F1 ≥80%, ROC-AUC ≥80%")
    print(f"Device: {config.device.upper()}")
    print("="*80)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    print(f"🔒 Random seed: {config.seed}")
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATASETS")
    print("="*80)
    
    train_samples = load_processed_dataset(config.data_dir / config.train_file)
    val_samples = load_processed_dataset(config.data_dir / config.val_file)
    
    train_stats = analyze_class_distribution(train_samples, "Training")
    val_stats = analyze_class_distribution(val_samples, "Validation")
    
    # Quick diagnostic: check if features contain signal
    linear_probe_diagnostic(train_samples, val_samples)
    
    # =========================================================================
    # STEP 2: Prepare Data Loaders
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: PREPARING DATA LOADERS")
    print("="*80)
    
    train_graphs = samples_to_pyg_data(train_samples)
    val_graphs = samples_to_pyg_data(val_samples)
    
    print(f"✅ Converted to PyG format: {len(train_graphs)} train, {len(val_graphs)} val")
    
    train_loader = DataLoader(
        train_graphs,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_graphs,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    print(f"📦 Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}")
    
    # =========================================================================
    # STEP 3: Initialize Model
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: INITIALIZING MODEL")
    print("="*80)
    
    # Load vocabulary
    vocab_path = config.data_dir / config.vocab_file
    if vocab_path.exists():
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        vocab_size = len(vocab_data['vocab'])
        print(f"✅ Vocabulary loaded: {vocab_size:,} tokens")
    else:
        vocab_size = 50000
        print(f"⚠️  No vocabulary file, using default: {vocab_size:,} tokens")
    
    # Initialize model
    model = HybridVulnerabilityModel(
        vocab_size=vocab_size,
        node_feature_dim=config.node_feature_dim,
        gnn_hidden_dim=config.hidden_dim,
        gnn_output_dim=64,
        lstm_embedding_dim=config.hidden_dim,
        lstm_hidden_dim=config.lstm_hidden_dim,
        lstm_output_dim=64,
        metrics_input_dim=20,
        metrics_output_dim=128,
        fusion_hidden_dim=config.hidden_dim,
        use_gat=config.use_gat,
        use_metrics=True,
        dropout=config.dropout
    ).to(config.device)
    
    # Weight initialization (critical for convergence)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
    
    model.apply(init_weights)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model initialized: {total_params:,} trainable parameters")
    
    # =========================================================================
    # STEP 4: Setup Loss, Optimizer, Scheduler
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: CONFIGURING TRAINING COMPONENTS")
    print("="*80)
    
    # Loss function with class weighting
    if config.use_focal_loss:
        pos_weight_tensor = None
        if config.use_pos_weight and abs(train_stats['imbalance_ratio'] - 1.0) > 0.2:
            pos_weight_value = train_stats['imbalance_ratio'] * config.pos_weight_boost
            pos_weight_tensor = torch.tensor([pos_weight_value], device=config.device)
        
        criterion = FocalLoss(
            gamma=config.focal_gamma,
            alpha=0.5,
            label_smoothing=config.label_smoothing,
            pos_weight=pos_weight_tensor
        )
        print(f"📊 Loss: FocalLoss (gamma={config.focal_gamma}, label_smoothing={config.label_smoothing})")
    elif config.use_pos_weight and abs(train_stats['imbalance_ratio'] - 1.0) > 0.2:
        pos_weight_value = train_stats['imbalance_ratio'] * config.pos_weight_boost
        pos_weight = torch.tensor([pos_weight_value], device=config.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"📊 Loss: BCEWithLogitsLoss with pos_weight={pos_weight_value:.3f}")
        print(f"   (Base ratio: {train_stats['imbalance_ratio']:.2f}, Boost: {config.pos_weight_boost}x)")
        print(f"   ⚠️  WARNING: Using pos_weight on balanced data may cause bias!")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"📊 Loss: BCEWithLogitsLoss (no pos_weight)")
        if abs(train_stats['imbalance_ratio'] - 1.0) < 0.2:
            print(f"   ✅ Data is balanced ({train_stats['imbalance_ratio']:.2f}:1) - no weighting needed")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    print(f"⚙️  Optimizer: AdamW (lr={config.learning_rate}, weight_decay={config.weight_decay})")
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.num_epochs,
        pct_start=config.warmup_pct,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor
    )
    print(f"📈 Scheduler: OneCycleLR")
    print(f"   Warmup: {int(config.num_epochs * config.warmup_pct)} epochs")
    print(f"   LR range: {config.learning_rate/config.div_factor:.2e} → {config.learning_rate:.2e} → {config.learning_rate/config.final_div_factor:.2e}")
    
    # =========================================================================
    # STEP 5: Training Loop
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: TRAINING MODEL")
    print("="*80)
    
    best_f1 = 0.0
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    training_history = []
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            config.device, config.gradient_clip
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, config.device)
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n{'='*80}")
        print(f"📊 EPOCH {epoch}/{config.num_epochs}")
        print(f"{'='*80}")
        print(f"🔵 Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
        print(f"🟢 Val Loss:   {val_loss:.4f} | Acc: {val_metrics['accuracy']*100:.2f}%")
        print(f"📈 Validation Metrics:")
        print(f"   • Precision:  {val_metrics['precision']:.4f}")
        print(f"   • Recall:     {val_metrics['recall']:.4f} {'✅' if val_metrics['recall'] >= 0.80 else '❌'}")
        print(f"   • F1-Score:   {val_metrics['f1']:.4f} {'✅' if val_metrics['f1'] >= 0.80 else '❌'}")
        print(f"   • ROC-AUC:    {val_metrics['roc_auc']:.4f} {'✅' if val_metrics['roc_auc'] >= 0.80 else '❌'}")
        print(f"   • MCC:        {val_metrics['mcc']:.4f}")
        
        # Show prediction distribution (diagnostic)
        pred_dist = val_metrics.get('pred_distribution', {})
        if pred_dist:
            print(f"🔍 Prediction Distribution:")
            print(f"   • Predicted Vulnerable: {pred_dist['vulnerable']} (Actual: {pred_dist['actual_vulnerable']})")
            print(f"   • Predicted Safe:       {pred_dist['safe']} (Actual: {pred_dist['actual_safe']})")
            bias_ratio = pred_dist['vulnerable'] / max(pred_dist['actual_vulnerable'], 1)
            if bias_ratio > 1.5:
                print(f"   ⚠️  WARNING: Model is over-predicting vulnerable by {bias_ratio:.1f}x")
            elif bias_ratio > 0 and bias_ratio < 0.5:
                print(f"   ⚠️  WARNING: Model is under-predicting vulnerable by {1/bias_ratio:.1f}x")
            elif bias_ratio == 0:
                print(f"   ⚠️  WARNING: Model predicting ZERO vulnerable samples!")
        
        print(f"⚙️  Learning Rate: {current_lr:.2e}")
        
        # Save training history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_roc_auc': val_metrics['roc_auc'],
            'val_mcc': val_metrics['mcc'],
            'learning_rate': current_lr
        })
        
        # Check for improvement (based on ROC-AUC - threshold independent)
        current_score = val_metrics['roc_auc']
        if current_score > best_auc + config.min_delta:
            improvement = current_score - best_auc
            print(f"\n🎉 NEW BEST ROC-AUC! ({best_auc:.4f} → {current_score:.4f}, +{improvement:.4f})")
            best_auc = current_score
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            save_model(model, optimizer, epoch, val_metrics, config, "best_model.pt")
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{config.patience}")
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"\n⏹️  EARLY STOPPING at epoch {epoch}")
            print(f"   Best ROC-AUC: {best_auc:.4f} | F1: {best_f1:.4f} at epoch {best_epoch}")
            break
    
    # =========================================================================
    # STEP 6: Save Results
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 6: SAVING RESULTS")
    print("="*80)
    
    # Save final model
    save_model(model, optimizer, epoch, val_metrics, config, "final_model.pt")
    
    # Save training history
    config.log_dir.mkdir(parents=True, exist_ok=True)
    history_file = config.log_dir / f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"📝 Training history saved: {history_file}")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"🏆 Best Validation Metrics (Epoch {best_epoch}):")
    best_metrics = training_history[best_epoch - 1]
    print(f"   • ROC-AUC:    {best_metrics['val_roc_auc']:.4f} {'✅' if best_metrics['val_roc_auc'] >= 0.80 else '❌'}")
    print(f"   • F1-Score:   {best_metrics['val_f1']:.4f} {'✅' if best_metrics['val_f1'] >= 0.80 else '❌'}")
    print(f"   • Recall:     {best_metrics['val_recall']:.4f} {'✅' if best_metrics['val_recall'] >= 0.80 else '❌'}")
    print(f"   • Precision:  {best_metrics['val_precision']:.4f} {'✅' if best_metrics['val_precision'] >= 0.80 else '❌'}")
    print(f"   • Accuracy:   {best_metrics['val_acc']:.4f} {'✅' if best_metrics['val_acc'] >= 0.80 else '❌'}")
    print(f"   • MCC:        {best_metrics['val_mcc']:.4f}")
    print("="*80)
    
    return model, training_history

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    config = TrainingConfig()
    model, history = train_vulnerability_model(config)