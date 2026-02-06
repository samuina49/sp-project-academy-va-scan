"""
Model Training Script for Hybrid GNN+LSTM Vulnerability Scanner
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from pathlib import Path
import json
import sys
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.ml.hybrid_model import HybridVulnerabilityModel
from app.ml.feature_extraction import FeatureExtractor

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def load_dataset(json_path: Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Support both list directly or {"samples": [...]} format
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get('samples', [])
    else:
        print(f"Warning: Unknown dataset format in {json_path}")
        return []

def prepare_data(train_path, val_path, extractor: FeatureExtractor):
    """
    Load data, build vocabulary, and convert to PyG Data objects
    """
    print("Loading datasets...")
    train_samples = load_dataset(train_path)
    val_samples = load_dataset(val_path)
    
    print(f"  Loaded {len(train_samples)} training samples")
    print(f"  Loaded {len(val_samples)} validation samples")
    
    # Count vulnerability types - support both 'label' and 'vuln_type' formats
    train_vuln = sum(1 for s in train_samples if s.get('label', 0) == 1 or s.get('vuln_type', 'none') != 'none')
    train_safe = len(train_samples) - train_vuln
    val_vuln = sum(1 for s in val_samples if s.get('label', 0) == 1 or s.get('vuln_type', 'none') != 'none')
    val_safe = len(val_samples) - val_vuln
    print(f"  Train: {train_vuln} vulnerable, {train_safe} safe")
    print(f"  Val: {val_vuln} vulnerable, {val_safe} safe")
    
    # 1. Build Vocabulary from Train Data
    print("\nBuilding vocabulary...")
    # Extract (code, language) tuples - use all training data for better vocab
    vocab_samples = [
        (s.get('cleaned_code') or s.get('code', ''), s.get('language', 'python'))
        for s in train_samples
        if (s.get('cleaned_code') or s.get('code'))
    ]  # Use all data
    extractor.build_vocabulary(vocab_samples)
    print(f"Vocab size: {extractor.get_vocab_size()}")

    def sample_is_vulnerable(sample: dict) -> float:
        """Derive binary vulnerable label from mixed dataset schemas."""
        # Priority 1: Check 'label' field (0/1 format from quick_prepare_dataset)
        if 'label' in sample:
            label = sample['label']
            if isinstance(label, (int, float)):
                return float(label)
            elif isinstance(label, str):
                return 1.0 if label == '1' else 0.0
        
        # Priority 2: Check 'vuln_type' field
        vuln_type = sample.get('vuln_type')
        if isinstance(vuln_type, str) and vuln_type.strip():
            return 0.0 if vuln_type.lower() == 'none' else 1.0

        # Priority 3: Check 'vulnerabilities' list
        vulns = sample.get('vulnerabilities')
        if isinstance(vulns, list):
            return 1.0 if len(vulns) > 0 else 0.0

        # Legacy boolean flag (rare)
        if sample.get('vulnerable') is True:
            return 1.0

        return 0.0
    
    def process_samples(samples, desc):
        graphs = []
        for s in tqdm(samples, desc=desc):
            try:
                code = s.get('cleaned_code') or s.get('code')
                lang = s.get('language', 'python')
                if not code or not isinstance(code, str):
                    continue
                
                # Structural Features (GNN)
                graph = extractor.code_to_graph(code, language=lang)
                if not graph: continue
                
                # Sequential Features (LSTM)
                seq = extractor.code_to_sequence(code, language=lang)
                if not seq or seq.token_ids is None: continue
                
                # Attach LSTM features to graph for batching
                # Ensure token_ids is [1, seq_len] so batching creates [batch_size, seq_len]
                graph.token_ids = seq.token_ids.unsqueeze(0)
                
                # Target Label
                is_vuln = sample_is_vulnerable(s)
                graph.y = torch.tensor([is_vuln], dtype=torch.float)
                
                graphs.append(graph)
            except Exception as e:
                # print(f"Error: {e}")
                continue
        return graphs

    print("Processing Training Set...")
    train_graphs = process_samples(train_samples, "Train")
    
    print("Processing Validation Set...")
    val_graphs = process_samples(val_samples, "Val")
    
    return train_graphs, val_graphs

def train_epoch(model, loader, optimizer, criterion, device, label_smoothing=0.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Determine token_ids shape
        # PyG batches custom attributes by concatenation.
        # If we added [1, seq_len], batch.token_ids is [batch_size, seq_len]
        token_ids = batch.token_ids
        
        # Forward pass
        predictions, _, _ = model(batch, token_ids)
        
        # ✅ Apply label smoothing to prevent overconfidence
        targets = batch.y.unsqueeze(1)
        if label_smoothing > 0:
            targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
        
        loss = criterion(predictions, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        
        total_loss += loss.item()
        predicted_labels = (torch.sigmoid(predictions) > 0.5).float()
        correct += (predicted_labels == batch.y.unsqueeze(1)).sum().item()
        total += batch.y.size(0)
        
    return total_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            batch = batch.to(device)
            token_ids = batch.token_ids
            
            predictions, _, _ = model(batch, token_ids)
            loss = criterion(predictions, batch.y.unsqueeze(1))
            
            total_loss += loss.item()
            probs = torch.sigmoid(predictions)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.unsqueeze(1).cpu().numpy())
            
    # Calculate metrics
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
            
    return total_loss / len(loader), acc, precision, recall, f1

if __name__ == "__main__":
    # ✅ IMPROVED Settings to Prevent Overfitting
    EPOCHS = 100  # Max epochs
    EARLY_STOP_PATIENCE = 10  # ⬆️ Increased patience (was 5)
    BATCH_SIZE = 32  # Balanced batch size
    LR = 0.0001  # ⬇️ Reduced learning rate (was 0.0005)
    WEIGHT_DECAY = 0.01  # ⬆️ Stronger regularization (was 0.001)
    MAX_SEQ_LEN = 256  # Longer sequences for more context
    GRADIENT_CLIP = 0.5  # ⬇️ Stricter gradient clipping (was 1.0)
    LABEL_SMOOTHING = 0.1  # ✅ Label smoothing to prevent overconfidence
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    # Robust path handling relative to this script
    script_dir = Path(__file__).parent
    backend_dir = script_dir.parent
    
    # PRIMARY: Use new training data from quick_prepare_dataset.py
    data_dir = backend_dir / "data" / "training"
    train_path = data_dir / "train_split.json"
    val_path = data_dir / "val_split.json"
    
    # FALLBACK: Use old processed data if new data doesn't exist
    if not train_path.exists():
        print("⚠️ New training data not found, trying fallback paths...")
        data_dir = backend_dir / "data" / "processed"
        train_path = data_dir / "merged_train.json"
        val_path = data_dir / "merged_val.json"
    
    if not train_path.exists():
        train_path = data_dir / "ds_train.json"
        val_path = data_dir / "ds_val.json"
    
    if not train_path.exists():
        train_path = data_dir / "final_train.json"
        val_path = data_dir / "final_val.json"
    
    if not train_path.exists():
        print(f"Error: {train_path} not found!")
        sys.exit(1)
        
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")
    
    set_seed(42)  # Ensure reproducibility
    
    # Initialize Extractor
    extractor = FeatureExtractor(max_seq_length=MAX_SEQ_LEN)
    
    # Prepare Data
    train_graphs, val_graphs = prepare_data(train_path, val_path, extractor)
    
    if not train_graphs:
        print("Error: No graphs generated. Check data or feature extractor.")
        sys.exit(1)
        
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)
    
    # Initialize Model with STRONGER REGULARIZATION
    # Note: feature dims must match what FeatureExtractor produces
    # AST graph node feature dim defaults to 64 in FeatureExtractor
    vocab_size = extractor.get_vocab_size()
    model = HybridVulnerabilityModel(
        vocab_size=vocab_size,  # Don't add +1, vocab already includes PAD/UNK
        node_feature_dim=64,
        lstm_embedding_dim=256,  # Increased embedding dimension
        dropout=0.4  # ⬆️ STRONGER dropout (was 0.2) to prevent overfitting
    ).to(device)
    
    # Calculate class weights for imbalanced data
    vuln_count = sum(1 for g in train_graphs if g.y.item() > 0.5)
    safe_count = len(train_graphs) - vuln_count
    
    # pos_weight should boost the MINORITY class
    # If vulnerable is majority (vuln > safe), don't use pos_weight or use 1.0
    # If safe is majority (safe > vuln), use pos_weight = safe/vuln to boost vulnerable
    if safe_count > vuln_count and vuln_count > 0:
        pos_weight = torch.tensor([safe_count / vuln_count]).to(device)
        print(f"Using pos_weight={pos_weight.item():.4f} to boost minority (vulnerable) class")
    else:
        pos_weight = None  # Don't weight - vulnerable is already majority
        print(f"No pos_weight needed - vulnerable is majority ({vuln_count}/{vuln_count+safe_count})")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # None means equal weight
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3  # Reduce LR after 3 epochs without improvement
    )
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Val graphs: {len(val_graphs)}")
    print(f"Class distribution - Vulnerable: {vuln_count}, Safe: {safe_count}")
    if pos_weight is not None:
        print(f"Positive weight: {pos_weight.item():.4f}")
    else:
        print("Positive weight: None (equal weighting)")
    print(f"Batch size: {BATCH_SIZE}, LR: {LR}, Max epochs: {EPOCHS}")
    print("="*60)
    
    best_f1 = 0.0  # Use F1 for model selection (better for imbalanced data)
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, label_smoothing=LABEL_SMOOTHING)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")
        
        # Update learning rate based on validation F1
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Use F1 for model selection (better for imbalanced data)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            patience_counter = 0
            
            # Save to: backend/training/models/
            output_model_dir = script_dir / "models"
            output_model_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(model.state_dict(), output_model_dir / "hybrid_model_best.pth")
            
            # Save Vocabulary
            with open(output_model_dir / "vocab.json", 'w') as f:
                json.dump(extractor.vocab, f)
            
            # Save training info
            training_info = {
                'best_f1': best_f1,
                'best_acc': best_acc,
                'epoch': epoch + 1,
                'vuln_count': vuln_count,
                'safe_count': safe_count,
                'vocab_size': vocab_size
            }
            with open(output_model_dir / "training_info.json", 'w') as f:
                json.dump(training_info, f, indent=2)
            
            print(f"✅ Best model saved! (F1: {best_f1:.4f}, Acc: {best_acc:.4f})")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter}/{EARLY_STOP_PATIENCE} epochs (LR: {current_lr:.6f})")
            
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            break
            
    print(f"\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Model saved to: {output_model_dir}")
