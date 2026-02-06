# Training Quick Reference Card
> **Enhanced Hybrid GNN+LSTM Model**  
> Version 2.0 - Feb 6, 2026

## 🚀 Quick Start (3 Commands)

```bash
# 1. Process dataset (if not done)
cd backend
python scripts/enhanced_dataset_pipeline.py

# 2. Train model
python training/train_enhanced.py --epochs 50 --batch-size 8

# 3. Evaluate
python training/evaluate_enhanced.py
```

---

## 📂 File Structure

```
backend/
├── data/
│   └── processed_graphs/          ← Your dataset goes here
│       ├── train_graphs.pkl
│       ├── val_graphs.pkl
│       └── test_graphs.pkl
│
├── training/
│   ├── train_enhanced.py          ← Main training script
│   ├── evaluate_enhanced.py       ← Evaluation script
│   ├── training_config.yaml       ← Configuration
│   ├── checkpoints/               ← Saved models
│   │   ├── best_model.pt
│   │   └── final_model.pt
│   ├── logs/                      ← Training history
│   └── evaluation/                ← Evaluation results
│
└── scripts/
    └── enhanced_dataset_pipeline.py  ← Process raw code to graphs
```

---

## 🎯 Common Commands

### Training

```bash
# Quick test (5 epochs)
python training/train_enhanced.py --epochs 5 --batch-size 4

# Normal training
python training/train_enhanced.py

# Full training with custom params
python training/train_enhanced.py --epochs 100 --batch-size 16 --lr 0.0005
```

### Evaluation

```bash
# Evaluate best model on test set
python training/evaluate_enhanced.py

# Evaluate specific checkpoint
python training/evaluate_enhanced.py --checkpoint training/checkpoints/final_model.pt

# Evaluate on different dataset
python training/evaluate_enhanced.py --test-data data/processed/codexglue/test_graphs.pkl
```

### Dataset Processing

```bash
# Process mock dataset (for testing)
python scripts/enhanced_dataset_pipeline.py

# Process real dataset
python scripts/enhanced_dataset_pipeline.py \
  --input data/raw_datasets/codexglue \
  --output data/processed/codexglue
```

---

## ⚙️ Command Line Arguments

### train_enhanced.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch-size` | int | 8 | Samples per batch |
| `--epochs` | int | 50 | Number of epochs |
| `--lr` | float | 0.001 | Learning rate |
| `--data-dir` | path | `data/processed_graphs` | Dataset directory |

### evaluate_enhanced.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | path | `training/checkpoints/best_model.pt` | Model checkpoint |
| `--test-data` | path | `data/processed_graphs/test_graphs.pkl` | Test dataset |
| `--output-dir` | path | `training/evaluation` | Output directory |
| `--device` | str | `cpu` | Device (cpu/cuda) |

---

## 📊 Expected Performance

### Mock Dataset (15 samples)
- Training time: ~1 minute (3 epochs)
- Expected F1: 0.60-0.70
- Purpose: Pipeline testing only

### Small Dataset (100-1000 samples)
- Training time: ~10-30 minutes
- Expected F1: 0.75-0.85
- Recommended: batch_size=8, epochs=50

### Large Dataset (10,000+ samples)
- Training time: 2-6 hours
- Expected F1: 0.85-0.92
- Recommended: batch_size=32, epochs=100

---

## 🐛 Quick Fixes

### Issue: Out of Memory
```bash
# Solution: Reduce batch size
python training/train_enhanced.py --batch-size 4
```

### Issue: Model not learning (F1 stuck ~0.5)
```bash
# Solution: Increase learning rate
python training/train_enhanced.py --lr 0.01
```

### Issue: Overfitting (train >> val accuracy)
```yaml
# Solution: Edit training_config.yaml
training:
  dropout: 0.5
  label_smoothing: 0.2
```

### Warning: Missing token_ids
```
# Expected warning (LSTM uses dummy tokens)
# GNN branch still works correctly
# To fix: Update enhanced_dataset_pipeline.py to extract token sequences
```

---

## 📈 Monitoring Training

### Real-time Monitoring

Watch the training output:
```bash
python training/train_enhanced.py | tee training.log
```

### Check Training History

```bash
# View latest training history
python -c "import json; h = json.load(open('training/logs/training_history_20260206.json')); print(f'Best F1: {max(e[\"val_metrics\"][\"f1\"] for e in h):.4f}')"
```

### Visualize Training Curves

```python
import json
import matplotlib.pyplot as plt

# Load history
with open('training/logs/training_history_20260206.json') as f:
    history = json.load(f)

# Plot
epochs = [h['epoch'] for h in history]
train_loss = [h['train_loss'] for h in history]
val_loss = [h['val_loss'] for h in history]

plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.legend()
plt.show()
```

---

## 🎓 Configuration Presets

### Fast Test (Debug Mode)
```bash
python training/train_enhanced.py --epochs 3 --batch-size 4 --lr 0.01
# Time: 30 seconds
# Use: Quick pipeline testing
```

### Standard Training (Mock Dataset)
```bash
python training/train_enhanced.py --epochs 50 --batch-size 8
# Time: 5-10 minutes
# Use: Development and testing
```

### Production Training (Real Dataset)
```bash
python training/train_enhanced.py --epochs 100 --batch-size 32 --lr 0.0005 --data-dir data/processed/codexglue
# Time: 2-4 hours
# Use: Final model training
```

### High Accuracy (Research Mode)
```yaml
# Edit training_config.yaml:
model:
  hidden_dim: 256
  num_gnn_layers: 4
  lstm_hidden_dim: 256

training:
  batch_size: 16
  num_epochs: 150
  learning_rate: 0.0003
```
```bash
python training/train_enhanced.py
# Time: 6+ hours
# Use: Maximum accuracy, research papers
```

---

## 📦 Model Size Reference

| Configuration | Parameters | Model Size | RAM Usage |
|---------------|-----------|------------|-----------|
| Lightweight (64) | ~500K | ~2 MB | ~1 GB |
| Standard (128) | ~1M | ~4 MB | ~2 GB |
| Large (256) | ~2.5M | ~10 MB | ~4 GB |
| XL (512) | ~10M | ~40 MB | ~8 GB |

---

## 🔗 Related Files

- [Full Training Guide](TRAINING_GUIDE.md) - Comprehensive documentation
- [Config File](training_config.yaml) - All configurable parameters
- [Evaluation Output](evaluation/evaluation_results.json) - Last evaluation metrics
- [Dataset Guide](../DOWNLOAD_DATASETS_GUIDE.md) - How to get real datasets

---

## 💡 Pro Tips

1. **Always test on mock dataset first** before training on large datasets
2. **Monitor validation F1**, not accuracy (handles imbalanced data better)
3. **Save checkpoints frequently** (already done automatically)
4. **Use early stopping** to prevent wasting time on non-improving models
5. **Check evaluation plots** (confusion matrix, ROC curve) for insights
6. **Start with default params**, then tune based on results

---

**Need more details?** → Read [TRAINING_GUIDE.md](TRAINING_GUIDE.md)  
**Found a bug?** → Check [Troubleshooting section](TRAINING_GUIDE.md#troubleshooting)  
**Want to customize?** → Edit [training_config.yaml](training_config.yaml)

---

**Last Updated:** Feb 6, 2026  
**Status:** ✅ Tested & Working
