# 📊 ตัวอย่าง Output จากการเทรนที่ปรับปรุงแล้ว

## 🚀 ตอนเริ่มต้น Training

```
================================================================================
📊 TRAINING DATASET STATISTICS
================================================================================
📦 Total samples:     1,234
🔴 Vulnerable:        654 (53.0%)
🟢 Safe:              580 (47.0%)
⚖️  Class Balance:     Balanced

💻 Languages:
   • Python      :  456 (37.0%)
   • JavaScript  :  345 (28.0%)
   • Java        :  234 (19.0%)
   • PHP         :  199 (16.0%)

🔐 Vulnerability Types (Top 5):
   • SQL Injection        :  123 (18.8% of vulnerable)
   • XSS                  :  98  (15.0% of vulnerable)
   • Command Injection    :  87  (13.3% of vulnerable)
   • Path Traversal       :  76  (11.6% of vulnerable)
   • Code Injection       :  65  (9.9% of vulnerable)
================================================================================

[3.5/6] Calculating Class Weights...
Training set distribution:
  Safe samples: 580 (47.0%)
  Vulnerable samples: 654 (53.0%)
✓ Using pos_weight=0.8868 to boost vulnerable class

================================================================================
🚀 TRAINING CONFIGURATION
================================================================================
📱 Device:            cuda
🔢 Total Epochs:      50
📦 Batch Size:        8
📈 Learning Rate:     0.001
🔥 Warmup Epochs:     5
⏸️  Early Stop Patience: 10
================================================================================
```

## 🏋️ ระหว่าง Training (แต่ละ Epoch)

```
================================================================================
📅 EPOCH 1/50 [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 2.0%
================================================================================

────────────────────────────────────────────────────────────────────────────────
📊 EPOCH 1/50 RESULTS
────────────────────────────────────────────────────────────────────────────────
🏋️  Training:
   Loss: 0.6543 | Accuracy: 0.6234 (62.3%)

✅ Validation:
   Loss:      0.6234
   Accuracy:  0.6456 (64.6%)
   Precision: 0.6234
   Recall:    0.6789
   F1 Score:  0.6498

📋 Confusion Matrix:
              Predicted
              Safe  Vuln
   Actual Safe   89    23
          Vuln   31   145

⚙️  Learning Rate: 0.000100

⏱️  Timing:
   This epoch:     125.3s
   Total elapsed:  2.1m
   ETA:            102.2m (49 epochs remaining)

🎉 NEW BEST MODEL!
   Previous best F1: 0.0000
   Current F1:       0.6498
   Improvement:      +0.6498 (64.98%)

============================================================
💾 MODEL SAVED: best_model.pt
============================================================
📍 Path: training/checkpoints/best_model.pt
📊 Size: 9.23 MB
🔢 Epoch: 1
📈 Metrics:
   • Accuracy:  0.6456
   • Precision: 0.6234
   • Recall:    0.6789
   • F1 Score:  0.6498
⏰ Time: 2026-02-06 14:23:45
============================================================
```

## 📈 Epoch ที่ไม่มี Improvement

```
================================================================================
📅 EPOCH 15/50 [████████████░░░░░░░░░░░░░░░░░░] 30.0%
================================================================================

────────────────────────────────────────────────────────────────────────────────
📊 EPOCH 15/50 RESULTS
────────────────────────────────────────────────────────────────────────────────
🏋️  Training:
   Loss: 0.3421 | Accuracy: 0.8234 (82.3%)

✅ Validation:
   Loss:      0.4123
   Accuracy:  0.7856 (78.6%)
   Precision: 0.7654
   Recall:    0.8123
   F1 Score:  0.7880

📋 Confusion Matrix:
              Predicted
              Safe  Vuln
   Actual Safe  102    10
          Vuln   24   152

⚙️  Learning Rate: 0.000456

⏱️  Timing:
   This epoch:     118.7s
   Total elapsed:  29.8m
   ETA:            69.3m (35 epochs remaining)

⚠️  No improvement (Best F1: 0.7923 at epoch 12)
   Early stopping patience: 3/10
   🔄 Will continue for 7 more epochs...
```

## ⏹️ Early Stopping Triggered

```
================================================================================
📅 EPOCH 22/50 [█████████████████░░░░░░░░░░░░░░░░░░░] 44.0%
================================================================================

────────────────────────────────────────────────────────────────────────────────
📊 EPOCH 22/50 RESULTS
────────────────────────────────────────────────────────────────────────────────
🏋️  Training:
   Loss: 0.2876 | Accuracy: 0.8567 (85.7%)

✅ Validation:
   Loss:      0.4234
   Accuracy:  0.7823 (78.2%)
   Precision: 0.7598
   Recall:    0.8067
   F1 Score:  0.7825

📋 Confusion Matrix:
              Predicted
              Safe  Vuln
   Actual Safe  101    11
          Vuln   26   150

⚙️  Learning Rate: 0.000312

⏱️  Timing:
   This epoch:     119.2s
   Total elapsed:  43.6m
   ETA:            55.4m (28 epochs remaining)

⚠️  No improvement (Best F1: 0.7923 at epoch 12)
   Early stopping patience: 10/10
   ❌ Patience exhausted!

================================================================================
⏹️  EARLY STOPPING TRIGGERED
================================================================================
📊 Training Statistics:
   • Reason:           No improvement for 10 consecutive epochs
   • Best F1 Score:    0.7923 (Epoch 12)
   • Current F1 Score: 0.7825 (Epoch 22)
   • Epochs Wasted:    10 epochs without improvement
   • Total Epochs:     22/50 (44.0%)
   • Training Time:    43.6 minutes

💡 Best model was saved at epoch 12
   Training stopped early to prevent overfitting.
================================================================================
```

## 🏆 สรุปผลการเทรน

```
[5/6] Saving Final Model...

============================================================
💾 MODEL SAVED: final_model.pt
============================================================
📍 Path: training/checkpoints/final_model.pt
📊 Size: 9.23 MB
🔢 Epoch: 22
📈 Metrics:
   • Accuracy:  0.7823
   • Precision: 0.7598
   • Recall:    0.8067
   • F1 Score:  0.7825
⏰ Time: 2026-02-06 15:07:21
============================================================

📝 Training history saved to training/logs/training_history_20260206_150721.json

[6/6] Training Complete!
================================================================================
🏆 FINAL TRAINING RESULTS
================================================================================

📊 Performance Metrics:
   • Best Validation F1:    0.7923 (Epoch 12)
   • Best Validation Loss:  0.3987
   • Final Validation F1:   0.7825

📈 Training Statistics:
   • Total Epochs:          22/50 (44.0%)
   • Best Epoch:            12
   • Early Stopped:         Yes
   • Total Time:            43.6 minutes
   • Avg Time per Epoch:    119.1 seconds

💾 Saved Files:
   • Best Model:   training/checkpoints/best_model.pt
   • Final Model:  training/checkpoints/final_model.pt
   • Training Log: training/logs/training_history_20260206_150721.json

================================================================================
✅ Training pipeline completed successfully!
================================================================================
```

## 📁 Training History JSON

```json
{
  "total_epochs": 22,
  "best_epoch": 12,
  "best_f1": 0.7923,
  "best_val_loss": 0.3987,
  "early_stopped": true,
  "total_time_seconds": 2616.3,
  "total_time_minutes": 43.6,
  "avg_epoch_time": 118.9,
  "config": {
    "batch_size": 8,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "patience": 10,
    ...
  },
  "training_history": [
    {
      "epoch": 1,
      "train_loss": 0.6543,
      "train_acc": 0.6234,
      "val_loss": 0.6234,
      "val_acc": 0.6456,
      "val_metrics": {
        "accuracy": 0.6456,
        "precision": 0.6234,
        "recall": 0.6789,
        "f1": 0.6498,
        "confusion_matrix": [[89, 23], [31, 145]]
      },
      "epoch_time": 125.3,
      "lr": 0.0001
    },
    ...
  ]
}
```

## 🎯 ข้อมูลที่จะดูเพื่อประเมินการเทรน

### ✅ สัญญาณที่ดี:
- ✓ Loss ลงทุก epoch ในช่วงแรก
- ✓ Train Acc และ Val Acc ใกล้เคียงกัน (ไม่ overfit)
- ✓ Precision และ Recall สมดุล (~0.70-0.85)
- ✓ Confusion matrix แสดง TP และ TN สูง
- ✓ Learning rate ลดลงตามเวลา

### ⚠️ สัญญาณที่ไม่ดี:
- ⚠ Val Loss เพิ่มขึ้นขณะที่ Train Loss ลง → Overfitting
- ⚠ Recall = 100%, Precision ต่ำ → Predict ทุกอย่างเป็น positive
- ⚠ Loss ไม่ลง → Learning rate สูงเกินไป หรือ imbalanced data
- ⚠ Gradients เป็น 0 หรือ explode → Architecture issue

## 🔧 การปรับแต่งถ้าผลไม่ดี:

```python
# ถ้า Loss ไม่ลง
learning_rate: 0.001 → 0.002  # เพิ่ม LR
batch_size: 8 → 16            # เพิ่ม batch size

# ถ้า Overfit
dropout: 0.2 → 0.3            # เพิ่ม dropout
weight_decay: 0.0001 → 0.001  # เพิ่ม regularization

# ถ้า Recall = 100%
pos_weight: auto → manual     # ปรับ class weight
label_smoothing: 0.05 → 0.0   # ปิด smoothing
```
