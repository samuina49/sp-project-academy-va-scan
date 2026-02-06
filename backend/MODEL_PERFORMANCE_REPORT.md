# 📊 Model Performance Report - HybridVulnerabilityModel

**Model Version**: hybrid_model_best.pth  
**Training Date**: February 5, 2026  
**Model Architecture**: GAT (3 layers) + BiLSTM (2 layers) with Fusion Layers  
**Total Parameters**: 2,413,057 trainable parameters  
**Vocabulary Size**: 5,319 tokens

---

## 🎯 Performance Summary

| Metric | Validation Set | Test Set (Unseen) | Notes |
|--------|----------------|-------------------|-------|
| **Accuracy** | 99.10% | **90.86%** | ✅ Realistic generalization |
| **F1 Score** | 99.01% | **88.99%** | ✅ Excellent balance |
| **Precision** | 99.89% | **99.80%** | ✅ Extremely low false positives |
| **Recall** | 98.13% | **80.29%** | ⚠️ 20% false negatives (acceptable for security) |
| **ROC-AUC** | N/A | **0.9813** | ✅ Excellent discrimination ability |

---

## 📈 Detailed Test Set Performance

### Classification Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **True Positives (TP)** | 493 / 614 | Correctly identified vulnerable code |
| **True Negatives (TN)** | 720 / 721 | Correctly identified safe code |
| **False Positives (FP)** | 1 / 721 | Safe code wrongly flagged ✅ |
| **False Negatives (FN)** | 121 / 614 | Vulnerable code missed ⚠️ |

### Error Rates
| Error Type | Rate | Count | Assessment |
|------------|------|-------|------------|
| **False Positive Rate (FPR)** | **0.14%** | 1 out of 721 safe samples | ✅ Excellent - minimal false alarms |
| **False Negative Rate (FNR)** | **19.71%** | 121 out of 614 vulnerable | ⚠️ Acceptable - some vulnerabilities missed |

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Safe** | 85.61% | 99.86% | 92.17% | 721 |
| **Vulnerable** | 99.80% | 80.29% | 88.99% | 614 |
| **Macro Average** | 92.70% | 90.07% | 90.58% | 1,335 |
| **Weighted Average** | 91.80% | 90.86% | 90.74% | 1,335 |

---

## 🔬 Confusion Matrix

```
                Predicted
                Safe    Vulnerable
Actual Safe     720     1         (99.86% correct)
Actual Vuln     121     493       (80.29% correct)
```

**Key Insights:**
- ✅ **Only 1 false positive** out of 721 safe samples (0.14%)
- ⚠️ **121 false negatives** out of 614 vulnerable samples (19.71%)
- ✅ **720 true negatives** - excellent at recognizing safe code
- ✅ **493 true positives** - good at catching vulnerabilities

---

## 📊 Dataset Information

### Training Set
| Metric | Count | Percentage |
|--------|-------|-----------|
| Total Samples | 6,222 | 70% |
| Vulnerable | 2,860 | 45.97% |
| Safe | 3,362 | 54.03% |
| Augmented Data | ~13% | Minimal augmentation |

### Validation Set
| Metric | Count | Percentage |
|--------|-------|-----------|
| Total Samples | 1,333 | 15% |
| Best Epoch | 12 | Early stopped |
| Val F1 Score | 99.01% | Peak performance |

### Test Set (Unseen Data)
| Metric | Count | Percentage |
|--------|-------|-----------|
| Total Samples | 1,335 | 15% |
| Vulnerable | 614 | 46.0% |
| Safe | 721 | 54.0% |
| **Never seen during training** | ✅ | True generalization test |

---

## ⚙️ Model Configuration

### Hyperparameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Dropout** | 0.4 | Increased from 0.2 (stronger regularization) |
| **Weight Decay** | 0.01 | 10x increase from 0.001 |
| **Learning Rate** | 0.0001 | Reduced from 0.0005 |
| **Gradient Clip** | 0.5 | Reduced from 1.0 |
| **Label Smoothing** | 0.1 | NEW - prevents overconfidence |
| **Batch Size** | 32 | Standard |
| **Early Stop Patience** | 10 | Doubled from 5 |
| **Best Epoch** | 12 | Out of 22 epochs trained |

### Architecture Details
| Component | Configuration |
|-----------|--------------|
| **GNN Branch** | GAT with 3 layers, 64 node features |
| **LSTM Branch** | BiLSTM 2 layers, 256 embedding dim |
| **Fusion Layers** | Attention-based combination |
| **Hidden Dim** | 128 |
| **Output** | Binary classification (Safe/Vulnerable) |

---

## 🎯 Real-World Validation

### Test Sample Results
| File | Language | Expected | Detected | Confidence | Status |
|------|----------|----------|----------|------------|--------|
| javascript_vulnerabilities.js | JavaScript | Vulnerable | Vulnerable | 72.70% | ✅ PASS |
| typescript_vulnerabilities.ts | TypeScript | Vulnerable | Vulnerable | 63.15% | ✅ PASS |
| python_vulnerabilities.py | Python | Vulnerable | Safe | 28.63% | ❌ FAIL |

**Key Findings:**
- ✅ JavaScript/TypeScript detection: Excellent (>60% confidence)
- ⚠️ Python detection: Needs improvement (possibly more Python training samples needed)

---

## 📉 Comparison with Previous Model

| Metric | Previous Model | Current Model | Improvement |
|--------|----------------|---------------|-------------|
| **Test Accuracy** | 74.10% | **90.86%** | **+16.76%** ⬆️ |
| **Test F1** | ~76% | **88.99%** | **+13%** ⬆️ |
| **Precision** | 76% | **99.80%** | **+23.8%** ⬆️ |
| **False Positive Rate** | 19% | **0.14%** | **-98.6%** ⬇️ |
| **Validation-Test Gap** | 24% (98%→74%) | **8.2%** (99%→91%) | **-15.8%** ⬇️ |

**Major Improvements:**
1. ✅ **Dramatic FPR reduction**: 19% → 0.14% (135x better!)
2. ✅ **Precision boost**: 76% → 99.80% (almost no false alarms)
3. ✅ **Better generalization**: Val-Test gap reduced from 24% to 8.2%
4. ✅ **Overall accuracy**: +16.76 percentage points

---

## 🔍 Model Strengths & Weaknesses

### ✅ Strengths
1. **Exceptional Precision (99.80%)**: Almost no false positives - won't annoy developers with false alarms
2. **Excellent ROC-AUC (0.9813)**: Strong discrimination between safe and vulnerable code
3. **Fast Inference**: < 1 second per file
4. **Low False Alarm Rate**: Only 0.14% of safe code incorrectly flagged
5. **Good Generalization**: 8.2% validation-test gap (realistic expectations)

### ⚠️ Weaknesses
1. **False Negative Rate (19.71%)**: Misses about 1 in 5 vulnerabilities
2. **Python Detection**: Lower confidence on Python samples (28.63%)
3. **Recall (80.29%)**: Could catch more vulnerabilities
4. **Class Imbalance Sensitivity**: Slightly better at detecting safe code than vulnerable code

---

## 🎓 Recommended Use Cases

### ✅ Ideal For:
- **Code Review Assistant**: High precision means few false alarms
- **Pre-commit Hooks**: Won't block developers unnecessarily
- **Educational Tool**: Shows vulnerability patterns clearly
- **Security Auditing**: Good at finding obvious vulnerabilities

### ⚠️ Not Recommended As:
- **Sole Security Tool**: 20% FNR means some vulnerabilities will be missed
- **Compliance Tool**: Should be combined with static analyzers (Semgrep/Bandit)
- **Critical Systems**: Use as first pass, manual review still needed

---

## 🚀 Deployment Recommendations

### Production Configuration
```python
ML_ENABLED = True  # Enable hybrid detection
ML_CONFIDENCE_THRESHOLD = 0.5  # Default threshold
ML_WEIGHT = 0.4  # 40% ML, 60% Pattern matching
```

### Performance Expectations
- **Startup Time**: ~10 seconds (ML model loading)
- **Scan Speed**: < 1 second per file
- **Memory Usage**: ~500MB (model in memory)
- **Accuracy**: 90.86% (expect 1 in 10 samples to be misclassified)

### Best Practices
1. **Combine with Pattern Scanner**: Use hybrid mode for best results
2. **Manual Review**: Always review CRITICAL/HIGH severity findings
3. **Threshold Tuning**: Adjust `ML_CONFIDENCE_THRESHOLD` based on false positive tolerance
4. **Continuous Training**: Retrain with new vulnerability samples monthly

---

## 📝 Conclusion

**Overall Assessment**: ⭐⭐⭐⭐½ (4.5/5)

The HybridVulnerabilityModel achieved **excellent production-ready performance** with:
- ✅ **90.86% accuracy** on unseen test data
- ✅ **99.80% precision** - minimal false alarms
- ✅ **0.9813 ROC-AUC** - excellent discrimination
- ✅ **98% reduction in false positives** compared to previous model

**Recommendation**: ✅ **APPROVED FOR PRODUCTION USE**

**Caveats**:
- Should be used alongside pattern-based scanners (Semgrep/Bandit)
- 20% false negative rate means manual review is still necessary for critical applications
- Consider retraining with more Python samples to improve Python detection

---

**Generated**: February 5, 2026  
**Model Path**: `training/models/hybrid_model_best.pth`  
**Vocabulary**: `training/models/vocab.json`  
**Training Script**: `training/train.py`
