# ML Model Improvements Plan

## Current Status
The GNN+LSTM ML model is **currently disabled** due to high false positive rate.

### Problems Identified:
1. **High False Positive Rate**: Model detects safe code (e.g., `import os`) as vulnerabilities
2. **Low Precision**: Confidence scores are misleading (100% confidence on false positives)
3. **No Line-Level Detection**: Cannot pinpoint exact vulnerability locations
4. **Training Data Quality**: Dataset may contain noise and mislabeled samples

## Comparison: Pattern Matching vs ML

| Metric | Pattern Matching | Current ML Model |
|--------|-----------------|------------------|
| **Precision** | ✅ High (~95%) | ❌ Low (~40%) |
| **False Positives** | ✅ Very Low | ❌ Very High |
| **Line Detection** | ✅ Accurate | ❌ Whole file only |
| **Speed** | ✅ Fast (<1s) | ⚠️ Slower (~3-5s) |
| **Maintainability** | ✅ Easy (add rules) | ❌ Hard (retrain) |

**Conclusion**: Pattern Matching is currently more reliable and accurate.

---

## Future Improvement Options

### Option 1: Retrain with Cleaned Dataset
**Steps:**
1. Clean training dataset
   - Remove false positives
   - Add more negative samples (safe code)
   - Balance vulnerability classes
2. Implement proper validation split
3. Add post-processing filters
4. Lower confidence threshold to reduce false positives

**Estimated Time**: 2-3 weeks
**Success Rate**: Medium (60-70%)

---

### Option 2: Use Pre-trained Models
**Recommended Models:**
1. **CodeBERT** (Microsoft)
   - Pre-trained on code understanding
   - Fine-tune for vulnerability detection
   - Better context understanding

2. **GraphCodeBERT** (Microsoft)
   - Understands code structure (AST)
   - Better for structural vulnerabilities

3. **Vuldeepecker** (Academic)
   - Specifically designed for vulnerability detection

**Estimated Time**: 1-2 weeks
**Success Rate**: High (80-90%)

---

### Option 3: Hybrid Approach with Confidence Filtering
Keep ML as a **secondary validator** only:
- Pattern Matching detects vulnerabilities (primary)
- ML validates findings (confidence boost)
- Only add ML findings if confidence > 90% AND pattern didn't detect

**Estimated Time**: 1 week
**Success Rate**: Medium-High (70-80%)

---

### Option 4: Keep Pattern-Only (Recommended for Production)
**Why?**
- Pattern Matching already provides excellent accuracy
- Easier to maintain and update
- Semgrep rules cover OWASP Top 10
- Can add custom rules quickly

**What to do with ML code?**
- Keep code archived for future research
- Document as "experimental feature"
- Can be enabled via config for testing

---

## Recommended Action

**For Production/Final Project:**
✅ **Use Pattern Matching Only**
- Set `ML_ENABLED = False`
- Focus on improving Semgrep rules
- Brand as "Advanced Pattern-Based Detection"

**For Future Research:**
🔬 **Explore Pre-trained Models**
- Try CodeBERT fine-tuning
- Compare with Pattern Matching baseline
- Only deploy if significantly better

---

## How to Enable ML (for testing)

1. Edit `backend/app/core/config.py`:
   ```python
   ML_ENABLED: bool = True
   ```

2. Restart backend:
   ```bash
   python -m uvicorn app.main:app --reload
   ```

3. Test with known vulnerabilities and check false positive rate

4. **Important**: Review all ML findings manually before trusting results

---

## Contact & References

**Training Code**: `backend/training/`
**Model Files**: `backend/models/best_model.pt`
**Dataset**: `backend/data/training/`

**Useful Papers**:
- VulDeePecker: https://arxiv.org/abs/1801.01681
- CodeBERT: https://arxiv.org/abs/2002.08155
- GraphCodeBERT: https://arxiv.org/abs/2009.08366
