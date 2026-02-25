"""
Deep Data Quality Diagnostic
=============================
Linear probe showed ALL features have Val AUC < 0.52, meaning the features
(or labels) contain no generalizable signal. This script investigates WHY.

Tests:
1. TF-IDF + LogReg on raw code text (is text itself discriminative?)
2. Inspect actual code samples (what do vuln vs safe look like?)
3. Check if labels correlate with code length, language, or source
4. Test vulnerability keywords presence
"""

import pickle
import numpy as np
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_geometric.data import Data
import torch

@dataclass
class ProcessedSample:
    code: str
    label: int
    language: str
    graph_data: object
    vulnerability_type: str
    source: str
    metadata: dict
    token_ids: Optional[torch.Tensor] = None
    code_metrics: Optional[np.ndarray] = None

data_dir = Path("data/processed_graphs")

# Load ORIGINAL data (before our modifications)
print("Loading ORIGINAL data (before feature enhancement)...")
train_path = data_dir / "train_graphs_ORIGINAL.pkl"
val_path = data_dir / "val_graphs_ORIGINAL.pkl"

if not train_path.exists():
    train_path = data_dir / "train_graphs.pkl"
    val_path = data_dir / "val_graphs.pkl"
    print("  (Using current data, originals not found)")

with open(train_path, 'rb') as f:
    train = pickle.load(f)
with open(val_path, 'rb') as f:
    val = pickle.load(f)

print(f"Train: {len(train)}, Val: {len(val)}")

# =============================================================================
# TEST 1: TF-IDF + LogReg on raw code
# =============================================================================
print("\n" + "="*80)
print("TEST 1: TF-IDF + Logistic Regression on raw code text")
print("="*80)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

train_codes = [s.code for s in train]
train_labels = [s.label for s in train]
val_codes = [s.code for s in val]
val_labels = [s.label for s in val]

# TF-IDF with code-relevant settings
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Unigrams + bigrams
    analyzer='word',
    token_pattern=r'\b\w+\b',
    min_df=2,
    max_df=0.95
)

X_train = tfidf.fit_transform(train_codes)
X_val = tfidf.transform(val_codes)

clf = LogisticRegression(max_iter=1000, C=0.1)
clf.fit(X_train, train_labels)

train_pred = clf.predict_proba(X_train)[:, 1]
val_pred = clf.predict_proba(X_val)[:, 1]

train_auc = roc_auc_score(train_labels, train_pred)
val_auc = roc_auc_score(val_labels, val_pred)
train_acc = accuracy_score(train_labels, (train_pred > 0.5).astype(int))
val_acc = accuracy_score(val_labels, (val_pred > 0.5).astype(int))
val_f1 = f1_score(val_labels, (val_pred > 0.5).astype(int))

print(f"  Train AUC: {train_auc:.4f} | Acc: {train_acc:.4f}")
print(f"  Val AUC:   {val_auc:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

if val_auc > 0.60:
    print(f"  *** CODE TEXT IS DISCRIMINATIVE (AUC={val_auc:.3f}) ***")
    print(f"  The model architecture/features are the bottleneck, not the data")
    
    # Show most discriminative words
    feature_names = tfidf.get_feature_names_out()
    coef = clf.coef_[0]
    top_vuln_idx = np.argsort(coef)[-15:]
    top_safe_idx = np.argsort(coef)[:15]
    
    print(f"\n  Top VULNERABLE indicators:")
    for idx in reversed(top_vuln_idx):
        print(f"    '{feature_names[idx]}' (coef={coef[idx]:.4f})")
    
    print(f"\n  Top SAFE indicators:")
    for idx in top_safe_idx:
        print(f"    '{feature_names[idx]}' (coef={coef[idx]:.4f})")
else:
    print(f"  *** CODE TEXT IS NOT DISCRIMINATIVE ***")
    print(f"  The labels may be noisy or the task is too hard for surface features")

# =============================================================================
# TEST 2: Inspect actual samples
# =============================================================================
print("\n" + "="*80)
print("TEST 2: Sample code inspection")
print("="*80)

vuln_samples = [s for s in train if s.label == 1]
safe_samples = [s for s in train if s.label == 0]

print(f"\n--- RANDOM VULNERABLE SAMPLES ---")
import random
random.seed(42)
for s in random.sample(vuln_samples, min(3, len(vuln_samples))):
    code_preview = s.code[:300].replace('\n', '\n    ')
    print(f"\n  [VULNERABLE] Source={s.source} | Lang={s.language} | Type={s.vulnerability_type}")
    print(f"    {code_preview}")
    if len(s.code) > 300:
        print(f"    ... [{len(s.code)} chars total]")

print(f"\n--- RANDOM SAFE SAMPLES ---")
for s in random.sample(safe_samples, min(3, len(safe_samples))):
    code_preview = s.code[:300].replace('\n', '\n    ')
    print(f"\n  [SAFE] Source={s.source} | Lang={s.language} | Type={s.vulnerability_type}")
    print(f"    {code_preview}")
    if len(s.code) > 300:
        print(f"    ... [{len(s.code)} chars total]")

# =============================================================================
# TEST 3: Label correlation with metadata
# =============================================================================
print("\n" + "="*80)
print("TEST 3: Label correlations (are labels just predicting source/language?)")
print("="*80)

# By source
print("\n  Label distribution by SOURCE:")
for source in set(s.source for s in train):
    src_samples = [s for s in train if s.source == source]
    vuln_pct = 100 * sum(1 for s in src_samples if s.label == 1) / len(src_samples)
    print(f"    {source:20s}: {len(src_samples):4d} samples, {vuln_pct:.1f}% vuln")

# By language
print("\n  Label distribution by LANGUAGE:")
for lang in set(s.language for s in train):
    lang_samples = [s for s in train if s.language == lang]
    vuln_pct = 100 * sum(1 for s in lang_samples if s.label == 1) / len(lang_samples)
    print(f"    {lang:20s}: {len(lang_samples):4d} samples, {vuln_pct:.1f}% vuln")

# By vulnerability type
print("\n  Vulnerability types (train):")
types = Counter(s.vulnerability_type for s in train if s.label == 1)
for vtype, count in types.most_common(15):
    vtype_str = str(vtype) if vtype is not None else "None"
    print(f"    {vtype_str:30s}: {count:4d}")

# =============================================================================
# TEST 4: Keyword-based features
# =============================================================================
print("\n" + "="*80)
print("TEST 4: Vulnerability keyword analysis")
print("="*80)

vuln_keywords = [
    'eval', 'exec', 'system', 'shell_exec', 'popen', 'subprocess',
    'sql', 'query', 'execute', 'cursor',
    'input', 'request', 'param', 'user_input', 'args',
    'innerHTML', 'document.write', 'outerHTML',
    'open(', 'read(', 'write(',
    'password', 'secret', 'key', 'token', 'credential',
    'pickle.loads', 'yaml.load', 'deserialize',
    'http://', 'https://', 'url',
    'os.path', 'os.system', '__import__',
]

print("\n  Keyword presence (% of samples containing keyword):")
print(f"  {'Keyword':30s} | {'Vuln %':>8s} | {'Safe %':>8s} | {'Diff':>8s}")
print(f"  {'-'*30} | {'-'*8} | {'-'*8} | {'-'*8}")

discriminative_keywords = []
for kw in vuln_keywords:
    vuln_pct = 100 * sum(1 for s in vuln_samples if kw.lower() in s.code.lower()) / len(vuln_samples)
    safe_pct = 100 * sum(1 for s in safe_samples if kw.lower() in s.code.lower()) / len(safe_samples)
    diff = vuln_pct - safe_pct
    if abs(diff) > 2:
        discriminative_keywords.append((kw, diff))
    print(f"  {kw:30s} | {vuln_pct:7.1f}% | {safe_pct:7.1f}% | {diff:+7.1f}%")

if discriminative_keywords:
    print(f"\n  Discriminative keywords (diff > 2%):")
    for kw, diff in sorted(discriminative_keywords, key=lambda x: abs(x[1]), reverse=True):
        direction = "VULN" if diff > 0 else "SAFE"
        print(f"    '{kw}' → {direction} (diff={diff:+.1f}%)")

# =============================================================================
# TEST 5: Cross-validation on TF-IDF (check if train/val split is the issue)
# =============================================================================
print("\n" + "="*80)
print("TEST 5: Cross-validation on combined data")
print("="*80)

from sklearn.model_selection import cross_val_score

all_codes = train_codes + val_codes
all_labels = np.array(train_labels + val_labels)

X_all = tfidf.fit_transform(all_codes)
cv_scores = cross_val_score(
    LogisticRegression(max_iter=1000, C=0.1),
    X_all, all_labels, cv=5, scoring='roc_auc'
)
print(f"  5-Fold CV AUC (TF-IDF + LogReg): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Individual folds: {[f'{s:.3f}' for s in cv_scores]}")

if cv_scores.mean() > 0.60:
    print(f"  *** DATA IS LEARNABLE (CV AUC={cv_scores.mean():.3f}) ***")
    print(f"  Problem is likely train/val split or feature extraction")
else:
    print(f"  *** DATA MAY HAVE LABEL QUALITY ISSUES ***")

# =============================================================================
# TEST 6: Check raw dataset files for label quality
# =============================================================================
print("\n" + "="*80)
print("TEST 6: Raw dataset inspection")
print("="*80)

raw_dir = Path("data/raw_datasets")
if raw_dir.exists():
    for f in raw_dir.glob("*.json"):
        import json
        with open(f, 'r', encoding='utf-8') as jf:
            try:
                data = json.load(jf)
                if isinstance(data, list):
                    n = len(data)
                    labels = Counter(d.get('label', 'missing') for d in data)
                    print(f"\n  {f.name}: {n} samples")
                    print(f"    Labels: {dict(labels)}")
                    if n > 0:
                        sample = data[0]
                        print(f"    Fields: {list(sample.keys())}")
                        code_len = len(sample.get('code', ''))
                        print(f"    Sample code length: {code_len}")
            except:
                print(f"  {f.name}: ERROR reading")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
