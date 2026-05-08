# Vulnerability Dataset Source Report

## Overview
This document details the dataset used to train the final AI scanner model (Hybrid GNN+LSTM). The dataset strategy employs a **Multi-Source Diverse Approach**, combining real-world vulnerability data from multiple sources with fingerprint-based splitting to ensure zero data leakage.

**Latest Dataset Status (February 3, 2026):**
- **Total Samples**: 3,117
- **Unique Fingerprints**: 96.5% (3,009 unique patterns)
- **Data Leakage**: 0% (verified by fingerprint analysis)
- **Training Result**: F1 Score 99.58%, Accuracy 99.37%

---

## 1. Dataset Methodology: Multi-Source Diverse Approach

### Phase 1: Data Collection
Multiple sources were merged to create a diverse, high-quality dataset:

| Source | Samples | Description |
|--------|---------|-------------|
| **Big-Vul** | Real CVE samples | Actual vulnerabilities from CVE database |
| **SARD** | Pattern-based | Software Assurance Reference Dataset patterns |
| **GitHub Advisory** | Advisory-inspired | Security advisory-inspired patterns |
| **Generated Diverse** | 247 samples | Variable-randomized vulnerability patterns |

### Phase 2: Pattern Generation
Diverse vulnerability patterns with variable name randomization:

| Vulnerability Type | Samples | Methods Used |
|-------------------|---------|--------------|
| **SQL Injection** | 57 | f-string, format(), concatenation, various contexts |
| **Command Injection** | 38 | subprocess, os.system, Popen, various wrappers |
| **Path Traversal** | 38 | directory traversal, file access patterns |
| **XSS** | 26 | innerHTML, document.write, various DOM methods |
| **SSRF** | 20 | fetch, requests, urllib to user-controlled URLs |
| **Deserialization** | 18 | pickle.loads, yaml.load, unsafe deserialization |
| **Safe Samples** | 50 | parameterized queries, validated inputs, safe patterns |

### Phase 3: Data Quality Assurance
- **Fingerprint-based Splitting**: Code fingerprints ensure no train/test overlap
- **Unique Fingerprints**: 96.5% (3,009 unique from 3,117 samples)
- **Variable Randomization**: Prevents template memorization
- **Class Balance**: 73.5% vulnerable / 26.5% safe (realistic distribution)

---

## 2. Final Dataset Composition

### 2.1 Distribution Summary
| Metric | Value |
|--------|-------|
| **Total Samples** | 3,117 |
| **Unique Fingerprints** | 3,009 (96.5%) |
| **Data Leakage** | 0% |

### 2.2 Distribution by Class (Training Set)
| Label | Count | Percentage |
|-------|-------|------------|
| **Vulnerable (1)** | 1,832 | 73.5% |
| **Safe (0)** | 659 | 26.5% |

---

## 3. Dataset Splits

Fingerprint-based splitting ensures no overlap between splits:

| Split | Samples | Percentage |
|-------|---------|------------|
| **Training Set** | 2,491 | 80% |
| **Validation Set** | 319 | 10% |
| **Test Set** | 307 | 10% |

*Note: Fingerprint overlap check confirmed 0% leakage between all splits.*

---

## 4. Training Results

| Metric | Value |
|--------|-------|
| **Best F1 Score** | 99.58% |
| **Best Accuracy** | 99.37% |
| **Training Epochs** | 25 |
| **Model Parameters** | 1,905,409 |
| **Vocabulary Size** | 3,336 tokens |

---

## 5. Data Files Location

```
backend/data/processed/
├── merged_train.json     # 2,491 samples (training)
├── merged_val.json       # 319 samples (validation)
├── merged_test.json      # 307 samples (test)
├── diverse_generated.json # 247 generated patterns
└── external_dataset.json  # 150 external samples

backend/training/models/
├── hybrid_model_best.pth  # Trained model (~8 MB)
├── vocab.json             # 3,336 tokens
└── training_info.json     # F1=99.58%, Acc=99.37%
```

---

## 6. Old Datasets (Archived)

*The following datasets were used in earlier phases but are now superseded by the multi-source dataset.*
- **CVE-Inspired (Phase 1)**: 14,255 samples (had template-like patterns, only 0.52% unique fingerprints)
- **Synthetic Original**: 20,000 random template samples (legacy)
- **Public Research**: MSR 20 / BigVul (used for reference only)
1.  **Normalization**: Removed comments, docstrings, and normalized whitespace.
2.  **Deduplication**: Removed exact code duplicates across all sources.
3.  **Strict Split**: 80/10/10 split with data leakage checks.

This dataset is stored in `backend/data/processed/final_{train|val|test}.json` and serves as the foundation for the new training run.
