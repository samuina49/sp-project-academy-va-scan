# 🧹 Project Cleanup Plan
**Date:** February 6, 2026  
**Purpose:** Remove duplicate and obsolete files

---

## 📋 Files to Keep (Essential)

### Core Application
- ✅ app/ (all files - core backend)
- ✅ training/ (all files - model training)
- ✅ models/ (trained models)
- ✅ data/ (datasets)

### Essential Scripts (New Enhanced Versions)
- ✅ scripts/download_quality_datasets.py (NEW)
- ✅ scripts/quick_download_datasets.py (NEW)
- ✅ scripts/enhanced_dataset_pipeline.py (NEW)

### Documentation
- ✅ README.md
- ✅ PROJECT_STRUCTURE.md
- ✅ MODEL_PERFORMANCE_REPORT.md
- ✅ DOWNLOAD_DATASETS_GUIDE.md (NEW)

---

## 🗑️ Files to DELETE

### 1. Debug/Test Files (One-off usage)
- ❌ test_dfg_debug.py
- ❌ test_api_with_new_model.py
- ❌ test_on_unseen_data.py
- ❌ test_severity_fix.py
- ❌ test_trained_model.py

### 2. Duplicate Dataset Scripts
**Old download scripts (replaced by new ones):**
- ❌ scripts/download_datasets.py (old)
- ❌ scripts/download_cve_datasets.py (old)
- ❌ scripts/fetch_real_world_datasets.py (old)

**Old merge scripts (replaced by enhanced pipeline):**
- ❌ scripts/merge_datasets.py
- ❌ scripts/merge_all_datasets.py
- ❌ scripts/merge_and_clean_datasets.py

**Old preparation scripts (replaced by enhanced pipeline):**
- ❌ scripts/prepare_dataset.py
- ❌ scripts/prepare_full_dataset.py
- ❌ scripts/quick_prepare_dataset.py
- ❌ scripts/improved_dataset_prep.py

### 3. Synthetic Data Generators (Not needed with real data)
- ❌ scripts/generate_synthetic_data.py
- ❌ scripts/generate_cve_inspired_dataset.py
- ❌ scripts/generate_diverse_patterns.py
- ❌ scripts/generate_robust_dataset.py
- ❌ scripts/generate_training_dataset.py
- ❌ scripts/fix_broken_templates.py

### 4. Analysis/Inspection Scripts (One-off usage)
- ❌ scripts/analyze_and_clean_dataset.py
- ❌ scripts/analyze_dataset_leakage.py
- ❌ scripts/analyze_data_sources.py
- ❌ scripts/analyze_devign_patterns.py
- ❌ scripts/inspect_cve_datasets.py
- ❌ scripts/inspect_dataset_quality.py

### 5. Old Dataset Processing Scripts
- ❌ scripts/build_dataset.py
- ❌ scripts/clean_dataset.py
- ❌ scripts/collect_production_dataset.py
- ❌ scripts/collect_safe_code.py
- ❌ scripts/parse_datasets.py
- ❌ scripts/split_dataset.py
- ❌ scripts/fix_dataset_split.py
- ❌ scripts/convert_linevul.py

### 6. Balance/Sampling Scripts (Replaced by pipeline)
- ❌ scripts/check_balance.py
- ❌ scripts/check_downsampled.py
- ❌ scripts/handle_imbalance.py
- ❌ scripts/downsample_by_fingerprint.py

### 7. Old Training Scripts (Keep only essential)
- ❌ scripts/train_model.py (duplicate)
- ❌ scripts/retrain_model.py
- ❌ scripts/retrain_model_clean.py

### 8. Old Test Scripts
- ❌ scripts/test_final_model.py
- ❌ scripts/test_hybrid.py
- ❌ scripts/test_js_scan.py
- ❌ scripts/test_ml_only.py
- ❌ scripts/stress_test_ml.py
- ❌ scripts/real_world_validation.py
- ❌ scripts/performance_benchmark.py

### 9. Utility Scripts (Redundant)
- ❌ scripts/cleanup_project.py (ironic!)
- ❌ scripts/debug_scanner.py
- ❌ scripts/debug_structure.py
- ❌ scripts/install_torch_geometric.py (one-time use)

### 10. Old Pipeline Scripts
- ❌ scripts/master_dataset_pipeline.py (replaced by enhanced version)

### 11. Old Documentation (Replaced)
- ❌ DATASET_COLLECTION_GUIDE.md (replaced by DOWNLOAD_DATASETS_GUIDE.md)
- ❌ DATASET_PIPELINE_README.md (info moved to enhanced pipeline)

### 12. CI/CD Scripts (If not used)
- ❌ scripts/cicd_scanner.py (check if used first)
- ❌ scripts/pre-commit-hook.py (check if used first)

---

## 📦 Recommended Structure After Cleanup

```
backend/
├── app/                              # Core application
├── training/                         # Model training
├── models/                           # Trained models
├── data/                            # Datasets
│   ├── raw_datasets/
│   └── processed_graphs/
├── scripts/                         # Essential scripts only
│   ├── download_quality_datasets.py
│   ├── quick_download_datasets.py
│   └── enhanced_dataset_pipeline.py
├── requirements.txt
├── requirements-ml.txt
├── Dockerfile
├── README.md
├── PROJECT_STRUCTURE.md
├── MODEL_PERFORMANCE_REPORT.md
└── DOWNLOAD_DATASETS_GUIDE.md
```

**Before:** 54 scripts + 5 test files = ~60 files  
**After:** 3 essential scripts = ~3 files  
**Reduction:** ~95% cleanup! 🎉

---

## ✅ Actions

1. Delete all files marked with ❌
2. Keep archived_files/ but document it's deprecated
3. Update PROJECT_STRUCTURE.md
4. Create CHANGELOG.md documenting improvements
