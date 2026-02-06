# 🎯 High-Quality Dataset Setup Guide
**Created:** February 6, 2026  
**Purpose:** Download real-world vulnerability datasets to replace synthetic data

---

## ⚠️ Prerequisites Required

### 1. Install Python 3.10+
**Windows:**
```powershell
# Download from official website
$url = "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe"
Invoke-WebRequest -Uri $url -OutFile "python_installer.exe"
.\python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
```

**Or manually:**
- Download: https://www.python.org/downloads/
- ✅ Check "Add Python to PATH"
- Install

**Verify:**
```bash
python --version  # Should show Python 3.10+
pip --version     # Should show pip
```

### 2. Install Git
**Windows:**
```powershell
# Download Git for Windows
$url = "https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe"
Invoke-WebRequest -Uri $url -OutFile "git_installer.exe"
.\git_installer.exe /VERYSILENT
```

**Or manually:**
- Download: https://git-scm.com/download/win
- Install with default options

**Verify:**
```bash
git --version  # Should show git version
```

---

## 📦 Step-by-Step: Download Real Datasets

### Step 1: Install Python Dependencies
```bash
cd backend
pip install requests tqdm pandas numpy
```

### Step 2: CodeXGLUE Defect Detection (Recommended)
**Dataset:** Real Python/C code from GitHub with defect labels  
**Size:** ~21,000+ samples  
**Quality:** ⭐⭐⭐⭐⭐

**Option A: Using Hugging Face (Easiest)**
```bash
pip install datasets

# Python script to download
python -c "
from datasets import load_dataset
dataset = load_dataset('code_x_glue_cc_defect_detection')
dataset.save_to_disk('data/raw_datasets/codexglue')
print('✅ CodeXGLUE downloaded!')
"
```

**Option B: Direct Git Clone**
```bash
cd data/raw_datasets
git clone https://github.com/microsoft/CodeXGLUE.git
cd CodeXGLUE/Code-Code/Defect-detection
# Data is in dataset/ folder
```

**Expected structure:**
```
data/raw_datasets/codexglue/
├── train.jsonl (21,854 samples)
├── valid.jsonl (2,732 samples)  
└── test.jsonl (2,732 samples)
```

---

### Step 3: Devign Dataset (Real C Vulnerabilities)
**Dataset:** FFmpeg, QEMU, Wireshark, Linux Kernel vulnerabilities  
**Size:** 27,000+ functions (12,000+ vulnerable)  
**Quality:** ⭐⭐⭐⭐⭐

**Download:**
```bash
cd data/raw_datasets
git clone https://github.com/saikat107/Devign.git
cd Devign

# The dataset is in: Devign/dataset/function.json
```

**Alternative (pre-processed):**
```bash
# If main repo doesn't work, try this fork:
git clone https://github.com/epicosy/devign.git
```

**Expected file:**
```
data/raw_datasets/Devign/dataset/function.json
# Contains 27,000+ C functions with labels
```

---

### Step 4: CVEFixes (Real CVE Patches)
**Dataset:** Real CVE patches with before/after code  
**Size:** 5,000+ CVEs across multiple languages  
**Quality:** ⭐⭐⭐⭐

**Download:**
```bash
cd data/raw_datasets
git clone https://github.com/secureIT-project/CVEfixes.git
cd CVEfixes

# Database file: CVEfixes.db (SQLite)
# Contains vulnerable and fixed code pairs
```

**Extract data:**
```python
# Python script to extract from SQLite
import sqlite3
import json

conn = sqlite3.connect('CVEfixes.db')
cursor = conn.cursor()

# Get vulnerable functions
query = """
SELECT file_change.old_code, file_change.new_code, cve.cve_id
FROM file_change 
JOIN cve ON file_change.cve_id = cve.cve_id
WHERE file_change.old_code IS NOT NULL
LIMIT 5000;
"""

cursor.execute(query)
data = cursor.fetchall()

# Save as JSON
with open('../cvefixes_extracted.json', 'w') as f:
    json.dump([{
        'vulnerable_code': row[0],
        'fixed_code': row[1],
        'cve_id': row[2],
        'label': 1  # vulnerable
    } for row in data], f, indent=2)

print(f'✅ Extracted {len(data)} CVE samples')
```

---

### Step 5: DiverseVul (High Diversity)
**Dataset:** 18,000+ diverse vulnerable functions  
**Size:** 150+ projects  
**Quality:** ⭐⭐⭐⭐

**Download:**
```bash
cd data/raw_datasets
git clone https://github.com/wagner-group/diversevul.git
cd diversevul

# Data in: data/*.json
```

---

### Step 6: JavaScript/Node.js Vulnerabilities
**For JavaScript/TypeScript scanning**

**Option A: npm Advisories Database**
```bash
cd data/raw_datasets
git clone https://github.com/nodejs/security-wg.git
cd security-wg/vuln

# Contains CVE details and vulnerable packages
```

**Option B: Snyk Vulnerability Database**
```bash
# Clone Snyk's open-source vulnerability database
git clone https://github.com/snyk/vulnerability-db.git
cd vulnerability-db

# npm packages vulnerabilities in npm/ folder
```

**Option C: OSV (Open Source Vulnerabilities)**
```bash
# Download PyPI and npm vulnerabilities
pip install osv

# Python script:
python -c "
import osv
import json

# Get npm vulnerabilities
ecosystems = ['npm', 'PyPI']
vulns = []

for ecosystem in ecosystems:
    result = osv.list_vulnerabilities(ecosystem=ecosystem)
    vulns.extend(result)
    
with open('data/raw_datasets/osv_vulns.json', 'w') as f:
    json.dump(vulns, f, indent=2)
    
print(f'✅ Downloaded {len(vulns)} vulnerabilities')
"
```

---

### Step 7: SARD (NIST Reference Dataset)
**Dataset:** Synthetic but high-quality test cases  
**Size:** 100,000+ test cases  
**Quality:** ⭐⭐⭐⭐ (Synthetic but comprehensive)

**Download:**
```bash
# Manual download from NIST SARD
# https://samate.nist.gov/SARD/testsuite.php

# Select:
# - Juliet Test Suite v1.3 (60,000+ cases)
# - Languages: C/C++, Java, Python

# Extract and organize:
# data/raw_datasets/sard/
#   ├── c_testcases/
#   ├── java_testcases/
#   └── python_testcases/
```

---

## 🔄 Unified Dataset Format

After downloading, convert all to **unified format**:

```json
{
  "code": "function vulnerable_code() { ... }",
  "label": 1,
  "language": "python|javascript|c|cpp|java",
  "vulnerability_type": "sql_injection|xss|buffer_overflow|...",
  "cwe_id": "CWE-79",
  "source": "codexglue|devign|cvefixes|...",
  "metadata": {
    "project": "project_name",
    "commit_id": "abc123",
    "file_path": "path/to/file.py"
  }
}
```

---

## 📊 Expected Dataset Summary

| Dataset | Samples | Languages | Type | Natural Code? |
|---------|---------|-----------|------|---------------|
| **CodeXGLUE** | 27,000+ | C, Python | Real GitHub | ✅ YES |
| **Devign** | 27,000+ | C | Real OSS | ✅ YES |
| **CVEFixes** | 5,000+ | Multi | Real CVEs | ✅ YES |
| **DiverseVul** | 18,000+ | C/C++ | Real OSS | ✅ YES |
| **npm/Snyk** | 10,000+ | JavaScript | Real packages | ✅ YES |
| **SARD/Juliet** | 60,000+ | Multi | Synthetic | ⚠️ Synthetic |

**Total Real Data:** ~75,000+ samples with natural code structure

---

## 🎯 Next Steps After Download

1. **Parse to unified format:**
   ```bash
   cd backend
   python scripts/parse_quality_datasets.py
   ```

2. **Verify data quality:**
   ```bash
   python scripts/verify_dataset_quality.py
   ```

3. **Build graph features:**
   ```bash
   python scripts/build_enhanced_graphs.py
   ```

4. **Train with real data:**
   ```bash
   python training/train_with_real_data.py --epochs 50
   ```

---

## 🔧 Troubleshooting

### Python not found
```powershell
# Check installation
python --version

# If not found, add to PATH:
$env:Path += ";C:\Python311;C:\Python311\Scripts"
```

### Git not found
```powershell
# Install Git for Windows
winget install --id Git.Git -e --source winget
```

### Dataset repository moved/deleted
- Check the official paper for the dataset
- Look for alternative mirrors on Zenodo, figshare, or Hugging Face
- Some datasets require registration (e.g., NIST SARD)

### Large downloads fail
```powershell
# Use Git LFS for large files
git lfs install
git lfs pull
```

---

## 📧 Support

- CodeXGLUE Issues: https://github.com/microsoft/CodeXGLUE/issues
- Devign Paper: https://arxiv.org/abs/1909.03496
- CVEFixes: https://github.com/secureIT-project/CVEfixes/issues

---

## ✅ Checklist

- [ ] Python 3.10+ installed
- [ ] Git installed  
- [ ] CodeXGLUE downloaded (27K samples)
- [ ] Devign downloaded (27K samples)
- [ ] CVEFixes downloaded (5K+ samples)
- [ ] JavaScript vulnerabilities (10K+ samples)
- [ ] All datasets in unified format
- [ ] Graph features extracted
- [ ] Ready for training

---

**Target:** 50,000+ real-world samples with natural code structure ✅  
**Replaces:** Synthetic/mapped C patterns ❌
