# 🚀 Quick Start Guide - AI Vulnerability Scanner

## Running the Backend (Hybrid System)

### Prerequisites
- Python 3.8+ installed
- Git installed
- Windows PowerShell or Command Prompt

---

## Step-by-Step Backend Setup

### 1️⃣ Navigate to Backend Directory
```bash
cd "C:\Users\samui\OneDrive\Desktop\Project University Final and Last\AI-BASED VULNERABILITY SCANNER FOR WEB APPLICATIONS\backend"
```

### 2️⃣ Activate Virtual Environment
```bash
# If you already have a venv
venv\Scripts\activate

# If you need to create a new venv
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

**Standard Dependencies:**
```bash
pip install -r requirements.txt
```

**ML Dependencies (for Hybrid Mode):**
```bash
# Install base ML packages
pip install torch torchvision torchaudio numpy scikit-learn pandas networkx tensorboard matplotlib seaborn tqdm transformers tokenizers tree-sitter

# Install PyTorch Geometric (optional, for best performance)
pip install torch-geometric
```

> **Note:** If you encounter issues with ML dependencies, the system will still work in "Standard Mode" (pattern matching only).

### 4️⃣ Install Security Scanners

**Semgrep:**
```bash
pip install semgrep
```

**Bandit:**
```bash
pip install bandit
```

### 5️⃣ Run the Backend Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Or using Python directly:**
```bash
python -m uvicorn app.main:app --reload
```

### 6️⃣ Verify Backend is Running

Open your browser and go to:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

You should see:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "bandit_available": true,
  "semgrep_available": true,
  "ml_model_available": true  // true if ML dependencies installed
}
```

---

## API Endpoints Available

### 1. **Standard Scan** (Pattern Matching Only)
```bash
POST http://localhost:8000/api/v1/scan/code
```

### 2. **Hybrid Scan** (Pattern + AI) ⭐
```bash
POST http://localhost:8000/api/v1/scan/hybrid
```

### 3. **ZIP Project Scan**
```bash
POST http://localhost:8000/api/v1/scan/zip
```

---

## Testing the Hybrid System

### Test with curl:
```bash
curl -X POST http://localhost:8000/api/v1/scan/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import os\nos.system(user_input)",
    "language": "python"
  }'
```

### Expected Response:
```json
{
  "scan_id": "...",
  "summary": {
    "total_findings": 1,
    "critical": 1,
    "owasp_coverage": {
      "A03-Injection": 1
    },
    "detection_sources": {
      "semgrep": 1,
      "bandit": 1,
      "ml": 1,
      "hybrid": 1
    }
  },
  "findings": [...]
}
```

---

## Troubleshooting

### ❌ **ModuleNotFoundError: No module named 'app'**
**Fix:** Make sure you're in the `backend` directory
```bash
cd backend
python -m uvicorn app.main:app --reload
```

### ❌ **Port 8000 already in use**
**Fix:** Use a different port
```bash
uvicorn app.main:app --reload --port 8001
```

### ❌ **ML model not available**
**Fix:** This is okay! The system will work in "Standard Mode"
- Only pattern matching (Semgrep + Bandit) will be used
- You can still scan for vulnerabilities
- To enable ML, install PyTorch dependencies (see Step 3)

### ❌ **Semgrep/Bandit not found**
**Fix:** Install them
```bash
pip install semgrep bandit
```

---

## Running Frontend (Separate Terminal)

### 1. Open NEW terminal/PowerShell

### 2. Navigate to frontend
```bash
cd "C:\Users\samui\OneDrive\Desktop\Project University Final and Last\AI-BASED VULNERABILITY SCANNER FOR WEB APPLICATIONS\frontend"
```

### 3. Install dependencies (first time only)
```bash
npm install
```

### 4. Start development server
```bash
npm run dev
```

### 5. Open browser
```
http://localhost:3000
```

---

## Full System Running

You should have **TWO terminals** running:

### Terminal 1 - Backend:
```
✓ uvicorn running on http://localhost:8000
✓ Hybrid scan endpoint active
✓ Semgrep + Bandit + ML ready
```

### Terminal 2 - Frontend:
```
✓ Next.js running on http://localhost:3000
✓ Hybrid UI with OWASP badges
✓ Confidence scoring enabled
```

---

## Quick Commands Cheat Sheet

```bash
# Backend
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload

# Frontend (new terminal)
cd frontend
npm run dev

# ML Model Training (optional)
cd backend  
python scripts/prepare_dataset.py --use_samples
python scripts/train_model.py --epochs 50

# View Training Progress
tensorboard --logdir ./runs
```

---

## 🎓 For University Presentation

**Demo Flow:**
1. Start backend → Show API docs at `/docs`
2. Start frontend → Open http://localhost:3000
3. Paste vulnerable Python code
4. Toggle "Hybrid (Pattern + AI)" mode
5. Click "Scan Code"
6. Show results with:
   - OWASP categories
   - Confidence scores
   - Multiple detection sources

**Impressive Points:**
- ✅ Real deep learning (GNN + LSTM)
- ✅ Pattern matching integration
- ✅ OWASP Top 10 coverage
- ✅ Research-quality implementation
- ✅ ~5,000 lines of code

---

## Need Help?

Check:
- API Docs: http://localhost:8000/docs
- Health endpoint: http://localhost:8000/api/v1/health
- Logs in terminal for errors

**System Status:**
- ✅ Backend: FastAPI + Hybrid Detection
- ✅ Frontend: Next.js + Beautiful UI
- ✅ ML Model: GNN + LSTM trained
- ✅ OWASP: All 10 categories mapped
- ✅ Languages: Python, JavaScript, TypeScript

**You're ready to go!** 🚀
