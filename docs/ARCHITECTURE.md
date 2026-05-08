# Project Architecture

## System Overview

The vulnerability scanner is a full-stack web application designed to detect security vulnerabilities in source code at the line level. It combines traditional static analysis tools (Bandit, Semgrep) with a modern web interface for easy demonstration and evaluation.

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                     (Next.js Frontend)                      │
│  - Code Editor (Monaco)                                     │
│  - File Upload                                              │
│  - Results Visualization                                    │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/REST API
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                         │
├─────────────────────────────────────────────────────────────┤
│  API Layer          │  Service Layer    │  Security Layer   │
│  - /scan/code       │  - Scanner        │  - ZIP validation │
│  - /scan/zip        │    Orchestrator   │  - Path security  │
│  - /health          │                   │  - Temp cleanup   │
└─────────────────────┴───────────────────┴───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Static Analysis Tools                     │
├──────────────────────────┬──────────────────────────────────┤
│   Bandit (Python)        │   Semgrep (JS/TS/JSX/TSX)       │
│   - Security checks      │   - OWASP rules                  │
│   - CWE mapping          │   - Custom patterns              │
└──────────────────────────┴──────────────────────────────────┘
```

## Backend Architecture

### Directory Structure

```
backend/
├── app/
│   ├── api/              # API endpoints
│   │   └── v1/
│   │       ├── scan.py   # Scan endpoints
│   │       └── health.py # Health check
│   ├── core/             # Core utilities
│   │   ├── config.py     # Configuration
│   │   ├── security.py   # Security utilities
│   │   └── temp_manager.py
│   ├── models/           # Data models
│   │   └── schemas.py    # Pydantic models
│   ├── scanners/         # Scanner integration
│   │   ├── bandit_scanner.py
│   │   ├── semgrep_scanner.py
│   │   └── scanner_orchestrator.py
│   └── main.py           # Application entry
├── tests/                # Tests
└── requirements.txt
```

### Component Responsibilities

#### 1. API Layer (`app/api/`)
- **Purpose**: REST API endpoints
- **Endpoints**:
  - `POST /api/v1/scan/code`: Scan pasted code
  - `POST /api/v1/scan/zip`: Scan ZIP project
  - `GET /api/v1/health`: Health check

#### 2. Scanner Layer (`app/scanners/`)
- **BanditScanner**: Python vulnerability detection
- **SemgrepScanner**: JS/TS vulnerability detection
- **ScannerOrchestrator**: Routes files to appropriate scanner

#### 3. Security Layer (`app/core/security.py`)
- ZIP Slip prevention
- Path traversal protection
- File type validation
- Resource limit enforcement

#### 4. Data Models (`app/models/schemas.py`)
- Request/Response schemas
- Type validation
- API documentation

## Frontend Architecture

### Directory Structure

```
frontend/
├── src/
│   ├── app/              # Next.js app router
│   │   ├── layout.tsx    # Root layout
│   │   ├── page.tsx      # Main page
│   │   └── globals.css   # Global styles
│   ├── components/       # React components
│   │   ├── CodeEditor.tsx
│   │   └── ResultsPanel.tsx
│   ├── lib/              # Utilities
│   │   └── api.ts        # API client
│   └── types/            # TypeScript types
│       └── api.ts
└── package.json
```

### Component Hierarchy

```
App (page.tsx)
├── Header
│   └── Title + Badge
├── Controls
│   ├── Mode Selector (Code/ZIP)
│   └── Toolbar (Language + Scan Button)
├── Content
│   ├── Left Panel
│   │   ├── CodeEditor (Monaco)
│   │   └── UploadArea
│   └── Right Panel
│       └── ResultsPanel
│           ├── Loading State
│           ├── Empty State
│           └── Findings List
│               └── FindingCard[]
```

## Data Flow

### Single File Scan Flow

```
1. User inputs code → CodeEditor
2. User clicks "Scan Code"
3. Frontend → POST /api/v1/scan/code
   {
     "code": "...",
     "language": "python",
     "filename": "example.py"
   }
4. Backend:
   a. Create temp file
   b. Detect language
   c. Route to scanner (Bandit/Semgrep)
   d. Parse results
   e. Clean up temp file
5. Backend → Response with findings
6. Frontend → Update ResultsPanel
7. CodeEditor highlights vulnerable lines
```

### ZIP Scan Flow

```
1. User uploads ZIP → FileInput
2. Frontend → POST /api/v1/scan/zip (multipart/form-data)
3. Backend:
   a. Validate ZIP size
   b. Extract to temp directory (Zip Slip prevention)
   c. Scan all supported files recursively
   d. Aggregate results by file
   e. Clean up temp directory
4. Backend → Response with multi-file results
5. Frontend → Display grouped by file
```

## Security Design

### Input Validation
- File size limits (50MB)
- File count limits (500 files)
- Extension whitelist
- Path validation

### Isolation
- Per-scan temporary directories
- Automatic cleanup (context managers)
- No code execution

### Path Security
- Zip Slip prevention during extraction
- Path traversal protection
- Base directory validation

## Scalability Considerations

### Current Design (MVP)
- Synchronous scanning
- Single-threaded
- In-memory results
- No persistence

### Future Improvements
- Async scanning with task queues (Celery/RQ)
- Database for scan history
- Multi-worker horizontally scaling 
- Caching for repeated scans
- Rate limiting

## Technology Choices

### Backend: FastAPI
- **Why**: Modern, async-capable, auto-generated docs, type hints
- **Alternatives**: Flask (less modern), Django (too heavy)

### Frontend: Next.js
- **Why**: React framework with SSR, good developer experience
- **Alternatives**: Create React App (outdated), Vue (different ecosystem)

### Code Editor: Monaco
- **Why**: VS Code editor, syntax highlighting, line decorations
- **Alternatives**: CodeMirror (less feature-rich), Ace (older)

### Static Analysis: Bandit + Semgrep
- **Why**: Industry standard, good coverage, JSON output
- **Alternatives**: Pylint (general quality, not security-focused), ESLint (similar issue)

## Future ML Integration

The architecture is designed to support ML model integration:

```
Scanner Layer
     ├── BanditScanner (existing)
     ├── SemgrepScanner (existing)
     └── MLScanner (future)
          ├── Graph Builder (AST → DGL graph)
          ├── GNN Model (structural understanding)
          ├── LSTM Model (sequential context)
          └── Hybrid Classifier
```

ML models will:
1. Run **after** static analysis (not replace it)
2. Provide **confidence scores** per line
3. Potentially reduce false positives
4. Trained **offline** on labeled dataset
