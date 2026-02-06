"""
FastAPI application entry point.
AI-Based Vulnerability Scanner for Web Applications
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

# Import existing routers
from app.api.v1 import scan, health, hybrid_scan, ai_scan, feedback, xai, report

# Import new Phase 8-15 routers
from app.api.v1 import auth  # Phase 8: Authentication
from app.api.v1 import project_scan  # Phase 9: Multi-file scanning
from app.api.v1 import dependency_scan  # Phase 10: Dependency scanning
from app.api.v1 import dashboard  # Phase 13: Historical tracking
from app.api.v1 import compliance  # Phase 14: Compliance & reporting
from app.api.v1 import infrastructure  # Phase 15: Infrastructure security


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    AI-Based Vulnerability Scanner for Web Applications
    
    Features:
    - Line-level vulnerability detection
    - ML-powered analysis with GNN+LSTM hybrid model
    - Pattern-based detection for OWASP Top 10
    - Multi-file and project scanning
    - Dependency vulnerability scanning
    - Infrastructure security (Docker, Kubernetes)
    - Compliance reporting (OWASP ASVS, PCI-DSS)
    - SARIF output format
    """,
    debug=settings.DEBUG
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"[REQUEST] {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"[RESPONSE] {request.method} {request.url.path} → {response.status_code}")
    return response

# ==================== Core Routers ====================
app.include_router(health.router, prefix="/api/v1")
app.include_router(scan.router, prefix="/api/v1")
app.include_router(hybrid_scan.router, prefix="/api/v1/scan", tags=["Hybrid Scan"])
app.include_router(ai_scan.router, prefix="/api/v1", tags=["AI Scanner"])
app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])
app.include_router(xai.router, prefix="/api/v1", tags=["Explainable AI"])
app.include_router(report.router, prefix="/api/v1", tags=["Reports"])

# ==================== Phase 8: Authentication ====================
app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])

# ==================== Phase 9: Multi-File Scanning ====================
app.include_router(project_scan.router, prefix="/api/v1", tags=["Project Scanning"])

# ==================== Phase 10: Dependency Scanning ====================
app.include_router(dependency_scan.router, prefix="/api/v1", tags=["Dependency Scanning"])

# ==================== Phase 13: Dashboard & Historical Tracking ====================
app.include_router(dashboard.router, prefix="/api/v1", tags=["Dashboard"])

# ==================== Phase 14: Compliance & Reporting ====================
app.include_router(compliance.router, prefix="/api/v1", tags=["Compliance"])

# ==================== Phase 15: Infrastructure Security ====================
app.include_router(infrastructure.router, prefix="/api/v1", tags=["Infrastructure Security"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
