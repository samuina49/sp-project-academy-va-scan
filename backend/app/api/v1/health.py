"""
Health check and system status endpoints.
"""
from fastapi import APIRouter
import os

from app.models.scan_models import HealthResponse
from app.scanners.scanner_orchestrator import ScannerOrchestrator
from app.core.config import settings


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify service status and scanner availability.
    
    Returns:
        HealthResponse with service status
    """
    scanner_status = ScannerOrchestrator.get_available_scanners()
    
    # Check if ML model is available
    ml_available = False
    if settings.ML_ENABLED:
        ml_model_exists = os.path.exists(settings.ML_MODEL_PATH)
        ml_vocab_exists = os.path.exists(settings.ML_VOCAB_PATH)
        ml_available = ml_model_exists and ml_vocab_exists
    
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        bandit_available=scanner_status.get("bandit", False),
        semgrep_available=scanner_status.get("semgrep", False),
        ml_model_available=ml_available
    )
