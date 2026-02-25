"""
Simple integration example showing how to use the hybrid model
in the existing FastAPI vulnerability scanner.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch

# Import hybrid model components
from app.ml.preprocessing import PreprocessingPipeline
from app.ml.feature_extraction import FeatureExtractor
from app.ml.hybrid_model import HybridVulnerabilityModel

router = APIRouter()

# Initialize components (in production, load from saved model)
extractor = FeatureExtractor(max_seq_length=512, node_feature_dim=64)
model = HybridVulnerabilityModel(
    vocab_size=5000,  # Should match trained model
    node_feature_dim=64
)

# Load trained model weights (when available)
# model.load_state_dict(torch.load('models/hybrid_model.pt'))
model.eval()


class MLPredictionRequest(BaseModel):
    code: str
    language: str
    confidence_threshold: float = 0.5


class MLPredictionResponse(BaseModel):
    is_vulnerable: bool
    confidence: float
    gnn_contribution: float
    lstm_contribution: float
    explanation: str


@router.post("/ml/predict", response_model=MLPredictionResponse)
async def predict_vulnerability(request: MLPredictionRequest):
    """
    Use hybrid GNN+LSTM model to predict vulnerabilities.
    
    This is the ML-only endpoint. For production, combine with
    pattern matching (Semgrep/Bandit) for best results.
    """
    try:
        # Step 1: Preprocess code
        pipeline = PreprocessingPipeline()
        cleaned_code = pipeline._clean_code(request.code, request.language)
        
        # Step 2: Extract features
        graph_data = extractor.code_to_graph(cleaned_code, request.language)
        seq_features = extractor.code_to_sequence(cleaned_code, request.language)
        
        # Step 3: Predict with hybrid model
        with torch.no_grad():
            predictions, gnn_features, lstm_features = model(
                graph_data,
                seq_features.token_ids.unsqueeze(0)
            )
        
        vulnerability_prob = predictions.item()
        is_vulnerable = vulnerability_prob >= request.confidence_threshold
        
        # Calculate branch contributions (normalized)
        gnn_norm = torch.norm(gnn_features, p=2).item()
        lstm_norm = torch.norm(lstm_features, p=2).item()
        total_norm = gnn_norm + lstm_norm
        
        gnn_contribution = gnn_norm / total_norm if total_norm > 0 else 0.5
        lstm_contribution = lstm_norm / total_norm if total_norm > 0 else 0.5
        
        # Generate explanation
        if is_vulnerable:
            if gnn_contribution > lstm_contribution:
                explanation = "Detected structural vulnerability patterns in code graph (e.g., unsafe API calls in dangerous contexts)"
            else:
                explanation = "Detected sequential vulnerability patterns in code flow (e.g., user input flowing into unsafe operations)"
        else:
            explanation = "No significant vulnerability patterns detected by ML model"
        
        # Cleanup
        pipeline.cleanup()
        
        return MLPredictionResponse(
            is_vulnerable=is_vulnerable,
            confidence=vulnerability_prob,
            gnn_contribution=gnn_contribution,
            lstm_contribution=lstm_contribution,
            explanation=explanation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML prediction failed: {str(e)}")


@router.get("/ml/model-info")
async def get_model_info():
    """Get information about the loaded ML model"""
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "model_type": "Hybrid GNN+LSTM",
        "total_parameters": total_params,
        "vocabulary_size": extractor.get_vocab_size(),
        "max_sequence_length": extractor.max_seq_length,
        "node_feature_dimension": extractor.node_feature_dim,
        "supported_languages": ["python", "javascript", "typescript"],
        "owasp_coverage": ["A01", "A03", "A04", "A05"]
    }


# Add to main FastAPI app:
# from app.api.v1.ml_endpoints import router as ml_router
# app.include_router(ml_router, prefix="/api/v1", tags=["ml"])
