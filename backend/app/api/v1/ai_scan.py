"""
Production-Ready ML Inference Service
Loads trained hybrid model and provides vulnerability scanning
Combines ML detection with pattern-based rules for comprehensive coverage
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
from pathlib import Path
import json
import logging

from app.ml.feature_extraction import FeatureExtractor
from app.ml.hybrid_model import HybridVulnerabilityModel  # ✅ Use NEW trained model
from app.scanners.simple_scanner import SimplePatternScanner

router = APIRouter()
logger = logging.getLogger(__name__)

# Global model cache
_model_cache = {
    "model": None,
    "extractor": None,
    "vocab": None,
    "pattern_scanner": None  # Add pattern scanner
}

def load_model():
    """Load trained model, vocabulary, and pattern scanner (singleton pattern)"""
    if _model_cache["model"] is not None:
        return _model_cache["model"], _model_cache["extractor"], _model_cache["pattern_scanner"]
    
    try:
        model_path = Path("training/models/hybrid_model_best.pth")
        vocab_path = Path("training/models/vocab.json")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")
        
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        # Handle nested structure if present
        if "token_to_id" in vocab_data:
            vocab = vocab_data["token_to_id"]
        else:
            vocab = vocab_data
        
        # Initialize extractor
        extractor = FeatureExtractor(max_seq_length=128)
        extractor.vocab = vocab
        extractor.vocab_size = len(vocab)
        extractor.vocab_frozen = True
        
        # Detect vocab size from checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Get actual vocab size from model weights (HybridVulnerabilityModel structure)
        if 'lstm_branch.embedding.weight' in state_dict:
            actual_vocab_size = state_dict['lstm_branch.embedding.weight'].shape[0]
        else:
            actual_vocab_size = len(vocab)
        
        # ✅ Initialize HybridVulnerabilityModel with correct parameters
        model = HybridVulnerabilityModel(
            vocab_size=actual_vocab_size,
            node_feature_dim=64,
            lstm_embedding_dim=256,
            dropout=0.4  # Match training config
        )
        
        # Load weights
        model.load_state_dict(state_dict)
        model.eval()
        
        # Initialize pattern scanner
        pattern_scanner = SimplePatternScanner()
        
        # Cache
        _model_cache["model"] = model
        _model_cache["extractor"] = extractor
        _model_cache["vocab"] = vocab
        _model_cache["pattern_scanner"] = pattern_scanner
        
        logger.info("✓ Model and Pattern Scanner loaded successfully")
        return model, extractor, pattern_scanner
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


class MLScanRequest(BaseModel):
    code: str
    language: str
    threshold: float = 0.5


class VulnerabilityDetail(BaseModel):
    cwe_id: str
    severity: str
    confidence: float
    message: str
    line: int = 1


class MLScanResponse(BaseModel):
    is_vulnerable: bool
    confidence: float
    vulnerabilities: List[VulnerabilityDetail]
    model_analysis: Dict[str, float]
    explanation: str


@router.post("/ml-scan", response_model=MLScanResponse, tags=["ML Scanner"])
async def ml_scan_endpoint(request: MLScanRequest):
    """
    **AI-Powered Vulnerability Scan**
    
    Uses the trained Hybrid GNN+LSTM model to detect vulnerabilities.
    
    **Supported Languages:**
    - Python
    - JavaScript
    
    **Detection Coverage:**
    - SQL Injection (CWE-89)
    - XSS (CWE-79)
    - Command Injection (CWE-78)
    - Code Injection (CWE-94)
    - Insecure Deserialization (CWE-502)
    - Path Traversal (CWE-22)
    - SSRF (CWE-918)
    - Cryptographic Failures (CWE-327)
    - Security Misconfiguration (CWE-489)
    
    **Detection Mode:** Hybrid (ML + Pattern-Based Rules)
    """
    try:
        # Load model and pattern scanner
        model, extractor, pattern_scanner = load_model()
        
        # Validate language
        lang = request.language.lower()
        processed_code = request.code
        
        # Determine language processing mode
        if lang == 'typescript':
            processing_lang = 'javascript' 
        elif lang in ['python', 'javascript']:
            processing_lang = lang
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {request.language}. Supported: python, javascript, typescript"
            )
        
        # === Phase 1: Pattern-Based Scan ===
        pattern_result = pattern_scanner.scan_code(processed_code, language=lang)
        pattern_vulns = []
        
        # Check if code contains safe patterns (used to override ML false positives)
        is_safe_code = pattern_scanner.is_safe_code(processed_code, lang)
        
        for finding in pattern_result.findings:
            pattern_vulns.append(VulnerabilityDetail(
                cwe_id=finding.cwe_id or "CWE-UNKNOWN",
                severity=finding.severity.value if hasattr(finding.severity, 'value') else str(finding.severity),
                confidence=0.95,  # Pattern match is high confidence
                message=finding.message,
                line=finding.start_line
            ))
        
        # === Phase 2: ML-Based Scan ===
        ml_confidence = 0.0
        gnn_contribution = 0.0
        lstm_contribution = 0.0
        
        try:
            # Extract features
            graph = extractor.code_to_graph(processed_code, language=processing_lang)
            seq = extractor.code_to_sequence(processed_code, language=processing_lang)
            
            if graph and seq:
                # Prepare inputs for CombinedModel
                # GNN inputs
                x = graph.x
                edge_index = graph.edge_index
                batch_tensor = torch.zeros(graph.x.size(0), dtype=torch.long)
                
                # LSTM inputs
                token_ids = seq.token_ids.unsqueeze(0)  # [1, seq_len]
                
                #  ✅ Inference with HybridVulnerabilityModel (new API)
                with torch.no_grad():
                    # HybridVulnerabilityModel.forward(graph_data, token_ids)
                    # Returns: predictions [batch, 1], gnn_features, lstm_features
                    predictions, gnn_feats, lstm_feats = model(graph, token_ids)
                    
                    # Get probability (sigmoid already applied in training, but not in inference)
                    ml_confidence = torch.sigmoid(predictions[0]).item()
                
                # Calculate branch contributions from feature tensors
                if gnn_feats is not None and lstm_feats is not None:
                    gnn_norm = torch.norm(gnn_feats, p=2).item()
                    lstm_norm = torch.norm(lstm_feats, p=2).item()
                    total_norm = gnn_norm + lstm_norm + 1e-8
                    
                    gnn_contribution = gnn_norm / total_norm
                    lstm_contribution = lstm_norm / total_norm
        except Exception as ml_error:
            logger.warning(f"ML scan failed, using pattern-only: {ml_error}")
        
        # === Phase 3: Combine Results ===
        vulnerabilities = pattern_vulns.copy()
        
        # Add ML detection if confident and no patterns found AND not safe code
        ml_is_vulnerable = ml_confidence >= request.threshold
        
        # If code has safe patterns but no vulnerability patterns, trust the safe patterns
        # This reduces false positives from ML model on safe code like SHA256, env variables, etc.
        if is_safe_code and not pattern_vulns:
            # Safe code override - don't add ML vulnerabilities
            ml_is_vulnerable = False
            logger.info("Safe code pattern detected - overriding ML prediction")
        
        if ml_is_vulnerable and not pattern_vulns:
            vulnerabilities.append(VulnerabilityDetail(
                cwe_id="CWE-MULTI",
                severity="HIGH" if ml_confidence > 0.8 else "MEDIUM",
                confidence=ml_confidence,
                message=f"AI model detected potential vulnerability pattern (confidence: {ml_confidence:.2%})",
                line=1
            ))
        
        # Final decision: vulnerable if patterns found OR ML confident
        is_vulnerable = len(vulnerabilities) > 0
        
        # Combined confidence score
        if pattern_vulns:
            final_confidence = max(0.95, ml_confidence)  # Pattern match = high confidence
        else:
            final_confidence = ml_confidence
        
        # Build explanation
        if is_vulnerable:
            if pattern_vulns:
                explanation = f"🔍 Found {len(pattern_vulns)} vulnerability pattern(s) in code. "
                if ml_is_vulnerable:
                    explanation += f"AI model confirms with {ml_confidence:.0%} confidence."
            elif gnn_contribution > 0.6:
                explanation = "⚠️ Structural vulnerability detected: Dangerous API calls or control flow patterns."
            elif lstm_contribution > 0.6:
                explanation = "⚠️ Sequential vulnerability detected: Unsafe data flow patterns."
            else:
                explanation = "⚠️ Hybrid vulnerability detected: Both structural and sequential risk factors."
        else:
            explanation = "✓ No vulnerability patterns detected. Code appears safe."
        
        return MLScanResponse(
            is_vulnerable=is_vulnerable,
            confidence=final_confidence,
            vulnerabilities=vulnerabilities,
            model_analysis={
                "gnn_contribution": round(gnn_contribution, 4),
                "lstm_contribution": round(lstm_contribution, 4),
                "threshold_used": request.threshold,
                "patterns_found": len(pattern_vulns),
                "ml_confidence": round(ml_confidence, 4)
            },
            explanation=explanation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ML scan error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ML scan failed: {str(e)}")


@router.get("/ml-scan/status", tags=["ML Scanner"])
async def ml_model_status():
    """
    **Check ML Model Status**
    
    Returns whether the model is loaded and ready.
    """
    try:
        load_model()
        return {
            "status": "ready",
            "model_loaded": True,
            "vocab_size": len(_model_cache["vocab"]),
            "supported_languages": ["python", "javascript"]
        }
    except Exception as e:
        return {
            "status": "error",
            "model_loaded": False,
            "error": str(e)
        }
