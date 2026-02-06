"""
Explainable AI (XAI) API for Vulnerability Scanner
Provides interpretability for ML model predictions

This module implements:
1. Token Importance: Which tokens contribute most to vulnerability detection
2. Feature Contribution: GNN vs LSTM branch contributions  
3. Code Highlight Map: Line-by-line risk scores for visualization
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import logging
import re

router = APIRouter()
logger = logging.getLogger(__name__)


class XAIRequest(BaseModel):
    code: str
    language: str


class TokenImportance(BaseModel):
    token: str
    position: int
    importance: float
    line: int


class LineRiskScore(BaseModel):
    line_number: int
    code_snippet: str
    risk_score: float
    contributing_tokens: List[str]


class XAIResponse(BaseModel):
    """Explainable AI response with interpretability data"""
    # Overall analysis
    is_vulnerable: bool
    confidence: float
    
    # Branch contributions (why the model decided)
    gnn_contribution: float  # Structural analysis weight
    lstm_contribution: float  # Sequential analysis weight
    interpretation: str  # Human-readable explanation
    
    # Token-level explanations
    top_tokens: List[TokenImportance]  # Most important tokens
    
    # Line-level risk map
    line_risk_scores: List[LineRiskScore]
    
    # Visualization data
    attention_summary: Dict[str, float]


def get_token_importance(code: str, model, extractor, language: str) -> List[TokenImportance]:
    """
    Calculate token importance using gradient-based attribution.
    Uses input gradients to determine which tokens influence the prediction.
    """
    try:
        # Tokenize code
        tokens = extractor.tokenize(code, language)
        if not tokens:
            return []
        
        # Get sequence
        seq = extractor.code_to_sequence(code, language)
        if seq is None:
            return []
        
        graph = extractor.code_to_graph(code, language)
        if graph is None:
            return []
        
        # Enable gradient tracking for embeddings
        batch = graph
        batch.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        token_ids = seq.token_ids.unsqueeze(0)
        
        # Get embedding layer
        embedding_layer = model.lstm_branch.embedding
        
        # Forward pass with gradient tracking
        model.eval()
        embedded = embedding_layer(token_ids)
        embedded.requires_grad_(True)
        embedded.retain_grad()
        
        # Manual forward through LSTM after embedding
        lstm_out, (hidden, cell) = model.lstm_branch.lstm(embedded)
        if model.lstm_branch.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_combined = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            hidden_combined = hidden[-1, :, :]
        
        x = F.relu(model.lstm_branch.fc1(hidden_combined))
        lstm_features = model.lstm_branch.fc2(x)
        
        # GNN features
        gnn_features = model.gnn_branch(batch)
        
        # Fusion
        fused = torch.cat([gnn_features, lstm_features], dim=1)
        fused = model.fusion_layers(fused)
        output = model.classifier(fused)
        
        # Backward pass to get gradients
        output.backward()
        
        # Get gradient importance scores
        if embedded.grad is not None:
            # Sum gradient magnitude across embedding dimension
            importance = embedded.grad.abs().sum(dim=-1).squeeze()
            importance = importance / (importance.max() + 1e-8)  # Normalize
            importance = importance.detach().numpy()
        else:
            # Fallback: use uniform importance
            importance = [0.5] * len(tokens)
        
        # Map tokens to lines
        lines = code.split('\n')
        token_line_map = {}
        for i, token in enumerate(tokens[:len(importance)]):
            # Find which line contains this token
            for line_idx, line in enumerate(lines):
                if token in line:
                    token_line_map[i] = line_idx + 1
                    break
            else:
                token_line_map[i] = 1
        
        # Create token importance list
        result = []
        for i, token in enumerate(tokens[:len(importance)]):
            if i < len(importance):
                result.append(TokenImportance(
                    token=token,
                    position=i,
                    importance=float(importance[i]),
                    line=token_line_map.get(i, 1)
                ))
        
        # Sort by importance and return top 20
        result.sort(key=lambda x: x.importance, reverse=True)
        return result[:20]
        
    except Exception as e:
        logger.warning(f"Token importance calculation failed: {e}")
        return []


def calculate_line_risk_scores(code: str, token_importance: List[TokenImportance]) -> List[LineRiskScore]:
    """
    Aggregate token importance to line-level risk scores.
    """
    lines = code.split('\n')
    line_scores = {}
    line_tokens = {}
    
    for token_imp in token_importance:
        line_num = token_imp.line
        if line_num not in line_scores:
            line_scores[line_num] = []
            line_tokens[line_num] = []
        line_scores[line_num].append(token_imp.importance)
        if token_imp.importance > 0.5:  # Only track high-importance tokens
            line_tokens[line_num].append(token_imp.token)
    
    result = []
    for i, line in enumerate(lines):
        line_num = i + 1
        if line_num in line_scores:
            avg_score = sum(line_scores[line_num]) / len(line_scores[line_num])
            max_score = max(line_scores[line_num])
            # Combined score: weighted avg of mean and max
            risk = 0.3 * avg_score + 0.7 * max_score
        else:
            risk = 0.0
        
        result.append(LineRiskScore(
            line_number=line_num,
            code_snippet=line[:100] if len(line) > 100 else line,
            risk_score=round(risk, 4),
            contributing_tokens=line_tokens.get(line_num, [])[:5]
        ))
    
    return result


def generate_interpretation(gnn_contrib: float, lstm_contrib: float, is_vulnerable: bool, confidence: float) -> str:
    """
    Generate human-readable interpretation of the model's decision.
    """
    if not is_vulnerable:
        return "✅ The AI model analyzed both code structure and token sequences. No vulnerability patterns were detected in either analysis path."
    
    # Determine dominant analysis
    if gnn_contrib > 0.6:
        dominant = "structural"
        detail = "The Graph Neural Network detected suspicious patterns in the code's Abstract Syntax Tree (AST), such as dangerous function calls or risky control flow."
    elif lstm_contrib > 0.6:
        dominant = "sequential"
        detail = "The LSTM detected suspicious token sequences, such as user input flowing into dangerous operations without sanitization."
    else:
        dominant = "hybrid"
        detail = "Both structural (AST patterns) and sequential (token flow) analyses contributed to this detection, indicating multiple risk factors."
    
    confidence_desc = "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
    
    return f"⚠️ **{dominant.title()} Vulnerability Detected** ({confidence_desc} confidence)\n\n{detail}\n\n**GNN (Structure):** {gnn_contrib:.0%} contribution\n**LSTM (Sequence):** {lstm_contrib:.0%} contribution"


@router.post("/explain", response_model=XAIResponse, tags=["Explainable AI"])
async def explain_prediction(request: XAIRequest):
    """
    **Explainable AI Analysis**
    
    Provides interpretability for vulnerability detection:
    
    1. **Feature Contributions**: How much each model branch (GNN vs LSTM) contributed
    2. **Token Importance**: Which code tokens triggered the detection
    3. **Line Risk Scores**: Per-line vulnerability risk for visualization
    4. **Human-Readable Interpretation**: Plain English explanation
    
    Use this endpoint to understand *why* code was flagged as vulnerable.
    """
    try:
        # Import here to avoid circular imports
        from app.api.v1.ai_scan import load_model
        
        model, extractor, pattern_scanner = load_model()
        
        # Validate language
        lang = request.language.lower()
        if lang == 'typescript':
            processing_lang = 'javascript'
        elif lang in ['python', 'javascript']:
            processing_lang = lang
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {request.language}"
            )
        
        # Extract features
        graph = extractor.code_to_graph(request.code, language=processing_lang)
        seq = extractor.code_to_sequence(request.code, language=processing_lang)
        
        if not graph or not seq:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract features from code"
            )
        
        # Prepare batch
        batch = graph
        batch.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        token_ids = seq.token_ids.unsqueeze(0)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            predictions, gnn_features, lstm_features = model(batch, token_ids)
            confidence = predictions.item()
        
        # Calculate branch contributions
        gnn_norm = torch.norm(gnn_features, p=2).item()
        lstm_norm = torch.norm(lstm_features, p=2).item()
        total_norm = gnn_norm + lstm_norm + 1e-8
        
        gnn_contribution = gnn_norm / total_norm
        lstm_contribution = lstm_norm / total_norm
        
        is_vulnerable = confidence >= 0.5
        
        # Get token importance (with gradients enabled)
        token_importance = get_token_importance(request.code, model, extractor, processing_lang)
        
        # Calculate line risk scores
        line_risks = calculate_line_risk_scores(request.code, token_importance)
        
        # Generate interpretation
        interpretation = generate_interpretation(
            gnn_contribution, lstm_contribution, is_vulnerable, confidence
        )
        
        # Build attention summary
        attention_summary = {
            "structural_analysis": round(gnn_contribution, 4),
            "sequential_analysis": round(lstm_contribution, 4),
            "high_risk_tokens": len([t for t in token_importance if t.importance > 0.7]),
            "high_risk_lines": len([l for l in line_risks if l.risk_score > 0.7]),
            "total_tokens_analyzed": len(token_importance)
        }
        
        return XAIResponse(
            is_vulnerable=is_vulnerable,
            confidence=round(confidence, 4),
            gnn_contribution=round(gnn_contribution, 4),
            lstm_contribution=round(lstm_contribution, 4),
            interpretation=interpretation,
            top_tokens=token_importance[:10],  # Top 10
            line_risk_scores=line_risks,
            attention_summary=attention_summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"XAI explanation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.get("/explain/info", tags=["Explainable AI"])
async def xai_info():
    """
    **XAI Feature Information**
    
    Returns documentation about the explainability features.
    """
    return {
        "feature": "Explainable AI (XAI)",
        "description": "Provides interpretability for AI-based vulnerability detection",
        "capabilities": {
            "token_importance": "Identifies which code tokens most influenced the prediction using gradient-based attribution",
            "branch_contribution": "Shows whether structural (GNN) or sequential (LSTM) analysis dominated",
            "line_risk_scores": "Aggregates token importance to line-level for easy visualization",
            "interpretation": "Generates human-readable explanation of the model's decision"
        },
        "methods_used": [
            "Gradient-based Input Attribution",
            "Feature Norm Comparison",
            "Token-to-Line Mapping"
        ],
        "academic_references": [
            "Integrated Gradients (Sundararajan et al., 2017)",
            "Attention Visualization for Code (Hellendoorn et al., 2020)"
        ]
    }
