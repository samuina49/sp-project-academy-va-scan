"""
Feedback API for Active Learning
Collects user feedback (false positives/negatives) to improve the model.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import json
from pathlib import Path
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Feedback storage path
FEEDBACK_FILE = Path("data/feedback/user_feedback.jsonl")
FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)


class FeedbackRequest(BaseModel):
    """Request model for user feedback"""
    cwe_id: str
    severity: str
    line_number: int
    code_snippet: str
    description: str
    feedback_type: str  # 'confirmed' or 'false_positive'
    language: str
    is_vulnerable: bool


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    success: bool
    message: str
    feedback_id: str


class FeedbackStats(BaseModel):
    """Statistics about collected feedback"""
    total_feedback: int
    confirmed_count: int
    false_positive_count: int
    cwe_breakdown: dict


@router.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for a vulnerability finding.
    
    This data is collected for Active Learning to:
    - Identify false positives and improve pattern matching
    - Collect labeled data for model retraining
    - Track common false positive patterns
    
    Args:
        request: Feedback details including code snippet and user classification
        
    Returns:
        Success confirmation with feedback ID
    """
    try:
        # Generate unique feedback ID
        feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.code_snippet) % 10000:04d}"
        
        # Create feedback record
        feedback_record = {
            "feedback_id": feedback_id,
            "timestamp": datetime.now().isoformat(),
            "cwe_id": request.cwe_id,
            "severity": request.severity,
            "line_number": request.line_number,
            "code_snippet": request.code_snippet,
            "description": request.description,
            "feedback_type": request.feedback_type,
            "language": request.language,
            "is_vulnerable": request.is_vulnerable,
            # For future model training
            "label": 1 if request.feedback_type == "confirmed" else 0,
        }
        
        # Append to JSONL file (one JSON object per line)
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_record, ensure_ascii=False) + "\n")
        
        logger.info(f"Feedback received: {feedback_id} - {request.feedback_type} for {request.cwe_id}")
        
        return FeedbackResponse(
            success=True,
            message=f"Thank you for your feedback! This helps improve our detection accuracy.",
            feedback_id=feedback_id
        )
        
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")


@router.get("/feedback/stats", response_model=FeedbackStats, tags=["Feedback"])
async def get_feedback_stats():
    """
    Get statistics about collected user feedback.
    
    Returns:
        Summary statistics including total count and breakdown by type/CWE
    """
    try:
        if not FEEDBACK_FILE.exists():
            return FeedbackStats(
                total_feedback=0,
                confirmed_count=0,
                false_positive_count=0,
                cwe_breakdown={}
            )
        
        total = 0
        confirmed = 0
        false_positive = 0
        cwe_counts: dict = {}
        
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    total += 1
                    
                    if record.get("feedback_type") == "confirmed":
                        confirmed += 1
                    else:
                        false_positive += 1
                    
                    cwe = record.get("cwe_id", "UNKNOWN")
                    if cwe not in cwe_counts:
                        cwe_counts[cwe] = {"confirmed": 0, "false_positive": 0}
                    
                    if record.get("feedback_type") == "confirmed":
                        cwe_counts[cwe]["confirmed"] += 1
                    else:
                        cwe_counts[cwe]["false_positive"] += 1
        
        return FeedbackStats(
            total_feedback=total,
            confirmed_count=confirmed,
            false_positive_count=false_positive,
            cwe_breakdown=cwe_counts
        )
        
    except Exception as e:
        logger.error(f"Failed to read feedback stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read feedback: {str(e)}")


@router.get("/feedback/export", tags=["Feedback"])
async def export_feedback():
    """
    Export all collected feedback for model retraining.
    
    Returns:
        List of all feedback records in training-ready format
    """
    try:
        if not FEEDBACK_FILE.exists():
            return {"feedback": [], "count": 0}
        
        feedback_list = []
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    feedback_list.append(json.loads(line))
        
        return {
            "feedback": feedback_list,
            "count": len(feedback_list),
            "export_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to export feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export feedback: {str(e)}")
