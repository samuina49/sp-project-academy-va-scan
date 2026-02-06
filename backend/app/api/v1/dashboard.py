"""
Dashboard API Endpoints for Historical Tracking
Provides trend analysis and statistics
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

from ...utils.scan_history import get_scan_history_db, TimeRange, TrendData

router = APIRouter()


# ==================== Response Models ====================

class ScanSummary(BaseModel):
    scan_id: str
    timestamp: str
    filename: Optional[str]
    language: str
    vulnerability_count: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    scan_time_ms: int


class TrendDataResponse(BaseModel):
    date: str
    total_scans: int
    total_vulnerabilities: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int


class DashboardStats(BaseModel):
    total_scans: int
    total_vulnerabilities: int
    critical: int
    high: int
    medium: int
    low: int
    avg_scan_time_ms: float
    top_vulnerability_types: List[Dict]
    by_language: List[Dict]


class FileHistory(BaseModel):
    scan_id: str
    timestamp: str
    vulnerability_count: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int


# ==================== Endpoints ====================

@router.get("/dashboard/stats", response_model=DashboardStats, tags=["Dashboard"])
async def get_dashboard_stats():
    """
    **Get Dashboard Statistics**
    
    Returns overall statistics including:
    - Total scans and vulnerabilities
    - Breakdown by severity
    - Top vulnerability types
    - Vulnerabilities by language
    """
    db = get_scan_history_db()
    stats = db.get_vulnerability_statistics()
    
    totals = stats.get('totals', {})
    
    return DashboardStats(
        total_scans=totals.get('total_scans', 0) or 0,
        total_vulnerabilities=totals.get('total_vulns', 0) or 0,
        critical=totals.get('critical', 0) or 0,
        high=totals.get('high', 0) or 0,
        medium=totals.get('medium', 0) or 0,
        low=totals.get('low', 0) or 0,
        avg_scan_time_ms=totals.get('avg_scan_time', 0) or 0,
        top_vulnerability_types=stats.get('top_vulnerability_types', []),
        by_language=stats.get('by_language', [])
    )


@router.get("/dashboard/trends", response_model=List[TrendDataResponse], tags=["Dashboard"])
async def get_trends(
    time_range: TimeRange = Query(TimeRange.WEEK, description="Time range for trend data")
):
    """
    **Get Trend Data**
    
    Returns vulnerability trends over time.
    
    Time ranges:
    - day: Hourly data for last 24 hours
    - week: Daily data for last 7 days
    - month: Daily data for last 30 days
    - quarter: Weekly data for last 90 days
    - year: Monthly data for last 365 days
    """
    db = get_scan_history_db()
    trends = db.get_trend_data(time_range)
    
    return [
        TrendDataResponse(
            date=t.date,
            total_scans=t.total_scans,
            total_vulnerabilities=t.total_vulnerabilities,
            critical_count=t.critical_count,
            high_count=t.high_count,
            medium_count=t.medium_count,
            low_count=t.low_count
        )
        for t in trends
    ]


@router.get("/dashboard/recent", response_model=List[ScanSummary], tags=["Dashboard"])
async def get_recent_scans(
    limit: int = Query(50, ge=1, le=500, description="Number of recent scans to return")
):
    """
    **Get Recent Scans**
    
    Returns the most recent scan results.
    """
    db = get_scan_history_db()
    scans = db.get_recent_scans(limit)
    
    return [
        ScanSummary(
            scan_id=s['scan_id'],
            timestamp=str(s['timestamp']),
            filename=s.get('filename'),
            language=s['language'],
            vulnerability_count=s['vulnerability_count'],
            critical_count=s['critical_count'],
            high_count=s['high_count'],
            medium_count=s['medium_count'],
            low_count=s['low_count'],
            scan_time_ms=s['scan_time_ms']
        )
        for s in scans
    ]


@router.get("/dashboard/scan/{scan_id}", tags=["Dashboard"])
async def get_scan_details(scan_id: str):
    """
    **Get Scan Details**
    
    Returns full details for a specific scan, including all findings.
    """
    db = get_scan_history_db()
    scan = db.get_scan_by_id(scan_id)
    
    if not scan:
        return {"error": "Scan not found"}
    
    return scan


@router.get("/dashboard/file-history", response_model=List[FileHistory], tags=["Dashboard"])
async def get_file_history(
    filename: str = Query(..., description="Filename to get history for")
):
    """
    **Get File History**
    
    Returns scan history for a specific file, showing how vulnerabilities
    have changed over time.
    """
    db = get_scan_history_db()
    history = db.get_file_history(filename)
    
    return [
        FileHistory(
            scan_id=h['scan_id'],
            timestamp=str(h['timestamp']),
            vulnerability_count=h['vulnerability_count'],
            critical_count=h['critical_count'],
            high_count=h['high_count'],
            medium_count=h['medium_count'],
            low_count=h['low_count']
        )
        for h in history
    ]


@router.get("/dashboard/mttr", tags=["Dashboard"])
async def get_mttr_metrics():
    """
    **Get MTTR Metrics**
    
    Returns Mean Time To Remediate (MTTR) and related metrics.
    
    Note: MTTR tracking requires vulnerability remediation data,
    which is collected when vulnerabilities are marked as fixed.
    """
    db = get_scan_history_db()
    return db.get_mttr_metrics()


@router.post("/dashboard/cleanup", tags=["Dashboard"])
async def cleanup_old_data(
    days: int = Query(90, ge=7, le=365, description="Delete records older than this many days")
):
    """
    **Cleanup Old Data**
    
    Removes scan records older than the specified number of days.
    This helps manage database size.
    """
    db = get_scan_history_db()
    deleted_count = db.cleanup_old_records(days)
    
    return {
        "status": "success",
        "deleted_records": deleted_count,
        "retention_days": days
    }


@router.get("/dashboard/summary", tags=["Dashboard"])
async def get_summary():
    """
    **Get Dashboard Summary**
    
    Returns a compact summary suitable for dashboard widgets.
    """
    db = get_scan_history_db()
    stats = db.get_vulnerability_statistics()
    trends = db.get_trend_data(TimeRange.WEEK)
    recent = db.get_recent_scans(5)
    
    totals = stats.get('totals', {})
    
    # Calculate week-over-week change
    if len(trends) >= 2:
        prev_week_vulns = sum(t.total_vulnerabilities for t in trends[:len(trends)//2])
        curr_week_vulns = sum(t.total_vulnerabilities for t in trends[len(trends)//2:])
        week_change = ((curr_week_vulns - prev_week_vulns) / max(prev_week_vulns, 1)) * 100
    else:
        week_change = 0
    
    return {
        "overview": {
            "total_scans": totals.get('total_scans', 0) or 0,
            "total_vulnerabilities": totals.get('total_vulns', 0) or 0,
            "week_change_percent": round(week_change, 1)
        },
        "severity": {
            "critical": totals.get('critical', 0) or 0,
            "high": totals.get('high', 0) or 0,
            "medium": totals.get('medium', 0) or 0,
            "low": totals.get('low', 0) or 0
        },
        "recent_scans": recent[:5],
        "trend_direction": "up" if week_change > 0 else "down" if week_change < 0 else "stable"
    }
