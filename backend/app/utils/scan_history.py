"""
Scan History & Trend Analysis Database
Tracks vulnerability scans over time for trend analysis

Features:
- SQLite-based persistent storage
- Scan history with full results
- Trend analysis and metrics
- Dashboard API endpoints
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TimeRange(str, Enum):
    """Time range for trend analysis"""
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


@dataclass
class ScanRecord:
    """Single scan record"""
    id: int
    scan_id: str
    timestamp: datetime
    filename: Optional[str]
    language: str
    code_size: int
    vulnerability_count: int
    severity_breakdown: Dict[str, int]
    ml_enabled: bool
    scan_time_ms: int
    findings_json: str


@dataclass
class TrendData:
    """Trend analysis data point"""
    date: str
    total_scans: int
    total_vulnerabilities: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int


class ScanHistoryDB:
    """SQLite database for scan history"""
    
    def __init__(self, db_path: str = "data/scan_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            logger.error(f"Database transaction failed: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Main scans table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id TEXT UNIQUE NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    filename TEXT,
                    language TEXT,
                    code_size INTEGER,
                    vulnerability_count INTEGER DEFAULT 0,
                    critical_count INTEGER DEFAULT 0,
                    high_count INTEGER DEFAULT 0,
                    medium_count INTEGER DEFAULT 0,
                    low_count INTEGER DEFAULT 0,
                    ml_enabled BOOLEAN DEFAULT FALSE,
                    scan_time_ms INTEGER,
                    findings_json TEXT
                )
            ''')
            
            # Findings table for detailed queries
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS findings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id TEXT NOT NULL,
                    line_number INTEGER,
                    vulnerability_type TEXT,
                    severity TEXT,
                    cwe_id TEXT,
                    owasp_category TEXT,
                    message TEXT,
                    confidence REAL,
                    FOREIGN KEY (scan_id) REFERENCES scans(scan_id)
                )
            ''')
            
            # Projects table for project-level tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_name TEXT UNIQUE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_scan_at DATETIME,
                    total_scans INTEGER DEFAULT 0,
                    current_vuln_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create indexes for common queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_scans_timestamp ON scans(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_scans_language ON scans(language)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_findings_type ON findings(vulnerability_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_findings_severity ON findings(severity)')
            
            logger.info(f"Initialized scan history database at {self.db_path}")
    
    def record_scan(
        self,
        scan_id: str,
        filename: Optional[str],
        language: str,
        code_size: int,
        findings: List[Dict],
        ml_enabled: bool,
        scan_time_ms: int
    ) -> int:
        """Record a new scan"""
        # Count by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for finding in findings:
            sev = finding.get('severity', 'low').lower()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert scan record
            cursor.execute('''
                INSERT INTO scans (
                    scan_id, filename, language, code_size,
                    vulnerability_count, critical_count, high_count, medium_count, low_count,
                    ml_enabled, scan_time_ms, findings_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                scan_id, filename, language, code_size,
                len(findings), severity_counts['critical'], severity_counts['high'],
                severity_counts['medium'], severity_counts['low'],
                ml_enabled, scan_time_ms, json.dumps(findings)
            ))
            
            # Insert individual findings
            for finding in findings:
                cursor.execute('''
                    INSERT INTO findings (
                        scan_id, line_number, vulnerability_type, severity,
                        cwe_id, owasp_category, message, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    scan_id,
                    finding.get('line', 0),
                    finding.get('type', 'unknown'),
                    finding.get('severity', 'low'),
                    finding.get('cwe', ''),
                    finding.get('owasp', ''),
                    finding.get('message', ''),
                    finding.get('confidence', 0.0)
                ))
            
            return cursor.lastrowid
    
    def get_scan_by_id(self, scan_id: str) -> Optional[Dict]:
        """Get a specific scan by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM scans WHERE scan_id = ?', (scan_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def get_recent_scans(self, limit: int = 50) -> List[Dict]:
        """Get recent scans"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT scan_id, timestamp, filename, language, 
                       vulnerability_count, critical_count, high_count, 
                       medium_count, low_count, scan_time_ms
                FROM scans 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_trend_data(self, time_range: TimeRange = TimeRange.WEEK) -> List[TrendData]:
        """Get trend data for specified time range"""
        # Calculate date range
        now = datetime.now()
        
        if time_range == TimeRange.DAY:
            start_date = now - timedelta(days=1)
            group_format = '%Y-%m-%d %H:00'
        elif time_range == TimeRange.WEEK:
            start_date = now - timedelta(weeks=1)
            group_format = '%Y-%m-%d'
        elif time_range == TimeRange.MONTH:
            start_date = now - timedelta(days=30)
            group_format = '%Y-%m-%d'
        elif time_range == TimeRange.QUARTER:
            start_date = now - timedelta(days=90)
            group_format = '%Y-%W'
        else:  # YEAR
            start_date = now - timedelta(days=365)
            group_format = '%Y-%m'
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    strftime(?, timestamp) as date,
                    COUNT(*) as total_scans,
                    SUM(vulnerability_count) as total_vulnerabilities,
                    SUM(critical_count) as critical_count,
                    SUM(high_count) as high_count,
                    SUM(medium_count) as medium_count,
                    SUM(low_count) as low_count
                FROM scans
                WHERE timestamp >= ?
                GROUP BY strftime(?, timestamp)
                ORDER BY date
            ''', (group_format, start_date.isoformat(), group_format))
            
            return [
                TrendData(
                    date=row['date'],
                    total_scans=row['total_scans'],
                    total_vulnerabilities=row['total_vulnerabilities'] or 0,
                    critical_count=row['critical_count'] or 0,
                    high_count=row['high_count'] or 0,
                    medium_count=row['medium_count'] or 0,
                    low_count=row['low_count'] or 0
                )
                for row in cursor.fetchall()
            ]
    
    def get_vulnerability_statistics(self) -> Dict:
        """Get overall vulnerability statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total counts
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_scans,
                    SUM(vulnerability_count) as total_vulns,
                    SUM(critical_count) as critical,
                    SUM(high_count) as high,
                    SUM(medium_count) as medium,
                    SUM(low_count) as low,
                    AVG(scan_time_ms) as avg_scan_time
                FROM scans
            ''')
            totals = dict(cursor.fetchone())
            
            # Top vulnerability types
            cursor.execute('''
                SELECT vulnerability_type, COUNT(*) as count
                FROM findings
                GROUP BY vulnerability_type
                ORDER BY count DESC
                LIMIT 10
            ''')
            top_types = [dict(row) for row in cursor.fetchall()]
            
            # Vulnerability by language
            cursor.execute('''
                SELECT language, SUM(vulnerability_count) as count
                FROM scans
                GROUP BY language
                ORDER BY count DESC
            ''')
            by_language = [dict(row) for row in cursor.fetchall()]
            
            return {
                'totals': totals,
                'top_vulnerability_types': top_types,
                'by_language': by_language
            }
    
    def get_file_history(self, filename: str) -> List[Dict]:
        """Get scan history for a specific file"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT scan_id, timestamp, vulnerability_count,
                       critical_count, high_count, medium_count, low_count
                FROM scans
                WHERE filename = ?
                ORDER BY timestamp DESC
                LIMIT 100
            ''', (filename,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_mttr_metrics(self) -> Dict:
        """Calculate Mean Time To Remediate (MTTR) metrics"""
        # This would require tracking when vulnerabilities are fixed
        # For now, return placeholder metrics
        return {
            'mttr_days': 0,
            'fix_rate': 0,
            'recurring_vulns': 0,
            'note': 'MTTR tracking requires vulnerability remediation data'
        }
    
    def cleanup_old_records(self, days: int = 90):
        """Clean up records older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get scan IDs to delete
            cursor.execute(
                'SELECT scan_id FROM scans WHERE timestamp < ?',
                (cutoff_date.isoformat(),)
            )
            scan_ids = [row['scan_id'] for row in cursor.fetchall()]
            
            if scan_ids:
                # Delete findings
                cursor.execute(
                    f"DELETE FROM findings WHERE scan_id IN ({','.join('?' * len(scan_ids))})",
                    scan_ids
                )
                
                # Delete scans
                cursor.execute('DELETE FROM scans WHERE timestamp < ?', (cutoff_date.isoformat(),))
                
                logger.info(f"Cleaned up {len(scan_ids)} old scan records")
            
            return len(scan_ids)


# Global instance
_db_instance: Optional[ScanHistoryDB] = None


def get_scan_history_db() -> ScanHistoryDB:
    """Get or create the scan history database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = ScanHistoryDB()
    return _db_instance
