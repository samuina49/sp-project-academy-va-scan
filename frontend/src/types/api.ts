/**
 * TypeScript type definitions for API responses.
 */

export type SeverityLevel = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'INFO';
export type DetectionSource = 'semgrep' | 'bandit' | 'ml' | 'hybrid';

export interface VulnerabilityFinding {
  owasp_category: any;
  tool: string;
  rule_id: string;
  severity: SeverityLevel;
  message: string;
  start_line: number;
  end_line: number;
  code_snippet?: string;
  cwe_id?: string;
}

export interface HybridFinding {
  line: number;
  end_line: number;
  cwe: string;
  rule_id: string;
  severity: string;
  confidence: string;
  message: string;
  explanation: string;
  code_snippet: string;
  verdict: string;
  ai_score: number;
  ai_available: boolean;
  language: string;
  // Legacy fields (kept for backward compat)
  vulnerability_type?: string;
  sources?: DetectionSource[];
  remediation?: string;
  cwe_id?: string;
  owasp_category?: string;
  semgrep_rule?: string;
  bandit_test?: string;
  ml_probability?: number;
}

export interface FileScanResult {
  file_path: string;
  language: string;
  findings: VulnerabilityFinding[];
  scan_duration_ms?: number;
  source_code?: string;
}

export interface CodeScanResponse {
  scan_id: string;
  file_result: FileScanResult;
  total_findings: number;
  success: boolean;
  error?: string;
}

export interface HybridScanResponse {
  scan_id: string;
  timestamp: string;
  file: string;
  language: string;
  original_language: string;
  total_candidates: number;
  confirmed_vulnerabilities: number;
  false_positives_filtered: number;
  ai_available: boolean;
  scan_duration_ms: number;
  findings: HybridFinding[];
  errors: string[];
}

export interface ZipScanResponse {
  scan_id: string;
  file_results: FileScanResult[];
  total_files_scanned: number;
  total_findings: number;
  scan_duration_ms?: number;
  success: boolean;
  error?: string;
}

export interface CodeScanRequest {
  code: string;
  language: 'python' | 'javascript' | 'typescript';
  filename?: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  bandit_available: boolean;
  semgrep_available: boolean;
  ml_model_available?: boolean;
}
