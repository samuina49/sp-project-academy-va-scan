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
  vulnerability_type: string;
  severity: SeverityLevel;
  confidence: number; // 0-1
  sources: DetectionSource[];
  code_snippet: string;
  explanation: string;
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
  code_language: string;
  scan_type: string;
  summary: {
    total_findings: number;
    critical: number;
    high: number;
    medium: number;
    low: number;
    info: number;
    owasp_coverage: Record<string, number>;
    detection_sources: {
      semgrep: number;
      bandit: number;
      ml: number;
      hybrid: number;
    };
    ml_enabled: boolean;
  };
  findings: HybridFinding[];
  success: boolean;
  error?: string;
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
