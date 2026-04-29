/**
 * API client functions
 * Handles communication with the backend API
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Health check endpoint
 */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_URL}/api/v1/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Health check error:', error);
    throw error;
  }
}

/**
 * Scan code for vulnerabilities
 */
export async function scanCode(code: string, language: string) {
  try {
    const response = await fetch(`${API_URL}/api/v1/scan/code`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ code, language }),
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `Scan failed: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Scan error:', error);
    throw error;
  }
}

/**
 * Upload and scan ZIP file
 */
export async function scanZipFile(file: File) {
  try {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_URL}/api/v1/scan/zip`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `ZIP scan failed: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('ZIP scan error:', error);
    throw error;
  }
}

/**
 * Hybrid Pattern + AI scan
 * Uses the hybrid pipeline: pattern matching (Phase 1) + AI refinement (Phase 2)
 */
export async function mlScan(
  codeOrRequest: string | { code: string; language: string; filename?: string },
  language?: string
) {
  try {
    // Handle both formats: mlScan(code, language) or mlScan({ code, language, filename })
    const requestBody = typeof codeOrRequest === 'string' 
      ? { code: codeOrRequest, language: language! }
      : { code: codeOrRequest.code, language: codeOrRequest.language, filename: codeOrRequest.filename };

    const response = await fetch(`${API_URL}/api/v1/hybrid-scan/code`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `Hybrid scan failed: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Hybrid scan error:', error);
    throw error;
  }
}

/**
 * Get scan history
 */
export async function getScanHistory(limit = 10) {
  try {
    const response = await fetch(`${API_URL}/api/v1/scans?limit=${limit}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch history: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Get history error:', error);
    throw error;
  }
}

/**
 * Export scan results
 */
export async function exportScanResults(scanId: string, format: 'json' | 'sarif' | 'pdf' = 'json') {
  try {
    const response = await fetch(`${API_URL}/api/v1/export/${scanId}?format=${format}`, {
      method: 'GET',
    });
    
    if (!response.ok) {
      throw new Error(`Export failed: ${response.status}`);
    }
    
    if (format === 'pdf') {
      return await response.blob();
    }
    
    return await response.json();
  } catch (error) {
    console.error('Export error:', error);
    throw error;
  }
}

/**
 * Hybrid code scan (alias for mlScan)
 * Used by scan page for hybrid AI + pattern detection
 */
export async function scanCodeHybrid(
  codeOrRequest: string | { code: string; language: string; filename?: string },
  language?: string
) {
  return mlScan(codeOrRequest, language);
}

/**
 * Scan ZIP file (alias for scanZipFile)
 * Used by scan page for project scanning
 */
export async function scanZip(file: File) {
  return scanZipFile(file);
}

export default {
  checkHealth,
  scanCode,
  scanZipFile,
  scanZip,
  mlScan,
  scanCodeHybrid,
  getScanHistory,
  exportScanResults,
};
