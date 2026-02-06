"""
Compliance & Reporting Module
OWASP ASVS, PCI-DSS, and other compliance framework mappings

Features:
- OWASP ASVS requirement mapping
- PCI-DSS requirement mapping
- SARIF output format
- PDF/HTML report generation
"""

from typing import List, Dict, Optional, Set
from pydantic import BaseModel
from enum import Enum
import json
from datetime import datetime


# ==================== OWASP ASVS Mapping ====================

# ASVS v4.0.3 - Verification Requirement Mapping
OWASP_ASVS_MAPPING = {
    "SQL Injection": {
        "asvs_section": "V5.3 Output Encoding and Injection Prevention",
        "requirements": ["5.3.4", "5.3.5"],
        "level": 1,
        "description": "Verify that data selection or database queries use parameterized queries"
    },
    "XSS": {
        "asvs_section": "V5.3 Output Encoding and Injection Prevention",
        "requirements": ["5.3.3"],
        "level": 1,
        "description": "Verify that context-aware output encoding is used"
    },
    "Command Injection": {
        "asvs_section": "V5.3 Output Encoding and Injection Prevention",
        "requirements": ["5.3.8"],
        "level": 1,
        "description": "Verify that OS command injection is prevented"
    },
    "Path Traversal": {
        "asvs_section": "V5.3 Output Encoding and Injection Prevention",
        "requirements": ["5.3.9"],
        "level": 1,
        "description": "Verify that local file inclusion is prevented"
    },
    "LDAP Injection": {
        "asvs_section": "V5.3 Output Encoding and Injection Prevention",
        "requirements": ["5.3.7"],
        "level": 1,
        "description": "Verify that LDAP injection is prevented"
    },
    "XXE": {
        "asvs_section": "V5.5 Deserialization Prevention",
        "requirements": ["5.5.2"],
        "level": 1,
        "description": "Verify that XML processors are configured to prevent XXE"
    },
    "Insecure Deserialization": {
        "asvs_section": "V5.5 Deserialization Prevention",
        "requirements": ["5.5.1", "5.5.3"],
        "level": 1,
        "description": "Verify that serialized objects are integrity checked"
    },
    "Hardcoded Credentials": {
        "asvs_section": "V2.10 Credential Storage",
        "requirements": ["2.10.4"],
        "level": 1,
        "description": "Verify that passwords are not stored in source code"
    },
    "Weak Cryptography": {
        "asvs_section": "V6.2 Algorithms",
        "requirements": ["6.2.1", "6.2.2"],
        "level": 1,
        "description": "Verify that approved cryptographic algorithms are used"
    },
    "SSRF": {
        "asvs_section": "V5.3 Output Encoding and Injection Prevention",
        "requirements": ["5.3.10"],
        "level": 1,
        "description": "Verify that SSRF attacks are prevented"
    },
    "Broken Access Control": {
        "asvs_section": "V4.1 Access Control Design",
        "requirements": ["4.1.1", "4.1.2", "4.1.3"],
        "level": 1,
        "description": "Verify that access control is enforced"
    },
    "Security Misconfiguration": {
        "asvs_section": "V14.2 Dependency",
        "requirements": ["14.2.1", "14.2.2"],
        "level": 1,
        "description": "Verify components are securely configured"
    },
    "Insufficient Logging": {
        "asvs_section": "V7.1 Log Content",
        "requirements": ["7.1.1", "7.1.2"],
        "level": 1,
        "description": "Verify that security events are logged"
    }
}


# ==================== PCI-DSS Mapping ====================

# PCI-DSS v4.0 Requirement Mapping
PCI_DSS_MAPPING = {
    "SQL Injection": {
        "requirement": "6.2.4",
        "category": "Develop and maintain secure systems and software",
        "description": "Software development processes prevent common coding vulnerabilities"
    },
    "XSS": {
        "requirement": "6.2.4",
        "category": "Develop and maintain secure systems and software",
        "description": "Software development processes prevent common coding vulnerabilities"
    },
    "Command Injection": {
        "requirement": "6.2.4",
        "category": "Develop and maintain secure systems and software",
        "description": "Software development processes prevent common coding vulnerabilities"
    },
    "Path Traversal": {
        "requirement": "6.2.4",
        "category": "Develop and maintain secure systems and software",
        "description": "Software development processes prevent common coding vulnerabilities"
    },
    "Hardcoded Credentials": {
        "requirement": "8.6.2",
        "category": "Identify users and authenticate access",
        "description": "Passwords/passphrases are not hard-coded in scripts or software"
    },
    "Weak Cryptography": {
        "requirement": "3.6.1",
        "category": "Protect stored account data",
        "description": "Strong cryptographic keys are used for encryption"
    },
    "Insecure Deserialization": {
        "requirement": "6.2.4",
        "category": "Develop and maintain secure systems and software",
        "description": "Software development processes prevent common coding vulnerabilities"
    },
    "Broken Access Control": {
        "requirement": "7.2.1",
        "category": "Restrict access to system components",
        "description": "Access control systems enforce role-based access"
    },
    "Insufficient Logging": {
        "requirement": "10.2.1",
        "category": "Log and monitor access",
        "description": "Audit logs record security events"
    }
}


# ==================== CWE to OWASP Top 10 2021 Mapping ====================

CWE_TO_OWASP_2021 = {
    "CWE-89": "A03:2021 - Injection",
    "CWE-79": "A03:2021 - Injection",
    "CWE-78": "A03:2021 - Injection",
    "CWE-90": "A03:2021 - Injection",
    "CWE-22": "A01:2021 - Broken Access Control",
    "CWE-611": "A05:2021 - Security Misconfiguration",
    "CWE-502": "A08:2021 - Software and Data Integrity Failures",
    "CWE-798": "A07:2021 - Identification and Authentication Failures",
    "CWE-327": "A02:2021 - Cryptographic Failures",
    "CWE-918": "A10:2021 - Server-Side Request Forgery",
    "CWE-200": "A01:2021 - Broken Access Control",
    "CWE-284": "A01:2021 - Broken Access Control",
    "CWE-306": "A07:2021 - Identification and Authentication Failures",
    "CWE-307": "A07:2021 - Identification and Authentication Failures",
    "CWE-434": "A04:2021 - Insecure Design",
    "CWE-352": "A01:2021 - Broken Access Control",
    "CWE-1021": "A05:2021 - Security Misconfiguration"
}


# ==================== Models ====================

class ComplianceLevel(str, Enum):
    ASVS_L1 = "ASVS Level 1"
    ASVS_L2 = "ASVS Level 2"
    ASVS_L3 = "ASVS Level 3"
    PCI_DSS = "PCI-DSS"


class ComplianceMapping(BaseModel):
    framework: str
    requirement_id: str
    section: str
    description: str
    level: Optional[int] = None


class ComplianceReport(BaseModel):
    scan_id: str
    generated_at: str
    total_findings: int
    compliance_score: float
    frameworks_checked: List[str]
    findings_by_requirement: Dict[str, List[Dict]]
    gaps: List[Dict]
    recommendations: List[str]


# ==================== SARIF Output ====================

class SARIFGenerator:
    """Generate SARIF (Static Analysis Results Interchange Format) output"""
    
    SARIF_VERSION = "2.1.0"
    SCHEMA = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
    
    def __init__(self, tool_name: str = "AI Vulnerability Scanner", version: str = "1.0.0"):
        self.tool_name = tool_name
        self.version = version
    
    def generate(self, findings: List[Dict], source_file: Optional[str] = None) -> Dict:
        """Generate SARIF report from findings"""
        
        rules = self._generate_rules(findings)
        results = self._generate_results(findings, source_file)
        
        sarif = {
            "$schema": self.SCHEMA,
            "version": self.SARIF_VERSION,
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": self.tool_name,
                            "version": self.version,
                            "informationUri": "https://github.com/your-org/ai-vulnerability-scanner",
                            "rules": rules
                        }
                    },
                    "results": results,
                    "invocations": [
                        {
                            "executionSuccessful": True,
                            "endTimeUtc": datetime.utcnow().isoformat() + "Z"
                        }
                    ]
                }
            ]
        }
        
        return sarif
    
    def _generate_rules(self, findings: List[Dict]) -> List[Dict]:
        """Generate SARIF rules from findings"""
        seen_types = set()
        rules = []
        
        for finding in findings:
            vuln_type = finding.get('type', 'unknown')
            if vuln_type not in seen_types:
                seen_types.add(vuln_type)
                
                # Get CWE info
                cwe = finding.get('cwe', '')
                owasp = CWE_TO_OWASP_2021.get(cwe, finding.get('owasp', ''))
                
                rules.append({
                    "id": vuln_type.replace(' ', '_').lower(),
                    "name": vuln_type,
                    "shortDescription": {
                        "text": f"{vuln_type} vulnerability"
                    },
                    "fullDescription": {
                        "text": finding.get('message', f"Potential {vuln_type} vulnerability detected")
                    },
                    "help": {
                        "text": finding.get('suggestion', f"Review and fix {vuln_type} vulnerability"),
                        "markdown": f"### {vuln_type}\n\n" + 
                                   f"**CWE**: {cwe}\n\n" +
                                   f"**OWASP**: {owasp}\n\n" +
                                   f"**Remediation**: {finding.get('suggestion', 'Review the flagged code')}"
                    },
                    "properties": {
                        "tags": [cwe, owasp] if cwe else [],
                        "security-severity": self._get_security_severity(finding.get('severity', 'medium'))
                    }
                })
        
        return rules
    
    def _generate_results(self, findings: List[Dict], source_file: Optional[str]) -> List[Dict]:
        """Generate SARIF results from findings"""
        results = []
        
        for finding in findings:
            result = {
                "ruleId": finding.get('type', 'unknown').replace(' ', '_').lower(),
                "level": self._map_severity_to_level(finding.get('severity', 'medium')),
                "message": {
                    "text": finding.get('message', 'Security vulnerability detected')
                },
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": source_file or finding.get('file', 'unknown')
                            },
                            "region": {
                                "startLine": finding.get('line', 1),
                                "startColumn": finding.get('column', 1)
                            }
                        }
                    }
                ],
                "properties": {
                    "confidence": finding.get('confidence', 0.0),
                    "cwe": finding.get('cwe', ''),
                    "owasp": finding.get('owasp', '')
                }
            }
            
            # Add fix if available
            if finding.get('suggestion'):
                result["fixes"] = [
                    {
                        "description": {
                            "text": finding['suggestion']
                        }
                    }
                ]
            
            results.append(result)
        
        return results
    
    def _map_severity_to_level(self, severity: str) -> str:
        """Map severity to SARIF level"""
        mapping = {
            'critical': 'error',
            'high': 'error',
            'medium': 'warning',
            'low': 'note',
            'info': 'note'
        }
        return mapping.get(severity.lower(), 'warning')
    
    def _get_security_severity(self, severity: str) -> str:
        """Get SARIF security severity score"""
        mapping = {
            'critical': '9.0',
            'high': '7.0',
            'medium': '4.0',
            'low': '1.0',
            'info': '0.0'
        }
        return mapping.get(severity.lower(), '4.0')


# ==================== Compliance Checker ====================

class ComplianceChecker:
    """Check findings against compliance frameworks"""
    
    def __init__(self):
        self.asvs_mapping = OWASP_ASVS_MAPPING
        self.pci_mapping = PCI_DSS_MAPPING
        self.cwe_owasp_mapping = CWE_TO_OWASP_2021
    
    def get_compliance_info(self, finding: Dict) -> Dict[str, ComplianceMapping]:
        """Get compliance mappings for a finding"""
        vuln_type = finding.get('type', '')
        cwe = finding.get('cwe', '')
        
        mappings = {}
        
        # OWASP ASVS mapping
        if vuln_type in self.asvs_mapping:
            asvs = self.asvs_mapping[vuln_type]
            mappings['OWASP_ASVS'] = ComplianceMapping(
                framework="OWASP ASVS 4.0",
                requirement_id=", ".join(asvs['requirements']),
                section=asvs['asvs_section'],
                description=asvs['description'],
                level=asvs['level']
            )
        
        # PCI-DSS mapping
        if vuln_type in self.pci_mapping:
            pci = self.pci_mapping[vuln_type]
            mappings['PCI_DSS'] = ComplianceMapping(
                framework="PCI-DSS 4.0",
                requirement_id=pci['requirement'],
                section=pci['category'],
                description=pci['description']
            )
        
        # OWASP Top 10 mapping
        if cwe in self.cwe_owasp_mapping:
            mappings['OWASP_TOP_10'] = ComplianceMapping(
                framework="OWASP Top 10 2021",
                requirement_id=self.cwe_owasp_mapping[cwe].split(' - ')[0],
                section=self.cwe_owasp_mapping[cwe],
                description=f"Maps to {self.cwe_owasp_mapping[cwe]}"
            )
        
        return mappings
    
    def generate_compliance_report(
        self,
        scan_id: str,
        findings: List[Dict],
        frameworks: List[str] = ["OWASP_ASVS", "PCI_DSS"]
    ) -> ComplianceReport:
        """Generate compliance report from findings"""
        
        findings_by_requirement: Dict[str, List[Dict]] = {}
        covered_requirements: Set[str] = set()
        
        for finding in findings:
            mappings = self.get_compliance_info(finding)
            
            for framework, mapping in mappings.items():
                if framework in frameworks:
                    req_id = f"{framework}:{mapping.requirement_id}"
                    covered_requirements.add(req_id)
                    
                    if req_id not in findings_by_requirement:
                        findings_by_requirement[req_id] = []
                    
                    findings_by_requirement[req_id].append({
                        'type': finding.get('type'),
                        'severity': finding.get('severity'),
                        'line': finding.get('line'),
                        'message': finding.get('message'),
                        'compliance': mapping.dict()
                    })
        
        # Calculate compliance score
        total_requirements = sum(
            len(self.asvs_mapping) if 'OWASP_ASVS' in frameworks else 0,
            len(self.pci_mapping) if 'PCI_DSS' in frameworks else 0
        )
        
        violations = len(findings_by_requirement)
        compliance_score = max(0, ((total_requirements - violations) / max(total_requirements, 1)) * 100)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(findings)
        
        return ComplianceReport(
            scan_id=scan_id,
            generated_at=datetime.utcnow().isoformat(),
            total_findings=len(findings),
            compliance_score=round(compliance_score, 1),
            frameworks_checked=frameworks,
            findings_by_requirement=findings_by_requirement,
            gaps=[],
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, findings: List[Dict]) -> List[str]:
        """Generate remediation recommendations"""
        recommendations = []
        seen_types = set()
        
        for finding in findings:
            vuln_type = finding.get('type', '')
            if vuln_type not in seen_types:
                seen_types.add(vuln_type)
                
                if vuln_type in self.asvs_mapping:
                    asvs = self.asvs_mapping[vuln_type]
                    recommendations.append(
                        f"Implement {asvs['description']} (ASVS {', '.join(asvs['requirements'])})"
                    )
        
        return recommendations[:10]  # Limit to top 10


# ==================== Factory Functions ====================

def get_sarif_generator() -> SARIFGenerator:
    """Get SARIF generator instance"""
    return SARIFGenerator()


def get_compliance_checker() -> ComplianceChecker:
    """Get compliance checker instance"""
    return ComplianceChecker()
