"""
OWASP Top 10 Mapper

Maps vulnerability findings to OWASP Top 10 (2021) categories.
Supports Semgrep rules, Bandit tests, and ML predictions.
"""

import json
import os
from typing import Optional, Dict, List
from pathlib import Path


class OWASPMapper:
    """
    Map vulnerabilities to OWASP Top 10 categories
    """
    
    def __init__(self, rules_file: Optional[str] = None):
        """
        Initialize OWASP mapper
        
        Args:
            rules_file: Path to OWASP rules JSON file
        """
        if rules_file is None:
            # Default to backend/data/owasp_rules.json
            backend_dir = Path(__file__).parent.parent.parent
            rules_file = backend_dir / 'data' / 'owasp_rules.json'
        
        self.rules = self._load_rules(rules_file)
        
        # Build reverse lookups for fast mapping
        self.semgrep_to_owasp = {}
        self.bandit_to_owasp = {}
        self.keyword_to_owasp = {}
        
        self._build_lookups()
    
    def _load_rules(self, rules_file: str) -> Dict:
        """Load OWASP rules from JSON file"""
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: OWASP rules file not found: {rules_file}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid OWASP rules JSON: {e}")
            return {}
    
    def _build_lookups(self):
        """Build reverse lookup dictionaries"""
        for owasp_id, category in self.rules.items():
            # Semgrep patterns
            for pattern in category.get('semgrep_patterns', []):
                self.semgrep_to_owasp[pattern.lower()] = owasp_id
            
            # Bandit tests
            for test_id in category.get('bandit_tests', []):
                self.bandit_to_owasp[test_id] = owasp_id
            
            # ML keywords
            for keyword in category.get('ml_keywords', []):
                self.keyword_to_owasp[keyword.lower()] = owasp_id
    
    def map_semgrep_rule(self, rule_id: str) -> Optional[str]:
        """
        Map Semgrep rule to OWASP category
        
        Args:
            rule_id: Semgrep rule identifier
            
        Returns:
            OWASP category ID or None
        """
        # Try exact match
        rule_lower = rule_id.lower()
        
        # Check each pattern
        for pattern, owasp_id in self.semgrep_to_owasp.items():
            if pattern in rule_lower:
                return owasp_id
        
        # Fallback: check rule name for common patterns
        if 'sql' in rule_lower or 'injection' in rule_lower:
            return 'A03-Injection'
        elif 'auth' in rule_lower or 'session' in rule_lower:
            return 'A07-IdentificationAuthenticationFailures'
        elif 'crypto' in rule_lower or 'hash' in rule_lower or 'password' in rule_lower:
            return 'A02-CryptographicFailures'
        elif 'deserial' in rule_lower or 'pickle' in rule_lower:
            return 'A08-SoftwareDataIntegrityFailures'
        elif 'ssrf' in rule_lower or 'redirect' in rule_lower:
            return 'A10-ServerSideRequestForgery'
        elif 'log' in rule_lower:
            return 'A09-SecurityLoggingMonitoringFailures'
        
        return None
    
    def map_bandit_test(self, test_id: str) -> Optional[str]:
        """
        Map Bandit test to OWASP category
        
        Args:
            test_id: Bandit test identifier (e.g., "B101")
            
        Returns:
            OWASP category ID or None
        """
        return self.bandit_to_owasp.get(test_id)
    
    def map_vulnerability_type(self, vuln_type: str) -> Optional[str]:
        """
        Map vulnerability type to OWASP category
        
        Args:
            vuln_type: Vulnerability type description
            
        Returns:
            OWASP category ID or None
        """
        vuln_lower = vuln_type.lower()
        
        # Check keywords
        for keyword, owasp_id in self.keyword_to_owasp.items():
            if keyword.lower() in vuln_lower:
                return owasp_id
        
        # Common patterns
        if 'injection' in vuln_lower:
            return 'A03-Injection'
        elif 'auth' in vuln_lower or 'session' in vuln_lower:
            return 'A07-IdentificationAuthenticationFailures'
        elif 'crypto' in vuln_lower or 'encrypt' in vuln_lower:
            return 'A02-CryptographicFailures'
        elif 'access' in vuln_lower or 'authorization' in vuln_lower:
            return 'A01-BrokenAccessControl'
        elif 'deserial' in vuln_lower:
            return 'A08-SoftwareDataIntegrityFailures'
        elif 'ssrf' in vuln_lower:
            return 'A10-ServerSideRequestForgery'
        elif 'config' in vuln_lower:
            return 'A05-SecurityMisconfiguration'
        
        return None
    
    def get_category_info(self, owasp_id: str) -> Optional[Dict]:
        """
        Get information about an OWASP category
        
        Args:
            owasp_id: OWASP category identifier
            
        Returns:
            Category information dictionary
        """
        return self.rules.get(owasp_id)
    
    def get_all_categories(self) -> List[Dict]:
        """Get all OWASP categories with their info"""
        return [
            {'id': owasp_id, **info}
            for owasp_id, info in self.rules.items()
        ]
    
    def get_coverage_stats(self, findings: List[Dict]) -> Dict:
        """
        Calculate OWASP coverage statistics
        
        Args:
            findings: List of vulnerability findings
            
        Returns:
            Coverage statistics
        """
        covered_categories = set()
        category_counts = {}
        
        for finding in findings:
            owasp_id = finding.get('owasp_category')
            if owasp_id:
                covered_categories.add(owasp_id)
                category_counts[owasp_id] = category_counts.get(owasp_id, 0) + 1
        
        return {
            'total_categories': len(self.rules),
            'covered_categories': len(covered_categories),
            'coverage_percentage': (len(covered_categories) / len(self.rules) * 100) if self.rules else 0,
            'category_counts': category_counts,
            'covered_ids': list(covered_categories)
        }


# Example usage
if __name__ == '__main__':
    # Initialize mapper
    mapper = OWASPMapper()
    
    print("OWASP Top 10 Categories Loaded:")
    for category in mapper.get_all_categories():
        print(f"  {category['id']}: {category['name']}")
    
    # Test mappings
    print("\nTest Mappings:")
    
    # Semgrep
    semgrep_rule = "python.lang.security.sql-injection"
    owasp = mapper.map_semgrep_rule(semgrep_rule)
    print(f"  Semgrep '{semgrep_rule}' -> {owasp}")
    
    # Bandit
    bandit_test = "B608"
    owasp = mapper.map_bandit_test(bandit_test)
    print(f"  Bandit '{bandit_test}' -> {owasp}")
    
    # Vulnerability type
    vuln = "SQL Injection"
    owasp = mapper.map_vulnerability_type(vuln)
    print(f"  Type '{vuln}' -> {owasp}")
    
    # Coverage stats
    sample_findings = [
        {'owasp_category': 'A03-Injection'},
        {'owasp_category': 'A03-Injection'},
        {'owasp_category': 'A02-CryptographicFailures'}
    ]
    stats = mapper.get_coverage_stats(sample_findings)
    print(f"\nCoverage: {stats['coverage_percentage']:.1f}% ({stats['covered_categories']}/{stats['total_categories']})")
