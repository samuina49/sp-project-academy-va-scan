"""
Infrastructure Security Scanner
Scans Dockerfiles, Kubernetes manifests, and detects secrets

Features:
- Dockerfile security best practices
- docker-compose.yml scanning
- Kubernetes YAML scanning
- Secret/credential detection
- Infrastructure-as-Code scanning
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class InfraType(str, Enum):
    DOCKERFILE = "dockerfile"
    DOCKER_COMPOSE = "docker-compose"
    KUBERNETES = "kubernetes"
    TERRAFORM = "terraform"
    GENERIC = "generic"


@dataclass
class InfraFinding:
    line: int
    type: str
    severity: str
    message: str
    rule_id: str
    remediation: str


# ==================== Dockerfile Security Rules ====================

DOCKERFILE_RULES = [
    {
        "id": "DOCKER001",
        "pattern": r"^FROM\s+.*:latest\s*$",
        "type": "Insecure Base Image",
        "severity": "MEDIUM",
        "message": "Using 'latest' tag makes builds non-reproducible and may introduce vulnerabilities",
        "remediation": "Pin to a specific image version (e.g., python:3.10-slim)"
    },
    {
        "id": "DOCKER002",
        "pattern": r"^USER\s+root\s*$",
        "type": "Root User",
        "severity": "HIGH",
        "message": "Container runs as root user, which is a security risk",
        "remediation": "Create and use a non-root user (USER appuser)"
    },
    {
        "id": "DOCKER003",
        "pattern": r"^RUN\s+.*curl\s+.*\|\s*(bash|sh)",
        "type": "Insecure Install",
        "severity": "CRITICAL",
        "message": "Piping curl output to shell is dangerous - script content not verified",
        "remediation": "Download script first, verify checksum, then execute"
    },
    {
        "id": "DOCKER004",
        "pattern": r"^RUN\s+.*apt-get\s+install.*-y(?!.*--no-install-recommends)",
        "type": "Large Image",
        "severity": "LOW",
        "message": "Installing packages without --no-install-recommends increases image size",
        "remediation": "Add --no-install-recommends to apt-get install"
    },
    {
        "id": "DOCKER005",
        "pattern": r"^EXPOSE\s+22\s*$",
        "type": "SSH Exposed",
        "severity": "HIGH",
        "message": "Exposing SSH port in container is a security risk",
        "remediation": "Remove SSH from container, use docker exec instead"
    },
    {
        "id": "DOCKER006",
        "pattern": r"^ENV\s+.*(?:PASSWORD|SECRET|API_KEY|TOKEN)\s*=\s*.+",
        "type": "Hardcoded Secret",
        "severity": "CRITICAL",
        "message": "Secrets hardcoded in Dockerfile will be visible in image layers",
        "remediation": "Use build args or runtime environment variables"
    },
    {
        "id": "DOCKER007",
        "pattern": r"^ADD\s+https?://",
        "type": "Remote ADD",
        "severity": "MEDIUM",
        "message": "ADD from URL doesn't verify file integrity",
        "remediation": "Use curl/wget with checksum verification instead"
    },
    {
        "id": "DOCKER008",
        "pattern": r"^RUN\s+chmod\s+777",
        "type": "Permissive Permissions",
        "severity": "HIGH",
        "message": "Setting 777 permissions is overly permissive",
        "remediation": "Use minimal required permissions (e.g., 755 or less)"
    },
    {
        "id": "DOCKER009",
        "pattern": r"^RUN\s+.*sudo\s+",
        "type": "Sudo in Container",
        "severity": "MEDIUM",
        "message": "Using sudo in containers suggests improper privilege management",
        "remediation": "Use USER directive to switch users properly"
    },
    {
        "id": "DOCKER010",
        "pattern": r"^COPY\s+\.\s+",
        "type": "Full Context Copy",
        "severity": "LOW",
        "message": "Copying entire context may include sensitive files",
        "remediation": "Use specific paths and ensure .dockerignore is configured"
    }
]


# ==================== Docker Compose Security Rules ====================

COMPOSE_RULES = [
    {
        "id": "COMPOSE001",
        "pattern": r"privileged:\s*true",
        "type": "Privileged Container",
        "severity": "CRITICAL",
        "message": "Privileged containers have full host access",
        "remediation": "Remove privileged flag, use specific capabilities instead"
    },
    {
        "id": "COMPOSE002",
        "pattern": r"network_mode:\s*['\"]?host['\"]?",
        "type": "Host Network",
        "severity": "HIGH",
        "message": "Using host network mode exposes all host ports",
        "remediation": "Use bridge network with specific port mappings"
    },
    {
        "id": "COMPOSE003",
        "pattern": r"pid:\s*['\"]?host['\"]?",
        "type": "Host PID Namespace",
        "severity": "HIGH",
        "message": "Sharing host PID namespace allows container to see host processes",
        "remediation": "Remove pid: host unless absolutely necessary"
    },
    {
        "id": "COMPOSE004",
        "pattern": r"volumes:.*:/.*:rw",
        "type": "Writable Host Mount",
        "severity": "MEDIUM",
        "message": "Writable mount to host filesystem can be dangerous",
        "remediation": "Use read-only mounts (:ro) where possible"
    },
    {
        "id": "COMPOSE005",
        "pattern": r"environment:.*(?:PASSWORD|SECRET|API_KEY)\s*[:=]\s*['\"]?[^$]",
        "type": "Hardcoded Secret",
        "severity": "CRITICAL",
        "message": "Secrets hardcoded in compose file",
        "remediation": "Use environment variable substitution or Docker secrets"
    },
    {
        "id": "COMPOSE006",
        "pattern": r"cap_add:.*SYS_ADMIN",
        "type": "Dangerous Capability",
        "severity": "CRITICAL",
        "message": "SYS_ADMIN capability grants extensive privileges",
        "remediation": "Use more specific capabilities instead"
    },
    {
        "id": "COMPOSE007",
        "pattern": r"security_opt:.*seccomp:unconfined",
        "type": "Disabled Seccomp",
        "severity": "HIGH",
        "message": "Disabling seccomp removes syscall filtering",
        "remediation": "Use a custom seccomp profile if default is too restrictive"
    }
]


# ==================== Kubernetes Security Rules ====================

KUBERNETES_RULES = [
    {
        "id": "K8S001",
        "pattern": r"privileged:\s*true",
        "type": "Privileged Pod",
        "severity": "CRITICAL",
        "message": "Privileged pods can escape container isolation",
        "remediation": "Set privileged: false and use specific securityContext"
    },
    {
        "id": "K8S002",
        "pattern": r"runAsUser:\s*0",
        "type": "Root User",
        "severity": "HIGH",
        "message": "Pod runs as root user",
        "remediation": "Set runAsUser to non-zero UID (e.g., 1000)"
    },
    {
        "id": "K8S003",
        "pattern": r"hostNetwork:\s*true",
        "type": "Host Network",
        "severity": "HIGH",
        "message": "Using host network bypasses network policies",
        "remediation": "Use pod network unless host access is required"
    },
    {
        "id": "K8S004",
        "pattern": r"hostPID:\s*true",
        "type": "Host PID",
        "severity": "HIGH",
        "message": "Sharing host PID namespace is a security risk",
        "remediation": "Set hostPID: false"
    },
    {
        "id": "K8S005",
        "pattern": r"hostIPC:\s*true",
        "type": "Host IPC",
        "severity": "HIGH",
        "message": "Sharing host IPC namespace allows inter-process attacks",
        "remediation": "Set hostIPC: false"
    },
    {
        "id": "K8S006",
        "pattern": r"allowPrivilegeEscalation:\s*true",
        "type": "Privilege Escalation",
        "severity": "HIGH",
        "message": "Container can gain more privileges than parent",
        "remediation": "Set allowPrivilegeEscalation: false"
    },
    {
        "id": "K8S007",
        "pattern": r"readOnlyRootFilesystem:\s*false",
        "type": "Writable Root FS",
        "severity": "MEDIUM",
        "message": "Container can write to root filesystem",
        "remediation": "Set readOnlyRootFilesystem: true, use volumes for writes"
    },
    {
        "id": "K8S008",
        "pattern": r"capabilities:.*add:.*-?\s*(?:ALL|SYS_ADMIN|NET_ADMIN)",
        "type": "Dangerous Capability",
        "severity": "CRITICAL",
        "message": "Adding dangerous capabilities to container",
        "remediation": "Remove dangerous capabilities, add only what's needed"
    },
    {
        "id": "K8S009",
        "pattern": r"image:.*:latest",
        "type": "Latest Tag",
        "severity": "MEDIUM",
        "message": "Using 'latest' tag makes deployments unpredictable",
        "remediation": "Pin to specific image versions"
    },
    {
        "id": "K8S010",
        "pattern": r"(?:password|secret|api[_-]?key|token):\s*['\"]?(?!\\$|\{\{)[a-zA-Z0-9+/=]{8,}",
        "type": "Hardcoded Secret",
        "severity": "CRITICAL",
        "message": "Secrets should not be hardcoded in manifests",
        "remediation": "Use Kubernetes Secrets or external secret management"
    },
    {
        "id": "K8S011",
        "pattern": r"resources:\s*\{\}|#.*no resources",
        "type": "No Resource Limits",
        "severity": "MEDIUM",
        "message": "No resource limits defined, risk of resource exhaustion",
        "remediation": "Define CPU and memory limits"
    }
]


# ==================== Secret Detection Patterns ====================

SECRET_PATTERNS = [
    {
        "id": "SECRET001",
        "pattern": r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?",
        "type": "API Key",
        "severity": "CRITICAL",
        "message": "Potential API key detected",
        "remediation": "Remove secret and use environment variables or secret management"
    },
    {
        "id": "SECRET002",
        "pattern": r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]{8,})['\"]?",
        "type": "Password",
        "severity": "CRITICAL",
        "message": "Potential password detected",
        "remediation": "Remove password and use secure secret management"
    },
    {
        "id": "SECRET003",
        "pattern": r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
        "type": "Private Key",
        "severity": "CRITICAL",
        "message": "Private key detected in code",
        "remediation": "Remove private key, use secure key management"
    },
    {
        "id": "SECRET004",
        "pattern": r"(?i)aws[_-]?(?:secret[_-]?access[_-]?key|access[_-]?key[_-]?id)\s*[:=]\s*['\"]?([A-Z0-9]{20,})['\"]?",
        "type": "AWS Credentials",
        "severity": "CRITICAL",
        "message": "AWS credentials detected",
        "remediation": "Use IAM roles or AWS Secrets Manager"
    },
    {
        "id": "SECRET005",
        "pattern": r"ghp_[a-zA-Z0-9]{36}",
        "type": "GitHub Token",
        "severity": "CRITICAL",
        "message": "GitHub personal access token detected",
        "remediation": "Revoke token and use GitHub Actions secrets"
    },
    {
        "id": "SECRET006",
        "pattern": r"sk-[a-zA-Z0-9]{48}",
        "type": "OpenAI API Key",
        "severity": "CRITICAL",
        "message": "OpenAI API key detected",
        "remediation": "Remove key and use environment variables"
    },
    {
        "id": "SECRET007",
        "pattern": r"(?i)bearer\s+[a-zA-Z0-9\-_\.]{20,}",
        "type": "Bearer Token",
        "severity": "HIGH",
        "message": "Bearer token detected",
        "remediation": "Remove token from code"
    },
    {
        "id": "SECRET008",
        "pattern": r"(?i)(?:mysql|postgres|mongodb)://[^:]+:([^@]+)@",
        "type": "Database Connection String",
        "severity": "CRITICAL",
        "message": "Database credentials in connection string",
        "remediation": "Use environment variables for connection strings"
    },
    {
        "id": "SECRET009",
        "pattern": r"xox[baprs]-[0-9a-zA-Z-]{10,}",
        "type": "Slack Token",
        "severity": "HIGH",
        "message": "Slack token detected",
        "remediation": "Revoke token and use secure storage"
    },
    {
        "id": "SECRET010",
        "pattern": r"(?i)(?:sendgrid|mailgun|twilio)[_-]?(?:api[_-]?key|token)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-\.]{20,}",
        "type": "Service API Key",
        "severity": "HIGH",
        "message": "Third-party service API key detected",
        "remediation": "Move to environment variables or secret manager"
    }
]


# ==================== Scanner Classes ====================

class InfrastructureScanner:
    """Base infrastructure scanner"""
    
    def __init__(self):
        self.findings: List[InfraFinding] = []
    
    def scan(self, content: str, infra_type: InfraType) -> List[InfraFinding]:
        """Scan content based on infrastructure type"""
        self.findings = []
        
        if infra_type == InfraType.DOCKERFILE:
            self._scan_with_rules(content, DOCKERFILE_RULES)
        elif infra_type == InfraType.DOCKER_COMPOSE:
            self._scan_with_rules(content, COMPOSE_RULES)
        elif infra_type == InfraType.KUBERNETES:
            self._scan_with_rules(content, KUBERNETES_RULES)
        
        # Always check for secrets
        self._scan_for_secrets(content)
        
        return self.findings
    
    def _scan_with_rules(self, content: str, rules: List[Dict]):
        """Scan content with a set of rules"""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for rule in rules:
                if re.search(rule['pattern'], line, re.IGNORECASE | re.MULTILINE):
                    self.findings.append(InfraFinding(
                        line=line_num,
                        type=rule['type'],
                        severity=rule['severity'],
                        message=rule['message'],
                        rule_id=rule['id'],
                        remediation=rule['remediation']
                    ))
    
    def _scan_for_secrets(self, content: str):
        """Scan for secrets in content"""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for pattern in SECRET_PATTERNS:
                if re.search(pattern['pattern'], line):
                    self.findings.append(InfraFinding(
                        line=line_num,
                        type=pattern['type'],
                        severity=pattern['severity'],
                        message=pattern['message'],
                        rule_id=pattern['id'],
                        remediation=pattern['remediation']
                    ))
    
    def detect_type(self, filename: str, content: str) -> InfraType:
        """Detect infrastructure file type"""
        filename_lower = filename.lower()
        
        if 'dockerfile' in filename_lower:
            return InfraType.DOCKERFILE
        elif 'docker-compose' in filename_lower or 'compose.y' in filename_lower:
            return InfraType.DOCKER_COMPOSE
        elif any(k in content for k in ['apiVersion:', 'kind:', 'metadata:']):
            return InfraType.KUBERNETES
        elif 'terraform' in filename_lower or filename_lower.endswith('.tf'):
            return InfraType.TERRAFORM
        else:
            return InfraType.GENERIC


class DockerfileScanner(InfrastructureScanner):
    """Specialized Dockerfile scanner"""
    
    def scan_dockerfile(self, content: str) -> List[InfraFinding]:
        """Scan Dockerfile content"""
        findings = self.scan(content, InfraType.DOCKERFILE)
        
        # Additional Dockerfile-specific checks
        self._check_no_user_directive(content)
        self._check_no_healthcheck(content)
        
        return self.findings
    
    def _check_no_user_directive(self, content: str):
        """Check if USER directive is missing"""
        if 'USER ' not in content.upper():
            self.findings.append(InfraFinding(
                line=1,
                type="Missing USER",
                severity="MEDIUM",
                message="No USER directive found, container will run as root",
                rule_id="DOCKER_USER",
                remediation="Add 'USER nonroot' before CMD/ENTRYPOINT"
            ))
    
    def _check_no_healthcheck(self, content: str):
        """Check if HEALTHCHECK is missing"""
        if 'HEALTHCHECK' not in content.upper():
            self.findings.append(InfraFinding(
                line=1,
                type="No Healthcheck",
                severity="LOW",
                message="No HEALTHCHECK instruction found",
                rule_id="DOCKER_HEALTH",
                remediation="Add HEALTHCHECK instruction for container health monitoring"
            ))


class SecretScanner:
    """Specialized secret detection scanner"""
    
    def __init__(self):
        self.patterns = SECRET_PATTERNS
    
    def scan(self, content: str) -> List[InfraFinding]:
        """Scan for secrets"""
        findings = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('//'):
                continue
            
            for pattern in self.patterns:
                match = re.search(pattern['pattern'], line)
                if match:
                    findings.append(InfraFinding(
                        line=line_num,
                        type=pattern['type'],
                        severity=pattern['severity'],
                        message=pattern['message'],
                        rule_id=pattern['id'],
                        remediation=pattern['remediation']
                    ))
        
        return findings
    
    def scan_file(self, content: str, filename: str) -> List[InfraFinding]:
        """Scan file for secrets with context"""
        # Skip binary files, lock files, etc.
        skip_extensions = {'.lock', '.min.js', '.min.css', '.map', '.svg', '.png', '.jpg'}
        
        for ext in skip_extensions:
            if filename.endswith(ext):
                return []
        
        return self.scan(content)


# ==================== Factory Functions ====================

def get_infrastructure_scanner() -> InfrastructureScanner:
    """Get infrastructure scanner instance"""
    return InfrastructureScanner()


def get_dockerfile_scanner() -> DockerfileScanner:
    """Get Dockerfile scanner instance"""
    return DockerfileScanner()


def get_secret_scanner() -> SecretScanner:
    """Get secret scanner instance"""
    return SecretScanner()
