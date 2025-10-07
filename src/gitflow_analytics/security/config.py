"""Security configuration module."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SecretScanningConfig:
    """Configuration for secret detection."""

    enabled: bool = True
    patterns: dict[str, str] = field(
        default_factory=lambda: {
            # AWS
            "aws_access_key": r"AKIA[0-9A-Z]{16}",
            "aws_secret_key": r"aws['\"][0-9a-zA-Z/+=]{40}['\"]",
            # GitHub
            "github_token": r"gh[ps]_[a-zA-Z0-9]{36}",
            "github_oauth": r"gho_[a-zA-Z0-9]{36}",
            "github_app_token": r"ghs_[a-zA-Z0-9]{36}",
            # Generic API Keys
            "api_key": r"(api[_-]?key|apikey)(.{0,20})?['\"]([0-9a-zA-Z]{32,45})['\"]",
            "secret": r"(secret|password|passwd|pwd)(.{0,20})?['\"]([0-9a-zA-Z]{8,})['\"]",
            # Private Keys
            "private_key": r"-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----",
            # Database URLs
            "db_url": r"(postgres|postgresql|mysql|mongodb|redis)://[^:]+:[^@]+@[^/]+",
            # JWT
            "jwt": r"eyJ[A-Za-z0-9-_]+\.eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+",
            # Slack
            "slack_token": r"xox[baprs]-[0-9a-zA-Z]{10,48}",
            # Google
            "google_api": r"AIza[0-9A-Za-z\\-_]{35}",
            # Stripe
            "stripe_key": r"(sk|pk)_(test|live)_[0-9a-zA-Z]{24,}",
        }
    )
    entropy_threshold: float = 4.5
    exclude_paths: list[str] = field(
        default_factory=lambda: [
            "*.test.*",
            "*.spec.*",
            "*_test.go",
            "test_*.py",
            "*/tests/*",
            "*/test/*",
            "*.md",
            "*.txt",
        ]
    )


@dataclass
class VulnerabilityScanningConfig:
    """Configuration for vulnerability scanning."""

    enabled: bool = True

    # Tool-specific configurations
    enable_semgrep: bool = True
    semgrep_rules: list[str] = field(
        default_factory=lambda: [
            "auto",  # Use Semgrep's auto configuration
            "p/security-audit",
            "p/owasp-top-ten",
        ]
    )

    enable_bandit: bool = True  # Python
    bandit_severity: str = "medium"  # low, medium, high

    enable_gosec: bool = True  # Go
    enable_eslint_security: bool = True  # JavaScript/TypeScript
    enable_brakeman: bool = False  # Ruby on Rails

    # Custom patterns for quick checks
    vulnerability_patterns: dict[str, str] = field(
        default_factory=lambda: {
            "sql_injection": r"(SELECT|DELETE|INSERT|UPDATE|DROP).*\+.*(?:request|params|input)",
            "command_injection": r"(exec|eval|system|popen|subprocess).*\+.*(?:request|params|input)",
            "xss": r"innerHTML\s*=.*(?:request|params|input)",
            "path_traversal": r"\.\./.*(?:request|params|input)",
            "weak_crypto": r"(md5|sha1|des|rc4)\s*\(",
            "hardcoded_sql": r"(SELECT|DELETE|INSERT|UPDATE).*FROM.*WHERE.*=\s*['\"]",
        }
    )


@dataclass
class DependencyScanningConfig:
    """Configuration for dependency vulnerability scanning."""

    enabled: bool = True
    check_npm: bool = True
    check_pip: bool = True
    check_go: bool = True
    check_ruby: bool = True
    vulnerability_db: str = "ghsa"  # GitHub Security Advisory Database
    severity_threshold: str = "medium"  # low, medium, high, critical


@dataclass
class LLMSecurityConfig:
    """Configuration for LLM-based security analysis."""

    enabled: bool = True
    model: str = "claude-3-haiku-20240307"  # Fast and cost-effective
    api_key: Optional[str] = None

    # LLM analysis prompts
    code_review_prompt: str = """Analyze this code change for security vulnerabilities:

Files changed: {files_changed}
Lines added:
{lines_added}

Focus on:
1. Authentication/authorization issues
2. Input validation problems
3. Data exposure risks
4. Injection vulnerabilities
5. Cryptographic weaknesses
6. Any other security concerns

Provide a brief, specific analysis. If no issues found, state "No security issues detected."
"""

    commit_review_prompt: str = """Review this git commit for security implications:

Message: {message}
Files: {files}
Category: {category}

Identify any security-relevant changes or potential risks. Be concise and specific.
"""

    max_lines_for_llm: int = 500  # Limit lines sent to LLM for cost control
    confidence_threshold: float = 0.7


@dataclass
class SecurityConfig:
    """Main security configuration."""

    enabled: bool = False  # Disabled by default, opt-in

    secret_scanning: SecretScanningConfig = field(default_factory=SecretScanningConfig)
    vulnerability_scanning: VulnerabilityScanningConfig = field(
        default_factory=VulnerabilityScanningConfig
    )
    dependency_scanning: DependencyScanningConfig = field(default_factory=DependencyScanningConfig)
    llm_security: LLMSecurityConfig = field(default_factory=LLMSecurityConfig)

    # Output configuration
    generate_sarif: bool = False  # Generate SARIF format for GitHub Security
    fail_on_critical: bool = False  # Fail analysis if critical issues found

    # Performance
    max_concurrent_scans: int = 4
    scan_timeout_seconds: int = 30

    @classmethod
    def from_dict(cls, data: dict) -> "SecurityConfig":
        """Create SecurityConfig from dictionary."""
        if not data:
            return cls()

        config = cls(enabled=data.get("enabled", False))

        if "secret_scanning" in data:
            config.secret_scanning = SecretScanningConfig(**data["secret_scanning"])

        if "vulnerability_scanning" in data:
            config.vulnerability_scanning = VulnerabilityScanningConfig(
                **data["vulnerability_scanning"]
            )

        if "dependency_scanning" in data:
            config.dependency_scanning = DependencyScanningConfig(**data["dependency_scanning"])

        if "llm_security" in data:
            config.llm_security = LLMSecurityConfig(**data["llm_security"])

        config.generate_sarif = data.get("generate_sarif", False)
        config.fail_on_critical = data.get("fail_on_critical", False)
        config.max_concurrent_scans = data.get("max_concurrent_scans", 4)
        config.scan_timeout_seconds = data.get("scan_timeout_seconds", 30)

        return config
