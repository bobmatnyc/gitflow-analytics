"""Secret detection in git commits."""

import logging
import math
import re
from typing import Optional

logger = logging.getLogger(__name__)


class SecretDetector:
    """Detect potential secrets and credentials in code changes."""

    def __init__(
        self,
        patterns: dict[str, str],
        entropy_threshold: float = 4.5,
        exclude_paths: list[str] = None,
    ):
        """Initialize secret detector.

        Args:
            patterns: Dictionary of secret type to regex pattern
            entropy_threshold: Shannon entropy threshold for detecting high-entropy strings
            exclude_paths: List of glob patterns for paths to exclude
        """
        self.patterns = {name: re.compile(pattern) for name, pattern in patterns.items()}
        self.entropy_threshold = entropy_threshold
        self.exclude_paths = exclude_paths or []

        # Common false positive patterns to exclude
        self.false_positive_patterns = [
            re.compile(r"example\.com"),
            re.compile(r"localhost"),
            re.compile(r"127\.0\.0\.1"),
            re.compile(r"test|demo|sample|example", re.IGNORECASE),
            re.compile(r"xxx+|placeholder|your[_-]?api[_-]?key", re.IGNORECASE),
        ]

    def scan_text(self, text: str, file_path: Optional[str] = None) -> list[dict]:
        """Scan text for potential secrets.

        Args:
            text: Text content to scan
            file_path: Optional file path for context

        Returns:
            List of detected secrets with metadata
        """
        findings = []

        # Skip if file should be excluded
        if file_path and self._should_exclude(file_path):
            return findings

        # Check against regex patterns
        for secret_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                secret_value = match.group(0)

                # Skip false positives
                if self._is_false_positive(secret_value):
                    continue

                finding = {
                    "type": "secret",
                    "secret_type": secret_type,
                    "severity": self._get_severity(secret_type),
                    "file": file_path,
                    "line": text[: match.start()].count("\n") + 1,
                    "column": match.start() - text.rfind("\n", 0, match.start()),
                    "match": secret_value[:20] + "..." if len(secret_value) > 20 else secret_value,
                    "confidence": "high",
                }
                findings.append(finding)

        # Check for high-entropy strings (potential secrets)
        for line_num, line in enumerate(text.split("\n"), 1):
            high_entropy_strings = self._find_high_entropy_strings(line)
            for string, entropy in high_entropy_strings:
                if not self._is_false_positive(string):
                    finding = {
                        "type": "secret",
                        "secret_type": "high_entropy_string",
                        "severity": "medium",
                        "file": file_path,
                        "line": line_num,
                        "entropy": round(entropy, 2),
                        "match": string[:20] + "..." if len(string) > 20 else string,
                        "confidence": "medium",
                    }
                    findings.append(finding)

        return findings

    def scan_commit(self, commit_data: dict) -> list[dict]:
        """Scan a commit for secrets.

        Args:
            commit_data: Commit data dictionary with message, files_changed, etc.

        Returns:
            List of security findings
        """
        findings = []

        # Scan commit message
        message_findings = self.scan_text(commit_data.get("message", ""), "commit_message")
        findings.extend(message_findings)

        # For actual file content scanning, we'd need to read the files
        # This is a placeholder for integration with the git diff analysis
        # In practice, you'd get the actual diff content here

        return findings

    def _should_exclude(self, file_path: str) -> bool:
        """Check if file should be excluded from scanning."""
        from fnmatch import fnmatch

        return any(fnmatch(file_path, pattern) for pattern in self.exclude_paths)

    def _is_false_positive(self, value: str) -> bool:
        """Check if a detected secret is likely a false positive."""
        return any(pattern.search(value) for pattern in self.false_positive_patterns)

    def _get_severity(self, secret_type: str) -> str:
        """Determine severity based on secret type."""
        critical_types = ["private_key", "aws_secret_key", "db_url"]
        high_types = ["aws_access_key", "github_token", "api_key", "stripe_key"]

        if secret_type in critical_types:
            return "critical"
        elif secret_type in high_types:
            return "high"
        else:
            return "medium"

    def _calculate_entropy(self, string: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not string:
            return 0.0

        # Count character frequencies
        char_counts = {}
        for char in string:
            char_counts[char] = char_counts.get(char, 0) + 1

        # Calculate entropy
        entropy = 0.0
        length = len(string)
        for count in char_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def _find_high_entropy_strings(
        self, text: str, min_length: int = 20
    ) -> list[tuple[str, float]]:
        """Find strings with high entropy (potential secrets).

        Args:
            text: Text to analyze
            min_length: Minimum string length to consider

        Returns:
            List of (string, entropy) tuples
        """
        high_entropy_strings = []

        # Look for quoted strings and continuous non-space sequences
        patterns = [
            r'"([^"]+)"',  # Double quoted strings
            r"'([^']+)'",  # Single quoted strings
            r"`([^`]+)`",  # Backtick strings
            r"=\s*([^\s;,]+)",  # Values after equals sign
            r':\s*"([^"]+)"',  # JSON-style values
            r":\s*\'([^\']+)\'",  # JSON-style values with single quotes
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                string = match.group(1)
                if len(string) >= min_length:
                    entropy = self._calculate_entropy(string)
                    if entropy >= self.entropy_threshold:
                        high_entropy_strings.append((string, entropy))

        return high_entropy_strings
