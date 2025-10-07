"""Main security analyzer that orchestrates all security checks."""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import SecurityConfig
from .extractors import DependencyChecker, SecretDetector, VulnerabilityScanner
from .llm_analyzer import LLMSecurityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SecurityAnalysis:
    """Results from security analysis of a commit."""

    commit_hash: str
    timestamp: datetime
    files_changed: list[str]

    # Findings by type
    secrets: list[dict]
    vulnerabilities: list[dict]
    dependency_issues: list[dict]
    llm_findings: list[dict]

    # Summary metrics
    total_findings: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int

    # Risk score (0-100)
    risk_score: float


class SecurityAnalyzer:
    """Orchestrates comprehensive security analysis of git commits."""

    def __init__(self, config: Optional[SecurityConfig] = None, repo_path: Optional[Path] = None):
        """Initialize security analyzer.

        Args:
            config: Security configuration
            repo_path: Repository path for context
        """
        self.config = config or SecurityConfig()
        self.repo_path = repo_path or Path.cwd()

        # Initialize components based on configuration
        self.components = []

        if self.config.secret_scanning.enabled:
            self.secret_detector = SecretDetector(
                patterns=self.config.secret_scanning.patterns,
                entropy_threshold=self.config.secret_scanning.entropy_threshold,
                exclude_paths=self.config.secret_scanning.exclude_paths,
            )
            self.components.append(("secrets", self.secret_detector))
        else:
            self.secret_detector = None

        if self.config.vulnerability_scanning.enabled:
            self.vulnerability_scanner = VulnerabilityScanner(self.config.vulnerability_scanning)
            self.components.append(("vulnerabilities", self.vulnerability_scanner))
        else:
            self.vulnerability_scanner = None

        if self.config.dependency_scanning.enabled:
            self.dependency_checker = DependencyChecker(self.config.dependency_scanning)
            self.components.append(("dependencies", self.dependency_checker))
        else:
            self.dependency_checker = None

        if self.config.llm_security.enabled:
            self.llm_analyzer = LLMSecurityAnalyzer(self.config.llm_security)
            self.components.append(("llm", self.llm_analyzer))
        else:
            self.llm_analyzer = None

    def analyze_commit(self, commit_data: dict) -> SecurityAnalysis:
        """Analyze a single commit for security issues.

        Args:
            commit_data: Dictionary with commit information

        Returns:
            SecurityAnalysis object with findings
        """
        findings = {"secrets": [], "vulnerabilities": [], "dependencies": [], "llm": []}

        # Run analyses in parallel for performance
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_scans) as executor:
            futures = []

            # Schedule secret detection
            if self.secret_detector:
                future = executor.submit(self._run_secret_detection, commit_data)
                futures.append(("secrets", future))

            # Schedule vulnerability scanning
            if self.vulnerability_scanner:
                future = executor.submit(self._run_vulnerability_scan, commit_data)
                futures.append(("vulnerabilities", future))

            # Schedule dependency checking
            if self.dependency_checker:
                future = executor.submit(self._run_dependency_check, commit_data)
                futures.append(("dependencies", future))

            # Schedule LLM analysis
            if self.llm_analyzer:
                future = executor.submit(self._run_llm_analysis, commit_data)
                futures.append(("llm", future))

            # Collect results
            for finding_type, future in futures:
                try:
                    result = future.result(timeout=self.config.scan_timeout_seconds)
                    findings[finding_type] = result
                except Exception as e:
                    logger.warning(f"Error in {finding_type} analysis: {e}")

        # Calculate summary metrics
        all_findings = (
            findings["secrets"]
            + findings["vulnerabilities"]
            + findings["dependencies"]
            + findings["llm"]
        )

        severity_counts = self._count_severities(all_findings)
        risk_score = self._calculate_risk_score(severity_counts)

        return SecurityAnalysis(
            commit_hash=commit_data.get("commit_hash", ""),
            timestamp=commit_data.get("timestamp", datetime.now()),
            files_changed=commit_data.get("files_changed", []),
            secrets=findings["secrets"],
            vulnerabilities=findings["vulnerabilities"],
            dependency_issues=findings["dependencies"],
            llm_findings=findings["llm"],
            total_findings=len(all_findings),
            critical_count=severity_counts.get("critical", 0),
            high_count=severity_counts.get("high", 0),
            medium_count=severity_counts.get("medium", 0),
            low_count=severity_counts.get("low", 0),
            risk_score=risk_score,
        )

    def analyze_batch(self, commits: list[dict]) -> list[SecurityAnalysis]:
        """Analyze multiple commits for security issues.

        Args:
            commits: List of commit data dictionaries

        Returns:
            List of SecurityAnalysis objects
        """
        results = []

        for commit in commits:
            try:
                analysis = self.analyze_commit(commit)
                results.append(analysis)

                # Check for critical issues
                if self.config.fail_on_critical and analysis.critical_count > 0:
                    logger.error(
                        f"Critical security issues found in commit {commit.get('commit_hash', '')}"
                    )
                    if self.config.fail_on_critical:
                        raise SecurityException(
                            f"Critical security issues detected: {analysis.critical_count}"
                        )

            except Exception as e:
                logger.error(f"Error analyzing commit {commit.get('commit_hash', '')}: {e}")

        return results

    def _run_secret_detection(self, commit_data: dict) -> list[dict]:
        """Run secret detection on commit."""
        try:
            return self.secret_detector.scan_commit(commit_data)
        except Exception as e:
            logger.warning(f"Secret detection error: {e}")
            return []

    def _run_vulnerability_scan(self, commit_data: dict) -> list[dict]:
        """Run vulnerability scanning on changed files."""
        try:
            files_changed = commit_data.get("files_changed", [])
            return self.vulnerability_scanner.scan_files(files_changed, self.repo_path)
        except Exception as e:
            logger.warning(f"Vulnerability scanning error: {e}")
            return []

    def _run_dependency_check(self, commit_data: dict) -> list[dict]:
        """Check for vulnerable dependencies."""
        try:
            files_changed = commit_data.get("files_changed", [])
            return self.dependency_checker.check_files(files_changed, self.repo_path)
        except Exception as e:
            logger.warning(f"Dependency checking error: {e}")
            return []

    def _run_llm_analysis(self, commit_data: dict) -> list[dict]:
        """Run LLM-based security analysis."""
        try:
            return self.llm_analyzer.analyze_commit(commit_data)
        except Exception as e:
            logger.warning(f"LLM analysis error: {e}")
            return []

    def _count_severities(self, findings: list[dict]) -> dict[str, int]:
        """Count findings by severity level."""
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for finding in findings:
            severity = finding.get("severity", "medium").lower()
            if severity in counts:
                counts[severity] += 1

        return counts

    def _calculate_risk_score(self, severity_counts: dict[str, int]) -> float:
        """Calculate overall risk score (0-100).

        Weighted formula:
        - Critical: 25 points each
        - High: 10 points each
        - Medium: 3 points each
        - Low: 1 point each

        Capped at 100.
        """
        score = (
            severity_counts.get("critical", 0) * 25
            + severity_counts.get("high", 0) * 10
            + severity_counts.get("medium", 0) * 3
            + severity_counts.get("low", 0) * 1
        )

        return min(100.0, float(score))

    def generate_summary_report(self, analyses: list[SecurityAnalysis]) -> dict:
        """Generate summary report from multiple analyses.

        Args:
            analyses: List of SecurityAnalysis objects

        Returns:
            Summary dictionary with statistics and insights
        """
        if not analyses:
            return {
                "total_commits": 0,
                "commits_with_issues": 0,
                "total_findings": 0,
                "risk_assessment": "No data available",
            }

        total_findings = sum(a.total_findings for a in analyses)
        commits_with_issues = sum(1 for a in analyses if a.total_findings > 0)

        # Aggregate findings by type
        all_secrets = []
        all_vulnerabilities = []
        all_dependencies = []
        all_llm = []

        for analysis in analyses:
            all_secrets.extend(analysis.secrets)
            all_vulnerabilities.extend(analysis.vulnerabilities)
            all_dependencies.extend(analysis.dependency_issues)
            all_llm.extend(analysis.llm_findings)

        # Calculate average risk score
        avg_risk_score = sum(a.risk_score for a in analyses) / len(analyses) if analyses else 0

        # Determine risk level
        if avg_risk_score >= 75:
            risk_level = "CRITICAL"
        elif avg_risk_score >= 50:
            risk_level = "HIGH"
        elif avg_risk_score >= 25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        summary = {
            "total_commits": len(analyses),
            "commits_with_issues": commits_with_issues,
            "total_findings": total_findings,
            "average_risk_score": round(avg_risk_score, 2),
            "risk_level": risk_level,
            "findings_by_type": {
                "secrets": len(all_secrets),
                "vulnerabilities": len(all_vulnerabilities),
                "dependency_issues": len(all_dependencies),
                "llm_findings": len(all_llm),
            },
            "severity_distribution": {
                "critical": sum(a.critical_count for a in analyses),
                "high": sum(a.high_count for a in analyses),
                "medium": sum(a.medium_count for a in analyses),
                "low": sum(a.low_count for a in analyses),
            },
            "top_issues": self._identify_top_issues(analyses),
            "recommendations": self._generate_recommendations(analyses),
        }

        # Add LLM insights if available
        if self.llm_analyzer and all_llm:
            all_findings = all_secrets + all_vulnerabilities + all_dependencies + all_llm
            insights = self.llm_analyzer.generate_security_insights(all_findings)
            summary["llm_insights"] = insights

        return summary

    def _identify_top_issues(self, analyses: list[SecurityAnalysis]) -> list[dict]:
        """Identify the most common/critical issues."""
        issue_counts = {}

        for analysis in analyses:
            for finding in (
                analysis.secrets
                + analysis.vulnerabilities
                + analysis.dependency_issues
                + analysis.llm_findings
            ):
                issue_type = finding.get(
                    "vulnerability_type", finding.get("secret_type", "unknown")
                )
                severity = finding.get("severity", "medium")

                key = f"{issue_type}:{severity}"
                if key not in issue_counts:
                    issue_counts[key] = {
                        "type": issue_type,
                        "severity": severity,
                        "count": 0,
                        "files": set(),
                    }

                issue_counts[key]["count"] += 1
                if "file" in finding:
                    issue_counts[key]["files"].add(finding["file"])

        # Sort by severity and count
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_issues = sorted(
            issue_counts.values(),
            key=lambda x: (severity_order.get(x["severity"], 999), -x["count"]),
        )

        # Return top 10 issues
        top_issues = []
        for issue in sorted_issues[:10]:
            top_issues.append(
                {
                    "type": issue["type"],
                    "severity": issue["severity"],
                    "occurrences": issue["count"],
                    "affected_files": len(issue["files"]),
                }
            )

        return top_issues

    def _generate_recommendations(self, analyses: list[SecurityAnalysis]) -> list[str]:
        """Generate actionable security recommendations."""
        recommendations = []

        # Count issue types
        has_secrets = any(a.secrets for a in analyses)
        has_vulnerabilities = any(a.vulnerabilities for a in analyses)
        has_dependencies = any(a.dependency_issues for a in analyses)
        critical_count = sum(a.critical_count for a in analyses)

        if critical_count > 0:
            recommendations.append(
                f"🚨 Address {critical_count} critical security issues immediately"
            )

        if has_secrets:
            recommendations.append("🔑 Implement pre-commit hooks to prevent secret commits")
            recommendations.append("📝 Rotate all exposed credentials and API keys")

        if has_vulnerabilities:
            recommendations.append("🛡️ Enable static analysis tools in CI/CD pipeline")
            recommendations.append("📚 Provide secure coding training for developers")

        if has_dependencies:
            recommendations.append("📦 Update vulnerable dependencies to patched versions")
            recommendations.append("🔄 Implement automated dependency scanning")

        if not recommendations:
            recommendations.append("✅ No significant security issues detected")
            recommendations.append("🔍 Continue regular security reviews")

        return recommendations


class SecurityException(Exception):
    """Exception raised for critical security issues."""

    pass
