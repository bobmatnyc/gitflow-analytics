#!/usr/bin/env python3
"""Test script for security analysis functionality."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add src to path for development testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gitflow_analytics.security import SecurityAnalyzer, SecurityConfig
from gitflow_analytics.security.reports import SecurityReportGenerator


def create_test_commit_data():
    """Create sample commit data for testing."""
    return [
        {
            "commit_hash": "abc123def456789",  # pragma: allowlist secret
            "commit_hash_short": "abc123d",
            "message": "Add database connection with hardcoded password",
            "author_name": "John Developer",
            "author_email": "john@example.com",
            "timestamp": datetime.now(timezone.utc),
            "files_changed": ["config/database.php", "app/db.js"],
            "lines_added": 50,
            "lines_deleted": 10,
            "category": "feature",
        },
        {
            "commit_hash": "def456ghi789012",
            "commit_hash_short": "def456g",
            "message": "Update API endpoint with AWS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE",  # pragma: allowlist secret
            "author_name": "Jane Developer",
            "author_email": "jane@example.com",
            "timestamp": datetime.now(timezone.utc) - timedelta(hours=2),
            "files_changed": ["api/endpoints.js", ".env.example"],
            "lines_added": 30,
            "lines_deleted": 5,
            "category": "feature",
        },
        {
            "commit_hash": "ghi789jkl012345",
            "commit_hash_short": "ghi789j",
            "message": "Fix SQL injection vulnerability in user search",
            "author_name": "Bob Security",
            "author_email": "bob@example.com",
            "timestamp": datetime.now(timezone.utc) - timedelta(hours=4),
            "files_changed": ["app/models/user.php", "app/search.php"],
            "lines_added": 25,
            "lines_deleted": 20,
            "category": "bugfix",
        },
        {
            "commit_hash": "jkl012mno345678",
            "commit_hash_short": "jkl012m",
            "message": "Update package.json dependencies",
            "author_name": "Alice Maintainer",
            "author_email": "alice@example.com",
            "timestamp": datetime.now(timezone.utc) - timedelta(days=1),
            "files_changed": ["package.json", "package-lock.json"],
            "lines_added": 100,
            "lines_deleted": 90,
            "category": "maintenance",
        },
    ]


def main():
    """Run security analysis test."""
    print("🔒 GitFlow Analytics Security Analysis Test")
    print("=" * 50)

    # Create test configuration
    config = SecurityConfig(enabled=True, fail_on_critical=False, generate_sarif=True)

    # Enable all scanners for testing
    config.secret_scanning.enabled = True
    config.vulnerability_scanning.enabled = True
    config.dependency_scanning.enabled = True
    config.llm_security.enabled = False  # Disable LLM for test (requires API key)

    # Initialize analyzer
    print("\n📊 Initializing Security Analyzer...")
    analyzer = SecurityAnalyzer(config=config)

    # Get test commit data
    commits = create_test_commit_data()
    print(f"✅ Created {len(commits)} test commits for analysis")

    # Analyze commits
    print("\n🔍 Analyzing commits for security issues...")
    analyses = []
    for commit in commits:
        print(f"  - Analyzing commit {commit['commit_hash_short']}...")
        analysis = analyzer.analyze_commit(commit)
        analyses.append(analysis)

        if analysis.total_findings > 0:
            print(
                f"    Found {analysis.total_findings} issues "
                f"(Critical: {analysis.critical_count}, "
                f"High: {analysis.high_count}, "
                f"Medium: {analysis.medium_count})"
            )

    # Generate summary
    print("\n📈 Generating Security Summary...")
    summary = analyzer.generate_summary_report(analyses)

    # Print summary
    print("\n" + "=" * 50)
    print("SECURITY ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total Commits Analyzed: {summary['total_commits']}")
    print(f"Commits with Issues: {summary['commits_with_issues']}")
    print(f"Total Security Findings: {summary['total_findings']}")
    print(f"Average Risk Score: {summary['average_risk_score']:.1f}/100")
    print(f"Risk Level: {summary['risk_level']}")

    print("\nFindings by Type:")
    for finding_type, count in summary["findings_by_type"].items():
        if count > 0:
            print(f"  - {finding_type}: {count}")

    print("\nSeverity Distribution:")
    for severity, count in summary["severity_distribution"].items():
        if count > 0:
            print(f"  - {severity.upper()}: {count}")

    # Top issues
    if summary["top_issues"]:
        print("\nTop Security Issues:")
        for i, issue in enumerate(summary["top_issues"][:5], 1):
            print(
                f"  {i}. {issue['type']} ({issue['severity']}) - {issue['occurrences']} occurrences"
            )

    # Recommendations
    print("\nRecommendations:")
    for rec in summary["recommendations"][:5]:
        print(f"  {rec}")

    # Generate reports
    print("\n📝 Generating Security Reports...")
    report_gen = SecurityReportGenerator(output_dir=Path("ewtn-test/reports"))
    reports = report_gen.generate_reports(analyses, summary)

    print("\n✅ Security Analysis Complete!")
    print("\nGenerated Reports:")
    for report_type, path in reports.items():
        print(f"  - {report_type.upper()}: {path}")

    # Check for critical issues
    if summary["severity_distribution"].get("critical", 0) > 0:
        print("\n⚠️  WARNING: Critical security issues detected!")
        print("Please review the security reports immediately.")
        return 1  # Non-zero exit code for critical issues

    return 0


if __name__ == "__main__":
    sys.exit(main())
