#!/usr/bin/env python3
"""Standalone security analysis for EWTN repositories."""

import hashlib
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gitflow_analytics.security import SecurityAnalyzer, SecurityConfig
from gitflow_analytics.security.reports import SecurityReportGenerator


def get_commits_from_repo(repo_path: Path, weeks: int = 4):
    """Get commits from a repository using git log."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(weeks=weeks)

    # Get commit data using git log
    cmd = [
        "git",
        "log",
        "--pretty=format:%H|%h|%an|%ae|%at|%s",
        "--name-only",
        f"--since={start_date.isoformat()}",
        "--all",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_path, check=True)
        lines = result.stdout.strip().split("\n")

        commits = []
        current_commit = None

        for line in lines:
            if not line:
                continue

            if "|" in line:
                # Parse commit info
                parts = line.split("|")
                if len(parts) >= 6:
                    current_commit = {
                        "commit_hash": parts[0],
                        "commit_hash_short": parts[1],
                        "author_name": parts[2],
                        "author_email": parts[3],
                        "timestamp": datetime.fromtimestamp(int(parts[4]), tz=timezone.utc),
                        "message": "|".join(parts[5:]),  # Handle messages with pipes
                        "files_changed": [],
                    }
                    commits.append(current_commit)
            elif current_commit:
                # Add file to current commit
                current_commit["files_changed"].append(line)

        return commits
    except subprocess.CalledProcessError as e:
        print(f"Error getting commits: {e}")
        return []


def main():
    """Run security analysis on EWTN repositories."""
    print("🔒 EWTN Security Analysis")
    print("=" * 60)

    # Configure repository path
    repo_path = Path("ewtn-test/repos/aciafrica")
    if not repo_path.exists():
        print(f"❌ Repository not found: {repo_path}")
        return 1

    print(f"📂 Analyzing repository: {repo_path}")

    # Get commits - use last 20 commits instead of date range
    print("📥 Fetching last 20 commits...")

    # Use git log with count instead of date
    import subprocess

    cmd = [
        "git",
        "log",
        "--pretty=format:%H|%h|%an|%ae|%at|%s",
        "--name-only",
        "-n",
        "20",  # Last 20 commits
        "--all",
    ]

    commits = []
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_path, check=True)
        lines = result.stdout.strip().split("\n")

        current_commit = None
        for line in lines:
            if not line:
                continue

            if "|" in line:
                parts = line.split("|")
                if len(parts) >= 6:
                    current_commit = {
                        "commit_hash": parts[0],
                        "commit_hash_short": parts[1],
                        "author_name": parts[2],
                        "author_email": parts[3],
                        "timestamp": datetime.fromtimestamp(int(parts[4]), tz=timezone.utc),
                        "message": "|".join(parts[5:]),
                        "files_changed": [],
                    }
                    commits.append(current_commit)
            elif current_commit:
                current_commit["files_changed"].append(line)
    except Exception as e:
        print(f"Error: {e}")

    print(f"✅ Found {len(commits)} commits")

    if not commits:
        print("❌ No commits found to analyze")
        return 1

    # Configure security analysis
    config = SecurityConfig(enabled=True, fail_on_critical=False, generate_sarif=False)

    # Enable basic scanners (no external tools required)
    config.secret_scanning.enabled = True
    config.vulnerability_scanning.enabled = True
    config.vulnerability_scanning.enable_semgrep = False
    config.vulnerability_scanning.enable_bandit = False
    config.dependency_scanning.enabled = True
    config.llm_security.enabled = False  # Skip LLM for now

    # Initialize analyzer
    print("\n🔍 Running security analysis...")
    analyzer = SecurityAnalyzer(config=config, repo_path=repo_path)

    # Analyze commits
    analyses = []
    issues_found = 0

    for i, commit in enumerate(commits, 1):
        print(f"  Analyzing commit {i}/{len(commits)}: {commit['commit_hash_short']}...", end="")
        analysis = analyzer.analyze_commit(commit)
        analyses.append(analysis)

        if analysis.total_findings > 0:
            print(f" ⚠️  {analysis.total_findings} issues")
            issues_found += analysis.total_findings
        else:
            print(" ✅")

    # Generate summary
    print("\n📊 Generating security report...")
    summary = analyzer.generate_summary_report(analyses)

    # Print summary
    print("\n" + "=" * 60)
    print("SECURITY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Repository: {repo_path.name}")
    print("Period: Last 4 weeks")
    print(f"Total Commits: {summary['total_commits']}")
    print(f"Commits with Issues: {summary['commits_with_issues']}")
    print(f"Total Findings: {summary['total_findings']}")
    print(f"Risk Level: {summary['risk_level']} (Score: {summary['average_risk_score']:.1f})")

    if summary["severity_distribution"]["critical"] > 0:
        print(f"\n🔴 Critical: {summary['severity_distribution']['critical']}")
    if summary["severity_distribution"]["high"] > 0:
        print(f"🟠 High: {summary['severity_distribution']['high']}")
    if summary["severity_distribution"]["medium"] > 0:
        print(f"🟡 Medium: {summary['severity_distribution']['medium']}")
    if summary["severity_distribution"]["low"] > 0:
        print(f"🟢 Low: {summary['severity_distribution']['low']}")

    # Generate detailed report
    report_dir = Path("ewtn-test/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_gen = SecurityReportGenerator(output_dir=report_dir)
    reports = report_gen.generate_reports(analyses, summary)

    print("\n📝 Reports Generated:")
    for report_type, path in reports.items():
        print(f"  - {report_type.upper()}: {path.name}")

    # Create qualitative security report with unique ID
    report_id = hashlib.sha256(f"{datetime.now().isoformat()}".encode()).hexdigest()[:8]
    qualitative_report_path = report_dir / f"security_qualitative_report_{report_id}.md"

    with open(qualitative_report_path, "w") as f:
        f.write("# 🔒 Security Qualitative Analysis Report\n\n")
        f.write(f"**Report ID**: {report_id}\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Repository**: {repo_path.name}\n")
        f.write("**Analysis Period**: Last 4 weeks\n\n")

        f.write("## Executive Summary\n\n")
        f.write(
            f"The security analysis of the {repo_path.name} repository reveals a **{summary['risk_level']}** "
        )
        f.write(
            f"risk profile with an average score of {summary['average_risk_score']:.1f}/100.\n\n"
        )

        if summary["total_findings"] > 0:
            f.write(
                f"A total of **{summary['total_findings']} security issues** were identified across "
            )
            f.write(f"{summary['commits_with_issues']} commits. ")

            if summary["severity_distribution"]["critical"] > 0:
                f.write(f"Notably, {summary['severity_distribution']['critical']} critical issues ")
                f.write("require immediate attention.\n\n")
            else:
                f.write("No critical issues were found, indicating good security hygiene.\n\n")
        else:
            f.write("**No security issues were detected** in the analyzed commits, ")
            f.write("demonstrating excellent security practices.\n\n")

        f.write("## Key Findings\n\n")

        if summary["top_issues"]:
            for i, issue in enumerate(summary["top_issues"][:5], 1):
                f.write(f"{i}. **{issue['type'].replace('_', ' ').title()}** ")
                f.write(f"({issue['severity'].upper()}): {issue['occurrences']} occurrences ")
                f.write(f"in {issue['affected_files']} files\n")
        else:
            f.write("- No significant security patterns detected\n")
            f.write("- Code follows secure development practices\n")

        f.write("\n## Recommendations\n\n")
        for rec in summary["recommendations"]:
            f.write(f"- {rec}\n")

        f.write("\n## Risk Assessment Matrix\n\n")
        f.write("| Risk Factor | Assessment | Score |\n")
        f.write("|-------------|------------|-------|\n")
        f.write(
            f"| Overall Risk | {summary['risk_level']} | {summary['average_risk_score']:.1f}/100 |\n"
        )
        f.write(
            f"| Secret Exposure | {'High' if summary['findings_by_type'].get('secrets', 0) > 0 else 'Low'} | "
        )
        f.write(f"{summary['findings_by_type'].get('secrets', 0)} findings |\n")
        f.write(
            f"| Code Vulnerabilities | {'High' if summary['findings_by_type'].get('vulnerabilities', 0) > 0 else 'Low'} | "
        )
        f.write(f"{summary['findings_by_type'].get('vulnerabilities', 0)} findings |\n")
        f.write(
            f"| Dependency Issues | {'High' if summary['findings_by_type'].get('dependency_issues', 0) > 0 else 'Low'} | "
        )
        f.write(f"{summary['findings_by_type'].get('dependency_issues', 0)} findings |\n")

        f.write("\n## Next Steps\n\n")
        f.write("1. Review detailed findings in the generated reports\n")
        f.write("2. Prioritize remediation of critical and high-severity issues\n")
        f.write("3. Implement recommended security improvements\n")
        f.write("4. Schedule regular security reviews (recommended: weekly)\n")
        f.write("5. Consider enabling advanced security tools (SAST, dependency scanning)\n")

        f.write("\n---\n")
        f.write(
            "*This qualitative security report provides strategic insights based on automated analysis "
        )
        f.write(
            "of commit history and code changes. For detailed technical findings, refer to the "
        )
        f.write("accompanying technical reports.*\n")

    print(f"\n✨ Qualitative Security Report: {qualitative_report_path.name}")

    # Show recommendations
    print("\n💡 Top Recommendations:")
    for i, rec in enumerate(summary["recommendations"][:3], 1):
        print(f"  {i}. {rec}")

    print("\n✅ Security analysis completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
