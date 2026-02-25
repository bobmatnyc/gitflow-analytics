#!/usr/bin/env python3
"""Run security analysis on all EWTN repositories."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gitflow_analytics.security import SecurityAnalyzer, SecurityConfig
from gitflow_analytics.security.reports import SecurityReportGenerator


def get_all_repos(repos_dir: Path):
    """Get all repository directories."""
    repos = []
    for item in repos_dir.iterdir():
        if item.is_dir() and (item / ".git").exists():
            repos.append(item)
    return sorted(repos)


def get_commits_from_repo(repo_path: Path, weeks: int = 4):
    """Get commits from a repository for the specified number of weeks."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(weeks=weeks)

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
        if not result.stdout.strip():
            return []

        lines = result.stdout.strip().split("\n")
        commits = []
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

        return commits
    except subprocess.CalledProcessError:
        return []


def analyze_repository(repo_path: Path, output_dir: Path):
    """Analyze a single repository for security issues."""
    repo_name = repo_path.name
    print(f"\n{'=' * 60}")
    print(f"📂 Analyzing: {repo_name}")
    print(f"{'=' * 60}")

    # Get commits from last 4 weeks
    print("📥 Fetching commits from last 4 weeks...")
    commits = get_commits_from_repo(repo_path, weeks=4)

    if not commits:
        print("⚠️  No commits found in the last 4 weeks")
        return {
            "repository": repo_name,
            "total_commits": 0,
            "commits_with_issues": 0,
            "total_findings": 0,
            "risk_level": "N/A",
            "status": "No recent activity",
        }

    print(f"✅ Found {len(commits)} commits")

    # Configure security analysis
    config = SecurityConfig(enabled=True, fail_on_critical=False, generate_sarif=False)

    # Enable scanners
    config.secret_scanning.enabled = True
    config.vulnerability_scanning.enabled = True
    config.vulnerability_scanning.enable_semgrep = False
    config.vulnerability_scanning.enable_bandit = False
    config.dependency_scanning.enabled = True
    config.llm_security.enabled = False

    # Initialize analyzer
    analyzer = SecurityAnalyzer(config=config, repo_path=repo_path)

    # Analyze commits
    print(f"🔍 Analyzing {len(commits)} commits...")
    analyses = []
    for i, commit in enumerate(commits, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(commits)}")
        analysis = analyzer.analyze_commit(commit)
        analyses.append(analysis)

    # Generate summary
    summary = analyzer.generate_summary_report(analyses)

    # Print summary for this repo
    print(f"\n📊 Results for {repo_name}:")
    print(f"  - Total Commits: {summary['total_commits']}")
    print(f"  - Commits with Issues: {summary['commits_with_issues']}")
    print(f"  - Total Findings: {summary['total_findings']}")
    print(f"  - Risk Level: {summary['risk_level']} (Score: {summary['average_risk_score']:.1f})")

    # Generate reports for this repository
    repo_output_dir = output_dir / repo_name
    repo_output_dir.mkdir(parents=True, exist_ok=True)

    report_gen = SecurityReportGenerator(output_dir=repo_output_dir)
    report_gen.generate_reports(analyses, summary)

    # Generate qualitative report
    report_id = hashlib.sha256(f"{repo_name}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
    qualitative_report_path = repo_output_dir / f"security_qualitative_report_{report_id}.md"

    with open(qualitative_report_path, "w") as f:
        f.write(f"# 🔒 Security Qualitative Analysis Report - {repo_name}\n\n")
        f.write(f"**Report ID**: {report_id}\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Repository**: {repo_name}\n")
        f.write("**Analysis Period**: Last 4 weeks\n")
        f.write(f"**Total Commits Analyzed**: {summary['total_commits']}\n\n")

        f.write("## Executive Summary\n\n")
        f.write(
            f"The security analysis of the **{repo_name}** repository reveals a **{summary['risk_level']}** "
        )
        f.write(
            f"risk profile with an average score of **{summary['average_risk_score']:.1f}/100**.\n\n"
        )

        if summary["total_findings"] > 0:
            f.write(
                f"A total of **{summary['total_findings']} security issues** were identified across "
            )
            f.write(f"**{summary['commits_with_issues']} commits**. ")

            if summary["severity_distribution"]["critical"] > 0:
                f.write(
                    f"**🔴 CRITICAL: {summary['severity_distribution']['critical']} critical issues** "
                )
                f.write("require immediate attention.\n\n")
            elif summary["severity_distribution"]["high"] > 0:
                f.write(
                    f"**🟠 HIGH: {summary['severity_distribution']['high']} high-severity issues** "
                )
                f.write("should be addressed urgently.\n\n")
            else:
                f.write("The issues found are of medium to low severity.\n\n")
        else:
            f.write("**✅ No security issues were detected** in the analyzed commits, ")
            f.write("demonstrating excellent security practices.\n\n")

        f.write("## Security Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total Commits | {summary['total_commits']} |\n")
        f.write(f"| Commits with Issues | {summary['commits_with_issues']} |\n")
        f.write(f"| Total Findings | {summary['total_findings']} |\n")
        f.write(f"| Risk Score | {summary['average_risk_score']:.1f}/100 |\n")
        f.write(f"| Risk Level | **{summary['risk_level']}** |\n\n")

        if (
            summary["severity_distribution"]["critical"]
            + summary["severity_distribution"]["high"]
            + summary["severity_distribution"]["medium"]
            + summary["severity_distribution"]["low"]
            > 0
        ):
            f.write("## Severity Distribution\n\n")
            f.write("| Severity | Count | Impact |\n")
            f.write("|----------|-------|--------|\n")
            if summary["severity_distribution"]["critical"] > 0:
                f.write(
                    f"| 🔴 Critical | {summary['severity_distribution']['critical']} | Immediate action required |\n"
                )
            if summary["severity_distribution"]["high"] > 0:
                f.write(
                    f"| 🟠 High | {summary['severity_distribution']['high']} | Urgent attention needed |\n"
                )
            if summary["severity_distribution"]["medium"] > 0:
                f.write(
                    f"| 🟡 Medium | {summary['severity_distribution']['medium']} | Should be addressed soon |\n"
                )
            if summary["severity_distribution"]["low"] > 0:
                f.write(
                    f"| 🟢 Low | {summary['severity_distribution']['low']} | Monitor and plan fixes |\n"
                )
            f.write("\n")

        if summary["top_issues"]:
            f.write("## Top Security Issues\n\n")
            for i, issue in enumerate(summary["top_issues"][:5], 1):
                severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
                    issue["severity"], "⚪"
                )
                f.write(f"{i}. {severity_emoji} **{issue['type'].replace('_', ' ').title()}** ")
                f.write(f"({issue['severity'].upper()}): {issue['occurrences']} occurrences ")
                f.write(f"across {issue['affected_files']} files\n")
            f.write("\n")

        f.write("## Recommendations\n\n")
        if summary["recommendations"]:
            for rec in summary["recommendations"]:
                f.write(f"- {rec}\n")
        else:
            f.write("- Continue following secure coding practices\n")
            f.write("- Maintain regular security reviews\n")
            f.write("- Keep dependencies updated\n")

        f.write("\n## Risk Assessment\n\n")
        f.write("| Category | Risk Level | Findings |\n")
        f.write("|----------|------------|----------|\n")
        f.write(
            f"| Secrets & Credentials | {'🔴 High' if summary['findings_by_type'].get('secrets', 0) > 0 else '🟢 Low'} | "
        )
        f.write(f"{summary['findings_by_type'].get('secrets', 0)} |\n")
        f.write(
            f"| Code Vulnerabilities | {'🔴 High' if summary['findings_by_type'].get('vulnerabilities', 0) > 0 else '🟢 Low'} | "
        )
        f.write(f"{summary['findings_by_type'].get('vulnerabilities', 0)} |\n")
        f.write(
            f"| Dependency Issues | {'🔴 High' if summary['findings_by_type'].get('dependency_issues', 0) > 0 else '🟢 Low'} | "
        )
        f.write(f"{summary['findings_by_type'].get('dependency_issues', 0)} |\n")

        f.write("\n## Next Steps\n\n")
        if summary["total_findings"] > 0:
            f.write("1. **Immediate**: Address any critical security issues\n")
            f.write("2. **Short-term**: Fix high and medium severity vulnerabilities\n")
            f.write("3. **Ongoing**: Implement security best practices in development workflow\n")
            f.write("4. **Prevention**: Enable pre-commit hooks for security scanning\n")
        else:
            f.write("1. **Continue** current security practices\n")
            f.write("2. **Monitor** for new vulnerabilities in dependencies\n")
            f.write("3. **Regular** security reviews (weekly recommended)\n")
            f.write("4. **Consider** enabling automated security scanning in CI/CD\n")

        f.write("\n---\n")
        f.write(f"*Generated by GitFlow Analytics Security Module | Report ID: {report_id}*\n")

    print(f"  ✅ Qualitative report: {qualitative_report_path.name}")

    return {
        "repository": repo_name,
        "report_id": report_id,
        "total_commits": summary["total_commits"],
        "commits_with_issues": summary["commits_with_issues"],
        "total_findings": summary["total_findings"],
        "risk_level": summary["risk_level"],
        "risk_score": summary["average_risk_score"],
        "critical": summary["severity_distribution"]["critical"],
        "high": summary["severity_distribution"]["high"],
        "medium": summary["severity_distribution"]["medium"],
        "low": summary["severity_distribution"]["low"],
    }


def main():
    """Main execution function."""
    print("🔒 EWTN Security Analysis - All Repositories")
    print("=" * 60)
    print("Analysis Period: Last 4 weeks")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup directories
    repos_dir = Path("ewtn-test/repos")
    output_dir = Path("ewtn-test/security-reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all repositories
    repos = get_all_repos(repos_dir)
    print(f"\n📦 Found {len(repos)} repositories to analyze")

    # Analyze each repository
    all_results = []
    total_findings = 0
    critical_repos = []
    high_risk_repos = []

    for repo in repos:
        result = analyze_repository(repo, output_dir)
        all_results.append(result)
        total_findings += result.get("total_findings", 0)

        if result.get("critical", 0) > 0:
            critical_repos.append(result["repository"])
        if result.get("risk_level") in ["CRITICAL", "HIGH"]:
            high_risk_repos.append(result["repository"])

    # Generate master summary report
    print(f"\n{'=' * 60}")
    print("📊 MASTER SECURITY SUMMARY")
    print(f"{'=' * 60}")

    master_report_id = hashlib.sha256(f"master_{datetime.now().isoformat()}".encode()).hexdigest()[
        :8
    ]
    master_report_path = output_dir / f"master_security_report_{master_report_id}.md"

    with open(master_report_path, "w") as f:
        f.write("# 🔒 Master Security Analysis Report - All EWTN Repositories\n\n")
        f.write(f"**Report ID**: {master_report_id}\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("**Analysis Period**: Last 4 weeks\n")
        f.write(f"**Repositories Analyzed**: {len(all_results)}\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"Security analysis completed for **{len(all_results)} repositories** ")
        f.write(f"with a total of **{total_findings} security findings**.\n\n")

        if critical_repos:
            f.write(
                f"**🔴 CRITICAL ALERT**: {len(critical_repos)} repositories have critical security issues:\n"
            )
            for repo in critical_repos:
                f.write(f"- {repo}\n")
            f.write("\n")

        if high_risk_repos:
            f.write(
                f"**🟠 HIGH RISK**: {len(high_risk_repos)} repositories require urgent attention:\n"
            )
            for repo in high_risk_repos:
                f.write(f"- {repo}\n")
            f.write("\n")

        f.write("## Repository Risk Matrix\n\n")
        f.write(
            "| Repository | Risk Level | Score | Total Issues | Critical | High | Medium | Low | Report ID |\n"
        )
        f.write(
            "|------------|------------|-------|--------------|----------|------|--------|-----|----------|\n"
        )

        # Sort by risk score descending
        sorted_results = sorted(all_results, key=lambda x: x.get("risk_score", 0), reverse=True)

        for result in sorted_results:
            risk_emoji = {
                "CRITICAL": "🔴",
                "HIGH": "🟠",
                "MEDIUM": "🟡",
                "LOW": "🟢",
                "N/A": "⚪",
            }.get(result["risk_level"], "⚪")

            f.write(f"| {result['repository']} ")
            f.write(f"| {risk_emoji} {result['risk_level']} ")
            f.write(f"| {result.get('risk_score', 0):.1f} ")
            f.write(f"| {result['total_findings']} ")
            f.write(f"| {result.get('critical', 0)} ")
            f.write(f"| {result.get('high', 0)} ")
            f.write(f"| {result.get('medium', 0)} ")
            f.write(f"| {result.get('low', 0)} ")
            f.write(f"| {result.get('report_id', 'N/A')} |\n")

        f.write("\n## Statistics\n\n")
        repos_with_issues = sum(1 for r in all_results if r["total_findings"] > 0)
        repos_without_issues = len(all_results) - repos_with_issues

        f.write(f"- **Total Repositories**: {len(all_results)}\n")
        f.write(f"- **Repositories with Issues**: {repos_with_issues}\n")
        f.write(f"- **Clean Repositories**: {repos_without_issues}\n")
        f.write(f"- **Total Security Findings**: {total_findings}\n")
        f.write(
            f"- **Average Findings per Repository**: {total_findings / len(all_results) if all_results else 0:.1f}\n"
        )

        f.write("\n## Priority Actions\n\n")
        if critical_repos:
            f.write("### 🔴 Immediate (Critical)\n")
            f.write("Address critical security issues in:\n")
            for repo in critical_repos:
                f.write(f"- [ ] {repo}\n")
            f.write("\n")

        if high_risk_repos:
            f.write("### 🟠 Urgent (High Risk)\n")
            f.write("Review and fix high-risk vulnerabilities in:\n")
            for repo in high_risk_repos:
                if repo not in critical_repos:
                    f.write(f"- [ ] {repo}\n")
            f.write("\n")

        f.write("### 🟢 Ongoing\n")
        f.write("- [ ] Implement security scanning in CI/CD pipeline\n")
        f.write("- [ ] Regular dependency updates\n")
        f.write("- [ ] Security training for development team\n")
        f.write("- [ ] Weekly security reviews\n")

        f.write("\n---\n")
        f.write(f"*Master Security Report | Report ID: {master_report_id}*\n")

    # Save summary JSON
    summary_json_path = output_dir / f"security_summary_{master_report_id}.json"
    with open(summary_json_path, "w") as f:
        json.dump(
            {
                "report_id": master_report_id,
                "generated": datetime.now().isoformat(),
                "total_repositories": len(all_results),
                "total_findings": total_findings,
                "repositories": all_results,
            },
            f,
            indent=2,
            default=str,
        )

    # Print final summary
    print("\n✅ Analysis Complete!")
    print(f"📊 Total Findings: {total_findings}")
    print(f"📁 Reports saved to: {output_dir}")
    print(f"📄 Master Report: {master_report_path.name}")
    print(f"📋 Summary JSON: {summary_json_path.name}")

    if critical_repos:
        print(f"\n🔴 CRITICAL: {len(critical_repos)} repositories need immediate attention!")
    elif high_risk_repos:
        print(f"\n🟠 HIGH RISK: {len(high_risk_repos)} repositories require urgent review")
    else:
        print("\n✅ Good security posture across all repositories")

    return 0


if __name__ == "__main__":
    sys.exit(main())
