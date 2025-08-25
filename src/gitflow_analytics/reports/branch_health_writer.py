"""Branch health report generation for GitFlow Analytics."""

import csv
import logging
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class BranchHealthReportGenerator:
    """Generate branch health reports in CSV and markdown formats."""

    def __init__(self):
        """Initialize branch health report generator."""
        pass

    def generate_csv_report(
        self, branch_health_metrics: dict[str, dict[str, Any]], output_path: Path
    ) -> Path:
        """Generate CSV report for branch health metrics.

        Args:
            branch_health_metrics: Dictionary mapping repo names to their branch health metrics
            output_path: Path where the CSV should be written

        Returns:
            Path to the generated CSV file
        """
        rows = []

        for repo_name, metrics in branch_health_metrics.items():
            # Add summary row for the repository
            summary = metrics.get("summary", {})
            health = metrics.get("health_indicators", {})

            summary_row = {
                "repository": repo_name,
                "branch_name": "[SUMMARY]",
                "total_branches": summary.get("total_branches", 0),
                "active_branches": summary.get("active_branches", 0),
                "stale_branches": summary.get("stale_branches", 0),
                "long_lived_branches": summary.get("long_lived_branches", 0),
                "overall_health": health.get("overall_health", "unknown"),
                "stale_percentage": health.get("stale_branch_percentage", 0),
                "branch_creation_rate_weekly": summary.get("branch_creation_rate_per_week", 0),
                "average_branch_age_days": summary.get("average_branch_age_days", 0),
                "average_commits_per_branch": summary.get("average_commits_per_branch", 0),
            }
            rows.append(summary_row)

            # Add individual branch rows
            branches = metrics.get("branches", {})
            for branch_name, branch_data in branches.items():
                branch_row = {
                    "repository": repo_name,
                    "branch_name": branch_name,
                    "age_days": branch_data.get("age_days", 0),
                    "is_stale": branch_data.get("is_stale", False),
                    "is_merged": branch_data.get("is_merged", False),
                    "total_commits": branch_data.get("total_commits", 0),
                    "unique_authors": branch_data.get("unique_authors", 0),
                    "ahead_of_main": branch_data.get("ahead_of_main", 0),
                    "behind_main": branch_data.get("behind_main", 0),
                    "divergence_score": branch_data.get("divergence_score", 0),
                    "health_score": branch_data.get("health_score", 0),
                    "latest_activity": branch_data.get("latest_activity", ""),
                    "daily_commit_average": branch_data.get("commit_frequency", {}).get(
                        "daily_average", 0
                    ),
                }
                rows.append(branch_row)

        # Write CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        else:
            # Write empty CSV with headers
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "repository",
                        "branch_name",
                        "total_branches",
                        "active_branches",
                        "stale_branches",
                        "long_lived_branches",
                        "overall_health",
                        "stale_percentage",
                        "branch_creation_rate_weekly",
                        "average_branch_age_days",
                        "average_commits_per_branch",
                        "age_days",
                        "is_stale",
                        "is_merged",
                        "total_commits",
                        "unique_authors",
                        "ahead_of_main",
                        "behind_main",
                        "divergence_score",
                        "health_score",
                        "latest_activity",
                        "daily_commit_average",
                    ],
                )
                writer.writeheader()

        return output_path

    def generate_markdown_section(self, branch_health_metrics: dict[str, dict[str, Any]]) -> str:
        """Generate markdown section for branch health to include in narrative reports.

        Args:
            branch_health_metrics: Dictionary mapping repo names to their branch health metrics

        Returns:
            Markdown formatted string with branch health insights
        """
        if not branch_health_metrics:
            return ""

        report = StringIO()
        report.write("\n## Branch Health Analysis\n\n")

        # Overall summary across all repositories
        total_repos = len(branch_health_metrics)
        total_branches_all = sum(
            m.get("summary", {}).get("total_branches", 0) for m in branch_health_metrics.values()
        )
        total_stale_all = sum(
            m.get("summary", {}).get("stale_branches", 0) for m in branch_health_metrics.values()
        )

        report.write("### Overview\n\n")
        report.write(
            f"Analyzed **{total_repos} repositories** with a total of **{total_branches_all} branches**.\n\n"
        )

        if total_stale_all > 0:
            stale_pct = (
                (total_stale_all / total_branches_all * 100) if total_branches_all > 0 else 0
            )
            report.write(
                f"⚠️ Found **{total_stale_all} stale branches** ({stale_pct:.1f}% of total)\n\n"
            )

        # Repository breakdown
        report.write("### Repository Branch Health\n\n")

        for repo_name, metrics in branch_health_metrics.items():
            summary = metrics.get("summary", {})
            health = metrics.get("health_indicators", {})

            # Repository header
            health_emoji = self._get_health_emoji(health.get("overall_health", "unknown"))
            report.write(f"#### {repo_name} {health_emoji}\n\n")

            # Key metrics
            report.write(f"- **Total Branches**: {summary.get('total_branches', 0)}\n")
            report.write(f"- **Active**: {summary.get('active_branches', 0)}\n")
            report.write(f"- **Stale**: {summary.get('stale_branches', 0)}\n")
            report.write(f"- **Long-lived**: {summary.get('long_lived_branches', 0)}\n")
            report.write(
                f"- **Average Age**: {summary.get('average_branch_age_days', 0):.1f} days\n"
            )
            report.write(
                f"- **Creation Rate**: {summary.get('branch_creation_rate_per_week', 0):.1f} branches/week\n"
            )
            report.write(
                f"- **Health Status**: {health.get('overall_health', 'unknown').title()}\n\n"
            )

            # Top unhealthy branches
            branches = metrics.get("branches", {})
            unhealthy_branches = [
                (name, data)
                for name, data in branches.items()
                if data.get("health_score", 100) < 60 and not data.get("is_merged", False)
            ]

            if unhealthy_branches:
                report.write("**Branches Needing Attention**:\n")
                # Sort by health score (lowest first)
                unhealthy_branches.sort(key=lambda x: x[1].get("health_score", 100))

                for branch_name, branch_data in unhealthy_branches[:5]:  # Top 5
                    age = branch_data.get("age_days", 0)
                    behind = branch_data.get("behind_main", 0)
                    score = branch_data.get("health_score", 0)

                    issues = []
                    if age > 30:
                        issues.append(f"{age} days old")
                    if behind > 50:
                        issues.append(f"{behind} commits behind")

                    report.write(f"- `{branch_name}` (score: {score:.0f}) - {', '.join(issues)}\n")

                if len(unhealthy_branches) > 5:
                    report.write(f"- ...and {len(unhealthy_branches) - 5} more\n")
                report.write("\n")

        # Recommendations section
        report.write("### Recommendations\n\n")

        all_recommendations = []
        for metrics in branch_health_metrics.values():
            all_recommendations.extend(metrics.get("recommendations", []))

        # Deduplicate and prioritize recommendations
        unique_recommendations = []
        seen = set()
        for rec in all_recommendations:
            # Create a simplified key for deduplication
            key = rec.split()[0]  # Use emoji as key
            if key not in seen:
                seen.add(key)
                unique_recommendations.append(rec)

        if unique_recommendations:
            for rec in unique_recommendations[:5]:  # Top 5 recommendations
                report.write(f"- {rec}\n")
        else:
            report.write("- ✅ All repositories show healthy branch management practices\n")

        report.write("\n")

        # Best practices reminder
        report.write("### Best Practices (2025 Standards)\n\n")
        report.write("- 🎯 **Elite teams** maintain <3% rework rate and <26 hour cycle times\n")
        report.write(
            "- 📏 **Small PRs** (<200 lines) correlate with better quality and faster reviews\n"
        )
        report.write(
            "- 🔄 **Frequent integration** reduces merge conflicts and improves deployment readiness\n"
        )
        report.write(
            "- 🧹 **Regular cleanup** of merged and stale branches keeps repositories manageable\n"
        )

        return report.getvalue()

    def _get_health_emoji(self, health_status: str) -> str:
        """Get emoji for health status."""
        emoji_map = {
            "excellent": "🟢",
            "good": "🟢",
            "fair": "🟡",
            "poor": "🔴",
            "unknown": "⚪",
        }
        return emoji_map.get(health_status.lower(), "⚪")

    def generate_detailed_branch_report(
        self, branch_health_metrics: dict[str, dict[str, Any]], output_path: Path
    ) -> Path:
        """Generate detailed branch-by-branch CSV report.

        Args:
            branch_health_metrics: Dictionary mapping repo names to their branch health metrics
            output_path: Path where the CSV should be written

        Returns:
            Path to the generated CSV file
        """
        rows = []

        for repo_name, metrics in branch_health_metrics.items():
            branches = metrics.get("branches", {})
            main_branch = metrics.get("main_branch", "main")

            for branch_name, branch_data in branches.items():
                # Skip main branch in detailed report
                if branch_name == main_branch:
                    continue

                freq = branch_data.get("commit_frequency", {})

                row = {
                    "repository": repo_name,
                    "branch": branch_name,
                    "age_days": branch_data.get("age_days", 0),
                    "health_score": round(branch_data.get("health_score", 0), 1),
                    "status": self._get_branch_status(branch_data),
                    "total_commits": branch_data.get("total_commits", 0),
                    "unique_authors": branch_data.get("unique_authors", 0),
                    "commits_ahead": branch_data.get("ahead_of_main", 0),
                    "commits_behind": branch_data.get("behind_main", 0),
                    "divergence_total": branch_data.get("divergence_score", 0),
                    "daily_commit_avg": round(freq.get("daily_average", 0), 2),
                    "weekly_commit_avg": round(freq.get("weekly_average", 0), 2),
                    "latest_activity": branch_data.get("latest_activity", ""),
                    "is_merged": branch_data.get("is_merged", False),
                    "is_stale": branch_data.get("is_stale", False),
                }
                rows.append(row)

        # Sort by repository and health score
        rows.sort(key=lambda x: (x["repository"], x["health_score"]))

        # Write CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)

        return output_path

    def _get_branch_status(self, branch_data: dict[str, Any]) -> str:
        """Determine branch status based on metrics."""
        if branch_data.get("is_merged", False):
            return "merged"
        elif branch_data.get("is_stale", False):
            return "stale"
        elif branch_data.get("age_days", 0) > 14:
            return "long-lived"
        else:
            return "active"
