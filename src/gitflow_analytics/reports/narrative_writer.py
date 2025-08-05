"""Narrative report generation in Markdown format."""

import logging
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

from ..metrics.activity_scoring import ActivityScorer

# Get logger for this module
logger = logging.getLogger(__name__)


class NarrativeReportGenerator:
    """Generate human-readable narrative reports in Markdown."""

    def __init__(self) -> None:
        """Initialize narrative report generator."""
        self.activity_scorer = ActivityScorer()
        self.templates = {
            "high_performer": "{name} led development with {commits} commits ({pct}% of total activity)",
            "multi_project": "{name} worked across {count} projects, primarily on {primary} ({primary_pct}%)",
            "focused_developer": "{name} showed strong focus on {project} with {pct}% of their time",
            "ticket_coverage": "The team maintained {coverage}% ticket coverage, indicating {quality} process adherence",
            "work_distribution": "Work distribution shows a {distribution} pattern with a Gini coefficient of {gini}",
        }

    def generate_narrative_report(
        self,
        commits: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        developer_stats: list[dict[str, Any]],
        activity_dist: list[dict[str, Any]],
        focus_data: list[dict[str, Any]],
        insights: list[dict[str, Any]],
        ticket_analysis: dict[str, Any],
        pr_metrics: dict[str, Any],
        output_path: Path,
        weeks: int,
        pm_data: dict[str, Any] = None,
        chatgpt_summary: str = None,
    ) -> Path:
        """Generate comprehensive narrative report."""
        report = StringIO()

        # Header
        report.write("# GitFlow Analytics Report\n\n")

        # Log datetime formatting
        now = datetime.now()
        logger.debug(
            f"Formatting current datetime for report header: {now} (tzinfo: {getattr(now, 'tzinfo', 'N/A')})"
        )
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        logger.debug(f"  Formatted time: {formatted_time}")

        report.write(f"**Generated**: {formatted_time}\n")
        report.write(f"**Analysis Period**: Last {weeks} weeks\n\n")

        # Executive Summary
        report.write("## Executive Summary\n\n")
        self._write_executive_summary(report, commits, developer_stats, ticket_analysis, prs)

        # Add ChatGPT qualitative insights if available
        if chatgpt_summary:
            report.write("\n## Qualitative Analysis\n\n")
            report.write(chatgpt_summary)
            report.write("\n")

        # Team Composition
        report.write("\n## Team Composition\n\n")
        self._write_team_composition(report, developer_stats, focus_data, commits, prs)

        # Project Activity
        report.write("\n## Project Activity\n\n")
        self._write_project_activity(report, activity_dist, commits)

        # Development Patterns
        report.write("\n## Development Patterns\n\n")
        self._write_development_patterns(report, insights, focus_data)

        # Pull Request Analysis (if available)
        if pr_metrics and pr_metrics.get("total_prs", 0) > 0:
            report.write("\n## Pull Request Analysis\n\n")
            self._write_pr_analysis(report, pr_metrics, prs)

        # Ticket Tracking
        report.write("\n## Issue Tracking\n\n")
        self._write_ticket_tracking(report, ticket_analysis, developer_stats)

        # PM Platform Insights
        if pm_data and "metrics" in pm_data:
            report.write("\n## PM Platform Integration\n\n")
            self._write_pm_insights(report, pm_data)

        # Recommendations
        report.write("\n## Recommendations\n\n")
        self._write_recommendations(report, insights, ticket_analysis, focus_data)

        # Write to file
        with open(output_path, "w") as f:
            f.write(report.getvalue())

        return output_path

    def _write_executive_summary(
        self,
        report: StringIO,
        commits: list[dict[str, Any]],
        developer_stats: list[dict[str, Any]],
        ticket_analysis: dict[str, Any],
        prs: list[dict[str, Any]],
    ) -> None:
        """Write executive summary section."""
        total_commits = len(commits)
        total_developers = len(developer_stats)
        total_lines = sum(
            c.get("filtered_insertions", c.get("insertions", 0))
            + c.get("filtered_deletions", c.get("deletions", 0))
            for c in commits
        )

        report.write(f"- **Total Commits**: {total_commits:,}\n")
        report.write(f"- **Active Developers**: {total_developers}\n")
        report.write(f"- **Lines Changed**: {total_lines:,}\n")
        report.write(f"- **Ticket Coverage**: {ticket_analysis['commit_coverage_pct']:.1f}%\n")

        # Projects worked on - show full list instead of just count
        projects = set(c.get("project_key", "UNKNOWN") for c in commits)
        projects_list = sorted(projects)
        report.write(f"- **Active Projects**: {', '.join(projects_list)}\n")

        # Top contributor with proper format matching old report
        if developer_stats:
            top_dev = developer_stats[0]
            # Handle both 'primary_name' (production) and 'name' (tests) for backward compatibility
            dev_name = top_dev.get("primary_name", top_dev.get("name", "Unknown Developer"))
            report.write(
                f"- **Top Contributor**: {dev_name} with {top_dev['total_commits']} commits\n"
            )

            # Calculate team average activity
            if commits:
                # Quick activity score calculation for executive summary
                # total_prs = len(prs) if prs else 0  # Not used yet
                total_lines = sum(
                    c.get("filtered_insertions", c.get("insertions", 0))
                    + c.get("filtered_deletions", c.get("deletions", 0))
                    for c in commits
                )

                # Basic team activity assessment
                avg_commits_per_dev = len(commits) / len(developer_stats) if developer_stats else 0
                if avg_commits_per_dev >= 10:
                    activity_assessment = "high activity"
                elif avg_commits_per_dev >= 5:
                    activity_assessment = "moderate activity"
                else:
                    activity_assessment = "low activity"

                report.write(
                    f"- **Team Activity**: {activity_assessment} (avg {avg_commits_per_dev:.1f} commits/developer)\n"
                )

    def _write_team_composition(
        self,
        report: StringIO,
        developer_stats: list[dict[str, Any]],
        focus_data: list[dict[str, Any]],
        commits: list[dict[str, Any]] = None,
        prs: list[dict[str, Any]] = None,
    ) -> None:
        """Write team composition analysis with activity scores."""
        report.write("### Developer Profiles\n\n")

        # Create developer lookup for focus data
        focus_lookup = {d["developer"]: d for d in focus_data}

        # Calculate activity scores for all developers
        activity_scores = {}
        if commits:
            # Aggregate metrics by developer
            dev_metrics = {}
            for commit in commits:
                canonical_id = commit.get("canonical_id", "")
                if canonical_id not in dev_metrics:
                    dev_metrics[canonical_id] = {
                        "commits": 0,
                        "lines_added": 0,
                        "lines_removed": 0,
                        "files_changed": set(),
                        "complexity_delta": 0,
                        "prs_involved": 0,
                    }

                metrics = dev_metrics[canonical_id]
                metrics["commits"] += 1
                metrics["lines_added"] += commit.get(
                    "filtered_insertions", commit.get("insertions", 0)
                )
                metrics["lines_removed"] += commit.get(
                    "filtered_deletions", commit.get("deletions", 0)
                )
                metrics["complexity_delta"] += commit.get("complexity_delta", 0)

                # Track unique files
                files = commit.get("files_changed", [])
                if isinstance(files, list):
                    metrics["files_changed"].update(files)
                elif isinstance(files, int):
                    # If it's already aggregated, just add the count
                    metrics["files_changed"] = len(metrics["files_changed"]) + files

            # Count PRs per developer
            if prs:
                for pr in prs:
                    author = pr.get("author", "")
                    # Map PR author to canonical ID - need to look up in developer_stats
                    for dev in developer_stats:
                        if (
                            dev.get("github_username") == author
                            or dev.get("primary_name") == author
                        ):
                            canonical_id = dev.get("canonical_id")
                            if canonical_id in dev_metrics:
                                dev_metrics[canonical_id]["prs_involved"] += 1
                            break

            # Calculate scores
            for canonical_id, metrics in dev_metrics.items():
                # Convert set to count
                if isinstance(metrics["files_changed"], set):
                    metrics["files_changed"] = len(metrics["files_changed"])

                score_result = self.activity_scorer.calculate_activity_score(metrics)
                activity_scores[canonical_id] = score_result

        # Calculate team scores for relative ranking
        all_scores = [score["raw_score"] for score in activity_scores.values()]

        for dev in developer_stats[:10]:  # Top 10 developers
            # Handle both 'primary_name' (production) and 'name' (tests) for backward compatibility
            name = dev.get("primary_name", dev.get("name", "Unknown Developer"))
            commits = dev["total_commits"]
            canonical_id = dev.get("canonical_id")

            report.write(f"**{name}**\n")
            report.write(f"- Commits: {commits}\n")

            # Add activity score if available
            if canonical_id and canonical_id in activity_scores:
                score_data = activity_scores[canonical_id]
                relative_data = self.activity_scorer.calculate_team_relative_score(
                    score_data["raw_score"], all_scores
                )

                report.write(
                    f"- Activity Score: {score_data['normalized_score']:.1f}/100 "
                    f"({score_data['activity_level']}, {relative_data['percentile']:.0f}th percentile)\n"
                )

            # Add focus data if available
            if name in focus_lookup:
                focus = focus_lookup[name]

                # Get all projects for this developer - check for both naming patterns
                project_percentages = []

                # First try the _dev_pct pattern - use 0.05 threshold to include small percentages but filter out noise
                for key in focus:
                    if key.endswith("_dev_pct") and focus[key] > 0.05:
                        project_name = key.replace("_dev_pct", "")
                        project_percentages.append((project_name, focus[key]))

                # If no _dev_pct found, try _pct pattern
                if not project_percentages:
                    for key in focus:
                        if (
                            key.endswith("_pct")
                            and not key.startswith("primary_")
                            and focus[key] > 0.05
                        ):
                            project_name = key.replace("_pct", "")
                            project_percentages.append((project_name, focus[key]))

                # Sort by percentage descending
                project_percentages.sort(key=lambda x: x[1], reverse=True)

                # Build projects string - show all projects above threshold with percentages
                if project_percentages:
                    projects_str = ", ".join(
                        f"{proj} ({pct:.1f}%)" for proj, pct in project_percentages
                    )
                    report.write(f"- Projects: {projects_str}\n")
                else:
                    # Fallback to primary project if no percentage fields found above threshold
                    primary_project = focus.get("primary_project", "UNKNOWN")
                    primary_pct = focus.get("primary_project_pct", 0)
                    if primary_pct > 0.05:  # Apply same threshold to fallback
                        report.write(f"- Projects: {primary_project} ({primary_pct:.1f}%)\n")
                    else:
                        # If even primary project is below threshold, show it anyway to avoid empty projects
                        report.write(f"- Projects: {primary_project} ({primary_pct:.1f}%)\n")

                report.write(f"- Work Style: {focus['work_style']}\n")
                report.write(f"- Active Pattern: {focus['time_pattern']}\n")

            report.write("\n")

    def _write_project_activity(
        self, report: StringIO, activity_dist: list[dict[str, Any]], commits: list[dict[str, Any]]
    ) -> None:
        """Write project activity breakdown."""
        # Aggregate by project with developer details
        project_totals: dict[str, dict[str, Any]] = {}
        project_developers: dict[str, dict[str, int]] = {}

        for row in activity_dist:
            # Handle missing fields gracefully for test compatibility
            project = row.get("project", "UNKNOWN")
            developer = row.get("developer", "Unknown Developer")

            if project not in project_totals:
                project_totals[project] = {"commits": 0, "lines": 0, "developers": set()}
                project_developers[project] = {}

            data = project_totals[project]
            # Handle missing fields gracefully for test compatibility
            data["commits"] += row.get("commits", 1)  # Default to 1 if missing
            data["lines"] += row.get("lines_changed", 0)
            developers_set: set[str] = data["developers"]
            developers_set.add(developer)

            # Track commits per developer per project
            if developer not in project_developers[project]:
                project_developers[project][developer] = 0
            project_developers[project][developer] += row.get(
                "commits", 1
            )  # Default to 1 if missing

        # Sort by commits
        sorted_projects = sorted(
            project_totals.items(), key=lambda x: x[1]["commits"], reverse=True
        )

        # Calculate total commits across all projects in activity distribution
        total_activity_commits = sum(data["commits"] for data in project_totals.values())

        report.write("### Activity by Project\n\n")
        for project, data in sorted_projects:
            report.write(f"**{project}**\n")
            report.write(f"- Commits: {data['commits']} ")
            report.write(f"({data['commits'] / total_activity_commits * 100:.1f}% of total)\n")
            report.write(f"- Lines Changed: {data['lines']:,}\n")

            # Get developer contributions for this project
            dev_contributions = project_developers[project]
            # Sort by commits descending
            sorted_devs = sorted(dev_contributions.items(), key=lambda x: x[1], reverse=True)

            # Build contributors string
            contributors = []
            for dev_name, dev_commits in sorted_devs:
                dev_pct = dev_commits / data["commits"] * 100
                contributors.append(f"{dev_name} ({dev_pct:.1f}%)")

            contributors_str = ", ".join(contributors)
            report.write(f"- Contributors: {contributors_str}\n\n")

    def _write_development_patterns(
        self, report: StringIO, insights: list[dict[str, Any]], focus_data: list[dict[str, Any]]
    ) -> None:
        """Write development patterns analysis."""
        report.write("### Key Patterns Identified\n\n")

        # Group insights by category (handle missing category field gracefully)
        by_category: dict[str, list[dict[str, Any]]] = {}
        for insight in insights:
            category = insight.get("category", "General")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(insight)

        for category, category_insights in by_category.items():
            report.write(f"**{category}**:\n")
            for insight in category_insights:
                # Handle missing fields gracefully for test compatibility
                insight_text = insight.get("insight", insight.get("metric", "Unknown"))
                insight_value = insight.get("value", "N/A")
                insight_impact = insight.get("impact", "No impact specified")
                report.write(f"- {insight_text}: {insight_value} ")
                report.write(f"({insight_impact})\n")
            report.write("\n")

        # Add focus insights (handle missing focus_score field gracefully)
        if focus_data:
            # Use focus_ratio if focus_score is not available
            focus_scores = []
            for d in focus_data:
                if "focus_score" in d:
                    focus_scores.append(d["focus_score"])
                elif "focus_ratio" in d:
                    focus_scores.append(d["focus_ratio"] * 100)  # Convert ratio to percentage
                else:
                    focus_scores.append(50)  # Default value

            if focus_scores:
                avg_focus = sum(focus_scores) / len(focus_scores)
                report.write(f"**Developer Focus**: Average focus score of {avg_focus:.1f}% ")

                if avg_focus > 80:
                    report.write("indicates strong project concentration\n")
                elif avg_focus > 60:
                    report.write("shows moderate multi-project work\n")
                else:
                    report.write("suggests high context switching\n")

    def _write_pr_analysis(
        self, report: StringIO, pr_metrics: dict[str, Any], prs: list[dict[str, Any]]
    ) -> None:
        """Write pull request analysis."""
        report.write(f"- **Total PRs Merged**: {pr_metrics.get('total_prs', 0)}\n")
        report.write(f"- **Average PR Size**: {pr_metrics.get('avg_pr_size', 0):.0f} lines\n")

        # Handle optional metrics gracefully
        if "avg_pr_lifetime_hours" in pr_metrics:
            report.write(
                f"- **Average PR Lifetime**: {pr_metrics['avg_pr_lifetime_hours']:.1f} hours\n"
            )

        if "story_point_coverage" in pr_metrics:
            report.write(f"- **Story Point Coverage**: {pr_metrics['story_point_coverage']:.1f}%\n")

        total_comments = pr_metrics.get("total_review_comments", 0)
        if total_comments > 0:
            report.write(f"- **Total Review Comments**: {total_comments}\n")
            total_prs = pr_metrics.get("total_prs", 1)
            avg_comments = total_comments / total_prs if total_prs > 0 else 0
            report.write(f"- **Average Comments per PR**: {avg_comments:.1f}\n")

    def _write_ticket_tracking(
        self,
        report: StringIO,
        ticket_analysis: dict[str, Any],
        developer_stats: list[dict[str, Any]],
    ) -> None:
        """Write ticket tracking analysis with simplified platform usage section."""
        # Simplified platform usage matching old report format
        ticket_summary = ticket_analysis.get("ticket_summary", {})
        total_tickets = sum(ticket_summary.values()) if ticket_summary else 0

        if total_tickets > 0:
            report.write("### Platform Usage\n\n")
            for platform, count in sorted(ticket_summary.items(), key=lambda x: x[1], reverse=True):
                pct = count / total_tickets * 100 if total_tickets > 0 else 0
                report.write(f"- **{platform.title()}**: {count} tickets ({pct:.1f}%)\n")

        report.write("\n### Coverage Analysis\n\n")

        # Handle missing fields gracefully
        commits_with_tickets = ticket_analysis.get("commits_with_tickets", 0)
        total_commits = ticket_analysis.get("total_commits", 0)
        coverage_pct = ticket_analysis.get("commit_coverage_pct", 0)

        report.write(f"- **Commits with Tickets**: {commits_with_tickets} ")
        report.write(f"of {total_commits} ")
        report.write(f"({coverage_pct:.1f}%)\n")

        # Enhanced untracked commits reporting
        untracked_commits = ticket_analysis.get("untracked_commits", [])
        if untracked_commits:
            self._write_enhanced_untracked_analysis(
                report, untracked_commits, ticket_analysis, developer_stats
            )

    def _write_enhanced_untracked_analysis(
        self,
        report: StringIO,
        untracked_commits: list[dict[str, Any]],
        ticket_analysis: dict[str, Any],
        developer_stats: list[dict[str, Any]],
    ) -> None:
        """Write comprehensive untracked commits analysis.

        WHY: Enhanced untracked analysis provides actionable insights into what
        types of work are happening outside the tracked process, helping identify
        process improvements and training opportunities.
        """
        report.write("\n### Untracked Work Analysis\n\n")

        total_untracked = len(untracked_commits)
        total_commits = ticket_analysis.get("total_commits", 0)
        untracked_pct = (total_untracked / total_commits * 100) if total_commits > 0 else 0

        report.write(
            f"**Summary**: {total_untracked} commits ({untracked_pct:.1f}% of total) lack ticket references.\n\n"
        )

        # Analyze categories
        categories = {}
        contributors = {}
        projects = {}

        for commit in untracked_commits:
            # Category analysis
            category = commit.get("category", "other")
            if category not in categories:
                categories[category] = {"count": 0, "lines": 0, "examples": []}
            categories[category]["count"] += 1
            categories[category]["lines"] += commit.get("lines_changed", 0)
            if len(categories[category]["examples"]) < 2:
                categories[category]["examples"].append(
                    {
                        "hash": commit.get("hash", ""),
                        "message": commit.get("message", ""),
                        "author": commit.get("author", ""),
                    }
                )

            # Contributor analysis
            author = commit.get("author", "Unknown")
            if author not in contributors:
                contributors[author] = {"count": 0, "categories": set()}
            contributors[author]["count"] += 1
            contributors[author]["categories"].add(category)

            # Project analysis
            project = commit.get("project_key", "UNKNOWN")
            if project not in projects:
                projects[project] = {"count": 0, "categories": set()}
            projects[project]["count"] += 1
            projects[project]["categories"].add(category)

        # Write category breakdown
        if categories:
            report.write("#### Work Categories\n\n")
            sorted_categories = sorted(
                categories.items(), key=lambda x: x[1]["count"], reverse=True
            )

            for category, data in sorted_categories[:8]:  # Show top 8 categories
                pct = (data["count"] / total_untracked) * 100
                avg_size = data["lines"] / data["count"] if data["count"] > 0 else 0

                # Categorize the impact
                if category in ["style", "documentation", "maintenance"]:
                    impact_note = " *(acceptable untracked)*"
                elif category in ["feature", "bug_fix"]:
                    impact_note = " *(should be tracked)*"
                else:
                    impact_note = ""

                report.write(f"- **{category.replace('_', ' ').title()}**: ")
                report.write(f"{data['count']} commits ({pct:.1f}%), ")
                report.write(f"avg {avg_size:.0f} lines{impact_note}\n")

                # Add examples
                if data["examples"]:
                    for example in data["examples"]:
                        report.write(f"  - `{example['hash']}`: {example['message'][:80]}...\n")
            report.write("\n")

        # Write top contributors to untracked work with enhanced percentage analysis
        if contributors:
            report.write("#### Top Contributors (Untracked Work)\n\n")

            # Create developer lookup for total commits
            dev_lookup = {}
            for dev in developer_stats:
                # Map canonical_id to developer data
                dev_lookup[dev["canonical_id"]] = dev
                # Also map primary name and primary email as fallbacks
                dev_lookup[dev["primary_name"]] = dev
                dev_lookup[dev["primary_email"]] = dev

            sorted_contributors = sorted(
                contributors.items(), key=lambda x: x[1]["count"], reverse=True
            )

            for author, data in sorted_contributors[:5]:  # Show top 5
                untracked_count = data["count"]
                pct_of_untracked = (untracked_count / total_untracked) * 100

                # Find developer's total commits to calculate percentage of their work that's untracked
                dev_data = dev_lookup.get(author)
                if dev_data:
                    total_dev_commits = dev_data["total_commits"]
                    pct_of_dev_work = (
                        (untracked_count / total_dev_commits) * 100 if total_dev_commits > 0 else 0
                    )
                    dev_context = f", {pct_of_dev_work:.1f}% of their work"
                else:
                    dev_context = ""

                categories_list = list(data["categories"])
                categories_str = ", ".join(categories_list[:3])  # Show up to 3 categories
                if len(categories_list) > 3:
                    categories_str += f" (+{len(categories_list) - 3} more)"

                report.write(f"- **{author}**: {untracked_count} commits ")
                report.write(f"({pct_of_untracked:.1f}% of untracked{dev_context}) - ")
                report.write(f"*{categories_str}*\n")
            report.write("\n")

        # Write project breakdown
        if len(projects) > 1:
            report.write("#### Projects with Untracked Work\n\n")
            sorted_projects = sorted(projects.items(), key=lambda x: x[1]["count"], reverse=True)

            for project, data in sorted_projects:
                pct = (data["count"] / total_untracked) * 100
                categories_list = list(data["categories"])
                report.write(f"- **{project}**: {data['count']} commits ({pct:.1f}%)\n")
            report.write("\n")

        # Write recent examples (configurable limit, default 15 for better visibility)
        if untracked_commits:
            report.write("#### Recent Untracked Commits\n\n")

            # Show configurable number of recent commits (increased from 10 to 15)
            max_recent_commits = 15
            recent_commits = sorted(
                untracked_commits, key=lambda x: x.get("timestamp", ""), reverse=True
            )[:max_recent_commits]

            if len(untracked_commits) > max_recent_commits:
                report.write(
                    f"*Showing {max_recent_commits} most recent of {len(untracked_commits)} untracked commits*\n\n"
                )

            for commit in recent_commits:
                # Format date
                timestamp = commit.get("timestamp")
                if timestamp and hasattr(timestamp, "strftime"):
                    date_str = timestamp.strftime("%Y-%m-%d")
                else:
                    date_str = "unknown date"

                report.write(f"- `{commit.get('hash', '')}` ({date_str}) ")
                report.write(f"**{commit.get('author', 'Unknown')}** ")
                report.write(f"[{commit.get('category', 'other')}]: ")
                report.write(f"{commit.get('message', '')[:100]}")
                if len(commit.get("message", "")) > 100:
                    report.write("...")
                report.write(f" *({commit.get('files_changed', 0)} files, ")
                report.write(f"{commit.get('lines_changed', 0)} lines)*\n")
            report.write("\n")

        # Add recommendations based on untracked analysis
        self._write_untracked_recommendations(
            report, categories, contributors, total_untracked, total_commits
        )

    def _write_untracked_recommendations(
        self,
        report: StringIO,
        categories: dict[str, Any],
        contributors: dict[str, Any],
        total_untracked: int,
        total_commits: int,
    ) -> None:
        """Write specific recommendations based on untracked commit analysis."""
        report.write("#### Recommendations for Untracked Work\n\n")

        recommendations = []

        # Category-based recommendations
        feature_count = categories.get("feature", {}).get("count", 0)
        bug_fix_count = categories.get("bug_fix", {}).get("count", 0)
        maintenance_count = categories.get("maintenance", {}).get("count", 0)
        docs_count = categories.get("documentation", {}).get("count", 0)
        style_count = categories.get("style", {}).get("count", 0)

        if feature_count > total_untracked * 0.2:
            recommendations.append(
                "🎫 **Require tickets for features**: Many feature developments lack ticket references. "
                "Consider enforcing ticket creation for new functionality."
            )

        if bug_fix_count > total_untracked * 0.15:
            recommendations.append(
                "🐛 **Track bug fixes**: Bug fixes should be linked to issue tickets for better "
                "visibility and follow-up."
            )

        # Positive recognition for appropriate untracked work
        acceptable_count = maintenance_count + docs_count + style_count
        if acceptable_count > total_untracked * 0.6:
            recommendations.append(
                "✅ **Good process balance**: Most untracked work consists of maintenance, "
                "documentation, and style improvements - this is acceptable and shows good "
                "development hygiene."
            )

        # Coverage recommendations
        untracked_pct = (total_untracked / total_commits * 100) if total_commits > 0 else 0
        if untracked_pct > 50:
            recommendations.append(
                "📈 **Improve overall tracking**: Over 50% of commits lack ticket references. "
                "Consider team training on linking commits to work items."
            )
        elif untracked_pct < 20:
            recommendations.append(
                "🎯 **Excellent tracking**: Less than 20% of commits are untracked - "
                "the team shows strong process adherence."
            )

        # Developer-specific recommendations
        if len(contributors) > 1:
            max_contributor_pct = max(
                (data["count"] / total_untracked * 100) for data in contributors.values()
            )
            if max_contributor_pct > 40:
                recommendations.append(
                    "👥 **Targeted training**: Some developers need additional guidance on "
                    "ticket referencing practices. Consider peer mentoring or process review."
                )

        if not recommendations:
            recommendations.append(
                "✅ **Balanced approach**: Untracked work appears well-balanced between "
                "necessary maintenance and tracked development work."
            )

        for rec in recommendations:
            report.write(f"{rec}\n\n")

    def _write_recommendations(
        self,
        report: StringIO,
        insights: list[dict[str, Any]],
        ticket_analysis: dict[str, Any],
        focus_data: list[dict[str, Any]],
    ) -> None:
        """Write recommendations based on analysis."""
        recommendations = []

        # Ticket coverage recommendations
        coverage = ticket_analysis["commit_coverage_pct"]
        if coverage < 50:
            recommendations.append(
                "🎫 **Improve ticket tracking**: Current coverage is below 50%. "
                "Consider enforcing ticket references in commit messages or PR descriptions."
            )

        # Work distribution recommendations (handle missing insight field gracefully)
        for insight in insights:
            insight_text = insight.get("insight", insight.get("metric", ""))
            if insight_text == "Work distribution":
                insight_value = str(insight.get("value", ""))
                if "unbalanced" in insight_value.lower():
                    recommendations.append(
                        "⚖️ **Balance workload**: Work is concentrated among few developers. "
                        "Consider distributing tasks more evenly or adding team members."
                    )

        # Focus recommendations (handle missing focus_score field gracefully)
        if focus_data:
            low_focus = []
            for d in focus_data:
                focus_score = d.get("focus_score", d.get("focus_ratio", 0.5) * 100)
                if focus_score < 50:
                    low_focus.append(d)
            if len(low_focus) > len(focus_data) / 2:
                recommendations.append(
                    "🎯 **Reduce context switching**: Many developers work across multiple projects. "
                    "Consider more focused project assignments to improve efficiency."
                )

        # Branching strategy (handle missing insight field gracefully)
        for insight in insights:
            insight_text = insight.get("insight", insight.get("metric", ""))
            insight_value = str(insight.get("value", ""))
            if insight_text == "Branching strategy" and "Heavy" in insight_value:
                recommendations.append(
                    "🌿 **Review branching strategy**: High percentage of merge commits suggests "
                    "complex branching. Consider simplifying the Git workflow."
                )

        if recommendations:
            for rec in recommendations:
                report.write(f"{rec}\n\n")
        else:
            report.write("✅ The team shows healthy development patterns. ")
            report.write("Continue current practices while monitoring for changes.\n")

    def _write_pm_insights(self, report: StringIO, pm_data: dict[str, Any]) -> None:
        """Write PM platform integration insights.

        WHY: PM platform integration provides valuable insights into work item
        tracking, story point accuracy, and development velocity that complement
        Git-based analytics. This section highlights the value of PM integration.
        """
        metrics = pm_data.get("metrics", {})

        # Platform overview
        platform_coverage = metrics.get("platform_coverage", {})
        total_issues = metrics.get("total_pm_issues", 0)
        correlations = len(pm_data.get("correlations", []))

        report.write(f"The team has integrated **{len(platform_coverage)} PM platforms** ")
        report.write(
            f"tracking **{total_issues:,} issues** with **{correlations} commit correlations**.\n\n"
        )

        # Story point analysis
        story_analysis = metrics.get("story_point_analysis", {})
        pm_story_points = story_analysis.get("pm_total_story_points", 0)
        git_story_points = story_analysis.get("git_total_story_points", 0)
        coverage_pct = story_analysis.get("story_point_coverage_pct", 0)

        if pm_story_points > 0:
            report.write("### Story Point Tracking\n\n")
            report.write(f"- **PM Platform Story Points**: {pm_story_points:,}\n")
            report.write(f"- **Git Extracted Story Points**: {git_story_points:,}\n")
            report.write(
                f"- **Story Point Coverage**: {coverage_pct:.1f}% of issues have story points\n"
            )

            if git_story_points > 0:
                accuracy = min(git_story_points / pm_story_points, 1.0) * 100
                report.write(
                    f"- **Extraction Accuracy**: {accuracy:.1f}% of PM story points found in Git\n"
                )
            report.write("\n")

        # Issue type distribution
        issue_types = metrics.get("issue_type_distribution", {})
        if issue_types:
            report.write("### Work Item Types\n\n")
            sorted_types = sorted(issue_types.items(), key=lambda x: x[1], reverse=True)
            total_typed_issues = sum(issue_types.values())

            for issue_type, count in sorted_types[:5]:  # Top 5 types
                pct = (count / total_typed_issues * 100) if total_typed_issues > 0 else 0
                report.write(f"- **{issue_type.title()}**: {count} issues ({pct:.1f}%)\n")
            report.write("\n")

        # Platform-specific insights
        if platform_coverage:
            report.write("### Platform Coverage\n\n")
            for platform, coverage_data in platform_coverage.items():
                platform_issues = coverage_data.get("total_issues", 0)
                linked_issues = coverage_data.get("linked_issues", 0)
                coverage_percentage = coverage_data.get("coverage_percentage", 0)

                report.write(f"**{platform.title()}**: ")
                report.write(f"{platform_issues} issues, {linked_issues} linked to commits ")
                report.write(f"({coverage_percentage:.1f}% coverage)\n")
            report.write("\n")

        # Correlation quality
        correlation_quality = metrics.get("correlation_quality", {})
        if correlation_quality.get("total_correlations", 0) > 0:
            avg_confidence = correlation_quality.get("average_confidence", 0)
            high_confidence = correlation_quality.get("high_confidence_correlations", 0)
            correlation_methods = correlation_quality.get("correlation_methods", {})

            report.write("### Correlation Quality\n\n")
            report.write(f"- **Average Confidence**: {avg_confidence:.2f} (0.0-1.0 scale)\n")
            report.write(f"- **High Confidence Matches**: {high_confidence} correlations\n")

            if correlation_methods:
                report.write("- **Methods Used**: ")
                method_list = [
                    f"{method.replace('_', ' ').title()} ({count})"
                    for method, count in correlation_methods.items()
                ]
                report.write(", ".join(method_list))
                report.write("\n")
            report.write("\n")

        # Key insights
        report.write("### Key Insights\n\n")

        if coverage_pct > 80:
            report.write(
                "✅ **Excellent story point coverage** - Most issues have effort estimates\n"
            )
        elif coverage_pct > 50:
            report.write(
                "⚠️ **Moderate story point coverage** - Consider improving estimation practices\n"
            )
        else:
            report.write(
                "❌ **Low story point coverage** - Story point tracking needs improvement\n"
            )

        if correlations > total_issues * 0.5:
            report.write(
                "✅ **Strong commit-issue correlation** - Good traceability between work items and code\n"
            )
        elif correlations > total_issues * 0.2:
            report.write(
                "⚠️ **Moderate commit-issue correlation** - Some work items lack code links\n"
            )
        else:
            report.write(
                "❌ **Weak commit-issue correlation** - Improve ticket referencing in commits\n"
            )

        if len(platform_coverage) > 1:
            report.write(
                "📊 **Multi-platform integration** - Comprehensive work item tracking across tools\n"
            )

        report.write("\n")
