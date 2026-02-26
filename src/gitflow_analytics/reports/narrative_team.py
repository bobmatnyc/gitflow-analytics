"""Narrative report mixin: team composition and project activity sections.

Extracted from narrative_writer.py to keep file sizes manageable.
"""

import logging
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)



class NarrativeTeamMixin:
    """Mixin: team composition and project activity writing methods."""

    def _write_team_composition(
        self,
        report: StringIO,
        developer_stats: list[dict[str, Any]],
        focus_data: list[dict[str, Any]],
        commits: list[dict[str, Any]] = None,
        prs: list[dict[str, Any]] = None,
        ticket_analysis: dict[str, Any] = None,
        weeks: int = 4,
    ) -> None:
        """Write team composition analysis with activity scores and commit classifications.

        WHY: Enhanced team composition shows not just how much each developer commits,
        but what types of work they're doing. This provides actionable insights into
        developer specializations, training needs, and work distribution patterns.
        """
        report.write("### Developer Profiles\n\n")

        # Create developer lookup for focus data
        focus_lookup = {d["developer"]: d for d in focus_data}

        # Calculate activity scores for all developers
        activity_scores = {}
        dev_metrics = {}  # Initialize outside if block to ensure it's always defined

        if commits:
            # Aggregate metrics by developer
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
                metrics["lines_added"] += (
                    commit.get("filtered_insertions", commit.get("insertions", 0)) or 0
                )
                metrics["lines_removed"] += (
                    commit.get("filtered_deletions", commit.get("deletions", 0)) or 0
                )
                metrics["complexity_delta"] += commit.get("complexity_delta", 0) or 0

                # Track unique files
                files = commit.get("files_changed", [])
                if isinstance(files, list):
                    # Only update if metrics["files_changed"] is still a set
                    if isinstance(metrics["files_changed"], set):
                        metrics["files_changed"].update(files)
                    else:
                        # If it's already an int, convert back to set and update
                        metrics["files_changed"] = set()
                        metrics["files_changed"].update(files)
                elif isinstance(files, int):
                    # If it's already aggregated, just add the count
                    if isinstance(metrics["files_changed"], set):
                        metrics["files_changed"] = len(metrics["files_changed"]) + files
                    else:
                        metrics["files_changed"] += files

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
            raw_scores_for_curve = {}
            for canonical_id, metrics in dev_metrics.items():
                # Convert set to count
                if isinstance(metrics["files_changed"], set):
                    metrics["files_changed"] = len(metrics["files_changed"])

                score_result = self.activity_scorer.calculate_activity_score(metrics)
                activity_scores[canonical_id] = score_result
                raw_scores_for_curve[canonical_id] = score_result["raw_score"]

            # Apply curve normalization
            curve_normalized = self.activity_scorer.normalize_scores_on_curve(raw_scores_for_curve)

            # Update activity scores with curve data
            for canonical_id, curve_data in curve_normalized.items():
                if canonical_id in activity_scores:
                    activity_scores[canonical_id]["curve_data"] = curve_data

        # Calculate team scores for relative ranking
        all_scores = [score["raw_score"] for score in activity_scores.values()]

        # Consolidate developer_stats by canonical_id to avoid duplicates from identity aliasing
        consolidated_devs = {}
        for dev in developer_stats:
            canonical_id = dev.get("canonical_id")
            if canonical_id and canonical_id not in consolidated_devs:
                consolidated_devs[canonical_id] = dev

        # BUGFIX: Only include developers who have commits in the analysis period
        # Filter using dev_metrics (period-specific) instead of developer_stats (all-time)
        active_devs = {}

        # Only process developers if we have commit data for the period
        for canonical_id, dev in consolidated_devs.items():
            # Only include developers who have commits in the current analysis period
            if canonical_id in dev_metrics:
                active_devs[canonical_id] = dev
        # If no commits in period, no developers will be shown
        # (This handles the case where all commits are outside the analysis period)

        for canonical_id, dev in active_devs.items():  # Only developers with commits in period
            # Handle both 'primary_name' (production) and 'name' (tests) for backward compatibility
            name = dev.get("primary_name", dev.get("name", "Unknown Developer"))

            # BUGFIX: Use period-specific commit count instead of all-time total
            # Safety check: dev_metrics should exist if we got here, but be defensive
            if canonical_id in dev_metrics:
                period_commits = dev_metrics[canonical_id]["commits"]
                total_commits = period_commits  # For backward compatibility with existing logic
            else:
                # Fallback (shouldn't happen with the filtering above)
                total_commits = 0

            report.write(f"**{name}**\n")

            # Try to get commit classification breakdown if available
            if ticket_analysis:
                classifications = self._aggregate_commit_classifications(
                    ticket_analysis, commits, developer_stats
                )
                dev_classifications = classifications.get(canonical_id, {})

                if dev_classifications:
                    # Sort categories by count (descending)
                    sorted_categories = sorted(
                        dev_classifications.items(), key=lambda x: x[1], reverse=True
                    )

                    # Format as "Features: 15 (45%), Bug Fixes: 8 (24%), etc."
                    total_classified = sum(dev_classifications.values())
                    if total_classified > 0:
                        category_parts = []
                        for category, count in sorted_categories:
                            pct = (count / total_classified) * 100
                            display_name = self._format_category_name(category)
                            category_parts.append(f"{display_name}: {count} ({pct:.0f}%)")

                        # Show top categories (limit to avoid excessive length)
                        max_categories = 5
                        if len(category_parts) > max_categories:
                            shown_parts = category_parts[:max_categories]
                            remaining = len(category_parts) - max_categories
                            shown_parts.append(f"({remaining} more)")
                            category_display = ", ".join(shown_parts)
                        else:
                            category_display = ", ".join(category_parts)

                        # Calculate ticket coverage for this developer
                        ticket_coverage_pct = dev.get("ticket_coverage_pct", 0)
                        report.write(f"- Commits: {category_display}\n")
                        report.write(f"- Ticket Coverage: {ticket_coverage_pct:.1f}%\n")

                        # Add weekly trend lines if available
                        if commits:
                            weekly_trends = self._calculate_weekly_classification_percentages(
                                commits,
                                developer_id=canonical_id,
                                weeks=weeks,
                                analysis_start_date=self._analysis_start_date,
                                analysis_end_date=self._analysis_end_date,
                            )
                            if weekly_trends:
                                self._write_weekly_trend_lines(report, weekly_trends)
                            else:
                                # Fallback to simple trend analysis
                                trends = self._calculate_classification_trends(
                                    commits, developer_id=canonical_id, weeks=weeks
                                )
                                trend_line = self._format_trend_line(trends)
                                if trend_line:
                                    report.write(f"- {trend_line}\n")
                    else:
                        # Fallback to simple count if no classifications
                        ticket_coverage_pct = dev.get("ticket_coverage_pct", 0)
                        report.write(f"- Commits: {total_commits}\n")
                        report.write(f"- Ticket Coverage: {ticket_coverage_pct:.1f}%\n")

                        # Still try to add weekly trend lines for simple commits
                        if commits:
                            weekly_trends = self._calculate_weekly_classification_percentages(
                                commits,
                                developer_id=canonical_id,
                                weeks=weeks,
                                analysis_start_date=self._analysis_start_date,
                                analysis_end_date=self._analysis_end_date,
                            )
                            if weekly_trends:
                                self._write_weekly_trend_lines(report, weekly_trends)
                            else:
                                # Fallback to simple trend analysis
                                trends = self._calculate_classification_trends(
                                    commits, developer_id=canonical_id, weeks=weeks
                                )
                                trend_line = self._format_trend_line(trends)
                                if trend_line:
                                    report.write(f"- {trend_line}\n")
                else:
                    # Fallback to simple count if no classification data for this developer
                    ticket_coverage_pct = dev.get("ticket_coverage_pct", 0)
                    report.write(f"- Commits: {total_commits}\n")
                    report.write(f"- Ticket Coverage: {ticket_coverage_pct:.1f}%\n")

                    # Still try to add weekly trend lines
                    if commits:
                        weekly_trends = self._calculate_weekly_classification_percentages(
                            commits,
                            developer_id=canonical_id,
                            weeks=weeks,
                            analysis_start_date=self._analysis_start_date,
                            analysis_end_date=self._analysis_end_date,
                        )
                        if weekly_trends:
                            self._write_weekly_trend_lines(report, weekly_trends)
                        else:
                            # Fallback to simple trend analysis
                            trends = self._calculate_classification_trends(
                                commits, developer_id=canonical_id, weeks=weeks
                            )
                            trend_line = self._format_trend_line(trends)
                            if trend_line:
                                report.write(f"- {trend_line}\n")
            else:
                # Fallback to simple count if no ticket analysis available
                report.write(f"- Commits: {total_commits}\n")
                # No ticket coverage info available in this case

                # Still try to add weekly trend lines if commits available
                if commits:
                    weekly_trends = self._calculate_weekly_classification_percentages(
                        commits,
                        developer_id=canonical_id,
                        weeks=weeks,
                        analysis_start_date=self._analysis_start_date,
                        analysis_end_date=self._analysis_end_date,
                    )
                    if weekly_trends:
                        self._write_weekly_trend_lines(report, weekly_trends)
                    else:
                        # Fallback to simple trend analysis
                        trends = self._calculate_classification_trends(
                            commits, developer_id=canonical_id, weeks=weeks
                        )
                        trend_line = self._format_trend_line(trends)
                        if trend_line:
                            report.write(f"- {trend_line}\n")

            # Add activity score if available
            if canonical_id and canonical_id in activity_scores:
                score_data = activity_scores[canonical_id]

                # Use curve data if available, otherwise fall back to relative scoring
                if "curve_data" in score_data:
                    curve_data = score_data["curve_data"]
                    report.write(
                        f"- Activity Score: {curve_data['curved_score']:.1f}/100 "
                        f"({curve_data['activity_level']}, {curve_data['level_description']})\n"
                    )
                else:
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
        self,
        report: StringIO,
        activity_dist: list[dict[str, Any]],
        commits: list[dict[str, Any]],
        branch_health_metrics: dict[str, dict[str, Any]] = None,
        ticket_analysis: dict[str, Any] = None,
        weeks: int = 4,
    ) -> None:
        """Write project activity breakdown with commit classifications.

        WHY: Enhanced project activity section now includes commit classification
        breakdown per project, providing insights into what types of work are
        happening in each project (features, bug fixes, refactoring, etc.).
        This helps identify project-specific development patterns.
        """
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
            report.write(f"- Contributors: {contributors_str}\n")

            # Add commit classification breakdown for this project
            if ticket_analysis:
                project_classifications = self._get_project_classifications(
                    project, commits, ticket_analysis
                )
                if project_classifications:
                    # Sort categories by count (descending)
                    sorted_categories = sorted(
                        project_classifications.items(), key=lambda x: x[1], reverse=True
                    )

                    # Calculate total for percentages
                    total_classified = sum(project_classifications.values())
                    if total_classified > 0:
                        category_parts = []
                        for category, count in sorted_categories:
                            pct = (count / total_classified) * 100
                            display_name = self._format_category_name(category)
                            category_parts.append(f"{display_name}: {count} ({pct:.0f}%)")

                        # Show top categories to avoid excessive length
                        max_categories = 4
                        if len(category_parts) > max_categories:
                            shown_parts = category_parts[:max_categories]
                            remaining = len(category_parts) - max_categories
                            shown_parts.append(f"({remaining} more)")
                            category_display = ", ".join(shown_parts)
                        else:
                            category_display = ", ".join(category_parts)

                        report.write(f"- Classifications: {category_display}\n")

                        # Add project-level weekly trend lines
                        if commits:
                            project_weekly_trends = (
                                self._calculate_weekly_classification_percentages(
                                    commits,
                                    project_key=project,
                                    weeks=weeks,
                                    analysis_start_date=self._analysis_start_date,
                                    analysis_end_date=self._analysis_end_date,
                                )
                            )
                            if project_weekly_trends:
                                self._write_weekly_trend_lines(
                                    report, project_weekly_trends, "Project "
                                )
                            else:
                                # Fallback to simple project trend analysis
                                project_trends = self._calculate_classification_trends(
                                    commits, project_key=project, weeks=weeks
                                )
                                project_trend_line = self._format_trend_line(
                                    project_trends, prefix="📊 Weekly Trend"
                                )
                                if project_trend_line:
                                    report.write(f"- {project_trend_line}\n")

            # Add branch health for this project/repository if available
            if branch_health_metrics and project in branch_health_metrics:
                repo_health = branch_health_metrics[project]
                summary = repo_health.get("summary", {})
                health_indicators = repo_health.get("health_indicators", {})
                branches = repo_health.get("branches", [])

                health_score = health_indicators.get("overall_health_score", 0)
                total_branches = summary.get("total_branches", 0)
                stale_branches = summary.get("stale_branches", 0)
                active_branches = summary.get("active_branches", 0)
                long_lived_branches = summary.get("long_lived_branches", 0)

                # Determine health status
                if health_score >= 80:
                    status_emoji = "🟢"
                    status_text = "Excellent"
                elif health_score >= 60:
                    status_emoji = "🟡"
                    status_text = "Good"
                elif health_score >= 40:
                    status_emoji = "🟠"
                    status_text = "Fair"
                else:
                    status_emoji = "🔴"
                    status_text = "Needs Attention"

                report.write("\n**Branch Management**\n")
                report.write(
                    f"- Overall Health: {status_emoji} {status_text} ({health_score:.0f}/100)\n"
                )
                report.write(f"- Total Branches: {total_branches}\n")
                report.write(f"  - Active: {active_branches} branches\n")
                report.write(f"  - Long-lived: {long_lived_branches} branches (>30 days)\n")
                report.write(f"  - Stale: {stale_branches} branches (>90 days)\n")

                # Show top problematic branches if any
                if branches:
                    # Sort branches by health score (ascending) to get worst first
                    problem_branches = [
                        b
                        for b in branches
                        if b.get("health_score", 100) < 60 and not b.get("is_merged", False)
                    ]
                    problem_branches.sort(key=lambda x: x.get("health_score", 100))

                    if problem_branches:
                        report.write("\n**Branches Needing Attention**:\n")
                        for i, branch in enumerate(problem_branches[:3]):  # Show top 3
                            name = branch.get("name", "unknown")
                            age = branch.get("age_days", 0)
                            behind = branch.get("behind_main", 0)
                            ahead = branch.get("ahead_of_main", 0)
                            score = branch.get("health_score", 0)

                            report.write(f"  {i + 1}. `{name}` (score: {score:.0f}/100)\n")
                            report.write(f"     - Age: {age} days\n")
                            if behind > 0:
                                report.write(f"     - Behind main: {behind} commits\n")
                            if ahead > 0:
                                report.write(f"     - Ahead of main: {ahead} commits\n")

                # Add recommendations
                recommendations = repo_health.get("recommendations", [])
                if recommendations:
                    report.write("\n**Recommended Actions**:\n")
                    for rec in recommendations[:3]:  # Show top 3 recommendations
                        report.write(f"- {rec}\n")

            report.write("\n")

