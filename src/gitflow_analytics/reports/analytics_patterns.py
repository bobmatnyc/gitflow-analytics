"""Pattern analysis helpers for analytics reports.

Extracted from analytics_writer.py to keep file sizes manageable.
These helpers analyze commit and developer patterns for qualitative insights,
and provide the weekly developer/project trend CSV writing helpers.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Get logger for this module
logger = logging.getLogger(__name__)


class AnalyticsPatternseMixin:
    """Mixin providing pattern analysis helpers for AnalyticsReportGenerator.

    Contains the private analysis methods that support generate_qualitative_insights_report
    and generate_weekly_trends_report. Kept separate to reduce file size.
    """

    def _analyze_commit_patterns(self, commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze patterns in commit data."""
        insights = []

        # Time-based patterns (use local hour if available)
        commit_hours = []
        for c in commits:
            if "local_hour" in c:
                commit_hours.append(c["local_hour"])
            elif hasattr(c["timestamp"], "hour"):
                commit_hours.append(c["timestamp"].hour)

        if commit_hours:
            peak_hour = max(set(commit_hours), key=commit_hours.count)
            insights.append(
                {
                    "category": "Timing",
                    "insight": "Peak commit hour",
                    "value": f"{peak_hour}:00",
                    "impact": "Indicates team working hours",
                }
            )

        # Commit message patterns
        message_lengths = [len(c["message"].split()) for c in commits]
        if message_lengths:
            avg_message_length = np.mean(message_lengths)

            if avg_message_length < 5:
                quality = "Very brief"
            elif avg_message_length < 10:
                quality = "Concise"
            elif avg_message_length < 20:
                quality = "Detailed"
            else:
                quality = "Very detailed"

            insights.append(
                {
                    "category": "Quality",
                    "insight": "Commit message quality",
                    "value": quality,
                    "impact": f"Average {avg_message_length:.1f} words per message",
                }
            )

        # Ticket coverage insights
        commits_with_tickets = sum(1 for c in commits if c.get("ticket_references"))
        coverage_pct = commits_with_tickets / len(commits) * 100 if commits else 0

        if coverage_pct < 30:
            tracking = "Poor tracking"
        elif coverage_pct < 60:
            tracking = "Moderate tracking"
        elif coverage_pct < 80:
            tracking = "Good tracking"
        else:
            tracking = "Excellent tracking"

        insights.append(
            {
                "category": "Process",
                "insight": "Ticket tracking adherence",
                "value": tracking,
                "impact": f"{coverage_pct:.1f}% commits have ticket references",
            }
        )

        return insights

    def _analyze_developer_patterns(
        self, commits: List[Dict[str, Any]], developer_stats: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze developer behavior patterns."""
        insights = []

        # Team size insights
        team_size = len(developer_stats)
        if team_size < 3:
            team_type = "Very small team"
        elif team_size < 6:
            team_type = "Small team"
        elif team_size < 12:
            team_type = "Medium team"
        else:
            team_type = "Large team"

        insights.append(
            {
                "category": "Team",
                "insight": "Team size",
                "value": team_type,
                "impact": f"{team_size} active developers",
            }
        )

        # Contribution distribution
        commit_counts = [dev["total_commits"] for dev in developer_stats]
        if commit_counts:
            gini_coef = self._calculate_gini_coefficient(commit_counts)

            if gini_coef < 0.3:
                distribution = "Very balanced"
            elif gini_coef < 0.5:
                distribution = "Moderately balanced"
            elif gini_coef < 0.7:
                distribution = "Somewhat unbalanced"
            else:
                distribution = "Highly concentrated"

            insights.append(
                {
                    "category": "Team",
                    "insight": "Work distribution",
                    "value": distribution,
                    "impact": f"Gini coefficient: {gini_coef:.2f}",
                }
            )

        return insights

    def _analyze_collaboration_patterns(
        self, commits: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze collaboration patterns."""
        insights = []

        # Merge commit analysis
        merge_commits = [c for c in commits if c.get("is_merge")]
        merge_pct = len(merge_commits) / len(commits) * 100 if commits else 0

        if merge_pct < 5:
            branching = "Minimal branching"
        elif merge_pct < 15:
            branching = "Moderate branching"
        elif merge_pct < 25:
            branching = "Active branching"
        else:
            branching = "Heavy branching"

        insights.append(
            {
                "category": "Workflow",
                "insight": "Branching strategy",
                "value": branching,
                "impact": f"{merge_pct:.1f}% merge commits",
            }
        )

        return insights

    def _analyze_work_distribution(self, commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze work distribution patterns."""
        insights = []

        # File change patterns
        file_changes = []
        for c in commits:
            files_count = self._get_files_changed_count(c)
            if files_count > 0:
                file_changes.append(files_count)
        if file_changes:
            avg_files = np.mean(file_changes)

            if avg_files < 3:
                pattern = "Focused changes"
            elif avg_files < 8:
                pattern = "Moderate scope changes"
            else:
                pattern = "Broad scope changes"

            insights.append(
                {
                    "category": "Workflow",
                    "insight": "Change scope pattern",
                    "value": pattern,
                    "impact": f"Average {avg_files:.1f} files per commit",
                }
            )

        return insights

    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for distribution analysis."""
        if not values or len(values) == 1:
            return 0.0

        sorted_values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        # Use builtin sum instead of np.sum for generator expression (numpy deprecation)
        return (2 * sum((i + 1) * sorted_values[i] for i in range(n))) / (n * cumsum[-1]) - (
            n + 1
        ) / n

    def _get_week_start(self, date: datetime) -> datetime:
        """Get Monday of the week for a given date."""
        logger.debug(
            f"Getting week start for date: {date} (tzinfo: {getattr(date, 'tzinfo', 'N/A')})"
        )

        # Ensure consistent timezone handling - keep timezone info
        if hasattr(date, "tzinfo") and date.tzinfo is not None:
            # Keep timezone-aware but ensure it's UTC
            if date.tzinfo != timezone.utc:
                date = date.astimezone(timezone.utc)
                logger.debug(f"  Converted to UTC: {date}")
        else:
            # Convert naive datetime to UTC timezone-aware
            date = date.replace(tzinfo=timezone.utc)
            logger.debug(f"  Made timezone-aware: {date}")

        days_since_monday = date.weekday()
        monday = date - timedelta(days=days_since_monday)
        result = monday.replace(hour=0, minute=0, second=0, microsecond=0)

        logger.debug(f"  Week start result: {result} (tzinfo: {result.tzinfo})")
        return result

    def _anonymize_value(self, value: str, field_type: str) -> str:
        """Anonymize a value if anonymization is enabled."""
        if not self.anonymize or not value:
            return value

        if value not in self._anonymization_map:
            self._anonymous_counter += 1
            if field_type == "name":
                anonymous = f"Developer{self._anonymous_counter}"
            else:
                anonymous = f"anon{self._anonymous_counter}"
            self._anonymization_map[value] = anonymous

        return self._anonymization_map[value]

    # ---------------------------------------------------------------------------
    # Developer focus report helpers
    # ---------------------------------------------------------------------------

    def _add_project_columns_to_focus_row(
        self,
        row: Dict[str, Any],
        all_projects: List[str],
        projects: Any,
        project_lines: Any,
        project_totals: Any,
        dev_commits: List[Any],
        commit_sizes: List[float],
        total_commits: int,
    ) -> None:
        """Add per-project metric columns to a developer focus row dict.

        Mutates ``row`` in place with gross/adjusted commits and percentage columns
        for each project in ``all_projects``.
        """
        for project in all_projects:
            gross_commits = projects.get(project, 0)
            row[f"{project}_gross_commits"] = gross_commits

            if gross_commits > 0 and project_lines[project] > 0:
                project_avg_lines = project_lines[project] / gross_commits
                overall_avg_lines = sum(commit_sizes) / len(commit_sizes) if commit_sizes else 1
                adjustment_factor = project_avg_lines / overall_avg_lines if overall_avg_lines > 0 else 1
                adjusted_commits = round(gross_commits * adjustment_factor, 1)
            else:
                adjusted_commits = 0
            row[f"{project}_adjusted_commits"] = adjusted_commits

            dev_pct = (
                round(gross_commits / len(dev_commits) * 100, 1) if dev_commits else 0
            )
            row[f"{project}_dev_pct"] = dev_pct

            proj_pct = (
                round(gross_commits / project_totals[project] * 100, 1)
                if project_totals[project] > 0
                else 0
            )
            row[f"{project}_proj_pct"] = proj_pct

            total_pct = round(gross_commits / total_commits * 100, 1) if total_commits > 0 else 0
            row[f"{project}_total_pct"] = total_pct

    # ---------------------------------------------------------------------------
    # Weekly trend CSV helpers (used by generate_weekly_trends_report)
    # ---------------------------------------------------------------------------

    def _write_developer_trends_csv(
        self,
        output_path: Path,
        developer_weekly: Dict[str, Any],
        dev_lookup: Dict[str, Any],
        sorted_weeks: List[str],
    ) -> None:
        """Write per-developer weekly trends to a companion CSV file."""
        dev_trends_path = (
            output_path.parent / f'developer_trends_{output_path.stem.split("_")[-1]}.csv'
        )
        dev_trend_rows = []

        for dev_id, weekly_commits in developer_weekly.items():
            dev_info = dev_lookup.get(dev_id, {})
            dev_name = self._anonymize_value(
                self._get_canonical_display_name(dev_id, dev_info.get("primary_name", "Unknown")),
                "name",
            )

            weekly_values = [
                weekly_commits.get(week, {}).get("commits", 0) for week in sorted_weeks
            ]

            if sum(weekly_values) == 0:
                continue

            changes = [weekly_values[i] - weekly_values[i - 1] for i in range(1, len(weekly_values))]
            avg_change = sum(changes) / len(changes) if changes else 0
            volatility = np.std(changes) if len(changes) > 1 else 0
            trend = (
                "increasing" if avg_change > 1 else "decreasing" if avg_change < -1 else "stable"
            )

            row: Dict[str, Any] = {
                "developer": dev_name,
                "total_commits": sum(weekly_values),
                "avg_weekly_commits": round(sum(weekly_values) / len(weekly_values), 1),
                "avg_weekly_change": round(avg_change, 1),
                "volatility": round(float(volatility), 1),
                "trend": trend,
                "total_weeks_active": sum(1 for v in weekly_values if v > 0),
                "max_week": max(weekly_values),
                "min_week": min((v for v in weekly_values if v > 0), default=0),
            }
            for i, week in enumerate(sorted_weeks):
                row[f"week_{i + 1}_{week}"] = weekly_values[i]

            dev_trend_rows.append(row)

        if dev_trend_rows:
            dev_trends_df = pd.DataFrame(dev_trend_rows)
            dev_trends_df.sort_values("total_commits", ascending=False, inplace=True)
            dev_trends_df.to_csv(dev_trends_path, index=False)

    def _write_project_trends_csv(
        self,
        output_path: Path,
        project_weekly: Dict[str, Any],
        sorted_weeks: List[str],
    ) -> None:
        """Write per-project weekly trends to a companion CSV file."""
        proj_trends_path = (
            output_path.parent / f'project_trends_{output_path.stem.split("_")[-1]}.csv'
        )
        proj_trend_rows = []

        for project, weekly_commits in project_weekly.items():
            weekly_values = [
                weekly_commits.get(week, {}).get("commits", 0) for week in sorted_weeks
            ]
            weekly_developers = [
                len(weekly_commits.get(week, {}).get("developers", set()))
                for week in sorted_weeks
            ]

            if sum(weekly_values) == 0:
                continue

            changes = [weekly_values[i] - weekly_values[i - 1] for i in range(1, len(weekly_values))]
            avg_change = sum(changes) / len(changes) if changes else 0
            volatility = np.std(changes) if len(changes) > 1 else 0
            trend = (
                "growing" if avg_change > 2 else "shrinking" if avg_change < -2 else "stable"
            )

            row: Dict[str, Any] = {
                "project": project,
                "total_commits": sum(weekly_values),
                "avg_weekly_commits": round(sum(weekly_values) / len(weekly_values), 1),
                "avg_weekly_developers": round(
                    sum(weekly_developers) / len(weekly_developers), 1
                ),
                "avg_weekly_change": round(avg_change, 1),
                "volatility": round(float(volatility), 1),
                "trend": trend,
                "total_weeks_active": sum(1 for v in weekly_values if v > 0),
                "max_week": max(weekly_values),
                "min_week": min((v for v in weekly_values if v > 0), default=0),
            }
            for i, week in enumerate(sorted_weeks):
                row[f"week_{i + 1}_{week}"] = weekly_values[i]
            for i in range(len(sorted_weeks)):
                row[f"devs_week_{i + 1}"] = weekly_developers[i]

            proj_trend_rows.append(row)

        if proj_trend_rows:
            proj_trends_df = pd.DataFrame(proj_trend_rows)
            proj_trends_df.sort_values("total_commits", ascending=False, inplace=True)
            proj_trends_df.to_csv(proj_trends_path, index=False)
