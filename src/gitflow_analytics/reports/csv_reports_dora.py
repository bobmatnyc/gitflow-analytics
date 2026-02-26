"""Story point, velocity, DORA metrics, and CI/CD CSV reports.

Extracted from csv_writer.py to keep file sizes manageable.
Contains generate_story_point_correlation_report, generate_weekly_velocity_report,
generate_weekly_dora_report, and generate_weekly_cicd_report.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CSVDoraReportsMixin:
    """Mixin providing DORA/velocity/CI/CD CSV reports for CSVReportGenerator.

    Attributes expected from host class:
        anonymize, _anonymization_map, _anonymous_counter
    """

    def generate_story_point_correlation_report(
        self,
        commits: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        pm_data: Optional[dict[str, Any]],
        output_path: Path,
        weeks: int = 12,
    ) -> Path:
        """Generate story point correlation analysis CSV report.

        WHY: Story point correlation analysis helps teams understand the relationship
        between estimated effort (story points) and actual work metrics (commits,
        lines of code, time). This enables process improvements and better estimation
        calibration.

        INTEGRATION: Uses the StoryPointCorrelationAnalyzer to provide comprehensive
        correlation metrics including weekly trends, developer accuracy, and velocity
        analysis in a format suitable for spreadsheet analysis.

        Args:
            commits: List of commit data with story points
            prs: List of pull request data
            pm_data: PM platform data with issue correlations
            output_path: Path for the output CSV file
            weeks: Number of weeks to analyze

        Returns:
            Path to the generated CSV report
        """
        try:
            # Import here to avoid circular imports
            from .story_point_correlation import StoryPointCorrelationAnalyzer

            # Create analyzer with same configuration as CSV writer
            analyzer = StoryPointCorrelationAnalyzer(
                anonymize=self.anonymize, identity_resolver=self.identity_resolver
            )

            # Apply exclusion filtering consistent with other reports
            commits = self._filter_excluded_authors_list(commits)

            # Generate the correlation report
            logger.debug(f"Generating story point correlation report: {output_path}")
            return analyzer.generate_correlation_report(commits, prs, pm_data, output_path, weeks)

        except Exception as e:
            logger.error(f"Error generating story point correlation report: {e}")

            # Create empty report as fallback
            headers = [
                "week_start",
                "metric_type",
                "developer_name",
                "sp_commits_correlation",
                "sp_lines_correlation",
                "sp_files_correlation",
                "sp_prs_correlation",
                "sp_complexity_correlation",
                "sample_size",
                "total_story_points",
                "total_commits",
                "story_points_completed",
                "commits_count",
                "prs_merged",
                "developers_active",
                "velocity_trend",
                "overall_accuracy",
                "avg_weekly_accuracy",
                "consistency",
                "weeks_active",
                "total_estimated_sp",
                "total_actual_sp",
                "estimation_ratio",
            ]

            df = pd.DataFrame(columns=headers)
            df.to_csv(output_path, index=False)

            raise

    def generate_weekly_velocity_report(
        self,
        commits: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        output_path: Path,
        weeks: int = 12,
    ) -> Path:
        """Generate weekly lines-per-story-point velocity analysis report.

        WHY: Velocity analysis helps teams understand the relationship between
        estimated effort (story points) and actual work performed (lines of code).
        This enables process improvements, better estimation calibration, and
        identification of efficiency trends over time.

        DESIGN DECISION: Combines both PR-based and commit-based story points
        to provide comprehensive coverage, as some organizations track story
        points differently across their development workflow.

        Args:
            commits: List of commit data dictionaries with story points
            prs: List of pull request data dictionaries with story points
            output_path: Path where the CSV report should be written
            weeks: Number of weeks to analyze (default: 12)

        Returns:
            Path to the generated CSV file
        """
        # Apply exclusion filtering in Phase 2
        commits = self._filter_excluded_authors_list(commits)

        # Calculate date range (timezone-aware to match commit timestamps)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)

        logger.debug("Weekly velocity report date range:")
        logger.debug(f"  start_date: {start_date} (tzinfo: {start_date.tzinfo})")
        logger.debug(f"  end_date: {end_date} (tzinfo: {end_date.tzinfo})")

        # Initialize weekly aggregation structures
        weekly_data: dict[datetime, dict[str, Any]] = defaultdict(
            lambda: {
                "total_story_points": 0,
                "pr_story_points": 0,
                "commit_story_points": 0,
                "total_lines": 0,
                "lines_added": 0,
                "lines_removed": 0,
                "files_changed": 0,
                "commits_count": 0,
                "developers": set(),
                "prs_with_sp": 0,
                "commits_with_sp": 0,
            }
        )

        # Process commits for weekly aggregation
        for commit in commits:
            timestamp = commit["timestamp"]
            logger.debug(
                f"Processing commit timestamp: {timestamp} (tzinfo: {getattr(timestamp, 'tzinfo', 'N/A')})"
            )

            # Ensure consistent timezone handling
            if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is not None:
                if timestamp.tzinfo != timezone.utc:
                    timestamp = timestamp.astimezone(timezone.utc)
            else:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            # Check date range
            if self._safe_datetime_compare(
                timestamp, start_date, "lt", "generate_weekly_velocity_report range check"
            ) or self._safe_datetime_compare(
                timestamp, end_date, "gt", "generate_weekly_velocity_report range check"
            ):
                continue

            # Get week start (Monday)
            week_start = self._get_week_start(timestamp)
            week_data = weekly_data[week_start]

            # Aggregate commit metrics
            story_points = commit.get("story_points", 0) or 0
            lines_added = commit.get("filtered_insertions", commit.get("insertions", 0)) or 0
            lines_removed = commit.get("filtered_deletions", commit.get("deletions", 0)) or 0
            files_changed = (
                commit.get("filtered_files_changed", commit.get("files_changed", 0)) or 0
            )

            week_data["commits_count"] += 1
            week_data["commit_story_points"] += story_points
            week_data["total_story_points"] += story_points
            week_data["lines_added"] += lines_added
            week_data["lines_removed"] += lines_removed
            week_data["total_lines"] += lines_added + lines_removed
            week_data["files_changed"] += files_changed

            # Track developers and story point coverage
            developer_id = commit.get("canonical_id", commit.get("author_email", "unknown"))
            week_data["developers"].add(developer_id)

            if story_points > 0:
                week_data["commits_with_sp"] += 1

        # Process PRs for weekly aggregation (by merge date or creation date)
        for pr in prs:
            # Use merged_at if available and valid, otherwise created_at
            pr_date = pr.get("merged_at") or pr.get("created_at")
            if not pr_date:
                continue

            # Handle string dates (convert to datetime if needed)
            if isinstance(pr_date, str):
                try:
                    from dateutil.parser import parse

                    pr_date = parse(pr_date)
                except Exception:
                    continue

            # Ensure timezone consistency
            if hasattr(pr_date, "tzinfo") and pr_date.tzinfo is not None:
                if pr_date.tzinfo != timezone.utc:
                    pr_date = pr_date.astimezone(timezone.utc)
            else:
                pr_date = pr_date.replace(tzinfo=timezone.utc)

            # Check date range
            if self._safe_datetime_compare(
                pr_date, start_date, "lt", "generate_weekly_velocity_report PR range check"
            ) or self._safe_datetime_compare(
                pr_date, end_date, "gt", "generate_weekly_velocity_report PR range check"
            ):
                continue

            # Get week start
            week_start = self._get_week_start(pr_date)
            week_data = weekly_data[week_start]

            # Aggregate PR metrics
            story_points = pr.get("story_points", 0) or 0
            if story_points > 0:
                week_data["pr_story_points"] += story_points
                week_data["total_story_points"] += story_points
                week_data["prs_with_sp"] += 1

                # Track developer from PR
                developer_id = pr.get("canonical_id", pr.get("author", "unknown"))
                week_data["developers"].add(developer_id)

        # Build CSV rows with velocity metrics
        rows = []
        previous_week_lines_per_point = None

        for week_start in sorted(weekly_data.keys()):
            week_data = weekly_data[week_start]
            total_story_points = week_data["total_story_points"]
            total_lines = week_data["total_lines"]

            # Calculate key metrics with division by zero protection
            lines_per_point = (total_lines / total_story_points) if total_story_points > 0 else 0
            commits_per_point = (
                (week_data["commits_count"] / total_story_points) if total_story_points > 0 else 0
            )

            # Calculate efficiency score (inverse of lines per point, normalized to 0-100 scale)
            # Higher efficiency = fewer lines needed per story point
            if lines_per_point > 0:
                # Use a logarithmic scale to handle wide ranges
                import math

                efficiency_score = max(0, 100 - (math.log10(max(lines_per_point, 1)) * 20))
            else:
                efficiency_score = 0

            # Calculate velocity trend (week-over-week change in lines per point)
            if previous_week_lines_per_point is not None and previous_week_lines_per_point > 0:
                if lines_per_point > 0:
                    velocity_trend = (
                        (lines_per_point - previous_week_lines_per_point)
                        / previous_week_lines_per_point
                    ) * 100
                else:
                    velocity_trend = -100  # Went from some lines per point to zero
            else:
                velocity_trend = 0  # No previous data for comparison

            row = {
                "week_start": week_start.strftime("%Y-%m-%d"),
                "total_story_points": total_story_points,
                "pr_story_points": week_data["pr_story_points"],
                "commit_story_points": week_data["commit_story_points"],
                "total_lines": total_lines,
                "lines_added": week_data["lines_added"],
                "lines_removed": week_data["lines_removed"],
                "files_changed": week_data["files_changed"],
                "lines_per_point": round(lines_per_point, 2) if lines_per_point > 0 else 0,
                "commits_per_point": round(commits_per_point, 2) if commits_per_point > 0 else 0,
                "developers_involved": len(week_data["developers"]),
                "efficiency_score": round(efficiency_score, 1),
                "velocity_trend": round(velocity_trend, 1),
                # Additional metrics for deeper analysis
                "commits_count": week_data["commits_count"],
                "prs_with_story_points": week_data["prs_with_sp"],
                "commits_with_story_points": week_data["commits_with_sp"],
                "story_point_coverage_pct": round(
                    (week_data["commits_with_sp"] / max(week_data["commits_count"], 1)) * 100, 1
                ),
                "avg_lines_per_commit": round(total_lines / max(week_data["commits_count"], 1), 1),
                "avg_files_per_commit": round(
                    week_data["files_changed"] / max(week_data["commits_count"], 1), 1
                ),
            }
            rows.append(row)

            # Store for next iteration's trend calculation
            previous_week_lines_per_point = lines_per_point if lines_per_point > 0 else None

        # Write CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        else:
            # Write empty CSV with headers
            headers = [
                "week_start",
                "total_story_points",
                "pr_story_points",
                "commit_story_points",
                "total_lines",
                "lines_added",
                "lines_removed",
                "files_changed",
                "lines_per_point",
                "commits_per_point",
                "developers_involved",
                "efficiency_score",
                "velocity_trend",
                "commits_count",
                "prs_with_story_points",
                "commits_with_story_points",
                "story_point_coverage_pct",
                "avg_lines_per_commit",
                "avg_files_per_commit",
            ]
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

        return output_path

    def generate_weekly_dora_report(
        self,
        commits: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        output_path: Path,
        weeks: int = 12,
    ) -> Path:
        """Generate weekly DORA metrics CSV report.

        WHY: Weekly DORA metrics provide trend analysis for software delivery
        performance, enabling teams to track improvements and identify periods
        of degraded performance across the four key metrics.

        DESIGN DECISION: Uses the DORAMetricsCalculator with weekly breakdown
        to provide consistent methodology while adding trend analysis and
        rolling averages for smoother interpretation.

        Args:
            commits: List of commit data dictionaries
            prs: List of pull request data dictionaries
            output_path: Path where the CSV report should be written
            weeks: Number of weeks to analyze (default: 12)

        Returns:
            Path to the generated CSV file
        """
        from ..metrics.dora import DORAMetricsCalculator

        # Apply exclusion filtering in Phase 2
        commits = self._filter_excluded_authors_list(commits)

        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)

        # Initialize DORA calculator
        dora_calculator = DORAMetricsCalculator()

        try:
            # Calculate weekly DORA metrics
            weekly_metrics = dora_calculator.calculate_weekly_dora_metrics(
                commits=commits,
                prs=prs,
                start_date=start_date,
                end_date=end_date,
            )

            if not weekly_metrics:
                # Generate empty report with headers
                headers = [
                    "week_start",
                    "week_end",
                    "deployment_frequency",
                    "lead_time_hours",
                    "change_failure_rate",
                    "mttr_hours",
                    "total_failures",
                    "total_commits",
                    "total_prs",
                    "deployment_frequency_4w_avg",
                    "lead_time_4w_avg",
                    "change_failure_rate_4w_avg",
                    "mttr_4w_avg",
                    "deployment_frequency_change_pct",
                    "lead_time_change_pct",
                    "change_failure_rate_change_pct",
                    "mttr_change_pct",
                    "deployment_frequency_trend",
                    "lead_time_trend",
                    "change_failure_rate_trend",
                    "mttr_trend",
                ]

                df = pd.DataFrame(columns=headers)
                df.to_csv(output_path, index=False)
                return output_path

            # Convert to DataFrame and write CSV
            df = pd.DataFrame(weekly_metrics)
            df.to_csv(output_path, index=False)

            return output_path

        except Exception as e:
            logger.error(f"Error generating weekly DORA report: {e}")

            # Create empty report as fallback
            headers = [
                "week_start",
                "week_end",
                "deployment_frequency",
                "lead_time_hours",
                "change_failure_rate",
                "mttr_hours",
                "total_failures",
                "total_commits",
                "total_prs",
                "deployment_frequency_4w_avg",
                "lead_time_4w_avg",
                "change_failure_rate_4w_avg",
                "mttr_4w_avg",
                "deployment_frequency_change_pct",
                "lead_time_change_pct",
                "change_failure_rate_change_pct",
                "mttr_change_pct",
                "deployment_frequency_trend",
                "lead_time_trend",
                "change_failure_rate_trend",
                "mttr_trend",
            ]

            df = pd.DataFrame(columns=headers)
            df.to_csv(output_path, index=False)

            raise

    def generate_weekly_cicd_report(
        self,
        cicd_data: dict[str, Any],
        commits: list[dict[str, Any]],
        output_path: Path,
        weeks: int = 12,
    ) -> Path:
        """Generate weekly CI/CD pipeline metrics CSV report.

        WHY: CI/CD pipeline success rates and build durations are key indicators of
        development velocity and system stability. This report enables teams to track
        pipeline health trends, identify periods of instability, and correlate with
        deployment frequency (DORA metrics).

        DESIGN DECISION: Processes all pipelines regardless of platform (GitHub Actions,
        Jenkins, etc.) to provide unified visibility into CI/CD health across the entire
        development workflow.

        Args:
            cicd_data: CI/CD data from orchestrator with pipelines and metrics
            commits: List of commit data for correlation
            output_path: Path where the CSV report should be written
            weeks: Number of weeks to analyze (default: 12)

        Returns:
            Path to the generated CSV file
        """
        # Calculate date range (timezone-aware to match timestamps)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)

        logger.debug("Weekly CI/CD report date range:")
        logger.debug(f"  start_date: {start_date} (tzinfo: {start_date.tzinfo})")
        logger.debug(f"  end_date: {end_date} (tzinfo: {end_date.tzinfo})")

        # Extract pipelines and metrics
        pipelines = cicd_data.get("pipelines", [])
        cicd_data.get("metrics", {})

        if not pipelines:
            # Generate empty report with headers
            headers = [
                "week_start",
                "platform",
                "total_pipelines",
                "successful_pipelines",
                "failed_pipelines",
                "success_rate",
                "avg_duration_minutes",
                "total_duration_minutes",
                "workflows_count",
                "unique_commits",
                "pipelines_per_commit",
            ]
            df = pd.DataFrame(columns=headers)
            df.to_csv(output_path, index=False)
            return output_path

        # Initialize weekly aggregation structures
        weekly_data: dict[tuple[datetime, str], dict[str, Any]] = defaultdict(
            lambda: {
                "total_pipelines": 0,
                "successful_pipelines": 0,
                "failed_pipelines": 0,
                "total_duration_minutes": 0.0,
                "workflows": set(),
                "commits": set(),
            }
        )

        # Process pipelines for weekly aggregation
        for pipeline in pipelines:
            # Parse created_at or started_at timestamp
            created_at = pipeline.get("created_at") or pipeline.get("started_at")
            if not created_at:
                continue

            # Handle string dates (convert to datetime if needed)
            if isinstance(created_at, str):
                try:
                    from dateutil.parser import parse

                    created_at = parse(created_at)
                except Exception:
                    continue

            # Ensure timezone consistency
            if hasattr(created_at, "tzinfo") and created_at.tzinfo is not None:
                if created_at.tzinfo != timezone.utc:
                    created_at = created_at.astimezone(timezone.utc)
            else:
                created_at = created_at.replace(tzinfo=timezone.utc)

            # Check date range
            if self._safe_datetime_compare(
                created_at, start_date, "lt", "generate_weekly_cicd_report range check"
            ) or self._safe_datetime_compare(
                created_at, end_date, "gt", "generate_weekly_cicd_report range check"
            ):
                continue

            # Get week start (Monday)
            week_start = self._get_week_start(created_at)

            # Get platform
            platform = pipeline.get("platform", "unknown")

            # Aggregate metrics by week and platform
            key = (week_start, platform)
            week_data = weekly_data[key]

            week_data["total_pipelines"] += 1

            # Count success/failure
            status = pipeline.get("status", "").lower()
            if status == "success":
                week_data["successful_pipelines"] += 1
            elif status in ["failure", "failed"]:
                week_data["failed_pipelines"] += 1

            # Aggregate duration (in minutes)
            duration_seconds = pipeline.get("duration_seconds", 0)
            if duration_seconds:
                week_data["total_duration_minutes"] += duration_seconds / 60.0

            # Track unique workflows and commits
            workflow_name = pipeline.get("workflow_name") or pipeline.get("name")
            if workflow_name:
                week_data["workflows"].add(workflow_name)

            commit_sha = pipeline.get("commit_sha") or pipeline.get("head_sha")
            if commit_sha:
                week_data["commits"].add(commit_sha)

        # Build CSV rows with CI/CD metrics
        rows = []

        for (week_start, platform), week_data in sorted(weekly_data.items()):
            total_pipelines = week_data["total_pipelines"]
            successful_pipelines = week_data["successful_pipelines"]
            failed_pipelines = week_data["failed_pipelines"]

            # Calculate success rate
            success_rate = (
                (successful_pipelines / total_pipelines * 100) if total_pipelines > 0 else 0
            )

            # Calculate average duration
            avg_duration_minutes = (
                week_data["total_duration_minutes"] / total_pipelines if total_pipelines > 0 else 0
            )

            # Calculate pipelines per commit
            unique_commits = len(week_data["commits"])
            pipelines_per_commit = total_pipelines / unique_commits if unique_commits > 0 else 0

            row = {
                "week_start": week_start.strftime("%Y-%m-%d"),
                "platform": platform,
                "total_pipelines": total_pipelines,
                "successful_pipelines": successful_pipelines,
                "failed_pipelines": failed_pipelines,
                "success_rate": round(success_rate, 1),
                "avg_duration_minutes": round(avg_duration_minutes, 2),
                "total_duration_minutes": round(week_data["total_duration_minutes"], 2),
                "workflows_count": len(week_data["workflows"]),
                "unique_commits": unique_commits,
                "pipelines_per_commit": round(pipelines_per_commit, 2),
            }
            rows.append(row)

        # Write CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        else:
            # Write empty CSV with headers
            headers = [
                "week_start",
                "platform",
                "total_pipelines",
                "successful_pipelines",
                "failed_pipelines",
                "success_rate",
                "avg_duration_minutes",
                "total_duration_minutes",
                "workflows_count",
                "unique_commits",
                "pipelines_per_commit",
            ]
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

        return output_path
