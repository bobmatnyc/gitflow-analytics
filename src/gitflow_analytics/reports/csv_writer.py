"""CSV report generation for GitFlow Analytics."""

import csv
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..metrics.activity_scoring import ActivityScorer

# Get logger for this module
logger = logging.getLogger(__name__)


class CSVReportGenerator:
    """Generate CSV reports with weekly metrics."""

    def __init__(self, anonymize: bool = False):
        """Initialize report generator."""
        self.anonymize = anonymize
        self._anonymization_map: dict[str, str] = {}
        self._anonymous_counter = 0
        self.activity_scorer = ActivityScorer()

    def _log_datetime_comparison(
        self, dt1: datetime, dt2: datetime, operation: str, location: str
    ) -> None:
        """Log datetime comparison details for debugging timezone issues."""
        logger.debug(f"Comparing dates in {location} ({operation}):")
        logger.debug(f"  dt1: {dt1} (tzinfo: {dt1.tzinfo}, aware: {dt1.tzinfo is not None})")
        logger.debug(f"  dt2: {dt2} (tzinfo: {dt2.tzinfo}, aware: {dt2.tzinfo is not None})")

    def _safe_datetime_compare(
        self, dt1: datetime, dt2: datetime, operation: str, location: str
    ) -> bool:
        """Safely compare datetimes with logging and error handling."""
        try:
            self._log_datetime_comparison(dt1, dt2, operation, location)

            if operation == "lt":
                result = dt1 < dt2
            elif operation == "gt":
                result = dt1 > dt2
            elif operation == "le":
                result = dt1 <= dt2
            elif operation == "ge":
                result = dt1 >= dt2
            elif operation == "eq":
                result = dt1 == dt2
            else:
                raise ValueError(f"Unknown operation: {operation}")

            logger.debug(f"  Result: {result}")
            return result

        except TypeError as e:
            logger.error(f"Timezone comparison error in {location}:")
            logger.error(
                f"  dt1: {dt1} (type: {type(dt1)}, tzinfo: {getattr(dt1, 'tzinfo', 'N/A')})"
            )
            logger.error(
                f"  dt2: {dt2} (type: {type(dt2)}, tzinfo: {getattr(dt2, 'tzinfo', 'N/A')})"
            )
            logger.error(f"  Operation: {operation}")
            logger.error(f"  Error: {e}")

            # Import traceback for detailed error info
            import traceback

            logger.error(f"  Full traceback:\n{traceback.format_exc()}")

            # Try to fix by making both timezone-aware in UTC
            try:
                if dt1.tzinfo is None:
                    dt1 = dt1.replace(tzinfo=timezone.utc)
                    logger.debug(f"  Fixed dt1 to UTC: {dt1}")
                if dt2.tzinfo is None:
                    dt2 = dt2.replace(tzinfo=timezone.utc)
                    logger.debug(f"  Fixed dt2 to UTC: {dt2}")

                # Retry comparison
                if operation == "lt":
                    result = dt1 < dt2
                elif operation == "gt":
                    result = dt1 > dt2
                elif operation == "le":
                    result = dt1 <= dt2
                elif operation == "ge":
                    result = dt1 >= dt2
                elif operation == "eq":
                    result = dt1 == dt2
                else:
                    raise ValueError(f"Unknown operation: {operation}")

                logger.info(f"  Fixed comparison result: {result}")
                return result

            except Exception as fix_error:
                logger.error(f"  Failed to fix timezone issue: {fix_error}")
                raise

    def _safe_datetime_format(self, dt: datetime, format_str: str) -> str:
        """Safely format datetime with logging."""
        try:
            logger.debug(
                f"Formatting datetime: {dt} (tzinfo: {getattr(dt, 'tzinfo', 'N/A')}) with format {format_str}"
            )
            result = dt.strftime(format_str)
            logger.debug(f"  Format result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error formatting datetime {dt}: {e}")
            return str(dt)

    def generate_weekly_report(
        self,
        commits: list[dict[str, Any]],
        developer_stats: list[dict[str, Any]],
        output_path: Path,
        weeks: int = 12,
    ) -> Path:
        """Generate weekly metrics CSV report."""
        # Calculate week boundaries (timezone-aware to match commit timestamps)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)

        logger.debug("Weekly report date range:")
        logger.debug(f"  start_date: {start_date} (tzinfo: {start_date.tzinfo})")
        logger.debug(f"  end_date: {end_date} (tzinfo: {end_date.tzinfo})")

        # Group commits by week and developer
        weekly_data: dict[tuple[datetime, str, str], dict[str, Any]] = self._aggregate_weekly_data(
            commits, start_date, end_date
        )

        # Create developer lookup
        dev_lookup = {dev["canonical_id"]: dev for dev in developer_stats}

        # First pass: collect all raw scores for curve normalization
        developer_raw_scores = {}
        weekly_scores = {}
        
        for (week_start, canonical_id, project_key), metrics in weekly_data.items():
            activity_result = self.activity_scorer.calculate_activity_score(metrics)
            raw_score = activity_result["raw_score"]
            
            # Store for curve normalization
            if canonical_id not in developer_raw_scores:
                developer_raw_scores[canonical_id] = 0
            developer_raw_scores[canonical_id] += raw_score
            
            # Store weekly result for later use
            weekly_scores[(week_start, canonical_id, project_key)] = activity_result
        
        # Apply curve normalization to developer totals
        curve_normalized = self.activity_scorer.normalize_scores_on_curve(developer_raw_scores)

        # Build CSV rows
        rows = []
        for (week_start, canonical_id, project_key), metrics in weekly_data.items():
            developer = dev_lookup.get(canonical_id, {})
            activity_result = weekly_scores[(week_start, canonical_id, project_key)]
            
            # Get curve data for this developer
            curve_data = curve_normalized.get(canonical_id, {})

            row = {
                "week_start": week_start.strftime("%Y-%m-%d"),
                "developer_id": self._anonymize_value(canonical_id, "id"),
                "developer_name": self._anonymize_value(
                    developer.get("primary_name", "Unknown"), "name"
                ),
                "developer_email": self._anonymize_value(
                    developer.get("primary_email", "unknown@example.com"), "email"
                ),
                "project": project_key,
                "commits": metrics["commits"],
                "story_points": metrics["story_points"],
                "lines_added": metrics["lines_added"],
                "lines_removed": metrics["lines_removed"],
                "files_changed": metrics["files_changed"],
                "complexity_delta": round(metrics["complexity_delta"], 2),
                "ticket_coverage_pct": round(metrics["ticket_coverage_pct"], 1),
                "avg_commit_size": round(metrics["avg_commit_size"], 1),
                "unique_tickets": metrics["unique_tickets"],
                "prs_involved": metrics["prs_involved"],
                # Activity score fields
                "activity_score": round(activity_result["normalized_score"], 1),
                "activity_level": activity_result["activity_level"],
                "commit_score": round(activity_result["components"]["commit_score"], 1),
                "pr_score": round(activity_result["components"]["pr_score"], 1),
                "code_impact_score": round(activity_result["components"]["code_impact_score"], 1),
                "complexity_score": round(activity_result["components"]["complexity_score"], 1),
                # Curve normalization fields
                "curved_score": curve_data.get("curved_score", 0),
                "percentile": curve_data.get("percentile", 0),
                "quintile": curve_data.get("quintile", 0),
                "curved_activity_level": curve_data.get("activity_level", "unknown"),
            }
            rows.append(row)

        # Sort by week and developer
        rows.sort(key=lambda x: (x["week_start"], x["developer_name"], x["project"]))

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
                        "week_start",
                        "developer_id",
                        "developer_name",
                        "developer_email",
                        "project",
                        "commits",
                        "story_points",
                        "lines_added",
                        "lines_removed",
                        "files_changed",
                        "complexity_delta",
                        "ticket_coverage_pct",
                        "avg_commit_size",
                        "unique_tickets",
                        "prs_involved",
                        "activity_score",
                        "activity_level",
                        "commit_score",
                        "pr_score",
                        "code_impact_score",
                        "complexity_score",
                        "curved_score",
                        "percentile",
                        "quintile",
                        "curved_activity_level",
                    ],
                )
                writer.writeheader()

        return output_path

    def generate_summary_report(
        self,
        commits: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        developer_stats: list[dict[str, Any]],
        ticket_analysis: dict[str, Any],
        output_path: Path,
        pm_data: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Generate summary statistics CSV."""
        summary_data = []

        # Overall statistics
        total_commits = len(commits)
        total_story_points = sum(c.get("story_points", 0) or 0 for c in commits)
        # Use filtered stats if available, otherwise fall back to raw stats
        total_lines = sum(
            c.get("filtered_insertions", c.get("insertions", 0))
            + c.get("filtered_deletions", c.get("deletions", 0))
            for c in commits
        )

        summary_data.append(
            {"metric": "Total Commits", "value": total_commits, "category": "Overall"}
        )

        summary_data.append(
            {"metric": "Total Story Points", "value": total_story_points, "category": "Overall"}
        )

        summary_data.append(
            {"metric": "Total Lines Changed", "value": total_lines, "category": "Overall"}
        )

        summary_data.append(
            {"metric": "Active Developers", "value": len(developer_stats), "category": "Overall"}
        )

        # Ticket coverage
        summary_data.append(
            {
                "metric": "Commit Ticket Coverage %",
                "value": round(ticket_analysis.get("commit_coverage_pct", 0), 1),
                "category": "Tracking",
            }
        )

        summary_data.append(
            {
                "metric": "PR Ticket Coverage %",
                "value": round(ticket_analysis.get("pr_coverage_pct", 0), 1),
                "category": "Tracking",
            }
        )

        # Platform breakdown
        for platform, count in ticket_analysis.get("ticket_summary", {}).items():
            summary_data.append(
                {"metric": f"{platform.title()} Tickets", "value": count, "category": "Platforms"}
            )

        # Developer statistics
        if developer_stats:
            top_contributor = max(developer_stats, key=lambda x: x["total_commits"])
            summary_data.append(
                {
                    "metric": "Top Contributor",
                    "value": self._anonymize_value(top_contributor["primary_name"], "name"),
                    "category": "Developers",
                }
            )

            summary_data.append(
                {
                    "metric": "Top Contributor Commits",
                    "value": top_contributor["total_commits"],
                    "category": "Developers",
                }
            )

        # PM Platform statistics
        if pm_data and "metrics" in pm_data:
            metrics = pm_data["metrics"]

            # Total PM issues
            summary_data.append(
                {
                    "metric": "Total PM Issues",
                    "value": metrics.get("total_pm_issues", 0),
                    "category": "PM Platforms",
                }
            )

            # Story point analysis
            story_analysis = metrics.get("story_point_analysis", {})
            summary_data.append(
                {
                    "metric": "PM Story Points",
                    "value": story_analysis.get("pm_total_story_points", 0),
                    "category": "PM Platforms",
                }
            )

            summary_data.append(
                {
                    "metric": "Story Point Coverage %",
                    "value": round(story_analysis.get("story_point_coverage_pct", 0), 1),
                    "category": "PM Platforms",
                }
            )

            # Issue type distribution
            issue_types = metrics.get("issue_type_distribution", {})
            for issue_type, count in issue_types.items():
                summary_data.append(
                    {
                        "metric": f"{issue_type.title()} Issues",
                        "value": count,
                        "category": "Issue Types",
                    }
                )

            # Platform coverage
            platform_coverage = metrics.get("platform_coverage", {})
            for platform, coverage_data in platform_coverage.items():
                summary_data.append(
                    {
                        "metric": f"{platform.title()} Issues",
                        "value": coverage_data.get("total_issues", 0),
                        "category": "Platform Coverage",
                    }
                )

                summary_data.append(
                    {
                        "metric": f"{platform.title()} Linked %",
                        "value": round(coverage_data.get("coverage_percentage", 0), 1),
                        "category": "Platform Coverage",
                    }
                )

            # Correlation quality
            correlation_quality = metrics.get("correlation_quality", {})
            summary_data.append(
                {
                    "metric": "Issue-Commit Correlations",
                    "value": correlation_quality.get("total_correlations", 0),
                    "category": "Correlation Quality",
                }
            )

            summary_data.append(
                {
                    "metric": "Avg Correlation Confidence",
                    "value": round(correlation_quality.get("average_confidence", 0), 2),
                    "category": "Correlation Quality",
                }
            )

        # Write summary CSV
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)

        return output_path

    def generate_developer_report(
        self, developer_stats: list[dict[str, Any]], output_path: Path
    ) -> Path:
        """Generate developer statistics CSV."""
        rows = []

        for dev in developer_stats:
            row = {
                "developer_id": self._anonymize_value(dev["canonical_id"], "id"),
                "name": self._anonymize_value(dev["primary_name"], "name"),
                "email": self._anonymize_value(dev["primary_email"], "email"),
                "github_username": (
                    self._anonymize_value(dev.get("github_username", ""), "username")
                    if dev.get("github_username")
                    else ""
                ),
                "total_commits": dev["total_commits"],
                "total_story_points": dev["total_story_points"],
                "alias_count": dev["alias_count"],
                "first_seen": (
                    self._safe_datetime_format(dev["first_seen"], "%Y-%m-%d")
                    if dev["first_seen"]
                    else ""
                ),
                "last_seen": (
                    self._safe_datetime_format(dev["last_seen"], "%Y-%m-%d")
                    if dev["last_seen"]
                    else ""
                ),
                "avg_story_points_per_commit": round(
                    dev["total_story_points"] / max(dev["total_commits"], 1), 2
                ),
            }
            rows.append(row)

        # Sort by total commits
        rows.sort(key=lambda x: x["total_commits"], reverse=True)

        # Write CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        return output_path

    def generate_pm_correlations_report(self, pm_data: dict[str, Any], output_path: Path) -> Path:
        """Generate PM platform correlations CSV report.

        WHY: PM platform integration provides valuable correlation data between
        work items and code changes. This report enables analysis of story point
        accuracy, development velocity, and work item completion patterns.

        Args:
            pm_data: PM platform data including correlations and metrics.
            output_path: Path where the CSV report should be written.

        Returns:
            Path to the generated CSV file.
        """
        if not pm_data or "correlations" not in pm_data:
            # Generate empty report if no PM data
            df = pd.DataFrame(
                columns=[
                    "commit_hash",
                    "commit_message",
                    "commit_author",
                    "commit_date",
                    "issue_key",
                    "issue_title",
                    "issue_type",
                    "issue_status",
                    "issue_platform",
                    "story_points",
                    "correlation_method",
                    "confidence",
                    "matched_text",
                ]
            )
            df.to_csv(output_path, index=False)
            return output_path

        correlations = pm_data["correlations"]
        rows = []

        for correlation in correlations:
            row = {
                "commit_hash": correlation.get("commit_hash", ""),
                "commit_message": correlation.get("commit_message", ""),
                "commit_author": self._anonymize_value(
                    correlation.get("commit_author", ""), "name"
                ),
                "commit_date": correlation.get("commit_date", ""),
                "issue_key": correlation.get("issue_key", ""),
                "issue_title": correlation.get("issue_title", ""),
                "issue_type": correlation.get("issue_type", ""),
                "issue_status": correlation.get("issue_status", ""),
                "issue_platform": correlation.get("issue_platform", ""),
                "story_points": correlation.get("story_points", 0) or 0,
                "correlation_method": correlation.get("correlation_method", ""),
                "confidence": round(correlation.get("confidence", 0), 3),
                "matched_text": correlation.get("matched_text", ""),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        return output_path

    def _aggregate_weekly_data(
        self, commits: list[dict[str, Any]], start_date: datetime, end_date: datetime
    ) -> dict[tuple[datetime, str, str], dict[str, Any]]:
        """Aggregate commit data by week."""
        weekly_data: defaultdict[tuple[datetime, str, str], dict[str, Any]] = defaultdict(
            lambda: {
                "commits": 0,
                "story_points": 0,
                "lines_added": 0,
                "lines_removed": 0,
                "files_changed": 0,
                "complexity_delta": 0.0,
                "commits_with_tickets": 0,
                "tickets": set(),
                "prs": set(),
            }
        )

        for commit in commits:
            timestamp = commit["timestamp"]
            logger.debug(
                f"Processing commit timestamp: {timestamp} (tzinfo: {getattr(timestamp, 'tzinfo', 'N/A')})"
            )

            # Ensure consistent timezone handling
            if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is not None:
                # Keep timezone-aware but ensure it's UTC
                if timestamp.tzinfo != timezone.utc:
                    timestamp = timestamp.astimezone(timezone.utc)
                    logger.debug(f"  Converted to UTC: {timestamp}")
            else:
                # Convert naive datetime to UTC timezone-aware
                timestamp = timestamp.replace(tzinfo=timezone.utc)
                logger.debug(f"  Made timezone-aware: {timestamp}")

            # Use safe comparison functions with logging
            if self._safe_datetime_compare(
                timestamp, start_date, "lt", "_aggregate_weekly_data range check"
            ) or self._safe_datetime_compare(
                timestamp, end_date, "gt", "_aggregate_weekly_data range check"
            ):
                logger.debug("  Skipping commit outside date range")
                continue

            # Get week start (Monday)
            week_start = self._get_week_start(timestamp)

            # Get project key (default to 'unknown')
            project_key = commit.get("project_key", "unknown")

            # Get canonical developer ID
            canonical_id = commit.get("canonical_id", commit.get("author_email", "unknown"))

            key = (week_start, canonical_id, project_key)

            # Aggregate metrics
            data = weekly_data[key]
            data["commits"] += 1
            data["story_points"] += commit.get("story_points", 0) or 0

            # Use filtered stats if available, otherwise fall back to raw stats
            data["lines_added"] += (
                commit.get("filtered_insertions", commit.get("insertions", 0)) or 0
            )
            data["lines_removed"] += (
                commit.get("filtered_deletions", commit.get("deletions", 0)) or 0
            )
            data["files_changed"] += (
                commit.get("filtered_files_changed", commit.get("files_changed", 0)) or 0
            )

            data["complexity_delta"] += commit.get("complexity_delta", 0.0) or 0.0

            # Track tickets
            ticket_refs = commit.get("ticket_references", [])
            if ticket_refs:
                data["commits_with_tickets"] += 1
                tickets_set = data["tickets"]
                for ticket in ticket_refs:
                    if isinstance(ticket, dict):
                        tickets_set.add(ticket.get("full_id", ""))
                    else:
                        tickets_set.add(str(ticket))

            # Track PRs (if available)
            pr_number = commit.get("pr_number")
            if pr_number:
                prs_set = data["prs"]
                prs_set.add(pr_number)

        # Calculate derived metrics
        result: dict[tuple[datetime, str, str], dict[str, Any]] = {}
        for key, metrics in weekly_data.items():
            commits_count = metrics["commits"]
            if commits_count > 0:
                metrics["ticket_coverage_pct"] = (
                    metrics["commits_with_tickets"] / commits_count * 100
                )
                metrics["avg_commit_size"] = (
                    metrics["lines_added"] + metrics["lines_removed"]
                ) / commits_count
            else:
                metrics["ticket_coverage_pct"] = 0
                metrics["avg_commit_size"] = 0

            tickets_set = metrics["tickets"]
            prs_set = metrics["prs"]
            metrics["unique_tickets"] = len(tickets_set)
            metrics["prs_involved"] = len(prs_set)

            # Remove sets before returning
            del metrics["tickets"]
            del metrics["prs"]
            del metrics["commits_with_tickets"]

            result[key] = metrics

        return result

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

    def generate_developer_activity_summary(
        self,
        commits: list[dict[str, Any]],
        developer_stats: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        output_path: Path,
        weeks: int = 12,
    ) -> Path:
        """Generate developer activity summary with curve-normalized scores.
        
        This report provides a high-level view of developer activity with
        curve-normalized scores that allow for fair comparison across the team.
        """
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)
        
        # Aggregate metrics by developer
        developer_metrics = defaultdict(lambda: {
            "commits": 0,
            "prs_involved": 0,
            "lines_added": 0,
            "lines_removed": 0,
            "files_changed": 0,
            "complexity_delta": 0.0,
            "story_points": 0,
            "unique_tickets": set(),
        })
        
        # Process commits
        for commit in commits:
            timestamp = commit["timestamp"]
            if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            
            if timestamp < start_date or timestamp > end_date:
                continue
                
            dev_id = commit.get("canonical_id", commit.get("author_email", "unknown"))
            metrics = developer_metrics[dev_id]
            
            metrics["commits"] += 1
            metrics["lines_added"] += commit.get("filtered_insertions", commit.get("insertions", 0)) or 0
            metrics["lines_removed"] += commit.get("filtered_deletions", commit.get("deletions", 0)) or 0
            metrics["files_changed"] += commit.get("filtered_files_changed", commit.get("files_changed", 0)) or 0
            metrics["complexity_delta"] += commit.get("complexity_delta", 0.0) or 0.0
            metrics["story_points"] += commit.get("story_points", 0) or 0
            
            ticket_refs = commit.get("ticket_references", [])
            for ticket in ticket_refs:
                if isinstance(ticket, dict):
                    metrics["unique_tickets"].add(ticket.get("full_id", ""))
                else:
                    metrics["unique_tickets"].add(str(ticket))
        
        # Process PRs
        for pr in prs:
            author_id = pr.get("canonical_id", pr.get("author", "unknown"))
            if author_id in developer_metrics:
                developer_metrics[author_id]["prs_involved"] += 1
        
        # Calculate activity scores
        developer_scores = {}
        developer_results = {}
        
        for dev_id, metrics in developer_metrics.items():
            # Convert sets to counts
            metrics["unique_tickets"] = len(metrics["unique_tickets"])
            
            # Calculate activity score
            activity_result = self.activity_scorer.calculate_activity_score(metrics)
            developer_scores[dev_id] = activity_result["raw_score"]
            developer_results[dev_id] = activity_result
        
        # Apply curve normalization
        curve_normalized = self.activity_scorer.normalize_scores_on_curve(developer_scores)
        
        # Create developer lookup
        dev_lookup = {dev["canonical_id"]: dev for dev in developer_stats}
        
        # Build rows
        rows = []
        for dev_id, metrics in developer_metrics.items():
            developer = dev_lookup.get(dev_id, {})
            activity_result = developer_results[dev_id]
            curve_data = curve_normalized.get(dev_id, {})
            
            row = {
                "developer_id": self._anonymize_value(dev_id, "id"),
                "developer_name": self._anonymize_value(
                    developer.get("primary_name", "Unknown"), "name"
                ),
                "commits": metrics["commits"],
                "prs": metrics["prs_involved"],
                "story_points": metrics["story_points"],
                "lines_added": metrics["lines_added"],
                "lines_removed": metrics["lines_removed"],
                "files_changed": metrics["files_changed"],
                "unique_tickets": metrics["unique_tickets"],
                # Raw activity scores
                "raw_activity_score": round(activity_result["raw_score"], 1),
                "normalized_activity_score": round(activity_result["normalized_score"], 1),
                "activity_level": activity_result["activity_level"],
                # Curve-normalized scores
                "curved_score": curve_data.get("curved_score", 0),
                "percentile": curve_data.get("percentile", 0),
                "quintile": curve_data.get("quintile", 0),
                "curved_activity_level": curve_data.get("activity_level", "unknown"),
                "level_description": curve_data.get("level_description", ""),
                # Component breakdown
                "commit_score": round(activity_result["components"]["commit_score"], 1),
                "pr_score": round(activity_result["components"]["pr_score"], 1),
                "code_impact_score": round(activity_result["components"]["code_impact_score"], 1),
                "complexity_score": round(activity_result["components"]["complexity_score"], 1),
            }
            rows.append(row)
        
        # Sort by curved score (highest first)
        rows.sort(key=lambda x: x["curved_score"], reverse=True)
        
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
                        "developer_id",
                        "developer_name",
                        "commits",
                        "prs",
                        "story_points",
                        "lines_added",
                        "lines_removed",
                        "files_changed",
                        "unique_tickets",
                        "raw_activity_score",
                        "normalized_activity_score",
                        "activity_level",
                        "curved_score",
                        "percentile",
                        "quintile",
                        "curved_activity_level",
                        "level_description",
                        "commit_score",
                        "pr_score",
                        "code_impact_score",
                        "complexity_score",
                    ],
                )
                writer.writeheader()
        
        return output_path

    def _anonymize_value(self, value: str, field_type: str) -> str:
        """Anonymize a value if anonymization is enabled."""
        if not self.anonymize or not value:
            return value

        if field_type == "email" and "@" in value:
            # Keep domain for email
            local, domain = value.split("@", 1)
            value = local  # Anonymize only local part
            suffix = f"@{domain}"
        else:
            suffix = ""

        if value not in self._anonymization_map:
            self._anonymous_counter += 1
            if field_type == "name":
                anonymous = f"Developer{self._anonymous_counter}"
            elif field_type == "email":
                anonymous = f"dev{self._anonymous_counter}"
            elif field_type == "id":
                anonymous = f"ID{self._anonymous_counter:04d}"
            else:
                anonymous = f"anon{self._anonymous_counter}"

            self._anonymization_map[value] = anonymous

        return self._anonymization_map[value] + suffix

    def generate_untracked_commits_report(
        self, ticket_analysis: dict[str, Any], output_path: Path
    ) -> Path:
        """Generate detailed CSV report for commits without ticket references.

        WHY: Untracked commits represent work that may not be visible to project
        management tools. This report enables analysis of what types of work are
        being performed outside the tracked process, helping identify process
        improvements and training needs.

        Args:
            ticket_analysis: Ticket analysis results containing untracked commits
            output_path: Path where the CSV report should be written

        Returns:
            Path to the generated CSV file
        """
        untracked_commits = ticket_analysis.get("untracked_commits", [])

        if not untracked_commits:
            # Generate empty report with headers
            headers = [
                "commit_hash",
                "short_hash",
                "author",
                "author_email",
                "canonical_id",
                "date",
                "project",
                "message",
                "category",
                "files_changed",
                "lines_added",
                "lines_removed",
                "lines_changed",
                "is_merge",
            ]
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
            return output_path

        # Process untracked commits into CSV rows
        rows = []
        for commit in untracked_commits:
            # Handle datetime formatting
            timestamp = commit.get("timestamp")
            if timestamp:
                if hasattr(timestamp, "strftime"):
                    date_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    date_str = str(timestamp)
            else:
                date_str = ""

            row = {
                "commit_hash": commit.get("full_hash", commit.get("hash", "")),
                "short_hash": commit.get("hash", ""),
                "author": self._anonymize_value(commit.get("author", "Unknown"), "name"),
                "author_email": self._anonymize_value(commit.get("author_email", ""), "email"),
                "canonical_id": self._anonymize_value(commit.get("canonical_id", ""), "id"),
                "date": date_str,
                "project": commit.get("project_key", "UNKNOWN"),
                "message": commit.get("message", ""),
                "category": commit.get("category", "other"),
                "files_changed": commit.get("files_changed", 0),
                "lines_added": commit.get("lines_added", 0),
                "lines_removed": commit.get("lines_removed", 0),
                "lines_changed": commit.get("lines_changed", 0),
                "is_merge": commit.get("is_merge", False),
            }
            rows.append(row)

        # Write CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)

        return output_path
