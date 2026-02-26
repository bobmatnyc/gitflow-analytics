"""Weekly, summary, developer, and PM correlation CSV reports.

Extracted from csv_writer.py to keep file sizes manageable.
Contains generate_weekly_report, generate_summary_report,
generate_developer_report, generate_pm_correlations_report,
and _aggregate_weekly_data helper.
"""

import csv
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..metrics.activity_scoring import ActivityScorer

logger = logging.getLogger(__name__)


class CSVWeeklyReportsMixin:
    """Mixin providing weekly/summary/PM reports for CSVReportGenerator.

    Attributes expected from host class:
        activity_scorer, anonymize, exclude_authors, identity_resolver,
        _anonymization_map, _anonymous_counter
    """

    def generate_weekly_report(
        self,
        commits: list[dict[str, Any]],
        developer_stats: list[dict[str, Any]],
        output_path: Path,
        weeks: int = 12,
    ) -> Path:
        """Generate weekly metrics CSV report."""
        # Apply exclusion filtering in Phase 2
        commits = self._filter_excluded_authors_list(commits)
        developer_stats = self._filter_excluded_authors_list(developer_stats)
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
                    self._get_canonical_display_name(
                        canonical_id, developer.get("primary_name", "Unknown")
                    ),
                    "name",
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
        pr_metrics: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Generate summary statistics CSV.

        Args:
            commits: List of commit data dictionaries.
            prs: List of pull request data dictionaries.
            developer_stats: Per-developer aggregated statistics.
            ticket_analysis: Ticket/issue coverage analysis.
            output_path: Path where the CSV file will be written.
            pm_data: Optional PM platform integration data.
            pr_metrics: Optional pre-calculated PR metrics dict from
                ``GitHubIntegration.calculate_pr_metrics()``.  When provided,
                enhanced review metrics (approval rate, time-to-review, etc.)
                are included in the summary.

        Returns:
            Path to the written CSV file.
        """
        # Apply exclusion filtering in Phase 2
        commits = self._filter_excluded_authors_list(commits)
        developer_stats = self._filter_excluded_authors_list(developer_stats)

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
                    "value": self._anonymize_value(
                        self._get_canonical_display_name(
                            top_contributor["canonical_id"], top_contributor["primary_name"]
                        ),
                        "name",
                    ),
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

        # PR metrics (basic — always from PR list)
        if prs:
            total_prs = len(prs)
            summary_data.append(
                {"metric": "Total PRs", "value": total_prs, "category": "Pull Requests"}
            )

            total_inline = sum(pr.get("review_comments", 0) or 0 for pr in prs)
            summary_data.append(
                {
                    "metric": "Total Inline Review Comments",
                    "value": total_inline,
                    "category": "Pull Requests",
                }
            )

            avg_size = (
                sum((pr.get("additions") or 0) + (pr.get("deletions") or 0) for pr in prs)
                / total_prs
            )
            summary_data.append(
                {
                    "metric": "Avg PR Size (lines)",
                    "value": round(avg_size, 1),
                    "category": "Pull Requests",
                }
            )

        # PR metrics (enhanced — only when review data was collected)
        if pr_metrics:
            _cat = "PR Review Metrics"

            if pr_metrics.get("total_prs", 0) > 0:
                summary_data.append(
                    {
                        "metric": "Story Point Coverage %",
                        "value": round(pr_metrics.get("story_point_coverage", 0.0), 1),
                        "category": _cat,
                    }
                )

            approval_rate = pr_metrics.get("approval_rate")
            if approval_rate is not None:
                summary_data.append(
                    {
                        "metric": "PR Approval Rate %",
                        "value": round(approval_rate, 1),
                        "category": _cat,
                    }
                )

            review_coverage = pr_metrics.get("review_coverage")
            if review_coverage is not None:
                summary_data.append(
                    {
                        "metric": "PR Review Coverage %",
                        "value": round(review_coverage, 1),
                        "category": _cat,
                    }
                )

            avg_approvals = pr_metrics.get("avg_approvals_per_pr")
            if avg_approvals is not None:
                summary_data.append(
                    {
                        "metric": "Avg Approvals per PR",
                        "value": round(avg_approvals, 2),
                        "category": _cat,
                    }
                )

            avg_cr = pr_metrics.get("avg_change_requests_per_pr")
            if avg_cr is not None:
                summary_data.append(
                    {
                        "metric": "Avg Change Requests per PR",
                        "value": round(avg_cr, 2),
                        "category": _cat,
                    }
                )

            avg_ttfr = pr_metrics.get("avg_time_to_first_review_hours")
            if avg_ttfr is not None:
                summary_data.append(
                    {
                        "metric": "Avg Time to First Review (hours)",
                        "value": round(avg_ttfr, 2),
                        "category": _cat,
                    }
                )

            median_ttfr = pr_metrics.get("median_time_to_first_review_hours")
            if median_ttfr is not None:
                summary_data.append(
                    {
                        "metric": "Median Time to First Review (hours)",
                        "value": round(median_ttfr, 2),
                        "category": _cat,
                    }
                )

            total_pr_comments = pr_metrics.get("total_pr_comments", 0)
            if total_pr_comments:
                summary_data.append(
                    {
                        "metric": "Total PR Comments",
                        "value": total_pr_comments,
                        "category": _cat,
                    }
                )

            avg_revisions = pr_metrics.get("avg_revision_count")
            if avg_revisions is not None:
                summary_data.append(
                    {
                        "metric": "Avg Revisions per PR",
                        "value": round(avg_revisions, 2),
                        "category": _cat,
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
                "name": self._anonymize_value(
                    self._get_canonical_display_name(dev["canonical_id"], dev["primary_name"]),
                    "name",
                ),
                "email": self._anonymize_value(dev["primary_email"], "email"),
                "github_username": (
                    self._anonymize_value(dev.get("github_username", ""), "username")
                    if dev.get("github_username")
                    else ""
                ),
                "total_commits": dev["total_commits"],
                "total_story_points": dev["total_story_points"],
                "alias_count": dev.get("alias_count", 1),
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

