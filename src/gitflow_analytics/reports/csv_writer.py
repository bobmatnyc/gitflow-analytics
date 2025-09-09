"""CSV report generation for GitFlow Analytics."""

import csv
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..metrics.activity_scoring import ActivityScorer
from .base import BaseReportGenerator, ReportData, ReportOutput
from .interfaces import ReportFormat

# Get logger for this module
logger = logging.getLogger(__name__)


class CSVReportGenerator(BaseReportGenerator):
    """Generate CSV reports with weekly metrics."""

    def __init__(self, anonymize: bool = False, exclude_authors: list[str] = None, identity_resolver=None, **kwargs):
        """Initialize report generator."""
        super().__init__(anonymize=anonymize, exclude_authors=exclude_authors, 
                        identity_resolver=identity_resolver, **kwargs)
        self.activity_scorer = ActivityScorer()
    
    # Implementation of abstract methods from BaseReportGenerator
    
    def generate(self, data: ReportData, output_path: Optional[Path] = None) -> ReportOutput:
        """Generate CSV report from standardized data.
        
        Args:
            data: Standardized report data
            output_path: Optional path to write the report to
            
        Returns:
            ReportOutput containing the results
        """
        try:
            # Validate data
            if not self.validate_data(data):
                return ReportOutput(
                    success=False,
                    errors=["Invalid or incomplete data provided"]
                )
            
            # Pre-process data (apply filters and anonymization)
            data = self.pre_process(data)
            
            # Generate appropriate CSV based on available data
            if output_path:
                # Determine report type based on filename or available data
                filename = output_path.name.lower()
                
                if "weekly" in filename and data.commits:
                    self.generate_weekly_report(data.commits, data.developer_stats, output_path)
                elif "developer" in filename and data.developer_stats:
                    self.generate_developer_report(data.developer_stats, output_path)
                elif "activity" in filename and data.activity_data:
                    # Write activity data directly
                    df = pd.DataFrame(data.activity_data)
                    df.to_csv(output_path, index=False)
                elif "focus" in filename and data.focus_data:
                    # Write focus data directly
                    df = pd.DataFrame(data.focus_data)
                    df.to_csv(output_path, index=False)
                elif data.commits:
                    # Default to weekly report
                    self.generate_weekly_report(data.commits, data.developer_stats, output_path)
                else:
                    return ReportOutput(
                        success=False,
                        errors=["No suitable data found for CSV generation"]
                    )
                
                # Calculate file size
                file_size = output_path.stat().st_size if output_path.exists() else 0
                
                return ReportOutput(
                    success=True,
                    file_path=output_path,
                    format="csv",
                    size_bytes=file_size
                )
            else:
                # Generate in-memory CSV
                import io
                buffer = io.StringIO()
                
                # Default to generating weekly report in memory
                if data.commits:
                    # Create temporary dataframe
                    df = pd.DataFrame(self._aggregate_weekly_data(data.commits,
                                                                 datetime.now(timezone.utc) - timedelta(weeks=52),
                                                                 datetime.now(timezone.utc)))
                    df.to_csv(buffer, index=False)
                    content = buffer.getvalue()
                    
                    return ReportOutput(
                        success=True,
                        content=content,
                        format="csv",
                        size_bytes=len(content)
                    )
                else:
                    return ReportOutput(
                        success=False,
                        errors=["No data available for CSV generation"]
                    )
                    
        except Exception as e:
            self.logger.error(f"Error generating CSV report: {e}")
            return ReportOutput(
                success=False,
                errors=[str(e)]
            )
    
    def get_required_fields(self) -> List[str]:
        """Get the list of required data fields for CSV generation.
        
        Returns:
            List of required field names
        """
        # CSV reports can work with various combinations of data
        # At minimum, we need either commits or developer_stats
        return ["commits"]  # Primary requirement
    
    def get_format_type(self) -> str:
        """Get the format type this generator produces.
        
        Returns:
            Format identifier
        """
        return ReportFormat.CSV.value

    def _filter_excluded_authors_list(self, data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Filter out excluded authors from any data list using canonical_id and enhanced bot detection.
        
        WHY: Bot exclusion happens in Phase 2 (reporting) instead of Phase 1 (data collection)
        to ensure manual identity mappings work correctly. This allows the system to see 
        consolidated bot identities via canonical_id instead of just original author_email/author_name.
        
        ENHANCEMENT: Added enhanced bot pattern matching to catch bots that weren't properly
        consolidated via manual mappings, preventing bot leakage in reports.
        
        Args:
            data_list: List of data dictionaries containing canonical_id field
            
        Returns:
            Filtered list with excluded authors removed
        """
        if not self.exclude_authors:
            return data_list
            
        logger.debug(f"DEBUG EXCLUSION: Starting filter with {len(self.exclude_authors)} excluded authors: {self.exclude_authors}")
        logger.debug(f"DEBUG EXCLUSION: Filtering {len(data_list)} items from data list")
        
        excluded_lower = [author.lower() for author in self.exclude_authors]
        logger.debug(f"DEBUG EXCLUSION: Excluded authors (lowercase): {excluded_lower}")
        
        # Separate explicit excludes from bot patterns  
        explicit_excludes = []
        bot_patterns = []
        
        for exclude in excluded_lower:
            if '[bot]' in exclude or 'bot' in exclude.split():
                bot_patterns.append(exclude)
            else:
                explicit_excludes.append(exclude)
        
        logger.debug(f"DEBUG EXCLUSION: Explicit excludes: {explicit_excludes}")
        logger.debug(f"DEBUG EXCLUSION: Bot patterns: {bot_patterns}")
        
        filtered_data = []
        excluded_count = 0
        
        # Sample first 5 items to see data structure
        for i, item in enumerate(data_list[:5]):
            logger.debug(f"DEBUG EXCLUSION: Sample item {i}: canonical_id='{item.get('canonical_id', '')}', "
                        f"author_email='{item.get('author_email', '')}', author_name='{item.get('author_name', '')}', "
                        f"author='{item.get('author', '')}', primary_name='{item.get('primary_name', '')}', "
                        f"name='{item.get('name', '')}', developer='{item.get('developer', '')}', "
                        f"display_name='{item.get('display_name', '')}'")  
        
        for item in data_list:
            canonical_id = item.get("canonical_id", "")
            # Also check original author fields as fallback for data without canonical_id
            author_email = item.get("author_email", "")
            author_name = item.get("author_name", "")
            
            # Check all possible author fields to ensure we catch every variation
            author = item.get("author", "")
            primary_name = item.get("primary_name", "")
            name = item.get("name", "")
            developer = item.get("developer", "")  # Common in CSV data
            display_name = item.get("display_name", "")  # Common in some data structures
            
            # Collect all identity fields for checking
            identity_fields = [
                canonical_id,
                item.get("primary_email", ""),
                author_email,
                author_name,
                author,
                primary_name,
                name,
                developer,
                display_name
            ]
            
            should_exclude = False
            exclusion_reason = ""
            
            # Check for exact matches with explicit excludes first
            for field in identity_fields:
                if field and field.lower() in explicit_excludes:
                    should_exclude = True
                    exclusion_reason = f"exact match with '{field}' in explicit excludes"
                    break
            
            # If not explicitly excluded, check for bot patterns
            if not should_exclude:
                for field in identity_fields:
                    if not field:
                        continue
                    field_lower = field.lower()
                    
                    # Enhanced bot detection: check if any field contains bot-like patterns
                    for bot_pattern in bot_patterns:
                        if bot_pattern in field_lower:
                            should_exclude = True
                            exclusion_reason = f"bot pattern '{bot_pattern}' matches field '{field}'"
                            break
                    
                    # Additional bot detection: check for common bot patterns not in explicit list
                    if not should_exclude:
                        bot_indicators = ['[bot]', 'bot@', '-bot', 'automated', 'github-actions', 'dependabot', 'renovate']
                        for indicator in bot_indicators:
                            if indicator in field_lower:
                                # Only exclude if this bot-like pattern matches something in our exclude list
                                for exclude in excluded_lower:
                                    if indicator.replace('[', '').replace(']', '') in exclude or exclude in field_lower:
                                        should_exclude = True
                                        exclusion_reason = f"bot indicator '{indicator}' in field '{field}' matches exclude pattern '{exclude}'"
                                        break
                                if should_exclude:
                                    break
                        
                    if should_exclude:
                        break
            
            if should_exclude:
                excluded_count += 1
                logger.debug(f"DEBUG EXCLUSION: EXCLUDING item - {exclusion_reason}")
                logger.debug(f"  canonical_id='{canonical_id}', primary_email='{item.get('primary_email', '')}', "
                           f"author_email='{author_email}', author_name='{author_name}', author='{author}', "
                           f"primary_name='{primary_name}', name='{name}', developer='{developer}', "
                           f"display_name='{display_name}'")
            else:
                filtered_data.append(item)
                
        logger.debug(f"DEBUG EXCLUSION: Excluded {excluded_count} items, kept {len(filtered_data)} items")
        return filtered_data

    def _get_canonical_display_name(self, canonical_id: str, fallback_name: str) -> str:
        """
        Get the canonical display name for a developer.
        
        WHY: Manual identity mappings may have updated display names that aren't
        reflected in the developer_stats data passed to report generators. This
        method ensures we get the most current display name from the identity resolver.
        
        Args:
            canonical_id: The canonical ID to get the display name for
            fallback_name: The fallback name to use if identity resolver is not available
            
        Returns:
            The canonical display name or fallback name
        """
        if self.identity_resolver and canonical_id:
            try:
                canonical_name = self.identity_resolver.get_canonical_name(canonical_id)
                if canonical_name and canonical_name != "Unknown":
                    return canonical_name
            except Exception as e:
                logger.debug(f"Error getting canonical name for {canonical_id}: {e}")
        
        return fallback_name

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
                        canonical_id, 
                        developer.get("primary_name", "Unknown")
                    ), "name"
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
                            top_contributor["canonical_id"], 
                            top_contributor["primary_name"]
                        ), "name"
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
                    self._get_canonical_display_name(
                        dev["canonical_id"], 
                        dev["primary_name"]
                    ), "name"
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
        # Apply exclusion filtering in Phase 2
        commits = self._filter_excluded_authors_list(commits)
        developer_stats = self._filter_excluded_authors_list(developer_stats)
        
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
                    self._get_canonical_display_name(
                        dev_id, 
                        developer.get("primary_name", "Unknown")
                    ), "name"
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
    def generate_weekly_categorization_report(
        self, 
        all_commits: list[dict[str, Any]], 
        ticket_extractor,  # TicketExtractor or MLTicketExtractor instance
        output_path: Path, 
        weeks: int = 12
    ) -> Path:
        """Generate weekly commit categorization metrics CSV report for ALL commits.
        
        WHY: Categorization trends provide insights into development patterns
        over time, helping identify process improvements and training needs.
        This enhanced version processes ALL commits (tracked and untracked) to provide
        complete visibility into work patterns across the entire development flow.
        
        DESIGN DECISION: Processes all commits using the same ML/rule-based categorization
        system used elsewhere in the application, ensuring consistent categorization
        across all reports and analysis.
        
        Args:
            all_commits: Complete list of commits to categorize
            ticket_extractor: TicketExtractor instance for commit categorization
            output_path: Path where the CSV report should be written
            weeks: Number of weeks to analyze
            
        Returns:
            Path to the generated CSV file
        """
        # Calculate week boundaries
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)
        
        # Initialize weekly aggregation structures
        weekly_categories = defaultdict(lambda: defaultdict(int))
        weekly_metrics = defaultdict(lambda: {
            'lines_added': 0,
            'lines_removed': 0, 
            'files_changed': 0,
            'developers': set()
        })
        
        # Process ALL commits with classification
        processed_commits = 0
        for commit in all_commits:
            if not isinstance(commit, dict):
                continue
                
            # Get timestamp and validate date range
            timestamp = commit.get("timestamp")
            if not timestamp:
                continue
                
            # Ensure timezone consistency
            if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            elif hasattr(timestamp, "tzinfo") and timestamp.tzinfo != timezone.utc:
                timestamp = timestamp.astimezone(timezone.utc)
                
            if timestamp < start_date or timestamp > end_date:
                continue
                
            # Skip merge commits (consistent with untracked analysis)
            if commit.get("is_merge", False):
                continue
                
            # Categorize the commit using the same system as untracked analysis
            message = commit.get("message", "")
            files_changed_raw = commit.get("files_changed", [])
            
            # Handle both int and list types for files_changed
            if isinstance(files_changed_raw, int):
                files_changed_count = files_changed_raw
                files_changed_list = []  # Can't provide file names, only count
            elif isinstance(files_changed_raw, list):
                files_changed_count = len(files_changed_raw)
                files_changed_list = files_changed_raw
            else:
                files_changed_count = 0
                files_changed_list = []
            
            # Handle both TicketExtractor and MLTicketExtractor signatures
            try:
                # Try ML signature first (message, files_changed as list)
                category = ticket_extractor.categorize_commit(message, files_changed_list)
            except TypeError:
                # Fall back to base signature (message only)
                category = ticket_extractor.categorize_commit(message)
            
            # Get week boundary (Monday start)
            week_start = self._get_week_start(timestamp)
            
            # Aggregate by category
            weekly_categories[week_start][category] += 1
            
            # Aggregate metrics
            weekly_metrics[week_start]['lines_added'] += commit.get("insertions", 0)
            weekly_metrics[week_start]['lines_removed'] += commit.get("deletions", 0)
            weekly_metrics[week_start]['files_changed'] += files_changed_count
            
            # Track unique developers (use canonical_id or fallback to email)
            developer_id = commit.get("canonical_id") or commit.get("author_email", "Unknown")
            weekly_metrics[week_start]['developers'].add(developer_id)
            
            processed_commits += 1
        
        # Build CSV rows with comprehensive metrics
        rows = []
        all_categories = set()
        
        # Collect all categories across all weeks
        for week_data in weekly_categories.values():
            all_categories.update(week_data.keys())
        
        # Ensure standard categories are included even if not found
        standard_categories = ["bug_fix", "feature", "refactor", "documentation", 
                             "maintenance", "test", "style", "build", "integration", "other"]
        all_categories.update(standard_categories)
        sorted_categories = sorted(all_categories)
        
        # Generate weekly rows
        for week_start in sorted(weekly_categories.keys()):
            week_data = weekly_categories[week_start]
            week_metrics = weekly_metrics[week_start]
            total_commits = sum(week_data.values())
            
            row = {
                "week_start": week_start.strftime("%Y-%m-%d"),
                "total_commits": total_commits,
                "lines_added": week_metrics['lines_added'],
                "lines_removed": week_metrics['lines_removed'],
                "files_changed": week_metrics['files_changed'],
                "developer_count": len(week_metrics['developers'])
            }
            
            # Add each category count and percentage
            for category in sorted_categories:
                count = week_data.get(category, 0)
                pct = (count / total_commits * 100) if total_commits > 0 else 0
                
                row[f"{category}_count"] = count
                row[f"{category}_pct"] = round(pct, 1)
            
            rows.append(row)
        
        # Write CSV with comprehensive headers
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        else:
            # Write empty CSV with comprehensive headers
            headers = ["week_start", "total_commits", "lines_added", "lines_removed", 
                      "files_changed", "developer_count"]
            
            for category in sorted_categories:
                headers.extend([f"{category}_count", f"{category}_pct"])
            
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
        
        return output_path
    
    def generate_story_point_correlation_report(
        self,
        commits: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        pm_data: Optional[dict[str, Any]],
        output_path: Path,
        weeks: int = 12
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
                anonymize=self.anonymize,
                identity_resolver=self.identity_resolver
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
                "week_start", "metric_type", "developer_name",
                "sp_commits_correlation", "sp_lines_correlation", "sp_files_correlation", 
                "sp_prs_correlation", "sp_complexity_correlation", "sample_size",
                "total_story_points", "total_commits", "story_points_completed",
                "commits_count", "prs_merged", "developers_active", "velocity_trend",
                "overall_accuracy", "avg_weekly_accuracy", "consistency",
                "weeks_active", "total_estimated_sp", "total_actual_sp", "estimation_ratio"
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
        weekly_data: dict[datetime, dict[str, Any]] = defaultdict(lambda: {
            'total_story_points': 0,
            'pr_story_points': 0,
            'commit_story_points': 0,
            'total_lines': 0,
            'lines_added': 0,
            'lines_removed': 0,
            'files_changed': 0,
            'commits_count': 0,
            'developers': set(),
            'prs_with_sp': 0,
            'commits_with_sp': 0,
        })

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
            files_changed = commit.get("filtered_files_changed", commit.get("files_changed", 0)) or 0

            week_data['commits_count'] += 1
            week_data['commit_story_points'] += story_points
            week_data['total_story_points'] += story_points
            week_data['lines_added'] += lines_added
            week_data['lines_removed'] += lines_removed
            week_data['total_lines'] += lines_added + lines_removed
            week_data['files_changed'] += files_changed

            # Track developers and story point coverage
            developer_id = commit.get("canonical_id", commit.get("author_email", "unknown"))
            week_data['developers'].add(developer_id)
            
            if story_points > 0:
                week_data['commits_with_sp'] += 1

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
                week_data['pr_story_points'] += story_points
                week_data['total_story_points'] += story_points
                week_data['prs_with_sp'] += 1

                # Track developer from PR
                developer_id = pr.get("canonical_id", pr.get("author", "unknown"))
                week_data['developers'].add(developer_id)

        # Build CSV rows with velocity metrics
        rows = []
        previous_week_lines_per_point = None

        for week_start in sorted(weekly_data.keys()):
            week_data = weekly_data[week_start]
            total_story_points = week_data['total_story_points']
            total_lines = week_data['total_lines']

            # Calculate key metrics with division by zero protection
            lines_per_point = (total_lines / total_story_points) if total_story_points > 0 else 0
            commits_per_point = (week_data['commits_count'] / total_story_points) if total_story_points > 0 else 0
            
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
                    velocity_trend = ((lines_per_point - previous_week_lines_per_point) / previous_week_lines_per_point) * 100
                else:
                    velocity_trend = -100  # Went from some lines per point to zero
            else:
                velocity_trend = 0  # No previous data for comparison

            row = {
                "week_start": week_start.strftime("%Y-%m-%d"),
                "total_story_points": total_story_points,
                "pr_story_points": week_data['pr_story_points'],
                "commit_story_points": week_data['commit_story_points'],
                "total_lines": total_lines,
                "lines_added": week_data['lines_added'],
                "lines_removed": week_data['lines_removed'],
                "files_changed": week_data['files_changed'],
                "lines_per_point": round(lines_per_point, 2) if lines_per_point > 0 else 0,
                "commits_per_point": round(commits_per_point, 2) if commits_per_point > 0 else 0,
                "developers_involved": len(week_data['developers']),
                "efficiency_score": round(efficiency_score, 1),
                "velocity_trend": round(velocity_trend, 1),
                # Additional metrics for deeper analysis
                "commits_count": week_data['commits_count'],
                "prs_with_story_points": week_data['prs_with_sp'],
                "commits_with_story_points": week_data['commits_with_sp'],
                "story_point_coverage_pct": round(
                    (week_data['commits_with_sp'] / max(week_data['commits_count'], 1)) * 100, 1
                ),
                "avg_lines_per_commit": round(
                    total_lines / max(week_data['commits_count'], 1), 1
                ),
                "avg_files_per_commit": round(
                    week_data['files_changed'] / max(week_data['commits_count'], 1), 1
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