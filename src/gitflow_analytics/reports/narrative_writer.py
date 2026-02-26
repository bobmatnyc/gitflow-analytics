"""Narrative report generation in Markdown format."""

import logging
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from ..core.progress import get_progress_service
from ..metrics.activity_scoring import ActivityScorer
from .narrative_executive import NarrativeExecutiveMixin
from .narrative_classification import NarrativeClassificationMixin
from .narrative_team import NarrativeTeamMixin
from .narrative_analysis import NarrativeAnalysisMixin
from .narrative_recommendations import NarrativeRecommendationsMixin

# Get logger for this module
logger = logging.getLogger(__name__)


class NarrativeReportGenerator(
    NarrativeExecutiveMixin,
    NarrativeClassificationMixin,
    NarrativeTeamMixin,
    NarrativeAnalysisMixin,
    NarrativeRecommendationsMixin,
):
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

    def _filter_excluded_authors(
        self, data_list: list[dict[str, Any]], exclude_authors: list[str]
    ) -> list[dict[str, Any]]:
        """
        Filter out excluded authors from any data list using canonical_id and enhanced bot detection.

        WHY: Bot exclusion happens in Phase 2 (reporting) instead of Phase 1 (data collection)
        to ensure manual identity mappings work correctly. This allows the system to see
        consolidated bot identities via canonical_id instead of just original author_email/author_name.

        ENHANCEMENT: Added enhanced bot pattern matching to catch bots that weren't properly
        consolidated via manual mappings, preventing bot leakage in reports.

        Args:
            data_list: List of data dictionaries containing canonical_id field
            exclude_authors: List of author identifiers to exclude (checked against canonical_id)

        Returns:
            Filtered list with excluded authors removed
        """
        if not exclude_authors:
            return data_list

        logger.debug(
            f"DEBUG EXCLUSION: Starting filter with {len(exclude_authors)} excluded authors: {exclude_authors}"
        )
        logger.debug(f"DEBUG EXCLUSION: Filtering {len(data_list)} items from data list")

        excluded_lower = [author.lower() for author in exclude_authors]
        logger.debug(f"DEBUG EXCLUSION: Excluded authors (lowercase): {excluded_lower}")

        # Separate explicit excludes from bot patterns
        explicit_excludes = []
        bot_patterns = []

        for exclude in excluded_lower:
            if "[bot]" in exclude or "bot" in exclude.split():
                bot_patterns.append(exclude)
            else:
                explicit_excludes.append(exclude)

        logger.debug(f"DEBUG EXCLUSION: Explicit excludes: {explicit_excludes}")
        logger.debug(f"DEBUG EXCLUSION: Bot patterns: {bot_patterns}")

        filtered_data = []
        excluded_count = 0

        # Sample first 5 items to see data structure
        for i, item in enumerate(data_list[:5]):
            logger.debug(
                f"DEBUG EXCLUSION: Sample item {i}: canonical_id='{item.get('canonical_id', '')}', "
                f"author_email='{item.get('author_email', '')}', author_name='{item.get('author_name', '')}', "
                f"author='{item.get('author', '')}', primary_name='{item.get('primary_name', '')}', "
                f"name='{item.get('name', '')}'"
            )

        for item in data_list:
            canonical_id = item.get("canonical_id", "")
            # Also check original author fields as fallback for data without canonical_id
            author_email = item.get("author_email", "")
            author_name = item.get("author_name", "")

            # Check all possible author fields
            author = item.get("author", "")
            primary_name = item.get("primary_name", "")
            name = item.get("name", "")

            # Collect all identity fields for checking
            identity_fields = [
                canonical_id,
                item.get("primary_email", ""),
                author_email,
                author_name,
                author,
                primary_name,
                name,
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
                            exclusion_reason = (
                                f"bot pattern '{bot_pattern}' matches field '{field}'"
                            )
                            break

                    # Additional bot detection: check for common bot patterns not in explicit list
                    if not should_exclude:
                        bot_indicators = [
                            "[bot]",
                            "bot@",
                            "-bot",
                            "automated",
                            "github-actions",
                            "dependabot",
                            "renovate",
                        ]
                        for indicator in bot_indicators:
                            if indicator in field_lower:
                                # Only exclude if this bot-like pattern matches something in our exclude list
                                for exclude in excluded_lower:
                                    if (
                                        indicator.replace("[", "").replace("]", "") in exclude
                                        or exclude in field_lower
                                    ):
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
                logger.debug(
                    f"  canonical_id='{canonical_id}', primary_email='{item.get('primary_email', '')}', "
                    f"author_email='{author_email}', author_name='{author_name}', author='{author}', "
                    f"primary_name='{primary_name}', name='{name}'"
                )
            else:
                filtered_data.append(item)

        logger.debug(
            f"DEBUG EXCLUSION: Excluded {excluded_count} items, kept {len(filtered_data)} items"
        )
        return filtered_data

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
        branch_health_metrics: dict[str, dict[str, Any]] = None,
        exclude_authors: list[str] = None,
        analysis_start_date: datetime = None,
        analysis_end_date: datetime = None,
        cicd_data: dict[str, Any] = None,
    ) -> Path:
        """Generate comprehensive narrative report."""
        # Store analysis period for use in weekly trends calculation
        self._analysis_start_date = analysis_start_date
        self._analysis_end_date = analysis_end_date

        logger.debug(
            f"DEBUG NARRATIVE: Starting report generation with exclude_authors: {exclude_authors}"
        )
        logger.debug(
            f"DEBUG NARRATIVE: Analysis period: {analysis_start_date} to {analysis_end_date}"
        )
        logger.debug(
            f"DEBUG NARRATIVE: Input data sizes - commits: {len(commits)}, developer_stats: {len(developer_stats)}, "
            f"activity_dist: {len(activity_dist)}, focus_data: {len(focus_data)}"
        )

        # Sample some developer_stats to see their structure
        if developer_stats:
            for i, dev in enumerate(developer_stats[:3]):
                logger.debug(
                    f"DEBUG NARRATIVE: Sample developer_stats[{i}]: canonical_id='{dev.get('canonical_id', '')}', "
                    f"primary_name='{dev.get('primary_name', '')}', name='{dev.get('name', '')}', "
                    f"primary_email='{dev.get('primary_email', '')}'"
                )

        # Filter out excluded authors in Phase 2 using canonical_id
        if exclude_authors:
            logger.debug(
                f"DEBUG NARRATIVE: Applying exclusion filter with {len(exclude_authors)} excluded authors"
            )

            original_commits = len(commits)
            commits = self._filter_excluded_authors(commits, exclude_authors)
            filtered_commits = original_commits - len(commits)

            # Filter other data structures too
            logger.debug(
                f"DEBUG NARRATIVE: Filtering developer_stats (original: {len(developer_stats)})"
            )
            developer_stats = self._filter_excluded_authors(developer_stats, exclude_authors)
            logger.debug(
                f"DEBUG NARRATIVE: After filtering developer_stats: {len(developer_stats)}"
            )

            activity_dist = self._filter_excluded_authors(activity_dist, exclude_authors)
            focus_data = self._filter_excluded_authors(focus_data, exclude_authors)

            if filtered_commits > 0:
                logger.info(
                    f"Filtered out {filtered_commits} commits from {len(exclude_authors)} excluded authors in narrative report"
                )

            # Log remaining developers after filtering
            if developer_stats:
                remaining_devs = [
                    dev.get("primary_name", dev.get("name", "Unknown")) for dev in developer_stats
                ]
                logger.debug(
                    f"DEBUG NARRATIVE: Remaining developers after filtering: {remaining_devs}"
                )
        else:
            logger.debug("DEBUG NARRATIVE: No exclusion filter applied")

        # Initialize progress tracking for narrative report generation
        progress_service = get_progress_service()

        # Count all sections to be generated (including conditional ones)
        sections = []
        sections.append(("Executive Summary", True))
        sections.append(("Qualitative Analysis", bool(chatgpt_summary)))
        sections.append(("Team Composition", True))
        sections.append(("Project Activity", True))
        sections.append(("Development Patterns", True))
        sections.append(
            (
                "Commit Classification Analysis",
                ticket_analysis.get("ml_analysis", {}).get("enabled", False),
            )
        )
        sections.append(
            ("Pull Request Analysis", bool(pr_metrics and pr_metrics.get("total_prs", 0) > 0))
        )
        sections.append(("Issue Tracking", True))
        sections.append(("PM Platform Integration", bool(pm_data and "metrics" in pm_data)))
        sections.append(("Recommendations", True))

        # Filter to only included sections
        active_sections = [name for name, include in sections if include]
        total_sections = len(active_sections)

        logger.debug(
            f"Generating narrative report with {total_sections} sections: {', '.join(active_sections)}"
        )

        # Create progress context for narrative report generation
        with progress_service.progress(
            total_sections, "Generating narrative report sections", unit="sections"
        ) as progress_ctx:
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

            # When there are no commits, emit a clear notice and skip the detailed sections.
            if not commits:
                report.write("## No Commits Found\n\n")
                report.write(
                    "No commits were found in the analysis period. "
                    "This report has been generated with empty data.\n\n"
                    "Possible reasons:\n"
                    "- The repository was quiet during this period\n"
                    "- The `--weeks` window is too short\n"
                    "- The date range does not overlap with any commits in the configured repositories\n"
                )
                with open(output_path, "w") as f:
                    f.write(report.getvalue())
                return output_path

            # Executive Summary
            progress_service.set_description(progress_ctx, "Generating Executive Summary")
            report.write("## Executive Summary\n\n")
            self._write_executive_summary(
                report,
                commits,
                developer_stats,
                ticket_analysis,
                prs,
                branch_health_metrics,
                pm_data,
            )
            progress_service.update(progress_ctx)

            # Add ChatGPT qualitative insights if available
            if chatgpt_summary:
                progress_service.set_description(progress_ctx, "Generating Qualitative Analysis")
                report.write("\n## Qualitative Analysis\n\n")
                report.write(chatgpt_summary)
                report.write("\n")
                progress_service.update(progress_ctx)

            # Team Composition
            progress_service.set_description(progress_ctx, "Generating Team Composition")
            report.write("\n## Team Composition\n\n")
            self._write_team_composition(
                report, developer_stats, focus_data, commits, prs, ticket_analysis, weeks
            )
            progress_service.update(progress_ctx)

            # Project Activity
            progress_service.set_description(progress_ctx, "Generating Project Activity")
            report.write("\n## Project Activity\n\n")
            self._write_project_activity(
                report, activity_dist, commits, branch_health_metrics, ticket_analysis, weeks
            )
            progress_service.update(progress_ctx)

            # Development Patterns
            progress_service.set_description(progress_ctx, "Generating Development Patterns")
            report.write("\n## Development Patterns\n\n")
            self._write_development_patterns(report, insights, focus_data)
            progress_service.update(progress_ctx)

            # Commit Classification Analysis (if ML analysis is available)
            if ticket_analysis.get("ml_analysis", {}).get("enabled", False):
                progress_service.set_description(
                    progress_ctx, "Generating Commit Classification Analysis"
                )
                report.write("\n## Commit Classification Analysis\n\n")
                self._write_commit_classification_analysis(report, ticket_analysis)
                progress_service.update(progress_ctx)

            # Pull Request Analysis (if available)
            if pr_metrics and pr_metrics.get("total_prs", 0) > 0:
                progress_service.set_description(progress_ctx, "Generating Pull Request Analysis")
                report.write("\n## Pull Request Analysis\n\n")
                self._write_pr_analysis(report, pr_metrics, prs)
                progress_service.update(progress_ctx)

            # Issue Tracking (includes Enhanced Untracked Analysis)
            progress_service.set_description(progress_ctx, "Generating Issue Tracking")
            report.write("\n## Issue Tracking\n\n")
            self._write_ticket_tracking(report, ticket_analysis, developer_stats)
            progress_service.update(progress_ctx)

            # PM Platform Insights
            if pm_data and "metrics" in pm_data:
                progress_service.set_description(progress_ctx, "Generating PM Platform Integration")
                report.write("\n## PM Platform Integration\n\n")
                self._write_pm_insights(report, pm_data)
                progress_service.update(progress_ctx)

            # CI/CD Pipeline Health
            if cicd_data and cicd_data.get("pipelines"):
                progress_service.set_description(progress_ctx, "Generating CI/CD Pipeline Health")
                report.write("\n## CI/CD Pipeline Health\n\n")
                self._write_cicd_health(report, cicd_data)
                progress_service.update(progress_ctx)

            # Recommendations
            progress_service.set_description(progress_ctx, "Generating Recommendations")
            report.write("\n## Recommendations\n\n")
            self._write_recommendations(report, insights, ticket_analysis, focus_data)
            progress_service.update(progress_ctx)

            # Write to file
            with open(output_path, "w") as f:
                f.write(report.getvalue())

        return output_path

