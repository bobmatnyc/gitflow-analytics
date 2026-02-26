"""Narrative report mixin: executive summary and commit classification helpers.

Extracted from narrative_writer.py to keep file sizes manageable.
"""

import logging
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)



class NarrativeExecutiveMixin:
    """Mixin: executive summary and initial classification methods."""

    def _write_executive_summary(
        self,
        report: StringIO,
        commits: list[dict[str, Any]],
        developer_stats: list[dict[str, Any]],
        ticket_analysis: dict[str, Any],
        prs: list[dict[str, Any]],
        branch_health_metrics: dict[str, dict[str, Any]] = None,
        pm_data: dict[str, Any] = None,
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

        # PM Platform Story Points (if available)
        if pm_data and "metrics" in pm_data:
            metrics = pm_data.get("metrics", {})
            story_analysis = metrics.get("story_point_analysis", {})
            pm_story_points = story_analysis.get("pm_total_story_points", 0)
            git_story_points = story_analysis.get("git_total_story_points", 0)

            if pm_story_points > 0 or git_story_points > 0:
                report.write(
                    f"- **PM Story Points**: {pm_story_points:,} (platform) / {git_story_points:,} (commit-linked)\n"
                )

        # Add repository branch health summary
        if branch_health_metrics:
            # Aggregate branch health across all repositories
            total_branches = 0
            total_stale = 0
            overall_health_scores = []

            for _repo_name, metrics in branch_health_metrics.items():
                summary = metrics.get("summary", {})
                health_indicators = metrics.get("health_indicators", {})

                total_branches += summary.get("total_branches", 0)
                total_stale += summary.get("stale_branches", 0)

                if health_indicators.get("overall_health_score") is not None:
                    overall_health_scores.append(health_indicators["overall_health_score"])

            # Calculate average health score
            avg_health_score = (
                sum(overall_health_scores) / len(overall_health_scores)
                if overall_health_scores
                else 0
            )

            # Determine health status
            if avg_health_score >= 80:
                health_status = "Excellent"
            elif avg_health_score >= 60:
                health_status = "Good"
            elif avg_health_score >= 40:
                health_status = "Fair"
            else:
                health_status = "Needs Attention"

            report.write(
                f"- **Branch Health**: {health_status} ({avg_health_score:.0f}/100) - "
                f"{total_branches} branches, {total_stale} stale\n"
            )

        # Projects worked on - show full list instead of just count
        projects = set(c.get("project_key", "UNKNOWN") for c in commits)
        projects_list = sorted(projects)
        report.write(f"- **Active Projects**: {', '.join(projects_list)}\n")

        # Top contributor with proper format matching old report
        if developer_stats and commits:
            # BUGFIX: Calculate period-specific commit counts instead of using all-time totals
            period_commit_counts = {}
            for commit in commits:
                canonical_id = commit.get("canonical_id", "")
                period_commit_counts[canonical_id] = period_commit_counts.get(canonical_id, 0) + 1

            # Find the developer with most commits in this period
            if period_commit_counts:
                top_canonical_id = max(period_commit_counts, key=period_commit_counts.get)
                top_period_commits = period_commit_counts[top_canonical_id]

                # Find the developer stats entry for this canonical_id
                top_dev = None
                for dev in developer_stats:
                    if dev.get("canonical_id") == top_canonical_id:
                        top_dev = dev
                        break

                if top_dev:
                    # Handle both 'primary_name' (production) and 'name' (tests) for backward compatibility
                    dev_name = top_dev.get("primary_name", top_dev.get("name", "Unknown Developer"))
                    report.write(
                        f"- **Top Contributor**: {dev_name} with {top_period_commits} commits\n"
                    )
            elif developer_stats:
                # Fallback: use first developer but with 0 commits (shouldn't happen with proper filtering)
                top_dev = developer_stats[0]
                dev_name = top_dev.get("primary_name", top_dev.get("name", "Unknown Developer"))
                report.write(f"- **Top Contributor**: {dev_name} with 0 commits\n")

            # Calculate team average activity
            if commits:
                # Quick activity score calculation for executive summary
                # total_prs = len(prs) if prs else 0  # Not used yet
                total_lines = sum(
                    c.get("filtered_insertions", c.get("insertions", 0))
                    + c.get("filtered_deletions", c.get("deletions", 0))
                    for c in commits
                )

                # BUGFIX: Basic team activity assessment using only active developers in period
                active_devs_in_period = len(period_commit_counts) if period_commit_counts else 0
                avg_commits_per_dev = (
                    len(commits) / active_devs_in_period if active_devs_in_period > 0 else 0
                )
                if avg_commits_per_dev >= 10:
                    activity_assessment = "high activity"
                elif avg_commits_per_dev >= 5:
                    activity_assessment = "moderate activity"
                else:
                    activity_assessment = "low activity"

                report.write(
                    f"- **Team Activity**: {activity_assessment} (avg {avg_commits_per_dev:.1f} commits/developer)\n"
                )

    def _aggregate_commit_classifications(
        self,
        ticket_analysis: dict[str, Any],
        commits: list[dict[str, Any]] = None,
        developer_stats: list[dict[str, Any]] = None,
    ) -> dict[str, dict[str, int]]:
        """Aggregate commit classifications per developer.

        WHY: This method provides detailed breakdown of commit types per developer,
        replacing simple commit counts with actionable insights into what types of
        work each developer is doing. This helps identify patterns and training needs.

        DESIGN DECISION: Classify ALL commits (tracked and untracked) into proper
        categories (feature, bug_fix, refactor, etc.) rather than using 'tracked_work'
        as a category. For tracked commits, use ticket information to enhance accuracy.

        Args:
            ticket_analysis: Ticket analysis data containing classification info
            commits: Optional list of all commits for complete categorization
            developer_stats: Developer statistics for mapping canonical IDs

        Returns:
            Dictionary mapping developer canonical_id to category counts:
            {
                'dev_canonical_id': {
                    'feature': 15,
                    'bug_fix': 8,
                    'maintenance': 5,
                    ...
                }
            }
        """
        # Defensive type checking
        if not isinstance(ticket_analysis, dict):
            return {}

        if commits is not None and not isinstance(commits, list):
            # Log the error and continue without commits data
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Expected commits to be list or None, got {type(commits)}: {commits}")
            commits = None

        if developer_stats is not None and not isinstance(developer_stats, list):
            developer_stats = None

        classifications = {}

        # If we have full commits data, classify ALL commits properly
        if commits and isinstance(commits, list):
            # Import the ticket extractor for classification
            try:
                from ..extractors.ml_tickets import MLTicketExtractor

                extractor = MLTicketExtractor(enable_ml=True)
            except Exception:
                # Fallback to basic ticket extractor
                from ..extractors.tickets import TicketExtractor

                extractor = TicketExtractor()

            # Classify all commits
            for commit in commits:
                canonical_id = commit.get("canonical_id", "Unknown")
                message = commit.get("message", "")

                # Get files_changed in proper format for classification
                files_changed = commit.get("files_changed", [])
                if isinstance(files_changed, int):
                    # If files_changed is just a count, we can't provide file names
                    files_changed = []
                elif not isinstance(files_changed, list):
                    files_changed = []

                # Use ticket information to enhance classification for tracked commits
                ticket_refs = commit.get("ticket_references", [])

                if ticket_refs and hasattr(extractor, "categorize_commit_with_confidence"):
                    # Use ML categorization with confidence for tracked commits
                    try:
                        result = extractor.categorize_commit_with_confidence(message, files_changed)
                        category = result["category"]
                        # For tracked commits with ticket info, try to infer better category from ticket type
                        category = self._enhance_category_with_ticket_info(
                            category, ticket_refs, message
                        )
                    except Exception:
                        # Fallback to basic categorization
                        category = extractor.categorize_commit(message)
                else:
                    # Use basic categorization for untracked commits
                    category = extractor.categorize_commit(message)

                # Initialize developer classification if not exists
                if canonical_id not in classifications:
                    classifications[canonical_id] = {}

                # Initialize category count if not exists
                if category not in classifications[canonical_id]:
                    classifications[canonical_id][category] = 0

                # Increment category count
                classifications[canonical_id][category] += 1

        else:
            # Fallback: Only process untracked commits (legacy behavior)
            untracked_commits = ticket_analysis.get("untracked_commits", [])

            # Process untracked commits (these have category information)
            for commit in untracked_commits:
                author = commit.get("author", "Unknown")
                category = commit.get("category", "other")

                # Map author to canonical_id if developer_stats is available
                canonical_id = author  # fallback
                if developer_stats:
                    for dev in developer_stats:
                        # Check multiple possible name mappings
                        if (
                            dev.get("primary_name") == author
                            or dev.get("primary_email") == author
                            or dev.get("canonical_id") == author
                        ):
                            canonical_id = dev.get("canonical_id", author)
                            break

                if canonical_id not in classifications:
                    classifications[canonical_id] = {}

                if category not in classifications[canonical_id]:
                    classifications[canonical_id][category] = 0

                classifications[canonical_id][category] += 1

        return classifications

    def _enhance_category_with_ticket_info(
        self, category: str, ticket_refs: list, message: str
    ) -> str:
        """Enhance commit categorization using ticket reference information.

        WHY: For tracked commits, we can often infer better categories by examining
        the ticket references and message content. This improves classification accuracy
        for tracked work versus relying purely on message patterns.

        Args:
            category: Base category from ML/rule-based classification
            ticket_refs: List of ticket references for this commit
            message: Commit message

        Returns:
            Enhanced category, potentially refined based on ticket information
        """
        if not ticket_refs:
            return category

        # Try to extract insights from ticket references and message
        message_lower = message.lower()

        # Look for ticket type patterns in the message or ticket IDs
        # These patterns suggest specific categories regardless of base classification
        if any(
            pattern in message_lower
            for pattern in ["hotfix", "critical", "urgent", "prod", "production"]
        ):
            return "bug_fix"  # Production/critical issues are typically bug fixes

        if any(pattern in message_lower for pattern in ["feature", "epic", "story", "user story"]):
            return "feature"  # Explicitly mentioned features

        # Look for JIRA/GitHub issue patterns that might indicate bug fixes
        for ticket_ref in ticket_refs:
            if isinstance(ticket_ref, dict):
                ticket_id = ticket_ref.get("id", "").lower()
            else:
                ticket_id = str(ticket_ref).lower()

            # Common bug fix patterns in ticket IDs
            if any(pattern in ticket_id for pattern in ["bug", "fix", "issue", "defect"]):
                return "bug_fix"

            # Feature patterns in ticket IDs
            if any(pattern in ticket_id for pattern in ["feat", "feature", "epic", "story"]):
                return "feature"

        # If no specific enhancement found, return original category
        return category

    def _get_project_classifications(
        self, project: str, commits: list[dict[str, Any]], ticket_analysis: dict[str, Any]
    ) -> dict[str, int]:
        """Get commit classification breakdown for a specific project.

        WHY: This method filters classification data to show only commits belonging
        to a specific project, enabling project-specific classification insights
        in the project activity section.

        DESIGN DECISION: Classify ALL commits (tracked and untracked) for this project
        into proper categories rather than lumping tracked commits as 'tracked_work'.

        Args:
            project: Project key to filter by
            commits: List of all commits for mapping
            ticket_analysis: Ticket analysis data containing classifications

        Returns:
            Dictionary mapping category names to commit counts for this project:
            {'feature': 15, 'bug_fix': 8, 'refactor': 5, ...}
        """
        if not isinstance(ticket_analysis, dict):
            return {}

        project_classifications = {}

        # First, try to use already classified untracked commits
        untracked_commits = ticket_analysis.get("untracked_commits", [])
        for commit in untracked_commits:
            commit_project = commit.get("project_key", "UNKNOWN")
            if commit_project == project:
                category = commit.get("category", "other")
                if category not in project_classifications:
                    project_classifications[category] = 0
                project_classifications[category] += 1

        # If we have classifications from untracked commits, use those
        if project_classifications:
            return project_classifications

        # Fallback: If no untracked commits data, classify all commits for this project
        if isinstance(commits, list):
            # Import the ticket extractor for classification
            try:
                from ..extractors.ml_tickets import MLTicketExtractor

                extractor = MLTicketExtractor(enable_ml=True)
            except Exception:
                # Fallback to basic ticket extractor
                from ..extractors.tickets import TicketExtractor

                extractor = TicketExtractor()

            # Classify all commits for this project
            for commit in commits:
                commit_project = commit.get("project_key", "UNKNOWN")
                if commit_project == project:
                    message = commit.get("message", "")

                    # Get files_changed in proper format for classification
                    files_changed = commit.get("files_changed", [])
                    if isinstance(files_changed, int):
                        # If files_changed is just a count, we can't provide file names
                        files_changed = []
                    elif not isinstance(files_changed, list):
                        files_changed = []

                    # Use ticket information to enhance classification for tracked commits
                    ticket_refs = commit.get("ticket_references", [])

                    if ticket_refs and hasattr(extractor, "categorize_commit_with_confidence"):
                        # Use ML categorization with confidence for tracked commits
                        try:
                            result = extractor.categorize_commit_with_confidence(
                                message, files_changed
                            )
                            category = result["category"]
                            # For tracked commits with ticket info, try to infer better category from ticket type
                            category = self._enhance_category_with_ticket_info(
                                category, ticket_refs, message
                            )
                        except Exception:
                            # Fallback to basic categorization
                            category = extractor.categorize_commit(message)
                    else:
                        # Use basic categorization for untracked commits
                        category = extractor.categorize_commit(message)

                    # Initialize category count if not exists
                    if category not in project_classifications:
                        project_classifications[category] = 0

                    # Increment category count
                    project_classifications[category] += 1

        return project_classifications

    def _format_category_name(self, category: str) -> str:
        """Convert internal category names to user-friendly display names.

        Args:
            category: Internal category name (e.g., 'bug_fix', 'feature', 'refactor')

        Returns:
            User-friendly display name (e.g., 'Bug Fixes', 'Features', 'Refactoring')
        """
        category_mapping = {
            "bug_fix": "Bug Fixes",
            "feature": "Features",
            "refactor": "Refactoring",
            "documentation": "Documentation",
            "maintenance": "Maintenance",
            "test": "Testing",
            "style": "Code Style",
            "build": "Build/CI",
            "other": "Other",
        }
        return category_mapping.get(category, category.replace("_", " ").title())

