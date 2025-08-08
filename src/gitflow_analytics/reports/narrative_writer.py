"""Narrative report generation in Markdown format."""

import logging
from datetime import datetime, timedelta, timezone
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
        branch_health_metrics: dict[str, dict[str, Any]] = None,
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
        self._write_executive_summary(report, commits, developer_stats, ticket_analysis, prs, branch_health_metrics)

        # Add ChatGPT qualitative insights if available
        if chatgpt_summary:
            report.write("\n## Qualitative Analysis\n\n")
            report.write(chatgpt_summary)
            report.write("\n")

        # Team Composition
        report.write("\n## Team Composition\n\n")
        self._write_team_composition(report, developer_stats, focus_data, commits, prs, ticket_analysis, weeks)

        # Project Activity
        report.write("\n## Project Activity\n\n")
        self._write_project_activity(report, activity_dist, commits, branch_health_metrics, ticket_analysis, weeks)


        # Development Patterns
        report.write("\n## Development Patterns\n\n")
        self._write_development_patterns(report, insights, focus_data)

        # Commit Classification Analysis (if ML analysis is available)
        if ticket_analysis.get("ml_analysis", {}).get("enabled", False):
            report.write("\n## Commit Classification Analysis\n\n")
            self._write_commit_classification_analysis(report, ticket_analysis)

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
        branch_health_metrics: dict[str, dict[str, Any]] = None,
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
            avg_health_score = sum(overall_health_scores) / len(overall_health_scores) if overall_health_scores else 0
            
            # Determine health status
            if avg_health_score >= 80:
                health_status = "Excellent"
            elif avg_health_score >= 60:
                health_status = "Good"
            elif avg_health_score >= 40:
                health_status = "Fair"
            else:
                health_status = "Needs Attention"
            
            report.write(f"- **Branch Health**: {health_status} ({avg_health_score:.0f}/100) - "
                        f"{total_branches} branches, {total_stale} stale\n")

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

    def _aggregate_commit_classifications(
        self, 
        ticket_analysis: dict[str, Any], 
        commits: list[dict[str, Any]] = None,
        developer_stats: list[dict[str, Any]] = None
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
                
                if ticket_refs and hasattr(extractor, 'categorize_commit_with_confidence'):
                    # Use ML categorization with confidence for tracked commits
                    try:
                        result = extractor.categorize_commit_with_confidence(message, files_changed)
                        category = result['category']
                        # For tracked commits with ticket info, try to infer better category from ticket type
                        category = self._enhance_category_with_ticket_info(category, ticket_refs, message)
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
                        if (dev.get("primary_name") == author or 
                            dev.get("primary_email") == author or
                            dev.get("canonical_id") == author):
                            canonical_id = dev.get("canonical_id", author)
                            break
                
                if canonical_id not in classifications:
                    classifications[canonical_id] = {}
                
                if category not in classifications[canonical_id]:
                    classifications[canonical_id][category] = 0
                
                classifications[canonical_id][category] += 1
        
        return classifications
    
    def _enhance_category_with_ticket_info(self, category: str, ticket_refs: list, message: str) -> str:
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
        if any(pattern in message_lower for pattern in ['hotfix', 'critical', 'urgent', 'prod', 'production']):
            return 'bug_fix'  # Production/critical issues are typically bug fixes
        
        if any(pattern in message_lower for pattern in ['feature', 'epic', 'story', 'user story']):
            return 'feature'  # Explicitly mentioned features
        
        # Look for JIRA/GitHub issue patterns that might indicate bug fixes
        for ticket_ref in ticket_refs:
            if isinstance(ticket_ref, dict):
                ticket_id = ticket_ref.get('id', '').lower()
            else:
                ticket_id = str(ticket_ref).lower()
            
            # Common bug fix patterns in ticket IDs
            if any(pattern in ticket_id for pattern in ['bug', 'fix', 'issue', 'defect']):
                return 'bug_fix'
            
            # Feature patterns in ticket IDs
            if any(pattern in ticket_id for pattern in ['feat', 'feature', 'epic', 'story']):
                return 'feature'
        
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
                    
                    if ticket_refs and hasattr(extractor, 'categorize_commit_with_confidence'):
                        # Use ML categorization with confidence for tracked commits
                        try:
                            result = extractor.categorize_commit_with_confidence(message, files_changed)
                            category = result['category']
                            # For tracked commits with ticket info, try to infer better category from ticket type
                            category = self._enhance_category_with_ticket_info(category, ticket_refs, message)
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
            'bug_fix': 'Bug Fixes',
            'feature': 'Features', 
            'refactor': 'Refactoring',
            'documentation': 'Documentation',
            'maintenance': 'Maintenance',
            'test': 'Testing',
            'style': 'Code Style',
            'build': 'Build/CI',
            'other': 'Other'
        }
        return category_mapping.get(category, category.replace('_', ' ').title())
    
    def _calculate_weekly_classification_percentages(
        self,
        commits: list[dict[str, Any]],
        developer_id: str = None,
        project_key: str = None,
        weeks: int = 4
    ) -> list[dict[str, Any]]:
        """Calculate weekly classification percentages for trend lines.
        
        WHY: This method creates detailed week-by-week breakdown of commit classifications
        showing how work type distribution changes over time, providing granular insights
        into development patterns and workload shifts.
        
        DESIGN DECISION: Break down the analysis period into individual weeks and show
        classification percentages for each week, with change indicators from previous week.
        Only show categories that represent >5% of work to avoid noise.
        
        Args:
            commits: List of all commits with timestamps and classifications
            developer_id: Optional canonical developer ID to filter by
            project_key: Optional project key to filter by
            weeks: Total analysis period in weeks
            
        Returns:
            List of weekly data dictionaries:
            [
                {
                    'week_start': datetime,
                    'week_display': 'Jul 7-13',
                    'classifications': {'Features': 45.0, 'Bug Fixes': 30.0, 'Maintenance': 25.0},
                    'changes': {'Features': 5.0, 'Bug Fixes': -5.0, 'Maintenance': 0.0}
                },
                ...
            ]
        """
        if not commits or weeks < 1:
            return []
        
        # Filter commits by developer or project if specified
        filtered_commits = []
        for commit in commits:
            if developer_id and commit.get('canonical_id') != developer_id:
                continue
            if project_key and commit.get('project_key') != project_key:
                continue
            filtered_commits.append(commit)
        
        if len(filtered_commits) < 2:
            return []
        
        # Group commits by week
        weekly_commits = {}
        
        for commit in filtered_commits:
            timestamp = commit.get('timestamp')
            if not timestamp:
                continue
            
            # Ensure timezone consistency
            if hasattr(timestamp, 'tzinfo'):
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                elif timestamp.tzinfo != timezone.utc:
                    timestamp = timestamp.astimezone(timezone.utc)
            
            # Get week start (Monday)
            week_start = self._get_week_start(timestamp)
            
            if week_start not in weekly_commits:
                weekly_commits[week_start] = []
            weekly_commits[week_start].append(commit)
        
        if len(weekly_commits) < 2:
            return []
        
        # Sort weeks chronologically
        sorted_weeks = sorted(weekly_commits.keys())
        
        # Import classifiers
        try:
            from ..extractors.ml_tickets import MLTicketExtractor
            extractor = MLTicketExtractor(enable_ml=True)
        except Exception:
            from ..extractors.tickets import TicketExtractor
            extractor = TicketExtractor()
        
        # Calculate classifications for each week
        weekly_data = []
        previous_percentages = {}
        
        for week_start in sorted_weeks:
            week_commits = weekly_commits[week_start]
            
            # Classify commits for this week
            week_classifications = {}
            for commit in week_commits:
                message = commit.get('message', '')
                files_changed = commit.get('files_changed', [])
                if isinstance(files_changed, int) or not isinstance(files_changed, list):
                    files_changed = []
                
                ticket_refs = commit.get('ticket_references', [])
                
                if ticket_refs and hasattr(extractor, 'categorize_commit_with_confidence'):
                    try:
                        result = extractor.categorize_commit_with_confidence(message, files_changed)
                        category = result['category']
                        category = self._enhance_category_with_ticket_info(category, ticket_refs, message)
                    except Exception:
                        category = extractor.categorize_commit(message)
                else:
                    category = extractor.categorize_commit(message)
                
                if category not in week_classifications:
                    week_classifications[category] = 0
                week_classifications[category] += 1
            
            # Calculate percentages
            total_commits = sum(week_classifications.values())
            if total_commits == 0:
                continue
            
            week_percentages = {}
            for category, count in week_classifications.items():
                percentage = (count / total_commits) * 100
                if percentage >= 5.0:  # Only include significant categories
                    display_name = self._format_category_name(category)
                    week_percentages[display_name] = percentage
            
            # Calculate changes from previous week
            changes = {}
            if previous_percentages:
                for category in set(week_percentages.keys()) | set(previous_percentages.keys()):
                    current_pct = week_percentages.get(category, 0.0)
                    prev_pct = previous_percentages.get(category, 0.0)
                    change = current_pct - prev_pct
                    if abs(change) >= 1.0:  # Only show changes >= 1%
                        changes[category] = change
            
            # Format week display
            week_end = week_start + timedelta(days=6)
            week_display = f"{week_start.strftime('%b %d')}-{week_end.strftime('%d')}"
            
            weekly_data.append({
                'week_start': week_start,
                'week_display': week_display,
                'classifications': week_percentages,
                'changes': changes
            })
            
            previous_percentages = week_percentages.copy()
        
        return weekly_data
    
    def _calculate_classification_trends(
        self, 
        commits: list[dict[str, Any]], 
        developer_id: str = None, 
        project_key: str = None,
        weeks: int = 4
    ) -> dict[str, float]:
        """Calculate week-over-week changes in classification percentages.
        
        WHY: This method provides trend analysis showing how development patterns
        change over time, helping identify shifts in work type distribution.
        
        DESIGN DECISION: Compare the most recent half of the analysis period
        with the earlier half to show meaningful trends. For shorter periods,
        compare week-to-week. Use percentage point changes for clarity.
        
        Args:
            commits: List of all commits with timestamps and classifications
            developer_id: Optional canonical developer ID to filter by
            project_key: Optional project key to filter by
            weeks: Total analysis period in weeks
            
        Returns:
            Dictionary mapping category names to percentage point changes:
            {'Features': 15.2, 'Bug Fixes': -8.1, 'Refactoring': 3.4}
            Positive values indicate increases, negative indicate decreases.
        """
        if not commits or len(commits) < 2:
            return {}
        
        # Filter commits by developer or project if specified
        filtered_commits = []
        for commit in commits:
            if developer_id and commit.get('canonical_id') != developer_id:
                continue
            if project_key and commit.get('project_key') != project_key:
                continue
            filtered_commits.append(commit)
        
        if len(filtered_commits) < 2:
            return {}
        
        # Sort commits by timestamp
        def safe_timestamp_key(commit):
            ts = commit.get('timestamp')
            if ts is None:
                return datetime.min.replace(tzinfo=timezone.utc)
            if hasattr(ts, 'tzinfo'):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return ts
            return ts
        
        sorted_commits = sorted(filtered_commits, key=safe_timestamp_key)
        
        if len(sorted_commits) < 4:  # Need at least 4 commits for meaningful trend
            return {}
        
        # Determine time split strategy based on analysis period
        if weeks <= 2:
            # For short periods (1-2 weeks), compare last 3 days vs previous 3+ days
            cutoff_days = 3
        elif weeks <= 4:
            # For 3-4 week periods, compare last week vs previous weeks
            cutoff_days = 7
        else:
            # For longer periods, compare recent half vs older half
            cutoff_days = (weeks * 7) // 2
        
        # Calculate cutoff timestamp
        latest_timestamp = safe_timestamp_key(sorted_commits[-1])
        cutoff_timestamp = latest_timestamp - timedelta(days=cutoff_days)
        
        # Split commits into recent and previous periods
        recent_commits = [c for c in sorted_commits if safe_timestamp_key(c) >= cutoff_timestamp]
        previous_commits = [c for c in sorted_commits if safe_timestamp_key(c) < cutoff_timestamp]
        
        if not recent_commits or not previous_commits:
            return {}
        
        # Classify commits for both periods
        def get_period_classifications(period_commits):
            period_classifications = {}
            
            # Import classifiers
            try:
                from ..extractors.ml_tickets import MLTicketExtractor
                extractor = MLTicketExtractor(enable_ml=True)
            except Exception:
                from ..extractors.tickets import TicketExtractor
                extractor = TicketExtractor()
            
            for commit in period_commits:
                message = commit.get('message', '')
                files_changed = commit.get('files_changed', [])
                if isinstance(files_changed, int) or not isinstance(files_changed, list):
                    files_changed = []
                
                # Get ticket info for enhancement
                ticket_refs = commit.get('ticket_references', [])
                
                if ticket_refs and hasattr(extractor, 'categorize_commit_with_confidence'):
                    try:
                        result = extractor.categorize_commit_with_confidence(message, files_changed)
                        category = result['category']
                        category = self._enhance_category_with_ticket_info(category, ticket_refs, message)
                    except Exception:
                        category = extractor.categorize_commit(message)
                else:
                    category = extractor.categorize_commit(message)
                
                if category not in period_classifications:
                    period_classifications[category] = 0
                period_classifications[category] += 1
            
            return period_classifications
        
        recent_classifications = get_period_classifications(recent_commits)
        previous_classifications = get_period_classifications(previous_commits)
        
        # Calculate percentage changes
        trends = {}
        all_categories = set(recent_classifications.keys()) | set(previous_classifications.keys())
        
        total_recent = sum(recent_classifications.values())
        total_previous = sum(previous_classifications.values())
        
        if total_recent == 0 or total_previous == 0:
            return {}
        
        for category in all_categories:
            recent_count = recent_classifications.get(category, 0)
            previous_count = previous_classifications.get(category, 0)
            
            recent_pct = (recent_count / total_recent) * 100
            previous_pct = (previous_count / total_previous) * 100
            
            change = recent_pct - previous_pct
            
            # Only include significant changes (>= 5% absolute change)
            if abs(change) >= 5.0:
                display_name = self._format_category_name(category)
                trends[display_name] = change
        
        return trends
    
    def _format_trend_line(self, trends: dict[str, float], prefix: str = "📈 Trends") -> str:
        """Format trend data into a readable line with appropriate icons.
        
        WHY: This method provides consistent formatting for trend display across
        different sections of the report, using visual indicators to highlight
        increases, decreases, and overall patterns.
        
        Args:
            trends: Dictionary of category name to percentage change
            prefix: Text prefix for the trend line
            
        Returns:
            Formatted trend line string, or empty string if no significant trends
        """
        if not trends:
            return ""
        
        # Sort by absolute change magnitude (largest first)
        sorted_trends = sorted(trends.items(), key=lambda x: abs(x[1]), reverse=True)
        
        trend_parts = []
        for category, change in sorted_trends[:4]:  # Show top 4 trends
            if change > 0:
                icon = "⬆️"
                sign = "+"
            else:
                icon = "⬇️"
                sign = ""
            
            trend_parts.append(f"{category} {icon}{sign}{change:.0f}%")
        
        if trend_parts:
            return f"{prefix}: {', '.join(trend_parts)}"
        
        return ""
    
    def _write_weekly_trend_lines(
        self,
        report: StringIO,
        weekly_trends: list[dict[str, Any]],
        prefix: str = ""
    ) -> None:
        """Write weekly trend lines showing week-by-week classification changes.
        
        WHY: This method provides detailed weekly breakdown of work patterns,
        showing how development focus shifts over time with specific percentages
        and change indicators from previous weeks.
        
        Args:
            report: StringIO buffer to write to
            weekly_trends: List of weekly classification data
            prefix: Optional prefix for the trend section (e.g., "Project ")
        """
        if not weekly_trends:
            return
        
        report.write(f"- {prefix}Weekly Trends:\n")
        
        for i, week_data in enumerate(weekly_trends):
            week_display = week_data['week_display']
            classifications = week_data['classifications']
            changes = week_data['changes']
            
            if not classifications:
                continue
            
            # Format classifications with percentages
            classification_parts = []
            for category in sorted(classifications.keys()):
                percentage = classifications[category]
                change = changes.get(category, 0.0)
                
                if i == 0 or abs(change) < 1.0:
                    # First week or no significant change
                    classification_parts.append(f"{category} {percentage:.0f}%")
                else:
                    # Show change from previous week
                    if change > 0:
                        change_indicator = f"(+{change:.0f}%)"
                    else:
                        change_indicator = f"({change:.0f}%)"
                    classification_parts.append(f"{category} {percentage:.0f}% {change_indicator}")
            
            if classification_parts:
                classifications_text = ", ".join(classification_parts)
                report.write(f"  - Week {i+1} ({week_display}): {classifications_text}\n")
        
        # Add a blank line after trend lines for spacing
        # (Note: Don't add extra newline here as the caller will handle spacing)
    
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
                ) or 0
                metrics["lines_removed"] += commit.get(
                    "filtered_deletions", commit.get("deletions", 0)
                ) or 0
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

        for dev in developer_stats:  # All developers
            # Handle both 'primary_name' (production) and 'name' (tests) for backward compatibility
            name = dev.get("primary_name", dev.get("name", "Unknown Developer"))
            total_commits = dev["total_commits"]
            canonical_id = dev.get("canonical_id")

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
                        dev_classifications.items(), 
                        key=lambda x: x[1], 
                        reverse=True
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
                                commits, developer_id=canonical_id, weeks=weeks
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
                                commits, developer_id=canonical_id, weeks=weeks
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
                            commits, developer_id=canonical_id, weeks=weeks
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
                        commits, developer_id=canonical_id, weeks=weeks
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
        self, report: StringIO, activity_dist: list[dict[str, Any]], commits: list[dict[str, Any]],
        branch_health_metrics: dict[str, dict[str, Any]] = None,
        ticket_analysis: dict[str, Any] = None,
        weeks: int = 4
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
                project_classifications = self._get_project_classifications(project, commits, ticket_analysis)
                if project_classifications:
                    # Sort categories by count (descending)
                    sorted_categories = sorted(
                        project_classifications.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    # Calculate total for percentages
                    total_classified = sum(project_classifications.values())
                    if total_classified > 0:
                        category_parts = []
                        for category, count in sorted_categories:
                            pct = (count / total_classified) * 100
                            display_name = self._format_category_name(category)
                            category_parts.append(f"{display_name} ({pct:.0f}%)")
                        
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
                            project_weekly_trends = self._calculate_weekly_classification_percentages(
                                commits, project_key=project, weeks=weeks
                            )
                            if project_weekly_trends:
                                self._write_weekly_trend_lines(report, project_weekly_trends, "Project ")
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
                report.write(f"- Overall Health: {status_emoji} {status_text} ({health_score:.0f}/100)\n")
                report.write(f"- Total Branches: {total_branches}\n")
                report.write(f"  - Active: {active_branches} branches\n")
                report.write(f"  - Long-lived: {long_lived_branches} branches (>30 days)\n")
                report.write(f"  - Stale: {stale_branches} branches (>90 days)\n")
                
                # Show top problematic branches if any
                if branches:
                    # Sort branches by health score (ascending) to get worst first
                    problem_branches = [b for b in branches if b.get("health_score", 100) < 60 and not b.get("is_merged", False)]
                    problem_branches.sort(key=lambda x: x.get("health_score", 100))
                    
                    if problem_branches:
                        report.write("\n**Branches Needing Attention**:\n")
                        for i, branch in enumerate(problem_branches[:3]):  # Show top 3
                            name = branch.get("name", "unknown")
                            age = branch.get("age_days", 0)
                            behind = branch.get("behind_main", 0)
                            ahead = branch.get("ahead_of_main", 0)
                            score = branch.get("health_score", 0)
                            
                            report.write(f"  {i+1}. `{name}` (score: {score:.0f}/100)\n")
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

    def _get_week_start(self, date: datetime) -> datetime:
        """Get Monday of the week for a given date."""
        # Ensure consistent timezone handling - keep timezone info
        if hasattr(date, "tzinfo") and date.tzinfo is not None:
            # Keep timezone-aware but ensure it's UTC
            if date.tzinfo != timezone.utc:
                date = date.astimezone(timezone.utc)
        else:
            # Convert naive datetime to UTC timezone-aware
            date = date.replace(tzinfo=timezone.utc)

        days_since_monday = date.weekday()
        monday = date - timedelta(days=days_since_monday)
        result = monday.replace(hour=0, minute=0, second=0, microsecond=0)

        return result

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

        # Debug logging for ticket coverage issues
        logger.debug(f"Ticket coverage analysis - commits_with_tickets: {commits_with_tickets}, total_commits: {total_commits}, coverage_pct: {coverage_pct}")
        if commits_with_tickets == 0 and total_commits > 0:
            logger.warning(f"No commits found with ticket references out of {total_commits} total commits")
            # Log sample of ticket_analysis structure for debugging
            if "ticket_summary" in ticket_analysis:
                logger.debug(f"Ticket summary: {ticket_analysis['ticket_summary']}")
            if "ticket_platforms" in ticket_analysis:
                logger.debug(f"Ticket platforms: {ticket_analysis['ticket_platforms']}")

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
            
            # Safe timestamp sorting that handles mixed timezone types
            def safe_timestamp_key(commit):
                ts = commit.get("timestamp")
                if ts is None:
                    return datetime.min.replace(tzinfo=timezone.utc)
                # If it's a datetime object, handle timezone issues
                if hasattr(ts, "tzinfo"):
                    # Make timezone-naive datetime UTC-aware for consistent comparison
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    return ts
                # If it's a string or other type, try to parse or use as-is
                return ts
            
            recent_commits = sorted(
                untracked_commits, key=safe_timestamp_key, reverse=True
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

    def _write_commit_classification_analysis(
        self, report: StringIO, ticket_analysis: dict[str, Any]
    ) -> None:
        """Write commit classification analysis section.
        
        WHY: This section provides insights into automated commit categorization
        quality and distribution, helping teams understand their development patterns
        and the effectiveness of ML-based categorization.
        
        Args:
            report: StringIO buffer to write to
            ticket_analysis: Ticket analysis data containing ML classification results
        """
        ml_analysis = ticket_analysis.get("ml_analysis", {})
        if not ml_analysis.get("enabled", False):
            return
        
        report.write("The team's commit patterns reveal the following automated classification insights:\n\n")
        
        # Overall classification statistics
        total_ml_predictions = ml_analysis.get("total_ml_predictions", 0)
        total_rule_predictions = ml_analysis.get("total_rule_predictions", 0)
        total_cached_predictions = ml_analysis.get("total_cached_predictions", 0)
        total_predictions = total_ml_predictions + total_rule_predictions + total_cached_predictions
        
        if total_predictions > 0:
            report.write("### Classification Method Distribution\n\n")
            
            # Calculate percentages
            ml_pct = (total_ml_predictions / total_predictions) * 100
            rules_pct = (total_rule_predictions / total_predictions) * 100
            cached_pct = (total_cached_predictions / total_predictions) * 100
            
            report.write(f"- **ML-based Classifications**: {total_ml_predictions} commits ({ml_pct:.1f}%)\n")
            report.write(f"- **Rule-based Classifications**: {total_rule_predictions} commits ({rules_pct:.1f}%)\n")
            report.write(f"- **Cached Results**: {total_cached_predictions} commits ({cached_pct:.1f}%)\n\n")
            
            # Classification confidence analysis
            avg_confidence = ml_analysis.get("avg_confidence", 0)
            confidence_dist = ml_analysis.get("confidence_distribution", {})
            
            if confidence_dist:
                report.write("### Classification Confidence\n\n")
                report.write(f"- **Average Confidence**: {avg_confidence:.1%} across all classifications\n")
                
                high_conf = confidence_dist.get("high", 0)
                medium_conf = confidence_dist.get("medium", 0)
                low_conf = confidence_dist.get("low", 0)
                total_conf_items = high_conf + medium_conf + low_conf
                
                if total_conf_items > 0:
                    high_pct = (high_conf / total_conf_items) * 100
                    medium_pct = (medium_conf / total_conf_items) * 100
                    low_pct = (low_conf / total_conf_items) * 100
                    
                    report.write(f"- **High Confidence** (≥80%): {high_conf} commits ({high_pct:.1f}%)\n")
                    report.write(f"- **Medium Confidence** (60-79%): {medium_conf} commits ({medium_pct:.1f}%)\n")
                    report.write(f"- **Low Confidence** (<60%): {low_conf} commits ({low_pct:.1f}%)\n\n")
            
            # Category confidence breakdown
            category_confidence = ml_analysis.get("category_confidence", {})
            if category_confidence:
                report.write("### Classification Categories\n\n")
                
                # Sort categories by count (descending)
                sorted_categories = sorted(
                    category_confidence.items(), 
                    key=lambda x: x[1].get("count", 0), 
                    reverse=True
                )
                
                # Calculate total commits for percentages
                total_categorized = sum(data.get("count", 0) for data in category_confidence.values())
                
                for category, data in sorted_categories:
                    count = data.get("count", 0)
                    avg_conf = data.get("avg", 0)
                    
                    if count > 0:
                        category_pct = (count / total_categorized) * 100
                        category_display = category.replace("_", " ").title()
                        report.write(f"- **{category_display}**: {count} commits ({category_pct:.1f}%, avg confidence: {avg_conf:.1%})\n")
                
                report.write("\n")
            
            # Performance metrics
            processing_stats = ml_analysis.get("processing_time_stats", {})
            if processing_stats.get("total_ms", 0) > 0:
                avg_ms = processing_stats.get("avg_ms", 0)
                total_ms = processing_stats.get("total_ms", 0)
                
                report.write("### Processing Performance\n\n")
                report.write(f"- **Average Processing Time**: {avg_ms:.1f}ms per commit\n")
                report.write(f"- **Total Processing Time**: {total_ms:.0f}ms ({total_ms/1000:.1f} seconds)\n\n")
            
        
        else:
            report.write("No classification data available for analysis.\n\n")

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
