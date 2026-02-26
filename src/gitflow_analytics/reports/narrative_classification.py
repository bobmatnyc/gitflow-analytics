"""Narrative report mixin: weekly classification percentages and trend analysis.

Extracted from narrative_writer.py to keep file sizes manageable.
"""

import logging
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)



class NarrativeClassificationMixin:
    """Mixin: weekly classification percentages and trend/formatting methods."""

    def _calculate_weekly_classification_percentages(
        self,
        commits: list[dict[str, Any]],
        developer_id: str = None,
        project_key: str = None,
        weeks: int = 4,
        analysis_start_date: datetime = None,
        analysis_end_date: datetime = None,
    ) -> list[dict[str, Any]]:
        """Calculate weekly classification percentages for trend lines.

        WHY: This method creates detailed week-by-week breakdown of commit classifications
        showing how work type distribution changes over time, providing granular insights
        into development patterns and workload shifts.

        DESIGN DECISION: Only show weeks that contain actual commit activity within the
        analysis period. This prevents phantom "No activity" weeks for periods outside
        the actual data collection range, providing more accurate and meaningful reports.

        Args:
            commits: List of all commits with timestamps and classifications
            developer_id: Optional canonical developer ID to filter by
            project_key: Optional project key to filter by
            weeks: Total analysis period in weeks
            analysis_start_date: Analysis period start (from CLI)
            analysis_end_date: Analysis period end (from CLI)

        Returns:
            List of weekly data dictionaries:
            [
                {
                    'week_start': datetime,
                    'week_display': 'Jul 7-13',
                    'classifications': {'Features': 45.0, 'Bug Fixes': 30.0, 'Maintenance': 25.0},
                    'changes': {'Features': 5.0, 'Bug Fixes': -5.0, 'Maintenance': 0.0},
                    'has_activity': True
                },
                ...
            ]
        """
        if not commits or weeks < 1:
            return []

        # Filter commits by developer or project if specified
        filtered_commits = []
        for commit in commits:
            if developer_id and commit.get("canonical_id") != developer_id:
                continue
            if project_key and commit.get("project_key") != project_key:
                continue
            filtered_commits.append(commit)

        # If no commits match the filter, return empty
        if not filtered_commits:
            return []

        # Determine the analysis period bounds
        if analysis_start_date and analysis_end_date:
            # Use the exact analysis period from the CLI
            analysis_start = analysis_start_date
            analysis_end = analysis_end_date
        else:
            # Fallback: Use the actual date range of the filtered commits
            # This ensures we only show weeks that have potential for activity
            filtered_timestamps = []
            for commit in filtered_commits:
                timestamp = commit.get("timestamp")
                if timestamp:
                    # Ensure timezone consistency
                    if hasattr(timestamp, "tzinfo"):
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                        elif timestamp.tzinfo != timezone.utc:
                            timestamp = timestamp.astimezone(timezone.utc)
                    filtered_timestamps.append(timestamp)

            if not filtered_timestamps:
                return []

            # Use the actual range of commits for this developer/project
            analysis_start = min(filtered_timestamps)
            analysis_end = max(filtered_timestamps)

        # Generate ALL weeks in the analysis period (not just weeks with commits)
        # This ensures complete week coverage from start to end
        # FIX: Only include complete weeks (Monday-Sunday) within the analysis period
        analysis_weeks = []
        current_week_start = self._get_week_start(analysis_start)

        # Bug 7 fix: the previous code computed week_end with microsecond=0 (via
        # timedelta arithmetic on a datetime that already had microsecond=0 from
        # _get_week_start), while analysis_end from get_week_end() has
        # microsecond=999999.  The strict `<=` comparison therefore excluded the
        # last week of every analysis period because:
        #   week_end  = Sunday 23:59:59.000000
        #   analysis_end = Sunday 23:59:59.999999
        #   week_end <= analysis_end  → True  (ok for last week)
        # Wait — actually the issue is the OPPOSITE: week_end has second=59 but
        # no microseconds, so it IS less than analysis_end.  The real failure was
        # that timedelta(days=6, hours=23, minutes=59, seconds=59) lands at
        # 23:59:59.000000 on Sunday while analysis_end is 23:59:59.999999, so the
        # comparison does include the last week... BUT when analysis_end is an
        # exact Monday 00:00:00 (from the fallback path that uses commit timestamps)
        # week_end (Sunday 23:59:59) < analysis_end (Monday 00:00:00) fails for
        # the week that *contains* analysis_end.
        #
        # Fix: compare against the *start* of the next day instead.  A week is
        # included if its Sunday falls before the start of the day after
        # analysis_end.  This is unambiguous regardless of the microsecond value
        # in analysis_end.
        next_day_after_end = analysis_end.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)

        while current_week_start <= analysis_end:
            # Sunday of the current week (start of the next week, exclusive)
            next_week_start = current_week_start + timedelta(weeks=1)
            # Include this week if its entire span ends before next_day_after_end,
            # i.e. the Sunday of this week is strictly before the day after analysis_end.
            if next_week_start <= next_day_after_end:
                analysis_weeks.append(current_week_start)
            current_week_start += timedelta(weeks=1)

        # Group commits by week
        weekly_commits = {}
        for week_start in analysis_weeks:
            weekly_commits[week_start] = []

        for commit in filtered_commits:
            timestamp = commit.get("timestamp")
            if not timestamp:
                continue

            # Ensure timezone consistency
            if hasattr(timestamp, "tzinfo"):
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                elif timestamp.tzinfo != timezone.utc:
                    timestamp = timestamp.astimezone(timezone.utc)

            # Only include commits within the analysis period bounds
            if (
                analysis_start_date
                and analysis_end_date
                and not (analysis_start <= timestamp <= analysis_end)
            ):
                continue

            # Get week start (Monday) for this commit
            commit_week_start = self._get_week_start(timestamp)

            # Only include commits in weeks we're tracking
            if commit_week_start in weekly_commits:
                weekly_commits[commit_week_start].append(commit)

        # Import classifiers
        try:
            from ..extractors.ml_tickets import MLTicketExtractor

            extractor = MLTicketExtractor(enable_ml=True)
        except Exception:
            from ..extractors.tickets import TicketExtractor

            extractor = TicketExtractor()

        # Calculate classifications for each week in the analysis period
        # This includes both weeks with activity and weeks with no commits
        weekly_data = []
        previous_percentages = {}

        for week_start in analysis_weeks:
            week_commits = weekly_commits[week_start]
            has_activity = len(week_commits) > 0

            # Classify commits for this week
            week_classifications = {}
            week_percentages = {}

            if has_activity:
                for commit in week_commits:
                    message = commit.get("message", "")
                    files_changed = commit.get("files_changed", [])
                    if isinstance(files_changed, int) or not isinstance(files_changed, list):
                        files_changed = []

                    ticket_refs = commit.get("ticket_references", [])

                    if ticket_refs and hasattr(extractor, "categorize_commit_with_confidence"):
                        try:
                            result = extractor.categorize_commit_with_confidence(
                                message, files_changed
                            )
                            category = result["category"]
                            category = self._enhance_category_with_ticket_info(
                                category, ticket_refs, message
                            )
                        except Exception:
                            category = extractor.categorize_commit(message)
                    else:
                        category = extractor.categorize_commit(message)

                    if category not in week_classifications:
                        week_classifications[category] = 0
                    week_classifications[category] += 1

                # Calculate percentages for weeks with activity
                total_commits = sum(week_classifications.values())
                if total_commits > 0:
                    for category, count in week_classifications.items():
                        percentage = (count / total_commits) * 100
                        if percentage >= 5.0:  # Only include significant categories
                            display_name = self._format_category_name(category)
                            week_percentages[display_name] = percentage

            # Calculate changes from previous week
            changes = {}
            if previous_percentages and week_percentages:
                for category in set(week_percentages.keys()) | set(previous_percentages.keys()):
                    current_pct = week_percentages.get(category, 0.0)
                    prev_pct = previous_percentages.get(category, 0.0)
                    change = current_pct - prev_pct
                    if abs(change) >= 1.0:  # Only show changes >= 1%
                        changes[category] = change

            # Format week display
            week_end = week_start + timedelta(days=6)
            week_display = f"{week_start.strftime('%b %d')}-{week_end.strftime('%d')}"

            # Calculate ticket coverage stats for this week
            total_commits_week = len(week_commits)
            commits_with_tickets = sum(
                1 for commit in week_commits if commit.get("ticket_references")
            )
            ticket_coverage_pct = (
                (commits_with_tickets / total_commits_week * 100) if total_commits_week > 0 else 0
            )

            # Calculate activity score for this week
            week_activity_score = 0.0
            if total_commits_week > 0:
                # Aggregate weekly metrics for activity score
                total_lines_added = sum(commit.get("lines_added", 0) for commit in week_commits)
                total_lines_deleted = sum(commit.get("lines_deleted", 0) for commit in week_commits)
                total_files_changed = sum(
                    commit.get("files_changed_count", 0) for commit in week_commits
                )

                week_metrics = {
                    "commits": total_commits_week,
                    "prs_involved": 0,  # PR data not available in commit data
                    "lines_added": total_lines_added,
                    "lines_removed": total_lines_deleted,
                    "files_changed_count": total_files_changed,
                    "complexity_delta": 0,  # Complexity data not available
                }

                activity_result = self.activity_scorer.calculate_activity_score(week_metrics)
                week_activity_score = activity_result.get("normalized_score", 0.0)

            weekly_data.append(
                {
                    "week_start": week_start,
                    "week_display": week_display,
                    "classifications": week_percentages,
                    "classification_counts": week_classifications,  # Absolute counts
                    "changes": changes,
                    "has_activity": has_activity,
                    "total_commits": total_commits_week,
                    "commits_with_tickets": commits_with_tickets,
                    "ticket_coverage": ticket_coverage_pct,
                    "activity_score": week_activity_score,
                }
            )

            # Update previous percentages only if there was activity
            if has_activity and week_percentages:
                previous_percentages = week_percentages.copy()

        return weekly_data

    def _calculate_classification_trends(
        self,
        commits: list[dict[str, Any]],
        developer_id: str = None,
        project_key: str = None,
        weeks: int = 4,
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
            if developer_id and commit.get("canonical_id") != developer_id:
                continue
            if project_key and commit.get("project_key") != project_key:
                continue
            filtered_commits.append(commit)

        if len(filtered_commits) < 2:
            return {}

        # Sort commits by timestamp
        def safe_timestamp_key(commit):
            ts = commit.get("timestamp")
            if ts is None:
                return datetime.min.replace(tzinfo=timezone.utc)
            if hasattr(ts, "tzinfo"):
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
                message = commit.get("message", "")
                files_changed = commit.get("files_changed", [])
                if isinstance(files_changed, int) or not isinstance(files_changed, list):
                    files_changed = []

                # Get ticket info for enhancement
                ticket_refs = commit.get("ticket_references", [])

                if ticket_refs and hasattr(extractor, "categorize_commit_with_confidence"):
                    try:
                        result = extractor.categorize_commit_with_confidence(message, files_changed)
                        category = result["category"]
                        category = self._enhance_category_with_ticket_info(
                            category, ticket_refs, message
                        )
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
        self, report: StringIO, weekly_trends: list[dict[str, Any]], prefix: str = ""
    ) -> None:
        """Write weekly trend lines showing week-by-week classification changes.

        WHY: This method provides detailed weekly breakdown of work patterns,
        showing how development focus shifts over time with specific percentages
        and change indicators from previous weeks. Shows ALL weeks in the analysis
        period, including weeks with no activity for complete timeline coverage.

        Args:
            report: StringIO buffer to write to
            weekly_trends: List of weekly classification data (all weeks in period)
            prefix: Optional prefix for the trend section (e.g., "Project ")
        """
        if not weekly_trends:
            return

        report.write(f"- {prefix}Weekly Trends:\n")

        for i, week_data in enumerate(weekly_trends):
            week_display = week_data["week_display"]
            classifications = week_data["classifications"]
            changes = week_data["changes"]
            has_activity = week_data.get("has_activity", True)

            # Get additional data from week_data
            classification_counts = week_data.get("classification_counts", {})
            total_commits = week_data.get("total_commits", 0)
            commits_with_tickets = week_data.get("commits_with_tickets", 0)
            ticket_coverage = week_data.get("ticket_coverage", 0)
            activity_score = week_data.get("activity_score", 0.0)

            # Handle weeks with no activity
            if not classifications and not has_activity:
                report.write(f"  - Week {i + 1} ({week_display}): No activity\n")
                continue
            elif not classifications:
                # Should not happen, but handle gracefully
                continue

            # Format classifications with absolute numbers and percentages
            classification_parts = []
            for category in sorted(classifications.keys()):
                percentage = classifications[category]

                # Find the count for this formatted category name by reverse mapping
                count = 0
                for raw_category, raw_count in classification_counts.items():
                    if self._format_category_name(raw_category) == category:
                        count = raw_count
                        break

                change = changes.get(category, 0.0)

                if i == 0 or abs(change) < 1.0:
                    # First week or no significant change - show count and percentage
                    classification_parts.append(f"{category} {count} ({percentage:.0f}%)")
                else:
                    # Show change from previous week
                    change_indicator = f"(+{change:.0f}%)" if change > 0 else f"({change:.0f}%)"
                    classification_parts.append(
                        f"{category} {count} ({percentage:.0f}% {change_indicator})"
                    )

            if classification_parts:
                classifications_text = ", ".join(classification_parts)
                # Add total commits, ticket coverage, and activity score to the week summary
                if total_commits > 0:
                    ticket_info = (
                        f" | {commits_with_tickets}/{total_commits} tickets ({ticket_coverage:.0f}%)"
                        if commits_with_tickets > 0
                        else f" | 0/{total_commits} tickets (0%)"
                    )
                    activity_info = f" | Activity: {activity_score:.1f}/100"
                    report.write(
                        f"  - Week {i + 1} ({week_display}): {classifications_text}{ticket_info}{activity_info}\n"
                    )
                else:
                    report.write(f"  - Week {i + 1} ({week_display}): {classifications_text}\n")
            else:
                # Fallback in case classifications exist but are empty
                report.write(f"  - Week {i + 1} ({week_display}): No significant activity\n")

        # Add a blank line after trend lines for spacing
        # (Note: Don't add extra newline here as the caller will handle spacing)

