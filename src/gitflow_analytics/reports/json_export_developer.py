"""JSON exporter mixin: developer analysis (projects, contributions, patterns, metrics, timeline).

Extracted from json_exporter.py to keep file sizes manageable.
"""

import json
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class JSONExportDeveloperMixin:
    """Mixin providing developer analysis (projects, contributions, patterns, metrics, timeline) for ComprehensiveJSONExporter."""

    def _get_developer_projects(self, commits: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Get projects a developer has worked on with contribution details."""

        project_contributions = defaultdict(
            lambda: {
                "commits": 0,
                "lines_changed": 0,
                "story_points": 0,
                "first_commit": None,
                "last_commit": None,
            }
        )

        for commit in commits:
            project_key = commit.get("project_key", "UNKNOWN")
            project_data = project_contributions[project_key]

            project_data["commits"] += 1

            lines = commit.get("filtered_insertions", commit.get("insertions", 0)) + commit.get(
                "filtered_deletions", commit.get("deletions", 0)
            )
            project_data["lines_changed"] += lines
            project_data["story_points"] += commit.get("story_points", 0) or 0

            # Track first and last commits
            commit_date = commit["timestamp"]
            if not project_data["first_commit"] or commit_date < project_data["first_commit"]:
                project_data["first_commit"] = commit_date
            if not project_data["last_commit"] or commit_date > project_data["last_commit"]:
                project_data["last_commit"] = commit_date

        # Convert to regular dict and add percentages
        total_commits = len(commits)
        projects = {}

        for project_key, data in project_contributions.items():
            projects[project_key] = {
                "commits": data["commits"],
                "commits_percentage": round((data["commits"] / total_commits) * 100, 1),
                "lines_changed": data["lines_changed"],
                "story_points": data["story_points"],
                "first_commit": data["first_commit"].isoformat() if data["first_commit"] else None,
                "last_commit": data["last_commit"].isoformat() if data["last_commit"] else None,
                "days_active": (data["last_commit"] - data["first_commit"]).days
                if data["first_commit"] and data["last_commit"]
                else 0,
            }

        return projects

    def _analyze_developer_contribution_patterns(
        self, commits: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze a developer's contribution patterns."""

        if not commits:
            return {}

        # Time-based patterns (use local hour if available)
        commit_hours = []
        for c in commits:
            if "local_hour" in c:
                commit_hours.append(c["local_hour"])
            elif hasattr(c["timestamp"], "hour"):
                commit_hours.append(c["timestamp"].hour)

        commit_days = [
            c["timestamp"].weekday() for c in commits if hasattr(c["timestamp"], "weekday")
        ]

        # Size patterns
        commit_sizes = []
        for commit in commits:
            lines = commit.get("filtered_insertions", commit.get("insertions", 0)) + commit.get(
                "filtered_deletions", commit.get("deletions", 0)
            )
            commit_sizes.append(lines)

        patterns = {
            "total_commits": len(commits),
            "avg_commit_size": round(statistics.mean(commit_sizes), 1) if commit_sizes else 0,
            "commit_size_stddev": round(statistics.pstdev(commit_sizes), 1)
            if len(commit_sizes) > 1
            else 0,
        }

        if commit_hours:
            patterns["peak_hour"] = max(set(commit_hours), key=commit_hours.count)
            patterns["time_distribution"] = self._get_time_distribution_pattern(commit_hours)

        if commit_days:
            patterns["peak_day"] = self._get_day_name(max(set(commit_days), key=commit_days.count))
            patterns["work_pattern"] = self._get_work_pattern(commit_days)

        # Consistency patterns
        weekly_commits = self._get_weekly_commit_counts(commits)
        if len(weekly_commits) > 1:
            patterns["consistency_score"] = round(
                100
                - (
                    statistics.pstdev(weekly_commits)
                    / max(statistics.mean(weekly_commits), 1)
                    * 100
                ),
                1,
            )
        else:
            patterns["consistency_score"] = 50

        return patterns

    def _get_time_distribution_pattern(self, hours: List[int]) -> str:
        """Determine time distribution pattern from commit hours."""
        avg_hour = statistics.mean(hours)

        if avg_hour < 10:
            return "early_bird"
        elif avg_hour < 14:
            return "morning_focused"
        elif avg_hour < 18:
            return "afternoon_focused"
        else:
            return "night_owl"

    def _get_day_name(self, day_index: int) -> str:
        """Convert day index to day name."""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return days[day_index] if 0 <= day_index < 7 else "Unknown"

    def _get_work_pattern(self, days: List[int]) -> str:
        """Determine work pattern from commit days."""
        weekday_commits = sum(1 for day in days if day < 5)  # Mon-Fri
        weekend_commits = sum(1 for day in days if day >= 5)  # Sat-Sun

        total = len(days)
        weekday_pct = (weekday_commits / total) * 100 if total > 0 else 0

        if weekday_pct > 90:
            return "strictly_weekdays"
        elif weekday_pct > 75:
            return "mostly_weekdays"
        elif weekday_pct > 50:
            return "mixed_schedule"
        else:
            return "weekend_warrior"

    def _calculate_developer_collaboration_metrics(
        self, commits: List[Dict[str, Any]], all_developer_stats: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate collaboration metrics for a developer."""

        # Get projects this developer worked on
        dev_projects = set(c.get("project_key", "UNKNOWN") for c in commits)

        # Find other developers on same projects
        collaborators = set()
        for dev in all_developer_stats:
            dev_id = dev["canonical_id"]
            # Simple check - assumes we can identify overlapping work
            # In real implementation, would need more sophisticated analysis
            if len(dev_projects) > 0:  # Placeholder logic
                collaborators.add(dev_id)

        # Remove self from collaborators
        dev_id = commits[0].get("canonical_id") if commits else None
        collaborators.discard(dev_id)

        return {
            "projects_count": len(dev_projects),
            "potential_collaborators": len(collaborators),
            "cross_project_work": len(dev_projects) > 1,
            "collaboration_score": min(100, len(collaborators) * 10),  # Simple scoring
        }

    def _calculate_developer_health_score(
        self, commits: List[Dict[str, Any]], dev_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate health score for a specific developer."""

        if not commits:
            return {"overall": 0, "components": {}, "rating": "no_data"}

        scores = {}

        # Activity score based on commits per week
        weekly_commits = self._get_weekly_commit_counts(commits)
        if weekly_commits:
            avg_weekly = statistics.mean(weekly_commits)
            activity_score = min(100, avg_weekly * 20)  # Scale appropriately
            scores["activity"] = activity_score
        else:
            scores["activity"] = 0

        # Consistency score
        if len(weekly_commits) > 1:
            consistency = max(
                0,
                100
                - (
                    statistics.pstdev(weekly_commits) / max(statistics.mean(weekly_commits), 1) * 50
                ),
            )
            scores["consistency"] = consistency
        else:
            scores["consistency"] = 50

        # Engagement score (based on projects and commit sizes)
        project_count = len(set(c.get("project_key", "UNKNOWN") for c in commits))
        engagement_score = min(100, project_count * 25 + 25)  # Bonus for multi-project work
        scores["engagement"] = engagement_score

        # Overall score
        overall_score = sum(scores.values()) / len(scores)

        return {
            "overall": round(overall_score, 1),
            "components": {k: round(v, 1) for k, v in scores.items()},
            "rating": self._get_health_rating(overall_score),
        }

    def _identify_developer_achievements(
        self, commits: List[Dict[str, Any]], dev_stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify achievements for a developer."""

        achievements = []

        # High commit count
        if dev_stats["total_commits"] > 50:
            achievements.append(
                {
                    "type": "productivity",
                    "title": "High Productivity",
                    "description": f"{dev_stats['total_commits']} commits in analysis period",
                    "badge": "prolific_contributor",
                }
            )

        # Multi-project contributor
        projects = set(c.get("project_key", "UNKNOWN") for c in commits)
        if len(projects) > 3:
            achievements.append(
                {
                    "type": "versatility",
                    "title": "Multi-Project Contributor",
                    "description": f"Contributed to {len(projects)} projects",
                    "badge": "versatile_developer",
                }
            )

        # Consistent contributor
        weekly_commits = self._get_weekly_commit_counts(commits)
        if len(weekly_commits) > 4:
            active_weeks = sum(1 for w in weekly_commits if w > 0)
            consistency_rate = active_weeks / len(weekly_commits)

            if consistency_rate > 0.8:
                achievements.append(
                    {
                        "type": "consistency",
                        "title": "Consistent Contributor",
                        "description": f"Active in {active_weeks} out of {len(weekly_commits)} weeks",
                        "badge": "reliable_contributor",
                    }
                )

        return achievements

    def _identify_improvement_areas(
        self, commits: List[Dict[str, Any]], dev_stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify areas for improvement for a developer."""

        improvements = []

        # Check ticket linking
        commits_with_tickets = sum(1 for c in commits if c.get("ticket_references"))
        ticket_rate = (commits_with_tickets / len(commits)) * 100 if commits else 0

        if ticket_rate < 50:
            improvements.append(
                {
                    "category": "process",
                    "title": "Improve Ticket Linking",
                    "description": f"Only {ticket_rate:.1f}% of commits reference tickets",
                    "priority": "medium",
                    "suggestion": "Include ticket references in commit messages",
                }
            )

        # Check commit size consistency
        commit_sizes = []
        for commit in commits:
            lines = commit.get("filtered_insertions", commit.get("insertions", 0)) + commit.get(
                "filtered_deletions", commit.get("deletions", 0)
            )
            commit_sizes.append(lines)

        if commit_sizes and len(commit_sizes) > 5:
            avg_size = statistics.mean(commit_sizes)
            if avg_size > 300:
                improvements.append(
                    {
                        "category": "quality",
                        "title": "Consider Smaller Commits",
                        "description": f"Average commit size is {avg_size:.0f} lines",
                        "priority": "low",
                        "suggestion": "Break down large changes into smaller, focused commits",
                    }
                )

        return improvements

    def _build_developer_activity_timeline(
        self, commits: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build activity timeline for a developer."""

        if not commits:
            return []

        # Group commits by week
        weekly_activity = defaultdict(lambda: {"commits": 0, "lines_changed": 0, "projects": set()})

        for commit in commits:
            week_start = self._get_week_start(commit["timestamp"])
            week_key = week_start.strftime("%Y-%m-%d")

            weekly_activity[week_key]["commits"] += 1

            lines = commit.get("filtered_insertions", commit.get("insertions", 0)) + commit.get(
                "filtered_deletions", commit.get("deletions", 0)
            )
            weekly_activity[week_key]["lines_changed"] += lines
            weekly_activity[week_key]["projects"].add(commit.get("project_key", "UNKNOWN"))

        # Convert to timeline format
        timeline = []
        for week_key in sorted(weekly_activity.keys()):
            data = weekly_activity[week_key]
            timeline.append(
                {
                    "week": week_key,
                    "commits": data["commits"],
                    "lines_changed": data["lines_changed"],
                    "projects": len(data["projects"]),
                    "project_list": sorted(list(data["projects"])),
                }
            )

        return timeline

    def _analyze_branching_patterns(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze branching and merge patterns."""

        merge_commits = sum(1 for c in commits if c.get("is_merge"))
        total_commits = len(commits)

        merge_rate = (merge_commits / total_commits) * 100 if total_commits > 0 else 0

        # Determine branching strategy
        if merge_rate < 5:
            strategy = "linear"
        elif merge_rate < 15:
            strategy = "feature_branches"
        elif merge_rate < 30:
            strategy = "git_flow"
        else:
            strategy = "complex_branching"

        return {
            "merge_commits": merge_commits,
            "merge_rate_percent": round(merge_rate, 1),
            "strategy": strategy,
            "complexity_rating": "low"
            if merge_rate < 15
            else "medium"
            if merge_rate < 30
            else "high",
        }

    def _analyze_commit_timing_patterns(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze when commits typically happen."""

        if not commits:
            return {}

        # Extract timing data
        hours = []
        days = []

        for commit in commits:
            timestamp = commit["timestamp"]
            # Use local hour if available
            if "local_hour" in commit:
                hours.append(commit["local_hour"])
            elif hasattr(timestamp, "hour"):
                hours.append(timestamp.hour)
            if hasattr(timestamp, "weekday"):
                days.append(timestamp.weekday())

        patterns = {}

        if hours:
            # Hour distribution
            hour_counts = defaultdict(int)
            for hour in hours:
                hour_counts[hour] += 1

            peak_hour = max(hour_counts, key=hour_counts.get)
            patterns["peak_hour"] = peak_hour
            patterns["peak_hour_commits"] = hour_counts[peak_hour]

            # Time periods
            morning = sum(1 for h in hours if 6 <= h < 12)
            afternoon = sum(1 for h in hours if 12 <= h < 18)
            evening = sum(1 for h in hours if 18 <= h < 24)
            night = sum(1 for h in hours if 0 <= h < 6)

            total = len(hours)
            patterns["time_distribution"] = {
                "morning_pct": round((morning / total) * 100, 1),
                "afternoon_pct": round((afternoon / total) * 100, 1),
                "evening_pct": round((evening / total) * 100, 1),
                "night_pct": round((night / total) * 100, 1),
            }

        if days:
            # Day distribution
            day_counts = defaultdict(int)
            for day in days:
                day_counts[day] += 1

            peak_day = max(day_counts, key=day_counts.get)
            patterns["peak_day"] = self._get_day_name(peak_day)
            patterns["peak_day_commits"] = day_counts[peak_day]

            # Weekday vs weekend
            weekday_commits = sum(1 for d in days if d < 5)
            weekend_commits = sum(1 for d in days if d >= 5)

            total = len(days)
            patterns["weekday_pct"] = round((weekday_commits / total) * 100, 1)
            patterns["weekend_pct"] = round((weekend_commits / total) * 100, 1)

        return patterns

    def _analyze_pr_workflow(self, prs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze pull request workflow patterns."""

        if not prs:
            return {}

        # PR lifecycle analysis
        lifetimes = []
        sizes = []
        review_counts = []

        for pr in prs:
            # Calculate PR lifetime
            created = pr.get("created_at")
            merged = pr.get("merged_at")

            if created and merged:
                if isinstance(created, str):
                    created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if isinstance(merged, str):
                    merged = datetime.fromisoformat(merged.replace("Z", "+00:00"))

                lifetime_hours = (merged - created).total_seconds() / 3600
                lifetimes.append(lifetime_hours)

            # PR size (additions + deletions)
            additions = pr.get("additions", 0)
            deletions = pr.get("deletions", 0)
            sizes.append(additions + deletions)

            # Review comments
            review_comments = pr.get("review_comments", 0)
            review_counts.append(review_comments)

        workflow = {}

        if lifetimes:
            workflow["avg_lifetime_hours"] = round(statistics.mean(lifetimes), 1)
            workflow["median_lifetime_hours"] = round(statistics.median(lifetimes), 1)

        if sizes:
            workflow["avg_pr_size"] = round(statistics.mean(sizes), 1)
            workflow["median_pr_size"] = round(statistics.median(sizes), 1)

        if review_counts:
            workflow["avg_review_comments"] = round(statistics.mean(review_counts), 1)
            workflow["prs_with_reviews"] = sum(1 for r in review_counts if r > 0)
            workflow["review_rate_pct"] = round((workflow["prs_with_reviews"] / len(prs)) * 100, 1)

        return workflow

    def _analyze_git_pm_correlation(
        self, commits: List[Dict[str, Any]], pm_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze correlation between Git activity and PM platform data."""

        correlations = pm_data.get("correlations", [])
        metrics = pm_data.get("metrics", {})

        if not correlations:
            return {"status": "no_correlations"}

        # Analyze correlation quality
        high_confidence = sum(1 for c in correlations if c.get("confidence", 0) > 0.8)
        medium_confidence = sum(1 for c in correlations if 0.5 <= c.get("confidence", 0) <= 0.8)
        low_confidence = sum(1 for c in correlations if c.get("confidence", 0) < 0.5)

        total_correlations = len(correlations)

        # Analyze correlation methods
        methods = defaultdict(int)
        for c in correlations:
            method = c.get("correlation_method", "unknown")
            methods[method] += 1

        # Story point accuracy analysis
        story_analysis = metrics.get("story_point_analysis", {})

        return {
            "total_correlations": total_correlations,
            "confidence_distribution": {
                "high": high_confidence,
                "medium": medium_confidence,
                "low": low_confidence,
            },
            "confidence_rates": {
                "high_pct": round((high_confidence / total_correlations) * 100, 1),
                "medium_pct": round((medium_confidence / total_correlations) * 100, 1),
                "low_pct": round((low_confidence / total_correlations) * 100, 1),
            },
            "correlation_methods": dict(methods),
            "story_point_analysis": story_analysis,
            "platforms": list(metrics.get("platform_coverage", {}).keys()),
        }

    def _calculate_merge_commit_rate(self, commits: List[Dict[str, Any]]) -> float:
        """Calculate percentage of merge commits."""
        if not commits:
            return 0

        merge_commits = sum(1 for c in commits if c.get("is_merge"))
        return round((merge_commits / len(commits)) * 100, 1)

    def _analyze_commit_message_quality(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze commit message quality patterns."""

        if not commits:
            return {}

        message_lengths = []
        has_ticket_ref = 0
        conventional_commits = 0

        # Conventional commit prefixes
        conventional_prefixes = ["feat:", "fix:", "docs:", "style:", "refactor:", "test:", "chore:"]

        for commit in commits:
            message = commit.get("message", "")

            # Message length (in words)
            word_count = len(message.split())
            message_lengths.append(word_count)

            # Ticket reference check
            if commit.get("ticket_references"):
                has_ticket_ref += 1

            # Conventional commit check
            if any(message.lower().startswith(prefix) for prefix in conventional_prefixes):
                conventional_commits += 1

        total_commits = len(commits)

        quality = {}

        if message_lengths:
            quality["avg_message_length_words"] = round(statistics.mean(message_lengths), 1)
            quality["median_message_length_words"] = round(statistics.median(message_lengths), 1)

        quality["ticket_reference_rate_pct"] = round((has_ticket_ref / total_commits) * 100, 1)
        quality["conventional_commit_rate_pct"] = round(
            (conventional_commits / total_commits) * 100, 1
        )

        # Quality rating
        score = 0
        if quality.get("avg_message_length_words", 0) >= 5:
            score += 25
        if quality.get("ticket_reference_rate_pct", 0) >= 50:
            score += 35
        if quality.get("conventional_commit_rate_pct", 0) >= 30:
            score += 40

        if score >= 80:
            quality["overall_rating"] = "excellent"
        elif score >= 60:
            quality["overall_rating"] = "good"
        elif score >= 40:
            quality["overall_rating"] = "fair"
        else:
            quality["overall_rating"] = "needs_improvement"

        return quality

