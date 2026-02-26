"""JSON exporter mixin: time series generation, insights, and untracked commit analysis.

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


class JSONExportInsightsMixin:
    """Mixin providing time series generation, insights, and untracked commit analysis for ComprehensiveJSONExporter."""

    def _generate_weekly_time_series(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Generate weekly time series data for charts."""

        weekly_data = []
        current_date = start_date

        while current_date <= end_date:
            week_end = current_date + timedelta(days=7)

            # Filter commits for this week
            week_commits = []
            for c in commits:
                # Ensure both timestamps are timezone-aware for comparison
                commit_ts = c["timestamp"]
                if hasattr(commit_ts, "tzinfo") and commit_ts.tzinfo is None:
                    # Make timezone-aware if needed
                    commit_ts = commit_ts.replace(tzinfo=timezone.utc)
                elif not hasattr(commit_ts, "tzinfo"):
                    # Convert to datetime if needed
                    commit_ts = datetime.fromisoformat(str(commit_ts))
                    if commit_ts.tzinfo is None:
                        commit_ts = commit_ts.replace(tzinfo=timezone.utc)

                if current_date <= commit_ts < week_end:
                    week_commits.append(c)

            # Filter PRs for this week (by merge date)
            week_prs = []
            for pr in prs:
                merged_at = pr.get("merged_at")
                if merged_at:
                    if isinstance(merged_at, str):
                        merged_at = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
                    # Ensure timezone-aware for comparison
                    if hasattr(merged_at, "tzinfo") and merged_at.tzinfo is None:
                        merged_at = merged_at.replace(tzinfo=timezone.utc)
                    if current_date <= merged_at < week_end:
                        week_prs.append(pr)

            # Calculate metrics
            lines_changed = sum(
                c.get("filtered_insertions", c.get("insertions", 0))
                + c.get("filtered_deletions", c.get("deletions", 0))
                for c in week_commits
            )

            story_points = sum(c.get("story_points", 0) or 0 for c in week_commits)

            active_developers = len(
                set(c.get("canonical_id", c.get("author_email")) for c in week_commits)
            )

            weekly_data.append(
                {
                    "date": current_date.strftime("%Y-%m-%d"),
                    "commits": len(week_commits),
                    "lines_changed": lines_changed,
                    "story_points": story_points,
                    "active_developers": active_developers,
                    "pull_requests": len(week_prs),
                }
            )

            current_date = week_end

        return weekly_data

    def _generate_daily_time_series(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Generate daily time series data for detailed analysis."""

        daily_data = []
        current_date = start_date

        while current_date <= end_date:
            day_end = current_date + timedelta(days=1)

            # Filter commits for this day
            day_commits = []
            for c in commits:
                # Ensure both timestamps are timezone-aware for comparison
                commit_ts = c["timestamp"]
                if hasattr(commit_ts, "tzinfo") and commit_ts.tzinfo is None:
                    # Make timezone-aware if needed
                    commit_ts = commit_ts.replace(tzinfo=timezone.utc)
                elif not hasattr(commit_ts, "tzinfo"):
                    # Convert to datetime if needed
                    commit_ts = datetime.fromisoformat(str(commit_ts))
                    if commit_ts.tzinfo is None:
                        commit_ts = commit_ts.replace(tzinfo=timezone.utc)

                if current_date <= commit_ts < day_end:
                    day_commits.append(c)

            daily_data.append(
                {"date": current_date.strftime("%Y-%m-%d"), "commits": len(day_commits)}
            )

            current_date = day_end

        return daily_data

    def _generate_quantitative_insights(
        self, commits: List[Dict[str, Any]], developer_stats: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate quantitative insights from data analysis."""

        insights = []

        # Team productivity insights
        total_commits = len(commits)
        if total_commits > 0:
            weekly_commits = self._get_weekly_commit_counts(commits)
            if weekly_commits:
                avg_weekly = statistics.mean(weekly_commits)
                insights.append(
                    {
                        "category": "productivity",
                        "type": "metric",
                        "title": "Weekly Commit Rate",
                        "description": f"Team averages {avg_weekly:.1f} commits per week",
                        "value": avg_weekly,
                        "trend": self._calculate_simple_trend(weekly_commits),
                        "priority": "medium",
                    }
                )

        # Developer distribution insights
        if len(developer_stats) > 1:
            commit_counts = [dev["total_commits"] for dev in developer_stats]
            gini = self._calculate_gini_coefficient(commit_counts)

            if gini > 0.7:
                insights.append(
                    {
                        "category": "team",
                        "type": "concern",
                        "title": "Unbalanced Contributions",
                        "description": f"Work is concentrated among few developers (Gini: {gini:.2f})",
                        "value": gini,
                        "priority": "high",
                        "recommendation": "Consider distributing work more evenly",
                    }
                )
            elif gini < 0.3:
                insights.append(
                    {
                        "category": "team",
                        "type": "positive",
                        "title": "Balanced Team Contributions",
                        "description": f"Work is well-distributed across the team (Gini: {gini:.2f})",
                        "value": gini,
                        "priority": "low",
                    }
                )

        # Code quality insights
        commit_sizes = []
        for commit in commits:
            lines = commit.get("filtered_insertions", commit.get("insertions", 0)) + commit.get(
                "filtered_deletions", commit.get("deletions", 0)
            )
            commit_sizes.append(lines)

        if commit_sizes:
            avg_size = statistics.mean(commit_sizes)
            if avg_size > 300:
                insights.append(
                    {
                        "category": "quality",
                        "type": "concern",
                        "title": "Large Commit Sizes",
                        "description": f"Average commit size is {avg_size:.0f} lines",
                        "value": avg_size,
                        "priority": "medium",
                        "recommendation": "Consider breaking down changes into smaller commits",
                    }
                )
            elif 20 <= avg_size <= 200:
                insights.append(
                    {
                        "category": "quality",
                        "type": "positive",
                        "title": "Optimal Commit Sizes",
                        "description": f"Average commit size of {avg_size:.0f} lines indicates good change management",
                        "value": avg_size,
                        "priority": "low",
                    }
                )

        return insights

    def _process_qualitative_insights(
        self, qualitative_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process qualitative analysis results into insights."""

        insights = []

        for item in qualitative_data:
            # Transform qualitative data into insight format
            insight = {
                "category": item.get("category", "general"),
                "type": "qualitative",
                "title": item.get("insight", "Qualitative Insight"),
                "description": item.get("description", ""),
                "priority": item.get("priority", "medium"),
                "confidence": item.get("confidence", 0.5),
            }

            if "recommendation" in item:
                insight["recommendation"] = item["recommendation"]

            insights.append(insight)

        return insights

    def _prioritize_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize insights by importance and impact."""

        def get_priority_score(insight):
            priority_scores = {"high": 3, "medium": 2, "low": 1}
            type_scores = {"concern": 3, "positive": 1, "metric": 2, "qualitative": 2}

            priority_score = priority_scores.get(insight.get("priority", "medium"), 2)
            type_score = type_scores.get(insight.get("type", "metric"), 2)

            return priority_score + type_score

        # Sort by priority score (descending)
        prioritized = sorted(insights, key=get_priority_score, reverse=True)

        return prioritized[:10]  # Return top 10 insights

    def _categorize_insights(
        self, insights: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize insights by category."""

        categories = defaultdict(list)

        for insight in insights:
            category = insight.get("category", "general")
            categories[category].append(insight)

        return dict(categories)

    def _build_untracked_analysis(
        self, commits: List[Dict[str, Any]], project_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build comprehensive untracked commit analysis for JSON export.

        WHY: Untracked work analysis is critical for understanding what development
        activities are happening outside the formal process. This data enables
        process improvements, training identification, and better project visibility.

        Args:
            commits: List of all commits
            project_metrics: Project metrics including ticket analysis

        Returns:
            Dictionary with comprehensive untracked analysis
        """
        ticket_analysis = project_metrics.get("ticket_analysis", {})
        untracked_commits = ticket_analysis.get("untracked_commits", [])

        if not untracked_commits:
            return {
                "summary": {
                    "total_untracked": 0,
                    "untracked_percentage": 0,
                    "analysis_status": "no_untracked_commits",
                },
                "categories": {},
                "contributors": {},
                "projects": {},
                "trends": {},
                "recommendations": [],
            }

        # Initialize analysis structures
        categories = {}
        contributors = {}
        projects = {}
        monthly_trends = {}

        total_commits = ticket_analysis.get("total_commits", len(commits))
        total_untracked = len(untracked_commits)

        # Process each untracked commit
        for commit in untracked_commits:
            # Category analysis
            category = commit.get("category", "other")
            if category not in categories:
                categories[category] = {
                    "count": 0,
                    "lines_changed": 0,
                    "files_changed": 0,
                    "examples": [],
                    "authors": set(),
                }

            categories[category]["count"] += 1
            categories[category]["lines_changed"] += commit.get("lines_changed", 0)
            categories[category]["files_changed"] += commit.get("files_changed", 0)
            categories[category]["authors"].add(
                commit.get("canonical_id", commit.get("author_email", "Unknown"))
            )

            if len(categories[category]["examples"]) < 3:
                categories[category]["examples"].append(
                    {
                        "hash": commit.get("hash", ""),
                        "message": commit.get("message", "")[:200],
                        "author": self._anonymize_value(commit.get("author", "Unknown"), "name"),
                        "timestamp": commit.get("timestamp"),
                        "lines_changed": commit.get("lines_changed", 0),
                        "files_changed": commit.get("files_changed", 0),
                    }
                )

            # Contributor analysis
            author_id = commit.get("canonical_id", commit.get("author_email", "Unknown"))
            author_name = self._anonymize_value(commit.get("author", "Unknown"), "name")

            if author_id not in contributors:
                contributors[author_id] = {
                    "name": author_name,
                    "count": 0,
                    "lines_changed": 0,
                    "categories": set(),
                    "projects": set(),
                    "recent_commits": [],
                }

            contributors[author_id]["count"] += 1
            contributors[author_id]["lines_changed"] += commit.get("lines_changed", 0)
            contributors[author_id]["categories"].add(category)
            contributors[author_id]["projects"].add(commit.get("project_key", "UNKNOWN"))

            if len(contributors[author_id]["recent_commits"]) < 5:
                contributors[author_id]["recent_commits"].append(
                    {
                        "hash": commit.get("hash", ""),
                        "message": commit.get("message", "")[:100],
                        "category": category,
                        "timestamp": commit.get("timestamp"),
                        "lines_changed": commit.get("lines_changed", 0),
                    }
                )

            # Project analysis
            project = commit.get("project_key", "UNKNOWN")
            if project not in projects:
                projects[project] = {
                    "count": 0,
                    "lines_changed": 0,
                    "categories": set(),
                    "contributors": set(),
                    "avg_commit_size": 0,
                }

            projects[project]["count"] += 1
            projects[project]["lines_changed"] += commit.get("lines_changed", 0)
            projects[project]["categories"].add(category)
            projects[project]["contributors"].add(author_id)

            # Monthly trend analysis
            timestamp = commit.get("timestamp")
            if timestamp and hasattr(timestamp, "strftime"):
                month_key = timestamp.strftime("%Y-%m")
                if month_key not in monthly_trends:
                    monthly_trends[month_key] = {
                        "count": 0,
                        "categories": {},
                        "contributors": set(),
                    }
                monthly_trends[month_key]["count"] += 1
                monthly_trends[month_key]["contributors"].add(author_id)

                if category not in monthly_trends[month_key]["categories"]:
                    monthly_trends[month_key]["categories"][category] = 0
                monthly_trends[month_key]["categories"][category] += 1

        # Convert sets to lists and calculate derived metrics
        for category_data in categories.values():
            category_data["authors"] = len(category_data["authors"])
            category_data["avg_lines_per_commit"] = (
                category_data["lines_changed"] / category_data["count"]
                if category_data["count"] > 0
                else 0
            )

        for contributor_data in contributors.values():
            contributor_data["categories"] = list(contributor_data["categories"])
            contributor_data["projects"] = list(contributor_data["projects"])
            contributor_data["avg_lines_per_commit"] = (
                contributor_data["lines_changed"] / contributor_data["count"]
                if contributor_data["count"] > 0
                else 0
            )

        for project_data in projects.values():
            project_data["categories"] = list(project_data["categories"])
            project_data["contributors"] = len(project_data["contributors"])
            project_data["avg_commit_size"] = (
                project_data["lines_changed"] / project_data["count"]
                if project_data["count"] > 0
                else 0
            )

        # Convert sets to counts in trends
        for trend_data in monthly_trends.values():
            trend_data["contributors"] = len(trend_data["contributors"])

        # Generate insights and recommendations
        insights = self._generate_untracked_insights(
            categories, contributors, projects, total_untracked, total_commits
        )
        recommendations = self._generate_untracked_recommendations_json(
            categories, contributors, total_untracked, total_commits
        )

        # Calculate quality scores
        quality_scores = self._calculate_untracked_quality_scores(
            categories, total_untracked, total_commits
        )

        return {
            "summary": {
                "total_untracked": total_untracked,
                "total_commits": total_commits,
                "untracked_percentage": round((total_untracked / total_commits * 100), 2)
                if total_commits > 0
                else 0,
                "avg_lines_per_untracked_commit": round(
                    sum(commit.get("lines_changed", 0) for commit in untracked_commits)
                    / total_untracked,
                    1,
                )
                if total_untracked > 0
                else 0,
                "analysis_status": "complete",
            },
            "categories": categories,
            "contributors": contributors,
            "projects": projects,
            "monthly_trends": monthly_trends,
            "insights": insights,
            "recommendations": recommendations,
            "quality_scores": quality_scores,
        }

    def _generate_untracked_insights(
        self,
        categories: Dict[str, Any],
        contributors: Dict[str, Any],
        projects: Dict[str, Any],
        total_untracked: int,
        total_commits: int,
    ) -> List[Dict[str, Any]]:
        """Generate insights from untracked commit analysis."""
        insights = []

        # Category insights
        if categories:
            top_category = max(categories.items(), key=lambda x: x[1]["count"])
            category_name, category_data = top_category
            category_pct = category_data["count"] / total_untracked * 100

            if category_name in ["feature", "bug_fix"]:
                insights.append(
                    {
                        "type": "concern",
                        "category": "process",
                        "title": f'High {category_name.replace("_", " ").title()} Untracked Rate',
                        "description": f'{category_pct:.1f}% of untracked work is {category_name.replace("_", " ")} development',
                        "impact": "high",
                        "value": category_pct,
                    }
                )
            elif category_name in ["maintenance", "style", "documentation"]:
                insights.append(
                    {
                        "type": "positive",
                        "category": "process",
                        "title": "Appropriate Untracked Work",
                        "description": f"{category_pct:.1f}% of untracked work is {category_name} - this is acceptable",
                        "impact": "low",
                        "value": category_pct,
                    }
                )

        # Contributor concentration insights
        if len(contributors) > 1:
            contributor_counts = [data["count"] for data in contributors.values()]
            max_contributor_count = max(contributor_counts)
            contributor_concentration = max_contributor_count / total_untracked * 100

            if contributor_concentration > 50:
                insights.append(
                    {
                        "type": "concern",
                        "category": "team",
                        "title": "Concentrated Untracked Work",
                        "description": f"One developer accounts for {contributor_concentration:.1f}% of untracked commits",
                        "impact": "medium",
                        "value": contributor_concentration,
                    }
                )

        # Overall coverage insight
        untracked_pct = (total_untracked / total_commits * 100) if total_commits > 0 else 0
        if untracked_pct > 40:
            insights.append(
                {
                    "type": "concern",
                    "category": "coverage",
                    "title": "High Untracked Rate",
                    "description": f"{untracked_pct:.1f}% of all commits lack ticket references",
                    "impact": "high",
                    "value": untracked_pct,
                }
            )
        elif untracked_pct < 15:
            insights.append(
                {
                    "type": "positive",
                    "category": "coverage",
                    "title": "Excellent Tracking Coverage",
                    "description": f"Only {untracked_pct:.1f}% of commits are untracked",
                    "impact": "low",
                    "value": untracked_pct,
                }
            )

        return insights

    def _generate_untracked_recommendations_json(
        self,
        categories: Dict[str, Any],
        contributors: Dict[str, Any],
        total_untracked: int,
        total_commits: int,
    ) -> List[Dict[str, Any]]:
        """Generate JSON-formatted recommendations for untracked work."""
        recommendations = []

        # Category-based recommendations
        feature_count = categories.get("feature", {}).get("count", 0)
        bug_fix_count = categories.get("bug_fix", {}).get("count", 0)

        if feature_count > total_untracked * 0.25:
            recommendations.append(
                {
                    "type": "process_improvement",
                    "priority": "high",
                    "title": "Enforce Feature Ticket Requirements",
                    "description": "Many feature developments lack ticket references",
                    "action": "Require ticket creation and referencing for all new features",
                    "expected_impact": "Improved project visibility and planning",
                    "effort": "low",
                }
            )

        if bug_fix_count > total_untracked * 0.20:
            recommendations.append(
                {
                    "type": "process_improvement",
                    "priority": "high",
                    "title": "Link Bug Fixes to Issues",
                    "description": "Bug fixes should be tracked through issue management",
                    "action": "Create issues for bugs and reference them in fix commits",
                    "expected_impact": "Better bug tracking and resolution visibility",
                    "effort": "low",
                }
            )

        # Coverage-based recommendations
        untracked_pct = (total_untracked / total_commits * 100) if total_commits > 0 else 0
        if untracked_pct > 40:
            recommendations.append(
                {
                    "type": "team_training",
                    "priority": "medium",
                    "title": "Team Process Training",
                    "description": "High percentage of untracked commits indicates process gaps",
                    "action": "Provide training on ticket referencing and commit best practices",
                    "expected_impact": "Improved process adherence and visibility",
                    "effort": "medium",
                }
            )

        # Developer-specific recommendations
        if len(contributors) > 1:
            max_contributor_pct = max(
                (data["count"] / total_untracked * 100) for data in contributors.values()
            )
            if max_contributor_pct > 40:
                recommendations.append(
                    {
                        "type": "individual_coaching",
                        "priority": "medium",
                        "title": "Targeted Developer Coaching",
                        "description": "Some developers need additional guidance on process",
                        "action": "Provide one-on-one coaching for developers with high untracked rates",
                        "expected_impact": "More consistent process adherence across the team",
                        "effort": "low",
                    }
                )

        return recommendations

