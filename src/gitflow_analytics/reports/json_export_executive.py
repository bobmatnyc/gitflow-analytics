"""JSON exporter mixin: executive analytics (trends, anomalies, wins/concerns, health scores, project analysis).

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


class JSONExportExecutiveMixin:
    """Mixin providing executive analytics (trends, anomalies, wins/concerns, health scores, project analysis) for ComprehensiveJSONExporter."""

    def _calculate_executive_trends(
        self, commits: List[Dict[str, Any]], prs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate trends by comparing first half vs second half of data."""

        if not commits:
            return {}

        # Sort commits by timestamp
        sorted_commits = sorted(commits, key=lambda x: x["timestamp"])
        midpoint = len(sorted_commits) // 2

        first_half = sorted_commits[:midpoint]
        second_half = sorted_commits[midpoint:]

        # Calculate metrics for each half
        def get_half_metrics(commit_list):
            return {
                "commits": len(commit_list),
                "lines": sum(
                    c.get("filtered_insertions", c.get("insertions", 0))
                    + c.get("filtered_deletions", c.get("deletions", 0))
                    for c in commit_list
                ),
                "story_points": sum(c.get("story_points", 0) or 0 for c in commit_list),
            }

        first_metrics = get_half_metrics(first_half)
        second_metrics = get_half_metrics(second_half)

        # Calculate percentage changes
        trends = {}
        for metric in ["commits", "lines", "story_points"]:
            if first_metrics[metric] > 0:
                change = (
                    (second_metrics[metric] - first_metrics[metric]) / first_metrics[metric]
                ) * 100
                trends[f"{metric}_trend"] = round(change, 1)
            else:
                trends[f"{metric}_trend"] = 0

        # PR trends if available
        if prs:
            sorted_prs = sorted(
                prs, key=lambda x: x.get("merged_at", x.get("created_at", datetime.now()))
            )
            pr_midpoint = len(sorted_prs) // 2

            first_pr_count = pr_midpoint
            second_pr_count = len(sorted_prs) - pr_midpoint

            if first_pr_count > 0:
                pr_change = ((second_pr_count - first_pr_count) / first_pr_count) * 100
                trends["prs_trend"] = round(pr_change, 1)
            else:
                trends["prs_trend"] = 0

        return trends

    def _detect_executive_anomalies(
        self, commits: List[Dict[str, Any]], developer_stats: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in executive-level data."""

        anomalies = []

        # Check for commit spikes/drops by week
        weekly_commits = self._get_weekly_commit_counts(commits)
        if len(weekly_commits) >= 3:
            mean_commits = statistics.mean(weekly_commits)
            std_commits = statistics.pstdev(weekly_commits) if len(weekly_commits) > 1 else 0

            for i, count in enumerate(weekly_commits):
                if std_commits > 0:
                    if count > mean_commits + (
                        std_commits * self.anomaly_thresholds["spike_multiplier"]
                    ):
                        anomalies.append(
                            {
                                "type": "spike",
                                "metric": "weekly_commits",
                                "value": count,
                                "expected": round(mean_commits, 1),
                                "severity": "high"
                                if count > mean_commits + (std_commits * 3)
                                else "medium",
                                "week_index": i,
                            }
                        )
                    elif count < mean_commits * self.anomaly_thresholds["drop_threshold"]:
                        anomalies.append(
                            {
                                "type": "drop",
                                "metric": "weekly_commits",
                                "value": count,
                                "expected": round(mean_commits, 1),
                                "severity": "high" if count < mean_commits * 0.1 else "medium",
                                "week_index": i,
                            }
                        )

        # Check for contributor anomalies
        commit_counts = [dev["total_commits"] for dev in developer_stats]
        if len(commit_counts) > 1:
            gini_coefficient = self._calculate_gini_coefficient(commit_counts)
            if gini_coefficient > 0.8:
                anomalies.append(
                    {
                        "type": "concentration",
                        "metric": "contribution_distribution",
                        "value": round(gini_coefficient, 2),
                        "threshold": 0.8,
                        "severity": "medium",
                        "description": "Highly concentrated contribution pattern",
                    }
                )

        return anomalies

    def _identify_wins_and_concerns(
        self,
        commits: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        project_metrics: Dict[str, Any],
        dora_metrics: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Identify key wins and concerns from the data."""

        wins = []
        concerns = []

        # Ticket coverage analysis
        ticket_coverage = project_metrics.get("ticket_analysis", {}).get("commit_coverage_pct", 0)
        if ticket_coverage > 80:
            wins.append(
                {
                    "category": "process",
                    "title": "Excellent Ticket Coverage",
                    "description": f"{ticket_coverage:.1f}% of commits linked to tickets",
                    "impact": "high",
                }
            )
        elif ticket_coverage < 30:
            concerns.append(
                {
                    "category": "process",
                    "title": "Low Ticket Coverage",
                    "description": f"Only {ticket_coverage:.1f}% of commits linked to tickets",
                    "impact": "high",
                    "recommendation": "Improve ticket referencing in commit messages",
                }
            )

        # Team activity analysis
        if len(developer_stats) > 1:
            commit_counts = [dev["total_commits"] for dev in developer_stats]
            avg_commits = sum(commit_counts) / len(commit_counts)

            if min(commit_counts) > avg_commits * 0.5:
                wins.append(
                    {
                        "category": "team",
                        "title": "Balanced Team Contributions",
                        "description": "All team members are actively contributing",
                        "impact": "medium",
                    }
                )
            elif max(commit_counts) > avg_commits * 3:
                concerns.append(
                    {
                        "category": "team",
                        "title": "Unbalanced Contributions",
                        "description": "Work is heavily concentrated among few developers",
                        "impact": "medium",
                        "recommendation": "Consider distributing work more evenly",
                    }
                )

        # Code quality indicators
        total_lines = sum(
            c.get("filtered_insertions", c.get("insertions", 0))
            + c.get("filtered_deletions", c.get("deletions", 0))
            for c in commits
        )
        avg_commit_size = total_lines / max(len(commits), 1)

        if 20 <= avg_commit_size <= 200:
            wins.append(
                {
                    "category": "quality",
                    "title": "Optimal Commit Size",
                    "description": f"Average commit size of {avg_commit_size:.0f} lines indicates good change management",
                    "impact": "low",
                }
            )
        elif avg_commit_size > 500:
            concerns.append(
                {
                    "category": "quality",
                    "title": "Large Commit Sizes",
                    "description": f"Average commit size of {avg_commit_size:.0f} lines may indicate batched changes",
                    "impact": "low",
                    "recommendation": "Consider breaking down changes into smaller commits",
                }
            )

        return wins, concerns

    def _calculate_overall_health_score(
        self,
        commits: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        project_metrics: Dict[str, Any],
        dora_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate overall project health score."""

        scores = {}

        # Activity consistency score (0-100)
        weekly_commits = self._get_weekly_commit_counts(commits)
        if weekly_commits:
            consistency = max(
                0,
                100
                - (
                    statistics.pstdev(weekly_commits)
                    / max(statistics.mean(weekly_commits), 1)
                    * 100
                ),
            )
            scores["activity_consistency"] = min(100, consistency)
        else:
            scores["activity_consistency"] = 0

        # Ticket coverage score
        ticket_coverage = project_metrics.get("ticket_analysis", {}).get("commit_coverage_pct", 0)
        scores["ticket_coverage"] = min(100, ticket_coverage)

        # Collaboration score (based on multi-project work and team balance)
        if len(developer_stats) > 1:
            commit_counts = [dev["total_commits"] for dev in developer_stats]
            gini = self._calculate_gini_coefficient(commit_counts)
            collaboration_score = max(0, 100 - (gini * 100))
            scores["collaboration"] = collaboration_score
        else:
            scores["collaboration"] = 50  # Neutral for single developer

        # Code quality score (based on commit size and patterns)
        total_lines = sum(
            c.get("filtered_insertions", c.get("insertions", 0))
            + c.get("filtered_deletions", c.get("deletions", 0))
            for c in commits
        )
        avg_commit_size = total_lines / max(len(commits), 1)

        # Optimal range is 20-200 lines per commit
        if 20 <= avg_commit_size <= 200:
            quality_score = 100
        elif avg_commit_size < 20:
            quality_score = max(0, (avg_commit_size / 20) * 100)
        else:
            quality_score = max(0, 100 - ((avg_commit_size - 200) / 500 * 100))

        scores["code_quality"] = min(100, quality_score)

        # Velocity score (commits per week vs. baseline)
        weeks_with_activity = len([w for w in weekly_commits if w > 0])
        velocity_score = min(100, (weeks_with_activity / max(len(weekly_commits), 1)) * 100)
        scores["velocity"] = velocity_score

        # Calculate weighted overall score
        overall_score = sum(
            scores.get(metric, 0) * weight for metric, weight in self.health_weights.items()
        )

        return {
            "overall": round(overall_score, 1),
            "components": {k: round(v, 1) for k, v in scores.items()},
            "weights": self.health_weights,
            "rating": self._get_health_rating(overall_score),
        }

    def _get_health_rating(self, score: float) -> str:
        """Get health rating based on score."""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        else:
            return "needs_improvement"

    def _get_trend_direction(self, trend_percent: float) -> str:
        """Get trend direction from percentage change."""
        if abs(trend_percent) < self.anomaly_thresholds["trend_threshold"] * 100:
            return "stable"
        elif trend_percent > 0:
            return "increasing"
        else:
            return "decreasing"

    def _get_coverage_quality_rating(self, coverage: float) -> str:
        """Get quality rating for ticket coverage."""
        if coverage >= 80:
            return "excellent"
        elif coverage >= 60:
            return "good"
        elif coverage >= 40:
            return "fair"
        else:
            return "poor"

    def _calculate_active_developer_percentage(
        self, developer_stats: List[Dict[str, Any]]
    ) -> float:
        """Calculate percentage of developers with meaningful activity."""
        if not developer_stats:
            return 0

        total_commits = sum(dev["total_commits"] for dev in developer_stats)
        avg_commits = total_commits / len(developer_stats)
        threshold = max(1, avg_commits * 0.1)  # 10% of average

        active_developers = sum(1 for dev in developer_stats if dev["total_commits"] >= threshold)
        return round((active_developers / len(developer_stats)) * 100, 1)

    def _calculate_avg_developers_per_project(self, commits: List[Dict[str, Any]]) -> float:
        """Calculate average number of developers per project."""
        project_developers = defaultdict(set)

        for commit in commits:
            project_key = commit.get("project_key", "UNKNOWN")
            dev_id = commit.get("canonical_id", commit.get("author_email"))
            project_developers[project_key].add(dev_id)

        if not project_developers:
            return 0

        avg = sum(len(devs) for devs in project_developers.values()) / len(project_developers)
        return round(avg, 1)

    def _count_cross_project_contributors(
        self, commits: List[Dict[str, Any]], developer_stats: List[Dict[str, Any]]
    ) -> int:
        """Count developers who contribute to multiple projects."""
        developer_projects = defaultdict(set)

        for commit in commits:
            project_key = commit.get("project_key", "UNKNOWN")
            dev_id = commit.get("canonical_id", commit.get("author_email"))
            developer_projects[dev_id].add(project_key)

        return sum(1 for projects in developer_projects.values() if len(projects) > 1)

    def _calculate_project_health_score(
        self, commits: List[Dict[str, Any]], contributors: Set[str]
    ) -> Dict[str, Any]:
        """Calculate health score for a specific project."""

        if not commits:
            return {"overall": 0, "components": {}, "rating": "no_data"}

        scores = {}

        # Activity score (commits per week)
        weekly_commits = self._get_weekly_commit_counts(commits)
        if weekly_commits:
            avg_weekly = statistics.mean(weekly_commits)
            activity_score = min(100, avg_weekly * 10)  # Scale appropriately
            scores["activity"] = activity_score
        else:
            scores["activity"] = 0

        # Contributor diversity score
        if len(contributors) == 1:
            diversity_score = 30  # Single contributor is risky
        elif len(contributors) <= 3:
            diversity_score = 60
        else:
            diversity_score = 100
        scores["contributor_diversity"] = diversity_score

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

        # Overall score (equal weights for now)
        overall_score = sum(scores.values()) / len(scores)

        return {
            "overall": round(overall_score, 1),
            "components": {k: round(v, 1) for k, v in scores.items()},
            "rating": self._get_health_rating(overall_score),
        }

    def _get_project_contributor_details(
        self, commits: List[Dict[str, Any]], developer_stats: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get detailed contributor information for a project."""

        # Create developer lookup
        dev_lookup = {dev["canonical_id"]: dev for dev in developer_stats}

        # Count contributions per developer
        contributor_commits = defaultdict(int)
        contributor_lines = defaultdict(int)

        for commit in commits:
            dev_id = commit.get("canonical_id", commit.get("author_email"))
            contributor_commits[dev_id] += 1

            lines = commit.get("filtered_insertions", commit.get("insertions", 0)) + commit.get(
                "filtered_deletions", commit.get("deletions", 0)
            )
            contributor_lines[dev_id] += lines

        # Build contributor details
        contributors = []
        total_commits = len(commits)

        for dev_id, commit_count in contributor_commits.items():
            dev = dev_lookup.get(dev_id, {})

            contributors.append(
                {
                    "id": dev_id,
                    "name": self._anonymize_value(dev.get("primary_name", "Unknown"), "name"),
                    "commits": commit_count,
                    "commits_percentage": round((commit_count / total_commits) * 100, 1),
                    "lines_changed": contributor_lines[dev_id],
                    "role": self._determine_contributor_role(commit_count, total_commits),
                }
            )

        # Sort by commits descending
        contributors.sort(key=lambda x: x["commits"], reverse=True)

        return contributors

    def _determine_contributor_role(self, commits: int, total_commits: int) -> str:
        """Determine contributor role based on contribution percentage."""
        percentage = (commits / total_commits) * 100

        if percentage >= 50:
            return "primary"
        elif percentage >= 25:
            return "major"
        elif percentage >= 10:
            return "regular"
        else:
            return "occasional"

    def _calculate_project_trends(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trends for a specific project."""

        if len(commits) < 4:  # Need sufficient data for trends
            return {"insufficient_data": True}

        # Sort by timestamp
        sorted_commits = sorted(commits, key=lambda x: x["timestamp"])

        # Split into quarters for trend analysis
        quarter_size = len(sorted_commits) // 4
        quarters = [sorted_commits[i * quarter_size : (i + 1) * quarter_size] for i in range(4)]

        # Handle remainder commits
        if len(sorted_commits) % 4:
            quarters[-1].extend(sorted_commits[4 * quarter_size :])

        # Calculate metrics per quarter
        quarter_metrics = []
        for quarter in quarters:
            metrics = {
                "commits": len(quarter),
                "lines": sum(
                    c.get("filtered_insertions", c.get("insertions", 0))
                    + c.get("filtered_deletions", c.get("deletions", 0))
                    for c in quarter
                ),
                "contributors": len(
                    set(c.get("canonical_id", c.get("author_email")) for c in quarter)
                ),
            }
            quarter_metrics.append(metrics)

        # Calculate trends (compare Q1 vs Q4)
        trends = {}
        for metric in ["commits", "lines", "contributors"]:
            q1_value = quarter_metrics[0][metric]
            q4_value = quarter_metrics[-1][metric]

            if q1_value > 0:
                change = ((q4_value - q1_value) / q1_value) * 100
                trends[f"{metric}_trend"] = round(change, 1)
            else:
                trends[f"{metric}_trend"] = 0

        return trends

    def _detect_project_anomalies(self, commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in project-specific data."""

        if len(commits) < 7:  # Need sufficient data
            return []

        anomalies = []

        # Get daily commit counts
        daily_commits = self._get_daily_commit_counts(commits)

        if len(daily_commits) >= 7:
            mean_daily = statistics.mean(daily_commits)
            std_daily = statistics.pstdev(daily_commits) if len(daily_commits) > 1 else 0

            # Find days with unusual activity
            for i, count in enumerate(daily_commits):
                if std_daily > 0 and count > mean_daily + (std_daily * 2):
                    anomalies.append(
                        {
                            "type": "activity_spike",
                            "value": count,
                            "expected": round(mean_daily, 1),
                            "day_index": i,
                            "severity": "medium",
                        }
                    )

        return anomalies

    def _identify_primary_contributors(
        self, commits: List[Dict[str, Any]], contributor_details: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify primary contributors (top 80% of activity)."""

        sorted_contributors = sorted(contributor_details, key=lambda x: x["commits"], reverse=True)
        total_commits = sum(c["commits"] for c in contributor_details)

        primary_contributors = []
        cumulative_commits = 0

        for contributor in sorted_contributors:
            cumulative_commits += contributor["commits"]
            primary_contributors.append(contributor["name"])

            if cumulative_commits >= total_commits * 0.8:
                break

        return primary_contributors

    def _calculate_contribution_distribution(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate distribution metrics for contributions."""

        contributor_commits = defaultdict(int)
        for commit in commits:
            dev_id = commit.get("canonical_id", commit.get("author_email"))
            contributor_commits[dev_id] += 1

        commit_counts = list(contributor_commits.values())

        if not commit_counts:
            return {}

        gini = self._calculate_gini_coefficient(commit_counts)

        return {
            "gini_coefficient": round(gini, 3),
            "concentration_level": "high" if gini > 0.7 else "medium" if gini > 0.4 else "low",
            "top_contributor_percentage": round((max(commit_counts) / sum(commit_counts)) * 100, 1),
            "contributor_count": len(commit_counts),
        }

