"""Developer analysis mixin for EnhancedQualitativeAnalyzer."""

"""Enhanced qualitative analyzer for GitFlow Analytics.

This module provides sophisticated qualitative analysis across four key dimensions:
1. Executive Summary Analysis - High-level team health and strategic insights
2. Project Analysis - Project-specific momentum and health assessment
3. Developer Analysis - Individual contribution patterns and career development
4. Workflow Analysis - Process effectiveness and Git-PM correlation analysis

WHY: Traditional quantitative metrics only tell part of the story. This enhanced analyzer
combines statistical analysis with pattern recognition to generate actionable insights
for different stakeholder levels - from executives to individual developers.

DESIGN DECISIONS:
- Confidence-based scoring: All insights include confidence scores for reliability
- Multi-dimensional analysis: Each section focuses on different aspects of team performance
- Natural language generation: Produces human-readable insights and recommendations
- Anomaly detection: Identifies unusual patterns that merit attention
- Risk assessment: Flags potential issues before they become critical

INTEGRATION: Works with existing qualitative pipeline and extends JSON export format
with structured analysis results that can be consumed by dashboards and reports.
"""

import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np

from .models.schemas import QualitativeCommitData
from .utils.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)




class DeveloperAnalysisMixin:
    """Mixin: contribution patterns, collaboration, expertise, growth, burnout, career."""

    def _analyze_contribution_patterns(
        self, dev_commits: list[dict[str, Any]], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze individual developer contribution patterns."""

        if not dev_commits:
            return {"pattern": "no_activity", "confidence": 0.0}

        # Temporal consistency analysis
        weekly_commits = self._get_weekly_commit_counts(dev_commits)
        active_weeks = sum(1 for w in weekly_commits if w > 0)
        total_weeks = len(weekly_commits) if weekly_commits else 1
        consistency_rate = active_weeks / total_weeks

        # Commit size consistency
        commit_sizes = []
        for commit in dev_commits:
            lines = commit.get("filtered_insertions", commit.get("insertions", 0)) + commit.get(
                "filtered_deletions", commit.get("deletions", 0)
            )
            commit_sizes.append(lines)

        avg_commit_size = statistics.mean(commit_sizes) if commit_sizes else 0
        size_consistency = (
            100 - (statistics.pstdev(commit_sizes) / max(avg_commit_size, 1) * 100)
            if len(commit_sizes) > 1
            else 50
        )

        # Pattern classification
        total_commits = len(dev_commits)

        if (
            total_commits >= self.thresholds["high_productivity_commits"]
            and consistency_rate >= 0.7
        ):
            pattern = "consistent_high_performer"
            confidence = 0.9
        elif total_commits >= self.thresholds["high_productivity_commits"]:
            pattern = "high_volume_irregular"
            confidence = 0.8
        elif consistency_rate >= 0.7:
            pattern = "consistent_steady"
            confidence = 0.8
        elif consistency_rate < 0.3:
            pattern = "sporadic"
            confidence = 0.7
        else:
            pattern = "moderate_irregular"
            confidence = 0.6

        return {
            "pattern": pattern,
            "confidence": confidence,
            "consistency_rate": round(consistency_rate, 2),
            "avg_commit_size": round(avg_commit_size, 1),
            "size_consistency_score": round(max(0, size_consistency), 1),
            "total_commits": total_commits,
            "active_weeks": active_weeks,
            "description": self._get_pattern_description(pattern, consistency_rate, total_commits),
        }

    def _get_pattern_description(
        self, pattern: str, consistency_rate: float, total_commits: int
    ) -> str:
        """Get human-readable description of contribution pattern."""

        descriptions = {
            "consistent_high_performer": f"Highly productive with {consistency_rate:.0%} week consistency",
            "high_volume_irregular": f"High output ({total_commits} commits) but irregular timing",
            "consistent_steady": f"Steady contributor active {consistency_rate:.0%} of weeks",
            "moderate_irregular": "Moderate activity with irregular patterns",
            "sporadic": f"Sporadic activity in {consistency_rate:.0%} of weeks",
            "no_activity": "No significant activity in analysis period",
        }

        return descriptions.get(pattern, "Unknown contribution pattern")

    def _calculate_collaboration_score(
        self, dev_commits: list[dict[str, Any]], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate collaboration metrics for a developer."""

        # Project diversity
        projects_worked = set(c.get("project_key", "UNKNOWN") for c in dev_commits)
        project_diversity_score = min(100, len(projects_worked) * 25)

        # Cross-project contribution consistency
        project_commit_counts = defaultdict(int)
        for commit in dev_commits:
            project_key = commit.get("project_key", "UNKNOWN")
            project_commit_counts[project_key] += 1

        if len(project_commit_counts) > 1:
            # Calculate how evenly distributed work is across projects
            commit_values = list(project_commit_counts.values())
            gini = self._calculate_gini_coefficient(commit_values)
            distribution_score = (1 - gini) * 100  # Lower Gini = more even distribution
        else:
            distribution_score = 50  # Neutral for single project

        # Overall collaboration score
        collaboration_score = project_diversity_score * 0.6 + distribution_score * 0.4

        # Collaboration level classification
        if collaboration_score >= 80:
            level = "highly_collaborative"
        elif collaboration_score >= 60:
            level = "moderately_collaborative"
        elif collaboration_score >= 40:
            level = "focused_contributor"
        else:
            level = "single_focus"

        return {
            "score": round(collaboration_score, 1),
            "level": level,
            "projects_count": len(projects_worked),
            "project_diversity_score": round(project_diversity_score, 1),
            "work_distribution_score": round(distribution_score, 1),
            "projects_list": sorted(list(projects_worked)),
            "description": f"{level.replace('_', ' ').title()} - active in {len(projects_worked)} projects",
        }

    def _identify_expertise_domains(
        self, dev_commits: list[dict[str, Any]], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Identify developer expertise domains based on file patterns and projects."""

        domains = []

        # Analyze file patterns (simplified - in real implementation would use file extensions)
        total_commits = len(dev_commits)
        project_contributions = defaultdict(int)

        for commit in dev_commits:
            project_key = commit.get("project_key", "UNKNOWN")
            project_contributions[project_key] += 1

        # Create expertise domains based on project contributions
        for project, commit_count in project_contributions.items():
            contribution_percentage = (commit_count / total_commits) * 100

            if contribution_percentage >= 30:
                expertise_level = "expert"
            elif contribution_percentage >= 15:
                expertise_level = "proficient"
            else:
                expertise_level = "familiar"

            domains.append(
                {
                    "domain": project,
                    "expertise_level": expertise_level,
                    "contribution_percentage": round(contribution_percentage, 1),
                    "commit_count": commit_count,
                    "confidence": min(
                        0.9, commit_count / 20
                    ),  # Higher confidence with more commits
                }
            )

        # Sort by contribution percentage
        domains.sort(key=lambda x: x["contribution_percentage"], reverse=True)

        return domains[:5]  # Top 5 domains

    def _analyze_growth_trajectory(
        self, dev_commits: list[dict[str, Any]], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze developer growth trajectory over time."""

        if len(dev_commits) < 4:
            return {
                "trajectory": "insufficient_data",
                "confidence": 0.1,
                "description": "Not enough data for growth analysis",
            }

        # Sort commits chronologically
        sorted_commits = sorted(dev_commits, key=lambda x: x["timestamp"])

        # Split into quarters for trend analysis
        quarter_size = len(sorted_commits) // 4
        if quarter_size == 0:
            return {
                "trajectory": "insufficient_data",
                "confidence": 0.2,
                "description": "Insufficient commit history for growth analysis",
            }

        quarters = []
        for i in range(4):
            start_idx = i * quarter_size
            end_idx = (i + 1) * quarter_size if i < 3 else len(sorted_commits)
            quarters.append(sorted_commits[start_idx:end_idx])

        # Analyze complexity trends (using commit size as proxy)
        quarter_complexities = []
        for quarter in quarters:
            if not quarter:
                continue
            quarter_complexity = statistics.mean(
                [
                    commit.get("filtered_insertions", commit.get("insertions", 0))
                    + commit.get("filtered_deletions", commit.get("deletions", 0))
                    for commit in quarter
                ]
            )
            quarter_complexities.append(quarter_complexity)

        # Analyze project diversity trends
        quarter_projects = []
        for quarter in quarters:
            projects = set(c.get("project_key", "UNKNOWN") for c in quarter)
            quarter_projects.append(len(projects))

        # Determine trajectory
        if len(quarter_complexities) >= 2 and len(quarter_projects) >= 2:
            complexity_trend = (quarter_complexities[-1] - quarter_complexities[0]) / max(
                quarter_complexities[0], 1
            )
            project_trend = quarter_projects[-1] - quarter_projects[0]

            if complexity_trend > 0.2 or project_trend > 0:
                trajectory = "growing"
                description = "Increasing complexity and scope of contributions"
            elif complexity_trend < -0.2 and project_trend < 0:
                trajectory = "declining"
                description = "Decreasing complexity and scope of work"
            else:
                trajectory = "stable"
                description = "Consistent level of contribution complexity"

            confidence = min(0.8, len(sorted_commits) / 50)  # Higher confidence with more data
        else:
            trajectory = "stable"
            description = "Stable contribution pattern"
            confidence = 0.5

        return {
            "trajectory": trajectory,
            "confidence": confidence,
            "description": description,
            "complexity_trend": (
                round(complexity_trend * 100, 1) if "complexity_trend" in locals() else 0
            ),
            "project_expansion": project_trend if "project_trend" in locals() else 0,
        }

    def _detect_burnout_indicators(
        self, dev_commits: list[dict[str, Any]], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Detect potential burnout indicators for a developer."""

        indicators = []

        # Weekend work pattern
        weekend_commits = sum(
            1
            for c in dev_commits
            if hasattr(c.get("timestamp"), "weekday") and c["timestamp"].weekday() >= 5
        )
        weekend_percentage = (weekend_commits / len(dev_commits)) * 100 if dev_commits else 0

        if weekend_percentage > 40:  # More than 40% weekend work
            indicators.append(
                {
                    "type": "excessive_weekend_work",
                    "severity": "medium",
                    "description": f"{weekend_percentage:.1f}% of commits made on weekends",
                    "risk_level": "work_life_balance",
                    "confidence": 0.7,
                }
            )

        # Late night commits (if timezone info available)
        late_night_commits = 0
        for commit in dev_commits:
            timestamp = commit.get("timestamp")
            if hasattr(timestamp, "hour") and (timestamp.hour >= 22 or timestamp.hour <= 5):
                # 10 PM to 5 AM
                late_night_commits += 1

        late_night_percentage = (late_night_commits / len(dev_commits)) * 100 if dev_commits else 0
        if late_night_percentage > 30:
            indicators.append(
                {
                    "type": "late_night_activity",
                    "severity": "medium",
                    "description": f"{late_night_percentage:.1f}% of commits made late night/early morning",
                    "risk_level": "work_life_balance",
                    "confidence": 0.6,
                }
            )

        # Declining commit quality (increasing size without proportional impact)
        recent_commits = sorted(dev_commits, key=lambda x: x["timestamp"])[-10:]  # Last 10 commits
        if len(recent_commits) >= 5:
            recent_sizes = [
                c.get("filtered_insertions", c.get("insertions", 0))
                + c.get("filtered_deletions", c.get("deletions", 0))
                for c in recent_commits
            ]
            avg_recent_size = statistics.mean(recent_sizes)

            if avg_recent_size > self.thresholds["large_commit_lines"]:
                indicators.append(
                    {
                        "type": "increasing_commit_sizes",
                        "severity": "low",
                        "description": f"Recent commits average {avg_recent_size:.0f} lines",
                        "risk_level": "productivity",
                        "confidence": 0.5,
                    }
                )

        return indicators

    def _generate_career_recommendations(
        self,
        contribution_pattern: dict[str, Any],
        collaboration_score: dict[str, Any],
        expertise_domains: list[dict[str, Any]],
        growth_trajectory: dict[str, Any],
        burnout_indicators: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate career development recommendations for a developer."""

        recommendations = []

        # Pattern-based recommendations
        pattern = contribution_pattern.get("pattern", "")
        if pattern == "sporadic":
            recommendations.append(
                {
                    "category": "consistency",
                    "title": "Improve Contribution Consistency",
                    "action": "Establish regular development schedule and focus on smaller, frequent commits",
                    "priority": "medium",
                    "expected_benefit": "Better project integration and skill development",
                }
            )
        elif pattern == "high_volume_irregular":
            recommendations.append(
                {
                    "category": "work_balance",
                    "title": "Balance Workload Distribution",
                    "action": "Spread work more evenly across time periods to improve sustainability",
                    "priority": "medium",
                    "expected_benefit": "Reduced burnout risk and more consistent output",
                }
            )

        # Collaboration recommendations
        collab_level = collaboration_score.get("level", "")
        if collab_level == "single_focus":
            recommendations.append(
                {
                    "category": "growth",
                    "title": "Expand Project Involvement",
                    "action": "Contribute to additional projects to broaden experience and impact",
                    "priority": "low",
                    "expected_benefit": "Increased versatility and cross-team collaboration",
                }
            )
        elif collab_level == "highly_collaborative":
            recommendations.append(
                {
                    "category": "leadership",
                    "title": "Consider Technical Leadership Role",
                    "action": "Leverage cross-project experience to mentor others and guide architecture decisions",
                    "priority": "low",
                    "expected_benefit": "Career advancement and increased impact",
                }
            )

        # Growth trajectory recommendations
        trajectory = growth_trajectory.get("trajectory", "")
        if trajectory == "declining":
            recommendations.append(
                {
                    "category": "engagement",
                    "title": "Address Declining Engagement",
                    "action": "Discuss career goals and explore new challenges or responsibilities",
                    "priority": "high",
                    "expected_benefit": "Renewed motivation and career development",
                }
            )
        elif trajectory == "stable":
            recommendations.append(
                {
                    "category": "development",
                    "title": "Seek New Challenges",
                    "action": "Take on more complex tasks or explore new technology areas",
                    "priority": "medium",
                    "expected_benefit": "Continued skill development and career growth",
                }
            )

        # Burnout prevention recommendations
        if burnout_indicators:
            high_severity = [i for i in burnout_indicators if i.get("severity") == "high"]
            if high_severity or len(burnout_indicators) >= 2:
                recommendations.append(
                    {
                        "category": "wellbeing",
                        "title": "Address Work-Life Balance",
                        "action": "Review working patterns and implement better time boundaries",
                        "priority": "high",
                        "expected_benefit": "Improved wellbeing and sustainable productivity",
                    }
                )

        return recommendations[:4]  # Top 4 recommendations

    def _generate_developer_narrative(
        self,
        developer_name: str,
        contribution_pattern: dict[str, Any],
        expertise_domains: list[dict[str, Any]],
        growth_trajectory: dict[str, Any],
    ) -> str:
        """Generate narrative summary for a developer."""

        narrative_parts = []

        # Developer introduction with pattern
        pattern_desc = contribution_pattern.get("description", "shows mixed activity patterns")
        narrative_parts.append(f"{developer_name} {pattern_desc}.")

        # Expertise areas
        if expertise_domains and len(expertise_domains) > 0:
            primary_domain = expertise_domains[0]
            if len(expertise_domains) == 1:
                narrative_parts.append(
                    f"Primary expertise in {primary_domain['domain']} with {primary_domain['expertise_level']} level proficiency."
                )
            else:
                narrative_parts.append(
                    f"Multi-domain contributor with {primary_domain['expertise_level']} expertise in {primary_domain['domain']} and experience across {len(expertise_domains)} areas."
                )

        # Growth trajectory
        trajectory = growth_trajectory.get("trajectory", "stable")
        trajectory_desc = growth_trajectory.get("description", "")
        if trajectory == "growing":
            narrative_parts.append(
                f"Shows positive growth trajectory with {trajectory_desc.lower()}."
            )
        elif trajectory == "declining":
            narrative_parts.append(f"Attention needed: {trajectory_desc.lower()}.")
        else:
            narrative_parts.append(f"Maintains {trajectory_desc.lower()}.")

        return " ".join(narrative_parts)

    # Workflow Analysis Helper Methods

