"""Project analysis mixin for EnhancedQualitativeAnalyzer."""

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




class ProjectAnalysisMixin:
    """Mixin: project momentum, health, tech debt, delivery, risks, recommendations."""

    def _classify_project_momentum(
        self, project_commits: list[dict[str, Any]], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Classify project momentum as growing, stable, or declining."""

        if len(project_commits) < 4:
            return {
                "classification": "insufficient_data",
                "confidence": 0.1,
                "trend_percentage": 0,
                "description": "Not enough data for momentum analysis",
            }

        # Analyze commit trends over time
        sorted_commits = sorted(project_commits, key=lambda x: x["timestamp"])
        midpoint = len(sorted_commits) // 2

        first_half = sorted_commits[:midpoint]
        second_half = sorted_commits[midpoint:]

        first_count = len(first_half)
        second_count = len(second_half)

        if first_count > 0:
            trend_percentage = ((second_count - first_count) / first_count) * 100
        else:
            trend_percentage = 0

        # Classification logic
        if trend_percentage > 20:
            classification = "growing"
            description = (
                f"Strong upward momentum with {trend_percentage:.1f}% increase in activity"
            )
        elif trend_percentage < -20:
            classification = "declining"
            description = (
                f"Concerning decline with {abs(trend_percentage):.1f}% decrease in activity"
            )
        else:
            classification = "stable"
            description = f"Consistent activity with {abs(trend_percentage):.1f}% variance"

        # Confidence based on data quality
        time_span = (sorted_commits[-1]["timestamp"] - sorted_commits[0]["timestamp"]).days
        confidence = min(0.9, time_span / (context["weeks_analyzed"] * 7))

        return {
            "classification": classification,
            "confidence": confidence,
            "trend_percentage": round(trend_percentage, 1),
            "description": description,
        }

    def _calculate_project_health_indicators(
        self, project_commits: list[dict[str, Any]], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate various health indicators for a project."""

        # Activity level
        weekly_commits = len(project_commits) / max(context["weeks_analyzed"], 1)
        activity_score = min(100, weekly_commits * 15)  # Scale appropriately

        # Contributor diversity
        contributors = set(c.get("canonical_id", c.get("author_email")) for c in project_commits)
        diversity_score = min(100, len(contributors) * 25)  # Max score with 4+ contributors

        # PR velocity (if available)
        pr_velocity_score = 75  # Default neutral score when PR data not available

        # Ticket coverage
        commits_with_tickets = sum(1 for c in project_commits if c.get("ticket_references"))
        ticket_coverage = (
            (commits_with_tickets / len(project_commits)) * 100 if project_commits else 0
        )

        # Overall health calculation
        indicators = {
            "activity_level": {
                "score": round(activity_score, 1),
                "description": f"{weekly_commits:.1f} commits per week",
                "status": (
                    "excellent"
                    if activity_score >= 80
                    else "good"
                    if activity_score >= 60
                    else "needs_improvement"
                ),
            },
            "contributor_diversity": {
                "score": round(diversity_score, 1),
                "description": f"{len(contributors)} active contributors",
                "status": (
                    "excellent"
                    if len(contributors) >= 4
                    else "good"
                    if len(contributors) >= 2
                    else "concerning"
                ),
            },
            "pr_velocity": {
                "score": pr_velocity_score,
                "description": "PR data not available",
                "status": "unknown",
            },
            "ticket_coverage": {
                "score": round(ticket_coverage, 1),
                "description": f"{ticket_coverage:.1f}% commits linked to tickets",
                "status": (
                    "excellent"
                    if ticket_coverage >= 80
                    else "good"
                    if ticket_coverage >= 60
                    else "needs_improvement"
                ),
            },
        }

        # Calculate overall health score
        overall_score = statistics.mean(
            [
                indicators["activity_level"]["score"],
                indicators["contributor_diversity"]["score"],
                indicators["ticket_coverage"]["score"],
            ]
        )

        indicators["overall_health"] = {
            "score": round(overall_score, 1),
            "status": (
                "excellent"
                if overall_score >= 80
                else "good"
                if overall_score >= 60
                else "needs_improvement"
            ),
        }

        return indicators

    def _detect_technical_debt_signals(
        self, project_commits: list[dict[str, Any]], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Detect signals of technical debt accumulation."""

        signals = []

        # Large commit pattern (potential code quality issue)
        large_commits = []
        for commit in project_commits:
            lines_changed = commit.get(
                "filtered_insertions", commit.get("insertions", 0)
            ) + commit.get("filtered_deletions", commit.get("deletions", 0))
            if lines_changed > self.thresholds["large_commit_lines"]:
                large_commits.append(commit)

        if len(large_commits) > len(project_commits) * 0.2:
            signals.append(
                {
                    "type": "large_commits",
                    "severity": "medium",
                    "description": f"{len(large_commits)} commits exceed {self.thresholds['large_commit_lines']} lines",
                    "impact": "Difficult code review, potential quality issues",
                    "recommendation": "Break down changes into smaller, focused commits",
                }
            )

        # Fix-heavy pattern analysis
        fix_commits = []
        for commit in project_commits:
            message = commit.get("message", "").lower()
            if any(keyword in message for keyword in ["fix", "bug", "hotfix", "patch"]):
                fix_commits.append(commit)

        fix_percentage = (len(fix_commits) / len(project_commits)) * 100 if project_commits else 0
        if fix_percentage > 30:  # More than 30% fix commits
            signals.append(
                {
                    "type": "high_fix_ratio",
                    "severity": "high",
                    "description": f"{fix_percentage:.1f}% of commits are fixes",
                    "impact": "Indicates quality issues in initial development",
                    "recommendation": "Improve testing and code review processes",
                }
            )

        return signals

    def _assess_delivery_predictability(
        self, project_commits: list[dict[str, Any]], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess how predictable project delivery patterns are."""

        if len(project_commits) < 7:
            return {
                "score": 0,
                "status": "insufficient_data",
                "description": "Not enough data for predictability analysis",
            }

        # Calculate weekly commit consistency
        weekly_counts = defaultdict(int)
        for commit in project_commits:
            week_key = self._get_week_start(commit["timestamp"]).strftime("%Y-%m-%d")
            weekly_counts[week_key] += 1

        weekly_values = list(weekly_counts.values())

        if len(weekly_values) < 2:
            predictability_score = 50  # Neutral score
        else:
            mean_weekly = statistics.mean(weekly_values)
            std_weekly = statistics.pstdev(weekly_values)

            # Lower standard deviation = higher predictability
            consistency = max(0, 100 - (std_weekly / max(mean_weekly, 1) * 100))
            predictability_score = min(100, consistency)

        # Determine status
        if predictability_score >= 80:
            status = "highly_predictable"
        elif predictability_score >= 60:
            status = "moderately_predictable"
        else:
            status = "unpredictable"

        return {
            "score": round(predictability_score, 1),
            "status": status,
            "description": f"Delivery shows {status.replace('_', ' ')} patterns",
        }

    def _assess_project_risks(
        self, project_commits: list[dict[str, Any]], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Assess various risks for the project."""

        risks = []

        # Single contributor dependency risk
        contributors = defaultdict(int)
        for commit in project_commits:
            dev_id = commit.get("canonical_id", commit.get("author_email"))
            contributors[dev_id] += 1

        if len(contributors) == 1:
            risks.append(
                {
                    "type": "single_contributor",
                    "severity": "high",
                    "description": "Project depends on single contributor",
                    "probability": "high",
                    "impact": "Project abandonment risk if contributor leaves",
                    "mitigation": "Involve additional team members in project",
                }
            )
        elif len(contributors) > 1:
            top_contributor_pct = (max(contributors.values()) / sum(contributors.values())) * 100
            if top_contributor_pct > 80:
                risks.append(
                    {
                        "type": "contributor_concentration",
                        "severity": "medium",
                        "description": f"Top contributor handles {top_contributor_pct:.1f}% of work",
                        "probability": "medium",
                        "impact": "Knowledge concentration risk",
                        "mitigation": "Distribute knowledge and responsibilities",
                    }
                )

        # Activity decline risk
        recent_commits = [
            c for c in project_commits if (datetime.now(timezone.utc) - c["timestamp"]).days <= 14
        ]

        if len(recent_commits) == 0 and len(project_commits) > 5:
            risks.append(
                {
                    "type": "abandonment_risk",
                    "severity": "high",
                    "description": "No activity in past 2 weeks",
                    "probability": "medium",
                    "impact": "Project may be abandoned",
                    "mitigation": "Review project status and resource allocation",
                }
            )

        return risks

    def _generate_project_recommendations(
        self,
        momentum: dict[str, Any],
        health_indicators: dict[str, Any],
        tech_debt_signals: list[dict[str, Any]],
        risk_assessment: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate project-specific recommendations."""

        recommendations = []

        # Momentum-based recommendations
        if momentum["classification"] == "declining":
            recommendations.append(
                {
                    "priority": "high",
                    "category": "momentum",
                    "title": "Address Declining Activity",
                    "action": "Investigate causes of reduced activity and reallocate resources",
                    "expected_outcome": "Restored project momentum",
                }
            )

        # Health-based recommendations
        overall_health = health_indicators.get("overall_health", {})
        if overall_health.get("status") == "needs_improvement":
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "health",
                    "title": "Improve Project Health Metrics",
                    "action": "Focus on activity consistency and contributor engagement",
                    "expected_outcome": "Better project sustainability",
                }
            )

        # Technical debt recommendations
        high_severity_debt = [s for s in tech_debt_signals if s.get("severity") == "high"]
        if high_severity_debt:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "quality",
                    "title": "Address Technical Debt",
                    "action": high_severity_debt[0].get(
                        "recommendation", "Improve code quality practices"
                    ),
                    "expected_outcome": "Reduced maintenance burden",
                }
            )

        # Risk-based recommendations
        high_severity_risks = [r for r in risk_assessment if r.get("severity") == "high"]
        if high_severity_risks:
            recommendations.append(
                {
                    "priority": "critical",
                    "category": "risk",
                    "title": "Mitigate Critical Risks",
                    "action": high_severity_risks[0].get(
                        "mitigation", "Address identified risk factors"
                    ),
                    "expected_outcome": "Improved project stability",
                }
            )

        return recommendations[:3]  # Top 3 recommendations per project

    def _generate_project_narrative(
        self,
        project_key: str,
        momentum: dict[str, Any],
        health_indicators: dict[str, Any],
        risk_assessment: list[dict[str, Any]],
    ) -> str:
        """Generate narrative summary for a project."""

        narrative_parts = []

        # Project momentum
        momentum_descriptions = {
            "growing": "showing strong growth momentum",
            "stable": "maintaining steady progress",
            "declining": "experiencing declining activity",
            "insufficient_data": "lacking sufficient activity data",
        }

        momentum_desc = momentum_descriptions.get(momentum["classification"], "in unclear state")
        narrative_parts.append(f"Project {project_key} is {momentum_desc}.")

        # Health status
        overall_health = health_indicators.get("overall_health", {})
        health_score = overall_health.get("score", 0)
        narrative_parts.append(f"Overall project health scores {health_score:.1f}/100.")

        # Key strengths or concerns
        activity = health_indicators.get("activity_level", {})
        contributors = health_indicators.get("contributor_diversity", {})

        if contributors.get("status") == "concerning":
            narrative_parts.append("Single-contributor dependency presents sustainability risk.")
        elif activity.get("status") == "excellent":
            narrative_parts.append("Strong activity levels indicate healthy development pace.")

        # Risk highlights
        high_risks = [r for r in risk_assessment if r.get("severity") == "high"]
        if high_risks:
            narrative_parts.append(
                f"Critical attention needed for {high_risks[0]['type'].replace('_', ' ')} risk."
            )

        return " ".join(narrative_parts)

    # Developer Analysis Helper Methods

