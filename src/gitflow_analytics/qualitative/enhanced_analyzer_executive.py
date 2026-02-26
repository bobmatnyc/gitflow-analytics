"""Executive analysis mixin for EnhancedQualitativeAnalyzer."""

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




class ExecutiveAnalysisMixin:
    """Mixin: executive summary analysis, achievements, concerns, risks, recommendations."""

    def _analyze_projects(self, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze project-level momentum and health indicators.

        WHY: Project managers need to understand individual project health, momentum,
        and contributor dynamics to make informed resource allocation decisions.
        """
        projects_analysis = {}
        commits_by_project = context["commits_by_project"]

        for project_key, project_commits in commits_by_project.items():
            if not project_commits:
                continue

            # Momentum classification
            momentum = self._classify_project_momentum(project_commits, context)

            # Health indicators
            health_indicators = self._calculate_project_health_indicators(project_commits, context)

            # Technical debt signals
            tech_debt_signals = self._detect_technical_debt_signals(project_commits, context)

            # Delivery predictability
            predictability = self._assess_delivery_predictability(project_commits, context)

            # Risk assessment
            risk_assessment = self._assess_project_risks(project_commits, context)

            # Project-specific recommendations
            recommendations = self._generate_project_recommendations(
                momentum, health_indicators, tech_debt_signals, risk_assessment
            )

            projects_analysis[project_key] = {
                "momentum": momentum,
                "health_indicators": health_indicators,
                "technical_debt_signals": tech_debt_signals,
                "delivery_predictability": predictability,
                "risk_assessment": risk_assessment,
                "recommendations": recommendations,
                "project_narrative": self._generate_project_narrative(
                    project_key, momentum, health_indicators, risk_assessment
                ),
            }

        return projects_analysis

    def _analyze_developers(self, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze individual developer patterns and career development insights.

        WHY: Developers and their managers need insights into contribution patterns,
        growth trajectory, and areas for professional development.
        """
        developers_analysis = {}
        commits_by_developer = context["commits_by_developer"]
        developer_stats = context["developer_stats"]

        # Create developer stats lookup
        dev_stats_by_id = {}
        for dev in developer_stats:
            dev_stats_by_id[dev.get("canonical_id")] = dev

        for dev_id, dev_commits in commits_by_developer.items():
            if not dev_commits:
                continue

            dev_stats = dev_stats_by_id.get(dev_id, {})

            # Contribution pattern analysis
            contribution_pattern = self._analyze_contribution_patterns(dev_commits, context)

            # Collaboration score
            collaboration_score = self._calculate_collaboration_score(dev_commits, context)

            # Expertise domains
            expertise_domains = self._identify_expertise_domains(dev_commits, context)

            # Growth trajectory analysis
            growth_trajectory = self._analyze_growth_trajectory(dev_commits, context)

            # Burnout indicators
            burnout_indicators = self._detect_burnout_indicators(dev_commits, context)

            # Career development recommendations
            career_recommendations = self._generate_career_recommendations(
                contribution_pattern,
                collaboration_score,
                expertise_domains,
                growth_trajectory,
                burnout_indicators,
            )

            developers_analysis[dev_id] = {
                "contribution_pattern": contribution_pattern,
                "collaboration_score": collaboration_score,
                "expertise_domains": expertise_domains,
                "growth_trajectory": growth_trajectory,
                "burnout_indicators": burnout_indicators,
                "career_recommendations": career_recommendations,
                "developer_narrative": self._generate_developer_narrative(
                    dev_stats.get("primary_name", f"Developer {dev_id}"),
                    contribution_pattern,
                    expertise_domains,
                    growth_trajectory,
                ),
            }

        return developers_analysis

    def _analyze_workflow(self, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze workflow effectiveness and Git-PM correlation.

        WHY: Team leads need to understand process effectiveness, identify bottlenecks,
        and optimize workflows for better productivity and quality.
        """
        commits = context["commits"]
        pm_data = context["pm_data"]
        project_metrics = context["project_metrics"]

        # Git-PM correlation effectiveness
        git_pm_effectiveness = self._assess_git_pm_correlation(commits, pm_data, context)

        # Process bottleneck identification
        bottlenecks = self._identify_process_bottlenecks(commits, context)

        # Automation opportunities
        automation_opportunities = self._identify_automation_opportunities(commits, context)

        # Compliance metrics
        compliance_metrics = self._calculate_compliance_metrics(commits, project_metrics, context)

        # Team collaboration patterns
        collaboration_patterns = self._analyze_team_collaboration_patterns(commits, context)

        # Process improvement recommendations
        process_recommendations = self._generate_process_recommendations(
            git_pm_effectiveness, bottlenecks, automation_opportunities, compliance_metrics
        )

        return {
            "git_pm_effectiveness": git_pm_effectiveness,
            "process_bottlenecks": bottlenecks,
            "automation_opportunities": automation_opportunities,
            "compliance_metrics": compliance_metrics,
            "team_collaboration_patterns": collaboration_patterns,
            "process_recommendations": process_recommendations,
            "workflow_narrative": self._generate_workflow_narrative(
                git_pm_effectiveness, bottlenecks, compliance_metrics
            ),
        }

    # Executive Analysis Helper Methods

    def _assess_team_health(self, context: dict[str, Any]) -> tuple[str, float]:
        """Assess overall team health with confidence score."""

        commits = context["commits"]
        developer_stats = context["developer_stats"]
        weeks = context["weeks_analyzed"]

        health_factors = []

        # Activity consistency factor
        weekly_commits = self._get_weekly_commit_counts(commits)
        if weekly_commits:
            consistency = 100 - (
                statistics.pstdev(weekly_commits) / max(statistics.mean(weekly_commits), 1) * 100
            )
            health_factors.append(max(0, min(100, consistency)))

        # Developer engagement factor
        if developer_stats:
            active_developers = sum(
                1
                for dev in developer_stats
                if dev.get("total_commits", 0) > self.thresholds["low_productivity_commits"]
            )
            engagement_score = (active_developers / len(developer_stats)) * 100
            health_factors.append(engagement_score)

        # Velocity factor
        avg_weekly_commits = len(commits) / max(weeks, 1)
        velocity_score = min(100, avg_weekly_commits * 10)  # Scale appropriately
        health_factors.append(velocity_score)

        # Overall health score
        if health_factors:
            overall_score = statistics.mean(health_factors)
            confidence = min(0.95, len(health_factors) / 5.0)  # More factors = higher confidence

            if overall_score >= self.thresholds["health_score_excellent"]:
                return "excellent", confidence
            elif overall_score >= self.thresholds["health_score_good"]:
                return "good", confidence
            elif overall_score >= self.thresholds["health_score_fair"]:
                return "fair", confidence
            else:
                return "needs_improvement", confidence

        return "insufficient_data", 0.2

    def _analyze_velocity_trends(self, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze velocity trends over the analysis period."""

        commits = context["commits"]
        weekly_commits = self._get_weekly_commit_counts(commits)

        if len(weekly_commits) < 4:
            return {
                "trend_direction": "insufficient_data",
                "trend_percentage": 0,
                "weekly_average": 0,
                "confidence": 0.1,
            }

        # Compare first quarter vs last quarter
        quarter_size = len(weekly_commits) // 4
        first_quarter = weekly_commits[:quarter_size] or [0]
        last_quarter = weekly_commits[-quarter_size:] or [0]

        first_avg = statistics.mean(first_quarter)
        last_avg = statistics.mean(last_quarter)

        trend_percentage = (last_avg - first_avg) / first_avg * 100 if first_avg > 0 else 0

        # Determine trend direction
        if abs(trend_percentage) < self.thresholds["velocity_trend_threshold"] * 100:
            trend_direction = "stable"
        elif trend_percentage > 0:
            trend_direction = "improving"
        else:
            trend_direction = "declining"

        # Calculate confidence based on data consistency
        weekly_std = statistics.pstdev(weekly_commits) if len(weekly_commits) > 1 else 0.1
        weekly_mean = statistics.mean(weekly_commits)
        consistency = max(0, 1 - (weekly_std / max(weekly_mean, 0.1)))
        confidence = min(0.95, consistency * 0.8 + 0.2)  # Base confidence + consistency bonus

        return {
            "trend_direction": trend_direction,
            "trend_percentage": round(trend_percentage, 1),
            "weekly_average": round(statistics.mean(weekly_commits), 1),
            "confidence": round(confidence, 2),
        }

    def _identify_key_achievements(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify key achievements during the analysis period."""

        achievements = []
        commits = context["commits"]
        context["developer_stats"]
        project_metrics = context["project_metrics"]

        # High productivity achievement
        total_commits = len(commits)
        if (
            total_commits
            > self.thresholds["high_productivity_commits"] * context["weeks_analyzed"] / 12
        ):
            achievements.append(
                {
                    "category": "productivity",
                    "title": "High Team Productivity",
                    "description": f"Team delivered {total_commits} commits across {context['unique_projects']} projects",
                    "impact": "high",
                    "confidence": 0.9,
                }
            )

        # Consistent delivery achievement
        weekly_commits = self._get_weekly_commit_counts(commits)
        if weekly_commits:
            active_weeks = sum(1 for w in weekly_commits if w > 0)
            consistency_rate = active_weeks / len(weekly_commits)

            if consistency_rate >= self.thresholds["consistent_activity_weeks"]:
                achievements.append(
                    {
                        "category": "consistency",
                        "title": "Consistent Delivery Rhythm",
                        "description": f"Team maintained activity in {active_weeks} of {len(weekly_commits)} weeks",
                        "impact": "medium",
                        "confidence": 0.8,
                    }
                )

        # Cross-project collaboration achievement
        if context["unique_developers"] > 1 and context["unique_projects"] > 2:
            cross_project_devs = 0
            for dev_commits in context["commits_by_developer"].values():
                projects = set(c.get("project_key", "UNKNOWN") for c in dev_commits)
                if len(projects) > 1:
                    cross_project_devs += 1

            if cross_project_devs > context["unique_developers"] * 0.5:
                achievements.append(
                    {
                        "category": "collaboration",
                        "title": "Strong Cross-Project Collaboration",
                        "description": f"{cross_project_devs} developers contributed to multiple projects",
                        "impact": "medium",
                        "confidence": 0.7,
                    }
                )

        # Ticket coverage achievement
        ticket_analysis = project_metrics.get("ticket_analysis", {})
        ticket_coverage = ticket_analysis.get("commit_coverage_pct", 0)
        if ticket_coverage >= self.thresholds["ticket_coverage_excellent"]:
            achievements.append(
                {
                    "category": "process",
                    "title": "Excellent Process Adherence",
                    "description": f"{ticket_coverage:.1f}% of commits properly linked to tickets",
                    "impact": "high",
                    "confidence": 0.9,
                }
            )

        return achievements

    def _identify_major_concerns(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify major concerns that need executive attention."""

        concerns = []
        context["commits"]
        developer_stats = context["developer_stats"]
        project_metrics = context["project_metrics"]

        # Bus factor concern (contribution concentration)
        if developer_stats and len(developer_stats) > 1:
            commit_counts = [dev.get("total_commits", 0) for dev in developer_stats]
            gini_coefficient = self._calculate_gini_coefficient(commit_counts)

            if gini_coefficient > self.thresholds["bus_factor_threshold"]:
                top_contributor = max(developer_stats, key=lambda x: x.get("total_commits", 0))
                top_percentage = (
                    top_contributor.get("total_commits", 0) / sum(commit_counts)
                ) * 100

                concerns.append(
                    {
                        "category": "risk",
                        "title": "High Bus Factor Risk",
                        "description": f"Work highly concentrated: top contributor handles {top_percentage:.1f}% of commits",
                        "severity": "high",
                        "impact": "critical",
                        "confidence": 0.9,
                        "recommendation": "Distribute knowledge and responsibilities more evenly across team",
                    }
                )

        # Declining velocity concern
        velocity_trends = self._analyze_velocity_trends(context)
        if (
            velocity_trends["trend_direction"] == "declining"
            and velocity_trends["trend_percentage"] < -20
        ):
            concerns.append(
                {
                    "category": "productivity",
                    "title": "Declining Team Velocity",
                    "description": f"Commit velocity declined by {abs(velocity_trends['trend_percentage']):.1f}% over analysis period",
                    "severity": "high",
                    "impact": "high",
                    "confidence": velocity_trends["confidence"],
                    "recommendation": "Investigate productivity bottlenecks and team capacity issues",
                }
            )

        # Poor ticket coverage concern
        ticket_analysis = project_metrics.get("ticket_analysis", {})
        ticket_coverage = ticket_analysis.get("commit_coverage_pct", 0)
        if ticket_coverage < self.thresholds["ticket_coverage_poor"]:
            concerns.append(
                {
                    "category": "process",
                    "title": "Poor Process Adherence",
                    "description": f"Only {ticket_coverage:.1f}% of commits linked to tickets",
                    "severity": "medium",
                    "impact": "medium",
                    "confidence": 0.8,
                    "recommendation": "Implement better ticket referencing practices and training",
                }
            )

        # Inactive developer concern
        if developer_stats:
            inactive_devs = sum(
                1
                for dev in developer_stats
                if dev.get("total_commits", 0) < self.thresholds["low_productivity_commits"]
            )

            if inactive_devs > len(developer_stats) * 0.3:
                concerns.append(
                    {
                        "category": "team",
                        "title": "Team Engagement Issues",
                        "description": f"{inactive_devs} of {len(developer_stats)} developers have minimal activity",
                        "severity": "medium",
                        "impact": "medium",
                        "confidence": 0.7,
                        "recommendation": "Review individual workloads and engagement levels",
                    }
                )

        return concerns

    def _assess_risk_indicators(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Assess various risk indicators for the team and projects."""

        risk_indicators = []
        commits = context["commits"]

        # Large commit size risk
        large_commits = sum(
            1
            for c in commits
            if (
                c.get("filtered_insertions", c.get("insertions", 0))
                + c.get("filtered_deletions", c.get("deletions", 0))
            )
            > self.thresholds["large_commit_lines"]
        )

        if large_commits > len(commits) * 0.2:  # More than 20% large commits
            risk_indicators.append(
                {
                    "type": "code_quality",
                    "title": "Large Commit Pattern",
                    "description": f"{large_commits} commits exceed {self.thresholds['large_commit_lines']} lines",
                    "risk_level": "medium",
                    "impact": "Code review difficulty, potential bugs",
                    "confidence": 0.8,
                }
            )

        # Weekend work pattern risk
        weekend_commits = 0
        for commit in commits:
            if hasattr(commit.get("timestamp"), "weekday") and commit["timestamp"].weekday() >= 5:
                weekend_commits += 1

        weekend_percentage = (weekend_commits / len(commits)) * 100 if commits else 0
        if weekend_percentage > 30:  # More than 30% weekend work
            risk_indicators.append(
                {
                    "type": "work_life_balance",
                    "title": "High Weekend Activity",
                    "description": f"{weekend_percentage:.1f}% of commits made on weekends",
                    "risk_level": "medium",
                    "impact": "Potential burnout, work-life balance issues",
                    "confidence": 0.7,
                }
            )

        return risk_indicators

    def _generate_executive_recommendations(
        self,
        health_assessment: str,
        velocity_trends: dict[str, Any],
        concerns: list[dict[str, Any]],
        risk_indicators: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate strategic recommendations for executive leadership."""

        recommendations = []

        # Health-based recommendations
        if health_assessment in ["needs_improvement", "fair"]:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "team_health",
                    "title": "Improve Team Health Metrics",
                    "action": "Focus on consistency, engagement, and velocity improvements",
                    "timeline": "1-2 quarters",
                    "expected_impact": "Improved productivity and team morale",
                }
            )

        # Velocity-based recommendations
        if velocity_trends["trend_direction"] == "declining":
            recommendations.append(
                {
                    "priority": "high",
                    "category": "productivity",
                    "title": "Address Velocity Decline",
                    "action": "Investigate bottlenecks and optimize development processes",
                    "timeline": "4-6 weeks",
                    "expected_impact": "Restored or improved delivery velocity",
                }
            )

        # Risk-based recommendations
        high_severity_concerns = [c for c in concerns if c.get("severity") == "high"]
        if high_severity_concerns:
            recommendations.append(
                {
                    "priority": "critical",
                    "category": "risk_mitigation",
                    "title": "Address Critical Risk Factors",
                    "action": f"Immediate attention needed for {len(high_severity_concerns)} high-severity issues",
                    "timeline": "2-4 weeks",
                    "expected_impact": "Reduced project risk and improved stability",
                }
            )

        # Process improvement recommendation
        if any(c.get("category") == "process" for c in concerns):
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "process",
                    "title": "Strengthen Development Processes",
                    "action": "Implement better tracking, documentation, and compliance practices",
                    "timeline": "6-8 weeks",
                    "expected_impact": "Improved visibility and process adherence",
                }
            )

        return recommendations[:5]  # Top 5 recommendations

    def _generate_executive_narrative(
        self,
        health_assessment: str,
        velocity_trends: dict[str, Any],
        achievements: list[dict[str, Any]],
        concerns: list[dict[str, Any]],
    ) -> str:
        """Generate executive narrative summary."""

        narrative_parts = []

        # Health assessment
        health_descriptions = {
            "excellent": "operating at peak performance with strong metrics across all dimensions",
            "good": "performing well with room for targeted improvements",
            "fair": "showing mixed results requiring focused attention",
            "needs_improvement": "facing significant challenges requiring immediate intervention",
        }

        narrative_parts.append(
            f"The development team is currently {health_descriptions.get(health_assessment, 'in an unclear state')}."
        )

        # Velocity trends
        if velocity_trends["trend_direction"] == "improving":
            narrative_parts.append(
                f"Team velocity is trending upward with a {velocity_trends['trend_percentage']:.1f}% improvement, averaging {velocity_trends['weekly_average']} commits per week."
            )
        elif velocity_trends["trend_direction"] == "declining":
            narrative_parts.append(
                f"Team velocity shows concerning decline of {abs(velocity_trends['trend_percentage']):.1f}%, requiring immediate attention to restore productivity."
            )
        else:
            narrative_parts.append(
                f"Team velocity remains stable at {velocity_trends['weekly_average']} commits per week, providing consistent delivery rhythm."
            )

        # Key achievements
        if achievements:
            high_impact_achievements = [a for a in achievements if a.get("impact") == "high"]
            if high_impact_achievements:
                narrative_parts.append(
                    f"Notable achievements include {', '.join([a['title'].lower() for a in high_impact_achievements[:2]])}."
                )

        # Major concerns
        critical_concerns = [c for c in concerns if c.get("severity") == "high"]
        if critical_concerns:
            narrative_parts.append(
                f"Critical attention needed for {critical_concerns[0]['title'].lower()} and other high-priority issues."
            )
        elif concerns:
            narrative_parts.append(
                f"Some areas require monitoring, particularly {concerns[0]['category']} aspects."
            )

        return " ".join(narrative_parts)

    # Project Analysis Helper Methods

