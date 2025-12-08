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


class EnhancedQualitativeAnalyzer:
    """Enhanced qualitative analyzer providing specialized analysis across four dimensions.

    This analyzer processes quantitative commit data and generates qualitative insights
    across executive, project, developer, and workflow dimensions. Each analysis includes
    confidence scores, risk assessments, and actionable recommendations.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize the enhanced analyzer.

        Args:
            config: Configuration dictionary with analysis thresholds and parameters
        """
        self.config = config or {}

        # Analysis thresholds and parameters
        self.thresholds = {
            "high_productivity_commits": 50,  # Commits per analysis period
            "low_productivity_commits": 5,  # Minimum meaningful activity
            "high_collaboration_projects": 3,  # Projects for versatility
            "consistent_activity_weeks": 0.7,  # Percentage of weeks active
            "large_commit_lines": 300,  # Lines changed threshold
            "critical_risk_score": 0.8,  # Risk level for critical issues
            "velocity_trend_threshold": 0.2,  # 20% change for significant trend
            "health_score_excellent": 80,  # Health score thresholds
            "health_score_good": 60,
            "health_score_fair": 40,
            "bus_factor_threshold": 0.7,  # Contribution concentration limit
            "ticket_coverage_excellent": 80,  # Ticket linking thresholds
            "ticket_coverage_poor": 30,
        }

        # Update thresholds from config
        if "analysis_thresholds" in self.config:
            self.thresholds.update(self.config["analysis_thresholds"])

        self.metrics = PerformanceMetrics()
        self.logger = logging.getLogger(__name__)

    def analyze_comprehensive(
        self,
        commits: list[dict[str, Any]],
        qualitative_data: Optional[list[QualitativeCommitData]] = None,
        developer_stats: Optional[list[dict[str, Any]]] = None,
        project_metrics: Optional[dict[str, Any]] = None,
        pm_data: Optional[dict[str, Any]] = None,
        weeks_analyzed: int = 12,
    ) -> dict[str, Any]:
        """Perform comprehensive enhanced qualitative analysis.

        Args:
            commits: List of commit data from GitFlow Analytics
            qualitative_data: Optional qualitative commit analysis results
            developer_stats: Optional developer statistics
            project_metrics: Optional project-level metrics
            pm_data: Optional PM platform integration data
            weeks_analyzed: Number of weeks in analysis period

        Returns:
            Dictionary containing all four analysis dimensions
        """
        self.logger.info(f"Starting enhanced qualitative analysis of {len(commits)} commits")

        # Prepare unified data structures
        analysis_context = self._prepare_analysis_context(
            commits, qualitative_data, developer_stats, project_metrics, pm_data, weeks_analyzed
        )

        # Perform four-dimensional analysis
        executive_analysis = self._analyze_executive_summary(analysis_context)
        project_analysis = self._analyze_projects(analysis_context)
        developer_analysis = self._analyze_developers(analysis_context)
        workflow_analysis = self._analyze_workflow(analysis_context)

        # Cross-reference insights for consistency
        comprehensive_analysis = {
            "metadata": {
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "commits_analyzed": len(commits),
                "weeks_analyzed": weeks_analyzed,
                "analysis_version": "2.0.0",
            },
            "executive_analysis": executive_analysis,
            "project_analysis": project_analysis,
            "developer_analysis": developer_analysis,
            "workflow_analysis": workflow_analysis,
            "cross_insights": self._generate_cross_insights(
                executive_analysis, project_analysis, developer_analysis, workflow_analysis
            ),
        }

        self.logger.info("Enhanced qualitative analysis completed")
        return comprehensive_analysis

    def _prepare_analysis_context(
        self,
        commits: list[dict[str, Any]],
        qualitative_data: Optional[list[QualitativeCommitData]],
        developer_stats: Optional[list[dict[str, Any]]],
        project_metrics: Optional[dict[str, Any]],
        pm_data: Optional[dict[str, Any]],
        weeks_analyzed: int,
    ) -> dict[str, Any]:
        """Prepare unified analysis context with all available data."""

        # Process commits data
        commits_by_project = defaultdict(list)
        commits_by_developer = defaultdict(list)

        for commit in commits:
            project_key = commit.get("project_key", "UNKNOWN")
            dev_id = commit.get("canonical_id", commit.get("author_email"))

            commits_by_project[project_key].append(commit)
            commits_by_developer[dev_id].append(commit)

        # Calculate time periods
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks_analyzed)

        # Prepare qualitative mapping
        qualitative_by_hash = {}
        if qualitative_data:
            # Handle both QualitativeCommitData objects and dictionaries
            qualitative_by_hash = {}
            for q in qualitative_data:
                if hasattr(q, "hash"):
                    # QualitativeCommitData object
                    qualitative_by_hash[q.hash] = q
                elif isinstance(q, dict) and "hash" in q:
                    # Dictionary format
                    qualitative_by_hash[q["hash"]] = q
                else:
                    # Skip invalid entries
                    self.logger.warning(f"Invalid qualitative data format: {type(q)}")

        return {
            "commits": commits,
            "commits_by_project": dict(commits_by_project),
            "commits_by_developer": dict(commits_by_developer),
            "qualitative_data": qualitative_by_hash,
            "developer_stats": developer_stats or [],
            "project_metrics": project_metrics or {},
            "pm_data": pm_data or {},
            "weeks_analyzed": weeks_analyzed,
            "analysis_period": {"start_date": start_date, "end_date": end_date},
            "total_commits": len(commits),
            "unique_projects": len(commits_by_project),
            "unique_developers": len(commits_by_developer),
        }

    def _analyze_executive_summary(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate executive-level analysis with strategic insights.

        WHY: Executives need high-level health assessment, trend analysis, and risk indicators
        without getting lost in technical details. This analysis focuses on team productivity,
        velocity trends, and strategic recommendations.
        """
        context["commits"]
        context["total_commits"]
        context["weeks_analyzed"]

        # Overall team health assessment
        health_assessment, health_confidence = self._assess_team_health(context)

        # Velocity trend analysis
        velocity_trends = self._analyze_velocity_trends(context)

        # Key achievements identification
        achievements = self._identify_key_achievements(context)

        # Major concerns and risks
        concerns = self._identify_major_concerns(context)

        # Risk indicators
        risk_indicators = self._assess_risk_indicators(context)

        # Strategic recommendations
        recommendations = self._generate_executive_recommendations(
            health_assessment, velocity_trends, concerns, risk_indicators
        )

        return {
            "health_assessment": health_assessment,
            "health_confidence": health_confidence,
            "velocity_trends": {
                "overall_trend": velocity_trends["trend_direction"],
                "trend_percentage": velocity_trends["trend_percentage"],
                "weekly_average": velocity_trends["weekly_average"],
                "trend_confidence": velocity_trends["confidence"],
            },
            "key_achievements": achievements,
            "major_concerns": concerns,
            "risk_indicators": risk_indicators,
            "recommendations": recommendations,
            "executive_summary": self._generate_executive_narrative(
                health_assessment, velocity_trends, achievements, concerns
            ),
        }

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

    def _assess_git_pm_correlation(
        self, commits: list[dict[str, Any]], pm_data: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess effectiveness of Git-PM platform correlation."""

        if not pm_data or not pm_data.get("correlations"):
            return {
                "effectiveness": "no_integration",
                "description": "No PM platform integration detected",
                "score": 0,
                "confidence": 0.9,
            }

        correlations = pm_data.get("correlations", [])
        total_correlations = len(correlations)

        # Analyze correlation quality
        high_confidence = sum(1 for c in correlations if c.get("confidence", 0) > 0.8)
        medium_confidence = sum(1 for c in correlations if 0.5 <= c.get("confidence", 0) <= 0.8)

        # Calculate effectiveness score
        if total_correlations > 0:
            quality_score = (
                (high_confidence * 1.0 + medium_confidence * 0.6) / total_correlations * 100
            )
        else:
            quality_score = 0

        # Determine effectiveness level
        if quality_score >= 80:
            effectiveness = "highly_effective"
        elif quality_score >= 60:
            effectiveness = "moderately_effective"
        elif quality_score >= 40:
            effectiveness = "partially_effective"
        else:
            effectiveness = "ineffective"

        return {
            "effectiveness": effectiveness,
            "description": f"{effectiveness.replace('_', ' ').title()} with {quality_score:.1f}% correlation quality",
            "score": round(quality_score, 1),
            "confidence": 0.8,
            "correlation_breakdown": {
                "total": total_correlations,
                "high_confidence": high_confidence,
                "medium_confidence": medium_confidence,
                "low_confidence": total_correlations - high_confidence - medium_confidence,
            },
        }

    def _identify_process_bottlenecks(
        self, commits: list[dict[str, Any]], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Identify potential process bottlenecks."""

        bottlenecks = []

        # Large commit bottleneck
        large_commits = sum(
            1
            for c in commits
            if (
                c.get("filtered_insertions", c.get("insertions", 0))
                + c.get("filtered_deletions", c.get("deletions", 0))
            )
            > self.thresholds["large_commit_lines"]
        )

        if large_commits > len(commits) * 0.25:  # More than 25% large commits
            bottlenecks.append(
                {
                    "type": "large_commit_pattern",
                    "severity": "medium",
                    "description": f"{large_commits} commits exceed {self.thresholds['large_commit_lines']} lines",
                    "impact": "Slower code reviews, increased merge conflicts",
                    "recommendation": "Implement commit size guidelines and encourage smaller, focused changes",
                }
            )

        # Irregular commit timing bottleneck
        daily_commits = defaultdict(int)
        for commit in commits:
            day_key = commit["timestamp"].strftime("%Y-%m-%d")
            daily_commits[day_key] += 1

        daily_values = list(daily_commits.values())
        if daily_values and len(daily_values) > 7:
            daily_std = statistics.pstdev(daily_values)
            daily_mean = statistics.mean(daily_values)

            if daily_std > daily_mean:  # High variability
                bottlenecks.append(
                    {
                        "type": "irregular_development_rhythm",
                        "severity": "low",
                        "description": "Highly variable daily commit patterns",
                        "impact": "Unpredictable integration and review workload",
                        "recommendation": "Encourage more consistent development and integration practices",
                    }
                )

        # Ticket linking bottleneck
        commits_with_tickets = sum(1 for c in commits if c.get("ticket_references"))
        ticket_coverage = (commits_with_tickets / len(commits)) * 100 if commits else 0

        if ticket_coverage < self.thresholds["ticket_coverage_poor"]:
            bottlenecks.append(
                {
                    "type": "poor_ticket_linking",
                    "severity": "medium",
                    "description": f"Only {ticket_coverage:.1f}% of commits reference tickets",
                    "impact": "Poor traceability and project management visibility",
                    "recommendation": "Implement mandatory ticket referencing and provide training",
                }
            )

        return bottlenecks

    def _identify_automation_opportunities(
        self, commits: list[dict[str, Any]], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Identify opportunities for process automation."""

        opportunities = []

        # Repetitive commit message patterns
        message_patterns = defaultdict(int)
        for commit in commits:
            # Simplified pattern detection - look for common prefixes
            message = commit.get("message", "").lower()
            words = message.split()
            if words:
                first_word = words[0]
                if first_word in ["fix", "update", "add", "remove", "refactor"]:
                    message_patterns[first_word] += 1

        total_commits = len(commits)
        for pattern, count in message_patterns.items():
            percentage = (count / total_commits) * 100
            if percentage > 30:  # More than 30% of commits follow this pattern
                opportunities.append(
                    {
                        "type": "commit_message_templates",
                        "description": f"{percentage:.1f}% of commits start with '{pattern}'",
                        "potential": "Implement commit message templates and validation",
                        "effort": "low",
                        "impact": "medium",
                    }
                )

        # Regular fix patterns suggesting test automation needs
        fix_commits = sum(
            1
            for c in commits
            if any(keyword in c.get("message", "").lower() for keyword in ["fix", "bug", "hotfix"])
        )
        fix_percentage = (fix_commits / total_commits) * 100 if total_commits else 0

        if fix_percentage > 25:
            opportunities.append(
                {
                    "type": "automated_testing",
                    "description": f"{fix_percentage:.1f}% of commits are fixes",
                    "potential": "Implement comprehensive automated testing to catch issues earlier",
                    "effort": "high",
                    "impact": "high",
                }
            )

        # Deployment frequency analysis
        deploy_keywords = ["deploy", "release", "version"]
        deploy_commits = sum(
            1
            for c in commits
            if any(keyword in c.get("message", "").lower() for keyword in deploy_keywords)
        )

        weeks_analyzed = context["weeks_analyzed"]
        deploy_frequency = deploy_commits / max(weeks_analyzed, 1)

        if deploy_frequency < 0.5:  # Less than 0.5 deploys per week
            opportunities.append(
                {
                    "type": "continuous_deployment",
                    "description": f"Low deployment frequency: {deploy_frequency:.1f} per week",
                    "potential": "Implement continuous deployment pipeline",
                    "effort": "high",
                    "impact": "high",
                }
            )

        return opportunities

    def _calculate_compliance_metrics(
        self,
        commits: list[dict[str, Any]],
        project_metrics: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate various compliance and process adherence metrics."""

        total_commits = len(commits)

        # Ticket coverage compliance
        commits_with_tickets = sum(1 for c in commits if c.get("ticket_references"))
        ticket_coverage = (commits_with_tickets / total_commits) * 100 if total_commits else 0

        # Commit message quality compliance
        descriptive_messages = sum(
            1 for c in commits if len(c.get("message", "").split()) >= 3
        )  # At least 3 words
        message_quality = (descriptive_messages / total_commits) * 100 if total_commits else 0

        # Size compliance (reasonable commit sizes)
        appropriate_size_commits = sum(
            1
            for c in commits
            if 10
            <= (
                c.get("filtered_insertions", c.get("insertions", 0))
                + c.get("filtered_deletions", c.get("deletions", 0))
            )
            <= 300
        )
        size_compliance = (appropriate_size_commits / total_commits) * 100 if total_commits else 0

        # PR approval compliance (if PR data available - placeholder)
        pr_approval_rate = 75  # Default assumption when PR data not available

        # Overall compliance score
        compliance_factors = [ticket_coverage, message_quality, size_compliance, pr_approval_rate]
        overall_compliance = statistics.mean(compliance_factors)

        return {
            "overall_score": round(overall_compliance, 1),
            "ticket_coverage": {
                "score": round(ticket_coverage, 1),
                "status": (
                    "excellent"
                    if ticket_coverage >= 80
                    else "good"
                    if ticket_coverage >= 60
                    else "needs_improvement"
                ),
            },
            "message_quality": {
                "score": round(message_quality, 1),
                "status": (
                    "excellent"
                    if message_quality >= 80
                    else "good"
                    if message_quality >= 60
                    else "needs_improvement"
                ),
            },
            "commit_size_compliance": {
                "score": round(size_compliance, 1),
                "status": (
                    "excellent"
                    if size_compliance >= 80
                    else "good"
                    if size_compliance >= 60
                    else "needs_improvement"
                ),
            },
            "pr_approval_rate": {"score": pr_approval_rate, "status": "good"},  # Placeholder
        }

    def _analyze_team_collaboration_patterns(
        self, commits: list[dict[str, Any]], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze team collaboration patterns."""

        # Cross-project collaboration analysis
        developer_projects = defaultdict(set)
        for commit in commits:
            dev_id = commit.get("canonical_id", commit.get("author_email"))
            project_key = commit.get("project_key", "UNKNOWN")
            developer_projects[dev_id].add(project_key)

        # Count cross-project contributors
        cross_project_devs = sum(1 for projects in developer_projects.values() if len(projects) > 1)
        total_devs = len(developer_projects)
        cross_collaboration_rate = (cross_project_devs / total_devs) * 100 if total_devs else 0

        # Project contributor diversity
        project_contributors = defaultdict(set)
        for commit in commits:
            dev_id = commit.get("canonical_id", commit.get("author_email"))
            project_key = commit.get("project_key", "UNKNOWN")
            project_contributors[project_key].add(dev_id)

        avg_contributors_per_project = (
            statistics.mean([len(contributors) for contributors in project_contributors.values()])
            if project_contributors
            else 0
        )

        # Collaboration score calculation
        collaboration_factors = [
            min(100, cross_collaboration_rate * 2),  # Cross-project work
            min(100, avg_contributors_per_project * 25),  # Multi-contributor projects
        ]

        collaboration_score = statistics.mean(collaboration_factors)

        return {
            "collaboration_score": round(collaboration_score, 1),
            "cross_project_contributors": cross_project_devs,
            "cross_collaboration_rate": round(cross_collaboration_rate, 1),
            "avg_contributors_per_project": round(avg_contributors_per_project, 1),
            "collaboration_level": (
                "high"
                if collaboration_score >= 70
                else "medium"
                if collaboration_score >= 40
                else "low"
            ),
            "patterns": {
                "multi_project_engagement": cross_collaboration_rate >= 50,
                "team_project_distribution": avg_contributors_per_project >= 2,
                "siloed_development": cross_collaboration_rate < 20,
            },
        }

    def _generate_process_recommendations(
        self,
        git_pm_effectiveness: dict[str, Any],
        bottlenecks: list[dict[str, Any]],
        automation_opportunities: list[dict[str, Any]],
        compliance_metrics: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate process improvement recommendations."""

        recommendations = []

        # Git-PM integration recommendations
        effectiveness = git_pm_effectiveness.get("effectiveness", "")
        if effectiveness in ["ineffective", "partially_effective"]:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "integration",
                    "title": "Improve Git-PM Integration",
                    "action": "Enhance ticket referencing and correlation accuracy",
                    "timeline": "4-6 weeks",
                    "expected_impact": "Better project visibility and tracking",
                }
            )

        # Bottleneck recommendations
        high_severity_bottlenecks = [b for b in bottlenecks if b.get("severity") == "high"]
        if high_severity_bottlenecks:
            bottleneck = high_severity_bottlenecks[0]
            recommendations.append(
                {
                    "priority": "high",
                    "category": "process_optimization",
                    "title": f"Address {bottleneck['type'].replace('_', ' ').title()}",
                    "action": bottleneck.get("recommendation", "Address identified bottleneck"),
                    "timeline": "2-4 weeks",
                    "expected_impact": bottleneck.get("impact", "Improved process efficiency"),
                }
            )

        # Automation recommendations
        high_impact_automation = [a for a in automation_opportunities if a.get("impact") == "high"]
        if high_impact_automation:
            automation = high_impact_automation[0]
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "automation",
                    "title": f"Implement {automation['type'].replace('_', ' ').title()}",
                    "action": automation["potential"],
                    "timeline": "6-12 weeks" if automation.get("effort") == "high" else "2-4 weeks",
                    "expected_impact": "Reduced manual effort and improved consistency",
                }
            )

        # Compliance recommendations
        overall_compliance = compliance_metrics.get("overall_score", 0)
        if overall_compliance < 70:
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "compliance",
                    "title": "Improve Process Compliance",
                    "action": "Focus on ticket linking, commit message quality, and size guidelines",
                    "timeline": "4-8 weeks",
                    "expected_impact": "Better process adherence and project visibility",
                }
            )

        return recommendations[:4]  # Top 4 recommendations

    def _generate_workflow_narrative(
        self,
        git_pm_effectiveness: dict[str, Any],
        bottlenecks: list[dict[str, Any]],
        compliance_metrics: dict[str, Any],
    ) -> str:
        """Generate workflow analysis narrative."""

        narrative_parts = []

        # Git-PM effectiveness
        git_pm_effectiveness.get("effectiveness", "unknown")
        effectiveness_desc = git_pm_effectiveness.get("description", "integration status unclear")
        narrative_parts.append(f"Git-PM platform integration is {effectiveness_desc.lower()}.")

        # Process health
        compliance_score = compliance_metrics.get("overall_score", 0)
        if compliance_score >= 80:
            narrative_parts.append("Development processes show strong compliance and adherence.")
        elif compliance_score >= 60:
            narrative_parts.append(
                "Development processes are generally well-followed with room for improvement."
            )
        else:
            narrative_parts.append(
                "Development processes need attention to improve compliance and effectiveness."
            )

        # Bottleneck highlights
        high_severity_bottlenecks = [b for b in bottlenecks if b.get("severity") == "high"]
        if high_severity_bottlenecks:
            narrative_parts.append(
                f"Critical bottleneck identified: {high_severity_bottlenecks[0]['type'].replace('_', ' ')}."
            )
        elif bottlenecks:
            narrative_parts.append(
                f"Some process inefficiencies detected, particularly in {bottlenecks[0]['type'].replace('_', ' ')}."
            )

        return " ".join(narrative_parts)

    # Cross-Analysis Helper Methods

    def _generate_cross_insights(
        self,
        executive_analysis: dict[str, Any],
        project_analysis: dict[str, Any],
        developer_analysis: dict[str, Any],
        workflow_analysis: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate insights that span multiple analysis dimensions."""

        cross_insights = []

        # Executive-Project alignment insight
        exec_health = executive_analysis.get("health_assessment", "unknown")
        project_health_scores = []
        for project_data in project_analysis.values():
            health_indicators = project_data.get("health_indicators", {})
            overall_health = health_indicators.get("overall_health", {})
            score = overall_health.get("score", 0)
            project_health_scores.append(score)

        if project_health_scores:
            avg_project_health = statistics.mean(project_health_scores)
            if exec_health == "excellent" and avg_project_health < 60:
                cross_insights.append(
                    {
                        "type": "alignment_mismatch",
                        "title": "Executive-Project Health Disconnect",
                        "description": "Overall team health excellent but individual projects show concerns",
                        "priority": "medium",
                        "recommendation": "Investigate project-specific issues that may not be visible at team level",
                    }
                )

        # Developer-Workflow correlation insight
        high_burnout_devs = 0
        for dev_data in developer_analysis.values():
            burnout_indicators = dev_data.get("burnout_indicators", [])
            if len(burnout_indicators) >= 2:
                high_burnout_devs += 1

        workflow_bottlenecks = workflow_analysis.get("process_bottlenecks", [])
        if high_burnout_devs > 0 and len(workflow_bottlenecks) > 2:
            cross_insights.append(
                {
                    "type": "systemic_issue",
                    "title": "Process Issues Contributing to Developer Stress",
                    "description": f"{high_burnout_devs} developers show burnout indicators alongside {len(workflow_bottlenecks)} process bottlenecks",
                    "priority": "high",
                    "recommendation": "Address workflow inefficiencies to reduce developer burden",
                }
            )

        # Project-Developer resource allocation insight
        declining_projects = sum(
            1
            for p in project_analysis.values()
            if p.get("momentum", {}).get("classification") == "declining"
        )
        declining_developers = sum(
            1
            for d in developer_analysis.values()
            if d.get("growth_trajectory", {}).get("trajectory") == "declining"
        )

        if declining_projects > 0 and declining_developers > 0:
            cross_insights.append(
                {
                    "type": "resource_allocation",
                    "title": "Coordinated Decline Pattern",
                    "description": f"{declining_projects} projects and {declining_developers} developers showing decline",
                    "priority": "high",
                    "recommendation": "Review resource allocation and team capacity planning",
                }
            )

        return cross_insights

    # Utility Helper Methods

    def _get_weekly_commit_counts(self, commits: list[dict[str, Any]]) -> list[int]:
        """Get commit counts grouped by week."""

        if not commits:
            return []

        weekly_counts = defaultdict(int)
        for commit in commits:
            week_start = self._get_week_start(commit["timestamp"])
            week_key = week_start.strftime("%Y-%m-%d")
            weekly_counts[week_key] += 1

        # Return counts in chronological order
        sorted_weeks = sorted(weekly_counts.keys())
        return [weekly_counts[week] for week in sorted_weeks]

    def _get_week_start(self, date: datetime) -> datetime:
        """Get Monday of the week for a given date."""

        # Ensure timezone consistency
        if hasattr(date, "tzinfo") and date.tzinfo is not None:
            if date.tzinfo != timezone.utc:
                date = date.astimezone(timezone.utc)
        else:
            date = date.replace(tzinfo=timezone.utc)

        days_since_monday = date.weekday()
        monday = date - timedelta(days=days_since_monday)
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)

    def _calculate_gini_coefficient(self, values: list[float]) -> float:
        """Calculate Gini coefficient for measuring inequality."""

        if not values or len(values) == 1:
            return 0.0

        sorted_values = sorted([v for v in values if v > 0])  # Filter out zeros
        if not sorted_values:
            return 0.0

        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)

        return (2 * np.sum((i + 1) * sorted_values[i] for i in range(n))) / (n * cumsum[-1]) - (
            n + 1
        ) / n
