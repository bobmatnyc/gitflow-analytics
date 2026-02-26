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



from .enhanced_analyzer_executive import ExecutiveAnalysisMixin
from .enhanced_analyzer_projects import ProjectAnalysisMixin
from .enhanced_analyzer_developer import DeveloperAnalysisMixin
from .enhanced_analyzer_workflow import WorkflowAnalysisMixin


class EnhancedQualitativeAnalyzer(
    ExecutiveAnalysisMixin,
    ProjectAnalysisMixin,
    DeveloperAnalysisMixin,
    WorkflowAnalysisMixin,
):
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

