"""Comprehensive JSON export system for GitFlow Analytics.

This module provides a comprehensive JSON export system that consolidates all report data
into a single structured JSON format optimized for web consumption and API integration.

WHY: Traditional CSV reports are excellent for analysis tools but lack the structure needed
for modern web applications and dashboards. This JSON exporter creates a self-contained,
hierarchical data structure that includes:
- Time series data for charts
- Cross-references between entities
- Anomaly detection and trend analysis
- Health scores and insights
- All existing report data in a unified format

DESIGN DECISIONS:
- Self-contained: All data needed for visualization is included
- Hierarchical: Supports drill-down from executive summary to detailed metrics
- Web-optimized: Compatible with common charting libraries (Chart.js, D3, etc.)
- Extensible: Easy to add new metrics and dimensions
- Consistent: Follows established patterns from existing report generators
"""

import json
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .base import BaseReportGenerator, ReportData, ReportOutput
from .interfaces import ReportFormat
from .json_export_builders import JSONExportBuildersMixin
from .json_export_executive import JSONExportExecutiveMixin
from .json_export_developer import JSONExportDeveloperMixin
from .json_export_insights import JSONExportInsightsMixin
from .json_export_utils import JSONExportUtilsMixin

# Get logger for this module
logger = logging.getLogger(__name__)


class ComprehensiveJSONExporter(
    JSONExportBuildersMixin,
    JSONExportExecutiveMixin,
    JSONExportDeveloperMixin,
    JSONExportInsightsMixin,
    JSONExportUtilsMixin,
    BaseReportGenerator,
):
    """Generate comprehensive JSON exports with advanced analytics and insights.

    This exporter consolidates all GitFlow Analytics data into a single, structured
    JSON format that's optimized for web consumption and includes:

    - Executive summary with key metrics and trends
    - Project-level data with health scores
    - Developer profiles with contribution patterns
    - Time series data for visualization
    - Anomaly detection and alerting
    - Cross-references between entities
    """

    def __init__(self, anonymize: bool = False, **kwargs):
        """Initialize the comprehensive JSON exporter.

        Args:
            anonymize: Whether to anonymize developer information
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(anonymize=anonymize, **kwargs)
        # Note: anonymization map and counter are now in base class

        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            "spike_multiplier": 2.0,  # 2x normal activity = spike
            "drop_threshold": 0.3,  # 30% of normal activity = drop
            "volatility_threshold": 1.5,  # Standard deviation threshold
            "trend_threshold": 0.2,  # 20% change = significant trend
        }

        # Health score weights
        self.health_weights = {
            "activity_consistency": 0.3,
            "ticket_coverage": 0.25,
            "collaboration": 0.2,
            "code_quality": 0.15,
            "velocity": 0.1,
        }

    def export_comprehensive_data(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        project_metrics: Dict[str, Any],
        dora_metrics: Dict[str, Any],
        output_path: Path,
        weeks: int = 12,
        pm_data: Optional[Dict[str, Any]] = None,
        qualitative_data: Optional[List[Dict[str, Any]]] = None,
        enhanced_qualitative_analysis: Optional[Dict[str, Any]] = None,
        cicd_data: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Export comprehensive analytics data to JSON format.

        Args:
            commits: List of commit data
            prs: List of pull request data
            developer_stats: Developer statistics
            project_metrics: Project-level metrics
            dora_metrics: DORA metrics data
            output_path: Path to write JSON file
            weeks: Number of weeks analyzed
            pm_data: PM platform integration data
            qualitative_data: Qualitative analysis results
            enhanced_qualitative_analysis: Enhanced multi-dimensional qualitative analysis
            cicd_data: CI/CD pipeline data with metrics

        Returns:
            Path to the generated JSON file
        """
        logger.info(f"Starting comprehensive JSON export with {len(commits)} commits")

        # Calculate analysis period
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)

        # Build comprehensive data structure
        export_data = {
            "metadata": self._build_metadata(commits, prs, developer_stats, start_date, end_date),
            "executive_summary": self._build_executive_summary(
                commits, prs, developer_stats, project_metrics, dora_metrics
            ),
            "projects": self._build_project_data(commits, prs, developer_stats, project_metrics),
            "developers": self._build_developer_profiles(commits, developer_stats),
            "workflow_analysis": self._build_workflow_analysis(
                commits, prs, project_metrics, pm_data
            ),
            "time_series": self._build_time_series_data(commits, prs, weeks),
            "insights": self._build_insights_data(commits, developer_stats, qualitative_data),
            "untracked_analysis": self._build_untracked_analysis(commits, project_metrics),
            "raw_data": self._build_raw_data_summary(commits, prs, developer_stats, dora_metrics),
        }

        # Add enhanced qualitative analysis if available
        if enhanced_qualitative_analysis:
            export_data["enhanced_qualitative_analysis"] = enhanced_qualitative_analysis

        # Add PM platform data if available
        if pm_data:
            export_data["pm_integration"] = self._build_pm_integration_data(pm_data)

        # Add CI/CD data if available
        if cicd_data:
            export_data["cicd_metrics"] = self._build_cicd_data(cicd_data)

        # Serialize and write JSON
        serialized_data = self._serialize_for_json(export_data)

        with open(output_path, "w") as f:
            json.dump(serialized_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Comprehensive JSON export written to {output_path}")
        return output_path

    def create_enhanced_qualitative_analysis(
        self,
        commits: List[Dict[str, Any]],
        qualitative_data: Optional[List[Any]] = None,
        developer_stats: Optional[List[Dict[str, Any]]] = None,
        project_metrics: Optional[Dict[str, Any]] = None,
        pm_data: Optional[Dict[str, Any]] = None,
        weeks_analyzed: int = 12,
    ) -> Optional[Dict[str, Any]]:
        """Create enhanced qualitative analysis using the EnhancedQualitativeAnalyzer.

        This method integrates with the enhanced analyzer to generate comprehensive
        qualitative insights across executive, project, developer, and workflow dimensions.

        Args:
            commits: List of commit data
            qualitative_data: Optional qualitative commit analysis results
            developer_stats: Optional developer statistics
            project_metrics: Optional project-level metrics
            pm_data: Optional PM platform integration data
            weeks_analyzed: Number of weeks in analysis period

        Returns:
            Enhanced qualitative analysis results or None if analyzer unavailable
        """
        try:
            # Import here to avoid circular dependencies
            from ..qualitative.enhanced_analyzer import EnhancedQualitativeAnalyzer

            # Initialize analyzer
            analyzer = EnhancedQualitativeAnalyzer()

            # Perform comprehensive analysis
            enhanced_analysis = analyzer.analyze_comprehensive(
                commits=commits,
                qualitative_data=qualitative_data,
                developer_stats=developer_stats,
                project_metrics=project_metrics,
                pm_data=pm_data,
                weeks_analyzed=weeks_analyzed,
            )

            logger.info("Enhanced qualitative analysis completed successfully")
            return enhanced_analysis

        except ImportError as e:
            logger.warning(f"Enhanced qualitative analyzer not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Enhanced qualitative analysis failed: {e}")
            return None

