"""JSON exporter mixin: recommendations, utility helpers, and BaseReportGenerator integration.

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

from .base import BaseReportGenerator, ReportData, ReportOutput
from .interfaces import ReportFormat

logger = logging.getLogger(__name__)


class JSONExportUtilsMixin:
    """Mixin providing recommendations, utility helpers, and BaseReportGenerator integration for ComprehensiveJSONExporter."""

    def _calculate_untracked_quality_scores(
        self, categories: Dict[str, Any], total_untracked: int, total_commits: int
    ) -> Dict[str, Any]:
        """Calculate quality scores for untracked work patterns."""
        scores = {}

        # Process adherence score (lower untracked % = higher score)
        untracked_pct = (total_untracked / total_commits * 100) if total_commits > 0 else 0
        process_score = max(0, 100 - untracked_pct * 2)  # Scale so 50% untracked = 0 score
        scores["process_adherence"] = round(min(100, process_score), 1)

        # Appropriate untracked score (higher % of maintenance/docs/style = higher score)
        appropriate_categories = ["maintenance", "documentation", "style", "test"]
        appropriate_count = sum(
            categories.get(cat, {}).get("count", 0) for cat in appropriate_categories
        )
        appropriate_pct = (appropriate_count / total_untracked * 100) if total_untracked > 0 else 0
        scores["appropriate_untracked"] = round(appropriate_pct, 1)

        # Work type balance score
        if categories:
            category_counts = [data["count"] for data in categories.values()]
            # Calculate distribution balance (lower Gini = more balanced)
            gini = self._calculate_gini_coefficient(category_counts)
            balance_score = max(0, 100 - (gini * 100))
            scores["work_type_balance"] = round(balance_score, 1)
        else:
            scores["work_type_balance"] = 100

        # Overall untracked quality score
        overall_score = (
            scores["process_adherence"] * 0.5
            + scores["appropriate_untracked"] * 0.3
            + scores["work_type_balance"] * 0.2
        )
        scores["overall"] = round(overall_score, 1)

        # Quality rating
        if overall_score >= 80:
            rating = "excellent"
        elif overall_score >= 60:
            rating = "good"
        elif overall_score >= 40:
            rating = "fair"
        else:
            rating = "needs_improvement"

        scores["rating"] = rating

        return scores

    def _generate_actionable_recommendations(
        self, insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from insights."""

        recommendations = []

        # Extract recommendations from insights
        for insight in insights:
            if "recommendation" in insight and insight.get("type") == "concern":
                recommendations.append(
                    {
                        "title": insight["title"],
                        "action": insight["recommendation"],
                        "priority": insight.get("priority", "medium"),
                        "category": insight.get("category", "general"),
                        "expected_impact": self._estimate_recommendation_impact(insight),
                    }
                )

        # Add general recommendations based on patterns
        self._add_general_recommendations(recommendations, insights)

        return recommendations[:5]  # Return top 5 recommendations

    def _estimate_recommendation_impact(self, insight: Dict[str, Any]) -> str:
        """Estimate the impact of implementing a recommendation."""

        category = insight.get("category", "")
        priority = insight.get("priority", "medium")

        if priority == "high":
            return "high"
        elif category in ["team", "productivity"]:
            return "medium"
        else:
            return "low"

    def _add_general_recommendations(
        self, recommendations: List[Dict[str, Any]], insights: List[Dict[str, Any]]
    ) -> None:
        """Add general recommendations based on insight patterns."""

        # Check for lack of ticket coverage insights
        ticket_insights = [i for i in insights if "ticket" in i.get("description", "").lower()]
        if not ticket_insights:
            recommendations.append(
                {
                    "title": "Improve Development Process Tracking",
                    "action": "Implement consistent ticket referencing in commits and PRs",
                    "priority": "medium",
                    "category": "process",
                    "expected_impact": "medium",
                }
            )

    def _calculate_simple_trend(self, values: List[float]) -> str:
        """Calculate simple trend direction from a list of values."""

        if len(values) < 2:
            return "stable"

        # Compare first half vs second half
        midpoint = len(values) // 2
        first_half = values[:midpoint]
        second_half = values[midpoint:]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if first_avg == 0:
            return "stable"

        change_pct = ((second_avg - first_avg) / first_avg) * 100

        if abs(change_pct) < 10:
            return "stable"
        elif change_pct > 0:
            return "increasing"
        else:
            return "decreasing"

    def _get_weekly_commit_counts(self, commits: List[Dict[str, Any]]) -> List[int]:
        """Get commit counts grouped by week."""

        if not commits:
            return []

        # Group commits by week
        weekly_counts = defaultdict(int)

        for commit in commits:
            week_start = self._get_week_start(commit["timestamp"])
            week_key = week_start.strftime("%Y-%m-%d")
            weekly_counts[week_key] += 1

        # Return counts in chronological order
        sorted_weeks = sorted(weekly_counts.keys())
        return [weekly_counts[week] for week in sorted_weeks]

    def _get_daily_commit_counts(self, commits: List[Dict[str, Any]]) -> List[int]:
        """Get commit counts grouped by day."""

        if not commits:
            return []

        # Group commits by day
        daily_counts = defaultdict(int)

        for commit in commits:
            day_key = commit["timestamp"].strftime("%Y-%m-%d")
            daily_counts[day_key] += 1

        # Return counts in chronological order
        sorted_days = sorted(daily_counts.keys())
        return [daily_counts[day] for day in sorted_days]

    def _calculate_weekly_commits(self, commits: List[Dict[str, Any]]) -> float:
        """Calculate average commits per week."""

        weekly_counts = self._get_weekly_commit_counts(commits)
        if not weekly_counts:
            return 0

        return round(statistics.mean(weekly_counts), 1)

    def _find_peak_activity_day(self, commits: List[Dict[str, Any]]) -> str:
        """Find the day of week with most commits."""

        if not commits:
            return "Unknown"

        day_counts = defaultdict(int)

        for commit in commits:
            if hasattr(commit["timestamp"], "weekday"):
                day_index = commit["timestamp"].weekday()
                day_counts[day_index] += 1

        if not day_counts:
            return "Unknown"

        peak_day_index = max(day_counts, key=day_counts.get)
        return self._get_day_name(peak_day_index)

    def _analyze_commit_size_distribution(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of commit sizes."""

        if not commits:
            return {}

        sizes = []
        for commit in commits:
            lines = commit.get("filtered_insertions", commit.get("insertions", 0)) + commit.get(
                "filtered_deletions", commit.get("deletions", 0)
            )
            sizes.append(lines)

        if not sizes:
            return {}

        return {
            "mean": round(statistics.mean(sizes), 1),
            "median": round(statistics.median(sizes), 1),
            "std_dev": round(statistics.pstdev(sizes), 1) if len(sizes) > 1 else 0,
            "min": min(sizes),
            "max": max(sizes),
            "small_commits": sum(1 for s in sizes if s < 50),  # < 50 lines
            "medium_commits": sum(1 for s in sizes if 50 <= s <= 200),  # 50-200 lines
            "large_commits": sum(1 for s in sizes if s > 200),  # > 200 lines
        }

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

    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for measuring inequality."""

        if not values or len(values) == 1:
            return 0.0

        sorted_values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)

        # Use builtin sum instead of np.sum for generator expression (numpy deprecation)
        return (2 * sum((i + 1) * sorted_values[i] for i in range(n))) / (n * cumsum[-1]) - (
            n + 1
        ) / n

    def _anonymize_value(self, value: str, field_type: str) -> str:
        """Anonymize a value if anonymization is enabled."""

        if not self.anonymize or not value:
            return value

        if field_type == "email" and "@" in value:
            # Keep domain for email
            local, domain = value.split("@", 1)
            value = local  # Anonymize only local part
            suffix = f"@{domain}"
        else:
            suffix = ""

        if value not in self._anonymization_map:
            self._anonymous_counter += 1
            if field_type == "name":
                anonymous = f"Developer{self._anonymous_counter}"
            elif field_type == "email":
                anonymous = f"dev{self._anonymous_counter}"
            elif field_type == "id":
                anonymous = f"ID{self._anonymous_counter:04d}"
            elif field_type == "username":
                anonymous = f"user{self._anonymous_counter}"
            else:
                anonymous = f"anon{self._anonymous_counter}"

            self._anonymization_map[value] = anonymous

        return self._anonymization_map[value] + suffix

    def _serialize_for_json(self, data: Any) -> Any:
        """Serialize data for JSON output, handling datetime objects."""

        if isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, dict):
            return {k: self._serialize_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_for_json(item) for item in data]
        elif isinstance(data, set):
            return list(data)  # Convert sets to lists
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)  # Convert numpy types to Python types
        else:
            return data

    # Implementation of abstract methods from BaseReportGenerator

    def generate(self, data: ReportData, output_path: Optional[Path] = None) -> ReportOutput:
        """Generate comprehensive JSON export from standardized data.

        Args:
            data: Standardized report data
            output_path: Optional path to write the JSON to

        Returns:
            ReportOutput containing the results
        """
        try:
            # Validate data
            if not self.validate_data(data):
                return ReportOutput(success=False, errors=["Invalid or incomplete data provided"])

            # Pre-process data
            data = self.pre_process(data)

            # Use the main export method with ReportData fields
            if output_path:
                self.export_comprehensive_data(
                    commits=data.commits or [],
                    prs=data.pull_requests or [],
                    developer_stats=data.developer_stats or [],
                    project_metrics=data.config.get("project_metrics", {}),
                    dora_metrics=data.dora_metrics or {},
                    output_path=output_path,
                    weeks=data.metadata.analysis_period_weeks or 12,
                    pm_data=data.pm_data,
                    qualitative_data=data.qualitative_results,
                    enhanced_qualitative_analysis=data.config.get("enhanced_qualitative_analysis"),
                )

                return ReportOutput(
                    success=True,
                    file_path=output_path,
                    format=self.get_format_type(),
                    size_bytes=output_path.stat().st_size if output_path.exists() else 0,
                )
            else:
                # Generate in-memory JSON
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(weeks=data.metadata.analysis_period_weeks or 12)

                export_data = {
                    "metadata": self._build_metadata(
                        data.commits or [],
                        data.pull_requests or [],
                        data.developer_stats or [],
                        start_date,
                        end_date,
                    ),
                    "executive_summary": self._build_executive_summary(
                        data.commits or [],
                        data.pull_requests or [],
                        data.developer_stats or [],
                        data.config.get("project_metrics", {}),
                        data.dora_metrics or {},
                    ),
                    "raw_data": self._build_raw_data_summary(
                        data.commits or [],
                        data.pull_requests or [],
                        data.developer_stats or [],
                        data.dora_metrics or {},
                    ),
                }

                serialized_data = self._serialize_for_json(export_data)
                json_content = json.dumps(serialized_data, indent=2, ensure_ascii=False)

                return ReportOutput(
                    success=True,
                    content=json_content,
                    format=self.get_format_type(),
                    size_bytes=len(json_content),
                )

        except Exception as e:
            logger.error(f"Error generating comprehensive JSON export: {e}")
            return ReportOutput(success=False, errors=[str(e)])

    def get_required_fields(self) -> List[str]:
        """Get the list of required data fields for JSON export.

        Returns:
            List of required field names
        """
        # Comprehensive JSON export can work with any combination of data
        # but works best with commits and developer_stats
        return []  # No strict requirements, flexible export

    def get_format_type(self) -> str:
        """Get the format type this generator produces.

        Returns:
            Format identifier
        """
        return ReportFormat.JSON.value
