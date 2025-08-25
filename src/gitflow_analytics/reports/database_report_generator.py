"""Database-backed report generator for GitFlow Analytics.

WHY: This module generates reports directly from the SQLite database,
providing fast retrieval and consistent formatting for daily metrics
and trend analysis.
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.metrics_storage import DailyMetricsStorage

logger = logging.getLogger(__name__)


class DatabaseReportGenerator:
    """Generate reports directly from database-stored daily metrics.
    
    WHY: Database-backed reporting provides fast, consistent report generation
    with built-in trend analysis and classification insights.
    """

    def __init__(self, metrics_storage: DailyMetricsStorage):
        """Initialize database report generator.
        
        Args:
            metrics_storage: DailyMetricsStorage instance for data access
        """
        self.storage = metrics_storage
        logger.info("Initialized database report generator")

    def generate_qualitative_report(
        self,
        start_date: date,
        end_date: date,
        output_path: Path,
        developer_ids: Optional[List[str]] = None,
        project_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive qualitative analysis report.
        
        WHY: Provides detailed insights into development patterns, 
        classification trends, and team productivity based on stored metrics.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            output_path: Path to write report file
            developer_ids: Optional filter by specific developers
            project_keys: Optional filter by specific projects
            
        Returns:
            Dict with report metadata and statistics
        """
        logger.info(f"Generating qualitative report for {start_date} to {end_date}")
        
        # Gather data
        daily_metrics = self.storage.get_date_range_metrics(
            start_date, end_date, developer_ids, project_keys
        )
        
        classification_summary = self.storage.get_classification_summary(
            start_date, end_date
        )
        
        trends = self.storage.calculate_weekly_trends(start_date, end_date)
        
        # Generate report content
        report_content = self._build_qualitative_report_content(
            daily_metrics, classification_summary, trends, start_date, end_date
        )
        
        # Write report to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Calculate report statistics
        report_stats = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'total_days': (end_date - start_date).days + 1,
            'unique_developers': len(set(m['developer_id'] for m in daily_metrics)),
            'unique_projects': len(set(m['project_key'] for m in daily_metrics)),
            'total_commits': sum(m['total_commits'] for m in daily_metrics),
            'total_records': len(daily_metrics),
            'trends_calculated': len(trends),
            'output_file': str(output_path),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Generated qualitative report with {report_stats['total_records']} records")
        return report_stats

    def _build_qualitative_report_content(
        self,
        daily_metrics: List[Dict[str, Any]],
        classification_summary: Dict[str, Dict[str, int]],
        trends: Dict[Tuple[str, str], Dict[str, float]],
        start_date: date,
        end_date: date
    ) -> str:
        """Build the complete qualitative report content.
        
        WHY: Structures the report with clear sections for executive summary,
        detailed analysis, and actionable insights based on database metrics.
        """
        lines = []
        
        # Header
        lines.extend([
            "# GitFlow Analytics - Qualitative Report",
            f"**Report Period:** {start_date.isoformat()} to {end_date.isoformat()}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            ""
        ])
        
        # Executive Summary
        lines.extend(self._build_executive_summary(daily_metrics, classification_summary))
        
        # Team Analysis
        lines.extend(self._build_team_analysis(classification_summary['by_developer']))
        
        # Project Analysis
        lines.extend(self._build_project_analysis(classification_summary['by_project']))
        
        # Weekly Trends Analysis
        lines.extend(self._build_trends_analysis(trends, daily_metrics))
        
        # Classification Insights
        lines.extend(self._build_classification_insights(daily_metrics))
        
        # Recommendations
        lines.extend(self._build_recommendations(daily_metrics, classification_summary, trends))
        
        return "\n".join(lines)

    def _build_executive_summary(
        self, 
        daily_metrics: List[Dict[str, Any]], 
        classification_summary: Dict[str, Dict[str, int]]
    ) -> List[str]:
        """Build executive summary section."""
        if not daily_metrics:
            return ["## Executive Summary", "", "No data available for the selected period.", ""]
        
        total_commits = sum(m['total_commits'] for m in daily_metrics)
        unique_developers = len(set(m['developer_id'] for m in daily_metrics))
        unique_projects = len(set(m['project_key'] for m in daily_metrics))
        
        # Calculate top categories
        category_totals = {}
        for metrics in daily_metrics:
            for category in ['feature', 'bug_fix', 'refactor', 'documentation', 'maintenance']:
                field = f"{category}_commits"
                category_totals[category] = category_totals.get(category, 0) + metrics.get(field, 0)
        
        top_category = max(category_totals, key=category_totals.get) if category_totals else "feature"
        top_category_count = category_totals.get(top_category, 0)
        top_category_pct = (top_category_count / total_commits * 100) if total_commits > 0 else 0
        
        # Top contributor
        dev_commits = {}
        for metrics in daily_metrics:
            dev_name = metrics['developer_name']
            dev_commits[dev_name] = dev_commits.get(dev_name, 0) + metrics['total_commits']
        
        top_contributor = max(dev_commits, key=dev_commits.get) if dev_commits else "Unknown"
        top_contributor_commits = dev_commits.get(top_contributor, 0)
        top_contributor_pct = (top_contributor_commits / total_commits * 100) if total_commits > 0 else 0
        
        return [
            "## Executive Summary",
            "",
            f"- **Total Activity:** {total_commits:,} commits across {unique_developers} developers and {unique_projects} projects",
            f"- **Primary Focus:** {top_category.replace('_', ' ').title()} development ({top_category_count} commits, {top_category_pct:.1f}%)",
            f"- **Top Contributor:** {top_contributor} ({top_contributor_commits} commits, {top_contributor_pct:.1f}%)",
            f"- **Average Daily Activity:** {total_commits / max(1, len(set(m['date'] for m in daily_metrics))):.1f} commits per day",
            "",
        ]

    def _build_team_analysis(self, developer_summary: Dict[str, Dict[str, int]]) -> List[str]:
        """Build team analysis section."""
        if not developer_summary:
            return ["## Team Analysis", "", "No developer data available.", ""]
        
        lines = ["## Team Analysis", ""]
        
        # Sort developers by total commits
        sorted_devs = sorted(
            developer_summary.items(), 
            key=lambda x: x[1]['total'], 
            reverse=True
        )
        
        for dev_name, stats in sorted_devs[:10]:  # Top 10 developers
            total = stats['total']
            features = stats['features']
            bugs = stats['bug_fixes']
            refactors = stats['refactors']
            
            feature_pct = (features / total * 100) if total > 0 else 0
            bug_pct = (bugs / total * 100) if total > 0 else 0
            refactor_pct = (refactors / total * 100) if total > 0 else 0
            
            lines.extend([
                f"### {dev_name}",
                f"- **Total Commits:** {total}",
                f"- **Features:** {features} ({feature_pct:.1f}%)",
                f"- **Bug Fixes:** {bugs} ({bug_pct:.1f}%)",
                f"- **Refactoring:** {refactors} ({refactor_pct:.1f}%)",
                ""
            ])
        
        return lines

    def _build_project_analysis(self, project_summary: Dict[str, Dict[str, int]]) -> List[str]:
        """Build project analysis section."""
        if not project_summary:
            return ["## Project Analysis", "", "No project data available.", ""]
        
        lines = ["## Project Analysis", ""]
        
        # Sort projects by total commits
        sorted_projects = sorted(
            project_summary.items(), 
            key=lambda x: x[1]['total'], 
            reverse=True
        )
        
        for project_key, stats in sorted_projects:
            total = stats['total']
            features = stats['features']
            bugs = stats['bug_fixes']
            refactors = stats['refactors']
            
            feature_pct = (features / total * 100) if total > 0 else 0
            bug_pct = (bugs / total * 100) if total > 0 else 0
            refactor_pct = (refactors / total * 100) if total > 0 else 0
            
            lines.extend([
                f"### {project_key}",
                f"- **Total Commits:** {total}",
                f"- **Features:** {features} ({feature_pct:.1f}%)",
                f"- **Bug Fixes:** {bugs} ({bug_pct:.1f}%)",
                f"- **Refactoring:** {refactors} ({refactor_pct:.1f}%)",
                ""
            ])
        
        return lines

    def _build_trends_analysis(
        self, 
        trends: Dict[Tuple[str, str], Dict[str, float]], 
        daily_metrics: List[Dict[str, Any]]
    ) -> List[str]:
        """Build weekly trends analysis section."""
        lines = ["## Weekly Trends Analysis", ""]
        
        if not trends:
            return lines + ["No trend data available (requires at least 2 weeks of data).", ""]
        
        # Group trends by developer
        dev_trends = {}
        for (dev_id, project_key), trend_data in trends.items():
            if dev_id not in dev_trends:
                dev_trends[dev_id] = {}
            dev_trends[dev_id][project_key] = trend_data
        
        # Get developer names mapping
        dev_names = {}
        for metrics in daily_metrics:
            dev_names[metrics['developer_id']] = metrics['developer_name']
        
        for dev_id, project_trends in dev_trends.items():
            dev_name = dev_names.get(dev_id, dev_id)
            lines.extend([f"### {dev_name}", ""])
            
            for project_key, trend_data in project_trends.items():
                total_change = trend_data['total_commits_change']
                feature_change = trend_data['feature_commits_change']
                bug_change = trend_data['bug_fix_commits_change']
                refactor_change = trend_data['refactor_commits_change']
                
                # Format trend direction
                def format_change(change: float) -> str:
                    if change > 5:
                        return f"+{change:.1f}% ⬆️"
                    elif change < -5:
                        return f"{change:.1f}% ⬇️"
                    else:
                        return f"{change:+.1f}% →"
                
                lines.extend([
                    f"**{project_key}:**",
                    f"- Total Commits: {format_change(total_change)}",
                    f"- Features: {format_change(feature_change)}",
                    f"- Bug Fixes: {format_change(bug_change)}",
                    f"- Refactoring: {format_change(refactor_change)}",
                    ""
                ])
        
        return lines

    def _build_classification_insights(self, daily_metrics: List[Dict[str, Any]]) -> List[str]:
        """Build classification insights section."""
        lines = ["## Classification Insights", ""]
        
        if not daily_metrics:
            return lines + ["No classification data available.", ""]
        
        # Calculate overall classification distribution
        total_commits = sum(m['total_commits'] for m in daily_metrics)
        
        category_totals = {}
        for metrics in daily_metrics:
            for category in ['feature', 'bug_fix', 'refactor', 'documentation', 
                           'maintenance', 'test', 'style', 'build', 'other']:
                field = f"{category}_commits"
                category_totals[category] = category_totals.get(category, 0) + metrics.get(field, 0)
        
        lines.append("### Overall Distribution")
        lines.append("")
        
        for category, count in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / total_commits * 100) if total_commits > 0 else 0
                category_name = category.replace('_', ' ').title()
                lines.append(f"- **{category_name}:** {count} commits ({percentage:.1f}%)")
        
        lines.append("")
        
        # Ticket tracking insights
        total_tracked = sum(m['tracked_commits'] for m in daily_metrics)
        total_untracked = sum(m['untracked_commits'] for m in daily_metrics)
        total_ticket_commits = total_tracked + total_untracked
        
        if total_ticket_commits > 0:
            tracking_rate = (total_tracked / total_ticket_commits * 100)
            lines.extend([
                "### Ticket Tracking",
                "",
                f"- **Tracked Commits:** {total_tracked} ({tracking_rate:.1f}%)",
                f"- **Untracked Commits:** {total_untracked} ({100 - tracking_rate:.1f}%)",
                ""
            ])
        
        return lines

    def _build_recommendations(
        self,
        daily_metrics: List[Dict[str, Any]],
        classification_summary: Dict[str, Dict[str, int]],
        trends: Dict[Tuple[str, str], Dict[str, float]]
    ) -> List[str]:
        """Build recommendations section."""
        lines = ["## Recommendations", ""]
        
        recommendations = []
        
        # Ticket tracking recommendations
        total_tracked = sum(m['tracked_commits'] for m in daily_metrics)
        total_untracked = sum(m['untracked_commits'] for m in daily_metrics)
        total_commits = total_tracked + total_untracked
        
        if total_commits > 0:
            tracking_rate = (total_tracked / total_commits * 100)
            if tracking_rate < 70:
                recommendations.append(
                    f"**Improve Ticket Tracking:** Only {tracking_rate:.1f}% of commits are linked to tickets. "
                    "Consider implementing commit message templates or pre-commit hooks."
                )
        
        # Classification balance recommendations
        total_feature = sum(m['feature_commits'] for m in daily_metrics)
        total_bugs = sum(m['bug_fix_commits'] for m in daily_metrics)
        total_refactor = sum(m['refactor_commits'] for m in daily_metrics)
        total_class_commits = total_feature + total_bugs + total_refactor
        
        if total_class_commits > 0:
            bug_ratio = (total_bugs / total_class_commits * 100)
            if bug_ratio > 40:
                recommendations.append(
                    f"**High Bug Fix Activity:** {bug_ratio:.1f}% of development time is spent on bug fixes. "
                    "Consider investing in code quality improvements and testing."
                )
            
            refactor_ratio = (total_refactor / total_class_commits * 100)
            if refactor_ratio < 10:
                recommendations.append(
                    f"**Low Refactoring Activity:** Only {refactor_ratio:.1f}% of commits are refactoring. "
                    "Regular refactoring helps maintain code quality and reduces technical debt."
                )
        
        # Trend-based recommendations
        declining_developers = []
        for (dev_id, project_key), trend_data in trends.items():
            if trend_data['total_commits_change'] < -20:
                # Find developer name
                dev_name = "Unknown"
                for metrics in daily_metrics:
                    if metrics['developer_id'] == dev_id:
                        dev_name = metrics['developer_name']
                        break
                declining_developers.append(dev_name)
        
        if declining_developers:
            dev_list = ", ".join(set(declining_developers))
            recommendations.append(
                f"**Monitor Developer Activity:** The following developers show declining activity: {dev_list}. "
                "Consider checking for blockers or workload balance issues."
            )
        
        # Output recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"{i}. {rec}")
                lines.append("")
        else:
            lines.append("No specific recommendations at this time. Overall development patterns look healthy.")
            lines.append("")
        
        return lines