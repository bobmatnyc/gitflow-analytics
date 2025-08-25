"""CLI integration for the new report abstraction layer.

This module shows how the CLI can be refactored to use the new
report generation abstraction layer while maintaining backward compatibility.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ReportData, ReportMetadata
from .factory import ReportBuilder, ReportFactory
from .interfaces import ReportFormat, ReportType

logger = logging.getLogger(__name__)


class ReportCLIAdapter:
    """Adapter to integrate the new report abstraction with the existing CLI.
    
    This class bridges the gap between the existing CLI code and the new
    report abstraction layer, allowing gradual migration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the CLI adapter.
        
        Args:
            config: Configuration dictionary from CLI
        """
        self.config = config
        self.factory = ReportFactory()
        
        # Set default configuration for all generators
        self.factory.set_default_config({
            "anonymize": config.get("anonymize", False),
            "exclude_authors": config.get("analysis", {}).get("exclude_authors", []),
            "identity_resolver": config.get("identity_resolver")
        })
    
    def prepare_report_data(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        activity_data: List[Dict[str, Any]] = None,
        focus_data: List[Dict[str, Any]] = None,
        insights_data: List[Dict[str, Any]] = None,
        ticket_analysis: Dict[str, Any] = None,
        pr_metrics: Dict[str, Any] = None,
        dora_metrics: Dict[str, Any] = None,
        branch_health_metrics: List[Dict[str, Any]] = None,
        pm_data: Dict[str, Any] = None,
        qualitative_results: List[Dict[str, Any]] = None,
        chatgpt_summary: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        weeks: int = 12
    ) -> ReportData:
        """Prepare standardized report data from CLI data.
        
        Args:
            Various data components from CLI analysis
            
        Returns:
            Standardized ReportData instance
        """
        # Calculate metadata
        metadata = ReportMetadata(
            analysis_period_weeks=weeks,
            start_date=start_date,
            end_date=end_date,
            total_commits=len(commits) if commits else 0,
            total_developers=len(developer_stats) if developer_stats else 0,
            source_repositories=self._extract_repositories(commits),
            excluded_authors=self.config.get("analysis", {}).get("exclude_authors", [])
        )
        
        # Create ReportData
        return ReportData(
            commits=commits or [],
            pull_requests=prs or [],
            developer_stats=developer_stats or [],
            activity_data=activity_data or [],
            focus_data=focus_data or [],
            insights_data=insights_data or [],
            ticket_analysis=ticket_analysis or {},
            pr_metrics=pr_metrics or {},
            dora_metrics=dora_metrics or {},
            branch_health_metrics=branch_health_metrics or [],
            pm_data=pm_data,
            story_points_data=self._extract_story_points(commits, prs),
            qualitative_results=qualitative_results or [],
            chatgpt_summary=chatgpt_summary,
            metadata=metadata,
            config=self.config
        )
    
    def generate_reports(
        self,
        report_data: ReportData,
        output_dir: Path,
        formats: List[str] = None
    ) -> Dict[str, Any]:
        """Generate all configured reports.
        
        Args:
            report_data: Standardized report data
            output_dir: Output directory for reports
            formats: List of format strings (csv, markdown, json, etc.)
            
        Returns:
            Dictionary of report results
        """
        if formats is None:
            formats = self.config.get("output", {}).get("formats", ["csv", "markdown"])
        
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # Use ReportBuilder to configure and generate reports
        builder = ReportBuilder(self.factory)
        builder.with_data(report_data)
        builder.with_output_dir(output_dir)
        
        # Add reports based on configured formats
        if "csv" in formats:
            self._add_csv_reports(builder)
        
        if "markdown" in formats:
            builder.add_report(ReportType.NARRATIVE, ReportFormat.MARKDOWN)
        
        if "json" in formats:
            builder.add_report(ReportType.COMPREHENSIVE, ReportFormat.JSON)
        
        if "html" in formats:
            builder.add_report(ReportType.COMPREHENSIVE, ReportFormat.HTML)
        
        # Generate all reports
        try:
            outputs = builder.generate()
            
            # Process outputs
            if isinstance(outputs, list):
                for i, output in enumerate(outputs):
                    if output.success:
                        report_name = f"report_{i}"
                        if output.file_path:
                            report_name = output.file_path.stem
                        results[report_name] = {
                            "success": True,
                            "path": str(output.file_path) if output.file_path else None,
                            "size": output.size_bytes
                        }
                    else:
                        results[f"report_{i}"] = {
                            "success": False,
                            "errors": output.errors
                        }
            else:
                # Single output
                results["report"] = {
                    "success": outputs.success,
                    "path": str(outputs.file_path) if outputs.file_path else None,
                    "errors": outputs.errors if not outputs.success else []
                }
                
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            results["error"] = str(e)
        
        return results
    
    def generate_legacy_reports(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        output_dir: Path,
        **kwargs
    ) -> Dict[str, Path]:
        """Generate reports using legacy method for backward compatibility.
        
        This method maintains the exact same interface as the existing CLI
        report generation code.
        
        Args:
            commits: Commit data
            prs: Pull request data
            developer_stats: Developer statistics
            output_dir: Output directory
            **kwargs: Additional data components
            
        Returns:
            Dictionary mapping report names to file paths
        """
        generated_reports = {}
        timestamp = datetime.now().strftime("%Y%m%d")
        
        try:
            # Create legacy generators with backward-compatible initialization
            from .analytics_writer import AnalyticsReportGenerator
            from .csv_writer import CSVReportGenerator
            from .narrative_writer import NarrativeReportGenerator

            # CSV Reports
            csv_gen = CSVReportGenerator(
                anonymize=self.config.get("anonymize", False),
                exclude_authors=self.config.get("analysis", {}).get("exclude_authors", []),
                identity_resolver=kwargs.get("identity_resolver")
            )
            
            # Weekly report
            weekly_path = output_dir / f"weekly_metrics_{timestamp}.csv"
            csv_gen.generate_weekly_report(commits, developer_stats, weekly_path)
            generated_reports["weekly_metrics"] = weekly_path
            
            # Developer report
            dev_path = output_dir / f"developer_stats_{timestamp}.csv"
            csv_gen.generate_developer_report(developer_stats, dev_path)
            generated_reports["developer_stats"] = dev_path
            
            # Analytics Reports
            analytics_gen = AnalyticsReportGenerator(
                anonymize=self.config.get("anonymize", False),
                exclude_authors=self.config.get("analysis", {}).get("exclude_authors", []),
                identity_resolver=kwargs.get("identity_resolver")
            )
            
            # Activity distribution
            if kwargs.get("activity_data"):
                activity_path = output_dir / f"activity_distribution_{timestamp}.csv"
                analytics_gen.generate_activity_distribution_report(
                    commits, developer_stats, activity_path
                )
                generated_reports["activity_distribution"] = activity_path
            
            # Developer focus
            if kwargs.get("focus_data"):
                focus_path = output_dir / f"developer_focus_{timestamp}.csv"
                analytics_gen.generate_developer_focus_report(
                    commits, developer_stats, focus_path, 
                    kwargs.get("weeks", 12)
                )
                generated_reports["developer_focus"] = focus_path
            
            # Narrative Report
            if "markdown" in self.config.get("output", {}).get("formats", []):
                narrative_gen = NarrativeReportGenerator()
                narrative_path = output_dir / f"narrative_report_{timestamp}.md"
                
                narrative_gen.generate_narrative_report(
                    commits,
                    prs,
                    developer_stats,
                    kwargs.get("activity_data", []),
                    kwargs.get("focus_data", []),
                    kwargs.get("insights_data", []),
                    kwargs.get("ticket_analysis", {}),
                    kwargs.get("pr_metrics", {}),
                    narrative_path,
                    kwargs.get("weeks", 12),
                    kwargs.get("pm_data"),
                    kwargs.get("chatgpt_summary"),
                    kwargs.get("branch_health_metrics"),
                    self.config.get("analysis", {}).get("exclude_authors", [])
                )
                generated_reports["narrative"] = narrative_path
            
            # JSON Export
            if "json" in self.config.get("output", {}).get("formats", []):
                from .json_exporter import ComprehensiveJSONExporter
                
                json_exporter = ComprehensiveJSONExporter(
                    anonymize=self.config.get("anonymize", False)
                )
                
                json_path = output_dir / f"comprehensive_export_{timestamp}.json"
                json_exporter.export_comprehensive_data(
                    commits,
                    prs,
                    developer_stats,
                    kwargs.get("project_metrics", {}),
                    kwargs.get("dora_metrics", {}),
                    json_path,
                    kwargs.get("weeks", 12),
                    kwargs.get("pm_data"),
                    kwargs.get("qualitative_data")
                )
                generated_reports["json_export"] = json_path
                
        except Exception as e:
            logger.error(f"Error generating legacy reports: {e}")
            raise
        
        return generated_reports
    
    def _add_csv_reports(self, builder: ReportBuilder) -> None:
        """Add CSV report types to builder.
        
        Args:
            builder: Report builder instance
        """
        csv_reports = [
            ReportType.WEEKLY_METRICS,
            ReportType.DEVELOPER_STATS,
            ReportType.ACTIVITY_DISTRIBUTION,
            ReportType.DEVELOPER_FOCUS,
            ReportType.QUALITATIVE_INSIGHTS
        ]
        
        for report_type in csv_reports:
            builder.add_report(report_type, ReportFormat.CSV)
    
    def _extract_repositories(self, commits: List[Dict[str, Any]]) -> List[str]:
        """Extract unique repository names from commits.
        
        Args:
            commits: List of commit data
            
        Returns:
            List of unique repository names
        """
        repos = set()
        for commit in commits or []:
            if "repository" in commit:
                repos.add(commit["repository"])
            elif "project_key" in commit:
                repos.add(commit["project_key"])
        return list(repos)
    
    def _extract_story_points(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract story points data from commits and PRs.
        
        Args:
            commits: Commit data
            prs: Pull request data
            
        Returns:
            Story points summary
        """
        total_points = 0.0
        commits_with_points = 0
        
        for commit in commits or []:
            points = commit.get("story_points", 0) or 0
            if points > 0:
                total_points += points
                commits_with_points += 1
        
        for pr in prs or []:
            points = pr.get("story_points", 0) or 0
            if points > 0:
                total_points += points
        
        return {
            "total_story_points": total_points,
            "commits_with_points": commits_with_points,
            "coverage_percentage": (commits_with_points / len(commits) * 100) if commits else 0
        }


def integrate_with_cli(cli_config: Dict[str, Any]) -> ReportCLIAdapter:
    """Create a CLI adapter for report generation.
    
    This function can be called from the existing CLI to use the new
    report abstraction layer.
    
    Args:
        cli_config: Configuration from CLI
        
    Returns:
        Configured ReportCLIAdapter instance
    """
    return ReportCLIAdapter(cli_config)


# Example usage in CLI:
"""
# In cli.py, replace direct report generation with:

from .reports.cli_integration import integrate_with_cli

# ... existing CLI code ...

# Create adapter
report_adapter = integrate_with_cli(cfg)

# Prepare standardized data
report_data = report_adapter.prepare_report_data(
    commits=all_commits,
    prs=all_prs,
    developer_stats=developer_stats,
    activity_data=activity_data,
    focus_data=focus_data,
    insights_data=insights_data,
    ticket_analysis=ticket_analysis,
    pr_metrics=pr_metrics,
    dora_metrics=dora_metrics,
    branch_health_metrics=branch_health_metrics,
    pm_data=aggregated_pm_data,
    qualitative_results=qualitative_results,
    chatgpt_summary=chatgpt_summary,
    start_date=start_date,
    end_date=end_date,
    weeks=weeks
)

# Generate reports using new abstraction
results = report_adapter.generate_reports(report_data, output_dir)

# Or use legacy method for gradual migration
legacy_reports = report_adapter.generate_legacy_reports(
    all_commits,
    all_prs,
    developer_stats,
    output_dir,
    activity_data=activity_data,
    focus_data=focus_data,
    # ... other kwargs
)
"""