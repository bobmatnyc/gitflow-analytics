"""Example usage of the report generation abstraction layer.

This module demonstrates how to use the new report abstraction layer
to generate reports in various formats with a unified interface.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .base import ReportData, ReportMetadata
from .factory import ReportBuilder, ReportFactory, create_multiple_reports, create_report
from .interfaces import ReportFormat, ReportType


def example_basic_usage():
    """Example of basic report generation using the factory."""
    
    # Create sample data
    sample_commits = [
        {
            "hash": "abc123",
            "author_email": "dev@example.com",
            "author_name": "Developer One",
            "timestamp": datetime.now(timezone.utc),
            "message": "feat: Add new feature",
            "insertions": 100,
            "deletions": 20,
            "files_changed": 5
        }
    ]
    
    sample_developer_stats = [
        {
            "canonical_id": "dev@example.com",
            "primary_email": "dev@example.com",
            "primary_name": "Developer One",
            "total_commits": 50,
            "total_story_points": 25,
            "ticket_coverage_pct": 85.0
        }
    ]
    
    # Create ReportData
    report_data = ReportData(
        commits=sample_commits,
        developer_stats=sample_developer_stats,
        metadata=ReportMetadata(
            analysis_period_weeks=4,
            total_commits=len(sample_commits),
            total_developers=len(sample_developer_stats)
        )
    )
    
    # Method 1: Using the factory directly
    factory = ReportFactory()
    csv_generator = factory.create_generator(ReportType.WEEKLY_METRICS, ReportFormat.CSV)
    output = csv_generator.generate(report_data, Path("weekly_metrics.csv"))
    
    if output.success:
        print(f"Report generated: {output.file_path}")
    else:
        print(f"Errors: {output.errors}")
    
    # Method 2: Using the convenience function
    output = create_report(
        ReportType.DEVELOPER_STATS,
        ReportFormat.CSV,
        report_data,
        "developer_stats.csv"
    )
    
    # Method 3: Using the builder pattern
    builder = ReportBuilder()
    generator = (builder
                .add_report(ReportType.NARRATIVE, ReportFormat.MARKDOWN)
                .with_config(anonymize=False, exclude_authors=["bot@example.com"])
                .with_data(report_data)
                .build())
    
    output = generator.generate(report_data, Path("narrative.md"))


def example_composite_reports():
    """Example of generating multiple report formats at once."""
    
    # Create sample data
    report_data = ReportData(
        commits=[{"hash": "test", "author_email": "dev@test.com", "timestamp": datetime.now(timezone.utc)}],
        developer_stats=[{"canonical_id": "dev@test.com", "total_commits": 10}]
    )
    
    # Generate multiple reports at once
    outputs = create_multiple_reports(
        [
            (ReportType.WEEKLY_METRICS, ReportFormat.CSV),
            (ReportType.DEVELOPER_STATS, ReportFormat.CSV),
            (ReportType.COMPREHENSIVE, ReportFormat.JSON)
        ],
        report_data,
        output_dir="reports/"
    )
    
    for output in outputs:
        if output.success:
            print(f"Generated: {output.file_path}")
        else:
            print(f"Failed: {output.errors}")


def example_custom_generator():
    """Example of creating a custom report generator."""
    
    from .base import BaseReportGenerator, ReportOutput
    
    class CustomHTMLGenerator(BaseReportGenerator):
        """Custom HTML report generator."""
        
        def generate(self, data: ReportData, output_path: Path = None) -> ReportOutput:
            """Generate HTML report."""
            # Pre-process data
            data = self.pre_process(data)
            
            # Generate HTML
            html_content = self._generate_html(data)
            
            if output_path:
                self.write_to_file(html_content, output_path)
                return ReportOutput(
                    success=True,
                    file_path=output_path,
                    format="html",
                    size_bytes=len(html_content)
                )
            else:
                return ReportOutput(
                    success=True,
                    content=html_content,
                    format="html",
                    size_bytes=len(html_content)
                )
        
        def get_required_fields(self) -> List[str]:
            return ["commits", "developer_stats"]
        
        def get_format_type(self) -> str:
            return "html"
        
        def _generate_html(self, data: ReportData) -> str:
            """Generate HTML content."""
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>GitFlow Analytics Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>GitFlow Analytics Report</h1>
                <h2>Summary</h2>
                <p>Total Commits: {len(data.commits)}</p>
                <p>Total Developers: {len(data.developer_stats)}</p>
                
                <h2>Top Contributors</h2>
                <table>
                    <tr>
                        <th>Developer</th>
                        <th>Commits</th>
                        <th>Story Points</th>
                    </tr>
            """
            
            for dev in sorted(data.developer_stats, 
                            key=lambda d: d.get("total_commits", 0), 
                            reverse=True)[:10]:
                html += f"""
                    <tr>
                        <td>{dev.get('primary_name', 'Unknown')}</td>
                        <td>{dev.get('total_commits', 0)}</td>
                        <td>{dev.get('total_story_points', 0)}</td>
                    </tr>
                """
            
            html += """
                </table>
            </body>
            </html>
            """
            
            return html
    
    # Register the custom generator
    factory = ReportFactory()
    factory.register_generator(
        ReportType.CUSTOM,
        ReportFormat.HTML,
        CustomHTMLGenerator
    )
    
    # Use the custom generator
    report_data = ReportData(
        commits=[{"hash": "test", "timestamp": datetime.now(timezone.utc)}],
        developer_stats=[{"primary_name": "Test Dev", "total_commits": 10}]
    )
    
    generator = factory.create_generator(ReportType.CUSTOM, ReportFormat.HTML)
    output = generator.generate(report_data, Path("custom_report.html"))
    
    if output.success:
        print(f"Custom report generated: {output.file_path}")


def example_report_chaining():
    """Example of chaining report generators."""
    
    from .base import ChainedReportGenerator

    # Create a chain of generators that process data sequentially
    factory = ReportFactory()
    
    # First generate CSV, then use that to generate a summary JSON
    csv_gen = factory.create_generator(ReportType.WEEKLY_METRICS, ReportFormat.CSV)
    json_gen = factory.create_generator(ReportType.COMPREHENSIVE, ReportFormat.JSON)
    
    chain = ChainedReportGenerator([csv_gen, json_gen])
    
    report_data = ReportData(
        commits=[{"hash": "test", "timestamp": datetime.now(timezone.utc)}],
        developer_stats=[{"canonical_id": "dev@test.com", "total_commits": 10}]
    )
    
    output = chain.generate(report_data, Path("final_output.json"))
    
    if output.success:
        print(f"Chained report generated: {output.file_path}")


def example_template_based_generation():
    """Example of using templates for report generation."""
    
    from string import Template
    
    class TemplateReportGenerator(BaseReportGenerator):
        """Template-based report generator."""
        
        def __init__(self, template_path: Path = None, **kwargs):
            super().__init__(**kwargs)
            self.template_path = template_path
        
        def generate(self, data: ReportData, output_path: Path = None) -> ReportOutput:
            """Generate report from template."""
            # Pre-process data
            data = self.pre_process(data)
            
            # Load template
            if self.template_path and self.template_path.exists():
                template_str = self.template_path.read_text()
            else:
                template_str = self._get_default_template()
            
            template = Template(template_str)
            
            # Prepare context
            context = {
                "total_commits": len(data.commits),
                "total_developers": len(data.developer_stats),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "top_developer": self._get_top_developer(data.developer_stats)
            }
            
            # Render template
            content = template.safe_substitute(context)
            
            if output_path:
                self.write_to_file(content, output_path)
                return ReportOutput(success=True, file_path=output_path, format="txt")
            else:
                return ReportOutput(success=True, content=content, format="txt")
        
        def get_required_fields(self) -> List[str]:
            return []
        
        def get_format_type(self) -> str:
            return "template"
        
        def _get_default_template(self) -> str:
            return """
GitFlow Analytics Report
========================
Date: $date

Summary:
- Total Commits: $total_commits
- Total Developers: $total_developers
- Top Contributor: $top_developer

Generated by GitFlow Analytics
            """
        
        def _get_top_developer(self, developers: List[Dict[str, Any]]) -> str:
            if not developers:
                return "N/A"
            
            top = max(developers, key=lambda d: d.get("total_commits", 0))
            return f"{top.get('primary_name', 'Unknown')} ({top.get('total_commits', 0)} commits)"
    
    # Use the template generator
    report_data = ReportData(
        commits=[{"hash": "test", "timestamp": datetime.now(timezone.utc)}],
        developer_stats=[
            {"primary_name": "Alice", "total_commits": 50},
            {"primary_name": "Bob", "total_commits": 30}
        ]
    )
    
    generator = TemplateReportGenerator()
    output = generator.generate(report_data, Path("template_report.txt"))
    
    if output.success:
        print(f"Template report generated: {output.file_path}")


if __name__ == "__main__":
    # Run examples
    print("Running basic usage example...")
    example_basic_usage()
    
    print("\nRunning composite reports example...")
    example_composite_reports()
    
    print("\nRunning custom generator example...")
    example_custom_generator()
    
    print("\nRunning report chaining example...")
    example_report_chaining()
    
    print("\nRunning template-based generation example...")
    example_template_based_generation()