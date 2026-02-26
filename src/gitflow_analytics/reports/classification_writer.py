"""Classification report generator for GitFlow Analytics.

This module provides comprehensive reporting capabilities for commit classification
results, including aggregate statistics, developer breakdowns, confidence analysis,
and temporal patterns. Designed to integrate with existing GitFlow Analytics
reporting infrastructure.

Detailed CSV reports are implemented in classification_csv.ClassificationCsvMixin.
"""

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .classification_csv import ClassificationCsvMixin

logger = logging.getLogger(__name__)


class ClassificationReportGenerator(ClassificationCsvMixin):
    """Generator for comprehensive commit classification reports.

    This class creates detailed reports from commit classification results,
    providing insights into development patterns, team productivity, and
    code quality metrics through the lens of commit categorization.

    Key capabilities:
    - Aggregate classification statistics
    - Per-developer activity breakdowns
    - Per-repository analysis
    - Confidence score analysis
    - Temporal pattern identification
    - Export to multiple formats (CSV, JSON, Markdown)
    """

    def __init__(self, output_directory: Path, config: Optional[Dict[str, Any]] = None):
        """Initialize the classification report generator.

        Args:
            output_directory: Directory where reports will be saved
            config: Optional configuration for report generation
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.config = config or {}
        self.include_low_confidence = self.config.get("include_low_confidence", True)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        self.min_commits_for_analysis = self.config.get("min_commits_for_analysis", 5)

        # Report metadata
        self.generated_at = datetime.now()
        self.reports_generated = []

        logger.info(
            f"Classification report generator initialized - output: {self.output_directory}"
        )

    def generate_comprehensive_report(
        self, classified_commits: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Generate all available classification reports.

        Args:
            classified_commits: List of commits with classification results
            metadata: Optional metadata about the analysis (date range, repos, etc.)

        Returns:
            Dictionary mapping report types to file paths
        """
        if not classified_commits:
            logger.warning("No classified commits provided - skipping report generation")
            return {}

        # Filter classified commits
        classified_only = [c for c in classified_commits if "predicted_class" in c]

        if not classified_only:
            logger.warning("No commits with classification results found")
            return {}

        logger.info(
            f"Generating comprehensive classification reports for {len(classified_only)} commits"
        )

        report_paths = {}

        try:
            # Generate individual reports
            report_paths["summary"] = self.generate_summary_report(classified_only, metadata)
            report_paths["detailed_csv"] = self.generate_detailed_csv_report(
                classified_only, metadata
            )
            report_paths["developer_breakdown"] = self.generate_developer_breakdown_report(
                classified_only, metadata
            )
            report_paths["repository_analysis"] = self.generate_repository_analysis_report(
                classified_only, metadata
            )
            report_paths["confidence_analysis"] = self.generate_confidence_analysis_report(
                classified_only, metadata
            )
            report_paths["temporal_patterns"] = self.generate_temporal_patterns_report(
                classified_only, metadata
            )
            report_paths["classification_matrix"] = self.generate_classification_matrix_report(
                classified_only, metadata
            )
            report_paths["executive_summary"] = self.generate_executive_summary_report(
                classified_only, metadata
            )

            # Generate comprehensive JSON export
            report_paths["comprehensive_json"] = self.generate_json_export(
                classified_only, metadata
            )

            # Generate markdown summary
            report_paths["markdown_summary"] = self.generate_markdown_summary(
                classified_only, metadata
            )

            self.reports_generated = list(report_paths.keys())
            logger.info(f"Generated {len(report_paths)} classification reports")

            return report_paths

        except Exception as e:
            logger.error(f"Failed to generate comprehensive reports: {e}")
            return report_paths

    def generate_summary_report(
        self, classified_commits: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate high-level summary report.

        Args:
            classified_commits: List of classified commits
            metadata: Optional analysis metadata

        Returns:
            Path to generated summary CSV file
        """
        output_path = self.output_directory / f"classification_summary_{self._get_timestamp()}.csv"

        # Calculate summary statistics
        total_commits = len(classified_commits)
        classification_counts = Counter(c["predicted_class"] for c in classified_commits)
        confidence_scores = [c.get("classification_confidence", 0) for c in classified_commits]

        high_confidence_count = sum(
            1 for score in confidence_scores if score >= self.confidence_threshold
        )
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        unique_developers = len(
            set(
                c.get("canonical_author_name", c.get("author_name", "unknown"))
                for c in classified_commits
            )
        )
        unique_repositories = len(set(c.get("repository", "unknown") for c in classified_commits))

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header information
            writer.writerow(["Classification Analysis Summary"])
            writer.writerow(["Generated:", self.generated_at.isoformat()])

            if metadata:
                writer.writerow(
                    [
                        "Analysis Period:",
                        f"{metadata.get('start_date', 'N/A')} to {metadata.get('end_date', 'N/A')}",
                    ]
                )
                writer.writerow(["Configuration:", metadata.get("config_path", "N/A")])

            writer.writerow([])

            # Overall statistics
            writer.writerow(["Overall Statistics"])
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Total Commits Analyzed", total_commits])
            writer.writerow(["Unique Developers", unique_developers])
            writer.writerow(["Unique Repositories", unique_repositories])
            writer.writerow(["Average Confidence Score", f"{avg_confidence:.3f}"])
            writer.writerow(
                [
                    "High Confidence Predictions",
                    f"{high_confidence_count} ({(high_confidence_count/total_commits)*100:.1f}%)",
                ]
            )
            writer.writerow([])

            # Classification distribution
            writer.writerow(["Classification Distribution"])
            writer.writerow(["Classification Type", "Count", "Percentage"])

            for class_type, count in classification_counts.most_common():
                percentage = (count / total_commits) * 100
                writer.writerow([class_type, count, f"{percentage:.1f}%"])

        logger.info(f"Summary report generated: {output_path}")
        return str(output_path)

    # generate_detailed_csv_report is in ClassificationCsvMixin (classification_csv.py)

    def generate_executive_summary_report(
        self, classified_commits: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate executive summary report for leadership.

        Args:
            classified_commits: List of classified commits
            metadata: Optional analysis metadata

        Returns:
            Path to generated executive summary CSV file
        """
        output_path = (
            self.output_directory / f"classification_executive_summary_{self._get_timestamp()}.csv"
        )

        # Calculate key metrics
        total_commits = len(classified_commits)
        unique_developers = len(
            set(
                c.get("canonical_author_name", c.get("author_name", "unknown"))
                for c in classified_commits
            )
        )
        unique_repositories = len(set(c.get("repository", "unknown") for c in classified_commits))

        classification_counts = Counter(
            c.get("predicted_class", "unknown") for c in classified_commits
        )
        confidence_scores = [c.get("classification_confidence", 0) for c in classified_commits]

        # Productivity metrics
        total_lines_changed = sum(
            c.get("insertions", 0) + c.get("deletions", 0) for c in classified_commits
        )
        avg_lines_per_commit = total_lines_changed / total_commits if total_commits > 0 else 0

        # Time span analysis
        commit_dates = [c["timestamp"] for c in classified_commits if c.get("timestamp")]
        if commit_dates:
            analysis_span = (max(commit_dates) - min(commit_dates)).days
        else:
            analysis_span = 0

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(["Executive Summary - Commit Classification Analysis"])
            writer.writerow(["Generated:", self.generated_at.strftime("%Y-%m-%d %H:%M:%S")])

            if metadata:
                writer.writerow(
                    [
                        "Analysis Period:",
                        f"{metadata.get('start_date', 'N/A')} to {metadata.get('end_date', 'N/A')}",
                    ]
                )

            writer.writerow([])

            # Key metrics
            writer.writerow(["KEY METRICS"])
            writer.writerow(["Total Development Activity", f"{total_commits:,} commits"])
            writer.writerow(["Team Size", f"{unique_developers} active developers"])
            writer.writerow(["Codebase Scope", f"{unique_repositories} repositories"])
            writer.writerow(["Analysis Timespan", f"{analysis_span} days"])
            writer.writerow(
                ["Average Code Changes per Commit", f"{avg_lines_per_commit:.0f} lines"]
            )

            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                high_confidence_pct = (
                    sum(1 for s in confidence_scores if s >= self.confidence_threshold)
                    / len(confidence_scores)
                ) * 100
                writer.writerow(["Classification Confidence", f"{avg_confidence:.1%} average"])
                writer.writerow(["High Confidence Predictions", f"{high_confidence_pct:.1f}%"])

            writer.writerow([])

            # Development focus areas
            writer.writerow(["DEVELOPMENT FOCUS AREAS"])
            writer.writerow(["Activity Type", "Commits", "% of Total", "Strategic Insight"])

            # Define strategic insights for each classification type
            strategic_insights = {
                "feature": "New capability development",
                "bugfix": "Quality maintenance and stability",
                "refactor": "Technical debt management",
                "docs": "Knowledge management and documentation",
                "test": "Quality assurance and testing",
                "config": "Infrastructure and configuration",
                "chore": "Maintenance and operational tasks",
                "security": "Security and compliance",
                "hotfix": "Critical issue resolution",
                "style": "Code quality and standards",
                "build": "Build system and deployment",
                "ci": "Automation and continuous integration",
            }

            for class_type, count in classification_counts.most_common():
                percentage = (count / total_commits) * 100
                insight = strategic_insights.get(class_type, "Unclassified development activity")
                writer.writerow([class_type.title(), f"{count:,}", f"{percentage:.1f}%", insight])

            writer.writerow([])

            # Recommendations
            writer.writerow(["STRATEGIC RECOMMENDATIONS"])

            # Generate recommendations based on the data
            recommendations = []

            # Feature vs maintenance balance
            feature_pct = (classification_counts.get("feature", 0) / total_commits) * 100
            maintenance_pct = (
                (
                    classification_counts.get("bugfix", 0)
                    + classification_counts.get("refactor", 0)
                    + classification_counts.get("chore", 0)
                )
                / total_commits
            ) * 100

            if feature_pct > 60:
                recommendations.append(
                    "High feature development velocity - consider increasing quality assurance"
                )
            elif feature_pct < 20:
                recommendations.append(
                    "Low feature development - may indicate focus on maintenance or technical debt"
                )

            if maintenance_pct > 40:
                recommendations.append(
                    "High maintenance overhead - consider technical debt reduction initiatives"
                )

            # Documentation analysis
            docs_pct = (classification_counts.get("docs", 0) / total_commits) * 100
            if docs_pct < 5:
                recommendations.append(
                    "Low documentation activity - consider improving documentation practices"
                )

            # Testing analysis
            test_pct = (classification_counts.get("test", 0) / total_commits) * 100
            if test_pct < 10:
                recommendations.append(
                    "Limited testing activity - consider strengthening testing practices"
                )

            # Security analysis
            security_pct = (classification_counts.get("security", 0) / total_commits) * 100
            if security_pct > 0:
                recommendations.append(
                    f"Active security focus ({security_pct:.1f}% of commits) - positive security posture"
                )

            # Confidence analysis
            if confidence_scores:
                low_confidence_pct = (
                    sum(1 for s in confidence_scores if s < 0.6) / len(confidence_scores)
                ) * 100
                if low_confidence_pct > 20:
                    recommendations.append(
                        "Consider improving commit message clarity for better classification"
                    )

            for i, recommendation in enumerate(recommendations, 1):
                writer.writerow([f"Recommendation {i}", recommendation])

        logger.info(f"Executive summary report generated: {output_path}")
        return str(output_path)

    def generate_json_export(
        self, classified_commits: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate comprehensive JSON export of all classification data.

        Args:
            classified_commits: List of classified commits
            metadata: Optional analysis metadata

        Returns:
            Path to generated JSON file
        """
        output_path = (
            self.output_directory / f"classification_comprehensive_{self._get_timestamp()}.json"
        )

        # Create comprehensive data structure
        export_data = {
            "metadata": {
                "generated_at": self.generated_at.isoformat(),
                "total_commits": len(classified_commits),
                "generator_version": "1.0",
                "config": self.config,
            },
            "summary_statistics": self._calculate_summary_statistics(classified_commits),
            "commits": classified_commits,
            "analysis_metadata": metadata or {},
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"JSON export generated: {output_path}")
        return str(output_path)

    def generate_markdown_summary(
        self, classified_commits: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate markdown summary report.

        Args:
            classified_commits: List of classified commits
            metadata: Optional analysis metadata

        Returns:
            Path to generated markdown file
        """
        output_path = self.output_directory / f"classification_summary_{self._get_timestamp()}.md"

        # Calculate statistics
        total_commits = len(classified_commits)
        classification_counts = Counter(
            c.get("predicted_class", "unknown") for c in classified_commits
        )
        confidence_scores = [c.get("classification_confidence", 0) for c in classified_commits]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Commit Classification Analysis Report\n\n")
            f.write(f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if metadata:
                f.write(
                    f"**Analysis Period:** {metadata.get('start_date', 'N/A')} to {metadata.get('end_date', 'N/A')}\n\n"
                )

            f.write("## Summary Statistics\n\n")
            f.write(f"- **Total Commits Analyzed:** {total_commits:,}\n")
            f.write(
                f"- **Unique Developers:** {len(set(c.get('canonical_author_name', c.get('author_name', 'unknown')) for c in classified_commits))}\n"
            )
            f.write(
                f"- **Unique Repositories:** {len(set(c.get('repository', 'unknown') for c in classified_commits))}\n"
            )

            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                high_confidence_count = sum(
                    1 for s in confidence_scores if s >= self.confidence_threshold
                )
                f.write(f"- **Average Confidence:** {avg_confidence:.1%}\n")
                f.write(
                    f"- **High Confidence Predictions:** {high_confidence_count:,} ({(high_confidence_count/total_commits)*100:.1f}%)\n"
                )

            f.write("\n## Classification Distribution\n\n")
            f.write("| Classification Type | Count | Percentage |\n")
            f.write("|-------------------|--------|------------|\n")

            for class_type, count in classification_counts.most_common():
                percentage = (count / total_commits) * 100
                f.write(f"| {class_type.title()} | {count:,} | {percentage:.1f}% |\n")

            f.write("\n## Analysis Details\n\n")
            f.write(
                "This report was generated using GitFlow Analytics commit classification system.\n"
            )
            f.write(f"Classification confidence threshold: {self.confidence_threshold}\n\n")

            f.write("For detailed analysis, see the accompanying CSV reports:\n")
            for report_type in self.reports_generated:
                f.write(f"- {report_type.replace('_', ' ').title()}\n")

        logger.info(f"Markdown summary generated: {output_path}")
        return str(output_path)

    def _calculate_summary_statistics(
        self, classified_commits: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics.

        Args:
            classified_commits: List of classified commits

        Returns:
            Dictionary containing summary statistics
        """
        total_commits = len(classified_commits)

        classification_counts = Counter(
            c.get("predicted_class", "unknown") for c in classified_commits
        )
        confidence_scores = [c.get("classification_confidence", 0) for c in classified_commits]

        developers = set(
            c.get("canonical_author_name", c.get("author_name", "unknown"))
            for c in classified_commits
        )
        repositories = set(c.get("repository", "unknown") for c in classified_commits)

        return {
            "total_commits": total_commits,
            "unique_developers": len(developers),
            "unique_repositories": len(repositories),
            "classification_distribution": dict(classification_counts),
            "confidence_statistics": {
                "average": sum(confidence_scores) / len(confidence_scores)
                if confidence_scores
                else 0,
                "minimum": min(confidence_scores) if confidence_scores else 0,
                "maximum": max(confidence_scores) if confidence_scores else 0,
                "high_confidence_count": sum(
                    1 for s in confidence_scores if s >= self.confidence_threshold
                ),
                "high_confidence_percentage": (
                    sum(1 for s in confidence_scores if s >= self.confidence_threshold)
                    / len(confidence_scores)
                )
                * 100
                if confidence_scores
                else 0,
            },
            "productivity_metrics": {
                "total_lines_changed": sum(
                    c.get("insertions", 0) + c.get("deletions", 0) for c in classified_commits
                ),
                "average_lines_per_commit": sum(
                    c.get("insertions", 0) + c.get("deletions", 0) for c in classified_commits
                )
                / total_commits
                if total_commits > 0
                else 0,
                "average_files_per_commit": sum(
                    c.get("files_changed", 0) for c in classified_commits
                )
                / total_commits
                if total_commits > 0
                else 0,
            },
        }

    def _get_timestamp(self) -> str:
        """Get timestamp string for file naming.

        Returns:
            Timestamp string in YYYYMMDD_HHMMSS format
        """
        return self.generated_at.strftime("%Y%m%d_%H%M%S")
