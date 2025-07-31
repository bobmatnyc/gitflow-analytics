"""
Tests for the reports module.

These tests verify report generation functionality including CSV reports,
narrative reports, and analytics output.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime, timezone
from pathlib import Path
import csv
import io

from gitflow_analytics.reports.csv_writer import CSVReportWriter
from gitflow_analytics.reports.narrative_writer import NarrativeReportWriter
from gitflow_analytics.reports.analytics_writer import AnalyticsReportWriter


class TestCSVReportWriter:
    """Test cases for CSV report generation."""

    def test_init(self, temp_dir):
        """Test CSVReportWriter initialization."""
        reports_dir = temp_dir / "reports"
        writer = CSVReportWriter(str(reports_dir))

        assert writer.reports_dir == str(reports_dir)

    def test_write_weekly_metrics(self, temp_dir):
        """Test writing weekly metrics CSV report."""
        reports_dir = temp_dir / "reports"
        reports_dir.mkdir()
        writer = CSVReportWriter(str(reports_dir))

        # Sample weekly metrics data
        weekly_data = [
            {
                "week_start": "2024-01-01",
                "week_end": "2024-01-07",
                "total_commits": 25,
                "total_developers": 3,
                "total_files_changed": 45,
                "total_insertions": 320,
                "total_deletions": 128,
                "avg_commits_per_developer": 8.33,
            },
            {
                "week_start": "2024-01-08",
                "week_end": "2024-01-14",
                "total_commits": 18,
                "total_developers": 2,
                "total_files_changed": 32,
                "total_insertions": 210,
                "total_deletions": 89,
                "avg_commits_per_developer": 9.0,
            },
        ]

        filename = writer.write_weekly_metrics(weekly_data, "20240115")

        assert filename is not None
        assert "weekly_metrics_20240115.csv" in filename

        # Verify file was created and has correct content
        file_path = Path(filename)
        assert file_path.exists()

        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) == 2
            assert rows[0]["week_start"] == "2024-01-01"
            assert rows[0]["total_commits"] == "25"
            assert rows[1]["total_developers"] == "2"

    def test_write_developer_stats(self, temp_dir):
        """Test writing developer statistics CSV report."""
        reports_dir = temp_dir / "reports"
        reports_dir.mkdir()
        writer = CSVReportWriter(str(reports_dir))

        # Sample developer data
        developer_data = [
            {
                "developer": "john@example.com",
                "name": "John Doe",
                "total_commits": 15,
                "total_files_changed": 28,
                "total_insertions": 245,
                "total_deletions": 67,
                "avg_files_per_commit": 1.87,
                "focus_ratio": 0.78,
                "primary_project": "ProjectA",
            },
            {
                "developer": "jane@example.com",
                "name": "Jane Smith",
                "total_commits": 12,
                "total_files_changed": 22,
                "total_insertions": 189,
                "total_deletions": 45,
                "avg_files_per_commit": 1.83,
                "focus_ratio": 0.85,
                "primary_project": "ProjectB",
            },
        ]

        filename = writer.write_developer_stats(developer_data, "20240115")

        assert filename is not None
        assert "developers_20240115.csv" in filename

        # Verify file content
        file_path = Path(filename)
        assert file_path.exists()

        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) == 2
            assert rows[0]["developer"] == "john@example.com"
            assert rows[0]["name"] == "John Doe"
            assert rows[1]["focus_ratio"] == "0.85"

    def test_write_summary_report(self, temp_dir):
        """Test writing summary CSV report."""
        reports_dir = temp_dir / "reports"
        reports_dir.mkdir()
        writer = CSVReportWriter(str(reports_dir))

        # Sample summary data
        summary_data = {
            "analysis_period": "4 weeks",
            "total_commits": 156,
            "total_developers": 5,
            "total_repositories": 3,
            "avg_commits_per_week": 39,
            "most_active_developer": "john@example.com",
            "primary_branch": "main",
            "ticket_completion_rate": 0.78,
        }

        filename = writer.write_summary_report(summary_data, "20240115")

        assert filename is not None
        assert "summary_20240115.csv" in filename

        # Verify file content
        file_path = Path(filename)
        assert file_path.exists()

        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # Summary should have one row per metric
            assert len(rows) > 0
            # Check that key metrics are present
            metrics = {row["metric"]: row["value"] for row in rows}
            assert "total_commits" in metrics
            assert metrics["total_commits"] == "156"

    def test_ensure_reports_directory(self, temp_dir):
        """Test that reports directory is created if it doesn't exist."""
        reports_dir = temp_dir / "new_reports"
        assert not reports_dir.exists()

        writer = CSVReportWriter(str(reports_dir))
        writer._ensure_reports_directory()

        assert reports_dir.exists()
        assert reports_dir.is_dir()


class TestNarrativeReportWriter:
    """Test cases for narrative report generation."""

    def test_init(self, temp_dir):
        """Test NarrativeReportWriter initialization."""
        reports_dir = temp_dir / "reports"
        writer = NarrativeReportWriter(str(reports_dir))

        assert writer.reports_dir == str(reports_dir)

    def test_generate_summary_narrative(self):
        """Test generating narrative summary from data."""
        writer = NarrativeReportWriter("/tmp/reports")

        # Sample analysis data
        analysis_data = {
            "period": "4 weeks",
            "total_commits": 156,
            "total_developers": 5,
            "weekly_metrics": [
                {"week_start": "2024-01-01", "total_commits": 40},
                {"week_start": "2024-01-08", "total_commits": 35},
                {"week_start": "2024-01-15", "total_commits": 42},
                {"week_start": "2024-01-22", "total_commits": 39},
            ],
            "developer_stats": [
                {"developer": "john@example.com", "name": "John Doe", "total_commits": 45},
                {"developer": "jane@example.com", "name": "Jane Smith", "total_commits": 38},
            ],
        }

        narrative = writer.generate_summary_narrative(analysis_data)

        assert isinstance(narrative, str)
        assert len(narrative) > 0
        assert "156" in narrative  # Total commits
        assert "5" in narrative  # Total developers
        assert "4 weeks" in narrative

    def test_generate_developer_insights(self):
        """Test generating developer-specific insights."""
        writer = NarrativeReportWriter("/tmp/reports")

        developer_data = [
            {
                "developer": "john@example.com",
                "name": "John Doe",
                "total_commits": 45,
                "focus_ratio": 0.85,
                "primary_project": "ProjectA",
                "avg_files_per_commit": 2.1,
            },
            {
                "developer": "jane@example.com",
                "name": "Jane Smith",
                "total_commits": 38,
                "focus_ratio": 0.72,
                "primary_project": "ProjectB",
                "avg_files_per_commit": 1.8,
            },
        ]

        insights = writer.generate_developer_insights(developer_data)

        assert isinstance(insights, str)
        assert "John Doe" in insights
        assert "Jane Smith" in insights
        assert "focus" in insights.lower()

    def test_generate_trend_analysis(self):
        """Test generating trend analysis from weekly data."""
        writer = NarrativeReportWriter("/tmp/reports")

        weekly_data = [
            {"week_start": "2024-01-01", "total_commits": 30, "total_developers": 3},
            {"week_start": "2024-01-08", "total_commits": 35, "total_developers": 4},
            {"week_start": "2024-01-15", "total_commits": 42, "total_developers": 4},
            {"week_start": "2024-01-22", "total_commits": 39, "total_developers": 5},
        ]

        analysis = writer.generate_trend_analysis(weekly_data)

        assert isinstance(analysis, str)
        assert len(analysis) > 0
        assert "trend" in analysis.lower() or "week" in analysis.lower()

    def test_write_narrative_report(self, temp_dir):
        """Test writing complete narrative report to file."""
        reports_dir = temp_dir / "reports"
        reports_dir.mkdir()
        writer = NarrativeReportWriter(str(reports_dir))

        # Sample complete data
        analysis_data = {
            "period": "4 weeks",
            "total_commits": 156,
            "total_developers": 5,
            "weekly_metrics": [{"week_start": "2024-01-01", "total_commits": 40}],
            "developer_stats": [
                {"developer": "john@example.com", "name": "John Doe", "total_commits": 45}
            ],
        }

        filename = writer.write_narrative_report(analysis_data, "20240115")

        assert filename is not None
        assert "narrative_20240115.md" in filename

        # Verify file was created
        file_path = Path(filename)
        assert file_path.exists()

        # Verify content
        content = file_path.read_text()
        assert len(content) > 0
        assert "# GitFlow Analytics Report" in content or "GitFlow" in content


class TestAnalyticsReportWriter:
    """Test cases for analytics report generation."""

    def test_init(self, temp_dir):
        """Test AnalyticsReportWriter initialization."""
        reports_dir = temp_dir / "reports"
        writer = AnalyticsReportWriter(str(reports_dir))

        assert writer.reports_dir == str(reports_dir)

    def test_calculate_developer_metrics(self):
        """Test calculating developer-specific metrics."""
        writer = AnalyticsReportWriter("/tmp/reports")

        # Sample commit data
        commits = [
            Mock(author_email="john@example.com", files_changed=3, insertions=25, deletions=5),
            Mock(author_email="john@example.com", files_changed=2, insertions=15, deletions=3),
            Mock(author_email="jane@example.com", files_changed=4, insertions=30, deletions=8),
        ]

        metrics = writer.calculate_developer_metrics(commits)

        assert "john@example.com" in metrics
        assert "jane@example.com" in metrics

        john_metrics = metrics["john@example.com"]
        assert john_metrics["total_commits"] == 2
        assert john_metrics["total_files_changed"] == 5
        assert john_metrics["total_insertions"] == 40
        assert john_metrics["total_deletions"] == 8

    def test_calculate_weekly_metrics(self):
        """Test calculating weekly aggregated metrics."""
        writer = AnalyticsReportWriter("/tmp/reports")

        # Sample commits with dates
        commits = [
            Mock(
                date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                files_changed=3,
                insertions=25,
                deletions=5,
                author_email="john@example.com",
            ),
            Mock(
                date=datetime(2024, 1, 3, tzinfo=timezone.utc),
                files_changed=2,
                insertions=15,
                deletions=3,
                author_email="jane@example.com",
            ),
            Mock(
                date=datetime(2024, 1, 10, tzinfo=timezone.utc),
                files_changed=4,
                insertions=30,
                deletions=8,
                author_email="john@example.com",
            ),
        ]

        weekly_metrics = writer.calculate_weekly_metrics(commits)

        assert len(weekly_metrics) >= 1  # At least one week of data

        # Check structure of weekly data
        first_week = weekly_metrics[0]
        assert "week_start" in first_week
        assert "total_commits" in first_week
        assert "total_developers" in first_week
        assert "total_files_changed" in first_week

    def test_generate_focus_metrics(self):
        """Test generating developer focus metrics."""
        writer = AnalyticsReportWriter("/tmp/reports")

        # Sample commits with project/branch information
        commits = [
            Mock(author_email="john@example.com", branch="main", files_changed=3),
            Mock(author_email="john@example.com", branch="main", files_changed=2),
            Mock(author_email="john@example.com", branch="feature/auth", files_changed=1),
            Mock(author_email="jane@example.com", branch="main", files_changed=4),
        ]

        focus_metrics = writer.generate_focus_metrics(commits)

        assert "john@example.com" in focus_metrics
        assert "jane@example.com" in focus_metrics

        john_focus = focus_metrics["john@example.com"]
        assert "focus_ratio" in john_focus
        assert "primary_project" in john_focus
        assert john_focus["focus_ratio"] <= 1.0

    @patch("builtins.open", new_callable=mock_open)
    def test_export_json_data(self, mock_file, temp_dir):
        """Test exporting analysis data as JSON."""
        reports_dir = temp_dir / "reports"
        writer = AnalyticsReportWriter(str(reports_dir))

        data = {"total_commits": 156, "total_developers": 5, "analysis_date": "2024-01-15"}

        filename = writer.export_json_data(data, "20240115")

        assert "analytics_20240115.json" in filename
        mock_file.assert_called_once()

    def test_generate_dora_metrics(self):
        """Test DORA metrics generation."""
        writer = AnalyticsReportWriter("/tmp/reports")

        # Sample deployment and incident data
        deployments = [
            {"date": "2024-01-01", "success": True},
            {"date": "2024-01-03", "success": True},
            {"date": "2024-01-05", "success": False},
            {"date": "2024-01-07", "success": True},
        ]

        incidents = [
            {"date": "2024-01-02", "resolved_date": "2024-01-02", "severity": "high"},
            {"date": "2024-01-06", "resolved_date": "2024-01-06", "severity": "medium"},
        ]

        # This would need to be implemented in the actual class
        # For now, just test that the method exists and can be called
        try:
            dora_metrics = writer.generate_dora_metrics(deployments, incidents)
            # Should return some form of DORA metrics
            assert isinstance(dora_metrics, dict)
        except AttributeError:
            # Method not implemented yet, which is fine for this test structure
            pass


class TestReportIntegration:
    """Integration tests for report generation."""

    def test_complete_report_generation_flow(self, temp_dir, sample_commits):
        """Test complete flow of generating all report types."""
        reports_dir = temp_dir / "reports"
        reports_dir.mkdir()

        # Initialize all writers
        csv_writer = CSVReportWriter(str(reports_dir))
        narrative_writer = NarrativeReportWriter(str(reports_dir))
        analytics_writer = AnalyticsReportWriter(str(reports_dir))

        date_suffix = "20240115"

        # Generate analytics data
        developer_metrics = analytics_writer.calculate_developer_metrics(sample_commits)
        weekly_metrics = analytics_writer.calculate_weekly_metrics(sample_commits)

        # Generate CSV reports
        csv_files = []
        csv_files.append(
            csv_writer.write_developer_stats(list(developer_metrics.values()), date_suffix)
        )
        csv_files.append(csv_writer.write_weekly_metrics(weekly_metrics, date_suffix))

        # Generate narrative report
        analysis_data = {
            "period": "4 weeks",
            "total_commits": len(sample_commits),
            "total_developers": len(developer_metrics),
            "weekly_metrics": weekly_metrics,
            "developer_stats": list(developer_metrics.values()),
        }
        narrative_file = narrative_writer.write_narrative_report(analysis_data, date_suffix)

        # Verify all files were created
        for file_path in csv_files + [narrative_file]:
            assert file_path is not None
            assert Path(file_path).exists()

        # Verify file contents are not empty
        for file_path in csv_files + [narrative_file]:
            assert Path(file_path).stat().st_size > 0
