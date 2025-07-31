"""
Tests for the reports module.

These tests verify report generation functionality including CSV reports,
narrative reports, and analytics output.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

from gitflow_analytics.reports.analytics_writer import AnalyticsReportGenerator
from gitflow_analytics.reports.csv_writer import CSVReportGenerator
from gitflow_analytics.reports.narrative_writer import NarrativeReportGenerator


class TestCSVReportGenerator:
    """Test cases for CSV report generation."""

    def test_init(self, temp_dir):
        """Test CSVReportGenerator initialization."""
        generator = CSVReportGenerator()
        
        assert generator.anonymize is False
        assert isinstance(generator._anonymization_map, dict)
        assert generator._anonymous_counter == 0

    def test_init_with_anonymization(self, temp_dir):
        """Test CSVReportGenerator initialization with anonymization enabled."""
        generator = CSVReportGenerator(anonymize=True)
        
        assert generator.anonymize is True

    def test_generate_weekly_report(self, temp_dir):
        """Test weekly report generation."""
        generator = CSVReportGenerator()
        output_path = temp_dir / "weekly_report.csv"
        
        # Sample data
        commits = [
            {
                "hash": "abc123",
                "author_name": "John Doe",
                "author_email": "john@example.com",
                "canonical_id": "john@example.com",
                "timestamp": datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                "insertions": 25,
                "deletions": 5,
                "files_changed": 3,
                "story_points": 3,
                "project": "FRONTEND"
            },
            {
                "hash": "def456",
                "author_name": "Jane Smith",
                "author_email": "jane@example.com",
                "canonical_id": "jane@example.com",
                "timestamp": datetime(2024, 1, 2, 14, 0, 0, tzinfo=timezone.utc),
                "insertions": 15,
                "deletions": 8,
                "files_changed": 2,
                "story_points": 2,
                "project": "BACKEND"
            }
        ]
        
        developer_stats = [
            {
                "canonical_id": "john@example.com",
                "canonical_email": "john@example.com",
                "name": "John Doe",
                "total_commits": 1,
                "avg_files_per_commit": 3,
                "story_points": 3
            },
            {
                "canonical_id": "jane@example.com",
                "canonical_email": "jane@example.com",
                "name": "Jane Smith",
                "total_commits": 1,
                "avg_files_per_commit": 2,
                "story_points": 2
            }
        ]
        
        # Generate report
        result_path = generator.generate_weekly_report(commits, developer_stats, output_path, weeks=4)
        
        assert result_path == output_path
        assert output_path.exists()
        
        # Verify file content structure
        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Should have data for weeks
            assert len(rows) > 0
            
            # Check column names
            if rows:
                expected_columns = {"week_start", "developer", "name", "commits", "story_points"}
                assert expected_columns.issubset(set(rows[0].keys()))

    def test_generate_developer_report(self, temp_dir):
        """Test developer summary report generation."""
        generator = CSVReportGenerator()
        output_path = temp_dir / "developer_report.csv"
        
        developer_stats = [
            {
                "canonical_id": "john@example.com",
                "canonical_email": "john@example.com", 
                "name": "John Doe",
                "total_commits": 25,
                "avg_files_per_commit": 2.5,
                "story_points": 45,
                "focus_ratio": 0.85
            },
            {
                "canonical_id": "jane@example.com",
                "canonical_email": "jane@example.com",
                "name": "Jane Smith", 
                "total_commits": 18,
                "avg_files_per_commit": 3.2,
                "story_points": 38,
                "focus_ratio": 0.75
            }
        ]
        
        result_path = generator.generate_developer_report(developer_stats, output_path)
        
        assert result_path == output_path
        assert output_path.exists()
        
        # Verify file content
        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            assert len(rows) == 2
            
            # Check data values
            john_row = next((r for r in rows if r["name"] == "John Doe"), None)
            assert john_row is not None
            assert john_row["total_commits"] == "25"


class TestAnalyticsReportGenerator:
    """Test cases for analytics report generation."""

    def test_init(self):
        """Test AnalyticsReportGenerator initialization."""
        generator = AnalyticsReportGenerator()
        
        assert generator.anonymize is False
        assert isinstance(generator._anonymization_map, dict)

    def test_init_with_anonymization(self):
        """Test AnalyticsReportGenerator initialization with anonymization."""
        generator = AnalyticsReportGenerator(anonymize=True)
        
        assert generator.anonymize is True

    def test_generate_activity_distribution_report(self, temp_dir):
        """Test activity distribution report generation."""
        generator = AnalyticsReportGenerator()
        output_path = temp_dir / "activity_dist.csv"
        
        commits = [
            {
                "canonical_id": "john@example.com",
                "insertions": 25,
                "deletions": 5,
                "files_changed": 3,
                "project": "FRONTEND"
            },
            {
                "canonical_id": "jane@example.com", 
                "insertions": 15,
                "deletions": 8,
                "files_changed": 2,
                "project": "BACKEND"
            }
        ]
        
        developer_stats = [
            {
                "canonical_id": "john@example.com",
                "name": "John Doe",
                "total_commits": 1
            },
            {
                "canonical_id": "jane@example.com",
                "name": "Jane Smith",
                "total_commits": 1
            }
        ]
        
        result_path = generator.generate_activity_distribution_report(commits, developer_stats, output_path)
        
        assert result_path == output_path
        assert output_path.exists()
        
        # Basic structure validation
        with open(output_path) as f:
            content = f.read()
            assert len(content) > 0


class TestNarrativeReportGenerator:
    """Test cases for narrative report generation."""

    def test_init(self):
        """Test NarrativeReportGenerator initialization."""
        generator = NarrativeReportGenerator()
        
        assert isinstance(generator.templates, dict)
        assert len(generator.templates) > 0

    def test_generate_narrative_report(self, temp_dir):
        """Test narrative report generation."""
        generator = NarrativeReportGenerator()
        output_path = temp_dir / "narrative.md"
        
        # Sample data
        commits = [
            {
                "hash": "abc123",
                "author_name": "John Doe",
                "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "project": "FRONTEND"
            }
        ]
        
        prs = [
            {
                "number": 123,
                "title": "Add feature",
                "author": "john@example.com"
            }
        ]
        
        developer_stats = [
            {
                "canonical_id": "john@example.com",
                "name": "John Doe",
                "total_commits": 25
            }
        ]
        
        activity_dist = [
            {
                "developer": "john@example.com",
                "commits_pct": 65.0
            }
        ]
        
        focus_data = [
            {
                "developer": "john@example.com", 
                "primary_project": "FRONTEND",
                "focus_ratio": 0.85
            }
        ]
        
        insights = [
            {
                "metric": "total_commits",
                "value": 25
            }
        ]
        
        ticket_analysis = {
            "coverage_percentage": 85.0,
            "total_commits": 25
        }
        
        pr_metrics = {
            "avg_pr_size": 150,
            "total_prs": 10
        }
        
        dora_metrics = {
            "deployment_frequency": {"daily_average": 0.5},
            "lead_time_hours": 48.0
        }
        
        result_path = generator.generate_narrative_report(
            commits, prs, developer_stats, activity_dist, focus_data,
            insights, ticket_analysis, pr_metrics, output_path, dora_metrics
        )
        
        assert result_path == output_path
        assert output_path.exists()
        
        # Verify it's a markdown file with content
        with open(output_path) as f:
            content = f.read()
            assert len(content) > 0
            assert "# GitFlow Analytics Report" in content or "#" in content  # Some markdown header


class TestReportingIntegration:
    """Test cases for integrated reporting functionality."""

    def test_multiple_report_generation(self, temp_dir):
        """Test generating multiple report types together."""
        csv_gen = CSVReportGenerator()
        analytics_gen = AnalyticsReportGenerator()
        narrative_gen = NarrativeReportGenerator()
        
        # Sample data
        commits = [{
            "hash": "abc123",
            "author_name": "John Doe",
            "author_email": "john@example.com",
            "canonical_id": "john@example.com",
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "insertions": 25,
            "deletions": 5,
            "files_changed": 3,
            "project": "FRONTEND"
        }]
        
        developer_stats = [{
            "canonical_id": "john@example.com",
            "canonical_email": "john@example.com",
            "name": "John Doe",
            "total_commits": 1,
            "avg_files_per_commit": 3.0
        }]
        
        # Generate all report types
        csv_path = csv_gen.generate_weekly_report(commits, developer_stats, temp_dir / "weekly.csv")
        dev_path = csv_gen.generate_developer_report(developer_stats, temp_dir / "developers.csv")
        activity_path = analytics_gen.generate_activity_distribution_report(commits, developer_stats, temp_dir / "activity.csv")
        
        # All reports should be generated
        assert csv_path.exists()
        assert dev_path.exists()
        assert activity_path.exists()
        
        # All should be valid CSV files
        for path in [csv_path, dev_path, activity_path]:
            with open(path) as f:
                content = f.read()
                assert len(content) > 0
                assert "," in content  # Basic CSV validation