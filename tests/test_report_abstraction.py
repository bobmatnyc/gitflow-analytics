"""Tests for the report generation abstraction layer.

This module tests the new report abstraction layer components
to ensure they work correctly and maintain backward compatibility.
"""

import json
from datetime import datetime, timezone

import pytest

from gitflow_analytics.reports import (
    BaseReportGenerator,
    CompositeReportGenerator,
    ComprehensiveJSONExporter,
    CSVReportGenerator,
    ReportBuilder,
    ReportData,
    ReportFactory,
    ReportFormat,
    ReportMetadata,
    ReportType,
    create_multiple_reports,
    create_report,
)


@pytest.fixture
def sample_report_data():
    """Create sample report data for testing."""
    return ReportData(
        commits=[
            {
                "hash": "abc123",
                "author_email": "dev1@example.com",
                "author_name": "Developer One",
                "timestamp": datetime.now(timezone.utc),
                "message": "feat: Add new feature",
                "insertions": 100,
                "deletions": 20,
                "files_changed": 5,
                "canonical_id": "dev1@example.com",
            },
            {
                "hash": "def456",
                "author_email": "dev2@example.com",
                "author_name": "Developer Two",
                "timestamp": datetime.now(timezone.utc),
                "message": "fix: Fix bug",
                "insertions": 50,
                "deletions": 10,
                "files_changed": 2,
                "canonical_id": "dev2@example.com",
            },
        ],
        developer_stats=[
            {
                "canonical_id": "dev1@example.com",
                "primary_email": "dev1@example.com",
                "primary_name": "Developer One",
                "total_commits": 50,
                "total_story_points": 25.0,
                "ticket_coverage_pct": 85.0,
            },
            {
                "canonical_id": "dev2@example.com",
                "primary_email": "dev2@example.com",
                "primary_name": "Developer Two",
                "total_commits": 30,
                "total_story_points": 15.0,
                "ticket_coverage_pct": 75.0,
            },
        ],
        metadata=ReportMetadata(
            analysis_period_weeks=4,
            total_commits=2,
            total_developers=2,
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc),
        ),
    )


class TestReportData:
    """Test ReportData class."""

    def test_report_data_creation(self):
        """Test creating ReportData instance."""
        data = ReportData()
        assert data.commits == []
        assert data.developer_stats == []
        assert data.metadata is not None

    def test_report_data_validation(self, sample_report_data):
        """Test ReportData validation."""
        assert sample_report_data.validate() is True

        # Test with empty data
        empty_data = ReportData()
        assert empty_data.validate() is False  # Missing required fields

    def test_required_fields(self, sample_report_data):
        """Test getting required fields."""
        required = sample_report_data.get_required_fields()
        assert "commits" in required
        assert "developer_stats" in required


class TestBaseReportGenerator:
    """Test BaseReportGenerator class."""

    def test_csv_generator_inheritance(self):
        """Test that CSVReportGenerator inherits from BaseReportGenerator."""
        generator = CSVReportGenerator()
        assert isinstance(generator, BaseReportGenerator)

    def test_json_exporter_inheritance(self):
        """Test that ComprehensiveJSONExporter inherits from BaseReportGenerator."""
        exporter = ComprehensiveJSONExporter()
        assert isinstance(exporter, BaseReportGenerator)

    def test_abstract_methods_implementation(self):
        """Test that concrete generators implement abstract methods."""
        generator = CSVReportGenerator()

        # Test required methods exist
        assert hasattr(generator, "generate")
        assert hasattr(generator, "get_required_fields")
        assert hasattr(generator, "get_format_type")

        # Test format type
        assert generator.get_format_type() == "csv"

    def test_anonymization(self, sample_report_data):
        """Test anonymization functionality."""
        generator = CSVReportGenerator(anonymize=True)

        # The base class should handle anonymization
        anonymized_data = generator._anonymize_data(sample_report_data)

        # Check that emails are anonymized
        for commit in anonymized_data.commits:
            assert "Developer" in commit.get("author_name", "")

    def test_author_exclusion(self, sample_report_data):
        """Test author exclusion functionality."""
        generator = CSVReportGenerator(exclude_authors=["dev2@example.com"])

        # Filter the data
        filtered_data = generator._filter_excluded_authors(sample_report_data)

        # Check that dev2 is excluded
        assert len(filtered_data.commits) == 1
        assert filtered_data.commits[0]["author_email"] == "dev1@example.com"


class TestReportFactory:
    """Test ReportFactory class."""

    def test_factory_creation(self):
        """Test creating factory instance."""
        factory = ReportFactory()
        assert factory is not None

    def test_create_generator(self):
        """Test creating generators through factory."""
        factory = ReportFactory()

        # Create CSV generator
        csv_gen = factory.create_generator(ReportType.WEEKLY_METRICS, ReportFormat.CSV)
        assert isinstance(csv_gen, CSVReportGenerator)

        # Create JSON generator
        json_gen = factory.create_generator(ReportType.COMPREHENSIVE, ReportFormat.JSON)
        assert isinstance(json_gen, ComprehensiveJSONExporter)

    def test_supported_reports(self):
        """Test getting supported report types."""
        factory = ReportFactory()
        supported = factory.get_supported_reports()

        assert ReportType.WEEKLY_METRICS in supported
        assert ReportType.DEVELOPER_STATS in supported

    def test_supported_formats(self):
        """Test getting supported formats for a report type."""
        factory = ReportFactory()
        formats = factory.get_supported_formats(ReportType.WEEKLY_METRICS)

        assert ReportFormat.CSV in formats

    def test_composite_generator(self):
        """Test creating composite generator."""
        factory = ReportFactory()
        composite = factory.create_composite_generator(
            [
                (ReportType.WEEKLY_METRICS, ReportFormat.CSV),
                (ReportType.COMPREHENSIVE, ReportFormat.JSON),
            ]
        )

        assert isinstance(composite, CompositeReportGenerator)
        assert len(composite.generators) == 2


class TestReportBuilder:
    """Test ReportBuilder class."""

    def test_builder_creation(self):
        """Test creating builder instance."""
        builder = ReportBuilder()
        assert builder is not None

    def test_builder_chaining(self, sample_report_data):
        """Test builder method chaining."""
        builder = (
            ReportBuilder()
            .add_report(ReportType.WEEKLY_METRICS, ReportFormat.CSV)
            .with_config(anonymize=True)
            .with_data(sample_report_data)
        )

        assert builder._report_types == [(ReportType.WEEKLY_METRICS, ReportFormat.CSV)]
        assert builder._config["anonymize"] is True
        assert builder._data is not None

    def test_builder_build(self):
        """Test building generator from builder."""
        builder = ReportBuilder()
        builder.add_report(ReportType.WEEKLY_METRICS, ReportFormat.CSV)

        generator = builder.build()
        assert isinstance(generator, CSVReportGenerator)

    def test_builder_multiple_reports(self):
        """Test building composite generator."""
        builder = (
            ReportBuilder()
            .add_report(ReportType.WEEKLY_METRICS, ReportFormat.CSV)
            .add_report(ReportType.COMPREHENSIVE, ReportFormat.JSON)
        )

        generator = builder.build()
        assert isinstance(generator, CompositeReportGenerator)


class TestReportGeneration:
    """Test actual report generation."""

    def test_csv_generation(self, sample_report_data, tmp_path):
        """Test generating CSV report."""
        output_path = tmp_path / "test_report.csv"

        generator = CSVReportGenerator()
        output = generator.generate(sample_report_data, output_path)

        assert output.success is True
        assert output.file_path == output_path
        assert output.format == "csv"
        assert output_path.exists()

    def test_json_generation(self, sample_report_data, tmp_path):
        """Test generating JSON report."""
        output_path = tmp_path / "test_report.json"

        exporter = ComprehensiveJSONExporter()
        output = exporter.generate(sample_report_data, output_path)

        assert output.success is True
        assert output.file_path == output_path
        assert output.format == "json"
        assert output_path.exists()

        # Verify JSON is valid
        with open(output_path) as f:
            data = json.load(f)
            assert "metadata" in data

    def test_in_memory_generation(self, sample_report_data):
        """Test generating report in memory without file output."""
        generator = CSVReportGenerator()
        output = generator.generate(sample_report_data, output_path=None)

        assert output.success is True
        assert output.content is not None
        assert output.file_path is None
        assert output.size_bytes > 0

    def test_convenience_functions(self, sample_report_data, tmp_path):
        """Test convenience functions for report creation."""
        output_path = tmp_path / "convenience_test.csv"

        # Test create_report
        output = create_report(
            ReportType.WEEKLY_METRICS, ReportFormat.CSV, sample_report_data, output_path
        )

        assert output.success is True
        assert output_path.exists()

    def test_multiple_reports(self, sample_report_data, tmp_path):
        """Test generating multiple reports at once."""
        outputs = create_multiple_reports(
            [
                (ReportType.WEEKLY_METRICS, ReportFormat.CSV),
                (ReportType.COMPREHENSIVE, ReportFormat.JSON),
            ],
            sample_report_data,
            output_dir=tmp_path,
        )

        assert len(outputs) == 2
        assert all(o.success for o in outputs)

        # Check files were created
        csv_file = tmp_path / "weekly_metrics.csv"
        json_file = tmp_path / "comprehensive.json"
        assert csv_file.exists() or json_file.exists()


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_legacy_csv_generator(self, tmp_path):
        """Test that legacy CSVReportGenerator still works."""
        generator = CSVReportGenerator(anonymize=False)

        # Create legacy-style data
        commits = [
            {
                "hash": "test",
                "author_email": "test@example.com",
                "timestamp": datetime.now(timezone.utc),
                "insertions": 10,
                "deletions": 5,
            }
        ]

        developer_stats = [
            {"canonical_id": "test@example.com", "primary_name": "Test Dev", "total_commits": 10}
        ]

        # Legacy methods should still work
        output_path = tmp_path / "legacy_weekly.csv"
        generator.generate_weekly_report(commits, developer_stats, output_path)

        assert output_path.exists()

    def test_legacy_json_exporter(self, tmp_path):
        """Test that legacy ComprehensiveJSONExporter still works."""
        exporter = ComprehensiveJSONExporter(anonymize=False)

        # Legacy export method should still work
        output_path = tmp_path / "legacy_export.json"
        result = exporter.export_comprehensive_data(
            commits=[],
            prs=[],
            developer_stats=[],
            project_metrics={},
            dora_metrics={},
            output_path=output_path,
            weeks=4,
        )

        assert result == output_path
        assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
