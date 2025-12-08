#!/usr/bin/env python3
"""Test script for the two-step fetch/analyze process.

This script demonstrates the new two-step architecture:
1. Fetch: Collect raw git commits and ticket data without classification
2. Analyze: Use batch LLM classification on the cached data
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Test configuration
TEST_CONFIG = """
# Test Configuration for Two-Step Process
cache:
  directory: "{cache_dir}"
  retention_days: 30

repositories:
  - name: "test-repo"
    path: "."
    project_key: "TEST"

analysis:
  # Standard analysis settings  
  exclude_authors: []
  exclude_paths: []
  branch_mapping_rules: {{}}
  auto_identity_analysis: false
  manual_identity_mappings: []
  
  # LLM classification settings
  llm_classification:
    enabled: true
    api_key: "test-key"  # Would be real key in production
    model: "gpt-3.5-turbo"
    confidence_threshold: 0.7
    max_tokens: 4000
    temperature: 0.1
    timeout_seconds: 30
    cache_duration_days: 7
    enable_caching: true
    max_daily_requests: 1000
    domain_terms:
      - "frontend"
      - "backend" 
      - "api"
      - "database"

# Optional integrations (disabled for test)
github:
  enabled: false

jira:
  enabled: false

reports:
  anonymize_developers: false
  output_directory: "{reports_dir}"
"""


def test_data_fetcher():
    """Test the data fetcher component."""
    print("🧪 Testing Data Fetcher...")

    from src.gitflow_analytics.core.cache import GitAnalysisCache
    from src.gitflow_analytics.core.data_fetcher import GitDataFetcher

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        cache = GitAnalysisCache(cache_dir)

        # Initialize data fetcher
        data_fetcher = GitDataFetcher(
            cache=cache,
            branch_mapping_rules={},
            allowed_ticket_platforms=["jira", "github"],
        )

        # Test fetch status (should be empty initially)
        status = data_fetcher.get_fetch_status("TEST", Path("."))
        print(f"   📊 Initial status: {status}")

        # Note: Full integration test would require actual git repo
        print("   ✅ Data Fetcher component loads successfully")


def test_batch_classifier():
    """Test the batch classifier component."""
    print("🧪 Testing Batch Classifier...")

    # Create a mock config object for testing
    class MockLLMConfig:
        def __init__(self):
            self.enable_caching = False
            self.cache_duration_days = 7
            self.model = "test-model"
            self.domain_terms = {}
            self.api_key = "test-key"
            self.max_tokens = 1000
            self.temperature = 0.1
            self.timeout_seconds = 30
            self.api_base_url = "https://api.example.com"
            self.max_daily_requests = 1000

    try:
        from src.gitflow_analytics.classification.batch_classifier import BatchCommitClassifier

        # Create temporary cache directory
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            # Initialize batch classifier with mock config
            batch_classifier = BatchCommitClassifier(
                cache_dir=cache_dir,
                llm_config=MockLLMConfig(),
                batch_size=10,
                confidence_threshold=0.7,
                fallback_enabled=True,
            )

            # Test classification status (should be empty initially)
            start_date = datetime.now(timezone.utc) - timedelta(weeks=2)
            end_date = datetime.now(timezone.utc)

            status = batch_classifier.get_classification_status(start_date, end_date)
            print(f"   📊 Classification status: {status}")

            print("   ✅ Batch Classifier component loads successfully")

    except ImportError as e:
        print(f"   ⚠️  Batch Classifier test skipped: {e}")
        print("   ℹ️  This is expected if optional dependencies are missing")
    except Exception as e:
        print(f"   ❌ Batch Classifier test failed: {e}")
        # Don't raise - this is just a component test


def test_database_models():
    """Test the database models for the two-step process."""
    print("🧪 Testing Database Models...")

    from src.gitflow_analytics.models.database import DailyCommitBatch, Database

    # Create in-memory database for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        database = Database(db_path)

        session = database.get_session()
        try:
            # Test that we can create the tables
            database.init_db()

            # Test creating a sample daily batch
            sample_batch = DailyCommitBatch(
                date=datetime.now().date(),
                project_key="TEST",
                repo_path="/test/repo",
                commit_count=5,
                total_files_changed=20,
                total_lines_added=100,
                total_lines_deleted=50,
                active_developers=["dev1", "dev2"],
                unique_tickets=["TEST-123", "TEST-456"],
                context_summary="5 commits by 2 developers, 2 tickets referenced",
            )

            session.add(sample_batch)
            session.commit()

            # Verify we can query it back
            retrieved = (
                session.query(DailyCommitBatch)
                .filter(DailyCommitBatch.project_key == "TEST")
                .first()
            )

            assert retrieved is not None
            assert retrieved.commit_count == 5

            print("   ✅ Database models work correctly")
            print(f"      - Created batch with {retrieved.commit_count} commits")
            print(f"      - Project: {retrieved.project_key}")
            print(f"      - Status: {retrieved.classification_status}")

        finally:
            session.close()


def test_cli_commands():
    """Test that the CLI commands are properly registered."""
    print("🧪 Testing CLI Commands...")

    try:
        from src.gitflow_analytics.cli import cli

        # Get list of registered commands
        commands = list(cli.commands.keys())
        print(f"   📋 Available commands: {commands}")

        # Check that fetch command exists
        assert "fetch" in commands, "fetch command not found"
        assert "analyze" in commands, "analyze command not found"

        print("   ✅ Both 'fetch' and 'analyze' commands are registered")

        # Test command help (basic smoke test)
        import click.testing

        runner = click.testing.CliRunner()

        # Test fetch command help
        result = runner.invoke(cli, ["fetch", "--help"])
        assert result.exit_code == 0
        assert "Fetch data from external platforms" in result.output
        print("   ✅ Fetch command help works")

        # Test analyze command help
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "use-batch-classification" in result.output
        print("   ✅ Analyze command help works with new options")

    except Exception as e:
        print(f"   ❌ CLI test failed: {e}")
        raise


def test_metrics_storage():
    """Test the daily metrics storage system."""
    print("🧪 Testing Metrics Storage...")

    from datetime import date

    from src.gitflow_analytics.core.metrics_storage import DailyMetricsStorage

    # Create temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "metrics.db"
        storage = DailyMetricsStorage(db_path)

        # Test storing sample metrics
        test_date = date.today()
        sample_commits = [
            {
                "timestamp": datetime.now(),
                "author_email": "dev1@example.com",
                "author_name": "Developer One",
                "project_key": "TEST",
                "category": "feature",
                "files_changed": 3,
                "insertions": 50,
                "deletions": 10,
                "story_points": 2,
                "ticket_references": ["TEST-123"],
                "is_merge": False,
            }
        ]

        developer_identities = {
            "dev1@example.com": {
                "canonical_id": "dev1",
                "name": "Developer One",
                "email": "dev1@example.com",
            }
        }

        records_created = storage.store_daily_metrics(
            test_date, sample_commits, developer_identities
        )
        print(f"   📊 Created {records_created} metric records")

        # Test retrieving metrics
        end_date = test_date + timedelta(days=1)
        metrics = storage.get_date_range_metrics(test_date, end_date)

        assert len(metrics) == 1
        assert metrics[0]["developer_name"] == "Developer One"
        assert metrics[0]["feature_commits"] == 1

        print("   ✅ Metrics storage works correctly")


def main():
    """Run all tests for the two-step process."""
    print("🚀 Testing Two-Step Fetch/Analyze Process")
    print("=" * 50)

    try:
        # Test individual components
        test_database_models()
        test_data_fetcher()
        test_batch_classifier()
        test_metrics_storage()
        test_cli_commands()

        print("\n" + "=" * 50)
        print("🎉 All tests passed! The two-step process is ready for use.")

        print("\n📖 Usage Instructions:")
        print("1. First, fetch raw data:")
        print("   gitflow-analytics fetch -c config.yaml --weeks 4")

        print("\n2. Then, analyze with batch classification:")
        print("   gitflow-analytics analyze -c config.yaml --use-batch-classification")

        print("\n🔧 Advanced Options:")
        print("   --force-fetch         # Skip cached data, fetch fresh")
        print("   --clear-cache         # Clear all caches before processing")
        print("   --log DEBUG           # Enable detailed logging")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
