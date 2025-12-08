"""Basic integration tests for qualitative analysis system."""

from datetime import datetime
from unittest.mock import patch

import pytest

from gitflow_analytics.models.database import Database
from gitflow_analytics.qualitative.core.processor import QualitativeProcessor
from gitflow_analytics.qualitative.models.schemas import (
    CacheConfig as QualitativeCacheConfig,
)
from gitflow_analytics.qualitative.models.schemas import (
    LLMConfig,
    NLPConfig,
    QualitativeConfig,
)


class TestQualitativeIntegration:
    """Integration tests for qualitative analysis system."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample qualitative configuration."""
        return QualitativeConfig(
            enabled=True,
            batch_size=10,
            confidence_threshold=0.7,
            nlp_config=NLPConfig(),
            llm_config=LLMConfig(openrouter_api_key="test-key"),
            cache_config=QualitativeCacheConfig(),
        )

    @pytest.fixture
    def sample_commits(self):
        """Create sample commit data for testing."""
        return [
            {
                "hash": "abc123",
                "message": "Add user authentication feature",
                "author_name": "John Doe",
                "author_email": "john@example.com",
                "timestamp": datetime.now(),
                "files_changed": ["src/auth.py", "src/user.py"],
                "insertions": 150,
                "deletions": 20,
            },
            {
                "hash": "def456",
                "message": "Fix bug in payment processing",
                "author_name": "Jane Smith",
                "author_email": "jane@example.com",
                "timestamp": datetime.now(),
                "files_changed": ["src/payment.py"],
                "insertions": 25,
                "deletions": 10,
            },
            {
                "hash": "ghi789",
                "message": "Update documentation for API endpoints",
                "author_name": "Bob Wilson",
                "author_email": "bob@example.com",
                "timestamp": datetime.now(),
                "files_changed": ["docs/api.md", "README.md"],
                "insertions": 80,
                "deletions": 5,
            },
        ]

    def test_config_creation(self, sample_config):
        """Test that configuration can be created properly."""
        assert sample_config.enabled is True
        assert sample_config.batch_size == 10
        assert sample_config.confidence_threshold == 0.7
        assert sample_config.nlp_config is not None
        assert sample_config.llm_config is not None
        assert sample_config.cache_config is not None

    def test_config_validation(self, sample_config):
        """Test configuration validation."""
        warnings = sample_config.validate()
        # Should not have warnings for reasonable config
        assert isinstance(warnings, list)

    @patch("gitflow_analytics.qualitative.core.processor.NLPEngine")
    @patch("gitflow_analytics.qualitative.core.processor.PatternCache")
    def test_processor_initialization(self, mock_cache, mock_nlp, sample_config, tmp_path):
        """Test that qualitative processor can be initialized."""
        # Create temporary database
        db = Database(tmp_path / "test.db")

        # Initialize processor
        processor = QualitativeProcessor(sample_config, db)

        assert processor.config == sample_config
        assert processor.database == db
        assert mock_nlp.called
        assert mock_cache.called

    @patch("gitflow_analytics.qualitative.core.processor.NLPEngine")
    @patch("gitflow_analytics.qualitative.core.processor.PatternCache")
    def test_disabled_analysis(self, mock_cache, mock_nlp, sample_config, sample_commits, tmp_path):
        """Test that disabled analysis returns appropriate results."""
        # Disable qualitative analysis
        sample_config.enabled = False

        db = Database(tmp_path / "test.db")
        processor = QualitativeProcessor(sample_config, db)

        results = processor.process_commits(sample_commits)

        assert len(results) == len(sample_commits)
        for result in results:
            assert result.change_type == "disabled"
            assert result.business_domain == "disabled"
            assert result.processing_method == "disabled"

    def test_empty_commits(self, sample_config, tmp_path):
        """Test processing empty commit list."""
        db = Database(tmp_path / "test.db")

        with (
            patch("gitflow_analytics.qualitative.core.processor.NLPEngine"),
            patch("gitflow_analytics.qualitative.core.processor.PatternCache"),
        ):
            processor = QualitativeProcessor(sample_config, db)
            results = processor.process_commits([])

        assert results == []

    @pytest.mark.skipif(
        not pytest.importorskip("spacy", minversion=None), reason="spaCy not available"
    )
    def test_nlp_processing_availability(self):
        """Test that NLP processing dependencies are available."""
        try:
            from gitflow_analytics.qualitative.core.nlp_engine import NLPEngine
            from gitflow_analytics.qualitative.models.schemas import NLPConfig

            # This should not raise ImportError
            config = NLPConfig()
            assert config.spacy_model == "en_core_web_sm"
        except ImportError:
            pytest.skip("NLP dependencies not available")

    def test_import_qualitative_components(self):
        """Test that qualitative components can be imported."""
        # Test main package imports
        from gitflow_analytics.qualitative import (
            QualitativeCommitData,
            QualitativeConfig,
            QualitativeProcessor,
        )

        assert QualitativeProcessor is not None
        assert QualitativeCommitData is not None
        assert QualitativeConfig is not None

    def test_database_schema_creation(self, tmp_path):
        """Test that qualitative database tables are created."""
        db = Database(tmp_path / "test.db")

        # Tables should be created during initialization
        with db.get_session() as session:
            # Test that we can execute a basic query (tables exist)
            from sqlalchemy import text

            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            table_names = [row[0] for row in result.fetchall()]

            # Check that qualitative tables are created
            expected_tables = ["qualitative_commits", "pattern_cache", "llm_usage_stats"]
            for table in expected_tables:
                assert table in table_names, f"Table {table} not found in database"

    def test_text_processor_utilities(self):
        """Test text processing utilities."""
        from gitflow_analytics.qualitative.utils.text_processing import TextProcessor

        processor = TextProcessor()

        # Test message normalization
        normalized = processor.normalize_message("Fix BUG-123: Handle null values")
        assert "[TICKET]" in normalized
        assert "handle null values" in normalized.lower()

        # Test keyword extraction
        keywords = processor.extract_keywords("implement user authentication system")
        assert "implement" in keywords
        assert "user" in keywords
        assert "authentication" in keywords

        # Test fingerprint creation
        fingerprint = processor.create_semantic_fingerprint(
            "Add login feature", ["src/auth.py", "src/login.js"]
        )
        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 32  # MD5 hash length

    def test_cost_tracker_initialization(self):
        """Test cost tracking functionality."""
        from gitflow_analytics.qualitative.utils.cost_tracker import CostTracker

        tracker = CostTracker(daily_budget=10.0)

        # Test recording a call
        cost = tracker.record_call(
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            processing_time=1.5,
            batch_size=5,
        )

        assert isinstance(cost, float)
        assert cost >= 0

        # Test budget checking
        remaining = tracker.check_budget_remaining()
        assert isinstance(remaining, float)

        # Test usage stats
        stats = tracker.get_usage_stats(days=1)
        assert isinstance(stats, dict)
        assert "total_calls" in stats
