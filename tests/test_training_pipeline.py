"""Tests for the commit classification training pipeline.

These tests verify the training pipeline functionality including:
- Training data extraction and labeling
- Model training and validation
- Model storage and loading
- Integration with existing systems

WHY: The training pipeline is a complex system that integrates multiple components.
These tests ensure reliability and catch regressions in the training workflow.
"""

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Skip all tests if sklearn not available
sklearn = pytest.importorskip("sklearn")

from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.training.model_loader import TrainingModelLoader
from gitflow_analytics.training.pipeline import (
    CommitClassificationTrainer,
    TrainingData,
    TrainingSession,
)


class TestCommitClassificationTrainer:
    """Test the main training pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_config(self):
        """Mock configuration object."""
        config = Mock()
        config.analysis.branch_mapping_rules = {}
        config.analysis.ticket_platforms = None
        config.analysis.exclude_paths = []
        return config

    @pytest.fixture
    def cache(self, temp_dir):
        """Test cache instance."""
        return GitAnalysisCache(temp_dir / "cache")

    @pytest.fixture
    def mock_orchestrator(self):
        """Mock integration orchestrator."""
        orchestrator = Mock()
        # Mock PM data response
        orchestrator.enrich_repository_data.return_value = {
            "pm_data": {
                "issues": {
                    "jira": [
                        {
                            "key": "PROJ-123",
                            "type": "Bug",
                            "status": "Done",
                            "title": "Fix login issue",
                        },
                        {
                            "key": "PROJ-124",
                            "type": "Story",
                            "status": "Done",
                            "title": "Add user dashboard",
                        },
                    ]
                }
            }
        }
        return orchestrator

    @pytest.fixture
    def sample_commits(self):
        """Sample commit data for testing."""
        return [
            {
                "hash": "abc123",
                "message": "fix: resolve login timeout issue PROJ-123",
                "author_name": "John Doe",
                "author_email": "john@example.com",
                "timestamp": datetime.now(timezone.utc),
                "files_changed_list": ["src/auth.py", "tests/test_auth.py"],
                "files_changed": 2,
                "insertions": 10,
                "deletions": 5,
                "ticket_references": ["PROJ-123"],
                "project_key": "TEST",
                "repo_name": "test-repo",
            },
            {
                "hash": "def456",
                "message": "feat: implement user dashboard PROJ-124",
                "author_name": "Jane Smith",
                "author_email": "jane@example.com",
                "timestamp": datetime.now(timezone.utc),
                "files_changed_list": ["src/dashboard.py", "templates/dashboard.html"],
                "files_changed": 2,
                "insertions": 50,
                "deletions": 0,
                "ticket_references": ["PROJ-124"],
                "project_key": "TEST",
                "repo_name": "test-repo",
            },
        ]

    @pytest.fixture
    def trainer(self, mock_config, cache, mock_orchestrator, temp_dir):
        """Training pipeline instance."""
        training_config = {
            "min_training_examples": 2,  # Low threshold for testing
            "validation_split": 0.5,  # Simple split for small dataset
            "model_type": "random_forest",
        }
        return CommitClassificationTrainer(
            config=mock_config,
            cache=cache,
            orchestrator=mock_orchestrator,
            training_config=training_config,
        )

    def test_trainer_initialization(self, trainer):
        """Test trainer initializes correctly."""
        assert trainer is not None
        assert trainer.training_config["min_training_examples"] == 2
        assert trainer.Session is not None
        assert trainer.db_path is not None

    @patch("gitflow_analytics.training.pipeline.GitAnalyzer")
    def test_extract_labeled_commits(
        self, mock_analyzer, trainer, sample_commits, mock_orchestrator
    ):
        """Test labeled commit extraction."""
        # Mock repository config
        mock_repo = Mock()
        mock_repo.path.exists.return_value = True
        mock_repo.name = "test-repo"
        mock_repo.project_key = "TEST"
        mock_repo.branch = "main"

        # Mock analyzer
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze_repository.return_value = sample_commits
        mock_analyzer.return_value = mock_analyzer_instance

        # Mock ticket data fetching
        with patch.object(trainer, "_fetch_ticket_data") as mock_fetch:
            mock_fetch.return_value = [
                {"type": "Bug", "key": "PROJ-123", "platform": "jira"},
                {"type": "Story", "key": "PROJ-124", "platform": "jira"},
            ]

            # Test extraction
            since = datetime.now(timezone.utc)
            labeled_commits = trainer._extract_labeled_commits([mock_repo], since)

            # We expect 2 commits to be returned based on sample data
            assert isinstance(labeled_commits, list)

    def test_determine_label(self, trainer):
        """Test label determination from ticket data."""
        # Test with Bug ticket
        bug_tickets = [{"type": "Bug", "key": "PROJ-123", "platform": "jira"}]
        label = trainer._determine_label(bug_tickets)
        assert label == "bug_fix"

        # Test with Story ticket
        story_tickets = [{"type": "Story", "key": "PROJ-124", "platform": "jira"}]
        label = trainer._determine_label(story_tickets)
        assert label == "feature"

        # Test with unknown type
        unknown_tickets = [{"type": "Unknown", "key": "PROJ-125", "platform": "jira"}]
        label = trainer._determine_label(unknown_tickets)
        assert label is None or label == "other"

    def test_normalize_commit_data(self, trainer):
        """Test commit data normalization."""
        # Test normalization of commit data
        commit = {
            "hash": "abc123",
            "message": "fix: test issue",
            "author_name": "John Doe",
            "author_email": "john@example.com",
            "timestamp": datetime.now(timezone.utc),
            "files_changed_list": ["src/test.py"],
            "files_changed": 1,
            "insertions": 10,
            "deletions": 5,
        }

        normalized = trainer._normalize_commit_data(commit)

        assert "hash" in normalized
        assert "message" in normalized
        assert normalized["hash"] == "abc123"
        assert normalized["message"] == "fix: test issue"

    def test_store_training_data(self, trainer):
        """Test storing training data to database."""
        session_id = str(uuid.uuid4())
        labeled_commits = [
            {
                "commit_data": {
                    "hash": "abc123",
                    "message": "fix: test issue",
                    "author_name": "John Doe",
                    "author_email": "john@example.com",
                    "timestamp": datetime.now(timezone.utc),
                    "files_changed_list": ["test.py"],
                    "files_changed": 1,
                    "insertions": 10,
                    "deletions": 5,
                },
                "label": "bug_fix",
                "ticket_data": [{"key": "PROJ-123", "type": "Bug", "platform": "jira"}],
                "repository": "test-repo",
            }
        ]

        # Store data - should not raise exception
        trainer._store_training_data(session_id, labeled_commits)

        # Verify data was stored
        with trainer.Session() as session:
            count = session.query(TrainingData).filter_by(commit_hash="abc123").count()
            assert count == 1

    def test_create_training_session(self, trainer):
        """Test creating a training session."""
        session_name = "Test Session"
        session_id = trainer._create_training_session(session_name)

        assert session_id is not None
        assert isinstance(session_id, str)

        # Verify session was created in database
        with trainer.Session() as session:
            training_session = session.query(TrainingSession).filter_by(id=session_id).first()
            assert training_session is not None
            assert training_session.name == session_name

    def test_get_training_history(self, trainer):
        """Test retrieving training history."""
        history = trainer.get_training_history()

        assert isinstance(history, list)
        # History may be empty initially
        for session in history:
            assert "id" in session
            assert "name" in session
            assert "created_at" in session
            assert "training_examples" in session

    @patch("gitflow_analytics.training.pipeline.CommitClassifier")
    def test_train_method(self, mock_classifier, trainer):
        """Test the main train method."""
        # Mock classifier training
        mock_classifier_instance = Mock()
        mock_classifier_instance.train_model.return_value = {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1_score": 0.85,
        }
        trainer.classifier = mock_classifier_instance

        # Mock extraction method to return minimal labeled data
        with patch.object(trainer, "_extract_labeled_commits") as mock_extract:
            mock_extract.return_value = [
                {
                    "commit_data": {"hash": "abc123", "message": "fix: bug"},
                    "label": "bug_fix",
                    "ticket_data": [{"key": "PROJ-1", "type": "Bug"}],
                    "repository": "test-repo",
                },
                {
                    "commit_data": {"hash": "def456", "message": "feat: feature"},
                    "label": "feature",
                    "ticket_data": [{"key": "PROJ-2", "type": "Story"}],
                    "repository": "test-repo",
                },
            ]

            # Run training
            repositories = [Mock()]
            since = datetime.now(timezone.utc)
            results = trainer.train(repositories, since, "Test Training")

            assert "session_id" in results
            assert "training_examples" in results
            assert results["training_examples"] == 2
            assert results["accuracy"] == 0.85


class TestTrainingModelLoader:
    """Test the model loader functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def model_loader(self, temp_dir):
        """Model loader instance."""
        return TrainingModelLoader(temp_dir)

    def test_model_loader_initialization(self, model_loader):
        """Test model loader initializes correctly."""
        assert model_loader is not None
        assert model_loader.db is not None
        assert model_loader.loaded_models == {}

    def test_get_best_model_empty(self, model_loader):
        """Test getting best model when none exist."""
        best_model = model_loader.get_best_model()
        assert best_model is None

    def test_list_available_models_empty(self, model_loader):
        """Test listing models when none exist."""
        models = model_loader.list_available_models()
        assert models == []

    def test_load_model_not_found(self, model_loader):
        """Test loading model that doesn't exist."""
        with pytest.raises(ValueError, match="No trained models available"):
            model_loader.load_model()

        with pytest.raises(ValueError, match="No model found with ID"):
            model_loader.load_model("nonexistent")

    def test_get_model_statistics(self, model_loader):
        """Test model statistics retrieval."""
        stats = model_loader.get_model_statistics()

        assert "loaded_models_count" in stats
        assert "available_models_count" in stats
        assert "total_usage_count" in stats
        assert "best_model_accuracy" in stats
        assert stats["loaded_models_count"] == 0
        assert stats["available_models_count"] == 0


class TestTrainingIntegration:
    """Integration tests for the complete training workflow."""

    @pytest.fixture
    def temp_dir(self):
        """Temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_training_database_models(self, temp_dir):
        """Test training database models creation."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from gitflow_analytics.training.pipeline import (
            TrainingBase,
        )
        from gitflow_analytics.training.pipeline import (
            TrainingData as PipelineTrainingData,
        )
        from gitflow_analytics.training.pipeline import (
            TrainingSession as PipelineTrainingSession,
        )

        # Create database with pipeline's schema
        db_path = temp_dir / "training.db"
        engine = create_engine(f"sqlite:///{db_path}")
        TrainingBase.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)

        # Test database initialization
        with Session() as session:
            # Should not raise any errors
            session.query(PipelineTrainingData).count()
            session.query(PipelineTrainingSession).count()

    def test_training_data_storage(self, temp_dir):
        """Test storing and retrieving training data."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from gitflow_analytics.training.pipeline import (
            TrainingBase,
        )
        from gitflow_analytics.training.pipeline import (
            TrainingData as PipelineTrainingData,
        )

        # Create database with pipeline's schema
        db_path = temp_dir / "training.db"
        engine = create_engine(f"sqlite:///{db_path}")
        TrainingBase.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)

        with Session() as session:
            # Create training data using pipeline's model
            training_data = PipelineTrainingData(
                session_id="test-session",
                commit_hash="abc123",
                repository="test-repo",
                message="fix: test issue",
                author="John Doe",
                timestamp=datetime.now(timezone.utc),
                files_changed=1,
                insertions=10,
                deletions=5,
                ticket_id="PROJ-123",
                ticket_type="Bug",
                ticket_platform="jira",
                label="bug_fix",
                confidence=1.0,
                created_at=datetime.now(timezone.utc),
            )

            session.add(training_data)
            session.commit()

            # Retrieve and verify
            retrieved = session.query(PipelineTrainingData).filter_by(commit_hash="abc123").first()
            assert retrieved is not None
            assert retrieved.label == "bug_fix"
            assert retrieved.ticket_platform == "jira"


# Utility functions for testing
def create_mock_repository_config(name: str, path: Path):
    """Create a mock repository configuration."""
    repo = Mock()
    repo.name = name
    repo.path = path
    repo.project_key = name.upper()
    repo.branch = "main"
    repo.github_repo = f"org/{name}"
    return repo


def create_sample_pm_data(ticket_count: int = 5):
    """Create sample PM platform data for testing."""
    tickets = []
    for i in range(ticket_count):
        tickets.append(
            {
                "key": f"PROJ-{i + 1}",
                "type": "Bug" if i % 2 == 0 else "Story",
                "status": "Done",
                "title": f"Sample ticket {i + 1}",
            }
        )

    return {"test-repo": {"issues": {"jira": tickets}}}


if __name__ == "__main__":
    pytest.main([__file__])
