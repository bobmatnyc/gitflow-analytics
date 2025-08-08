"""Tests for the commit classification training pipeline.

These tests verify the training pipeline functionality including:
- Training data extraction and labeling
- Model training and validation  
- Model storage and loading
- Integration with existing systems

WHY: The training pipeline is a complex system that integrates multiple components.
These tests ensure reliability and catch regressions in the training workflow.
"""

import pytest
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Skip all tests if sklearn not available
sklearn = pytest.importorskip("sklearn")

from gitflow_analytics.training.pipeline import CommitClassificationTrainer
from gitflow_analytics.training.model_loader import TrainingModelLoader
from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.models.database import Database, TrainingData, TrainingSession


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
                            "title": "Fix login issue"
                        },
                        {
                            "key": "PROJ-124", 
                            "type": "Story",
                            "status": "Done",
                            "title": "Add user dashboard"
                        }
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
                "repo_name": "test-repo"
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
                "repo_name": "test-repo"
            }
        ]
    
    @pytest.fixture
    def trainer(self, mock_config, cache, mock_orchestrator, temp_dir):
        """Training pipeline instance."""
        training_config = {
            'min_training_examples': 2,  # Low threshold for testing
            'validation_split': 0.5,    # Simple split for small dataset
            'model_type': 'random_forest'
        }
        return CommitClassificationTrainer(
            config=mock_config,
            cache=cache,
            orchestrator=mock_orchestrator,
            training_config=training_config
        )
    
    def test_trainer_initialization(self, trainer):
        """Test trainer initializes correctly."""
        assert trainer is not None
        assert trainer.training_config['min_training_examples'] == 2
        assert trainer.db is not None
    
    @patch('gitflow_analytics.training.pipeline.GitAnalyzer')
    def test_extract_commits_and_tickets(self, mock_analyzer, trainer, sample_commits, mock_orchestrator):
        """Test commit and ticket extraction."""
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
        
        # Test extraction
        commits, pm_data = trainer._extract_commits_and_tickets([mock_repo], 4)
        
        assert len(commits) == 2
        assert "test-repo" in pm_data
        assert commits[0]["project_key"] == "TEST"
        assert commits[0]["repo_name"] == "test-repo"
    
    def test_create_training_data(self, trainer, sample_commits):
        """Test training data creation from commits and PM data."""
        pm_data = {
            "test-repo": {
                "issues": {
                    "jira": [
                        {"key": "PROJ-123", "type": "Bug", "status": "Done"},
                        {"key": "PROJ-124", "type": "Story", "status": "Done"}
                    ]
                }
            }
        }
        
        session_id = str(uuid.uuid4())
        training_data = trainer._create_training_data(sample_commits, pm_data, session_id)
        
        assert len(training_data) == 2
        assert training_data[0]["category"] == "bug_fix"  # Bug -> bug_fix
        assert training_data[1]["category"] == "feature"  # Story -> feature
        assert training_data[0]["source_platform"] == "jira"
        assert training_data[0]["confidence"] == 1.0
    
    def test_resolve_multi_ticket_category(self, trainer):
        """Test intelligent handling of commits with multiple tickets."""
        # Test same category
        tickets_same = [
            {"type": "Bug", "platform": "jira", "issue": {"key": "PROJ-1"}},
            {"type": "Bug", "platform": "jira", "issue": {"key": "PROJ-2"}}
        ]
        
        category, confidence, source_info = trainer._resolve_multi_ticket_category(tickets_same)
        assert category == "bug_fix"
        assert confidence == 1.0
        
        # Test mixed categories (bug_fix should have priority)
        tickets_mixed = [
            {"type": "Story", "platform": "jira", "issue": {"key": "PROJ-1"}},
            {"type": "Bug", "platform": "jira", "issue": {"key": "PROJ-2"}}
        ]
        
        category, confidence, source_info = trainer._resolve_multi_ticket_category(tickets_mixed)
        assert category == "bug_fix"  # Higher priority
        assert confidence == 0.7  # Lower confidence for mixed
    
    def test_extract_commit_features(self, trainer, sample_commits):
        """Test feature extraction from commits."""
        commit = sample_commits[0]
        features = trainer._extract_commit_features(commit)
        
        assert "message_length" in features
        assert "word_count" in features
        assert "files_count" in features
        assert "lines_changed" in features
        assert features["files_count"] == 2
        assert features["lines_changed"] == 15  # 10 + 5
        assert features["has_test_files"] == True  # has test_auth.py
    
    def test_prepare_features_and_labels(self, trainer):
        """Test feature preparation for model training."""
        training_data = [
            {
                "commit_message": "fix: resolve login issue",
                "category": "bug_fix"
            },
            {
                "commit_message": "feat: add new dashboard",
                "category": "feature"
            }
        ]
        
        X, y, features_info = trainer._prepare_features_and_labels(training_data)
        
        assert X.shape[0] == 2  # Two examples
        assert len(y) == 2
        assert y == ["bug_fix", "feature"]
        assert "vectorizer" in features_info
        assert "categories" in features_info
        assert features_info["categories"] == ["bug_fix", "feature"]
    
    def test_split_training_data(self, trainer):
        """Test data splitting for training/validation/test."""
        # Create mock data
        import numpy as np
        from scipy.sparse import csr_matrix
        
        X = csr_matrix(np.random.rand(10, 5))  # 10 samples, 5 features
        y = ["bug_fix"] * 5 + ["feature"] * 5  # Balanced classes
        
        splits = trainer._split_training_data(X, y)
        
        assert "X_train" in splits
        assert "X_val" in splits
        assert "X_test" in splits
        assert "y_train" in splits
        assert "y_val" in splits
        assert "y_test" in splits
        
        # Check split sizes (approximately correct)
        total_size = len(y)
        train_size = splits["X_train"].shape[0]
        val_size = splits["X_val"].shape[0]
        test_size = splits["X_test"].shape[0]
        
        assert train_size + val_size + test_size == total_size
    
    def test_training_statistics(self, trainer):
        """Test training statistics retrieval."""
        stats = trainer.get_training_statistics()
        
        assert "total_sessions" in stats
        assert "completed_sessions" in stats
        assert "failed_sessions" in stats
        assert "total_models" in stats
        assert "total_training_examples" in stats
        assert stats["total_sessions"] == 0  # No sessions yet


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
        db = Database(temp_dir / "training.db")
        
        # Test database initialization
        with db.get_session() as session:
            # Should not raise any errors
            session.query(TrainingData).count()
            session.query(TrainingSession).count()
    
    def test_training_data_storage(self, temp_dir):
        """Test storing and retrieving training data."""
        db = Database(temp_dir / "training.db")
        
        with db.get_session() as session:
            # Create training data
            training_data = TrainingData(
                commit_hash="abc123",
                commit_message="fix: test issue",
                files_changed=["test.py"],
                repo_path="test-repo",
                category="bug_fix",
                confidence=1.0,
                source_type="pm_platform",
                source_platform="jira",
                source_ticket_id="PROJ-123",
                source_ticket_type="Bug",
                training_session_id="test-session"
            )
            
            session.add(training_data)
            session.commit()
            
            # Retrieve and verify
            retrieved = session.query(TrainingData).filter_by(commit_hash="abc123").first()
            assert retrieved is not None
            assert retrieved.category == "bug_fix"
            assert retrieved.source_platform == "jira"


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
        tickets.append({
            "key": f"PROJ-{i+1}",
            "type": "Bug" if i % 2 == 0 else "Story",
            "status": "Done",
            "title": f"Sample ticket {i+1}"
        })
    
    return {
        "test-repo": {
            "issues": {
                "jira": tickets
            }
        }
    }


if __name__ == "__main__":
    pytest.main([__file__])