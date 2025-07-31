"""
Pytest configuration and shared fixtures for GitFlow Analytics tests.

This module provides common test fixtures and configuration that can be shared
across all test modules.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from gitflow_analytics.config import Config
from gitflow_analytics.models.database import Base


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_git_repo(temp_dir):
    """Create a mock git repository for testing."""
    repo_path = temp_dir / "test_repo"
    repo_path.mkdir()

    # Create a minimal git structure
    git_dir = repo_path / ".git"
    git_dir.mkdir()

    # Create some test files
    (repo_path / "README.md").write_text("# Test Repository")
    (repo_path / "src").mkdir()
    (repo_path / "src" / "main.py").write_text("print('Hello, World!')")

    return repo_path


@pytest.fixture
def test_database():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample configuration for testing."""
    config_data = {
        "repositories": [
            {
                "name": "test-repo",
                "path": str(temp_dir / "test-repo"),
                "url": "https://github.com/test/test-repo.git",
            }
        ],
        "cache_dir": str(temp_dir / ".gitflow-cache"),
        "reports_dir": str(temp_dir / "reports"),
        "developers": {"identity_threshold": 0.85, "manual_mappings": {}},
        "tickets": {"default_platform": "github"},
    }

    return Config(config_data)


@pytest.fixture
def mock_github_client():
    """Create a mock GitHub client for testing."""
    client = Mock()
    client.get_repository.return_value = Mock()
    client.get_commits.return_value = []
    client.get_pull_requests.return_value = []
    return client


@pytest.fixture
def sample_commits():
    """Provide sample commit data for testing."""
    return [
        {
            "hash": "abc123",
            "author_name": "John Doe",
            "author_email": "john@example.com",
            "committer_name": "John Doe",
            "committer_email": "john@example.com",
            "date": "2024-01-01T10:00:00Z",
            "message": "feat: add new feature [PROJ-123]",
            "files_changed": 3,
            "insertions": 25,
            "deletions": 5,
            "branch": "main",
        },
        {
            "hash": "def456",
            "author_name": "Jane Smith",
            "author_email": "jane@example.com",
            "committer_name": "Jane Smith",
            "committer_email": "jane@example.com",
            "date": "2024-01-02T14:30:00Z",
            "message": "fix: resolve bug in user authentication",
            "files_changed": 1,
            "insertions": 10,
            "deletions": 8,
            "branch": "feature/auth-fix",
        },
    ]


@pytest.fixture
def sample_developers():
    """Provide sample developer identity data for testing."""
    return {
        "john@example.com": {
            "primary_email": "john@example.com",
            "all_emails": ["john@example.com", "john.doe@company.com"],
            "name": "John Doe",
            "commit_count": 15,
        },
        "jane@example.com": {
            "primary_email": "jane@example.com",
            "all_emails": ["jane@example.com"],
            "name": "Jane Smith",
            "commit_count": 8,
        },
    }
