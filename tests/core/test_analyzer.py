"""
Tests for the Git analyzer module.

These tests verify git repository analysis functionality including commit parsing,
branch detection, and file change tracking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from pathlib import Path

from gitflow_analytics.core.analyzer import GitAnalyzer, CommitData


class TestGitAnalyzer:
    """Test cases for the GitAnalyzer class."""

    def test_init(self, temp_dir):
        """Test GitAnalyzer initialization."""
        repo_path = temp_dir / "test_repo"
        repo_path.mkdir()

        analyzer = GitAnalyzer(str(repo_path))

        assert analyzer.repo_path == str(repo_path)
        assert analyzer.repo is None  # Not initialized until needed

    @patch("gitflow_analytics.core.analyzer.git.Repo")
    def test_get_commits_basic(self, mock_repo_class, temp_dir):
        """Test basic commit retrieval functionality."""
        # Setup mock repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo

        # Setup mock commit
        mock_commit = Mock()
        mock_commit.hexsha = "abc123"
        mock_commit.author.name = "John Doe"
        mock_commit.author.email = "john@example.com"
        mock_commit.committer.name = "John Doe"
        mock_commit.committer.email = "john@example.com"
        mock_commit.committed_datetime = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        mock_commit.message = "feat: add new feature"
        mock_commit.stats.files = {"file1.py": {"insertions": 10, "deletions": 2}}
        mock_commit.stats.total = {"insertions": 10, "deletions": 2, "files": 1}

        # Setup mock branch reference
        mock_refs = Mock()
        mock_refs.name = "origin/main"
        mock_commit.refs = [mock_refs]

        mock_repo.iter_commits.return_value = [mock_commit]

        repo_path = temp_dir / "test_repo"
        repo_path.mkdir()
        analyzer = GitAnalyzer(str(repo_path))

        commits = list(analyzer.get_commits(weeks=4))

        assert len(commits) == 1
        commit = commits[0]
        assert isinstance(commit, CommitData)
        assert commit.hash == "abc123"
        assert commit.author_name == "John Doe"
        assert commit.author_email == "john@example.com"
        assert commit.message == "feat: add new feature"
        assert commit.files_changed == 1
        assert commit.insertions == 10
        assert commit.deletions == 2

    @patch("gitflow_analytics.core.analyzer.git.Repo")
    def test_get_commits_with_date_filter(self, mock_repo_class, temp_dir):
        """Test commit retrieval with date filtering."""
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo

        # Create commits from different time periods
        old_commit = Mock()
        old_commit.hexsha = "old123"
        old_commit.committed_datetime = datetime(2023, 1, 1, tzinfo=timezone.utc)

        recent_commit = Mock()
        recent_commit.hexsha = "recent123"
        recent_commit.committed_datetime = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # Setup common attributes
        for commit in [old_commit, recent_commit]:
            commit.author.name = "Test Author"
            commit.author.email = "test@example.com"
            commit.committer.name = "Test Author"
            commit.committer.email = "test@example.com"
            commit.message = "test commit"
            commit.stats.files = {}
            commit.stats.total = {"insertions": 0, "deletions": 0, "files": 0}
            commit.refs = []

        mock_repo.iter_commits.return_value = [recent_commit, old_commit]

        repo_path = temp_dir / "test_repo"
        repo_path.mkdir()
        analyzer = GitAnalyzer(str(repo_path))

        # Test with recent date filter
        with patch("gitflow_analytics.core.analyzer.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, tzinfo=timezone.utc)
            mock_datetime.timedelta.return_value.total_seconds.return_value = 604800 * 2  # 2 weeks

            commits = list(analyzer.get_commits(weeks=2))

            # Should only return recent commit based on date filtering logic
            assert len(commits) >= 1  # At least the recent commit

    @patch("gitflow_analytics.core.analyzer.git.Repo")
    def test_get_branches(self, mock_repo_class, temp_dir):
        """Test branch retrieval functionality."""
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo

        # Setup mock branches
        mock_branch1 = Mock()
        mock_branch1.name = "origin/main"
        mock_branch2 = Mock()
        mock_branch2.name = "origin/feature/new-feature"

        mock_repo.remote_refs = [mock_branch1, mock_branch2]

        repo_path = temp_dir / "test_repo"
        repo_path.mkdir()
        analyzer = GitAnalyzer(str(repo_path))

        branches = analyzer.get_branches()

        assert "main" in branches
        assert "feature/new-feature" in branches

    def test_extract_branch_from_commit(self, temp_dir):
        """Test branch extraction from commit information."""
        repo_path = temp_dir / "test_repo"
        repo_path.mkdir()
        analyzer = GitAnalyzer(str(repo_path))

        # Test with mock commit that has branch refs
        mock_commit = Mock()
        mock_ref = Mock()
        mock_ref.name = "origin/feature/awesome-feature"
        mock_commit.refs = [mock_ref]

        branch = analyzer._extract_branch_from_commit(mock_commit)
        assert branch == "feature/awesome-feature"

        # Test with commit that has no refs
        mock_commit.refs = []
        branch = analyzer._extract_branch_from_commit(mock_commit)
        assert branch == "unknown"

    def test_calculate_commit_metrics(self, temp_dir):
        """Test commit metrics calculation."""
        repo_path = temp_dir / "test_repo"
        repo_path.mkdir()
        analyzer = GitAnalyzer(str(repo_path))

        # Create mock commit stats
        mock_stats = Mock()
        mock_stats.files = {
            "src/file1.py": {"insertions": 10, "deletions": 2},
            "src/file2.py": {"insertions": 5, "deletions": 1},
            "README.md": {"insertions": 3, "deletions": 0},
        }
        mock_stats.total = {"insertions": 18, "deletions": 3, "files": 3}

        files_changed, insertions, deletions = analyzer._calculate_commit_metrics(mock_stats)

        assert files_changed == 3
        assert insertions == 18
        assert deletions == 3

    @patch("gitflow_analytics.core.analyzer.git.Repo")
    def test_repository_error_handling(self, mock_repo_class, temp_dir):
        """Test error handling for repository access."""
        mock_repo_class.side_effect = Exception("Repository not found")

        repo_path = temp_dir / "nonexistent_repo"
        analyzer = GitAnalyzer(str(repo_path))

        with pytest.raises(Exception):
            list(analyzer.get_commits())


class TestCommitData:
    """Test cases for the CommitData data class."""

    def test_commit_data_creation(self):
        """Test CommitData object creation and attributes."""
        commit_date = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        commit = CommitData(
            hash="abc123",
            author_name="John Doe",
            author_email="john@example.com",
            committer_name="John Doe",
            committer_email="john@example.com",
            date=commit_date,
            message="feat: add new feature",
            files_changed=3,
            insertions=25,
            deletions=5,
            branch="main",
        )

        assert commit.hash == "abc123"
        assert commit.author_name == "John Doe"
        assert commit.author_email == "john@example.com"
        assert commit.date == commit_date
        assert commit.message == "feat: add new feature"
        assert commit.files_changed == 3
        assert commit.insertions == 25
        assert commit.deletions == 5
        assert commit.branch == "main"

    def test_commit_data_equality(self):
        """Test CommitData equality comparison."""
        commit_date = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        commit1 = CommitData(
            hash="abc123",
            author_name="John Doe",
            author_email="john@example.com",
            committer_name="John Doe",
            committer_email="john@example.com",
            date=commit_date,
            message="feat: add new feature",
            files_changed=3,
            insertions=25,
            deletions=5,
            branch="main",
        )

        commit2 = CommitData(
            hash="abc123",
            author_name="John Doe",
            author_email="john@example.com",
            committer_name="John Doe",
            committer_email="john@example.com",
            date=commit_date,
            message="feat: add new feature",
            files_changed=3,
            insertions=25,
            deletions=5,
            branch="main",
        )

        assert commit1 == commit2

        # Test inequality
        commit3 = CommitData(
            hash="def456",  # Different hash
            author_name="John Doe",
            author_email="john@example.com",
            committer_name="John Doe",
            committer_email="john@example.com",
            date=commit_date,
            message="feat: add new feature",
            files_changed=3,
            insertions=25,
            deletions=5,
            branch="main",
        )

        assert commit1 != commit3
