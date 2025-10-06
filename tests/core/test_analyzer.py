"""
Tests for the Git analyzer module.

These tests verify git repository analysis functionality including commit parsing,
branch detection, and file change tracking.
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from gitflow_analytics.core.analyzer import GitAnalyzer
from gitflow_analytics.core.cache import GitAnalysisCache


class TestGitAnalyzer:
    """Test cases for the GitAnalyzer class."""

    def test_init(self, temp_dir):
        """Test GitAnalyzer initialization."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        analyzer = GitAnalyzer(cache)

        assert analyzer.cache == cache
        assert analyzer.batch_size == 1000
        assert analyzer.exclude_paths == []

    @patch("gitflow_analytics.core.analyzer.Repo")
    def test_analyze_repository_basic(self, mock_repo_class, temp_dir):
        """Test basic repository analysis functionality."""
        # Setup cache
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

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
        mock_commit.parents = []
        mock_commit.diff.return_value = []

        # Setup mock branch reference
        mock_refs = Mock()
        mock_refs.name = "origin/main"
        mock_commit.refs = [mock_refs]

        mock_repo.iter_commits.return_value = iter([mock_commit])
        mock_repo.remote_refs = [mock_refs]
        mock_repo.refs = [mock_refs]
        mock_repo.branches = []  # Add branches attribute for the analyzer

        repo_path = temp_dir / "test_repo"
        repo_path.mkdir()
        analyzer = GitAnalyzer(cache)

        since = datetime(2023, 12, 1, tzinfo=timezone.utc)
        commits = analyzer.analyze_repository(repo_path, since)

        assert len(commits) == 1
        commit = commits[0]
        assert isinstance(commit, dict)
        assert commit["hash"] == "abc123"
        assert commit["author_name"] == "John Doe"
        assert commit["author_email"] == "john@example.com"
        assert commit["message"] == "feat: add new feature"

    @patch("gitflow_analytics.core.analyzer.Repo")
    def test_analyze_repository_with_date_filter(self, mock_repo_class, temp_dir):
        """Test repository analysis with date filtering."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

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
            commit.parents = []
            commit.diff.return_value = []

        # Setup a mock branch for the repository
        mock_branch = Mock()
        mock_branch.name = "main"

        # Setup mock remote for update
        mock_remote = Mock()
        mock_remote.fetch.return_value = None
        mock_repo.remotes = [mock_remote]

        # Setup mock head
        mock_repo.head.is_detached = False
        mock_repo.active_branch = mock_branch
        mock_repo.active_branch.tracking_branch.return_value = mock_branch

        # Add the main branch to refs so it can be found
        mock_ref = Mock()
        mock_ref.name = "refs/heads/main"

        mock_repo.iter_commits.return_value = iter(
            [recent_commit]
        )  # Only recent commits returned by git
        mock_repo.remote_refs = []
        mock_repo.refs = [mock_ref]  # Include the main branch ref
        mock_repo.branches = [mock_branch]  # Add a main branch for the analyzer

        repo_path = temp_dir / "test_repo"
        repo_path.mkdir()
        analyzer = GitAnalyzer(cache)

        # Test with recent date filter
        since = datetime(2023, 12, 1, tzinfo=timezone.utc)
        commits = analyzer.analyze_repository(repo_path, since)

        # Should only return recent commit based on date filtering logic
        assert len(commits) == 1
        assert commits[0]["hash"] == "recent123"

    def test_extract_branch_mapping(self, temp_dir):
        """Test branch to project mapping functionality."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        # Custom branch mapping rules
        branch_rules = {"FRONTEND": ["frontend/*", "fe/*"], "BACKEND": ["backend/*", "api/*"]}

        analyzer = GitAnalyzer(cache, branch_mapping_rules=branch_rules)

        # Test branch mapping
        frontend_project = analyzer.branch_mapper.map_branch_to_project("frontend/new-feature")
        backend_project = analyzer.branch_mapper.map_branch_to_project("api/user-service")
        unknown_project = analyzer.branch_mapper.map_branch_to_project("random/branch")

        assert frontend_project == "FRONTEND"
        assert backend_project == "BACKEND"
        assert unknown_project == "UNKNOWN"

    def test_ticket_extraction(self, temp_dir):
        """Test ticket reference extraction from commit messages."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        analyzer = GitAnalyzer(cache, allowed_ticket_platforms=["jira", "github"])

        # Test ticket extraction
        commit_message = "feat: add user authentication [PROJ-123] fixes #456"
        tickets = analyzer.ticket_extractor.extract_from_text(commit_message)

        assert len(tickets) == 2
        ticket_ids = [t["id"] for t in tickets]
        assert "PROJ-123" in ticket_ids
        assert "456" in ticket_ids

    def test_story_point_extraction(self, temp_dir):
        """Test story point extraction from commit messages."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        analyzer = GitAnalyzer(cache)

        # Test story point extraction
        commit_message = "feat: implement user dashboard [SP: 5]"
        points = analyzer.story_point_extractor.extract_from_text(commit_message)

        assert points == 5

        # Test no story points
        no_points_message = "fix: typo in documentation"
        points = analyzer.story_point_extractor.extract_from_text(no_points_message)

        assert points is None

    @patch("gitflow_analytics.core.analyzer.Repo")
    def test_repository_error_handling(self, mock_repo_class, temp_dir):
        """Test error handling for repository access."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        mock_repo_class.side_effect = Exception("Repository not found")

        repo_path = temp_dir / "nonexistent_repo"
        analyzer = GitAnalyzer(cache)

        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            analyzer.analyze_repository(repo_path, since)


class TestExcludePatterns:
    """Test cases for file exclusion patterns."""

    def test_exclude_paths_filtering(self, temp_dir):
        """Test that specified paths are excluded from analysis."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        exclude_patterns = ["*.md", "*.txt", "node_modules/*"]
        analyzer = GitAnalyzer(cache, exclude_paths=exclude_patterns)

        # Test file exclusion logic
        assert analyzer._should_exclude_file("README.md") is True
        assert analyzer._should_exclude_file("notes.txt") is True
        assert analyzer._should_exclude_file("node_modules/package.json") is True
        assert analyzer._should_exclude_file("src/main.py") is False
        assert analyzer._should_exclude_file("tests/test_main.py") is False

    def test_batch_processing(self, temp_dir):
        """Test batch processing of commits."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        # Set small batch size for testing
        analyzer = GitAnalyzer(cache, batch_size=2)

        assert analyzer.batch_size == 2

        # Test batch generation
        test_commits = [Mock() for _ in range(5)]
        batches = list(analyzer._batch_commits(test_commits, 2))

        assert len(batches) == 3  # 2 + 2 + 1
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1
