"""
Tests for the caching system module.

These tests verify caching functionality including commit caching,
database operations, and cache invalidation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import os

from gitflow_analytics.core.cache import CacheManager, CommitCache
from gitflow_analytics.core.analyzer import CommitData


class TestCacheManager:
    """Test cases for the CacheManager class."""

    def test_init(self, temp_dir):
        """Test CacheManager initialization."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache_manager = CacheManager(str(cache_dir))

        assert cache_manager.cache_dir == str(cache_dir)
        assert cache_manager.db_path == str(cache_dir / "gitflow_cache.db")

    def test_init_creates_directory(self, temp_dir):
        """Test that CacheManager creates cache directory if it doesn't exist."""
        cache_dir = temp_dir / "new_cache_dir"
        assert not cache_dir.exists()

        cache_manager = CacheManager(str(cache_dir))

        # Directory should be created when first accessed
        cache_manager.ensure_cache_dir()
        assert cache_dir.exists()

    def test_clear_cache(self, temp_dir):
        """Test cache clearing functionality."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache_dir.mkdir()

        # Create some dummy cache files
        (cache_dir / "gitflow_cache.db").touch()
        (cache_dir / "identities.db").touch()
        (cache_dir / "temp_file.tmp").touch()

        cache_manager = CacheManager(str(cache_dir))
        cache_manager.clear_cache()

        # Cache files should be removed
        assert not (cache_dir / "gitflow_cache.db").exists()
        assert not (cache_dir / "identities.db").exists()
        # Directory should still exist
        assert cache_dir.exists()

    def test_get_cache_info(self, temp_dir):
        """Test getting cache information."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache_dir.mkdir()

        # Create cache file with some content
        cache_file = cache_dir / "gitflow_cache.db"
        cache_file.write_text("dummy cache content")

        cache_manager = CacheManager(str(cache_dir))
        info = cache_manager.get_cache_info()

        assert "gitflow_cache.db" in info
        assert info["gitflow_cache.db"]["exists"] is True
        assert info["gitflow_cache.db"]["size"] > 0

    @patch("gitflow_analytics.core.cache.create_engine")
    @patch("gitflow_analytics.core.cache.sessionmaker")
    def test_get_session(self, mock_sessionmaker, mock_create_engine, temp_dir):
        """Test database session creation."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache_manager = CacheManager(str(cache_dir))

        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        mock_session_class = Mock()
        mock_sessionmaker.return_value = mock_session_class

        mock_session = Mock()
        mock_session_class.return_value = mock_session

        session = cache_manager.get_session()

        mock_create_engine.assert_called_once()
        mock_sessionmaker.assert_called_once_with(bind=mock_engine)
        assert session == mock_session


class TestCommitCache:
    """Test cases for the CommitCache class."""

    def test_init(self, temp_dir):
        """Test CommitCache initialization."""
        cache_dir = temp_dir / ".gitflow-cache"
        commit_cache = CommitCache(str(cache_dir))

        assert commit_cache.cache_manager.cache_dir == str(cache_dir)

    def test_cache_key_generation(self, temp_dir):
        """Test cache key generation for repositories."""
        cache_dir = temp_dir / ".gitflow-cache"
        commit_cache = CommitCache(str(cache_dir))

        repo_path = "/path/to/repo"
        weeks = 4

        key = commit_cache._generate_cache_key(repo_path, weeks)

        assert repo_path in key
        assert str(weeks) in key

    @patch("gitflow_analytics.core.cache.session_scope")
    def test_get_cached_commits_found(self, mock_session_scope, temp_dir):
        """Test retrieving cached commits when they exist."""
        mock_session = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Mock cached commit data
        mock_cached_commit = Mock()
        mock_cached_commit.commit_hash = "abc123"
        mock_cached_commit.author_name = "John Doe"
        mock_cached_commit.author_email = "john@example.com"
        mock_cached_commit.committer_name = "John Doe"
        mock_cached_commit.committer_email = "john@example.com"
        mock_cached_commit.commit_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_cached_commit.message = "feat: add feature"
        mock_cached_commit.files_changed = 3
        mock_cached_commit.insertions = 25
        mock_cached_commit.deletions = 5
        mock_cached_commit.branch = "main"

        mock_session.query.return_value.filter.return_value.all.return_value = [mock_cached_commit]

        cache_dir = temp_dir / ".gitflow-cache"
        commit_cache = CommitCache(str(cache_dir))

        commits = commit_cache.get_cached_commits("/path/to/repo", 4)

        assert len(commits) == 1
        commit = commits[0]
        assert isinstance(commit, CommitData)
        assert commit.hash == "abc123"
        assert commit.author_name == "John Doe"
        assert commit.message == "feat: add feature"

    @patch("gitflow_analytics.core.cache.session_scope")
    def test_get_cached_commits_not_found(self, mock_session_scope, temp_dir):
        """Test retrieving cached commits when they don't exist."""
        mock_session = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        mock_session.query.return_value.filter.return_value.all.return_value = []

        cache_dir = temp_dir / ".gitflow-cache"
        commit_cache = CommitCache(str(cache_dir))

        commits = commit_cache.get_cached_commits("/path/to/repo", 4)

        assert commits is None

    @patch("gitflow_analytics.core.cache.session_scope")
    def test_cache_commits(self, mock_session_scope, temp_dir):
        """Test caching commits to database."""
        mock_session = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Create sample commits to cache
        commit_date = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        commits = [
            CommitData(
                hash="abc123",
                author_name="John Doe",
                author_email="john@example.com",
                committer_name="John Doe",
                committer_email="john@example.com",
                date=commit_date,
                message="feat: add feature",
                files_changed=3,
                insertions=25,
                deletions=5,
                branch="main",
            )
        ]

        cache_dir = temp_dir / ".gitflow-cache"
        commit_cache = CommitCache(str(cache_dir))

        commit_cache.cache_commits("/path/to/repo", 4, commits)

        # Verify that commits were added to session
        mock_session.add.assert_called()
        mock_session.commit.assert_called_once()

    @patch("gitflow_analytics.core.cache.session_scope")
    def test_invalidate_cache(self, mock_session_scope, temp_dir):
        """Test cache invalidation for specific repository."""
        mock_session = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.delete.return_value = 5  # 5 records deleted

        cache_dir = temp_dir / ".gitflow-cache"
        commit_cache = CommitCache(str(cache_dir))

        deleted_count = commit_cache.invalidate_cache("/path/to/repo")

        assert deleted_count == 5
        mock_session.commit.assert_called_once()

    def test_is_cache_valid_fresh(self, temp_dir):
        """Test cache validity check for fresh cache."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache_dir.mkdir()

        # Create a fresh cache file
        cache_file = cache_dir / "gitflow_cache.db"
        cache_file.touch()

        commit_cache = CommitCache(str(cache_dir))

        # Fresh cache should be valid
        assert commit_cache.is_cache_valid("/path/to/repo", max_age_hours=24) is True

    def test_is_cache_valid_stale(self, temp_dir):
        """Test cache validity check for stale cache."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache_dir.mkdir()

        # Create a cache file and make it old
        cache_file = cache_dir / "gitflow_cache.db"
        cache_file.touch()

        # Modify file time to be older than max_age
        old_time = datetime.now().timestamp() - (25 * 3600)  # 25 hours ago
        os.utime(cache_file, (old_time, old_time))

        commit_cache = CommitCache(str(cache_dir))

        # Stale cache should be invalid
        assert commit_cache.is_cache_valid("/path/to/repo", max_age_hours=24) is False

    def test_is_cache_valid_missing(self, temp_dir):
        """Test cache validity check for missing cache file."""
        cache_dir = temp_dir / ".gitflow-cache"
        commit_cache = CommitCache(str(cache_dir))

        # Missing cache should be invalid
        assert commit_cache.is_cache_valid("/path/to/repo", max_age_hours=24) is False

    @patch("gitflow_analytics.core.cache.session_scope")
    def test_get_cache_statistics(self, mock_session_scope, temp_dir):
        """Test getting cache statistics."""
        mock_session = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Mock statistics queries
        mock_session.query.return_value.count.return_value = 100
        mock_session.query.return_value.distinct.return_value.count.return_value = 5

        cache_dir = temp_dir / ".gitflow-cache"
        commit_cache = CommitCache(str(cache_dir))

        stats = commit_cache.get_cache_statistics()

        assert "total_commits" in stats
        assert "total_repositories" in stats
        assert stats["total_commits"] == 100
        assert stats["total_repositories"] == 5


class TestCacheIntegration:
    """Integration tests for cache functionality."""

    def test_full_cache_cycle(self, temp_dir):
        """Test complete cache cycle: store, retrieve, invalidate."""
        cache_dir = temp_dir / ".gitflow-cache"
        commit_cache = CommitCache(str(cache_dir))

        # Create sample commits
        commit_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        original_commits = [
            CommitData(
                hash="abc123",
                author_name="John Doe",
                author_email="john@example.com",
                committer_name="John Doe",
                committer_email="john@example.com",
                date=commit_date,
                message="feat: add feature",
                files_changed=3,
                insertions=25,
                deletions=5,
                branch="main",
            )
        ]

        repo_path = "/test/repo"
        weeks = 4

        with patch("gitflow_analytics.core.cache.session_scope") as mock_session_scope:
            mock_session = Mock()
            mock_session_scope.return_value.__enter__.return_value = mock_session

            # Test caching
            commit_cache.cache_commits(repo_path, weeks, original_commits)
            mock_session.add.assert_called()
            mock_session.commit.assert_called()

            # Test retrieval
            mock_cached_commit = Mock()
            mock_cached_commit.commit_hash = "abc123"
            mock_cached_commit.author_name = "John Doe"
            mock_cached_commit.author_email = "john@example.com"
            mock_cached_commit.committer_name = "John Doe"
            mock_cached_commit.committer_email = "john@example.com"
            mock_cached_commit.commit_date = commit_date
            mock_cached_commit.message = "feat: add feature"
            mock_cached_commit.files_changed = 3
            mock_cached_commit.insertions = 25
            mock_cached_commit.deletions = 5
            mock_cached_commit.branch = "main"

            mock_session.query.return_value.filter.return_value.all.return_value = [
                mock_cached_commit
            ]

            retrieved_commits = commit_cache.get_cached_commits(repo_path, weeks)
            assert len(retrieved_commits) == 1
            assert retrieved_commits[0].hash == "abc123"

            # Test invalidation
            mock_session.query.return_value.filter.return_value.delete.return_value = 1
            deleted_count = commit_cache.invalidate_cache(repo_path)
            assert deleted_count == 1
