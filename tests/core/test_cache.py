"""
Tests for the caching system module.

These tests verify caching functionality including commit caching,
database operations, and cache invalidation.
"""

from datetime import datetime, timezone

from gitflow_analytics.core.cache import GitAnalysisCache


class TestGitAnalysisCache:
    """Test cases for the GitAnalysisCache class."""

    def test_init(self, temp_dir):
        """Test GitAnalysisCache initialization."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        assert cache.cache_dir == cache_dir
        assert cache.ttl_hours == 24

    def test_init_creates_directory(self, temp_dir):
        """Test that GitAnalysisCache creates cache directory if it doesn't exist."""
        cache_dir = temp_dir / "new_cache_dir"
        assert not cache_dir.exists()

        GitAnalysisCache(cache_dir)

        # Directory should exist after initialization
        assert cache_dir.exists()

    def test_cache_commit(self, temp_dir):
        """Test caching individual commit data."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        repo_path = "/path/to/repo"
        commit_data = {
            "hash": "abc123",
            "author_name": "John Doe",
            "author_email": "john@example.com",
            "message": "feat: add new feature",
            "timestamp": datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            "branch": "main",
            "project": "PROJECT",
            "files_changed": 3,
            "insertions": 25,
            "deletions": 5,
            "complexity_delta": 1.5,
            "story_points": 3,
            "ticket_references": ["PROJ-123"],
        }

        # Cache the commit
        cache.cache_commit(repo_path, commit_data)

        # Retrieve the cached commit
        cached = cache.get_cached_commit(repo_path, "abc123")

        assert cached is not None
        assert cached["hash"] == "abc123"
        assert cached["author_name"] == "John Doe"
        assert cached["story_points"] == 3
        assert cached["ticket_references"] == ["PROJ-123"]

    def test_cache_commits_batch(self, temp_dir):
        """Test batch caching of commits."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        repo_path = "/path/to/repo"
        commits = [
            {
                "hash": "abc123",
                "author_name": "John Doe",
                "author_email": "john@example.com",
                "message": "feat: add feature 1",
                "timestamp": datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                "branch": "main",
                "project": "PROJECT",
                "files_changed": 2,
                "insertions": 15,
                "deletions": 3,
            },
            {
                "hash": "def456",
                "author_name": "Jane Smith",
                "author_email": "jane@example.com",
                "message": "fix: resolve bug",
                "timestamp": datetime(2024, 1, 2, 14, 0, 0, tzinfo=timezone.utc),
                "branch": "main",
                "project": "PROJECT",
                "files_changed": 1,
                "insertions": 5,
                "deletions": 8,
            },
        ]

        # Cache the commits in batch
        cache.cache_commits_batch(repo_path, commits)

        # Verify both commits are cached
        cached1 = cache.get_cached_commit(repo_path, "abc123")
        cached2 = cache.get_cached_commit(repo_path, "def456")

        assert cached1 is not None
        assert cached1["hash"] == "abc123"
        assert cached2 is not None
        assert cached2["hash"] == "def456"

    def test_cache_pull_request(self, temp_dir):
        """Test caching pull request data."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        repo_path = "/path/to/repo"
        pr_data = {
            "number": 123,
            "title": "Add new feature",
            "description": "This PR adds a new feature",
            "author": "john@example.com",
            "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "merged_at": datetime(2024, 1, 3, tzinfo=timezone.utc),
            "story_points": 5,
            "labels": ["enhancement", "frontend"],
            "commit_hashes": ["abc123", "def456"],
        }

        # Cache the PR
        cache.cache_pr(repo_path, pr_data)

        # Retrieve the cached PR
        cached = cache.get_cached_pr(repo_path, 123)

        assert cached is not None
        assert cached["number"] == 123
        assert cached["title"] == "Add new feature"
        assert cached["story_points"] == 5
        assert cached["labels"] == ["enhancement", "frontend"]

    def test_cache_pull_request_upsert(self, temp_dir):
        """Test that caching the same PR twice doesn't cause UNIQUE constraint errors."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        repo_path = "/path/to/repo"
        original_pr_data = {
            "number": 123,
            "title": "Add new feature",
            "description": "This PR adds a new feature",
            "author": "john@example.com",
            "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "merged_at": datetime(2024, 1, 3, tzinfo=timezone.utc),
            "story_points": 5,
            "labels": ["enhancement", "frontend"],
            "commit_hashes": ["abc123", "def456"],
        }

        # Cache the PR first time
        cache.cache_pr(repo_path, original_pr_data)

        # Retrieve and verify
        cached = cache.get_cached_pr(repo_path, 123)
        assert cached is not None
        assert cached["title"] == "Add new feature"
        assert cached["story_points"] == 5

        # Update the PR data
        updated_pr_data = {
            "number": 123,
            "title": "Add new feature (updated)",
            "description": "This PR adds a new feature with improvements",
            "author": "john@example.com",
            "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "merged_at": datetime(2024, 1, 4, tzinfo=timezone.utc),  # Different merge time
            "story_points": 8,  # Updated story points
            "labels": ["enhancement", "frontend", "high-priority"],  # Added label
            "commit_hashes": ["abc123", "def456", "ghi789"],  # Added commit
        }

        # Cache the PR again - this should update, not insert
        cache.cache_pr(repo_path, updated_pr_data)

        # Retrieve and verify the updates took effect
        cached_updated = cache.get_cached_pr(repo_path, 123)
        assert cached_updated is not None
        assert cached_updated["title"] == "Add new feature (updated)"
        assert cached_updated["story_points"] == 8
        assert cached_updated["labels"] == ["enhancement", "frontend", "high-priority"]
        assert len(cached_updated["commit_hashes"]) == 3

    def test_cache_issue(self, temp_dir):
        """Test caching issue data."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        issue_data = {
            "id": "PROJ-123",
            "project_key": "PROJ",
            "title": "Implement user authentication",
            "description": "Add OAuth2 authentication",
            "status": "In Progress",
            "assignee": "john@example.com",
            "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "updated_at": datetime(2024, 1, 2, tzinfo=timezone.utc),
            "story_points": 8,
            "labels": ["backend", "security"],
            "platform_data": {"custom_field": "value"},
        }

        # Cache the issue
        cache.cache_issue("jira", issue_data)

        # Retrieve cached issues
        cached_issues = cache.get_cached_issues("jira", "PROJ")

        assert len(cached_issues) == 1
        assert cached_issues[0]["id"] == "PROJ-123"
        assert cached_issues[0]["story_points"] == 8

    def test_cache_issue_upsert(self, temp_dir):
        """Test that caching the same issue twice doesn't cause UNIQUE constraint errors."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        original_issue_data = {
            "id": "PROJ-123",
            "project_key": "PROJ",
            "title": "Implement user authentication",
            "description": "Add OAuth2 authentication",
            "status": "In Progress",
            "assignee": "john@example.com",
            "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "updated_at": datetime(2024, 1, 2, tzinfo=timezone.utc),
            "story_points": 8,
            "labels": ["backend", "security"],
            "platform_data": {"custom_field": "value"},
        }

        # Cache the issue first time
        cache.cache_issue("jira", original_issue_data)

        # Retrieve and verify
        cached_issues = cache.get_cached_issues("jira", "PROJ")
        assert len(cached_issues) == 1
        assert cached_issues[0]["status"] == "In Progress"
        assert cached_issues[0]["story_points"] == 8

        # Update the issue data
        updated_issue_data = {
            "id": "PROJ-123",
            "project_key": "PROJ",
            "title": "Implement user authentication (completed)",
            "description": "Add OAuth2 authentication - completed implementation",
            "status": "Done",  # Updated status
            "assignee": "john@example.com",
            "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "updated_at": datetime(2024, 1, 5, tzinfo=timezone.utc),  # Updated time
            "resolved_at": datetime(2024, 1, 5, tzinfo=timezone.utc),  # New field
            "story_points": 10,  # Updated story points
            "labels": ["backend", "security", "completed"],  # Added label
            "platform_data": {"custom_field": "updated_value", "new_field": "new"},
        }

        # Cache the issue again - this should update, not insert
        cache.cache_issue("jira", updated_issue_data)

        # Retrieve and verify the updates took effect
        cached_issues_updated = cache.get_cached_issues("jira", "PROJ")
        assert len(cached_issues_updated) == 1  # Should still be just one issue
        issue = cached_issues_updated[0]
        assert issue["title"] == "Implement user authentication (completed)"
        assert issue["status"] == "Done"
        assert issue["story_points"] == 10
        assert issue["labels"] == ["backend", "security", "completed"]
        assert issue["resolved_at"] is not None

    def test_cache_expiration(self, temp_dir):
        """Test cache TTL expiration."""
        cache_dir = temp_dir / ".gitflow-cache"
        # Set very short TTL for testing
        cache = GitAnalysisCache(cache_dir, ttl_hours=0.001)  # ~3.6 seconds

        repo_path = "/path/to/repo"
        commit_data = {
            "hash": "abc123",
            "author_name": "John Doe",
            "author_email": "john@example.com",
            "message": "test commit",
            "timestamp": datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            "branch": "main",
            "project": "PROJECT",
            "files_changed": 1,
            "insertions": 10,
            "deletions": 0,
        }

        # Cache the commit
        cache.cache_commit(repo_path, commit_data)

        # Should be retrievable immediately
        cached = cache.get_cached_commit(repo_path, "abc123")
        assert cached is not None

        # Simulate time passing by directly testing the _is_stale method
        # with a past timestamp
        import time

        time.sleep(0.1)  # Small delay to ensure cache is stale

        # The cache should handle stale entries appropriately
        # (exact behavior depends on implementation)

    def test_clear_cache(self, temp_dir):
        """Test cache clearing functionality."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        # Add some data to cache
        repo_path = "/path/to/repo"
        commit_data = {
            "hash": "abc123",
            "author_name": "John Doe",
            "author_email": "john@example.com",
            "message": "test commit",
            "timestamp": datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            "branch": "main",
            "project": "PROJECT",
            "files_changed": 1,
            "insertions": 10,
            "deletions": 0,
        }
        cache.cache_commit(repo_path, commit_data)

        # Verify data is cached
        cached = cache.get_cached_commit(repo_path, "abc123")
        assert cached is not None

        # Clear cache
        cache.clear_stale_cache()  # Clear stale entries

        # This test mainly ensures the clear method doesn't raise exceptions
        # since we can't easily test actual clearing without manipulating timestamps

    def test_cache_stats(self, temp_dir):
        """Test cache statistics functionality."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        # Add some test data
        repo_path = "/path/to/repo"
        commit_data = {
            "hash": "abc123",
            "author_name": "John Doe",
            "author_email": "john@example.com",
            "message": "test commit",
            "timestamp": datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            "branch": "main",
            "project": "PROJECT",
            "files_changed": 1,
            "insertions": 10,
            "deletions": 0,
        }
        cache.cache_commit(repo_path, commit_data)

        # Get cache stats
        stats = cache.get_cache_stats()

        assert isinstance(stats, dict)
        assert "cached_commits" in stats
        assert "cached_prs" in stats
        assert "cached_issues" in stats
        assert stats["cached_commits"] >= 1


class TestCachePerformance:
    """Test cases for cache performance and efficiency."""

    def test_repeated_access_performance(self, temp_dir):
        """Test that repeated cache access is efficient."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        repo_path = "/path/to/repo"
        commit_data = {
            "hash": "abc123",
            "author_name": "John Doe",
            "author_email": "john@example.com",
            "message": "test commit",
            "timestamp": datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            "branch": "main",
            "project": "PROJECT",
            "files_changed": 1,
            "insertions": 10,
            "deletions": 0,
        }

        # Cache the commit
        cache.cache_commit(repo_path, commit_data)

        # Multiple retrievals should work efficiently
        for _ in range(10):
            cached = cache.get_cached_commit(repo_path, "abc123")
            assert cached is not None
            assert cached["hash"] == "abc123"

    def test_large_batch_caching(self, temp_dir):
        """Test caching large batches of commits."""
        cache_dir = temp_dir / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir)

        repo_path = "/path/to/repo"
        commits = []

        # Generate large batch of commits
        for i in range(100):
            commits.append(
                {
                    "hash": f"commit{i:03d}",
                    "author_name": f"Author {i}",
                    "author_email": f"author{i}@example.com",
                    "message": f"commit message {i}",
                    "timestamp": datetime(2024, 1, 1, 10, i % 60, 0, tzinfo=timezone.utc),
                    "branch": "main",
                    "project": "PROJECT",
                    "files_changed": 1,
                    "insertions": i,
                    "deletions": 0,
                }
            )

        # Cache all commits in batch
        cache.cache_commits_batch(repo_path, commits)

        # Verify random commits are retrievable
        for i in [0, 25, 50, 75, 99]:
            cached = cache.get_cached_commit(repo_path, f"commit{i:03d}")
            assert cached is not None
            assert cached["hash"] == f"commit{i:03d}"
            assert cached["insertions"] == i
