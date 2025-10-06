"""Tests for GitDataFetcher module."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.core.data_fetcher import GitDataFetcher


class TestGitDataFetcher:
    """Test suite for GitDataFetcher class."""

    @pytest.fixture
    def temp_cache(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = GitAnalysisCache(Path(temp_dir))
            yield cache

    @pytest.fixture
    def data_fetcher(self, temp_cache):
        """Create a GitDataFetcher instance."""
        return GitDataFetcher(cache=temp_cache)

    def test_init(self, temp_cache):
        """Test GitDataFetcher initialization."""
        fetcher = GitDataFetcher(
            cache=temp_cache,
            branch_mapping_rules={"main": ["main", "master"]},
            allowed_ticket_platforms=["jira", "github"],
            exclude_paths=["**/node_modules/**", "**/vendor/**"],
        )

        assert fetcher.cache == temp_cache
        assert fetcher.exclude_paths == ["**/node_modules/**", "**/vendor/**"]
        assert fetcher.database is not None
        assert fetcher.story_point_extractor is not None
        assert fetcher.ticket_extractor is not None
        assert fetcher.branch_mapper is not None
        assert fetcher.identity_resolver is not None

    def test_should_exclude_file_basic_patterns(self, data_fetcher):
        """Test basic file exclusion patterns."""
        data_fetcher.exclude_paths = ["*.log", "*.tmp", "test_*"]

        # Should exclude
        assert data_fetcher._should_exclude_file("debug.log") is True
        assert data_fetcher._should_exclude_file("temp.tmp") is True
        assert data_fetcher._should_exclude_file("test_file.py") is True

        # Should not exclude
        assert data_fetcher._should_exclude_file("main.py") is False
        assert data_fetcher._should_exclude_file("config.json") is False

    def test_should_exclude_file_recursive_patterns(self, data_fetcher):
        """Test recursive ** patterns for file exclusion."""
        data_fetcher.exclude_paths = [
            "**/node_modules/**",
            "**/vendor/**",
            "**/*.min.js",
            "**/package-lock.json",
        ]

        # Should exclude
        assert data_fetcher._should_exclude_file("node_modules/lib/index.js") is True
        assert data_fetcher._should_exclude_file("src/node_modules/package.json") is True
        assert data_fetcher._should_exclude_file("vendor/jquery.js") is True
        assert data_fetcher._should_exclude_file("dist/vendor/bootstrap.css") is True
        assert data_fetcher._should_exclude_file("assets/script.min.js") is True
        assert data_fetcher._should_exclude_file("src/assets/app.min.js") is True
        assert data_fetcher._should_exclude_file("package-lock.json") is True
        assert data_fetcher._should_exclude_file("frontend/package-lock.json") is True

        # Should not exclude
        assert data_fetcher._should_exclude_file("src/main.js") is False
        assert data_fetcher._should_exclude_file("app.js") is False
        assert data_fetcher._should_exclude_file("package.json") is False

    def test_calculate_commit_stats_with_exclusions(self, data_fetcher):
        """Test that _calculate_commit_stats properly filters excluded files."""
        # Mock commit with git diff output
        mock_commit = MagicMock()
        mock_repo = MagicMock()
        mock_commit.repo = mock_repo
        mock_commit.hexsha = "abc123def456"
        mock_commit.parents = [MagicMock()]  # Has parent

        # Simulate git diff --numstat output
        diff_output = """100\t0\tmain.py
1000\t0\tpackage-lock.json
500\t0\tvendor/lib.js
50\t0\tutils.py
200\t50\tnode_modules/some_lib.js
300\t100\tdist/app.min.js"""

        mock_repo.git.diff.return_value = diff_output

        # Test without exclusions
        data_fetcher.exclude_paths = []
        stats = data_fetcher._calculate_commit_stats(mock_commit)

        assert stats["files"] == 6
        assert stats["insertions"] == 2150  # 100+1000+500+50+200+300
        assert stats["deletions"] == 150  # 0+0+0+0+50+100

        # Test with exclusions
        data_fetcher.exclude_paths = [
            "**/package-lock.json",
            "**/vendor/**",
            "**/node_modules/**",
            "**/*.min.js",
        ]
        stats = data_fetcher._calculate_commit_stats(mock_commit)

        assert stats["files"] == 2  # Only main.py and utils.py
        assert stats["insertions"] == 150  # 100+50
        assert stats["deletions"] == 0  # 0+0

    def test_calculate_commit_stats_initial_commit(self, data_fetcher):
        """Test _calculate_commit_stats for initial commit (no parent)."""
        # Mock initial commit
        mock_commit = MagicMock()
        mock_repo = MagicMock()
        mock_commit.repo = mock_repo
        mock_commit.hexsha = "initial123"
        mock_commit.parents = []  # No parent (initial commit)

        # Simulate git show --numstat output for initial commit
        show_output = """50\t0\tREADME.md
100\t0\tsrc/main.py
1500\t0\tpackage-lock.json"""

        mock_repo.git.show.return_value = show_output

        # Test with exclusions
        data_fetcher.exclude_paths = ["**/package-lock.json"]
        stats = data_fetcher._calculate_commit_stats(mock_commit)

        assert stats["files"] == 2  # README.md and src/main.py
        assert stats["insertions"] == 150  # 50+100
        assert stats["deletions"] == 0

    def test_matches_glob_pattern_edge_cases(self, data_fetcher):
        """Test edge cases in glob pattern matching."""
        # Empty inputs
        assert data_fetcher._matches_glob_pattern("", "*.py") is False
        assert data_fetcher._matches_glob_pattern("file.py", "") is False
        assert data_fetcher._matches_glob_pattern("", "") is False

        # Complex patterns
        assert (
            data_fetcher._matches_glob_pattern("src/components/Button/index.js", "**/components/**")
            is True
        )

        assert (
            data_fetcher._matches_glob_pattern("build/static/js/main.chunk.js", "**/build/**")
            is True
        )

    def test_match_recursive_pattern(self, data_fetcher):
        """Test complex recursive pattern matching."""
        # Multiple ** patterns
        pattern = "**/src/**/test/**/*.spec.js"

        assert (
            data_fetcher._match_recursive_pattern(
                "project/src/components/test/button.spec.js", pattern
            )
            is True
        )

        assert (
            data_fetcher._match_recursive_pattern("src/utils/test/helper.spec.js", pattern) is True
        )

        assert data_fetcher._match_recursive_pattern("src/main.js", pattern) is False

    @patch("git.Repo")
    def test_fetch_repository_data_integration(self, mock_repo_class, data_fetcher):
        """Test full integration of fetch_repository_data with exclude patterns."""
        # This is a more complex integration test
        # In a real scenario, you might want to use a real test repository

        with tempfile.TemporaryDirectory() as repo_dir:
            repo_path = Path(repo_dir)

            # Mock repository
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo

            # Configure exclude patterns
            data_fetcher.exclude_paths = [
                "**/node_modules/**",
                "**/vendor/**",
                "**/*.min.js",
                "**/package-lock.json",
            ]

            # Note: This would require more complex mocking of the git operations
            # For now, we just verify the method doesn't crash with exclude_paths set

            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=7)

            # Mock the internal methods to prevent actual git operations
            data_fetcher._fetch_commits_by_day = MagicMock(return_value={})
            data_fetcher._extract_all_ticket_references = MagicMock(return_value=set())
            data_fetcher._build_commit_ticket_correlations = MagicMock(return_value=0)
            data_fetcher._store_daily_batches = MagicMock(return_value=0)
            data_fetcher._verify_commit_storage = MagicMock(
                return_value={"actual_stored": 0, "total_found": 0, "expected_new": 0}
            )

            result = data_fetcher.fetch_repository_data(
                repo_path=repo_path, project_key="TEST", start_date=start_date, end_date=end_date
            )

            assert result["project_key"] == "TEST"
            assert result["stats"]["stored_commits"] == 0
