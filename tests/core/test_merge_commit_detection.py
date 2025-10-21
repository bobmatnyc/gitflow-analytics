"""
Comprehensive unit tests for merge commit detection logic.

Tests the core merge commit detection and filtering in both GitDataFetcher
and GitAnalyzer, covering edge cases like initial commits, regular commits,
two-parent merges, and octopus merges.
"""

from unittest.mock import MagicMock

import pytest

from gitflow_analytics.core.analyzer import GitAnalyzer
from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.core.data_fetcher import GitDataFetcher


def create_mock_commit(
    hexsha: str = "abc123def456",
    parent_count: int = 1,
    insertions: int = 100,
    deletions: int = 50,
    files_changed: int = 1,
) -> MagicMock:
    """Create a mock Git commit object for testing.

    Args:
        hexsha: Commit hash (default: "abc123def456")
        parent_count: Number of parents (0=initial, 1=regular, 2+=merge)
        insertions: Number of lines inserted
        deletions: Number of lines deleted
        files_changed: Number of files changed

    Returns:
        Mock commit object with configured parents and diff output
    """
    mock_commit = MagicMock()
    mock_commit.hexsha = hexsha

    # Create parent mocks
    mock_commit.parents = [MagicMock() for _ in range(parent_count)]
    if parent_count > 0:
        # Set hexsha for parent commits
        for i, parent in enumerate(mock_commit.parents):
            parent.hexsha = f"parent{i}_{hexsha}"

    # Configure repo and git operations
    mock_repo = MagicMock()
    mock_commit.repo = mock_repo
    mock_repo.working_dir = "/tmp/test_repo"

    # Create numstat output: insertions\tdeletions\tfilename
    diff_lines = []
    for i in range(files_changed):
        diff_lines.append(f"{insertions}\t{deletions}\tfile{i}.py")
    diff_output = "\n".join(diff_lines)

    if parent_count > 0:
        # Regular commit uses git diff
        mock_repo.git.diff.return_value = diff_output
    else:
        # Initial commit uses git show
        mock_repo.git.show.return_value = diff_output

    return mock_commit


class TestMergeCommitDetectionDataFetcher:
    """Test merge commit detection in GitDataFetcher."""

    @pytest.fixture
    def data_fetcher(self, temp_dir):
        """Create a GitDataFetcher instance."""
        cache = GitAnalysisCache(temp_dir / ".gitflow-cache")
        return GitDataFetcher(cache=cache)

    def test_single_parent_commit_not_merge(self, data_fetcher):
        """Regular commit with 1 parent should not be detected as merge."""
        data_fetcher.exclude_merge_commits = True

        # Create regular commit with 1 parent
        mock_commit = create_mock_commit(
            hexsha="regular123",
            parent_count=1,
            insertions=100,
            deletions=50,
            files_changed=2,
        )

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # Regular commits should have normal stats (not filtered)
        assert stats["files"] == 2
        assert stats["insertions"] == 200  # 100 per file * 2 files
        assert stats["deletions"] == 100  # 50 per file * 2 files
        assert stats["raw_insertions"] == 200
        assert stats["raw_deletions"] == 100

    def test_two_parent_commit_is_merge(self, data_fetcher):
        """Commit with 2 parents should be detected as merge."""
        data_fetcher.exclude_merge_commits = True

        # Create merge commit with 2 parents
        mock_commit = create_mock_commit(
            hexsha="merge123", parent_count=2, insertions=500, deletions=200, files_changed=5
        )

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # Merge commits should have filtered stats = 0
        assert stats["files"] == 0
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0

        # Raw stats should still be calculated
        assert stats["raw_insertions"] == 2500  # 500 per file * 5 files
        assert stats["raw_deletions"] == 1000  # 200 per file * 5 files

    def test_octopus_merge_detection(self, data_fetcher):
        """Commit with 3+ parents should be detected as merge."""
        data_fetcher.exclude_merge_commits = True

        # Create octopus merge with 3 parents
        mock_commit = create_mock_commit(
            hexsha="octopus123", parent_count=3, insertions=300, deletions=100, files_changed=10
        )

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # Octopus merges should also be excluded
        assert stats["files"] == 0
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0

        # Raw stats preserved
        assert stats["raw_insertions"] == 3000  # 300 per file * 10 files
        assert stats["raw_deletions"] == 1000  # 100 per file * 10 files

    def test_initial_commit_not_merge(self, data_fetcher):
        """Commit with 0 parents should not be detected as merge."""
        data_fetcher.exclude_merge_commits = True

        # Create initial commit (no parents)
        mock_commit = create_mock_commit(
            hexsha="initial123", parent_count=0, insertions=1000, deletions=0, files_changed=5
        )

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # Initial commits are not merges, should count normally
        assert stats["files"] == 5
        assert stats["insertions"] == 5000  # 1000 per file * 5 files
        assert stats["deletions"] == 0
        assert stats["raw_insertions"] == 5000
        assert stats["raw_deletions"] == 0

    def test_merge_commit_filtered_stats_zero(self, data_fetcher):
        """When exclude_merge_commits=True, filtered stats should be 0."""
        data_fetcher.exclude_merge_commits = True

        # Create merge commit
        mock_commit = create_mock_commit(parent_count=2, insertions=500, deletions=250)

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # All filtered stats must be exactly 0
        assert stats["files"] == 0, "Filtered files should be 0 for merge commits"
        assert stats["insertions"] == 0, "Filtered insertions should be 0 for merge commits"
        assert stats["deletions"] == 0, "Filtered deletions should be 0 for merge commits"

    def test_merge_commit_raw_stats_preserved(self, data_fetcher):
        """When exclude_merge_commits=True, raw stats should still be calculated."""
        data_fetcher.exclude_merge_commits = True

        # Create merge commit with known stats
        mock_commit = create_mock_commit(
            parent_count=2, insertions=750, deletions=350, files_changed=3
        )

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # Raw stats must be preserved
        assert stats["raw_insertions"] == 2250, "Raw insertions should be calculated"
        assert stats["raw_deletions"] == 1050, "Raw deletions should be calculated"

        # But filtered stats are 0
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0

    def test_exclude_merge_commits_disabled(self, data_fetcher):
        """When exclude_merge_commits=False, merge commits should have normal stats."""
        data_fetcher.exclude_merge_commits = False

        # Create merge commit
        mock_commit = create_mock_commit(
            parent_count=2, insertions=400, deletions=200, files_changed=4
        )

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # When exclusion disabled, merge commits count normally
        assert stats["files"] == 4
        assert stats["insertions"] == 1600  # 400 per file * 4 files
        assert stats["deletions"] == 800  # 200 per file * 4 files
        assert stats["raw_insertions"] == 1600
        assert stats["raw_deletions"] == 800


class TestMergeCommitDetectionAnalyzer:
    """Test merge commit detection in GitAnalyzer."""

    @pytest.fixture
    def analyzer(self, temp_dir):
        """Create a GitAnalyzer instance."""
        cache = GitAnalysisCache(temp_dir / ".gitflow-cache")
        return GitAnalyzer(cache=cache)

    def test_single_parent_commit_not_merge_analyzer(self, analyzer):
        """GitAnalyzer: Regular commit with 1 parent should not be detected as merge."""
        analyzer.exclude_merge_commits = True

        # Create regular commit with 1 parent
        mock_commit = create_mock_commit(
            hexsha="regular456", parent_count=1, insertions=150, deletions=75, files_changed=3
        )

        stats = analyzer._calculate_filtered_stats(mock_commit)

        # Regular commits should have normal stats
        assert stats["files"] == 3
        assert stats["insertions"] == 450  # 150 per file * 3 files
        assert stats["deletions"] == 225  # 75 per file * 3 files

    def test_two_parent_commit_is_merge_analyzer(self, analyzer):
        """GitAnalyzer: Commit with 2 parents should be detected as merge."""
        analyzer.exclude_merge_commits = True

        # Create merge commit with 2 parents
        mock_commit = create_mock_commit(
            hexsha="merge456", parent_count=2, insertions=600, deletions=300, files_changed=6
        )

        stats = analyzer._calculate_filtered_stats(mock_commit)

        # Merge commits should have filtered stats = 0
        assert stats["files"] == 0
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0

    def test_octopus_merge_detection_analyzer(self, analyzer):
        """GitAnalyzer: Commit with 3+ parents should be detected as merge."""
        analyzer.exclude_merge_commits = True

        # Create octopus merge with 4 parents
        mock_commit = create_mock_commit(
            hexsha="octopus456", parent_count=4, insertions=800, deletions=400, files_changed=8
        )

        stats = analyzer._calculate_filtered_stats(mock_commit)

        # Octopus merges should be excluded
        assert stats["files"] == 0
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0

    def test_initial_commit_not_merge_analyzer(self, analyzer):
        """GitAnalyzer: Commit with 0 parents should not be detected as merge."""
        analyzer.exclude_merge_commits = True

        # Create initial commit (no parents)
        mock_commit = create_mock_commit(
            hexsha="initial456", parent_count=0, insertions=2000, deletions=0, files_changed=10
        )

        stats = analyzer._calculate_filtered_stats(mock_commit)

        # Initial commits should count normally
        assert stats["files"] == 10
        assert stats["insertions"] == 20000  # 2000 per file * 10 files
        assert stats["deletions"] == 0

    def test_exclude_merge_commits_disabled_analyzer(self, analyzer):
        """GitAnalyzer: When exclude_merge_commits=False, merge commits should have normal stats."""
        analyzer.exclude_merge_commits = False

        # Create merge commit
        mock_commit = create_mock_commit(
            parent_count=2, insertions=500, deletions=250, files_changed=5
        )

        stats = analyzer._calculate_filtered_stats(mock_commit)

        # When exclusion disabled, merge commits count normally
        assert stats["files"] == 5
        assert stats["insertions"] == 2500  # 500 per file * 5 files
        assert stats["deletions"] == 1250  # 250 per file * 5 files


class TestEdgeCases:
    """Test edge cases in merge commit detection."""

    @pytest.fixture
    def data_fetcher(self, temp_dir):
        """Create a GitDataFetcher instance."""
        cache = GitAnalysisCache(temp_dir / ".gitflow-cache")
        return GitDataFetcher(cache=cache)

    def test_empty_merge_commit(self, data_fetcher):
        """Empty merge commits (no file changes) should still be excluded."""
        data_fetcher.exclude_merge_commits = True

        # Create empty merge commit (no changes)
        mock_commit = create_mock_commit(
            hexsha="empty_merge", parent_count=2, insertions=0, deletions=0, files_changed=0
        )

        # Mock empty diff output
        mock_commit.repo.git.diff.return_value = ""

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # Empty merge commit should still be excluded (0 stats)
        assert stats["files"] == 0
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0
        assert stats["raw_insertions"] == 0
        assert stats["raw_deletions"] == 0

    def test_parent_count_boundary_values(self, data_fetcher):
        """Test parent count boundary values for merge detection."""
        data_fetcher.exclude_merge_commits = True

        # Test 0 parents (initial commit) - NOT a merge
        commit_0 = create_mock_commit(hexsha="c0", parent_count=0, insertions=100)
        stats_0 = data_fetcher._calculate_commit_stats(commit_0)
        assert stats_0["insertions"] > 0, "0 parents should NOT be merge"

        # Test 1 parent (regular commit) - NOT a merge
        commit_1 = create_mock_commit(hexsha="c1", parent_count=1, insertions=100)
        stats_1 = data_fetcher._calculate_commit_stats(commit_1)
        assert stats_1["insertions"] > 0, "1 parent should NOT be merge"

        # Test 2 parents (merge commit) - IS a merge
        commit_2 = create_mock_commit(hexsha="c2", parent_count=2, insertions=100)
        stats_2 = data_fetcher._calculate_commit_stats(commit_2)
        assert stats_2["insertions"] == 0, "2 parents IS a merge"

        # Test 5 parents (octopus merge) - IS a merge
        commit_5 = create_mock_commit(hexsha="c5", parent_count=5, insertions=100)
        stats_5 = data_fetcher._calculate_commit_stats(commit_5)
        assert stats_5["insertions"] == 0, "5 parents IS a merge"

    def test_consistency_between_fetcher_and_analyzer(self, temp_dir):
        """Ensure GitDataFetcher and GitAnalyzer use same merge detection logic."""
        cache = GitAnalysisCache(temp_dir / ".gitflow-cache")
        fetcher = GitDataFetcher(cache=cache, exclude_merge_commits=True)
        analyzer = GitAnalyzer(cache=cache, exclude_merge_commits=True)

        # Create same merge commit for both
        merge_commit = create_mock_commit(
            hexsha="consistency_test", parent_count=2, insertions=500, deletions=250
        )

        # Test both implementations
        fetcher_stats = fetcher._calculate_commit_stats(merge_commit)
        analyzer_stats = analyzer._calculate_filtered_stats(merge_commit)

        # Both should detect as merge and return 0 filtered stats
        assert fetcher_stats["files"] == 0
        assert fetcher_stats["insertions"] == 0
        assert fetcher_stats["deletions"] == 0

        assert analyzer_stats["files"] == 0
        assert analyzer_stats["insertions"] == 0
        assert analyzer_stats["deletions"] == 0
