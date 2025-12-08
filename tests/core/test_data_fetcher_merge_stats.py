"""
Unit tests for GitDataFetcher._calculate_commit_stats merge commit handling.

Tests specifically focus on the merge commit exclusion logic in the
_calculate_commit_stats method, including raw stats preservation and
filtered stats calculation.
"""

from unittest.mock import MagicMock

import pytest

from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.core.data_fetcher import GitDataFetcher


def create_mock_commit_for_stats(
    hexsha: str = "abc123",
    parent_count: int = 1,
    diff_output: str = "100\t50\tmain.py",
) -> MagicMock:
    """Create a mock commit for stats calculation testing.

    Args:
        hexsha: Commit hash
        parent_count: Number of parents (0=initial, 1=regular, 2+=merge)
        diff_output: Git diff --numstat output format

    Returns:
        Mock commit object configured for stats testing
    """
    mock_commit = MagicMock()
    mock_commit.hexsha = hexsha

    # Create parent mocks
    mock_commit.parents = [MagicMock() for _ in range(parent_count)]
    if parent_count > 0:
        for i, parent in enumerate(mock_commit.parents):
            parent.hexsha = f"parent{i}_{hexsha}"

    # Configure repo
    mock_repo = MagicMock()
    mock_commit.repo = mock_repo
    mock_repo.working_dir = "/tmp/test_repo"

    # Configure git operations
    if parent_count > 0:
        # Regular commit or merge commit uses git diff
        mock_repo.git.diff.return_value = diff_output
    else:
        # Initial commit uses git show
        mock_repo.git.show.return_value = diff_output

    return mock_commit


class TestDataFetcherMergeStats:
    """Test GitDataFetcher._calculate_commit_stats merge commit handling."""

    @pytest.fixture
    def data_fetcher(self, temp_dir):
        """Create a GitDataFetcher instance."""
        cache = GitAnalysisCache(temp_dir / ".gitflow-cache")
        return GitDataFetcher(cache=cache)

    def test_calculate_stats_merge_commit_excluded(self, data_fetcher):
        """_calculate_commit_stats should return 0 filtered stats for merge commits."""
        data_fetcher.exclude_merge_commits = True

        # Create merge commit with significant changes
        diff_output = """500\t250\tsrc/main.py
300\t150\tsrc/utils.py
200\t100\ttests/test_main.py"""

        mock_commit = create_mock_commit_for_stats(
            hexsha="merge_excluded",
            parent_count=2,  # Merge commit
            diff_output=diff_output,
        )

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # Filtered stats should be 0
        assert stats["files"] == 0, "Merge commit should have 0 filtered files"
        assert stats["insertions"] == 0, "Merge commit should have 0 filtered insertions"
        assert stats["deletions"] == 0, "Merge commit should have 0 filtered deletions"

    def test_calculate_stats_regular_commit(self, data_fetcher):
        """_calculate_commit_stats should calculate normal stats for regular commits."""
        data_fetcher.exclude_merge_commits = True

        # Create regular commit with 1 parent
        diff_output = """150\t75\tsrc/service.py
200\t100\tsrc/models.py
50\t25\tREADME.md"""

        mock_commit = create_mock_commit_for_stats(
            hexsha="regular_commit",
            parent_count=1,  # Regular commit
            diff_output=diff_output,
        )

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # Regular commits should have normal calculated stats
        assert stats["files"] == 3, "Regular commit should count all files"
        assert stats["insertions"] == 400, "Regular commit should sum insertions (150+200+50)"
        assert stats["deletions"] == 200, "Regular commit should sum deletions (75+100+25)"

    def test_calculate_stats_preserves_raw_stats(self, data_fetcher):
        """Raw stats should be preserved even when filtered stats are 0."""
        data_fetcher.exclude_merge_commits = True

        # Create merge commit
        diff_output = """1000\t500\tbig_file.py
250\t125\tsmall_file.py"""

        mock_commit = create_mock_commit_for_stats(
            hexsha="merge_raw_preserved",
            parent_count=2,  # Merge commit
            diff_output=diff_output,
        )

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # Filtered stats are 0
        assert stats["files"] == 0
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0

        # But raw stats are preserved
        assert stats["raw_insertions"] == 1250, "Raw insertions should be 1000+250"
        assert stats["raw_deletions"] == 625, "Raw deletions should be 500+125"

    def test_calculate_stats_initial_commit_counts(self, data_fetcher):
        """Initial commits (0 parents) should have normal stats when exclusion enabled."""
        data_fetcher.exclude_merge_commits = True

        # Create initial commit (no parents)
        diff_output = """2000\t0\tinitial_file.py
500\t0\tREADME.md"""

        mock_commit = create_mock_commit_for_stats(
            hexsha="initial_commit",
            parent_count=0,  # Initial commit
            diff_output=diff_output,
        )

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # Initial commits are NOT merges, should count normally
        assert stats["files"] == 2
        assert stats["insertions"] == 2500, "Initial commit insertions: 2000+500"
        assert stats["deletions"] == 0
        assert stats["raw_insertions"] == 2500
        assert stats["raw_deletions"] == 0

    def test_calculate_stats_when_exclusion_disabled(self, data_fetcher):
        """When exclude_merge_commits=False, merge commits should have normal stats."""
        data_fetcher.exclude_merge_commits = False

        # Create merge commit
        diff_output = """600\t300\tmerged_file.py
400\t200\tanother_file.py"""

        mock_commit = create_mock_commit_for_stats(
            hexsha="merge_not_excluded",
            parent_count=2,  # Merge commit
            diff_output=diff_output,
        )

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # When exclusion disabled, merge commits count normally
        assert stats["files"] == 2
        assert stats["insertions"] == 1000, "Merge commit insertions: 600+400"
        assert stats["deletions"] == 500, "Merge commit deletions: 300+200"
        assert stats["raw_insertions"] == 1000
        assert stats["raw_deletions"] == 500

    def test_calculate_stats_empty_merge_commit(self, data_fetcher):
        """Empty merge commits should still be excluded when flag is set."""
        data_fetcher.exclude_merge_commits = True

        # Create empty merge commit (no changes)
        mock_commit = create_mock_commit_for_stats(
            hexsha="empty_merge",
            parent_count=2,
            diff_output="",  # Empty diff
        )

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # Empty merge commit should have all 0s
        assert stats["files"] == 0
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0
        assert stats["raw_insertions"] == 0
        assert stats["raw_deletions"] == 0

    def test_calculate_stats_binary_files_in_merge(self, data_fetcher):
        """Test merge commits with binary files (marked with -)."""
        data_fetcher.exclude_merge_commits = True

        # Binary files show as "-\t-" in numstat
        diff_output = """100\t50\ttext_file.py
-\t-\timage.png
200\t100\tanother_text.py
-\t-\tbinary_data.bin"""

        mock_commit = create_mock_commit_for_stats(
            hexsha="merge_with_binary", parent_count=2, diff_output=diff_output
        )

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # Merge commit: filtered stats should be 0
        assert stats["files"] == 0
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0

        # Raw stats should count text files (binary files are skipped)
        # 100+200 insertions, 50+100 deletions, 2 text files
        assert stats["raw_insertions"] == 300
        assert stats["raw_deletions"] == 150

    def test_calculate_stats_with_exclude_paths_and_merge(self, data_fetcher):
        """Test interaction between exclude_paths and merge commit exclusion."""
        data_fetcher.exclude_merge_commits = True
        data_fetcher.exclude_paths = ["node_modules/**", "*.min.js"]

        # Merge commit with some excluded files
        diff_output = """100\t50\tsrc/main.py
1000\t500\tnode_modules/lib.js
200\t100\tdist/app.min.js
50\t25\ttests/test.py"""

        mock_commit = create_mock_commit_for_stats(
            hexsha="merge_with_exclusions", parent_count=2, diff_output=diff_output
        )

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # Merge commit: filtered stats should be 0 (merge takes precedence)
        assert stats["files"] == 0
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0

        # Raw stats should include all files
        assert stats["raw_insertions"] == 1350  # 100+1000+200+50
        assert stats["raw_deletions"] == 675  # 50+500+100+25

    def test_calculate_stats_octopus_merge(self, data_fetcher):
        """Test octopus merges (3+ parents) are excluded."""
        data_fetcher.exclude_merge_commits = True

        # Octopus merge with 4 parents
        diff_output = """1000\t500\toctopus_file1.py
2000\t1000\toctopus_file2.py
500\t250\toctopus_file3.py"""

        mock_commit = create_mock_commit_for_stats(
            hexsha="octopus_merge",
            parent_count=4,  # Octopus merge
            diff_output=diff_output,
        )

        stats = data_fetcher._calculate_commit_stats(mock_commit)

        # Octopus merges should also be excluded
        assert stats["files"] == 0
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0

        # Raw stats preserved
        assert stats["raw_insertions"] == 3500  # 1000+2000+500
        assert stats["raw_deletions"] == 1750  # 500+1000+250

    def test_calculate_stats_raw_stats_always_present(self, data_fetcher):
        """Raw stats fields should always be present in return dict."""
        data_fetcher.exclude_merge_commits = True

        # Test merge commit
        merge_commit = create_mock_commit_for_stats(
            hexsha="has_raw_merge", parent_count=2, diff_output="100\t50\tfile.py"
        )

        merge_stats = data_fetcher._calculate_commit_stats(merge_commit)
        assert "raw_insertions" in merge_stats, "raw_insertions should be in stats dict"
        assert "raw_deletions" in merge_stats, "raw_deletions should be in stats dict"

        # Test regular commit
        data_fetcher.exclude_merge_commits = False
        regular_commit = create_mock_commit_for_stats(
            hexsha="has_raw_regular", parent_count=1, diff_output="200\t100\tfile.py"
        )

        regular_stats = data_fetcher._calculate_commit_stats(regular_commit)
        assert "raw_insertions" in regular_stats, "raw_insertions should be in stats dict"
        assert "raw_deletions" in regular_stats, "raw_deletions should be in stats dict"
