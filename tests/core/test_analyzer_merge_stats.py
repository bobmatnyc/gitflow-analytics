"""
Unit tests for GitAnalyzer._calculate_filtered_stats merge commit handling.

Tests specifically focus on the merge commit exclusion logic in the
_calculate_filtered_stats method, verifying early return pattern and
consistency with GitDataFetcher.
"""

from unittest.mock import MagicMock

import pytest

from gitflow_analytics.core.analyzer import GitAnalyzer
from gitflow_analytics.core.cache import GitAnalysisCache


def create_mock_commit_for_analyzer(
    hexsha: str = "xyz789",
    parent_count: int = 1,
    diff_output: str = "150\t75\tanalyzer_test.py",
) -> MagicMock:
    """Create a mock commit for analyzer stats testing.

    Args:
        hexsha: Commit hash
        parent_count: Number of parents (0=initial, 1=regular, 2+=merge)
        diff_output: Git diff --numstat output format

    Returns:
        Mock commit object configured for analyzer testing
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

    # Configure git operations
    if parent_count > 0:
        mock_repo.git.diff.return_value = diff_output
    else:
        mock_repo.git.show.return_value = diff_output

    return mock_commit


class TestAnalyzerMergeStats:
    """Test GitAnalyzer._calculate_filtered_stats merge commit handling."""

    @pytest.fixture
    def analyzer(self, temp_dir):
        """Create a GitAnalyzer instance."""
        cache = GitAnalysisCache(temp_dir / ".gitflow-cache")
        return GitAnalyzer(cache=cache)

    def test_filtered_stats_merge_commit_excluded(self, analyzer):
        """_calculate_filtered_stats should return 0 for merge commits."""
        analyzer.exclude_merge_commits = True

        # Create merge commit
        diff_output = """800\t400\tmerge_feature.py
600\t300\tmerge_tests.py
200\t100\tmerge_docs.md"""

        mock_commit = create_mock_commit_for_analyzer(
            hexsha="analyzer_merge",
            parent_count=2,  # Merge commit
            diff_output=diff_output,
        )

        stats = analyzer._calculate_filtered_stats(mock_commit)

        # Merge commits should return 0 for all filtered stats
        assert stats["files"] == 0, "Merge commit should have 0 files"
        assert stats["insertions"] == 0, "Merge commit should have 0 insertions"
        assert stats["deletions"] == 0, "Merge commit should have 0 deletions"

    def test_filtered_stats_regular_commit(self, analyzer):
        """_calculate_filtered_stats should calculate normal stats for regular commits."""
        analyzer.exclude_merge_commits = True

        # Create regular commit
        diff_output = """250\t125\tregular_feature.py
350\t175\tregular_utils.py
100\t50\tregular_test.py"""

        mock_commit = create_mock_commit_for_analyzer(
            hexsha="analyzer_regular",
            parent_count=1,  # Regular commit
            diff_output=diff_output,
        )

        stats = analyzer._calculate_filtered_stats(mock_commit)

        # Regular commits should have normal stats
        assert stats["files"] == 3
        assert stats["insertions"] == 700, "Regular commit insertions: 250+350+100"
        assert stats["deletions"] == 350, "Regular commit deletions: 125+175+50"

    def test_filtered_stats_initial_commit(self, analyzer):
        """Initial commits should have normal stats calculated."""
        analyzer.exclude_merge_commits = True

        # Create initial commit (no parents)
        diff_output = """3000\t0\tinitial_setup.py
1500\t0\tinitial_config.yaml
500\t0\tREADME.md"""

        mock_commit = create_mock_commit_for_analyzer(
            hexsha="analyzer_initial",
            parent_count=0,  # Initial commit
            diff_output=diff_output,
        )

        stats = analyzer._calculate_filtered_stats(mock_commit)

        # Initial commits are NOT merges
        assert stats["files"] == 3
        assert stats["insertions"] == 5000, "Initial commit insertions: 3000+1500+500"
        assert stats["deletions"] == 0

    def test_filtered_stats_when_exclusion_disabled(self, analyzer):
        """When exclude_merge_commits=False, merge commits should have normal stats."""
        analyzer.exclude_merge_commits = False

        # Create merge commit
        diff_output = """1000\t500\tmerge_not_excluded.py
750\t375\tanother_file.py"""

        mock_commit = create_mock_commit_for_analyzer(
            hexsha="analyzer_merge_counted",
            parent_count=2,  # Merge commit
            diff_output=diff_output,
        )

        stats = analyzer._calculate_filtered_stats(mock_commit)

        # When exclusion disabled, merge commits count normally
        assert stats["files"] == 2
        assert stats["insertions"] == 1750, "Merge commit insertions: 1000+750"
        assert stats["deletions"] == 875, "Merge commit deletions: 500+375"

    def test_filtered_stats_octopus_merge(self, analyzer):
        """Octopus merges (3+ parents) should be excluded."""
        analyzer.exclude_merge_commits = True

        # Create octopus merge with 3 parents
        diff_output = """1500\t750\toctopus_file1.py
2500\t1250\toctopus_file2.py
1000\t500\toctopus_file3.py"""

        mock_commit = create_mock_commit_for_analyzer(
            hexsha="analyzer_octopus",
            parent_count=3,  # Octopus merge
            diff_output=diff_output,
        )

        stats = analyzer._calculate_filtered_stats(mock_commit)

        # Octopus merges should be excluded
        assert stats["files"] == 0
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0

    def test_filtered_stats_empty_merge(self, analyzer):
        """Empty merge commits should return 0 stats."""
        analyzer.exclude_merge_commits = True

        # Create empty merge commit
        mock_commit = create_mock_commit_for_analyzer(
            hexsha="analyzer_empty_merge",
            parent_count=2,  # Merge commit
            diff_output="",  # No changes
        )

        stats = analyzer._calculate_filtered_stats(mock_commit)

        # Empty merge should have all 0s
        assert stats["files"] == 0
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0

    def test_filtered_stats_early_return_pattern(self, analyzer):
        """Verify that early return pattern is used for merge commits."""
        analyzer.exclude_merge_commits = True

        # Create merge commit
        mock_commit = create_mock_commit_for_analyzer(
            hexsha="early_return_test", parent_count=2, diff_output="100\t50\tfile.py"
        )

        # The method should return early with 0 stats for merge commits
        # without processing the diff
        stats = analyzer._calculate_filtered_stats(mock_commit)

        assert stats == {
            "files": 0,
            "insertions": 0,
            "deletions": 0,
        }, "Early return should produce exact 0 stats dict"

    def test_filtered_stats_with_exclude_paths(self, analyzer):
        """Test interaction between exclude_paths and merge commit exclusion."""
        analyzer.exclude_merge_commits = True
        analyzer.exclude_paths = ["*.log", "node_modules/**"]

        # Merge commit with some files that would be excluded
        diff_output = """100\t50\tsrc/app.py
500\t250\tdebug.log
1000\t500\tnode_modules/package.json
200\t100\ttests/test.py"""

        mock_commit = create_mock_commit_for_analyzer(
            hexsha="analyzer_with_paths",
            parent_count=2,  # Merge commit
            diff_output=diff_output,
        )

        stats = analyzer._calculate_filtered_stats(mock_commit)

        # Merge exclusion takes precedence: all stats should be 0
        # (early return prevents exclude_paths processing)
        assert stats["files"] == 0
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0

    def test_filtered_stats_binary_files(self, analyzer):
        """Test merge commits with binary files."""
        analyzer.exclude_merge_commits = True

        # Binary files marked with "-\t-"
        diff_output = """200\t100\ttext_file.py
-\t-\timage.png
300\t150\tconfig.json
-\t-\tbinary.bin"""

        mock_commit = create_mock_commit_for_analyzer(
            hexsha="analyzer_binary",
            parent_count=2,  # Merge commit
            diff_output=diff_output,
        )

        stats = analyzer._calculate_filtered_stats(mock_commit)

        # Merge commit: early return with 0 stats
        assert stats["files"] == 0
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0

    def test_filtered_stats_consistency_single_vs_two_parents(self, analyzer):
        """Verify boundary between regular commit (1 parent) and merge (2 parents)."""
        analyzer.exclude_merge_commits = True

        same_diff = "500\t250\tboundary_test.py"

        # 1 parent: should count
        commit_1_parent = create_mock_commit_for_analyzer(
            hexsha="one_parent", parent_count=1, diff_output=same_diff
        )

        stats_1 = analyzer._calculate_filtered_stats(commit_1_parent)
        assert stats_1["insertions"] == 500, "1 parent should count as regular commit"

        # 2 parents: should NOT count
        commit_2_parents = create_mock_commit_for_analyzer(
            hexsha="two_parents", parent_count=2, diff_output=same_diff
        )

        stats_2 = analyzer._calculate_filtered_stats(commit_2_parents)
        assert stats_2["insertions"] == 0, "2 parents should count as merge commit"

    def test_filtered_stats_no_raw_stats_in_return(self, analyzer):
        """GitAnalyzer._calculate_filtered_stats does NOT include raw stats."""
        analyzer.exclude_merge_commits = True

        # Create regular commit
        mock_commit = create_mock_commit_for_analyzer(
            hexsha="no_raw_stats", parent_count=1, diff_output="300\t150\tfile.py"
        )

        stats = analyzer._calculate_filtered_stats(mock_commit)

        # Verify only filtered stats are returned (no raw_insertions/raw_deletions)
        assert "files" in stats
        assert "insertions" in stats
        assert "deletions" in stats
        assert "raw_insertions" not in stats, (
            "GitAnalyzer should NOT include raw_insertions in filtered stats"
        )
        assert "raw_deletions" not in stats, (
            "GitAnalyzer should NOT include raw_deletions in filtered stats"
        )
