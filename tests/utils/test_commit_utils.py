"""Tests for commit utility functions."""

from unittest.mock import Mock

from gitflow_analytics.utils.commit_utils import (
    get_parent_count,
    is_initial_commit,
    is_merge_commit,
)


class TestIsMergeCommit:
    """Tests for is_merge_commit function."""

    def test_single_parent_not_merge(self):
        """Regular commit with 1 parent is not a merge commit."""
        commit = Mock()
        commit.parents = [Mock()]  # 1 parent
        assert is_merge_commit(commit) is False

    def test_two_parents_is_merge(self):
        """Commit with 2 parents is a merge commit."""
        commit = Mock()
        commit.parents = [Mock(), Mock()]  # 2 parents
        assert is_merge_commit(commit) is True

    def test_octopus_merge(self):
        """Commit with 3+ parents is a merge commit (octopus merge)."""
        commit = Mock()
        commit.parents = [Mock(), Mock(), Mock()]  # 3 parents
        assert is_merge_commit(commit) is True

    def test_initial_commit_not_merge(self):
        """Initial commit with 0 parents is not a merge commit."""
        commit = Mock()
        commit.parents = []  # 0 parents
        assert is_merge_commit(commit) is False


class TestGetParentCount:
    """Tests for get_parent_count function."""

    def test_initial_commit(self):
        commit = Mock()
        commit.parents = []
        assert get_parent_count(commit) == 0

    def test_regular_commit(self):
        commit = Mock()
        commit.parents = [Mock()]
        assert get_parent_count(commit) == 1

    def test_merge_commit(self):
        commit = Mock()
        commit.parents = [Mock(), Mock()]
        assert get_parent_count(commit) == 2


class TestIsInitialCommit:
    """Tests for is_initial_commit function."""

    def test_initial_commit(self):
        commit = Mock()
        commit.parents = []
        assert is_initial_commit(commit) is True

    def test_regular_commit_not_initial(self):
        commit = Mock()
        commit.parents = [Mock()]
        assert is_initial_commit(commit) is False
