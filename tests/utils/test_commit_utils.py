"""Tests for commit utility functions."""

from unittest.mock import Mock

from gitflow_analytics.utils.commit_utils import (
    extract_co_authors,
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


class TestExtractCoAuthors:
    """Tests for extract_co_authors function (Gap 4: Co-authored-by trailers)."""

    def test_single_co_author(self) -> None:
        """A standard Co-authored-by trailer is parsed correctly."""
        message = "Fix auth bug\n\nCo-authored-by: Alice <alice@example.com>"
        result = extract_co_authors(message)
        assert result == [{"name": "Alice", "email": "alice@example.com"}]

    def test_multiple_co_authors(self) -> None:
        """Multiple Co-authored-by trailers are all extracted."""
        message = (
            "Pair-programming session\n\n"
            "Co-authored-by: Bob <bob@example.com>\n"
            "Co-authored-by: Carol <carol@test.io>"
        )
        result = extract_co_authors(message)
        assert len(result) == 2
        names = {r["name"] for r in result}
        emails = {r["email"] for r in result}
        assert names == {"Bob", "Carol"}
        assert emails == {"bob@example.com", "carol@test.io"}

    def test_email_normalized_to_lowercase(self) -> None:
        """Email addresses are lowercased for consistent matching."""
        message = "Refactor\n\nCo-authored-by: Dev <DEV@EXAMPLE.COM>"
        result = extract_co_authors(message)
        assert result == [{"name": "Dev", "email": "dev@example.com"}]

    def test_no_co_authors_returns_empty_list(self) -> None:
        """Plain commit messages return an empty list."""
        message = "chore: bump version\n\nSome description."
        result = extract_co_authors(message)
        assert result == []

    def test_case_insensitive_trailer_key(self) -> None:
        """Trailer key matching is case-insensitive per Git convention."""
        message = "Mob programming\n\nCO-AUTHORED-BY: Eve <eve@example.com>"
        result = extract_co_authors(message)
        assert len(result) == 1
        assert result[0]["name"] == "Eve"

    def test_github_copilot_style_trailer(self) -> None:
        """GitHub Copilot / VS Code auto-inserts a bot trailer — it should be extracted."""
        message = (
            "Generate boilerplate\n\n"
            "Co-authored-by: GitHub Copilot <copilot@github.com>"
        )
        result = extract_co_authors(message)
        assert len(result) == 1
        assert result[0]["name"] == "GitHub Copilot"
        assert result[0]["email"] == "copilot@github.com"

    def test_trailer_with_extra_whitespace(self) -> None:
        """Leading/trailing whitespace around name is stripped."""
        message = "Fix\n\nCo-authored-by:  Alice   <alice@example.com>"
        result = extract_co_authors(message)
        assert result[0]["name"] == "Alice"
        assert result[0]["email"] == "alice@example.com"

    def test_empty_message_returns_empty_list(self) -> None:
        """Empty commit message does not raise and returns empty list."""
        result = extract_co_authors("")
        assert result == []

    def test_trailer_without_angle_brackets_not_matched(self) -> None:
        """Malformed trailer without proper <email> syntax is skipped."""
        message = "Fix\n\nCo-authored-by: Alice alice@example.com"
        result = extract_co_authors(message)
        assert result == []

    def test_trailer_at_start_of_message(self) -> None:
        """Trailer works even at the very start of the message (edge case)."""
        message = "Co-authored-by: Start <start@example.com>"
        result = extract_co_authors(message)
        assert len(result) == 1
        assert result[0]["name"] == "Start"
