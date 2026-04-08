"""Tests for AI tool detection utility (gitflow_analytics/utils/ai_detection.py)."""

from __future__ import annotations

import pytest

from gitflow_analytics.utils.ai_detection import (
    AI_PATTERNS,
    detect_ai_tool,
    is_ai_assisted,
    is_ai_generated,
)

# ---------------------------------------------------------------------------
# detect_ai_tool
# ---------------------------------------------------------------------------


class TestDetectAiTool:
    """Tests for detect_ai_tool()."""

    # --- Claude Code ---

    def test_claude_code_anthropic_noreply_trailer(self) -> None:
        msg = "feat: implement thing\n\nCo-authored-by: Claude <noreply@anthropic.com>"
        assert detect_ai_tool(msg) == "claude_code"

    def test_claude_code_generated_with_marker(self) -> None:
        msg = "fix: patch bug\n\n🤖 Generated with [Claude Code]\nhttps://claude.ai/claude-code"
        assert detect_ai_tool(msg) == "claude_code"

    def test_claude_code_case_insensitive(self) -> None:
        msg = "CO-AUTHORED-BY: Claude Sonnet <NOREPLY@ANTHROPIC.COM>"
        assert detect_ai_tool(msg) == "claude_code"

    # --- Copilot ---

    def test_copilot_bot_trailer(self) -> None:
        msg = "feat: add endpoint\n\nCo-authored-by: Copilot[bot] <175728472+Copilot@users.noreply.github.com>"
        assert detect_ai_tool(msg) == "copilot"

    def test_copilot_github_email(self) -> None:
        msg = "refactor: cleanup\n\nCo-authored-by: github-copilot <copilot@github.com>"
        assert detect_ai_tool(msg) == "copilot"

    def test_copilot_suggestion_marker(self) -> None:
        msg = "chore: update deps\n\nSuggestion from @copilot"
        assert detect_ai_tool(msg) == "copilot"

    def test_copilot_suggestion_no_at_sign(self) -> None:
        msg = "Suggestion from copilot applied"
        assert detect_ai_tool(msg) == "copilot"

    # --- Cursor ---

    def test_cursor_co_author_trailer(self) -> None:
        msg = "feat: new feature\n\nCo-authored-by: Cursor <cursor@anysvc.com>"
        assert detect_ai_tool(msg) == "cursor"

    def test_cursor_generated_with(self) -> None:
        msg = "docs: update readme\n\nGenerated with Cursor"
        assert detect_ai_tool(msg) == "cursor"

    def test_cursor_ai_label(self) -> None:
        msg = "style: format files\nCursor AI"
        assert detect_ai_tool(msg) == "cursor"

    # --- Mixed ---

    def test_mixed_when_multiple_tools_detected(self) -> None:
        msg = (
            "feat: big commit\n\n"
            "Co-authored-by: Copilot[bot] <copilot@github.com>\n"
            "Co-authored-by: Claude <noreply@anthropic.com>"
        )
        assert detect_ai_tool(msg) == "mixed"

    def test_mixed_cursor_and_copilot(self) -> None:
        msg = "Co-authored-by: Cursor <x@y.com>\nSuggestion from @copilot"
        assert detect_ai_tool(msg) == "mixed"

    # --- No AI ---

    def test_no_ai_returns_none_for_plain_commit(self) -> None:
        msg = "fix: resolve race condition in async handler"
        assert detect_ai_tool(msg) is None

    def test_no_ai_returns_none_for_empty_message(self) -> None:
        assert detect_ai_tool("") is None

    def test_no_ai_returns_none_for_human_co_author(self) -> None:
        msg = "feat: pair programming\n\nCo-authored-by: Alice <alice@example.com>"
        assert detect_ai_tool(msg) is None

    def test_no_false_positive_on_anthropic_in_body(self) -> None:
        # 'anthropic' in body text but not in co-author trailer should NOT match
        msg = "docs: mention Anthropic as an AI company in README"
        assert detect_ai_tool(msg) is None


# ---------------------------------------------------------------------------
# is_ai_assisted
# ---------------------------------------------------------------------------


class TestIsAiAssisted:
    """Tests for is_ai_assisted()."""

    def test_returns_true_for_claude_code_commit(self) -> None:
        msg = "feat: x\n\nCo-authored-by: Claude <noreply@anthropic.com>"
        assert is_ai_assisted(msg) is True

    def test_returns_true_for_copilot_commit(self) -> None:
        msg = "fix: y\n\nCo-authored-by: Copilot[bot] <copilot@github.com>"
        assert is_ai_assisted(msg) is True

    def test_returns_true_for_cursor_commit(self) -> None:
        msg = "Generated with Cursor"
        assert is_ai_assisted(msg) is True

    def test_returns_false_for_plain_commit(self) -> None:
        msg = "chore: bump version"
        assert is_ai_assisted(msg) is False

    def test_returns_false_for_empty_message(self) -> None:
        assert is_ai_assisted("") is False


# ---------------------------------------------------------------------------
# is_ai_generated
# ---------------------------------------------------------------------------


class TestIsAiGenerated:
    """Tests for is_ai_generated().

    is_ai_generated() is narrower than is_ai_assisted() — it returns True only
    when the commit bears a 'generated with' marker, not merely a co-author
    trailer.
    """

    def test_true_for_generated_with_claude_code(self) -> None:
        msg = "feat: implement X\n\n🤖 Generated with [Claude Code]"
        assert is_ai_generated(msg) is True

    def test_true_for_bot_generated_emoji_marker(self) -> None:
        msg = "🤖 generated with something"
        assert is_ai_generated(msg) is True

    def test_false_for_plain_co_authored_by_trailer(self) -> None:
        # Co-author trailer indicates assistance, not full generation
        msg = "feat: new thing\n\nCo-authored-by: Claude <noreply@anthropic.com>"
        assert is_ai_generated(msg) is False

    def test_false_for_copilot_suggestion_marker(self) -> None:
        msg = "Suggestion from @copilot"
        assert is_ai_generated(msg) is False

    def test_false_for_plain_human_commit(self) -> None:
        assert is_ai_generated("fix: null pointer dereference") is False

    def test_false_for_empty_message(self) -> None:
        assert is_ai_generated("") is False

    def test_assisted_but_not_generated_distinction(self) -> None:
        """A commit can be AI-assisted (has trailer) but not AI-generated."""
        msg = "feat: pair coding\n\nCo-authored-by: Claude <noreply@anthropic.com>"
        assert is_ai_assisted(msg) is True
        assert is_ai_generated(msg) is False


# ---------------------------------------------------------------------------
# AI_PATTERNS integrity
# ---------------------------------------------------------------------------


class TestAiPatternsStructure:
    """Sanity-check the AI_PATTERNS constant."""

    def test_all_expected_tools_present(self) -> None:
        assert set(AI_PATTERNS.keys()) == {"claude_code", "copilot", "cursor"}

    def test_each_tool_has_at_least_one_pattern(self) -> None:
        for tool, patterns in AI_PATTERNS.items():
            assert len(patterns) >= 1, f"Tool '{tool}' has no detection patterns"

    def test_all_patterns_are_strings(self) -> None:
        for tool, patterns in AI_PATTERNS.items():
            for p in patterns:
                assert isinstance(p, str), f"Pattern for '{tool}' is not a string: {p!r}"
