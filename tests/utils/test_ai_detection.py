"""Tests for AI tool detection utility (gitflow_analytics/utils/ai_detection.py)."""

from __future__ import annotations

from gitflow_analytics.utils.ai_detection import (
    AI_PATTERNS,
    detect_ai_commit,
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


# ---------------------------------------------------------------------------
# detect_ai_commit — issue #47 signal-based detection
# ---------------------------------------------------------------------------


class TestDetectAiCommitCoAuthorSignals:
    """Co-author trailer signals (highest confidence)."""

    def test_co_author_claude_fires_on_claude_trailer(self) -> None:
        msg = (
            "feat: implement auth service\n\n"
            "Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>"
        )
        confidence, method = detect_ai_commit(msg)
        assert method == "co_author_claude"
        assert confidence == 0.95

    def test_co_author_claude_case_insensitive(self) -> None:
        msg = "feat: x\n\nCO-AUTHORED-BY: Claude <NOREPLY@ANTHROPIC.COM>"
        confidence, method = detect_ai_commit(msg)
        assert method == "co_author_claude"
        assert confidence == 0.95

    def test_co_author_copilot_fires_on_copilot_trailer(self) -> None:
        msg = (
            "fix: null pointer\n\n"
            "Co-authored-by: GitHub Copilot <175728472+Copilot@users.noreply.github.com>"
        )
        confidence, method = detect_ai_commit(msg)
        assert method == "co_author_copilot"
        assert confidence == 0.95

    def test_co_author_copilot_bracket_bot_variant(self) -> None:
        msg = "fix: bug\n\nCo-authored-by: Copilot[bot] <copilot@github.com>"
        confidence, method = detect_ai_commit(msg)
        assert method == "co_author_copilot"
        assert confidence == 0.95

    def test_cursor_pattern_fires_on_cursor_trailer(self) -> None:
        msg = "feat: x\n\nCo-authored-by: cursor <cursor@anysvc.com>"
        confidence, method = detect_ai_commit(msg)
        assert method == "cursor_pattern"
        assert confidence == 0.90

    def test_cursor_pattern_fires_on_generated_with_cursor(self) -> None:
        msg = "docs: readme\n\nGenerated with Cursor"
        confidence, method = detect_ai_commit(msg)
        assert method == "cursor_pattern"
        assert confidence == 0.90


class TestDetectAiCommitFileSignals:
    """File-based signals fire when the corresponding file is in changed_files."""

    def test_cursorrules_touch_fires_on_cursorrules_in_changed_files(self) -> None:
        msg = "chore: update project rules"
        confidence, method = detect_ai_commit(msg, [".cursorrules"])
        assert method == "cursorrules_touch"
        assert confidence == 0.75

    def test_cursorrules_touch_matches_nested_path(self) -> None:
        msg = "chore: update nested cursor rules"
        confidence, method = detect_ai_commit(msg, ["subdir/.cursorrules"])
        assert method == "cursorrules_touch"
        assert confidence == 0.75

    def test_copilot_instructions_touch_fires_on_instructions_file(self) -> None:
        msg = "docs: update copilot instructions"
        confidence, method = detect_ai_commit(msg, [".github/copilot-instructions.md"])
        assert method == "copilot_instructions_touch"
        assert confidence == 0.70

    def test_file_signals_skipped_when_changed_files_none(self) -> None:
        # Backfill path — only message-based detection should run.  With a
        # plain message and no file list, we expect 'none'.
        msg = "chore: update project config"
        confidence, method = detect_ai_commit(msg, None)
        assert method == "none"
        assert confidence == 0.0

    def test_file_signals_skipped_when_changed_files_empty(self) -> None:
        msg = "chore: update project config"
        confidence, method = detect_ai_commit(msg, [])
        assert method == "none"
        assert confidence == 0.0


class TestDetectAiCommitMessageSignals:
    """Message-body phrase signals are the weakest — only fire when nothing stronger does."""

    def test_message_pattern_fires_on_generated_by_ai(self) -> None:
        msg = "feat: add caching layer\n\nThis module was generated by AI."
        confidence, method = detect_ai_commit(msg)
        assert method == "message_pattern"
        assert confidence == 0.60

    def test_message_pattern_fires_on_ai_assisted_phrase(self) -> None:
        msg = "feat: implement X (AI-assisted refactor)"
        confidence, method = detect_ai_commit(msg)
        assert method == "message_pattern"
        assert confidence == 0.60

    def test_message_pattern_fires_on_bot_generated_marker(self) -> None:
        msg = "feat: X\n\n🤖 Generated with [Claude Code]\nhttps://claude.ai/code"
        confidence, method = detect_ai_commit(msg)
        assert method == "message_pattern"
        assert confidence == 0.60


class TestDetectAiCommitNoneSignal:
    """Plain commits with no AI signals return the 'none' signal."""

    def test_none_for_plain_commit(self) -> None:
        msg = "fix: resolve race condition in async handler"
        confidence, method = detect_ai_commit(msg)
        assert method == "none"
        assert confidence == 0.0

    def test_none_for_empty_message(self) -> None:
        confidence, method = detect_ai_commit("")
        assert method == "none"
        assert confidence == 0.0

    def test_none_for_human_co_author(self) -> None:
        msg = "feat: pair programming\n\nCo-authored-by: Alice <alice@example.com>"
        confidence, method = detect_ai_commit(msg)
        assert method == "none"
        assert confidence == 0.0


class TestDetectAiCommitPrecedence:
    """Highest-confidence signal wins when multiple signals fire."""

    def test_claude_trailer_beats_cursorrules_file(self) -> None:
        # Claude trailer (0.95) should win over cursorrules touch (0.75)
        msg = "feat: huge change\n\n" "Co-authored-by: Claude <noreply@anthropic.com>"
        confidence, method = detect_ai_commit(msg, [".cursorrules", "src/main.py"])
        assert method == "co_author_claude"
        assert confidence == 0.95

    def test_cursor_trailer_beats_cursorrules_file(self) -> None:
        # Cursor trailer (0.90) beats cursorrules touch (0.75)
        msg = "feat: x\n\nCo-authored-by: cursor <c@c.com>"
        confidence, method = detect_ai_commit(msg, [".cursorrules"])
        assert method == "cursor_pattern"
        assert confidence == 0.90

    def test_cursorrules_file_beats_copilot_instructions_file(self) -> None:
        # Cursorrules (0.75) beats copilot instructions (0.70)
        msg = "chore: update rules"
        confidence, method = detect_ai_commit(
            msg, [".cursorrules", ".github/copilot-instructions.md"]
        )
        assert method == "cursorrules_touch"
        assert confidence == 0.75

    def test_copilot_trailer_beats_message_pattern(self) -> None:
        # Copilot trailer (0.95) beats message pattern (0.60)
        msg = "feat: code was ai-assisted\n\n" "Co-authored-by: Copilot[bot] <copilot@github.com>"
        confidence, method = detect_ai_commit(msg)
        assert method == "co_author_copilot"
        assert confidence == 0.95

    def test_copilot_instructions_file_beats_message_pattern(self) -> None:
        # Copilot instructions touch (0.70) beats message pattern (0.60)
        msg = "docs: ai-assisted update to instructions"
        confidence, method = detect_ai_commit(msg, [".github/copilot-instructions.md"])
        assert method == "copilot_instructions_touch"
        assert confidence == 0.70


class TestDetectAiCommitBackfillBehavior:
    """Backfill path (changed_files=None) skips file-based signals gracefully."""

    def test_backfill_preserves_co_author_signals(self) -> None:
        msg = "feat: x\n\nCo-authored-by: Claude <noreply@anthropic.com>"
        confidence, method = detect_ai_commit(msg, None)
        assert method == "co_author_claude"
        assert confidence == 0.95

    def test_backfill_preserves_message_signals(self) -> None:
        msg = "feat: add caching (AI-assisted)"
        confidence, method = detect_ai_commit(msg, None)
        assert method == "message_pattern"
        assert confidence == 0.60

    def test_backfill_without_file_info_does_not_crash(self) -> None:
        # Even a commit that WOULD match .cursorrules touch should gracefully
        # fall back to 'none' when we can't see the file list.
        msg = "chore: update rules"
        confidence, method = detect_ai_commit(msg, None)
        # No file info → file signal can't fire → falls through to 'none'
        assert method == "none"
        assert confidence == 0.0

    def test_backfill_return_types(self) -> None:
        confidence, method = detect_ai_commit("any message", None)
        assert isinstance(confidence, float)
        assert isinstance(method, str)
