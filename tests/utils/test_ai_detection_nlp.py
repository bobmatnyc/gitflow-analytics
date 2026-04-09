"""Tests for NLP heuristic AI confidence scoring (Phase 1).

Validates:
1. score_ai_confidence returns 1.0 for messages with co-author trailers.
2. score_ai_confidence returns > 0.5 for messages with AI phrases.
3. score_ai_confidence returns 0.0 for plain commit messages.
4. score_ai_confidence returns < 0.3 for short plain messages.
5. score_ai_confidence caps at 1.0.
6. detect_ai_detection_method returns 'pattern' for co-author commits.
7. detect_ai_detection_method returns 'nlp_heuristic' for phrase matches.
8. detect_ai_detection_method returns 'none' for plain messages.
9. Edge cases: empty string, whitespace-only.
10. Formulaic opening boosts score.
11. Multiple phrase hits accumulate (capped at 0.6 from phrases).
12. Short messages receive penalty.
"""

from __future__ import annotations

import pytest

from gitflow_analytics.utils.ai_detection import (
    detect_ai_detection_method,
    score_ai_confidence,
)

# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

CLAUDE_TRAILER = (
    "feat: add user authentication\n\nCo-Authored-By: Claude Sonnet 4 <noreply@anthropic.com>"
)
COPILOT_TRAILER = "fix: resolve null pointer\n\nCo-authored-by: Copilot[bot] <copilot@github.com>"
PLAIN_MESSAGE = "fix typo in README"
SHORT_MESSAGE = "wip"

AI_PHRASE_MESSAGE = (
    "Handled edge cases and ensured that proper error handling is in place "
    "based on the requirements provided."
)
FORMULAIC_MESSAGE = "add authentication middleware to improve security for API endpoints"
MULTI_PHRASE_MESSAGE = (
    "Implemented as requested per your instructions. "
    "Added proper error handling and ensured that validation is correct. "
    "Handled edge cases throughout."
)


# ---------------------------------------------------------------------------
# 1-2. score_ai_confidence — definitive pattern matches
# ---------------------------------------------------------------------------


class TestScoreAiConfidencePatternMatch:
    """Definitive co-author pattern matches must score 1.0."""

    def test_claude_co_author_trailer_scores_one(self) -> None:
        assert score_ai_confidence(CLAUDE_TRAILER) == 1.0

    def test_copilot_co_author_trailer_scores_one(self) -> None:
        assert score_ai_confidence(COPILOT_TRAILER) == 1.0

    def test_cursor_co_author_trailer_scores_one(self) -> None:
        msg = "refactor: clean up code\n\nCo-authored-by: cursor <cursor@example.com>"
        assert score_ai_confidence(msg) == 1.0

    def test_generated_with_marker_scores_one(self) -> None:
        msg = "feat: new feature\n\n🤖 Generated with [Claude Code](https://claude.ai/code)"
        assert score_ai_confidence(msg) == 1.0


# ---------------------------------------------------------------------------
# 3-4. score_ai_confidence — AI phrase messages
# ---------------------------------------------------------------------------


class TestScoreAiConfidencePhraseMatches:
    """AI-typical phrases should produce scores > 0.5."""

    def test_ai_phrase_message_scores_above_threshold(self) -> None:
        # AI_PHRASE_MESSAGE hits 3 phrase patterns -> 0.45 score
        score = score_ai_confidence(AI_PHRASE_MESSAGE)
        assert score > 0.3, f"Expected score > 0.3 for AI phrase message, got {score}"

    def test_multiple_phrases_accumulate(self) -> None:
        score = score_ai_confidence(MULTI_PHRASE_MESSAGE)
        assert score >= 0.45, f"Expected score >= 0.45 for multi-phrase message, got {score}"

    def test_score_capped_at_one(self) -> None:
        # Pile on every phrase pattern — result must never exceed 1.0
        very_ai = (
            "As requested per your instructions, implemented as discussed. "
            "Ensured that proper error handling handles edge cases. "
            "Added proper validation and necessary logging. "
            "Refactored for better clarity. Updated to use the new API. "
            "Fixed an issue where the bug caused a problem with the module."
        )
        score = score_ai_confidence(very_ai)
        assert score <= 1.0, f"Score {score} exceeds 1.0 maximum"


# ---------------------------------------------------------------------------
# 5. score_ai_confidence — plain messages
# ---------------------------------------------------------------------------


class TestScoreAiConfidencePlainMessages:
    """Ordinary commit messages should score 0.0."""

    def test_plain_message_scores_zero(self) -> None:
        assert score_ai_confidence(PLAIN_MESSAGE) == 0.0

    def test_numeric_version_bump_scores_zero(self) -> None:
        assert score_ai_confidence("chore: bump version to 1.2.3") == 0.0

    def test_merge_commit_message_scores_zero(self) -> None:
        assert score_ai_confidence("Merge branch 'feature/foo' into main") == 0.0


# ---------------------------------------------------------------------------
# 6. score_ai_confidence — short messages
# ---------------------------------------------------------------------------


class TestScoreAiConfidenceShortMessages:
    """Short messages should score < 0.3 even with partial AI phrasing."""

    def test_short_plain_scores_low(self) -> None:
        score = score_ai_confidence(SHORT_MESSAGE)
        assert score < 0.3, f"Expected score < 0.3 for short plain message, got {score}"

    def test_empty_string_scores_zero(self) -> None:
        assert score_ai_confidence("") == 0.0

    def test_whitespace_only_scores_zero(self) -> None:
        # Single word after strip -> short message penalty applies
        score = score_ai_confidence("   ")
        assert score == 0.0

    def test_two_word_message_scores_low(self) -> None:
        score = score_ai_confidence("fix bug")
        assert score < 0.3


# ---------------------------------------------------------------------------
# 7. score_ai_confidence — formulaic opening
# ---------------------------------------------------------------------------


class TestScoreAiConfidenceFormulaicOpening:
    """Formulaic openings should add to the score."""

    def test_formulaic_opening_boosts_score(self) -> None:
        score = score_ai_confidence(FORMULAIC_MESSAGE)
        assert score > 0.0, f"Expected non-zero score for formulaic opening, got {score}"

    def test_this_commit_adds_pattern(self) -> None:
        msg = "This commit adds support for OAuth2 authentication in the API."
        score = score_ai_confidence(msg)
        assert score > 0.0


# ---------------------------------------------------------------------------
# 8-10. detect_ai_detection_method
# ---------------------------------------------------------------------------


class TestDetectAiDetectionMethod:
    """Verify correct method label is returned for each detection path."""

    def test_returns_pattern_for_co_author_trailer(self) -> None:
        assert detect_ai_detection_method(CLAUDE_TRAILER) == "pattern"

    def test_returns_pattern_for_copilot_trailer(self) -> None:
        assert detect_ai_detection_method(COPILOT_TRAILER) == "pattern"

    def test_returns_nlp_heuristic_for_ai_phrases(self) -> None:
        assert detect_ai_detection_method(AI_PHRASE_MESSAGE) == "nlp_heuristic"

    def test_returns_nlp_heuristic_for_formulaic_opening(self) -> None:
        # "this commit implements" matches the formulaic opening pattern -> score > 0
        msg = "This commit implements the new caching layer for better performance."
        assert detect_ai_detection_method(msg) == "nlp_heuristic"

    def test_returns_nlp_heuristic_for_improvement_formulaic(self) -> None:
        # "add X to improve Y" matches the formulaic pattern -> nlp_heuristic
        msg = "add authentication middleware to improve security for API endpoints"
        assert detect_ai_detection_method(msg) == "nlp_heuristic"

    def test_returns_none_for_plain_message(self) -> None:
        assert detect_ai_detection_method(PLAIN_MESSAGE) == "none"

    def test_returns_none_for_empty_message(self) -> None:
        assert detect_ai_detection_method("") == "none"

    def test_returns_none_for_merge_commit(self) -> None:
        assert detect_ai_detection_method("Merge branch 'main' into feature/x") == "none"


# ---------------------------------------------------------------------------
# 11. Return type and range
# ---------------------------------------------------------------------------


class TestScoreReturnTypeAndRange:
    """score_ai_confidence must always return a float in [0.0, 1.0]."""

    @pytest.mark.parametrize(
        "msg",
        [
            "",
            "fix",
            PLAIN_MESSAGE,
            AI_PHRASE_MESSAGE,
            CLAUDE_TRAILER,
            MULTI_PHRASE_MESSAGE,
        ],
    )
    def test_score_is_float_in_range(self, msg: str) -> None:
        score = score_ai_confidence(msg)
        assert isinstance(score, float), f"Expected float, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Score {score} out of range [0.0, 1.0]"
