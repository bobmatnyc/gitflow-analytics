"""Tests for JIRA adapter story-point preservation (issue #56).

WHY: JIRA supports fractional story points (e.g., modified Fibonacci scale
0.5, 1.5, 3.5).  The previous implementation cast to ``int``, silently
truncating ``3.5`` to ``3``.  These tests pin the behavior that the
extractor now returns ``float`` values without truncation.
"""

from __future__ import annotations

from gitflow_analytics.pm_framework.adapters.jira_adapter_converters import (
    JIRAAdapterConvertersMixin,
)


class _BaseFallback:
    """Stand-in for the PMPlatformAdapter base class fallback.

    The real mixin chains to ``super()._extract_story_points`` if the
    configured fields don't yield a value.  In tests we don't want to drag in
    the full adapter base class, so this returns ``None`` (matches the
    "nothing found" branch of the real base implementation).
    """

    def _extract_story_points(self, _fields: dict[str, object]) -> float | None:
        del _fields
        return None


class _FakeJiraConverter(JIRAAdapterConvertersMixin, _BaseFallback):
    """Minimal stand-in that exposes ``_extract_story_points`` for testing.

    We avoid instantiating the full JIRAAdapter (which requires HTTP credentials
    and a session) by constructing a bare object and attaching only the
    attributes the converter mixin reads.
    """

    def __init__(self, story_point_fields: list[str]) -> None:
        self.story_point_fields = story_point_fields


def test_extract_story_points_preserves_fractional_values() -> None:
    """3.5 must come through as 3.5, not 3."""
    converter = _FakeJiraConverter(story_point_fields=["customfield_10016"])
    fields = {"customfield_10016": 3.5}

    result = converter._extract_story_points(fields)

    assert result == 3.5
    assert isinstance(result, float)


def test_extract_story_points_preserves_fractional_string_values() -> None:
    """String '1.5' must parse to 1.5, not 1."""
    converter = _FakeJiraConverter(story_point_fields=["customfield_10016"])
    fields = {"customfield_10016": "1.5"}

    result = converter._extract_story_points(fields)

    assert result == 1.5
    assert isinstance(result, float)


def test_extract_story_points_integer_values_returned_as_float() -> None:
    """Integer 5 should still work and come through as 5.0 (float)."""
    converter = _FakeJiraConverter(story_point_fields=["customfield_10016"])
    fields = {"customfield_10016": 5}

    result = converter._extract_story_points(fields)

    assert result == 5.0
    assert isinstance(result, float)


def test_extract_story_points_returns_none_when_missing() -> None:
    """Missing fields must still return None."""
    converter = _FakeJiraConverter(story_point_fields=["customfield_10016"])
    fields: dict[str, object] = {}

    assert converter._extract_story_points(fields) is None


def test_extract_story_points_returns_none_for_unparseable_string() -> None:
    """Garbage strings should not raise and should fall through to None."""
    converter = _FakeJiraConverter(story_point_fields=["customfield_10016"])
    fields = {"customfield_10016": "not-a-number"}

    assert converter._extract_story_points(fields) is None


def test_extract_story_points_zero_point_five() -> None:
    """0.5 (smallest common fractional point) must round-trip exactly."""
    converter = _FakeJiraConverter(story_point_fields=["customfield_10016"])
    fields = {"customfield_10016": 0.5}

    result = converter._extract_story_points(fields)

    assert result == 0.5
    assert isinstance(result, float)
