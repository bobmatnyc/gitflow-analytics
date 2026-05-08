"""Tests for the JIRA project-key → work_type tier-3 classification signal.

Issue #62: when a commit message contains a JIRA ticket reference whose
project key (e.g. "ADV" in "ADV-1234") appears in the configured
``jira_project_mappings``, the batch classifier should short-circuit the
LLM call and assign the configured work_type with high confidence (0.95).

These tests focus on the pure logic in
``BatchClassifierImplMixin._classify_via_jira_project_key`` and the merge
helper ``_merge_jira_with_llm_results``. They do not exercise the database
layer or LLM provider, so they run quickly and have no external deps.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from gitflow_analytics.classification.batch_classifier import BatchCommitClassifier


class _MockLLMConfig:
    """Minimal LLM config stub matching the duck-typed interface used by
    ``BatchCommitClassifier.__init__`` (sufficient for these unit tests)."""

    def __init__(self) -> None:
        self.enable_caching = False
        self.cache_duration_days = 7
        self.model = "test-model"
        self.api_key = ""  # Empty so llm_enabled is False — keeps tests offline.
        self.max_tokens = 1000
        self.temperature = 0.1
        self.timeout_seconds = 30
        self.confidence_threshold = 0.7
        self.batch_size = 10
        self.max_daily_requests = 1000
        self.aws_region = None
        self.aws_profile = None
        self.bedrock_model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        self.provider = "auto"


def _make_classifier(
    mappings: dict[str, str] | None,
    *,
    show_jira_signals: bool = False,
) -> BatchCommitClassifier:
    """Construct a BatchCommitClassifier with no real LLM provider."""
    tmp = tempfile.mkdtemp()
    return BatchCommitClassifier(
        cache_dir=Path(tmp),
        llm_config=_MockLLMConfig(),
        batch_size=10,
        confidence_threshold=0.7,
        fallback_enabled=True,
        jira_project_mappings=mappings,
        show_jira_signals=show_jira_signals,
    )


def _commit(commit_hash: str, ticket_refs: list[Any], message: str = "test") -> dict[str, Any]:
    return {
        "commit_hash": commit_hash,
        "message": message,
        "ticket_references": ticket_refs,
    }


# ---------------------------------------------------------------------------
# _classify_via_jira_project_key
# ---------------------------------------------------------------------------


def test_jira_project_key_matches_canonical_extractor_format() -> None:
    """Canonical extractor format: list of dicts with platform + id keys."""
    classifier = _make_classifier({"ADV": "feature", "BI": "analytics"})

    commit = _commit(
        "abc123",
        [{"platform": "jira", "id": "ADV-1234", "full_id": "ADV-1234"}],
    )

    result = classifier._classify_via_jira_project_key(commit)
    assert result == ("feature", "ADV")


def test_jira_project_key_matches_bare_string_format() -> None:
    """Back-compat: list of bare ticket ID strings (older cached data)."""
    classifier = _make_classifier({"FD": "feature"})

    commit = _commit("abc123", ["FD-987"])

    result = classifier._classify_via_jira_project_key(commit)
    assert result == ("feature", "FD")


def test_jira_project_key_unmapped_falls_through() -> None:
    """Unmapped project key returns None so caller falls through to LLM."""
    classifier = _make_classifier({"ADV": "feature"})

    commit = _commit(
        "abc123",
        [{"platform": "jira", "id": "ZZZ-1", "full_id": "ZZZ-1"}],
    )

    assert classifier._classify_via_jira_project_key(commit) is None


def test_jira_project_key_no_ticket_refs_returns_none() -> None:
    """Commit with no ticket references returns None."""
    classifier = _make_classifier({"ADV": "feature"})

    commit = _commit("abc123", [])

    assert classifier._classify_via_jira_project_key(commit) is None


def test_jira_project_key_no_mapping_returns_none() -> None:
    """When no mapping is configured, every commit falls through."""
    classifier = _make_classifier(None)

    commit = _commit(
        "abc123",
        [{"platform": "jira", "id": "ADV-1"}],
    )

    assert classifier._classify_via_jira_project_key(commit) is None


def test_jira_project_key_first_match_wins() -> None:
    """When multiple ticket refs are present, the first mapped one wins."""
    classifier = _make_classifier({"ADV": "feature", "OPS": "maintenance"})

    commit = _commit(
        "abc123",
        [
            {"platform": "jira", "id": "OPS-5"},
            {"platform": "jira", "id": "ADV-9"},
        ],
    )

    result = classifier._classify_via_jira_project_key(commit)
    assert result == ("maintenance", "OPS")


def test_jira_project_key_normalises_case() -> None:
    """Mapping keys are case-insensitive (configured as ADV matches adv-1)."""
    classifier = _make_classifier({"ADV": "feature"})

    commit = _commit(
        "abc123",
        [{"platform": "jira", "id": "adv-42"}],
    )

    result = classifier._classify_via_jira_project_key(commit)
    assert result == ("feature", "ADV")


def test_jira_project_key_skips_non_jira_platforms() -> None:
    """Non-JIRA platforms (e.g. github) are not matched against the mapping."""
    classifier = _make_classifier({"ADV": "feature"})

    commit = _commit(
        "abc123",
        [{"platform": "github", "id": "ADV-1"}],
    )

    assert classifier._classify_via_jira_project_key(commit) is None


def test_jira_project_key_handles_id_without_dash() -> None:
    """Malformed ticket id without '-' is skipped without raising."""
    classifier = _make_classifier({"ADV": "feature"})

    commit = _commit(
        "abc123",
        [{"platform": "jira", "id": "ADV1234"}],
    )

    assert classifier._classify_via_jira_project_key(commit) is None


# ---------------------------------------------------------------------------
# _merge_jira_with_llm_results
# ---------------------------------------------------------------------------


def test_merge_preserves_caller_order() -> None:
    """Merged results are returned in the original commit order."""
    classifier = _make_classifier({"ADV": "feature"})

    commits = [
        _commit("h1", [{"platform": "jira", "id": "ADV-1"}]),
        _commit("h2", []),
        _commit("h3", [{"platform": "jira", "id": "ADV-2"}]),
    ]
    jira_classified = {
        "h1": {"commit_hash": "h1", "category": "feature", "method": "jira_project_key"},
        "h3": {"commit_hash": "h3", "category": "feature", "method": "jira_project_key"},
    }
    llm_results = [{"commit_hash": "h2", "category": "bug_fix", "method": "llm"}]

    merged = classifier._merge_jira_with_llm_results(commits, jira_classified, llm_results)

    assert [r["commit_hash"] for r in merged] == ["h1", "h2", "h3"]
    assert [r["method"] for r in merged] == ["jira_project_key", "llm", "jira_project_key"]


# ---------------------------------------------------------------------------
# Full _classify_commit_batch_with_llm short-circuit (LLM disabled path)
# ---------------------------------------------------------------------------


def test_full_batch_short_circuit_skips_llm_entirely() -> None:
    """When every commit hits the JIRA project-key mapping, the LLM is never called."""
    classifier = _make_classifier({"ADV": "feature", "OPS": "maintenance"})
    # Force LLM disabled so test stays offline; the short-circuit path should
    # trigger before the fallback path is even consulted.
    classifier.llm_enabled = False

    commits = [
        _commit("h1", [{"platform": "jira", "id": "ADV-1"}], message="cleanup whitespace"),
        _commit("h2", [{"platform": "jira", "id": "OPS-9"}], message="implement new feature"),
    ]

    results = classifier._classify_commit_batch_with_llm(commits, ticket_context={})

    by_hash = {r["commit_hash"]: r for r in results}
    assert by_hash["h1"]["category"] == "feature"
    assert by_hash["h1"]["method"] == "jira_project_key"
    assert by_hash["h1"]["confidence"] == pytest.approx(0.95)
    # "implement new feature" message would normally be flagged "feature" by
    # the rule-based fallback, but here OPS → maintenance must win.
    assert by_hash["h2"]["category"] == "maintenance"
    assert by_hash["h2"]["method"] == "jira_project_key"


def test_partial_batch_falls_through_for_unmapped_commits() -> None:
    """Commits without a mapped JIRA key still go through the fallback path."""
    classifier = _make_classifier({"ADV": "feature"})
    classifier.llm_enabled = False  # Force fallback for the unmapped commit.

    commits = [
        _commit("h1", [{"platform": "jira", "id": "ADV-1"}], message="anything"),
        _commit("h2", [], message="fix: null pointer crash"),
    ]

    results = classifier._classify_commit_batch_with_llm(commits, ticket_context={})

    by_hash = {r["commit_hash"]: r for r in results}
    assert by_hash["h1"]["method"] == "jira_project_key"
    assert by_hash["h1"]["category"] == "feature"
    # Unmapped commit goes through the rule-based fallback.
    assert by_hash["h2"]["method"] == "fallback_only"
    assert by_hash["h2"]["category"] == "bug_fix"


def test_show_jira_signals_logs_matches(caplog: pytest.LogCaptureFixture) -> None:
    """``show_jira_signals=True`` emits an INFO log per short-circuited commit."""
    import logging

    classifier = _make_classifier({"ADV": "feature"}, show_jira_signals=True)
    classifier.llm_enabled = False

    commits = [
        _commit("abcdef0", [{"platform": "jira", "id": "ADV-42"}]),
    ]

    with caplog.at_level(
        logging.INFO, logger="gitflow_analytics.classification.batch_classifier_impl"
    ):
        classifier._classify_commit_batch_with_llm(commits, ticket_context={})

    assert any(
        "JIRA project-key signal" in r.message and "ADV" in r.message for r in caplog.records
    )
