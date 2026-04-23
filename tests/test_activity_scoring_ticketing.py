"""Tests for ActivityScorer ticketing_weight integration (Feature 1, #35).

Covers:
- ticketing_weight=0 → identical score to pre-merger behaviour
- ticketing_weight>0 → score blends correctly with ticketing events
- Missing ticketing data → treat as 0, no crash
- Explicit ticketing_score passed via metrics bypasses DB lookup
- Config drives weights correctly
- Ticketing event key mapping
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gitflow_analytics.config.schema import ActivityScoringConfig
from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.metrics.activity_scoring import ActivityScorer
from gitflow_analytics.models.database import TicketingActivityCache


@pytest.fixture
def tmp_cache() -> Any:
    with tempfile.TemporaryDirectory() as tmp:
        cache = GitAnalysisCache(Path(tmp))
        yield cache


def _base_metrics() -> dict[str, Any]:
    return {
        "commits": 5,
        "prs_involved": 2,
        "lines_added": 100,
        "lines_removed": 20,
        "files_changed_count": 3,
        "complexity_delta": 0,
    }


def test_ticketing_weight_zero_matches_historical(tmp_cache: GitAnalysisCache) -> None:
    """With ticketing_weight=0, raw_score equals code-only score."""
    cfg_no_ticket = ActivityScoringConfig(
        commits_weight=0.25,
        prs_weight=0.30,
        code_impact_weight=0.30,
        complexity_weight=0.15,
        ticketing_weight=0.0,
    )
    scorer = ActivityScorer(config=cfg_no_ticket, cache=tmp_cache)
    result = scorer.calculate_activity_score(_base_metrics())
    # Ticketing score component recorded but not contributing to raw_score
    assert result["components"]["ticketing_score"] == 0.0
    # Re-calculate expected from pieces
    c = result["components"]
    expected = (
        c["commit_score"] * 0.25
        + c["pr_score"] * 0.30
        + c["code_impact_score"] * 0.30
        + c["complexity_score"] * 0.15
    )
    assert result["raw_score"] == pytest.approx(expected)


def test_ticketing_weight_positive_blends(tmp_cache: GitAnalysisCache) -> None:
    cfg = ActivityScoringConfig(ticketing_weight=0.15)
    scorer = ActivityScorer(config=cfg, cache=tmp_cache)
    metrics = _base_metrics()
    metrics["developer_id"] = "alice"
    metrics["ticketing_score"] = 10.0  # explicit injection

    result = scorer.calculate_activity_score(metrics)
    c = result["components"]
    expected_code = (
        c["commit_score"] * cfg.commits_weight
        + c["pr_score"] * cfg.prs_weight
        + c["code_impact_score"] * cfg.code_impact_weight
        + c["complexity_score"] * cfg.complexity_weight
    )
    expected_total = expected_code + 10.0 * cfg.ticketing_weight
    assert result["raw_score"] == pytest.approx(expected_total)
    assert c["ticketing_score"] == 10.0


def test_missing_ticketing_data_treated_as_zero(tmp_cache: GitAnalysisCache) -> None:
    """No cache rows and no explicit score → ticketing_score=0, no crash."""
    cfg = ActivityScoringConfig(ticketing_weight=0.15)
    scorer = ActivityScorer(config=cfg, cache=tmp_cache)
    metrics = _base_metrics()
    metrics["developer_id"] = "nonexistent_dev"

    result = scorer.calculate_activity_score(metrics)
    assert result["components"]["ticketing_score"] == 0.0
    # raw_score still computed
    assert result["raw_score"] >= 0


def test_no_config_uses_defaults(tmp_cache: GitAnalysisCache) -> None:
    _ = tmp_cache
    scorer = ActivityScorer()
    assert scorer.WEIGHTS["commits"] == 0.22
    assert scorer.WEIGHTS["ticketing"] == 0.15


def test_cache_reads_ticketing_events(tmp_cache: GitAnalysisCache) -> None:
    """Ticketing scores loaded from cache and mapped per-actor."""
    with tmp_cache.get_session() as session:
        session.add(
            TicketingActivityCache(
                platform="github_issues",
                item_id="1",
                item_type="issue",
                repo_or_space="org/repo",
                actor="alice",
                action="opened",
                activity_at=datetime(2024, 2, 1),
                comment_count=0,
                reaction_count=0,
                platform_data={},
            )
        )
        session.add(
            TicketingActivityCache(
                platform="jira",
                item_id="P-1",
                item_type="issue_closed",
                repo_or_space="P",
                actor="alice",
                action="closed",
                activity_at=datetime(2024, 2, 2),
                comment_count=0,
                reaction_count=0,
                platform_data={},
            )
        )
        session.commit()

    cfg = ActivityScoringConfig(ticketing_weight=0.15)
    scorer = ActivityScorer(
        config=cfg,
        cache=tmp_cache,
        since=datetime(2024, 1, 1, tzinfo=timezone.utc),
        until=datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    # issues_opened (1.0) + jira_issues_closed (2.0) = 3.0
    assert scorer.get_ticketing_score("alice") == pytest.approx(3.0)


def test_get_ticketing_score_zero_weight(tmp_cache: GitAnalysisCache) -> None:
    cfg = ActivityScoringConfig(ticketing_weight=0.0)
    scorer = ActivityScorer(config=cfg, cache=tmp_cache)
    assert scorer.get_ticketing_score("alice") == 0.0


def test_ticketing_event_key_mapping() -> None:
    assert (
        ActivityScorer._ticketing_event_key("github_issues", "issue", "opened") == "issues_opened"
    )
    assert (
        ActivityScorer._ticketing_event_key("github_issues", "issue", "closed") == "issues_closed"
    )
    assert (
        ActivityScorer._ticketing_event_key("github_issues", "comment", "commented")
        == "comments_posted"
    )
    assert (
        ActivityScorer._ticketing_event_key("confluence", "page_create", "created")
        == "pages_created"
    )
    assert (
        ActivityScorer._ticketing_event_key("confluence", "page_edit", "edited") == "pages_edited"
    )
    assert (
        ActivityScorer._ticketing_event_key("jira", "issue_created", "opened")
        == "jira_issues_opened"
    )
    assert (
        ActivityScorer._ticketing_event_key("jira", "issue_closed", "closed")
        == "jira_issues_closed"
    )
    assert (
        ActivityScorer._ticketing_event_key("jira", "comment", "commented")
        == "jira_comments_posted"
    )
    assert ActivityScorer._ticketing_event_key("unknown", "x", "y") is None


def test_actor_lowercased_in_lookup(tmp_cache: GitAnalysisCache) -> None:
    with tmp_cache.get_session() as session:
        session.add(
            TicketingActivityCache(
                platform="jira",
                item_id="P-1",
                item_type="issue_created",
                repo_or_space="P",
                actor="alice@example.com",
                action="opened",
                activity_at=datetime(2024, 2, 1),
                comment_count=0,
                reaction_count=0,
                platform_data={},
            )
        )
        session.commit()
    cfg = ActivityScoringConfig(ticketing_weight=0.15)
    scorer = ActivityScorer(
        config=cfg,
        cache=tmp_cache,
        since=datetime(2024, 1, 1, tzinfo=timezone.utc),
        until=datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    # Upper-case query should still match due to lowercasing
    assert scorer.get_ticketing_score("ALICE@example.com") == pytest.approx(1.5)


def test_no_cache_ticketing_score_zero() -> None:
    cfg = ActivityScoringConfig(ticketing_weight=0.15)
    scorer = ActivityScorer(config=cfg, cache=None)
    assert scorer.get_ticketing_score("alice") == 0.0


class _StubIdentityResolver:
    """Minimal stub mirroring ``DeveloperIdentityResolver.resolve_by_github_username``.

    Avoids the cost of spinning up a full ``DeveloperIdentityResolver`` (which
    requires a SQLite DB and pulls in the alias-resolution machinery) for
    tests that only need the canonical-id bridge.
    """

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = {k.lower(): v for k, v in mapping.items()}

    def resolve_by_github_username(self, github_username: str) -> str | None:
        if not github_username:
            return None
        return self._mapping.get(github_username.lower())


def test_ticketing_score_resolved_via_github_username(tmp_cache: GitAnalysisCache) -> None:
    """Regression test for #41 — ticketing_score looked up by canonical_id.

    When the developer-activity CSV passes ``developer_id=<canonical_id UUID>``
    into ``calculate_activity_score``, the ticketing lookup must still succeed
    even though ``TicketingActivityCache.actor`` is keyed by lowercase GitHub
    login.  The identity resolver bridges the two key spaces.
    """
    with tmp_cache.get_session() as session:
        session.add(
            TicketingActivityCache(
                platform="github_issues",
                item_id="1",
                item_type="issue",
                repo_or_space="org/repo",
                actor="bob-duetto",
                action="opened",
                activity_at=datetime(2024, 2, 1),
                comment_count=0,
                reaction_count=0,
                platform_data={},
            )
        )
        session.add(
            TicketingActivityCache(
                platform="github_issues",
                item_id="2",
                item_type="issue",
                repo_or_space="org/repo",
                actor="bob-duetto",
                action="closed",
                activity_at=datetime(2024, 2, 2),
                comment_count=0,
                reaction_count=0,
                platform_data={},
            )
        )
        session.commit()

    canonical_id = "11111111-2222-3333-4444-555555555555"
    resolver = _StubIdentityResolver({"bob-duetto": canonical_id})

    cfg = ActivityScoringConfig(ticketing_weight=0.15)
    scorer = ActivityScorer(
        config=cfg,
        cache=tmp_cache,
        since=datetime(2024, 1, 1, tzinfo=timezone.utc),
        until=datetime(2024, 12, 31, tzinfo=timezone.utc),
        identity_resolver=resolver,
    )

    # Direct lookup by canonical_id should succeed (1.0 opened + 1.0 closed)
    assert scorer.get_ticketing_score(canonical_id) == pytest.approx(2.0)

    # Original actor-key lookup must still work for backward compatibility
    assert scorer.get_ticketing_score("bob-duetto") == pytest.approx(2.0)

    # And calculate_activity_score should populate ticketing_score via the
    # developer_id → canonical_id path used by csv_reports_developer.py.
    metrics = _base_metrics()
    metrics["developer_id"] = canonical_id
    result = scorer.calculate_activity_score(metrics)
    assert result["components"]["ticketing_score"] == pytest.approx(2.0)


def test_ticketing_score_zero_when_no_resolver(tmp_cache: GitAnalysisCache) -> None:
    """Without an identity resolver, a canonical_id lookup gracefully returns 0.0.

    This documents the pre-fix behaviour that issue #41 reported: when the
    actor key space (GitHub logins) and the lookup key space (canonical_id
    UUIDs) are disjoint and no bridge is provided, the ticketing score is 0.0
    — no crash, just a silent miss.
    """
    with tmp_cache.get_session() as session:
        session.add(
            TicketingActivityCache(
                platform="github_issues",
                item_id="1",
                item_type="issue",
                repo_or_space="org/repo",
                actor="bob-duetto",
                action="opened",
                activity_at=datetime(2024, 2, 1),
                comment_count=0,
                reaction_count=0,
                platform_data={},
            )
        )
        session.commit()

    cfg = ActivityScoringConfig(ticketing_weight=0.15)
    scorer = ActivityScorer(
        config=cfg,
        cache=tmp_cache,
        since=datetime(2024, 1, 1, tzinfo=timezone.utc),
        until=datetime(2024, 12, 31, tzinfo=timezone.utc),
        # identity_resolver intentionally omitted
    )
    canonical_id = "11111111-2222-3333-4444-555555555555"
    assert scorer.get_ticketing_score(canonical_id) == 0.0


def test_config_loader_processes_activity_scoring() -> None:
    """Config loader creates ActivityScoringConfig with defaults when absent."""
    from gitflow_analytics.config.loader_sections import ConfigLoaderSectionsMixin

    cfg = ConfigLoaderSectionsMixin._process_activity_scoring_config({})
    assert isinstance(cfg, ActivityScoringConfig)
    assert cfg.ticketing_weight == 0.15

    cfg2 = ConfigLoaderSectionsMixin._process_activity_scoring_config(
        {"ticketing_weight": 0.0, "commits_weight": 0.5}
    )
    assert cfg2.ticketing_weight == 0.0
    assert cfg2.commits_weight == 0.5
