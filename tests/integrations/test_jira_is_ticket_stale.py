"""Tests for ``JIRAIntegration._is_ticket_stale`` (Fix 4).

Verifies that the deprecated ``datetime.utcnow()`` call has been replaced with
a timezone-aware ``datetime.now(timezone.utc)`` and that naive cached_at
values are normalized to UTC consistently.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.integrations.jira_integration import JIRAIntegration


@pytest.fixture
def tmp_cache():
    with tempfile.TemporaryDirectory() as tmp:
        yield GitAnalysisCache(Path(tmp))


def _make_integration(cache: GitAnalysisCache) -> JIRAIntegration:
    return JIRAIntegration(
        base_url="https://x.atlassian.net",
        username="u",
        api_token="t",
        cache=cache,
    )


def test_aware_recent_is_not_stale(tmp_cache: GitAnalysisCache) -> None:
    integ = _make_integration(tmp_cache)
    recent = datetime.now(timezone.utc) - timedelta(minutes=5)
    assert not integ._is_ticket_stale(recent)


def test_aware_old_is_stale(tmp_cache: GitAnalysisCache) -> None:
    integ = _make_integration(tmp_cache)
    # cache.ttl_hours defaults to 168 (7d) — pick something definitely older.
    old = datetime.now(timezone.utc) - timedelta(days=365)
    assert integ._is_ticket_stale(old)


def test_naive_cached_at_treated_as_utc_recent(tmp_cache: GitAnalysisCache) -> None:
    """Naive datetime (SQLite-style) recent value must NOT be considered stale."""
    integ = _make_integration(tmp_cache)
    naive_recent = (datetime.now(timezone.utc) - timedelta(minutes=5)).replace(tzinfo=None)
    assert not integ._is_ticket_stale(naive_recent)


def test_naive_cached_at_treated_as_utc_old(tmp_cache: GitAnalysisCache) -> None:
    """Naive datetime (SQLite-style) old value must be considered stale."""
    integ = _make_integration(tmp_cache)
    naive_old = (datetime.now(timezone.utc) - timedelta(days=365)).replace(tzinfo=None)
    assert integ._is_ticket_stale(naive_old)


def test_zero_ttl_means_never_stale(tmp_cache: GitAnalysisCache) -> None:
    integ = _make_integration(tmp_cache)
    integ.cache.ttl_hours = 0
    very_old = datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert not integ._is_ticket_stale(very_old)
