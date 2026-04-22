"""Tests for GitHubIssuesIntegration.

Covers:
- Issue activity event construction (opened, closed)
- Comment activity events (fetch_comments flag)
- Actor lowercasing
- Incremental fetch date logic
- allowed_repos filtering
- Rate limit handling (403/429)
- Empty response handling
- Storage via cache.get_session
- get_activity_summary aggregation
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.integrations.github_issues_integration import (
    GitHubIssuesIntegration,
    _build_comment_event,
    _parse_github_datetime,
    _to_github_iso,
    _to_naive_utc,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cache() -> Any:
    """Create a temporary cache backed by a real SQLite database."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = GitAnalysisCache(Path(tmp))
        yield cache


@pytest.fixture
def integration(tmp_cache: GitAnalysisCache) -> GitHubIssuesIntegration:
    return GitHubIssuesIntegration(
        token="test-token",
        cache=tmp_cache,
        fetch_comments=False,
        rate_limit_retries=1,
        backoff_factor=0.01,
    )


def _make_issue(
    number: int = 1,
    user_login: str = "AliceDev",
    state: str = "open",
    created_at: str = "2024-01-01T10:00:00Z",
    closed_at: str | None = None,
    comments: int = 0,
    title: str = "Test issue",
    html_url: str = "https://github.com/org/repo/issues/1",
    labels: list[dict[str, Any]] | None = None,
    closed_by: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "state": state,
        "created_at": created_at,
        "closed_at": closed_at,
        "comments": comments,
        "html_url": html_url,
        "user": {"login": user_login, "email": None},
        "labels": labels or [],
        "assignees": [],
        "reactions": {"total_count": 0},
        "closed_by": closed_by,
    }


def _make_response(status_code: int = 200, payload: Any = None) -> Mock:
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = payload if payload is not None else []
    resp.text = str(payload)
    return resp


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


def test_parse_github_datetime_iso() -> None:
    dt = _parse_github_datetime("2024-01-15T09:00:00Z")
    assert dt == datetime(2024, 1, 15, 9, 0, 0)
    assert dt is not None
    assert dt.tzinfo is None


def test_parse_github_datetime_none() -> None:
    assert _parse_github_datetime(None) is None
    assert _parse_github_datetime("") is None


def test_parse_github_datetime_invalid() -> None:
    assert _parse_github_datetime("not-a-date") is None


def test_to_naive_utc_converts_aware() -> None:
    aware = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    naive = _to_naive_utc(aware)
    assert naive.tzinfo is None
    assert naive == datetime(2024, 1, 1, 12, 0, 0)


def test_to_github_iso_naive() -> None:
    d = datetime(2024, 1, 1, 9, 30, 0)
    assert _to_github_iso(d).endswith("Z")


def test_to_github_iso_aware() -> None:
    d = datetime(2024, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
    assert _to_github_iso(d) == "2024-01-01T09:30:00Z"


def test_build_comment_event_lowercases_actor() -> None:
    issue = _make_issue(number=5, title="X", html_url="https://x")
    comment = {
        "id": 123,
        "user": {"login": "BOBMatsuoka", "email": "bob@x"},
        "created_at": "2024-02-01T00:00:00Z",
        "html_url": "https://x/comments/123",
        "reactions": {"total_count": 2},
    }
    event = _build_comment_event("org/repo", issue, comment)
    assert event["actor"] == "bobmatsuoka"
    assert event["item_type"] == "comment"
    assert event["action"] == "commented"
    assert event["reaction_count"] == 2
    assert event["platform"] == "github_issues"


# ---------------------------------------------------------------------------
# Integration __init__ / config
# ---------------------------------------------------------------------------


def test_init_sets_fields(tmp_cache: GitAnalysisCache) -> None:
    integ = GitHubIssuesIntegration(
        token="abc",
        cache=tmp_cache,
        fetch_comments=True,
        allowed_repos=["a/b"],
        issue_state="closed",
        max_issues_per_repo=10,
    )
    assert integ.fetch_comments is True
    assert integ.allowed_repos == ["a/b"]
    assert integ.issue_state == "closed"
    assert integ.max_issues_per_repo == 10


def test_session_auth_header(integration: GitHubIssuesIntegration) -> None:
    assert "Authorization" in integration._session.headers
    auth_header = str(integration._session.headers["Authorization"])
    assert auth_header.startswith("token ")


# ---------------------------------------------------------------------------
# Event building
# ---------------------------------------------------------------------------


def test_build_issue_events_opened_only(integration: GitHubIssuesIntegration) -> None:
    issue = _make_issue(number=1, state="open", closed_at=None)
    events = integration._build_issue_events("org/repo", issue)
    assert len(events) == 1
    assert events[0]["action"] == "opened"
    assert events[0]["actor"] == "alicedev"  # lowercased
    assert events[0]["item_type"] == "issue"
    assert events[0]["repo_or_space"] == "org/repo"


def test_build_issue_events_opened_and_closed(integration: GitHubIssuesIntegration) -> None:
    issue = _make_issue(
        number=2,
        state="closed",
        closed_at="2024-01-05T00:00:00Z",
        closed_by={"login": "Closer", "email": "c@x"},
    )
    events = integration._build_issue_events("org/repo", issue)
    assert len(events) == 2
    actions = [e["action"] for e in events]
    assert "opened" in actions
    assert "closed" in actions
    # Closer's login should be lowercased
    close_event = next(e for e in events if e["action"] == "closed")
    assert close_event["actor"] == "closer"


def test_build_issue_events_handles_missing_user(
    integration: GitHubIssuesIntegration,
) -> None:
    issue = _make_issue()
    issue["user"] = None
    events = integration._build_issue_events("org/repo", issue)
    # Actor becomes None because empty string is normalized
    assert events[0]["actor"] is None


def test_build_issue_events_labels_captured(
    integration: GitHubIssuesIntegration,
) -> None:
    issue = _make_issue(labels=[{"name": "bug"}, {"name": "high-priority"}])
    events = integration._build_issue_events("org/repo", issue)
    assert events[0]["platform_data"]["labels"] == ["bug", "high-priority"]


# ---------------------------------------------------------------------------
# Fetch flow
# ---------------------------------------------------------------------------


def test_fetch_issues_activity_empty_repos(integration: GitHubIssuesIntegration) -> None:
    assert integration.fetch_issues_activity([], datetime(2024, 1, 1)) == []


def test_fetch_issues_activity_allowed_repos_filter(
    tmp_cache: GitAnalysisCache,
) -> None:
    integ = GitHubIssuesIntegration(
        token="x",
        cache=tmp_cache,
        allowed_repos=["only/this"],
    )
    with patch.object(integ, "_fetch_repo_issue_activity") as mock_fetch:
        mock_fetch.return_value = []
        integ.fetch_issues_activity(["other/repo"], datetime(2024, 1, 1))
        mock_fetch.assert_not_called()


def test_fetch_issues_activity_allowed_repos_passes(
    tmp_cache: GitAnalysisCache,
) -> None:
    integ = GitHubIssuesIntegration(
        token="x",
        cache=tmp_cache,
        allowed_repos=["org/repo"],
    )
    with patch.object(integ, "_fetch_repo_issue_activity") as mock_fetch:
        mock_fetch.return_value = []
        integ.fetch_issues_activity(["org/repo"], datetime(2024, 1, 1))
        mock_fetch.assert_called_once()


def test_fetch_repo_issue_activity_paginated(
    integration: GitHubIssuesIntegration,
) -> None:
    issues_page1 = [_make_issue(number=i) for i in range(1, 3)]
    # Two issues returned, < per_page=100, so loop terminates after page 1
    with patch.object(
        integration, "_get_with_retries", return_value=_make_response(payload=issues_page1)
    ):
        events = integration._fetch_repo_issue_activity(
            "org/repo", datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
    assert len(events) == 2  # 2 'opened' events


def test_fetch_repo_issue_activity_skips_prs(
    integration: GitHubIssuesIntegration,
) -> None:
    # Issues endpoint can return PRs too; they must be skipped
    pr_payload = _make_issue(number=99)
    pr_payload["pull_request"] = {"url": "..."}
    with patch.object(
        integration, "_get_with_retries", return_value=_make_response(payload=[pr_payload])
    ):
        events = integration._fetch_repo_issue_activity(
            "org/repo", datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
    assert events == []


def test_fetch_repo_issue_activity_fetches_comments(
    tmp_cache: GitAnalysisCache,
) -> None:
    integ = GitHubIssuesIntegration(
        token="x", cache=tmp_cache, fetch_comments=True, rate_limit_retries=0
    )
    issue = _make_issue(number=1, comments=1)
    comment = {
        "id": 999,
        "user": {"login": "commenter"},
        "created_at": "2024-01-02T00:00:00Z",
        "html_url": "https://x/comments/999",
        "reactions": {"total_count": 0},
    }

    # First call returns issues; subsequent comment-fetch calls get the comment list
    responses = [
        _make_response(payload=[issue]),
        _make_response(payload=[comment]),
    ]

    def side_effect(*args: Any, **kwargs: Any) -> Any:
        _ = (args, kwargs)
        return responses.pop(0) if responses else _make_response(payload=[])

    with patch.object(integ, "_get_with_retries", side_effect=side_effect):
        events = integ._fetch_repo_issue_activity(
            "org/repo", datetime(2024, 1, 1, tzinfo=timezone.utc)
        )

    actions = [e["action"] for e in events]
    assert "opened" in actions
    assert "commented" in actions


def test_fetch_repo_issue_activity_http_error(
    integration: GitHubIssuesIntegration,
) -> None:
    with patch.object(integration, "_get_with_retries", return_value=None):
        events = integration._fetch_repo_issue_activity(
            "org/repo", datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
    assert events == []


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_get_with_retries_rate_limited(integration: GitHubIssuesIntegration) -> None:
    bad = _make_response(status_code=429)
    good = _make_response(status_code=200, payload=[])
    integration.rate_limit_retries = 2
    mock_get = MagicMock(side_effect=[bad, good])
    with patch.object(integration._session, "get", mock_get), patch("time.sleep"):
        resp = integration._get_with_retries("https://api.example/x")
    assert resp is good
    assert mock_get.call_count == 2


def test_get_with_retries_403_rate_limit(integration: GitHubIssuesIntegration) -> None:
    bad = _make_response(status_code=403)
    good = _make_response(status_code=200, payload=[])
    integration.rate_limit_retries = 2
    mock_get = MagicMock(side_effect=[bad, good])
    with patch.object(integration._session, "get", mock_get), patch("time.sleep"):
        resp = integration._get_with_retries("https://api.example/x")
    assert resp is good


def test_get_with_retries_exhausted(integration: GitHubIssuesIntegration) -> None:
    bad = _make_response(status_code=429)
    integration.rate_limit_retries = 1
    mock_get = MagicMock(return_value=bad)
    with patch.object(integration._session, "get", mock_get), patch("time.sleep"):
        resp = integration._get_with_retries("https://api.example/x")
    assert resp is None


def test_get_with_retries_client_error(integration: GitHubIssuesIntegration) -> None:
    bad = _make_response(status_code=404)
    with patch.object(integration._session, "get", return_value=bad):
        resp = integration._get_with_retries("https://api.example/x")
    assert resp is None


# ---------------------------------------------------------------------------
# Incremental fetch logic
# ---------------------------------------------------------------------------


def test_get_effective_since_no_prior(integration: GitHubIssuesIntegration) -> None:
    with patch.object(integration.schema_manager, "get_last_processed_date", return_value=None):
        result = integration._get_effective_since(datetime(2024, 1, 1))
    assert result.tzinfo is not None


def test_get_effective_since_uses_later(integration: GitHubIssuesIntegration) -> None:
    last = datetime(2024, 3, 1, tzinfo=timezone.utc)
    requested = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with patch.object(integration.schema_manager, "get_last_processed_date", return_value=last):
        result = integration._get_effective_since(requested)
    assert result == last


def test_get_effective_since_does_not_go_backwards(
    integration: GitHubIssuesIntegration,
) -> None:
    last = datetime(2024, 1, 1, tzinfo=timezone.utc)
    requested = datetime(2024, 3, 1, tzinfo=timezone.utc)
    with patch.object(integration.schema_manager, "get_last_processed_date", return_value=last):
        result = integration._get_effective_since(requested)
    assert result == requested


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_store_and_summary_roundtrip(integration: GitHubIssuesIntegration) -> None:
    events = [
        {
            "platform": "github_issues",
            "item_id": "1",
            "item_type": "issue",
            "repo_or_space": "org/repo",
            "actor": "alice",
            "actor_display_name": "Alice",
            "actor_email": None,
            "action": "opened",
            "activity_at": datetime(2024, 2, 1, 10, 0, 0, tzinfo=timezone.utc),
            "item_title": "A",
            "item_status": "open",
            "item_url": "https://x",
            "linked_ticket_id": None,
            "comment_count": 0,
            "reaction_count": 0,
            "platform_data": {},
        },
        {
            "platform": "github_issues",
            "item_id": "1",
            "item_type": "comment",
            "repo_or_space": "org/repo",
            "actor": "bob",
            "actor_display_name": "Bob",
            "actor_email": None,
            "action": "commented",
            "activity_at": datetime(2024, 2, 2, 12, 0, 0, tzinfo=timezone.utc),
            "item_title": "A",
            "item_status": "open",
            "item_url": "https://x",
            "linked_ticket_id": None,
            "comment_count": 0,
            "reaction_count": 0,
            "platform_data": {},
        },
    ]
    integration._store_activity_events_bulk(events)

    summary = integration.get_activity_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 3, 1, tzinfo=timezone.utc),
    )
    assert summary["total_events"] == 2
    assert summary["per_repo"]["org/repo"] == 2
    assert summary["per_actor"]["alice"] == 1
    assert summary["per_actor"]["bob"] == 1


def test_store_skips_events_without_activity_at(
    integration: GitHubIssuesIntegration,
) -> None:
    events = [{"platform": "github_issues", "activity_at": None}]
    # Should not raise
    integration._store_activity_events_bulk(events)
