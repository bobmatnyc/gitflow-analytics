"""Tests for JIRAActivityIntegration.

Covers:
- Issue event construction (issue_created, issue_closed)
- Comment event construction
- Actor extraction via email then name fallback
- Pagination (multiple pages)
- JQL construction with project keys
- Date filtering (events outside window dropped)
- Retry on 429
- Empty project_keys no-op
- Incremental fetch COMPONENT_NAME = "jira_activity"
- Storage in ticketing_activity_cache with platform='jira'
- get_activity_summary aggregation
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.integrations.jira_activity_integration import (  # type: ignore[import-untyped]
    JIRAActivityIntegration,
    _build_comment_event,
    _extract_actor,
    _parse_jira_date,
    _project_key_from_issue_key,
    _to_naive_utc,
    _within,
)
from gitflow_analytics.models.database import TicketingActivityCache

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cache() -> Any:
    with tempfile.TemporaryDirectory() as tmp:
        cache = GitAnalysisCache(Path(tmp))
        yield cache


@pytest.fixture
def integration(tmp_cache: GitAnalysisCache) -> JIRAActivityIntegration:
    return JIRAActivityIntegration(
        base_url="https://example.atlassian.net",
        username="u@example.com",
        api_token="tok",
        cache=tmp_cache,
        max_retries=1,
        backoff_factor=0.01,
    )


def _make_issue(
    key: str = "PROJ-1",
    reporter: dict[str, Any] | None = None,
    assignee: dict[str, Any] | None = None,
    created: str = "2024-02-01T10:00:00.000+0000",
    resolutiondate: str | None = None,
    summary: str = "A summary",
    status: str = "Open",
) -> dict[str, Any]:
    return {
        "key": key,
        "fields": {
            "summary": summary,
            "status": {"name": status},
            "created": created,
            "updated": created,
            "resolutiondate": resolutiondate,
            "reporter": reporter
            or {
                "emailAddress": "Alice@example.com",
                "displayName": "Alice",
                "name": "alice",
            },
            "assignee": assignee,
        },
    }


def _make_response(status_code: int = 200, payload: Any = None) -> Mock:
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = payload if payload is not None else {}
    resp.text = str(payload or "")
    resp.content = b'{"x": 1}' if payload is not None else b""
    return resp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_extract_actor_email_preferred() -> None:
    actor = _extract_actor({"emailAddress": "Bob@X", "name": "bob", "accountId": "abc"})
    assert actor == "bob@x"


def test_extract_actor_name_fallback() -> None:
    assert _extract_actor({"name": "Carol"}) == "carol"


def test_extract_actor_account_id_fallback() -> None:
    assert _extract_actor({"accountId": "ACC123"}) == "acc123"


def test_extract_actor_empty() -> None:
    assert _extract_actor({}) is None
    assert _extract_actor({"emailAddress": None}) is None


def test_parse_jira_date_iso() -> None:
    dt = _parse_jira_date("2024-01-15T09:00:00.000+0000")
    assert dt is not None
    assert dt.tzinfo is None


def test_parse_jira_date_none() -> None:
    assert _parse_jira_date(None) is None
    assert _parse_jira_date("") is None


def test_project_key_extraction() -> None:
    assert _project_key_from_issue_key("PROJ-123") == "PROJ"
    assert _project_key_from_issue_key("") == ""


def test_to_naive_utc_converts() -> None:
    aware = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    naive = _to_naive_utc(aware)
    assert naive is not None and naive.tzinfo is None


def test_within_window() -> None:
    dt = datetime(2024, 2, 5)
    assert _within(dt, datetime(2024, 2, 1), datetime(2024, 2, 10))
    assert not _within(dt, datetime(2024, 3, 1), datetime(2024, 3, 10))


def test_component_name_is_jira_activity() -> None:
    assert JIRAActivityIntegration.COMPONENT_NAME == "jira_activity"


# ---------------------------------------------------------------------------
# Init / session
# ---------------------------------------------------------------------------


def test_init_sets_base_url_without_trailing_slash(tmp_cache: GitAnalysisCache) -> None:
    integ = JIRAActivityIntegration(
        base_url="https://x.atlassian.net/",
        username="u",
        api_token="t",
        cache=tmp_cache,
    )
    assert integ.base_url == "https://x.atlassian.net"


def test_session_has_basic_auth(integration: JIRAActivityIntegration) -> None:
    auth = str(integration._session.headers["Authorization"])
    assert auth.startswith("Basic ")


# ---------------------------------------------------------------------------
# Event building
# ---------------------------------------------------------------------------


def test_build_issue_events_created_only(integration: JIRAActivityIntegration) -> None:
    issue = _make_issue(created="2024-02-01T10:00:00.000+0000", resolutiondate=None)
    events = integration._build_issue_events(
        issue,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert len(events) == 1
    assert events[0]["item_type"] == "issue_created"
    assert events[0]["action"] == "opened"
    assert events[0]["actor"] == "alice@example.com"  # lowercased email
    assert events[0]["platform"] == "jira"
    assert events[0]["repo_or_space"] == "PROJ"
    assert events[0]["linked_ticket_id"] == "PROJ-1"


def test_build_issue_events_created_and_closed(
    integration: JIRAActivityIntegration,
) -> None:
    issue = _make_issue(
        created="2024-02-01T10:00:00.000+0000",
        resolutiondate="2024-02-03T10:00:00.000+0000",
        assignee={"emailAddress": "Bob@x", "displayName": "Bob"},
    )
    events = integration._build_issue_events(
        issue,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    types = [e["item_type"] for e in events]
    assert "issue_created" in types
    assert "issue_closed" in types
    close = next(e for e in events if e["item_type"] == "issue_closed")
    assert close["actor"] == "bob@x"


def test_build_issue_events_actor_email_missing_fallback_to_name(
    integration: JIRAActivityIntegration,
) -> None:
    issue = _make_issue(
        reporter={"name": "UsernameOnly", "displayName": "U"},
        created="2024-02-01T10:00:00.000+0000",
    )
    events = integration._build_issue_events(
        issue,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert events[0]["actor"] == "usernameonly"


def test_build_issue_events_respects_date_window(
    integration: JIRAActivityIntegration,
) -> None:
    issue = _make_issue(created="2023-01-01T10:00:00.000+0000", resolutiondate=None)
    events = integration._build_issue_events(
        issue,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert events == []


def test_build_comment_event() -> None:
    issue = _make_issue(key="PROJ-7")
    comment = {
        "id": "99",
        "created": "2024-03-01T00:00:00.000+0000",
        "author": {"emailAddress": "COMMENTER@x", "displayName": "C"},
        "body": "hi",
    }
    ev = _build_comment_event(issue, comment)
    assert ev is not None
    assert ev["item_type"] == "comment"
    assert ev["action"] == "commented"
    assert ev["actor"] == "commenter@x"
    assert ev["item_id"] == "PROJ-7"
    assert ev["repo_or_space"] == "PROJ"


def test_build_comment_event_missing_date() -> None:
    issue = _make_issue()
    comment = {"id": "1", "author": {"name": "x"}}
    assert _build_comment_event(issue, comment) is None


# ---------------------------------------------------------------------------
# Fetch flow
# ---------------------------------------------------------------------------


def test_fetch_project_activity_empty_project_keys(
    integration: JIRAActivityIntegration,
) -> None:
    events = integration.fetch_project_activity(
        [], datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 12, 31, tzinfo=timezone.utc)
    )
    assert events == []


def test_fetch_project_activity_missing_credentials(tmp_cache: GitAnalysisCache) -> None:
    integ = JIRAActivityIntegration(base_url="", username="", api_token="", cache=tmp_cache)
    assert (
        integ.fetch_project_activity(
            ["PROJ"],
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        == []
    )


def test_fetch_issues_jql_contains_project_keys(
    integration: JIRAActivityIntegration,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, json_body: dict[str, Any]) -> Any:
        captured["url"] = url
        captured["json_body"] = json_body
        return _make_response(payload={"issues": [], "isLast": True})

    with patch.object(integration, "_post_with_retries", side_effect=fake_post):
        integration._fetch_issues(["PROJ", "ENG"], datetime(2024, 1, 1, tzinfo=timezone.utc))
    jql = captured["json_body"]["jql"]
    assert '"PROJ"' in jql and '"ENG"' in jql
    assert "created >=" in jql or "updated >=" in jql


def test_fetch_issues_pagination(integration: JIRAActivityIntegration) -> None:
    page1 = {
        "issues": [_make_issue(key=f"PROJ-{i}") for i in range(1, 51)],
        "isLast": False,
        "nextPageToken": "token-page-2",
    }
    page2 = {
        "issues": [_make_issue(key=f"PROJ-{i}") for i in range(51, 61)],
        "isLast": True,
    }
    responses = [_make_response(payload=page1), _make_response(payload=page2)]
    with patch.object(integration, "_post_with_retries", side_effect=responses):
        issues = integration._fetch_issues(["PROJ"], datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert len(issues) == 60


def test_get_with_retries_retries_on_429(
    integration: JIRAActivityIntegration,
) -> None:
    responses = [_make_response(status_code=429), _make_response(payload={"x": 1})]
    with patch.object(integration._session, "get", side_effect=responses):
        resp = integration._get_with_retries("https://x")
    assert resp is not None
    assert resp.status_code == 200


def test_get_with_retries_exhausts(integration: JIRAActivityIntegration) -> None:
    responses = [_make_response(status_code=429)] * 10
    with patch.object(integration._session, "get", side_effect=responses):
        resp = integration._get_with_retries("https://x")
    assert resp is None


def test_fetch_project_activity_end_to_end(
    integration: JIRAActivityIntegration,
) -> None:
    issue = _make_issue(
        key="PROJ-1",
        created="2024-02-01T10:00:00.000+0000",
        resolutiondate="2024-02-03T10:00:00.000+0000",
        assignee={"emailAddress": "Bob@x"},
    )

    def fake_post_with_retries(url: str, json_body: Any = None) -> Any:
        # _fetch_issues uses POST /rest/api/3/search/jql
        return _make_response(payload={"issues": [issue], "isLast": True})

    def fake_get_with_retries(url: str, params: Any = None) -> Any:
        if "/comment" in url:
            return _make_response(
                payload={
                    "comments": [
                        {
                            "id": "1",
                            "created": "2024-02-02T00:00:00.000+0000",
                            "author": {"emailAddress": "X@y"},
                        }
                    ]
                }
            )
        return _make_response(payload={})

    with (
        patch.object(integration, "_post_with_retries", side_effect=fake_post_with_retries),
        patch.object(integration, "_get_with_retries", side_effect=fake_get_with_retries),
    ):
        events = integration.fetch_project_activity(
            ["PROJ"],
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        )

    types = sorted(e["item_type"] for e in events)
    assert types == ["comment", "issue_closed", "issue_created"]
    # Verify stored in cache with platform='jira'
    with integration.cache.get_session() as session:
        rows = (
            session.query(TicketingActivityCache)
            .filter(TicketingActivityCache.platform == "jira")
            .all()
        )
        assert len(rows) == 3


def test_fetch_project_activity_drops_out_of_window_comments(
    integration: JIRAActivityIntegration,
) -> None:
    issue = _make_issue(key="PROJ-1", created="2024-02-01T10:00:00.000+0000")

    def fake_post_with_retries(url: str, json_body: Any = None) -> Any:
        return _make_response(payload={"issues": [issue], "isLast": True})

    def fake_get_with_retries(url: str, params: Any = None) -> Any:
        if "/comment" in url:
            return _make_response(
                payload={
                    "comments": [
                        {
                            "id": "1",
                            "created": "2023-01-01T00:00:00.000+0000",  # before window
                            "author": {"emailAddress": "old@x"},
                        }
                    ]
                }
            )
        return _make_response(payload={})

    with (
        patch.object(integration, "_post_with_retries", side_effect=fake_post_with_retries),
        patch.object(integration, "_get_with_retries", side_effect=fake_get_with_retries),
    ):
        events = integration.fetch_project_activity(
            ["PROJ"],
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
    # Only issue_created, no comment
    types = [e["item_type"] for e in events]
    assert "comment" not in types
    assert "issue_created" in types


# ---------------------------------------------------------------------------
# Storage + summary
# ---------------------------------------------------------------------------


def test_store_activity_events_persists(integration: JIRAActivityIntegration) -> None:
    events = [
        {
            "platform": "jira",
            "item_id": "PROJ-1",
            "item_type": "issue_created",
            "repo_or_space": "PROJ",
            "actor": "alice@x",
            "action": "opened",
            "activity_at": datetime(2024, 2, 1),
        }
    ]
    integration._store_activity_events_bulk(events)
    with integration.cache.get_session() as session:
        rows = session.query(TicketingActivityCache).all()
        assert len(rows) == 1
        assert rows[0].platform == "jira"
        assert rows[0].item_id == "PROJ-1"


def test_get_activity_summary_aggregates(integration: JIRAActivityIntegration) -> None:
    events = [
        {
            "platform": "jira",
            "item_id": "PROJ-1",
            "item_type": "issue_created",
            "repo_or_space": "PROJ",
            "actor": "alice@x",
            "action": "opened",
            "activity_at": datetime(2024, 2, 1),
        },
        {
            "platform": "jira",
            "item_id": "PROJ-1",
            "item_type": "issue_closed",
            "repo_or_space": "PROJ",
            "actor": "bob@x",
            "action": "closed",
            "activity_at": datetime(2024, 2, 3),
        },
        {
            "platform": "jira",
            "item_id": "PROJ-1",
            "item_type": "comment",
            "repo_or_space": "PROJ",
            "actor": "alice@x",
            "action": "commented",
            "activity_at": datetime(2024, 2, 2),
        },
    ]
    integration._store_activity_events_bulk(events)
    summary = integration.get_activity_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert summary["total_events"] == 3
    assert summary["per_developer"]["alice@x"]["jira_issues_opened"] == 1
    assert summary["per_developer"]["alice@x"]["jira_comments_posted"] == 1
    assert summary["per_developer"]["bob@x"]["jira_issues_closed"] == 1
