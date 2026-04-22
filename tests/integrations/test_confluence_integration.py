"""Tests for ConfluenceIntegration.

Covers:
- Session auth header construction (Basic auth)
- Page record extraction
- Activity event generation (page_create, page_edit)
- Author / last_editor extraction with lowercasing
- fetch_page_history flag behaviour
- Pagination via start/limit params
- Page upsert (by page_id)
- fetch_all_spaces iterates configured spaces
- Rate-limit handling (429)
"""

from __future__ import annotations

import base64
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.integrations.confluence_integration import (  # type: ignore[import-not-found]
    ConfluenceIntegration,
    _ensure_aware,
    _parse_confluence_datetime,
    _to_naive_utc,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cache() -> Any:
    with tempfile.TemporaryDirectory() as tmp:
        cache = GitAnalysisCache(Path(tmp))
        yield cache


@pytest.fixture
def integration(tmp_cache: GitAnalysisCache) -> ConfluenceIntegration:
    return ConfluenceIntegration(
        base_url="https://example.atlassian.net/wiki",
        username="user@x",
        api_token="token",
        cache=tmp_cache,
        spaces=["DOC"],
        fetch_page_history=False,
        max_retries=1,
        backoff_factor=0.01,
    )


def _make_page(
    page_id: str = "100",
    space_key: str = "DOC",
    title: str = "Page Title",
    version_number: int = 1,
    created_date: str = "2024-01-01T00:00:00.000Z",
    when: str = "2024-01-05T00:00:00.000Z",
    created_by_username: str = "Alice",
    last_editor_username: str = "Bob",
    labels: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": page_id,
        "type": "page",
        "status": "current",
        "title": title,
        "version": {
            "number": version_number,
            "when": when,
            "by": {
                "username": last_editor_username,
                "displayName": last_editor_username,
                "email": None,
            },
        },
        "history": {
            "createdDate": created_date,
            "createdBy": {
                "username": created_by_username,
                "displayName": created_by_username,
                "email": None,
            },
        },
        "metadata": {"labels": {"results": [{"name": n} for n in (labels or [])]}},
        "ancestors": [],
        "_links": {"webui": f"/spaces/{space_key}/pages/{page_id}"},
    }


def _make_response(status_code: int = 200, payload: Any = None) -> Mock:
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = payload if payload is not None else {"results": []}
    resp.text = str(payload)
    return resp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_parse_confluence_datetime_iso() -> None:
    dt = _parse_confluence_datetime("2024-01-01T00:00:00.000Z")
    assert dt is not None
    assert dt.tzinfo is None


def test_parse_confluence_datetime_none() -> None:
    assert _parse_confluence_datetime(None) is None


def test_parse_confluence_datetime_invalid() -> None:
    assert _parse_confluence_datetime("not-a-date") is None


def test_to_naive_utc() -> None:
    aware = datetime(2024, 1, 1, tzinfo=timezone.utc)
    naive = _to_naive_utc(aware)
    assert naive is not None
    assert naive.tzinfo is None


def test_ensure_aware_returns_aware() -> None:
    naive = datetime(2024, 1, 1)
    assert _ensure_aware(naive).tzinfo is not None


# ---------------------------------------------------------------------------
# Init + session
# ---------------------------------------------------------------------------


def test_auth_header_uses_basic(integration: ConfluenceIntegration) -> None:
    header = str(integration._session.headers["Authorization"])
    assert header.startswith("Basic ")
    decoded = base64.b64decode(header.split()[1]).decode()
    assert decoded == "user@x:token"


def test_base_url_trailing_slash_stripped(tmp_cache: GitAnalysisCache) -> None:
    integ = ConfluenceIntegration(
        base_url="https://x/wiki/", username="u", api_token="t", cache=tmp_cache
    )
    assert integ.base_url == "https://x/wiki"


def test_init_default_spaces(tmp_cache: GitAnalysisCache) -> None:
    integ = ConfluenceIntegration(
        base_url="https://x", username="u", api_token="t", cache=tmp_cache
    )
    assert integ.spaces == []


# ---------------------------------------------------------------------------
# Record / event building
# ---------------------------------------------------------------------------


def test_build_page_records_author_lowercased(
    integration: ConfluenceIntegration,
) -> None:
    page = _make_page(created_by_username="AliceUser", last_editor_username="BOB")
    row, _ = integration._build_page_records("DOC", page)
    assert row["author"] == "aliceuser"
    assert row["last_editor"] == "bob"


def test_build_page_records_single_event_no_history(
    integration: ConfluenceIntegration,
) -> None:
    # fetch_page_history defaults to False in fixture
    page = _make_page(version_number=3)
    _, events = integration._build_page_records("DOC", page)
    assert len(events) == 1
    assert events[0]["item_type"] == "page_edit"
    assert events[0]["action"] == "edited"


def test_build_page_records_initial_version_is_page_create(
    integration: ConfluenceIntegration,
) -> None:
    page = _make_page(version_number=1, when="2024-01-01T00:00:00.000Z")
    _, events = integration._build_page_records("DOC", page)
    assert len(events) == 1
    assert events[0]["item_type"] == "page_create"
    assert events[0]["action"] == "created"


def test_build_page_records_with_history(tmp_cache: GitAnalysisCache) -> None:
    integ = ConfluenceIntegration(
        base_url="https://x",
        username="u",
        api_token="t",
        cache=tmp_cache,
        fetch_page_history=True,
    )
    page = _make_page(version_number=3)
    _, events = integ._build_page_records("DOC", page)
    actions = [e["action"] for e in events]
    assert "created" in actions
    assert "edited" in actions


def test_build_page_records_labels_and_url(
    integration: ConfluenceIntegration,
) -> None:
    page = _make_page(labels=["how-to", "infra"])
    row, _ = integration._build_page_records("DOC", page)
    assert row["labels"] == ["how-to", "infra"]
    assert row["page_url"].endswith("/spaces/DOC/pages/100")


def test_build_page_records_space_key_preserved(
    integration: ConfluenceIntegration,
) -> None:
    page = _make_page()
    row, _ = integration._build_page_records("ENG", page)
    assert row["space_key"] == "ENG"


# ---------------------------------------------------------------------------
# Fetch flow
# ---------------------------------------------------------------------------


def test_fetch_space_activity_empty_space(integration: ConfluenceIntegration) -> None:
    assert integration.fetch_space_activity("", datetime(2024, 1, 1)) == []


def test_fetch_space_activity_no_results(
    integration: ConfluenceIntegration,
) -> None:
    with patch.object(
        integration,
        "_get_with_retries",
        return_value=_make_response(payload={"results": [], "size": 0}),
    ):
        events = integration.fetch_space_activity("DOC", datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert events == []


def test_fetch_space_activity_filters_before_since(
    integration: ConfluenceIntegration,
) -> None:
    page = _make_page(when="2023-01-01T00:00:00.000Z")
    with patch.object(
        integration,
        "_get_with_retries",
        return_value=_make_response(payload={"results": [page], "size": 1}),
    ):
        events = integration.fetch_space_activity("DOC", datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert events == []


def test_fetch_space_activity_returns_events(
    integration: ConfluenceIntegration,
) -> None:
    page = _make_page(when="2024-06-01T00:00:00.000Z")
    with patch.object(
        integration,
        "_get_with_retries",
        return_value=_make_response(payload={"results": [page], "size": 1}),
    ):
        events = integration.fetch_space_activity("DOC", datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert len(events) == 1


def test_fetch_space_activity_pagination(integration: ConfluenceIntegration) -> None:
    page = _make_page(when="2024-06-01T00:00:00.000Z")
    # Two pages of 50, then empty
    full_page = {"results": [page] * 50, "size": 50}
    partial_page = {"results": [page] * 5, "size": 5}

    responses = [
        _make_response(payload=full_page),
        _make_response(payload=partial_page),
    ]

    def side_effect(*args: Any, **kwargs: Any) -> Any:
        _ = (args, kwargs)
        return responses.pop(0) if responses else _make_response(payload={"results": []})

    with patch.object(integration, "_get_with_retries", side_effect=side_effect):
        events = integration.fetch_space_activity("DOC", datetime(2024, 1, 1, tzinfo=timezone.utc))
    # 50 + 5 = 55 events (one per page)
    assert len(events) == 55


def test_fetch_space_activity_http_error(
    integration: ConfluenceIntegration,
) -> None:
    with patch.object(integration, "_get_with_retries", return_value=None):
        events = integration.fetch_space_activity("DOC", datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert events == []


def test_fetch_all_spaces(integration: ConfluenceIntegration) -> None:
    with patch.object(integration, "fetch_space_activity", return_value=[{"x": 1}]):
        events = integration.fetch_all_spaces(datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert events == [{"x": 1}]


def test_fetch_all_spaces_handles_space_failure(tmp_cache: GitAnalysisCache) -> None:
    integ = ConfluenceIntegration(
        base_url="https://x",
        username="u",
        api_token="t",
        cache=tmp_cache,
        spaces=["A", "B"],
    )
    with patch.object(integ, "fetch_space_activity", side_effect=[Exception("boom"), []]):
        result = integ.fetch_all_spaces(datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert result == []


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_get_with_retries_429_then_ok(integration: ConfluenceIntegration) -> None:
    bad = _make_response(status_code=429)
    good = _make_response(status_code=200, payload={"results": []})
    mock_get = MagicMock(side_effect=[bad, good])
    with patch.object(integration._session, "get", mock_get), patch("time.sleep"):
        resp = integration._get_with_retries("https://x/api")
    assert resp is good


def test_get_with_retries_exhausts(integration: ConfluenceIntegration) -> None:
    bad = _make_response(status_code=429)
    mock_get = MagicMock(return_value=bad)
    with patch.object(integration._session, "get", mock_get), patch("time.sleep"):
        resp = integration._get_with_retries("https://x/api")
    assert resp is None


def test_get_with_retries_client_error(integration: ConfluenceIntegration) -> None:
    bad = _make_response(status_code=404)
    with patch.object(integration._session, "get", return_value=bad):
        resp = integration._get_with_retries("https://x/api")
    assert resp is None


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_upsert_pages_bulk_inserts(integration: ConfluenceIntegration) -> None:
    rows = [
        {
            "page_id": "42",
            "space_key": "DOC",
            "title": "First",
            "version": 1,
            "author": "alice",
            "author_email": None,
            "last_editor": "alice",
            "last_editor_email": None,
            "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "updated_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "labels": [],
            "ancestor_ids": [],
            "page_url": "https://x/42",
            "platform_data": {},
        }
    ]
    integration._upsert_pages_bulk(rows)

    from gitflow_analytics.models.database import ConfluencePageCache

    with integration.cache.get_session() as session:
        result = session.query(ConfluencePageCache).all()
        assert len(result) == 1
        assert result[0].page_id == "42"


def test_upsert_pages_bulk_updates_existing(
    integration: ConfluenceIntegration,
) -> None:
    initial = [
        {
            "page_id": "42",
            "space_key": "DOC",
            "title": "Old",
            "version": 1,
            "author": "alice",
            "last_editor": "alice",
            "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "updated_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "labels": [],
            "ancestor_ids": [],
            "page_url": "",
            "platform_data": {},
        }
    ]
    integration._upsert_pages_bulk(initial)
    updated = [
        {
            **initial[0],
            "title": "New",
            "version": 2,
            "last_editor": "bob",
            "updated_at": datetime(2024, 2, 1, tzinfo=timezone.utc),
        }
    ]
    integration._upsert_pages_bulk(updated)

    from gitflow_analytics.models.database import ConfluencePageCache

    with integration.cache.get_session() as session:
        rows = session.query(ConfluencePageCache).all()
        assert len(rows) == 1
        assert rows[0].title == "New"
        assert rows[0].version == 2
        assert rows[0].last_editor == "bob"


def test_store_activity_events_skips_missing_activity_at(
    integration: ConfluenceIntegration,
) -> None:
    integration._store_activity_events_bulk([{"activity_at": None}])  # no raise


def test_store_activity_events_persists(integration: ConfluenceIntegration) -> None:
    events = [
        {
            "platform": "confluence",
            "item_id": "1",
            "item_type": "page_edit",
            "repo_or_space": "DOC",
            "actor": "alice",
            "actor_display_name": "Alice",
            "actor_email": None,
            "action": "edited",
            "activity_at": datetime(2024, 1, 5, tzinfo=timezone.utc),
            "item_title": "T",
            "item_status": "current",
            "item_url": "https://x",
            "linked_ticket_id": None,
            "comment_count": 0,
            "reaction_count": 0,
            "platform_data": {},
        }
    ]
    integration._store_activity_events_bulk(events)

    from gitflow_analytics.models.database import TicketingActivityCache

    with integration.cache.get_session() as session:
        rows = (
            session.query(TicketingActivityCache)
            .filter(TicketingActivityCache.platform == "confluence")
            .all()
        )
        assert len(rows) == 1
        assert rows[0].actor == "alice"


# ---------------------------------------------------------------------------
# Issue #33: Env var expansion + pre-flight credential verification
# ---------------------------------------------------------------------------


def test_confluence_env_var_expansion_in_config(
    tmp_cache: GitAnalysisCache, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``api_token: "${CONFLUENCE_API_TOKEN}"`` is expanded via ``os.environ``.

    Regression test for issue #33: prior to the fix, a legitimate token in
    ``.env.local`` would still propagate through the loader, but any failure
    to pick it up would silently collapse to an empty string.  This test
    exercises the ``_process_confluence_config`` -> ``_resolve_env_var`` path
    directly.
    """
    from gitflow_analytics.config.loader_sections import ConfigLoaderSectionsMixin

    monkeypatch.setenv("CONFLUENCE_API_TOKEN", "tok-from-env-abc123")
    monkeypatch.setenv("CONFLUENCE_USER", "dev@example.com")

    cfg = ConfigLoaderSectionsMixin._process_confluence_config(
        {
            "enabled": True,
            "base_url": "https://example.atlassian.net/wiki",
            "username": "${CONFLUENCE_USER}",
            "api_token": "${CONFLUENCE_API_TOKEN}",
            "spaces": ["ENG"],
        }
    )

    assert cfg is not None
    assert cfg.api_token == "tok-from-env-abc123"
    assert cfg.username == "dev@example.com"
    assert cfg.enabled is True


def test_confluence_verify_credentials_raises_on_401(
    integration: ConfluenceIntegration,
) -> None:
    """``verify_credentials`` raises a clear ``RuntimeError`` on HTTP 401."""
    resp = Mock()
    resp.status_code = 401
    resp.text = "Unauthorized"

    with (
        patch.object(integration._session, "get", return_value=resp) as mock_get,
        pytest.raises(RuntimeError) as excinfo,
    ):
        integration.verify_credentials()

    # Called the space-list endpoint with limit=1
    assert mock_get.called
    args, kwargs = mock_get.call_args
    assert args[0].endswith("/rest/api/space")
    assert kwargs.get("params") == {"limit": 1}

    msg = str(excinfo.value)
    assert "Confluence authentication failed" in msg
    assert "base_url" in msg
    assert "api_token" in msg


def test_confluence_verify_credentials_succeeds_on_200(
    integration: ConfluenceIntegration,
) -> None:
    """``verify_credentials`` returns silently on a valid 200 response."""
    resp = Mock()
    resp.status_code = 200
    resp.text = "{}"

    with patch.object(integration._session, "get", return_value=resp):
        # Should not raise
        integration.verify_credentials()


def test_confluence_verify_credentials_empty_token_raises(
    tmp_cache: GitAnalysisCache,
) -> None:
    """Empty api_token triggers a pre-flight RuntimeError without HTTP call.

    This is the exact failure mode described in issue #33: when
    ``${CONFLUENCE_API_TOKEN}`` resolves to an empty string, the integration
    must surface a clear error rather than issuing a guaranteed-401 request.
    """
    integ = ConfluenceIntegration(
        base_url="https://example.atlassian.net/wiki",
        username="user@x",
        api_token="",  # simulates missing env var
        cache=tmp_cache,
    )
    with patch.object(integ._session, "get") as mock_get, pytest.raises(RuntimeError) as excinfo:
        integ.verify_credentials()
    # No HTTP request should be issued when creds are missing.
    mock_get.assert_not_called()
    assert "Confluence authentication failed" in str(excinfo.value)
