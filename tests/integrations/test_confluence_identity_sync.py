"""Tests for ConfluenceIdentitySync.

Covers:
- ``get_unresolved_confluence_actors`` returns only UUID-like actors
- Non-UUID actors (emails, GitHub logins) are skipped
- ``resolve_actor_to_email`` happy path + 404 handling
- ``sync`` rewrites ``actor`` column to canonical email
- ``sync`` populates empty ``actor_email`` column
- ``sync`` skips accountIds whose email has no canonical id
- ``/wiki`` suffix is stripped from base_url when building API URL
- Missing credentials → 0 (no-op)
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.integrations.confluence_identity_sync import (  # type: ignore[import-not-found]
    ConfluenceIdentitySync,
    _looks_like_account_id,
)
from gitflow_analytics.models.database_metrics_models import TicketingActivityCache

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cache() -> Any:
    with tempfile.TemporaryDirectory() as tmp:
        cache = GitAnalysisCache(Path(tmp))
        yield cache


@pytest.fixture
def sync_inst() -> ConfluenceIdentitySync:
    return ConfluenceIdentitySync(
        base_url="https://example.atlassian.net/wiki",
        username="admin@example.com",
        api_token="token",
        max_retries=2,
        backoff_factor=0.0,
    )


def _seed_cache(cache: GitAnalysisCache, rows: list[dict[str, Any]]) -> None:
    """Insert the given TicketingActivityCache rows into the cache DB."""
    with cache.get_session() as session:
        for r in rows:
            session.add(
                TicketingActivityCache(
                    platform=r.get("platform", "confluence"),
                    item_id=r.get("item_id", "p1"),
                    item_type=r.get("item_type", "page_edit"),
                    repo_or_space=r.get("repo_or_space", "SPACE"),
                    actor=r["actor"],
                    actor_email=r.get("actor_email"),
                    action=r.get("action", "edited"),
                    activity_at=r.get("activity_at", datetime(2024, 1, 1, tzinfo=timezone.utc)),
                )
            )
        session.commit()


def _make_response(status_code: int = 200, json_data: Any = None, text: str = "") -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.text = text
    return resp


# ---------------------------------------------------------------------------
# _looks_like_account_id
# ---------------------------------------------------------------------------


def test_looks_like_account_id_positive() -> None:
    assert _looks_like_account_id("712020:d2d0c7b6-a1b2-4c3d-9e8f-0123456789ab")
    assert _looks_like_account_id("557058:abcd1234")


def test_looks_like_account_id_negative() -> None:
    assert not _looks_like_account_id("")
    assert not _looks_like_account_id(None)
    assert not _looks_like_account_id("alice@example.com")
    assert not _looks_like_account_id("alice")  # No colon
    assert not _looks_like_account_id("  ")


# ---------------------------------------------------------------------------
# base_url stripping
# ---------------------------------------------------------------------------


def test_wiki_suffix_stripped_from_base_url() -> None:
    inst = ConfluenceIdentitySync(
        base_url="https://company.atlassian.net/wiki",
        username="u",
        api_token="t",
    )
    assert inst.base_url == "https://company.atlassian.net"


def test_base_url_without_wiki_preserved() -> None:
    inst = ConfluenceIdentitySync(
        base_url="https://company.atlassian.net/",
        username="u",
        api_token="t",
    )
    assert inst.base_url == "https://company.atlassian.net"


# ---------------------------------------------------------------------------
# get_unresolved_confluence_actors
# ---------------------------------------------------------------------------


def test_get_unresolved_returns_only_uuid_actors(
    sync_inst: ConfluenceIdentitySync, tmp_cache: GitAnalysisCache
) -> None:
    _seed_cache(
        tmp_cache,
        [
            {"actor": "712020:abc-def-123"},
            {"actor": "alice@example.com"},  # already resolved
            {"actor": "557058:ghi-jkl-456"},
            {"actor": "jdoe"},  # no colon — github-like login
        ],
    )
    unresolved = sync_inst.get_unresolved_confluence_actors(tmp_cache)
    assert unresolved == ["557058:ghi-jkl-456", "712020:abc-def-123"]


# ---------------------------------------------------------------------------
# resolve_actor_to_email
# ---------------------------------------------------------------------------


def test_resolve_actor_to_email_happy_path(
    sync_inst: ConfluenceIdentitySync,
) -> None:
    with patch.object(
        sync_inst._session,
        "get",
        return_value=_make_response(200, {"emailAddress": "Alice@Example.COM"}),
    ):
        email = sync_inst.resolve_actor_to_email("712020:xyz")
    assert email == "alice@example.com"


def test_resolve_actor_to_email_404_returns_none(
    sync_inst: ConfluenceIdentitySync,
) -> None:
    with patch.object(
        sync_inst._session,
        "get",
        return_value=_make_response(404, text="Not found"),
    ):
        email = sync_inst.resolve_actor_to_email("712020:missing")
    assert email is None


def test_resolve_actor_to_email_no_email_field(
    sync_inst: ConfluenceIdentitySync,
) -> None:
    """Atlassian can return a user record with no emailAddress (hidden)."""
    with patch.object(
        sync_inst._session,
        "get",
        return_value=_make_response(200, {"accountId": "712020:xyz"}),
    ):
        email = sync_inst.resolve_actor_to_email("712020:xyz")
    assert email is None


# ---------------------------------------------------------------------------
# sync — end-to-end
# ---------------------------------------------------------------------------


def test_sync_rewrites_actor_and_actor_email(
    sync_inst: ConfluenceIdentitySync, tmp_cache: GitAnalysisCache
) -> None:
    """After sync, Confluence rows have actor = canonical email."""
    uuid = "712020:alice-uuid"
    _seed_cache(
        tmp_cache,
        [
            {"actor": uuid, "item_id": "p1"},
            {"actor": uuid, "item_id": "p2"},
            {"actor": "712020:unknown-uuid", "item_id": "p3"},
            {"actor": "bob@example.com", "item_id": "p4"},  # should not change
        ],
    )

    resolver = MagicMock()

    def _find(email: str) -> str | None:
        return "canon-alice" if email == "alice@example.com" else None

    resolver.find_canonical_id_by_email.side_effect = _find

    def _get(url: str, params: Any = None, timeout: int = 30) -> MagicMock:
        assert "/rest/api/3/user" in url
        assert "/wiki" not in url  # Must target parent domain
        account_id = params["accountId"]
        if account_id == uuid:
            return _make_response(200, {"emailAddress": "alice@example.com"})
        if account_id == "712020:unknown-uuid":
            return _make_response(200, {"emailAddress": "ghost@example.com"})
        return _make_response(404)

    with patch.object(sync_inst._session, "get", side_effect=_get):
        resolved = sync_inst.sync(tmp_cache, resolver)

    assert resolved == 1

    # Verify DB state — must read attributes inside the session.
    with tmp_cache.get_session() as session:
        rows = session.query(TicketingActivityCache).order_by(TicketingActivityCache.item_id).all()
        by_item = {r.item_id: (r.actor, r.actor_email) for r in rows}

    assert by_item["p1"] == ("alice@example.com", "alice@example.com")
    assert by_item["p2"][0] == "alice@example.com"
    # Unknown-UUID actor wasn't re-keyed because email wasn't in identity db.
    assert by_item["p3"][0] == "712020:unknown-uuid"
    # Non-UUID actor untouched.
    assert by_item["p4"][0] == "bob@example.com"


def test_sync_skips_non_uuid_actors(
    sync_inst: ConfluenceIdentitySync, tmp_cache: GitAnalysisCache
) -> None:
    """Actors that are already emails or GitHub logins are never looked up."""
    _seed_cache(
        tmp_cache,
        [
            {"actor": "alice@example.com", "item_id": "p1"},
            {"actor": "bobhandle", "item_id": "p2"},
        ],
    )

    resolver = MagicMock()
    resolver.find_canonical_id_by_email.return_value = "canon-1"

    get_mock = MagicMock(return_value=_make_response(200, {"emailAddress": "x@x"}))
    with patch.object(sync_inst._session, "get", get_mock):
        resolved = sync_inst.sync(tmp_cache, resolver)

    assert resolved == 0
    get_mock.assert_not_called()


def test_sync_preserves_existing_actor_email(
    sync_inst: ConfluenceIdentitySync, tmp_cache: GitAnalysisCache
) -> None:
    """When actor_email is already set we do NOT overwrite it."""
    uuid = "712020:alice-uuid"
    _seed_cache(
        tmp_cache,
        [
            {
                "actor": uuid,
                "item_id": "p1",
                "actor_email": "old-alice@legacy.com",
            },
        ],
    )

    resolver = MagicMock()
    resolver.find_canonical_id_by_email.return_value = "canon-alice"

    with patch.object(
        sync_inst._session,
        "get",
        return_value=_make_response(200, {"emailAddress": "alice@example.com"}),
    ):
        resolved = sync_inst.sync(tmp_cache, resolver)

    assert resolved == 1
    with tmp_cache.get_session() as session:
        row = session.query(TicketingActivityCache).first()
        actor_value = row.actor
        actor_email_value = row.actor_email
    assert actor_value == "alice@example.com"
    # Already-populated actor_email is NOT overwritten.
    assert actor_email_value == "old-alice@legacy.com"


def test_sync_missing_credentials_is_noop(
    tmp_cache: GitAnalysisCache,
) -> None:
    inst = ConfluenceIdentitySync(base_url="", username="", api_token="")
    resolver = MagicMock()
    assert inst.sync(tmp_cache, resolver) == 0
    resolver.find_canonical_id_by_email.assert_not_called()


def test_sync_no_unresolved_actors(
    sync_inst: ConfluenceIdentitySync, tmp_cache: GitAnalysisCache
) -> None:
    """Empty cache → no API calls, 0 resolved."""
    resolver = MagicMock()
    get_mock = MagicMock()
    with patch.object(sync_inst._session, "get", get_mock):
        resolved = sync_inst.sync(tmp_cache, resolver)
    assert resolved == 0
    get_mock.assert_not_called()
