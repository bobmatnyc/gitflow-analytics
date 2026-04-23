"""Tests for GitHubOrgSync.

Covers:
- Org member fetch + per-user email lookup happy path
- ``_set_github_username_if_missing`` called with correct login
- Pagination via Link header (two pages)
- Graceful skip when user has no public email
- Retry on 429 (secondary rate limit)
- Skip when identity resolver does not know the email
- No-op when token/org missing
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from gitflow_analytics.integrations.github_username_sync import (
    GitHubOrgSync,  # type: ignore[import-not-found]
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    status_code: int = 200,
    json_data: Any = None,
    links: dict[str, dict[str, str]] | None = None,
    text: str = "",
) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.links = links or {}
    resp.text = text
    return resp


@pytest.fixture
def sync_inst() -> GitHubOrgSync:
    return GitHubOrgSync(
        token="ghp_test",
        org="acme",
        max_retries=2,
        backoff_factor=0.0,
    )


@pytest.fixture
def mock_resolver() -> MagicMock:
    resolver = MagicMock()
    resolver._cache = {}

    # find_canonical_id_by_email returns canonical_id for known emails.
    def _find(email: str) -> str | None:
        mapping = {
            "alice@example.com": "canon-alice",
            "bob@example.com": "canon-bob",
        }
        return mapping.get((email or "").lower())

    resolver.find_canonical_id_by_email.side_effect = _find

    # _set_github_username_if_missing updates the cache so the sync can
    # detect that an update happened.
    def _set(canonical_id: str, login: str) -> None:
        resolver._cache[canonical_id] = {"github_username": login.lower()}

    resolver._set_github_username_if_missing.side_effect = _set

    return resolver


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sync_happy_path_sets_username(sync_inst: GitHubOrgSync, mock_resolver: MagicMock) -> None:
    """Single page of members with public emails resolves to canonical IDs."""
    members_page = [
        {"login": "alice"},
        {"login": "bob"},
    ]
    user_alice = {"login": "alice", "email": "alice@example.com"}
    user_bob = {"login": "bob", "email": "bob@example.com"}

    def _get(url: str, _params: Any = None, _timeout: int = 30) -> MagicMock:
        if "/orgs/acme/members" in url:
            return _make_response(200, members_page)
        if url.endswith("/users/alice"):
            return _make_response(200, user_alice)
        if url.endswith("/users/bob"):
            return _make_response(200, user_bob)
        return _make_response(404)

    with patch.object(sync_inst._session, "get", side_effect=_get):
        updated = sync_inst.sync(mock_resolver)

    assert updated == 2
    calls = mock_resolver._set_github_username_if_missing.call_args_list
    logins_set = {c.args[1] for c in calls}
    assert logins_set == {"alice", "bob"}


def test_pagination_two_pages(sync_inst: GitHubOrgSync, mock_resolver: MagicMock) -> None:
    """Link header pagination fetches all pages."""
    page1 = [{"login": "alice"}]
    page2 = [{"login": "bob"}]

    next_url = "https://api.github.com/orgs/acme/members?page=2"

    def _get(url: str, _params: Any = None, _timeout: int = 30) -> MagicMock:
        if url.endswith("/orgs/acme/members") and _params and _params.get("per_page") == 100:
            return _make_response(200, page1, links={"next": {"url": next_url}})
        if url == next_url:
            return _make_response(200, page2)
        if url.endswith("/users/alice"):
            return _make_response(200, {"email": "alice@example.com"})
        if url.endswith("/users/bob"):
            return _make_response(200, {"email": "bob@example.com"})
        return _make_response(404)

    with patch.object(sync_inst._session, "get", side_effect=_get):
        updated = sync_inst.sync(mock_resolver)

    assert updated == 2


def test_skip_user_without_public_email(sync_inst: GitHubOrgSync, mock_resolver: MagicMock) -> None:
    """A user whose /users/{login} returns email=None is skipped."""
    members_page = [{"login": "alice"}, {"login": "ghost"}]

    def _get(url: str, _params: Any = None, _timeout: int = 30) -> MagicMock:
        if "/orgs/acme/members" in url:
            return _make_response(200, members_page)
        if url.endswith("/users/alice"):
            return _make_response(200, {"email": "alice@example.com"})
        if url.endswith("/users/ghost"):
            return _make_response(200, {"email": None})
        return _make_response(404)

    with patch.object(sync_inst._session, "get", side_effect=_get):
        updated = sync_inst.sync(mock_resolver)

    assert updated == 1
    mock_resolver._set_github_username_if_missing.assert_called_once()
    assert mock_resolver._set_github_username_if_missing.call_args.args[1] == "alice"


def test_retry_on_429(sync_inst: GitHubOrgSync, mock_resolver: MagicMock) -> None:
    """Sync retries on 429 rate-limit responses before succeeding."""
    members_page = [{"login": "alice"}]
    call_counter = {"members": 0, "user": 0}

    def _get(url: str, _params: Any = None, _timeout: int = 30) -> MagicMock:
        if "/orgs/acme/members" in url:
            call_counter["members"] += 1
            if call_counter["members"] == 1:
                return _make_response(429, text="rate limit")
            return _make_response(200, members_page)
        if url.endswith("/users/alice"):
            call_counter["user"] += 1
            return _make_response(200, {"email": "alice@example.com"})
        return _make_response(404)

    with patch.object(sync_inst._session, "get", side_effect=_get):
        updated = sync_inst.sync(mock_resolver)

    assert updated == 1
    assert call_counter["members"] >= 2  # first 429, then 200


def test_unknown_email_is_skipped(sync_inst: GitHubOrgSync, mock_resolver: MagicMock) -> None:
    """A user whose email is not in the identity resolver is skipped."""
    members_page = [{"login": "stranger"}]

    def _get(url: str, _params: Any = None, _timeout: int = 30) -> MagicMock:
        if "/orgs/acme/members" in url:
            return _make_response(200, members_page)
        if url.endswith("/users/stranger"):
            return _make_response(200, {"email": "stranger@elsewhere.com"})
        return _make_response(404)

    with patch.object(sync_inst._session, "get", side_effect=_get):
        updated = sync_inst.sync(mock_resolver)

    assert updated == 0
    mock_resolver._set_github_username_if_missing.assert_not_called()


def test_missing_token_is_noop(mock_resolver: MagicMock) -> None:
    """Empty token short-circuits to 0 updates."""
    inst = GitHubOrgSync(token="", org="acme")
    assert inst.sync(mock_resolver) == 0
    mock_resolver._set_github_username_if_missing.assert_not_called()


def test_empty_org_members_list(sync_inst: GitHubOrgSync, mock_resolver: MagicMock) -> None:
    """Empty org returns 0 and performs no identity updates."""
    with patch.object(
        sync_inst._session,
        "get",
        return_value=_make_response(200, []),
    ):
        updated = sync_inst.sync(mock_resolver)

    assert updated == 0
    mock_resolver._set_github_username_if_missing.assert_not_called()
