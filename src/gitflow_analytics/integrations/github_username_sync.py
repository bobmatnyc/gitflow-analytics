"""GitHub organization member sync for developer identities.

WHY (#45): Developers whose public GitHub profile has an email address that
matches an existing identity in ``developer_identities`` can have their
``github_username`` auto-populated.  This enables the ticketing score
lookup (which keys by github login) to find developers who were initially
created from a corporate email — avoiding a 0.0 ticketing score purely
because ``github_username`` was NULL.

The sync is purely additive: ``_set_github_username_if_missing`` never
overwrites an existing username.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class GitHubOrgSync:
    """Fetch GitHub organization members and back-fill ``github_username``."""

    COMPONENT_NAME = "github_org_sync"

    def __init__(
        self,
        token: str,
        org: str,
        base_url: str = "https://api.github.com",
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        connection_timeout: int = 30,
    ) -> None:
        """Initialize GitHub org sync.

        Args:
            token: GitHub personal access token with ``read:org`` scope.
            org: GitHub organization login (e.g. ``"acme"``).
            base_url: GitHub API base URL.  Override for GitHub Enterprise.
            max_retries: Retry attempts on 429/5xx.
            backoff_factor: Exponential backoff base (seconds).
            connection_timeout: HTTP timeout (seconds).
        """
        self.token = token
        self.org = org
        self.base_url = (base_url or "https://api.github.com").rstrip("/")
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.connection_timeout = connection_timeout

        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "GitFlow-Analytics-GitHubOrgSync/1.0",
        }
        self._session = self._build_session()

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------
    def _build_session(self) -> requests.Session:
        """Build resilient requests session."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(self.headers)
        return session

    def _get_with_retries(
        self, url: str, params: dict[str, Any] | None = None
    ) -> requests.Response | None:
        """GET with manual retry on 429/secondary-rate-limit responses."""
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.get(url, params=params, timeout=self.connection_timeout)
            except requests.RequestException as exc:
                logger.warning("GitHub org sync request error: %s", exc)
                return None

            if response.status_code == 429 or (
                response.status_code == 403 and "rate limit" in (response.text or "").lower()
            ):
                wait_time = self.backoff_factor * (2**attempt)
                logger.debug(
                    "GitHub org sync rate-limited (status %s), waiting %ss (attempt %d)",
                    response.status_code,
                    wait_time,
                    attempt + 1,
                )
                time.sleep(wait_time)
                continue

            if response.status_code == 404:
                logger.debug("GitHub org sync: 404 for %s", url)
                return None

            if response.status_code >= 400:
                logger.warning(
                    "GitHub org sync API error %s: %s",
                    response.status_code,
                    (response.text or "")[:200],
                )
                return None

            return response

        logger.warning("GitHub org sync retries exhausted for %s", url)
        return None

    # ------------------------------------------------------------------
    # Pagination helper
    # ------------------------------------------------------------------
    def _iter_org_members(self) -> list[dict[str, Any]]:
        """Fetch every organization member using Link-header pagination."""
        members: list[dict[str, Any]] = []
        url: str | None = f"{self.base_url}/orgs/{self.org}/members"
        params: dict[str, Any] | None = {"per_page": 100}

        while url:
            response = self._get_with_retries(url, params=params)
            if response is None:
                break
            try:
                page = response.json()
            except ValueError:
                logger.warning("GitHub org sync: failed to parse JSON page")
                break

            if not isinstance(page, list) or not page:
                break

            members.extend(page)

            # Parse Link header for next page; requests has a helper.
            next_link = response.links.get("next") if hasattr(response, "links") else None
            if next_link and "url" in next_link:
                url = next_link["url"]
                # Query params are already embedded in the "next" URL.
                params = None
            else:
                url = None

        return members

    def _get_user_email(self, login: str) -> str | None:
        """Fetch a user's public email from ``/users/{login}``."""
        if not login:
            return None
        response = self._get_with_retries(f"{self.base_url}/users/{login}")
        if response is None:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        email = payload.get("email") if isinstance(payload, dict) else None
        if not email or not isinstance(email, str):
            return None
        return email.lower().strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sync(self, identity_resolver: Any) -> int:
        """Populate ``github_username`` for developers discovered in the org.

        Args:
            identity_resolver: ``DeveloperIdentityResolver`` instance.

        Returns:
            Number of developers whose ``github_username`` was populated.
        """
        if not self.token or not self.org:
            return 0

        members = self._iter_org_members()
        if not members:
            return 0

        updated = 0
        for member in members:
            login = member.get("login") if isinstance(member, dict) else None
            if not isinstance(login, str) or not login:
                continue

            email = self._get_user_email(login)
            if not email:
                # User has no public email — can't map to a canonical identity.
                continue

            canonical_id = identity_resolver.find_canonical_id_by_email(email)
            if not canonical_id:
                continue

            # Check current username before the call so we can count updates.
            cached = identity_resolver._cache.get(canonical_id)
            had_username = isinstance(cached, dict) and bool(cached.get("github_username"))

            identity_resolver._set_github_username_if_missing(canonical_id, login)

            # Re-check cache: if the username is now populated and wasn't before,
            # count it as an update.  Falls back to True if cache wasn't a dict.
            if not had_username:
                new_cached = identity_resolver._cache.get(canonical_id)
                if isinstance(new_cached, dict) and new_cached.get("github_username"):
                    updated += 1
                elif not isinstance(new_cached, dict):
                    # Cache not populated yet for this id; assume update happened.
                    updated += 1

        return updated
