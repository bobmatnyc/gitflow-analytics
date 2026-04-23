"""Confluence actor-UUID → canonical identity re-keying.

WHY (#45): On Atlassian Cloud, the Confluence REST API returns an
``accountId`` that looks like ``712020:d2d0c7b6-…`` for each actor.
``ticketing_activity_cache`` stores this raw UUID in the ``actor``
column, but the ticketing-score lookup keys by canonical email (or
GitHub login).  The result: developers whose Confluence activity is
indexed by UUID never match their own commits, and their
``ticketing_score`` is 0.

This sync queries the Atlassian User API for each distinct UUID actor,
resolves the accountId to an email, looks that email up in the identity
resolver, and — when a canonical identity is found — rewrites the
``actor`` column on affected ``ticketing_activity_cache`` rows to the
canonical email.  The ``actor_email`` column is also populated so the
resolution is preserved for future runs.

The Atlassian User API lives at the parent domain (``/rest/api/3/user``),
NOT at ``/wiki/…`` — so we strip the ``/wiki`` suffix from
``confluence.base_url`` when building the request URL.
"""

from __future__ import annotations

import base64
import logging
import re
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Atlassian accountId pattern: "<realm>:<uuid>" or bare UUID.  We simply
# require that the string contain a ":" and NOT be an email address.
_UUID_CHARS = re.compile(r"^[A-Za-z0-9:\-]+$")


def _looks_like_account_id(actor: str | None) -> bool:
    """Heuristic: actor is an Atlassian accountId, not an email/username."""
    if not actor:
        return False
    actor = actor.strip()
    if not actor:
        return False
    if "@" in actor:
        return False  # Email address — already resolved.
    if ":" not in actor:
        return False  # Confluence Cloud accountIds always contain ":".
    return bool(_UUID_CHARS.match(actor))


class ConfluenceIdentitySync:
    """Resolve raw Confluence UUID actors into canonical identities."""

    COMPONENT_NAME = "confluence_identity_sync"

    def __init__(
        self,
        base_url: str,
        username: str,
        api_token: str,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        connection_timeout: int = 30,
    ) -> None:
        """Initialize Confluence identity sync.

        Args:
            base_url: Confluence base URL (``https://co.atlassian.net/wiki``
                or the parent domain).  ``/wiki`` suffix is stripped so the
                Atlassian User API URL is built correctly.
            username: Atlassian account username / email.
            api_token: Atlassian API token.
            max_retries: Retry attempts on 429/5xx.
            backoff_factor: Exponential backoff base (seconds).
            connection_timeout: HTTP timeout (seconds).
        """
        # Strip trailing slash, then strip "/wiki" if present.
        cleaned = (base_url or "").rstrip("/")
        if cleaned.endswith("/wiki"):
            cleaned = cleaned[: -len("/wiki")]
        self.base_url = cleaned
        self.username = username
        self.api_token = api_token
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.connection_timeout = connection_timeout

        credentials = base64.b64encode(f"{username}:{api_token}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {credentials}",
            "Accept": "application/json",
            "User-Agent": "GitFlow-Analytics-ConfluenceIdentitySync/1.0",
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
        """GET with manual retry on 429/503."""
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.get(url, params=params, timeout=self.connection_timeout)
            except requests.RequestException as exc:
                logger.warning("Confluence identity sync request error: %s", exc)
                return None

            if response.status_code in (429, 503):
                wait_time = self.backoff_factor * (2**attempt)
                logger.debug(
                    "Confluence identity sync rate-limited (status %s), "
                    "waiting %ss (attempt %d)",
                    response.status_code,
                    wait_time,
                    attempt + 1,
                )
                time.sleep(wait_time)
                continue

            if response.status_code == 404:
                # Account not found — caller handles None as a skip signal.
                return None

            if response.status_code >= 400:
                logger.warning(
                    "Confluence identity sync API error %s: %s",
                    response.status_code,
                    (response.text or "")[:200],
                )
                return None

            return response

        logger.warning("Confluence identity sync retries exhausted for %s", url)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_unresolved_confluence_actors(self, cache: Any) -> list[str]:
        """Return distinct Confluence actors that look like unresolved UUIDs.

        Args:
            cache: ``GitAnalysisCache`` instance.

        Returns:
            List of distinct accountId strings from ``ticketing_activity_cache``
            where ``platform='confluence'`` and the actor value looks like a
            raw Atlassian accountId (contains ``":"`` and no ``"@"``).
        """
        from ..models.database_metrics_models import TicketingActivityCache

        actors: set[str] = set()
        with cache.get_session() as session:
            rows = (
                session.query(TicketingActivityCache.actor)
                .filter(TicketingActivityCache.platform == "confluence")
                .distinct()
                .all()
            )
            for (actor,) in rows:
                if _looks_like_account_id(actor):
                    actors.add(actor)
        return sorted(actors)

    def resolve_actor_to_email(self, account_id: str) -> str | None:
        """Resolve an Atlassian accountId to an email address.

        Args:
            account_id: Atlassian accountId (e.g. ``"712020:d2d0…"``).

        Returns:
            Lowercased email address, or ``None`` if the user has no email
            or the lookup failed.
        """
        if not account_id or not self.base_url:
            return None
        url = f"{self.base_url}/rest/api/3/user"
        response = self._get_with_retries(url, params={"accountId": account_id})
        if response is None:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        email = payload.get("emailAddress")
        if not isinstance(email, str) or not email.strip():
            return None
        return email.lower().strip()

    def _update_actor_rows(self, cache: Any, account_id: str, canonical_email: str) -> int:
        """Rewrite ``actor`` column on all rows matching the accountId.

        Also populates ``actor_email`` when empty so future scoring lookups
        have the resolved address without rerunning the sync.

        Returns:
            Number of rows updated.
        """
        from ..models.database_metrics_models import TicketingActivityCache

        with cache.get_session() as session:
            rows = (
                session.query(TicketingActivityCache)
                .filter(
                    TicketingActivityCache.platform == "confluence",
                    TicketingActivityCache.actor == account_id,
                )
                .all()
            )
            count = 0
            for row in rows:
                row.actor = canonical_email
                if not row.actor_email:
                    row.actor_email = canonical_email
                count += 1
        return count

    def sync(self, cache: Any, identity_resolver: Any) -> int:
        """Resolve all unresolved Confluence UUID actors.

        Args:
            cache: ``GitAnalysisCache`` instance.
            identity_resolver: ``DeveloperIdentityResolver`` instance.

        Returns:
            Number of distinct accountIds that were successfully re-keyed.
            (Not the number of rows updated, which is reported via logs.)
        """
        if not self.base_url or not self.username or not self.api_token:
            return 0

        unresolved = self.get_unresolved_confluence_actors(cache)
        if not unresolved:
            return 0

        resolved_count = 0
        for account_id in unresolved:
            email = self.resolve_actor_to_email(account_id)
            if not email:
                continue

            canonical_id = identity_resolver.find_canonical_id_by_email(email)
            if not canonical_id:
                # User exists in Confluence but not in our developer db — skip.
                continue

            rows_updated = self._update_actor_rows(cache, account_id, email)
            if rows_updated:
                logger.debug(
                    "Confluence identity sync: re-keyed %d rows for %s → %s",
                    rows_updated,
                    account_id,
                    email,
                )
                resolved_count += 1

        return resolved_count
