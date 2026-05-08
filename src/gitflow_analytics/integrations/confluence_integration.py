"""Confluence Cloud REST API integration for page activity tracking.

WHY: Confluence page creation/editing activity is a productivity signal for
developers working on documentation.  Activity events are stored in
``ticketing_activity_cache`` (platform='confluence') and the latest page
metadata is upserted into ``confluence_page_cache`` (unique by page_id).

Actor usernames are always lowercased to match identity resolution conventions.
"""

from __future__ import annotations

import base64
import time
from datetime import datetime, timezone
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..core.cache import GitAnalysisCache
from ..core.schema_version import create_schema_manager
from ..utils.debug import is_debug_mode


class ConfluenceIntegration:
    """Integrate with Confluence Cloud REST API for page activity tracking."""

    COMPONENT_NAME = "confluence"

    def __init__(
        self,
        base_url: str,
        username: str,
        api_token: str,
        cache: GitAnalysisCache,
        spaces: list[str] | None = None,
        fetch_page_history: bool = False,
        dns_timeout: int = 10,
        connection_timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        full_scan_ttl_hours: float = 1.0,
    ):
        """Initialize Confluence integration.

        Args:
            base_url: Confluence instance base URL
                (e.g. ``https://company.atlassian.net/wiki``).
            username: Atlassian account username / email.
            api_token: Atlassian API token.
            cache: Shared :class:`GitAnalysisCache` instance.
            spaces: List of space keys to scan (empty = none).
            fetch_page_history: When True, emit a ``page_edit`` event per version
                (requires additional API calls); when False, emit one event per
                page at its ``updated_at`` timestamp.
            dns_timeout: DNS resolution timeout (seconds).
            connection_timeout: HTTP timeout (seconds).
            max_retries: Retry attempts on 429/5xx.
            backoff_factor: Exponential backoff base (seconds).
            full_scan_ttl_hours: Skip the per-space content scan entirely if
                the last successful scan completed within this many hours.
                Avoids re-paginating every page on back-to-back runs.
                Defaults to 1 hour.
        """
        self.base_url = (base_url or "").rstrip("/")
        self.username = username
        self.api_token = api_token
        self.cache = cache
        self.spaces = spaces or []
        self.fetch_page_history = fetch_page_history
        self.dns_timeout = dns_timeout
        self.connection_timeout = connection_timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.full_scan_ttl_hours = full_scan_ttl_hours
        self.debug_mode = is_debug_mode()

        self.schema_manager = create_schema_manager(cache.cache_dir)

        credentials = base64.b64encode(f"{username}:{api_token}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {credentials}",
            "Accept": "application/json",
            "User-Agent": "GitFlow-Analytics-Confluence/1.0",
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
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(self.headers)
        return session

    # ------------------------------------------------------------------
    # Pre-flight credential check
    # ------------------------------------------------------------------
    def verify_credentials(self) -> None:
        """Perform a lightweight authenticated probe against Confluence.

        Issues a single ``GET {base_url}/rest/api/space?limit=1`` request and
        raises :class:`RuntimeError` with a descriptive message on any
        authentication failure.  This surfaces credential problems during
        orchestrator initialization instead of during the first
        ``fetch_all_spaces`` call, where failures would previously be
        swallowed and silently produce an empty report (see issue #33).

        Raises:
            RuntimeError: If credentials appear invalid (missing creds, HTTP
                401/403, or other request-level failure).
        """
        # Fail fast on obviously-missing credentials rather than issuing a
        # guaranteed-401 request.
        if not self.base_url:
            raise RuntimeError(
                "Confluence authentication failed: base_url is empty. "
                "Check your config's 'confluence.base_url'."
            )
        if not self.username or not self.api_token:
            raise RuntimeError(
                "Confluence authentication failed: check base_url, username, "
                "and api_token. The username or api_token resolved to an empty "
                "string — most likely the ${CONFLUENCE_API_TOKEN} environment "
                "variable is not set. Verify your .env / .env.local file is "
                "loaded before `gfa analyze` runs."
            )

        url = f"{self.base_url}/rest/api/space"
        try:
            response = self._session.get(
                url,
                params={"limit": 1},
                timeout=self.connection_timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Confluence authentication failed: request error contacting "
                f"{url}: {exc}. Check base_url and network connectivity."
            ) from exc

        if response.status_code in (401, 403):
            raise RuntimeError(
                "Confluence authentication failed: check base_url, username, "
                f"and api_token (HTTP {response.status_code} from {url})."
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Confluence pre-flight check failed with HTTP "
                f"{response.status_code} from {url}: {response.text[:200]}"
            )

    # ------------------------------------------------------------------
    # Incremental fetch helpers
    # ------------------------------------------------------------------
    def _get_effective_since(self, requested_since: datetime) -> datetime:
        """Compute incremental fetch start date via schema-version manager.

        WHY: Mirror the pattern from JIRAActivityIntegration._get_effective_since.
        Without this, every run re-scans the full content listing from the user-
        supplied ``since`` even when the last successful scan completed minutes
        ago.  Anchoring to ``last_processed_date`` lets us skip pages that have
        not been updated since.
        """
        if requested_since.tzinfo is None:
            requested_since = requested_since.replace(tzinfo=timezone.utc)

        try:
            last_processed = self.schema_manager.get_last_processed_date(self.COMPONENT_NAME)
        except Exception:  # noqa: BLE001
            last_processed = None

        if last_processed is None:
            return requested_since

        if last_processed.tzinfo is None:
            last_processed = last_processed.replace(tzinfo=timezone.utc)

        return max(last_processed, requested_since)

    def _is_within_full_scan_ttl(self) -> bool:
        """True if last successful scan is newer than the full-scan TTL window.

        WHY: The Confluence v1 REST content endpoint does not accept a
        ``lastModified`` filter directly, so we fall back to skipping the
        full-space scan entirely when our checkpoint is fresh.  This is a
        coarse-grained but cheap optimization for back-to-back runs.
        """
        if self.full_scan_ttl_hours <= 0:
            return False

        try:
            last_processed = self.schema_manager.get_last_processed_date(self.COMPONENT_NAME)
        except Exception:  # noqa: BLE001
            return False

        if last_processed is None:
            return False
        if last_processed.tzinfo is None:
            last_processed = last_processed.replace(tzinfo=timezone.utc)

        from datetime import timedelta

        ttl_threshold = datetime.now(timezone.utc) - timedelta(hours=self.full_scan_ttl_hours)
        return last_processed >= ttl_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_space_activity(self, space_key: str, since: datetime) -> list[dict[str, Any]]:
        """Fetch pages updated since ``since`` for a single space.

        Args:
            space_key: Confluence space key.
            since: Lower bound on page ``updated_at``.

        Returns:
            List of activity event dicts (written to cache before returning).
        """
        if not space_key:
            return []

        since = _ensure_aware(since)

        # Fix 3: Skip the entire content scan if the last successful run was
        # within the TTL window.  Confluence v1 REST does not support a
        # ``lastModified`` filter, so without this guard every run re-paginates
        # the full space listing.
        if self._is_within_full_scan_ttl():
            if self.debug_mode:
                print(
                    f"   ⏭️  Confluence: skipping space={space_key} scan "
                    f"(last scan within {self.full_scan_ttl_hours}h TTL)"
                )
            return []

        # Fix 3: Anchor the lower bound to max(last_processed, requested_since)
        # so re-runs do not re-fetch pages that have not changed.
        effective_since = self._get_effective_since(since)
        if self.debug_mode:
            print(
                f"   🔍 Confluence: fetching space={space_key} since={effective_since.isoformat()}"
            )
        since = effective_since

        events: list[dict[str, Any]] = []
        pages_to_upsert: list[dict[str, Any]] = []

        start = 0
        limit = 50
        url = f"{self.base_url}/rest/api/content"

        while True:
            params = {
                "spaceKey": space_key,
                "expand": "version,history,metadata.labels,ancestors",
                "limit": limit,
                "start": start,
            }
            response = self._get_with_retries(url, params=params)
            if response is None:
                break
            data = response.json()
            results = data.get("results", []) if isinstance(data, dict) else []
            if not results:
                break

            for page in results:
                # Filter by updated_at
                version = page.get("version") or {}
                updated_raw = version.get("when") or (page.get("history") or {}).get("createdDate")
                updated_dt = _parse_confluence_datetime(updated_raw)
                if updated_dt is None:
                    continue
                since_naive = _to_naive_utc(since)
                if since_naive is not None and updated_dt < since_naive:
                    continue

                page_row, page_events = self._build_page_records(space_key, page)
                pages_to_upsert.append(page_row)
                events.extend(page_events)

            # Pagination
            size = data.get("size", len(results)) if isinstance(data, dict) else len(results)
            if size < limit:
                break
            start += limit
            time.sleep(0.1)

        # Persist
        if pages_to_upsert:
            self._upsert_pages_bulk(pages_to_upsert)
        if events:
            self._store_activity_events_bulk(events)
            print(f"   💾 Cached {len(events)} Confluence activity event(s) from space={space_key}")

        try:
            self.schema_manager.mark_date_processed(
                self.COMPONENT_NAME,
                datetime.now(timezone.utc),
                {"fetch_page_history": self.fetch_page_history},
            )
        except Exception as exc:  # noqa: BLE001
            if self.debug_mode:
                print(f"   🔍 Confluence schema_manager mark failed: {exc}")

        return events

    def fetch_all_spaces(self, since: datetime) -> list[dict[str, Any]]:
        """Fetch activity for every configured space."""
        all_events: list[dict[str, Any]] = []
        for space in self.spaces:
            try:
                all_events.extend(self.fetch_space_activity(space, since))
            except Exception as exc:  # noqa: BLE001
                print(f"   ⚠️  Confluence fetch failed for space={space}: {exc}")
        return all_events

    # ------------------------------------------------------------------
    # Record building
    # ------------------------------------------------------------------
    def _build_page_records(
        self, space_key: str, page: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Extract page-cache row + activity events from a single page payload."""
        history = page.get("history") or {}
        version = page.get("version") or {}
        created_by = history.get("createdBy") or {}
        last_editor = version.get("by") or created_by

        author = (created_by.get("username") or created_by.get("accountId") or "").lower()
        editor = (last_editor.get("username") or last_editor.get("accountId") or "").lower()

        author_display = created_by.get("displayName") or created_by.get("publicName")
        editor_display = last_editor.get("displayName") or last_editor.get("publicName")

        author_email = created_by.get("email")
        editor_email = last_editor.get("email")

        created_at = _parse_confluence_datetime(history.get("createdDate"))
        updated_at = _parse_confluence_datetime(version.get("when"))

        labels = [
            lbl.get("name")
            for lbl in ((page.get("metadata") or {}).get("labels") or {}).get("results", [])
            if lbl and lbl.get("name")
        ]
        ancestor_ids = [a.get("id") for a in (page.get("ancestors") or []) if a]

        page_id = str(page.get("id") or "")
        page_url_raw = (page.get("_links") or {}).get("webui") or ""
        page_url = f"{self.base_url}{page_url_raw}" if page_url_raw else ""

        page_row = {
            "page_id": page_id,
            "space_key": space_key,
            "title": page.get("title") or "",
            "version": int(version.get("number") or 0),
            "author": author or None,
            "author_email": author_email,
            "last_editor": editor or None,
            "last_editor_email": editor_email,
            "created_at": created_at,
            "updated_at": updated_at,
            "labels": labels,
            "ancestor_ids": ancestor_ids,
            "page_url": page_url,
            "platform_data": {
                "type": page.get("type"),
                "status": page.get("status"),
            },
        }

        events: list[dict[str, Any]] = []

        base_event = {
            "platform": "confluence",
            "item_id": page_id,
            "repo_or_space": space_key,
            "item_title": page.get("title") or "",
            "item_status": page.get("status") or "",
            "item_url": page_url,
            "linked_ticket_id": None,
            "comment_count": 0,
            "reaction_count": 0,
            "platform_data": {"page_type": page.get("type")},
        }

        if self.fetch_page_history:
            # Emit page_create event + page_edit event per version
            if created_at is not None:
                ev_create = dict(base_event)
                ev_create.update(
                    {
                        "item_type": "page_create",
                        "action": "created",
                        "activity_at": created_at,
                        "actor": author or None,
                        "actor_display_name": author_display,
                        "actor_email": author_email,
                    }
                )
                events.append(ev_create)
            # For each subsequent version edition up to current
            version_number = int(version.get("number") or 1)
            if version_number > 1 and updated_at is not None:
                ev_edit = dict(base_event)
                ev_edit.update(
                    {
                        "item_type": "page_edit",
                        "action": "edited",
                        "activity_at": updated_at,
                        "actor": editor or None,
                        "actor_display_name": editor_display,
                        "actor_email": editor_email,
                    }
                )
                events.append(ev_edit)
        else:
            # Single event per page at updated_at
            activity_at = updated_at or created_at
            actor = editor or author
            display = editor_display or author_display
            email = editor_email or author_email
            item_type = (
                "page_edit" if updated_at and int(version.get("number") or 1) > 1 else "page_create"
            )
            action = "edited" if item_type == "page_edit" else "created"
            if activity_at is not None:
                ev = dict(base_event)
                ev.update(
                    {
                        "item_type": item_type,
                        "action": action,
                        "activity_at": activity_at,
                        "actor": actor or None,
                        "actor_display_name": display,
                        "actor_email": email,
                    }
                )
                events.append(ev)

        return page_row, events

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def _upsert_pages_bulk(self, pages: list[dict[str, Any]]) -> None:
        """Upsert pages into ``confluence_page_cache`` by page_id."""
        from ..models.database import ConfluencePageCache

        with self.cache.get_session() as session:
            for p in pages:
                page_id = p.get("page_id")
                if not page_id:
                    continue
                existing = (
                    session.query(ConfluencePageCache)
                    .filter(ConfluencePageCache.page_id == page_id)
                    .first()
                )
                created_at = _to_naive_utc(p.get("created_at")) if p.get("created_at") else None
                updated_at = _to_naive_utc(p.get("updated_at")) if p.get("updated_at") else None
                if existing:
                    existing.space_key = p.get("space_key") or existing.space_key
                    existing.title = p.get("title") or existing.title
                    existing.version = p.get("version") or existing.version
                    existing.author = p.get("author") or existing.author
                    existing.author_email = p.get("author_email") or existing.author_email
                    existing.last_editor = p.get("last_editor") or existing.last_editor
                    existing.last_editor_email = (
                        p.get("last_editor_email") or existing.last_editor_email
                    )
                    existing.created_at = created_at or existing.created_at
                    existing.updated_at = updated_at or existing.updated_at
                    existing.labels = p.get("labels") or existing.labels
                    existing.ancestor_ids = p.get("ancestor_ids") or existing.ancestor_ids
                    existing.page_url = p.get("page_url") or existing.page_url
                    existing.platform_data = p.get("platform_data") or existing.platform_data
                else:
                    session.add(
                        ConfluencePageCache(
                            page_id=page_id,
                            space_key=p.get("space_key") or "",
                            title=p.get("title") or "",
                            version=int(p.get("version") or 0),
                            author=p.get("author"),
                            author_email=p.get("author_email"),
                            last_editor=p.get("last_editor"),
                            last_editor_email=p.get("last_editor_email"),
                            created_at=created_at,
                            updated_at=updated_at,
                            labels=p.get("labels") or [],
                            ancestor_ids=p.get("ancestor_ids") or [],
                            page_url=p.get("page_url") or "",
                            platform_data=p.get("platform_data") or {},
                        )
                    )
            session.commit()

    def _store_activity_events_bulk(self, events: list[dict[str, Any]]) -> None:
        """Persist activity events."""
        from ..models.database import TicketingActivityCache

        with self.cache.get_session() as session:
            for ev in events:
                activity_at = ev.get("activity_at")
                if activity_at is None:
                    continue
                activity_at_naive = _to_naive_utc(activity_at)

                row = TicketingActivityCache(
                    platform=ev.get("platform", "confluence"),
                    item_id=str(ev.get("item_id") or ""),
                    item_type=ev.get("item_type") or "page_edit",
                    repo_or_space=ev.get("repo_or_space") or "",
                    actor=ev.get("actor"),
                    actor_display_name=ev.get("actor_display_name"),
                    actor_email=ev.get("actor_email"),
                    action=ev.get("action"),
                    activity_at=activity_at_naive,
                    item_title=ev.get("item_title"),
                    item_status=ev.get("item_status"),
                    item_url=ev.get("item_url"),
                    linked_ticket_id=ev.get("linked_ticket_id"),
                    comment_count=int(ev.get("comment_count") or 0),
                    reaction_count=int(ev.get("reaction_count") or 0),
                    platform_data=ev.get("platform_data") or {},
                )
                session.add(row)
            session.commit()

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------
    def _get_with_retries(
        self, url: str, params: dict[str, Any] | None = None
    ) -> requests.Response | None:
        """GET with manual retry on 429."""
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.get(url, params=params, timeout=self.connection_timeout)
            except requests.RequestException as exc:
                print(f"   ⚠️  Confluence request error: {exc}")
                return None

            if response.status_code in (429, 503):
                wait_time = self.backoff_factor * (2**attempt)
                if self.debug_mode:
                    print(
                        f"   🔍 Confluence rate-limited (status {response.status_code}), "
                        f"waiting {wait_time}s (attempt {attempt + 1})"
                    )
                time.sleep(wait_time)
                continue

            if response.status_code >= 400:
                print(f"   ⚠️  Confluence API error {response.status_code}: {response.text[:200]}")
                return None

            return response

        print("   ❌ Confluence retries exhausted")
        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_confluence_datetime(value: Any) -> datetime | None:
    """Parse a Confluence ISO-8601 timestamp, returning naive UTC."""
    if not value:
        return None
    if isinstance(value, datetime):
        return _to_naive_utc(value)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        try:
            from dateutil import parser  # type: ignore

            dt = parser.parse(str(value))
        except Exception:  # noqa: BLE001
            return None
    return _to_naive_utc(dt)


def _to_naive_utc(value: datetime | None) -> datetime | None:
    """Convert a datetime to naive UTC per project convention.

    Returns None when the input is None.
    """
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _ensure_aware(value: datetime) -> datetime:
    """Return an aware UTC datetime."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value
