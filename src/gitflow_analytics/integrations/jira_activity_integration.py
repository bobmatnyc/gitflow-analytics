"""JIRA Cloud REST API integration for issue + comment activity tracking.

WHY: Tracks per-developer JIRA issue opens/closes and comment activity as a
productivity signal distinct from story-point enrichment (handled by
``JIRAIntegration``).  Activity events are normalized into one row per
event in ``ticketing_activity_cache`` (platform='jira').

Actors (reporter email or username) are always stored lowercased so that
identity matching is consistent with other integrations.
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


class JIRAActivityIntegration:
    """Fetch JIRA issue/comment activity into ``ticketing_activity_cache``."""

    COMPONENT_NAME = "jira_activity"

    def __init__(
        self,
        base_url: str,
        username: str,
        api_token: str,
        cache: GitAnalysisCache,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        connection_timeout: int = 30,
    ):
        """Initialize JIRA activity integration.

        Args:
            base_url: JIRA instance base URL (e.g. ``https://company.atlassian.net``).
            username: Atlassian username / email.
            api_token: Atlassian API token.
            cache: Shared :class:`GitAnalysisCache` instance.
            max_retries: Retry attempts on 429/5xx.
            backoff_factor: Exponential backoff base (seconds).
            connection_timeout: HTTP timeout (seconds).
        """
        self.base_url = (base_url or "").rstrip("/")
        self.username = username
        self.api_token = api_token
        self.cache = cache
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.connection_timeout = connection_timeout
        self.debug_mode = is_debug_mode()

        self.schema_manager = create_schema_manager(cache.cache_dir)

        credentials = base64.b64encode(f"{username}:{api_token}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {credentials}",
            "Accept": "application/json",
            "User-Agent": "GitFlow-Analytics-JIRAActivity/1.0",
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
    # Public API
    # ------------------------------------------------------------------
    def fetch_project_activity(
        self, project_keys: list[str], since: datetime, until: datetime
    ) -> list[dict[str, Any]]:
        """Fetch issue + comment activity for the given projects.

        Args:
            project_keys: List of JIRA project keys to scan (e.g. ``["PROJ"]``).
            since: Lower bound on issue created/updated timestamps.
            until: Upper bound for filtering events (inclusive).

        Returns:
            List of activity event dicts (one per issue_created /
            issue_closed / comment).  Events are also persisted to
            ``ticketing_activity_cache``.
        """
        if not project_keys:
            return []
        if not self.base_url or not self.username or not self.api_token:
            return []

        since = _ensure_aware(since)
        until = _ensure_aware(until)
        effective_since = self._get_effective_since(since)

        if self.debug_mode:
            print(
                f"   🔍 JIRAActivity: fetching projects={project_keys} "
                f"since={effective_since.isoformat()}"
            )

        all_events: list[dict[str, Any]] = []
        issues = self._fetch_issues(project_keys, effective_since)

        for issue in issues:
            issue_events = self._build_issue_events(issue, effective_since, until)
            all_events.extend(issue_events)

            comment_events = self._fetch_issue_comments(issue, effective_since, until)
            all_events.extend(comment_events)

            time.sleep(0.1)

        if all_events:
            self._store_activity_events_bulk(all_events)
            print(f"   💾 Cached {len(all_events)} JIRA activity event(s)")

        try:
            self.schema_manager.mark_date_processed(
                self.COMPONENT_NAME,
                datetime.now(timezone.utc),
                {"project_keys": project_keys},
            )
        except Exception as exc:  # noqa: BLE001
            if self.debug_mode:
                print(f"   🔍 JIRAActivity schema_manager mark failed: {exc}")

        return all_events

    def get_activity_summary(self, since: datetime, until: datetime) -> dict[str, Any]:
        """Aggregate stored JIRA activity events for a time window.

        Returns a dict with per-developer jira_issues_opened,
        jira_issues_closed, jira_comments_posted counts.
        """
        from ..models.database import TicketingActivityCache

        since_naive = _to_naive_utc(since)
        until_naive = _to_naive_utc(until)

        per_developer: dict[str, dict[str, Any]] = {}
        total_events = 0

        with self.cache.get_session() as session:
            rows = (
                session.query(TicketingActivityCache)
                .filter(
                    TicketingActivityCache.platform == "jira",
                    TicketingActivityCache.activity_at >= since_naive,
                    TicketingActivityCache.activity_at <= until_naive,
                )
                .all()
            )
            for row in rows:
                total_events += 1
                actor = (getattr(row, "actor", None) or "unknown").lower() or "unknown"
                item_type = getattr(row, "item_type", None) or ""
                entry = per_developer.setdefault(
                    actor,
                    {
                        "jira_issues_opened": 0,
                        "jira_issues_closed": 0,
                        "jira_comments_posted": 0,
                    },
                )
                if item_type == "issue_created":
                    entry["jira_issues_opened"] += 1
                elif item_type == "issue_closed":
                    entry["jira_issues_closed"] += 1
                elif item_type == "comment":
                    entry["jira_comments_posted"] += 1

        return {
            "total_events": total_events,
            "per_developer": per_developer,
        }

    # ------------------------------------------------------------------
    # Issue + comment fetching
    # ------------------------------------------------------------------
    def _fetch_issues(self, project_keys: list[str], since: datetime) -> list[dict[str, Any]]:
        """Fetch issues using JQL via the v3 /rest/api/3/search/jql POST endpoint.

        WHY: Atlassian retired ``/rest/api/2/search`` (and legacy v3 ``/search``)
        — a 410 is now returned.  The replacement endpoint is POST
        ``/rest/api/3/search/jql`` with token-based pagination
        (``nextPageToken``/``isLast``) rather than ``startAt``/``total``.
        See CHANGE-2046 in Atlassian changelog.
        """
        # JQL — quote each project key to avoid collisions with JQL reserved
        # words (e.g. project key ``IS`` is parsed as the ``IS`` operator
        # without quotes and yields a 400).
        keys_csv = ",".join(f'"{k}"' for k in project_keys)
        since_iso = since.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
        jql = f'project in ({keys_csv}) AND (created >= "{since_iso}" OR updated >= "{since_iso}")'

        issues: list[dict[str, Any]] = []
        max_results = 50
        url = f"{self.base_url}/rest/api/3/search/jql"
        next_page_token: str | None = None

        while True:
            body: dict[str, Any] = {
                "jql": jql,
                "maxResults": max_results,
                "fields": [
                    "summary",
                    "status",
                    "created",
                    "updated",
                    "resolutiondate",
                    "reporter",
                    "assignee",
                ],
            }
            if next_page_token:
                body["nextPageToken"] = next_page_token

            response = self._post_with_retries(url, json_body=body)
            if response is None:
                break
            data = response.json() if response.content else {}
            page_issues = data.get("issues", []) if isinstance(data, dict) else []
            if not page_issues:
                break
            issues.extend(page_issues)

            is_last = bool(data.get("isLast", True)) if isinstance(data, dict) else True
            next_page_token = data.get("nextPageToken") if isinstance(data, dict) else None
            if is_last or not next_page_token:
                break
            time.sleep(0.1)

        return issues

    def _fetch_issue_comments(
        self, issue: dict[str, Any], since: datetime, until: datetime
    ) -> list[dict[str, Any]]:
        """Fetch comments for a single issue and build comment events."""
        key = issue.get("key")
        if not key:
            return []

        url = f"{self.base_url}/rest/api/2/issue/{key}/comment"
        response = self._get_with_retries(url)
        if response is None:
            return []
        data = response.json() if response.content else {}
        comments = data.get("comments", []) if isinstance(data, dict) else []

        events: list[dict[str, Any]] = []
        since_naive = _to_naive_utc(since)
        until_naive = _to_naive_utc(until)
        for comment in comments:
            ev = _build_comment_event(issue, comment)
            if ev is None:
                continue
            activity_at = ev.get("activity_at")
            if activity_at is None:
                continue
            if since_naive is not None and activity_at < since_naive:
                continue
            if until_naive is not None and activity_at > until_naive:
                continue
            events.append(ev)

        return events

    # ------------------------------------------------------------------
    # Event building
    # ------------------------------------------------------------------
    def _build_issue_events(
        self, issue: dict[str, Any], since: datetime, until: datetime
    ) -> list[dict[str, Any]]:
        """Build issue_created + issue_closed events for a JIRA issue."""
        fields = issue.get("fields", {}) or {}
        reporter = fields.get("reporter") or {}

        actor = _extract_actor(reporter)
        display_name = reporter.get("displayName") or reporter.get("name")
        actor_email = reporter.get("emailAddress")

        created_at = _parse_jira_date(fields.get("created"))
        resolved_at = _parse_jira_date(fields.get("resolutiondate"))

        key = issue.get("key") or ""
        summary = fields.get("summary") or ""
        status = (fields.get("status") or {}).get("name") or ""
        item_url = f"{self.base_url}/browse/{key}" if key else ""

        since_naive = _to_naive_utc(since)
        until_naive = _to_naive_utc(until)

        base = {
            "platform": "jira",
            "item_id": key,
            "repo_or_space": _project_key_from_issue_key(key),
            "item_title": summary,
            "item_status": status,
            "item_url": item_url,
            "linked_ticket_id": key,
            "comment_count": 0,
            "reaction_count": 0,
            "platform_data": {
                "assignee": (
                    (fields.get("assignee") or {}).get("emailAddress")
                    or (fields.get("assignee") or {}).get("name")
                ),
            },
        }

        events: list[dict[str, Any]] = []

        if created_at is not None and _within(created_at, since_naive, until_naive):
            ev_open = dict(base)
            ev_open.update(
                {
                    "item_type": "issue_created",
                    "action": "opened",
                    "activity_at": created_at,
                    "actor": actor,
                    "actor_display_name": display_name,
                    "actor_email": actor_email,
                }
            )
            events.append(ev_open)

        if resolved_at is not None and _within(resolved_at, since_naive, until_naive):
            # JIRA v2 REST does not reliably expose "closer" so we attribute
            # to the assignee when available, else to the reporter.
            assignee = fields.get("assignee") or {}
            closer_actor = _extract_actor(assignee) or actor
            closer_display = assignee.get("displayName") or assignee.get("name") or display_name
            closer_email = assignee.get("emailAddress") or actor_email
            ev_close = dict(base)
            ev_close.update(
                {
                    "item_type": "issue_closed",
                    "action": "closed",
                    "activity_at": resolved_at,
                    "actor": closer_actor,
                    "actor_display_name": closer_display,
                    "actor_email": closer_email,
                }
            )
            events.append(ev_close)

        return events

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def _store_activity_events_bulk(self, events: list[dict[str, Any]]) -> None:
        """Persist activity events in a single session transaction."""
        from ..models.database import TicketingActivityCache

        with self.cache.get_session() as session:
            for ev in events:
                activity_at = ev.get("activity_at")
                if activity_at is None:
                    continue
                activity_at_naive = _to_naive_utc(activity_at)

                row = TicketingActivityCache(
                    platform=ev.get("platform", "jira"),
                    item_id=str(ev.get("item_id") or ""),
                    item_type=ev.get("item_type") or "issue_created",
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
    # Incremental fetch + HTTP helpers
    # ------------------------------------------------------------------
    def _get_effective_since(self, requested_since: datetime) -> datetime:
        """Compute incremental fetch start date via schema-version manager."""
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

    def _get_with_retries(
        self, url: str, params: dict[str, Any] | None = None
    ) -> requests.Response | None:
        """GET with manual retry on 429/503."""
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.get(url, params=params, timeout=self.connection_timeout)
            except requests.RequestException as exc:
                print(f"   ⚠️  JIRA activity request error: {exc}")
                return None

            if response.status_code in (429, 503):
                wait_time = self.backoff_factor * (2**attempt)
                if self.debug_mode:
                    print(
                        f"   🔍 JIRAActivity rate-limited (status {response.status_code}), "
                        f"waiting {wait_time}s (attempt {attempt + 1})"
                    )
                time.sleep(wait_time)
                continue

            if response.status_code >= 400:
                print(
                    f"   ⚠️  JIRA activity API error {response.status_code}: {response.text[:200]}"
                )
                return None

            return response

        print("   ❌ JIRA activity retries exhausted")
        return None

    def _post_with_retries(self, url: str, json_body: dict[str, Any]) -> requests.Response | None:
        """POST with manual retry on 429/503.

        Why: the v3 ``/rest/api/3/search/jql`` endpoint requires POST with a
        JSON body rather than GET with query params.  Mirrors the retry
        semantics of :meth:`_get_with_retries` so the caller behaves the same.
        Test: mock :attr:`_session` to return a 429 then a 200 and assert the
        function retries and returns the 200 response.
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.post(url, json=json_body, timeout=self.connection_timeout)
            except requests.RequestException as exc:
                print(f"   ⚠️  JIRA activity request error: {exc}")
                return None

            if response.status_code in (429, 503):
                wait_time = self.backoff_factor * (2**attempt)
                if self.debug_mode:
                    print(
                        f"   🔍 JIRAActivity rate-limited (status {response.status_code}), "
                        f"waiting {wait_time}s (attempt {attempt + 1})"
                    )
                time.sleep(wait_time)
                continue

            if response.status_code >= 400:
                print(
                    f"   ⚠️  JIRA activity API error {response.status_code}: {response.text[:200]}"
                )
                return None

            return response

        print("   ❌ JIRA activity retries exhausted")
        return None


# ---------------------------------------------------------------------------
# Module-level helpers (testable / picklable)
# ---------------------------------------------------------------------------


def _build_comment_event(issue: dict[str, Any], comment: dict[str, Any]) -> dict[str, Any] | None:
    """Build a single JIRA comment activity event."""
    author = comment.get("author") or {}
    actor = _extract_actor(author)
    created_at = _parse_jira_date(comment.get("created"))
    if created_at is None:
        return None

    key = issue.get("key") or ""
    fields = issue.get("fields", {}) or {}
    return {
        "platform": "jira",
        "item_id": key,
        "item_type": "comment",
        "repo_or_space": _project_key_from_issue_key(key),
        "actor": actor,
        "actor_display_name": author.get("displayName") or author.get("name"),
        "actor_email": author.get("emailAddress"),
        "action": "commented",
        "activity_at": created_at,
        "item_title": fields.get("summary") or "",
        "item_status": (fields.get("status") or {}).get("name") or "",
        "item_url": "",
        "linked_ticket_id": key,
        "comment_count": 0,
        "reaction_count": 0,
        "platform_data": {"comment_id": comment.get("id")},
    }


def _extract_actor(user: dict[str, Any]) -> str | None:
    """Extract a canonical lowercased actor identifier from a JIRA user dict.

    Prefers ``emailAddress`` (most stable), falls back to ``name`` then
    ``accountId``.  Returns None when no identifier is available.
    """
    if not user:
        return None
    email = user.get("emailAddress")
    if email:
        return str(email).lower()
    name = user.get("name")
    if name:
        return str(name).lower()
    account_id = user.get("accountId")
    if account_id:
        return str(account_id).lower()
    return None


def _parse_jira_date(value: Any) -> datetime | None:
    """Parse a JIRA ISO-8601 timestamp, returning naive UTC."""
    if not value:
        return None
    if isinstance(value, datetime):
        return _to_naive_utc(value)
    try:
        from dateutil import parser  # type: ignore

        dt = parser.parse(str(value))
    except Exception:  # noqa: BLE001
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return _to_naive_utc(dt)


def _to_naive_utc(value: datetime | None) -> datetime | None:
    """Convert datetime to naive UTC per project convention."""
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


def _within(value: datetime, since: datetime | None, until: datetime | None) -> bool:
    """Return True iff ``value`` is within ``[since, until]`` (naive UTC)."""
    if since is not None and value < since:
        return False
    return not (until is not None and value > until)


def _project_key_from_issue_key(issue_key: str) -> str:
    """Extract JIRA project key (``PROJ``) from an issue key (``PROJ-123``)."""
    if not issue_key:
        return ""
    return issue_key.split("-", 1)[0]
