"""GitHub Issues API integration for issue and comment activity tracking.

WHY: Tracks per-developer issue activity (opens, closes, comments) as a
productivity signal distinct from PR and commit activity.  Follows the same
bulk-read-first -> API-fetch-misses -> bulk-write pattern as other integrations.

Actors (GitHub logins) are always stored lowercased so identity matching works
consistently with PR author/reviewer storage.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..core.cache import GitAnalysisCache
from ..core.schema_version import create_schema_manager
from ..utils.debug import is_debug_mode


class GitHubIssuesIntegration:
    """Integrate with GitHub Issues REST API to track issue/comment activity.

    Activity events are normalized into one row per event in
    ``ticketing_activity_cache`` (platform='github_issues').
    """

    COMPONENT_NAME = "github_issues"
    API_ROOT = "https://api.github.com"

    def __init__(
        self,
        token: str,
        cache: GitAnalysisCache,
        fetch_comments: bool = False,
        allowed_repos: list[str] | None = None,
        issue_state: str = "all",
        max_issues_per_repo: int = 500,
        rate_limit_retries: int = 3,
        backoff_factor: float = 2.0,
    ):
        """Initialize GitHub Issues integration.

        Args:
            token: GitHub personal access token.  Reused from the main GitHub
                configuration when called by the orchestrator.
            cache: Shared :class:`GitAnalysisCache` instance for DB access.
            fetch_comments: When True, additionally fetch issue comments and
                emit ``comment`` activity events (one API call per issue).
            allowed_repos: Optional whitelist of ``owner/repo`` slugs — other
                repos are skipped even if supplied to :meth:`fetch_issues_activity`.
            issue_state: GitHub issue state filter ('open' | 'closed' | 'all').
            max_issues_per_repo: Safety cap on number of issues fetched per repo
                per run to bound API usage.
            rate_limit_retries: Number of retry attempts on 403/429 responses.
            backoff_factor: Exponential backoff base (seconds).
        """
        self.token = token
        self.cache = cache
        self.fetch_comments = fetch_comments
        self.allowed_repos = allowed_repos
        self.issue_state = issue_state
        self.max_issues_per_repo = max_issues_per_repo
        self.rate_limit_retries = rate_limit_retries
        self.backoff_factor = backoff_factor
        self.debug_mode = is_debug_mode()

        self.schema_manager = create_schema_manager(cache.cache_dir)

        self._session = self._build_session()

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------
    def _build_session(self) -> requests.Session:
        """Build a resilient requests session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.rate_limit_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(
            {
                "Authorization": f"token {self.token}" if self.token else "",
                "Accept": "application/vnd.github+json",
                "User-Agent": "GitFlow-Analytics-GitHubIssues/1.0",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        return session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_issues_activity(self, repos: list[str], since: datetime) -> list[dict[str, Any]]:
        """Fetch issue activity events for the given repos since the given date.

        Args:
            repos: List of ``owner/repo`` slugs to scan.
            since: Fetch issues updated at or after this datetime.

        Returns:
            List of activity event dicts (one row per event, i.e. one per
            opened/closed/comment).  Events are also written to the cache.
        """
        if not repos:
            return []

        # Apply allowed_repos whitelist when configured
        if self.allowed_repos:
            allowed_set = {r.lower() for r in self.allowed_repos}
            repos = [r for r in repos if r.lower() in allowed_set]

        if not repos:
            if self.debug_mode:
                print("   🔍 GitHubIssues: no repos after allowed_repos filter")
            return []

        # Incremental fetch: take the later of requested-since and last-processed
        since = self._get_effective_since(since)

        all_events: list[dict[str, Any]] = []
        for repo in repos:
            try:
                repo_events = self._fetch_repo_issue_activity(repo, since)
                all_events.extend(repo_events)
            except Exception as exc:  # noqa: BLE001
                print(f"   ⚠️  GitHubIssues fetch failed for {repo}: {exc}")

        # Bulk persist
        if all_events:
            self._store_activity_events_bulk(all_events)
            print(f"   💾 Cached {len(all_events)} GitHub Issues activity event(s)")

        # Update schema tracking
        try:
            self.schema_manager.mark_date_processed(
                self.COMPONENT_NAME,
                datetime.now(timezone.utc),
                {
                    "fetch_comments": self.fetch_comments,
                    "issue_state": self.issue_state,
                },
            )
        except Exception as exc:  # noqa: BLE001
            if self.debug_mode:
                print(f"   🔍 schema_manager.mark_date_processed failed: {exc}")

        return all_events

    def get_activity_summary(self, since: datetime, until: datetime) -> dict[str, Any]:
        """Aggregate stored activity events for a time window.

        Args:
            since: Lower bound (inclusive) on activity_at.
            until: Upper bound (inclusive) on activity_at.

        Returns:
            Dict with per-repo and per-actor counts.
        """
        from ..models.database import TicketingActivityCache

        since_naive = _to_naive_utc(since)
        until_naive = _to_naive_utc(until)

        summary: dict[str, Any] = {
            "total_events": 0,
            "per_repo": {},
            "per_actor": {},
            "per_action": {},
        }

        with self.cache.get_session() as session:
            rows = (
                session.query(TicketingActivityCache)
                .filter(
                    TicketingActivityCache.platform == "github_issues",
                    TicketingActivityCache.activity_at >= since_naive,
                    TicketingActivityCache.activity_at <= until_naive,
                )
                .all()
            )

            for row in rows:
                summary["total_events"] += 1
                repo = getattr(row, "repo_or_space", None) or "unknown"
                actor = getattr(row, "actor", None) or "unknown"
                action = getattr(row, "action", None) or "unknown"

                summary["per_repo"][repo] = summary["per_repo"].get(repo, 0) + 1
                summary["per_actor"][actor] = summary["per_actor"].get(actor, 0) + 1
                summary["per_action"][action] = summary["per_action"].get(action, 0) + 1

        return summary

    # ------------------------------------------------------------------
    # Repo-level fetch
    # ------------------------------------------------------------------
    def _fetch_repo_issue_activity(self, repo: str, since: datetime) -> list[dict[str, Any]]:
        """Fetch issues + comments for a single repo since the given date."""
        if self.debug_mode:
            print(f"   🔍 GitHubIssues: fetching {repo} since {since.isoformat()}")

        events: list[dict[str, Any]] = []
        page = 1
        per_page = 100
        issues_fetched = 0
        since_iso = _to_github_iso(since)

        while issues_fetched < self.max_issues_per_repo:
            url = f"{self.API_ROOT}/repos/{repo}/issues"
            params = {
                "state": self.issue_state,
                "since": since_iso,
                "per_page": per_page,
                "page": page,
                # exclude PRs — GitHub's /issues endpoint includes PRs otherwise
            }
            response = self._get_with_retries(url, params=params)
            if response is None:
                break

            payload = response.json()
            if not isinstance(payload, list) or not payload:
                break

            for item in payload:
                # Skip pull requests — the /issues endpoint includes them
                if "pull_request" in item:
                    continue

                issues_fetched += 1
                if issues_fetched > self.max_issues_per_repo:
                    break

                issue_events = self._build_issue_events(repo, item)
                events.extend(issue_events)

                # Optional: fetch comments
                if self.fetch_comments and (item.get("comments") or 0) > 0:
                    comment_events = self._fetch_issue_comments(repo, item.get("number"), item)
                    events.extend(comment_events)

                # Rate-limit friendly pause between issue-level operations
                time.sleep(0.1)

            if len(payload) < per_page:
                break
            page += 1

        return events

    def _fetch_issue_comments(
        self, repo: str, issue_number: Any, issue_payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Fetch comments for a single issue and build comment activity events."""
        events: list[dict[str, Any]] = []
        if issue_number is None:
            return events

        url = f"{self.API_ROOT}/repos/{repo}/issues/{issue_number}/comments"
        page = 1
        per_page = 100

        while True:
            response = self._get_with_retries(url, params={"per_page": per_page, "page": page})
            if response is None:
                break

            payload = response.json()
            if not isinstance(payload, list) or not payload:
                break

            for comment in payload:
                events.append(_build_comment_event(repo, issue_payload, comment))

            if len(payload) < per_page:
                break
            page += 1
            time.sleep(0.1)

        return events

    # ------------------------------------------------------------------
    # Event building
    # ------------------------------------------------------------------
    def _build_issue_events(self, repo: str, issue: dict[str, Any]) -> list[dict[str, Any]]:
        """Build activity event rows from a GitHub issue payload."""
        events: list[dict[str, Any]] = []

        creator = issue.get("user") or {}
        actor = (creator.get("login") or "").lower() or None
        display_name = creator.get("login")  # GitHub doesn't expose display name here
        actor_email = creator.get("email")

        created_at = _parse_github_datetime(issue.get("created_at"))
        closed_at = _parse_github_datetime(issue.get("closed_at"))

        base = {
            "platform": "github_issues",
            "item_id": str(issue.get("number")) if issue.get("number") is not None else "",
            "item_type": "issue",
            "repo_or_space": repo,
            "item_title": issue.get("title") or "",
            "item_status": issue.get("state") or "",
            "item_url": issue.get("html_url") or "",
            "linked_ticket_id": None,
            "comment_count": int(issue.get("comments") or 0),
            "reaction_count": int((issue.get("reactions") or {}).get("total_count", 0) or 0),
            "platform_data": {
                "labels": [lbl.get("name") for lbl in issue.get("labels", []) if lbl],
                "assignees": [
                    (a.get("login") or "").lower() for a in issue.get("assignees", []) if a
                ],
            },
        }

        # Opened event
        if created_at is not None:
            event_open = dict(base)
            event_open.update(
                {
                    "action": "opened",
                    "activity_at": created_at,
                    "actor": actor,
                    "actor_display_name": display_name,
                    "actor_email": actor_email,
                }
            )
            events.append(event_open)

        # Closed event (if applicable)
        if closed_at is not None:
            closer = issue.get("closed_by") or {}
            closer_login = (closer.get("login") or "").lower() or actor
            event_close = dict(base)
            event_close.update(
                {
                    "action": "closed",
                    "activity_at": closed_at,
                    "actor": closer_login,
                    "actor_display_name": closer.get("login") or display_name,
                    "actor_email": closer.get("email") or actor_email,
                }
            )
            events.append(event_close)

        return events

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def _store_activity_events_bulk(self, events: list[dict[str, Any]]) -> None:
        """Persist activity events in a single session transaction.

        WHY: GitHub REST API only returns ``user.email`` when the user has set
        it public, which is rare for corporate accounts.  Before writing, we
        enrich null ``actor_email`` values from ``identities.db`` using the
        actor (GitHub login) so downstream consumers that key by email have
        resolvable identities.  See GitHub issue #52 (662 unresolved events).
        Test: ingest a GH issue where API returns null email for a user whose
        GitHub login is mapped in ``developer_identities.github_username``;
        assert the stored row has a non-null ``actor_email``.
        """
        from ..models.database import TicketingActivityCache

        username_to_email = self._build_username_email_map(events)

        with self.cache.get_session() as session:
            for ev in events:
                activity_at = ev.get("activity_at")
                if activity_at is None:
                    continue
                # Store as naive UTC per project convention
                activity_at_naive = _to_naive_utc(activity_at)

                actor_email = ev.get("actor_email")
                if not actor_email:
                    actor_key = (ev.get("actor") or "").lower()
                    if actor_key:
                        actor_email = username_to_email.get(actor_key)

                row = TicketingActivityCache(
                    platform=ev.get("platform", "github_issues"),
                    item_id=str(ev.get("item_id") or ""),
                    item_type=ev.get("item_type") or "issue",
                    repo_or_space=ev.get("repo_or_space") or "",
                    actor=ev.get("actor"),
                    actor_display_name=ev.get("actor_display_name"),
                    actor_email=actor_email,
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

    def _build_username_email_map(self, events: list[dict[str, Any]]) -> dict[str, str]:
        """Return lowercased ``github_username -> primary_email`` map.

        Why: Closes the GitHub-Issues actor-email gap in one bulk read of
        ``identities.db`` instead of per-row lookups.  Only usernames that
        appear in the current event batch are queried.
        Test: provide events with actors ``[\"bob-duetto\"]`` where
        ``developer_identities`` has ``github_username='bob-duetto'``; assert
        map contains ``{'bob-duetto': '<primary_email>'}``.
        """
        usernames: set[str] = set()
        for ev in events:
            if ev.get("actor_email"):
                continue
            actor = (ev.get("actor") or "").lower()
            if actor:
                usernames.add(actor)
        if not usernames:
            return {}

        try:
            from pathlib import Path

            from sqlalchemy import create_engine, text

            identities_path = Path(self.cache.cache_dir) / "identities.db"
            if not identities_path.exists():
                return {}

            engine = create_engine(f"sqlite:///{identities_path}")
            with engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT LOWER(github_username) AS u, primary_email "
                        "FROM developer_identities "
                        "WHERE github_username IS NOT NULL"
                    )
                ).fetchall()
            engine.dispose()
            return {r[0]: r[1] for r in rows if r[0] in usernames and r[1]}
        except Exception as exc:  # noqa: BLE001
            if self.debug_mode:
                print(f"   🔍 GitHub Issues email backfill skipped: {exc}")
            return {}

    # ------------------------------------------------------------------
    # Incremental fetch + HTTP helpers
    # ------------------------------------------------------------------
    def _get_effective_since(self, requested_since: datetime) -> datetime:
        """Compute incremental fetch start date.

        Uses the schema-version manager to track last-processed date.  Never
        goes backwards from the caller-supplied date.
        """
        if requested_since.tzinfo is None:
            requested_since = requested_since.replace(tzinfo=timezone.utc)

        try:
            last_processed = self.schema_manager.get_last_processed_date(self.COMPONENT_NAME)
        except Exception:
            last_processed = None

        if last_processed is None:
            return requested_since

        if last_processed.tzinfo is None:
            last_processed = last_processed.replace(tzinfo=timezone.utc)

        return max(last_processed, requested_since)

    def _get_with_retries(
        self, url: str, params: dict[str, Any] | None = None
    ) -> requests.Response | None:
        """Perform GET with manual retry on 403/429.

        Returns the response on success, None if all retries are exhausted.
        """
        for attempt in range(self.rate_limit_retries + 1):
            try:
                response = self._session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                print(f"   ⚠️  GitHub Issues request error: {exc}")
                return None

            if response.status_code in (403, 429):
                # Rate limited — back off and retry
                wait_time = self.backoff_factor**attempt
                if self.debug_mode:
                    print(
                        f"   🔍 GitHub Issues rate-limited (status {response.status_code}), "
                        f"waiting {wait_time}s (attempt {attempt + 1})"
                    )
                time.sleep(wait_time)
                continue

            if response.status_code >= 400:
                print(
                    f"   ⚠️  GitHub Issues API error {response.status_code}: {response.text[:200]}"
                )
                return None

            return response

        print("   ❌ GitHub Issues rate limit exhausted")
        return None


# ---------------------------------------------------------------------------
# Module-level helpers (keep testable / picklable)
# ---------------------------------------------------------------------------


def _build_comment_event(
    repo: str, issue: dict[str, Any], comment: dict[str, Any]
) -> dict[str, Any]:
    """Build a single comment activity event row."""
    commenter = comment.get("user") or {}
    actor = (commenter.get("login") or "").lower() or None
    return {
        "platform": "github_issues",
        "item_id": str(issue.get("number")) if issue.get("number") is not None else "",
        "item_type": "comment",
        "repo_or_space": repo,
        "actor": actor,
        "actor_display_name": commenter.get("login"),
        "actor_email": commenter.get("email"),
        "action": "commented",
        "activity_at": _parse_github_datetime(comment.get("created_at")),
        "item_title": issue.get("title") or "",
        "item_status": issue.get("state") or "",
        "item_url": comment.get("html_url") or issue.get("html_url") or "",
        "linked_ticket_id": None,
        "comment_count": 0,
        "reaction_count": int((comment.get("reactions") or {}).get("total_count", 0) or 0),
        "platform_data": {"comment_id": comment.get("id")},
    }


def _parse_github_datetime(value: Any) -> datetime | None:
    """Parse a GitHub ISO-8601 timestamp, return naive UTC datetime."""
    if not value:
        return None
    if isinstance(value, datetime):
        return _to_naive_utc(value)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return _to_naive_utc(dt)
    except ValueError:
        return None


def _to_naive_utc(value: datetime) -> datetime:
    """Convert a datetime to naive UTC per project convention."""
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _to_github_iso(value: datetime) -> str:
    """Format a datetime as GitHub-compatible ISO-8601 (UTC, with Z)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
