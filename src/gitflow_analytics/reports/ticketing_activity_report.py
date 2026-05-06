"""Ticketing Activity Report — GitHub Issues + Confluence summaries.

WHY: Combines per-developer and per-repo/space ticketing activity (issues,
comments, page edits) into JSON summary reports for downstream consumption.
Reads from ``ticketing_activity_cache`` and ``confluence_page_cache``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.cache import GitAnalysisCache

logger = logging.getLogger(__name__)


class TicketingActivityReport:
    """Generate ticketing activity summary reports from cached data."""

    # Weights used to compute the composite ticketing_score per developer.
    # Values are intentionally small so the score trends to single- or
    # double-digit numbers for typical weekly activity.
    _WEIGHTS = {
        "issues_opened": 1.0,
        "issues_closed": 1.0,
        "comments_posted": 0.5,
        "pages_created": 2.0,
        "pages_edited": 1.0,
        "jira_issues_opened": 1.5,
        "jira_issues_closed": 2.0,
        "jira_comments_posted": 0.5,
        # Azure DevOps weights mirror JIRA's. The adapter does not yet emit
        # ticketing-cache rows (Phase 3+); registering the keys now keeps the
        # combined summary forward-compatible without changing existing
        # behaviour for JIRA-only deployments.
        "azure_devops_issues_opened": 1.5,
        "azure_devops_issues_closed": 2.0,
        "azure_devops_comments_posted": 0.5,
    }

    def __init__(
        self,
        cache: GitAnalysisCache,
        identity_resolver: Any | None = None,
    ) -> None:
        """Initialize report generator.

        Args:
            cache: Shared :class:`GitAnalysisCache` instance.
            identity_resolver: Optional identity resolver used to collapse
                actor aliases to a canonical developer id.  When None the
                raw lowercased actor string is used.
        """
        self.cache = cache
        self.identity_resolver = identity_resolver

    # ------------------------------------------------------------------
    # Summary builders
    # ------------------------------------------------------------------
    def generate_github_issues_summary(self, since: datetime, until: datetime) -> dict[str, Any]:
        """Build a summary of GitHub Issues activity for the given window."""
        from ..models.database import TicketingActivityCache

        since_naive = _to_naive(since)
        until_naive = _to_naive(until)

        per_repo: dict[str, dict[str, Any]] = {}
        per_developer: dict[str, dict[str, Any]] = {}
        total_issues: set[tuple[str, str]] = set()
        total_events = 0
        issue_opened_to_closed: dict[tuple[str, str], dict[str, datetime]] = {}

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
                total_events += 1
                repo = getattr(row, "repo_or_space", None) or "unknown"
                actor = getattr(row, "actor", None) or "unknown"
                action = getattr(row, "action", None) or ""
                item_id = getattr(row, "item_id", None) or ""
                item_type = getattr(row, "item_type", None) or ""
                activity_at = getattr(row, "activity_at", None)

                canonical = self._canonical(actor)

                # Per-repo stats
                repo_entry = per_repo.setdefault(
                    repo,
                    {
                        "issue_count": 0,
                        "opened": 0,
                        "closed": 0,
                        "comments": 0,
                    },
                )

                if item_type == "issue" and action == "opened":
                    repo_entry["opened"] += 1
                    total_issues.add((repo, item_id))
                    if activity_at is not None:
                        issue_opened_to_closed.setdefault((repo, item_id), {})["opened_at"] = (
                            activity_at
                        )
                elif item_type == "issue" and action == "closed":
                    repo_entry["closed"] += 1
                    total_issues.add((repo, item_id))
                    if activity_at is not None:
                        issue_opened_to_closed.setdefault((repo, item_id), {})["closed_at"] = (
                            activity_at
                        )
                elif item_type == "comment":
                    repo_entry["comments"] += 1

                repo_entry["issue_count"] = len({i for (r, i) in total_issues if r == repo})

                # Per-developer stats
                dev = per_developer.setdefault(
                    canonical,
                    {
                        "issues_opened": 0,
                        "issues_closed": 0,
                        "comments_posted": 0,
                        "total_activity": 0,
                    },
                )
                if item_type == "issue" and action == "opened":
                    dev["issues_opened"] += 1
                elif item_type == "issue" and action == "closed":
                    dev["issues_closed"] += 1
                elif item_type == "comment":
                    dev["comments_posted"] += 1
                dev["total_activity"] += 1

        # Compute avg resolution hours per repo
        for repo, entry in per_repo.items():
            deltas: list[float] = []
            for (r, _), times in issue_opened_to_closed.items():
                if r != repo:
                    continue
                opened = times.get("opened_at")
                closed = times.get("closed_at")
                if opened and closed:
                    deltas.append((closed - opened).total_seconds() / 3600.0)
            entry["avg_resolution_hours"] = round(sum(deltas) / len(deltas), 2) if deltas else None

        top_contributors = _sorted_top(per_developer, key="total_activity", limit=10)

        return {
            "period": {
                "since": since_naive.isoformat() if since_naive else None,
                "until": until_naive.isoformat() if until_naive else None,
            },
            "total_events": total_events,
            "total_issues": len(total_issues),
            "per_repo": per_repo,
            "per_developer": per_developer,
            "top_contributors": top_contributors,
        }

    def generate_confluence_summary(self, since: datetime, until: datetime) -> dict[str, Any]:
        """Build a summary of Confluence page activity for the given window."""
        from ..models.database import TicketingActivityCache

        since_naive = _to_naive(since)
        until_naive = _to_naive(until)

        per_space: dict[str, dict[str, Any]] = {}
        per_developer: dict[str, dict[str, Any]] = {}
        total_events = 0

        with self.cache.get_session() as session:
            rows = (
                session.query(TicketingActivityCache)
                .filter(
                    TicketingActivityCache.platform == "confluence",
                    TicketingActivityCache.activity_at >= since_naive,
                    TicketingActivityCache.activity_at <= until_naive,
                )
                .all()
            )

            for row in rows:
                total_events += 1
                space = getattr(row, "repo_or_space", None) or "unknown"
                actor = getattr(row, "actor", None) or "unknown"
                item_type = getattr(row, "item_type", None) or ""

                canonical = self._canonical(actor)

                space_entry = per_space.setdefault(
                    space,
                    {
                        "pages_created": 0,
                        "pages_edited": 0,
                        "unique_editors": set(),
                    },
                )

                dev_entry = per_developer.setdefault(
                    canonical,
                    {
                        "pages_created": 0,
                        "pages_edited": 0,
                        "spaces_active": set(),
                        "total_activity": 0,
                    },
                )

                if item_type == "page_create":
                    space_entry["pages_created"] += 1
                    dev_entry["pages_created"] += 1
                elif item_type == "page_edit":
                    space_entry["pages_edited"] += 1
                    dev_entry["pages_edited"] += 1

                space_entry["unique_editors"].add(canonical)
                dev_entry["spaces_active"].add(space)
                dev_entry["total_activity"] += 1

        # Convert sets -> counts/lists for JSON serialization
        for _, entry in per_space.items():
            entry["unique_editors_count"] = len(entry["unique_editors"])
            entry["unique_editors"] = sorted(entry["unique_editors"])
        for _, entry in per_developer.items():  # pyright: ignore[reportUnusedVariable]
            entry["spaces_active_count"] = len(entry["spaces_active"])
            entry["spaces_active"] = sorted(entry["spaces_active"])

        return {
            "period": {
                "since": since_naive.isoformat() if since_naive else None,
                "until": until_naive.isoformat() if until_naive else None,
            },
            "total_events": total_events,
            "per_space": per_space,
            "per_developer": per_developer,
        }

    def generate_jira_summary(self, since: datetime, until: datetime) -> dict[str, Any]:
        """Build a summary of JIRA issue + comment activity for the given window.

        Reads rows where ``platform='jira'`` from ``ticketing_activity_cache``.
        """
        from ..models.database import TicketingActivityCache

        since_naive = _to_naive(since)
        until_naive = _to_naive(until)

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
                actor = getattr(row, "actor", None) or "unknown"
                item_type = getattr(row, "item_type", None) or ""
                canonical = self._canonical(actor)
                entry = per_developer.setdefault(
                    canonical,
                    {
                        "jira_issues_opened": 0,
                        "jira_issues_closed": 0,
                        "jira_comments_posted": 0,
                        "total_activity": 0,
                    },
                )
                if item_type == "issue_created":
                    entry["jira_issues_opened"] += 1
                elif item_type == "issue_closed":
                    entry["jira_issues_closed"] += 1
                elif item_type == "comment":
                    entry["jira_comments_posted"] += 1
                entry["total_activity"] += 1

        return {
            "period": {
                "since": since_naive.isoformat() if since_naive else None,
                "until": until_naive.isoformat() if until_naive else None,
            },
            "total_events": total_events,
            "per_developer": per_developer,
        }

    def generate_combined_summary(self, since: datetime, until: datetime) -> dict[str, Any]:
        """Build a combined per-developer summary across all platforms."""
        gh_summary = self.generate_github_issues_summary(since, until)
        conf_summary = self.generate_confluence_summary(since, until)
        jira_summary = self.generate_jira_summary(since, until)

        combined: dict[str, dict[str, Any]] = {}

        for dev, stats in gh_summary.get("per_developer", {}).items():
            c = combined.setdefault(dev, _empty_combined_row())
            c["issues_opened"] += stats.get("issues_opened", 0)
            c["issues_closed"] += stats.get("issues_closed", 0)
            c["comments_posted"] += stats.get("comments_posted", 0)

        for dev, stats in conf_summary.get("per_developer", {}).items():
            c = combined.setdefault(dev, _empty_combined_row())
            c["pages_created"] += stats.get("pages_created", 0)
            c["pages_edited"] += stats.get("pages_edited", 0)

        for dev, stats in jira_summary.get("per_developer", {}).items():
            c = combined.setdefault(dev, _empty_combined_row())
            c["jira_issues_opened"] += stats.get("jira_issues_opened", 0)
            c["jira_issues_closed"] += stats.get("jira_issues_closed", 0)
            c["jira_comments_posted"] += stats.get("jira_comments_posted", 0)

        for _, row in combined.items():
            row["ticketing_score"] = round(
                sum(row.get(k, 0) * w for k, w in self._WEIGHTS.items()),
                2,
            )
            row["total_activity"] = (
                row["issues_opened"]
                + row["issues_closed"]
                + row["comments_posted"]
                + row["pages_created"]
                + row["pages_edited"]
                + row["jira_issues_opened"]
                + row["jira_issues_closed"]
                + row["jira_comments_posted"]
            )

        # Sort developers by ticketing_score descending for convenience
        top = sorted(
            ({"developer": dev, **row} for dev, row in combined.items()),
            key=lambda r: r.get("ticketing_score", 0),
            reverse=True,
        )

        return {
            "period": gh_summary.get("period"),
            "per_developer": combined,
            "top_contributors": top[:10],
            "github_issues": {
                "total_events": gh_summary.get("total_events", 0),
                "total_issues": gh_summary.get("total_issues", 0),
            },
            "confluence": {
                "total_events": conf_summary.get("total_events", 0),
            },
            "jira": {
                "total_events": jira_summary.get("total_events", 0),
            },
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def write_reports(self, output_dir: Path, since: datetime, until: datetime) -> list[str]:
        """Write all three summary JSON files and return their file paths."""
        output_dir.mkdir(parents=True, exist_ok=True)

        gh_summary = self.generate_github_issues_summary(since, until)
        conf_summary = self.generate_confluence_summary(since, until)
        combined_summary = self.generate_combined_summary(since, until)

        written: list[str] = []

        gh_path = output_dir / "github_issues_summary.json"
        _write_json(gh_path, gh_summary)
        written.append(str(gh_path))

        conf_path = output_dir / "confluence_activity_summary.json"
        _write_json(conf_path, conf_summary)
        written.append(str(conf_path))

        combined_path = output_dir / "ticketing_activity_summary.json"
        _write_json(combined_path, combined_summary)
        written.append(str(combined_path))

        return written

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _canonical(self, actor: str | None) -> str:
        """Resolve actor to canonical developer id when a resolver is available."""
        if not actor:
            return "unknown"
        if self.identity_resolver is None:
            return actor.lower()
        try:
            # Best-effort: try common method names without hard-coupling.
            if hasattr(self.identity_resolver, "resolve_identity"):
                resolved = self.identity_resolver.resolve_identity(None, actor)
                if isinstance(resolved, str) and resolved:
                    return resolved.lower()
        except Exception:  # noqa: BLE001  # nosec B110 - best-effort identity resolution, fallback to raw actor
            pass
        return actor.lower()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _empty_combined_row() -> dict[str, Any]:
    return {
        "issues_opened": 0,
        "issues_closed": 0,
        "comments_posted": 0,
        "pages_created": 0,
        "pages_edited": 0,
        "jira_issues_opened": 0,
        "jira_issues_closed": 0,
        "jira_comments_posted": 0,
        "total_activity": 0,
        "ticketing_score": 0.0,
    }


def _sorted_top(mapping: dict[str, dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    return sorted(
        ({"developer": dev, **stats} for dev, stats in mapping.items()),
        key=lambda r: r.get(key, 0),
        reverse=True,
    )[:limit]


def _to_naive(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _write_json(path: Path, data: Any) -> None:
    def _default(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_default)
