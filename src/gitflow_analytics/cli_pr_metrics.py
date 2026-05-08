"""CLI command for aggregating weekly per-engineer PR metrics (issue #49).

WHY: This module implements the ``gfa pr-metrics`` command which aggregates
per-engineer pull request activity from the existing ``pull_request_cache``
table into a dedicated ``weekly_pr_metrics`` table keyed by
(engineer_identifier, iso_week).

The command is **non-destructive** — it never touches ``pull_request_cache``
or any other table — and uses upsert semantics so that re-running the same
week is safe and idempotent.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import click
from sqlalchemy import and_

from .cli_utils import setup_logging
from .config import ConfigLoader
from .models.database import (
    Database,
    PullRequestCache,
    WeeklyPRMetrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions (pure / unit-testable)
# ---------------------------------------------------------------------------


def format_iso_week(d: date) -> str:
    """Return the ISO-week label for a date in 'YYYY-WNN' format.

    Uses Python's ``date.isocalendar()`` which returns the ISO year/week pair.
    The ISO year may differ from the calendar year for dates near year boundaries
    (e.g. 2024-12-30 is in ISO week 2025-W01).

    Args:
        d: A ``date`` or ``datetime`` instance.

    Returns:
        The ISO week label, e.g. ``'2026-W16'``.
    """
    iso_year, iso_week, _ = d.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def parse_iso_week(week_str: str) -> tuple[datetime, datetime]:
    """Parse an ISO-week label into a (week_start, week_end) UTC tuple.

    The week_start is Monday 00:00:00 UTC of the given ISO week and the
    week_end is Sunday 23:59:59.999999 UTC.

    Args:
        week_str: ISO week label, e.g. ``'2026-W16'``.

    Returns:
        ``(week_start, week_end)`` as timezone-aware UTC datetimes.

    Raises:
        ValueError: If ``week_str`` is not in ``YYYY-WNN`` format.
    """
    # %G is the ISO-year, %V is the ISO week number, %u is the ISO weekday (1=Mon).
    # Appending '-1' selects Monday of the given ISO week.
    try:
        monday = datetime.strptime(week_str + "-1", "%G-W%V-%u")
    except ValueError as exc:
        raise ValueError(
            f"Invalid ISO week '{week_str}'. Expected format 'YYYY-WNN' (e.g. '2026-W16')."
        ) from exc

    week_start = monday.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999)
    return week_start, week_end


def calculate_week_range(
    week: str | None,
    since: str | None,
    now: datetime | None = None,
) -> list[tuple[str, datetime, datetime]]:
    """Resolve the CLI flags into a list of ISO weeks to process.

    Precedence:
      1. ``--week 2026-W16`` -> only that week
      2. ``--since YYYY-MM-DD`` -> all weeks from that date through the
         current week (inclusive)
      3. Neither flag -> the current ISO week only

    Args:
        week: Optional explicit ISO week label.
        since: Optional ``YYYY-MM-DD`` start date.
        now: Optional override for "current time" (mostly for tests).

    Returns:
        A list of ``(iso_week_label, week_start_utc, week_end_utc)`` tuples,
        ordered chronologically and de-duplicated.

    Raises:
        ValueError: If both flags are provided, or if a value can't be parsed.
    """
    if week and since:
        raise ValueError("--week and --since are mutually exclusive; pick one.")

    if now is None:
        now = datetime.now(timezone.utc)

    # Single explicit week
    if week:
        week_start, week_end = parse_iso_week(week)
        return [(week, week_start, week_end)]

    # Backfill from a start date
    if since:
        try:
            since_date = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValueError(f"Invalid --since date '{since}'. Expected YYYY-MM-DD.") from exc

        weeks: list[tuple[str, datetime, datetime]] = []
        seen: set[str] = set()
        # Walk one day at a time across the range; collect each unique ISO week.
        # Cheap enough for any sane backfill window and avoids edge-case bugs
        # around ISO year boundaries that arise with naive timedelta(weeks=1).
        cursor = since_date
        while cursor <= now:
            label = format_iso_week(cursor)
            if label not in seen:
                seen.add(label)
                ws, we = parse_iso_week(label)
                weeks.append((label, ws, we))
            cursor += timedelta(days=1)
        # Make sure the current week is included even if `now` lands mid-week.
        current_label = format_iso_week(now)
        if current_label not in seen:
            ws, we = parse_iso_week(current_label)
            weeks.append((current_label, ws, we))
        return weeks

    # Default: current ISO week only
    current_label = format_iso_week(now)
    ws, we = parse_iso_week(current_label)
    return [(current_label, ws, we)]


def aggregate_week(
    db: Database,
    iso_week: str,
    week_start: datetime,
    week_end: datetime,
) -> dict[str, dict[str, Any]]:
    """Compute per-engineer aggregates for a single ISO week.

    Aggregation rules (issue #49 + #66):
      * ``prs_opened``               — count of PRs whose ``created_at`` is in the week
                                        grouped by ``author``.
      * ``prs_merged``               — count of PRs whose ``merged_at`` is in the week
                                        AND ``is_merged`` is truthy, grouped by ``author``.
      * ``pr_reviews_given``         — for each PR opened in the week, increment by 1
                                        for every login in the ``reviewers`` JSON list.
      * ``pr_comments_given``        — same as ``pr_reviews_given`` (proxy).
      * ``pr_merge_rate``            — prs_merged / prs_opened (None when prs_opened = 0).
                                        Issue #66.
      * ``avg_cycle_time_hrs``       — Avg ``(merged_at - created_at).total_seconds() / 3600``
                                        across PRs *merged* in the week (None when no
                                        merges or timestamps missing). Issue #66.
      * ``change_requests_received`` — Sum of ``change_requests_count`` on PRs the
                                        engineer authored that opened in the week.
                                        Issue #66.
      * ``avg_revisions_per_pr``     — Avg ``revision_count`` across PRs the engineer
                                        authored that opened in the week. Issue #66.

    Args:
        db: Database wrapper exposing ``get_session``.
        iso_week: ISO week label (only used for logging).
        week_start: Inclusive lower bound (UTC, tz-aware).
        week_end: Inclusive upper bound (UTC, tz-aware).

    Returns:
        Mapping ``engineer -> {<metric_keys>}``.  Engineers with zero activity
        for the week are omitted.
    """
    # Per-engineer counters; revision/cycle-time totals are tracked separately
    # so we can compute averages at the end without losing precision.
    counts: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "prs_opened": 0,
            "prs_merged": 0,
            "pr_reviews_given": 0,
            "pr_comments_given": 0,
            "change_requests_received": 0,
            # Internal accumulators (popped before return)
            "_revision_total": 0,
            "_revision_count_prs": 0,
            "_cycle_time_hours_total": 0.0,
            "_cycle_time_count": 0,
        }
    )

    session = db.get_session()
    try:
        # 1. PRs opened in the week (by created_at)
        opened = (
            session.query(PullRequestCache)
            .filter(
                and_(
                    PullRequestCache.created_at >= week_start,
                    PullRequestCache.created_at <= week_end,
                )
            )
            .all()
        )
        for pr in opened:
            if not cast(str | None, pr.author):
                continue
            entry = counts[cast(str, pr.author)]
            entry["prs_opened"] += 1

            # Issue #66: aggregate change_requests_received and revision stats
            # over PRs *opened in this week* by the engineer.  The columns may
            # be NULL on legacy rows (pre-v3 pull_request_cache) — coerce to 0.
            # cast() to int silences pyright's Column[int] arithmetic complaints
            # without changing runtime behavior.
            cr_count = cast(int, pr.change_requests_count) or 0
            entry["change_requests_received"] += cr_count

            rev_count = pr.revision_count
            if rev_count is not None:
                entry["_revision_total"] += cast(int, rev_count)
                entry["_revision_count_prs"] += 1

        # 2. PRs merged in the week (by merged_at AND is_merged truthy)
        merged = (
            session.query(PullRequestCache)
            .filter(
                and_(
                    PullRequestCache.merged_at.isnot(None),
                    PullRequestCache.merged_at >= week_start,
                    PullRequestCache.merged_at <= week_end,
                )
            )
            .all()
        )
        for pr in merged:
            # is_merged may be NULL on legacy rows — fall back to merged_at presence
            is_merged = (
                bool(pr.is_merged) if pr.is_merged is not None else (pr.merged_at is not None)
            )
            author = cast(str, pr.author) if pr.author is not None else None
            if not is_merged or author is None:
                continue
            entry = counts[author]
            entry["prs_merged"] += 1

            # Issue #66: cycle time uses (merged_at - created_at).  Both must
            # exist and merged_at must be after created_at for the delta to be
            # meaningful.  Skip negative or null-edge cases silently.
            if pr.created_at is not None and pr.merged_at is not None:
                delta = pr.merged_at - pr.created_at
                hours = delta.total_seconds() / 3600.0
                if hours >= 0:
                    entry["_cycle_time_hours_total"] += hours
                    entry["_cycle_time_count"] += 1

        # 3. Reviewer / commenter aggregation across PRs touching the week
        # WHY: A reviewer is "active" in the week the PR is opened (proxy).
        # We use opened-in-week as the activity window which matches the
        # semantics of "PRs in the week where engineer appears in reviewers".
        for pr in opened:
            reviewers = _normalize_reviewers(pr.reviewers)
            for login in reviewers:
                if not login:
                    continue
                counts[login]["pr_reviews_given"] += 1
                counts[login]["pr_comments_given"] += 1
    finally:
        session.close()

    # Finalize derived metrics for each engineer (issue #66).
    # WHY: We compute averages and ratios *after* the loop so each engineer's
    # numerator/denominator pair stays atomic.  None values represent "no data"
    # (e.g. zero opens => merge_rate is undefined, not 0.0) so consumers can
    # distinguish "no activity" from "all opens, zero merges".
    finalized: dict[str, dict[str, Any]] = {}
    for engineer, entry in counts.items():
        prs_opened = entry["prs_opened"]
        prs_merged = entry["prs_merged"]

        # Engineers who only reviewed (no opens) get None rather than 0,
        # since a 0/0 ratio is mathematically undefined.
        pr_merge_rate: float | None = prs_merged / prs_opened if prs_opened > 0 else None

        cycle_count = entry.pop("_cycle_time_count")
        cycle_total = entry.pop("_cycle_time_hours_total")
        avg_cycle_time_hrs: float | None = cycle_total / cycle_count if cycle_count > 0 else None

        rev_count_prs = entry.pop("_revision_count_prs")
        rev_total = entry.pop("_revision_total")
        avg_revisions_per_pr: float | None = (
            rev_total / rev_count_prs if rev_count_prs > 0 else None
        )

        entry["pr_merge_rate"] = pr_merge_rate
        entry["avg_cycle_time_hrs"] = avg_cycle_time_hrs
        entry["avg_revisions_per_pr"] = avg_revisions_per_pr
        finalized[engineer] = entry

    logger.debug("Aggregated %d engineers for week %s", len(finalized), iso_week)
    return finalized


def _normalize_reviewers(raw: Any) -> list[str]:
    """Return a list[str] from a reviewers field that may be JSON-encoded text.

    SQLAlchemy's JSON column auto-deserializes on most backends but the
    underlying SQLite TEXT-storage path can occasionally return the raw string.
    We accept either form and always return a list.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if x]
        except (ValueError, TypeError):
            return []
    return []


def upsert_weekly_metrics(
    db: Database,
    iso_week: str,
    aggregates: dict[str, dict[str, Any]],
) -> int:
    """Upsert per-engineer aggregates for the given week.

    Uses SQLite's ``INSERT ... ON CONFLICT DO UPDATE`` so that re-running the
    same week is safe and produces identical row contents (with refreshed
    ``computed_at``).

    Args:
        db: Database wrapper.
        iso_week: ISO week label.
        aggregates: Output of :func:`aggregate_week`.

    Returns:
        The number of rows upserted (== ``len(aggregates)``).
    """
    if not aggregates:
        return 0

    now = datetime.now(timezone.utc)
    session = db.get_session()
    try:
        # We use session.merge for portability across SQLite backends,
        # which performs SELECT-then-INSERT-or-UPDATE based on PK.
        for engineer, vals in aggregates.items():
            session.merge(
                WeeklyPRMetrics(
                    engineer_identifier=engineer,
                    iso_week=iso_week,
                    prs_opened=vals.get("prs_opened", 0),
                    prs_merged=vals.get("prs_merged", 0),
                    pr_comments_given=vals.get("pr_comments_given", 0),
                    pr_reviews_given=vals.get("pr_reviews_given", 0),
                    # Issue #66: nullable derived metrics.  ``vals.get(..., None)``
                    # preserves the explicit None for "undefined" cases (e.g. no
                    # PRs opened => merge_rate undefined).
                    pr_merge_rate=vals.get("pr_merge_rate"),
                    avg_cycle_time_hrs=vals.get("avg_cycle_time_hrs"),
                    change_requests_received=vals.get("change_requests_received", 0),
                    avg_revisions_per_pr=vals.get("avg_revisions_per_pr"),
                    computed_at=now,
                )
            )
        session.commit()
    finally:
        session.close()

    return len(aggregates)


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------


@click.command(name="pr-metrics")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--week",
    type=str,
    default=None,
    help="Aggregate a single ISO week, e.g. 2026-W17",
)
@click.option(
    "--since",
    type=str,
    default=None,
    help="Backfill all weeks from this date (YYYY-MM-DD) through the current week",
)
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="none",
    help="Enable logging with specified level",
)
def pr_metrics_command(
    config_path: Path,
    week: str | None,
    since: str | None,
    log: str,
) -> None:
    """Aggregate weekly per-engineer PR metrics into the weekly_pr_metrics table.

    \b
    Reads from pull_request_cache (read-only) and upserts per-engineer
    aggregates into weekly_pr_metrics.  Re-running the same week is safe.

    \b
    EXAMPLES:
      # Aggregate the current ISO week
      gfa pr-metrics -c config.yaml

      # Aggregate a specific week
      gfa pr-metrics -c config.yaml --week 2026-W17

      # Backfill from January
      gfa pr-metrics -c config.yaml --since 2026-01-01
    """
    setup_logging(log, __name__)

    try:
        cfg = ConfigLoader.load(config_path)
        db_path = cfg.cache.directory / "gitflow_cache.db"
        db = Database(db_path)

        weeks = calculate_week_range(week=week, since=since)
        click.echo(f"Aggregating PR metrics for {len(weeks)} week(s)...")

        total_rows = 0
        for iso_week, week_start, week_end in weeks:
            aggregates = aggregate_week(db, iso_week, week_start, week_end)
            rows = upsert_weekly_metrics(db, iso_week, aggregates)
            total_rows += rows
            click.echo(f"  {iso_week}: {rows} engineer rows upserted")

        click.echo(f"\nDone. {total_rows} rows written to weekly_pr_metrics.")

    except ValueError as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(2)
    except Exception as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)


def register_pr_metrics_commands(cli_group: click.Group) -> None:
    """Register the pr-metrics command on the given Click group.

    Args:
        cli_group: The root ``gfa`` / ``gitflow-analytics`` Click group.
    """
    cli_group.add_command(pr_metrics_command)
