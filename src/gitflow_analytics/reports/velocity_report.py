"""Velocity report generator — PR cycle time, throughput, revision rate.

Computes per-developer and per-week velocity metrics from merged PR data
and writes a velocity_summary.json file to the output directory.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _to_dt(val: Any) -> datetime | None:
    """Coerce a value to a timezone-aware datetime, or return None."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _cycle_time_hrs(pr: dict[str, Any]) -> float | None:
    """Return PR open→merge duration in hours, or None if data is missing."""
    created = _to_dt(pr.get("created_at"))
    merged = _to_dt(pr.get("merged_at"))
    if not created or not merged:
        return None
    delta = (merged - created).total_seconds() / 3600
    return delta if delta >= 0 else None


def _week_start(dt: datetime | date) -> date:
    """Return the Monday of the ISO week containing *dt*."""
    d = dt.date() if isinstance(dt, datetime) else dt
    return d - timedelta(days=d.weekday())


class VelocityReportGenerator:
    """Compute and write velocity metrics from PR data.

    Attributes:
        min_hrs: Cycle times below this threshold are excluded as outliers.
        max_hrs: Cycle times above this threshold are excluded as outliers.
        top_n: Number of fastest / slowest PRs to include in top lists.
    """

    def __init__(self, config: Any = None) -> None:
        """Initialise the generator from an optional VelocityConfig object."""
        self.min_hrs: float = getattr(config, "cycle_time_outlier_min_hrs", 0.5) if config else 0.5
        self.max_hrs: float = (
            getattr(config, "cycle_time_outlier_max_hrs", 720.0) if config else 720.0
        )
        self.top_n: int = getattr(config, "top_n", 5) if config else 5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prs: list[dict[str, Any]],
        output_dir: Path,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, Any]:
        """Compute velocity metrics and write velocity_summary.json.

        Args:
            prs: Raw PR dicts loaded from pull_request_cache.
            output_dir: Directory where velocity_summary.json will be written.
            start_date: Inclusive lower bound on merged_at date (optional).
            end_date: Inclusive upper bound on merged_at date (optional).

        Returns:
            The summary dict that was serialised to disk.
        """
        merged = self._filter_merged(prs, start_date, end_date)
        per_developer = self._by_developer(merged)
        per_week = self._by_week(merged)
        top_fastest, top_slowest = self._top_prs(merged)

        summary: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_prs_analyzed": len(merged),
            "per_developer": per_developer,
            "per_week": per_week,
            "top_fastest_prs": top_fastest,
            "top_slowest_prs": top_slowest,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "velocity_summary.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("Wrote velocity_summary.json (%d merged PRs)", len(merged))
        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_merged(
        self,
        prs: list[dict[str, Any]],
        start_date: date | None,
        end_date: date | None,
    ) -> list[dict[str, Any]]:
        """Return only merged PRs within the optional date window."""
        merged = [pr for pr in prs if pr.get("is_merged") or pr.get("pr_state") == "merged"]
        if not (start_date or end_date):
            return merged

        filtered: list[dict[str, Any]] = []
        for pr in merged:
            merged_at = _to_dt(pr.get("merged_at"))
            if merged_at:
                d = merged_at.date()
                if start_date and d < start_date:
                    continue
                if end_date and d > end_date:
                    continue
            filtered.append(pr)
        return filtered

    def _cycle_times(self, prs: list[dict[str, Any]]) -> list[float]:
        """Return cycle times (hours) for *prs*, excluding outliers."""
        times: list[float] = []
        for pr in prs:
            ct = _cycle_time_hrs(pr)
            if ct is not None and self.min_hrs <= ct <= self.max_hrs:
                times.append(ct)
        return times

    def _pr_metrics(self, prs: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregated velocity metrics for a cohort of PRs."""
        cycle_times = self._cycle_times(prs)
        revision_counts = [pr.get("revision_count") or 0 for pr in prs]
        sp = sum((pr.get("story_points") or 0) for pr in prs)
        ttfr = [
            pr["time_to_first_review_hours"]
            for pr in prs
            if pr.get("time_to_first_review_hours") is not None
        ]
        return {
            "prs_merged": len(prs),
            "avg_cycle_time_hrs": (round(statistics.mean(cycle_times), 2) if cycle_times else None),
            "median_cycle_time_hrs": (
                round(statistics.median(cycle_times), 2) if cycle_times else None
            ),
            "avg_revision_count": (
                round(statistics.mean(revision_counts), 2) if revision_counts else 0.0
            ),
            "story_points_delivered": sp,
            "avg_time_to_first_review_hrs": (round(statistics.mean(ttfr), 2) if ttfr else None),
        }

    def _by_developer(self, prs: list[dict[str, Any]]) -> dict[str, Any]:
        """Group *prs* by author and return per-developer metrics."""
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pr in prs:
            author = pr.get("author") or "unknown"
            buckets[author].append(pr)
        return {dev: self._pr_metrics(dev_prs) for dev, dev_prs in sorted(buckets.items())}

    def _by_week(self, prs: list[dict[str, Any]]) -> dict[str, Any]:
        """Group *prs* by ISO week (Monday-aligned) of their merged_at date."""
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pr in prs:
            merged_at = _to_dt(pr.get("merged_at"))
            if merged_at:
                week = str(_week_start(merged_at))
                buckets[week].append(pr)
        return {week: self._pr_metrics(week_prs) for week, week_prs in sorted(buckets.items())}

    def _top_prs(
        self, prs: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return the top-N fastest and slowest PRs by cycle time."""
        with_ct = [(pr, _cycle_time_hrs(pr)) for pr in prs]
        with_ct = [
            (pr, ct) for pr, ct in with_ct if ct is not None and self.min_hrs <= ct <= self.max_hrs
        ]
        with_ct.sort(key=lambda x: x[1])  # ascending by cycle time

        def _fmt(pr: dict[str, Any], ct: float) -> dict[str, Any]:
            return {
                "title": pr.get("title", ""),
                "author": pr.get("author", ""),
                "repo": pr.get("repo_path", ""),
                "cycle_time_hrs": round(ct, 2),
                "merged_at": str(pr.get("merged_at", "")),
            }

        fastest = [_fmt(pr, ct) for pr, ct in with_ct[: self.top_n]]
        slowest = [_fmt(pr, ct) for pr, ct in with_ct[-self.top_n :][::-1]]
        return fastest, slowest
