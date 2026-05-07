"""Boilerplate / bulk auto-generated commit filter (Issue #28).

Detects developer-weeks whose line-count metrics exceed configured thresholds
(indicating bulk auto-generated commits — Swagger schemas, DB migrations,
vendored dependencies, generated clients, etc.) and applies one of three
actions to the resulting report data:

* ``flag`` — annotate developer rows with a ``boilerplate_flag: true`` marker
  and a ``boilerplate_label`` but leave them in all averages.
* ``exclude_from_averages`` — keep developer rows in per-developer output but
  exclude them when computing team/org averages.
* ``exclude`` — remove developer rows entirely from the report output.

WHY: Velocity metrics are only meaningful when outliers from bulk-generated
code are detectable and optionally removable. Hard-coding detection rules in
the report writer would mean every new report type re-implements the same
logic; extracting a dedicated filter keeps the contract explicit and testable.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal

from ..config.schema import BoilerplateFilterConfig

logger = logging.getLogger(__name__)

# Classification constants (avoid magic strings in callers).
CLASSIFICATION_CLEAN: Literal["clean"] = "clean"
CLASSIFICATION_FLAGGED: Literal["flagged"] = "flagged"
CLASSIFICATION_EXCLUDED: Literal["excluded"] = "excluded"

# Action constants
ACTION_FLAG = "flag"
ACTION_EXCLUDE_FROM_AVERAGES = "exclude_from_averages"
ACTION_EXCLUDE = "exclude"

# Annotation keys applied to report rows.
FIELD_BOILERPLATE_FLAG = "boilerplate_flag"
FIELD_BOILERPLATE_LABEL = "boilerplate_label"
FIELD_EXCLUDED_FROM_AVERAGES = "excluded_from_averages"


@dataclass(frozen=True)
class BoilerplateClassification:
    """Result of classifying a single developer-week metric row.

    Attributes:
        classification: One of ``"clean"``, ``"flagged"``, ``"excluded"``.
        reason: Short human-readable explanation (empty for ``"clean"``).
        avg_lines_per_commit: Computed average — may be 0.0 when no commits.
        total_lines: Total lines added considered (typically ``lines_added``).
    """

    classification: str
    reason: str
    avg_lines_per_commit: float
    total_lines: int


class BoilerplateFilter:
    """Detect and act on developer-weeks dominated by bulk auto-generated commits.

    Instantiate with a :class:`BoilerplateFilterConfig`.  When the config is
    disabled (``enabled=False``) all calls are no-ops: :meth:`classify` always
    returns ``"clean"`` and :meth:`apply` returns the input unchanged.
    """

    def __init__(self, config: BoilerplateFilterConfig | None) -> None:
        """Create a filter from a ``BoilerplateFilterConfig`` (or ``None``).

        Args:
            config: Parsed ``BoilerplateFilterConfig`` instance, or ``None`` to
                treat the filter as disabled.
        """
        self.config = config or BoilerplateFilterConfig()

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """``True`` when the filter is enabled in config."""
        return bool(self.config.enabled)

    def classify(self, metrics: dict[str, Any]) -> BoilerplateClassification:
        """Classify a single developer-week metrics row.

        A row is **flagged** when either threshold is strictly exceeded:

        * average lines per commit > ``avg_lines_per_commit_threshold``
        * total lines added       > ``total_lines_threshold``

        When the filter is disabled the result is always ``"clean"`` and no
        thresholds are checked.

        The returned classification is always ``"clean"`` or ``"flagged"``
        — a ``"flagged"`` row becomes ``"excluded"`` only after :meth:`apply`
        is called with an action of ``exclude`` or ``exclude_from_averages``.

        Args:
            metrics: Developer-week metrics dict.  Expected keys:

                * ``total_commits`` — number of commits that week (int)
                * ``lines_added``   — total added lines that week (int)

                Missing keys are treated as 0.  The dict itself is not
                mutated.

        Returns:
            :class:`BoilerplateClassification` describing whether the row
            should be flagged and why.
        """
        total_commits = int(metrics.get("total_commits") or 0)
        total_lines = int(metrics.get("lines_added") or 0)

        avg = (total_lines / total_commits) if total_commits > 0 else 0.0

        if not self.enabled:
            return BoilerplateClassification(
                classification=CLASSIFICATION_CLEAN,
                reason="",
                avg_lines_per_commit=avg,
                total_lines=total_lines,
            )

        reasons: list[str] = []
        if avg > self.config.avg_lines_per_commit_threshold:
            reasons.append(
                f"avg_lines_per_commit={avg:.1f} > {self.config.avg_lines_per_commit_threshold}"
            )
        if total_lines > self.config.total_lines_threshold:
            reasons.append(f"total_lines={total_lines} > {self.config.total_lines_threshold}")

        classification = CLASSIFICATION_FLAGGED if reasons else CLASSIFICATION_CLEAN
        return BoilerplateClassification(
            classification=classification,
            reason="; ".join(reasons),
            avg_lines_per_commit=avg,
            total_lines=total_lines,
        )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(self, report_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply the configured action to a list of developer-week metric rows.

        When the filter is disabled the input is returned unchanged (no copy).

        * ``flag`` — flagged rows are copied and annotated with
          ``boilerplate_flag: true`` and ``boilerplate_label: <flag_label>``.
          Clean rows pass through unchanged.

        * ``exclude_from_averages`` — flagged rows are annotated as above AND
          marked ``excluded_from_averages: true``.  Callers computing team
          averages should skip rows with this flag set.

        * ``exclude`` — flagged rows are dropped entirely from the output.

        Args:
            report_data: List of developer-week metric dicts.  Each dict
                should contain at least ``total_commits`` and ``lines_added``.

        Returns:
            A new list of metric dicts with the action applied.  The input
            list is never mutated.
        """
        if not self.enabled:
            return list(report_data)

        action = self.config.action
        label = self.config.flag_label

        result: list[dict[str, Any]] = []
        for row in report_data:
            classification = self.classify(row)

            if classification.classification == CLASSIFICATION_CLEAN:
                result.append(dict(row))
                continue

            # Row is flagged — branch on action.
            if action == ACTION_EXCLUDE:
                logger.debug(
                    "Boilerplate filter excluded developer=%s reason=%s",
                    row.get("developer_id") or row.get("developer_email"),
                    classification.reason,
                )
                continue

            annotated = dict(row)
            annotated[FIELD_BOILERPLATE_FLAG] = True
            annotated[FIELD_BOILERPLATE_LABEL] = label
            annotated["boilerplate_reason"] = classification.reason

            if action == ACTION_EXCLUDE_FROM_AVERAGES:
                annotated[FIELD_EXCLUDED_FROM_AVERAGES] = True

            # For ACTION_FLAG the row is kept in averages — no extra key set.
            result.append(annotated)

        return result

    # ------------------------------------------------------------------
    # Helpers for averaging consumers
    # ------------------------------------------------------------------

    @staticmethod
    def rows_for_averaging(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the subset of rows that should count toward team averages.

        Rows marked ``excluded_from_averages: true`` by :meth:`apply` are
        skipped; every other row is included.  This helper centralises the
        convention so every downstream aggregator doesn't re-implement the
        same filter.

        Args:
            rows: A list of developer-week metric dicts, typically the output
                of :meth:`apply`.

        Returns:
            A new list containing only rows eligible for averaging.
        """
        return [r for r in rows if not r.get(FIELD_EXCLUDED_FROM_AVERAGES)]


# ---------------------------------------------------------------------------
# Pipeline integration helpers
# ---------------------------------------------------------------------------


def _week_start(ts: Any) -> date | None:
    """Return the Monday (ISO week start) date for a timestamp.

    Args:
        ts: A ``datetime``, ``date``, or anything truthy with a ``.date()`` /
            ``.weekday()`` method.  ``None`` returns ``None``.

    Returns:
        The Monday-anchored ``date`` for the timestamp's week, or ``None`` if
        ``ts`` cannot be converted to a date.
    """
    if ts is None:
        return None
    if isinstance(ts, datetime):
        d = ts.date()
    elif isinstance(ts, date):
        d = ts
    elif hasattr(ts, "date"):
        try:
            d = ts.date()
        except Exception:
            return None
    else:
        return None
    return d - timedelta(days=d.weekday())


def _developer_key(row: dict[str, Any]) -> str:
    """Derive a stable per-developer identity key from a proxy row.

    Uses ``developer_id`` if present, falling back through ``developer_email``,
    ``author_email``, ``developer_name``, ``author``. Returns ``"unknown"``
    when no identity key is available so anonymous rows aggregate together.

    Args:
        row: A commit-proxy or metric dict.

    Returns:
        A non-empty string suitable for use as a dict key.
    """
    for key in (
        "developer_id",
        "developer_email",
        "author_email",
        "developer_name",
        "author",
    ):
        val = row.get(key)
        if val:
            return str(val)
    return "unknown"


def aggregate_weekly_developer_metrics(
    commit_proxy: list[dict[str, Any]],
) -> dict[tuple[str, date | None], dict[str, Any]]:
    """Aggregate per-commit proxy rows into per-developer-per-week metrics.

    WHY: The boilerplate filter operates on weekly aggregates (thresholds are
    expressed in lines-per-commit-per-week and total-lines-per-week).
    The pipeline's commit_proxy list has one row per commit, so we need
    a lightweight weekly fold before classification.

    Args:
        commit_proxy: List of per-commit proxy dicts.  Each row should have
            ``timestamp`` (or be undated — those bucket into the ``None``
            week), ``total_commits`` (typically 1), and ``lines_added``.

    Returns:
        Mapping of ``(developer_key, week_start_date_or_None)`` →
        aggregated weekly metric dict.  Each value has ``developer_id``,
        ``week_start`` (``None`` when undated), ``total_commits``, and
        ``lines_added`` keys.
    """
    buckets: dict[tuple[str, date | None], dict[str, Any]] = defaultdict(
        lambda: {
            "developer_id": "",
            "week_start": None,
            "total_commits": 0,
            "lines_added": 0,
            "lines_deleted": 0,
        }
    )

    for row in commit_proxy:
        dev = _developer_key(row)
        week = _week_start(row.get("timestamp"))
        key = (dev, week)
        bucket = buckets[key]
        bucket["developer_id"] = dev
        bucket["week_start"] = week
        bucket["total_commits"] += int(row.get("total_commits") or 1)
        bucket["lines_added"] += int(row.get("lines_added") or 0)
        bucket["lines_deleted"] += int(row.get("lines_deleted") or 0)

    return dict(buckets)


def apply_boilerplate_filter_to_commits(
    bp_filter: BoilerplateFilter,
    commit_proxy: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply a boilerplate filter to a list of per-commit proxy rows.

    Aggregates the commit proxy to per-developer-per-week buckets, classifies
    each bucket, then maps the classification back down to the original
    commit rows according to the configured action:

    * ``flag`` — commits in flagged dev-weeks are annotated with
      ``boilerplate_flag: true`` and ``boilerplate_label``.
    * ``exclude_from_averages`` — commits in flagged dev-weeks are annotated
      with ``excluded_from_averages: true`` (team aggregator ignores them
      when computing averages but per-developer output still includes them).
    * ``exclude`` — commits in flagged dev-weeks are dropped entirely.

    When the filter is disabled a shallow copy of the input is returned.

    Args:
        bp_filter: A :class:`BoilerplateFilter` instance (may be disabled).
        commit_proxy: List of per-commit proxy dicts.

    Returns:
        A new list of (possibly annotated, possibly filtered) commit dicts.
        Original rows are never mutated.
    """
    if not bp_filter.enabled:
        return list(commit_proxy)

    weekly = aggregate_weekly_developer_metrics(commit_proxy)

    # Classify each dev-week; collect those that are flagged.
    flagged: dict[tuple[str, date | None], BoilerplateClassification] = {}
    for key, metrics in weekly.items():
        classification = bp_filter.classify(metrics)
        if classification.classification == CLASSIFICATION_FLAGGED:
            flagged[key] = classification

    if not flagged:
        return list(commit_proxy)

    action = bp_filter.config.action
    label = bp_filter.config.flag_label
    result: list[dict[str, Any]] = []

    for row in commit_proxy:
        key = (_developer_key(row), _week_start(row.get("timestamp")))
        classification = flagged.get(key)

        if classification is None:
            result.append(dict(row))
            continue

        if action == ACTION_EXCLUDE:
            # Drop this commit from the output entirely.
            continue

        annotated = dict(row)
        annotated[FIELD_BOILERPLATE_FLAG] = True
        annotated[FIELD_BOILERPLATE_LABEL] = label
        annotated["boilerplate_reason"] = classification.reason

        if action == ACTION_EXCLUDE_FROM_AVERAGES:
            annotated[FIELD_EXCLUDED_FROM_AVERAGES] = True

        result.append(annotated)

    return result
