"""14-day code churn rate calculator.

WHY: Churn rate (lines modified/deleted within 14 days of being written) is
the strongest available proxy for code quality without external tooling.
Higher churn = more rework = lower quality. GitClear research shows AI-assisted
code has 2-3x higher churn rates, making this a useful quality signal alongside
AI adoption metrics.
"""

from datetime import date, timedelta
from typing import Any


def calculate_churn_rate_14d(
    developer_id: str,
    week_start: date,
    daily_metrics: list[dict[str, Any]],
) -> float:
    """Calculate 14-day churn rate for a developer starting from a week.

    Algorithm:
    1. Sum lines_added for the developer in [week_start, week_start+6]
    2. Sum lines_deleted for the developer in [week_start+7, week_start+20] (next 14 days)
    3. churn_rate = min(deletions_in_next_14d / lines_added_in_week, 1.0)

    Returns 0.0 if no lines were added (no churn possible).

    Args:
        developer_id: Developer canonical ID
        week_start: Monday of the week being analyzed
        daily_metrics: List of daily_metrics dicts with date, developer_id,
            lines_added, lines_deleted

    Returns:
        Churn rate as float in [0.0, 1.0]
    """
    week_end = week_start + timedelta(days=6)
    churn_window_start = week_start + timedelta(days=7)
    churn_window_end = week_start + timedelta(days=20)

    # Filter to this developer's records
    dev_metrics = [m for m in daily_metrics if m.get("developer_id") == developer_id]

    # Lines added during the week
    lines_added = sum(
        m.get("lines_added", 0) or 0
        for m in dev_metrics
        if week_start <= _to_date(m.get("date")) <= week_end
    )

    if lines_added == 0:
        return 0.0

    # Lines deleted in the following 14 days (proxy for churn of previous week's code)
    lines_deleted_next = sum(
        m.get("lines_deleted", 0) or 0
        for m in dev_metrics
        if churn_window_start <= _to_date(m.get("date")) <= churn_window_end
    )

    return min(lines_deleted_next / lines_added, 1.0)


def _to_date(d: Any) -> date:
    """Convert various date representations to date object."""
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return date.fromisoformat(str(d)[:10])
    return date.today()  # fallback


def calculate_org_churn_rate(
    week_start: date,
    daily_metrics: list[dict[str, Any]],
) -> float:
    """Calculate org-level 14-day churn rate for a week.

    Weighted average: total_churned_lines / total_lines_added across all
    developers. This avoids skew from outlier developers with very low volume.

    Args:
        week_start: Monday of the week being analyzed
        daily_metrics: List of daily_metrics dicts with date, developer_id,
            lines_added, lines_deleted

    Returns:
        Org-level churn rate as float in [0.0, 1.0]
    """
    week_end = week_start + timedelta(days=6)
    churn_window_start = week_start + timedelta(days=7)
    churn_window_end = week_start + timedelta(days=20)

    total_lines_added = sum(
        m.get("lines_added", 0) or 0
        for m in daily_metrics
        if week_start <= _to_date(m.get("date")) <= week_end
    )

    if total_lines_added == 0:
        return 0.0

    total_deleted_next = sum(
        m.get("lines_deleted", 0) or 0
        for m in daily_metrics
        if churn_window_start <= _to_date(m.get("date")) <= churn_window_end
    )

    return min(total_deleted_next / total_lines_added, 1.0)
