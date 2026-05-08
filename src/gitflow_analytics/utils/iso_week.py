"""ISO week string parsing utilities."""

from __future__ import annotations

import re
from datetime import date, timedelta

_ISO_WEEK_RE = re.compile(r"^(\d{4})-W(\d{2})$")


def parse_iso_week(week_str: str) -> tuple[date, date]:
    """Parse an ISO week string (YYYY-Www) into (monday, sunday) date pair.

    Args:
        week_str: ISO week string, e.g. "2026-W07"

    Returns:
        (week_start, week_end) as date objects (Monday-Sunday)

    Raises:
        ValueError: if the string is not a valid ISO week
    """
    m = _ISO_WEEK_RE.match(week_str.strip())
    if not m:
        raise ValueError(
            f"Invalid ISO week format {week_str!r}. Expected YYYY-Www (e.g. 2026-W07)."
        )
    year, week = int(m.group(1)), int(m.group(2))
    # Python's date.fromisocalendar(year, week, 1) = Monday of that week
    try:
        monday = date.fromisocalendar(year, week, 1)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO week {week_str!r}: {exc}") from exc
    sunday = monday + timedelta(days=6)
    return monday, sunday


def iso_week_range(from_week: str, to_week: str) -> tuple[date, date]:
    """Return (start_date, end_date) spanning from_week Monday to to_week Sunday.

    Raises:
        ValueError: if from_week > to_week
    """
    from_start, _ = parse_iso_week(from_week)
    _, to_end = parse_iso_week(to_week)
    if from_start > to_end:
        raise ValueError(f"--from {from_week} is after --to {to_week}.")
    return from_start, to_end
