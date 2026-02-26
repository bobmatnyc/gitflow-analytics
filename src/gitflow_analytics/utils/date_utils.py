"""Date utility functions for week boundary calculations.

These are pure date math functions used across the CLI and report generation
to ensure consistent Monday-aligned week boundaries.
"""

from datetime import datetime, timedelta, timezone


def get_week_start(date: datetime) -> datetime:
    """Get Monday of the week for a given date, ensuring week boundary alignment.

    WHY: This provides consistent week boundary calculation across CLI date calculation
    and report generation, ensuring that filenames and week displays align properly.

    DESIGN DECISION: Always returns Monday 00:00:00 UTC as the week start to ensure
    consistent week boundaries regardless of the input date's day of week or time.

    Args:
        date: Input date (timezone-aware or naive)

    Returns:
        Monday of the week containing the input date, as timezone-aware UTC datetime
        at 00:00:00 (start of day)
    """
    # Ensure timezone consistency - convert to UTC if needed
    if hasattr(date, "tzinfo") and date.tzinfo is not None:
        # Keep timezone-aware but ensure it's UTC
        if date.tzinfo != timezone.utc:
            date = date.astimezone(timezone.utc)
    else:
        # Convert naive datetime to UTC timezone-aware
        date = date.replace(tzinfo=timezone.utc)

    # Get days since Monday (0=Monday, 6=Sunday)
    days_since_monday = date.weekday()

    # Calculate Monday of this week
    monday = date - timedelta(days=days_since_monday)

    # Reset to start of day (00:00:00)
    result = monday.replace(hour=0, minute=0, second=0, microsecond=0)

    return result


def get_week_end(date: datetime) -> datetime:
    """Get Sunday end of the week for a given date.

    WHY: Provides the end boundary for week ranges to ensure complete week coverage
    in analysis periods and consistent date range calculations.

    Args:
        date: Input date (timezone-aware or naive)

    Returns:
        Sunday 23:59:59.999999 UTC of the week containing the input date
    """
    # Get the Monday start of this week
    week_start = get_week_start(date)

    # Add 6 days to get to Sunday, and set to end of day
    week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999)

    return week_end


def get_monday_aligned_start(weeks_back: int) -> datetime:
    """Calculate the Monday-aligned start date for N complete weeks of analysis.

    Bug 3 fix: Both the fetch sub-command and the main batch-mode path must use
    identical Monday-anchored week boundaries.  Previously the fetch path used a
    raw timedelta rollback (not aligned to Monday) while the batch-mode path
    correctly anchored to the last complete Monday-to-Sunday week.

    Logic matches the batch-mode calculation in the main `analyze` command:
      - Find the Monday of the *current* week.
      - Step back one week to get the last *complete* week's Monday.
      - Step back (weeks_back - 1) more weeks for the start of the N-week window.

    Args:
        weeks_back: Number of complete Monday-to-Sunday weeks to include.

    Returns:
        Monday 00:00:00 UTC marking the start of the analysis window.
    """
    current_time = datetime.now(timezone.utc)
    current_week_start = get_week_start(current_time)
    last_complete_week_start = current_week_start - timedelta(weeks=1)
    return last_complete_week_start - timedelta(weeks=weeks_back - 1)
