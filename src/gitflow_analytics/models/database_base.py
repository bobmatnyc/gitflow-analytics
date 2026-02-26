"""Base declarative class and utility for GitFlow Analytics database models."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import declarative_base

Base: Any = declarative_base()


def utcnow_tz_aware() -> datetime:
    """Return current UTC time as timezone-aware datetime.

    WHY: SQLAlchemy DateTime(timezone=True) requires timezone-aware datetimes.
    Using timezone-naive datetime.utcnow() causes query mismatches when filtering
    by timezone-aware date ranges.

    Returns:
        Timezone-aware datetime in UTC
    """
    return datetime.now(timezone.utc)

