"""Tests for 14-day code churn rate calculator.

Validates:
1. Basic churn rate calculation for a developer
2. Zero lines_added returns 0.0
3. Churn is capped at 1.0
4. Org-level calculation (weighted average across developers)
5. Date range boundaries (only week and churn window dates count)
6. _to_date helper handles str, date, and fallback
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from gitflow_analytics.core.churn_calculator import (
    _to_date,
    calculate_churn_rate_14d,
    calculate_org_churn_rate,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

ALICE = "alice@example.com"
BOB = "bob@example.com"


def _metric(developer_id: str, day: date, lines_added: int, lines_deleted: int) -> dict:
    return {
        "developer_id": developer_id,
        "date": day,
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
    }


# ---------------------------------------------------------------------------
# 1. Basic churn rate calculation
# ---------------------------------------------------------------------------


class TestCalculateChurnRate14d:
    """Basic churn rate scenarios for calculate_churn_rate_14d."""

    def test_zero_churn_when_no_deletions_in_window(self) -> None:
        week_start = date(2025, 1, 6)  # Monday
        metrics = [
            _metric(ALICE, date(2025, 1, 6), lines_added=100, lines_deleted=0),
            _metric(ALICE, date(2025, 1, 7), lines_added=50, lines_deleted=0),
        ]
        rate = calculate_churn_rate_14d(ALICE, week_start, metrics)
        assert rate == 0.0

    def test_churn_rate_basic(self) -> None:
        """50 lines added in week, 25 deleted in next 14 days → 0.5."""
        week_start = date(2025, 1, 6)
        metrics = [
            # Week (days 0-6)
            _metric(ALICE, date(2025, 1, 6), lines_added=50, lines_deleted=0),
            # Churn window (days 7-20): Jan 13 = day 7
            _metric(ALICE, date(2025, 1, 13), lines_added=0, lines_deleted=25),
        ]
        rate = calculate_churn_rate_14d(ALICE, week_start, metrics)
        assert rate == pytest.approx(0.5)

    def test_full_churn_returns_one(self) -> None:
        """Deletions equal to additions → churn rate exactly 1.0."""
        week_start = date(2025, 1, 6)
        metrics = [
            _metric(ALICE, date(2025, 1, 6), lines_added=100, lines_deleted=0),
            _metric(ALICE, date(2025, 1, 15), lines_added=0, lines_deleted=100),
        ]
        rate = calculate_churn_rate_14d(ALICE, week_start, metrics)
        assert rate == 1.0

    def test_ignores_other_developers(self) -> None:
        """Only the target developer's lines should be counted."""
        week_start = date(2025, 1, 6)
        metrics = [
            _metric(ALICE, date(2025, 1, 6), lines_added=100, lines_deleted=0),
            _metric(BOB, date(2025, 1, 6), lines_added=500, lines_deleted=0),
            _metric(BOB, date(2025, 1, 13), lines_added=0, lines_deleted=400),
            # Alice has no deletions in churn window → 0.0
        ]
        rate = calculate_churn_rate_14d(ALICE, week_start, metrics)
        assert rate == 0.0

    def test_churn_window_includes_day_7_and_day_20(self) -> None:
        """Churn window is [week_start+7, week_start+20] inclusive."""
        week_start = date(2025, 1, 6)
        # day 7 = Jan 13, day 20 = Jan 26
        metrics = [
            _metric(ALICE, date(2025, 1, 6), lines_added=100, lines_deleted=0),
            _metric(ALICE, date(2025, 1, 13), lines_added=0, lines_deleted=10),  # day 7, included
            _metric(ALICE, date(2025, 1, 26), lines_added=0, lines_deleted=10),  # day 20, included
        ]
        rate = calculate_churn_rate_14d(ALICE, week_start, metrics)
        assert rate == pytest.approx(0.2)  # 20 / 100

    def test_day_before_window_excluded(self) -> None:
        """Deletions on day 6 (same week) must not count as churn."""
        week_start = date(2025, 1, 6)
        metrics = [
            _metric(ALICE, date(2025, 1, 6), lines_added=100, lines_deleted=0),
            # Day 12 = week_start + 6: still inside the base week, not churn window
            _metric(ALICE, date(2025, 1, 12), lines_added=0, lines_deleted=50),
        ]
        rate = calculate_churn_rate_14d(ALICE, week_start, metrics)
        assert rate == 0.0

    def test_day_after_window_excluded(self) -> None:
        """Deletions on day 21 (outside churn window) must not be counted."""
        week_start = date(2025, 1, 6)
        metrics = [
            _metric(ALICE, date(2025, 1, 6), lines_added=100, lines_deleted=0),
            # Day 21 = Jan 27, outside [day 7, day 20]
            _metric(ALICE, date(2025, 1, 27), lines_added=0, lines_deleted=100),
        ]
        rate = calculate_churn_rate_14d(ALICE, week_start, metrics)
        assert rate == 0.0

    def test_none_values_treated_as_zero(self) -> None:
        """None for lines_added or lines_deleted must not raise errors."""
        week_start = date(2025, 1, 6)
        metrics = [
            {
                "developer_id": ALICE,
                "date": date(2025, 1, 6),
                "lines_added": None,
                "lines_deleted": None,
            },
            _metric(ALICE, date(2025, 1, 13), lines_added=0, lines_deleted=10),
        ]
        # lines_added is None → treated as 0 → return 0.0 immediately
        rate = calculate_churn_rate_14d(ALICE, week_start, metrics)
        assert rate == 0.0


# ---------------------------------------------------------------------------
# 2. Zero lines_added returns 0.0
# ---------------------------------------------------------------------------


class TestZeroLinesAdded:
    def test_no_additions_returns_zero(self) -> None:
        week_start = date(2025, 1, 6)
        metrics = [
            _metric(ALICE, date(2025, 1, 6), lines_added=0, lines_deleted=50),
        ]
        rate = calculate_churn_rate_14d(ALICE, week_start, metrics)
        assert rate == 0.0

    def test_empty_metrics_returns_zero(self) -> None:
        week_start = date(2025, 1, 6)
        rate = calculate_churn_rate_14d(ALICE, week_start, [])
        assert rate == 0.0

    def test_unknown_developer_returns_zero(self) -> None:
        week_start = date(2025, 1, 6)
        metrics = [_metric(BOB, date(2025, 1, 6), lines_added=200, lines_deleted=0)]
        rate = calculate_churn_rate_14d(ALICE, week_start, metrics)
        assert rate == 0.0


# ---------------------------------------------------------------------------
# 3. Churn capped at 1.0
# ---------------------------------------------------------------------------


class TestChurnCappedAtOne:
    def test_churn_capped_when_deletions_exceed_additions(self) -> None:
        """If deletions in churn window far exceed additions, cap at 1.0."""
        week_start = date(2025, 1, 6)
        metrics = [
            _metric(ALICE, date(2025, 1, 6), lines_added=10, lines_deleted=0),
            _metric(ALICE, date(2025, 1, 13), lines_added=0, lines_deleted=9999),
        ]
        rate = calculate_churn_rate_14d(ALICE, week_start, metrics)
        assert rate == 1.0

    def test_churn_never_exceeds_one(self) -> None:
        week_start = date(2025, 3, 3)
        metrics = [
            _metric(ALICE, date(2025, 3, 3), lines_added=1, lines_deleted=0),
            _metric(ALICE, date(2025, 3, 10), lines_added=0, lines_deleted=1000),
        ]
        rate = calculate_churn_rate_14d(ALICE, week_start, metrics)
        assert 0.0 <= rate <= 1.0
        assert rate == 1.0


# ---------------------------------------------------------------------------
# 4. Org-level calculation
# ---------------------------------------------------------------------------


class TestCalculateOrgChurnRate:
    def test_org_churn_weighted_average(self) -> None:
        """Org churn = total_deleted_next / total_added_in_week across all devs."""
        week_start = date(2025, 1, 6)
        metrics = [
            # Alice: 100 added, 30 churned
            _metric(ALICE, date(2025, 1, 6), lines_added=100, lines_deleted=0),
            _metric(ALICE, date(2025, 1, 13), lines_added=0, lines_deleted=30),
            # Bob: 200 added, 70 churned
            _metric(BOB, date(2025, 1, 7), lines_added=200, lines_deleted=0),
            _metric(BOB, date(2025, 1, 14), lines_added=0, lines_deleted=70),
        ]
        rate = calculate_org_churn_rate(week_start, metrics)
        # (30 + 70) / (100 + 200) = 100/300 ≈ 0.333
        assert rate == pytest.approx(100 / 300, rel=1e-6)

    def test_org_churn_no_additions_returns_zero(self) -> None:
        week_start = date(2025, 1, 6)
        metrics = [
            _metric(ALICE, date(2025, 1, 13), lines_added=0, lines_deleted=50),
        ]
        rate = calculate_org_churn_rate(week_start, metrics)
        assert rate == 0.0

    def test_org_churn_capped_at_one(self) -> None:
        week_start = date(2025, 1, 6)
        metrics = [
            _metric(ALICE, date(2025, 1, 6), lines_added=10, lines_deleted=0),
            _metric(BOB, date(2025, 1, 6), lines_added=5, lines_deleted=0),
            _metric(ALICE, date(2025, 1, 13), lines_added=0, lines_deleted=50000),
        ]
        rate = calculate_org_churn_rate(week_start, metrics)
        assert rate == 1.0

    def test_org_churn_empty_metrics_returns_zero(self) -> None:
        rate = calculate_org_churn_rate(date(2025, 1, 6), [])
        assert rate == 0.0


# ---------------------------------------------------------------------------
# 5. Date boundary tests
# ---------------------------------------------------------------------------


class TestDateBoundaries:
    def test_week_start_itself_is_included(self) -> None:
        """The first day of the week (week_start) must be included in the base window."""
        week_start = date(2025, 2, 3)  # Monday
        metrics = [
            _metric(ALICE, week_start, lines_added=80, lines_deleted=0),
            _metric(
                ALICE,
                week_start + __import__("datetime").timedelta(days=7),
                lines_added=0,
                lines_deleted=40,
            ),
        ]
        rate = calculate_churn_rate_14d(ALICE, week_start, metrics)
        assert rate == pytest.approx(0.5)

    def test_week_end_day6_included(self) -> None:
        """Day 6 (week_end = week_start + 6) is the last day of the base window."""
        week_start = date(2025, 2, 3)
        from datetime import timedelta

        metrics = [
            _metric(ALICE, week_start + timedelta(days=6), lines_added=60, lines_deleted=0),
            _metric(ALICE, week_start + timedelta(days=7), lines_added=0, lines_deleted=30),
        ]
        rate = calculate_churn_rate_14d(ALICE, week_start, metrics)
        assert rate == pytest.approx(0.5)

    def test_churn_day20_boundary(self) -> None:
        """Day 20 (week_start + 20) is the last day of the churn window."""
        week_start = date(2025, 2, 3)
        from datetime import timedelta

        metrics = [
            _metric(ALICE, week_start, lines_added=100, lines_deleted=0),
            _metric(ALICE, week_start + timedelta(days=20), lines_added=0, lines_deleted=100),
        ]
        rate = calculate_churn_rate_14d(ALICE, week_start, metrics)
        assert rate == 1.0

    def test_churn_day21_excluded(self) -> None:
        """Day 21 (week_start + 21) is beyond the churn window."""
        week_start = date(2025, 2, 3)
        from datetime import timedelta

        metrics = [
            _metric(ALICE, week_start, lines_added=100, lines_deleted=0),
            _metric(ALICE, week_start + timedelta(days=21), lines_added=0, lines_deleted=100),
        ]
        rate = calculate_churn_rate_14d(ALICE, week_start, metrics)
        assert rate == 0.0


# ---------------------------------------------------------------------------
# 6. _to_date helper
# ---------------------------------------------------------------------------


class TestToDate:
    def test_date_passthrough(self) -> None:
        d = date(2025, 6, 15)
        assert _to_date(d) == d

    def test_string_iso_format(self) -> None:
        assert _to_date("2025-06-15") == date(2025, 6, 15)

    def test_string_with_time_component(self) -> None:
        """Strings like '2025-06-15 10:30:00' should use only the date part."""
        assert _to_date("2025-06-15 10:30:00") == date(2025, 6, 15)

    def test_datetime_object(self) -> None:
        """datetime is a subclass of date, so it should pass through directly."""
        dt = datetime(2025, 6, 15, 12, 0, 0)
        result = _to_date(dt)
        # datetime.date() would strip time; since datetime IS a date subclass,
        # the isinstance(d, date) branch fires and returns the datetime itself.
        assert result == dt

    def test_none_returns_today(self) -> None:
        """None falls to the fallback branch and returns date.today()."""
        result = _to_date(None)
        assert result == date.today()

    def test_unknown_type_returns_today(self) -> None:
        """Non-date, non-str types fall to the fallback branch."""
        result = _to_date(12345)
        assert result == date.today()
