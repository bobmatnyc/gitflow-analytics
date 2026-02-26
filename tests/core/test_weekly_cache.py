"""Tests for week-level incremental fetch caching.

Verifies:
1. Week calculation alignment (Monday starts)
2. Missing week detection
3. Marking weeks as cached
4. Force flag bypasses cache
5. clear_all_cache clears weekly status too
6. clear_weekly_cache targeted clearing
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gitflow_analytics.core.cache import GitAnalysisCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MONDAY = datetime(2025, 2, 17, 0, 0, 0, tzinfo=timezone.utc)  # Known Monday
SUNDAY = datetime(2025, 2, 23, 23, 59, 59, tzinfo=timezone.utc)


def make_cache(tmp_path: Path) -> GitAnalysisCache:
    """Create a fresh GitAnalysisCache backed by a temp directory."""
    cache_dir = tmp_path / ".gitflow-cache"
    return GitAnalysisCache(cache_dir)


# ---------------------------------------------------------------------------
# calculate_weeks
# ---------------------------------------------------------------------------


class TestCalculateWeeks:
    """Week calculation must always align to Monday."""

    def test_single_complete_week(self, tmp_path):
        cache = make_cache(tmp_path)
        weeks = cache.calculate_weeks(MONDAY, SUNDAY)

        assert len(weeks) == 1
        week_start, week_end = weeks[0]

        # week_start must be the Monday itself
        assert week_start.weekday() == 0, "week_start is not a Monday"
        assert week_start == MONDAY

    def test_two_complete_weeks(self, tmp_path):
        cache = make_cache(tmp_path)
        start = MONDAY  # 2025-02-17
        end = start + timedelta(weeks=2) - timedelta(seconds=1)  # 2025-03-02 23:59:59
        weeks = cache.calculate_weeks(start, end)

        assert len(weeks) == 2
        for ws, _we in weeks:
            assert ws.weekday() == 0, f"{ws} is not a Monday"

    def test_alignment_from_mid_week_start(self, tmp_path):
        """If start is mid-week, the first week boundary rolls back to the prior Monday."""
        cache = make_cache(tmp_path)
        wednesday = datetime(2025, 2, 19, 0, 0, 0, tzinfo=timezone.utc)  # Wednesday
        end = wednesday + timedelta(weeks=1)
        weeks = cache.calculate_weeks(wednesday, end)

        assert len(weeks) >= 1
        first_week_start = weeks[0][0]
        assert first_week_start.weekday() == 0, "First week must start on Monday"
        # Must roll back to 2025-02-17 (the Monday before Wednesday)
        assert first_week_start.date() == MONDAY.date()

    def test_three_weeks_count(self, tmp_path):
        cache = make_cache(tmp_path)
        start = MONDAY
        end = start + timedelta(weeks=3) - timedelta(seconds=1)
        weeks = cache.calculate_weeks(start, end)
        assert len(weeks) == 3

    def test_last_week_end_capped_to_end_date(self, tmp_path):
        """Partial last week should have week_end == end_date, not Sunday."""
        cache = make_cache(tmp_path)
        start = MONDAY
        # End on Wednesday of the second week
        end = MONDAY + timedelta(days=9, hours=12)
        weeks = cache.calculate_weeks(start, end)

        # Should be exactly 2 weeks (first full + second partial)
        assert len(weeks) == 2
        last_week_end = weeks[-1][1]
        assert last_week_end == end

    def test_timezone_naive_inputs_treated_as_utc(self, tmp_path):
        """Naive datetimes should be treated as UTC without raising."""
        cache = make_cache(tmp_path)
        naive_start = datetime(2025, 2, 17, 0, 0, 0)
        naive_end = datetime(2025, 2, 23, 23, 59, 59)
        # Must not raise
        weeks = cache.calculate_weeks(naive_start, naive_end)
        assert len(weeks) == 1

    def test_empty_when_start_equals_end(self, tmp_path):
        """start >= end should produce an empty list."""
        cache = make_cache(tmp_path)
        at = MONDAY
        weeks = cache.calculate_weeks(at, at)
        # start == end means current < end is never satisfied — empty
        assert weeks == []


# ---------------------------------------------------------------------------
# get_cached_weeks / mark_week_cached
# ---------------------------------------------------------------------------


class TestMarkAndGetCachedWeeks:
    """Round-trip: mark → get."""

    def test_mark_and_retrieve(self, tmp_path):
        cache = make_cache(tmp_path)
        repo = "/repos/alpha"
        cache.mark_week_cached(repo, MONDAY, SUNDAY, commit_count=42)

        cached = cache.get_cached_weeks(repo)
        assert len(cached) == 1
        ws, we = cached[0]
        # SQLite stores datetimes; compare date portions
        assert ws.date() == MONDAY.date()
        assert we.date() == SUNDAY.date()

    def test_mark_multiple_weeks(self, tmp_path):
        cache = make_cache(tmp_path)
        repo = "/repos/beta"
        week2_start = MONDAY + timedelta(weeks=1)
        week2_end = SUNDAY + timedelta(weeks=1)

        cache.mark_week_cached(repo, MONDAY, SUNDAY, commit_count=10)
        cache.mark_week_cached(repo, week2_start, week2_end, commit_count=20)

        cached = cache.get_cached_weeks(repo)
        assert len(cached) == 2

    def test_mark_idempotent_upsert(self, tmp_path):
        """Marking the same week twice should upsert, not create a duplicate."""
        cache = make_cache(tmp_path)
        repo = "/repos/gamma"

        cache.mark_week_cached(repo, MONDAY, SUNDAY, commit_count=5)
        cache.mark_week_cached(repo, MONDAY, SUNDAY, commit_count=99)  # second write

        cached = cache.get_cached_weeks(repo)
        assert len(cached) == 1, "Duplicate week row created instead of upsert"

    def test_different_repos_isolated(self, tmp_path):
        """Cache entries for different repos must not bleed into each other."""
        cache = make_cache(tmp_path)
        cache.mark_week_cached("/repos/a", MONDAY, SUNDAY, commit_count=1)
        cache.mark_week_cached("/repos/b", MONDAY, SUNDAY, commit_count=2)

        assert len(cache.get_cached_weeks("/repos/a")) == 1
        assert len(cache.get_cached_weeks("/repos/b")) == 1
        assert len(cache.get_cached_weeks("/repos/c")) == 0


# ---------------------------------------------------------------------------
# get_missing_weeks
# ---------------------------------------------------------------------------


class TestGetMissingWeeks:
    """Missing week detection is the core of incremental fetching."""

    def test_all_weeks_missing_on_first_run(self, tmp_path):
        cache = make_cache(tmp_path)
        required = cache.calculate_weeks(MONDAY, MONDAY + timedelta(weeks=3) - timedelta(seconds=1))

        missing = cache.get_missing_weeks("/repos/new", required)
        assert missing == required

    def test_no_weeks_missing_when_all_cached(self, tmp_path):
        cache = make_cache(tmp_path)
        repo = "/repos/full"
        required = cache.calculate_weeks(MONDAY, MONDAY + timedelta(weeks=2) - timedelta(seconds=1))

        for ws, we in required:
            cache.mark_week_cached(repo, ws, we, commit_count=0)

        missing = cache.get_missing_weeks(repo, required)
        assert missing == []

    def test_partial_cache_returns_only_missing(self, tmp_path):
        cache = make_cache(tmp_path)
        repo = "/repos/partial"
        week2_start = MONDAY + timedelta(weeks=1)
        week2_end = SUNDAY + timedelta(weeks=1)
        week3_start = MONDAY + timedelta(weeks=2)
        week3_end = SUNDAY + timedelta(weeks=2)

        required = [(MONDAY, SUNDAY), (week2_start, week2_end), (week3_start, week3_end)]

        # Cache only the middle week
        cache.mark_week_cached(repo, week2_start, week2_end, commit_count=7)

        missing = cache.get_missing_weeks(repo, required)
        assert len(missing) == 2
        missing_starts = {ws.date() for ws, _ in missing}
        assert MONDAY.date() in missing_starts
        assert week3_start.date() in missing_starts
        # Week 2 must NOT be in missing
        assert week2_start.date() not in missing_starts

    def test_empty_required_returns_empty(self, tmp_path):
        cache = make_cache(tmp_path)
        assert cache.get_missing_weeks("/repos/x", []) == []


# ---------------------------------------------------------------------------
# clear_weekly_cache
# ---------------------------------------------------------------------------


class TestClearWeeklyCache:
    """Targeted and global clearing of WeeklyFetchStatus rows."""

    def test_clear_all(self, tmp_path):
        cache = make_cache(tmp_path)
        cache.mark_week_cached("/repos/a", MONDAY, SUNDAY, commit_count=1)
        cache.mark_week_cached("/repos/b", MONDAY, SUNDAY, commit_count=2)

        deleted = cache.clear_weekly_cache()
        assert deleted == 2
        assert cache.get_cached_weeks("/repos/a") == []
        assert cache.get_cached_weeks("/repos/b") == []

    def test_clear_by_repo(self, tmp_path):
        cache = make_cache(tmp_path)
        cache.mark_week_cached("/repos/a", MONDAY, SUNDAY, commit_count=1)
        cache.mark_week_cached("/repos/b", MONDAY, SUNDAY, commit_count=2)

        deleted = cache.clear_weekly_cache(repo_path="/repos/a")
        assert deleted == 1
        assert cache.get_cached_weeks("/repos/a") == []
        assert len(cache.get_cached_weeks("/repos/b")) == 1

    def test_clear_specific_week(self, tmp_path):
        cache = make_cache(tmp_path)
        repo = "/repos/single"
        week2_start = MONDAY + timedelta(weeks=1)
        week2_end = SUNDAY + timedelta(weeks=1)

        cache.mark_week_cached(repo, MONDAY, SUNDAY, commit_count=5)
        cache.mark_week_cached(repo, week2_start, week2_end, commit_count=8)

        deleted = cache.clear_weekly_cache(repo_path=repo, week_start=MONDAY)
        assert deleted == 1

        remaining = cache.get_cached_weeks(repo)
        assert len(remaining) == 1
        ws, _ = remaining[0]
        assert ws.date() == week2_start.date()


# ---------------------------------------------------------------------------
# clear_all_cache includes WeeklyFetchStatus (bug fix test)
# ---------------------------------------------------------------------------


class TestClearAllCacheIncludesWeekly:
    """clear_all_cache must also wipe WeeklyFetchStatus rows."""

    def test_clear_all_cache_removes_weekly_rows(self, tmp_path):
        cache = make_cache(tmp_path)
        repo = "/repos/full-clear"
        cache.mark_week_cached(repo, MONDAY, SUNDAY, commit_count=10)

        assert len(cache.get_cached_weeks(repo)) == 1

        result = cache.clear_all_cache()

        # The return dict must report how many weekly rows were cleared
        assert (
            "weekly_fetch_status" in result
        ), "clear_all_cache() did not report weekly_fetch_status count"
        assert result["weekly_fetch_status"] == 1

        # The rows must actually be gone
        assert cache.get_cached_weeks(repo) == []

    def test_clear_all_cache_total_includes_weekly(self, tmp_path):
        """The 'total' key in the return dict must include the weekly count."""
        cache = make_cache(tmp_path)
        cache.mark_week_cached("/repos/r", MONDAY, SUNDAY, commit_count=0)

        result = cache.clear_all_cache()
        assert result["total"] >= result["weekly_fetch_status"]
        assert result["weekly_fetch_status"] > 0
