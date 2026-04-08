"""Tests for velocity_report.py — PR cycle time, throughput, revision rate.

Coverage targets:
  1.  _cycle_time_hrs with valid dates
  2.  _cycle_time_hrs with missing merged_at
  3.  _cycle_time_hrs with missing created_at
  4.  _cycle_time_hrs with string ISO dates
  5.  Outlier filtering (below min_hrs / above max_hrs)
  6.  _by_developer grouping
  7.  _by_week bucketing (Monday-aligned)
  8.  _top_prs fastest / slowest ordering
  9.  generate() writes velocity_summary.json
  10. generate() with empty PR list
  11. generate() date-range filtering
  12. generate() with no merged PRs
  13. avg/median/revision metrics correctness
  14. story_points_delivered aggregation
  15. avg_time_to_first_review_hrs aggregation
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gitflow_analytics.reports.velocity_report import (
    VelocityReportGenerator,
    _cycle_time_hrs,
    _to_dt,
    _week_start,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _pr(
    *,
    author: str = "alice",
    created_at: str = "2024-01-01T09:00:00+00:00",
    merged_at: str | None = "2024-01-03T09:00:00+00:00",
    is_merged: bool = True,
    pr_state: str = "merged",
    revision_count: int = 2,
    story_points: int = 3,
    time_to_first_review_hours: float | None = 4.0,
    repo_path: str = "org/repo",
    title: str = "My PR",
) -> dict[str, Any]:
    """Build a minimal PR dict for testing."""
    return {
        "author": author,
        "created_at": created_at,
        "merged_at": merged_at,
        "is_merged": is_merged,
        "pr_state": pr_state,
        "revision_count": revision_count,
        "story_points": story_points,
        "time_to_first_review_hours": time_to_first_review_hours,
        "repo_path": repo_path,
        "title": title,
    }


@pytest.fixture()
def gen() -> VelocityReportGenerator:
    return VelocityReportGenerator()


# ---------------------------------------------------------------------------
# 1–4. _cycle_time_hrs helpers
# ---------------------------------------------------------------------------


class TestCycleTimeHrs:
    def test_valid_dates_returns_hours(self) -> None:
        pr = _pr(
            created_at="2024-01-01T00:00:00+00:00",
            merged_at="2024-01-02T00:00:00+00:00",
        )
        result = _cycle_time_hrs(pr)
        assert result == pytest.approx(24.0)

    def test_missing_merged_at_returns_none(self) -> None:
        pr = _pr(merged_at=None)
        assert _cycle_time_hrs(pr) is None

    def test_missing_created_at_returns_none(self) -> None:
        pr = _pr(created_at=None)  # type: ignore[arg-type]
        pr["created_at"] = None
        assert _cycle_time_hrs(pr) is None

    def test_string_iso_dates_parsed_correctly(self) -> None:
        pr = _pr(
            created_at="2024-03-01T06:00:00+00:00",
            merged_at="2024-03-01T18:00:00+00:00",
        )
        result = _cycle_time_hrs(pr)
        assert result == pytest.approx(12.0)

    def test_native_datetime_objects(self) -> None:
        pr = dict(
            is_merged=True,
            pr_state="merged",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            merged_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        assert _cycle_time_hrs(pr) == pytest.approx(24.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. Outlier filtering
# ---------------------------------------------------------------------------


class TestOutlierFiltering:
    def test_below_min_hrs_excluded(self) -> None:
        gen = VelocityReportGenerator()
        pr = _pr(
            created_at="2024-01-01T00:00:00+00:00",
            merged_at="2024-01-01T00:10:00+00:00",  # 10 minutes — below 0.5 h
        )
        times = gen._cycle_times([pr])
        assert times == []

    def test_above_max_hrs_excluded(self) -> None:
        gen = VelocityReportGenerator()
        pr = _pr(
            created_at="2024-01-01T00:00:00+00:00",
            merged_at="2024-02-15T00:00:00+00:00",  # > 720 h
        )
        times = gen._cycle_times([pr])
        assert times == []

    def test_within_range_included(self) -> None:
        gen = VelocityReportGenerator()
        pr = _pr(
            created_at="2024-01-01T00:00:00+00:00",
            merged_at="2024-01-03T00:00:00+00:00",  # 48 h
        )
        times = gen._cycle_times([pr])
        assert times == [pytest.approx(48.0)]

    def test_custom_thresholds_respected(self) -> None:
        class Cfg:
            cycle_time_outlier_min_hrs = 10.0
            cycle_time_outlier_max_hrs = 100.0
            top_n = 5

        gen = VelocityReportGenerator(Cfg())
        fast_pr = _pr(
            created_at="2024-01-01T00:00:00+00:00",
            merged_at="2024-01-01T05:00:00+00:00",  # 5 h — below custom min
        )
        assert gen._cycle_times([fast_pr]) == []


# ---------------------------------------------------------------------------
# 6. _by_developer grouping
# ---------------------------------------------------------------------------


class TestByDeveloper:
    def test_groups_prs_by_author(self, gen: VelocityReportGenerator) -> None:
        prs = [
            _pr(author="alice"),
            _pr(author="bob"),
            _pr(author="alice"),
        ]
        result = gen._by_developer(prs)
        assert set(result.keys()) == {"alice", "bob"}
        assert result["alice"]["prs_merged"] == 2
        assert result["bob"]["prs_merged"] == 1

    def test_unknown_author_handled(self, gen: VelocityReportGenerator) -> None:
        pr = _pr(author=None)  # type: ignore[arg-type]
        pr["author"] = None
        result = gen._by_developer([pr])
        assert "unknown" in result

    def test_keys_sorted_alphabetically(self, gen: VelocityReportGenerator) -> None:
        prs = [_pr(author="zara"), _pr(author="alice"), _pr(author="mike")]
        result = gen._by_developer(prs)
        assert list(result.keys()) == sorted(result.keys())


# ---------------------------------------------------------------------------
# 7. _by_week bucketing
# ---------------------------------------------------------------------------


class TestByWeek:
    def test_groups_by_monday_week(self, gen: VelocityReportGenerator) -> None:
        # Wednesday 2024-01-03 and Thursday 2024-01-04 — same ISO week
        prs = [
            _pr(merged_at="2024-01-03T12:00:00+00:00"),
            _pr(merged_at="2024-01-04T12:00:00+00:00"),
        ]
        result = gen._by_week(prs)
        # Both should land in week starting 2024-01-01 (Monday)
        assert "2024-01-01" in result
        assert result["2024-01-01"]["prs_merged"] == 2

    def test_prs_in_different_weeks_separated(self, gen: VelocityReportGenerator) -> None:
        prs = [
            _pr(merged_at="2024-01-01T12:00:00+00:00"),  # week 2024-01-01
            _pr(merged_at="2024-01-08T12:00:00+00:00"),  # week 2024-01-08
        ]
        result = gen._by_week(prs)
        assert len(result) == 2

    def test_pr_missing_merged_at_not_bucketed(self, gen: VelocityReportGenerator) -> None:
        pr = _pr(merged_at=None)
        result = gen._by_week([pr])
        assert result == {}


# ---------------------------------------------------------------------------
# 8. _top_prs
# ---------------------------------------------------------------------------


class TestTopPrs:
    def test_fastest_pr_first(self, gen: VelocityReportGenerator) -> None:
        fast = _pr(
            title="fast",
            created_at="2024-01-01T00:00:00+00:00",
            merged_at="2024-01-01T01:00:00+00:00",  # 1 h
        )
        slow = _pr(
            title="slow",
            created_at="2024-01-01T00:00:00+00:00",
            merged_at="2024-01-10T00:00:00+00:00",  # 216 h
        )
        fastest, _ = gen._top_prs([slow, fast])
        assert fastest[0]["title"] == "fast"

    def test_slowest_pr_first_in_slowest_list(self, gen: VelocityReportGenerator) -> None:
        fast = _pr(
            title="fast",
            created_at="2024-01-01T00:00:00+00:00",
            merged_at="2024-01-01T01:00:00+00:00",
        )
        slow = _pr(
            title="slow",
            created_at="2024-01-01T00:00:00+00:00",
            merged_at="2024-01-10T00:00:00+00:00",
        )
        _, slowest = gen._top_prs([fast, slow])
        assert slowest[0]["title"] == "slow"

    def test_top_n_respected(self) -> None:
        class Cfg:
            cycle_time_outlier_min_hrs = 0.5
            cycle_time_outlier_max_hrs = 720.0
            top_n = 2

        gen = VelocityReportGenerator(Cfg())
        prs = [
            _pr(
                title=f"pr{i}",
                created_at="2024-01-01T00:00:00+00:00",
                merged_at=f"2024-01-{i + 2:02d}T00:00:00+00:00",
            )
            for i in range(5)
        ]
        fastest, slowest = gen._top_prs(prs)
        assert len(fastest) == 2
        assert len(slowest) == 2


# ---------------------------------------------------------------------------
# 9. generate() writes velocity_summary.json
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_writes_velocity_summary_json(
        self, gen: VelocityReportGenerator, tmp_path: Path
    ) -> None:
        prs = [_pr()]
        gen.generate(prs, tmp_path)
        out = tmp_path / "velocity_summary.json"
        assert out.exists()

    def test_output_structure(self, gen: VelocityReportGenerator, tmp_path: Path) -> None:
        prs = [_pr()]
        summary = gen.generate(prs, tmp_path)
        assert "generated_at" in summary
        assert "total_prs_analyzed" in summary
        assert "per_developer" in summary
        assert "per_week" in summary
        assert "top_fastest_prs" in summary
        assert "top_slowest_prs" in summary

    def test_json_is_valid(self, gen: VelocityReportGenerator, tmp_path: Path) -> None:
        gen.generate([_pr()], tmp_path)
        raw = (tmp_path / "velocity_summary.json").read_text()
        data = json.loads(raw)
        assert data["total_prs_analyzed"] == 1

    # 10. Empty PR list
    def test_empty_pr_list(self, gen: VelocityReportGenerator, tmp_path: Path) -> None:
        summary = gen.generate([], tmp_path)
        assert summary["total_prs_analyzed"] == 0
        assert summary["per_developer"] == {}
        assert summary["per_week"] == {}

    # 11. Date-range filtering
    def test_date_range_filters_out_of_range_prs(
        self, gen: VelocityReportGenerator, tmp_path: Path
    ) -> None:
        prs = [
            _pr(merged_at="2024-01-01T12:00:00+00:00"),  # inside
            _pr(merged_at="2024-03-01T12:00:00+00:00"),  # outside
        ]
        summary = gen.generate(
            prs,
            tmp_path,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )
        assert summary["total_prs_analyzed"] == 1

    # 12. No merged PRs
    def test_no_merged_prs_produces_empty_metrics(
        self, gen: VelocityReportGenerator, tmp_path: Path
    ) -> None:
        open_pr = _pr(is_merged=False, pr_state="open", merged_at=None)
        summary = gen.generate([open_pr], tmp_path)
        assert summary["total_prs_analyzed"] == 0


# ---------------------------------------------------------------------------
# 13. Metric correctness
# ---------------------------------------------------------------------------


class TestMetricCorrectness:
    def test_avg_cycle_time_calculated_correctly(self, gen: VelocityReportGenerator) -> None:
        prs = [
            _pr(
                created_at="2024-01-01T00:00:00+00:00",
                merged_at="2024-01-02T00:00:00+00:00",  # 24 h
            ),
            _pr(
                created_at="2024-01-01T00:00:00+00:00",
                merged_at="2024-01-03T00:00:00+00:00",  # 48 h
            ),
        ]
        metrics = gen._pr_metrics(prs)
        assert metrics["avg_cycle_time_hrs"] == pytest.approx(36.0)
        assert metrics["median_cycle_time_hrs"] == pytest.approx(36.0)

    def test_avg_revision_count(self, gen: VelocityReportGenerator) -> None:
        prs = [_pr(revision_count=2), _pr(revision_count=4)]
        metrics = gen._pr_metrics(prs)
        assert metrics["avg_revision_count"] == pytest.approx(3.0)

    # 14. story_points_delivered
    def test_story_points_summed(self, gen: VelocityReportGenerator) -> None:
        prs = [_pr(story_points=3), _pr(story_points=5)]
        metrics = gen._pr_metrics(prs)
        assert metrics["story_points_delivered"] == 8

    # 15. avg_time_to_first_review_hrs
    def test_avg_time_to_first_review_calculated(self, gen: VelocityReportGenerator) -> None:
        prs = [
            _pr(time_to_first_review_hours=2.0),
            _pr(time_to_first_review_hours=6.0),
        ]
        metrics = gen._pr_metrics(prs)
        assert metrics["avg_time_to_first_review_hrs"] == pytest.approx(4.0)

    def test_none_time_to_first_review_excluded(self, gen: VelocityReportGenerator) -> None:
        prs = [_pr(time_to_first_review_hours=None)]
        metrics = gen._pr_metrics(prs)
        assert metrics["avg_time_to_first_review_hrs"] is None
