"""Tests for TicketingActivityReport.

Covers:
- generate_github_issues_summary aggregation
- generate_confluence_summary aggregation
- generate_combined_summary scoring and ordering
- write_reports file outputs
- Empty-data handling
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.models.database import TicketingActivityCache
from gitflow_analytics.reports.ticketing_activity_report import (
    TicketingActivityReport,
    _empty_combined_row,
    _sorted_top,
    _to_naive,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cache() -> Any:
    with tempfile.TemporaryDirectory() as tmp:
        cache = GitAnalysisCache(Path(tmp))
        yield cache


def _add_event(
    cache: GitAnalysisCache,
    *,
    platform: str,
    item_id: str,
    item_type: str,
    action: str,
    actor: str,
    activity_at: datetime,
    repo_or_space: str = "org/repo",
) -> None:
    with cache.get_session() as session:
        session.add(
            TicketingActivityCache(
                platform=platform,
                item_id=item_id,
                item_type=item_type,
                repo_or_space=repo_or_space,
                actor=actor,
                action=action,
                activity_at=activity_at.replace(tzinfo=None) if activity_at.tzinfo else activity_at,
                comment_count=0,
                reaction_count=0,
                platform_data={},
            )
        )
        session.commit()


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


def test_empty_combined_row_shape() -> None:
    row = _empty_combined_row()
    for key in (
        "issues_opened",
        "issues_closed",
        "comments_posted",
        "pages_created",
        "pages_edited",
        "total_activity",
        "ticketing_score",
    ):
        assert key in row


def test_sorted_top_ordering() -> None:
    mapping = {"a": {"x": 1}, "b": {"x": 5}, "c": {"x": 3}}
    top = _sorted_top(mapping, "x", 2)
    assert [t["developer"] for t in top] == ["b", "c"]


def test_to_naive_with_tz() -> None:
    aware = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert _to_naive(aware).tzinfo is None


def test_to_naive_already_naive() -> None:
    naive = datetime(2024, 1, 1)
    assert _to_naive(naive) is naive


# ---------------------------------------------------------------------------
# GitHub Issues summary
# ---------------------------------------------------------------------------


def test_github_issues_summary_empty(tmp_cache: GitAnalysisCache) -> None:
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_github_issues_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert summary["total_events"] == 0
    assert summary["per_repo"] == {}
    assert summary["per_developer"] == {}


def test_github_issues_summary_aggregates(tmp_cache: GitAnalysisCache) -> None:
    # Two opens, one close, two comments
    _add_event(
        tmp_cache,
        platform="github_issues",
        item_id="1",
        item_type="issue",
        action="opened",
        actor="alice",
        activity_at=datetime(2024, 2, 1),
    )
    _add_event(
        tmp_cache,
        platform="github_issues",
        item_id="1",
        item_type="issue",
        action="closed",
        actor="bob",
        activity_at=datetime(2024, 2, 3),
    )
    _add_event(
        tmp_cache,
        platform="github_issues",
        item_id="2",
        item_type="issue",
        action="opened",
        actor="alice",
        activity_at=datetime(2024, 2, 2),
    )
    _add_event(
        tmp_cache,
        platform="github_issues",
        item_id="1",
        item_type="comment",
        action="commented",
        actor="alice",
        activity_at=datetime(2024, 2, 2),
    )

    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_github_issues_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )

    assert summary["total_events"] == 4
    assert summary["per_developer"]["alice"]["issues_opened"] == 2
    assert summary["per_developer"]["alice"]["comments_posted"] == 1
    assert summary["per_developer"]["bob"]["issues_closed"] == 1
    assert summary["per_repo"]["org/repo"]["opened"] == 2
    assert summary["per_repo"]["org/repo"]["closed"] == 1
    assert summary["per_repo"]["org/repo"]["comments"] == 1


def test_github_issues_summary_avg_resolution_hours(
    tmp_cache: GitAnalysisCache,
) -> None:
    # Issue #1 opened at 2024-02-01 00:00, closed at 2024-02-01 04:00 → 4h
    _add_event(
        tmp_cache,
        platform="github_issues",
        item_id="1",
        item_type="issue",
        action="opened",
        actor="alice",
        activity_at=datetime(2024, 2, 1, 0, 0),
    )
    _add_event(
        tmp_cache,
        platform="github_issues",
        item_id="1",
        item_type="issue",
        action="closed",
        actor="alice",
        activity_at=datetime(2024, 2, 1, 4, 0),
    )
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_github_issues_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert summary["per_repo"]["org/repo"]["avg_resolution_hours"] == 4.0


def test_github_issues_top_contributors_limited(
    tmp_cache: GitAnalysisCache,
) -> None:
    # Add 12 developers with 1 event each
    for i in range(12):
        _add_event(
            tmp_cache,
            platform="github_issues",
            item_id=str(i),
            item_type="issue",
            action="opened",
            actor=f"dev{i}",
            activity_at=datetime(2024, 2, 1),
        )
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_github_issues_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert len(summary["top_contributors"]) == 10


def test_github_issues_summary_respects_date_window(
    tmp_cache: GitAnalysisCache,
) -> None:
    _add_event(
        tmp_cache,
        platform="github_issues",
        item_id="1",
        item_type="issue",
        action="opened",
        actor="alice",
        activity_at=datetime(2023, 2, 1),
    )
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_github_issues_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert summary["total_events"] == 0


# ---------------------------------------------------------------------------
# Confluence summary
# ---------------------------------------------------------------------------


def test_confluence_summary_empty(tmp_cache: GitAnalysisCache) -> None:
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_confluence_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert summary["total_events"] == 0


def test_confluence_summary_aggregates(tmp_cache: GitAnalysisCache) -> None:
    _add_event(
        tmp_cache,
        platform="confluence",
        item_id="p1",
        item_type="page_create",
        action="created",
        actor="alice",
        activity_at=datetime(2024, 2, 1),
        repo_or_space="DOC",
    )
    _add_event(
        tmp_cache,
        platform="confluence",
        item_id="p1",
        item_type="page_edit",
        action="edited",
        actor="bob",
        activity_at=datetime(2024, 2, 2),
        repo_or_space="DOC",
    )
    _add_event(
        tmp_cache,
        platform="confluence",
        item_id="p2",
        item_type="page_edit",
        action="edited",
        actor="bob",
        activity_at=datetime(2024, 2, 3),
        repo_or_space="ENG",
    )
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_confluence_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert summary["total_events"] == 3
    assert summary["per_space"]["DOC"]["pages_created"] == 1
    assert summary["per_space"]["DOC"]["pages_edited"] == 1
    assert summary["per_space"]["DOC"]["unique_editors_count"] == 2
    assert summary["per_developer"]["bob"]["pages_edited"] == 2
    assert summary["per_developer"]["bob"]["spaces_active_count"] == 2


def test_confluence_summary_spaces_active_list_sorted(
    tmp_cache: GitAnalysisCache,
) -> None:
    for space in ["Z", "A", "M"]:
        _add_event(
            tmp_cache,
            platform="confluence",
            item_id="p",
            item_type="page_edit",
            action="edited",
            actor="alice",
            activity_at=datetime(2024, 2, 1),
            repo_or_space=space,
        )
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_confluence_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert summary["per_developer"]["alice"]["spaces_active"] == ["A", "M", "Z"]


# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------


def test_combined_summary_merges_platforms(tmp_cache: GitAnalysisCache) -> None:
    # GitHub issue open
    _add_event(
        tmp_cache,
        platform="github_issues",
        item_id="1",
        item_type="issue",
        action="opened",
        actor="alice",
        activity_at=datetime(2024, 2, 1),
    )
    # Confluence page create
    _add_event(
        tmp_cache,
        platform="confluence",
        item_id="p1",
        item_type="page_create",
        action="created",
        actor="alice",
        activity_at=datetime(2024, 2, 2),
        repo_or_space="DOC",
    )
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_combined_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    alice = summary["per_developer"]["alice"]
    assert alice["issues_opened"] == 1
    assert alice["pages_created"] == 1
    # ticketing_score = 1 * 1.0 (issues_opened) + 1 * 2.0 (pages_created) = 3.0
    assert alice["ticketing_score"] == 3.0
    assert alice["total_activity"] == 2


def test_combined_summary_top_contributors_sorted(
    tmp_cache: GitAnalysisCache,
) -> None:
    # alice: 1 page_create (score 2.0)
    _add_event(
        tmp_cache,
        platform="confluence",
        item_id="p1",
        item_type="page_create",
        action="created",
        actor="alice",
        activity_at=datetime(2024, 2, 1),
        repo_or_space="DOC",
    )
    # bob: 5 issues_opened (score 5.0)
    for i in range(5):
        _add_event(
            tmp_cache,
            platform="github_issues",
            item_id=str(i),
            item_type="issue",
            action="opened",
            actor="bob",
            activity_at=datetime(2024, 2, 1),
        )
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_combined_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    top = summary["top_contributors"]
    assert top[0]["developer"] == "bob"


def test_combined_summary_empty(tmp_cache: GitAnalysisCache) -> None:
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_combined_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert summary["per_developer"] == {}
    assert summary["top_contributors"] == []


# ---------------------------------------------------------------------------
# write_reports output
# ---------------------------------------------------------------------------


def test_write_reports_creates_three_files(tmp_cache: GitAnalysisCache) -> None:
    report = TicketingActivityReport(cache=tmp_cache)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        paths = report.write_reports(
            out,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        assert len(paths) == 3
        names = {Path(p).name for p in paths}
        assert names == {
            "github_issues_summary.json",
            "confluence_activity_summary.json",
            "ticketing_activity_summary.json",
        }


def test_write_reports_writes_valid_json(tmp_cache: GitAnalysisCache) -> None:
    report = TicketingActivityReport(cache=tmp_cache)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        paths = report.write_reports(
            out,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        for p in paths:
            with open(p) as f:
                data = json.load(f)
            assert isinstance(data, dict)


def test_write_reports_creates_missing_output_dir(
    tmp_cache: GitAnalysisCache,
) -> None:
    report = TicketingActivityReport(cache=tmp_cache)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "new_subdir"
        paths = report.write_reports(
            out,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        assert out.exists()
        assert len(paths) == 3


# ---------------------------------------------------------------------------
# Canonical resolution
# ---------------------------------------------------------------------------


def test_canonical_without_resolver(tmp_cache: GitAnalysisCache) -> None:
    report = TicketingActivityReport(cache=tmp_cache)
    assert report._canonical("BOBbb") == "bobbb"
    assert report._canonical(None) == "unknown"


def test_canonical_with_resolver(tmp_cache: GitAnalysisCache) -> None:
    class _Resolver:
        def resolve_identity(self, name: Any, email: Any) -> str:
            _ = (name, email)
            return "CanonicalBob"

    report = TicketingActivityReport(cache=tmp_cache, identity_resolver=_Resolver())
    assert report._canonical("bob") == "canonicalbob"
