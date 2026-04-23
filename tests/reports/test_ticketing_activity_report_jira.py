"""Tests for TicketingActivityReport JIRA integration.

Covers:
- generate_jira_summary aggregation
- combined_summary includes jira fields
- ticketing_score increases with jira activity
- jira-only developer (no git commits) appears in summary
- empty jira data → zero counts, no error
- JIRA weights applied correctly
"""

from __future__ import annotations

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
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
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
    repo_or_space: str = "PROJ",
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
# _empty_combined_row extension
# ---------------------------------------------------------------------------


def test_empty_combined_row_has_jira_fields() -> None:
    row = _empty_combined_row()
    for key in (
        "jira_issues_opened",
        "jira_issues_closed",
        "jira_comments_posted",
    ):
        assert key in row
        assert row[key] == 0


def test_weights_contain_jira_entries() -> None:
    w = TicketingActivityReport._WEIGHTS
    assert w["jira_issues_opened"] == 1.5
    assert w["jira_issues_closed"] == 2.0
    assert w["jira_comments_posted"] == 0.5


# ---------------------------------------------------------------------------
# generate_jira_summary
# ---------------------------------------------------------------------------


def test_jira_summary_empty(tmp_cache: GitAnalysisCache) -> None:
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_jira_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert summary["total_events"] == 0
    assert summary["per_developer"] == {}


def test_jira_summary_aggregates(tmp_cache: GitAnalysisCache) -> None:
    _add_event(
        tmp_cache,
        platform="jira",
        item_id="PROJ-1",
        item_type="issue_created",
        action="opened",
        actor="alice",
        activity_at=datetime(2024, 2, 1),
    )
    _add_event(
        tmp_cache,
        platform="jira",
        item_id="PROJ-1",
        item_type="issue_closed",
        action="closed",
        actor="bob",
        activity_at=datetime(2024, 2, 3),
    )
    _add_event(
        tmp_cache,
        platform="jira",
        item_id="PROJ-1",
        item_type="comment",
        action="commented",
        actor="alice",
        activity_at=datetime(2024, 2, 2),
    )
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_jira_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert summary["total_events"] == 3
    assert summary["per_developer"]["alice"]["jira_issues_opened"] == 1
    assert summary["per_developer"]["alice"]["jira_comments_posted"] == 1
    assert summary["per_developer"]["bob"]["jira_issues_closed"] == 1


def test_jira_summary_respects_date_window(tmp_cache: GitAnalysisCache) -> None:
    _add_event(
        tmp_cache,
        platform="jira",
        item_id="PROJ-1",
        item_type="issue_created",
        action="opened",
        actor="alice",
        activity_at=datetime(2023, 1, 1),
    )
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_jira_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert summary["total_events"] == 0


# ---------------------------------------------------------------------------
# generate_combined_summary with JIRA
# ---------------------------------------------------------------------------


def test_combined_summary_includes_jira_fields(tmp_cache: GitAnalysisCache) -> None:
    _add_event(
        tmp_cache,
        platform="jira",
        item_id="PROJ-1",
        item_type="issue_created",
        action="opened",
        actor="alice",
        activity_at=datetime(2024, 2, 1),
    )
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_combined_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    alice = summary["per_developer"]["alice"]
    assert alice["jira_issues_opened"] == 1
    assert alice["jira_issues_closed"] == 0
    assert alice["jira_comments_posted"] == 0
    # ticketing_score = 1 * 1.5
    assert alice["ticketing_score"] == 1.5
    assert alice["total_activity"] == 1
    assert "jira" in summary
    assert summary["jira"]["total_events"] == 1


def test_combined_summary_jira_score_blends(tmp_cache: GitAnalysisCache) -> None:
    # GitHub issue (score 1.0) + JIRA issue_closed (score 2.0) = 3.0
    _add_event(
        tmp_cache,
        platform="github_issues",
        item_id="1",
        item_type="issue",
        action="opened",
        actor="alice",
        activity_at=datetime(2024, 2, 1),
        repo_or_space="org/repo",
    )
    _add_event(
        tmp_cache,
        platform="jira",
        item_id="PROJ-1",
        item_type="issue_closed",
        action="closed",
        actor="alice",
        activity_at=datetime(2024, 2, 2),
    )
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_combined_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    alice = summary["per_developer"]["alice"]
    assert alice["ticketing_score"] == 3.0
    assert alice["total_activity"] == 2


def test_combined_summary_jira_only_developer(tmp_cache: GitAnalysisCache) -> None:
    # Developer with only JIRA activity, no git/github
    _add_event(
        tmp_cache,
        platform="jira",
        item_id="PROJ-7",
        item_type="comment",
        action="commented",
        actor="jira_only_dev",
        activity_at=datetime(2024, 2, 5),
    )
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_combined_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert "jira_only_dev" in summary["per_developer"]
    assert summary["per_developer"]["jira_only_dev"]["jira_comments_posted"] == 1


def test_combined_summary_empty_jira(tmp_cache: GitAnalysisCache) -> None:
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_combined_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert summary["jira"]["total_events"] == 0
    assert summary["per_developer"] == {}


def test_combined_summary_jira_score_more_than_comments(
    tmp_cache: GitAnalysisCache,
) -> None:
    # jira_issues_closed (2.0) beats jira_comments_posted (0.5) for same count
    _add_event(
        tmp_cache,
        platform="jira",
        item_id="P-1",
        item_type="issue_closed",
        action="closed",
        actor="alice",
        activity_at=datetime(2024, 2, 1),
    )
    _add_event(
        tmp_cache,
        platform="jira",
        item_id="P-2",
        item_type="comment",
        action="commented",
        actor="bob",
        activity_at=datetime(2024, 2, 1),
    )
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_combined_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert summary["per_developer"]["alice"]["ticketing_score"] == 2.0
    assert summary["per_developer"]["bob"]["ticketing_score"] == 0.5
    # top contributor should be alice
    assert summary["top_contributors"][0]["developer"] == "alice"


def test_combined_summary_multiple_jira_events_aggregate(
    tmp_cache: GitAnalysisCache,
) -> None:
    for i in range(3):
        _add_event(
            tmp_cache,
            platform="jira",
            item_id=f"PROJ-{i}",
            item_type="issue_created",
            action="opened",
            actor="alice",
            activity_at=datetime(2024, 2, 1),
        )
    report = TicketingActivityReport(cache=tmp_cache)
    summary = report.generate_combined_summary(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    alice = summary["per_developer"]["alice"]
    assert alice["jira_issues_opened"] == 3
    assert alice["ticketing_score"] == 4.5  # 3 * 1.5
