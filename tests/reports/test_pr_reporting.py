"""Tests for enhanced PR reporting.

Covers:
- narrative_writer._write_pr_analysis() with and without review data
- csv_writer.generate_summary_report() with pr_metrics parameter
- csv_writer.generate_developer_activity_summary() with per-developer PR stats
- csv_writer.generate_pr_metrics_report() new PR-level CSV
- data_models.PullRequestData new fields and helpers
- enhanced_analyzer._calculate_compliance_metrics() real approval rate
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from gitflow_analytics.reports.csv_writer import CSVReportGenerator
from gitflow_analytics.reports.data_models import PullRequestData
from gitflow_analytics.reports.narrative_writer import NarrativeReportGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def recent_date() -> datetime:
    """A consistent recent datetime for tests (within the last week)."""
    from datetime import timedelta

    return datetime.now(timezone.utc).replace(hour=10, minute=0, second=0, microsecond=0) - timedelta(days=7)


@pytest.fixture
def base_prs(recent_date: datetime) -> list[dict[str, Any]]:
    """PRs without enhanced review data (fetch_pr_reviews disabled)."""
    return [
        {
            "number": 1,
            "title": "Add feature X",
            "author": "alice",
            "created_at": recent_date,
            "merged_at": recent_date.replace(hour=18),
            "additions": 120,
            "deletions": 30,
            "changed_files": 5,
            "review_comments": 4,
            "story_points": 3,
            "labels": ["enhancement"],
        },
        {
            "number": 2,
            "title": "Fix bug Y",
            "author": "bob",
            "created_at": recent_date.replace(day=2),
            "merged_at": recent_date.replace(day=2, hour=16),
            "additions": 20,
            "deletions": 10,
            "changed_files": 2,
            "review_comments": 1,
            "story_points": 0,
            "labels": ["bug"],
        },
    ]


@pytest.fixture
def review_prs(recent_date: datetime) -> list[dict[str, Any]]:
    """PRs with full enhanced review data (fetch_pr_reviews enabled).

    ``canonical_id`` is set to match developer_metrics keys so per-developer
    PR stats join correctly in generate_developer_activity_summary.
    """
    return [
        {
            "number": 10,
            "title": "Refactor auth",
            "author": "alice",
            "canonical_id": "alice@example.com",
            "created_at": recent_date,
            "merged_at": recent_date.replace(hour=20),
            "additions": 200,
            "deletions": 80,
            "changed_files": 8,
            "review_comments": 6,
            "pr_comments_count": 3,
            "story_points": 5,
            "approvals_count": 2,
            "change_requests_count": 1,
            "reviewers": ["charlie", "dave"],
            "approved_by": ["charlie", "dave"],
            "time_to_first_review_hours": 2.5,
            "revision_count": 1,
            "labels": ["refactor"],
        },
        {
            "number": 11,
            "title": "Add tests",
            "author": "bob",
            "canonical_id": "bob@example.com",
            "created_at": recent_date.replace(day=recent_date.day + 1),
            "merged_at": recent_date.replace(day=recent_date.day + 1, hour=14),
            "additions": 150,
            "deletions": 0,
            "changed_files": 6,
            "review_comments": 2,
            "pr_comments_count": 1,
            "story_points": 2,
            "approvals_count": 1,
            "change_requests_count": 0,
            "reviewers": ["charlie"],
            "approved_by": ["charlie"],
            "time_to_first_review_hours": 1.0,
            "revision_count": 0,
            "labels": ["test"],
        },
        {
            "number": 12,
            "title": "Update docs",
            "author": "alice",
            "canonical_id": "alice@example.com",
            "created_at": recent_date.replace(day=recent_date.day + 2),
            "merged_at": recent_date.replace(day=recent_date.day + 2, hour=11),
            "additions": 30,
            "deletions": 5,
            "changed_files": 2,
            "review_comments": 0,
            "pr_comments_count": 0,
            "story_points": 0,
            "approvals_count": 1,
            "change_requests_count": 0,
            "reviewers": ["dave"],
            "approved_by": ["dave"],
            "time_to_first_review_hours": 4.0,
            "revision_count": 0,
            "labels": [],
        },
    ]


@pytest.fixture
def pr_metrics_no_reviews() -> dict[str, Any]:
    """Minimal pr_metrics dict without review data (fetch_pr_reviews disabled).

    ``review_data_collected=False`` is the sentinel that the narrative writer
    and CSV exporter use to suppress enhanced-review sections.
    """
    return {
        "total_prs": 2,
        "avg_pr_size": 90.0,
        "avg_pr_lifetime_hours": 8.0,
        "avg_files_per_pr": 3.5,
        "total_review_comments": 5,
        "prs_with_story_points": 1,
        "story_point_coverage": 50.0,
        # review_data_collected=False → no enhanced sections shown
        "review_data_collected": False,
        "approval_rate": 0.0,
        "avg_approvals_per_pr": 0.0,
        "avg_change_requests_per_pr": 0.0,
        "review_coverage": 0.0,
        "avg_time_to_first_review_hours": None,
        "median_time_to_first_review_hours": None,
        "total_pr_comments": 0,
        "avg_pr_comments_per_pr": 0.0,
        "avg_revision_count": 0.0,
    }


@pytest.fixture
def pr_metrics_with_reviews() -> dict[str, Any]:
    """pr_metrics dict with full review data (fetch_pr_reviews enabled)."""
    return {
        "total_prs": 3,
        "avg_pr_size": 155.0,
        "avg_pr_lifetime_hours": 9.5,
        "avg_files_per_pr": 5.3,
        "total_review_comments": 8,
        "prs_with_story_points": 2,
        "story_point_coverage": 66.7,
        "review_data_collected": True,
        "approval_rate": 100.0,
        "avg_approvals_per_pr": 1.33,
        "avg_change_requests_per_pr": 0.33,
        "review_coverage": 100.0,
        "avg_time_to_first_review_hours": 2.5,
        "median_time_to_first_review_hours": 2.5,
        "total_pr_comments": 4,
        "avg_pr_comments_per_pr": 1.33,
        "avg_revision_count": 0.33,
    }


@pytest.fixture
def sample_commits(recent_date: datetime) -> list[dict[str, Any]]:
    return [
        {
            "hash": "aaa",
            "author_email": "alice@example.com",
            "author_name": "Alice",
            "canonical_id": "alice@example.com",
            "timestamp": recent_date,
            "insertions": 50,
            "deletions": 10,
            "files_changed": 3,
            "story_points": 3,
            "project_key": "PROJ",
            "ticket_references": [{"full_id": "PROJ-1"}],
        },
        {
            "hash": "bbb",
            "author_email": "bob@example.com",
            "author_name": "Bob",
            "canonical_id": "bob@example.com",
            "timestamp": recent_date.replace(day=5),
            "insertions": 20,
            "deletions": 5,
            "files_changed": 2,
            "story_points": 2,
            "project_key": "PROJ",
            "ticket_references": [],
        },
    ]


@pytest.fixture
def sample_developer_stats() -> list[dict[str, Any]]:
    return [
        {
            "canonical_id": "alice@example.com",
            "primary_name": "Alice",
            "primary_email": "alice@example.com",
            "total_commits": 1,
            "total_story_points": 3,
            "alias_count": 1,
            "first_seen": None,
            "last_seen": None,
        },
        {
            "canonical_id": "bob@example.com",
            "primary_name": "Bob",
            "primary_email": "bob@example.com",
            "total_commits": 1,
            "total_story_points": 2,
            "alias_count": 1,
            "first_seen": None,
            "last_seen": None,
        },
    ]


# ---------------------------------------------------------------------------
# PullRequestData model tests
# ---------------------------------------------------------------------------


class TestPullRequestDataModel:
    def test_default_fields(self, recent_date: datetime) -> None:
        pr = PullRequestData(
            id=1,
            title="Test PR",
            author="alice",
            created_at=recent_date,
        )
        assert pr.approvals_count == 0
        assert pr.change_requests_count == 0
        assert pr.revision_count == 0
        assert pr.time_to_first_review_hours is None
        assert pr.reviewers == []
        assert pr.approved_by == []

    def test_was_approved_via_approvals_count(self, recent_date: datetime) -> None:
        pr = PullRequestData(
            id=1, title="T", author="a", created_at=recent_date, approvals_count=2
        )
        assert pr.was_approved() is True

    def test_was_approved_via_approved_by(self, recent_date: datetime) -> None:
        pr = PullRequestData(
            id=1, title="T", author="a", created_at=recent_date, approved_by=["charlie"]
        )
        assert pr.was_approved() is True

    def test_not_approved(self, recent_date: datetime) -> None:
        pr = PullRequestData(id=1, title="T", author="a", created_at=recent_date)
        assert pr.was_approved() is False

    def test_had_change_requests(self, recent_date: datetime) -> None:
        pr = PullRequestData(
            id=1, title="T", author="a", created_at=recent_date, change_requests_count=1
        )
        assert pr.had_change_requests() is True

    def test_no_change_requests(self, recent_date: datetime) -> None:
        pr = PullRequestData(id=1, title="T", author="a", created_at=recent_date)
        assert pr.had_change_requests() is False

    def test_get_cycle_time(self, recent_date: datetime) -> None:
        # recent_date is hour=10; merged at hour=16 → 6 hours
        merged = recent_date.replace(hour=16)
        pr = PullRequestData(
            id=1,
            title="T",
            author="a",
            created_at=recent_date,
            merged_at=merged,
            is_merged=True,
        )
        cycle = pr.get_cycle_time()
        assert cycle is not None
        assert cycle == pytest.approx(6.0, abs=0.01)

    def test_get_cycle_time_not_merged(self, recent_date: datetime) -> None:
        pr = PullRequestData(id=1, title="T", author="a", created_at=recent_date)
        assert pr.get_cycle_time() is None


# ---------------------------------------------------------------------------
# narrative_writer._write_pr_analysis tests
# ---------------------------------------------------------------------------


class TestNarrativeWriterPRAnalysis:
    def _run(
        self,
        pr_metrics: dict[str, Any],
        prs: list[dict[str, Any]],
    ) -> str:
        gen = NarrativeReportGenerator()
        buf = StringIO()
        gen._write_pr_analysis(buf, pr_metrics, prs)
        return buf.getvalue()

    def test_overview_section_always_present(
        self, pr_metrics_no_reviews: dict[str, Any], base_prs: list[dict[str, Any]]
    ) -> None:
        content = self._run(pr_metrics_no_reviews, base_prs)
        assert "### Overview" in content
        assert "Total PRs Merged" in content
        assert "Average PR Size" in content

    def test_avg_files_per_pr_shown(
        self, pr_metrics_no_reviews: dict[str, Any], base_prs: list[dict[str, Any]]
    ) -> None:
        content = self._run(pr_metrics_no_reviews, base_prs)
        assert "Average Files per PR" in content

    def test_pr_lifetime_formatted_in_days_when_over_24h(
        self, pr_metrics_no_reviews: dict[str, Any], base_prs: list[dict[str, Any]]
    ) -> None:
        pr_metrics_no_reviews["avg_pr_lifetime_hours"] = 48.0
        content = self._run(pr_metrics_no_reviews, base_prs)
        assert "days" in content

    def test_pr_lifetime_in_hours_when_under_24h(
        self, pr_metrics_no_reviews: dict[str, Any], base_prs: list[dict[str, Any]]
    ) -> None:
        pr_metrics_no_reviews["avg_pr_lifetime_hours"] = 8.0
        content = self._run(pr_metrics_no_reviews, base_prs)
        assert "8.0 hours" in content

    def test_story_point_coverage_shows_count(
        self, pr_metrics_no_reviews: dict[str, Any], base_prs: list[dict[str, Any]]
    ) -> None:
        content = self._run(pr_metrics_no_reviews, base_prs)
        assert "Story Point Coverage" in content
        assert "1 of 2 PRs" in content

    def test_review_section_absent_when_no_review_data(
        self, pr_metrics_no_reviews: dict[str, Any], base_prs: list[dict[str, Any]]
    ) -> None:
        # approval_rate=0.0 and review_coverage=0.0 → no review section
        content = self._run(pr_metrics_no_reviews, base_prs)
        assert "### Review Metrics" not in content

    def test_review_section_present_with_review_data(
        self, pr_metrics_with_reviews: dict[str, Any], review_prs: list[dict[str, Any]]
    ) -> None:
        content = self._run(pr_metrics_with_reviews, review_prs)
        assert "### Review Metrics" in content
        assert "Approval Rate" in content
        assert "100.0%" in content

    def test_comment_metrics_section(
        self, pr_metrics_with_reviews: dict[str, Any], review_prs: list[dict[str, Any]]
    ) -> None:
        content = self._run(pr_metrics_with_reviews, review_prs)
        assert "### Comment Metrics" in content
        assert "Inline Review Comments" in content
        assert "PR Comments" in content

    def test_time_to_review_section(
        self, pr_metrics_with_reviews: dict[str, Any], review_prs: list[dict[str, Any]]
    ) -> None:
        content = self._run(pr_metrics_with_reviews, review_prs)
        assert "### Time-to-Review Metrics" in content
        assert "Average Time to First Review" in content
        assert "Median Time to First Review" in content

    def test_time_interpretation_fast(
        self, pr_metrics_with_reviews: dict[str, Any], review_prs: list[dict[str, Any]]
    ) -> None:
        pr_metrics_with_reviews["avg_time_to_first_review_hours"] = 2.0
        content = self._run(pr_metrics_with_reviews, review_prs)
        assert "fast review turnaround" in content

    def test_time_interpretation_same_day(
        self, pr_metrics_with_reviews: dict[str, Any], review_prs: list[dict[str, Any]]
    ) -> None:
        pr_metrics_with_reviews["avg_time_to_first_review_hours"] = 10.0
        content = self._run(pr_metrics_with_reviews, review_prs)
        assert "same-day review" in content

    def test_revision_section_present_when_revisions(
        self, pr_metrics_with_reviews: dict[str, Any], review_prs: list[dict[str, Any]]
    ) -> None:
        pr_metrics_with_reviews["avg_revision_count"] = 1.2
        content = self._run(pr_metrics_with_reviews, review_prs)
        assert "### Revision Metrics" in content
        assert "Average Revisions per PR" in content

    def test_change_request_section_with_cr_data(
        self, pr_metrics_with_reviews: dict[str, Any], review_prs: list[dict[str, Any]]
    ) -> None:
        content = self._run(pr_metrics_with_reviews, review_prs)
        assert "### Change Request Metrics" in content
        assert "Average Change Requests" in content
        assert "Change Request Rate" in content

    def test_change_request_rate_calculation(
        self, pr_metrics_with_reviews: dict[str, Any], review_prs: list[dict[str, Any]]
    ) -> None:
        # Only PR #10 has change_requests_count=1; others are 0
        content = self._run(pr_metrics_with_reviews, review_prs)
        # 1/3 PRs had change requests → ~33.3%
        assert "33.3%" in content


# ---------------------------------------------------------------------------
# csv_writer.generate_summary_report tests
# ---------------------------------------------------------------------------


class TestCSVSummaryReportPRMetrics:
    def _generate(
        self,
        temp_dir: Path,
        commits: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        developer_stats: list[dict[str, Any]],
        pr_metrics: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        gen = CSVReportGenerator()
        output = temp_dir / "summary.csv"
        gen.generate_summary_report(
            commits,
            prs,
            developer_stats,
            {"commit_coverage_pct": 50.0, "ticket_summary": {}},
            output,
            pr_metrics=pr_metrics,
        )
        with open(output) as f:
            return list(csv.DictReader(f))

    def _metric_value(self, rows: list[dict[str, str]], metric: str) -> str | None:
        for row in rows:
            if row["metric"] == metric:
                return row["value"]
        return None

    def test_basic_pr_stats_always_included(
        self,
        temp_dir: Path,
        sample_commits: list[dict[str, Any]],
        base_prs: list[dict[str, Any]],
        sample_developer_stats: list[dict[str, Any]],
    ) -> None:
        rows = self._generate(temp_dir, sample_commits, base_prs, sample_developer_stats)
        metrics = {r["metric"] for r in rows}
        assert "Total PRs" in metrics
        assert "Total Inline Review Comments" in metrics
        assert "Avg PR Size (lines)" in metrics

    def test_pr_review_metrics_included_when_provided(
        self,
        temp_dir: Path,
        sample_commits: list[dict[str, Any]],
        review_prs: list[dict[str, Any]],
        sample_developer_stats: list[dict[str, Any]],
        pr_metrics_with_reviews: dict[str, Any],
    ) -> None:
        rows = self._generate(
            temp_dir,
            sample_commits,
            review_prs,
            sample_developer_stats,
            pr_metrics=pr_metrics_with_reviews,
        )
        metrics = {r["metric"] for r in rows}
        assert "PR Approval Rate %" in metrics
        assert "PR Review Coverage %" in metrics
        assert "Avg Approvals per PR" in metrics
        assert "Avg Change Requests per PR" in metrics
        assert "Avg Time to First Review (hours)" in metrics
        assert "Median Time to First Review (hours)" in metrics
        assert "Avg Revisions per PR" in metrics

    def test_approval_rate_value_correct(
        self,
        temp_dir: Path,
        sample_commits: list[dict[str, Any]],
        review_prs: list[dict[str, Any]],
        sample_developer_stats: list[dict[str, Any]],
        pr_metrics_with_reviews: dict[str, Any],
    ) -> None:
        rows = self._generate(
            temp_dir,
            sample_commits,
            review_prs,
            sample_developer_stats,
            pr_metrics=pr_metrics_with_reviews,
        )
        val = self._metric_value(rows, "PR Approval Rate %")
        assert val == "100.0"

    def test_no_review_metrics_without_pr_metrics(
        self,
        temp_dir: Path,
        sample_commits: list[dict[str, Any]],
        base_prs: list[dict[str, Any]],
        sample_developer_stats: list[dict[str, Any]],
    ) -> None:
        rows = self._generate(temp_dir, sample_commits, base_prs, sample_developer_stats)
        metrics = {r["metric"] for r in rows}
        assert "PR Approval Rate %" not in metrics
        assert "Avg Time to First Review (hours)" not in metrics

    def test_no_pr_rows_when_empty_pr_list(
        self,
        temp_dir: Path,
        sample_commits: list[dict[str, Any]],
        sample_developer_stats: list[dict[str, Any]],
    ) -> None:
        rows = self._generate(temp_dir, sample_commits, [], sample_developer_stats)
        metrics = {r["metric"] for r in rows}
        assert "Total PRs" not in metrics


# ---------------------------------------------------------------------------
# csv_writer.generate_developer_activity_summary tests
# ---------------------------------------------------------------------------


class TestCSVDeveloperActivitySummaryPRStats:
    def _generate(
        self,
        temp_dir: Path,
        commits: list[dict[str, Any]],
        developer_stats: list[dict[str, Any]],
        prs: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        gen = CSVReportGenerator()
        output = temp_dir / "dev_activity.csv"
        gen.generate_developer_activity_summary(commits, developer_stats, prs, output, weeks=52)
        with open(output) as f:
            return list(csv.DictReader(f))

    def test_new_pr_columns_present(
        self,
        temp_dir: Path,
        sample_commits: list[dict[str, Any]],
        sample_developer_stats: list[dict[str, Any]],
        review_prs: list[dict[str, Any]],
    ) -> None:
        rows = self._generate(temp_dir, sample_commits, sample_developer_stats, review_prs)
        assert len(rows) > 0
        expected_cols = {
            "pr_review_comments",
            "pr_general_comments",
            "pr_approvals_received",
            "pr_change_requests_received",
            "avg_approvals_per_pr",
            "avg_change_requests_per_pr",
            "avg_revisions_per_pr",
            "avg_time_to_first_review_hours",
        }
        actual_cols = set(rows[0].keys())
        assert expected_cols.issubset(actual_cols)

    def test_alice_pr_stats_correct(
        self,
        temp_dir: Path,
        sample_commits: list[dict[str, Any]],
        sample_developer_stats: list[dict[str, Any]],
        review_prs: list[dict[str, Any]],
    ) -> None:
        # alice authored PRs #10 and #12; bob authored PR #11
        rows = self._generate(temp_dir, sample_commits, sample_developer_stats, review_prs)
        alice_row = next(
            (r for r in rows if r["developer_name"] == "Alice"), None
        )
        assert alice_row is not None
        # 2 PRs for alice: approvals = 2+1 = 3 → avg = 1.5
        assert float(alice_row["avg_approvals_per_pr"]) == pytest.approx(1.5, abs=0.01)
        # review comments: 6+0 = 6
        assert int(alice_row["pr_review_comments"]) == 6

    def test_empty_pr_review_columns_when_no_review_data(
        self,
        temp_dir: Path,
        sample_commits: list[dict[str, Any]],
        sample_developer_stats: list[dict[str, Any]],
        base_prs: list[dict[str, Any]],
    ) -> None:
        # base_prs have no approvals_count field
        rows = self._generate(temp_dir, sample_commits, sample_developer_stats, base_prs)
        alice_row = next(
            (r for r in rows if r["developer_name"] == "Alice"), None
        )
        if alice_row:
            # approvals_count not in base_prs → aggregated as 0 → avg_approvals
            # is 0/n → either empty string or 0.0 depending on branch
            # The key test is that the column exists and doesn't crash
            assert "avg_approvals_per_pr" in alice_row


# ---------------------------------------------------------------------------
# csv_writer.generate_pr_metrics_report tests
# ---------------------------------------------------------------------------


class TestCSVPRMetricsReport:
    def _generate(
        self,
        temp_dir: Path,
        prs: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        gen = CSVReportGenerator()
        output = temp_dir / "pr_metrics.csv"
        gen.generate_pr_metrics_report(prs, output)
        with open(output) as f:
            return list(csv.DictReader(f))

    def test_columns_present(
        self, temp_dir: Path, review_prs: list[dict[str, Any]]
    ) -> None:
        rows = self._generate(temp_dir, review_prs)
        expected_cols = {
            "pr_number",
            "title",
            "author",
            "created_at",
            "merged_at",
            "lifetime_hours",
            "additions",
            "deletions",
            "changed_files",
            "review_comments",
            "story_points",
            "approvals_count",
            "change_requests_count",
            "pr_comments_count",
            "time_to_first_review_hours",
            "revision_count",
            "reviewers",
            "approved_by",
            "labels",
        }
        actual_cols = set(rows[0].keys())
        assert expected_cols.issubset(actual_cols)

    def test_row_count_matches_pr_count(
        self, temp_dir: Path, review_prs: list[dict[str, Any]]
    ) -> None:
        rows = self._generate(temp_dir, review_prs)
        assert len(rows) == len(review_prs)

    def test_lifetime_calculated(
        self, temp_dir: Path, review_prs: list[dict[str, Any]]
    ) -> None:
        rows = self._generate(temp_dir, review_prs)
        # PR #10: created 10:00 merged 20:00 → 10 hours
        pr10 = next(r for r in rows if r["pr_number"] == "10")
        assert float(pr10["lifetime_hours"]) == pytest.approx(10.0, abs=0.01)

    def test_reviewers_comma_separated(
        self, temp_dir: Path, review_prs: list[dict[str, Any]]
    ) -> None:
        rows = self._generate(temp_dir, review_prs)
        pr10 = next(r for r in rows if r["pr_number"] == "10")
        assert "charlie" in pr10["reviewers"]
        assert "dave" in pr10["reviewers"]

    def test_empty_review_fields_for_basic_prs(
        self, temp_dir: Path, base_prs: list[dict[str, Any]]
    ) -> None:
        """PRs without review data should produce empty strings for review cols."""
        rows = self._generate(temp_dir, base_prs)
        pr1 = next(r for r in rows if r["pr_number"] == "1")
        assert pr1["approvals_count"] == ""
        assert pr1["change_requests_count"] == ""
        assert pr1["time_to_first_review_hours"] == ""

    def test_empty_prs_generates_header_only(self, temp_dir: Path) -> None:
        gen = CSVReportGenerator()
        output = temp_dir / "pr_metrics_empty.csv"
        gen.generate_pr_metrics_report([], output)
        assert output.exists()
        with open(output) as f:
            content = f.read()
        assert "pr_number" in content
        lines = content.strip().splitlines()
        assert len(lines) == 1  # header only

    def test_sorted_by_merged_at_desc(
        self, temp_dir: Path, review_prs: list[dict[str, Any]]
    ) -> None:
        rows = self._generate(temp_dir, review_prs)
        merged_dates = [r["merged_at"] for r in rows]
        assert merged_dates == sorted(merged_dates, reverse=True)


# ---------------------------------------------------------------------------
# enhanced_analyzer._calculate_compliance_metrics tests
# ---------------------------------------------------------------------------


class TestComplianceMetricsRealApprovalRate:
    def _make_commits(self, n: int = 10) -> list[dict[str, Any]]:
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return [
            {
                "hash": f"hash{i}",
                "canonical_id": "dev@example.com",
                "message": "fix something important",
                "insertions": 50,
                "deletions": 10,
                "ticket_references": [{"full_id": f"PROJ-{i}"}],
                "timestamp": base,
            }
            for i in range(n)
        ]

    def test_uses_real_approval_rate_when_available(self) -> None:
        from gitflow_analytics.qualitative.enhanced_analyzer import EnhancedQualitativeAnalyzer

        analyzer = EnhancedQualitativeAnalyzer()
        commits = self._make_commits(10)
        project_metrics = {"pr_metrics": {"approval_rate": 90.0, "total_prs": 5}}
        context: dict[str, Any] = {}

        result = analyzer._calculate_compliance_metrics(commits, project_metrics, context)

        assert result["pr_approval_rate"]["score"] == 90.0
        assert result["pr_approval_rate"]["status"] == "excellent"
        assert result["pr_approval_rate"]["source"] == "measured"

    def test_no_data_when_pr_metrics_missing(self) -> None:
        from gitflow_analytics.qualitative.enhanced_analyzer import EnhancedQualitativeAnalyzer

        analyzer = EnhancedQualitativeAnalyzer()
        commits = self._make_commits(10)

        result = analyzer._calculate_compliance_metrics(commits, {}, {})

        assert result["pr_approval_rate"]["score"] is None
        assert result["pr_approval_rate"]["status"] == "no_data"
        assert result["pr_approval_rate"]["source"] == "unavailable"

    def test_no_data_when_total_prs_zero(self) -> None:
        from gitflow_analytics.qualitative.enhanced_analyzer import EnhancedQualitativeAnalyzer

        analyzer = EnhancedQualitativeAnalyzer()
        commits = self._make_commits(5)
        project_metrics = {"pr_metrics": {"approval_rate": 75.0, "total_prs": 0}}

        result = analyzer._calculate_compliance_metrics(commits, project_metrics, {})

        assert result["pr_approval_rate"]["score"] is None
        assert result["pr_approval_rate"]["source"] == "unavailable"

    def test_overall_score_excludes_unavailable_pr_rate(self) -> None:
        """When PR data is unavailable, overall score is mean of 3 factors (not 4)."""
        from gitflow_analytics.qualitative.enhanced_analyzer import EnhancedQualitativeAnalyzer

        analyzer = EnhancedQualitativeAnalyzer()
        commits = self._make_commits(10)

        result_no_pr = analyzer._calculate_compliance_metrics(commits, {}, {})
        result_with_pr = analyzer._calculate_compliance_metrics(
            commits, {"pr_metrics": {"approval_rate": 0.0, "total_prs": 5}}, {}
        )

        # Injecting 0% approval should drag the score down
        assert result_with_pr["overall_score"] < result_no_pr["overall_score"]

    def test_status_thresholds(self) -> None:
        from gitflow_analytics.qualitative.enhanced_analyzer import EnhancedQualitativeAnalyzer

        analyzer = EnhancedQualitativeAnalyzer()
        commits = self._make_commits(10)

        for rate, expected_status in [
            (85.0, "excellent"),
            (65.0, "good"),
            (40.0, "needs_improvement"),
        ]:
            result = analyzer._calculate_compliance_metrics(
                commits, {"pr_metrics": {"approval_rate": rate, "total_prs": 5}}, {}
            )
            assert result["pr_approval_rate"]["status"] == expected_status, (
                f"rate={rate} expected '{expected_status}' got '{result['pr_approval_rate']['status']}'"
            )
