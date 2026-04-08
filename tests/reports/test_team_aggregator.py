"""Tests for TeamAggregator — team/pod rollup metrics (Issue #23)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gitflow_analytics.config.schema import (
    PodConfig,
    TeamConfig,
    TeamMemberConfig,
    TeamsConfig,
)
from gitflow_analytics.reports.team_aggregator import TeamAggregator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_teams_config() -> TeamsConfig:
    """Build a representative TeamsConfig with two teams and one pod."""
    return TeamsConfig(
        enabled=True,
        teams=[
            TeamConfig(
                name="Backend",
                lead="alice@example.com",
                members=[
                    TeamMemberConfig(email="alice@example.com", github="alice-gh", name="Alice"),
                    TeamMemberConfig(email="bob@example.com", github="bob-gh", name="Bob"),
                ],
                pods=[
                    PodConfig(
                        name="API Pod",
                        members=[
                            TeamMemberConfig(email="alice@example.com"),
                        ],
                    )
                ],
            ),
            TeamConfig(
                name="Frontend",
                members=[
                    TeamMemberConfig(email="carol@example.com", github="carol-gh", name="Carol"),
                ],
                pods=[],
            ),
        ],
    )


def _make_row(
    email: str,
    name: str = "Dev",
    total: int = 3,
    feature: int = 1,
    bug: int = 1,
    refactor: int = 0,
    lines_added: int = 100,
    lines_deleted: int = 20,
    ai_assisted: int = 0,
    churn: float | None = None,
) -> dict:
    row: dict = {
        "developer_id": email,
        "developer_email": email,
        "developer_name": name,
        "total_commits": total,
        "feature_commits": feature,
        "bug_fix_commits": bug,
        "refactor_commits": refactor,
        "documentation_commits": 0,
        "maintenance_commits": 0,
        "test_commits": 0,
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
        "story_points": 0,
        "tracked_commits": 1,
        "untracked_commits": total - 1,
        "ai_assisted_commits": ai_assisted,
        "ai_generated_commits": 0,
    }
    if churn is not None:
        row["churn_rate_14d"] = churn
    return row


# ---------------------------------------------------------------------------
# Constructor / lookup-table tests
# ---------------------------------------------------------------------------


class TestTeamAggregatorInit:
    def test_empty_config_produces_empty_lookups(self) -> None:
        agg = TeamAggregator(TeamsConfig())
        assert agg._dev_to_team == {}
        assert agg._dev_to_pod == {}

    def test_none_config_produces_empty_lookups(self) -> None:
        agg = TeamAggregator(None)
        assert agg._dev_to_team == {}

    def test_email_indexed_lowercase(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        assert agg._dev_to_team["alice@example.com"] == "Backend"

    def test_github_handle_indexed(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        assert agg._dev_to_team["alice-gh"] == "Backend"

    def test_display_name_indexed(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        assert agg._dev_to_team["alice"] == "Backend"

    def test_pod_member_indexed(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        assert agg._dev_to_pod["alice@example.com"] == "API Pod"

    def test_non_pod_member_not_in_pod_lookup(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        assert "bob@example.com" not in agg._dev_to_pod


# ---------------------------------------------------------------------------
# resolve_team tests
# ---------------------------------------------------------------------------


class TestResolveTeam:
    def test_resolve_by_developer_email(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        row = {"developer_email": "alice@example.com"}
        assert agg.resolve_team(row) == "Backend"

    def test_resolve_by_developer_id(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        row = {"developer_id": "bob@example.com"}
        assert agg.resolve_team(row) == "Backend"

    def test_resolve_by_developer_name(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        row = {"developer_name": "Carol"}
        assert agg.resolve_team(row) == "Frontend"

    def test_resolve_by_author_email(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        row = {"author_email": "carol@example.com"}
        assert agg.resolve_team(row) == "Frontend"

    def test_resolve_by_author(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        row = {"author": "bob-gh"}
        assert agg.resolve_team(row) == "Backend"

    def test_unknown_developer_returns_none(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        row = {"developer_email": "nobody@example.com"}
        assert agg.resolve_team(row) is None

    def test_case_insensitive_match(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        row = {"developer_email": "ALICE@EXAMPLE.COM"}
        assert agg.resolve_team(row) == "Backend"


# ---------------------------------------------------------------------------
# resolve_pod tests
# ---------------------------------------------------------------------------


class TestResolvePod:
    def test_resolve_pod_by_email(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        row = {"developer_email": "alice@example.com"}
        assert agg.resolve_pod(row) == "API Pod"

    def test_non_pod_member_returns_none(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        row = {"developer_email": "bob@example.com"}
        assert agg.resolve_pod(row) is None

    def test_unknown_developer_returns_none(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        row = {"developer_email": "nobody@example.com"}
        assert agg.resolve_pod(row) is None


# ---------------------------------------------------------------------------
# _sum_metrics tests
# ---------------------------------------------------------------------------


class TestSumMetrics:
    def test_integer_keys_summed(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        rows = [_make_row("a@x.com", total=2), _make_row("b@x.com", total=4)]
        result = agg._sum_metrics(rows)
        assert result["total_commits"] == 6

    def test_lines_summed(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        rows = [
            _make_row("a@x.com", lines_added=50),
            _make_row("b@x.com", lines_added=150),
        ]
        result = agg._sum_metrics(rows)
        assert result["lines_added"] == 200

    def test_churn_averaged(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        rows = [
            _make_row("a@x.com", churn=0.2),
            _make_row("b@x.com", churn=0.4),
        ]
        result = agg._sum_metrics(rows)
        assert result["churn_rate_14d"] == pytest.approx(0.3, abs=1e-4)

    def test_churn_zero_when_missing(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        rows = [_make_row("a@x.com")]  # no churn key
        result = agg._sum_metrics(rows)
        assert result["churn_rate_14d"] == 0.0

    def test_developer_count(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        rows = [_make_row("a@x.com"), _make_row("a@x.com"), _make_row("b@x.com")]
        result = agg._sum_metrics(rows)
        assert result["developer_count"] == 2

    def test_ai_adoption_pct_calculated(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        rows = [_make_row("a@x.com", total=10, ai_assisted=4)]
        result = agg._sum_metrics(rows)
        assert result["ai_adoption_pct"] == pytest.approx(40.0, abs=0.1)

    def test_ai_adoption_pct_zero_when_no_commits(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        rows = [_make_row("a@x.com", total=0, ai_assisted=0)]
        result = agg._sum_metrics(rows)
        assert result["ai_adoption_pct"] == 0.0


# ---------------------------------------------------------------------------
# aggregate_metrics tests
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_groups_rows_by_team(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        rows = [
            _make_row("alice@example.com", total=5),
            _make_row("bob@example.com", total=3),
            _make_row("carol@example.com", total=7),
        ]
        result = agg.aggregate_metrics(rows)
        assert set(result.keys()) == {"Backend", "Frontend"}
        assert result["Backend"]["total_commits"] == 8
        assert result["Frontend"]["total_commits"] == 7

    def test_unknown_developer_excluded(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        rows = [
            _make_row("alice@example.com", total=5),
            _make_row("nobody@example.com", total=99),
        ]
        result = agg.aggregate_metrics(rows)
        assert "Backend" in result
        assert result["Backend"]["total_commits"] == 5
        # nobody@example.com must not inflate any team total
        assert all(t["total_commits"] < 99 for t in result.values())

    def test_empty_metrics_returns_empty_dict(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        assert agg.aggregate_metrics([]) == {}

    def test_result_keys_sorted_alphabetically(self) -> None:
        agg = TeamAggregator(_make_teams_config())
        rows = [
            _make_row("carol@example.com"),
            _make_row("alice@example.com"),
        ]
        result = agg.aggregate_metrics(rows)
        assert list(result.keys()) == sorted(result.keys())


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_writes_weekly_summary_json(self, tmp_path: Path) -> None:
        agg = TeamAggregator(_make_teams_config())
        rows = [_make_row("alice@example.com"), _make_row("carol@example.com")]
        agg.generate(rows, tmp_path)
        out = tmp_path / "weekly_summary.json"
        assert out.exists()

    def test_json_structure(self, tmp_path: Path) -> None:
        agg = TeamAggregator(_make_teams_config())
        rows = [_make_row("alice@example.com")]
        summary = agg.generate(rows, tmp_path)
        assert "generated_at" in summary
        assert "team_count" in summary
        assert "teams" in summary
        assert summary["team_count"] == 1

    def test_json_file_valid(self, tmp_path: Path) -> None:
        agg = TeamAggregator(_make_teams_config())
        rows = [_make_row("bob@example.com")]
        agg.generate(rows, tmp_path)
        data = json.loads((tmp_path / "weekly_summary.json").read_text())
        assert "Backend" in data["teams"]

    def test_empty_teams_config_returns_empty_dict(self, tmp_path: Path) -> None:
        agg = TeamAggregator(TeamsConfig())
        result = agg.generate([_make_row("x@x.com")], tmp_path)
        assert result == {}
        assert not (tmp_path / "weekly_summary.json").exists()

    def test_none_config_returns_empty_dict(self, tmp_path: Path) -> None:
        agg = TeamAggregator(None)
        result = agg.generate([_make_row("x@x.com")], tmp_path)
        assert result == {}

    def test_disabled_teams_returns_empty_dict(self, tmp_path: Path) -> None:
        cfg = TeamsConfig(
            enabled=False,
            teams=[
                TeamConfig(
                    name="Backend",
                    members=[TeamMemberConfig(email="alice@example.com")],
                )
            ],
        )
        agg = TeamAggregator(cfg)
        result = agg.generate([_make_row("alice@example.com")], tmp_path)
        assert result == {}
        assert not (tmp_path / "weekly_summary.json").exists()

    def test_all_developers_unknown_writes_empty_teams(self, tmp_path: Path) -> None:
        agg = TeamAggregator(_make_teams_config())
        rows = [_make_row("unknown@other.com")]
        summary = agg.generate(rows, tmp_path)
        # File should be written but teams will be empty
        assert summary["team_count"] == 0
        assert summary["teams"] == {}
