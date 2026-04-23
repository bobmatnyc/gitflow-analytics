"""Tests for BoilerplateFilter — detect and act on bulk auto-generated commits (Issue #28).

Covers:
    * Config dataclass defaults
    * Flag detection at, above, and below thresholds
    * All three action modes: flag / exclude_from_averages / exclude
    * Edge cases: zero commits, single commit, exactly at threshold
    * Disabled filter = pure no-op
    * Weekly aggregation and commit-level integration helpers
    * TeamAggregator honours ``excluded_from_averages``
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gitflow_analytics.config.schema import (
    BoilerplateFilterConfig,
    TeamConfig,
    TeamMemberConfig,
    TeamsConfig,
)
from gitflow_analytics.reports.boilerplate_filter import (
    ACTION_EXCLUDE,
    ACTION_EXCLUDE_FROM_AVERAGES,
    ACTION_FLAG,
    CLASSIFICATION_CLEAN,
    CLASSIFICATION_FLAGGED,
    FIELD_BOILERPLATE_FLAG,
    FIELD_BOILERPLATE_LABEL,
    FIELD_EXCLUDED_FROM_AVERAGES,
    BoilerplateClassification,
    BoilerplateFilter,
    _developer_key,
    _week_start,
    aggregate_weekly_developer_metrics,
    apply_boilerplate_filter_to_commits,
)
from gitflow_analytics.reports.team_aggregator import TeamAggregator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics(
    *,
    total_commits: int = 0,
    lines_added: int = 0,
    developer_id: str = "alice@example.com",
) -> dict[str, Any]:
    """Build a minimal per-developer-week metrics dict."""
    return {
        "developer_id": developer_id,
        "developer_email": developer_id,
        "total_commits": total_commits,
        "lines_added": lines_added,
    }


def _commit(
    *,
    developer_id: str = "alice@example.com",
    lines_added: int = 100,
    ts: datetime | None = None,
) -> dict[str, Any]:
    """Build a minimal per-commit proxy row."""
    return {
        "developer_id": developer_id,
        "developer_email": developer_id,
        "total_commits": 1,
        "lines_added": lines_added,
        "lines_deleted": 0,
        "timestamp": ts or datetime(2026, 4, 13, 12, 0, tzinfo=timezone.utc),
    }


# ---------------------------------------------------------------------------
# BoilerplateFilterConfig dataclass
# ---------------------------------------------------------------------------


class TestBoilerplateFilterConfigDefaults:
    def test_defaults_match_spec(self) -> None:
        cfg = BoilerplateFilterConfig()
        assert cfg.enabled is False
        assert cfg.avg_lines_per_commit_threshold == 500
        assert cfg.total_lines_threshold == 10000
        assert cfg.action == "flag"
        assert cfg.flag_label == "boilerplate"

    def test_custom_values(self) -> None:
        cfg = BoilerplateFilterConfig(
            enabled=True,
            avg_lines_per_commit_threshold=200,
            total_lines_threshold=5000,
            action="exclude",
            flag_label="auto_generated",
        )
        assert cfg.enabled is True
        assert cfg.action == "exclude"
        assert cfg.flag_label == "auto_generated"


# ---------------------------------------------------------------------------
# Filter classify() — disabled, clean, flagged
# ---------------------------------------------------------------------------


class TestClassifyDisabled:
    def test_disabled_filter_always_clean(self) -> None:
        f = BoilerplateFilter(BoilerplateFilterConfig(enabled=False))
        result = f.classify(_metrics(total_commits=1, lines_added=1_000_000))
        assert result.classification == CLASSIFICATION_CLEAN

    def test_none_config_treated_as_disabled(self) -> None:
        f = BoilerplateFilter(None)
        assert f.enabled is False
        result = f.classify(_metrics(total_commits=1, lines_added=1_000_000))
        assert result.classification == CLASSIFICATION_CLEAN


class TestClassifyClean:
    def _filter(self) -> BoilerplateFilter:
        return BoilerplateFilter(
            BoilerplateFilterConfig(
                enabled=True,
                avg_lines_per_commit_threshold=500,
                total_lines_threshold=10000,
            )
        )

    def test_small_activity_is_clean(self) -> None:
        result = self._filter().classify(_metrics(total_commits=5, lines_added=400))
        assert result.classification == CLASSIFICATION_CLEAN
        assert result.reason == ""

    def test_average_below_threshold_clean(self) -> None:
        # 5 commits × 499 = 2495 lines, avg=499
        result = self._filter().classify(_metrics(total_commits=5, lines_added=2_495))
        assert result.classification == CLASSIFICATION_CLEAN

    def test_total_below_threshold_clean(self) -> None:
        # 100 commits × 99 = 9900 lines, avg=99
        result = self._filter().classify(_metrics(total_commits=100, lines_added=9_900))
        assert result.classification == CLASSIFICATION_CLEAN

    def test_zero_commits_zero_lines_clean(self) -> None:
        result = self._filter().classify(_metrics(total_commits=0, lines_added=0))
        assert result.classification == CLASSIFICATION_CLEAN
        assert result.avg_lines_per_commit == 0.0


class TestClassifyFlagged:
    def _filter(self) -> BoilerplateFilter:
        return BoilerplateFilter(
            BoilerplateFilterConfig(
                enabled=True,
                avg_lines_per_commit_threshold=500,
                total_lines_threshold=10000,
            )
        )

    def test_avg_above_threshold_flagged(self) -> None:
        # 2 commits × 1000 = 2000 lines, avg=1000 > 500
        result = self._filter().classify(_metrics(total_commits=2, lines_added=2_000))
        assert result.classification == CLASSIFICATION_FLAGGED
        assert "avg_lines_per_commit" in result.reason

    def test_total_above_threshold_flagged(self) -> None:
        # 100 commits × 200 = 20000 lines, avg=200 < 500 but total > 10000
        result = self._filter().classify(_metrics(total_commits=100, lines_added=20_000))
        assert result.classification == CLASSIFICATION_FLAGGED
        assert "total_lines" in result.reason

    def test_both_thresholds_exceeded_both_reasons(self) -> None:
        # avg=2000 > 500, total=20000 > 10000
        result = self._filter().classify(_metrics(total_commits=10, lines_added=20_000))
        assert result.classification == CLASSIFICATION_FLAGGED
        assert "avg_lines_per_commit" in result.reason
        assert "total_lines" in result.reason

    def test_exactly_at_avg_threshold_clean(self) -> None:
        # Strictly greater than is required — exactly at threshold is clean.
        # 2 commits × 500 = 1000 lines, avg=500.0
        result = self._filter().classify(_metrics(total_commits=2, lines_added=1_000))
        assert result.classification == CLASSIFICATION_CLEAN

    def test_exactly_at_total_threshold_clean(self) -> None:
        # 100 commits × 100 = 10000 total, avg=100 (both at/below thresholds)
        result = self._filter().classify(_metrics(total_commits=100, lines_added=10_000))
        assert result.classification == CLASSIFICATION_CLEAN

    def test_one_above_avg_threshold_flagged(self) -> None:
        # 1 commit × 501 lines → avg=501 > 500 → flagged
        result = self._filter().classify(_metrics(total_commits=1, lines_added=501))
        assert result.classification == CLASSIFICATION_FLAGGED

    def test_one_above_total_threshold_flagged(self) -> None:
        # total=10001 > 10000 → flagged
        result = self._filter().classify(_metrics(total_commits=1000, lines_added=10_001))
        assert result.classification == CLASSIFICATION_FLAGGED


# ---------------------------------------------------------------------------
# Filter apply() — all three actions on weekly metrics
# ---------------------------------------------------------------------------


class TestApplyDisabled:
    def test_disabled_returns_copy_unchanged(self) -> None:
        f = BoilerplateFilter(BoilerplateFilterConfig(enabled=False))
        rows = [_metrics(total_commits=1, lines_added=5_000_000)]
        result = f.apply(rows)
        assert result == rows
        # Ensure FIELD_BOILERPLATE_FLAG was NOT added.
        assert FIELD_BOILERPLATE_FLAG not in result[0]

    def test_disabled_does_not_mutate_input(self) -> None:
        f = BoilerplateFilter(BoilerplateFilterConfig(enabled=False))
        row = _metrics(total_commits=10, lines_added=50_000)
        original = dict(row)
        f.apply([row])
        assert row == original


class TestApplyActionFlag:
    def _filter(self) -> BoilerplateFilter:
        return BoilerplateFilter(
            BoilerplateFilterConfig(
                enabled=True,
                avg_lines_per_commit_threshold=500,
                total_lines_threshold=10_000,
                action=ACTION_FLAG,
                flag_label="boilerplate",
            )
        )

    def test_flagged_row_annotated(self) -> None:
        rows = [_metrics(total_commits=2, lines_added=5_000)]  # avg=2500
        result = self._filter().apply(rows)
        assert len(result) == 1
        assert result[0][FIELD_BOILERPLATE_FLAG] is True
        assert result[0][FIELD_BOILERPLATE_LABEL] == "boilerplate"

    def test_flag_has_reason(self) -> None:
        rows = [_metrics(total_commits=2, lines_added=5_000)]
        result = self._filter().apply(rows)
        assert "boilerplate_reason" in result[0]
        assert "avg_lines_per_commit" in result[0]["boilerplate_reason"]

    def test_clean_row_not_flagged(self) -> None:
        rows = [_metrics(total_commits=10, lines_added=1000)]
        result = self._filter().apply(rows)
        assert FIELD_BOILERPLATE_FLAG not in result[0]

    def test_mixed_rows(self) -> None:
        rows = [
            _metrics(developer_id="alice", total_commits=10, lines_added=1_000),
            _metrics(developer_id="bob", total_commits=2, lines_added=10_000),
        ]
        result = self._filter().apply(rows)
        assert len(result) == 2
        alice, bob = result
        assert FIELD_BOILERPLATE_FLAG not in alice
        assert bob[FIELD_BOILERPLATE_FLAG] is True

    def test_flag_does_not_set_exclude_from_averages(self) -> None:
        rows = [_metrics(total_commits=2, lines_added=5_000)]
        result = self._filter().apply(rows)
        assert result[0].get(FIELD_EXCLUDED_FROM_AVERAGES) is not True

    def test_custom_flag_label(self) -> None:
        cfg = BoilerplateFilterConfig(
            enabled=True,
            avg_lines_per_commit_threshold=100,
            total_lines_threshold=10_000,
            action=ACTION_FLAG,
            flag_label="auto_generated",
        )
        result = BoilerplateFilter(cfg).apply([_metrics(total_commits=1, lines_added=500)])
        assert result[0][FIELD_BOILERPLATE_LABEL] == "auto_generated"


class TestApplyActionExcludeFromAverages:
    def _filter(self) -> BoilerplateFilter:
        return BoilerplateFilter(
            BoilerplateFilterConfig(
                enabled=True,
                avg_lines_per_commit_threshold=500,
                total_lines_threshold=10_000,
                action=ACTION_EXCLUDE_FROM_AVERAGES,
            )
        )

    def test_flagged_row_kept(self) -> None:
        rows = [_metrics(total_commits=2, lines_added=5_000)]
        result = self._filter().apply(rows)
        assert len(result) == 1

    def test_flagged_row_has_exclude_marker(self) -> None:
        rows = [_metrics(total_commits=2, lines_added=5_000)]
        result = self._filter().apply(rows)
        assert result[0][FIELD_EXCLUDED_FROM_AVERAGES] is True

    def test_clean_row_kept_without_marker(self) -> None:
        rows = [_metrics(total_commits=10, lines_added=1000)]
        result = self._filter().apply(rows)
        assert result[0].get(FIELD_EXCLUDED_FROM_AVERAGES) is not True


class TestApplyActionExclude:
    def _filter(self) -> BoilerplateFilter:
        return BoilerplateFilter(
            BoilerplateFilterConfig(
                enabled=True,
                avg_lines_per_commit_threshold=500,
                total_lines_threshold=10_000,
                action=ACTION_EXCLUDE,
            )
        )

    def test_flagged_row_removed(self) -> None:
        rows = [_metrics(total_commits=2, lines_added=5_000)]
        result = self._filter().apply(rows)
        assert result == []

    def test_clean_row_kept(self) -> None:
        rows = [_metrics(total_commits=10, lines_added=1_000)]
        result = self._filter().apply(rows)
        assert len(result) == 1

    def test_mixed_exclude_drops_only_flagged(self) -> None:
        rows = [
            _metrics(developer_id="alice", total_commits=10, lines_added=1_000),
            _metrics(developer_id="bob", total_commits=2, lines_added=5_000),
        ]
        result = self._filter().apply(rows)
        assert len(result) == 1
        assert result[0]["developer_id"] == "alice"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_commit_small_flagged(self) -> None:
        # 1 commit × 10_000 lines → avg=10000 > 500 → flagged
        f = BoilerplateFilter(
            BoilerplateFilterConfig(
                enabled=True, avg_lines_per_commit_threshold=500, total_lines_threshold=10_000
            )
        )
        result = f.classify(_metrics(total_commits=1, lines_added=10_000))
        assert result.classification == CLASSIFICATION_FLAGGED

    def test_zero_commits_zero_lines_clean_even_enabled(self) -> None:
        f = BoilerplateFilter(BoilerplateFilterConfig(enabled=True))
        result = f.classify(_metrics(total_commits=0, lines_added=0))
        assert result.classification == CLASSIFICATION_CLEAN

    def test_zero_commits_nonzero_lines_does_not_crash(self) -> None:
        # Defensive: a malformed input with lines but no commits shouldn't
        # DivisionByZero.  Avg becomes 0; only the total_lines check matters.
        f = BoilerplateFilter(
            BoilerplateFilterConfig(
                enabled=True, avg_lines_per_commit_threshold=500, total_lines_threshold=10_000
            )
        )
        result = f.classify(_metrics(total_commits=0, lines_added=20_000))
        assert result.classification == CLASSIFICATION_FLAGGED
        assert "total_lines" in result.reason
        assert result.avg_lines_per_commit == 0.0

    def test_empty_metrics_list_returns_empty(self) -> None:
        f = BoilerplateFilter(BoilerplateFilterConfig(enabled=True))
        assert f.apply([]) == []

    def test_missing_keys_default_to_zero(self) -> None:
        f = BoilerplateFilter(BoilerplateFilterConfig(enabled=True))
        result = f.classify({})  # no total_commits, no lines_added
        assert result.classification == CLASSIFICATION_CLEAN
        assert result.total_lines == 0

    def test_apply_does_not_mutate_input_rows(self) -> None:
        f = BoilerplateFilter(
            BoilerplateFilterConfig(
                enabled=True, avg_lines_per_commit_threshold=100, action=ACTION_FLAG
            )
        )
        row = _metrics(total_commits=1, lines_added=500)
        original = dict(row)
        f.apply([row])
        assert row == original


# ---------------------------------------------------------------------------
# rows_for_averaging helper
# ---------------------------------------------------------------------------


class TestRowsForAveraging:
    def test_excludes_rows_marked_excluded(self) -> None:
        rows = [
            {FIELD_EXCLUDED_FROM_AVERAGES: True, "total_commits": 5},
            {"total_commits": 3},
        ]
        assert len(BoilerplateFilter.rows_for_averaging(rows)) == 1

    def test_keeps_all_when_none_excluded(self) -> None:
        rows = [{"total_commits": 1}, {"total_commits": 2}]
        assert BoilerplateFilter.rows_for_averaging(rows) == rows

    def test_empty_list(self) -> None:
        assert BoilerplateFilter.rows_for_averaging([]) == []


# ---------------------------------------------------------------------------
# Week-start and developer-key helpers
# ---------------------------------------------------------------------------


class TestWeekStart:
    def test_none_returns_none(self) -> None:
        assert _week_start(None) is None

    def test_monday_is_unchanged(self) -> None:
        monday = date(2026, 4, 20)  # a Monday
        assert _week_start(monday) == monday

    def test_sunday_maps_to_previous_monday(self) -> None:
        sunday = date(2026, 4, 19)
        assert _week_start(sunday) == date(2026, 4, 13)

    def test_datetime_accepted(self) -> None:
        dt = datetime(2026, 4, 17, 15, 30, tzinfo=timezone.utc)  # Friday
        assert _week_start(dt) == date(2026, 4, 13)


class TestDeveloperKey:
    def test_prefers_developer_id(self) -> None:
        row = {"developer_id": "id1", "developer_email": "e@x.com"}
        assert _developer_key(row) == "id1"

    def test_fallback_to_email(self) -> None:
        row = {"developer_email": "e@x.com"}
        assert _developer_key(row) == "e@x.com"

    def test_fallback_to_author(self) -> None:
        row = {"author": "alice"}
        assert _developer_key(row) == "alice"

    def test_unknown_when_no_identity(self) -> None:
        assert _developer_key({}) == "unknown"


# ---------------------------------------------------------------------------
# aggregate_weekly_developer_metrics
# ---------------------------------------------------------------------------


class TestAggregateWeekly:
    def test_single_developer_single_week(self) -> None:
        commits = [_commit(lines_added=100), _commit(lines_added=200)]
        result = aggregate_weekly_developer_metrics(commits)
        assert len(result) == 1
        key = next(iter(result))
        bucket = result[key]
        assert bucket["total_commits"] == 2
        assert bucket["lines_added"] == 300

    def test_two_developers_split(self) -> None:
        commits = [
            _commit(developer_id="alice", lines_added=100),
            _commit(developer_id="bob", lines_added=200),
        ]
        result = aggregate_weekly_developer_metrics(commits)
        assert len(result) == 2

    def test_two_weeks_split(self) -> None:
        w1 = datetime(2026, 4, 13, tzinfo=timezone.utc)  # Monday wk1
        w2 = datetime(2026, 4, 20, tzinfo=timezone.utc)  # Monday wk2
        commits = [
            _commit(ts=w1, lines_added=100),
            _commit(ts=w2, lines_added=200),
        ]
        result = aggregate_weekly_developer_metrics(commits)
        assert len(result) == 2

    def test_empty_input(self) -> None:
        assert aggregate_weekly_developer_metrics([]) == {}

    def test_undated_commits_bucketed_to_none_week(self) -> None:
        commits = [_commit(ts=None)]  # no timestamp
        commits[0]["timestamp"] = None
        result = aggregate_weekly_developer_metrics(commits)
        assert len(result) == 1
        key = next(iter(result))
        assert key[1] is None


# ---------------------------------------------------------------------------
# apply_boilerplate_filter_to_commits — integration helper
# ---------------------------------------------------------------------------


class TestApplyToCommits:
    def _filter(self, action: str = ACTION_FLAG) -> BoilerplateFilter:
        return BoilerplateFilter(
            BoilerplateFilterConfig(
                enabled=True,
                avg_lines_per_commit_threshold=500,
                total_lines_threshold=10_000,
                action=action,
            )
        )

    def test_disabled_is_passthrough(self) -> None:
        f = BoilerplateFilter(BoilerplateFilterConfig(enabled=False))
        commits = [_commit(lines_added=1_000_000)]
        result = apply_boilerplate_filter_to_commits(f, commits)
        assert len(result) == 1
        assert FIELD_BOILERPLATE_FLAG not in result[0]

    def test_flag_action_annotates_all_commits_in_flagged_week(self) -> None:
        # 2 commits, each 5000 lines → weekly avg=5000 > 500 → flagged
        ts = datetime(2026, 4, 13, 12, tzinfo=timezone.utc)
        commits = [
            _commit(ts=ts, lines_added=5000),
            _commit(ts=ts, lines_added=5000),
        ]
        result = apply_boilerplate_filter_to_commits(self._filter(ACTION_FLAG), commits)
        assert len(result) == 2
        assert all(c[FIELD_BOILERPLATE_FLAG] for c in result)

    def test_exclude_action_drops_flagged_commits(self) -> None:
        ts = datetime(2026, 4, 13, 12, tzinfo=timezone.utc)
        commits = [_commit(ts=ts, lines_added=20_000)]
        result = apply_boilerplate_filter_to_commits(self._filter(ACTION_EXCLUDE), commits)
        assert result == []

    def test_exclude_from_averages_marks_flagged_commits(self) -> None:
        ts = datetime(2026, 4, 13, 12, tzinfo=timezone.utc)
        commits = [_commit(ts=ts, lines_added=20_000)]
        result = apply_boilerplate_filter_to_commits(
            self._filter(ACTION_EXCLUDE_FROM_AVERAGES), commits
        )
        assert len(result) == 1
        assert result[0][FIELD_EXCLUDED_FROM_AVERAGES] is True

    def test_clean_weeks_pass_through_unchanged(self) -> None:
        ts = datetime(2026, 4, 13, 12, tzinfo=timezone.utc)
        commits = [_commit(ts=ts, lines_added=100) for _ in range(3)]
        result = apply_boilerplate_filter_to_commits(self._filter(ACTION_FLAG), commits)
        assert len(result) == 3
        assert all(FIELD_BOILERPLATE_FLAG not in c for c in result)

    def test_mixed_clean_and_flagged_developers(self) -> None:
        ts = datetime(2026, 4, 13, 12, tzinfo=timezone.utc)
        commits = [
            _commit(developer_id="alice", ts=ts, lines_added=100),
            _commit(developer_id="bob", ts=ts, lines_added=20_000),
        ]
        result = apply_boilerplate_filter_to_commits(self._filter(ACTION_FLAG), commits)
        alice_rows = [r for r in result if r["developer_id"] == "alice"]
        bob_rows = [r for r in result if r["developer_id"] == "bob"]
        assert len(alice_rows) == 1
        assert FIELD_BOILERPLATE_FLAG not in alice_rows[0]
        assert len(bob_rows) == 1
        assert bob_rows[0][FIELD_BOILERPLATE_FLAG] is True

    def test_same_developer_two_weeks_flagged_separately(self) -> None:
        # Week 1: 20000 lines (flagged), Week 2: 100 lines (clean)
        w1 = datetime(2026, 4, 13, 12, tzinfo=timezone.utc)
        w2 = datetime(2026, 4, 20, 12, tzinfo=timezone.utc)
        commits = [
            _commit(ts=w1, lines_added=20_000),
            _commit(ts=w2, lines_added=100),
        ]
        result = apply_boilerplate_filter_to_commits(self._filter(ACTION_FLAG), commits)
        assert len(result) == 2
        flagged = [r for r in result if r.get(FIELD_BOILERPLATE_FLAG)]
        clean = [r for r in result if not r.get(FIELD_BOILERPLATE_FLAG)]
        assert len(flagged) == 1 and len(clean) == 1

    def test_empty_commit_list_returns_empty(self) -> None:
        assert apply_boilerplate_filter_to_commits(self._filter(), []) == []

    def test_does_not_mutate_input_commits(self) -> None:
        ts = datetime(2026, 4, 13, 12, tzinfo=timezone.utc)
        row = _commit(ts=ts, lines_added=50_000)
        original = dict(row)
        apply_boilerplate_filter_to_commits(self._filter(ACTION_FLAG), [row])
        assert row == original


# ---------------------------------------------------------------------------
# Integration with TeamAggregator
# ---------------------------------------------------------------------------


class TestTeamAggregatorIntegration:
    def _teams(self) -> TeamsConfig:
        return TeamsConfig(
            enabled=True,
            teams=[
                TeamConfig(
                    name="Backend",
                    members=[
                        TeamMemberConfig(email="alice@example.com"),
                        TeamMemberConfig(email="bob@example.com"),
                    ],
                )
            ],
        )

    def _team_row(
        self, email: str, *, total: int = 3, lines_added: int = 300, excluded: bool = False
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "developer_id": email,
            "developer_email": email,
            "developer_name": email.split("@")[0],
            "total_commits": total,
            "feature_commits": 1,
            "bug_fix_commits": 0,
            "refactor_commits": 0,
            "documentation_commits": 0,
            "maintenance_commits": 0,
            "test_commits": 0,
            "lines_added": lines_added,
            "lines_deleted": 0,
            "story_points": 0,
            "tracked_commits": 0,
            "untracked_commits": total,
            "ai_assisted_commits": 0,
            "ai_generated_commits": 0,
        }
        if excluded:
            row[FIELD_EXCLUDED_FROM_AVERAGES] = True
        return row

    def test_excluded_rows_ignored_by_team_aggregator(self) -> None:
        agg = TeamAggregator(self._teams())
        rows = [
            self._team_row("alice@example.com", total=5, lines_added=500),
            self._team_row("bob@example.com", total=2, lines_added=50_000, excluded=True),
        ]
        result = agg.aggregate_metrics(rows)
        # Only alice's 5 commits count — bob is excluded from team averages.
        assert result["Backend"]["total_commits"] == 5
        assert result["Backend"]["lines_added"] == 500

    def test_non_excluded_rows_still_aggregate(self) -> None:
        agg = TeamAggregator(self._teams())
        rows = [
            self._team_row("alice@example.com", total=3, lines_added=300),
            self._team_row("bob@example.com", total=4, lines_added=400),
        ]
        result = agg.aggregate_metrics(rows)
        assert result["Backend"]["total_commits"] == 7
        assert result["Backend"]["lines_added"] == 700


# ---------------------------------------------------------------------------
# Config loader integration (YAML → BoilerplateFilterConfig)
# ---------------------------------------------------------------------------


class TestConfigLoaderIntegration:
    def _write_config(self, tmp_path: Path, bp_block: str) -> Path:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"""
version: "1.0"
github:
  token: ""
  owner: "test"
repositories:
  - name: "test-repo"
    path: "{tmp_path}"

{bp_block}
""".strip()
        )
        return cfg

    def test_absent_block_uses_defaults(self, tmp_path: Path, monkeypatch: Any) -> None:
        from gitflow_analytics.config.loader import ConfigLoader

        monkeypatch.setenv("GITHUB_TOKEN", "x")
        cfg_path = self._write_config(tmp_path, "")
        cfg = ConfigLoader.load(cfg_path)
        assert cfg.boilerplate_filter.enabled is False
        assert cfg.boilerplate_filter.action == "flag"
        assert cfg.boilerplate_filter.avg_lines_per_commit_threshold == 500
        assert cfg.boilerplate_filter.total_lines_threshold == 10_000

    def test_enabled_with_custom_values(self, tmp_path: Path, monkeypatch: Any) -> None:
        from gitflow_analytics.config.loader import ConfigLoader

        monkeypatch.setenv("GITHUB_TOKEN", "x")
        cfg_path = self._write_config(
            tmp_path,
            """
boilerplate_filter:
  enabled: true
  avg_lines_per_commit_threshold: 200
  total_lines_threshold: 5000
  action: "exclude_from_averages"
  flag_label: "bulk_auto_generated"
""",
        )
        cfg = ConfigLoader.load(cfg_path)
        assert cfg.boilerplate_filter.enabled is True
        assert cfg.boilerplate_filter.avg_lines_per_commit_threshold == 200
        assert cfg.boilerplate_filter.total_lines_threshold == 5000
        assert cfg.boilerplate_filter.action == "exclude_from_averages"
        assert cfg.boilerplate_filter.flag_label == "bulk_auto_generated"

    def test_invalid_action_falls_back_to_flag(self, tmp_path: Path, monkeypatch: Any) -> None:
        from gitflow_analytics.config.loader import ConfigLoader

        monkeypatch.setenv("GITHUB_TOKEN", "x")
        cfg_path = self._write_config(
            tmp_path,
            """
boilerplate_filter:
  enabled: true
  action: "not_a_real_action"
""",
        )
        cfg = ConfigLoader.load(cfg_path)
        assert cfg.boilerplate_filter.action == "flag"


# ---------------------------------------------------------------------------
# BoilerplateClassification dataclass
# ---------------------------------------------------------------------------


class TestClassificationDataclass:
    def test_is_frozen(self) -> None:
        import dataclasses

        c = BoilerplateClassification(
            classification=CLASSIFICATION_CLEAN,
            reason="",
            avg_lines_per_commit=0.0,
            total_lines=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.classification = CLASSIFICATION_FLAGGED  # type: ignore[misc]

    def test_fields_exposed(self) -> None:
        c = BoilerplateClassification(
            classification=CLASSIFICATION_FLAGGED,
            reason="avg exceeded",
            avg_lines_per_commit=750.0,
            total_lines=1500,
        )
        assert c.classification == CLASSIFICATION_FLAGGED
        assert c.reason == "avg exceeded"
        assert c.avg_lines_per_commit == 750.0
        assert c.total_lines == 1500
