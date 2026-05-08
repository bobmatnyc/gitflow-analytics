"""Unit tests for the weekly_pr_metrics feature (issue #49).

These tests cover:
  * The aggregation helpers (prs_opened, prs_merged, pr_reviews_given).
  * CLI flag parsing (--week, --since).
  * Upsert idempotency.
  * Migration safety on an existing database with pre-existing
    pull_request_cache rows.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from gitflow_analytics.cli_pr_metrics import (
    aggregate_week,
    calculate_week_range,
    format_iso_week,
    parse_iso_week,
    upsert_weekly_metrics,
)
from gitflow_analytics.models.database import (
    Base,
    Database,
    PullRequestCache,
    WeeklyPRMetrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_db(tmp_path: Path) -> Database:
    """Spin up a fresh on-disk Database for each test."""
    db = Database(tmp_path / "gitflow_cache.db")
    return db


def _add_pr(
    db: Database,
    *,
    pr_number: int,
    repo_path: str = "octocat/hello",
    author: str = "alice",
    created_at: datetime,
    merged_at: datetime | None = None,
    is_merged: bool | None = None,
    reviewers: list[str] | None = None,
    change_requests_count: int = 0,
    revision_count: int | None = None,
) -> None:
    """Insert a synthetic PR row directly into pull_request_cache."""
    session = db.get_session()
    try:
        session.add(
            PullRequestCache(
                repo_path=repo_path,
                pr_number=pr_number,
                title=f"PR #{pr_number}",
                author=author,
                created_at=created_at,
                merged_at=merged_at,
                is_merged=is_merged,
                pr_state=("merged" if is_merged else "open"),
                reviewers=reviewers or [],
                approved_by=[],
                labels=[],
                commit_hashes=[],
                change_requests_count=change_requests_count,
                revision_count=revision_count,
            )
        )
        session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Helper / pure-function tests
# ---------------------------------------------------------------------------


class TestIsoWeekFormatting:
    def test_format_iso_week_typical(self):
        assert format_iso_week(datetime(2026, 4, 22, tzinfo=timezone.utc)) == "2026-W17"

    def test_format_iso_week_year_boundary(self):
        # 2024-12-30 is in ISO week 2025-W01
        assert format_iso_week(datetime(2024, 12, 30, tzinfo=timezone.utc)) == "2025-W01"

    def test_parse_iso_week_roundtrip(self):
        ws, we = parse_iso_week("2026-W17")
        assert ws.weekday() == 0  # Monday
        assert ws.tzinfo == timezone.utc
        assert we - ws == timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999)
        assert format_iso_week(ws) == "2026-W17"

    def test_parse_iso_week_invalid(self):
        with pytest.raises(ValueError):
            parse_iso_week("not-a-week")


class TestCalculateWeekRange:
    def test_default_returns_current_week(self):
        now = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)  # Wed of W17
        weeks = calculate_week_range(week=None, since=None, now=now)
        assert len(weeks) == 1
        assert weeks[0][0] == "2026-W17"

    def test_explicit_week(self):
        weeks = calculate_week_range(week="2026-W16", since=None)
        assert len(weeks) == 1
        label, ws, _ = weeks[0]
        assert label == "2026-W16"
        assert ws.weekday() == 0
        assert format_iso_week(ws) == "2026-W16"

    def test_since_backfill(self):
        now = datetime(2026, 4, 22, tzinfo=timezone.utc)
        weeks = calculate_week_range(week=None, since="2026-04-01", now=now)
        labels = [w[0] for w in weeks]
        # 2026-04-01 is in W14; current is W17 → expect W14, W15, W16, W17
        assert labels == ["2026-W14", "2026-W15", "2026-W16", "2026-W17"]

    def test_mutually_exclusive(self):
        with pytest.raises(ValueError):
            calculate_week_range(week="2026-W16", since="2026-01-01")

    def test_invalid_since(self):
        with pytest.raises(ValueError):
            calculate_week_range(week=None, since="not-a-date")


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------


class TestAggregateWeek:
    def test_prs_opened_counted_by_author(self, fresh_db: Database):
        ws, we = parse_iso_week("2026-W17")
        opened_at = ws + timedelta(days=1)
        _add_pr(fresh_db, pr_number=1, author="alice", created_at=opened_at)
        _add_pr(fresh_db, pr_number=2, author="alice", created_at=opened_at)
        _add_pr(fresh_db, pr_number=3, author="bob", created_at=opened_at)
        # Outside the week — must be ignored
        _add_pr(
            fresh_db,
            pr_number=4,
            author="alice",
            created_at=ws - timedelta(days=2),
        )

        result = aggregate_week(fresh_db, "2026-W17", ws, we)
        assert result["alice"]["prs_opened"] == 2
        assert result["bob"]["prs_opened"] == 1

    def test_prs_merged_counted_by_author(self, fresh_db: Database):
        ws, we = parse_iso_week("2026-W17")
        merge_time = ws + timedelta(days=2)
        _add_pr(
            fresh_db,
            pr_number=10,
            author="alice",
            created_at=ws - timedelta(days=14),
            merged_at=merge_time,
            is_merged=True,
        )
        # Closed-without-merge: must NOT count as merged
        _add_pr(
            fresh_db,
            pr_number=11,
            author="bob",
            created_at=ws - timedelta(days=14),
            merged_at=None,
            is_merged=False,
        )

        result = aggregate_week(fresh_db, "2026-W17", ws, we)
        assert result["alice"]["prs_merged"] == 1
        assert "bob" not in result or result["bob"]["prs_merged"] == 0

    def test_pr_reviews_given_from_reviewers_list(self, fresh_db: Database):
        ws, we = parse_iso_week("2026-W17")
        opened_at = ws + timedelta(days=1)
        _add_pr(
            fresh_db,
            pr_number=20,
            author="alice",
            created_at=opened_at,
            reviewers=["bob", "carol"],
        )
        _add_pr(
            fresh_db,
            pr_number=21,
            author="alice",
            created_at=opened_at,
            reviewers=["bob"],
        )

        result = aggregate_week(fresh_db, "2026-W17", ws, we)
        assert result["bob"]["pr_reviews_given"] == 2
        assert result["bob"]["pr_comments_given"] == 2  # proxy
        assert result["carol"]["pr_reviews_given"] == 1
        # alice authored but did not review herself
        assert result["alice"]["pr_reviews_given"] == 0

    def test_empty_week_returns_empty_dict(self, fresh_db: Database):
        ws, we = parse_iso_week("2026-W17")
        result = aggregate_week(fresh_db, "2026-W17", ws, we)
        assert result == {}

    def test_reviewers_stored_as_json_text(self, fresh_db: Database):
        """Reviewers may come back as a JSON-encoded string from older rows."""
        ws, we = parse_iso_week("2026-W17")
        opened_at = ws + timedelta(days=1)
        # Use the SQLAlchemy session to write a row whose reviewers value is a
        # raw JSON string rather than a list.
        session = fresh_db.get_session()
        try:
            session.execute(
                text(
                    "INSERT INTO pull_request_cache "
                    "(repo_path, pr_number, author, created_at, reviewers) "
                    "VALUES (:r, :n, :a, :c, :rev)"
                ),
                {
                    "r": "octocat/hello",
                    "n": 999,
                    "a": "alice",
                    "c": opened_at,
                    "rev": json.dumps(["dave"]),
                },
            )
            session.commit()
        finally:
            session.close()

        result = aggregate_week(fresh_db, "2026-W17", ws, we)
        assert result["dave"]["pr_reviews_given"] == 1


# ---------------------------------------------------------------------------
# Upsert idempotency
# ---------------------------------------------------------------------------


class TestUpsertIdempotency:
    def test_rerun_same_week_overwrites(self, fresh_db: Database):
        ws, we = parse_iso_week("2026-W17")
        opened_at = ws + timedelta(days=1)
        _add_pr(fresh_db, pr_number=1, author="alice", created_at=opened_at)

        agg1 = aggregate_week(fresh_db, "2026-W17", ws, we)
        upsert_weekly_metrics(fresh_db, "2026-W17", agg1)

        # Verify single row
        session = fresh_db.get_session()
        try:
            rows = (
                session.query(WeeklyPRMetrics)
                .filter_by(engineer_identifier="alice", iso_week="2026-W17")
                .all()
            )
            assert len(rows) == 1
            assert cast(int, rows[0].prs_opened) == 1
        finally:
            session.close()

        # Add another PR for alice and re-run
        _add_pr(fresh_db, pr_number=2, author="alice", created_at=opened_at)
        agg2 = aggregate_week(fresh_db, "2026-W17", ws, we)
        upsert_weekly_metrics(fresh_db, "2026-W17", agg2)

        session = fresh_db.get_session()
        try:
            rows = (
                session.query(WeeklyPRMetrics)
                .filter_by(engineer_identifier="alice", iso_week="2026-W17")
                .all()
            )
            assert len(rows) == 1  # still single row — upserted, not duplicated
            assert cast(int, rows[0].prs_opened) == 2
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Migration safety
# ---------------------------------------------------------------------------


class TestMigrationSafety:
    """The new migration must not touch any pre-existing data."""

    def test_migration_preserves_existing_pull_request_cache(self, tmp_path: Path):
        db_path = tmp_path / "legacy.db"

        # First, build a database WITHOUT the weekly_pr_metrics table by
        # creating only the tables that exist on the old schema, then write a
        # PR row to it.
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        # Drop the new table to simulate an "older" schema
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS weekly_pr_metrics"))
            conn.commit()

        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        session.add(
            PullRequestCache(
                repo_path="octocat/hello",
                pr_number=42,
                title="legacy PR",
                author="alice",
                created_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
            )
        )
        session.commit()
        session.close()
        engine.dispose()

        # Re-open via the production Database wrapper, which will run
        # _apply_migrations() and create weekly_pr_metrics non-destructively.
        db = Database(db_path)

        # The legacy PR must still be there.
        s = db.get_session()
        try:
            existing = s.query(PullRequestCache).filter_by(pr_number=42).one()
            assert str(existing.author) == "alice"
            assert str(existing.title) == "legacy PR"
        finally:
            s.close()

        # The new table must exist and be empty.
        s = db.get_session()
        try:
            new_rows = s.query(WeeklyPRMetrics).all()
            assert new_rows == []
        finally:
            s.close()

        # Confirm the table is queryable via raw SQL too.
        assert db.engine is not None, "Database engine not initialized"
        with db.engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='weekly_pr_metrics'"
                )
            )
            assert result.fetchone() is not None

    def test_migration_idempotent(self, tmp_path: Path):
        db_path = tmp_path / "idempotent.db"
        # Two consecutive opens must not error out.
        Database(db_path)
        Database(db_path)


# ---------------------------------------------------------------------------
# Issue #66: pr_merge_rate, avg_cycle_time_hrs, change_requests_received,
# avg_revisions_per_pr
# ---------------------------------------------------------------------------


class TestIssue66DerivedMetrics:
    """Tests for the four derived metrics added in issue #66."""

    def test_pr_merge_rate_basic(self, fresh_db: Database):
        ws, we = parse_iso_week("2026-W17")
        opened_at = ws + timedelta(days=1)
        merge_time = ws + timedelta(days=3)

        # alice: 2 opens, 1 merge -> 0.5
        _add_pr(
            fresh_db,
            pr_number=1,
            author="alice",
            created_at=opened_at,
            merged_at=merge_time,
            is_merged=True,
        )
        _add_pr(fresh_db, pr_number=2, author="alice", created_at=opened_at)

        result = aggregate_week(fresh_db, "2026-W17", ws, we)
        assert result["alice"]["prs_opened"] == 2
        assert result["alice"]["prs_merged"] == 1
        assert result["alice"]["pr_merge_rate"] == 0.5

    def test_pr_merge_rate_undefined_when_no_opens(self, fresh_db: Database):
        """Engineer who only reviewed (no opens) gets pr_merge_rate=None."""
        ws, we = parse_iso_week("2026-W17")
        opened_at = ws + timedelta(days=1)

        # alice opens, bob only reviews.
        _add_pr(
            fresh_db,
            pr_number=1,
            author="alice",
            created_at=opened_at,
            reviewers=["bob"],
        )

        result = aggregate_week(fresh_db, "2026-W17", ws, we)
        # bob: prs_opened=0 -> pr_merge_rate must be None (undefined, not 0.0)
        assert result["bob"]["prs_opened"] == 0
        assert result["bob"]["pr_merge_rate"] is None

    def test_pr_merge_rate_one_when_all_merged(self, fresh_db: Database):
        ws, we = parse_iso_week("2026-W17")
        opened_at = ws + timedelta(days=1)
        merge_time = ws + timedelta(days=2)

        _add_pr(
            fresh_db,
            pr_number=1,
            author="alice",
            created_at=opened_at,
            merged_at=merge_time,
            is_merged=True,
        )
        result = aggregate_week(fresh_db, "2026-W17", ws, we)
        assert result["alice"]["pr_merge_rate"] == 1.0

    def test_avg_cycle_time_hrs(self, fresh_db: Database):
        """avg_cycle_time_hrs averages (merged_at - created_at) for week's merges."""
        ws, we = parse_iso_week("2026-W17")
        # PR #1: opened 24h before merge in week
        _add_pr(
            fresh_db,
            pr_number=1,
            author="alice",
            created_at=ws + timedelta(days=1),
            merged_at=ws + timedelta(days=2),  # delta = 24h
            is_merged=True,
        )
        # PR #2: opened 12h before merge in week
        _add_pr(
            fresh_db,
            pr_number=2,
            author="alice",
            created_at=ws + timedelta(days=3),
            merged_at=ws + timedelta(days=3, hours=12),  # delta = 12h
            is_merged=True,
        )

        result = aggregate_week(fresh_db, "2026-W17", ws, we)
        # Average of 24h and 12h = 18h
        assert result["alice"]["avg_cycle_time_hrs"] == pytest.approx(18.0)

    def test_avg_cycle_time_hrs_none_when_no_merges(self, fresh_db: Database):
        """Engineer with opens but no merges in week gets None for cycle time."""
        ws, we = parse_iso_week("2026-W17")
        _add_pr(
            fresh_db,
            pr_number=1,
            author="alice",
            created_at=ws + timedelta(days=1),
        )
        result = aggregate_week(fresh_db, "2026-W17", ws, we)
        assert result["alice"]["avg_cycle_time_hrs"] is None

    def test_change_requests_received(self, fresh_db: Database):
        """change_requests_received sums change_requests_count for the week's opens."""
        ws, we = parse_iso_week("2026-W17")
        opened_at = ws + timedelta(days=1)

        _add_pr(
            fresh_db,
            pr_number=1,
            author="alice",
            created_at=opened_at,
            change_requests_count=2,
        )
        _add_pr(
            fresh_db,
            pr_number=2,
            author="alice",
            created_at=opened_at,
            change_requests_count=1,
        )
        # Out-of-week PR — must NOT contribute
        _add_pr(
            fresh_db,
            pr_number=3,
            author="alice",
            created_at=ws - timedelta(days=10),
            change_requests_count=99,
        )

        result = aggregate_week(fresh_db, "2026-W17", ws, we)
        assert result["alice"]["change_requests_received"] == 3

    def test_avg_revisions_per_pr(self, fresh_db: Database):
        """avg_revisions_per_pr averages revision_count across week's opens."""
        ws, we = parse_iso_week("2026-W17")
        opened_at = ws + timedelta(days=1)

        _add_pr(
            fresh_db,
            pr_number=1,
            author="alice",
            created_at=opened_at,
            revision_count=4,
        )
        _add_pr(
            fresh_db,
            pr_number=2,
            author="alice",
            created_at=opened_at,
            revision_count=2,
        )

        result = aggregate_week(fresh_db, "2026-W17", ws, we)
        assert result["alice"]["avg_revisions_per_pr"] == pytest.approx(3.0)

    def test_avg_revisions_per_pr_none_when_no_data(self, fresh_db: Database):
        """When all revision_count values are NULL, avg_revisions_per_pr is None.

        Use raw SQL to bypass the SQLAlchemy column default of 0 and create a
        genuine NULL revision_count, simulating a pre-v3 legacy PR row.
        """
        ws, we = parse_iso_week("2026-W17")
        opened_at = ws + timedelta(days=1)
        session = fresh_db.get_session()
        try:
            session.execute(
                text(
                    "INSERT INTO pull_request_cache "
                    "(repo_path, pr_number, author, created_at, revision_count) "
                    "VALUES (:r, :n, :a, :c, NULL)"
                ),
                {"r": "octocat/hello", "n": 500, "a": "alice", "c": opened_at},
            )
            session.commit()
        finally:
            session.close()
        result = aggregate_week(fresh_db, "2026-W17", ws, we)
        assert result["alice"]["avg_revisions_per_pr"] is None

    def test_upsert_persists_new_columns(self, fresh_db: Database):
        """Upsert writes the four new columns and they round-trip through the DB."""
        ws, we = parse_iso_week("2026-W17")
        opened_at = ws + timedelta(days=1)
        merge_time = ws + timedelta(days=2)

        _add_pr(
            fresh_db,
            pr_number=1,
            author="alice",
            created_at=opened_at,
            merged_at=merge_time,
            is_merged=True,
            change_requests_count=2,
            revision_count=3,
        )
        agg = aggregate_week(fresh_db, "2026-W17", ws, we)
        upsert_weekly_metrics(fresh_db, "2026-W17", agg)

        session = fresh_db.get_session()
        try:
            row = (
                session.query(WeeklyPRMetrics)
                .filter_by(engineer_identifier="alice", iso_week="2026-W17")
                .one()
            )
            assert cast(float, row.pr_merge_rate) == 1.0
            assert cast(float, row.avg_cycle_time_hrs) == pytest.approx(24.0)
            assert cast(int, row.change_requests_received) == 2
            assert cast(float, row.avg_revisions_per_pr) == pytest.approx(3.0)
        finally:
            session.close()


class TestIssue66MigrationSafety:
    """The v13 migration must add columns without disturbing existing rows."""

    def test_v13_migration_adds_columns_to_existing_table(self, tmp_path: Path):
        """An existing weekly_pr_metrics table without v13 columns gets them added."""
        db_path = tmp_path / "legacy_v11.db"

        # Build a database that has the v11 schema but NOT v13 columns.
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE weekly_pr_metrics (
                      engineer_identifier TEXT NOT NULL,
                      iso_week TEXT NOT NULL,
                      prs_opened INTEGER NOT NULL DEFAULT 0,
                      prs_merged INTEGER NOT NULL DEFAULT 0,
                      pr_comments_given INTEGER NOT NULL DEFAULT 0,
                      pr_reviews_given INTEGER NOT NULL DEFAULT 0,
                      computed_at DATETIME,
                      PRIMARY KEY (engineer_identifier, iso_week)
                    )
                    """
                )
            )
            # Insert a legacy row WITHOUT the v13 columns.
            conn.execute(
                text(
                    "INSERT INTO weekly_pr_metrics "
                    "(engineer_identifier, iso_week, prs_opened, prs_merged, "
                    " pr_comments_given, pr_reviews_given) "
                    "VALUES ('legacy_user', '2025-W01', 5, 3, 1, 1)"
                )
            )
            conn.commit()
        engine.dispose()

        # Re-open via the production wrapper which runs _apply_migrations
        # and therefore _migrate_weekly_pr_metrics_v13.
        db = Database(db_path)

        # Verify columns now exist.
        assert db.engine is not None
        with db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(weekly_pr_metrics)"))
            columns = {row[1] for row in result}
        for col in (
            "pr_merge_rate",
            "avg_cycle_time_hrs",
            "change_requests_received",
            "avg_revisions_per_pr",
        ):
            assert col in columns, f"v13 migration failed to add column {col}"

        # Legacy row preserved with NULL/0 in new columns.
        s = db.get_session()
        try:
            row = (
                s.query(WeeklyPRMetrics)
                .filter_by(engineer_identifier="legacy_user", iso_week="2025-W01")
                .one()
            )
            assert cast(int, row.prs_opened) == 5
            assert cast(int, row.prs_merged) == 3
            # New columns: REAL defaults to NULL, INTEGER defaults to 0.
            assert row.pr_merge_rate is None
            assert row.avg_cycle_time_hrs is None
            assert cast(int, row.change_requests_received) == 0
            assert row.avg_revisions_per_pr is None
        finally:
            s.close()
