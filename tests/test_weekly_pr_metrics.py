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
