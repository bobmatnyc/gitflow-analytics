"""Tests for WeeklyTrends velocity columns (v8.0 migration).

Validates that:
1. All five velocity columns exist on the WeeklyTrends model.
2. Column types are correct (Integer / Float).
3. A fresh database (create_all) includes all velocity columns.
4. _migrate_weekly_trends_velocity_columns() adds columns to a legacy DB.
5. The migration preserves existing rows.
6. The migration is idempotent (safe to run twice).
7. Correct default values are applied to existing rows after migration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Float, Integer, create_engine, inspect, text

from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.models.database import Base
from gitflow_analytics.models.database_commit_models import WeeklyTrends

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VELOCITY_COLUMNS = [
    "prs_merged",
    "avg_cycle_time_hrs",
    "median_cycle_time_hrs",
    "avg_revision_count",
    "story_points_delivered",
]

INTEGER_VELOCITY_COLUMNS = {"prs_merged", "story_points_delivered"}
FLOAT_VELOCITY_COLUMNS = {"avg_cycle_time_hrs", "median_cycle_time_hrs", "avg_revision_count"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_legacy_weekly_trends_db(db_path: Path) -> None:
    """Create a weekly_trends table without the v8 velocity columns."""
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(
            text("""
            CREATE TABLE weekly_trends (
                id INTEGER PRIMARY KEY,
                week_start DATETIME NOT NULL,
                week_end DATETIME NOT NULL,
                developer_id VARCHAR NOT NULL,
                project_key VARCHAR NOT NULL,
                total_commits INTEGER DEFAULT 0,
                feature_commits INTEGER DEFAULT 0,
                bug_fix_commits INTEGER DEFAULT 0,
                refactor_commits INTEGER DEFAULT 0,
                total_commits_change REAL DEFAULT 0.0,
                feature_commits_change REAL DEFAULT 0.0,
                bug_fix_commits_change REAL DEFAULT 0.0,
                refactor_commits_change REAL DEFAULT 0.0,
                days_active INTEGER DEFAULT 0,
                avg_commits_per_day REAL DEFAULT 0.0,
                churn_rate_14d REAL DEFAULT 0.0,
                calculated_at DATETIME
            )
        """)
        )
        # Insert a legacy row to verify data preservation
        conn.execute(
            text("""
            INSERT INTO weekly_trends
                (week_start, week_end, developer_id, project_key, total_commits)
            VALUES
                ('2025-01-06', '2025-01-12', 'alice@example.com', 'PROJ', 7)
        """)
        )
        conn.commit()
    engine.dispose()


# ---------------------------------------------------------------------------
# 1 & 2. Model column presence and types
# ---------------------------------------------------------------------------


class TestWeeklyTrendsVelocityColumnsOnModel:
    """All velocity columns must be declared on the SQLAlchemy model."""

    def test_all_velocity_columns_exist(self) -> None:
        mapper = inspect(WeeklyTrends)
        column_names = {col.key for col in mapper.columns}
        for col in VELOCITY_COLUMNS:
            assert col in column_names, (
                f"Expected column '{col}' on WeeklyTrends. "
                "Check models/database_commit_models.py."
            )

    def test_integer_columns_have_integer_type(self) -> None:
        mapper = inspect(WeeklyTrends)
        for col_name in INTEGER_VELOCITY_COLUMNS:
            col = mapper.columns[col_name]
            assert isinstance(
                col.type, Integer
            ), f"Column '{col_name}' should be Integer but is {type(col.type).__name__}"

    def test_float_columns_have_float_type(self) -> None:
        mapper = inspect(WeeklyTrends)
        for col_name in FLOAT_VELOCITY_COLUMNS:
            col = mapper.columns[col_name]
            assert isinstance(
                col.type, Float
            ), f"Column '{col_name}' should be Float but is {type(col.type).__name__}"

    def test_pre_existing_columns_still_present(self) -> None:
        """Velocity additions must not remove any pre-existing WeeklyTrends columns."""
        pre_existing = [
            "total_commits",
            "feature_commits",
            "bug_fix_commits",
            "refactor_commits",
            "days_active",
            "avg_commits_per_day",
            "churn_rate_14d",
        ]
        mapper = inspect(WeeklyTrends)
        column_names = {col.key for col in mapper.columns}
        for col in pre_existing:
            assert col in column_names, f"Pre-existing column '{col}' unexpectedly missing."


# ---------------------------------------------------------------------------
# 3. Fresh database includes all velocity columns
# ---------------------------------------------------------------------------


class TestFreshDatabaseIncludesVelocityColumns:
    def test_fresh_db_has_velocity_columns(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(weekly_trends)"))
            actual_columns = {row[1] for row in result}

        for col in VELOCITY_COLUMNS:
            assert col in actual_columns, f"Fresh DB is missing column '{col}' in weekly_trends"


# ---------------------------------------------------------------------------
# 4 & 5. Migration adds columns without destroying rows
# ---------------------------------------------------------------------------


class TestMigrationAddsVelocityColumns:
    def test_migration_adds_all_velocity_columns(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        _create_legacy_weekly_trends_db(cache_dir / "gitflow_cache.db")

        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache.db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(weekly_trends)"))
            actual_columns = {row[1] for row in result}

        for col in VELOCITY_COLUMNS:
            assert (
                col in actual_columns
            ), f"v8.0 migration failed to add column '{col}' to legacy weekly_trends"

    def test_migration_preserves_existing_rows(self, tmp_path: Path) -> None:
        """Existing weekly_trends rows must survive the v8.0 migration."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        _create_legacy_weekly_trends_db(cache_dir / "gitflow_cache.db")

        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache.db.engine.connect() as conn:
            result = conn.execute(text("SELECT developer_id, total_commits FROM weekly_trends"))
            row = result.fetchone()

        assert row is not None, "Existing row was deleted during v8.0 migration."
        assert row[0] == "alice@example.com"
        assert row[1] == 7


# ---------------------------------------------------------------------------
# 6. Idempotency
# ---------------------------------------------------------------------------


class TestMigrationIdempotent:
    def test_double_migration_no_duplicate_columns(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        _create_legacy_weekly_trends_db(cache_dir / "gitflow_cache.db")

        cache1 = GitAnalysisCache(cache_dir, ttl_hours=24)
        del cache1

        cache2 = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache2.db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(weekly_trends)"))
            columns = [row[1] for row in result]

        assert len(columns) == len(set(columns)), "Duplicate columns after double migration."
        column_set = set(columns)
        for col in VELOCITY_COLUMNS:
            assert col in column_set, f"Column '{col}' missing after double migration."


# ---------------------------------------------------------------------------
# 7. Default values applied to migrated rows
# ---------------------------------------------------------------------------


class TestMigrationDefaultValues:
    def test_integer_defaults_are_zero(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        _create_legacy_weekly_trends_db(cache_dir / "gitflow_cache.db")

        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache.db.engine.connect() as conn:
            result = conn.execute(
                text("SELECT prs_merged, story_points_delivered " "FROM weekly_trends LIMIT 1")
            )
            row = result.fetchone()

        assert row is not None
        # SQLite DEFAULT 0 means existing rows will have 0 after the migration
        assert row[0] == 0, "prs_merged default should be 0 for pre-existing rows"
        assert row[1] == 0, "story_points_delivered default should be 0"
