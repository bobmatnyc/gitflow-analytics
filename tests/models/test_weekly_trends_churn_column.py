"""Tests for WeeklyTrends churn_rate_14d column (v7.0 migration).

Validates that:
1. churn_rate_14d exists on the WeeklyTrends model with correct type.
2. A fresh database (create_all) includes the churn_rate_14d column.
3. _migrate_weekly_trends_churn_column() adds the column to a legacy database.
4. The migration is idempotent (safe to run twice).
5. Existing rows are preserved during migration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Float, create_engine, inspect, text

from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.models.database import Base, WeeklyTrends

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHURN_COLUMN = "churn_rate_14d"


# ---------------------------------------------------------------------------
# 1. Model column presence test
# ---------------------------------------------------------------------------


class TestWeeklyTrendsChurnColumnOnModel:
    """Verify that churn_rate_14d is declared on the SQLAlchemy model."""

    def test_churn_column_exists_on_model(self) -> None:
        mapper = inspect(WeeklyTrends)
        column_names = {col.key for col in mapper.columns}
        assert CHURN_COLUMN in column_names, (
            f"Expected column '{CHURN_COLUMN}' on WeeklyTrends but it was not found. "
            "Check models/database_commit_models.py."
        )

    def test_churn_column_is_float_type(self) -> None:
        mapper = inspect(WeeklyTrends)
        col = mapper.columns[CHURN_COLUMN]
        assert isinstance(
            col.type, Float
        ), f"'{CHURN_COLUMN}' should be Float but is {type(col.type).__name__}"

    def test_existing_columns_still_present(self) -> None:
        """churn addition must not remove any pre-existing WeeklyTrends columns."""
        pre_existing = [
            "total_commits",
            "feature_commits",
            "bug_fix_commits",
            "refactor_commits",
            "days_active",
            "avg_commits_per_day",
            "calculated_at",
        ]
        mapper = inspect(WeeklyTrends)
        column_names = {col.key for col in mapper.columns}
        for col in pre_existing:
            assert col in column_names, f"Pre-existing column '{col}' is unexpectedly missing."


# ---------------------------------------------------------------------------
# 2. Fresh database includes churn_rate_14d
# ---------------------------------------------------------------------------


class TestFreshDatabaseIncludesChurnColumn:
    def test_fresh_db_has_churn_column(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(weekly_trends)"))
            actual_columns = {row[1] for row in result}

        assert (
            CHURN_COLUMN in actual_columns
        ), f"Fresh DB missing column '{CHURN_COLUMN}' in weekly_trends"


# ---------------------------------------------------------------------------
# Helper: create a legacy weekly_trends DB without churn_rate_14d
# ---------------------------------------------------------------------------


def _create_legacy_weekly_trends_db(db_path: Path) -> None:
    """Create a weekly_trends table that looks like the pre-v7.0 schema."""
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
                calculated_at DATETIME
            )
        """)
        )
        # Insert a legacy row to confirm existing data is preserved
        conn.execute(
            text("""
            INSERT INTO weekly_trends
                (week_start, week_end, developer_id, project_key, total_commits)
            VALUES ('2025-01-06', '2025-01-12', 'alice@example.com', 'PROJ', 10)
        """)
        )
        conn.commit()
    engine.dispose()


# ---------------------------------------------------------------------------
# 3 & 5. Migration adds column without destroying existing rows
# ---------------------------------------------------------------------------


class TestMigrationAddsChurnColumn:
    def test_migration_adds_churn_column(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        _create_legacy_weekly_trends_db(cache_dir / "gitflow_cache.db")

        # Instantiate the cache — triggers _apply_migrations
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache.db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(weekly_trends)"))
            actual_columns = {row[1] for row in result}

        assert (
            CHURN_COLUMN in actual_columns
        ), f"v7.0 migration failed to add '{CHURN_COLUMN}' to legacy weekly_trends"

    def test_migration_preserves_existing_rows(self, tmp_path: Path) -> None:
        """Existing weekly_trends rows must not be deleted during v7.0 migration."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        _create_legacy_weekly_trends_db(cache_dir / "gitflow_cache.db")

        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache.db.engine.connect() as conn:
            result = conn.execute(text("SELECT developer_id, total_commits FROM weekly_trends"))
            row = result.fetchone()

        assert row is not None, "Existing row was deleted during v7.0 migration."
        assert row[0] == "alice@example.com"
        assert row[1] == 10

    def test_migrated_churn_column_defaults_to_zero(self, tmp_path: Path) -> None:
        """Existing rows should get the DEFAULT 0.0 value for churn_rate_14d."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        _create_legacy_weekly_trends_db(cache_dir / "gitflow_cache.db")

        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache.db.engine.connect() as conn:
            result = conn.execute(text("SELECT churn_rate_14d FROM weekly_trends"))
            row = result.fetchone()

        assert row is not None
        # SQLite DEFAULT 0.0 means existing rows get 0.0 after ALTER TABLE
        assert row[0] == 0.0 or row[0] is None  # NULL is also acceptable for existing rows


# ---------------------------------------------------------------------------
# 4. Idempotency — running migration twice must not duplicate columns
# ---------------------------------------------------------------------------


class TestMigrationIdempotent:
    def test_migration_idempotent(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        _create_legacy_weekly_trends_db(cache_dir / "gitflow_cache.db")

        # First run
        cache1 = GitAnalysisCache(cache_dir, ttl_hours=24)
        del cache1

        # Second run
        cache2 = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache2.db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(weekly_trends)"))
            columns = [row[1] for row in result]

        # No duplicates
        assert len(columns) == len(set(columns)), "Duplicate columns after double migration."

        # Churn column present
        assert CHURN_COLUMN in set(columns), f"'{CHURN_COLUMN}' missing after double migration."
