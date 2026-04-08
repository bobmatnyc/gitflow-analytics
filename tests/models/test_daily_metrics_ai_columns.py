"""Tests for DailyMetrics AI tool tracking columns (v6.0 migration).

Validates that:
1. New AI columns exist on the DailyMetrics model with correct types.
2. A fresh database (create_all) includes all AI columns.
3. _migrate_daily_metrics_ai_columns() adds columns to a legacy database.
4. The migration is idempotent (safe to run twice).
5. Existing rows are preserved during migration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Integer, String, create_engine, inspect, text

from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.models.database import Base, DailyMetrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AI_COLUMNS = [
    "ai_assisted_commits",
    "ai_generated_commits",
    "ai_tool_primary",
    "ai_assisted_lines",
    "ai_generated_lines",
]


# ---------------------------------------------------------------------------
# 1. Model column presence tests
# ---------------------------------------------------------------------------


class TestDailyMetricsAiColumnsOnModel:
    """Verify that all AI columns are declared on the SQLAlchemy model."""

    def test_all_ai_columns_exist_on_model(self) -> None:
        mapper = inspect(DailyMetrics)
        column_names = {col.key for col in mapper.columns}
        for col in AI_COLUMNS:
            assert col in column_names, (
                f"Expected column '{col}' on DailyMetrics but it was not found. "
                "Check models/database_commit_models.py."
            )

    def test_integer_columns_are_integer_type(self) -> None:
        mapper = inspect(DailyMetrics)
        integer_cols = [
            "ai_assisted_commits",
            "ai_generated_commits",
            "ai_assisted_lines",
            "ai_generated_lines",
        ]
        for col_name in integer_cols:
            col = mapper.columns[col_name]
            assert isinstance(
                col.type, Integer
            ), f"Column '{col_name}' should be Integer but is {type(col.type).__name__}"

    def test_ai_tool_primary_is_string_type(self) -> None:
        mapper = inspect(DailyMetrics)
        col = mapper.columns["ai_tool_primary"]
        assert isinstance(
            col.type, String
        ), f"ai_tool_primary should be String but is {type(col.type).__name__}"

    def test_existing_columns_still_present(self) -> None:
        """AI additions must not remove any pre-existing DailyMetrics columns."""
        pre_existing = [
            "total_commits",
            "feature_commits",
            "bug_fix_commits",
            "merge_commits",
            "complex_commits",
            "lines_added",
            "lines_deleted",
        ]
        mapper = inspect(DailyMetrics)
        column_names = {col.key for col in mapper.columns}
        for col in pre_existing:
            assert col in column_names, f"Pre-existing column '{col}' is unexpectedly missing."


# ---------------------------------------------------------------------------
# 2. Fresh database includes all AI columns
# ---------------------------------------------------------------------------


class TestFreshDatabaseIncludesAiColumns:
    """Verify that create_all() on a fresh DB produces all AI columns."""

    def test_fresh_db_has_ai_columns(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(daily_metrics)"))
            actual_columns = {row[1] for row in result}

        for col in AI_COLUMNS:
            assert col in actual_columns, f"Fresh DB missing column '{col}' in daily_metrics"


# ---------------------------------------------------------------------------
# 3 & 5. Migration adds columns to legacy DB without destroying existing rows
# ---------------------------------------------------------------------------


def _create_legacy_daily_metrics_db(db_path: Path) -> None:
    """Create a daily_metrics table that looks like the pre-v6.0 schema (no AI columns)."""
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(
            text("""
            CREATE TABLE daily_metrics (
                id INTEGER PRIMARY KEY,
                date DATETIME NOT NULL,
                developer_id VARCHAR NOT NULL,
                project_key VARCHAR NOT NULL,
                developer_name VARCHAR NOT NULL,
                developer_email VARCHAR NOT NULL,
                feature_commits INTEGER DEFAULT 0,
                bug_fix_commits INTEGER DEFAULT 0,
                refactor_commits INTEGER DEFAULT 0,
                documentation_commits INTEGER DEFAULT 0,
                maintenance_commits INTEGER DEFAULT 0,
                test_commits INTEGER DEFAULT 0,
                style_commits INTEGER DEFAULT 0,
                build_commits INTEGER DEFAULT 0,
                other_commits INTEGER DEFAULT 0,
                total_commits INTEGER DEFAULT 0,
                files_changed INTEGER DEFAULT 0,
                lines_added INTEGER DEFAULT 0,
                lines_deleted INTEGER DEFAULT 0,
                story_points INTEGER DEFAULT 0,
                tracked_commits INTEGER DEFAULT 0,
                untracked_commits INTEGER DEFAULT 0,
                unique_tickets INTEGER DEFAULT 0,
                merge_commits INTEGER DEFAULT 0,
                complex_commits INTEGER DEFAULT 0,
                created_at DATETIME,
                updated_at DATETIME
            )
        """)
        )
        # Insert a legacy row to confirm existing data is preserved
        conn.execute(
            text("""
            INSERT INTO daily_metrics
                (date, developer_id, project_key, developer_name, developer_email, total_commits)
            VALUES ('2025-01-01', 'alice@example.com', 'PROJ', 'Alice', 'alice@example.com', 5)
        """)
        )
        conn.commit()
    engine.dispose()


class TestMigrationAddsAiColumns:
    """Verify _migrate_daily_metrics_ai_columns adds the five missing columns."""

    def test_migration_adds_all_ai_columns(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        _create_legacy_daily_metrics_db(cache_dir / "gitflow_cache.db")

        # Instantiate the cache — this triggers _apply_migrations
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache.db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(daily_metrics)"))
            actual_columns = {row[1] for row in result}

        for col in AI_COLUMNS:
            assert (
                col in actual_columns
            ), f"v6.0 migration failed to add column '{col}' to legacy daily_metrics"

    def test_migration_preserves_existing_rows(self, tmp_path: Path) -> None:
        """Existing daily_metrics rows must not be deleted during v6.0 migration."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        _create_legacy_daily_metrics_db(cache_dir / "gitflow_cache.db")

        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache.db.engine.connect() as conn:
            result = conn.execute(text("SELECT developer_id, total_commits FROM daily_metrics"))
            row = result.fetchone()

        assert row is not None, "Existing row was deleted during v6.0 migration."
        assert row[0] == "alice@example.com"
        assert row[1] == 5


# ---------------------------------------------------------------------------
# 4. Idempotency — running migration twice must not duplicate columns
# ---------------------------------------------------------------------------


class TestMigrationIdempotent:
    """Running v6.0 migration twice must not raise errors or duplicate columns."""

    def test_migration_idempotent(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        _create_legacy_daily_metrics_db(cache_dir / "gitflow_cache.db")

        # First run
        cache1 = GitAnalysisCache(cache_dir, ttl_hours=24)
        del cache1

        # Second run
        cache2 = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache2.db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(daily_metrics)"))
            columns = [row[1] for row in result]

        # No duplicates
        assert len(columns) == len(set(columns)), "Duplicate columns after double migration."

        # All AI columns present
        column_set = set(columns)
        for col in AI_COLUMNS:
            assert col in column_set, f"Column '{col}' missing after double migration."
