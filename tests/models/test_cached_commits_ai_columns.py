"""Tests for CachedCommit AI detection columns (v9.0 migration).

Validates:
1. New AI columns exist on the CachedCommit model with correct types.
2. A fresh database (create_all) includes both AI columns.
3. _migrate_cached_commits_ai_columns() adds columns to a legacy database.
4. The migration is idempotent (safe to run twice).
5. Existing rows are preserved during migration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Float, String, create_engine, inspect, text

from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.models.database import Base
from gitflow_analytics.models.database_commit_models import CachedCommit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AI_COLUMNS = [
    "ai_confidence_score",
    "ai_detection_method",
]


# ---------------------------------------------------------------------------
# 1. Model column presence tests
# ---------------------------------------------------------------------------


class TestCachedCommitAiColumnsOnModel:
    """Verify that all AI columns are declared on the SQLAlchemy model."""

    def test_all_ai_columns_exist_on_model(self) -> None:
        mapper = inspect(CachedCommit)
        column_names = {col.key for col in mapper.columns}
        for col in AI_COLUMNS:
            assert col in column_names, (
                f"Expected column '{col}' on CachedCommit but it was not found. "
                "Check models/database_commit_models.py."
            )

    def test_ai_confidence_score_is_float_type(self) -> None:
        mapper = inspect(CachedCommit)
        col = mapper.columns["ai_confidence_score"]
        assert isinstance(
            col.type, Float
        ), f"ai_confidence_score should be Float but is {type(col.type).__name__}"

    def test_ai_detection_method_is_string_type(self) -> None:
        mapper = inspect(CachedCommit)
        col = mapper.columns["ai_detection_method"]
        assert isinstance(
            col.type, String
        ), f"ai_detection_method should be String but is {type(col.type).__name__}"

    def test_existing_columns_still_present(self) -> None:
        """AI additions must not remove any pre-existing CachedCommit columns."""
        pre_existing = [
            "id",
            "repo_path",
            "commit_hash",
            "author_name",
            "author_email",
            "message",
            "timestamp",
            "is_merge",
            "files_changed",
            "insertions",
            "deletions",
            "story_points",
            "ticket_references",
        ]
        mapper = inspect(CachedCommit)
        column_names = {col.key for col in mapper.columns}
        for col in pre_existing:
            assert col in column_names, f"Pre-existing column '{col}' is unexpectedly missing."


# ---------------------------------------------------------------------------
# 2. Fresh database includes all AI columns
# ---------------------------------------------------------------------------


class TestFreshDatabaseIncludesAiColumns:
    """Verify that create_all() on a fresh DB produces the AI columns."""

    def test_fresh_db_has_ai_columns(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(cached_commits)"))
            actual_columns = {row[1] for row in result}

        for col in AI_COLUMNS:
            assert col in actual_columns, f"Fresh DB missing column '{col}' in cached_commits"


# ---------------------------------------------------------------------------
# 3 & 5. Migration adds columns to legacy DB without destroying existing rows
# ---------------------------------------------------------------------------


def _create_legacy_cached_commits_db(db_path: Path) -> None:
    """Create a cached_commits table that looks like the pre-v9.0 schema (no AI columns).

    WHY: We must also create the schema_version table at the current version so
    that GitAnalysisCache does not trigger the v1.0 → v2.0 timezone migration,
    which would DELETE all rows before our v9.0 column additions can be tested.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        # Create schema_version so the DB is not treated as legacy v1.0
        conn.execute(
            text("""
            CREATE TABLE schema_version (
                id INTEGER PRIMARY KEY,
                version VARCHAR NOT NULL,
                upgraded_at DATETIME,
                previous_version VARCHAR,
                migration_notes VARCHAR
            )
        """)
        )
        conn.execute(
            text("""
            INSERT INTO schema_version (version, migration_notes)
            VALUES ('2.0', 'test legacy db at v2.0 without AI columns')
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE cached_commits (
                id INTEGER PRIMARY KEY,
                repo_path VARCHAR NOT NULL,
                commit_hash VARCHAR NOT NULL,
                author_name VARCHAR,
                author_email VARCHAR,
                message VARCHAR,
                timestamp DATETIME,
                branch VARCHAR,
                is_merge BOOLEAN DEFAULT 0,
                files_changed INTEGER,
                insertions INTEGER,
                deletions INTEGER,
                filtered_insertions INTEGER DEFAULT 0,
                filtered_deletions INTEGER DEFAULT 0,
                complexity_delta REAL,
                story_points INTEGER,
                ticket_references JSON,
                cached_at DATETIME,
                cache_version VARCHAR DEFAULT '1.0'
            )
        """)
        )
        # Insert a legacy row to confirm existing data is preserved
        conn.execute(
            text("""
            INSERT INTO cached_commits
                (repo_path, commit_hash, author_name, author_email, message, insertions)
            VALUES ('/repo', 'abc123', 'Alice', 'alice@example.com', 'fix bug', 10)
        """)
        )
        conn.commit()
    engine.dispose()


class TestMigrationAddsAiColumns:
    """Verify _migrate_cached_commits_ai_columns adds the missing columns."""

    def test_migration_adds_all_ai_columns(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        _create_legacy_cached_commits_db(cache_dir / "gitflow_cache.db")

        # Instantiate the cache — this triggers _apply_migrations
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache.db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(cached_commits)"))
            actual_columns = {row[1] for row in result}

        for col in AI_COLUMNS:
            assert (
                col in actual_columns
            ), f"v9.0 migration failed to add column '{col}' to legacy cached_commits"

    def test_migration_preserves_existing_rows(self, tmp_path: Path) -> None:
        """Existing cached_commits rows must not be deleted during v9.0 migration."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        _create_legacy_cached_commits_db(cache_dir / "gitflow_cache.db")

        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache.db.engine.connect() as conn:
            result = conn.execute(text("SELECT commit_hash, insertions FROM cached_commits"))
            row = result.fetchone()

        assert row is not None, "Existing row was deleted during v9.0 migration."
        assert row[0] == "abc123"
        assert row[1] == 10


# ---------------------------------------------------------------------------
# 4. Idempotency — running migration twice must not duplicate columns
# ---------------------------------------------------------------------------


class TestMigrationIdempotent:
    """Running v9.0 migration twice must not raise errors or duplicate columns."""

    def test_migration_idempotent(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        _create_legacy_cached_commits_db(cache_dir / "gitflow_cache.db")

        # First run
        cache1 = GitAnalysisCache(cache_dir, ttl_hours=24)
        del cache1

        # Second run
        cache2 = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache2.db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(cached_commits)"))
            columns = [row[1] for row in result]

        # No duplicates
        assert len(columns) == len(set(columns)), "Duplicate columns after double migration."

        # All AI columns present
        column_set = set(columns)
        for col in AI_COLUMNS:
            assert col in column_set, f"Column '{col}' missing after double migration."
