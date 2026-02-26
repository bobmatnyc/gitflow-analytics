"""Tests for PullRequestCache v3.0 schema extension.

Validates that:
1. New columns exist on the model with correct types and defaults.
2. The _apply_migrations / _migrate_pull_request_cache_v3 path adds columns to
   an existing database that was created without them.
3. cache_pr() stores all new fields correctly (both create and update paths).
4. _pr_to_dict() returns all new fields (including backward-compatible defaults
   for pre-v3.0 rows).
5. Existing functionality (commit caching, issue caching) is unaffected.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Column, DateTime, Integer, String, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.models.database import Base, PullRequestCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PR_V3_COLUMNS = [
    "review_comments_count",
    "pr_comments_count",
    "approvals_count",
    "change_requests_count",
    "reviewers",
    "approved_by",
    "time_to_first_review_hours",
    "revision_count",
    "changed_files",
    "additions",
    "deletions",
]


def _make_pr_data(**overrides: Any) -> dict[str, Any]:
    """Return a minimal PR data dict suitable for cache_pr()."""
    base: dict[str, Any] = {
        "number": 42,
        "title": "feat: shiny new thing",
        "description": "A description",
        "author": "alice",
        "created_at": datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        "merged_at": datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
        "story_points": 3,
        "labels": ["enhancement"],
        "commit_hashes": ["abc123", "def456"],
        # v3.0 enhanced fields
        "review_comments": 5,
        "pr_comments_count": 2,
        "approvals_count": 2,
        "change_requests_count": 1,
        "reviewers": ["bob", "carol"],
        "approved_by": ["bob", "carol"],
        "time_to_first_review_hours": 4.5,
        "revision_count": 3,
        "changed_files": 7,
        "additions": 120,
        "deletions": 30,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Model column presence tests
# ---------------------------------------------------------------------------


class TestPullRequestCacheModelColumns:
    """Verify that all v3.0 columns are declared on the SQLAlchemy model."""

    def test_v3_columns_exist_on_model(self) -> None:
        """All eleven new columns must be present as mapped attributes."""
        mapper = inspect(PullRequestCache)
        column_names = {col.key for col in mapper.columns}

        for col in PR_V3_COLUMNS:
            assert col in column_names, (
                f"Expected column '{col}' on PullRequestCache but it was not found. "
                "Did you forget to add it to models/database.py?"
            )

    def test_original_columns_still_present(self) -> None:
        """Existing columns must not be removed by the v3.0 extension."""
        required_original = [
            "id",
            "repo_path",
            "pr_number",
            "title",
            "description",
            "author",
            "created_at",
            "merged_at",
            "story_points",
            "labels",
            "commit_hashes",
            "cached_at",
        ]
        mapper = inspect(PullRequestCache)
        column_names = {col.key for col in mapper.columns}

        for col in required_original:
            assert col in column_names, f"Original column '{col}' is unexpectedly missing."

    def test_new_integer_columns_have_integer_type(self) -> None:
        """Integer columns must map to Integer type (not String, Float, etc.)."""
        from sqlalchemy import Integer as SAInteger

        mapper = inspect(PullRequestCache)
        integer_cols = [
            "review_comments_count",
            "pr_comments_count",
            "approvals_count",
            "change_requests_count",
            "revision_count",
            "changed_files",
            "additions",
            "deletions",
        ]
        for col_name in integer_cols:
            col = mapper.columns[col_name]
            assert isinstance(
                col.type, SAInteger
            ), f"Column '{col_name}' should be Integer but is {type(col.type).__name__}"

    def test_time_to_first_review_is_float(self) -> None:
        """time_to_first_review_hours must be Float to store fractional hours."""
        from sqlalchemy import Float as SAFloat

        mapper = inspect(PullRequestCache)
        col = mapper.columns["time_to_first_review_hours"]
        assert isinstance(
            col.type, SAFloat
        ), f"time_to_first_review_hours should be Float but is {type(col.type).__name__}"

    def test_json_list_columns_are_nullable(self) -> None:
        """reviewers and approved_by are optional (PR may not have been reviewed)."""
        mapper = inspect(PullRequestCache)
        for col_name in ("reviewers", "approved_by"):
            col = mapper.columns[col_name]
            assert col.nullable, f"Column '{col_name}' should be nullable."


# ---------------------------------------------------------------------------
# 2. In-memory database – fresh creation includes all columns
# ---------------------------------------------------------------------------


class TestFreshDatabaseSchema:
    """Verify that create_all() on a fresh database includes all v3.0 columns."""

    def test_fresh_db_has_v3_columns(self, tmp_path: Path) -> None:
        """A freshly created database must contain all v3.0 pull_request_cache columns."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(pull_request_cache)"))
            actual_columns = {row[1] for row in result}

        for col in PR_V3_COLUMNS:
            assert col in actual_columns, f"Fresh DB missing column '{col}' in pull_request_cache"


# ---------------------------------------------------------------------------
# 3. Migration test – simulate pre-v3.0 database
# ---------------------------------------------------------------------------


class TestMigrationAppliesV3Columns:
    """Verify _migrate_pull_request_cache_v3 adds missing columns to an old DB."""

    def _create_legacy_db(self, db_path: Path) -> None:
        """Create a pull_request_cache table that looks like the pre-v3.0 schema."""
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            conn.execute(
                text("""
                CREATE TABLE pull_request_cache (
                    id INTEGER PRIMARY KEY,
                    repo_path TEXT NOT NULL,
                    pr_number INTEGER NOT NULL,
                    title TEXT,
                    description TEXT,
                    author TEXT,
                    created_at DATETIME,
                    merged_at DATETIME,
                    story_points INTEGER,
                    labels TEXT,
                    commit_hashes TEXT,
                    cached_at DATETIME
                )
            """)
            )
            # Insert a legacy row to confirm existing data is preserved
            conn.execute(
                text("""
                INSERT INTO pull_request_cache
                    (repo_path, pr_number, title, author, cached_at)
                VALUES ('owner/repo', 1, 'Legacy PR', 'dev', datetime('now'))
            """)
            )
            conn.commit()
        engine.dispose()

    def test_migration_adds_all_v3_columns(self, tmp_path: Path) -> None:
        """Opening a legacy DB through GitAnalysisCache must add all v3.0 columns."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()

        # Create the legacy DB in the expected cache location
        self._create_legacy_db(cache_dir / "gitflow_cache.db")

        # Instantiate the cache – this triggers _apply_migrations
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        # Verify columns were added
        with cache.db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(pull_request_cache)"))
            actual_columns = {row[1] for row in result}

        for col in PR_V3_COLUMNS:
            assert (
                col in actual_columns
            ), f"Migration failed to add column '{col}' to legacy pull_request_cache"

    def test_migration_preserves_existing_rows(self, tmp_path: Path) -> None:
        """Existing PR rows must not be deleted during migration."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        db_path = cache_dir / "gitflow_cache.db"

        self._create_legacy_db(db_path)

        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache.db.engine.connect() as conn:
            result = conn.execute(
                text("SELECT pr_number, title, author FROM pull_request_cache WHERE pr_number = 1")
            )
            row = result.fetchone()

        assert row is not None, "Existing row was deleted during migration."
        assert row[1] == "Legacy PR"
        assert row[2] == "dev"

    def test_migration_idempotent(self, tmp_path: Path) -> None:
        """Running migration twice must not raise errors or duplicate columns."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        self._create_legacy_db(cache_dir / "gitflow_cache.db")

        # First open triggers migration
        cache1 = GitAnalysisCache(cache_dir, ttl_hours=24)
        del cache1

        # Second open should be a no-op (columns already exist)
        cache2 = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache2.db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(pull_request_cache)"))
            columns = [row[1] for row in result]

        # No duplicates
        assert len(columns) == len(set(columns)), "Duplicate columns after double migration."

        for col in PR_V3_COLUMNS:
            assert col in columns


# ---------------------------------------------------------------------------
# 4. cache_pr() create path
# ---------------------------------------------------------------------------


class TestCachePrCreate:
    """Verify cache_pr() persists all v3.0 fields when creating a new entry."""

    def test_creates_pr_with_all_v3_fields(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)
        repo = "owner/repo"
        pr_data = _make_pr_data()

        cache.cache_pr(repo, pr_data)

        result = cache.get_cached_pr(repo, pr_data["number"])

        assert result is not None
        assert result["review_comments"] == 5
        assert result["pr_comments_count"] == 2
        assert result["approvals_count"] == 2
        assert result["change_requests_count"] == 1
        assert result["reviewers"] == ["bob", "carol"]
        assert result["approved_by"] == ["bob", "carol"]
        assert result["time_to_first_review_hours"] == pytest.approx(4.5)
        assert result["revision_count"] == 3
        assert result["changed_files"] == 7
        assert result["additions"] == 120
        assert result["deletions"] == 30

    def test_creates_pr_with_zero_defaults_when_fields_absent(self, tmp_path: Path) -> None:
        """cache_pr() must not fail when v3.0 fields are missing from the payload."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)
        repo = "owner/repo"

        # Minimal payload without any v3.0 fields
        minimal_pr: dict[str, Any] = {
            "number": 10,
            "title": "Minimal PR",
            "author": "alice",
            "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "merged_at": datetime(2025, 1, 2, tzinfo=timezone.utc),
            "labels": [],
            "commit_hashes": [],
        }
        cache.cache_pr(repo, minimal_pr)

        result = cache.get_cached_pr(repo, 10)
        assert result is not None
        assert result["review_comments"] == 0
        assert result["approvals_count"] == 0
        assert result["reviewers"] == []
        assert result["time_to_first_review_hours"] is None
        assert result["changed_files"] == 0


# ---------------------------------------------------------------------------
# 5. cache_pr() update path
# ---------------------------------------------------------------------------


class TestCachePrUpdate:
    """Verify that updating an existing cached PR preserves and overwrites v3.0 fields."""

    def test_update_overwrites_v3_fields_when_present(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)
        repo = "owner/repo"

        # Initial write
        cache.cache_pr(repo, _make_pr_data())

        # Second write with updated values
        updated = _make_pr_data(
            review_comments=10,
            approvals_count=3,
            reviewers=["bob", "carol", "dave"],
            time_to_first_review_hours=1.25,
            revision_count=5,
        )
        cache.cache_pr(repo, updated)

        result = cache.get_cached_pr(repo, 42)
        assert result is not None
        assert result["review_comments"] == 10
        assert result["approvals_count"] == 3
        assert result["reviewers"] == ["bob", "carol", "dave"]
        assert result["time_to_first_review_hours"] == pytest.approx(1.25)
        assert result["revision_count"] == 5

    def test_update_does_not_zero_out_v3_fields_when_absent(self, tmp_path: Path) -> None:
        """If v3.0 keys are absent from the update payload they must not be zeroed."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)
        repo = "owner/repo"

        # Initial write with full v3.0 data
        cache.cache_pr(repo, _make_pr_data())

        # Update payload that deliberately omits the v3.0 fields
        update_without_v3: dict[str, Any] = {
            "number": 42,
            "title": "Updated title",
            "author": "alice",
            "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "merged_at": datetime(2025, 1, 2, tzinfo=timezone.utc),
            "labels": [],
            "commit_hashes": ["abc123"],
        }
        cache.cache_pr(repo, update_without_v3)

        result = cache.get_cached_pr(repo, 42)
        assert result is not None
        # v3.0 values from the first write must survive the partial update
        assert (
            result["review_comments"] == 5
        ), "review_comments should not be zeroed by a payload that omits it"
        assert result["approvals_count"] == 2
        assert result["reviewers"] == ["bob", "carol"]


# ---------------------------------------------------------------------------
# 6. _pr_to_dict() backward-compatibility
# ---------------------------------------------------------------------------


class TestPrToDictBackwardCompatibility:
    """Verify _pr_to_dict() returns safe defaults for pre-v3.0 rows."""

    def test_pr_to_dict_returns_zero_defaults_for_missing_v3_attributes(
        self, tmp_path: Path
    ) -> None:
        """A PullRequestCache object without v3.0 columns must still serialise cleanly."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        # Build a PullRequestCache instance with only original fields set,
        # simulating a row read from a database before migration ran.
        pr_obj = PullRequestCache(
            repo_path="owner/repo",
            pr_number=99,
            title="Old PR",
            author="dev",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            merged_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
            labels=[],
            commit_hashes=[],
        )
        # Simulate missing v3.0 attributes by deleting them from instance __dict__
        # (they won't be present on a row fetched before migration).
        for attr in [
            "review_comments_count",
            "pr_comments_count",
            "approvals_count",
            "change_requests_count",
            "reviewers",
            "approved_by",
            "time_to_first_review_hours",
            "revision_count",
            "changed_files",
            "additions",
            "deletions",
        ]:
            pr_obj.__dict__.pop(attr, None)

        result = cache._pr_to_dict(pr_obj)

        assert result["review_comments"] == 0
        assert result["pr_comments_count"] == 0
        assert result["approvals_count"] == 0
        assert result["change_requests_count"] == 0
        assert result["reviewers"] == []
        assert result["approved_by"] == []
        assert result["time_to_first_review_hours"] is None
        assert result["revision_count"] == 0
        assert result["changed_files"] == 0
        assert result["additions"] == 0
        assert result["deletions"] == 0


# ---------------------------------------------------------------------------
# 7. Regression – existing functionality unaffected
# ---------------------------------------------------------------------------


class TestExistingFunctionalityUnaffected:
    """Guard against regressions in commit and issue caching after schema changes."""

    def test_commit_caching_still_works(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        repo = "/path/to/repo"
        commit_data = {
            "hash": "deadbeef",
            "author_name": "Dev",
            "author_email": "dev@example.com",
            "message": "chore: bump version",
            "timestamp": datetime(2025, 1, 10, tzinfo=timezone.utc),
            "branch": "main",
            "project": "PROJ",
            "files_changed": 1,
            "insertions": 2,
            "deletions": 1,
            "complexity_delta": 0.0,
            "story_points": None,
            "ticket_references": [],
        }

        cache.cache_commit(repo, commit_data)
        cached = cache.get_cached_commit(repo, "deadbeef")

        assert cached is not None
        assert cached["hash"] == "deadbeef"
        assert cached["author_name"] == "Dev"

    def test_pr_number_and_title_still_stored(self, tmp_path: Path) -> None:
        """Original PR fields must remain intact after the v3.0 schema extension."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)
        repo = "owner/repo"

        cache.cache_pr(repo, _make_pr_data(number=7, title="My PR"))
        result = cache.get_cached_pr(repo, 7)

        assert result is not None
        assert result["number"] == 7
        assert result["title"] == "My PR"
        assert result["author"] == "alice"
        assert result["labels"] == ["enhancement"]
        assert result["commit_hashes"] == ["abc123", "def456"]
