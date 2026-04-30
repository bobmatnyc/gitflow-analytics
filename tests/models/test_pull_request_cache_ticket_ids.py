"""Tests for PullRequestCache v5.0 schema extension (commit_count, ticket_ids).

Issue #53.

Validates that:
1. New columns exist on the model with correct types and nullability.
2. Fresh database (Base.metadata.create_all) includes the new columns.
3. _migrate_pull_request_cache_v5 adds columns to a pre-v5 database without
   destroying existing rows, and is idempotent.
4. cache_pr() correctly derives commit_count from commit_hashes.
5. cache_pr() correctly extracts JIRA-style ticket IDs from commit_messages,
   deduplicates them, and stores them as JSON.
6. PRs without ticket-bearing messages produce ticket_ids = "[]".
7. The backfill command populates NULL rows from cached_commits.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Integer, String, Text, create_engine, inspect, text

from gitflow_analytics.cli_backfill_ticket_ids import backfill_pr_cache
from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.core.cache_commits import _extract_ticket_ids_from_messages
from gitflow_analytics.models.database import (
    Base,
    CachedCommit,
    Database,
    PullRequestCache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PR_V5_COLUMNS = ["commit_count", "ticket_ids"]


def _make_pr_data(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "number": 42,
        "title": "feat: shiny",
        "description": "",
        "author": "alice",
        "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "merged_at": datetime(2025, 1, 2, tzinfo=timezone.utc),
        "labels": [],
        "commit_hashes": ["abc123", "def456", "ghi789"],
        "commit_messages": [
            "DUE-1234: implement feature",
            "fix(auth): resolve issue per CORE-567",
            "chore: cleanup (still touches DUE-1234)",
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Model column presence
# ---------------------------------------------------------------------------


class TestPullRequestCacheModelColumnsV5:
    def test_v5_columns_exist_on_model(self) -> None:
        mapper = inspect(PullRequestCache)
        column_names = {col.key for col in mapper.columns}
        for col in PR_V5_COLUMNS:
            assert col in column_names, f"Missing column '{col}' on PullRequestCache."

    def test_commit_count_is_integer(self) -> None:
        mapper = inspect(PullRequestCache)
        col = mapper.columns["commit_count"]
        assert isinstance(col.type, Integer)
        assert col.nullable is True

    def test_ticket_ids_is_text(self) -> None:
        mapper = inspect(PullRequestCache)
        col = mapper.columns["ticket_ids"]
        # Text and String are both valid; Text is preferred for JSON-encoded blobs.
        assert isinstance(col.type, (Text, String))
        assert col.nullable is True


# ---------------------------------------------------------------------------
# 2. Fresh DB schema includes v5 columns
# ---------------------------------------------------------------------------


class TestFreshDatabaseSchemaV5:
    def test_fresh_db_has_v5_columns(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(pull_request_cache)"))
            actual_columns = {row[1] for row in result}
        for col in PR_V5_COLUMNS:
            assert col in actual_columns


# ---------------------------------------------------------------------------
# 3. Migration from pre-v5 schema is additive and idempotent
# ---------------------------------------------------------------------------


class TestMigrationAppliesV5Columns:
    def _create_pre_v5_db(self, db_path: Path) -> None:
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
            conn.execute(
                text("""
                INSERT INTO pull_request_cache
                  (repo_path, pr_number, title, author, cached_at, merged_at, commit_hashes)
                VALUES
                  ('owner/repo', 1, 'Legacy PR', 'dev', datetime('now'), datetime('now'),
                   '["aaa","bbb"]')
                """)
            )
            conn.commit()
        engine.dispose()

    def test_migration_adds_v5_columns(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        self._create_pre_v5_db(cache_dir / "gitflow_cache.db")

        cache = GitAnalysisCache(cache_dir, ttl_hours=24)
        with cache.db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(pull_request_cache)"))
            actual_columns = {row[1] for row in result}

        for col in PR_V5_COLUMNS:
            assert col in actual_columns

    def test_migration_preserves_existing_rows(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        self._create_pre_v5_db(cache_dir / "gitflow_cache.db")

        cache = GitAnalysisCache(cache_dir, ttl_hours=24)
        with cache.db.engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT pr_number, title, commit_count, ticket_ids "
                    "FROM pull_request_cache WHERE pr_number = 1"
                )
            )
            row = result.fetchone()
        assert row is not None
        assert row[1] == "Legacy PR"
        # New columns must be NULL on legacy rows (until backfill runs).
        assert row[2] is None
        assert row[3] is None

    def test_migration_idempotent(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        self._create_pre_v5_db(cache_dir / "gitflow_cache.db")

        cache1 = GitAnalysisCache(cache_dir, ttl_hours=24)
        del cache1
        cache2 = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache2.db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(pull_request_cache)"))
            columns = [row[1] for row in result]

        assert len(columns) == len(set(columns)), "Duplicate columns after double migration"
        for col in PR_V5_COLUMNS:
            assert col in columns


# ---------------------------------------------------------------------------
# 4. Pure helper: ticket extraction
# ---------------------------------------------------------------------------


class TestTicketExtractionHelper:
    def test_extracts_jira_style_tickets(self) -> None:
        msgs = ["DUE-1234: feature", "Fix per CORE-567 and ABC-1"]
        assert _extract_ticket_ids_from_messages(msgs) == ["ABC-1", "CORE-567", "DUE-1234"]

    def test_deduplicates(self) -> None:
        msgs = ["DUE-1234 first", "still DUE-1234 here", "and DUE-1234 again"]
        assert _extract_ticket_ids_from_messages(msgs) == ["DUE-1234"]

    def test_no_tickets_returns_empty(self) -> None:
        msgs = ["chore: misc cleanup", "fix typo"]
        assert _extract_ticket_ids_from_messages(msgs) == []

    def test_ignores_empty_and_none_messages(self) -> None:
        msgs = ["DUE-1234", "", None]  # type: ignore[list-item]
        assert _extract_ticket_ids_from_messages(msgs) == ["DUE-1234"]

    def test_empty_input(self) -> None:
        assert _extract_ticket_ids_from_messages([]) == []


# ---------------------------------------------------------------------------
# 5. cache_pr() populates commit_count and ticket_ids
# ---------------------------------------------------------------------------


class TestCachePrV5:
    def test_commit_count_derived_from_commit_hashes(self, tmp_path: Path) -> None:
        cache = GitAnalysisCache(tmp_path / ".gfa", ttl_hours=24)
        cache.cache_pr("owner/repo", _make_pr_data())
        result = cache.get_cached_pr("owner/repo", 42)
        assert result is not None
        assert result["commit_count"] == 3

    def test_commit_count_zero_when_no_hashes(self, tmp_path: Path) -> None:
        cache = GitAnalysisCache(tmp_path / ".gfa", ttl_hours=24)
        pr = _make_pr_data(commit_hashes=[], commit_messages=[])
        cache.cache_pr("owner/repo", pr)
        result = cache.get_cached_pr("owner/repo", 42)
        assert result is not None
        assert result["commit_count"] == 0

    def test_ticket_ids_extracted_from_messages(self, tmp_path: Path) -> None:
        cache = GitAnalysisCache(tmp_path / ".gfa", ttl_hours=24)
        cache.cache_pr("owner/repo", _make_pr_data())
        result = cache.get_cached_pr("owner/repo", 42)
        assert result is not None
        assert result["ticket_ids"] == ["CORE-567", "DUE-1234"]

    def test_ticket_ids_deduplicates(self, tmp_path: Path) -> None:
        """Same ticket appearing in multiple commits must appear once."""
        cache = GitAnalysisCache(tmp_path / ".gfa", ttl_hours=24)
        pr = _make_pr_data(
            commit_hashes=["a", "b"],
            commit_messages=["DUE-1234: do it", "DUE-1234 follow-up"],
        )
        cache.cache_pr("owner/repo", pr)
        result = cache.get_cached_pr("owner/repo", 42)
        assert result is not None
        assert result["ticket_ids"] == ["DUE-1234"]

    def test_no_tickets_returns_empty_list(self, tmp_path: Path) -> None:
        cache = GitAnalysisCache(tmp_path / ".gfa", ttl_hours=24)
        pr = _make_pr_data(
            commit_hashes=["a"],
            commit_messages=["chore: cleanup"],
        )
        cache.cache_pr("owner/repo", pr)
        result = cache.get_cached_pr("owner/repo", 42)
        assert result is not None
        assert result["ticket_ids"] == []
        # Ensure the underlying cell is "[]" JSON, not NULL.
        with cache.db.engine.connect() as conn:
            row = conn.execute(
                text("SELECT ticket_ids FROM pull_request_cache WHERE pr_number = 42")
            ).fetchone()
        assert row is not None
        assert row[0] == "[]"

    def test_payload_without_commit_messages_leaves_ticket_ids_null(self, tmp_path: Path) -> None:
        """If commit_messages omitted, ticket_ids stays NULL (waiting for backfill)."""
        cache = GitAnalysisCache(tmp_path / ".gfa", ttl_hours=24)
        pr = _make_pr_data()
        pr.pop("commit_messages")
        cache.cache_pr("owner/repo", pr)

        with cache.db.engine.connect() as conn:
            row = conn.execute(
                text("SELECT commit_count, ticket_ids FROM pull_request_cache WHERE pr_number = 42")
            ).fetchone()
        assert row is not None
        # commit_count is always derived (it doesn't need messages)
        assert row[0] == 3
        # ticket_ids is left untouched (NULL on first write)
        assert row[1] is None


# ---------------------------------------------------------------------------
# 6. Backfill command
# ---------------------------------------------------------------------------


class TestBackfillCommand:
    def _setup_db_with_legacy_pr(self, tmp_path: Path) -> tuple[Database, GitAnalysisCache]:
        """Create a cache with one PR cached without commit_messages, plus the
        underlying CachedCommit rows so the backfill has data to join against."""
        cache = GitAnalysisCache(tmp_path / ".gfa", ttl_hours=24)
        # Insert cached commits with messages
        with cache.db.SessionLocal() as session:
            for h, msg in [
                ("h1", "DUE-1234: do thing"),
                ("h2", "follow-up CORE-567"),
                ("h3", "chore: noise"),
            ]:
                session.add(
                    CachedCommit(
                        repo_path="owner/repo",
                        commit_hash=h,
                        author_name="dev",
                        author_email="dev@example.com",
                        message=msg,
                        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                        files_changed=1,
                        insertions=1,
                        deletions=0,
                        complexity_delta=0.0,
                        is_merge=False,
                    )
                )
            session.commit()

        # Cache a PR without supplying commit_messages, so ticket_ids stays NULL.
        pr = _make_pr_data(commit_hashes=["h1", "h2", "h3"])
        pr.pop("commit_messages")
        cache.cache_pr("owner/repo", pr)

        # Force the legacy state: NULL out commit_count and ticket_ids so we
        # exercise the backfill path on both columns.  (commit_count would
        # otherwise have been derived to 3.)
        with cache.db.engine.connect() as conn:
            conn.execute(
                text("UPDATE pull_request_cache SET commit_count = NULL, ticket_ids = NULL")
            )
            conn.commit()

        return cache.db, cache

    def test_backfill_populates_commit_count_and_ticket_ids(self, tmp_path: Path) -> None:
        db, cache = self._setup_db_with_legacy_pr(tmp_path)

        stats = backfill_pr_cache(db)
        assert stats["prs_examined"] == 1
        assert stats["commit_count_set"] == 1
        assert stats["ticket_ids_set"] == 1

        # Verify persisted values
        assert db.engine is not None
        with db.engine.connect() as conn:
            row = conn.execute(
                text("SELECT commit_count, ticket_ids FROM pull_request_cache WHERE pr_number = 42")
            ).fetchone()
        assert row is not None
        assert row[0] == 3
        assert json.loads(row[1]) == ["CORE-567", "DUE-1234"]

    def test_backfill_idempotent(self, tmp_path: Path) -> None:
        db, _ = self._setup_db_with_legacy_pr(tmp_path)

        first = backfill_pr_cache(db)
        second = backfill_pr_cache(db)

        assert first["prs_examined"] == 1
        # Second run finds nothing left to update.
        assert second["prs_examined"] == 0
        assert second["commit_count_set"] == 0
        assert second["ticket_ids_set"] == 0

    def test_backfill_handles_pr_with_no_matching_commits(self, tmp_path: Path) -> None:
        """A PR whose commit_hashes do not exist in cached_commits gets ticket_ids = []."""
        cache = GitAnalysisCache(tmp_path / ".gfa", ttl_hours=24)
        pr = _make_pr_data(commit_hashes=["unknown-hash"])
        pr.pop("commit_messages")
        cache.cache_pr("owner/repo", pr)

        with cache.db.engine.connect() as conn:
            conn.execute(
                text("UPDATE pull_request_cache SET commit_count = NULL, ticket_ids = NULL")
            )
            conn.commit()

        stats = backfill_pr_cache(cache.db)
        assert stats["prs_examined"] == 1

        with cache.db.engine.connect() as conn:
            row = conn.execute(
                text("SELECT commit_count, ticket_ids FROM pull_request_cache")
            ).fetchone()
        assert row is not None
        assert row[0] == 1  # one hash, even if unmatched
        assert json.loads(row[1]) == []
