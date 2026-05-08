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

    # Issue #54: false-positive suppression for known non-ticket prefixes
    # ----------------------------------------------------------------------

    def test_excludes_cve_identifiers(self) -> None:
        """CVE-2024-1234 must not be treated as a ticket reference."""
        assert _extract_ticket_ids_from_messages(["fix CVE-2024-1234 vulnerability"]) == []

    def test_excludes_cwe_identifiers(self) -> None:
        """CWE-89 (SQL injection weakness ID) is not a project ticket."""
        assert _extract_ticket_ids_from_messages(["address CWE-89 injection"]) == []

    def test_excludes_rfc_identifiers(self) -> None:
        """RFC-2616 is an HTTP spec, not a ticket."""
        assert _extract_ticket_ids_from_messages(["per RFC-2616 section 5"]) == []

    def test_excludes_short_prefix(self) -> None:
        """Single-letter prefixes (A-1) are excluded by the length>=2 rule."""
        assert _extract_ticket_ids_from_messages(["A-1 placeholder"]) == []

    def test_includes_real_tickets_alongside_false_positives(self) -> None:
        """Real tickets are kept; CVE/CWE noise in the same message is dropped."""
        msgs = ["PROJ-123 fixes CVE-2024-1234 and CWE-89"]
        assert _extract_ticket_ids_from_messages(msgs) == ["PROJ-123"]

    def test_includes_short_real_tickets(self) -> None:
        """2-letter prefixes like FD- and BI- are valid project keys."""
        msgs = ["FD-789 done; also BI-101"]
        assert _extract_ticket_ids_from_messages(msgs) == ["BI-101", "FD-789"]

    def test_excludes_other_known_prefixes(self) -> None:
        """All exclusion-list prefixes are filtered."""
        msgs = [
            "ISO-8601 date format",
            "HTTP-200 means OK",
            "API-1 spec",
            "SQL-92 standard",
            "URL-3986 syntax",
            "CSS-3 features",
            "HTML-5 element",
            "JSON-7159 format",
            "XML-1 spec",
            "PDF-1 reader",
            "AWS-2 region",
            "GCP-3 zone",
            "SDK-4 release",
        ]
        assert _extract_ticket_ids_from_messages(msgs) == []


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

    def test_payload_without_commit_messages_extracts_from_title_only(self, tmp_path: Path) -> None:
        """If commit_messages omitted, ticket_ids is still derived from the PR title.

        Issue #54: PR title is a first-class source of ticket refs.  When the
        title carries no ticket-shaped tokens we persist an empty JSON list
        (not NULL) so the row no longer requires backfill.
        """
        cache = GitAnalysisCache(tmp_path / ".gfa", ttl_hours=24)
        pr = _make_pr_data()  # title = "feat: shiny" — no tickets
        pr.pop("commit_messages")
        cache.cache_pr("owner/repo", pr)

        with cache.db.engine.connect() as conn:
            row = conn.execute(
                text("SELECT commit_count, ticket_ids FROM pull_request_cache WHERE pr_number = 42")
            ).fetchone()
        assert row is not None
        # commit_count is always derived (it doesn't need messages)
        assert row[0] == 3
        # Title has no tickets -> empty list, not NULL
        assert row[1] == "[]"

    def test_ticket_ids_extracted_from_pr_title(self, tmp_path: Path) -> None:
        """Issue #54: ticket IDs in PR titles are extracted at cache time."""
        cache = GitAnalysisCache(tmp_path / ".gfa", ttl_hours=24)
        pr = _make_pr_data(
            title="PROJ-101: implement new endpoint",
            commit_hashes=["a"],
            commit_messages=["chore: cleanup"],  # no ticket in commit msg
        )
        cache.cache_pr("owner/repo", pr)
        result = cache.get_cached_pr("owner/repo", 42)
        assert result is not None
        assert result["ticket_ids"] == ["PROJ-101"]

    def test_ticket_ids_merged_from_title_and_commit_messages(self, tmp_path: Path) -> None:
        """Issue #54: tickets in title and in commit messages are merged & deduplicated."""
        cache = GitAnalysisCache(tmp_path / ".gfa", ttl_hours=24)
        pr = _make_pr_data(
            title="PROJ-101: refactor",
            commit_hashes=["a", "b"],
            commit_messages=[
                "PROJ-101: extract helper",  # duplicates the title
                "fix: also addresses CORE-42",  # unique to commit msg
            ],
        )
        cache.cache_pr("owner/repo", pr)
        result = cache.get_cached_pr("owner/repo", 42)
        assert result is not None
        assert result["ticket_ids"] == ["CORE-42", "PROJ-101"]


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

    def test_backfill_extracts_ticket_from_pr_title(self, tmp_path: Path) -> None:
        """Issue #54: backfill scans PR title for ticket refs.

        Set up a PR whose commit messages contain no tickets but whose title
        does — the backfill must surface the title ticket.
        """
        cache = GitAnalysisCache(tmp_path / ".gfa", ttl_hours=24)

        # Cached commits with NO ticket refs in messages.
        with cache.db.SessionLocal() as session:
            session.add(
                CachedCommit(
                    repo_path="owner/repo",
                    commit_hash="hx",
                    author_name="dev",
                    author_email="dev@example.com",
                    message="chore: misc cleanup",
                    timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    files_changed=1,
                    insertions=1,
                    deletions=0,
                    complexity_delta=0.0,
                    is_merge=False,
                )
            )
            session.commit()

        # PR with ticket ref in TITLE, none in commit messages.
        pr = _make_pr_data(
            title="ADV-456: implement feature X",
            commit_hashes=["hx"],
        )
        pr.pop("commit_messages")
        cache.cache_pr("owner/repo", pr)

        # Force NULL state to exercise backfill.
        with cache.db.engine.connect() as conn:
            conn.execute(
                text("UPDATE pull_request_cache SET commit_count = NULL, ticket_ids = NULL")
            )
            conn.commit()

        stats = backfill_pr_cache(cache.db)
        assert stats["prs_examined"] == 1

        with cache.db.engine.connect() as conn:
            row = conn.execute(
                text("SELECT ticket_ids FROM pull_request_cache WHERE pr_number = 42")
            ).fetchone()
        assert row is not None
        assert json.loads(row[0]) == ["ADV-456"]

    def test_backfill_merges_title_and_commit_message_tickets(self, tmp_path: Path) -> None:
        """Issue #54: backfill deduplicates tickets across title + commit messages."""
        cache = GitAnalysisCache(tmp_path / ".gfa", ttl_hours=24)

        with cache.db.SessionLocal() as session:
            session.add(
                CachedCommit(
                    repo_path="owner/repo",
                    commit_hash="ha",
                    author_name="dev",
                    author_email="dev@example.com",
                    message="PROJ-100 work item",
                    timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    files_changed=1,
                    insertions=1,
                    deletions=0,
                    complexity_delta=0.0,
                    is_merge=False,
                )
            )
            session.commit()

        # Title shares one ticket with the commit message and adds another.
        pr = _make_pr_data(
            title="PROJ-100, FD-200: combined fix",
            commit_hashes=["ha"],
        )
        pr.pop("commit_messages")
        cache.cache_pr("owner/repo", pr)

        with cache.db.engine.connect() as conn:
            conn.execute(
                text("UPDATE pull_request_cache SET commit_count = NULL, ticket_ids = NULL")
            )
            conn.commit()

        backfill_pr_cache(cache.db)

        with cache.db.engine.connect() as conn:
            row = conn.execute(
                text("SELECT ticket_ids FROM pull_request_cache WHERE pr_number = 42")
            ).fetchone()
        assert row is not None
        # PROJ-100 appears once (deduplicated); FD-200 only in title.
        assert json.loads(row[0]) == ["FD-200", "PROJ-100"]

    def test_backfill_suppresses_cve_false_positives(self, tmp_path: Path) -> None:
        """Issue #54: CVE-XXXX in title or message is filtered out."""
        cache = GitAnalysisCache(tmp_path / ".gfa", ttl_hours=24)

        with cache.db.SessionLocal() as session:
            session.add(
                CachedCommit(
                    repo_path="owner/repo",
                    commit_hash="hc",
                    author_name="dev",
                    author_email="dev@example.com",
                    message="patch CWE-89 weakness",
                    timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    files_changed=1,
                    insertions=1,
                    deletions=0,
                    complexity_delta=0.0,
                    is_merge=False,
                )
            )
            session.commit()

        pr = _make_pr_data(
            title="security: fix CVE-2024-1234",
            commit_hashes=["hc"],
        )
        pr.pop("commit_messages")
        cache.cache_pr("owner/repo", pr)

        with cache.db.engine.connect() as conn:
            conn.execute(
                text("UPDATE pull_request_cache SET commit_count = NULL, ticket_ids = NULL")
            )
            conn.commit()

        backfill_pr_cache(cache.db)

        with cache.db.engine.connect() as conn:
            row = conn.execute(
                text("SELECT ticket_ids FROM pull_request_cache WHERE pr_number = 42")
            ).fetchone()
        assert row is not None
        assert json.loads(row[0]) == []  # no real tickets, CVE/CWE suppressed

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
