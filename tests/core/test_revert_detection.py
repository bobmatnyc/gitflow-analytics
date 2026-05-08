"""Tests for revert detection wiring (issue #64).

Covers:
    - is_revert_commit() pattern matching for all documented variants.
    - cache_commit() / cache_commits_batch() persist is_revert at write time.
    - bulk_store_commits() persists is_revert.
    - DailyMetrics.reversion_commits is aggregated correctly from commits.
    - backfill_revert_flags() flips legacy FALSE rows whose messages match.
    - gfa backfill-revert-flags --help is registered on the CLI.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from click.testing import CliRunner

from gitflow_analytics.cli import cli
from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.core.metrics_storage import DailyMetricsStorage
from gitflow_analytics.models.database import CachedCommit, DailyMetrics
from gitflow_analytics.utils.revert_detection import is_revert_commit


def _base_commit(hash_: str, message: str) -> dict:
    """Return a minimally-populated commit dict."""
    return {
        "hash": hash_,
        "author_name": "Dev",
        "author_email": "dev@example.com",
        "message": message,
        "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "branch": "main",
        "files_changed": 1,
        "insertions": 10,
        "deletions": 0,
    }


class TestIsRevertCommitPatterns:
    """Pure regex coverage — independent of database / cache plumbing."""

    @pytest.mark.parametrize(
        "message",
        [
            'Revert "feat: add new endpoint"',
            'revert "feat: add new endpoint"',  # case-insensitive
            "revert: rollback bad migration",
            "Revert: rollback bad migration",
            "revert bad commit 1234567",
            "reverts commit 1234567",
            "feat: something\n\nThis reverts commit abc123def456.",
            "FEAT: ROLLBACK\n\nTHIS REVERTS COMMIT ABC.",  # case-insensitive body
            '  Revert "oops"',  # leading whitespace tolerated
        ],
    )
    def test_matches(self, message: str) -> None:
        assert is_revert_commit(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "feat: add new endpoint",
            "fix: handle reverted state in UI",  # 'reverted' as substring, no anchor
            "chore: bump version",
            "refactor: extract helper",
            "",
            "   ",
        ],
    )
    def test_non_matches(self, message: str) -> None:
        assert is_revert_commit(message) is False

    def test_handles_none_and_bytes(self) -> None:
        assert is_revert_commit(None) is False
        assert is_revert_commit(b'Revert "feat: x"') is True
        assert is_revert_commit(b"feat: x") is False


class TestCacheCommitPersistsIsRevert:
    """is_revert must be persisted at ingestion through the single-row path."""

    def test_revert_commit_flagged(self, temp_dir) -> None:
        cache = GitAnalysisCache(temp_dir / ".cache")
        cache.cache_commit("/repo", _base_commit("rev1", 'Revert "feat: bad change"'))
        cached = cache.get_cached_commit("/repo", "rev1")
        assert cached is not None
        assert cached["is_revert"] is True

    def test_non_revert_commit_not_flagged(self, temp_dir) -> None:
        cache = GitAnalysisCache(temp_dir / ".cache")
        cache.cache_commit("/repo", _base_commit("nrev1", "feat: regular change"))
        cached = cache.get_cached_commit("/repo", "nrev1")
        assert cached is not None
        assert cached["is_revert"] is False

    def test_explicit_is_revert_preserved(self, temp_dir) -> None:
        """If caller explicitly sets is_revert, helper must not overwrite."""
        cache = GitAnalysisCache(temp_dir / ".cache")
        commit = _base_commit("override1", "feat: regular change")
        commit["is_revert"] = True
        cache.cache_commit("/repo", commit)
        cached = cache.get_cached_commit("/repo", "override1")
        assert cached is not None
        assert cached["is_revert"] is True


class TestCacheCommitsBatchPersistsIsRevert:
    """is_revert must be persisted through the batch write path."""

    def test_batch_flags_each_row(self, temp_dir) -> None:
        cache = GitAnalysisCache(temp_dir / ".cache")
        commits = [
            _base_commit("b1", 'Revert "feat: x"'),
            _base_commit("b2", "feat: regular"),
            _base_commit("b3", "revert: rollback bad migration"),
        ]
        cache.cache_commits_batch("/repo", commits)
        c1 = cache.get_cached_commit("/repo", "b1")
        c2 = cache.get_cached_commit("/repo", "b2")
        c3 = cache.get_cached_commit("/repo", "b3")
        assert c1 is not None and c2 is not None and c3 is not None
        assert c1["is_revert"] is True
        assert c2["is_revert"] is False
        assert c3["is_revert"] is True


class TestBulkStoreCommitsPersistsIsRevert:
    """is_revert must be persisted through the bulk_insert_mappings path."""

    def test_bulk_store_flags_revert(self, temp_dir) -> None:
        cache = GitAnalysisCache(temp_dir / ".cache")
        commits = [
            _base_commit("bulk_rev", 'Revert "x"'),
            _base_commit("bulk_plain", "feat: x"),
        ]
        result = cache.bulk_store_commits("/repo", commits)
        assert result["inserted"] == 2

        rev = cache.get_cached_commit("/repo", "bulk_rev")
        plain = cache.get_cached_commit("/repo", "bulk_plain")
        assert rev is not None and plain is not None
        assert rev["is_revert"] is True
        assert plain["is_revert"] is False


class TestDailyMetricsReversionCommits:
    """daily_metrics.reversion_commits is aggregated from commit input."""

    def test_aggregates_from_is_revert_flag(self, temp_dir) -> None:
        storage = DailyMetricsStorage(temp_dir / "metrics.db")
        target = date(2024, 1, 15)
        commits = [
            {
                "hash": "h1",
                "author_email": "dev@example.com",
                "author_name": "Dev",
                "message": 'Revert "feat: x"',
                "timestamp": datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
                "files_changed": 1,
                "insertions": 1,
                "deletions": 1,
                "category": "other",
                "ticket_references": [],
                "is_revert": True,
                "project_key": "PROJ",
            },
            {
                "hash": "h2",
                "author_email": "dev@example.com",
                "author_name": "Dev",
                "message": "feat: regular",
                "timestamp": datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
                "files_changed": 1,
                "insertions": 5,
                "deletions": 0,
                "category": "feature",
                "ticket_references": [],
                "is_revert": False,
                "project_key": "PROJ",
            },
        ]
        identities = {
            "dev@example.com": {
                "canonical_id": "dev@example.com",
                "name": "Dev",
                "email": "dev@example.com",
            }
        }
        storage.store_daily_metrics(target, commits, identities)

        with storage.get_session() as session:
            row = session.query(DailyMetrics).filter(DailyMetrics.project_key == "PROJ").one()
            assert int(row.reversion_commits) == 1  # type: ignore[arg-type]
            assert int(row.total_commits) == 2  # type: ignore[arg-type]

    def test_falls_back_to_message_regex_when_flag_missing(self, temp_dir) -> None:
        """When is_revert key is absent, the aggregator re-derives from message."""
        storage = DailyMetricsStorage(temp_dir / "metrics2.db")
        target = date(2024, 2, 1)
        commits = [
            {
                "hash": "fallback1",
                "author_email": "dev@example.com",
                "author_name": "Dev",
                "message": 'Revert "feat: legacy"',
                "timestamp": datetime(2024, 2, 1, 10, 0, tzinfo=timezone.utc),
                "files_changed": 1,
                "insertions": 1,
                "deletions": 1,
                "category": "other",
                "ticket_references": [],
                # NO is_revert key — exercise the fallback path.
                "project_key": "PROJ",
            },
        ]
        identities = {
            "dev@example.com": {
                "canonical_id": "dev@example.com",
                "name": "Dev",
                "email": "dev@example.com",
            }
        }
        storage.store_daily_metrics(target, commits, identities)
        with storage.get_session() as session:
            row = session.query(DailyMetrics).one()
            assert int(row.reversion_commits) == 1  # type: ignore[arg-type]


class TestBackfillRevertFlags:
    """gfa backfill-revert-flags equivalent: cache.backfill_revert_flags()."""

    def _insert_legacy_commit(
        self,
        cache: GitAnalysisCache,
        hash_: str,
        message: str,
    ) -> None:
        """Insert a CachedCommit row directly with is_revert=False, bypassing
        cache_commit() — simulates rows written before issue #64.
        """
        with cache.get_session() as session:
            session.add(
                CachedCommit(
                    repo_path="/repo",
                    commit_hash=hash_,
                    author_name="Dev",
                    author_email="dev@example.com",
                    message=message,
                    timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
                    branch="main",
                    is_merge=False,
                    is_revert=False,
                    files_changed=1,
                    insertions=10,
                    deletions=0,
                )
            )

    def test_backfill_flips_revert_messages(self, temp_dir) -> None:
        cache = GitAnalysisCache(temp_dir / ".cache")
        self._insert_legacy_commit(cache, "legacy_rev", 'Revert "feat: bad"')
        self._insert_legacy_commit(cache, "legacy_plain", "feat: regular")

        result = cache.backfill_revert_flags()

        assert result["scanned"] == 2
        assert result["updated"] == 1
        assert result["revert_count"] == 1

        rev = cache.get_cached_commit("/repo", "legacy_rev")
        plain = cache.get_cached_commit("/repo", "legacy_plain")
        assert rev is not None and plain is not None
        assert rev["is_revert"] is True
        assert plain["is_revert"] is False

    def test_backfill_idempotent(self, temp_dir) -> None:
        cache = GitAnalysisCache(temp_dir / ".cache")
        self._insert_legacy_commit(cache, "legacy_rev2", 'Revert "feat: x"')
        first = cache.backfill_revert_flags()
        second = cache.backfill_revert_flags()
        assert first["updated"] == 1
        # After first run the row is True, so second run scans only
        # remaining FALSE rows (none here).
        assert second["scanned"] == 0
        assert second["updated"] == 0

    def test_backfill_cli_command_registered(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["backfill-revert-flags", "--help"])
        assert result.exit_code == 0
        assert "revert" in result.output.lower()
        assert "--config" in result.output
