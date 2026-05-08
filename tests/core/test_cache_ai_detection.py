"""Tests for AI detection wiring in the cache write path + backfill (issue #47).

Covers:
    - Commits with AI signals get non-NULL ai_confidence_score at write time.
    - Commits without AI signals get 0.0 score and "" detection method.
    - Pre-populated AI fields are preserved (no double-detection).
    - backfill_ai_detection() updates only NULL rows and is idempotent.
"""

from datetime import datetime, timezone

from click.testing import CliRunner

from gitflow_analytics.cli import cli
from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.models.database import CachedCommit


def _base_commit(hash_: str, message: str) -> dict:
    """Return a minimally-populated commit dict (no AI fields set)."""
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


class TestCacheCommitAIDetection:
    """AI detection at commit write time (single-row write path)."""

    def test_ai_signal_commit_gets_non_null_score(self, temp_dir) -> None:
        """A commit with a Claude co-author trailer should yield a high score."""
        cache = GitAnalysisCache(temp_dir / ".cache")
        message = "feat: refactor cache layer\n\n" "Co-Authored-By: Claude <noreply@anthropic.com>"
        cache.cache_commit("/repo", _base_commit("aaa111", message))

        cached = cache.get_cached_commit("/repo", "aaa111")
        assert cached is not None
        # Strong signal -> >= 0.9 confidence, recognized as a known AI signal.
        assert cached["ai_confidence_score"] is not None
        assert cached["ai_confidence_score"] >= 0.9
        assert cached["ai_detection_method"] not in ("", "none")

    def test_plain_commit_gets_zero_score(self, temp_dir) -> None:
        """A vanilla commit message should be 0.0 confidence + '' method."""
        cache = GitAnalysisCache(temp_dir / ".cache")
        cache.cache_commit("/repo", _base_commit("bbb222", "chore: bump version to 1.2.3"))

        cached = cache.get_cached_commit("/repo", "bbb222")
        assert cached is not None
        # No signal -> 0.0 + empty/none method.
        assert cached["ai_confidence_score"] == 0.0
        assert cached["ai_detection_method"] in ("", "none")

    def test_pre_populated_fields_are_preserved(self, temp_dir) -> None:
        """If a caller already set the AI fields, the cache must not overwrite."""
        cache = GitAnalysisCache(temp_dir / ".cache")
        commit = _base_commit("ccc333", "chore: nothing AI here")
        # Pretend an upstream stage already scored this.
        commit["ai_confidence_score"] = 0.42
        commit["ai_detection_method"] = "nlp_heuristic"
        cache.cache_commit("/repo", commit)

        cached = cache.get_cached_commit("/repo", "ccc333")
        assert cached is not None
        assert cached["ai_confidence_score"] == 0.42
        assert cached["ai_detection_method"] == "nlp_heuristic"


class TestCacheCommitsBatchAIDetection:
    """AI detection at commit write time (batch write path)."""

    def test_batch_populates_ai_fields(self, temp_dir) -> None:
        """cache_commits_batch should score each commit individually."""
        cache = GitAnalysisCache(temp_dir / ".cache")
        commits = [
            _base_commit(
                "h1",
                "feat: thing\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
            ),
            _base_commit("h2", "chore: bump deps"),
        ]
        cache.cache_commits_batch("/repo", commits)

        c1 = cache.get_cached_commit("/repo", "h1")
        c2 = cache.get_cached_commit("/repo", "h2")
        assert c1 is not None and c2 is not None
        assert c1["ai_confidence_score"] is not None
        assert c1["ai_confidence_score"] >= 0.9
        assert c1["ai_detection_method"] not in ("", "none")
        assert c2["ai_confidence_score"] == 0.0


class TestBackfillAIDetection:
    """gfa backfill-ai-detection equivalent: cache.backfill_ai_detection()."""

    def _insert_commit_with_null_ai(
        self, cache: GitAnalysisCache, hash_: str, message: str
    ) -> None:
        """Insert a CachedCommit row directly with NULL ai_confidence_score.

        Bypasses cache_commit() so we can simulate legacy rows written before
        issue #47.
        """
        with cache.get_session() as session:
            session.add(
                CachedCommit(
                    repo_path="/repo",
                    commit_hash=hash_,
                    author_name="Dev",
                    author_email="dev@example.com",
                    message=message,
                    timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                    branch="main",
                    is_merge=False,
                    files_changed=1,
                    insertions=10,
                    deletions=0,
                    ai_confidence_score=None,
                    ai_detection_method="",
                )
            )

    def test_backfill_updates_null_rows(self, temp_dir) -> None:
        """Rows with NULL ai_confidence_score should be scored after backfill."""
        cache = GitAnalysisCache(temp_dir / ".cache")
        self._insert_commit_with_null_ai(
            cache,
            "ai1",
            "feat: rework cache\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
        )
        self._insert_commit_with_null_ai(cache, "plain1", "chore: bump version")

        result = cache.backfill_ai_detection()

        assert result["scanned"] == 2
        assert result["updated"] == 2

        ai_row = cache.get_cached_commit("/repo", "ai1")
        plain_row = cache.get_cached_commit("/repo", "plain1")
        assert ai_row is not None
        assert plain_row is not None
        assert ai_row["ai_confidence_score"] is not None
        assert ai_row["ai_confidence_score"] >= 0.9
        assert ai_row["ai_detection_method"] not in ("", "none")
        assert plain_row["ai_confidence_score"] == 0.0

    def test_backfill_is_idempotent(self, temp_dir) -> None:
        """A second backfill run should find zero NULL rows."""
        cache = GitAnalysisCache(temp_dir / ".cache")
        self._insert_commit_with_null_ai(cache, "x1", "chore: bump version")

        first = cache.backfill_ai_detection()
        second = cache.backfill_ai_detection()

        assert first["scanned"] == 1
        # All rows now populated; second pass scans nothing.
        assert second["scanned"] == 0
        assert second["updated"] == 0

    def test_backfill_cli_command_registered(self) -> None:
        """`gfa backfill-ai-detection --help` must succeed."""
        runner = CliRunner()
        result = runner.invoke(cli, ["backfill-ai-detection", "--help"])
        assert result.exit_code == 0
        assert "backfill" in result.output.lower()
        assert "--config" in result.output

    def test_backfill_skips_already_scored_rows(self, temp_dir) -> None:
        """Rows with non-NULL ai_confidence_score must not be touched."""
        cache = GitAnalysisCache(temp_dir / ".cache")
        # Write through the normal path so AI fields get populated.
        cache.cache_commit("/repo", _base_commit("scored1", "chore: bump"))
        # Insert one legacy NULL row.
        self._insert_commit_with_null_ai(cache, "legacy1", "chore: legacy")

        result = cache.backfill_ai_detection()

        # Only the legacy row should be scanned/updated.
        assert result["scanned"] == 1
        assert result["updated"] == 1
