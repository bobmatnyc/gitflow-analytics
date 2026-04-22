"""Tests for metrics_storage constraint-violation log levels.

WHY: During re-classification runs the same commit may be inserted more than
once, producing a UNIQUE constraint conflict.  That is an *expected* condition
and must not pollute logs at ERROR level.  These tests verify:

1. A UNIQUE constraint violation causes logger.debug (not logger.error).
2. An unexpected non-UNIQUE exception still causes logger.warning (not DEBUG,
   not ERROR) so it remains visible without being noisy on re-runs.
3. The fallback lookup path (record found after rollback) causes logger.info.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from gitflow_analytics.core.metrics_storage import DailyMetricsStorage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_storage(tmp_path: Path) -> DailyMetricsStorage:
    """Return a DailyMetricsStorage wired to a temporary SQLite database."""
    db_path = tmp_path / "test_metrics.db"
    return DailyMetricsStorage(db_path)


def _minimal_commit(analysis_date: date) -> dict:
    """Return a commit dict that aggregates cleanly into one dev/project bucket."""
    return {
        "timestamp": datetime(analysis_date.year, analysis_date.month, analysis_date.day, 12, 0, 0),
        "author_email": "dev@example.com",
        "author_name": "Dev User",
        "project_key": "test-project",
        "category": "feature",
        "files_changed": 2,
        "insertions": 10,
        "deletions": 3,
        "filtered_insertions": 10,
        "filtered_deletions": 3,
        "story_points": 1,
        "ticket_references": [],
        "is_merge": False,
        "message": "feat: add something",
    }


# ---------------------------------------------------------------------------
# Test: UNIQUE constraint violation → DEBUG, not ERROR
# ---------------------------------------------------------------------------


class TestUniqueConstraintLoggingLevel:
    """Duplicate insert should log at DEBUG, never ERROR."""

    def test_unique_violation_logs_debug_not_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Simulates a UNIQUE constraint failure during insert.

        The outer except block must call logger.debug and must NOT call
        logger.error for 'UNIQUE constraint failed' exceptions.
        """
        analysis_date = date(2025, 1, 15)
        storage = _make_storage(tmp_path)
        commits = [_minimal_commit(analysis_date)]
        identities: dict = {}

        unique_exc = Exception(
            "UNIQUE constraint failed: daily_metrics.date, daily_metrics.developer_id, daily_metrics.project_key"
        )

        # Patch the session so that session.commit() raises a UNIQUE conflict
        # on the first call (simulating a concurrent / re-classification insert),
        # then succeeds on the retry path.
        original_get_session = storage.db.get_session

        call_count = 0

        def patched_get_session():
            nonlocal call_count
            call_count += 1
            session = original_get_session()
            original_commit = session.commit

            commit_calls = 0

            def commit_side_effect():
                nonlocal commit_calls
                commit_calls += 1
                if commit_calls == 1:
                    raise unique_exc
                return original_commit()

            session.commit = commit_side_effect
            return session

        with (
            caplog.at_level(logging.DEBUG, logger="gitflow_analytics.core.metrics_storage"),
            patch.object(storage.db, "get_session", side_effect=patched_get_session),
        ):
            storage.store_daily_metrics(analysis_date, commits, identities)

        debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]

        # Must have at least one DEBUG message mentioning the constraint
        assert any(
            "UNIQUE" in m or "constraint" in m.lower() for m in debug_messages
        ), f"Expected a DEBUG message about the UNIQUE constraint, got debug={debug_messages}"
        # Must have NO ERROR messages
        assert (
            error_messages == []
        ), f"Expected no ERROR messages for a UNIQUE violation, got: {error_messages}"

    def test_unique_violation_does_not_call_logger_error(self, tmp_path: Path) -> None:
        """Direct mock test: logger.error must NOT be called for UNIQUE violations."""
        analysis_date = date(2025, 2, 20)
        storage = _make_storage(tmp_path)
        commits = [_minimal_commit(analysis_date)]
        identities: dict = {}

        unique_exc = Exception("UNIQUE constraint failed: daily_metrics.date")

        original_get_session = storage.db.get_session

        def patched_get_session():
            session = original_get_session()
            original_commit = session.commit
            commit_calls: list[int] = [0]

            def commit_side_effect():
                commit_calls[0] += 1
                if commit_calls[0] == 1:
                    raise unique_exc
                return original_commit()

            session.commit = commit_side_effect
            return session

        with (
            patch.object(storage.db, "get_session", side_effect=patched_get_session),
            patch("gitflow_analytics.core.metrics_storage.logger") as mock_logger,
        ):
            storage.store_daily_metrics(analysis_date, commits, identities)

        # logger.error must never have been called
        mock_logger.error.assert_not_called()
        # logger.debug must have been called at least once
        assert (
            mock_logger.debug.call_count >= 1
        ), "Expected logger.debug to be called for UNIQUE violation"


# ---------------------------------------------------------------------------
# Test: Unexpected (non-UNIQUE) exception → WARNING, not DEBUG, not ERROR
# ---------------------------------------------------------------------------


class TestNonUniqueConstraintLoggingLevel:
    """Unexpected DB errors should log at WARNING — visible but not ERROR."""

    def test_non_unique_exception_logs_warning_not_error(self, tmp_path: Path) -> None:
        """A generic DB error (not UNIQUE) must produce logger.warning, not logger.error."""
        analysis_date = date(2025, 3, 10)
        storage = _make_storage(tmp_path)
        commits = [_minimal_commit(analysis_date)]
        identities: dict = {}

        # A database error that is NOT a UNIQUE constraint failure
        non_unique_exc = Exception("database disk image is malformed")

        original_get_session = storage.db.get_session

        def patched_get_session():
            session = original_get_session()
            original_commit = session.commit
            commit_calls: list[int] = [0]

            def commit_side_effect():
                commit_calls[0] += 1
                if commit_calls[0] == 1:
                    raise non_unique_exc
                return original_commit()

            session.commit = commit_side_effect
            return session

        with (
            patch.object(storage.db, "get_session", side_effect=patched_get_session),
            patch("gitflow_analytics.core.metrics_storage.logger") as mock_logger,
        ):
            storage.store_daily_metrics(analysis_date, commits, identities)

        # logger.error must never be called
        mock_logger.error.assert_not_called()
        # logger.warning must have been called (for the non-UNIQUE failure)
        assert (
            mock_logger.warning.call_count >= 1
        ), "Expected logger.warning for a non-UNIQUE DB error"


# ---------------------------------------------------------------------------
# Test: resolve-constraint path (record found after retry) → INFO
# ---------------------------------------------------------------------------


class TestConstraintResolutionLogging:
    """When the retry lookup finds the record, it should log at INFO."""

    def test_resolved_constraint_logs_info(self, tmp_path: Path) -> None:
        """Record found after retry should produce logger.info, not logger.error."""
        analysis_date = date(2025, 4, 5)
        storage = _make_storage(tmp_path)
        commits = [_minimal_commit(analysis_date)]
        identities: dict = {}

        # First store succeeds so there IS a real record in the DB
        storage.store_daily_metrics(analysis_date, commits, identities)

        # Second store of the same data: the insert will conflict but the retry
        # query will find the existing record and update it → should log INFO.
        unique_exc = Exception("UNIQUE constraint failed: daily_metrics.date")

        original_get_session = storage.db.get_session

        def patched_get_session():
            session = original_get_session()
            original_commit = session.commit
            commit_calls: list[int] = [0]

            def commit_side_effect():
                commit_calls[0] += 1
                if commit_calls[0] == 1:
                    raise unique_exc
                return original_commit()

            session.commit = commit_side_effect
            return session

        with (
            patch.object(storage.db, "get_session", side_effect=patched_get_session),
            patch("gitflow_analytics.core.metrics_storage.logger") as mock_logger,
        ):
            storage.store_daily_metrics(analysis_date, commits, identities)

        mock_logger.error.assert_not_called()
        # The "Updated metrics after constraint violation" path logs at INFO
        info_calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any(
            "constraint violation" in c.lower() or "updated" in c.lower() for c in info_calls
        ), f"Expected logger.info for resolved constraint. Got info calls: {info_calls}"
