"""Tests for the --backfill-since flag wiring (issue #52).

Coverage targets:
- ``GitHubIntegration._get_incremental_fetch_date`` honours ``force_since``
  and bypasses both schema and last-processed checkpoints.
- ``gfa fetch --backfill-since`` CLI argument validation: valid + invalid
  date strings.
- ``GitHubIntegration.enrich_repository_with_prs`` forwards ``backfill_since``
  to the gate so the actual fetch date matches the requested override.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

from click.testing import CliRunner

from gitflow_analytics.integrations.github_integration import GitHubIntegration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_integration() -> GitHubIntegration:
    """Instantiate GitHubIntegration with all external deps mocked out."""
    with (
        patch("gitflow_analytics.integrations.github_integration.Github"),
        patch("gitflow_analytics.integrations.github_integration.create_schema_manager"),
    ):
        cache = Mock()
        cache.cache_dir = "/tmp/test-cache"
        integration = GitHubIntegration(token="fake-token", cache=cache)
    return integration


# ---------------------------------------------------------------------------
# _get_incremental_fetch_date(force_since=...)
# ---------------------------------------------------------------------------


class TestForceSinceBypassesIncrementalGate:
    """When ``force_since`` is supplied, the gate must return that date verbatim."""

    def test_force_since_overrides_more_recent_last_processed(self) -> None:
        integration = _make_integration()

        # Schema unchanged + a much-more-recent last_processed_date.  Without
        # force_since, this would normally cause `max(last, requested)` to
        # advance fetch_since to last_processed.
        integration.schema_manager.has_schema_changed = MagicMock(return_value=False)
        recent = datetime(2026, 3, 23, tzinfo=timezone.utc)
        integration.schema_manager.get_last_processed_date = MagicMock(return_value=recent)

        force = datetime(2025, 1, 1, tzinfo=timezone.utc)
        requested = datetime(2026, 1, 1, tzinfo=timezone.utc)

        result = integration._get_incremental_fetch_date("github", requested, {}, force_since=force)

        assert result == force

    def test_force_since_naive_datetime_is_made_tz_aware(self) -> None:
        integration = _make_integration()
        integration.schema_manager.has_schema_changed = MagicMock(return_value=False)
        integration.schema_manager.get_last_processed_date = MagicMock(return_value=None)

        naive = datetime(2025, 1, 1)
        result = integration._get_incremental_fetch_date(
            "github", datetime(2026, 1, 1, tzinfo=timezone.utc), {}, force_since=naive
        )

        assert result.tzinfo is not None
        assert result == datetime(2025, 1, 1, tzinfo=timezone.utc)

    def test_requested_since_is_always_honored_even_when_older_than_last_processed(
        self,
    ) -> None:
        """``--weeks N`` (any N) must bypass the incremental gate implicitly.

        Previously ``_get_incremental_fetch_date`` returned
        ``max(last_processed, requested_since)``, silently advancing the fetch
        date when the user requested a wider window.  We now always honor the
        caller's ``requested_since`` — ``cache_pr()`` upsert keeps re-fetches
        safe.
        """
        integration = _make_integration()
        integration.schema_manager.has_schema_changed = MagicMock(return_value=False)
        recent = datetime(2026, 3, 23, tzinfo=timezone.utc)
        integration.schema_manager.get_last_processed_date = MagicMock(return_value=recent)

        # Simulate `--weeks 24`: requested_since is far older than last_processed.
        requested = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = integration._get_incremental_fetch_date("github", requested, {})

        # Must honor the requested date verbatim, NOT advance to last_processed.
        assert result == requested

    def test_requested_since_more_recent_than_last_processed_is_honored(self) -> None:
        """When the user asks for a smaller window than the checkpoint, honor it.

        Symmetric guarantee: the caller is the source of truth.  We don't second-
        guess them by walking the date backwards either.
        """
        integration = _make_integration()
        integration.schema_manager.has_schema_changed = MagicMock(return_value=False)
        old = datetime(2025, 1, 1, tzinfo=timezone.utc)
        integration.schema_manager.get_last_processed_date = MagicMock(return_value=old)

        requested = datetime(2026, 3, 1, tzinfo=timezone.utc)
        result = integration._get_incremental_fetch_date("github", requested, {})

        assert result == requested


# ---------------------------------------------------------------------------
# enrich_repository_with_prs forwards backfill_since
# ---------------------------------------------------------------------------


class TestEnrichRepositoryForwardsBackfillSince:
    def test_enrich_uses_backfill_since_for_api_call(self) -> None:
        """The since used for cache + API calls must match backfill_since."""
        integration = _make_integration()
        integration.schema_manager.has_schema_changed = MagicMock(return_value=False)
        # Last-processed is recent — without backfill we'd skip historical PRs.
        integration.schema_manager.get_last_processed_date = MagicMock(
            return_value=datetime(2026, 3, 23, tzinfo=timezone.utc)
        )
        integration.schema_manager.mark_date_processed = MagicMock()

        # Stub the GitHub repo lookup + PR fetch path
        repo_obj = Mock()
        integration.github.get_repo = MagicMock(return_value=repo_obj)
        integration._get_pull_requests = MagicMock(return_value=[])
        integration._get_cached_prs_bulk = MagicMock(return_value=[])
        integration._refresh_stale_open_prs = MagicMock(return_value=[])

        backfill = datetime(2025, 6, 1, tzinfo=timezone.utc)
        integration.enrich_repository_with_prs(
            "owner/repo",
            commits=[],
            since=datetime(2026, 3, 1, tzinfo=timezone.utc),
            backfill_since=backfill,
        )

        # _get_pull_requests must have been called with the BACKFILL date,
        # not the more-recent last_processed_date.
        called_args = integration._get_pull_requests.call_args
        assert called_args is not None
        _, since_arg = called_args.args
        assert since_arg == backfill


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCliBackfillSinceArgument:
    def test_help_lists_backfill_since_flag(self) -> None:
        from gitflow_analytics.cli_fetch import fetch

        runner = CliRunner()
        result = runner.invoke(fetch, ["--help"])
        assert result.exit_code == 0
        assert "--backfill-since" in result.output

    def test_invalid_date_format_exits_with_code_2(self, tmp_path) -> None:
        """Bad YYYY-MM-DD strings should fail fast before any heavy work."""
        from gitflow_analytics.cli_fetch import fetch

        # Need a real (empty) config file to satisfy click.Path(exists=True).
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# placeholder\n")

        runner = CliRunner()
        result = runner.invoke(
            fetch,
            ["-c", str(config_file), "--backfill-since", "not-a-date"],
        )
        assert result.exit_code == 2
        assert "Invalid --backfill-since" in result.output

    def test_valid_date_passes_argument_validation(self, tmp_path) -> None:
        """A well-formed date should not be rejected by the CLI parser itself.

        We don't run the full fetch (no real config / repos), so we only
        assert that argument parsing doesn't fail with exit code 2 (validation
        error).  Subsequent failures (config loading, etc.) are unrelated.
        """
        from gitflow_analytics.cli_fetch import fetch

        config_file = tmp_path / "config.yaml"
        config_file.write_text("# placeholder\n")

        runner = CliRunner()
        result = runner.invoke(
            fetch,
            ["-c", str(config_file), "--backfill-since", "2025-01-01"],
        )
        # Argument validation passes — exit code 2 specifically means our
        # validator rejected the value.  Anything else (1, etc.) means we
        # got past validation.
        assert result.exit_code != 2 or "Invalid --backfill-since" not in result.output


# ---------------------------------------------------------------------------
# _get_pull_requests: pagination correctness (issue #52)
# ---------------------------------------------------------------------------


def _make_pr(
    *,
    number: int,
    updated_at: datetime,
    merged_at: datetime | None = None,
    closed_at: datetime | None = None,
    state: str = "closed",
) -> Mock:
    """Build a mock PyGitHub PullRequest object with the fields we use."""
    pr = Mock()
    pr.number = number
    pr.updated_at = updated_at
    pr.merged_at = merged_at
    pr.merged = merged_at is not None
    pr.closed_at = closed_at if closed_at is not None else merged_at
    pr.state = state
    return pr


class TestGetPullRequestsPagination:
    """Issue #52: pagination must not abort on the first out-of-window PR.

    GitHub returns PRs sorted by ``updated_at`` desc.  A PR's ``updated_at`` is
    bumped by any activity (comments, labels, CI) even years after merge.  So a
    "quiet" PR (merged within the window but no post-merge activity) can appear
    AFTER PRs with recent activity but NEWER PRs merged in-window may still
    follow it in the stream.  Previously the loop ``break``-ed on the first
    such PR, silently dropping the rest.
    """

    def test_does_not_abort_on_quiet_pr_before_in_window_prs(self) -> None:
        """A quiet PR (updated_at < since) must NOT terminate pagination.

        Reproduces the exact scenario from issue #52:
        - PR#500 updated=11/22 merged=11/20 → included (in window)
        - PR#490 updated=11/14 merged=11/13 → included (in window)
        - PR#480 updated=10/28 merged=10/27 → out-of-window (the buggy break trigger)
        - PR#485 updated=11/05 merged=11/04 → MUST still be reached & included
        """
        integration = _make_integration()
        since = datetime(2025, 11, 1, tzinfo=timezone.utc)

        pr500 = _make_pr(
            number=500,
            updated_at=datetime(2025, 11, 22, tzinfo=timezone.utc),
            merged_at=datetime(2025, 11, 20, tzinfo=timezone.utc),
        )
        pr490 = _make_pr(
            number=490,
            updated_at=datetime(2025, 11, 14, tzinfo=timezone.utc),
            merged_at=datetime(2025, 11, 13, tzinfo=timezone.utc),
        )
        pr480 = _make_pr(  # out-of-window "quiet" PR (the bug trigger)
            number=480,
            updated_at=datetime(2025, 10, 28, tzinfo=timezone.utc),
            merged_at=datetime(2025, 10, 27, tzinfo=timezone.utc),
        )
        pr485 = _make_pr(  # in-window, but appears AFTER pr480 in the stream
            number=485,
            updated_at=datetime(2025, 11, 5, tzinfo=timezone.utc),
            merged_at=datetime(2025, 11, 4, tzinfo=timezone.utc),
        )

        repo = Mock()
        repo.get_pulls = MagicMock(return_value=iter([pr500, pr490, pr480, pr485]))

        result = integration._get_pull_requests(repo, since)
        numbers = [pr.number for pr in result]

        # pr480 is out-of-window so excluded; pr485 IS in-window and MUST be present.
        assert 500 in numbers
        assert 490 in numbers
        assert 485 in numbers, "Bug #52: in-window PR after a quiet PR was dropped"
        assert 480 not in numbers

    def test_safety_valve_stops_after_max_consecutive_misses(self) -> None:
        """After MAX_CONSECUTIVE_MISSES consecutive out-of-window PRs, stop.

        Uses MAX_CONSECUTIVE_MISSES + 2 = 5 out-of-window PRs to trigger the
        valve, ensuring the trap PR placed after them is never reached.
        """
        integration = _make_integration()
        since = datetime(2025, 11, 1, tzinfo=timezone.utc)

        # 1 in-window PR followed by MAX_CONSECUTIVE_MISSES + 2 = 5 out-of-window
        # PRs (enough to exceed the valve threshold of 3).
        in_window = _make_pr(
            number=1000,
            updated_at=datetime(2025, 11, 20, tzinfo=timezone.utc),
            merged_at=datetime(2025, 11, 19, tzinfo=timezone.utc),
        )
        out_of_window = [
            _make_pr(
                number=900 - i,
                updated_at=datetime(2025, 10, 20, tzinfo=timezone.utc),
                merged_at=datetime(2025, 10, 19, tzinfo=timezone.utc),
            )
            for i in range(5)  # MAX_CONSECUTIVE_MISSES + 2 = 5
        ]
        # A "trap" in-window PR placed after the valve trigger zone.
        # We expect pagination to STOP before reaching it.
        trap = _make_pr(
            number=42,
            updated_at=datetime(2025, 11, 25, tzinfo=timezone.utc),
            merged_at=datetime(2025, 11, 24, tzinfo=timezone.utc),
        )
        stream = [in_window, *out_of_window, trap]

        repo = Mock()
        repo.get_pulls = MagicMock(return_value=iter(stream))

        result = integration._get_pull_requests(repo, since)
        numbers = [pr.number for pr in result]

        assert 1000 in numbers
        # The trap MUST NOT be reached — safety valve fired.
        assert 42 not in numbers
        # None of the out-of-window PRs should be included.
        for pr in out_of_window:
            assert pr.number not in numbers

    def test_consecutive_miss_counter_resets_on_in_window_pr(self) -> None:
        """A mix of in-window and out-of-window PRs resets the miss counter.

        Reaches MAX_CONSECUTIVE_MISSES - 1 = 2 misses, then an in-window PR
        resets the counter, then more out-of-window PRs should be tolerated again.
        """
        integration = _make_integration()
        since = datetime(2025, 11, 1, tzinfo=timezone.utc)

        in1 = _make_pr(
            number=1,
            updated_at=datetime(2025, 11, 25, tzinfo=timezone.utc),
            merged_at=datetime(2025, 11, 24, tzinfo=timezone.utc),
        )
        # MAX_CONSECUTIVE_MISSES - 1 = 2 out-of-window (one shy of the valve)
        misses_a = [
            _make_pr(
                number=100 + i,
                updated_at=datetime(2025, 10, 20, tzinfo=timezone.utc),
                merged_at=datetime(2025, 10, 19, tzinfo=timezone.utc),
            )
            for i in range(2)
        ]
        # In-window PR resets the counter
        reset = _make_pr(
            number=2,
            updated_at=datetime(2025, 11, 10, tzinfo=timezone.utc),
            merged_at=datetime(2025, 11, 9, tzinfo=timezone.utc),
        )
        # 2 more out-of-window — total accumulated misses would be 4 without
        # reset (triggering the valve), but with reset we never hit 3 in a row.
        misses_b = [
            _make_pr(
                number=200 + i,
                updated_at=datetime(2025, 10, 15, tzinfo=timezone.utc),
                merged_at=datetime(2025, 10, 14, tzinfo=timezone.utc),
            )
            for i in range(2)
        ]
        # Final in-window PR proves we kept paginating after the reset.
        final = _make_pr(
            number=3,
            updated_at=datetime(2025, 11, 8, tzinfo=timezone.utc),
            merged_at=datetime(2025, 11, 7, tzinfo=timezone.utc),
        )

        stream = [in1, *misses_a, reset, *misses_b, final]
        repo = Mock()
        repo.get_pulls = MagicMock(return_value=iter(stream))

        result = integration._get_pull_requests(repo, since)
        numbers = [pr.number for pr in result]

        assert numbers == [1, 2, 3], f"Counter reset failed — expected [1,2,3], got {numbers}"
