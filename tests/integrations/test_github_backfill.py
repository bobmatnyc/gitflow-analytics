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

    def test_no_force_since_uses_max_of_last_and_requested(self) -> None:
        """Sanity check: default behaviour is preserved when force_since is None."""
        integration = _make_integration()
        integration.schema_manager.has_schema_changed = MagicMock(return_value=False)
        recent = datetime(2026, 3, 23, tzinfo=timezone.utc)
        integration.schema_manager.get_last_processed_date = MagicMock(return_value=recent)

        requested = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = integration._get_incremental_fetch_date("github", requested, {})

        # Must advance to the more recent date — that's the *bug* backfill fixes.
        assert result == recent


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
