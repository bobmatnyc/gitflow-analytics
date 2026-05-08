"""CLI command for backfilling ``is_revert`` on cached_commits (issue #64).

The ``cached_commits`` table gained an ``is_revert`` column in v15.0,
populated at ingestion by ``cache_commit()`` / ``cache_commits_batch()`` /
``bulk_store_commits()``.  Commits cached by older builds have
``is_revert = FALSE`` regardless of their message content, which causes
``daily_metrics.reversion_commits`` and ``weekly_trends.reversion_commits``
to under-count reverts on historical data.

This subcommand re-runs
:func:`gitflow_analytics.utils.revert_detection.is_revert_commit` over every
cached row whose ``is_revert`` is currently ``FALSE`` (or NULL on rare
half-migrated databases) and persists the result.  Idempotent — re-running
finds zero un-flipped rows.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from .cli_utils import setup_logging
from .config import ConfigLoader
from .core.cache import GitAnalysisCache

logger = logging.getLogger(__name__)


@click.command("backfill-revert-flags")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to GitFlow Analytics config file (YAML).",
)
@click.option(
    "--batch-size",
    type=int,
    default=1000,
    show_default=True,
    help="Rows per UPDATE flush; tuned for SQLite bulk performance.",
)
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="none",
    help="Enable logging with specified level.",
)
def backfill_revert_flags_command(
    config_path: Path,
    batch_size: int,
    log: str,
) -> None:
    """Backfill is_revert on cached_commits (issue #64).

    \b
    Scans cached_commits for rows whose is_revert is FALSE (or NULL),
    re-evaluates the revert regex set on each commit message, and flips the
    flag to TRUE for matches.  Idempotent — safe to run multiple times.

    \b
    EXAMPLES:
      gfa backfill-revert-flags -c config.yaml
      gfa backfill-revert-flags -c config.yaml --batch-size 5000 --log INFO
    """
    setup_logging(log, __name__)

    try:
        cfg = ConfigLoader.load(config_path)
    except Exception as exc:  # noqa: BLE001 — surface config errors to user
        click.echo(f"Error: failed to load config: {exc}", err=True)
        sys.exit(1)

    cache = GitAnalysisCache(cfg.cache.directory, ttl_hours=cfg.cache.ttl_hours)

    click.echo("Backfilling is_revert flag on cached_commits...")
    try:
        result = cache.backfill_revert_flags(batch_size=batch_size)
    except Exception as exc:  # noqa: BLE001 — surface DB errors to user
        click.echo(f"Error: backfill failed: {exc}", err=True)
        sys.exit(1)

    scanned = int(result.get("scanned", 0))
    updated = int(result.get("updated", 0))

    click.echo(f"  Scanned:        {scanned}")
    click.echo(f"  Reverts found:  {updated}")
    click.echo("Done.")


def register_backfill_revert_flags_commands(cli_group: click.Group) -> None:
    """Register the backfill-revert-flags command on the given Click group."""
    cli_group.add_command(backfill_revert_flags_command)
