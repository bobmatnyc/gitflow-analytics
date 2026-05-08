"""CLI command for backfilling AI detection signals on cached_commits (issue #47).

The ``cached_commits`` table gained two columns in v9.0 — ``ai_confidence_score``
and ``ai_detection_method`` — populated at collection time by
``analyzer_commit.py`` and (defensively) by the cache write path.  Commits
written before those changes have NULL ``ai_confidence_score``.

This subcommand re-runs ``detect_ai_commit()`` over every cached row that has
NULL ``ai_confidence_score`` and persists the results in place.  It is
idempotent: re-running scans only the rows that are still NULL.

Notes:
    File-based signals (e.g., ``.cursorrules`` adds) are skipped because
    cached rows do not retain the changed-file list.  Detection falls back
    to message-only heuristics, which is what is available on cached data.
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


@click.command("backfill-ai-detection")
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
    default=500,
    show_default=True,
    help="Rows per UPDATE flush; tuned for SQLite bulk performance.",
)
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="none",
    help="Enable logging with specified level.",
)
def backfill_ai_detection_command(
    config_path: Path,
    batch_size: int,
    log: str,
) -> None:
    """Backfill ai_confidence_score / ai_detection_method on cached_commits (issue #47).

    \b
    Scans cached_commits for rows with NULL ai_confidence_score, runs
    detect_ai_commit() on each commit message, and persists the resulting
    confidence + method back to the row.  Idempotent — safe to run multiple
    times.  Emits a summary including AI adoption rate.

    \b
    EXAMPLES:
      gfa backfill-ai-detection -c config.yaml
      gfa backfill-ai-detection -c config.yaml --batch-size 1000 --log INFO
    """
    setup_logging(log, __name__)

    try:
        cfg = ConfigLoader.load(config_path)
    except Exception as exc:  # noqa: BLE001 — surface config errors to user
        click.echo(f"Error: failed to load config: {exc}", err=True)
        sys.exit(1)

    cache = GitAnalysisCache(cfg.cache.directory, ttl_hours=cfg.cache.ttl_hours)

    click.echo("Backfilling AI detection signals on cached_commits...")
    try:
        result = cache.backfill_ai_detection(batch_size=batch_size)
    except Exception as exc:  # noqa: BLE001 — surface DB errors to user
        click.echo(f"Error: backfill failed: {exc}", err=True)
        sys.exit(1)

    scanned = int(result.get("scanned", 0))
    updated = int(result.get("updated", 0))
    method_counts: dict[str, int] = result.get("method_counts", {}) or {}

    # Compute AI adoption rate over scanned rows.  A commit is "AI-attributed"
    # when its detection method is anything other than the empty/none sentinels.
    ai_methods = {m: c for m, c in method_counts.items() if m not in ("", "none")}
    ai_count = sum(ai_methods.values())
    adoption_rate = (ai_count / scanned * 100.0) if scanned else 0.0

    click.echo(f"  Scanned:           {scanned}")
    click.echo(f"  Updated:           {updated}")
    click.echo(f"  AI commits found:  {ai_count}")
    click.echo(f"  AI adoption rate:  {adoption_rate:.1f}%")
    if method_counts:
        click.echo("  Method breakdown:")
        for method, count in sorted(method_counts.items(), key=lambda kv: -kv[1]):
            label = method or "(empty)"
            click.echo(f"    - {label}: {count}")
    click.echo("Done.")


def register_backfill_ai_detection_commands(cli_group: click.Group) -> None:
    """Register the backfill-ai-detection command on the given Click group."""
    cli_group.add_command(backfill_ai_detection_command)
