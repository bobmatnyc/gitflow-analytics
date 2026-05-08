"""CLI commands for managing manual classification overrides.

Issue #63: Manual classification corrections persist in the
``classification_overrides`` table so reruns of the classify pipeline never
overwrite curated work_type values. This module exposes the ``gfa override``
command group with three subcommands:

    gfa override set    COMMIT_HASH WORK_TYPE --repo REPO_PATH --reason "..."
    gfa override list   [--repo REPO_PATH]
    gfa override remove COMMIT_HASH --repo REPO_PATH

The ``set`` command is an upsert keyed on (commit_hash, repo_path); each
commit can have at most one override per repository.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from .cli_utils import setup_logging
from .config import ConfigLoader
from .models.database import ClassificationOverride, Database

logger = logging.getLogger(__name__)


def _resolve_db(config_path: Path) -> Database:
    """Load config and return the underlying gitflow_cache Database."""
    cfg = ConfigLoader.load(config_path)
    db_path = cfg.cache.directory / "gitflow_cache.db"
    return Database(db_path)


@click.group("override")
def override_group() -> None:
    """Manage manual classification overrides (issue #63).

    \b
    Persistent corrections that take priority over every classifier
    (LLM, JIRA project key, fallback). Reruns of ``gfa classify`` will
    NOT overwrite commits that have an override.
    """


@override_group.command("set")
@click.argument("commit_hash", type=str)
@click.argument("work_type", type=str)
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to GitFlow Analytics config file (YAML).",
)
@click.option(
    "--repo",
    "repo_path",
    type=str,
    required=True,
    help="Repository path (matches CachedCommit.repo_path).",
)
@click.option(
    "--reason",
    type=str,
    default=None,
    help="Optional human-readable note explaining the override.",
)
@click.option(
    "--created-by",
    type=str,
    default=None,
    help="Optional username/identifier for the operator.",
)
@click.option(
    "--confidence",
    type=float,
    default=1.0,
    show_default=True,
    help="Confidence value to record alongside the override.",
)
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="none",
    help="Enable logging with specified level.",
)
def override_set(
    commit_hash: str,
    work_type: str,
    config_path: Path,
    repo_path: str,
    reason: str | None,
    created_by: str | None,
    confidence: float,
    log: str,
) -> None:
    """Upsert a manual classification override for a single commit.

    \b
    EXAMPLES:
      gfa override set abc123def feature -c config.yaml --repo /path/to/repo
      gfa override set abc123def bug_fix -c config.yaml --repo /path/to/repo \\
          --reason "Misclassified by LLM"
    """
    setup_logging(log, __name__)
    try:
        db = _resolve_db(config_path)
        if db.SessionLocal is None:
            raise click.ClickException("Database is unavailable (read-only filesystem?)")

        with db.SessionLocal() as session:
            existing = (
                session.query(ClassificationOverride)
                .filter(
                    ClassificationOverride.commit_hash == commit_hash,
                    ClassificationOverride.repo_path == repo_path,
                )
                .first()
            )
            now = datetime.now(timezone.utc)
            if existing is not None:
                existing.work_type = work_type  # type: ignore[assignment]
                existing.confidence = confidence  # type: ignore[assignment]
                existing.reason = reason  # type: ignore[assignment]
                if created_by is not None:
                    existing.created_by = created_by  # type: ignore[assignment]
                existing.updated_at = now  # type: ignore[assignment]
                action = "updated"
            else:
                session.add(
                    ClassificationOverride(
                        commit_hash=commit_hash,
                        repo_path=repo_path,
                        work_type=work_type,
                        confidence=confidence,
                        reason=reason,
                        created_by=created_by,
                    )
                )
                action = "created"
            session.commit()

        click.echo(f"Override {action}: {commit_hash[:7]} -> {work_type} (repo={repo_path})")
    except click.ClickException:
        raise
    except Exception as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)


@override_group.command("list")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to GitFlow Analytics config file (YAML).",
)
@click.option(
    "--repo",
    "repo_path",
    type=str,
    default=None,
    help="Optional repo_path filter (exact match).",
)
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="none",
    help="Enable logging with specified level.",
)
def override_list(
    config_path: Path,
    repo_path: str | None,
    log: str,
) -> None:
    """List manual classification overrides.

    \b
    EXAMPLES:
      gfa override list -c config.yaml
      gfa override list -c config.yaml --repo /path/to/repo
    """
    setup_logging(log, __name__)
    try:
        db = _resolve_db(config_path)
        if db.SessionLocal is None:
            raise click.ClickException("Database is unavailable (read-only filesystem?)")

        with db.SessionLocal() as session:
            query = session.query(ClassificationOverride)
            if repo_path:
                query = query.filter(ClassificationOverride.repo_path == repo_path)
            rows = query.order_by(
                ClassificationOverride.repo_path,
                ClassificationOverride.commit_hash,
            ).all()

        if not rows:
            click.echo("No overrides found.")
            return

        click.echo(f"Found {len(rows)} override(s):")
        click.echo(f"  {'COMMIT':<10} {'WORK_TYPE':<14} {'CONF':<5} {'REPO':<40} REASON")
        for row in rows:
            commit_short = str(row.commit_hash)[:9]
            work_type = str(row.work_type)
            conf_value = float(row.confidence) if row.confidence is not None else 1.0  # type: ignore[arg-type]
            conf = f"{conf_value:.2f}"
            repo = str(row.repo_path)
            reason = str(row.reason or "")
            click.echo(f"  {commit_short:<10} {work_type:<14} {conf:<5} {repo:<40} {reason}")
    except click.ClickException:
        raise
    except Exception as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)


@override_group.command("remove")
@click.argument("commit_hash", type=str)
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to GitFlow Analytics config file (YAML).",
)
@click.option(
    "--repo",
    "repo_path",
    type=str,
    required=True,
    help="Repository path (must match the override's repo_path).",
)
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="none",
    help="Enable logging with specified level.",
)
def override_remove(
    commit_hash: str,
    config_path: Path,
    repo_path: str,
    log: str,
) -> None:
    """Delete a manual classification override.

    \b
    EXAMPLES:
      gfa override remove abc123def -c config.yaml --repo /path/to/repo
    """
    setup_logging(log, __name__)
    try:
        db = _resolve_db(config_path)
        if db.SessionLocal is None:
            raise click.ClickException("Database is unavailable (read-only filesystem?)")

        with db.SessionLocal() as session:
            existing = (
                session.query(ClassificationOverride)
                .filter(
                    ClassificationOverride.commit_hash == commit_hash,
                    ClassificationOverride.repo_path == repo_path,
                )
                .first()
            )
            if existing is None:
                click.echo(f"No override found for {commit_hash[:7]} in {repo_path}.")
                sys.exit(1)

            session.delete(existing)
            session.commit()
        click.echo(f"Override removed: {commit_hash[:7]} (repo={repo_path})")
    except click.ClickException:
        raise
    except Exception as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)


def register_override_commands(cli_group: click.Group) -> None:
    """Register the ``override`` command group on the given Click group."""
    cli_group.add_command(override_group)
