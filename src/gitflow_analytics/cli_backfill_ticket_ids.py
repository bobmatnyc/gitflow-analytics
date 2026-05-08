"""CLI command for backfilling commit_count + ticket_ids on pull_request_cache.

Issue #53: pull_request_cache gained two derived columns
    - commit_count: len(commit_hashes)
    - ticket_ids:   JSON list of [A-Z]{2,10}-\\d+ tickets extracted from commit
                    messages and PR titles.

For PRs cached before this change both columns are NULL.  This command joins
``pull_request_cache`` against ``cached_commits`` (by commit hash membership)
to backfill the values without re-hitting the GitHub API.

Issue #54: also scans ``pull_request_cache.title`` for ticket references —
many PRs reference tickets only in the title, not in their commit messages
(414 such PRs in the original report, lifting linkage from 33.7% -> ~60.3%).
The regex was tightened in the same change to exclude known non-ticket
prefixes (CVE, CWE, RFC, ...) which were inflating false-positive matches.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from sqlalchemy import or_, update

from .cli_utils import setup_logging
from .config import ConfigLoader
from .core.cache_commits import _extract_ticket_ids_from_messages
from .models.database import (
    CachedCommit,
    Database,
    PullRequestCache,
)

logger = logging.getLogger(__name__)


def backfill_pr_cache(db: Database, repo_filter: str | None = None) -> dict[str, int]:
    """Backfill commit_count and ticket_ids on pull_request_cache.

    WHY: PRs cached before issue #53 have NULL ``commit_count`` and NULL
    ``ticket_ids``.  This function joins each such PR's ``commit_hashes`` JSON
    list against the ``cached_commits`` table (by hash) to recover commit
    messages, then extracts ticket IDs from those messages.

    The function is idempotent — re-running it is safe and updates only rows
    where one or both columns are NULL.

    Args:
        db: Database wrapper exposing get_session().
        repo_filter: Optional repo_path substring to limit scope.

    Returns:
        Dict with counts: ``prs_examined``, ``commit_count_set``, ``ticket_ids_set``.
    """
    stats = {"prs_examined": 0, "commit_count_set": 0, "ticket_ids_set": 0}

    if db.SessionLocal is None:
        # Database initialization failed (e.g., readonly filesystem). Nothing to do.
        logger.warning("Database SessionLocal is None — skipping backfill")
        return stats

    with db.SessionLocal() as session:
        query = session.query(PullRequestCache).filter(
            or_(
                PullRequestCache.commit_count.is_(None),
                PullRequestCache.ticket_ids.is_(None),
            )
        )
        if repo_filter:
            query = query.filter(PullRequestCache.repo_path == repo_filter)

        prs = query.all()
        stats["prs_examined"] = len(prs)

        for pr in prs:
            hashes = pr.commit_hashes or []
            # commit_hashes may be stored as a JSON-decoded list (SQLAlchemy JSON)
            # or as a stringified JSON in older rows.
            if isinstance(hashes, str):
                try:
                    hashes = json.loads(hashes)
                except (ValueError, TypeError):
                    hashes = []
            if not isinstance(hashes, list):
                hashes = []

            # Build update values dict so we can issue a single UPDATE per PR
            # rather than ORM attribute assignments (which Pyright flags
            # because Column[T] is the descriptor type, not the runtime value).
            values: dict[str, object] = {}

            # Always (re)compute commit_count when NULL — cheap and unambiguous.
            if pr.commit_count is None:
                values["commit_count"] = len(hashes)
                stats["commit_count_set"] += 1

            # Backfill ticket_ids when NULL by joining against cached_commits
            # AND scanning the PR title (issue #54).  Both sources contribute;
            # results are deduplicated by _extract_ticket_ids_from_messages.
            if pr.ticket_ids is None:
                # Issue #54: PR title is a first-class source of ticket refs.
                # Include it even when there are no commit hashes — many PRs
                # have the ticket only in the title.
                # WHY: pr.title is typed as Column[str] by SQLAlchemy; coerce
                # via str() and then check the runtime value to satisfy
                # Pyright's reportGeneralTypeIssues check on Column conditionals.
                raw_title = pr.title
                title_text = raw_title if isinstance(raw_title, str) else None
                title_messages: list[str] = (
                    [title_text] if title_text is not None and len(title_text) > 0 else []
                )

                if not hashes:
                    ids = _extract_ticket_ids_from_messages(title_messages)
                    values["ticket_ids"] = json.dumps(ids)
                    stats["ticket_ids_set"] += 1
                else:
                    rows = (
                        session.query(CachedCommit.message)
                        .filter(CachedCommit.commit_hash.in_(hashes))
                        .all()
                    )
                    commit_messages = [str(r[0]) for r in rows if r and r[0]]
                    ids = _extract_ticket_ids_from_messages(title_messages + commit_messages)
                    values["ticket_ids"] = json.dumps(ids)
                    stats["ticket_ids_set"] += 1

            if values:
                session.execute(
                    update(PullRequestCache).where(PullRequestCache.id == pr.id).values(**values)
                )

        session.commit()

    return stats


@click.command("backfill-ticket-ids")
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
    "repo_filter",
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
def backfill_ticket_ids_command(
    config_path: Path,
    repo_filter: str | None,
    log: str,
) -> None:
    """Backfill commit_count and ticket_ids on pull_request_cache (issue #53).

    \b
    Reads from cached_commits.message (no GitHub API calls) and updates rows
    in pull_request_cache where commit_count or ticket_ids is NULL.  Idempotent.

    \b
    EXAMPLES:
      gfa backfill-ticket-ids -c config.yaml
      gfa backfill-ticket-ids -c config.yaml --repo owner/repo
    """
    setup_logging(log, __name__)

    try:
        cfg = ConfigLoader.load(config_path)
        db_path = cfg.cache.directory / "gitflow_cache.db"
        db = Database(db_path)

        click.echo("Backfilling pull_request_cache.commit_count and ticket_ids...")
        stats = backfill_pr_cache(db, repo_filter=repo_filter)

        click.echo(f"  PRs examined:        {stats['prs_examined']}")
        click.echo(f"  commit_count set:    {stats['commit_count_set']}")
        click.echo(f"  ticket_ids set:      {stats['ticket_ids_set']}")
        click.echo("Done.")

    except Exception as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)


def register_backfill_ticket_ids_commands(cli_group: click.Group) -> None:
    """Register the backfill-ticket-ids command on the given Click group."""
    cli_group.add_command(backfill_ticket_ids_command)
