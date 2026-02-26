"""Identity and alias management CLI commands for GitFlow Analytics.

This module contains core identity commands (merge_identity, list_developers)
and the registration function. Heavier analysis commands are split across:
- cli_identity_commands.py: identities, aliases_command
- cli_identity_alias_ops.py: create_alias_interactive, alias_rename

All symbols are re-exported here for backward compatibility with existing
importers (e.g. cli.py imports all 7 names from this module).
"""

import logging
import sys
from pathlib import Path

import click

from .cli_identity_alias_ops import alias_rename, create_alias_interactive  # noqa: F401
from .cli_identity_commands import aliases_command, identities  # noqa: F401
from .config import ConfigLoader

logger = logging.getLogger(__name__)


def register_identity_commands(cli_group: click.Group) -> None:
    """Register all identity/alias commands onto the provided Click group.

    Args:
        cli_group: The Click group to register commands on (typically the main cli group)
    """
    cli_group.add_command(merge_identity)
    cli_group.add_command(identities)
    cli_group.add_command(aliases_command)
    cli_group.add_command(list_developers)
    cli_group.add_command(create_alias_interactive)
    cli_group.add_command(alias_rename)


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.argument("dev1", metavar="PRIMARY_EMAIL")
@click.argument("dev2", metavar="ALIAS_EMAIL")
def merge_identity(config: Path, dev1: str, dev2: str) -> None:
    """Merge two developer identities into one.

    \b
    Consolidates commits from ALIAS_EMAIL under PRIMARY_EMAIL.
    This is useful when a developer has multiple email addresses
    that weren't automatically detected.

    \b
    EXAMPLES:
      # Merge john's gmail into his work email
      gitflow-analytics merge-identity -c config.yaml john@work.com john@gmail.com

    \b
    The merge:
    - Updates all historical commits
    - Refreshes cached statistics
    - Updates identity mappings
    """
    from .core.identity import DeveloperIdentityResolver

    try:
        cfg = ConfigLoader.load(config)
        identity_resolver = DeveloperIdentityResolver(cfg.cache.directory / "identities.db")

        click.echo(f"Merging {dev2} into {dev1}...")
        identity_resolver.merge_identities(dev1, dev2)
        click.echo("Identities merged successfully!")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def list_developers(config: Path) -> None:
    """List all known developers with statistics.

    \b
    Displays a table of developers showing:
    - Primary name and email
    - Total commit count
    - Story points delivered
    - Number of identity aliases

    \b
    EXAMPLES:
      # List all developers
      gitflow-analytics list-developers -c config.yaml

    \b
    Useful for:
    - Verifying identity resolution
    - Finding developer email addresses
    - Checking contribution statistics
    """
    from .core.identity import DeveloperIdentityResolver

    try:
        cfg = ConfigLoader.load(config)
        identity_resolver = DeveloperIdentityResolver(cfg.cache.directory / "identities.db")

        developers = identity_resolver.get_developer_stats()

        if not developers:
            click.echo("No developers found. Run analysis first.")
            return

        click.echo("Known Developers:")
        click.echo(f"{'Name':<30} {'Email':<40} {'Commits':<10} {'Points':<10} {'Aliases'}")
        click.echo("-" * 100)

        for dev in developers[:20]:  # Show top 20
            click.echo(
                f"{dev['primary_name']:<30} "
                f"{dev['primary_email']:<40} "
                f"{dev['total_commits']:<10} "
                f"{dev['total_story_points']:<10} "
                f"{dev['alias_count']}"
            )

        if len(developers) > 20:
            click.echo(f"\n... and {len(developers) - 20} more developers")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
