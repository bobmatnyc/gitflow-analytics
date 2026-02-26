"""Interactive alias management CLI commands for GitFlow Analytics.

This module contains:
- create_alias_interactive: Interactive alias creation from numbered list
- alias_rename: Rename a developer's canonical display name

Analysis commands (identities, aliases_command) live in cli_identity_commands.py.
Core commands (merge_identity, list_developers) live in cli_identity.py.
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

import click
import yaml

from .config import ConfigLoader

logger = logging.getLogger(__name__)


@click.command(name="create-alias-interactive")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for aliases.yaml (default: same dir as config)",
)
def create_alias_interactive(config: Path, output: Optional[Path]) -> None:
    """Create developer aliases interactively with numbered selection.

    \b
    This command provides an interactive interface to create developer
    aliases by selecting from a numbered list of developers in the database.
    You can merge multiple developer identities and save them to aliases.yaml.

    \b
    EXAMPLES:
      # Start interactive alias creation
      gitflow-analytics create-alias-interactive -c config.yaml

      # Save to specific location
      gitflow-analytics create-alias-interactive -c config.yaml -o ~/shared/aliases.yaml

    \b
    WORKFLOW:
      1. Displays numbered list of all developers from database
      2. Select multiple developer numbers to merge (space-separated)
      3. Choose which one should be the primary identity
      4. Create alias mapping
      5. Option to save to aliases.yaml
      6. Option to continue creating more aliases

    \b
    Useful for:
    - Consolidating developer identities across email addresses
    - Cleaning up duplicate developer entries
    - Maintaining consistent identity resolution
    """
    from .config.aliases import AliasesManager, DeveloperAlias
    from .core.identity import DeveloperIdentityResolver

    try:
        cfg = ConfigLoader.load(config)

        if not output:
            output = config.parent / "aliases.yaml"

        identity_resolver = DeveloperIdentityResolver(cfg.cache.directory / "identities.db")
        aliases_manager = AliasesManager(output if output.exists() else None)

        click.echo("\n" + "=" * 80)
        click.echo(click.style("Interactive Alias Creator", fg="cyan", bold=True))
        click.echo("=" * 80 + "\n")

        continue_creating = True

        while continue_creating:
            developers = identity_resolver.get_developer_stats()

            if not developers:
                click.echo("No developers found. Run analysis first.")
                sys.exit(1)

            click.echo(
                click.style(f"\nFound {len(developers)} developers:\n", fg="green", bold=True)
            )
            click.echo(f"{'#':<6} {'Name':<30} {'Email':<40} {'Commits':<10}")
            click.echo("-" * 86)

            for idx, dev in enumerate(developers, start=1):
                click.echo(
                    f"{idx:<6} "
                    f"{dev['primary_name']:<30} "
                    f"{dev['primary_email']:<40} "
                    f"{dev['total_commits']:<10}"
                )

            click.echo()

            while True:
                try:
                    selection_input = click.prompt(
                        click.style(
                            "Select developers to merge (enter numbers separated by spaces, or 'q' to quit)",
                            fg="yellow",
                        ),
                        type=str,
                    ).strip()

                    if selection_input.lower() in ["q", "quit", "exit"]:
                        click.echo("\nExiting alias creation.")
                        sys.exit(0)

                    selected_indices = []
                    for num_str in selection_input.split():
                        try:
                            num = int(num_str)
                            if 1 <= num <= len(developers):
                                selected_indices.append(num)
                            else:
                                click.echo(
                                    click.style(
                                        f"Number {num} is out of range (1-{len(developers)})",
                                        fg="red",
                                    )
                                )
                                raise ValueError("Invalid range")
                        except ValueError:
                            click.echo(
                                click.style(
                                    f"Invalid input: '{num_str}' is not a valid number", fg="red"
                                )
                            )
                            raise

                    if len(selected_indices) < 2:
                        click.echo(
                            click.style(
                                "You must select at least 2 developers to merge", fg="red"
                            )
                        )
                        continue

                    selected_indices = sorted(set(selected_indices))
                    break

                except ValueError:
                    continue
                except click.exceptions.Abort:
                    click.echo("\n\nExiting alias creation.")
                    sys.exit(0)

            selected_devs = [developers[idx - 1] for idx in selected_indices]

            click.echo(click.style("\nSelected developers:", fg="green", bold=True))
            for idx, dev in zip(selected_indices, selected_devs):
                click.echo(
                    f"  [{idx}] {dev['primary_name']} <{dev['primary_email']}> "
                    f"({dev['total_commits']} commits)"
                )

            click.echo()
            while True:
                try:
                    primary_input = click.prompt(
                        click.style(
                            f"Which developer should be the primary identity? "
                            f"Enter number ({', '.join(map(str, selected_indices))})",
                            fg="yellow",
                        ),
                        type=int,
                    )

                    if primary_input in selected_indices:
                        primary_idx = primary_input
                        break
                    else:
                        click.echo(
                            click.style(
                                f"Please select one of: {', '.join(map(str, selected_indices))}",
                                fg="red",
                            )
                        )
                except ValueError:
                    click.echo(click.style("Please enter a valid number", fg="red"))
                except click.exceptions.Abort:
                    click.echo("\n\nExiting alias creation.")
                    sys.exit(0)

            primary_dev = developers[primary_idx - 1]
            alias_emails = [
                dev["primary_email"]
                for idx, dev in zip(selected_indices, selected_devs)
                if idx != primary_idx
            ]

            new_alias = DeveloperAlias(
                primary_email=primary_dev["primary_email"],
                aliases=alias_emails,
                name=primary_dev["primary_name"],
                confidence=1.0,
                reasoning="Manually created via interactive CLI",
            )

            click.echo(click.style("\nAlias Configuration:", fg="cyan", bold=True))
            click.echo(f"  Primary: {new_alias.name} <{new_alias.primary_email}>")
            click.echo("  Aliases:")
            for alias_email in new_alias.aliases:
                click.echo(f"    - {alias_email}")

            aliases_manager.add_alias(new_alias)

            click.echo()
            if click.confirm(click.style(f"Save alias to {output}?", fg="green"), default=True):
                try:
                    aliases_manager.save()
                    click.echo(click.style(f"Alias saved to {output}", fg="green"))

                    for alias_email in alias_emails:
                        alias_dev = next(
                            (dev for dev in developers if dev["primary_email"] == alias_email),
                            None,
                        )

                        if alias_dev:
                            identity_resolver.merge_identities(
                                primary_dev["canonical_id"],
                                alias_dev["canonical_id"],
                            )
                        else:
                            click.echo(
                                click.style(
                                    f"Warning: Could not find developer entry for {alias_email}",
                                    fg="yellow",
                                )
                            )

                    click.echo(
                        click.style("Database updated with merged identities", fg="green")
                    )

                except Exception as e:
                    click.echo(click.style(f"Error saving alias: {e}", fg="red"), err=True)
            else:
                click.echo(click.style("Alias not saved", fg="yellow"))

            click.echo()
            if not click.confirm(click.style("Create another alias?", fg="cyan"), default=True):
                continue_creating = False

        click.echo(click.style("\nAlias creation completed!", fg="green", bold=True))
        click.echo(f"Aliases file: {output}")
        click.echo(f"\nTo use these aliases, ensure your config references: {output}\n")

    except KeyboardInterrupt:
        click.echo("\n\nInterrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        click.echo(click.style(f"\nError: {e}", fg="red"), err=True)

        traceback.print_exc()
        sys.exit(1)


@click.command(name="alias-rename")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--old-name",
    help="Current canonical name to rename (must match a name in manual_mappings)",
)
@click.option(
    "--new-name",
    help="New canonical display name to use in reports",
)
@click.option(
    "--update-cache",
    is_flag=True,
    help="Update cached database records with the new name",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be changed without applying changes",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Interactive mode: select developer from numbered list",
)
def alias_rename(
    config: Path,
    old_name: str,
    new_name: str,
    update_cache: bool,
    dry_run: bool,
    interactive: bool,
) -> None:
    """Rename a developer's canonical display name.

    \b
    Updates the developer's name in:
    - Configuration file (analysis.identity.manual_mappings)
    - Database cache (if --update-cache is specified)

    \b
    EXAMPLES:
      # Interactive mode: select from numbered list
      gitflow-analytics alias-rename -c config.yaml --interactive

      # Rename with dry-run to see changes
      gitflow-analytics alias-rename -c config.yaml \\
        --old-name "bianco-zaelot" \\
        --new-name "Emiliozzo Bianco" \\
        --dry-run

      # Apply rename to config only
      gitflow-analytics alias-rename -c config.yaml \\
        --old-name "bianco-zaelot" \\
        --new-name "Emiliozzo Bianco"

      # Apply rename to config and update cache
      gitflow-analytics alias-rename -c config.yaml \\
        --old-name "bianco-zaelot" \\
        --new-name "Emiliozzo Bianco" \\
        --update-cache

    \b
    NOTE:
      This command searches through analysis.identity.manual_mappings
      in your config file and updates the 'name' field for the matching
      entry. It preserves all other fields (primary_email, aliases).
    """
    try:
        from .core.identity import DeveloperIdentityResolver

        click.echo(f"\nLoading configuration from {config}...")

        try:
            with open(config, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            click.echo(f"Error loading config file: {e}", err=True)
            sys.exit(1)

        if "analysis" not in config_data:
            click.echo("Error: 'analysis' section not found in config", err=True)
            sys.exit(1)

        if "identity" not in config_data["analysis"]:
            click.echo("Error: 'analysis.identity' section not found in config", err=True)
            sys.exit(1)

        if "manual_mappings" not in config_data["analysis"]["identity"]:
            click.echo(
                "Error: 'analysis.identity.manual_mappings' not found in config", err=True
            )
            sys.exit(1)

        manual_mappings = config_data["analysis"]["identity"]["manual_mappings"]

        if not manual_mappings:
            click.echo("Error: manual_mappings is empty", err=True)
            sys.exit(1)

        if old_name is not None and not old_name.strip():
            click.echo("Error: --old-name cannot be empty", err=True)
            sys.exit(1)

        if new_name is not None and not new_name.strip():
            click.echo("Error: --new-name cannot be empty", err=True)
            sys.exit(1)

        if interactive or old_name is None or new_name is None:
            click.echo("\n" + "=" * 60)
            click.echo(click.style("Current Developers:", fg="cyan", bold=True))
            click.echo("=" * 60 + "\n")

            developer_names = []
            for idx, mapping in enumerate(manual_mappings, 1):
                name = mapping.get("name", "Unknown")
                email = mapping.get("primary_email", "N/A")
                alias_count = len(mapping.get("aliases", []))

                developer_names.append(name)
                click.echo(f"  {idx}. {click.style(name, fg='green')}")
                click.echo(f"     Email: {email}")
                click.echo(f"     Aliases: {alias_count} email(s)")
                click.echo()

            try:
                selection = click.prompt(
                    "Select developer number to rename (or 0 to cancel)",
                    type=click.IntRange(0, len(developer_names)),
                )
            except click.Abort:
                click.echo("\nCancelled by user.")
                sys.exit(0)

            if selection == 0:
                click.echo("\nCancelled.")
                sys.exit(0)

            old_name = developer_names[selection - 1]
            click.echo(f"\nSelected: {click.style(old_name, fg='green')}")

            if not new_name:
                new_name = click.prompt("Enter new canonical name", type=str)

        if not old_name or not old_name.strip():
            click.echo("Error: --old-name cannot be empty", err=True)
            sys.exit(1)

        if not new_name or not new_name.strip():
            click.echo("Error: --new-name cannot be empty", err=True)
            sys.exit(1)

        old_name = old_name.strip()
        new_name = new_name.strip()

        if old_name == new_name:
            click.echo("Error: old-name and new-name are identical", err=True)
            sys.exit(1)

        matching_entry = None
        matching_index = None

        for idx, mapping in enumerate(manual_mappings):
            if mapping.get("name") == old_name:
                matching_entry = mapping
                matching_index = idx
                break

        if not matching_entry:
            click.echo(f"Error: No manual mapping found with name '{old_name}'", err=True)
            click.echo("\nAvailable names in manual_mappings:")
            for mapping in manual_mappings:
                if "name" in mapping:
                    click.echo(f"  - {mapping['name']}")
            sys.exit(1)

        click.echo("\nFound matching entry:")
        click.echo(f"   Current name: {old_name}")
        click.echo(f"   New name:     {new_name}")
        click.echo(f"   Email:        {matching_entry.get('primary_email', 'N/A')}")
        click.echo(f"   Aliases:      {len(matching_entry.get('aliases', []))} email(s)")

        if dry_run:
            click.echo("\nDRY RUN - No changes will be made")

        if not dry_run:
            click.echo("\nUpdating configuration file...")
            manual_mappings[matching_index]["name"] = new_name

            try:
                with open(config, "w", encoding="utf-8") as f:
                    yaml.dump(
                        config_data,
                        f,
                        default_flow_style=False,
                        allow_unicode=True,
                        sort_keys=False,
                    )
                click.echo("Configuration file updated")
            except Exception as e:
                click.echo(f"Error writing config file: {e}", err=True)
                sys.exit(1)
        else:
            click.echo(f"   [Would update config: {config}]")

        if update_cache:
            click.echo("\nChecking database cache...")

            cfg = ConfigLoader.load(config)
            identity_db_path = cfg.cache.directory / "identities.db"

            if not identity_db_path.exists():
                click.echo(f"Warning: Identity database not found at {identity_db_path}")
                click.echo("   Skipping cache update")
            else:
                identity_resolver = DeveloperIdentityResolver(
                    str(identity_db_path),
                    manual_mappings=None,
                )

                from sqlalchemy import text

                with identity_resolver.get_session() as session:
                    result = session.execute(
                        text(
                            "SELECT COUNT(*) FROM developer_identities WHERE primary_name = :old_name"
                        ),
                        {"old_name": old_name},
                    )
                    identity_count = result.scalar()

                    result = session.execute(
                        text("SELECT COUNT(*) FROM developer_aliases WHERE name = :old_name"),
                        {"old_name": old_name},
                    )
                    alias_count = result.scalar()

                click.echo(f"   Found {identity_count} identity record(s)")
                click.echo(f"   Found {alias_count} alias record(s)")

                if identity_count == 0 and alias_count == 0:
                    click.echo("   No database records to update")
                elif not dry_run:
                    click.echo("   Updating database records...")

                    with identity_resolver.get_session() as session:
                        if identity_count > 0:
                            session.execute(
                                text(
                                    "UPDATE developer_identities SET primary_name = :new_name "
                                    "WHERE primary_name = :old_name"
                                ),
                                {"new_name": new_name, "old_name": old_name},
                            )

                        if alias_count > 0:
                            session.execute(
                                text(
                                    "UPDATE developer_aliases SET name = :new_name "
                                    "WHERE name = :old_name"
                                ),
                                {"new_name": new_name, "old_name": old_name},
                            )

                    click.echo("   Database updated")
                else:
                    click.echo(
                        f"   [Would update {identity_count + alias_count} database record(s)]"
                    )

        click.echo(f"\n{'DRY RUN SUMMARY' if dry_run else 'RENAME COMPLETE'}")
        click.echo(f"   Old name: {old_name}")
        click.echo(f"   New name: {new_name}")
        click.echo(f"   Config:   {'Would update' if dry_run else 'Updated'}")
        if update_cache:
            click.echo(f"   Cache:    {'Would update' if dry_run else 'Updated'}")
        else:
            click.echo("   Cache:    Skipped (use --update-cache to update)")

        if dry_run:
            click.echo("\nRun without --dry-run to apply changes")
        else:
            click.echo("\nNext steps:")
            click.echo(f"   - Review the updated config file: {config}")
            click.echo("   - Re-run analysis to see updated reports with new name")

    except KeyboardInterrupt:
        click.echo("\n\nInterrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)

        traceback.print_exc()
