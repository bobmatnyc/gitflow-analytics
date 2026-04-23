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

        identity_resolver = DeveloperIdentityResolver(str(cfg.cache.directory / "identities.db"))
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
                            click.style("You must select at least 2 developers to merge", fg="red")
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

                    click.echo(click.style("Database updated with merged identities", fg="green"))

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

        # Bug #30 fix: Also support the legacy top-level 'developer_aliases' config key.
        # The rest of the tool (see ConfigLoader) auto-migrates this on load, but
        # alias-rename reads/writes YAML directly, so we need to handle the legacy
        # format here as well.  When we detect it we convert it in-memory to the
        # modern 'analysis.identity.manual_mappings' layout and, on save, the
        # legacy key is removed — effectively migrating the user to the new format.
        migrated_from_legacy = False
        if (
            "developer_aliases" in config_data
            and isinstance(config_data.get("developer_aliases"), dict)
            and (
                "analysis" not in config_data
                or "identity" not in (config_data.get("analysis") or {})
                or "manual_mappings"
                not in ((config_data.get("analysis") or {}).get("identity") or {})
            )
        ):
            click.echo(
                "Detected legacy 'developer_aliases' configuration — "
                "migrating to 'analysis.identity.manual_mappings'."
            )
            legacy_aliases = config_data["developer_aliases"]

            config_data.setdefault("analysis", {})
            config_data["analysis"].setdefault("identity", {})
            config_data["analysis"]["identity"].setdefault("manual_mappings", [])

            # Mirror the conversion in ConfigLoader so behaviour is consistent.
            for canonical_name, emails in legacy_aliases.items():
                if isinstance(emails, list) and emails:
                    primary_email = next((e for e in emails if "@" in e), emails[0])
                    config_data["analysis"]["identity"]["manual_mappings"].append(
                        {
                            "name": canonical_name,
                            "primary_email": primary_email,
                            "aliases": list(emails),
                        }
                    )

            del config_data["developer_aliases"]
            migrated_from_legacy = True

        if "analysis" not in config_data:
            click.echo("Error: 'analysis' section not found in config", err=True)
            sys.exit(1)

        if "identity" not in config_data["analysis"]:
            click.echo("Error: 'analysis.identity' section not found in config", err=True)
            sys.exit(1)

        if "manual_mappings" not in config_data["analysis"]["identity"]:
            click.echo("Error: 'analysis.identity.manual_mappings' not found in config", err=True)
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
                if migrated_from_legacy:
                    click.echo(
                        "Migrated legacy 'developer_aliases' to 'analysis.identity.manual_mappings'"
                    )
            except Exception as e:
                click.echo(f"Error writing config file: {e}", err=True)
                sys.exit(1)
        else:
            click.echo(f"   [Would update config: {config}]")
            if migrated_from_legacy:
                click.echo(
                    "   [Would migrate legacy 'developer_aliases' to "
                    "'analysis.identity.manual_mappings']"
                )

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


def _load_alias_file(file_path: str) -> list[dict]:
    """Load alias mappings from YAML or JSON file.

    Accepts two formats:

    Format A (GFA aliases.yaml format):
      developer_aliases:
        - primary_email: john@company.com
          aliases: [john@gmail.com]
          name: John Doe

    Format B (flat list):
      - canonical: john@company.com
        aliases: [john@gmail.com]
        name: John Doe
    """
    import json

    path = Path(file_path)
    with open(path) as f:
        data = json.load(f) if path.suffix.lower() == ".json" else yaml.safe_load(f)

    if isinstance(data, dict) and "developer_aliases" in data:
        # GFA aliases.yaml format
        entries = data["developer_aliases"]
        result = []
        for entry in entries if isinstance(entries, list) else []:
            canonical = (
                entry.get("primary_email")
                or entry.get("canonical_email")
                or entry.get("canonical", "")
            )
            if canonical:
                result.append(
                    {
                        "canonical": canonical,
                        "aliases": entry.get("aliases", []),
                        "name": entry.get("name"),
                    }
                )
        return result
    elif isinstance(data, list):
        # Flat list format
        return [
            {
                "canonical": (
                    e.get("canonical") or e.get("primary_email") or e.get("canonical_email", "")
                ),
                "aliases": e.get("aliases", []),
                "name": e.get("name"),
            }
            for e in data
            if isinstance(e, dict)
        ]
    else:
        raise click.ClickException(
            f"Unrecognised format in {file_path}. Expected 'developer_aliases' list or flat list."
        )


def _migrate_legacy_developer_aliases(config_data: dict) -> bool:
    """Migrate top-level ``developer_aliases`` key into ``analysis.identity.manual_mappings``.

    Mirrors the migration logic in :mod:`config.loader` and :func:`alias_rename`
    so that ``add-alias`` can safely operate on configs that still use the
    legacy format. Mutates ``config_data`` in place and returns ``True`` when
    a migration occurred.

    Only migrates when the modern location is NOT already populated to avoid
    clobbering hand-maintained mappings.
    """
    legacy = config_data.get("developer_aliases")
    if not isinstance(legacy, dict) or not legacy:
        return False

    modern_present = (
        isinstance(config_data.get("analysis"), dict)
        and isinstance((config_data.get("analysis") or {}).get("identity"), dict)
        and "manual_mappings" in (config_data["analysis"].get("identity") or {})
    )

    if modern_present:
        # Modern key already set — just drop the legacy key to avoid confusion.
        del config_data["developer_aliases"]
        return True

    config_data.setdefault("analysis", {})
    config_data["analysis"].setdefault("identity", {})
    config_data["analysis"]["identity"].setdefault("manual_mappings", [])

    for canonical_name, emails in legacy.items():
        if isinstance(emails, list) and emails:
            primary_email = next((e for e in emails if "@" in e), emails[0])
            config_data["analysis"]["identity"]["manual_mappings"].append(
                {
                    "name": canonical_name,
                    "primary_email": primary_email,
                    "aliases": list(emails),
                }
            )

    del config_data["developer_aliases"]
    return True


def _resolve_cache_dir(config_path: Path, config_data: dict) -> Optional[Path]:
    """Figure out the cache directory without requiring a fully valid config.

    Tries (in order):
      1. ``cache.directory`` from the parsed YAML, resolved relative to the
         config file when the path is not absolute.
      2. ``ConfigLoader.load(config_path).cache.directory`` — the authoritative
         path but needs a valid config (repositories / org defined).
      3. Returns ``None`` when neither succeeds.

    Returning ``None`` lets callers emit a friendly warning rather than crash
    out when run against a skeleton/in-progress config.
    """
    cache_section = config_data.get("cache")
    if isinstance(cache_section, dict):
        directory = cache_section.get("directory")
        if directory:
            cache_path = Path(directory).expanduser()
            if not cache_path.is_absolute():
                cache_path = (config_path.parent / cache_path).resolve()
            return cache_path

    try:
        cfg = ConfigLoader.load(config_path)
    except Exception:
        return None
    return cfg.cache.directory


def _apply_aliases_to_identity_db(
    config_path: Path,
    config_data: dict,
    mappings: list[dict],
    dry_run: bool,
) -> None:
    """Propagate alias mappings to the ``developer_identities`` cache DB.

    For each canonical/alias pair:
    - Locate the :class:`DeveloperIdentity` rows matching the alias email (or name)
    - Merge them into the canonical identity via
      :meth:`DeveloperIdentityResolver.merge_identities` so subsequent reports
      reflect the unified identity without requiring a re-collect.
    - Set ``primary_name`` on the canonical row when an explicit name was supplied.

    Aliases that don't match any cached row are reported as "not in cache"
    (no-op) rather than treated as errors.
    """
    from sqlalchemy import text

    from .core.identity import DeveloperIdentityResolver

    cache_dir = _resolve_cache_dir(config_path, config_data)
    if cache_dir is None:
        click.echo(
            click.style(
                "Warning: could not resolve cache directory from config; skipping cache update",
                fg="yellow",
            )
        )
        return

    identity_db_path = cache_dir / "identities.db"
    if not identity_db_path.exists():
        click.echo(
            f"Note: identity database not found at {identity_db_path} — skipping cache update"
        )
        return

    resolver = DeveloperIdentityResolver(str(identity_db_path), manual_mappings=None)

    total_merged = 0
    total_renamed = 0
    total_missing = 0

    for mapping in mappings:
        canonical_email = (mapping.get("canonical") or "").strip().lower()
        if not canonical_email:
            continue
        alias_values = [a.strip() for a in mapping.get("aliases", []) if a.strip()]
        explicit_name = mapping.get("name")

        # Look up the canonical identity row (case-insensitive on email).
        with resolver.get_session() as session:
            canonical_row = session.execute(
                text(
                    "SELECT canonical_id, primary_name FROM developer_identities "
                    "WHERE LOWER(primary_email) = :email"
                ),
                {"email": canonical_email},
            ).fetchone()

        if canonical_row is None:
            total_missing += 1
            click.echo(
                f"   [cache] canonical '{canonical_email}' not in DB — skipping "
                "(will be created on next analysis run)"
            )
            continue

        canonical_id = canonical_row[0]

        for alias_value in alias_values:
            alias_lower = alias_value.lower()
            with resolver.get_session() as session:
                if "@" in alias_value:
                    alias_row = session.execute(
                        text(
                            "SELECT canonical_id FROM developer_identities "
                            "WHERE LOWER(primary_email) = :email"
                        ),
                        {"email": alias_lower},
                    ).fetchone()
                else:
                    alias_row = session.execute(
                        text(
                            "SELECT canonical_id FROM developer_identities "
                            "WHERE LOWER(primary_name) = :name"
                        ),
                        {"name": alias_lower},
                    ).fetchone()

            if alias_row is None:
                # Not yet in cache — nothing to merge. This is fine; future
                # analyses will pick up the mapping via manual_mappings.
                continue

            alias_canonical_id = alias_row[0]
            if alias_canonical_id == canonical_id:
                # Already merged — idempotent no-op.
                continue

            if dry_run:
                click.echo(
                    f"   [cache/dry-run] Would merge '{alias_value}' into '{canonical_email}'"
                )
                continue

            try:
                resolver.merge_identities(canonical_id, alias_canonical_id)
                total_merged += 1
            except ValueError as e:
                # merge_identities raises if rows disappear mid-flight.
                click.echo(
                    click.style(
                        f"   [cache] merge skipped for '{alias_value}' ({e})",
                        fg="yellow",
                    )
                )

        # Optionally update primary_name when caller supplied one.
        if explicit_name and not dry_run:
            with resolver.get_session() as session:
                result = session.execute(
                    text(
                        "UPDATE developer_identities SET primary_name = :name "
                        "WHERE canonical_id = :cid AND primary_name != :name"
                    ),
                    {"name": explicit_name, "cid": canonical_id},
                )
                # SQLAlchemy's rowcount is only reliable for UPDATE here.
                if getattr(result, "rowcount", 0):
                    total_renamed += 1
        elif explicit_name and dry_run:
            click.echo(
                f"   [cache/dry-run] Would set primary_name='{explicit_name}' "
                f"for '{canonical_email}'"
            )

    if total_merged or total_renamed or total_missing:
        click.echo(
            f"   [cache] merged {total_merged} identity rows, "
            f"renamed {total_renamed}, {total_missing} canonical(s) not in cache"
        )
    else:
        click.echo("   [cache] no cache changes needed (nothing to merge)")


@click.command(name="add-alias")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML configuration file",
)
@click.option(
    "--canonical",
    required=False,
    default=None,
    help="Primary/canonical email for this developer identity",
)
@click.option(
    "--alias",
    "aliases",
    multiple=True,
    help="Email or name to map to canonical (repeatable, e.g. --alias old@co.com --alias 'Jane Smith')",
)
@click.option(
    "--name",
    "name",
    required=False,
    default=None,
    help="Optional display name for the canonical developer (sets primary_name in cache DB)",
)
@click.option(
    "--from-file",
    "from_file",
    required=False,
    default=None,
    type=click.Path(exists=True),
    help="YAML or JSON file with batch alias mappings (see format below)",
)
@click.option(
    "--apply/--no-apply",
    "apply",
    default=True,
    help=(
        "Also update the developer_identities cache DB so the change takes "
        "effect immediately (default: enabled). Use --no-apply to only "
        "modify the config YAML."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be changed without writing to config or cache",
)
def add_alias_command(
    config: str,
    canonical: Optional[str],
    aliases: tuple[str, ...],
    name: Optional[str],
    from_file: Optional[str],
    apply: bool,
    dry_run: bool,
) -> None:
    """Add alias mappings to config non-interactively.

    Maps one or more alias emails/names to a canonical developer identity
    in analysis.identity.manual_mappings. Supports single mappings via
    --canonical/--alias flags, or bulk import via --from-file. Writes are
    idempotent: re-adding an alias that already exists is a no-op.

    \b
    USAGE MODES:

      Single mapping (--canonical + --alias):
        Adds or updates one developer's alias list. Multiple --alias
        flags may be provided to add several aliases at once.

      Batch import (--from-file):
        Reads a YAML or JSON file containing multiple mappings.
        Mutually exclusive with --canonical.

    \b
    FILE FORMATS (--from-file):

      GFA native YAML (developer_aliases list):
        developer_aliases:
          - canonical: john@work.com
            aliases:
              - john@gmail.com
              - "John Doe"
          - canonical: jane@corp.com
            aliases:
              - jane@personal.com

      Flat YAML list:
        - canonical: john@work.com
          aliases: [john@gmail.com]

      JSON array:
        [{"canonical": "john@work.com", "aliases": ["john@gmail.com"]}]

    \b
    EXAMPLES:

      # Add single alias (updates config AND cache DB by default)
      gfa add-alias -c config.yaml --canonical john@work.com --alias john@gmail.com

      # Add multiple aliases with an explicit display name
      gfa add-alias -c config.yaml --canonical john@work.com \\
          --alias john@gmail.com --alias "John Doe" --name "John Doe"

      # Preview without writing anything
      gfa add-alias -c config.yaml --canonical john@work.com \\
          --alias john@gmail.com --dry-run

      # Batch import from file, skip cache update
      gfa add-alias -c config.yaml --from-file aliases.yaml --no-apply

    \b
    NOTE:
      Existing canonical entries are merged (not replaced). Duplicate
      aliases are silently skipped. Legacy top-level 'developer_aliases'
      configs are auto-migrated to 'analysis.identity.manual_mappings'
      on save. Use --dry-run to preview changes.
    """
    # Validate args
    if not from_file and not canonical:
        raise click.UsageError("Either --canonical + --alias or --from-file is required")
    if from_file and canonical:
        raise click.UsageError("--from-file and --canonical are mutually exclusive")
    if canonical and not aliases:
        raise click.UsageError("--canonical requires at least one --alias")

    # Build list of mappings to add
    if from_file:
        mappings_to_add = _load_alias_file(from_file)
    else:
        single_mapping: dict = {"canonical": canonical, "aliases": list(aliases)}
        if name:
            single_mapping["name"] = name
        mappings_to_add = [single_mapping]

    # Load config YAML
    with open(config) as f:
        config_data = yaml.safe_load(f) or {}

    # Legacy handling: auto-migrate top-level ``developer_aliases`` into the
    # modern ``analysis.identity.manual_mappings`` layout. Mirrors the
    # ConfigLoader behaviour so scripted users aren't surprised.
    migrated_from_legacy = _migrate_legacy_developer_aliases(config_data)
    if migrated_from_legacy:
        click.echo(
            "Detected legacy 'developer_aliases' — migrated to 'analysis.identity.manual_mappings'."
        )

    # Ensure path exists
    config_data.setdefault("analysis", {})
    config_data["analysis"].setdefault("identity", {})
    config_data["analysis"]["identity"].setdefault("manual_mappings", [])
    current_mappings = config_data["analysis"]["identity"]["manual_mappings"]

    added: list[tuple[str, list[str]]] = []
    skipped: list[str] = []

    for mapping in mappings_to_add:
        canonical_email = mapping["canonical"].strip().lower()
        alias_list = [a.strip() for a in mapping.get("aliases", []) if a.strip()]

        # Find existing mapping for this canonical
        existing = next(
            (m for m in current_mappings if m.get("primary_email", "").lower() == canonical_email),
            None,
        )

        if existing is None:
            # Create new mapping
            new_mapping: dict = {
                "primary_email": canonical_email,
                "aliases": alias_list,
            }
            if mapping.get("name"):
                new_mapping["name"] = mapping["name"]
            if not dry_run:
                current_mappings.append(new_mapping)
            added.append((canonical_email, alias_list))
        else:
            # Merge aliases into existing mapping (idempotent)
            existing_aliases = [a.lower() for a in existing.get("aliases", [])]
            new_aliases = [a for a in alias_list if a.lower() not in existing_aliases]
            # Also update the display name when a new one was provided.
            if mapping.get("name") and not dry_run:
                existing["name"] = mapping["name"]
            if new_aliases:
                if not dry_run:
                    existing.setdefault("aliases", []).extend(new_aliases)
                added.append((canonical_email, new_aliases))
            else:
                skipped.append(canonical_email)

    # Write back to config when either we added something OR we performed a
    # legacy migration (so the user is moved off the deprecated format).
    should_write = (not dry_run) and (added or migrated_from_legacy)
    if should_write:
        with open(config, "w", encoding="utf-8") as f:
            yaml.dump(
                config_data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    # Report
    for canonical_email, alias_list in added:
        action = "[DRY RUN] Would add" if dry_run else "Added"
        click.echo(f"{action}: {canonical_email} <- {', '.join(alias_list)}")
    for c in skipped:
        click.echo(f"Skipped (already exists): {c}")

    if not added and not skipped:
        click.echo("No changes made.")
    elif not dry_run:
        click.echo(f"\nConfig updated: {config}")

    # Optionally propagate to the cache DB so subsequent reports reflect the
    # change without a re-collect. Only attempted when at least one mapping
    # actually changed config (or we're in dry-run mode to preview).
    if apply and (added or dry_run):
        click.echo(
            "\nApplying changes to developer_identities cache DB..."
            if not dry_run
            else "\n[DRY RUN] Cache DB changes that would be applied:"
        )
        _apply_aliases_to_identity_db(Path(config), config_data, mappings_to_add, dry_run=dry_run)
