"""Interactive CLI menu system for gitflow-analytics.

This module provides an interactive menu interface that appears when the tool
is run without arguments, offering options for configuration, alias management,
analysis execution, and more.
"""

import contextlib
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import click
import yaml

logger = logging.getLogger(__name__)


def _validate_subprocess_path(path: Path) -> None:
    """Validate path is safe for subprocess execution.

    Args:
        path: Path to validate

    Raises:
        ValueError: If path contains dangerous characters or traversal patterns
    """
    path_str = str(path.resolve())

    # Check for shell metacharacters
    dangerous_chars = [";", "&", "|", "$", "`", "\n", "\r", "<", ">"]
    if any(char in path_str for char in dangerous_chars):
        raise ValueError(f"Path contains invalid characters: {path}")

    # No path traversal
    if ".." in path.parts:
        raise ValueError(f"Path cannot contain '..' traversal: {path}")


def _validate_editor_command(editor: str) -> None:
    """Validate editor command is safe.

    Args:
        editor: Editor command from environment

    Raises:
        ValueError: If editor contains spaces or dangerous characters
    """
    # Editor should be a single command without arguments
    if " " in editor:
        raise ValueError(
            f"EDITOR environment variable contains spaces/arguments: {editor}. "
            "Please set EDITOR to command name only (e.g., 'vim' not 'vim -x')"
        )

    # Check for shell metacharacters
    dangerous_chars = [";", "&", "|", "$", "`", "\n", "\r", "<", ">"]
    if any(char in editor for char in dangerous_chars):
        raise ValueError(f"EDITOR contains invalid characters: {editor}")


def _atomic_yaml_write(config_path: Path, config_data: dict) -> None:
    """Atomically write YAML config to prevent corruption.

    Uses temp file + atomic rename pattern to ensure config is never
    partially written or corrupted.

    Args:
        config_path: Destination config file path
        config_data: Configuration data to write

    Raises:
        IOError: If write fails
    """
    temp_fd = None
    temp_path = None

    try:
        # Create temp file in same directory as target (required for atomic rename)
        temp_fd, temp_path_str = tempfile.mkstemp(
            dir=config_path.parent, prefix=f".{config_path.name}.", suffix=".tmp", text=True
        )

        temp_path = Path(temp_path_str)

        # Write to temp file
        with os.fdopen(temp_fd, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            temp_fd = None  # fdopen closes the fd

        # Atomic rename
        os.replace(temp_path, config_path)
        logger.debug(f"Atomically wrote config to {config_path}")

    except Exception as e:
        # Cleanup temp file on error
        if temp_fd is not None:
            with contextlib.suppress(Exception):
                os.close(temp_fd)

        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)

        raise OSError(f"Failed to write config atomically: {e}") from e


def _run_subprocess_safely(cmd: list[str], operation_name: str, timeout: int = 300) -> bool:
    """Run subprocess with comprehensive error handling.

    Args:
        cmd: Command list to execute (no shell=True)
        operation_name: Human-readable operation name for error messages
        timeout: Timeout in seconds (default: 300s for interactive operations)

    Returns:
        True if subprocess succeeded (exit code 0), False otherwise
    """
    try:
        result = subprocess.run(
            cmd,
            shell=False,
            check=False,
            timeout=timeout,
        )

        if result.returncode == 0:
            return True
        elif result.returncode < 0:
            # Process was terminated by signal
            signal_num = -result.returncode
            click.echo(
                click.style(
                    f"\n⚠️  {operation_name} was terminated (signal {signal_num})", fg="yellow"
                )
            )
            logger.warning(f"{operation_name} terminated by signal {signal_num}")
            return False
        else:
            # Process exited with non-zero code
            click.echo(
                click.style(
                    f"\n⚠️  {operation_name} exited with code {result.returncode}", fg="yellow"
                )
            )
            logger.warning(f"{operation_name} exited with code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        click.echo(
            click.style(f"\n❌ {operation_name} timed out after {timeout} seconds", fg="red"),
            err=True,
        )
        logger.error(f"{operation_name} timeout after {timeout}s")
        return False

    except FileNotFoundError as e:
        click.echo(click.style(f"\n❌ Command not found: {e}", fg="red"), err=True)
        logger.error(f"{operation_name} - command not found: {e}")
        return False

    except Exception as e:
        click.echo(click.style(f"\n❌ Error running {operation_name}: {e}", fg="red"), err=True)
        logger.error(f"{operation_name} error: {type(e).__name__}: {e}")
        return False


def find_or_prompt_config() -> Optional[Path]:
    """Find config.yaml or prompt user for path.

    Returns:
        Path to config file, or None if user cancels.
    """
    # Check for config.yaml in current directory
    cwd_config = Path.cwd() / "config.yaml"
    if cwd_config.exists():
        click.echo(
            click.style(f"✅ Found config.yaml in current directory: {cwd_config}", fg="green")
        )
        if click.confirm("Use this config?", default=True):
            return cwd_config

    # Prompt for config path
    click.echo("\n" + "=" * 60)
    click.echo(click.style("Configuration File Required", fg="yellow", bold=True))
    click.echo("=" * 60)
    click.echo("\nPlease provide the path to your config.yaml file:")
    click.echo("  • Enter absolute path: /path/to/config.yaml")
    click.echo("  • Enter relative path: ./config.yaml")
    click.echo("  • Press Ctrl+C to exit\n")

    while True:
        try:
            config_path_str = click.prompt("Config path", type=str)
            config_path = Path(config_path_str).expanduser().resolve()

            if config_path.exists() and config_path.is_file():
                return config_path
            else:
                click.echo(click.style(f"❌ File not found: {config_path}", fg="red"))
                if not click.confirm("Try again?", default=True):
                    return None
        except click.Abort:
            click.echo("\n\n👋 Exiting menu.")
            return None


def validate_config(config_path: Path) -> bool:
    """Validate configuration file with helpful error messages.

    Args:
        config_path: Path to config.yaml file

    Returns:
        True if config is valid, False otherwise.
    """
    try:
        from gitflow_analytics.config.loader import ConfigLoader

        ConfigLoader.load(config_path)
        return True

    except yaml.YAMLError as e:
        click.echo(click.style("\n❌ YAML Syntax Error", fg="red", bold=True), err=True)

        # Show location if available
        if hasattr(e, "problem_mark"):
            mark = e.problem_mark
            click.echo(f"   Location: Line {mark.line + 1}, Column {mark.column + 1}", err=True)

            # Show problematic line with context
            try:
                with open(config_path) as f:
                    lines = f.readlines()

                if 0 <= mark.line < len(lines):
                    click.echo("\n   Context:", err=True)

                    # Line before
                    if mark.line > 0:
                        click.echo(f"   {mark.line}: {lines[mark.line - 1].rstrip()}", err=True)

                    # Problematic line (highlighted)
                    click.echo(
                        click.style(f"   {mark.line + 1}: {lines[mark.line].rstrip()}", fg="red"),
                        err=True,
                    )

                    # Pointer to column
                    pointer = " " * (len(str(mark.line + 1)) + 2 + mark.column) + "^"
                    click.echo(click.style(pointer, fg="red"), err=True)

                    # Line after
                    if mark.line + 1 < len(lines):
                        click.echo(f"   {mark.line + 2}: {lines[mark.line + 1].rstrip()}", err=True)
            except Exception:
                # If we can't read file, just skip context
                pass

        # Show problem description
        if hasattr(e, "problem"):
            click.echo(f"\n   Problem: {e.problem}", err=True)

        # Add helpful tips
        click.echo("\n💡 Common YAML issues:", err=True)
        click.echo("   • Check for unmatched quotes or brackets", err=True)
        click.echo("   • Ensure proper indentation (use spaces, not tabs)", err=True)
        click.echo("   • Verify colons have space after them (key: value)", err=True)
        click.echo("   • Check for special characters that need quoting", err=True)

        logger.error(f"YAML syntax error in config: {e}")
        return False

    except Exception as e:
        click.echo(click.style(f"❌ Configuration error: {e}", fg="red"), err=True)
        logger.error(f"Config validation failed: {e}")
        return False


def edit_configuration(config_path: Path) -> bool:
    """Open config.yaml in user's editor.

    Args:
        config_path: Path to config.yaml file

    Returns:
        True if edit succeeded and config is valid, False otherwise.
    """
    # Get editor from environment, fallback to vi
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vi"))

    try:
        # Validate editor command
        _validate_editor_command(editor)
    except ValueError as e:
        click.echo(click.style(f"❌ {e}", fg="red"), err=True)
        logger.error(f"Editor validation failed: {e}")
        return False

    try:
        # Validate config path
        _validate_subprocess_path(config_path)
    except ValueError as e:
        click.echo(click.style(f"❌ Invalid config path: {e}", fg="red"), err=True)
        logger.error(f"Config path validation failed: {e}")
        return False

    click.echo(f"\n📝 Opening {config_path} with {editor}...")

    # Run editor with timeout (5 minutes default for interactive editing)
    success = _run_subprocess_safely(
        [editor, str(config_path)], operation_name="Editor", timeout=300
    )

    if not success:
        return False

    # Validate after edit
    click.echo("\n🔍 Validating configuration...")
    if validate_config(config_path):
        click.echo(click.style("✅ Configuration is valid!", fg="green"))
        return True
    else:
        click.echo(
            click.style(
                "⚠️  Configuration has errors. Please fix before running analysis.",
                fg="yellow",
            )
        )
        return False


def fix_aliases(config_path: Path) -> bool:
    """Launch interactive alias creator.

    Args:
        config_path: Path to config.yaml file

    Returns:
        True if alias creation succeeded, False otherwise.
    """
    click.echo(
        "\n" + click.style("🔧 Launching Interactive Alias Creator...", fg="cyan", bold=True) + "\n"
    )

    try:
        # Validate config path
        _validate_subprocess_path(config_path)
    except ValueError as e:
        click.echo(click.style(f"❌ Invalid config path: {e}", fg="red"), err=True)
        logger.error(f"Config path validation failed: {e}")
        return False

    cmd = [
        sys.executable,
        "-m",
        "gitflow_analytics.cli",
        "create-alias-interactive",
        "-c",
        str(config_path),
    ]

    # Run with 5 minute timeout for interactive session
    success = _run_subprocess_safely(cmd, operation_name="Alias creator", timeout=300)

    if success:
        click.echo(click.style("\n✅ Alias creation completed!", fg="green"))

    return success


def get_current_weeks(config_path: Path) -> int:
    """Get current weeks setting from config.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Current weeks setting, or 12 as default.
    """
    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        return config_data.get("analysis", {}).get("weeks_back", 12)
    except Exception as e:
        logger.warning(f"Could not read weeks from config: {e}")
        return 12


def repull_data(config_path: Path) -> bool:
    """Re-run analysis with optional cache clear.

    Args:
        config_path: Path to config.yaml file

    Returns:
        True if analysis succeeded, False otherwise.
    """
    click.echo("\n" + "=" * 60)
    click.echo(click.style("Re-pull Data (Re-run Analysis)", fg="cyan", bold=True))
    click.echo("=" * 60 + "\n")

    # Ask about clearing cache
    clear_cache = click.confirm("🗑️  Clear cache before re-pull?", default=True)

    # Ask about number of weeks
    current_weeks = get_current_weeks(config_path)
    use_current = click.confirm(f"📅 Use current setting ({current_weeks} weeks)?", default=True)

    if use_current:
        weeks = current_weeks
    else:
        weeks = click.prompt(
            "Number of weeks to analyze",
            type=click.IntRange(1, 52),
            default=current_weeks,
        )

    try:
        # Validate config path
        _validate_subprocess_path(config_path)
    except ValueError as e:
        click.echo(click.style(f"❌ Invalid config path: {e}", fg="red"), err=True)
        logger.error(f"Config path validation failed: {e}")
        return False

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "gitflow_analytics.cli",
        "analyze",
        "-c",
        str(config_path),
        "--weeks",
        str(weeks),
    ]

    if clear_cache:
        cmd.append("--clear-cache")

    # Display what will be run
    click.echo("\n🚀 Running analysis...")
    click.echo(f"   Config: {config_path}")
    click.echo(f"   Weeks: {weeks}")
    click.echo(f"   Clear cache: {'Yes' if clear_cache else 'No'}\n")

    # Run with 10 minute timeout for analysis
    success = _run_subprocess_safely(cmd, operation_name="Analysis", timeout=600)

    if success:
        click.echo(click.style("\n✅ Analysis completed successfully!", fg="green"))

    return success


def set_weeks(config_path: Path) -> bool:
    """Update analysis.weeks_back in config.

    Args:
        config_path: Path to config.yaml file

    Returns:
        True if config update succeeded, False otherwise.
    """
    click.echo("\n" + "=" * 60)
    click.echo(click.style("Set Number of Weeks", fg="cyan", bold=True))
    click.echo("=" * 60 + "\n")

    try:
        # Load current config
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Get current value
        current = config_data.get("analysis", {}).get("weeks_back", 12)
        click.echo(f"Current setting: {current} weeks\n")

        # Prompt for new value
        weeks = click.prompt(
            "Number of weeks to analyze",
            type=click.IntRange(1, 52),
            default=current,
        )

        # Update config
        if "analysis" not in config_data:
            config_data["analysis"] = {}
        config_data["analysis"]["weeks_back"] = weeks

        # Write back to file atomically
        _atomic_yaml_write(config_path, config_data)

        click.echo(click.style(f"\n✅ Set to {weeks} weeks", fg="green"))
        click.echo(f"💾 Saved to {config_path}")

        # Validate config after modification
        if not validate_config(config_path):
            click.echo(
                click.style(
                    "\n⚠️  Warning: Config may have validation issues after update.",
                    fg="yellow",
                )
            )
            return False

        return True

    except Exception as e:
        click.echo(click.style(f"\n❌ Error updating config: {e}", fg="red"), err=True)
        logger.error(f"Config update error: {type(e).__name__}: {e}")
        return False


def run_full_analysis(config_path: Path) -> bool:
    """Launch full analysis with current config settings.

    Args:
        config_path: Path to config.yaml file

    Returns:
        True if analysis succeeded, False otherwise.
    """
    click.echo("\n" + "=" * 60)
    click.echo(click.style("Run Full Analysis", fg="cyan", bold=True))
    click.echo("=" * 60 + "\n")

    # Get current weeks setting
    weeks = get_current_weeks(config_path)
    click.echo("📊 Running analysis with current settings:")
    click.echo(f"   Config: {config_path}")
    click.echo(f"   Weeks: {weeks}\n")

    try:
        # Validate config path
        _validate_subprocess_path(config_path)
    except ValueError as e:
        click.echo(click.style(f"❌ Invalid config path: {e}", fg="red"), err=True)
        logger.error(f"Config path validation failed: {e}")
        return False

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "gitflow_analytics.cli",
        "analyze",
        "-c",
        str(config_path),
    ]

    # Run with 10 minute timeout for analysis
    success = _run_subprocess_safely(cmd, operation_name="Analysis", timeout=600)

    if success:
        click.echo(click.style("\n✅ Analysis completed successfully!", fg="green"))

    return success


def rename_developer_alias(config_path: Path) -> bool:
    """Interactive interface for renaming developer aliases.

    Args:
        config_path: Path to config.yaml file

    Returns:
        True if rename succeeded, False otherwise.
    """
    click.echo("\n" + "=" * 60)
    click.echo(click.style("Rename Developer Alias", fg="cyan", bold=True))
    click.echo("=" * 60 + "\n")

    click.echo("Update a developer's canonical display name in reports.")
    click.echo("This updates the configuration file and optionally the cache.\n")

    try:
        # Load config to get manual_mappings
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Navigate to manual_mappings
        manual_mappings = (
            config_data.get("analysis", {}).get("identity", {}).get("manual_mappings", [])
        )

        if not manual_mappings:
            click.echo(
                click.style(
                    "❌ No manual_mappings found in config. Please add developers first.", fg="red"
                ),
                err=True,
            )
            return False

        # Display numbered list of developers
        click.echo(click.style("Current Developers:", fg="cyan", bold=True))
        click.echo()

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

        # Prompt for selection
        try:
            selection = click.prompt(
                "Select developer number to rename (or 0 to cancel)",
                type=click.IntRange(0, len(developer_names)),
            )
        except click.Abort:
            click.echo(click.style("\n❌ Cancelled", fg="yellow"))
            return False

        if selection == 0:
            click.echo(click.style("\n❌ Cancelled", fg="yellow"))
            return False

        # Get selected developer name
        old_name = developer_names[selection - 1]
        click.echo(f"\n📝 Selected: {click.style(old_name, fg='green')}")

        # Prompt for new name
        new_name = click.prompt("Enter new canonical name", type=str)

        # Validate new name
        new_name = new_name.strip()
        if not new_name:
            click.echo(click.style("❌ New name cannot be empty", fg="red"), err=True)
            return False

        if new_name == old_name:
            click.echo(click.style("❌ New name is identical to current name", fg="yellow"))
            return False

        # Ask about cache update
        update_cache = click.confirm("\nAlso update database cache?", default=True)

        # Show what will be done
        click.echo("\n" + "=" * 60)
        click.echo(click.style("Summary", fg="yellow", bold=True))
        click.echo("=" * 60)
        click.echo(f"  Old name: {old_name}")
        click.echo(f"  New name: {new_name}")
        click.echo(f"  Update cache: {'Yes' if update_cache else 'No'}")
        click.echo()

        # Confirm
        if not click.confirm("Proceed with rename?", default=True):
            click.echo(click.style("\n❌ Cancelled", fg="yellow"))
            return False

    except Exception as e:
        click.echo(click.style(f"❌ Error reading config: {e}", fg="red"), err=True)
        logger.error(f"Config read error: {type(e).__name__}: {e}")
        return False

    try:
        # Validate config path
        _validate_subprocess_path(config_path)
    except ValueError as e:
        click.echo(click.style(f"❌ Invalid config path: {e}", fg="red"), err=True)
        logger.error(f"Config path validation failed: {e}")
        return False

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "gitflow_analytics.cli",
        "alias-rename",
        "-c",
        str(config_path),
        "--old-name",
        old_name,
        "--new-name",
        new_name,
    ]

    if update_cache:
        cmd.append("--update-cache")

    # Run with timeout
    success = _run_subprocess_safely(cmd, operation_name="Alias Rename", timeout=60)

    if success:
        click.echo(click.style("\n✅ Rename completed successfully!", fg="green"))
        click.echo(f"Future reports will show '{new_name}' instead of '{old_name}'")

    return success


def show_main_menu(config_path: Optional[Path] = None) -> None:
    """Display main interactive menu.

    Args:
        config_path: Optional path to config.yaml file. If not provided,
                    will attempt to find or prompt for it.
    """
    # If no config, find or prompt for it
    if not config_path:
        config_path = find_or_prompt_config()
        if not config_path:
            click.echo(click.style("\n❌ No config file provided. Exiting.", fg="red"))
            sys.exit(1)

    # Validate config exists
    if not config_path.exists():
        click.echo(click.style(f"\n❌ Config file not found: {config_path}", fg="red"))
        sys.exit(1)

    # Main menu loop
    while True:
        try:
            # Display menu header
            click.echo("\n" + "=" * 60)
            click.echo(click.style("GitFlow Analytics - Interactive Menu", fg="cyan", bold=True))
            click.echo("=" * 60)
            click.echo(f"\nConfig: {click.style(str(config_path), fg='green')}\n")

            # Display menu options
            click.echo(click.style("Choose an option:", fg="white", bold=True))
            click.echo("  1. Edit Configuration")
            click.echo("  2. Fix Developer Aliases")
            click.echo("  3. Re-pull Data (Re-run Analysis)")
            click.echo("  4. Set Number of Weeks")
            click.echo("  5. Run Full Analysis")
            click.echo("  6. Rename Developer Alias")
            click.echo("  0. Exit")

            # Get user choice
            click.echo()
            choice = click.prompt(
                click.style("Enter your choice", fg="yellow"),
                type=click.Choice(["0", "1", "2", "3", "4", "5", "6"], case_sensitive=False),
                show_choices=False,
            )

            # Handle choice
            success = True

            if choice == "0":
                click.echo(click.style("\n👋 Goodbye!", fg="green"))
                break
            elif choice == "1":
                success = edit_configuration(config_path)
            elif choice == "2":
                success = fix_aliases(config_path)
            elif choice == "3":
                success = repull_data(config_path)
            elif choice == "4":
                success = set_weeks(config_path)
            elif choice == "5":
                success = run_full_analysis(config_path)
            elif choice == "6":
                success = rename_developer_alias(config_path)

            # Show warning if operation failed
            if not success and choice != "0":
                click.echo(
                    click.style(
                        "\n⚠️  Operation did not complete successfully. Check messages above.",
                        fg="yellow",
                    )
                )

            # Pause before showing menu again
            if choice != "0":
                click.echo()
                click.prompt(
                    click.style("Press Enter to continue", fg="cyan"),
                    default="",
                    show_default=False,
                )

        except click.Abort:
            click.echo(click.style("\n\n👋 Interrupted. Exiting.", fg="yellow"))
            sys.exit(0)
        except KeyboardInterrupt:
            click.echo(click.style("\n\n👋 Interrupted. Exiting.", fg="yellow"))
            sys.exit(0)
        except Exception as e:
            click.echo(click.style(f"\n❌ Error: {e}", fg="red"), err=True)
            logger.error(f"Menu error: {type(e).__name__}: {e}")

            # Ask if user wants to continue
            if not click.confirm("Continue?", default=True):
                sys.exit(1)
