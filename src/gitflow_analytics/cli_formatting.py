"""CLI formatting and error handling helpers for GitFlow Analytics.

Contains presentation-layer utilities used by the Click CLI to produce
friendly help text and helpful error suggestions.
"""

import logging
import re
import traceback
from difflib import get_close_matches
from typing import Optional

import click


class RichHelpFormatter:
    """Rich help formatter for enhanced CLI help display."""

    @staticmethod
    def format_command_help(command: str, description: str, examples: list[str]) -> str:
        """Format command help with examples."""
        help_text = description
        if examples:
            help_text += "\n\nExamples:\n"
            for example in examples:
                help_text += f"  {example}\n"
        return help_text

    @staticmethod
    def format_option_help(
        description: str, default: Optional[str] = None, choices: Optional[list[str]] = None
    ) -> str:
        """Format option help with default and choices.

        Args:
            description: Option description text
            default: Default value to display (optional)
            choices: List of valid choices (optional)

        Returns:
            Formatted help text string
        """
        help_text = description
        if default is not None:
            help_text += f" [default: {default}]"
        if choices:
            help_text += f" [choices: {', '.join(map(str, choices))}]"
        return help_text

    @staticmethod
    def suggest_command(invalid_cmd: str, available_cmds: list[str]) -> str:
        """Suggest similar commands for typos."""
        matches = get_close_matches(invalid_cmd, available_cmds, n=3, cutoff=0.6)
        if matches:
            return f"Did you mean: {', '.join(matches)}?"
        return ""


class ImprovedErrorHandler:
    """Enhanced error handling with helpful suggestions."""

    @staticmethod
    def handle_command_error(ctx: click.Context, error: Exception) -> None:
        """Handle command errors with helpful suggestions."""
        error_msg = str(error)

        # Check for common errors and provide suggestions
        if "no such option" in error_msg.lower():
            # Extract the invalid option
            match = re.search(r"no such option: (--?[\w-]+)", error_msg.lower())
            if match:
                invalid_option = match.group(1)
                available_options = [
                    "--config",
                    "--weeks",
                    "--output",
                    "--format",
                    "--clear-cache",
                    "--validate-only",
                    "--anonymize",
                    "--help",
                ]
                suggestion = RichHelpFormatter.suggest_command(invalid_option, available_options)
                if suggestion:
                    click.echo(f"\n❗ {error_msg}", err=True)
                    click.echo(f"\n💡 {suggestion}", err=True)
                    click.echo("\nUse 'gitflow-analytics --help' for available options.", err=True)
                    return

        elif "no such command" in error_msg.lower():
            # Extract the invalid command
            match = re.search(r"no such command[:'] (\w+)", error_msg.lower())
            if match:
                invalid_cmd = match.group(1)
                available_cmds = [
                    "analyze",
                    "collect",
                    "classify",
                    "report",
                    "fetch",
                    "identities",
                    "train",
                    "cache-stats",
                    "list-developers",
                    "merge-identity",
                    "help",
                    "train-stats",
                ]
                suggestion = RichHelpFormatter.suggest_command(invalid_cmd, available_cmds)
                if suggestion:
                    click.echo(f"\n❗ Unknown command: '{invalid_cmd}'", err=True)
                    click.echo(f"\n💡 {suggestion}", err=True)
                    click.echo("\nAvailable commands:", err=True)
                    for cmd in available_cmds[:5]:  # Show first 5
                        click.echo(f"  • {cmd}", err=True)
                    click.echo("\nUse 'gitflow-analytics help' for more information.", err=True)
                    return

        elif "file not found" in error_msg.lower() or "no such file" in error_msg.lower():
            click.echo(f"\n❗ {error_msg}", err=True)
            click.echo("\n💡 Suggestions:", err=True)
            click.echo("  • Check if the file path is correct", err=True)
            click.echo("  • Use absolute paths for clarity", err=True)
            click.echo("  • Create a config file: cp config-sample.yaml myconfig.yaml", err=True)
            return

        elif "permission denied" in error_msg.lower():
            click.echo(f"\n❗ {error_msg}", err=True)
            click.echo("\n💡 Suggestions:", err=True)
            click.echo("  • Check file/directory permissions", err=True)
            click.echo("  • Ensure you have read access to repositories", err=True)
            click.echo("  • Try running with appropriate user permissions", err=True)
            return

        elif "git repository" in error_msg.lower():
            click.echo(f"\n❗ {error_msg}", err=True)
            click.echo("\n💡 Suggestions:", err=True)
            click.echo("  • Verify repository paths in configuration", err=True)
            click.echo("  • Ensure repositories are cloned locally", err=True)
            click.echo("  • Check that .git directory exists", err=True)
            return

        # Default error display
        click.echo(f"\n❗ Error: {error_msg}", err=True)
        click.echo("\nFor help: gitflow-analytics help", err=True)


def handle_timezone_error(
    e: Exception, report_name: str, all_commits: list, logger: logging.Logger
) -> None:
    """Handle timezone comparison errors with detailed logging."""
    if isinstance(e, TypeError) and (
        "can't compare" in str(e).lower() or "timezone" in str(e).lower()
    ):
        logger.error(f"Timezone comparison error in {report_name}:")
        logger.error(f"  Error: {e}")

        logger.error(f"  Full traceback:\n{traceback.format_exc()}")

        # Log context information
        sample_commits = all_commits[:5] if all_commits else []
        for i, commit in enumerate(sample_commits):
            timestamp = commit.get("timestamp")
            logger.error(
                f"  Sample commit {i}: timestamp={timestamp} "
                f"(tzinfo: {getattr(timestamp, 'tzinfo', 'N/A')})"
            )

        click.echo(f"   ❌ Timezone comparison error in {report_name}")
        click.echo("   🔍 See logs with --log DEBUG for detailed information")
        click.echo("   💡 This usually indicates mixed timezone-aware and naive datetime objects")
        raise
    else:
        # Re-raise other errors
        raise
