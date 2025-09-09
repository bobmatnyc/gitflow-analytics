"""Command-line interface for GitFlow Analytics."""

import builtins
import contextlib
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Optional, cast

import click
import git
import pandas as pd
import yaml

from ._version import __version__
from .cli_rich import create_rich_display
from .config import ConfigLoader
from .core.analyzer import GitAnalyzer
from .core.cache import GitAnalysisCache
from .core.identity import DeveloperIdentityResolver
from .integrations.orchestrator import IntegrationOrchestrator
from .metrics.dora import DORAMetricsCalculator
from .reports.analytics_writer import AnalyticsReportGenerator
from .reports.csv_writer import CSVReportGenerator
from .reports.json_exporter import ComprehensiveJSONExporter
from .reports.narrative_writer import NarrativeReportGenerator
from .reports.weekly_trends_writer import WeeklyTrendsWriter
from .training.pipeline import CommitClassificationTrainer


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
    def format_option_help(description: str, default: Any = None, choices: list = None) -> str:
        """Format option help with default and choices."""
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


def get_week_start(date: datetime) -> datetime:
    """Get Monday of the week for a given date, ensuring week boundary alignment.

    WHY: This provides consistent week boundary calculation across CLI date calculation
    and report generation, ensuring that filenames and week displays align properly.

    DESIGN DECISION: Always returns Monday 00:00:00 UTC as the week start to ensure
    consistent week boundaries regardless of the input date's day of week or time.

    Args:
        date: Input date (timezone-aware or naive)

    Returns:
        Monday of the week containing the input date, as timezone-aware UTC datetime
        at 00:00:00 (start of day)
    """
    # Ensure timezone consistency - convert to UTC if needed
    if hasattr(date, "tzinfo") and date.tzinfo is not None:
        # Keep timezone-aware but ensure it's UTC
        if date.tzinfo != timezone.utc:
            date = date.astimezone(timezone.utc)
    else:
        # Convert naive datetime to UTC timezone-aware
        date = date.replace(tzinfo=timezone.utc)

    # Get days since Monday (0=Monday, 6=Sunday)
    days_since_monday = date.weekday()

    # Calculate Monday of this week
    monday = date - timedelta(days=days_since_monday)

    # Reset to start of day (00:00:00)
    result = monday.replace(hour=0, minute=0, second=0, microsecond=0)

    return result


def get_week_end(date: datetime) -> datetime:
    """Get Sunday end of the week for a given date.

    WHY: Provides the end boundary for week ranges to ensure complete week coverage
    in analysis periods and consistent date range calculations.

    Args:
        date: Input date (timezone-aware or naive)

    Returns:
        Sunday 23:59:59.999999 UTC of the week containing the input date
    """
    # Get the Monday start of this week
    week_start = get_week_start(date)

    # Add 6 days to get to Sunday, and set to end of day
    week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999)

    return week_end


def handle_timezone_error(
    e: Exception, report_name: str, all_commits: list, logger: logging.Logger
) -> None:
    """Handle timezone comparison errors with detailed logging."""
    if isinstance(e, TypeError) and (
        "can't compare" in str(e).lower() or "timezone" in str(e).lower()
    ):
        logger.error(f"Timezone comparison error in {report_name}:")
        logger.error(f"  Error: {e}")
        import traceback

        logger.error(f"  Full traceback:\n{traceback.format_exc()}")

        # Log context information
        sample_commits = all_commits[:5] if all_commits else []
        for i, commit in enumerate(sample_commits):
            timestamp = commit.get("timestamp")
            logger.error(
                f"  Sample commit {i}: timestamp={timestamp} (tzinfo: {getattr(timestamp, 'tzinfo', 'N/A')})"
            )

        click.echo(f"   ❌ Timezone comparison error in {report_name}")
        click.echo("   🔍 See logs with --log DEBUG for detailed information")
        click.echo("   💡 This usually indicates mixed timezone-aware and naive datetime objects")
        raise
    else:
        # Re-raise other errors
        raise


class ImprovedErrorHandler:
    """Enhanced error handling with helpful suggestions."""

    @staticmethod
    def handle_command_error(ctx: click.Context, error: Exception) -> None:
        """Handle command errors with helpful suggestions."""
        error_msg = str(error)

        # Check for common errors and provide suggestions
        if "no such option" in error_msg.lower():
            # Extract the invalid option
            import re

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
            import re

            match = re.search(r"no such command[:'] (\w+)", error_msg.lower())
            if match:
                invalid_cmd = match.group(1)
                available_cmds = [
                    "analyze",
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


class AnalyzeAsDefaultGroup(click.Group):
    """
    Custom Click group that treats unrecognized options as analyze command arguments.
    This allows 'gitflow-analytics -c config.yaml' to work like 'gitflow-analytics analyze -c config.yaml'
    """

    def parse_args(self, ctx, args):
        """Override parse_args to redirect unrecognized patterns to analyze command."""
        # Check if the first argument is a known subcommand
        if args and args[0] in self.list_commands(ctx):
            return super().parse_args(ctx, args)

        # Check for global options that should NOT be routed to analyze
        global_options = {"--version", "--help", "-h"}
        if args and args[0] in global_options:
            return super().parse_args(ctx, args)

        # Check if we have arguments that look like analyze options
        analyze_indicators = {
            "-c",
            "--config",
            "-w",
            "--weeks",
            "--output",
            "-o",
            "--anonymize",
            "--no-cache",
            "--validate-only",
            "--clear-cache",
            "--enable-qualitative",
            "--qualitative-only",
            "--enable-pm",
            "--pm-platform",
            "--disable-pm",
            "--rich",
            "--log",
            "--skip-identity-analysis",
            "--apply-identity-suggestions",
            "--warm-cache",
            "--validate-cache",
            "--generate-csv",
        }

        # If first arg starts with - and looks like an analyze option, prepend 'analyze'
        if args and args[0].startswith("-"):
            # Check if any of the args are analyze indicators
            has_analyze_indicators = any(arg in analyze_indicators for arg in args)
            if has_analyze_indicators:
                # This looks like it should be an analyze command
                new_args = ["analyze"] + args
                return super().parse_args(ctx, new_args)

        # Otherwise, use default behavior
        return super().parse_args(ctx, args)


@click.group(cls=AnalyzeAsDefaultGroup, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="GitFlow Analytics")
@click.help_option("-h", "--help")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """GitFlow Analytics - Developer productivity insights from Git history.

    \b
    A comprehensive tool for analyzing Git repositories to generate developer
    productivity metrics, DORA metrics, and team insights without requiring
    external project management tools.

    \b
    QUICK START:
      1. Create a configuration file (see config-sample.yaml)
      2. Run analysis: gitflow-analytics -c config.yaml --weeks 4
      3. View reports in the output directory

    \b
    COMMON WORKFLOWS:
      Analyze last 4 weeks:     gitflow-analytics -c config.yaml --weeks 4
      Generate specific report: gitflow-analytics -c config.yaml --format csv
      Clear cache and analyze:  gitflow-analytics -c config.yaml --clear-cache
      Validate configuration:   gitflow-analytics -c config.yaml --validate-only

    \b
    COMMANDS:
      analyze    Analyze repositories and generate reports (default)
      identities Manage developer identity resolution
      train      Train ML models for commit classification
      fetch      Fetch external data (GitHub PRs, PM tickets)
      help       Show detailed help and documentation

    \b
    For detailed command help: gitflow-analytics COMMAND --help
    For documentation: https://github.com/yourusername/gitflow-analytics
    """
    # If no subcommand was invoked, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)


# TUI command removed - replaced with rich CLI output
# Legacy TUI code preserved but not exposed


@cli.command(name="analyze")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--weeks", "-w", type=int, default=12, help="Number of weeks to analyze (default: 12)"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for reports (overrides config file)",
)
@click.option("--anonymize", is_flag=True, help="Anonymize developer information in reports")
@click.option("--no-cache", is_flag=True, help="Disable caching (slower but always fresh)")
@click.option(
    "--validate-only", is_flag=True, help="Validate configuration without running analysis"
)
@click.option("--clear-cache", is_flag=True, help="Clear cache before running analysis")
@click.option(
    "--enable-qualitative",
    is_flag=True,
    help="Enable qualitative analysis (requires additional dependencies)",
)
@click.option(
    "--qualitative-only", is_flag=True, help="Run only qualitative analysis on existing commits"
)
@click.option(
    "--enable-pm", is_flag=True, help="Enable PM platform integration (overrides config setting)"
)
@click.option(
    "--pm-platform",
    multiple=True,
    help="Enable specific PM platforms (e.g., --pm-platform jira --pm-platform azure)",
)
@click.option(
    "--disable-pm", is_flag=True, help="Disable PM platform integration (overrides config setting)"
)
@click.option(
    "--rich", is_flag=True, default=True, help="Use rich terminal output (default: enabled)"
)
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="none",
    help="Enable logging with specified level (default: none)",
)
@click.option("--skip-identity-analysis", is_flag=True, help="Skip automatic identity analysis")
@click.option(
    "--apply-identity-suggestions",
    is_flag=True,
    help="Apply identity analysis suggestions without prompting",
)
@click.option(
    "--warm-cache", is_flag=True, help="Pre-warm cache with all commits for faster subsequent runs"
)
@click.option("--validate-cache", is_flag=True, help="Validate cache integrity and consistency")
@click.option(
    "--generate-csv",
    is_flag=True,
    help="Generate CSV reports (disabled by default, only narrative report is generated)",
)
@click.option(
    "--use-batch-classification/--use-legacy-classification",
    default=True,
    help="Use batch LLM classification on pre-fetched data (Step 2 of 2) - now the default behavior",
)
@click.option(
    "--force-fetch", is_flag=True, help="Force fetch fresh data even if cached data exists"
)
def analyze_subcommand(
    config: Path,
    weeks: int,
    output: Optional[Path],
    anonymize: bool,
    no_cache: bool,
    validate_only: bool,
    clear_cache: bool,
    enable_qualitative: bool,
    qualitative_only: bool,
    enable_pm: bool,
    pm_platform: tuple[str, ...],
    disable_pm: bool,
    rich: bool,
    log: str,
    skip_identity_analysis: bool,
    apply_identity_suggestions: bool,
    warm_cache: bool,
    validate_cache: bool,
    generate_csv: bool,
    use_batch_classification: bool,
    force_fetch: bool,
) -> None:
    """Analyze Git repositories and generate comprehensive productivity reports.

    \b
    This is the main command for GitFlow Analytics. It:
    - Analyzes commit history from configured repositories
    - Resolves developer identities across different email addresses
    - Extracts ticket references and categorizes commits
    - Calculates productivity metrics and DORA metrics
    - Generates reports in various formats

    \b
    EXAMPLES:
      # Basic analysis of last 4 weeks
      gitflow-analytics analyze -c config.yaml --weeks 4

      # Generate CSV reports with fresh data
      gitflow-analytics analyze -c config.yaml --generate-csv --clear-cache

      # Quick validation of configuration
      gitflow-analytics analyze -c config.yaml --validate-only

      # Analyze with qualitative insights
      gitflow-analytics analyze -c config.yaml --enable-qualitative

    \b
    OUTPUT FILES:
      - developer_metrics_YYYYMMDD.csv: Individual developer statistics
      - weekly_metrics_YYYYMMDD.csv: Week-by-week team metrics
      - narrative_report_YYYYMMDD.md: Executive summary and insights
      - comprehensive_export_YYYYMMDD.json: Complete data export

    \b
    PERFORMANCE TIPS:
      - Use --no-cache for latest data but slower performance
      - Use --clear-cache when configuration changes
      - Smaller --weeks values analyze faster
      - Enable caching for repeated analyses
    """
    # Call the main analyze function
    analyze(
        config=config,
        weeks=weeks,
        output=output,
        anonymize=anonymize,
        no_cache=no_cache,
        validate_only=validate_only,
        clear_cache=clear_cache,
        enable_qualitative=enable_qualitative,
        qualitative_only=qualitative_only,
        enable_pm=enable_pm,
        pm_platform=pm_platform,
        disable_pm=disable_pm,
        rich=rich,
        log=log,
        skip_identity_analysis=skip_identity_analysis,
        apply_identity_suggestions=apply_identity_suggestions,
        warm_cache=warm_cache,
        validate_cache=validate_cache,
        generate_csv=generate_csv,
        use_batch_classification=use_batch_classification,
        force_fetch=force_fetch,
    )


def analyze(
    config: Path,
    weeks: int,
    output: Optional[Path],
    anonymize: bool,
    no_cache: bool,
    validate_only: bool,
    clear_cache: bool,
    enable_qualitative: bool,
    qualitative_only: bool,
    enable_pm: bool,
    pm_platform: tuple[str, ...],
    disable_pm: bool,
    rich: bool,
    log: str,
    skip_identity_analysis: bool,
    apply_identity_suggestions: bool,
    warm_cache: bool = False,
    validate_cache: bool = False,
    generate_csv: bool = False,
    use_batch_classification: bool = True,
    force_fetch: bool = False,
) -> None:
    """Analyze Git repositories using configuration file."""

    # Initialize display - use rich by default, fall back to simple output if needed
    display = create_rich_display() if rich else None

    # Configure logging based on the --log option
    if log.upper() != "NONE":
        # Configure structured logging with detailed formatter
        log_level = getattr(logging, log.upper())
        logging.basicConfig(
            level=log_level,
            format="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            handlers=[logging.StreamHandler(sys.stderr)],
            force=True,  # Ensure reconfiguration of existing loggers
        )

        # Ensure all GitFlow Analytics loggers are configured properly
        root_logger = logging.getLogger("gitflow_analytics")
        root_logger.setLevel(log_level)

        # Create logger for this module
        logger = logging.getLogger(__name__)
        logger.info(f"Logging enabled at {log.upper()} level")

        # Log that logging is properly configured for all modules
        logger.debug("Logging configuration applied to all gitflow_analytics modules")
    else:
        # Disable logging
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger("gitflow_analytics").setLevel(logging.CRITICAL)
        logger = logging.getLogger(__name__)

    try:
        if display:
            display.show_header()

        # Load configuration
        if display:
            display.print_status(f"Loading configuration from {config}...", "info")
        else:
            click.echo(f"📋 Loading configuration from {config}...")

        cfg = ConfigLoader.load(config)

        # Apply CLI overrides for PM integration
        if disable_pm:
            # Disable PM integration if explicitly requested
            if cfg.pm_integration:
                cfg.pm_integration.enabled = False
            if display:
                display.print_status("PM integration disabled via CLI flag", "info")
            else:
                click.echo("🚫 PM integration disabled via CLI flag")
        elif enable_pm:
            # Enable PM integration if explicitly requested
            if not cfg.pm_integration:
                from .config import PMIntegrationConfig

                cfg.pm_integration = PMIntegrationConfig(enabled=True)
            else:
                cfg.pm_integration.enabled = True
            if display:
                display.print_status("PM integration enabled via CLI flag", "info")
            else:
                click.echo("📋 PM integration enabled via CLI flag")

        # Filter PM platforms if specific ones are requested
        if pm_platform and cfg.pm_integration:
            requested_platforms = set(pm_platform)
            # Disable platforms not requested
            for platform_name in list(cfg.pm_integration.platforms.keys()):
                if platform_name not in requested_platforms:
                    cfg.pm_integration.platforms[platform_name].enabled = False
            if display:
                display.print_status(
                    f"PM integration limited to platforms: {', '.join(pm_platform)}", "info"
                )
            else:
                click.echo(f"📋 PM integration limited to platforms: {', '.join(pm_platform)}")

        # Validate configuration
        warnings = ConfigLoader.validate_config(cfg)
        if warnings:
            warning_msg = "Configuration warnings:\n" + "\n".join(f"• {w}" for w in warnings)
            if display:
                display.show_warning(warning_msg)
            else:
                click.echo("⚠️  Configuration warnings:")
                for warning in warnings:
                    click.echo(f"   - {warning}")

        if validate_only:
            if not warnings:
                if display:
                    display.print_status("Configuration is valid!", "success")
                else:
                    click.echo("✅ Configuration is valid!")
            else:
                if display:
                    display.print_status(
                        "Configuration has issues that should be addressed.", "error"
                    )
                else:
                    click.echo("❌ Configuration has issues that should be addressed.")
            return

        # Use output directory from CLI or config
        if output is None:
            # cfg.output.directory is already resolved relative to config file by ConfigLoader
            output = cfg.output.directory if cfg.output.directory else Path("./reports")

        # Setup output directory
        output.mkdir(parents=True, exist_ok=True)

        # Show configuration status in rich display
        if display:
            github_org = cfg.github.organization if cfg.github else None
            github_token_valid = bool(cfg.github and cfg.github.token)
            jira_configured = bool(cfg.jira and cfg.jira.base_url)
            jira_valid = jira_configured  # Simplified validation

            display.show_configuration_status(
                config,
                github_org=github_org,
                github_token_valid=github_token_valid,
                jira_configured=jira_configured,
                jira_valid=jira_valid,
                analysis_weeks=weeks,
            )

        # Initialize components
        cache_dir = cfg.cache.directory
        cache = GitAnalysisCache(cache_dir, ttl_hours=0 if no_cache else cfg.cache.ttl_hours)

        if clear_cache:
            if display:
                display.print_status("Clearing cache...", "info")
            else:
                click.echo("🗑️  Clearing cache...")

            try:
                # Use the new method that provides detailed feedback
                cleared_counts = cache.clear_all_cache()
                if display:
                    display.print_status(
                        f"Cache cleared: {cleared_counts['commits']} commits, "
                        f"{cleared_counts['repository_status']} repo status records, "
                        f"{cleared_counts['total']} total entries",
                        "success",
                    )
                else:
                    click.echo(
                        f"✅ Cache cleared: {cleared_counts['commits']} commits, "
                        f"{cleared_counts['repository_status']} repo status records, "
                        f"{cleared_counts['total']} total entries"
                    )
            except Exception:
                # Fallback to old method if database methods fail
                if display:
                    display.print_status("Using fallback cache clearing...", "info")
                else:
                    click.echo("🗑️  Using fallback cache clearing...")
                import shutil

                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    if display:
                        display.print_status("Cache directory removed", "success")
                    else:
                        click.echo("✅ Cache directory removed")

        # Handle cache validation if requested
        if validate_cache:
            if display:
                display.print_status("Validating cache integrity...", "info")
            validation_result = cache.validate_cache()

            if display:
                if validation_result["is_valid"]:
                    display.print_status("✅ Cache validation passed", "success")
                else:
                    display.print_status("❌ Cache validation failed", "error")
                    for issue in validation_result["issues"]:
                        display.print_status(f"  Issue: {issue}", "error")

                if validation_result["warnings"]:
                    for warning in validation_result["warnings"]:
                        display.print_status(f"  Warning: {warning}", "warning")

                # Show validation statistics
                stats = validation_result["stats"]
                display.print_status(f"Cache contains {stats['total_commits']} commits", "info")
                if stats["duplicates"] > 0:
                    display.print_status(
                        f"Found {stats['duplicates']} duplicate entries", "warning"
                    )
            else:
                # Simple output when not using rich display
                if validation_result["is_valid"]:
                    click.echo("✅ Cache validation passed")
                else:
                    click.echo("❌ Cache validation failed:")
                    for issue in validation_result["issues"]:
                        click.echo(f"  Issue: {issue}")

                if validation_result["warnings"]:
                    click.echo("Warnings:")
                    for warning in validation_result["warnings"]:
                        click.echo(f"  {warning}")

            # Exit after validation if no other action requested
            if not warm_cache:
                return

        # Handle cache warming if requested
        if warm_cache:
            if display:
                display.print_status("Warming cache with all repository commits...", "info")

            # Get all repository paths from configuration
            repo_paths = []
            for repo_config in cfg.repositories:
                repo_paths.append(repo_config.path)

            warming_result = cache.warm_cache(repo_paths, weeks=weeks)

            if display:
                display.print_status("✅ Cache warming completed", "success")
                display.print_status(
                    f"  Repositories processed: {warming_result['repos_processed']}", "info"
                )
                display.print_status(
                    f"  Total commits found: {warming_result['total_commits_found']}", "info"
                )
                display.print_status(
                    f"  Commits cached: {warming_result['commits_cached']}", "info"
                )
                display.print_status(
                    f"  Already cached: {warming_result['commits_already_cached']}", "info"
                )
                display.print_status(
                    f"  Duration: {warming_result['duration_seconds']:.1f}s", "info"
                )

                if warming_result["errors"]:
                    for error in warming_result["errors"]:
                        display.print_status(f"  Error: {error}", "error")
            else:
                # Simple output when not using rich display
                click.echo(
                    f"✅ Cache warming completed in {warming_result['duration_seconds']:.1f}s"
                )
                click.echo(f"  Repositories: {warming_result['repos_processed']}")
                click.echo(f"  Commits found: {warming_result['total_commits_found']}")
                click.echo(f"  Newly cached: {warming_result['commits_cached']}")
                click.echo(f"  Already cached: {warming_result['commits_already_cached']}")

                if warming_result["errors"]:
                    click.echo("Errors encountered:")
                    for error in warming_result["errors"]:
                        click.echo(f"  {error}")

            # Exit after warming unless we're also doing normal analysis
            if validate_only:
                return

        # Initialize identity resolver with comprehensive error handling
        identity_db_path = cache_dir / "identities.db"
        try:
            identity_resolver = DeveloperIdentityResolver(
                identity_db_path,
                similarity_threshold=cfg.analysis.similarity_threshold,
                manual_mappings=cfg.analysis.manual_identity_mappings,
            )

            # Check if we're using fallback and inform user
            if (
                hasattr(identity_resolver, "_database_available")
                and not identity_resolver._database_available
            ):
                click.echo(
                    click.style("⚠️  Warning: ", fg="yellow", bold=True)
                    + "Identity database unavailable. Using in-memory fallback."
                )
                click.echo("   Identity mappings will not persist between runs.")
                click.echo(f"   Check permissions on: {identity_db_path.parent}")

            elif (
                hasattr(identity_resolver.db, "is_readonly_fallback")
                and identity_resolver.db.is_readonly_fallback
            ):
                click.echo(
                    click.style("⚠️  Warning: ", fg="yellow", bold=True)
                    + "Using temporary database for identity resolution."
                )
                click.echo("   Identity mappings will not persist between runs.")
                click.echo(f"   Check permissions on: {identity_db_path.parent}")

        except Exception as e:
            click.echo(
                click.style("❌ Error: ", fg="red", bold=True)
                + f"Failed to initialize identity resolver: {e}"
            )
            click.echo(
                click.style("💡 Fix: ", fg="blue", bold=True) + "Try one of these solutions:"
            )
            click.echo(f"   • Check directory permissions: {cache_dir}")
            click.echo(f"   • Check available disk space: {cache_dir}")
            click.echo("   • Run with different cache directory:")
            click.echo("     export GITFLOW_CACHE_DIR=/tmp/gitflow-cache")
            click.echo("   • Run in readonly mode (analysis will work, no persistence):")
            click.echo(f"     chmod -w {cache_dir}")
            raise click.ClickException(f"Identity resolver initialization failed: {e}") from e

        # Prepare ML categorization config for analyzer
        ml_config = None
        if hasattr(cfg.analysis, "ml_categorization"):
            ml_config = {
                "enabled": cfg.analysis.ml_categorization.enabled,
                "min_confidence": cfg.analysis.ml_categorization.min_confidence,
                "semantic_weight": cfg.analysis.ml_categorization.semantic_weight,
                "file_pattern_weight": cfg.analysis.ml_categorization.file_pattern_weight,
                "hybrid_threshold": cfg.analysis.ml_categorization.hybrid_threshold,
                "cache_duration_days": cfg.analysis.ml_categorization.cache_duration_days,
                "batch_size": cfg.analysis.ml_categorization.batch_size,
                "enable_caching": cfg.analysis.ml_categorization.enable_caching,
                "spacy_model": cfg.analysis.ml_categorization.spacy_model,
            }

        # LLM classification configuration
        llm_config = {
            "enabled": cfg.analysis.llm_classification.enabled,
            "api_key": cfg.analysis.llm_classification.api_key,
            "model": cfg.analysis.llm_classification.model,
            "confidence_threshold": cfg.analysis.llm_classification.confidence_threshold,
            "max_tokens": cfg.analysis.llm_classification.max_tokens,
            "temperature": cfg.analysis.llm_classification.temperature,
            "timeout_seconds": cfg.analysis.llm_classification.timeout_seconds,
            "cache_duration_days": cfg.analysis.llm_classification.cache_duration_days,
            "enable_caching": cfg.analysis.llm_classification.enable_caching,
            "max_daily_requests": cfg.analysis.llm_classification.max_daily_requests,
            "domain_terms": cfg.analysis.llm_classification.domain_terms,
        }

        # Configure branch analysis
        branch_analysis_config = {
            "strategy": cfg.analysis.branch_analysis.strategy,
            "max_branches_per_repo": cfg.analysis.branch_analysis.max_branches_per_repo,
            "active_days_threshold": cfg.analysis.branch_analysis.active_days_threshold,
            "include_main_branches": cfg.analysis.branch_analysis.include_main_branches,
            "always_include_patterns": cfg.analysis.branch_analysis.always_include_patterns,
            "always_exclude_patterns": cfg.analysis.branch_analysis.always_exclude_patterns,
            "enable_progress_logging": cfg.analysis.branch_analysis.enable_progress_logging,
            "branch_commit_limit": cfg.analysis.branch_analysis.branch_commit_limit,
        }

        analyzer = GitAnalyzer(
            cache,
            branch_mapping_rules=cfg.analysis.branch_mapping_rules,
            allowed_ticket_platforms=getattr(cfg.analysis, "ticket_platforms", None),
            exclude_paths=cfg.analysis.exclude_paths,
            story_point_patterns=cfg.analysis.story_point_patterns,
            ml_categorization_config=ml_config,
            llm_config=llm_config,
            branch_analysis_config=branch_analysis_config,
        )
        orchestrator = IntegrationOrchestrator(cfg, cache)

        # Discovery organization repositories if needed
        repositories_to_analyze = cfg.repositories
        if cfg.github.organization and not repositories_to_analyze:
            if display:
                display.print_status(
                    f"Discovering repositories from organization: {cfg.github.organization}", "info"
                )
            else:
                click.echo(
                    f"🔍 Discovering repositories from organization: {cfg.github.organization}"
                )
            try:
                # Use a 'repos' directory in the config directory for cloned repositories
                config_dir = Path(config).parent if config else Path.cwd()
                repos_dir = config_dir / "repos"
                discovered_repos = cfg.discover_organization_repositories(clone_base_path=repos_dir)
                repositories_to_analyze = discovered_repos

                if display:
                    display.print_status(
                        f"Found {len(discovered_repos)} repositories in organization", "success"
                    )
                    # Show repository discovery in structured format
                    repo_data = [
                        {
                            "name": repo.name,
                            "github_repo": repo.github_repo,
                            "exists": repo.path.exists(),
                        }
                        for repo in discovered_repos
                    ]
                    display.show_repository_discovery(repo_data)
                else:
                    click.echo(f"   ✅ Found {len(discovered_repos)} repositories in organization")
                    for repo in discovered_repos:
                        click.echo(f"      - {repo.name} ({repo.github_repo})")
            except Exception as e:
                if display:
                    display.show_error(f"Failed to discover repositories: {e}")
                else:
                    click.echo(f"   ❌ Failed to discover repositories: {e}")
                return

        # Analysis period (timezone-aware to match commit timestamps)
        # Calculate week-aligned boundaries for exact N-week analysis period
        current_time = datetime.now(timezone.utc)

        # Calculate dates to use last N complete weeks (not including current week)
        # Get the start of current week, then go back 1 week to get last complete week
        current_week_start = get_week_start(current_time)
        last_complete_week_start = current_week_start - timedelta(weeks=1)

        # Start date is N weeks back from the last complete week
        start_date = last_complete_week_start - timedelta(weeks=weeks - 1)

        # End date is the end of the last complete week (last Sunday)
        end_date = get_week_end(last_complete_week_start + timedelta(days=6))

        if display:
            display.print_status(
                f"Analyzing {len(repositories_to_analyze)} repositories...", "info"
            )
            display.print_status(
                f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "info",
            )
            # Start live progress display
            display.start_live_display()
            display.add_progress_task(
                "repos", "Processing repositories", len(repositories_to_analyze)
            )
        else:
            click.echo(f"\n🚀 Analyzing {len(repositories_to_analyze)} repositories...")
            click.echo(
                f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            )

        # Initialize variables for both batch and non-batch modes
        developer_stats = None
        ticket_analysis = None

        # Generate configuration hash for cache validation
        config_hash = cache.generate_config_hash(
            branch_mapping_rules=getattr(cfg.analysis, "branch_mapping_rules", {}),
            ticket_platforms=getattr(
                cfg.analysis, "ticket_platforms", ["jira", "github", "clickup", "linear"]
            ),
            exclude_paths=getattr(cfg.analysis, "exclude_paths", None),
            ml_categorization_enabled=ml_config.get("enabled", False) if ml_config else False,
            additional_config={
                "weeks": weeks,
                "enable_qualitative": enable_qualitative,
                "enable_pm": enable_pm,
                "pm_platforms": list(pm_platform) if pm_platform else [],
            },
        )

        # Check if we should use batch classification (two-step process)
        if use_batch_classification:
            if display:
                display.print_status("Using two-step process: fetch then classify...", "info")
            else:
                click.echo("🔄 Using two-step process: fetch then classify...")

            # Cache-first logic: Check if analysis is already complete
            repos_needing_analysis = []
            cached_repos = []

            if not force_fetch:
                if display:
                    display.print_status("Checking cache completeness...", "info")
                else:
                    click.echo("🔍 Checking cache completeness...")

                for repo_config in repositories_to_analyze:
                    repo_path = str(Path(repo_config.path))
                    status = cache.get_repository_analysis_status(
                        repo_path=repo_path,
                        analysis_start=start_date,
                        analysis_end=end_date,
                        config_hash=config_hash,
                    )

                    if status:
                        cached_repos.append((repo_config, status))
                        if display:
                            display.print_status(
                                f"   ✅ {repo_config.name}: Using cached data ({status['commit_count']} commits)",
                                "success",
                            )
                    else:
                        repos_needing_analysis.append(repo_config)
                        if display:
                            display.print_status(
                                f"   📥 {repo_config.name}: Analysis needed", "info"
                            )

                if cached_repos:
                    total_cached_commits = sum(status["commit_count"] for _, status in cached_repos)
                    if display:
                        display.print_status(
                            f"Found {len(cached_repos)} repos with cached data ({total_cached_commits} commits)",
                            "success",
                        )
                    else:
                        click.echo(
                            f"✅ Found {len(cached_repos)} repos with cached data ({total_cached_commits} commits)"
                        )
            else:
                # Force fetch: analyze all repositories
                repos_needing_analysis = repositories_to_analyze
                if display:
                    display.print_status("Force fetch enabled - analyzing all repositories", "info")
                else:
                    click.echo("🔄 Force fetch enabled - analyzing all repositories")

            # Step 1: Fetch data only for repos that need analysis
            if repos_needing_analysis:
                if display:
                    display.print_status(
                        f"Step 1: Fetching data for {len(repos_needing_analysis)} repositories...",
                        "info",
                    )
                else:
                    click.echo(
                        f"📥 Step 1: Fetching data for {len(repos_needing_analysis)} repositories..."
                    )

                # Perform data fetch for repositories that need analysis
                from .core.data_fetcher import GitDataFetcher

                data_fetcher = GitDataFetcher(
                    cache=cache,
                    branch_mapping_rules=getattr(cfg.analysis, "branch_mapping_rules", {}),
                    allowed_ticket_platforms=getattr(
                        cfg.analysis, "ticket_platforms", ["jira", "github", "clickup", "linear"]
                    ),
                    exclude_paths=getattr(cfg.analysis, "exclude_paths", None),
                )

                # Initialize integrations for ticket fetching
                orchestrator = IntegrationOrchestrator(cfg, cache)
                jira_integration = orchestrator.integrations.get("jira")

                # Fetch data for repositories that need analysis
                total_commits = 0
                total_tickets = 0

                for repo_config in repos_needing_analysis:
                    try:
                        repo_path = Path(repo_config.path)
                        project_key = repo_config.project_key or repo_path.name

                        # Progress callback for fetch
                        def progress_callback(message: str):
                            if display:
                                display.print_status(f"   {message}", "info")

                        # Fetch repository data
                        result = data_fetcher.fetch_repository_data(
                            repo_path=repo_path,
                            project_key=project_key,
                            weeks_back=weeks,
                            branch_patterns=getattr(repo_config, "branch_patterns", None),
                            jira_integration=jira_integration,
                            progress_callback=progress_callback,
                            start_date=start_date,
                            end_date=end_date,
                        )

                        total_commits += result["stats"]["total_commits"]
                        total_tickets += result["stats"]["unique_tickets"]

                        if display:
                            display.print_status(
                                f"   ✅ {project_key}: {result['stats']['total_commits']} commits, "
                                f"{result['stats']['unique_tickets']} tickets",
                                "success",
                            )

                        # Mark repository analysis as complete
                        cache.mark_repository_analysis_complete(
                            repo_path=str(repo_path),
                            repo_name=repo_config.name,
                            project_key=project_key,
                            analysis_start=start_date,
                            analysis_end=end_date,
                            weeks_analyzed=weeks,
                            commit_count=result["stats"]["total_commits"],
                            ticket_count=result["stats"]["unique_tickets"],
                            config_hash=config_hash,
                        )

                    except Exception as e:
                        if display:
                            display.print_status(
                                f"   ❌ Error fetching {project_key}: {e}", "error"
                            )
                        else:
                            click.echo(f"   ❌ Error fetching {project_key}: {e}")

                        # Mark repository analysis as failed
                        with contextlib.suppress(Exception):
                            cache.mark_repository_analysis_failed(
                                repo_path=str(repo_path),
                                repo_name=repo_config.name,
                                analysis_start=start_date,
                                analysis_end=end_date,
                                error_message=str(e),
                                config_hash=config_hash,
                            )
                        continue

                if display:
                    display.print_status(
                        f"Step 1 complete: {total_commits} commits, {total_tickets} tickets fetched",
                        "success",
                    )
                else:
                    click.echo(
                        f"📥 Step 1 complete: {total_commits} commits, {total_tickets} tickets fetched"
                    )
            else:
                if display:
                    display.print_status(
                        "All repositories use cached data - skipping data fetch", "success"
                    )
                else:
                    click.echo("✅ All repositories use cached data - skipping data fetch")

            # Step 2: Batch classification
            # ENHANCED VALIDATION: Verify both commits and batches exist before proceeding
            from sqlalchemy import and_

            from .models.database import CachedCommit, DailyCommitBatch

            validation_passed = False

            with cache.get_session() as session:
                # Check 1: Verify commits were actually stored for the date range
                stored_commits = (
                    session.query(CachedCommit)
                    .filter(
                        and_(
                            CachedCommit.timestamp >= start_date, CachedCommit.timestamp <= end_date
                        )
                    )
                    .count()
                )

                # Check 2: Verify daily batches exist for classification
                existing_batches = (
                    session.query(DailyCommitBatch)
                    .filter(
                        and_(
                            DailyCommitBatch.date >= start_date.date(),
                            DailyCommitBatch.date <= end_date.date(),
                        )
                    )
                    .count()
                )

                # VALIDATION LOGIC
                if stored_commits > 0 and existing_batches > 0:
                    # All good - we have both commits and batches
                    validation_passed = True
                    if display:
                        display.print_status(
                            f"✅ Data validation passed: {stored_commits} commits, {existing_batches} batches ready",
                            "success",
                        )
                    else:
                        click.echo(
                            f"✅ Data validation passed: {stored_commits} commits, {existing_batches} batches ready"
                        )

                elif stored_commits > 0 and existing_batches == 0:
                    # We have commits but no batches - this shouldn't happen but we can recover
                    if display:
                        display.print_status(
                            f"⚠️ Found {stored_commits} commits but no daily batches - data inconsistency detected",
                            "warning",
                        )
                    else:
                        click.echo(
                            f"⚠️ Found {stored_commits} commits but no daily batches - data inconsistency detected"
                        )

                elif stored_commits == 0 and total_commits > 0:
                    # Step 1 claimed success but no commits were stored - critical error
                    error_msg = f"❌ VALIDATION FAILED: Step 1 reported {total_commits} commits but database contains 0 commits for date range"
                    if display:
                        display.print_status(error_msg, "error")
                    else:
                        click.echo(error_msg)
                    click.echo(
                        f"   📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    )
                    click.echo(
                        f"   📊 Step 1 stats: {total_commits} commits, {total_tickets} tickets"
                    )
                    click.echo(
                        f"   🗃️ Database reality: {stored_commits} commits, {existing_batches} batches"
                    )
                    click.echo(
                        "   💡 This suggests a timezone, date filtering, or database storage issue"
                    )
                    raise click.ClickException(
                        "Data validation failed - Step 1 success was misleading"
                    )

                elif stored_commits == 0 and existing_batches == 0:
                    # No data at all - need to fetch or explain why
                    if display:
                        display.print_status(
                            "📊 No commits or batches found for date range - proceeding with data fetch",
                            "warning",
                        )
                    else:
                        click.echo(
                            "📊 No commits or batches found for date range - proceeding with data fetch"
                        )

            # PROCEED WITH EMERGENCY FETCH if validation didn't pass
            if not validation_passed:
                if display:
                    display.print_status(
                        "Data validation failed - running emergency data fetch", "warning"
                    )
                else:
                    click.echo("⚠️ Data validation failed - running emergency data fetch")

                # Force data fetch for all repositories since we have no batches
                repos_needing_analysis = repositories_to_analyze

                # Run the data fetch step that was skipped
                if repos_needing_analysis:
                    if display:
                        display.print_status(
                            f"Emergency fetch: Fetching data for {len(repos_needing_analysis)} repositories...",
                            "info",
                        )
                    else:
                        click.echo(
                            f"🚨 Emergency fetch: Fetching data for {len(repos_needing_analysis)} repositories..."
                        )
                        click.echo(
                            "   📋 Reason: Need to ensure commits and batches exist for classification"
                        )

                    # Perform data fetch for repositories that need analysis
                    from .core.data_fetcher import GitDataFetcher

                    data_fetcher = GitDataFetcher(
                        cache=cache,
                        branch_mapping_rules=getattr(cfg.analysis, "branch_mapping_rules", {}),
                        allowed_ticket_platforms=getattr(
                            cfg.analysis,
                            "ticket_platforms",
                            ["jira", "github", "clickup", "linear"],
                        ),
                        exclude_paths=getattr(cfg.analysis, "exclude_paths", None),
                    )

                    # Initialize integrations for ticket fetching
                    orchestrator = IntegrationOrchestrator(cfg, cache)
                    jira_integration = orchestrator.integrations.get("jira")

                    # Fetch data for repositories that need analysis
                    total_commits = 0
                    total_tickets = 0

                    for repo_config in repos_needing_analysis:
                        try:
                            repo_path = Path(repo_config.path)
                            project_key = repo_config.project_key or repo_path.name

                            # Progress callback for fetch
                            def progress_callback(message: str):
                                if display:
                                    display.print_status(f"   {message}", "info")

                            # Fetch repository data
                            result = data_fetcher.fetch_repository_data(
                                repo_path=repo_path,
                                project_key=project_key,
                                weeks_back=weeks,
                                branch_patterns=getattr(repo_config, "branch_patterns", None),
                                jira_integration=jira_integration,
                                progress_callback=progress_callback,
                                start_date=start_date,
                                end_date=end_date,
                            )

                            total_commits += result["stats"]["total_commits"]
                            total_tickets += result["stats"]["unique_tickets"]

                            if display:
                                display.print_status(
                                    f"   ✅ {project_key}: {result['stats']['total_commits']} commits, "
                                    f"{result['stats']['unique_tickets']} tickets",
                                    "success",
                                )

                            # Mark repository analysis as complete
                            cache.mark_repository_analysis_complete(
                                repo_path=str(repo_path),
                                repo_name=repo_config.name,
                                project_key=project_key,
                                analysis_start=start_date,
                                analysis_end=end_date,
                                weeks_analyzed=weeks,
                                commit_count=result["stats"]["total_commits"],
                                ticket_count=result["stats"]["unique_tickets"],
                                config_hash=config_hash,
                            )

                        except Exception as e:
                            if display:
                                display.print_status(
                                    f"   ❌ Error fetching {project_key}: {e}", "error"
                                )
                            else:
                                click.echo(f"   ❌ Error fetching {project_key}: {e}")
                            continue

                    if display:
                        display.print_status(
                            f"Emergency fetch complete: {total_commits} commits, {total_tickets} tickets",
                            "success",
                        )
                    else:
                        click.echo(
                            f"🚨 Emergency fetch complete: {total_commits} commits, {total_tickets} tickets"
                        )

                    # RE-VALIDATE after emergency fetch
                    with cache.get_session() as session:
                        final_commits = (
                            session.query(CachedCommit)
                            .filter(
                                and_(
                                    CachedCommit.timestamp >= start_date,
                                    CachedCommit.timestamp <= end_date,
                                )
                            )
                            .count()
                        )

                        final_batches = (
                            session.query(DailyCommitBatch)
                            .filter(
                                and_(
                                    DailyCommitBatch.date >= start_date.date(),
                                    DailyCommitBatch.date <= end_date.date(),
                                )
                            )
                            .count()
                        )

                        if final_commits == 0:
                            error_msg = "❌ CRITICAL: Emergency fetch completed but still 0 commits stored in database"
                            if display:
                                display.print_status(error_msg, "error")
                            else:
                                click.echo(error_msg)
                            click.echo(
                                f"   📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                            )
                            click.echo(f"   📊 Emergency stats: {total_commits} commits reported")
                            click.echo(
                                f"   🗃️ Database result: {final_commits} commits, {final_batches} batches"
                            )
                            click.echo("   🔍 Possible causes:")
                            click.echo(
                                "      - Timezone mismatch between commit timestamps and analysis range"
                            )
                            click.echo("      - Date filtering excluding all commits")
                            click.echo("      - Database transaction not committed")
                            click.echo(
                                "      - Repository has no commits in the specified time range"
                            )
                            raise click.ClickException(
                                "Emergency fetch failed validation - no data available for classification"
                            )

                        if display:
                            display.print_status(
                                f"✅ Post-fetch validation: {final_commits} commits, {final_batches} batches confirmed",
                                "success",
                            )
                        else:
                            click.echo(
                                f"✅ Post-fetch validation: {final_commits} commits, {final_batches} batches confirmed"
                            )

            # FINAL PRE-CLASSIFICATION CHECK: Ensure we have data before starting batch classifier
            with cache.get_session() as session:
                pre_classification_commits = (
                    session.query(CachedCommit)
                    .filter(
                        and_(
                            CachedCommit.timestamp >= start_date, CachedCommit.timestamp <= end_date
                        )
                    )
                    .count()
                )

                pre_classification_batches = (
                    session.query(DailyCommitBatch)
                    .filter(
                        and_(
                            DailyCommitBatch.date >= start_date.date(),
                            DailyCommitBatch.date <= end_date.date(),
                        )
                    )
                    .count()
                )

                if pre_classification_commits == 0:
                    error_msg = "❌ PRE-CLASSIFICATION CHECK FAILED: No commits available for batch classification"
                    if display:
                        display.print_status(error_msg, "error")
                    else:
                        click.echo(error_msg)
                    click.echo(
                        f"   📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    )
                    click.echo(
                        f"   🗃️ Database state: {pre_classification_commits} commits, {pre_classification_batches} batches"
                    )
                    click.echo(
                        "   💡 This indicates all previous validation and fetch steps failed to store any data"
                    )
                    raise click.ClickException(
                        "No data available for batch classification - cannot proceed"
                    )

                if pre_classification_batches == 0:
                    error_msg = "❌ PRE-CLASSIFICATION CHECK FAILED: No daily batches available for classification"
                    if display:
                        display.print_status(error_msg, "error")
                    else:
                        click.echo(error_msg)
                    click.echo(
                        f"   📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    )
                    click.echo(
                        f"   🗃️ Database state: {pre_classification_commits} commits, {pre_classification_batches} batches"
                    )
                    click.echo("   💡 Commits exist but no daily batches - batch creation failed")
                    raise click.ClickException(
                        "No batches available for classification - batch creation process failed"
                    )

            if display:
                display.print_status("Step 2: Batch classification...", "info")
            else:
                click.echo("🧠 Step 2: Batch classification...")

            # Use batch classifier instead of regular analyzer
            from .classification.batch_classifier import BatchCommitClassifier

            # Extract LLM config for batch classifier
            llm_config = {
                "enabled": cfg.analysis.llm_classification.enabled,
                "api_key": cfg.analysis.llm_classification.api_key,
                "model": cfg.analysis.llm_classification.model,
                "confidence_threshold": cfg.analysis.llm_classification.confidence_threshold,
                "max_tokens": cfg.analysis.llm_classification.max_tokens,
                "temperature": cfg.analysis.llm_classification.temperature,
                "timeout_seconds": cfg.analysis.llm_classification.timeout_seconds,
                "cache_duration_days": cfg.analysis.llm_classification.cache_duration_days,
                "enable_caching": cfg.analysis.llm_classification.enable_caching,
                "max_daily_requests": cfg.analysis.llm_classification.max_daily_requests,
            }

            batch_classifier = BatchCommitClassifier(
                cache_dir=cfg.cache.directory,
                llm_config=llm_config,
                batch_size=50,
                confidence_threshold=cfg.analysis.llm_classification.confidence_threshold,
                fallback_enabled=True,
            )

            # Get project keys from repositories
            project_keys = []
            for repo_config in repositories_to_analyze:
                project_key = repo_config.project_key or repo_config.name
                project_keys.append(project_key)

            # Run batch classification
            classification_result = batch_classifier.classify_date_range(
                start_date=start_date,
                end_date=end_date,
                project_keys=project_keys,
                force_reclassify=clear_cache,
            )

            if display:
                display.print_status(
                    f"✅ Batch classification completed: {classification_result['processed_batches']} batches, "
                    f"{classification_result['total_commits']} commits",
                    "success",
                )
            else:
                click.echo("   ✅ Batch classification completed:")
                click.echo(
                    f"      - Processed batches: {classification_result['processed_batches']}"
                )
                click.echo(f"      - Total commits: {classification_result['total_commits']}")

            # Display LLM usage statistics if available
            if hasattr(batch_classifier, "llm_classifier"):
                llm_stats = batch_classifier.llm_classifier.get_statistics()
                if llm_stats.get("api_calls_made", 0) > 0:
                    click.echo("\n📊 LLM Classification Statistics:")
                    click.echo(f"   - Model: {llm_stats.get('model', 'Unknown')}")
                    click.echo(f"   - API calls: {llm_stats.get('api_calls_made', 0)}")
                    click.echo(f"   - Total tokens: {llm_stats.get('total_tokens_used', 0):,}")
                    click.echo(f"   - Total cost: ${llm_stats.get('total_cost', 0):.4f}")
                    click.echo(
                        f"   - Avg tokens/call: {llm_stats.get('average_tokens_per_call', 0):.0f}"
                    )

                    # Show cache statistics if available
                    cache_stats = llm_stats.get("cache_statistics", {})
                    if cache_stats.get("active_entries", 0) > 0:
                        click.echo(
                            f"   - Cache hits: {cache_stats.get('active_entries', 0)} cached predictions"
                        )

            # Skip to report generation by setting empty collections but continue with the flow
            # The reports will read from the database instead of memory
            all_commits = []
            all_prs = []
            all_enrichments = {}
            branch_health_metrics = {}

            # Skip repository analysis loop by setting a flag
            skip_repository_analysis = True

            # Load classified commits from database for report generation
            if display:
                display.print_status("Loading classified commits from database...", "info")
            else:
                click.echo("📊 Loading classified commits from database...")

            # Load commits from cache for the analysis period
            with cache.get_session() as session:
                from sqlalchemy import and_

                from .models.database import CachedCommit

                # Query commits for the analysis period across all repositories
                cached_commits = (
                    session.query(CachedCommit)
                    .filter(
                        and_(
                            CachedCommit.timestamp >= start_date,
                            CachedCommit.timestamp <= end_date,
                            # Filter by project repositories if needed
                            CachedCommit.repo_path.in_(
                                [str(Path(repo.path)) for repo in repositories_to_analyze]
                            ),
                        )
                    )
                    .order_by(CachedCommit.timestamp.desc())
                    .all()
                )

                # Convert cached commits to the format expected by reports
                all_commits = []
                for cached_commit in cached_commits:
                    commit_dict = cache._commit_to_dict(cached_commit)
                    # Ensure required fields for report generation
                    if "project_key" not in commit_dict:
                        # Infer project key from repo path
                        repo_path = Path(cached_commit.repo_path)
                        # Find matching repository config
                        for repo_config in repositories_to_analyze:
                            if repo_config.path == repo_path:
                                commit_dict["project_key"] = (
                                    repo_config.project_key or repo_config.name
                                )
                                break
                        else:
                            commit_dict["project_key"] = repo_path.name

                    # Add canonical_id field needed for reports
                    # This will be properly resolved by identity_resolver later
                    commit_dict["canonical_id"] = cached_commit.author_email or "unknown"

                    all_commits.append(commit_dict)

                if display:
                    display.print_status(
                        f"Loaded {len(all_commits)} classified commits from database", "success"
                    )
                else:
                    click.echo(f"✅ Loaded {len(all_commits)} classified commits from database")

            # Process the loaded commits to generate required statistics
            # Update developer identities
            if display:
                display.print_status("Processing developer identities...", "info")
            else:
                click.echo("👥 Processing developer identities...")

            identity_resolver.update_commit_stats(all_commits)

            # Analyze ticket references using loaded commits
            if display:
                display.print_status("Analyzing ticket references...", "info")
            else:
                click.echo("🎫 Analyzing ticket references...")

            ticket_analysis = analyzer.ticket_extractor.analyze_ticket_coverage(
                all_commits, all_prs
            )

            # Calculate per-developer ticket coverage and get updated developer stats
            developer_ticket_coverage = (
                analyzer.ticket_extractor.calculate_developer_ticket_coverage(all_commits)
            )
            developer_stats = identity_resolver.get_developer_stats(
                ticket_coverage=developer_ticket_coverage
            )

            if display:
                display.print_status(
                    f"Identified {len(developer_stats)} unique developers", "success"
                )
            else:
                click.echo(f"   ✅ Identified {len(developer_stats)} unique developers")

        else:
            skip_repository_analysis = False
            # Initialize collections for traditional mode
            all_commits = []
            all_prs = []
            all_enrichments = {}
            branch_health_metrics = {}  # Store branch health metrics per repository

        # Analyze repositories (traditional mode or forced fetch)
        # Note: In batch mode, these are already populated from database

        if not skip_repository_analysis:
            for idx, repo_config in enumerate(repositories_to_analyze, 1):
                if display:
                    display.update_progress_task(
                        "repos",
                        description=f"Analyzing {repo_config.name}... ({idx}/{len(repositories_to_analyze)})",
                    )
                else:
                    click.echo(
                        f"\n📁 Analyzing {repo_config.name}... ({idx}/{len(repositories_to_analyze)})"
                    )

                # Check if repo exists, clone if needed
                if not repo_config.path.exists():
                    # Try to clone if we have a github_repo configured
                    if repo_config.github_repo and cfg.github.organization:
                        if display:
                            display.print_status(
                                f"Cloning {repo_config.github_repo} from GitHub...", "info"
                            )
                        else:
                            click.echo(f"   📥 Cloning {repo_config.github_repo} from GitHub...")
                        try:
                            # Ensure parent directory exists
                            repo_config.path.parent.mkdir(parents=True, exist_ok=True)

                            # Clone the repository
                            clone_url = f"https://github.com/{repo_config.github_repo}.git"
                            if cfg.github.token:
                                # Use token for authentication
                                clone_url = f"https://{cfg.github.token}@github.com/{repo_config.github_repo}.git"

                            # Try to clone with specified branch, fall back to default if it fails
                            try:
                                # Use subprocess for better control over git command
                                env = os.environ.copy()
                                env["GIT_TERMINAL_PROMPT"] = "0"
                                env["GIT_ASKPASS"] = ""
                                env["GCM_INTERACTIVE"] = "never"

                                # Build git clone command
                                cmd = ["git", "clone", "--config", "credential.helper="]
                                if repo_config.branch:
                                    cmd.extend(["-b", repo_config.branch])
                                cmd.extend([clone_url, str(repo_config.path)])

                                # Run with timeout to prevent hanging
                                result = subprocess.run(
                                    cmd,
                                    env=env,
                                    capture_output=True,
                                    text=True,
                                    timeout=30,  # 30 second timeout
                                )

                                if result.returncode != 0:
                                    raise git.GitCommandError(
                                        cmd, result.returncode, stderr=result.stderr
                                    )
                            except subprocess.TimeoutExpired:
                                if display:
                                    display.print_status(
                                        f"Clone timeout for {repo_config.github_repo} (authentication may have failed)",
                                        "error",
                                    )
                                else:
                                    click.echo(
                                        "   ❌ Clone timeout - likely authentication failure"
                                    )
                                continue
                            except git.GitCommandError as e:
                                error_str = str(e)
                                # Check for authentication failures
                                if any(
                                    x in error_str.lower()
                                    for x in ["authentication", "permission denied", "401", "403"]
                                ):
                                    if display:
                                        display.print_status(
                                            f"Authentication failed for {repo_config.github_repo}. Check GitHub token.",
                                            "error",
                                        )
                                    else:
                                        click.echo(
                                            "   ❌ Authentication failed. Check GitHub token."
                                        )
                                    continue
                                elif (
                                    repo_config.branch
                                    and "Remote branch" in error_str
                                    and "not found" in error_str
                                ):
                                    # Branch doesn't exist, try cloning without specifying branch
                                    if display:
                                        display.print_status(
                                            f"Branch '{repo_config.branch}' not found, using repository default",
                                            "warning",
                                        )
                                    else:
                                        click.echo(
                                            f"   ⚠️  Branch '{repo_config.branch}' not found, using repository default"
                                        )
                                    # Try again without branch specification
                                    cmd = [
                                        "git",
                                        "clone",
                                        "--config",
                                        "credential.helper=",
                                        clone_url,
                                        str(repo_config.path),
                                    ]
                                    result = subprocess.run(
                                        cmd, env=env, capture_output=True, text=True, timeout=30
                                    )
                                    if result.returncode != 0:
                                        raise git.GitCommandError(
                                            cmd, result.returncode, stderr=result.stderr
                                        ) from e
                                else:
                                    raise
                            if display:
                                display.print_status(
                                    f"Successfully cloned {repo_config.github_repo}", "success"
                                )
                            else:
                                click.echo(f"   ✅ Successfully cloned {repo_config.github_repo}")
                        except Exception as e:
                            if display:
                                display.print_status(f"Failed to clone repository: {e}", "error")
                            else:
                                click.echo(f"   ❌ Failed to clone repository: {e}")
                            continue
                    else:
                        if display:
                            display.print_status(
                                f"Repository path not found: {repo_config.path}", "error"
                            )
                        else:
                            click.echo(f"   ❌ Repository path not found: {repo_config.path}")
                        continue

                # Analyze repository
                try:
                    commits = analyzer.analyze_repository(
                        repo_config.path, start_date, repo_config.branch
                    )

                    # Add project key and resolve developer identities
                    for commit in commits:
                        # Use configured project key or fall back to inferred project
                        if repo_config.project_key and repo_config.project_key != "UNKNOWN":
                            commit["project_key"] = repo_config.project_key
                        else:
                            commit["project_key"] = commit.get("inferred_project", "UNKNOWN")

                        commit["canonical_id"] = identity_resolver.resolve_developer(
                            commit["author_name"], commit["author_email"]
                        )

                    all_commits.extend(commits)
                    if display:
                        display.print_status(f"Found {len(commits)} commits", "success")
                    else:
                        click.echo(f"   ✅ Found {len(commits)} commits")

                    # Analyze branch health
                    from .metrics.branch_health import BranchHealthAnalyzer

                    branch_analyzer = BranchHealthAnalyzer()
                    branch_metrics = branch_analyzer.analyze_repository_branches(
                        str(repo_config.path)
                    )
                    branch_health_metrics[repo_config.name] = branch_metrics

                    # Log branch health summary
                    health_summary = branch_metrics.get("summary", {})
                    health_indicators = branch_metrics.get("health_indicators", {})
                    if display:
                        display.print_status(
                            f"Branch health: {health_indicators.get('overall_health', 'unknown')} "
                            f"({health_summary.get('total_branches', 0)} branches, "
                            f"{health_summary.get('stale_branches', 0)} stale)",
                            "info",
                        )
                    else:
                        click.echo(
                            f"   📊 Branch health: {health_indicators.get('overall_health', 'unknown')} "
                            f"({health_summary.get('total_branches', 0)} branches, "
                            f"{health_summary.get('stale_branches', 0)} stale)"
                        )

                    # Enrich with integration data
                    enrichment = orchestrator.enrich_repository_data(
                        repo_config, commits, start_date
                    )
                    all_enrichments[repo_config.name] = enrichment

                    if enrichment["prs"]:
                        all_prs.extend(enrichment["prs"])
                        if display:
                            display.print_status(
                                f"Found {len(enrichment['prs'])} pull requests", "success"
                            )
                        else:
                            click.echo(f"   ✅ Found {len(enrichment['prs'])} pull requests")

                except Exception as e:
                    if display:
                        display.print_status(f"Error: {e}", "error")
                    else:
                        click.echo(f"   ❌ Error: {e}")
                finally:
                    if display:
                        display.update_progress_task("repos", advance=1)

        # Stop repository progress and clean up display
        if display:
            display.complete_progress_task("repos", "Repository analysis complete")
            display.stop_live_display()

        if not all_commits:
            if display:
                display.show_error("No commits found in the specified period!")
            else:
                click.echo("\n❌ No commits found in the specified period!")
            return

        # NOTE: Bot exclusion moved to Phase 2 (reporting) to work with canonical_id
        # after identity resolution. This ensures manual identity mappings work correctly.
        # The exclusion logic now happens in report generators using canonical_id field.

        # Update developer statistics
        if display:
            display.print_status("Resolving developer identities...", "info")
        else:
            click.echo("\n👥 Resolving developer identities...")

        identity_resolver.update_commit_stats(all_commits)
        # Initialize empty ticket coverage - will be calculated after ticket analysis
        developer_stats = identity_resolver.get_developer_stats()

        if display:
            display.print_status(f"Identified {len(developer_stats)} unique developers", "success")
        else:
            click.echo(f"   ✅ Identified {len(developer_stats)} unique developers")

        # Check if we should run identity analysis
        should_check_identities = (
            not skip_identity_analysis
            and cfg.analysis.auto_identity_analysis  # Not explicitly skipped
            and not cfg.analysis.manual_identity_mappings  # Auto analysis is enabled
            and len(developer_stats) > 1  # No manual mappings  # Multiple developers to analyze
        )

        # Debug identity analysis decision
        if not should_check_identities:
            reasons = []
            if skip_identity_analysis:
                reasons.append("--skip-identity-analysis flag used")
            if not cfg.analysis.auto_identity_analysis:
                reasons.append("auto_identity_analysis disabled in config")
            if cfg.analysis.manual_identity_mappings:
                reasons.append(
                    f"manual identity mappings already exist ({len(cfg.analysis.manual_identity_mappings)} mappings)"
                )
            if len(developer_stats) <= 1:
                reasons.append(f"only {len(developer_stats)} developer(s) detected")

            if reasons and not skip_identity_analysis:
                if display:
                    display.print_status(f"Identity analysis skipped: {', '.join(reasons)}", "info")
                else:
                    click.echo(f"   ℹ️  Identity analysis skipped: {', '.join(reasons)}")

        if should_check_identities:
            from .identity_llm.analysis_pass import IdentityAnalysisPass

            try:
                # Check when we last prompted for identity suggestions
                last_prompt_file = cache_dir / ".identity_last_prompt"
                should_prompt = True

                if last_prompt_file.exists():
                    last_prompt_age = datetime.now() - datetime.fromtimestamp(
                        os.path.getmtime(last_prompt_file)
                    )
                    if last_prompt_age < timedelta(days=7):
                        should_prompt = False

                if should_prompt:
                    if display:
                        display.print_status("Analyzing developer identities...", "info")
                    else:
                        click.echo("\n🔍 Analyzing developer identities...")

                    analysis_pass = IdentityAnalysisPass(config)

                    # Run analysis
                    identity_cache_file = cache_dir / "identity_analysis_cache.yaml"
                    identity_result = analysis_pass.run_analysis(
                        all_commits, output_path=identity_cache_file, apply_to_config=False
                    )

                    if identity_result.clusters:
                        # Generate suggested configuration
                        suggested_config = analysis_pass.generate_suggested_config(identity_result)

                        # Show suggestions
                        if display:
                            display.print_status(
                                f"Found {len(identity_result.clusters)} potential identity clusters",
                                "warning",
                            )
                        else:
                            click.echo(
                                f"\n⚠️  Found {len(identity_result.clusters)} potential identity clusters:"
                            )

                        # Display all mappings
                        if suggested_config.get("analysis", {}).get("manual_identity_mappings"):
                            click.echo("\n📋 Suggested identity mappings:")
                            for mapping in suggested_config["analysis"]["manual_identity_mappings"]:
                                canonical = mapping["canonical_email"]
                                aliases = mapping.get("aliases", [])
                                if aliases:
                                    click.echo(f"   {canonical}")
                                    for alias in aliases:
                                        click.echo(f"     → {alias}")

                        # Check for bot exclusions
                        if suggested_config.get("exclude", {}).get("authors"):
                            bot_count = len(suggested_config["exclude"]["authors"])
                            click.echo(f"\n🤖 Found {bot_count} bot accounts to exclude:")
                            for bot in suggested_config["exclude"]["authors"][:5]:  # Show first 5
                                click.echo(f"   - {bot}")
                            if bot_count > 5:
                                click.echo(f"   ... and {bot_count - 5} more")

                        # Prompt user
                        click.echo("\n" + "─" * 60)
                        if click.confirm(
                            "Apply these identity mappings to your configuration?", default=True
                        ):
                            # Apply mappings to config
                            try:
                                # Reload config to ensure we have latest
                                with open(config) as f:
                                    config_data = yaml.safe_load(f)

                                # Update analysis section
                                if "analysis" not in config_data:
                                    config_data["analysis"] = {}
                                if "identity" not in config_data["analysis"]:
                                    config_data["analysis"]["identity"] = {}

                                # Apply manual mappings
                                existing_mappings = config_data["analysis"]["identity"].get(
                                    "manual_mappings", []
                                )
                                new_mappings = suggested_config.get("analysis", {}).get(
                                    "manual_identity_mappings", []
                                )

                                # Merge mappings
                                existing_emails = {
                                    m.get("canonical_email", "").lower() for m in existing_mappings
                                }
                                for new_mapping in new_mappings:
                                    if (
                                        new_mapping["canonical_email"].lower()
                                        not in existing_emails
                                    ):
                                        existing_mappings.append(new_mapping)

                                config_data["analysis"]["identity"][
                                    "manual_mappings"
                                ] = existing_mappings

                                # Apply bot exclusions
                                if suggested_config.get("exclude", {}).get("authors"):
                                    if "exclude" not in config_data["analysis"]:
                                        config_data["analysis"]["exclude"] = {}
                                    if "authors" not in config_data["analysis"]["exclude"]:
                                        config_data["analysis"]["exclude"]["authors"] = []

                                    existing_excludes = set(
                                        config_data["analysis"]["exclude"]["authors"]
                                    )
                                    for bot in suggested_config["exclude"]["authors"]:
                                        if bot not in existing_excludes:
                                            config_data["analysis"]["exclude"]["authors"].append(
                                                bot
                                            )

                                # Write updated config
                                with open(config, "w") as f:
                                    yaml.dump(
                                        config_data, f, default_flow_style=False, sort_keys=False
                                    )

                                if display:
                                    display.print_status(
                                        "Applied identity mappings to configuration", "success"
                                    )
                                else:
                                    click.echo("✅ Applied identity mappings to configuration")

                                # Reload configuration with new mappings
                                cfg = ConfigLoader.load(config)

                                # Re-initialize identity resolver with new mappings
                                identity_resolver = DeveloperIdentityResolver(
                                    cache_dir / "identities.db",
                                    similarity_threshold=cfg.analysis.similarity_threshold,
                                    manual_mappings=cfg.analysis.manual_identity_mappings,
                                )

                                # Re-resolve identities with new mappings
                                click.echo(
                                    "\n🔄 Re-resolving developer identities with new mappings..."
                                )
                                identity_resolver.update_commit_stats(all_commits)
                                # Use the previously calculated ticket coverage for accurate reporting
                                developer_stats = identity_resolver.get_developer_stats(
                                    ticket_coverage=developer_ticket_coverage
                                )

                                if display:
                                    display.print_status(
                                        f"Consolidated to {len(developer_stats)} unique developers",
                                        "success",
                                    )
                                else:
                                    click.echo(
                                        f"✅ Consolidated to {len(developer_stats)} unique developers"
                                    )

                            except Exception as e:
                                logger.error(f"Failed to apply identity mappings: {e}")
                                click.echo(f"❌ Failed to apply identity mappings: {e}")
                        else:
                            click.echo("⏭️  Skipping identity mapping suggestions")
                            click.echo("💡 Run with --analyze-identities to see suggestions again")

                        # Update last prompt timestamp
                        last_prompt_file.touch()

                    else:
                        if display:
                            display.print_status(
                                "No identity clusters found - all developers appear unique",
                                "success",
                            )
                        else:
                            click.echo(
                                "✅ No identity clusters found - all developers appear unique"
                            )

                        # Still update timestamp so we don't check again for 7 days
                        last_prompt_file.touch()

            except Exception as e:
                if display:
                    display.print_status(f"Identity analysis failed: {e}", "warning")
                else:
                    click.echo(f"⚠️  Identity analysis failed: {e}")
                logger.debug(f"Identity analysis error: {e}", exc_info=True)

        # Analyze tickets
        if display:
            display.print_status("Analyzing ticket references...", "info")
        else:
            click.echo("\n🎫 Analyzing ticket references...")

        # Use the analyzer's ticket extractor which may be ML-enhanced
        ticket_analysis = analyzer.ticket_extractor.analyze_ticket_coverage(all_commits, all_prs)

        # Calculate per-developer ticket coverage and update developer stats with accurate coverage
        developer_ticket_coverage = analyzer.ticket_extractor.calculate_developer_ticket_coverage(
            all_commits
        )
        developer_stats = identity_resolver.get_developer_stats(
            ticket_coverage=developer_ticket_coverage
        )

        for platform, count in ticket_analysis["ticket_summary"].items():
            if display:
                display.print_status(f"{platform.title()}: {count} unique tickets", "success")
            else:
                click.echo(f"   - {platform.title()}: {count} unique tickets")

        # Store daily metrics in database for reporting
        if display:
            display.print_status("Storing daily metrics for database-backed reporting...", "info")
        else:
            click.echo("\n💾 Storing daily metrics for database-backed reporting...")

        try:
            from .core.metrics_storage import DailyMetricsStorage

            # Initialize daily metrics storage
            metrics_db_path = cfg.cache.directory / "daily_metrics.db"
            metrics_storage = DailyMetricsStorage(metrics_db_path)

            # Get developer identities for storage
            developer_identities = {}
            for commit in all_commits:
                email = commit.get("author_email", "")
                canonical_id = commit.get("canonical_id", email)
                name = commit.get("author_name", "Unknown")
                developer_identities[email] = {
                    "canonical_id": canonical_id,
                    "name": name,
                    "email": email,
                }

            # Store metrics for each day in the analysis period
            current_date = start_date.date() if hasattr(start_date, "date") else start_date
            end_date_obj = (
                (start_date + timedelta(weeks=weeks)).date()
                if hasattr(start_date, "date")
                else (start_date + timedelta(weeks=weeks))
            )

            daily_commits = {}
            for commit in all_commits:
                commit_date = commit.get("timestamp")
                if commit_date:
                    if hasattr(commit_date, "date"):
                        commit_date = commit_date.date()
                    elif isinstance(commit_date, str):
                        from datetime import datetime as dt

                        commit_date = dt.fromisoformat(commit_date.replace("Z", "+00:00")).date()

                    if commit_date not in daily_commits:
                        daily_commits[commit_date] = []
                    daily_commits[commit_date].append(commit)

            total_records = 0
            for analysis_date, day_commits in daily_commits.items():
                if current_date <= analysis_date <= end_date_obj:
                    records = metrics_storage.store_daily_metrics(
                        analysis_date, day_commits, developer_identities
                    )
                    total_records += records

            if display:
                display.print_status(f"Stored {total_records} daily metric records", "success")
            else:
                click.echo(f"   ✅ Stored {total_records} daily metric records")

        except Exception as e:
            logger.error(f"Failed to store daily metrics: {e}")
            if display:
                display.print_status(f"Failed to store daily metrics: {e}", "warning")
            else:
                click.echo(f"   ⚠️  Failed to store daily metrics: {e}")

        # Perform qualitative analysis if enabled
        qualitative_results = []
        qual_cost_stats = None
        if (
            enable_qualitative or qualitative_only or (cfg.qualitative and cfg.qualitative.enabled)
        ) and cfg.qualitative:
            if display:
                display.print_status("Performing qualitative analysis...", "info")
            else:
                click.echo("\n🧠 Performing qualitative analysis...")

            try:
                from .models.database import Database
                from .qualitative import QualitativeProcessor

                # Initialize qualitative analysis components
                qual_db = Database(cfg.cache.directory / "qualitative.db")
                qual_processor = QualitativeProcessor(cfg.qualitative, qual_db)

                # Validate setup
                is_valid, issues = qual_processor.validate_setup()
                if not is_valid:
                    issue_msg = "Qualitative analysis setup issues:\n" + "\n".join(
                        f"• {issue}" for issue in issues
                    )
                    if issues:
                        issue_msg += "\n\n💡 Install dependencies: pip install spacy scikit-learn openai tiktoken"
                        issue_msg += (
                            "\n💡 Download spaCy model: python -m spacy download en_core_web_sm"
                        )

                    if display:
                        display.show_warning(issue_msg)
                    else:
                        click.echo("   ⚠️  Qualitative analysis setup issues:")
                        for issue in issues:
                            click.echo(f"      - {issue}")
                        if issues:
                            click.echo(
                                "   💡 Install dependencies: pip install spacy scikit-learn openai tiktoken"
                            )
                            click.echo(
                                "   💡 Download spaCy model: python -m spacy download en_core_web_sm"
                            )

                # Convert commits to qualitative format
                commits_for_qual = []
                for commit in all_commits:
                    # Handle both dict and object formats
                    if isinstance(commit, dict):
                        commit_dict = {
                            "hash": commit.get("hash") or commit.get("commit_hash"),
                            "message": commit.get("message"),
                            "author_name": commit.get("author_name"),
                            "author_email": commit.get("author_email"),
                            "timestamp": commit.get("timestamp"),
                            "files_changed": commit.get("files_changed") or [],
                            "insertions": commit.get("insertions", 0),
                            "deletions": commit.get("deletions", 0),
                            "branch": commit.get("branch", "main"),
                        }
                    else:
                        commit_dict = {
                            "hash": commit.hash,
                            "message": commit.message,
                            "author_name": commit.author_name,
                            "author_email": commit.author_email,
                            "timestamp": commit.timestamp,
                            "files_changed": commit.files_changed or [],
                            "insertions": commit.insertions,
                            "deletions": commit.deletions,
                            "branch": getattr(commit, "branch", "main"),
                        }
                    commits_for_qual.append(commit_dict)

                # Perform qualitative analysis with progress tracking
                if display:
                    display.start_live_display()
                    display.add_progress_task(
                        "qualitative",
                        "Analyzing commits with qualitative insights",
                        len(commits_for_qual),
                    )

                qualitative_results = qual_processor.process_commits(
                    commits_for_qual, show_progress=True
                )

                if display:
                    display.complete_progress_task("qualitative", "Qualitative analysis complete")
                    display.stop_live_display()
                    display.print_status(
                        f"Analyzed {len(qualitative_results)} commits with qualitative insights",
                        "success",
                    )
                else:
                    click.echo(
                        f"   ✅ Analyzed {len(qualitative_results)} commits with qualitative insights"
                    )

                # Get processing statistics and show them
                qual_stats = qual_processor.get_processing_statistics()

                # Extract cost statistics for later display
                if qual_stats and "llm_statistics" in qual_stats:
                    llm_stats = qual_stats["llm_statistics"]
                    if llm_stats.get("model_usage") == "available":
                        qual_cost_stats = llm_stats.get("cost_tracking", {})

                if display:
                    display.show_qualitative_stats(qual_stats)
                else:
                    processing_summary = qual_stats["processing_summary"]
                    click.echo(
                        f"   📈 Processing: {processing_summary['commits_per_second']:.1f} commits/sec"
                    )
                    click.echo(
                        f"   🎯 Methods: {processing_summary['method_breakdown']['cache']:.1f}% cached, "
                        f"{processing_summary['method_breakdown']['nlp']:.1f}% NLP, "
                        f"{processing_summary['method_breakdown']['llm']:.1f}% LLM"
                    )

                    if qual_stats["llm_statistics"]["model_usage"] == "available":
                        llm_stats = qual_stats["llm_statistics"]["cost_tracking"]
                        if llm_stats["total_cost"] > 0:
                            click.echo(f"   💰 LLM Cost: ${llm_stats['total_cost']:.4f}")

            except ImportError as e:
                error_msg = f"Qualitative analysis dependencies missing: {e}\n\n💡 Install with: pip install spacy scikit-learn openai tiktoken"
                if display:
                    display.show_error(error_msg)
                else:
                    click.echo(f"   ❌ Qualitative analysis dependencies missing: {e}")
                    click.echo("   💡 Install with: pip install spacy scikit-learn openai tiktoken")

                if not qualitative_only:
                    if display:
                        display.print_status("Continuing with standard analysis...", "info")
                    else:
                        click.echo("   ⏭️  Continuing with standard analysis...")
                else:
                    if display:
                        display.show_error(
                            "Cannot perform qualitative-only analysis without dependencies"
                        )
                    else:
                        click.echo(
                            "   ❌ Cannot perform qualitative-only analysis without dependencies"
                        )
                    return
            except Exception as e:
                error_msg = f"Qualitative analysis failed: {e}"
                if display:
                    display.show_error(error_msg)
                else:
                    click.echo(f"   ❌ Qualitative analysis failed: {e}")

                if qualitative_only:
                    if display:
                        display.show_error("Cannot continue with qualitative-only analysis")
                    else:
                        click.echo("   ❌ Cannot continue with qualitative-only analysis")
                    return
                else:
                    if display:
                        display.print_status("Continuing with standard analysis...", "info")
                    else:
                        click.echo("   ⏭️  Continuing with standard analysis...")
        elif enable_qualitative and not cfg.qualitative:
            warning_msg = "Qualitative analysis requested but not configured in config file\n\nAdd a 'qualitative:' section to your configuration"
            if display:
                display.show_warning(warning_msg)
            else:
                click.echo("\n⚠️  Qualitative analysis requested but not configured in config file")
                click.echo("   Add a 'qualitative:' section to your configuration")

        # Skip standard analysis if qualitative-only mode
        if qualitative_only:
            if display:
                display.print_status("Qualitative-only analysis completed!", "success")
            else:
                click.echo("\n✅ Qualitative-only analysis completed!")
            return

        # Aggregate PM platform data BEFORE report generation
        if not disable_pm and cfg.pm_integration and cfg.pm_integration.enabled:
            try:
                logger.debug("Starting PM data aggregation")
                aggregated_pm_data = {"issues": {}, "correlations": [], "metrics": {}}

                for _repo_name, enrichment in all_enrichments.items():
                    pm_data = enrichment.get("pm_data", {})
                    if pm_data:
                        # Aggregate issues by platform
                        for platform, issues in pm_data.get("issues", {}).items():
                            if platform not in aggregated_pm_data["issues"]:
                                aggregated_pm_data["issues"][platform] = []
                            aggregated_pm_data["issues"][platform].extend(issues)

                        # Aggregate correlations
                        aggregated_pm_data["correlations"].extend(pm_data.get("correlations", []))

                        # Use metrics from last repository with PM data (could be enhanced to merge)
                        if pm_data.get("metrics"):
                            aggregated_pm_data["metrics"] = pm_data["metrics"]

                # Only keep PM data if we actually have some
                if not aggregated_pm_data["correlations"] and not aggregated_pm_data["issues"]:
                    aggregated_pm_data = None

                logger.debug("PM data aggregation completed successfully")
            except Exception as e:
                logger.error(f"Error in PM data aggregation: {e}")
                click.echo(f"   ⚠️ Warning: PM data aggregation failed: {e}")
                aggregated_pm_data = None
        else:
            aggregated_pm_data = None

        # Generate reports
        if display:
            if generate_csv:
                display.print_status("Generating reports...", "info")
            else:
                display.print_status(
                    "Generating narrative report (CSV generation disabled)...", "info"
                )
        else:
            if generate_csv:
                click.echo("\n📊 Generating reports...")
            else:
                click.echo("\n📊 Generating narrative report (CSV generation disabled)...")

        logger.debug(f"Starting report generation with {len(all_commits)} commits")

        report_gen = CSVReportGenerator(
            anonymize=anonymize or cfg.output.anonymize_enabled,
            exclude_authors=cfg.analysis.exclude_authors,
            identity_resolver=identity_resolver,
        )
        analytics_gen = AnalyticsReportGenerator(
            anonymize=anonymize or cfg.output.anonymize_enabled,
            exclude_authors=cfg.analysis.exclude_authors,
            identity_resolver=identity_resolver,
        )

        # Collect generated report files for display
        generated_reports = []

        # Weekly metrics report (only if CSV generation is enabled)
        if generate_csv:
            weekly_report = (
                output / f'weekly_metrics_{datetime.now(timezone.utc).strftime("%Y%m%d")}.csv'
            )
            try:
                logger.debug("Starting weekly metrics report generation")
                report_gen.generate_weekly_report(
                    all_commits, developer_stats, weekly_report, weeks
                )
                logger.debug("Weekly metrics report completed successfully")
                generated_reports.append(weekly_report.name)
                if not display:
                    click.echo(f"   ✅ Weekly metrics: {weekly_report}")
            except Exception as e:
                logger.error(f"Error in weekly metrics report generation: {e}")
                with contextlib.suppress(builtins.BaseException):
                    handle_timezone_error(e, "weekly metrics report", all_commits, logger)

        # Developer activity summary with curve normalization (only if CSV generation is enabled)
        if generate_csv:
            activity_summary_report = (
                output
                / f'developer_activity_summary_{datetime.now(timezone.utc).strftime("%Y%m%d")}.csv'
            )
            try:
                logger.debug("Starting developer activity summary report generation")
                report_gen.generate_developer_activity_summary(
                    all_commits, developer_stats, all_prs, activity_summary_report, weeks
                )
                logger.debug("Developer activity summary report completed successfully")
                generated_reports.append(activity_summary_report.name)
                if not display:
                    click.echo(f"   ✅ Developer activity summary: {activity_summary_report}")
            except Exception as e:
                logger.error(f"Error in developer activity summary report generation: {e}")
                with contextlib.suppress(Exception):
                    handle_timezone_error(
                        e, "developer activity summary report", all_commits, logger
                    )
                click.echo(f"   ❌ Error generating weekly metrics report: {e}")
                click.echo(f"   🔍 Error type: {type(e).__name__}")
                click.echo(f"   📍 Error details: {str(e)}")
                import traceback

                traceback.print_exc()
                raise

        # Summary report (only if CSV generation is enabled)
        if generate_csv:
            summary_report = output / f'summary_{datetime.now().strftime("%Y%m%d")}.csv'
            try:
                report_gen.generate_summary_report(
                    all_commits,
                    all_prs,
                    developer_stats,
                    ticket_analysis,
                    summary_report,
                    aggregated_pm_data,
                )
                generated_reports.append(summary_report.name)
                if not display:
                    click.echo(f"   ✅ Summary stats: {summary_report}")
            except Exception as e:
                click.echo(f"   ❌ Error generating summary report: {e}")
                click.echo(f"   🔍 Error type: {type(e).__name__}")
                click.echo(f"   📍 Error details: {str(e)}")
                import traceback

                traceback.print_exc()
                raise

        # Developer report (only if CSV generation is enabled)
        if generate_csv:
            developer_report = output / f'developers_{datetime.now().strftime("%Y%m%d")}.csv'
            try:
                report_gen.generate_developer_report(developer_stats, developer_report)
                generated_reports.append(developer_report.name)
                if not display:
                    click.echo(f"   ✅ Developer stats: {developer_report}")
            except Exception as e:
                click.echo(f"   ❌ Error generating developer report: {e}")
                click.echo(f"   🔍 Error type: {type(e).__name__}")
                click.echo(f"   📍 Error details: {str(e)}")
                import traceback

                traceback.print_exc()
                raise

        # Untracked commits report (only if CSV generation is enabled)
        if generate_csv:
            untracked_commits_report = (
                output / f'untracked_commits_{datetime.now().strftime("%Y%m%d")}.csv'
            )
            try:
                report_gen.generate_untracked_commits_report(
                    ticket_analysis, untracked_commits_report
                )
                generated_reports.append(untracked_commits_report.name)
                if not display:
                    click.echo(f"   ✅ Untracked commits: {untracked_commits_report}")
            except Exception as e:
                click.echo(f"   ❌ Error generating untracked commits report: {e}")
                click.echo(f"   🔍 Error type: {type(e).__name__}")
                click.echo(f"   📍 Error details: {str(e)}")
                import traceback

                traceback.print_exc()
                raise

        # Weekly Categorization report (only if CSV generation is enabled)
        if generate_csv:
            weekly_categorization_report = (
                output / f'weekly_categorization_{datetime.now().strftime("%Y%m%d")}.csv'
            )
            try:
                logger.debug("Starting weekly categorization report generation")
                report_gen.generate_weekly_categorization_report(
                    all_commits, analyzer.ticket_extractor, weekly_categorization_report, weeks
                )
                logger.debug("Weekly categorization report completed successfully")
                generated_reports.append(weekly_categorization_report.name)
                if not display:
                    click.echo(f"   ✅ Weekly categorization: {weekly_categorization_report}")
            except Exception as e:
                logger.error(f"Error generating weekly categorization report: {e}")
                click.echo(f"   ⚠️ Warning: Weekly categorization report failed: {e}")

        # PM Correlations report (if PM data is available and CSV generation is enabled)
        if aggregated_pm_data and generate_csv:
            pm_correlations_report = (
                output / f'pm_correlations_{datetime.now().strftime("%Y%m%d")}.csv'
            )
            try:
                report_gen.generate_pm_correlations_report(
                    aggregated_pm_data, pm_correlations_report
                )
                generated_reports.append(pm_correlations_report.name)
                if not display:
                    click.echo(f"   ✅ PM correlations: {pm_correlations_report}")
            except Exception as e:
                click.echo(f"   ⚠️ Warning: PM correlations report failed: {e}")

        # Story Point Correlation report (only if CSV generation is enabled)
        if generate_csv:
            story_point_correlation_report = (
                output / f'story_point_correlation_{datetime.now().strftime("%Y%m%d")}.csv'
            )
            try:
                logger.debug("Starting story point correlation report generation")
                report_gen.generate_story_point_correlation_report(
                    all_commits, all_prs, aggregated_pm_data, story_point_correlation_report, weeks
                )
                logger.debug("Story point correlation report completed successfully")
                generated_reports.append(story_point_correlation_report.name)
                if not display:
                    click.echo(f"   ✅ Story point correlation: {story_point_correlation_report}")
            except Exception as e:
                logger.error(f"Error generating story point correlation report: {e}")
                click.echo(f"   ⚠️ Warning: Story point correlation report failed: {e}")

        # Activity distribution report (always generate data, optionally write CSV)
        activity_report = output / f'activity_distribution_{datetime.now().strftime("%Y%m%d")}.csv'
        try:
            logger.debug("Starting activity distribution report generation")
            analytics_gen.generate_activity_distribution_report(
                all_commits, developer_stats, activity_report
            )
            logger.debug("Activity distribution report completed successfully")
            if generate_csv:
                generated_reports.append(activity_report.name)
                if not display:
                    click.echo(f"   ✅ Activity distribution: {activity_report}")
        except Exception as e:
            logger.error(f"Error in activity distribution report generation: {e}")
            with contextlib.suppress(Exception):
                handle_timezone_error(e, "activity distribution report", all_commits, logger)
            click.echo(f"   ❌ Error generating activity distribution report: {e}")
            click.echo(f"   🔍 Error type: {type(e).__name__}")
            click.echo(f"   📍 Error details: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

        # Developer focus report (always generate data, optionally write CSV)
        focus_report = output / f'developer_focus_{datetime.now().strftime("%Y%m%d")}.csv'
        try:
            logger.debug("Starting developer focus report generation")
            analytics_gen.generate_developer_focus_report(
                all_commits, developer_stats, focus_report, weeks
            )
            logger.debug("Developer focus report completed successfully")
            if generate_csv:
                generated_reports.append(focus_report.name)
                if not display:
                    click.echo(f"   ✅ Developer focus: {focus_report}")
        except Exception as e:
            logger.error(f"Error in developer focus report generation: {e}")
            with contextlib.suppress(Exception):
                handle_timezone_error(e, "developer focus report", all_commits, logger)
            click.echo(f"   ❌ Error generating developer focus report: {e}")
            click.echo(f"   🔍 Error type: {type(e).__name__}")
            click.echo(f"   📍 Error details: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

        # Qualitative insights report (always generate data, optionally write CSV)
        insights_report = output / f'qualitative_insights_{datetime.now().strftime("%Y%m%d")}.csv'
        try:
            logger.debug("Starting qualitative insights report generation")
            analytics_gen.generate_qualitative_insights_report(
                all_commits, developer_stats, ticket_analysis, insights_report
            )
            logger.debug("Qualitative insights report completed successfully")
            if generate_csv:
                generated_reports.append(insights_report.name)
                if not display:
                    click.echo(f"   ✅ Qualitative insights: {insights_report}")
        except Exception as e:
            logger.error(f"Error in qualitative insights report generation: {e}")

        # Branch health report (only if CSV generation is enabled)
        if branch_health_metrics and generate_csv:
            from .reports.branch_health_writer import BranchHealthReportGenerator

            branch_health_gen = BranchHealthReportGenerator()

            branch_health_report = output / f'branch_health_{datetime.now().strftime("%Y%m%d")}.csv'
            try:
                logger.debug("Starting branch health report generation")
                branch_health_gen.generate_csv_report(branch_health_metrics, branch_health_report)
                logger.debug("Branch health report completed successfully")
                generated_reports.append(branch_health_report.name)
                if not display:
                    click.echo(f"   ✅ Branch health: {branch_health_report}")
            except Exception as e:
                logger.error(f"Error in branch health report generation: {e}")
                click.echo(f"   ❌ Error generating branch health report: {e}")

            # Detailed branch report
            detailed_branch_report = (
                output / f'branch_details_{datetime.now().strftime("%Y%m%d")}.csv'
            )
            try:
                branch_health_gen.generate_detailed_branch_report(
                    branch_health_metrics, detailed_branch_report
                )
                generated_reports.append(detailed_branch_report.name)
                if not display:
                    click.echo(f"   ✅ Branch details: {detailed_branch_report}")
            except Exception as e:
                logger.error(f"Error in detailed branch report generation: {e}")
                click.echo(f"   ❌ Error generating detailed branch report: {e}")

        # Weekly classification trends reports (developer and project) (only if CSV generation is enabled)
        if generate_csv:
            weekly_trends_writer = WeeklyTrendsWriter()
            date_suffix = f"_{datetime.now().strftime('%Y%m%d')}"

            try:
                logger.debug("Starting weekly classification trends report generation")
                trends_paths = weekly_trends_writer.generate_weekly_trends_reports(
                    all_commits, output, weeks, date_suffix
                )
                logger.debug("Weekly classification trends reports completed successfully")

                # Add both reports to generated reports list
                for report_type, report_path in trends_paths.items():
                    generated_reports.append(report_path.name)
                    if not display:
                        trend_type = "Developer" if "developer" in report_type else "Project"
                        click.echo(f"   ✅ {trend_type} weekly trends: {report_path}")

            except Exception as e:
                logger.error(f"Error in weekly classification trends report generation: {e}")
                click.echo(f"   ❌ Error generating weekly classification trends reports: {e}")
                click.echo(f"   🔍 Error type: {type(e).__name__}")
                click.echo(f"   📍 Error details: {str(e)}")

        # Weekly trends report (includes developer and project trends) (only if CSV generation is enabled)
        if generate_csv:
            trends_report = output / f'weekly_trends_{datetime.now().strftime("%Y%m%d")}.csv'
            try:
                logger.debug("Starting weekly trends report generation")
                analytics_gen.generate_weekly_trends_report(
                    all_commits, developer_stats, trends_report, weeks
                )
                logger.debug("Weekly trends report completed successfully")
                generated_reports.append(trends_report.name)

                # Check for additional trend files generated
                timestamp = trends_report.stem.split("_")[-1]
                dev_trends_file = output / f"developer_trends_{timestamp}.csv"
                proj_trends_file = output / f"project_trends_{timestamp}.csv"

                if dev_trends_file.exists():
                    generated_reports.append(dev_trends_file.name)
                if proj_trends_file.exists():
                    generated_reports.append(proj_trends_file.name)

                if not display:
                    click.echo(f"   ✅ Weekly trends: {trends_report}")
                    if dev_trends_file.exists():
                        click.echo(f"   ✅ Developer trends: {dev_trends_file}")
                    if proj_trends_file.exists():
                        click.echo(f"   ✅ Project trends: {proj_trends_file}")
            except Exception as e:
                logger.error(f"Error in weekly trends report generation: {e}")
                handle_timezone_error(e, "weekly trends report", all_commits, logger)
                click.echo(f"   ❌ Error generating weekly trends report: {e}")
                raise

        # Calculate DORA metrics
        try:
            logger.debug("Starting DORA metrics calculation")
            dora_calculator = DORAMetricsCalculator()
            dora_metrics = dora_calculator.calculate_dora_metrics(
                all_commits, all_prs, start_date, end_date
            )
            logger.debug("DORA metrics calculation completed successfully")
        except Exception as e:
            logger.error(f"Error in DORA metrics calculation: {e}")
            with contextlib.suppress(Exception):
                handle_timezone_error(e, "DORA metrics calculation", all_commits, logger)
            click.echo(f"   ❌ Error calculating DORA metrics: {e}")
            click.echo(f"   🔍 Error type: {type(e).__name__}")
            click.echo(f"   📍 Error details: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

        # Aggregate PR metrics
        try:
            logger.debug("Starting PR metrics aggregation")
            pr_metrics = {}
            for enrichment in all_enrichments.values():
                if enrichment.get("pr_metrics"):
                    # Combine metrics (simplified - in production would properly aggregate)
                    pr_metrics = enrichment["pr_metrics"]
                    break
            logger.debug("PR metrics aggregation completed successfully")
        except Exception as e:
            logger.error(f"Error in PR metrics aggregation: {e}")
            with contextlib.suppress(Exception):
                handle_timezone_error(e, "PR metrics aggregation", all_commits, logger)
            click.echo(f"   ❌ Error aggregating PR metrics: {e}")
            click.echo(f"   🔍 Error type: {type(e).__name__}")
            click.echo(f"   📍 Error details: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

        # Weekly velocity report (only if CSV generation is enabled)
        if generate_csv:
            weekly_velocity_report = (
                output / f'weekly_velocity_{datetime.now().strftime("%Y%m%d")}.csv'
            )
            try:
                logger.debug("Starting weekly velocity report generation")
                report_gen.generate_weekly_velocity_report(
                    all_commits, all_prs, weekly_velocity_report, weeks
                )
                logger.debug("Weekly velocity report completed successfully")
                generated_reports.append(weekly_velocity_report.name)
                if not display:
                    click.echo(f"   ✅ Weekly velocity: {weekly_velocity_report}")
            except Exception as e:
                logger.error(f"Error in weekly velocity report generation: {e}")
                with contextlib.suppress(Exception):
                    handle_timezone_error(e, "weekly velocity report", all_commits, logger)
                click.echo(f"   ❌ Error generating weekly velocity report: {e}")
                click.echo(f"   🔍 Error type: {type(e).__name__}")
                click.echo(f"   📍 Error details: {str(e)}")
                import traceback

                traceback.print_exc()
                raise

        # Weekly DORA metrics report (only if CSV generation is enabled)
        if generate_csv:
            weekly_dora_report = (
                output / f'weekly_dora_metrics_{datetime.now().strftime("%Y%m%d")}.csv'
            )
            try:
                logger.debug("Starting weekly DORA metrics report generation")
                report_gen.generate_weekly_dora_report(
                    all_commits, all_prs, weekly_dora_report, weeks
                )
                logger.debug("Weekly DORA metrics report completed successfully")
                generated_reports.append(weekly_dora_report.name)
                if not display:
                    click.echo(f"   ✅ Weekly DORA metrics: {weekly_dora_report}")
            except Exception as e:
                logger.error(f"Error in weekly DORA metrics report generation: {e}")
                with contextlib.suppress(Exception):
                    handle_timezone_error(e, "weekly DORA metrics report", all_commits, logger)
                click.echo(f"   ❌ Error generating weekly DORA metrics report: {e}")
                click.echo(f"   🔍 Error type: {type(e).__name__}")
                click.echo(f"   📍 Error details: {str(e)}")
                import traceback

                traceback.print_exc()
                raise

        # Generate markdown reports if enabled
        if "markdown" in cfg.output.formats:
            # Calculate date range for consistent filename formatting across all markdown reports
            date_range = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

            try:
                logger.debug("Starting narrative report generation")
                narrative_gen = NarrativeReportGenerator()

                # Load activity distribution data
                logger.debug("Loading activity distribution data")
                activity_df = pd.read_csv(activity_report)
                activity_data = cast(list[dict[str, Any]], activity_df.to_dict("records"))

                # Load focus data
                logger.debug("Loading focus data")
                focus_df = pd.read_csv(focus_report)
                focus_data = cast(list[dict[str, Any]], focus_df.to_dict("records"))

                # Load insights data
                logger.debug("Loading insights data")
                insights_df = pd.read_csv(insights_report)
                insights_data = cast(list[dict[str, Any]], insights_df.to_dict("records"))

                logger.debug("Generating narrative report")
                # Use pre-calculated date range for filename consistency
                narrative_report = output / f"narrative_report_{date_range}.md"

                # Try to generate ChatGPT summary
                chatgpt_summary = None

                openai_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
                if openai_key:
                    try:
                        # Create temporary comprehensive data for ChatGPT
                        from .qualitative.chatgpt_analyzer import ChatGPTQualitativeAnalyzer

                        logger.debug("Preparing data for ChatGPT analysis")

                        # Create minimal comprehensive data structure
                        comprehensive_data = {
                            "metadata": {
                                "analysis_weeks": weeks,
                                "generated_at": datetime.now(timezone.utc).isoformat(),
                            },
                            "executive_summary": {
                                "key_metrics": {
                                    "commits": {"total": len(all_commits)},
                                    "developers": {"total": len(developer_stats)},
                                    "lines_changed": {
                                        "total": sum(
                                            c.get("filtered_insertions", c.get("insertions", 0))
                                            + c.get("filtered_deletions", c.get("deletions", 0))
                                            for c in all_commits
                                        )
                                    },
                                    "story_points": {
                                        "total": sum(
                                            c.get("story_points", 0) or 0 for c in all_commits
                                        )
                                    },
                                    "ticket_coverage": {
                                        "percentage": ticket_analysis.get("commit_coverage_pct", 0)
                                    },
                                },
                                "health_score": {"overall": 75, "rating": "good"},  # Placeholder
                                "trends": {"velocity": {"direction": "stable"}},
                                "wins": [],
                                "concerns": [],
                            },
                            "developers": {},
                            "projects": {},
                        }

                        # Add developer data
                        for dev in developer_stats:  # All developers
                            dev_id = dev.get("canonical_id", dev.get("primary_email", "unknown"))
                            comprehensive_data["developers"][dev_id] = {
                                "identity": {"name": dev.get("primary_name", "Unknown")},
                                "summary": {
                                    "total_commits": dev.get("total_commits", 0),
                                    "total_story_points": dev.get("total_story_points", 0),
                                },
                                "projects": {},
                            }

                        analyzer = ChatGPTQualitativeAnalyzer(openai_key)
                        logger.debug("Generating ChatGPT qualitative summary")
                        chatgpt_summary = analyzer.generate_executive_summary(comprehensive_data)
                        logger.debug("ChatGPT summary generated successfully")

                    except Exception as e:
                        logger.warning(f"ChatGPT summary generation failed: {e}")
                        click.echo(f"   ⚠️ ChatGPT analysis skipped: {str(e)[:100]}")

                narrative_gen.generate_narrative_report(
                    all_commits,
                    all_prs,
                    developer_stats,
                    activity_data,
                    focus_data,
                    insights_data,
                    ticket_analysis,
                    pr_metrics,
                    narrative_report,
                    weeks,
                    aggregated_pm_data,
                    chatgpt_summary,
                    branch_health_metrics,
                    cfg.analysis.exclude_authors,
                    analysis_start_date=start_date,
                    analysis_end_date=end_date,
                )
                generated_reports.append(narrative_report.name)
                logger.debug("Narrative report generation completed successfully")
                if not display:
                    click.echo(f"   ✅ Narrative report: {narrative_report}")
            except Exception as e:
                logger.error(f"Error in narrative report generation: {e}")
                with contextlib.suppress(Exception):
                    handle_timezone_error(e, "narrative report generation", all_commits, logger)
                click.echo(f"   ❌ Error generating narrative report: {e}")
                click.echo(f"   🔍 Error type: {type(e).__name__}")
                click.echo(f"   📍 Error details: {str(e)}")
                import traceback

                traceback.print_exc()
                raise

        # Generate database-backed qualitative report
        if "markdown" in cfg.output.formats:
            try:
                logger.debug("Starting database-backed qualitative report generation")
                if display:
                    display.print_status("Generating database-backed qualitative report...", "info")
                else:
                    click.echo("   🔄 Generating database-backed qualitative report...")

                # Initialize database report generator
                from .core.metrics_storage import DailyMetricsStorage
                from .reports.database_report_generator import DatabaseReportGenerator

                # Use existing metrics storage from earlier in the pipeline
                metrics_db_path = cfg.cache.directory / "daily_metrics.db"
                metrics_storage = DailyMetricsStorage(metrics_db_path)
                db_report_gen = DatabaseReportGenerator(metrics_storage)

                # Generate report for the analysis period
                # Use pre-calculated date range for filename consistency
                db_qualitative_report = output / f"database_qualitative_report_{date_range}.md"
                analysis_start_date = (
                    start_date.date() if hasattr(start_date, "date") else start_date
                )
                analysis_end_date = (
                    (start_date + timedelta(weeks=weeks)).date()
                    if hasattr(start_date, "date")
                    else (start_date + timedelta(weeks=weeks))
                )

                report_stats = db_report_gen.generate_qualitative_report(
                    analysis_start_date, analysis_end_date, db_qualitative_report
                )

                generated_reports.append(db_qualitative_report.name)
                logger.debug(f"Database qualitative report generated: {report_stats}")

                if display:
                    display.print_status(
                        f"Generated report with {report_stats['total_records']} records, "
                        f"{report_stats['trends_calculated']} trends",
                        "success",
                    )
                else:
                    click.echo(f"   ✅ Database qualitative report: {db_qualitative_report}")
                    click.echo(
                        f"      📊 {report_stats['total_records']} records, {report_stats['trends_calculated']} trends analyzed"
                    )

            except Exception as e:
                logger.error(f"Error in database qualitative report generation: {e}")
                if display:
                    display.print_status(f"Database report generation failed: {e}", "warning")
                else:
                    click.echo(f"   ❌ Error generating database qualitative report: {e}")
                # Don't raise - this is a new feature and shouldn't break existing functionality

        # Generate comprehensive JSON export if enabled
        if "json" in cfg.output.formats:
            try:
                logger.debug("Starting comprehensive JSON export generation")
                click.echo("   🔄 Generating comprehensive JSON export...")
                json_report = (
                    output / f'comprehensive_export_{datetime.now().strftime("%Y%m%d")}.json'
                )

                # Initialize comprehensive JSON exporter
                json_exporter = ComprehensiveJSONExporter(anonymize=anonymize)

                # Enhanced qualitative analysis if available
                enhanced_analysis = None
                if qualitative_results:
                    try:
                        from .qualitative.enhanced_analyzer import EnhancedQualitativeAnalyzer

                        logger.debug("Running enhanced qualitative analysis")
                        enhanced_analyzer = EnhancedQualitativeAnalyzer()
                        enhanced_analysis = enhanced_analyzer.analyze_comprehensive(
                            commits=commits_for_qual,
                            qualitative_data=qualitative_results,
                            developer_stats=developer_stats,
                            project_metrics={
                                "ticket_analysis": ticket_analysis,
                                "pr_metrics": pr_metrics,
                                "enrichments": all_enrichments,
                            },
                            pm_data=aggregated_pm_data,
                            weeks_analyzed=weeks,
                        )
                        logger.debug("Enhanced qualitative analysis completed")
                    except Exception as e:
                        logger.warning(f"Enhanced qualitative analysis failed: {e}")
                        enhanced_analysis = None

                # Prepare project metrics
                project_metrics = {
                    "ticket_analysis": ticket_analysis,
                    "pr_metrics": pr_metrics,
                    "enrichments": all_enrichments,
                }

                # Generate comprehensive export
                logger.debug("Calling comprehensive JSON exporter")
                json_exporter.export_comprehensive_data(
                    commits=all_commits,
                    prs=all_prs,
                    developer_stats=developer_stats,
                    project_metrics=project_metrics,
                    dora_metrics=dora_metrics,
                    output_path=json_report,
                    weeks=weeks,
                    pm_data=aggregated_pm_data if aggregated_pm_data else None,
                    qualitative_data=qualitative_results if qualitative_results else None,
                    enhanced_qualitative_analysis=enhanced_analysis,
                )
                generated_reports.append(json_report.name)
                logger.debug("Comprehensive JSON export generation completed successfully")
                if not display:
                    click.echo(f"   ✅ Comprehensive JSON export: {json_report}")

                # Generate HTML report from JSON if requested
                # NOTE: HTML report generation temporarily disabled for database-backed reporting implementation
                # if "html" in cfg.output.formats:
                #     try:
                #         click.echo("   🔄 Generating HTML report...")
                #         from .reports.html_generator import HTMLReportGenerator
                #         html_report = output / f'gitflow_report_{datetime.now().strftime("%Y%m%d")}.html'
                #         logger.debug("Generating HTML report from JSON data")
                #
                #         # Read the JSON data we just wrote
                #         if not json_report.exists():
                #             # Check for alternative JSON file name
                #             alt_json = output / f'gitflow_export_{datetime.now().strftime("%Y%m%d")}.json'
                #             if alt_json.exists():
                #                 click.echo(f"   ⚠️ Using alternative JSON file: {alt_json.name}")
                #                 json_report = alt_json
                #
                #         with open(json_report) as f:
                #             import json
                #             json_data = json.load(f)
                #
                #         html_generator = HTMLReportGenerator()
                #         html_generator.generate_report(
                #             json_data=json_data,
                #             output_path=html_report,
                #             title=f"GitFlow Analytics Report - {datetime.now().strftime('%B %Y')}"
                #         )
                #         generated_reports.append(html_report.name)
                #         if not display:
                #             click.echo(f"   ✅ HTML report: {html_report}")
                #         logger.debug("HTML report generation completed successfully")
                #     except Exception as e:
                #         logger.error(f"Error generating HTML report: {e}")
                #         click.echo(f"   ⚠️ Warning: HTML report generation failed: {e}")
            except Exception as e:
                logger.error(f"Error in comprehensive JSON export generation: {e}")
                with contextlib.suppress(Exception):
                    handle_timezone_error(
                        e, "comprehensive JSON export generation", all_commits, logger
                    )
                click.echo(f"   ❌ Error generating comprehensive JSON export: {e}")
                click.echo(f"   🔍 Error type: {type(e).__name__}")
                click.echo(f"   📍 Error details: {str(e)}")
                import traceback

                traceback.print_exc()
                raise

        try:
            logger.debug("Starting final summary calculations")
            total_story_points = sum(c.get("story_points", 0) or 0 for c in all_commits)
            qualitative_count = len(qualitative_results) if qualitative_results else 0
            logger.debug("Final summary calculations completed successfully")

            # Show results summary
            if display:
                logger.debug("Starting display.show_analysis_summary")
                display.show_analysis_summary(
                    total_commits=len(all_commits),
                    total_prs=len(all_prs),
                    active_developers=len(developer_stats),
                    ticket_coverage=ticket_analysis["commit_coverage_pct"],
                    story_points=total_story_points,
                    qualitative_analyzed=qualitative_count,
                )
                logger.debug("display.show_analysis_summary completed successfully")

                # Show DORA metrics
                if dora_metrics:
                    logger.debug("Starting display.show_dora_metrics")
                    display.show_dora_metrics(dora_metrics)
                    logger.debug("display.show_dora_metrics completed successfully")

                # Show generated reports
                logger.debug("Starting display.show_reports_generated")
                display.show_reports_generated(output, generated_reports)
                logger.debug("display.show_reports_generated completed successfully")

                # Show LLM cost summary if cost tracking data is available
                if qual_cost_stats:
                    logger.debug("Starting display.show_llm_cost_summary")
                    display.show_llm_cost_summary(qual_cost_stats)
                    logger.debug("display.show_llm_cost_summary completed successfully")

                logger.debug("Starting display.print_status")
                display.print_status("Analysis complete!", "success")
                logger.debug("display.print_status completed successfully")

                # Display cache statistics
                logger.debug("Starting cache statistics display")
                try:
                    cache_stats = cache.get_cache_stats()

                    # Display cache performance summary
                    display.print_status("📊 Cache Performance Summary", "info")
                    display.print_status(
                        f"  Total requests: {cache_stats['total_requests']}", "info"
                    )
                    display.print_status(
                        f"  Cache hits: {cache_stats['cache_hits']} ({cache_stats['hit_rate_percent']:.1f}%)",
                        "info",
                    )
                    display.print_status(f"  Cache misses: {cache_stats['cache_misses']}", "info")

                    if cache_stats["time_saved_seconds"] > 0:
                        if cache_stats["time_saved_minutes"] >= 1:
                            display.print_status(
                                f"  Time saved: {cache_stats['time_saved_minutes']:.1f} minutes",
                                "success",
                            )
                        else:
                            display.print_status(
                                f"  Time saved: {cache_stats['time_saved_seconds']:.1f} seconds",
                                "success",
                            )

                    # Display cache storage info
                    display.print_status("💾 Cache Storage", "info")
                    display.print_status(
                        f"  Cached commits: {cache_stats['fresh_commits']}", "info"
                    )
                    if cache_stats["stale_commits"] > 0:
                        display.print_status(
                            f"  Stale commits: {cache_stats['stale_commits']}", "warning"
                        )
                    display.print_status(
                        f"  Database size: {cache_stats['database_size_mb']:.1f} MB", "info"
                    )

                    if cache_stats["debug_mode"]:
                        display.print_status("🔍 Debug mode active (GITFLOW_DEBUG=1)", "info")

                except Exception as e:
                    logger.error(f"Error displaying cache statistics: {e}")
                    display.print_status(
                        f"Warning: Could not display cache statistics: {e}", "warning"
                    )

                logger.debug("Cache statistics display completed")
        except Exception as e:
            logger.error(f"Error in final summary/display: {e}")
            with contextlib.suppress(Exception):
                handle_timezone_error(e, "final summary/display", all_commits, logger)
            click.echo(f"   ❌ Error in final summary/display: {e}")
            click.echo(f"   🔍 Error type: {type(e).__name__}")
            click.echo(f"   📍 Error details: {str(e)}")
            import traceback

            traceback.print_exc()
            raise
        else:
            # Print summary in simple format
            click.echo("\n📈 Analysis Summary:")
            click.echo(f"   - Total commits: {len(all_commits)}")
            click.echo(f"   - Total PRs: {len(all_prs)}")
            click.echo(f"   - Active developers: {len(developer_stats)}")
            click.echo(f"   - Ticket coverage: {ticket_analysis['commit_coverage_pct']:.1f}%")
            click.echo(f"   - Total story points: {total_story_points}")

            if dora_metrics:
                click.echo("\n🎯 DORA Metrics:")
                click.echo(
                    f"   - Deployment frequency: {dora_metrics['deployment_frequency']['category']}"
                )
                click.echo(f"   - Lead time: {dora_metrics['lead_time_hours']:.1f} hours")
                click.echo(f"   - Change failure rate: {dora_metrics['change_failure_rate']:.1f}%")
                click.echo(f"   - MTTR: {dora_metrics['mttr_hours']:.1f} hours")
                click.echo(f"   - Performance level: {dora_metrics['performance_level']}")

            # Show LLM cost summary if available
            if qual_cost_stats and qual_cost_stats.get("total_cost", 0) > 0:
                click.echo("\n🤖 LLM Usage Summary:")
                total_calls = qual_cost_stats.get("total_calls", 0)
                total_tokens = qual_cost_stats.get("total_tokens", 0)
                total_cost = qual_cost_stats.get("total_cost", 0)
                click.echo(
                    f"   - Qualitative Analysis: {total_calls:,} calls, {total_tokens:,} tokens (${total_cost:.4f})"
                )

                # Show budget info if available
                daily_budget = 5.0  # Default budget from CostTracker
                remaining = daily_budget - total_cost
                utilization = (total_cost / daily_budget) * 100 if daily_budget > 0 else 0
                click.echo(
                    f"   - Budget: ${daily_budget:.2f}, Remaining: ${remaining:.2f}, Utilization: {utilization:.1f}%"
                )

                # Show optimization suggestions if cost is significant
                if total_cost > 0.01:
                    model_usage = qual_cost_stats.get("model_usage", {})
                    expensive_models = ["anthropic/claude-3-opus", "openai/gpt-4"]
                    expensive_cost = sum(
                        model_usage.get(model, {}).get("cost", 0) for model in expensive_models
                    )

                    if expensive_cost > total_cost * 0.3:
                        click.echo(
                            "   💡 Cost Optimization: Consider using cheaper models (Claude Haiku, GPT-3.5) for routine tasks (save ~40%)"
                        )
            else:
                # Show note about token tracking when qualitative analysis is not configured
                if not (enable_qualitative or (cfg.qualitative and cfg.qualitative.enabled)):
                    click.echo(
                        "\n💡 Note: Token/cost tracking is only available with qualitative analysis enabled."
                    )
                    click.echo(
                        "   Add 'qualitative:' section to your config to enable detailed LLM cost tracking."
                    )

            # Display cache statistics in simple format
            try:
                cache_stats = cache.get_cache_stats()
                click.echo("\n📊 Cache Performance:")
                click.echo(f"   - Total requests: {cache_stats['total_requests']}")
                click.echo(
                    f"   - Cache hits: {cache_stats['cache_hits']} ({cache_stats['hit_rate_percent']:.1f}%)"
                )
                click.echo(f"   - Cache misses: {cache_stats['cache_misses']}")

                if cache_stats["time_saved_seconds"] > 0:
                    if cache_stats["time_saved_minutes"] >= 1:
                        click.echo(
                            f"   - Time saved: {cache_stats['time_saved_minutes']:.1f} minutes"
                        )
                    else:
                        click.echo(
                            f"   - Time saved: {cache_stats['time_saved_seconds']:.1f} seconds"
                        )

                click.echo(f"   - Cached commits: {cache_stats['fresh_commits']}")
                if cache_stats["stale_commits"] > 0:
                    click.echo(f"   - Stale commits: {cache_stats['stale_commits']}")
                click.echo(f"   - Database size: {cache_stats['database_size_mb']:.1f} MB")

                if cache_stats["debug_mode"]:
                    click.echo("   - Debug mode: ACTIVE (GITFLOW_DEBUG=1)")

            except Exception as e:
                click.echo(f"   Warning: Could not display cache statistics: {e}")

            click.echo(f"\n✅ Analysis complete! Reports saved to {output}")

    except click.ClickException:
        # Let Click handle its own exceptions
        raise
    except Exception as e:
        error_msg = str(e)

        # Check if this is already a formatted YAML configuration error
        if "❌ YAML configuration error" in error_msg or "❌ Configuration file" in error_msg:
            # This is already a user-friendly error, display it as-is
            if display:
                display.show_error(error_msg, show_debug_hint=False)
            else:
                click.echo(f"\n{error_msg}", err=True)
        else:
            # Use improved error handler for better suggestions
            ImprovedErrorHandler.handle_command_error(click.get_current_context(), e)

            # Still show rich display error if available
            if display and "--debug" not in sys.argv:
                display.show_error(error_msg, show_debug_hint=True)

        if "--debug" in sys.argv:
            raise
        sys.exit(1)


@cli.command(name="fetch")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option("--weeks", "-w", type=int, default=4, help="Number of weeks to fetch (default: 4)")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for cache (overrides config file)",
)
@click.option("--clear-cache", is_flag=True, help="Clear cache before fetching data")
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="none",
    help="Enable logging with specified level (default: none)",
)
@click.option(
    "--rich", is_flag=True, default=True, help="Use rich terminal output (default: enabled)"
)
def fetch(
    config: Path,
    weeks: int,
    output: Optional[Path],
    clear_cache: bool,
    log: str,
    rich: bool,
) -> None:
    """Fetch data from external platforms for enhanced analysis.

    \b
    This command retrieves data from:
    - Git repositories: Commits, branches, authors
    - GitHub: Pull requests, issues, reviews (if configured)
    - JIRA: Tickets, story points, sprint data (if configured)
    - ClickUp: Tasks, time tracking (if configured)

    \b
    The fetched data enhances reports with:
    - DORA metrics (deployment frequency, lead time)
    - Story point velocity and estimation accuracy
    - PR review turnaround times
    - Issue resolution metrics

    \b
    EXAMPLES:
      # Fetch last 4 weeks of data
      gitflow-analytics fetch -c config.yaml --weeks 4

      # Fetch fresh data, clearing old cache
      gitflow-analytics fetch -c config.yaml --clear-cache

      # Debug API connectivity issues
      gitflow-analytics fetch -c config.yaml --log DEBUG

    \b
    REQUIREMENTS:
      - API credentials in configuration or environment
      - Network access to platform APIs
      - Appropriate permissions for repositories/projects

    \b
    PERFORMANCE:
      - First fetch may take several minutes for large repos
      - Subsequent fetches use cache for unchanged data
      - Use --clear-cache to force fresh fetch
    """
    # Initialize display
    display = create_rich_display() if rich else None

    # Configure logging
    if log.upper() != "NONE":
        log_level = getattr(logging, log.upper())
        logging.basicConfig(
            level=log_level,
            format="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            handlers=[logging.StreamHandler(sys.stderr)],
            force=True,
        )
        logging.getLogger("gitflow_analytics").setLevel(log_level)
        logger = logging.getLogger(__name__)
        logger.info(f"Logging enabled at {log.upper()} level")
    else:
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger("gitflow_analytics").setLevel(logging.CRITICAL)
        logger = logging.getLogger(__name__)

    try:
        if display:
            display.show_header()

        # Load configuration
        if display:
            display.print_status(f"Loading configuration from {config}...", "info")
        else:
            click.echo(f"📋 Loading configuration from {config}...")

        cfg = ConfigLoader.load(config)

        # Override output directory if provided
        if output:
            cfg.cache.directory = output

        # Initialize cache
        cache = GitAnalysisCache(cfg.cache.directory)

        # Clear cache if requested
        if clear_cache:
            if display:
                display.print_status("Clearing cache...", "info")
            else:
                click.echo("🗑️  Clearing cache...")
            cache.clear_all()

        # Initialize data fetcher
        from .core.data_fetcher import GitDataFetcher

        data_fetcher = GitDataFetcher(
            cache=cache,
            branch_mapping_rules=getattr(cfg.analysis, "branch_mapping_rules", {}),
            allowed_ticket_platforms=getattr(
                cfg.analysis, "ticket_platforms", ["jira", "github", "clickup", "linear"]
            ),
            exclude_paths=getattr(cfg.analysis, "exclude_paths", None),
        )

        # Initialize integrations for ticket fetching
        orchestrator = IntegrationOrchestrator(cfg, cache)
        jira_integration = orchestrator.integrations.get("jira")

        # Discovery organization repositories if needed
        repositories_to_fetch = cfg.repositories
        if cfg.github.organization and not repositories_to_fetch:
            if display:
                display.print_status(
                    f"Discovering repositories from organization: {cfg.github.organization}", "info"
                )
            else:
                click.echo(
                    f"🔍 Discovering repositories from organization: {cfg.github.organization}"
                )
            try:
                # Use a 'repos' directory in the config directory for cloned repositories
                config_dir = Path(config).parent if config else Path.cwd()
                repos_dir = config_dir / "repos"
                discovered_repos = cfg.discover_organization_repositories(clone_base_path=repos_dir)
                repositories_to_fetch = discovered_repos

                if display:
                    display.print_status(
                        f"Found {len(discovered_repos)} repositories in organization", "success"
                    )
                    # Show repository discovery in structured format
                    repo_data = [
                        {
                            "name": repo.name,
                            "github_repo": repo.github_repo,
                            "exists": repo.path.exists(),
                        }
                        for repo in discovered_repos
                    ]
                    display.show_repository_discovery(repo_data)
                else:
                    click.echo(f"   ✅ Found {len(discovered_repos)} repositories in organization")
                    for repo in discovered_repos:
                        click.echo(f"      - {repo.name} ({repo.github_repo})")
            except Exception as e:
                if display:
                    display.show_error(f"Failed to discover repositories: {e}")
                else:
                    click.echo(f"   ❌ Failed to discover repositories: {e}")
                return

        # Calculate analysis period with week-aligned boundaries
        current_time = datetime.now(timezone.utc)

        # Calculate dates to use last N complete weeks (not including current week)
        # Get the start of current week, then go back 1 week to get last complete week
        current_week_start = get_week_start(current_time)
        last_complete_week_start = current_week_start - timedelta(weeks=1)

        # Start date is N weeks back from the last complete week
        start_date = last_complete_week_start - timedelta(weeks=weeks - 1)

        # End date is the end of the last complete week (last Sunday)
        end_date = get_week_end(last_complete_week_start + timedelta(days=6))

        # Progress tracking
        total_repos = len(repositories_to_fetch)
        processed_repos = 0
        total_commits = 0
        total_tickets = 0

        if display:
            display.print_status(f"Starting data fetch for {total_repos} repositories...", "info")
            display.print_status(
                f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "info",
            )
        else:
            click.echo(f"🔄 Starting data fetch for {total_repos} repositories...")
            click.echo(
                f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            )

        # Process each repository
        for repo_config in repositories_to_fetch:
            try:
                repo_path = Path(repo_config.path)
                project_key = repo_config.project_key or repo_path.name

                if display:
                    display.print_status(f"Fetching data for {project_key}...", "info")
                else:
                    click.echo(f"📦 Fetching data for {project_key}...")

                # Progress callback
                def progress_callback(message: str):
                    if display:
                        display.print_status(message, "info")
                    else:
                        click.echo(f"   {message}")

                # Fetch repository data
                result = data_fetcher.fetch_repository_data(
                    repo_path=repo_path,
                    project_key=project_key,
                    weeks_back=weeks,
                    branch_patterns=getattr(repo_config, "branch_patterns", None),
                    jira_integration=jira_integration,
                    progress_callback=progress_callback,
                    start_date=start_date,
                    end_date=end_date,
                )

                # Update totals
                total_commits += result["stats"]["total_commits"]
                total_tickets += result["stats"]["unique_tickets"]
                processed_repos += 1

                if display:
                    display.print_status(
                        f"✅ {project_key}: {result['stats']['total_commits']} commits, "
                        f"{result['stats']['unique_tickets']} tickets",
                        "success",
                    )
                else:
                    click.echo(
                        f"   ✅ {result['stats']['total_commits']} commits, {result['stats']['unique_tickets']} tickets"
                    )

            except Exception as e:
                logger.error(f"Error fetching data for {repo_config.path}: {e}")
                if display:
                    display.print_status(f"❌ Error fetching {project_key}: {e}", "error")
                else:
                    click.echo(f"   ❌ Error: {e}")
                continue

        # Show final summary
        if display:
            display.print_status(
                f"🎉 Data fetch completed: {processed_repos}/{total_repos} repositories, "
                f"{total_commits} commits, {total_tickets} tickets",
                "success",
            )
        else:
            click.echo("\n🎉 Data fetch completed!")
            click.echo(f"   📊 Processed: {processed_repos}/{total_repos} repositories")
            click.echo(f"   📝 Commits: {total_commits}")
            click.echo(f"   🎫 Tickets: {total_tickets}")
            click.echo(
                f"\n💡 Next step: Run 'gitflow-analytics analyze -c {config}' to classify the data"
            )

    except Exception as e:
        logger.error(f"Fetch command failed: {e}")
        error_msg = f"Data fetch failed: {e}"

        if display:
            display.show_error(error_msg, show_debug_hint=True)
        else:
            click.echo(f"\n❌ Error: {error_msg}", err=True)

        if "--debug" in sys.argv:
            raise
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def cache_stats(config: Path) -> None:
    """Display cache statistics and performance metrics.

    \b
    Shows detailed information about:
    - Cache hit/miss rates
    - Number of cached commits, PRs, and issues
    - Database size and storage usage
    - Time saved through caching
    - Stale entries that need refresh

    \b
    EXAMPLES:
      # Check cache status
      gitflow-analytics cache-stats -c config.yaml

    \b
    Use this to:
    - Monitor cache performance
    - Decide when to clear cache
    - Troubleshoot slow analyses
    """
    try:
        cfg = ConfigLoader.load(config)
        cache = GitAnalysisCache(cfg.cache.directory)

        stats = cache.get_cache_stats()

        click.echo("📊 Cache Statistics:")
        click.echo(f"   - Cached commits: {stats['cached_commits']}")
        click.echo(f"   - Cached PRs: {stats['cached_prs']}")
        click.echo(f"   - Cached issues: {stats['cached_issues']}")
        click.echo(f"   - Stale entries: {stats['stale_commits']}")

        # Calculate cache size
        cache_size = 0
        for root, _dirs, files in os.walk(cfg.cache.directory):
            for f in files:
                cache_size += os.path.getsize(os.path.join(root, f))

        click.echo(f"   - Cache size: {cache_size / 1024 / 1024:.1f} MB")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
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
    try:
        cfg = ConfigLoader.load(config)
        identity_resolver = DeveloperIdentityResolver(cfg.cache.directory / "identities.db")

        click.echo(f"🔄 Merging {dev2} into {dev1}...")
        identity_resolver.merge_identities(dev1, dev2)
        click.echo("✅ Identities merged successfully!")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command(name="discover-storypoint-fields")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def discover_storypoint_fields(config: Path) -> None:
    """Discover available story point fields in your PM platform (JIRA, ClickUp, etc.)."""
    try:
        cfg = ConfigLoader.load(config)

        # Check which PM platform is configured
        # Currently only JIRA is supported, but this can be extended for other platforms
        if not (cfg.pm and cfg.pm.jira and cfg.pm.jira.base_url):
            click.echo("❌ No PM platform configured. Currently supports:")
            click.echo("   • JIRA (via pm.jira section)")
            click.echo("   • Future: ClickUp, Azure DevOps, etc.")
            return

        # Initialize PM integration (currently JIRA)
        from .integrations.jira_integration import JIRAIntegration

        # Create minimal cache for integration
        cache = GitAnalysisCache(cfg.cache.directory)
        jira = JIRAIntegration(
            cfg.pm.jira.base_url,
            cfg.pm.jira.username,
            cfg.pm.jira.api_token,
            cache,
        )

        # Validate connection
        click.echo(f"🔗 Connecting to PM platform (JIRA) at {cfg.pm.jira.base_url}...")
        if not jira.validate_connection():
            click.echo("❌ Failed to connect to PM platform. Check your credentials.")
            return

        click.echo("✅ Connected successfully!\n")
        click.echo("🔍 Discovering fields with potential story point data...")

        fields = jira.discover_fields()

        if not fields:
            click.echo("No potential story point fields found.")
        else:
            click.echo(f"\nFound {len(fields)} potential story point fields:")
            click.echo("\nAdd these to your configuration under the PM platform section:")
            click.echo("```yaml")
            click.echo("# For JIRA:")
            click.echo("jira_integration:")
            click.echo("  story_point_fields:")
            for field_id, field_info in fields.items():
                click.echo(f'    - "{field_id}"  # {field_info["name"]}')
            click.echo("```")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--weeks", "-w", type=int, default=12, help="Number of weeks to analyze (default: 12)"
)
@click.option("--apply", is_flag=True, help="Apply suggestions to configuration")
def identities(config: Path, weeks: int, apply: bool) -> None:
    """Analyze and manage developer identity resolution.

    \b
    This command helps consolidate multiple email addresses and names
    that belong to the same developer. It uses intelligent analysis to:
    - Detect similar names (John Smith vs J. Smith)
    - Identify GitHub noreply addresses
    - Find bot accounts to exclude
    - Suggest identity mappings for your configuration

    \b
    EXAMPLES:
      # Analyze identities from last 12 weeks
      gitflow-analytics identities -c config.yaml --weeks 12

      # Auto-apply identity suggestions
      gitflow-analytics identities -c config.yaml --apply

    \b
    IDENTITY RESOLUTION PROCESS:
      1. Analyzes commit authors from recent history
      2. Groups similar identities using fuzzy matching
      3. Suggests consolidated mappings
      4. Updates configuration with approved mappings

    \b
    CONFIGURATION:
      Mappings are saved to 'analysis.identity.manual_mappings'
      Bot exclusions go to 'analysis.exclude.authors'
    """
    try:
        cfg = ConfigLoader.load(config)
        cache = GitAnalysisCache(cfg.cache.directory)

        # Get recent commits with week-aligned boundaries for exact N-week period
        current_time = datetime.now(timezone.utc)

        # Calculate dates to use last N complete weeks (not including current week)
        # Get the start of current week, then go back 1 week to get last complete week
        current_week_start = get_week_start(current_time)
        last_complete_week_start = current_week_start - timedelta(weeks=1)

        # Start date is N weeks back from the last complete week
        start_date = last_complete_week_start - timedelta(weeks=weeks - 1)

        # Prepare ML categorization config for analyzer
        ml_config = None
        if hasattr(cfg.analysis, "ml_categorization"):
            ml_config = {
                "enabled": cfg.analysis.ml_categorization.enabled,
                "min_confidence": cfg.analysis.ml_categorization.min_confidence,
                "semantic_weight": cfg.analysis.ml_categorization.semantic_weight,
                "file_pattern_weight": cfg.analysis.ml_categorization.file_pattern_weight,
                "hybrid_threshold": cfg.analysis.ml_categorization.hybrid_threshold,
                "cache_duration_days": cfg.analysis.ml_categorization.cache_duration_days,
                "batch_size": cfg.analysis.ml_categorization.batch_size,
                "enable_caching": cfg.analysis.ml_categorization.enable_caching,
                "spacy_model": cfg.analysis.ml_categorization.spacy_model,
            }

        # LLM classification configuration
        llm_config = {
            "enabled": cfg.analysis.llm_classification.enabled,
            "api_key": cfg.analysis.llm_classification.api_key,
            "model": cfg.analysis.llm_classification.model,
            "confidence_threshold": cfg.analysis.llm_classification.confidence_threshold,
            "max_tokens": cfg.analysis.llm_classification.max_tokens,
            "temperature": cfg.analysis.llm_classification.temperature,
            "timeout_seconds": cfg.analysis.llm_classification.timeout_seconds,
            "cache_duration_days": cfg.analysis.llm_classification.cache_duration_days,
            "enable_caching": cfg.analysis.llm_classification.enable_caching,
            "max_daily_requests": cfg.analysis.llm_classification.max_daily_requests,
            "domain_terms": cfg.analysis.llm_classification.domain_terms,
        }

        # Configure branch analysis
        branch_analysis_config = {
            "strategy": cfg.analysis.branch_analysis.strategy,
            "max_branches_per_repo": cfg.analysis.branch_analysis.max_branches_per_repo,
            "active_days_threshold": cfg.analysis.branch_analysis.active_days_threshold,
            "include_main_branches": cfg.analysis.branch_analysis.include_main_branches,
            "always_include_patterns": cfg.analysis.branch_analysis.always_include_patterns,
            "always_exclude_patterns": cfg.analysis.branch_analysis.always_exclude_patterns,
            "enable_progress_logging": cfg.analysis.branch_analysis.enable_progress_logging,
            "branch_commit_limit": cfg.analysis.branch_analysis.branch_commit_limit,
        }

        analyzer = GitAnalyzer(
            cache,
            branch_mapping_rules=cfg.analysis.branch_mapping_rules,
            allowed_ticket_platforms=getattr(cfg.analysis, "ticket_platforms", None),
            exclude_paths=cfg.analysis.exclude_paths,
            story_point_patterns=cfg.analysis.story_point_patterns,
            ml_categorization_config=ml_config,
            llm_config=llm_config,
            branch_analysis_config=branch_analysis_config,
        )

        click.echo("🔍 Analyzing repositories for developer identities...")

        all_commits = []
        for repo_config in cfg.repositories:
            if repo_config.path.exists():
                commits = analyzer.analyze_repository(
                    repo_config.path, start_date, repo_config.branch
                )
                all_commits.extend(commits)

        if not all_commits:
            click.echo("❌ No commits found in the specified period!")
            return

        click.echo(f"✅ Found {len(all_commits)} commits")

        from .identity_llm.analysis_pass import IdentityAnalysisPass

        analysis_pass = IdentityAnalysisPass(config)

        # Run analysis
        identity_report_path = (
            cfg.cache.directory / f'identity_analysis_{datetime.now().strftime("%Y%m%d")}.yaml'
        )
        identity_result = analysis_pass.run_analysis(
            all_commits, output_path=identity_report_path, apply_to_config=False
        )

        click.echo(f"\n📄 Analysis report saved to: {identity_report_path}")

        if identity_result.clusters:
            # Generate suggested configuration
            suggested_config = analysis_pass.generate_suggested_config(identity_result)

            # Show suggestions
            click.echo(f"\n⚠️  Found {len(identity_result.clusters)} potential identity clusters:")

            # Display all mappings
            if suggested_config.get("analysis", {}).get("manual_identity_mappings"):
                click.echo("\n📋 Suggested identity mappings:")
                for mapping in suggested_config["analysis"]["manual_identity_mappings"]:
                    canonical = mapping["canonical_email"]
                    aliases = mapping.get("aliases", [])
                    if aliases:
                        click.echo(f"   {canonical}")
                        for alias in aliases:
                            click.echo(f"     → {alias}")

            # Check for bot exclusions
            if suggested_config.get("exclude", {}).get("authors"):
                bot_count = len(suggested_config["exclude"]["authors"])
                click.echo(f"\n🤖 Found {bot_count} bot accounts to exclude:")
                for bot in suggested_config["exclude"]["authors"]:
                    click.echo(f"   - {bot}")

            # Apply if requested
            if apply or click.confirm(
                "\nApply these identity mappings to your configuration?", default=True
            ):
                analysis_pass._apply_to_config(identity_result)
                click.echo("✅ Applied identity mappings to configuration")

                # Clear the prompt timestamp
                last_prompt_file = cfg.cache.directory / ".identity_last_prompt"
                last_prompt_file.unlink(missing_ok=True)
        else:
            click.echo("✅ No identity clusters found - all developers appear unique")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
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
    try:
        cfg = ConfigLoader.load(config)
        identity_resolver = DeveloperIdentityResolver(cfg.cache.directory / "identities.db")

        developers = identity_resolver.get_developer_stats()

        if not developers:
            click.echo("No developers found. Run analysis first.")
            return

        click.echo("👥 Known Developers:")
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
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--weeks", "-w", type=int, default=12, help="Number of weeks to analyze (default: 12)"
)
@click.option(
    "--session-name", type=str, default=None, help="Optional name for the training session"
)
@click.option(
    "--min-examples",
    type=int,
    default=50,
    help="Minimum number of training examples required (default: 50)",
)
@click.option(
    "--validation-split",
    type=float,
    default=0.2,
    help="Fraction of data to use for validation (default: 0.2)",
)
@click.option(
    "--model-type",
    type=click.Choice(["random_forest", "svm", "naive_bayes"]),
    default="random_forest",
    help="Type of model to train (default: random_forest)",
)
@click.option(
    "--incremental", is_flag=True, help="Add to existing training data instead of starting fresh"
)
@click.option(
    "--save-training-data", is_flag=True, help="Save extracted training data as CSV for inspection"
)
@click.option("--clear-cache", is_flag=True, help="Clear cache before training")
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="INFO",
    help="Enable logging with specified level (default: INFO)",
)
def train(
    config: Path,
    weeks: int,
    session_name: Optional[str],
    min_examples: int,
    validation_split: float,
    model_type: str,
    incremental: bool,
    save_training_data: bool,
    clear_cache: bool,
    log: str,
) -> None:
    """Train custom ML models for improved commit classification.

    \b
    This command trains machine learning models on your repository's
    commit history to improve classification accuracy. The models learn:
    - Project-specific commit message patterns
    - Team coding conventions and terminology
    - Domain-specific keywords and concepts
    - File path patterns for different change types

    \b
    EXAMPLES:
      # Train on last 12 weeks of commits
      gitflow-analytics train -c config.yaml --weeks 12

      # Train with custom session name
      gitflow-analytics train -c config.yaml --session-name "q4-training"

      # Save training data for inspection
      gitflow-analytics train -c config.yaml --save-training-data

      # Incremental training on new data
      gitflow-analytics train -c config.yaml --incremental

    \b
    MODEL TYPES:
      - random_forest: Best general performance (default)
      - svm: Good for clear category boundaries
      - naive_bayes: Fast, works well with small datasets

    \b
    TRAINING PROCESS:
      1. Extracts commits with ticket references
      2. Fetches ticket types from PM platforms
      3. Maps ticket types to commit categories
      4. Trains model with cross-validation
      5. Saves model with performance metrics

    \b
    REQUIREMENTS:
      - PM platform integration configured
      - Minimum 50 commits with ticket references
      - scikit-learn and pandas dependencies
      - ~100MB disk space for model storage
    """

    # Configure logging
    if log.upper() != "NONE":
        log_level = getattr(logging, log.upper())
        logging.basicConfig(
            level=log_level,
            format="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            handlers=[logging.StreamHandler(sys.stderr)],
            force=True,
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Training logging enabled at {log.upper()} level")
    else:
        logging.getLogger().setLevel(logging.CRITICAL)
        logger = logging.getLogger(__name__)

    try:
        click.echo("🚀 GitFlow Analytics - Commit Classification Training")
        click.echo("=" * 60)

        # Load configuration
        click.echo(f"📋 Loading configuration from {config}...")
        cfg = ConfigLoader.load(config)

        # Validate PM integration is enabled
        if not cfg.pm_integration or not cfg.pm_integration.enabled:
            click.echo("❌ Error: PM integration must be enabled for training")
            click.echo("   Add PM platform configuration to your config file:")
            click.echo("   ")
            click.echo("   pm_integration:")
            click.echo("     enabled: true")
            click.echo("     platforms:")
            click.echo("       jira:")
            click.echo("         enabled: true")
            click.echo("         config: {...}")
            sys.exit(1)

        # Check if any PM platforms are configured
        active_platforms = [
            name for name, platform in cfg.pm_integration.platforms.items() if platform.enabled
        ]

        if not active_platforms:
            click.echo("❌ Error: No PM platforms are enabled")
            click.echo(
                f"   Configure at least one platform: {list(cfg.pm_integration.platforms.keys())}"
            )
            sys.exit(1)

        click.echo(f"✅ PM integration enabled with platforms: {', '.join(active_platforms)}")

        # Setup cache
        cache_dir = cfg.cache.directory
        if clear_cache:
            click.echo("🗑️  Clearing cache...")
            import shutil

            if cache_dir.exists():
                shutil.rmtree(cache_dir)

        cache = GitAnalysisCache(cache_dir, ttl_hours=cfg.cache.ttl_hours)

        # Initialize integrations
        click.echo("🔧 Initializing integrations...")
        orchestrator = IntegrationOrchestrator(cfg, cache)

        # Check PM orchestrator is available
        if not orchestrator.pm_orchestrator or not orchestrator.pm_orchestrator.is_enabled():
            click.echo("❌ Error: PM framework orchestrator failed to initialize")
            click.echo("   Check your PM platform configurations and credentials")
            sys.exit(1)

        click.echo(
            f"✅ PM framework initialized with {len(orchestrator.pm_orchestrator.get_active_platforms())} platforms"
        )

        # Get repositories to analyze
        repositories_to_analyze = cfg.repositories
        if cfg.github.organization and not repositories_to_analyze:
            click.echo(f"🔍 Discovering repositories from organization: {cfg.github.organization}")
            try:
                config_dir = Path(config).parent if config else Path.cwd()
                repos_dir = config_dir / "repos"
                discovered_repos = cfg.discover_organization_repositories(clone_base_path=repos_dir)
                repositories_to_analyze = discovered_repos
                click.echo(f"✅ Found {len(discovered_repos)} repositories in organization")
            except Exception as e:
                click.echo(f"❌ Failed to discover repositories: {e}")
                sys.exit(1)

        if not repositories_to_analyze:
            click.echo("❌ Error: No repositories configured for analysis")
            click.echo(
                "   Configure repositories in your config file or use GitHub organization discovery"
            )
            sys.exit(1)

        click.echo(f"📁 Analyzing {len(repositories_to_analyze)} repositories")

        # Training configuration
        training_config = {
            "min_training_examples": min_examples,
            "validation_split": validation_split,
            "model_type": model_type,
            "save_training_data": save_training_data,
        }

        # Initialize trainer
        click.echo("🧠 Initializing training pipeline...")
        try:
            trainer = CommitClassificationTrainer(
                config=cfg, cache=cache, orchestrator=orchestrator, training_config=training_config
            )
        except ImportError as e:
            click.echo(f"❌ Error: {e}")
            click.echo("\n💡 Install training dependencies:")
            click.echo("   pip install scikit-learn")
            sys.exit(1)

        # Start training
        click.echo("\n🎯 Starting training session...")
        click.echo(f"   Time period: {weeks} weeks")
        click.echo(f"   Repositories: {len(repositories_to_analyze)}")
        click.echo(f"   Model type: {model_type}")
        click.echo(f"   Min examples: {min_examples}")
        click.echo(f"   Validation split: {validation_split:.1%}")

        start_time = time.time()

        try:
            # Calculate since date with week-aligned boundaries for exact N-week period
            current_time = datetime.now(timezone.utc)

            # Calculate dates to use last N complete weeks (not including current week)
            # Get the start of current week, then go back 1 week to get last complete week
            current_week_start = get_week_start(current_time)
            last_complete_week_start = current_week_start - timedelta(weeks=1)

            # Start date is N weeks back from the last complete week
            since = last_complete_week_start - timedelta(weeks=weeks - 1)

            results = trainer.train(
                repositories=repositories_to_analyze, since=since, session_name=session_name
            )

            training_time = time.time() - start_time

            # Display results
            click.echo("\n🎉 Training completed successfully!")
            click.echo("=" * 50)
            click.echo(f"Session ID: {results['session_id']}")
            click.echo(f"Training examples: {results['training_examples']}")
            click.echo(f"Model accuracy: {results['accuracy']:.1%}")
            click.echo(f"Training time: {training_time:.1f} seconds")
            click.echo(f"Model saved to: {trainer.classifier.model_path}")

            # Show per-category performance
            if "results" in results and "class_metrics" in results["results"]:
                click.echo("\n📊 Per-category performance:")
                for category, metrics in results["results"]["class_metrics"].items():
                    if isinstance(metrics, dict) and "precision" in metrics:
                        precision = metrics["precision"]
                        recall = metrics["recall"]
                        f1 = metrics["f1-score"]
                        support = metrics["support"]
                        click.echo(
                            f"   {category:12} - P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f} (n={support})"
                        )

            # Show PM platform coverage
            if orchestrator.pm_orchestrator:
                platforms = orchestrator.pm_orchestrator.get_active_platforms()
                if platforms:
                    click.echo(f"\n🔗 PM platforms used: {', '.join(platforms)}")

            # Save training data if requested
            if save_training_data:
                try:
                    training_data_path = trainer._export_training_data(results["session_id"])
                    click.echo(f"📄 Training data saved to: {training_data_path}")
                except Exception as e:
                    click.echo(f"⚠️ Warning: Failed to save training data: {e}")

            # Show next steps
            click.echo("\n✨ Next steps:")
            click.echo("   1. Review the training metrics above")
            click.echo("   2. Test the model with 'gitflow-analytics analyze --enable-ml'")
            click.echo("   3. Monitor model performance and retrain as needed")
            click.echo("   4. Use 'gitflow-analytics train-stats' to view training history")

        except ValueError as e:
            click.echo(f"\n❌ Training failed: {e}")
            if "Insufficient training data" in str(e):
                click.echo("\n💡 Suggestions to get more training data:")
                click.echo("   - Increase --weeks to analyze more history")
                click.echo("   - Ensure commits reference ticket IDs (e.g., PROJ-123)")
                click.echo("   - Check PM platform connectivity and permissions")
                click.echo("   - Lower --min-examples threshold (not recommended)")
            sys.exit(1)

        except Exception as e:
            click.echo(f"\n❌ Training failed with error: {e}")
            if log.upper() == "DEBUG":
                import traceback

                traceback.print_exc()
            sys.exit(1)

    except Exception as e:
        click.echo(f"\n❌ Configuration or setup error: {e}")
        if log.upper() == "DEBUG":
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command(name="help")
def show_help() -> None:
    """Show comprehensive help and usage guide.

    \b
    Displays detailed information about:
    - Getting started with GitFlow Analytics
    - Common workflows and use cases
    - Configuration file setup
    - Troubleshooting tips
    - Available commands and options
    """
    help_text = """
╔════════════════════════════════════════════════════════════════════╗
║                    GitFlow Analytics Help Guide                    ║
╚════════════════════════════════════════════════════════════════════╝

📚 QUICK START GUIDE
────────────────────
  1. Create a configuration file:
     cp config-sample.yaml myconfig.yaml
     
  2. Edit configuration with your repositories:
     repositories:
       - path: /path/to/repo
         branch: main
     
  3. Run your first analysis:
     gitflow-analytics -c myconfig.yaml --weeks 4
     
  4. View reports in the output directory

🔧 COMMON WORKFLOWS
──────────────────
  Weekly team report:
    gitflow-analytics -c config.yaml --weeks 1
    
  Monthly metrics with all formats:
    gitflow-analytics -c config.yaml --weeks 4 --generate-csv
    
  Identity resolution:
    gitflow-analytics identities -c config.yaml
    
  Fresh analysis (bypass cache):
    gitflow-analytics -c config.yaml --clear-cache
    
  Quick config validation:
    gitflow-analytics -c config.yaml --validate-only

⚙️ CONFIGURATION TIPS
────────────────────
  • Use environment variables: ${GITHUB_TOKEN}
  • Store credentials in .env file (same directory as config)
  • Enable ML categorization for better accuracy
  • Configure identity mappings to consolidate developers
  • Set appropriate cache TTL for your workflow

🐛 TROUBLESHOOTING
─────────────────
  Slow analysis?
    → Use caching (default) or reduce --weeks
    → Check cache stats: cache-stats command
    
  Wrong developer names?
    → Run: identities command
    → Add manual mappings to config
    
  Missing ticket references?
    → Check ticket_platforms configuration
    → Verify commit message format
    
  API errors?
    → Verify credentials in config or .env
    → Check rate limits
    → Use --log DEBUG for details

📊 REPORT TYPES
──────────────
  CSV Reports (--generate-csv):
    • developer_metrics: Individual statistics
    • weekly_metrics: Time-based trends
    • activity_distribution: Work patterns
    • untracked_commits: Process gaps
    
  Narrative Report (default):
    • Executive summary
    • Team composition analysis
    • Development patterns
    • Recommendations
    
  JSON Export:
    • Complete data for integration
    • All metrics and metadata

🔗 INTEGRATIONS
──────────────
  GitHub:
    • Pull requests and reviews
    • Issues and milestones
    • DORA metrics
    
  JIRA:
    • Story points and velocity
    • Sprint tracking
    • Issue types
    
  ClickUp:
    • Task tracking
    • Time estimates

📖 DOCUMENTATION
───────────────
  • README: https://github.com/yourusername/gitflow-analytics
  • Config Guide: docs/configuration.md
  • API Reference: docs/api.md
  • Contributing: docs/contributing.md

💡 TIPS
──────
  • Use --weeks wisely: smaller = faster
  • Enable rich output for better visuals
  • Save different configs for different teams
  • Use --anonymize for external reports
  • Regular identity resolution improves accuracy

For detailed command help: gitflow-analytics COMMAND --help
    """
    click.echo(help_text)


@cli.command(name="train-stats")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def training_statistics(config: Path) -> None:
    """Display ML model training statistics and performance history.

    \b
    Shows comprehensive training metrics:
    - Total training sessions and success rate
    - Model accuracy and validation scores
    - Training data statistics
    - Best performing model details
    - Recent training session results

    \b
    EXAMPLES:
      # View training statistics
      gitflow-analytics train-stats -c config.yaml

    \b
    Use this to:
    - Monitor model performance over time
    - Identify when retraining is needed
    - Compare different model versions
    """
    try:
        cfg = ConfigLoader.load(config)
        cache = GitAnalysisCache(cfg.cache.directory)

        # Initialize trainer to access statistics
        trainer = CommitClassificationTrainer(
            config=cfg, cache=cache, orchestrator=None, training_config={}  # Not needed for stats
        )

        stats = trainer.get_training_statistics()

        click.echo("📊 Training Statistics")
        click.echo("=" * 40)
        click.echo(f"Total sessions: {stats['total_sessions']}")
        click.echo(f"Completed sessions: {stats['completed_sessions']}")
        click.echo(f"Failed sessions: {stats['failed_sessions']}")
        click.echo(f"Total models: {stats['total_models']}")
        click.echo(f"Active models: {stats['active_models']}")
        click.echo(f"Training examples: {stats['total_training_examples']}")

        if stats["latest_session"]:
            latest = stats["latest_session"]
            click.echo("\n🕒 Latest Session:")
            click.echo(f"   ID: {latest['session_id']}")
            click.echo(f"   Status: {latest['status']}")
            if latest["accuracy"]:
                click.echo(f"   Accuracy: {latest['accuracy']:.1%}")
            if latest["training_time_minutes"]:
                click.echo(f"   Training time: {latest['training_time_minutes']:.1f} minutes")

        if stats["best_model"]:
            best = stats["best_model"]
            click.echo("\n🏆 Best Model:")
            click.echo(f"   ID: {best['model_id']}")
            click.echo(f"   Version: {best['version']}")
            click.echo(f"   Accuracy: {best['accuracy']:.1%}")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
