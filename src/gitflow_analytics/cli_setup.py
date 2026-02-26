"""Setup, utility, and miscellaneous commands for GitFlow Analytics CLI."""

import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import click

from .config import ConfigLoader


def register_setup_commands(cli: click.Group) -> None:
    """Register setup/utility commands onto the CLI group."""
    cli.add_command(run_launcher, name="run")
    cli.add_command(install_command, name="install")
    cli.add_command(discover_storypoint_fields, name="discover-storypoint-fields")
    cli.add_command(cache_stats, name="cache-stats")
    cli.add_command(verify_activity, name="verify-activity")
    cli.add_command(show_help, name="help")


def _resolve_config_path(config: Optional[Path]) -> Optional[Path]:
    """Resolve configuration file path, offering to create if missing."""
    default_locations = [
        Path.cwd() / "config.yaml",
        Path.cwd() / ".gitflow-analytics.yaml",
        Path.home() / ".gitflow-analytics" / "config.yaml",
    ]

    if config:
        config_path = Path(config).resolve()
        if not config_path.exists():
            click.echo(f"Configuration file not found: {config_path}\n", err=True)

            if click.confirm("Would you like to create a new configuration?", default=True):
                click.echo("\nLaunching installation wizard...\n")

                from .cli_wizards.install_wizard import InstallWizard

                wizard = InstallWizard(output_dir=config_path.parent, skip_validation=False)
                wizard.config_filename = config_path.name
                success = wizard.run()

                if not success:
                    click.echo("\nInstallation wizard cancelled or failed.", err=True)
                    return None

                click.echo(f"\nConfiguration created: {config_path}")
                click.echo("\nReady to run analysis!\n")
                return config_path
            else:
                click.echo("\nCreate a configuration file with:")
                click.echo("   gitflow-analytics install")
                click.echo(f"\nOr manually create: {config_path}\n")
                return None

        return config_path

    click.echo("Looking for configuration files...\n")

    for location in default_locations:
        if location.exists():
            click.echo(f"Found configuration: {location}\n")
            return location

    click.echo("No configuration file found. Let's create one!\n")

    locations = [
        ("./config.yaml", "Current directory"),
        (str(Path.home() / ".gitflow-analytics" / "config.yaml"), "User directory"),
    ]

    click.echo("Where would you like to save the configuration?")
    for i, (path, desc) in enumerate(locations, 1):
        click.echo(f"  {i}. {path} ({desc})")
    click.echo("  3. Custom path")

    try:
        choice = click.prompt("\nSelect option", type=click.Choice(["1", "2", "3"]), default="1")
    except (click.exceptions.Abort, EOFError):
        click.echo("\nCancelled by user.")
        return None

    if choice == "1":
        config_path = Path.cwd() / "config.yaml"
    elif choice == "2":
        config_path = Path.home() / ".gitflow-analytics" / "config.yaml"
    else:
        try:
            custom_path = click.prompt("Enter configuration file path")
            config_path = Path(custom_path).expanduser().resolve()
        except (click.exceptions.Abort, EOFError):
            click.echo("\nCancelled by user.")
            return None

    click.echo(f"\nCreating configuration at: {config_path}")
    click.echo("Launching installation wizard...\n")

    from .cli_wizards.install_wizard import InstallWizard

    wizard = InstallWizard(output_dir=config_path.parent, skip_validation=False)
    wizard.config_filename = config_path.name
    success = wizard.run()

    if not success:
        click.echo("\nInstallation wizard cancelled or failed.", err=True)
        return None

    click.echo(f"\nConfiguration created: {config_path}")
    click.echo("\nReady to run analysis!\n")
    return config_path


@click.command(name="run")
@click.option(
    "--config",
    "-c",
    type=click.Path(path_type=Path),
    help="Path to configuration file (optional, will search for default)",
)
def run_launcher(config: Optional[Path]) -> None:
    """Interactive launcher for gitflow-analytics.

    \b
    This interactive command guides you through:
      - Repository selection (multi-select)
      - Analysis period configuration
      - Cache management
      - Identity analysis preferences
      - Preferences storage

    \b
    EXAMPLES:
      # Launch interactive mode
      gitflow-analytics run

      # Launch with specific config
      gitflow-analytics run -c config.yaml
    """
    try:
        config_path = _resolve_config_path(config)

        if not config_path:
            sys.exit(1)

        from .cli_wizards.run_launcher import run_interactive_launcher

        success = run_interactive_launcher(config_path=config_path)
        sys.exit(0 if success else 1)

    except (KeyboardInterrupt, click.exceptions.Abort):
        click.echo("\n\nLauncher cancelled by user.")
        sys.exit(130)
    except Exception as e:
        click.echo(f"Launcher failed: {e}", err=True)
        sys.exit(1)


@click.command(name="install")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=".",
    help="Directory for config files (default: current directory)",
)
@click.option(
    "--skip-validation",
    is_flag=True,
    help="Skip credential validation (for testing)",
)
def install_command(output_dir: Path, skip_validation: bool) -> None:
    """Interactive installation wizard for GitFlow Analytics.

    \b
    This wizard will guide you through setting up GitFlow Analytics:
    - GitHub credentials and repository configuration
    - Optional JIRA integration
    - Optional AI-powered insights (OpenRouter/ChatGPT)
    - Analysis settings and defaults

    \b
    EXAMPLES:
      # Run installation wizard in current directory
      gitflow-analytics install

      # Install to specific directory
      gitflow-analytics install --output-dir ./my-config
    """
    try:
        from .cli_wizards.install_wizard import InstallWizard

        wizard = InstallWizard(output_dir=Path(output_dir), skip_validation=skip_validation)
        success = wizard.run()
        sys.exit(0 if success else 1)

    except Exception as e:
        click.echo(f"Installation failed: {e}", err=True)
        sys.exit(1)


@click.command(name="discover-storypoint-fields")
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

        if not (cfg.pm and cfg.pm.jira and cfg.pm.jira.base_url):
            click.echo("No PM platform configured. Currently supports:")
            click.echo("   - JIRA (via pm.jira section)")
            click.echo("   - Future: ClickUp, Azure DevOps, etc.")
            return

        from .core.cache import GitAnalysisCache
        from .integrations.jira_integration import JIRAIntegration

        cache = GitAnalysisCache(cfg.cache.directory)
        jira = JIRAIntegration(
            cfg.pm.jira.base_url,
            cfg.pm.jira.username,
            cfg.pm.jira.api_token,
            cache,
        )

        click.echo(f"Connecting to PM platform (JIRA) at {cfg.pm.jira.base_url}...")
        if not jira.validate_connection():
            click.echo("Failed to connect to PM platform. Check your credentials.")
            return

        click.echo("Connected successfully!\n")
        click.echo("Discovering fields with potential story point data...")

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
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@click.command(name="cache-stats")
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
    """
    from .core.cache import GitAnalysisCache

    try:
        cfg = ConfigLoader.load(config)
        cache = GitAnalysisCache(cfg.cache.directory)

        stats = cache.get_cache_stats()

        click.echo("Cache Statistics:")
        click.echo(f"   - Cached commits: {stats['cached_commits']}")
        click.echo(f"   - Cached PRs: {stats['cached_prs']}")
        click.echo(f"   - Cached issues: {stats['cached_issues']}")
        click.echo(f"   - Stale entries: {stats['stale_commits']}")

        cache_size = 0
        for root, _dirs, files in os.walk(cfg.cache.directory):
            for f in files:
                cache_size += os.path.getsize(os.path.join(root, f))

        click.echo(f"   - Cache size: {cache_size / 1024 / 1024:.1f} MB")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@click.command(name="verify-activity")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--weeks",
    "-w",
    type=int,
    default=4,
    help="Number of weeks to analyze (default: 4)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Optional path to save the report",
)
def verify_activity(config: Path, weeks: int, output: Optional[Path]) -> None:
    """Verify day-by-day project activity without pulling code.

    \b
    This command helps verify if reports showing "No Activity" are accurate by:
    - Querying repositories for activity summaries
    - Showing day-by-day activity for each project
    - Listing all branches and their last activity dates
    - Highlighting days with zero activity
    - Using GitHub API for remote repos or git commands for local repos

    \b
    EXAMPLES:
      # Verify activity for last 4 weeks
      gitflow-analytics verify-activity -c config.yaml --weeks 4

      # Save report to file
      gitflow-analytics verify-activity -c config.yaml --weeks 8 -o activity_report.txt

    \b
    NOTE: This command does NOT pull or fetch code, it only queries metadata.
    """
    try:
        from .verify_activity import verify_activity_command

        verify_activity_command(config, weeks, output)

    except ImportError as e:
        click.echo(f"Missing dependency for activity verification: {e}")
        click.echo("Please install required packages: pip install tabulate")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during activity verification: {e}")
        traceback.print_exc()
        sys.exit(1)


@click.command(name="help")
def show_help() -> None:
    """Show comprehensive help and usage guide."""
    help_text = """
GitFlow Analytics Help Guide

QUICK START GUIDE
-----------------
  1. Create a configuration file:
     cp config-sample.yaml myconfig.yaml

  2. Edit configuration with your repositories:
     repositories:
       - path: /path/to/repo
         branch: main

  3. Run your first analysis:
     gitflow-analytics -c myconfig.yaml --weeks 4

  4. View reports in the output directory

COMMON WORKFLOWS
----------------
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

CONFIGURATION TIPS
------------------
  - Use environment variables: ${GITHUB_TOKEN}
  - Store credentials in .env file (same directory as config)
  - Enable ML categorization for better accuracy
  - Configure identity mappings to consolidate developers
  - Set appropriate cache TTL for your workflow

TROUBLESHOOTING
---------------
  Slow analysis?
    -> Use caching (default) or reduce --weeks
    -> Check cache stats: cache-stats command

  Wrong developer names?
    -> Run: identities command
    -> Add manual mappings to config

  Missing ticket references?
    -> Check ticket_platforms configuration
    -> Verify commit message format

  API errors?
    -> Verify credentials in config or .env
    -> Check rate limits
    -> Use --log DEBUG for details

REPORT TYPES
------------
  CSV Reports (--generate-csv):
    - developer_metrics: Individual statistics
    - weekly_metrics: Time-based trends
    - activity_distribution: Work patterns
    - untracked_commits: Process gaps

  Narrative Report (default):
    - Executive summary
    - Team composition analysis
    - Development patterns
    - Recommendations

  JSON Export:
    - Complete data for integration
    - All metrics and metadata

INTEGRATIONS
------------
  GitHub:
    - Pull requests and reviews
    - Issues and milestones
    - DORA metrics

  JIRA:
    - Story points and velocity
    - Sprint tracking
    - Issue types

  ClickUp:
    - Task tracking
    - Time estimates

DOCUMENTATION
-------------
  - README: https://github.com/bobmatnyc/gitflow-analytics
  - Config Guide: docs/configuration.md
  - API Reference: docs/api.md
  - Contributing: docs/contributing.md

TIPS
----
  - Use --weeks wisely: smaller = faster
  - Enable rich output for better visuals
  - Save different configs for different teams
  - Use --anonymize for external reports
  - Regular identity resolution improves accuracy

For detailed command help: gitflow-analytics COMMAND --help
    """
    click.echo(help_text)
