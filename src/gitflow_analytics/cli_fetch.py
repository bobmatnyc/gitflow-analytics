"""Fetch command for GitFlow Analytics CLI."""

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import click

from ._version import __version__
from .cli_formatting import handle_timezone_error
from .cli_utils import setup_logging
from .config import ConfigLoader
from .ui.progress_display import create_progress_display
from .utils.date_utils import get_week_end, get_week_start

logger = logging.getLogger(__name__)


@click.command(name="fetch")
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
    "--no-rich",
    is_flag=True,
    default=True,
    help="Disable rich terminal output (use simple text progress instead)",
)
def fetch(
    config: Path,
    weeks: int,
    output: Optional[Path],
    clear_cache: bool,
    log: str,
    no_rich: bool,
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
    # Create display - simple output by default for better compatibility, rich only when explicitly enabled
    display = (
        create_progress_display(style="simple" if no_rich else "rich", version=__version__)
        if not no_rich
        else None
    )

    logger = setup_logging(log, __name__)

    try:
        # Lazy imports
        from .core.cache import GitAnalysisCache
        from .integrations.orchestrator import IntegrationOrchestrator

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
            allowed_ticket_platforms=cfg.get_effective_ticket_platforms(),
            exclude_paths=getattr(cfg.analysis, "exclude_paths", None),
            exclude_merge_commits=cfg.analysis.exclude_merge_commits,
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

                # Progress callback for repository discovery
                def discovery_progress(repo_name, count):
                    if display:
                        display.print_status(f"   📦 Checking: {repo_name} ({count})", "info")
                    else:
                        click.echo(f"\r   📦 Checking repositories... {count}", nl=False)

                discovered_repos = cfg.discover_organization_repositories(
                    clone_base_path=repos_dir, progress_callback=discovery_progress
                )
                repositories_to_fetch = discovered_repos

                # Clear the progress line
                if not display:
                    click.echo("\r" + " " * 60 + "\r", nl=False)  # Clear line

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
                # For organization discovery, use branch patterns from analysis config
                # Default to ["*"] to analyze all branches when not specified
                branch_patterns = None
                if hasattr(cfg.analysis, "branch_patterns"):
                    branch_patterns = cfg.analysis.branch_patterns
                elif cfg.github.organization:
                    # For organization discovery, default to analyzing all branches
                    branch_patterns = ["*"]

                result = data_fetcher.fetch_repository_data(
                    repo_path=repo_path,
                    project_key=project_key,
                    weeks_back=weeks,
                    branch_patterns=branch_patterns,
                    jira_integration=jira_integration,
                    progress_callback=progress_callback,
                    start_date=start_date,
                    end_date=end_date,
                    # The standalone `fetch` command uses --clear-cache to force a full
                    # re-fetch; pass that flag through as the week-level force parameter.
                    force=clear_cache,
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
