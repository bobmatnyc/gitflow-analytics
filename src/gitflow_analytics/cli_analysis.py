"""Analysis commands (analyze, fetch, collect, classify, report) for GitFlow Analytics CLI."""

import contextlib
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import click
import yaml

from ._version import __version__
from .cli_formatting import ImprovedErrorHandler, handle_timezone_error
from .cli_utils import setup_logging
from .config import ConfigLoader
from .config.errors import ConfigurationError
from .core.repo_cloner import CloneResult, clone_repository
from .ui.progress_display import create_progress_display
from .utils.date_utils import get_monday_aligned_start, get_week_end, get_week_start

logger = logging.getLogger(__name__)


def register_analysis_commands(cli: click.Group) -> None:
    """Register analysis-related commands onto the CLI group."""
    cli.add_command(analyze_subcommand, name="analyze")
    cli.add_command(fetch, name="fetch")
    cli.add_command(collect_command, name="collect")
    cli.add_command(classify_command, name="classify")
    cli.add_command(report_command, name="report")


@click.command(name="analyze")
@click.option(
    "--config",
    "-c",
    type=click.Path(path_type=Path),
    default="config.yaml",
    help="Path to YAML configuration file (default: config.yaml)",
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
    "--no-rich",
    is_flag=True,
    default=True,
    help="Disable rich terminal output (use simple text progress instead)",
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
    help=(
        "Use batch LLM classification on pre-fetched data (Step 2 of 2) - now the default behavior"
    ),
)
@click.option(
    "--force-fetch",
    "-f",
    is_flag=True,
    help=(
        "Force fetch fresh data even if cached data exists. "
        "Clears per-week cache entries and re-fetches all weeks. "
        "Alias: -f"
    ),
)
@click.option(
    "--progress-style",
    type=click.Choice(["rich", "simple", "auto"], case_sensitive=False),
    default="simple",
    help="Progress display style: rich (beautiful terminal UI), simple (tqdm), auto (detect)",
)
@click.option(
    "--cicd-metrics/--no-cicd-metrics",
    default=True,
    help="Enable CI/CD pipeline metrics collection (enabled by default, use --no-cicd-metrics to disable)",
)
@click.option(
    "--cicd-platforms",
    multiple=True,
    type=click.Choice(["github-actions"], case_sensitive=False),
    default=["github-actions"],
    help="CI/CD platforms to integrate (default: github-actions)",
)
@click.option(
    "--security-only",
    is_flag=True,
    help="Run only security analysis (skip productivity metrics)",
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
    no_rich: bool,
    log: str,
    skip_identity_analysis: bool,
    apply_identity_suggestions: bool,
    warm_cache: bool,
    validate_cache: bool,
    generate_csv: bool,
    use_batch_classification: bool,
    force_fetch: bool,
    progress_style: str,
    cicd_metrics: bool,
    cicd_platforms: tuple[str, ...],
    security_only: bool,
) -> None:
    """Run the complete analysis pipeline: collect → classify → report.

    \b
    This is the all-in-one command that internally runs all three pipeline
    stages in sequence:
      1. Collect  — fetch raw commits from git repositories into the cache
      2. Classify — run batch LLM classification on cached commits
      3. Report   — read classified commits and generate report files

    \b
    For finer-grained control you can run each stage independently:
      gfa collect -c config.yaml --weeks 4
      gfa classify -c config.yaml
      gfa report   -c config.yaml

    \b
    EXAMPLES:
      # Basic analysis of last 4 weeks (uses config.yaml by default)
      gitflow-analytics analyze --weeks 4

      # Use a custom configuration file
      gitflow-analytics analyze -c myconfig.yaml --weeks 4

      # Generate CSV reports with fresh data
      gitflow-analytics analyze --generate-csv --clear-cache

      # Quick validation of configuration
      gitflow-analytics analyze --validate-only

      # Analyze with qualitative insights
      gitflow-analytics analyze --enable-qualitative

      # Run only security analysis (requires security config)
      gitflow-analytics analyze --security-only

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
        no_rich=no_rich,
        log=log,
        skip_identity_analysis=skip_identity_analysis,
        apply_identity_suggestions=apply_identity_suggestions,
        warm_cache=warm_cache,
        validate_cache=validate_cache,
        generate_csv=generate_csv,
        use_batch_classification=use_batch_classification,
        force_fetch=force_fetch,
        progress_style=progress_style,
        cicd_metrics=cicd_metrics,
        cicd_platforms=cicd_platforms,
        security_only=security_only,
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
    no_rich: bool,
    log: str,
    skip_identity_analysis: bool,
    apply_identity_suggestions: bool,
    warm_cache: bool = False,
    validate_cache: bool = False,
    generate_csv: bool = False,
    use_batch_classification: bool = True,
    force_fetch: bool = False,
    progress_style: str = "simple",
    cicd_metrics: bool = False,
    cicd_platforms: tuple[str, ...] = ("github-actions",),
    security_only: bool = False,
) -> None:
    """Analyze Git repositories using configuration file."""

    # Lazy imports: Only load heavy dependencies when actually running analysis
    # This improves CLI startup time from ~2s to <100ms for commands like --help
    from .core.analyzer import GitAnalyzer
    from .core.cache import GitAnalysisCache
    from .core.identity import DeveloperIdentityResolver
    from .core.progress import get_progress_service

    # Pipeline stage functions (extracted from this function)
    from .core.analyze_pipeline import (
        ClassificationResult,
        CommitLoadResult,
        QualitativeResult,
        analyze_tickets_and_store_metrics,
        calculate_date_range,
        classify_commits_batch,
        discover_repositories,
        fetch_repositories_batch,
        generate_all_reports,
        load_and_validate_config,
        load_commits_from_db,
        resolve_developer_identities,
        run_qualitative_analysis,
        aggregate_pm_data,
        validate_batch_state,
    )
    from .core.analyze_pipeline_helpers import get_qualitative_config, is_qualitative_enabled

    try:
        from ._version import __version__

        version = __version__
    except ImportError:
        version = "1.3.11"

    # Initialize progress service with user's preference
    progress = get_progress_service(display_style=progress_style, version=version)

    # Initialize display - simple output by default for better compatibility
    # Create display - only create if rich output is explicitly enabled (--no-rich=False)
    display = (
        create_progress_display(style="simple" if no_rich else "rich", version=__version__)
        if not no_rich
        else None
    )

    logger = setup_logging(log, __name__)

    try:
        if display:
            display.show_header()

        # ------------------------------------------------------------------
        # STAGE 1 – Config loading & validation
        # ------------------------------------------------------------------
        if display:
            display.print_status(f"Loading configuration from {config}...", "info")
        else:
            click.echo(f"📋 Loading configuration from {config}...")

        try:
            from .config.errors import ConfigurationError

            cfg_result = load_and_validate_config(
                config=config,
                enable_pm=enable_pm,
                disable_pm=disable_pm,
                pm_platform=pm_platform,
                cicd_metrics=cicd_metrics,
                cicd_platforms=cicd_platforms,
            )
        except (FileNotFoundError, ConfigurationError) as e:
            error_msg = str(e)
            if "not found" in error_msg.lower() or isinstance(e, FileNotFoundError):
                friendly_msg = (
                    f"❌ Configuration file not found: {config}\n\n"
                    "To get started:\n"
                    "  1. Copy the sample: cp examples/config/config-sample.yaml config.yaml\n"
                    "  2. Edit config.yaml with your repository settings\n"
                    "  3. Run: gitflow-analytics -w 4\n\n"
                    "Or use the interactive installer: gitflow-analytics install"
                )
                if display:
                    display.print_status(friendly_msg, "error")
                else:
                    click.echo(friendly_msg, err=True)
                sys.exit(1)
            else:
                raise

        cfg = cfg_result.cfg
        if cfg_result.warnings:
            warning_msg = "Configuration warnings:\n" + "\n".join(
                f"• {w}" for w in cfg_result.warnings
            )
            if display:
                display.show_warning(warning_msg)
            else:
                click.echo("⚠️  Configuration warnings:")
                for warning in cfg_result.warnings:
                    click.echo(f"   - {warning}")

        # PM / CI-CD override feedback
        if disable_pm:
            if display:
                display.print_status("PM integration disabled via CLI flag", "info")
            else:
                click.echo("🚫 PM integration disabled via CLI flag")
        elif enable_pm:
            if display:
                display.print_status("PM integration enabled via CLI flag", "info")
            else:
                click.echo("📋 PM integration enabled via CLI flag")
        if pm_platform and cfg.pm_integration:
            if display:
                display.print_status(
                    f"PM integration limited to platforms: {', '.join(pm_platform)}", "info"
                )
            else:
                click.echo(f"📋 PM integration limited to platforms: {', '.join(pm_platform)}")
        if cicd_metrics:
            if display:
                display.print_status(
                    f"CI/CD metrics enabled for platforms: {', '.join(cicd_platforms)}", "info"
                )
            else:
                click.echo(f"🔄 CI/CD metrics enabled for platforms: {', '.join(cicd_platforms)}")

        # ------------------------------------------------------------------
        # STAGE 2 – GitHub authentication pre-flight
        # ------------------------------------------------------------------
        github_auth_needed = bool(
            (cfg.repositories and any(getattr(r, "github_repo", None) for r in cfg.repositories))
            or (cfg.github and cfg.github.organization)
        )

        if github_auth_needed:
            if display:
                display.print_status("Verifying GitHub authentication...", "info")
            else:
                click.echo("Verifying GitHub authentication...")
            from .core.analyze_pipeline import check_github_auth

            if not check_github_auth(cfg):
                if display:
                    display.print_status(
                        "GitHub authentication failed. Cannot proceed with analysis.", "error"
                    )
                else:
                    click.echo("GitHub authentication failed. Cannot proceed with analysis.")
                sys.exit(1)
        else:
            if display:
                display.print_status(
                    "Running in local-only mode (no GitHub features configured).", "info"
                )
            else:
                click.echo("Running in local-only mode (no GitHub features configured).")

        if validate_only:
            if not cfg_result.warnings:
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

        # ------------------------------------------------------------------
        # STAGE 3 – Output directory & display setup
        # ------------------------------------------------------------------
        if output is None:
            output = cfg.output.directory if cfg.output.directory else Path("./reports")
        output.mkdir(parents=True, exist_ok=True)

        if display:
            github_org = cfg.github.organization if cfg.github else None
            github_token_valid = bool(cfg.github and cfg.github.token)
            jira_configured = bool(cfg.jira and cfg.jira.base_url)
            display.show_configuration_status(
                config,
                github_org=github_org,
                github_token_valid=github_token_valid,
                jira_configured=jira_configured,
                jira_valid=jira_configured,
                analysis_weeks=weeks,
            )
            try:
                if hasattr(display, "start_live_display"):
                    display.start_live_display()
                elif hasattr(display, "start"):
                    display.start(total_items=100, description="Initializing GitFlow Analytics")
                if hasattr(display, "add_progress_task"):
                    display.add_progress_task("main", "Initializing GitFlow Analytics", 100)
            except Exception as e:
                click.echo(f"⚠️ Rich display initialization failed: {e}")
                click.echo("   Continuing with simple output mode...")
                display = None

        # ------------------------------------------------------------------
        # STAGE 4 – Cache initialisation / warm / validate / clear
        # ------------------------------------------------------------------
        cache_dir = cfg.cache.directory
        cache = GitAnalysisCache(cache_dir, ttl_hours=0 if no_cache else cfg.cache.ttl_hours)

        if clear_cache:
            if display and display._live:
                display.update_progress_task("main", description="Clearing cache...", completed=5)
            elif display:
                display.print_status("Clearing cache...", "info")
            else:
                click.echo("🗑️  Clearing cache...")
            try:
                cleared_counts = cache.clear_all_cache()
                msg = (
                    f"Cache cleared: {cleared_counts['commits']} commits, "
                    f"{cleared_counts['total']} total"
                )
                if display and display._live:
                    display.update_progress_task("main", description=msg, completed=10)
                elif display:
                    display.print_status(msg, "success")
                else:
                    click.echo(f"✅ {msg}")
            except Exception:
                import shutil

                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                if display:
                    display.print_status("Cache directory removed", "success")
                else:
                    click.echo("✅ Cache directory removed")

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
                for warning in validation_result.get("warnings", []):
                    display.print_status(f"  Warning: {warning}", "warning")
                stats = validation_result["stats"]
                display.print_status(
                    f"Cache contains {stats['total_commits']} commits", "info"
                )
                if stats.get("duplicates", 0) > 0:
                    display.print_status(
                        f"Found {stats['duplicates']} duplicate entries", "warning"
                    )
            else:
                if validation_result["is_valid"]:
                    click.echo("✅ Cache validation passed")
                else:
                    click.echo("❌ Cache validation failed:")
                    for issue in validation_result["issues"]:
                        click.echo(f"  Issue: {issue}")
                for warning in validation_result.get("warnings", []):
                    click.echo(f"  {warning}")
            if not warm_cache:
                return

        if warm_cache:
            if display:
                display.print_status("Warming cache with all repository commits...", "info")
            repo_paths = [rc.path for rc in cfg.repositories]
            warming_result = cache.warm_cache(repo_paths, weeks=weeks)
            if display:
                display.print_status("✅ Cache warming completed", "success")
                display.print_status(
                    f"  Repositories processed: {warming_result['repos_processed']}", "info"
                )
                display.print_status(
                    f"  Commits cached: {warming_result['commits_cached']}", "info"
                )
            else:
                click.echo(
                    f"✅ Cache warming completed in {warming_result['duration_seconds']:.1f}s"
                )
                click.echo(f"  Repositories: {warming_result['repos_processed']}")
                click.echo(f"  Newly cached: {warming_result['commits_cached']}")
                if warming_result.get("errors"):
                    for error in warming_result["errors"]:
                        click.echo(f"  {error}")
            if validate_only:
                return

        # ------------------------------------------------------------------
        # STAGE 5 – Security-only mode
        # ------------------------------------------------------------------
        if security_only:
            _run_security_only_analysis(
                cfg=cfg,
                cache=cache,
                cache_dir=cache_dir,
                config=config,
                no_cache=no_cache,
                output=output,
                display=display,
                weeks=weeks,
            )
            return

        # ------------------------------------------------------------------
        # STAGE 6 – Identity resolver initialisation
        # ------------------------------------------------------------------
        identity_db_path = cache_dir / "identities.db"
        try:
            identity_resolver = DeveloperIdentityResolver(
                identity_db_path,
                similarity_threshold=cfg.analysis.similarity_threshold,
                manual_mappings=cfg.analysis.manual_identity_mappings,
            )
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
            raise click.ClickException(f"Identity resolver initialization failed: {e}") from e

        # ------------------------------------------------------------------
        # STAGE 7 – Analyzer initialisation
        # ------------------------------------------------------------------
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
            allowed_ticket_platforms=cfg.get_effective_ticket_platforms(),
            exclude_paths=cfg.analysis.exclude_paths,
            story_point_patterns=cfg.analysis.story_point_patterns,
            ml_categorization_config=ml_config,
            llm_config=llm_config,
            branch_analysis_config=branch_analysis_config,
            exclude_merge_commits=cfg.analysis.exclude_merge_commits,
        )

        # ------------------------------------------------------------------
        # STAGE 8 – Repository discovery (org)
        # ------------------------------------------------------------------
        def _discovery_progress(repo_name: str, count: int) -> None:
            if display and display._live:
                display.update_progress_task(
                    "main",
                    description=f"🔍 Discovering: {repo_name} ({count} repos checked)",
                    completed=15 + min(count % 5, 4),
                )
            else:
                click.echo(f"\r   📦 Checking repositories... {count}", nl=False)

        if display and display._live:
            display.update_progress_task(
                "main",
                description=(
                    f"🔍 Discovering repositories from organization: "
                    f"{cfg.github.organization}"
                    if cfg.github.organization
                    else "Preparing analysis"
                ),
                completed=15,
            )
        elif cfg.github.organization and not cfg.repositories:
            click.echo(
                f"🔍 Discovering repositories from organization: {cfg.github.organization}"
            )

        try:
            repositories_to_analyze = discover_repositories(cfg, config, _discovery_progress)
        except Exception as e:
            if display and display._live:
                display.update_progress_task(
                    "main",
                    description=f"❌ Failed to discover repositories: {e}",
                    completed=20,
                )
            else:
                click.echo(f"   ❌ Failed to discover repositories: {e}")
            return

        if not (display and display._live):
            click.echo("")  # clear progress line after discovery

        if display and display._live:
            display.update_progress_task(
                "main",
                description=f"Analyzing {len(repositories_to_analyze)} repositories",
                completed=25,
            )
            repo_list = [
                {"name": repo.name or repo.project_key or Path(repo.path).name, "status": "pending"}
                for repo in repositories_to_analyze
            ]
            display.initialize_repositories(repo_list)
        else:
            click.echo(f"\n🚀 Analyzing {len(repositories_to_analyze)} repositories...")

        # ------------------------------------------------------------------
        # STAGE 9 – Date range calculation
        # ------------------------------------------------------------------
        date_range_result = calculate_date_range(weeks)
        start_date = date_range_result.start_date
        end_date = date_range_result.end_date

        if not (display and display._live):
            click.echo(
                f"   Period: {start_date.strftime('%Y-%m-%d')} to "
                f"{end_date.strftime('%Y-%m-%d')}"
            )

        # Generate config hash for cache validation
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
                "exclude_merge_commits": cfg.analysis.exclude_merge_commits,
            },
        )

        # Initialise variables for downstream stages
        developer_stats: list[dict] = []
        ticket_analysis: dict = {}
        all_commits: list[dict] = []
        all_prs: list = []
        all_enrichments: dict = {}
        branch_health_metrics: dict = {}

        # ------------------------------------------------------------------
        # STAGE 10 – Batch fetch + classify (two-step process)
        # ------------------------------------------------------------------
        if use_batch_classification:
            if display:
                display.add_progress_task(
                    "repos", "Checking cache and preparing analysis",
                    len(repositories_to_analyze),
                )
            else:
                click.echo("🔄 Using two-step process: fetch then classify...")

            # Step 1 – Fetch
            if display and display._live:
                display.update_progress_task(
                    "repos",
                    description=(
                        f"Step 1: Fetching data for {len(repositories_to_analyze)} repositories..."
                    ),
                    completed=15,
                )
            else:
                click.echo(
                    f"📥 Step 1: Fetching data for {len(repositories_to_analyze)} repositories..."
                )

            fetch_result = fetch_repositories_batch(
                cfg=cfg,
                cache=cache,
                repositories=repositories_to_analyze,
                start_date=start_date,
                end_date=end_date,
                weeks=weeks,
                config_hash=config_hash,
                force_fetch=force_fetch,
                progress_callback=lambda msg: (
                    display.print_status(f"   {msg}", "info") if display else None
                ),
            )

            if display and display._live:
                display.update_progress_task(
                    "repos",
                    description=(
                        f"Step 1 complete: {fetch_result.total_commits} commits, "
                        f"{fetch_result.total_tickets} tickets fetched"
                    ),
                    completed=100,
                )
                display.stop_live_display()
            else:
                click.echo(
                    f"📥 Step 1 complete: {fetch_result.total_commits} commits, "
                    f"{fetch_result.total_tickets} tickets fetched"
                )

            # Validate DB state
            validation_passed, stored_commits, existing_batches = validate_batch_state(
                cache=cache,
                start_date=start_date,
                end_date=end_date,
                total_commits_fetched=fetch_result.total_commits,
            )

            if stored_commits == 0:
                # No commits at all – generate empty reports gracefully
                empty_msg = (
                    f"No commits found in the analysis period "
                    f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}). "
                    "Generating empty reports."
                )
                if display:
                    display.print_status(empty_msg, "warning")
                else:
                    click.echo(f"   ℹ️  {empty_msg}")
                identity_resolver.update_commit_stats([])
                developer_ticket_coverage: dict = {}
                developer_stats = identity_resolver.get_developer_stats(
                    ticket_coverage=developer_ticket_coverage
                )
                ticket_analysis = analyzer.ticket_extractor.analyze_ticket_coverage(
                    [], [], display
                )
            else:
                if validation_passed:
                    if display:
                        display.print_status(
                            f"✅ Data validation passed: {stored_commits} commits, "
                            f"{existing_batches} batches ready",
                            "success",
                        )
                    else:
                        click.echo(
                            f"✅ Data validation passed: {stored_commits} commits, "
                            f"{existing_batches} batches ready"
                        )

                # Step 2 – Classify
                if display:
                    display.print_status("Step 2: Batch classification...", "info")
                    display.start_live_display()
                    display.add_progress_task(
                        "repos", f"Classifying batches", existing_batches or 1
                    )
                else:
                    click.echo("🧠 Step 2: Batch classification...")

                classification_result = classify_commits_batch(
                    cfg=cfg,
                    cache=cache,
                    repositories=repositories_to_analyze,
                    start_date=start_date,
                    end_date=end_date,
                    force_reclassify=clear_cache,
                )

                if display:
                    display.complete_progress_task("repos", "Batch classification complete")
                    display.stop_live_display()
                    display.print_status(
                        f"✅ Batch classification completed: "
                        f"{classification_result.processed_batches} batches, "
                        f"{classification_result.total_commits} commits",
                        "success",
                    )
                else:
                    click.echo("   ✅ Batch classification completed:")
                    click.echo(
                        f"      - Processed batches: {classification_result.processed_batches}"
                    )
                    click.echo(
                        f"      - Total commits: {classification_result.total_commits}"
                    )

                # Load classified commits from DB
                if display:
                    display.print_status(
                        "Loading classified commits from database...", "info"
                    )
                else:
                    click.echo("📊 Loading classified commits from database...")

                commit_load = load_commits_from_db(
                    cache=cache,
                    repositories=repositories_to_analyze,
                    start_date=start_date,
                    end_date=end_date,
                )
                all_commits = commit_load.all_commits
                all_prs = commit_load.all_prs
                all_enrichments = commit_load.all_enrichments
                branch_health_metrics = commit_load.branch_health_metrics

                if display and display._live:
                    display.update_progress_task(
                        "main",
                        description=f"Loaded {len(all_commits)} classified commits from database",
                        completed=85,
                    )
                else:
                    click.echo(
                        f"✅ Loaded {len(all_commits)} classified commits from database"
                    )

                # Identity resolution
                if display and display._live:
                    display.update_progress_task(
                        "main",
                        description="Processing developer identities...",
                        completed=90,
                    )
                else:
                    click.echo("👥 Processing developer identities...")

                identity_result = resolve_developer_identities(
                    identity_resolver=identity_resolver,
                    all_commits=all_commits,
                    ticket_extractor=analyzer.ticket_extractor,
                )
                developer_stats = identity_result.developer_stats
                developer_ticket_coverage = identity_result.developer_ticket_coverage

                # Ticket analysis
                if display and display._live:
                    display.update_progress_task(
                        "main",
                        description="Analyzing ticket references...",
                        completed=95,
                    )
                else:
                    click.echo("🎫 Analyzing ticket references...")

                ticket_analysis = analyzer.ticket_extractor.analyze_ticket_coverage(
                    all_commits, all_prs, display
                )
                developer_stats = identity_resolver.get_developer_stats(
                    ticket_coverage=analyzer.ticket_extractor.calculate_developer_ticket_coverage(
                        all_commits
                    )
                )

                if display and display._live:
                    display.update_progress_task(
                        "main",
                        description=f"Identified {len(developer_stats)} unique developers",
                        completed=98,
                    )
                else:
                    click.echo(f"   ✅ Identified {len(developer_stats)} unique developers")

        else:
            # ------------------------------------------------------------------
            # STAGE 10b – Traditional (non-batch) repository analysis
            # ------------------------------------------------------------------
            if display and display._live:
                display.add_progress_task(
                    "repos", "Processing repositories", len(repositories_to_analyze)
                )

            from .integrations.orchestrator import IntegrationOrchestrator

            orchestrator = IntegrationOrchestrator(cfg, cache)

            for idx, repo_config in enumerate(repositories_to_analyze, 1):
                if display:
                    display.update_progress_task(
                        "repos",
                        description=(
                            f"Analyzing {repo_config.name}... "
                            f"({idx}/{len(repositories_to_analyze)})"
                        ),
                    )
                else:
                    click.echo(
                        f"\n📁 Analyzing {repo_config.name}... "
                        f"({idx}/{len(repositories_to_analyze)})"
                    )

                if not repo_config.path.exists():
                    if repo_config.github_repo and cfg.github.organization:
                        def _clone_p(msg: str) -> None:
                            if display:
                                display.print_status(f"   {msg}", "info")
                            else:
                                click.echo(f"   {msg}")

                        clone_result = clone_repository(
                            repo_path=repo_config.path,
                            github_repo=repo_config.github_repo,
                            token=cfg.github.token if cfg.github else None,
                            branch=getattr(repo_config, "branch", None),
                            timeout_seconds=120,
                            max_retries=1,
                            progress_callback=_clone_p,
                        )
                        if not clone_result.success:
                            continue
                    else:
                        if display:
                            display.print_status(
                                f"Repository path not found: {repo_config.path}", "error"
                            )
                        else:
                            click.echo(
                                f"   ❌ Repository path not found: {repo_config.path}"
                            )
                        continue

                try:
                    commits = analyzer.analyze_repository(
                        repo_config.path, start_date, repo_config.branch
                    )
                    for commit in commits:
                        if repo_config.project_key and repo_config.project_key != "UNKNOWN":
                            commit["project_key"] = repo_config.project_key
                        else:
                            commit["project_key"] = commit.get("inferred_project", "UNKNOWN")
                        canonical_id = identity_resolver.resolve_developer(
                            commit["author_name"], commit["author_email"]
                        )
                        commit["canonical_id"] = canonical_id
                        commit["canonical_name"] = identity_resolver.get_canonical_name(
                            canonical_id
                        )
                    all_commits.extend(commits)
                    if display:
                        display.print_status(f"Found {len(commits)} commits", "success")
                    else:
                        click.echo(f"   ✅ Found {len(commits)} commits")

                    from .metrics.branch_health import BranchHealthAnalyzer

                    branch_metrics = BranchHealthAnalyzer().analyze_repository_branches(
                        str(repo_config.path)
                    )
                    branch_health_metrics[repo_config.name] = branch_metrics

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
                            click.echo(
                                f"   ✅ Found {len(enrichment['prs'])} pull requests"
                            )
                except Exception as e:
                    if display:
                        display.print_status(f"Error: {e}", "error")
                    else:
                        click.echo(f"   ❌ Error: {e}")
                finally:
                    if display:
                        display.update_progress_task("repos", advance=1)

            if display:
                display.complete_progress_task("repos", "Repository analysis complete")
                display.stop_live_display()

            if not all_commits:
                empty_msg = (
                    f"No commits found in the analysis period "
                    f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}). "
                    "Generating empty reports."
                )
                if display:
                    display.print_status(empty_msg, "warning")
                else:
                    click.echo(f"\n   ℹ️  {empty_msg}")
                identity_resolver.update_commit_stats([])
                developer_stats = identity_resolver.get_developer_stats(ticket_coverage={})
                ticket_analysis = analyzer.ticket_extractor.analyze_ticket_coverage(
                    [], [], display
                )
            else:
                # Identity resolution
                if display:
                    display.print_status("Resolving developer identities...", "info")
                else:
                    click.echo("\n👥 Resolving developer identities...")
                identity_resolver.update_commit_stats(all_commits)
                developer_stats = identity_resolver.get_developer_stats()

            if display:
                display.print_status(
                    f"Identified {len(developer_stats)} unique developers", "success"
                )
            else:
                click.echo(f"   ✅ Identified {len(developer_stats)} unique developers")

            # Auto identity analysis (traditional mode only)
            should_check_identities = (
                not skip_identity_analysis
                and cfg.analysis.auto_identity_analysis
                and not cfg.analysis.manual_identity_mappings
                and len(developer_stats) > 1
            )
            if should_check_identities:
                _run_identity_analysis(
                    config=config,
                    cfg=cfg,
                    cache_dir=cache_dir,
                    identity_resolver=identity_resolver,
                    all_commits=all_commits,
                    developer_stats=developer_stats,
                    display=display,
                    logger=logger,
                )

            # Ticket analysis
            if display:
                display.print_status("Analyzing ticket references...", "info")
            else:
                click.echo("\n🎫 Analyzing ticket references...")
            ticket_analysis = analyzer.ticket_extractor.analyze_ticket_coverage(
                all_commits, all_prs, display
            )
            developer_ticket_coverage = (
                analyzer.ticket_extractor.calculate_developer_ticket_coverage(all_commits)
            )
            developer_stats = identity_resolver.get_developer_stats(
                ticket_coverage=developer_ticket_coverage
            )
            for platform, count in ticket_analysis["ticket_summary"].items():
                if display:
                    display.print_status(
                        f"{platform.title()}: {count} unique tickets", "success"
                    )
                else:
                    click.echo(f"   - {platform.title()}: {count} unique tickets")

        # ------------------------------------------------------------------
        # STAGE 11 – Store daily metrics
        # ------------------------------------------------------------------
        if display:
            display.print_status(
                "Storing daily metrics for database-backed reporting...", "info"
            )
        else:
            click.echo("\n💾 Storing daily metrics for database-backed reporting...")

        ticket_result = analyze_tickets_and_store_metrics(
            analyzer=analyzer,
            identity_resolver=identity_resolver,
            all_commits=all_commits,
            all_prs=all_prs,
            display=display,
            cfg=cfg,
            start_date=start_date,
            weeks=weeks,
        )
        # Update with re-calculated ticket data
        ticket_analysis = ticket_result.ticket_analysis
        developer_stats = ticket_result.developer_stats

        # ------------------------------------------------------------------
        # STAGE 12 – Qualitative analysis
        # ------------------------------------------------------------------
        qualitative_result = QualitativeResult(
            results=[], cost_stats=None, commits_for_qual=[]
        )
        if (enable_qualitative or qualitative_only or is_qualitative_enabled(cfg)) and get_qualitative_config(cfg):
            if display:
                display.print_status("Performing qualitative analysis...", "info")
            else:
                click.echo("\n🧠 Performing qualitative analysis...")
            try:
                if display:
                    display.start_live_display()
                    display.add_progress_task(
                        "qualitative",
                        "Analyzing commits with qualitative insights",
                        len(all_commits),
                    )
                qualitative_result = run_qualitative_analysis(
                    cfg=cfg,
                    all_commits=all_commits,
                    enable_qualitative=enable_qualitative,
                    qualitative_only=qualitative_only,
                    display=display,
                )
                if display:
                    display.complete_progress_task(
                        "qualitative", "Qualitative analysis complete"
                    )
                    display.stop_live_display()
                    display.print_status(
                        f"Analyzed {len(qualitative_result.results)} commits with qualitative insights",
                        "success",
                    )
                else:
                    click.echo(
                        f"   ✅ Analyzed {len(qualitative_result.results)} commits with qualitative insights"
                    )
            except ImportError as e:
                if display:
                    display.show_error(f"Qualitative analysis dependencies missing: {e}")
                else:
                    click.echo(
                        f"   ❌ Qualitative analysis dependencies missing: {e}"
                    )
                if qualitative_only:
                    return
            except Exception as e:
                if display:
                    display.show_error(f"Qualitative analysis failed: {e}")
                else:
                    click.echo(f"   ❌ Qualitative analysis failed: {e}")
                if qualitative_only:
                    return
        elif enable_qualitative and not get_qualitative_config(cfg):
            warning_msg = (
                "Qualitative analysis requested but not configured in config file\n\n"
                "Add a 'qualitative:' section to your configuration"
            )
            if display:
                display.show_warning(warning_msg)
            else:
                click.echo("\n⚠️  Qualitative analysis requested but not configured")

        if qualitative_only:
            if display:
                display.print_status("Qualitative-only analysis completed!", "success")
            else:
                click.echo("\n✅ Qualitative-only analysis completed!")
            return

        # ------------------------------------------------------------------
        # STAGE 13 – PM data aggregation
        # ------------------------------------------------------------------
        aggregated_pm_data = aggregate_pm_data(
            cfg=cfg,
            all_enrichments=all_enrichments,
            disable_pm=disable_pm,
        )

        # ------------------------------------------------------------------
        # STAGE 14 – Report generation
        # ------------------------------------------------------------------
        if display:
            display.print_status(
                "Generating reports..." if generate_csv else
                "Generating narrative report (CSV generation disabled)...",
                "info",
            )
        else:
            click.echo(
                "\n📊 Generating reports..."
                if generate_csv
                else "\n📊 Generating narrative report (CSV generation disabled)..."
            )

        report_result = generate_all_reports(
            cfg=cfg,
            output=output,
            all_commits=all_commits,
            all_prs=all_prs,
            all_enrichments=all_enrichments,
            developer_stats=developer_stats,
            ticket_analysis=ticket_analysis,
            branch_health_metrics=branch_health_metrics,
            start_date=start_date,
            end_date=end_date,
            weeks=weeks,
            anonymize=anonymize,
            generate_csv=generate_csv,
            aggregated_pm_data=aggregated_pm_data,
            qualitative_result=qualitative_result,
            analyzer=analyzer,
            identity_resolver=identity_resolver,
        )

        # ------------------------------------------------------------------
        # STAGE 15 – Final summary display
        # ------------------------------------------------------------------
        try:
            total_story_points = sum(c.get("story_points", 0) or 0 for c in all_commits)
            dora_metrics = report_result.dora_metrics

            if display:
                display.show_analysis_summary(
                    len(all_commits),
                    len(developer_stats),
                    ticket_analysis.get("commits_with_tickets", 0),
                    prs=len(all_prs),
                )
                if dora_metrics:
                    display.show_dora_metrics(dora_metrics)
                display.show_reports_generated(output, report_result.generated_reports)
                if qualitative_result.cost_stats:
                    display.show_llm_cost_summary(qualitative_result.cost_stats)
                display.print_status("Analysis complete!", "success")

                # Cache statistics
                try:
                    cache_stats = cache.get_cache_stats()
                    display.print_status("📊 Cache Performance Summary", "info")
                    display.print_status(
                        f"  Total requests: {cache_stats['total_requests']}", "info"
                    )
                    display.print_status(
                        f"  Cache hits: {cache_stats['cache_hits']} "
                        f"({cache_stats['hit_rate_percent']:.1f}%)",
                        "info",
                    )
                    display.print_status(
                        f"  Cache misses: {cache_stats['cache_misses']}", "info"
                    )
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
                    display.print_status(
                        f"  Cached commits: {cache_stats['fresh_commits']}", "info"
                    )
                    if cache_stats.get("stale_commits", 0) > 0:
                        display.print_status(
                            f"  Stale commits: {cache_stats['stale_commits']}", "warning"
                        )
                    display.print_status(
                        f"  Database size: {cache_stats['database_size_mb']:.1f} MB", "info"
                    )
                except Exception as e:
                    logger.error("Error displaying cache statistics: %s", e)
            else:
                click.echo("\n📈 Analysis Summary:")
                click.echo(f"   - Total commits: {len(all_commits)}")
                click.echo(f"   - Total PRs: {len(all_prs)}")
                click.echo(f"   - Active developers: {len(developer_stats)}")
                click.echo(
                    f"   - Ticket coverage: {ticket_analysis.get('commit_coverage_pct', 0):.1f}%"
                )
                click.echo(f"   - Total story points: {total_story_points}")

                if dora_metrics:
                    click.echo("\n🎯 DORA Metrics:")
                    click.echo(
                        f"   - Deployment frequency: "
                        f"{dora_metrics['deployment_frequency']['category']}"
                    )
                    click.echo(
                        f"   - Lead time: {dora_metrics['lead_time_hours']:.1f} hours"
                    )
                    click.echo(
                        f"   - Change failure rate: "
                        f"{dora_metrics['change_failure_rate']:.1f}%"
                    )
                    click.echo(f"   - MTTR: {dora_metrics['mttr_hours']:.1f} hours")
                    click.echo(
                        f"   - Performance level: {dora_metrics['performance_level']}"
                    )

                qual_cost_stats = qualitative_result.cost_stats
                if qual_cost_stats and qual_cost_stats.get("total_cost", 0) > 0:
                    click.echo("\n🤖 LLM Usage Summary:")
                    total_calls = qual_cost_stats.get("total_calls", 0)
                    total_tokens = qual_cost_stats.get("total_tokens", 0)
                    total_cost = qual_cost_stats.get("total_cost", 0)
                    click.echo(
                        f"   - Qualitative Analysis: {total_calls:,} calls, "
                        f"{total_tokens:,} tokens (${total_cost:.4f})"
                    )
                    daily_budget = 5.0
                    remaining = daily_budget - total_cost
                    utilization = (total_cost / daily_budget) * 100 if daily_budget > 0 else 0
                    click.echo(
                        f"   - Budget: ${daily_budget:.2f}, Remaining: ${remaining:.2f}, "
                        f"Utilization: {utilization:.1f}%"
                    )

                try:
                    cache_stats = cache.get_cache_stats()
                    click.echo("\n📊 Cache Performance:")
                    click.echo(
                        f"   - Total requests: {cache_stats['total_requests']}"
                    )
                    click.echo(
                        f"   - Cache hits: {cache_stats['cache_hits']} "
                        f"({cache_stats['hit_rate_percent']:.1f}%)"
                    )
                    click.echo(
                        f"   - Cache misses: {cache_stats['cache_misses']}"
                    )
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
                    if cache_stats.get("stale_commits", 0) > 0:
                        click.echo(
                            f"   - Stale commits: {cache_stats['stale_commits']}"
                        )
                    click.echo(
                        f"   - Database size: {cache_stats['database_size_mb']:.1f} MB"
                    )
                except Exception as e:
                    click.echo(f"   Warning: Could not display cache statistics: {e}")

                click.echo(f"\n✅ Analysis complete! Reports saved to {output}")

        except Exception as e:
            logger.error("Error in final summary/display: %s", e)
            click.echo(f"   ❌ Error in final summary/display: {e}")
            raise

        # Stop Rich display if it was started
        if (
            "progress" in locals()
            and progress
            and hasattr(progress, "_use_rich")
            and progress._use_rich
        ):
            progress.stop_rich_display()

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


def _run_security_only_analysis(
    cfg: Any,
    cache: Any,
    cache_dir: Path,
    config: Path,
    no_cache: bool,
    output: Optional[Path],
    display: Any,
    weeks: int,
) -> None:
    """Run the security-only analysis path and print results to console."""
    from .core.data_fetcher import GitDataFetcher
    from .security import SecurityAnalyzer, SecurityConfig
    from .security.reports import SecurityReportGenerator
    from .utils.date_utils import get_monday_aligned_start, get_week_end

    if display:
        display.print_status("🔒 Running security-only analysis...", "info")
    else:
        click.echo("\n🔒 Running security-only analysis...")

    security_config = SecurityConfig.from_dict(
        cfg.analysis.security if hasattr(cfg.analysis, "security") else {}
    )
    if not security_config.enabled:
        if display:
            display.show_error("Security analysis is not enabled in configuration")
        else:
            click.echo("❌ Security analysis is not enabled in configuration")
            click.echo("💡 Add 'security:' section to your config with 'enabled: true'")
        return

    _cache_dir = cfg.cache.directory
    if not _cache_dir.is_absolute():
        _cache_dir = config.parent / _cache_dir
    _cache_dir.mkdir(parents=True, exist_ok=True)

    from .core.cache import GitAnalysisCache

    _cache = GitAnalysisCache(
        cache_dir=_cache_dir,
        ttl_hours=cfg.cache.ttl_hours if not no_cache else 0,
    )
    data_fetcher = GitDataFetcher(
        cache=_cache,
        branch_mapping_rules=cfg.analysis.branch_mapping_rules,
        allowed_ticket_platforms=cfg.get_effective_ticket_platforms(),
        exclude_paths=cfg.analysis.exclude_paths,
        exclude_merge_commits=cfg.analysis.exclude_merge_commits,
    )

    all_commits: list[Any] = []
    for repo_config in cfg.repositories:
        repo_path = Path(repo_config["path"])
        if not repo_path.exists():
            click.echo(f"⚠️  Repository not found: {repo_path}")
            continue

        start_date = get_monday_aligned_start(weeks)
        from datetime import timedelta

        end_date = get_week_end(start_date + timedelta(weeks=weeks) - timedelta(days=1))

        if display:
            display.print_status(f"Fetching commits from {repo_config['name']}...", "info")
        else:
            click.echo(f"📥 Fetching commits from {repo_config['name']}...")

        raw_data = data_fetcher.fetch_raw_data(
            repositories=[repo_config],
            start_date=start_date,
            end_date=end_date,
        )
        commits = raw_data["commits"] if raw_data and raw_data.get("commits") else []
        all_commits.extend(commits)

    if not all_commits:
        if display:
            display.show_error("No commits found to analyze")
        else:
            click.echo("❌ No commits found to analyze")
        return

    security_analyzer = SecurityAnalyzer(config=security_config)
    if display:
        display.print_status(
            f"Analyzing {len(all_commits)} commits for security issues...", "info"
        )
    else:
        click.echo(f"\n🔍 Analyzing {len(all_commits)} commits for security issues...")

    analyses = [security_analyzer.analyze_commit(c) for c in all_commits]
    summary = security_analyzer.generate_summary_report(analyses)

    click.echo("\n" + "=" * 60)
    click.echo("SECURITY ANALYSIS SUMMARY")
    click.echo("=" * 60)
    click.echo(f"Total Commits Analyzed: {summary['total_commits']}")
    click.echo(f"Commits with Issues: {summary['commits_with_issues']}")
    click.echo(f"Total Security Findings: {summary['total_findings']}")
    click.echo(
        f"Risk Level: {summary['risk_level']} (Score: {summary['average_risk_score']:.1f})"
    )

    for severity, label in [("critical", "🔴"), ("high", "🟠"), ("medium", "🟡")]:
        count = summary["severity_distribution"].get(severity, 0)
        if count > 0:
            click.echo(f"\n{label} {severity.title()} Issues: {count}")

    report_dir = output or Path(cfg.output.directory)
    report_dir.mkdir(parents=True, exist_ok=True)
    reports = SecurityReportGenerator(output_dir=report_dir).generate_reports(analyses, summary)
    click.echo("\n✅ Security Reports Generated:")
    for report_type, path in reports.items():
        click.echo(f"  - {report_type.upper()}: {path}")

    if summary.get("recommendations"):
        click.echo("\n💡 Recommendations:")
        for rec in summary["recommendations"][:5]:
            click.echo(f"  {rec}")

    if display:
        display.print_status("Security analysis completed!", "success")


def _run_identity_analysis(
    config: Path,
    cfg: Any,
    cache_dir: Path,
    identity_resolver: Any,
    all_commits: list[Any],
    developer_stats: list[Any],
    display: Any,
    logger: Any,
) -> None:
    """Run the optional identity-cluster analysis and prompt user to apply mappings."""
    from .identity_llm.analysis_pass import IdentityAnalysisPass
    from datetime import datetime as _dt

    try:
        last_prompt_file = cache_dir / ".identity_last_prompt"
        should_prompt = True
        if last_prompt_file.exists():
            last_prompt_age = _dt.now() - _dt.fromtimestamp(
                os.path.getmtime(last_prompt_file)
            )
            if last_prompt_age < timedelta(days=7):
                should_prompt = False

        if not should_prompt:
            return

        if display:
            display.print_status("Analyzing developer identities...", "info")
        else:
            click.echo("\n🔍 Analyzing developer identities...")

        analysis_pass = IdentityAnalysisPass(config)
        identity_cache_file = cache_dir / "identity_analysis_cache.yaml"
        identity_result = analysis_pass.run_analysis(
            all_commits, output_path=identity_cache_file, apply_to_config=False
        )

        if not identity_result.clusters:
            if display:
                display.print_status(
                    "No identity clusters found - all developers appear unique", "success"
                )
            else:
                click.echo("✅ No identity clusters found - all developers appear unique")
            last_prompt_file.touch()
            return

        suggested_config = analysis_pass.generate_suggested_config(identity_result)

        if display:
            display.print_status(
                f"Found {len(identity_result.clusters)} potential identity clusters",
                "warning",
            )
        else:
            click.echo(
                f"\n⚠️  Found {len(identity_result.clusters)} potential identity clusters:"
            )

        if suggested_config.get("analysis", {}).get("manual_identity_mappings"):
            click.echo("\n📋 Suggested identity mappings:")
            for mapping in suggested_config["analysis"]["manual_identity_mappings"]:
                canonical = mapping["canonical_email"]
                aliases = mapping.get("aliases", [])
                if aliases:
                    click.echo(f"   {canonical}")
                    for alias in aliases:
                        click.echo(f"     → {alias}")

        if suggested_config.get("exclude", {}).get("authors"):
            bot_count = len(suggested_config["exclude"]["authors"])
            click.echo(f"\n🤖 Found {bot_count} bot accounts to exclude:")
            for bot in suggested_config["exclude"]["authors"][:5]:
                click.echo(f"   - {bot}")
            if bot_count > 5:
                click.echo(f"   ... and {bot_count - 5} more")

        click.echo("\n" + "─" * 60)
        if click.confirm(
            "Apply these identity mappings to your configuration?", default=True
        ):
            try:
                with open(config) as f:
                    config_data = yaml.safe_load(f)

                config_data.setdefault("analysis", {}).setdefault("identity", {})
                existing_mappings = config_data["analysis"]["identity"].get(
                    "manual_mappings", []
                )
                new_mappings = suggested_config.get("analysis", {}).get(
                    "manual_identity_mappings", []
                )
                existing_emails = {
                    m.get("canonical_email", "").lower() for m in existing_mappings
                }
                for new_mapping in new_mappings:
                    if new_mapping["canonical_email"].lower() not in existing_emails:
                        existing_mappings.append(new_mapping)
                config_data["analysis"]["identity"]["manual_mappings"] = existing_mappings

                if suggested_config.get("exclude", {}).get("authors"):
                    config_data["analysis"].setdefault("exclude", {}).setdefault(
                        "authors", []
                    )
                    existing_excludes = set(config_data["analysis"]["exclude"]["authors"])
                    for bot in suggested_config["exclude"]["authors"]:
                        if bot not in existing_excludes:
                            config_data["analysis"]["exclude"]["authors"].append(bot)

                with open(config, "w") as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

                if display:
                    display.print_status(
                        "Applied identity mappings to configuration", "success"
                    )
                else:
                    click.echo("✅ Applied identity mappings to configuration")

                # Reload config & re-init identity resolver
                from .config import ConfigLoader

                cfg_new = ConfigLoader.load(config)
                identity_resolver.__init__(
                    cache_dir / "identities.db",
                    similarity_threshold=cfg_new.analysis.similarity_threshold,
                    manual_mappings=cfg_new.analysis.manual_identity_mappings,
                )
                click.echo("\n🔄 Re-resolving developer identities with new mappings...")
                identity_resolver.update_commit_stats(all_commits)

            except Exception as e:
                logger.error("Failed to apply identity mappings: %s", e)
                click.echo(f"❌ Failed to apply identity mappings: {e}")
        else:
            click.echo("⏭️  Skipping identity mapping suggestions")

        last_prompt_file.touch()

    except Exception as e:
        if display:
            display.print_status(f"Identity analysis failed: {e}", "warning")
        else:
            click.echo(f"⚠️  Identity analysis failed: {e}")
        logger.debug("Identity analysis error: %s", e, exc_info=True)




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


@click.command(name="collect")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--weeks",
    "-w",
    type=int,
    default=4,
    show_default=True,
    help="Number of complete weeks to collect",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force re-fetch even for weeks already in the cache",
)
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="none",
    help="Enable logging with specified level",
)
def collect_command(config_path: Path, weeks: int, force: bool, log: str) -> None:
    """Stage 1: Collect raw commit data from repositories into the weekly cache.

    \b
    Fetches commits from every repository listed in the configuration file and
    stores them in the local SQLite cache.  Weeks that are already cached are
    skipped unless --force is given.

    \b
    EXAMPLES:
      # Collect 4 weeks of data
      gfa collect -c config.yaml --weeks 4

      # Force a fresh fetch, ignoring cached weeks
      gfa collect -c config.yaml --weeks 4 -f

    \b
    NEXT STEP:
      gfa classify -c config.yaml
    """
    setup_logging(log, __name__)

    try:
        from .pipeline import run_collect

        cfg = ConfigLoader.load(config_path)
        click.echo(f"Stage 1: Collecting data ({weeks} weeks)...")

        result = run_collect(
            cfg=cfg,
            weeks=weeks,
            force=force,
            progress_callback=lambda msg: click.echo(f"  {msg}"),
        )

        if result.errors:
            for err in result.errors:
                click.echo(f"  Warning: {err}", err=True)

        click.echo(
            f"\nCollect complete: "
            f"{result.total_commits} commits from "
            f"{result.repos_fetched + result.repos_cached} repositories "
            f"({result.repos_cached} cached, {result.repos_fetched} fetched"
            + (f", {result.repos_failed} failed" if result.repos_failed else "")
            + ")"
        )

    except (FileNotFoundError, Exception) as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)


@click.command(name="classify")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--weeks",
    "-w",
    type=int,
    default=4,
    show_default=True,
    help="Number of weeks to classify (must match the collect --weeks value)",
)
@click.option(
    "--reclassify",
    is_flag=True,
    help="Re-classify commits that were already classified",
)
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="none",
    help="Enable logging with specified level",
)
def classify_command(config_path: Path, weeks: int, reclassify: bool, log: str) -> None:
    """Stage 2: Classify collected commits using batch LLM classification.

    \b
    Reads commits from the local cache (written by 'gfa collect') and runs
    batch classification on them.  Commits that were already classified are
    skipped unless --reclassify is given.

    \b
    EXAMPLES:
      # Classify the last 4 weeks
      gfa classify -c config.yaml

      # Force re-classification of all commits
      gfa classify -c config.yaml --reclassify

    \b
    PREREQUISITE:
      gfa collect -c config.yaml

    \b
    NEXT STEP:
      gfa report -c config.yaml
    """
    setup_logging(log, __name__)

    try:
        from .pipeline import run_classify

        cfg = ConfigLoader.load(config_path)
        click.echo(f"Stage 2: Classifying commits ({weeks} weeks)...")

        result = run_classify(
            cfg=cfg,
            weeks=weeks,
            reclassify=reclassify,
            progress_callback=lambda msg: click.echo(f"  {msg}"),
        )

        if result.errors:
            for err in result.errors:
                click.echo(f"  Warning: {err}", err=True)
            if any("Run 'gfa collect' first" in e for e in result.errors):
                click.echo("\nHint: run 'gfa collect -c config.yaml' before classify", err=True)
                sys.exit(1)

        click.echo(
            f"\nClassify complete: "
            f"{result.processed_batches} batches, "
            f"{result.total_commits} commits"
            + (f" ({result.skipped_batches} skipped)" if result.skipped_batches else "")
        )

    except Exception as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)


@click.command(name="report")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--weeks",
    "-w",
    type=int,
    default=4,
    show_default=True,
    help="Number of weeks to include in reports",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for reports (overrides config file)",
)
@click.option(
    "--generate-csv",
    is_flag=True,
    help="Generate CSV reports in addition to the narrative markdown report",
)
@click.option(
    "--anonymize",
    is_flag=True,
    help="Anonymize developer information in reports",
)
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="none",
    help="Enable logging with specified level",
)
def report_command(
    config_path: Path,
    weeks: int,
    output_path: Optional[Path],
    generate_csv: bool,
    anonymize: bool,
    log: str,
) -> None:
    """Stage 3: Generate reports from classified commit data.

    \b
    Reads classified commits from the cache (written by 'gfa collect' and
    'gfa classify') and generates report files.  No git operations are
    performed.

    \b
    EXAMPLES:
      # Generate the narrative markdown report
      gfa report -c config.yaml

      # Generate CSV reports as well
      gfa report -c config.yaml --generate-csv

      # Write reports to a custom directory
      gfa report -c config.yaml -o /tmp/my-reports --generate-csv

    \b
    PREREQUISITE:
      gfa collect -c config.yaml
      gfa classify -c config.yaml
    """
    setup_logging(log, __name__)

    try:
        from .pipeline import run_report

        cfg = ConfigLoader.load(config_path)

        if output_path is None:
            output_path = cfg.output.directory if cfg.output.directory else Path("./reports")

        click.echo(f"Stage 3: Generating reports ({weeks} weeks) → {output_path}")

        result = run_report(
            cfg=cfg,
            weeks=weeks,
            output_dir=output_path,
            generate_csv=generate_csv,
            anonymize=anonymize,
            progress_callback=lambda msg: click.echo(f"  {msg}"),
        )

        if result.errors:
            for err in result.errors:
                click.echo(f"  Warning: {err}", err=True)

        click.echo(f"\nReport complete: {len(result.generated_reports)} files in {output_path}")
        for name in result.generated_reports:
            click.echo(f"  {name}")

    except Exception as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)


