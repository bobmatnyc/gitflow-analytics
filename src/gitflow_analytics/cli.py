"""Command-line interface for GitFlow Analytics."""

import logging
import sys
from datetime import datetime, timedelta, timezone
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
from .extractors.tickets import TicketExtractor
from .integrations.orchestrator import IntegrationOrchestrator
from .metrics.dora import DORAMetricsCalculator
from .reports.analytics_writer import AnalyticsReportGenerator
from .reports.csv_writer import CSVReportGenerator
from .reports.json_exporter import ComprehensiveJSONExporter
from .reports.narrative_writer import NarrativeReportGenerator


def handle_timezone_error(e: Exception, report_name: str, all_commits: list, logger: logging.Logger) -> None:
    """Handle timezone comparison errors with detailed logging."""
    if isinstance(e, TypeError) and ("can't compare" in str(e).lower() or "timezone" in str(e).lower()):
        logger.error(f"Timezone comparison error in {report_name}:")
        logger.error(f"  Error: {e}")
        import traceback
        logger.error(f"  Full traceback:\n{traceback.format_exc()}")
        
        # Log context information
        sample_commits = all_commits[:5] if all_commits else []
        for i, commit in enumerate(sample_commits):
            timestamp = commit.get('timestamp')
            logger.error(f"  Sample commit {i}: timestamp={timestamp} (tzinfo: {getattr(timestamp, 'tzinfo', 'N/A')})")
        
        click.echo(f"   ❌ Timezone comparison error in {report_name}")
        click.echo("   🔍 See logs with --log DEBUG for detailed information")
        click.echo("   💡 This usually indicates mixed timezone-aware and naive datetime objects")
        raise
    else:
        # Re-raise other errors
        raise


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
        global_options = {'--version', '--help', '-h'}
        if args and args[0] in global_options:
            return super().parse_args(ctx, args)
        
        # Check if we have arguments that look like analyze options
        analyze_indicators = {'-c', '--config', '-w', '--weeks', '--output', '-o', 
                            '--anonymize', '--no-cache', '--validate-only', '--clear-cache',
                            '--enable-qualitative', '--qualitative-only', '--enable-pm',
                            '--pm-platform', '--disable-pm', '--rich', '--log',
                            '--skip-identity-analysis', '--apply-identity-suggestions'}
        
        # If first arg starts with - and looks like an analyze option, prepend 'analyze'
        if args and args[0].startswith('-'):
            # Check if any of the args are analyze indicators
            has_analyze_indicators = any(arg in analyze_indicators for arg in args)
            if has_analyze_indicators:
                # This looks like it should be an analyze command
                new_args = ['analyze'] + args
                return super().parse_args(ctx, new_args)
        
        # Otherwise, use default behavior
        return super().parse_args(ctx, args)


@click.group(cls=AnalyzeAsDefaultGroup, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="GitFlow Analytics")
@click.help_option('-h', '--help')
@click.pass_context
def cli(ctx: click.Context) -> None:
    """GitFlow Analytics - Analyze Git repositories for productivity insights.
    
    If no subcommand is provided, the analyze command will be executed by default.
    You can use analysis options directly: gitflow-analytics -c config.yaml --weeks 2
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
@click.option("--enable-qualitative", is_flag=True, help="Enable qualitative analysis (requires additional dependencies)")
@click.option("--qualitative-only", is_flag=True, help="Run only qualitative analysis on existing commits")
@click.option("--enable-pm", is_flag=True, help="Enable PM platform integration (overrides config setting)")
@click.option("--pm-platform", multiple=True, help="Enable specific PM platforms (e.g., --pm-platform jira --pm-platform azure)")
@click.option("--disable-pm", is_flag=True, help="Disable PM platform integration (overrides config setting)")
@click.option("--rich", is_flag=True, default=True, help="Use rich terminal output (default: enabled)")
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="none",
    help="Enable logging with specified level (default: none)"
)
@click.option("--skip-identity-analysis", is_flag=True, help="Skip automatic identity analysis")
@click.option("--apply-identity-suggestions", is_flag=True, help="Apply identity analysis suggestions without prompting")
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
) -> None:
    """Analyze Git repositories using configuration file (explicit command)."""
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
            format='[%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stderr)
            ],
            force=True  # Ensure reconfiguration of existing loggers
        )
        
        # Ensure all GitFlow Analytics loggers are configured properly
        root_logger = logging.getLogger('gitflow_analytics')
        root_logger.setLevel(log_level)
        
        # Create logger for this module
        logger = logging.getLogger(__name__)
        logger.info(f"Logging enabled at {log.upper()} level")
        
        # Log that logging is properly configured for all modules
        logger.debug("Logging configuration applied to all gitflow_analytics modules")
    else:
        # Disable logging
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger('gitflow_analytics').setLevel(logging.CRITICAL)
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
                display.print_status(f"PM integration limited to platforms: {', '.join(pm_platform)}", "info")
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
                    display.print_status("Configuration has issues that should be addressed.", "error")
                else:
                    click.echo("❌ Configuration has issues that should be addressed.")
            return

        # Use output directory from CLI or config
        if output is None:
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
                analysis_weeks=weeks
            )

        # Initialize components
        cache_dir = cfg.cache.directory
        if clear_cache:
            if display:
                display.print_status("Clearing cache...", "info")
            else:
                click.echo("🗑️  Clearing cache...")
            import shutil

            if cache_dir.exists():
                shutil.rmtree(cache_dir)

        cache = GitAnalysisCache(cache_dir, ttl_hours=0 if no_cache else cfg.cache.ttl_hours)

        identity_resolver = DeveloperIdentityResolver(
            cache_dir / "identities.db",
            similarity_threshold=cfg.analysis.similarity_threshold,
            manual_mappings=cfg.analysis.manual_identity_mappings,
        )

        # Prepare ML categorization config for analyzer
        ml_config = None
        if hasattr(cfg.analysis, 'ml_categorization'):
            ml_config = {
                'enabled': cfg.analysis.ml_categorization.enabled,
                'min_confidence': cfg.analysis.ml_categorization.min_confidence,
                'semantic_weight': cfg.analysis.ml_categorization.semantic_weight,
                'file_pattern_weight': cfg.analysis.ml_categorization.file_pattern_weight,
                'hybrid_threshold': cfg.analysis.ml_categorization.hybrid_threshold,
                'cache_duration_days': cfg.analysis.ml_categorization.cache_duration_days,
                'batch_size': cfg.analysis.ml_categorization.batch_size,
                'enable_caching': cfg.analysis.ml_categorization.enable_caching,
                'spacy_model': cfg.analysis.ml_categorization.spacy_model,
            }

        analyzer = GitAnalyzer(
            cache,
            branch_mapping_rules=cfg.analysis.branch_mapping_rules,
            allowed_ticket_platforms=getattr(cfg.analysis, "ticket_platforms", None),
            exclude_paths=cfg.analysis.exclude_paths,
            ml_categorization_config=ml_config,
        )
        orchestrator = IntegrationOrchestrator(cfg, cache)

        # Discovery organization repositories if needed
        repositories_to_analyze = cfg.repositories
        if cfg.github.organization and not repositories_to_analyze:
            if display:
                display.print_status(f"Discovering repositories from organization: {cfg.github.organization}", "info")
            else:
                click.echo(f"🔍 Discovering repositories from organization: {cfg.github.organization}")
            try:
                # Use a 'repos' directory in the config directory for cloned repositories
                config_dir = Path(config).parent if config else Path.cwd()
                repos_dir = config_dir / "repos"
                discovered_repos = cfg.discover_organization_repositories(clone_base_path=repos_dir)
                repositories_to_analyze = discovered_repos
                
                if display:
                    display.print_status(f"Found {len(discovered_repos)} repositories in organization", "success")
                    # Show repository discovery in structured format
                    repo_data = [{
                        "name": repo.name,
                        "github_repo": repo.github_repo,
                        "exists": repo.path.exists()
                    } for repo in discovered_repos]
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
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)

        if display:
            display.print_status(f"Analyzing {len(repositories_to_analyze)} repositories...", "info")
            display.print_status(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}", "info")
            # Start live progress display
            display.start_live_display()
            display.add_progress_task("repos", "Processing repositories", len(repositories_to_analyze))
        else:
            click.echo(f"\n🚀 Analyzing {len(repositories_to_analyze)} repositories...")
            click.echo(
                f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            )

        # Analyze repositories
        all_commits = []
        all_prs = []
        all_enrichments = {}

        for repo_config in repositories_to_analyze:
            if display:
                display.update_progress_task("repos", description=f"Analyzing {repo_config.name}...")
            else:
                click.echo(f"\n📁 Analyzing {repo_config.name}...")

            # Check if repo exists, clone if needed
            if not repo_config.path.exists():
                # Try to clone if we have a github_repo configured
                if repo_config.github_repo and cfg.github.organization:
                    if display:
                        display.print_status("Cloning repository from GitHub...", "info")
                    else:
                        click.echo("   📥 Cloning repository from GitHub...")
                    try:
                        # Ensure parent directory exists
                        repo_config.path.parent.mkdir(parents=True, exist_ok=True)

                        # Clone the repository
                        clone_url = f"https://github.com/{repo_config.github_repo}.git"
                        if cfg.github.token:
                            # Use token for authentication
                            clone_url = f"https://{cfg.github.token}@github.com/{repo_config.github_repo}.git"

                        # Don't specify branch if None - let git use the default branch
                        if repo_config.branch:
                            git.Repo.clone_from(clone_url, repo_config.path, branch=repo_config.branch)
                        else:
                            git.Repo.clone_from(clone_url, repo_config.path)
                        if display:
                            display.print_status(f"Successfully cloned {repo_config.github_repo}", "success")
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
                        display.print_status(f"Repository path not found: {repo_config.path}", "error")
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

                # Enrich with integration data
                enrichment = orchestrator.enrich_repository_data(repo_config, commits, start_date)
                all_enrichments[repo_config.name] = enrichment

                if enrichment["prs"]:
                    all_prs.extend(enrichment["prs"])
                    if display:
                        display.print_status(f"Found {len(enrichment['prs'])} pull requests", "success")
                    else:
                        click.echo(f"   ✅ Found {len(enrichment['prs'])} pull requests")

            except Exception as e:
                if display:
                    display.print_status(f"Error: {e}", "error")
                else:
                    click.echo(f"   ❌ Error: {e}")
                continue
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

        # Filter out excluded authors (bots)
        if cfg.analysis.exclude_authors:
            original_count = len(all_commits)
            excluded_authors_lower = [author.lower() for author in cfg.analysis.exclude_authors]
            all_commits = [
                commit for commit in all_commits 
                if commit["author_email"].lower() not in excluded_authors_lower and
                   commit["author_name"].lower() not in excluded_authors_lower
            ]
            filtered_count = original_count - len(all_commits)
            if filtered_count > 0:
                if display:
                    display.print_status(f"Filtered out {filtered_count} commits from excluded authors", "info")
                else:
                    click.echo(f"\n🤖 Filtered out {filtered_count} commits from excluded authors")
                    
            # Also filter PRs from excluded authors
            if all_prs:
                original_pr_count = len(all_prs)
                filtered_prs = []
                for pr in all_prs:
                    # Skip non-dict PR entries (probably an error in data)
                    if not isinstance(pr, dict):
                        logger.warning(f"Skipping non-dict PR entry: {type(pr)} = {pr}")
                        continue
                    
                    # Check if PR author is in excluded list
                    author_info = pr.get("author", {})
                    if isinstance(author_info, dict):
                        author_login = author_info.get("login", "")
                        if author_login.lower() not in excluded_authors_lower:
                            filtered_prs.append(pr)
                    else:
                        # Keep PRs without proper author info
                        filtered_prs.append(pr)
                
                all_prs = filtered_prs
                filtered_pr_count = original_pr_count - len(all_prs)
                if filtered_pr_count > 0:
                    if display:
                        display.print_status(f"Filtered out {filtered_pr_count} PRs from excluded authors", "info")
                    else:
                        click.echo(f"   🤖 Filtered out {filtered_pr_count} PRs from excluded authors")

        # Update developer statistics
        if display:
            display.print_status("Resolving developer identities...", "info")
        else:
            click.echo("\n👥 Resolving developer identities...")
            
        identity_resolver.update_commit_stats(all_commits)
        developer_stats = identity_resolver.get_developer_stats()
        
        if display:
            display.print_status(f"Identified {len(developer_stats)} unique developers", "success")
        else:
            click.echo(f"   ✅ Identified {len(developer_stats)} unique developers")
        
        # Check if we should run identity analysis
        should_check_identities = (
            not skip_identity_analysis and  # Not explicitly skipped
            cfg.analysis.auto_identity_analysis and  # Auto analysis is enabled
            not cfg.analysis.manual_identity_mappings and  # No manual mappings
            len(developer_stats) > 1  # Multiple developers to analyze
        )
        
        # Debug identity analysis decision
        if not should_check_identities:
            reasons = []
            if skip_identity_analysis:
                reasons.append("--skip-identity-analysis flag used")
            if not cfg.analysis.auto_identity_analysis:
                reasons.append("auto_identity_analysis disabled in config")
            if cfg.analysis.manual_identity_mappings:
                reasons.append(f"manual identity mappings already exist ({len(cfg.analysis.manual_identity_mappings)} mappings)")
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
                    import os
                    last_prompt_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(last_prompt_file))
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
                        all_commits,
                        output_path=identity_cache_file,
                        apply_to_config=False
                    )
                    
                    if identity_result.clusters:
                        # Generate suggested configuration
                        suggested_config = analysis_pass.generate_suggested_config(identity_result)
                        
                        # Show suggestions
                        if display:
                            display.print_status(f"Found {len(identity_result.clusters)} potential identity clusters", "warning")
                        else:
                            click.echo(f"\n⚠️  Found {len(identity_result.clusters)} potential identity clusters:")
                        
                        # Display all mappings
                        if suggested_config.get('analysis', {}).get('manual_identity_mappings'):
                            click.echo("\n📋 Suggested identity mappings:")
                            for mapping in suggested_config['analysis']['manual_identity_mappings']:
                                canonical = mapping['canonical_email']
                                aliases = mapping.get('aliases', [])
                                if aliases:
                                    click.echo(f"   {canonical}")
                                    for alias in aliases:
                                        click.echo(f"     → {alias}")
                        
                        # Check for bot exclusions
                        if suggested_config.get('exclude', {}).get('authors'):
                            bot_count = len(suggested_config['exclude']['authors'])
                            click.echo(f"\n🤖 Found {bot_count} bot accounts to exclude:")
                            for bot in suggested_config['exclude']['authors'][:5]:  # Show first 5
                                click.echo(f"   - {bot}")
                            if bot_count > 5:
                                click.echo(f"   ... and {bot_count - 5} more")
                        
                        # Prompt user
                        click.echo("\n" + "─" * 60)
                        if click.confirm("Apply these identity mappings to your configuration?", default=True):
                            # Apply mappings to config
                            try:
                                # Reload config to ensure we have latest
                                with open(config) as f:
                                    config_data = yaml.safe_load(f)
                                
                                # Update analysis section
                                if 'analysis' not in config_data:
                                    config_data['analysis'] = {}
                                if 'identity' not in config_data['analysis']:
                                    config_data['analysis']['identity'] = {}
                                
                                # Apply manual mappings
                                existing_mappings = config_data['analysis']['identity'].get('manual_mappings', [])
                                new_mappings = suggested_config.get('analysis', {}).get('manual_identity_mappings', [])
                                
                                # Merge mappings
                                existing_emails = {m.get('canonical_email', '').lower() for m in existing_mappings}
                                for new_mapping in new_mappings:
                                    if new_mapping['canonical_email'].lower() not in existing_emails:
                                        existing_mappings.append(new_mapping)
                                
                                config_data['analysis']['identity']['manual_mappings'] = existing_mappings
                                
                                # Apply bot exclusions
                                if suggested_config.get('exclude', {}).get('authors'):
                                    if 'exclude' not in config_data['analysis']:
                                        config_data['analysis']['exclude'] = {}
                                    if 'authors' not in config_data['analysis']['exclude']:
                                        config_data['analysis']['exclude']['authors'] = []
                                    
                                    existing_excludes = set(config_data['analysis']['exclude']['authors'])
                                    for bot in suggested_config['exclude']['authors']:
                                        if bot not in existing_excludes:
                                            config_data['analysis']['exclude']['authors'].append(bot)
                                
                                # Write updated config
                                with open(config, 'w') as f:
                                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                                
                                if display:
                                    display.print_status("Applied identity mappings to configuration", "success")
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
                                click.echo("\n🔄 Re-resolving developer identities with new mappings...")
                                identity_resolver.update_commit_stats(all_commits)
                                developer_stats = identity_resolver.get_developer_stats()
                                
                                if display:
                                    display.print_status(f"Consolidated to {len(developer_stats)} unique developers", "success")
                                else:
                                    click.echo(f"✅ Consolidated to {len(developer_stats)} unique developers")
                                
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
                            display.print_status("No identity clusters found - all developers appear unique", "success")
                        else:
                            click.echo("✅ No identity clusters found - all developers appear unique")
                        
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
            
        ticket_extractor = TicketExtractor(
            allowed_platforms=getattr(cfg.analysis, "ticket_platforms", None)
        )
        ticket_analysis = ticket_extractor.analyze_ticket_coverage(all_commits, all_prs)

        for platform, count in ticket_analysis["ticket_summary"].items():
            if display:
                display.print_status(f"{platform.title()}: {count} unique tickets", "success")
            else:
                click.echo(f"   - {platform.title()}: {count} unique tickets")

        # Perform qualitative analysis if enabled
        qualitative_results = []
        if (enable_qualitative or qualitative_only) and cfg.qualitative and cfg.qualitative.enabled:
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
                    issue_msg = "Qualitative analysis setup issues:\n" + "\n".join(f"• {issue}" for issue in issues)
                    if issues:
                        issue_msg += "\n\n💡 Install dependencies: pip install spacy scikit-learn openai tiktoken"
                        issue_msg += "\n💡 Download spaCy model: python -m spacy download en_core_web_sm"
                    
                    if display:
                        display.show_warning(issue_msg)
                    else:
                        click.echo("   ⚠️  Qualitative analysis setup issues:")
                        for issue in issues:
                            click.echo(f"      - {issue}")
                        if issues:
                            click.echo("   💡 Install dependencies: pip install spacy scikit-learn openai tiktoken")
                            click.echo("   💡 Download spaCy model: python -m spacy download en_core_web_sm")
                
                # Convert commits to qualitative format
                commits_for_qual = []
                for commit in all_commits:
                    commit_dict = {
                        'hash': commit.hash,
                        'message': commit.message,
                        'author_name': commit.author_name,
                        'author_email': commit.author_email,
                        'timestamp': commit.timestamp,
                        'files_changed': commit.files_changed or [],
                        'insertions': commit.insertions,
                        'deletions': commit.deletions,
                        'branch': getattr(commit, 'branch', 'main')
                    }
                    commits_for_qual.append(commit_dict)
                
                # Perform qualitative analysis with progress tracking
                if display:
                    display.start_live_display()
                    display.add_progress_task("qualitative", "Analyzing commits with qualitative insights", len(commits_for_qual))
                
                qualitative_results = qual_processor.process_commits(commits_for_qual, show_progress=True)
                
                if display:
                    display.complete_progress_task("qualitative", "Qualitative analysis complete")
                    display.stop_live_display()
                    display.print_status(f"Analyzed {len(qualitative_results)} commits with qualitative insights", "success")
                else:
                    click.echo(f"   ✅ Analyzed {len(qualitative_results)} commits with qualitative insights")
                
                # Get processing statistics and show them
                qual_stats = qual_processor.get_processing_statistics()
                if display:
                    display.show_qualitative_stats(qual_stats)
                else:
                    processing_summary = qual_stats['processing_summary']
                    click.echo(f"   📈 Processing: {processing_summary['commits_per_second']:.1f} commits/sec")
                    click.echo(f"   🎯 Methods: {processing_summary['method_breakdown']['cache']:.1f}% cached, "
                              f"{processing_summary['method_breakdown']['nlp']:.1f}% NLP, "
                              f"{processing_summary['method_breakdown']['llm']:.1f}% LLM")
                    
                    if qual_stats['llm_statistics']['model_usage'] == 'available':
                        llm_stats = qual_stats['llm_statistics']['cost_tracking']
                        if llm_stats['total_cost'] > 0:
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
                        display.show_error("Cannot perform qualitative-only analysis without dependencies")
                    else:
                        click.echo("   ❌ Cannot perform qualitative-only analysis without dependencies")
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
                aggregated_pm_data = {
                    "issues": {},
                    "correlations": [],
                    "metrics": {}
                }
                
                for repo_name, enrichment in all_enrichments.items():
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
            display.print_status("Generating reports...", "info")
        else:
            click.echo("\n📊 Generating reports...")
        
        logger.debug(f"Starting report generation with {len(all_commits)} commits")
        
        report_gen = CSVReportGenerator(anonymize=anonymize or cfg.output.anonymize_enabled)
        analytics_gen = AnalyticsReportGenerator(
            anonymize=anonymize or cfg.output.anonymize_enabled
        )

        # Collect generated report files for display
        generated_reports = []
        
        # Weekly metrics report
        weekly_report = output / f'weekly_metrics_{datetime.now(timezone.utc).strftime("%Y%m%d")}.csv'
        try:
            logger.debug("Starting weekly metrics report generation")
            report_gen.generate_weekly_report(all_commits, developer_stats, weekly_report, weeks)
            logger.debug("Weekly metrics report completed successfully")
            generated_reports.append(weekly_report.name)
            if not display:
                click.echo(f"   ✅ Weekly metrics: {weekly_report}")
        except Exception as e:
            logger.error(f"Error in weekly metrics report generation: {e}")
            try:
                handle_timezone_error(e, "weekly metrics report", all_commits, logger)
            except:
                pass  # Let the original error handling below take over
            click.echo(f"   ❌ Error generating weekly metrics report: {e}")
            click.echo(f"   🔍 Error type: {type(e).__name__}")
            click.echo(f"   📍 Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        # Summary report
        summary_report = output / f'summary_{datetime.now().strftime("%Y%m%d")}.csv'
        try:
            report_gen.generate_summary_report(
                all_commits, all_prs, developer_stats, ticket_analysis, summary_report, aggregated_pm_data
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

        # Developer report
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

        # Untracked commits report  
        untracked_commits_report = output / f'untracked_commits_{datetime.now().strftime("%Y%m%d")}.csv'
        try:
            report_gen.generate_untracked_commits_report(ticket_analysis, untracked_commits_report)
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

        # PM Correlations report (if PM data is available)
        if aggregated_pm_data:
            pm_correlations_report = output / f'pm_correlations_{datetime.now().strftime("%Y%m%d")}.csv'
            try:
                report_gen.generate_pm_correlations_report(aggregated_pm_data, pm_correlations_report)
                generated_reports.append(pm_correlations_report.name)
                if not display:
                    click.echo(f"   ✅ PM correlations: {pm_correlations_report}")
            except Exception as e:
                click.echo(f"   ⚠️ Warning: PM correlations report failed: {e}")

        # Activity distribution report
        activity_report = output / f'activity_distribution_{datetime.now().strftime("%Y%m%d")}.csv'
        try:
            logger.debug("Starting activity distribution report generation")
            analytics_gen.generate_activity_distribution_report(
                all_commits, developer_stats, activity_report
            )
            logger.debug("Activity distribution report completed successfully")
            generated_reports.append(activity_report.name)
            if not display:
                click.echo(f"   ✅ Activity distribution: {activity_report}")
        except Exception as e:
            logger.error(f"Error in activity distribution report generation: {e}")
            try:
                handle_timezone_error(e, "activity distribution report", all_commits, logger)
            except:
                pass  # Let the original error handling below take over
            click.echo(f"   ❌ Error generating activity distribution report: {e}")
            click.echo(f"   🔍 Error type: {type(e).__name__}")
            click.echo(f"   📍 Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        # Developer focus report
        focus_report = output / f'developer_focus_{datetime.now().strftime("%Y%m%d")}.csv'
        try:
            logger.debug("Starting developer focus report generation")
            analytics_gen.generate_developer_focus_report(
                all_commits, developer_stats, focus_report, weeks
            )
            logger.debug("Developer focus report completed successfully")
            generated_reports.append(focus_report.name)
            if not display:
                click.echo(f"   ✅ Developer focus: {focus_report}")
        except Exception as e:
            logger.error(f"Error in developer focus report generation: {e}")
            try:
                handle_timezone_error(e, "developer focus report", all_commits, logger)
            except:
                pass  # Let the original error handling below take over
            click.echo(f"   ❌ Error generating developer focus report: {e}")
            click.echo(f"   🔍 Error type: {type(e).__name__}")
            click.echo(f"   📍 Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        # Qualitative insights report
        insights_report = output / f'qualitative_insights_{datetime.now().strftime("%Y%m%d")}.csv'
        try:
            logger.debug("Starting qualitative insights report generation")
            analytics_gen.generate_qualitative_insights_report(
                all_commits, developer_stats, ticket_analysis, insights_report
            )
            logger.debug("Qualitative insights report completed successfully")
            generated_reports.append(insights_report.name)
            if not display:
                click.echo(f"   ✅ Qualitative insights: {insights_report}")
        except Exception as e:
            logger.error(f"Error in qualitative insights report generation: {e}")
            try:
                handle_timezone_error(e, "qualitative insights report", all_commits, logger)
            except:
                pass  # Let the original error handling below take over
            click.echo(f"   ❌ Error generating qualitative insights report: {e}")
            click.echo(f"   🔍 Error type: {type(e).__name__}")
            click.echo(f"   📍 Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Weekly trends report (includes developer and project trends)
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
            dev_trends_file = output / f'developer_trends_{timestamp}.csv'
            proj_trends_file = output / f'project_trends_{timestamp}.csv'
            
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
            try:
                handle_timezone_error(e, "DORA metrics calculation", all_commits, logger)
            except:
                pass  # Let the original error handling below take over
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
            try:
                handle_timezone_error(e, "PR metrics aggregation", all_commits, logger)
            except:
                pass  # Let the original error handling below take over
            click.echo(f"   ❌ Error aggregating PR metrics: {e}")
            click.echo(f"   🔍 Error type: {type(e).__name__}")
            click.echo(f"   📍 Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


        # Generate narrative report if markdown format is enabled
        if "markdown" in cfg.output.formats:
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
                narrative_report = output / f'narrative_report_{datetime.now().strftime("%Y%m%d")}.md'
                
                # Try to generate ChatGPT summary
                chatgpt_summary = None
                import os as os_module
                openai_key = os_module.getenv("OPENROUTER_API_KEY") or os_module.getenv("OPENAI_API_KEY")
                if openai_key:
                    try:
                        # Create temporary comprehensive data for ChatGPT
                        from .qualitative.chatgpt_analyzer import ChatGPTQualitativeAnalyzer
                        
                        logger.debug("Preparing data for ChatGPT analysis")
                        
                        # Create minimal comprehensive data structure
                        comprehensive_data = {
                            "metadata": {
                                "analysis_weeks": weeks,
                                "generated_at": datetime.now(timezone.utc).isoformat()
                            },
                            "executive_summary": {
                                "key_metrics": {
                                    "commits": {"total": len(all_commits)},
                                    "developers": {"total": len(developer_stats)},
                                    "lines_changed": {"total": sum(
                                        c.get('filtered_insertions', c.get('insertions', 0)) + 
                                        c.get('filtered_deletions', c.get('deletions', 0)) 
                                        for c in all_commits
                                    )},
                                    "story_points": {"total": sum(c.get('story_points', 0) or 0 for c in all_commits)},
                                    "ticket_coverage": {"percentage": ticket_analysis.get('commit_coverage_pct', 0)}
                                },
                                "health_score": {"overall": 75, "rating": "good"},  # Placeholder
                                "trends": {"velocity": {"direction": "stable"}},
                                "wins": [],
                                "concerns": []
                            },
                            "developers": {},
                            "projects": {}
                        }
                        
                        # Add developer data
                        for dev in developer_stats[:10]:  # Top 10 developers
                            dev_id = dev.get('canonical_id', dev.get('primary_email', 'unknown'))
                            comprehensive_data["developers"][dev_id] = {
                                "identity": {"name": dev.get('primary_name', 'Unknown')},
                                "summary": {
                                    "total_commits": dev.get('total_commits', 0),
                                    "total_story_points": dev.get('total_story_points', 0)
                                },
                                "projects": {}
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
                    chatgpt_summary
                )
                generated_reports.append(narrative_report.name)
                logger.debug("Narrative report generation completed successfully")
                if not display:
                    click.echo(f"   ✅ Narrative report: {narrative_report}")
            except Exception as e:
                logger.error(f"Error in narrative report generation: {e}")
                try:
                    handle_timezone_error(e, "narrative report generation", all_commits, logger)
                except:
                    pass  # Let the original error handling below take over
                click.echo(f"   ❌ Error generating narrative report: {e}")
                click.echo(f"   🔍 Error type: {type(e).__name__}")
                click.echo(f"   📍 Error details: {str(e)}")
                import traceback
                traceback.print_exc()
                raise

        # Generate comprehensive JSON export if enabled
        if "json" in cfg.output.formats:
            try:
                logger.debug("Starting comprehensive JSON export generation")
                click.echo("   🔄 Generating comprehensive JSON export...")
                json_report = output / f'comprehensive_export_{datetime.now().strftime("%Y%m%d")}.json'

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
                            commits=all_commits,
                            developer_stats=developer_stats,
                            project_metrics={
                                "ticket_analysis": ticket_analysis,
                                "pr_metrics": pr_metrics,
                                "enrichments": all_enrichments,
                            },
                            pm_data=aggregated_pm_data,
                            weeks_analyzed=weeks
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
                    enhanced_qualitative_analysis=enhanced_analysis
                )
                generated_reports.append(json_report.name)
                logger.debug("Comprehensive JSON export generation completed successfully")
                if not display:
                    click.echo(f"   ✅ Comprehensive JSON export: {json_report}")
                
                # Generate HTML report from JSON if requested
                if "html" in cfg.output.formats:
                    try:
                        click.echo("   🔄 Generating HTML report...")
                        from .reports.html_generator import HTMLReportGenerator
                        html_report = output / f'gitflow_report_{datetime.now().strftime("%Y%m%d")}.html'
                        logger.debug("Generating HTML report from JSON data")
                        
                        # Read the JSON data we just wrote
                        if not json_report.exists():
                            # Check for alternative JSON file name
                            alt_json = output / f'gitflow_export_{datetime.now().strftime("%Y%m%d")}.json'
                            if alt_json.exists():
                                click.echo(f"   ⚠️ Using alternative JSON file: {alt_json.name}")
                                json_report = alt_json
                        
                        with open(json_report) as f:
                            import json
                            json_data = json.load(f)
                        
                        html_generator = HTMLReportGenerator()
                        html_generator.generate_report(
                            json_data=json_data,
                            output_path=html_report,
                            title=f"GitFlow Analytics Report - {datetime.now().strftime('%B %Y')}"
                        )
                        generated_reports.append(html_report.name)
                        if not display:
                            click.echo(f"   ✅ HTML report: {html_report}")
                        logger.debug("HTML report generation completed successfully")
                    except Exception as e:
                        logger.error(f"Error generating HTML report: {e}")
                        click.echo(f"   ⚠️ Warning: HTML report generation failed: {e}")
            except Exception as e:
                logger.error(f"Error in comprehensive JSON export generation: {e}")
                try:
                    handle_timezone_error(e, "comprehensive JSON export generation", all_commits, logger)
                except:
                    pass  # Let the original error handling below take over
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
                    ticket_coverage=ticket_analysis['commit_coverage_pct'],
                    story_points=total_story_points,
                    qualitative_analyzed=qualitative_count
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
                
                logger.debug("Starting display.print_status")
                display.print_status("Analysis complete!", "success")
                logger.debug("display.print_status completed successfully")
        except Exception as e:
            logger.error(f"Error in final summary/display: {e}")
            try:
                handle_timezone_error(e, "final summary/display", all_commits, logger)
            except:
                pass  # Let the original error handling below take over
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

            click.echo(f"\n✅ Analysis complete! Reports saved to {output}")

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
            # This is a generic error, add the standard error formatting
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
    """Show cache statistics."""
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
        import os

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
@click.argument("dev1")
@click.argument("dev2")
def merge_identity(config: Path, dev1: str, dev2: str) -> None:
    """Merge two developer identities."""
    try:
        cfg = ConfigLoader.load(config)
        identity_resolver = DeveloperIdentityResolver(cfg.cache.directory / "identities.db")

        click.echo(f"🔄 Merging {dev2} into {dev1}...")
        identity_resolver.merge_identities(dev1, dev2)
        click.echo("✅ Identities merged successfully!")

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
def discover_jira_fields(config: Path) -> None:
    """Discover available JIRA fields, particularly story point fields."""
    try:
        cfg = ConfigLoader.load(config)

        # Check if JIRA is configured
        if not cfg.jira or not cfg.jira.base_url:
            click.echo("❌ JIRA is not configured in the configuration file")
            return

        # Initialize JIRA integration
        from .integrations.jira_integration import JIRAIntegration

        # Create minimal cache for JIRA integration
        cache = GitAnalysisCache(cfg.cache.directory)
        jira = JIRAIntegration(
            cfg.jira.base_url,
            cfg.jira.access_user,
            cfg.jira.access_token,
            cache,
        )

        # Validate connection
        click.echo(f"🔗 Connecting to JIRA at {cfg.jira.base_url}...")
        if not jira.validate_connection():
            click.echo("❌ Failed to connect to JIRA. Check your credentials.")
            return

        click.echo("✅ Connected successfully!\n")
        click.echo("🔍 Discovering fields with potential story point data...")

        fields = jira.discover_fields()

        if not fields:
            click.echo("No potential story point fields found.")
        else:
            click.echo(f"\nFound {len(fields)} potential story point fields:")
            click.echo(
                "\nAdd these to your configuration under jira_integration.story_point_fields:"
            )
            click.echo("```yaml")
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
    """Analyze and manage developer identities."""
    try:
        cfg = ConfigLoader.load(config)
        cache = GitAnalysisCache(cfg.cache.directory)
        
        # Get recent commits
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)
        
        # Prepare ML categorization config for analyzer
        ml_config = None
        if hasattr(cfg.analysis, 'ml_categorization'):
            ml_config = {
                'enabled': cfg.analysis.ml_categorization.enabled,
                'min_confidence': cfg.analysis.ml_categorization.min_confidence,
                'semantic_weight': cfg.analysis.ml_categorization.semantic_weight,
                'file_pattern_weight': cfg.analysis.ml_categorization.file_pattern_weight,
                'hybrid_threshold': cfg.analysis.ml_categorization.hybrid_threshold,
                'cache_duration_days': cfg.analysis.ml_categorization.cache_duration_days,
                'batch_size': cfg.analysis.ml_categorization.batch_size,
                'enable_caching': cfg.analysis.ml_categorization.enable_caching,
                'spacy_model': cfg.analysis.ml_categorization.spacy_model,
            }

        analyzer = GitAnalyzer(
            cache,
            branch_mapping_rules=cfg.analysis.branch_mapping_rules,
            allowed_ticket_platforms=getattr(cfg.analysis, "ticket_platforms", None),
            exclude_paths=cfg.analysis.exclude_paths,
            ml_categorization_config=ml_config,
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
        identity_report_path = cfg.cache.directory / f'identity_analysis_{datetime.now().strftime("%Y%m%d")}.yaml'
        identity_result = analysis_pass.run_analysis(
            all_commits,
            output_path=identity_report_path,
            apply_to_config=False
        )
        
        click.echo(f"\n📄 Analysis report saved to: {identity_report_path}")
        
        if identity_result.clusters:
            # Generate suggested configuration
            suggested_config = analysis_pass.generate_suggested_config(identity_result)
            
            # Show suggestions
            click.echo(f"\n⚠️  Found {len(identity_result.clusters)} potential identity clusters:")
            
            # Display all mappings
            if suggested_config.get('analysis', {}).get('manual_identity_mappings'):
                click.echo("\n📋 Suggested identity mappings:")
                for mapping in suggested_config['analysis']['manual_identity_mappings']:
                    canonical = mapping['canonical_email']
                    aliases = mapping.get('aliases', [])
                    if aliases:
                        click.echo(f"   {canonical}")
                        for alias in aliases:
                            click.echo(f"     → {alias}")
            
            # Check for bot exclusions
            if suggested_config.get('exclude', {}).get('authors'):
                bot_count = len(suggested_config['exclude']['authors'])
                click.echo(f"\n🤖 Found {bot_count} bot accounts to exclude:")
                for bot in suggested_config['exclude']['authors']:
                    click.echo(f"   - {bot}")
            
            # Apply if requested
            if apply or click.confirm("\nApply these identity mappings to your configuration?", default=True):
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
    """List all known developers."""
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


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
