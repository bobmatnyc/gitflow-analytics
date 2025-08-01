"""Command-line interface for GitFlow Analytics."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, cast

import click
import git
import pandas as pd

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
from .reports.narrative_writer import NarrativeReportGenerator


@click.group()
@click.version_option(version=__version__, prog_name="GitFlow Analytics")
def cli() -> None:
    """GitFlow Analytics - Analyze Git repositories for productivity insights."""
    pass


# TUI command removed - replaced with rich CLI output
# Legacy TUI code preserved but not exposed


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
@click.option("--rich", is_flag=True, default=True, help="Use rich terminal output (default: enabled)")
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
    rich: bool,
) -> None:
    """Analyze Git repositories using configuration file."""

    # Initialize display - use rich by default, fall back to simple output if needed
    display = create_rich_display() if rich else None
    
    try:
        if display:
            display.show_header()
        
        # Load configuration
        if display:
            display.print_status(f"Loading configuration from {config}...", "info")
        else:
            click.echo(f"📋 Loading configuration from {config}...")
            
        cfg = ConfigLoader.load(config)

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

        analyzer = GitAnalyzer(
            cache,
            branch_mapping_rules=cfg.analysis.branch_mapping_rules,
            allowed_ticket_platforms=getattr(cfg.analysis, "ticket_platforms", None),
            exclude_paths=cfg.analysis.exclude_paths,
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

                        git.Repo.clone_from(clone_url, repo_config.path, branch=repo_config.branch)
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
                from .qualitative import QualitativeProcessor
                from .models.database import Database
                
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

        # Generate reports
        if display:
            display.print_status("Generating reports...", "info")
        else:
            click.echo("\n📊 Generating reports...")
        report_gen = CSVReportGenerator(anonymize=anonymize or cfg.output.anonymize_enabled)
        analytics_gen = AnalyticsReportGenerator(
            anonymize=anonymize or cfg.output.anonymize_enabled
        )

        # Collect generated report files for display
        generated_reports = []
        
        # Weekly metrics report
        weekly_report = output / f'weekly_metrics_{datetime.now(timezone.utc).strftime("%Y%m%d")}.csv'
        try:
            report_gen.generate_weekly_report(all_commits, developer_stats, weekly_report, weeks)
            generated_reports.append(weekly_report.name)
            if not display:
                click.echo(f"   ✅ Weekly metrics: {weekly_report}")
        except Exception as e:
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
                all_commits, all_prs, developer_stats, ticket_analysis, summary_report
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

        # Activity distribution report
        activity_report = output / f'activity_distribution_{datetime.now().strftime("%Y%m%d")}.csv'
        try:
            analytics_gen.generate_activity_distribution_report(
                all_commits, developer_stats, activity_report
            )
            generated_reports.append(activity_report.name)
            if not display:
                click.echo(f"   ✅ Activity distribution: {activity_report}")
        except Exception as e:
            click.echo(f"   ❌ Error generating activity distribution report: {e}")
            click.echo(f"   🔍 Error type: {type(e).__name__}")
            click.echo(f"   📍 Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        # Developer focus report
        focus_report = output / f'developer_focus_{datetime.now().strftime("%Y%m%d")}.csv'
        try:
            analytics_gen.generate_developer_focus_report(
                all_commits, developer_stats, focus_report, weeks
            )
            generated_reports.append(focus_report.name)
            if not display:
                click.echo(f"   ✅ Developer focus: {focus_report}")
        except Exception as e:
            click.echo(f"   ❌ Error generating developer focus report: {e}")
            click.echo(f"   🔍 Error type: {type(e).__name__}")
            click.echo(f"   📍 Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        # Qualitative insights report
        insights_report = output / f'qualitative_insights_{datetime.now().strftime("%Y%m%d")}.csv'
        try:
            analytics_gen.generate_qualitative_insights_report(
                all_commits, developer_stats, ticket_analysis, insights_report
            )
            generated_reports.append(insights_report.name)
            if not display:
                click.echo(f"   ✅ Qualitative insights: {insights_report}")
        except Exception as e:
            click.echo(f"   ❌ Error generating qualitative insights report: {e}")
            click.echo(f"   🔍 Error type: {type(e).__name__}")
            click.echo(f"   📍 Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        # Calculate DORA metrics
        dora_calculator = DORAMetricsCalculator()
        dora_metrics = dora_calculator.calculate_dora_metrics(
            all_commits, all_prs, start_date, end_date
        )

        # Aggregate PR metrics
        pr_metrics = {}
        for enrichment in all_enrichments.values():
            if enrichment.get("pr_metrics"):
                # Combine metrics (simplified - in production would properly aggregate)
                pr_metrics = enrichment["pr_metrics"]
                break

        # Generate narrative report if markdown format is enabled
        if "markdown" in cfg.output.formats:
            narrative_gen = NarrativeReportGenerator()

            # Load activity distribution data
            activity_df = pd.read_csv(activity_report)
            activity_data = cast(list[dict[str, Any]], activity_df.to_dict("records"))

            # Load focus data
            focus_df = pd.read_csv(focus_report)
            focus_data = cast(list[dict[str, Any]], focus_df.to_dict("records"))

            # Load insights data
            insights_df = pd.read_csv(insights_report)
            insights_data = cast(list[dict[str, Any]], insights_df.to_dict("records"))

            narrative_report = output / f'narrative_report_{datetime.now().strftime("%Y%m%d")}.md'
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
            )
            generated_reports.append(narrative_report.name)
            if not display:
                click.echo(f"   ✅ Narrative report: {narrative_report}")

        # Generate JSON export if enabled
        if "json" in cfg.output.formats:
            json_report = output / f'gitflow_export_{datetime.now().strftime("%Y%m%d")}.json'

            project_metrics = {
                "ticket_analysis": ticket_analysis,
                "pr_metrics": pr_metrics,
                "enrichments": all_enrichments,
            }

            orchestrator.export_to_json(
                all_commits,
                all_prs,
                developer_stats,
                project_metrics,
                dora_metrics,
                str(json_report),
            )
            generated_reports.append(json_report.name)
            if not display:
                click.echo(f"   ✅ JSON export: {json_report}")

        total_story_points = sum(c.get("story_points", 0) or 0 for c in all_commits)
        qualitative_count = len(qualitative_results) if qualitative_results else 0
        
        # Show results summary
        if display:
            display.show_analysis_summary(
                total_commits=len(all_commits),
                total_prs=len(all_prs),
                active_developers=len(developer_stats),
                ticket_coverage=ticket_analysis['commit_coverage_pct'],
                story_points=total_story_points,
                qualitative_analyzed=qualitative_count
            )
            
            # Show DORA metrics
            if dora_metrics:
                display.show_dora_metrics(dora_metrics)
            
            # Show generated reports
            display.show_reports_generated(output, generated_reports)
            
            display.print_status("Analysis complete!", "success")
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
        if display:
            display.show_error(str(e), show_debug_hint=True)
        else:
            click.echo(f"\n❌ Error: {e}", err=True)
        
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
