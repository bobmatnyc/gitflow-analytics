"""Command-line interface for GitFlow Analytics."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, cast, Dict, Any, List

import click
import git
import pandas as pd

from ._version import __version__
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
def analyze(
    config: Path,
    weeks: int,
    output: Optional[Path],
    anonymize: bool,
    no_cache: bool,
    validate_only: bool,
    clear_cache: bool,
) -> None:
    """Analyze Git repositories using configuration file."""

    try:
        # Load configuration
        click.echo(f"📋 Loading configuration from {config}...")
        cfg = ConfigLoader.load(config)

        # Validate configuration
        warnings = ConfigLoader.validate_config(cfg)
        if warnings:
            click.echo("⚠️  Configuration warnings:")
            for warning in warnings:
                click.echo(f"   - {warning}")

        if validate_only:
            if not warnings:
                click.echo("✅ Configuration is valid!")
            else:
                click.echo("❌ Configuration has issues that should be addressed.")
            return

        # Use output directory from CLI or config
        if output is None:
            output = cfg.output.directory if cfg.output.directory else Path("./reports")

        # Setup output directory
        output.mkdir(parents=True, exist_ok=True)

        # Initialize components
        cache_dir = cfg.cache.directory
        if clear_cache:
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
            click.echo(f"🔍 Discovering repositories from organization: {cfg.github.organization}")
            try:
                # Use a 'repos' directory in the config directory for cloned repositories
                config_dir = Path(config).parent if config else Path.cwd()
                repos_dir = config_dir / "repos"
                discovered_repos = cfg.discover_organization_repositories(clone_base_path=repos_dir)
                repositories_to_analyze = discovered_repos
                click.echo(f"   ✅ Found {len(discovered_repos)} repositories in organization")
                for repo in discovered_repos:
                    click.echo(f"      - {repo.name} ({repo.github_repo})")
            except Exception as e:
                click.echo(f"   ❌ Failed to discover repositories: {e}")
                return

        # Analysis period
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=weeks)

        click.echo(f"\n🚀 Analyzing {len(repositories_to_analyze)} repositories...")
        click.echo(
            f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        # Analyze repositories
        all_commits = []
        all_prs = []
        all_enrichments = {}

        for repo_config in repositories_to_analyze:
            click.echo(f"\n📁 Analyzing {repo_config.name}...")

            # Check if repo exists, clone if needed
            if not repo_config.path.exists():
                # Try to clone if we have a github_repo configured
                if repo_config.github_repo and cfg.github.organization:
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
                        click.echo(f"   ✅ Successfully cloned {repo_config.github_repo}")
                    except Exception as e:
                        click.echo(f"   ❌ Failed to clone repository: {e}")
                        continue
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
                click.echo(f"   ✅ Found {len(commits)} commits")

                # Enrich with integration data
                enrichment = orchestrator.enrich_repository_data(repo_config, commits, start_date)
                all_enrichments[repo_config.name] = enrichment

                if enrichment["prs"]:
                    all_prs.extend(enrichment["prs"])
                    click.echo(f"   ✅ Found {len(enrichment['prs'])} pull requests")

            except Exception as e:
                click.echo(f"   ❌ Error: {e}")
                continue

        if not all_commits:
            click.echo("\n❌ No commits found in the specified period!")
            return

        # Update developer statistics
        click.echo("\n👥 Resolving developer identities...")
        identity_resolver.update_commit_stats(all_commits)
        developer_stats = identity_resolver.get_developer_stats()
        click.echo(f"   ✅ Identified {len(developer_stats)} unique developers")

        # Analyze tickets
        click.echo("\n🎫 Analyzing ticket references...")
        ticket_extractor = TicketExtractor(
            allowed_platforms=getattr(cfg.analysis, "ticket_platforms", None)
        )
        ticket_analysis = ticket_extractor.analyze_ticket_coverage(all_commits, all_prs)

        for platform, count in ticket_analysis["ticket_summary"].items():
            click.echo(f"   - {platform.title()}: {count} unique tickets")

        # Generate reports
        click.echo("\n📊 Generating reports...")
        report_gen = CSVReportGenerator(anonymize=anonymize or cfg.output.anonymize_enabled)
        analytics_gen = AnalyticsReportGenerator(
            anonymize=anonymize or cfg.output.anonymize_enabled
        )

        # Weekly metrics report
        weekly_report = output / f'weekly_metrics_{datetime.now().strftime("%Y%m%d")}.csv'
        report_gen.generate_weekly_report(all_commits, developer_stats, weekly_report, weeks)
        click.echo(f"   ✅ Weekly metrics: {weekly_report}")

        # Summary report
        summary_report = output / f'summary_{datetime.now().strftime("%Y%m%d")}.csv'
        report_gen.generate_summary_report(
            all_commits, all_prs, developer_stats, ticket_analysis, summary_report
        )
        click.echo(f"   ✅ Summary stats: {summary_report}")

        # Developer report
        developer_report = output / f'developers_{datetime.now().strftime("%Y%m%d")}.csv'
        report_gen.generate_developer_report(developer_stats, developer_report)
        click.echo(f"   ✅ Developer stats: {developer_report}")

        # Activity distribution report
        activity_report = output / f'activity_distribution_{datetime.now().strftime("%Y%m%d")}.csv'
        analytics_gen.generate_activity_distribution_report(
            all_commits, developer_stats, activity_report
        )
        click.echo(f"   ✅ Activity distribution: {activity_report}")

        # Developer focus report
        focus_report = output / f'developer_focus_{datetime.now().strftime("%Y%m%d")}.csv'
        analytics_gen.generate_developer_focus_report(
            all_commits, developer_stats, focus_report, weeks
        )
        click.echo(f"   ✅ Developer focus: {focus_report}")

        # Qualitative insights report
        insights_report = output / f'qualitative_insights_{datetime.now().strftime("%Y%m%d")}.csv'
        analytics_gen.generate_qualitative_insights_report(
            all_commits, developer_stats, ticket_analysis, insights_report
        )
        click.echo(f"   ✅ Qualitative insights: {insights_report}")

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
            activity_data = cast(List[Dict[str, Any]], activity_df.to_dict("records"))

            # Load focus data
            focus_df = pd.read_csv(focus_report)
            focus_data = cast(List[Dict[str, Any]], focus_df.to_dict("records"))

            # Load insights data
            insights_df = pd.read_csv(insights_report)
            insights_data = cast(List[Dict[str, Any]], insights_df.to_dict("records"))

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
            click.echo(f"   ✅ JSON export: {json_report}")

        # Print summary
        click.echo("\n📈 Analysis Summary:")
        click.echo(f"   - Total commits: {len(all_commits)}")
        click.echo(f"   - Total PRs: {len(all_prs)}")
        click.echo(f"   - Active developers: {len(developer_stats)}")
        click.echo(f"   - Ticket coverage: {ticket_analysis['commit_coverage_pct']:.1f}%")

        total_story_points = sum(c.get("story_points", 0) or 0 for c in all_commits)
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
