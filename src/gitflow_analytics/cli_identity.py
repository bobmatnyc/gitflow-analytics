"""Identity and alias management CLI commands for GitFlow Analytics.

This module contains all commands related to developer identity resolution
and alias management, extracted from cli.py for maintainability.
"""

import logging
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import click
import yaml

from .config import ConfigLoader
from .utils.date_utils import get_week_start

logger = logging.getLogger(__name__)


def register_identity_commands(cli_group: click.Group) -> None:
    """Register all identity/alias commands onto the provided Click group.

    Args:
        cli_group: The Click group to register commands on (typically the main cli group)
    """
    cli_group.add_command(merge_identity)
    cli_group.add_command(identities)
    cli_group.add_command(aliases_command)
    cli_group.add_command(list_developers)
    cli_group.add_command(create_alias_interactive)
    cli_group.add_command(alias_rename)


@click.command()
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
    from .core.identity import DeveloperIdentityResolver

    try:
        cfg = ConfigLoader.load(config)
        identity_resolver = DeveloperIdentityResolver(cfg.cache.directory / "identities.db")

        click.echo(f"🔄 Merging {dev2} into {dev1}...")
        identity_resolver.merge_identities(dev1, dev2)
        click.echo("✅ Identities merged successfully!")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@click.command()
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
    from .core.analyzer import GitAnalyzer
    from .core.cache import GitAnalysisCache

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
            allowed_ticket_platforms=cfg.get_effective_ticket_platforms(),
            exclude_paths=cfg.analysis.exclude_paths,
            story_point_patterns=cfg.analysis.story_point_patterns,
            ml_categorization_config=ml_config,
            llm_config=llm_config,
            branch_analysis_config=branch_analysis_config,
            exclude_merge_commits=cfg.analysis.exclude_merge_commits,
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
            cfg.cache.directory / f"identity_analysis_{datetime.now().strftime('%Y%m%d')}.yaml"
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

            # Display all mappings with confidence scores
            if suggested_config.get("analysis", {}).get("manual_identity_mappings"):
                click.echo("\n📋 Suggested identity mappings:")
                for i, mapping in enumerate(
                    suggested_config["analysis"]["manual_identity_mappings"], 1
                ):
                    canonical = mapping["primary_email"]
                    aliases = mapping.get("aliases", [])
                    confidence = mapping.get("confidence", 0.0)
                    reasoning = mapping.get("reasoning", "")

                    # Color-code based on confidence (90%+ threshold)
                    if confidence >= 0.95:
                        confidence_indicator = "🟢"  # Very high confidence
                    elif confidence >= 0.90:
                        confidence_indicator = "🟡"  # High confidence (above threshold)
                    else:
                        confidence_indicator = "🟠"  # Medium confidence (below threshold)

                    if aliases:
                        click.echo(
                            f"\n   {confidence_indicator} Cluster {i} "
                            f"(Confidence: {confidence:.1%}):"
                        )
                        click.echo(f"      Primary: {canonical}")
                        for alias in aliases:
                            click.echo(f"      Alias:   {alias}")

                        # Show reasoning if available
                        if reasoning:
                            # Truncate reasoning for display
                            display_reasoning = (
                                reasoning if len(reasoning) <= 80 else reasoning[:77] + "..."
                            )
                            click.echo(f"      Reason:  {display_reasoning}")

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


@click.command(name="aliases")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to configuration file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for aliases.yaml (default: same dir as config)",
)
@click.option(
    "--confidence-threshold",
    type=float,
    default=0.9,
    help="Minimum confidence threshold for LLM matches (default: 0.9)",
)
@click.option(
    "--apply", is_flag=True, help="Automatically update config to use generated aliases file"
)
@click.option(
    "--weeks", type=int, default=12, help="Number of weeks of history to analyze (default: 12)"
)
def aliases_command(
    config: Path,
    output: Optional[Path],
    confidence_threshold: float,
    apply: bool,
    weeks: int,
) -> None:
    """Generate developer identity aliases using LLM analysis.

    \b
    This command analyzes commit history and uses LLM to identify
    developer aliases (same person with different email addresses).
    Results are saved to aliases.yaml which can be shared across
    multiple config files.

    \b
    EXAMPLES:
        # Generate aliases and review
        gitflow-analytics aliases -c config.yaml

        # Generate and apply automatically
        gitflow-analytics aliases -c config.yaml --apply

        # Save to specific location
        gitflow-analytics aliases -c config.yaml -o ~/shared/aliases.yaml

        # Use longer history for better accuracy
        gitflow-analytics aliases -c config.yaml --weeks 24

    \b
    CONFIGURATION:
        Aliases are saved to aliases.yaml and can be referenced in
        multiple config files for consistent identity resolution.
    """
    try:
        from .config.aliases import AliasesManager, DeveloperAlias
        from .core.analyzer import GitAnalyzer
        from .core.cache import GitAnalysisCache
        from .identity_llm.analyzer import LLMIdentityAnalyzer

        # Load configuration
        click.echo(f"\n📋 Loading configuration from {config}...")
        cfg = ConfigLoader.load(config)

        # Determine output path
        if not output:
            output = config.parent / "aliases.yaml"

        click.echo(f"🔍 Analyzing developer identities (last {weeks} weeks)")
        click.echo(f"📊 Confidence threshold: {confidence_threshold:.0%}")
        click.echo(f"💾 Output: {output}\n")

        # Set up date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)

        # Analyze repositories to collect commits
        click.echo("📥 Fetching commit history...\n")
        cache = GitAnalysisCache(cfg.cache.directory)

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
            allowed_ticket_platforms=cfg.get_effective_ticket_platforms(),
            exclude_paths=cfg.analysis.exclude_paths,
            story_point_patterns=cfg.analysis.story_point_patterns,
            ml_categorization_config=ml_config,
            llm_config=llm_config,
            branch_analysis_config=branch_analysis_config,
            exclude_merge_commits=cfg.analysis.exclude_merge_commits,
        )

        all_commits = []

        # Get repositories to analyze
        repositories = cfg.repositories if cfg.repositories else []

        if not repositories:
            click.echo("❌ No repositories configured", err=True)
            sys.exit(1)

        # Collect commits from all repositories
        with click.progressbar(
            repositories,
            label="Analyzing repositories",
            item_show_func=lambda r: r.name if r else "",
        ) as repos:
            for repo_config in repos:
                try:
                    if not repo_config.path.exists():
                        continue

                    # Fetch commits
                    repo_commits = analyzer.analyze_repository(
                        repo_config.path, since=start_date, branch=repo_config.branch
                    )

                    if repo_commits:
                        all_commits.extend(repo_commits)

                except Exception as e:
                    click.echo(f"\n⚠️  Warning: Failed to analyze repository: {e}", err=True)
                    continue

        click.echo(f"\n✅ Collected {len(all_commits)} commits\n")

        if not all_commits:
            click.echo("❌ No commits found to analyze", err=True)
            sys.exit(1)

        # Initialize LLM identity analyzer
        click.echo("🤖 Running LLM identity analysis...\n")

        # Get OpenRouter API key from config
        api_key = None
        if cfg.chatgpt and cfg.chatgpt.api_key:
            # Resolve environment variable if needed
            api_key_value = cfg.chatgpt.api_key
            if api_key_value.startswith("${") and api_key_value.endswith("}"):
                var_name = api_key_value[2:-1]
                api_key = os.getenv(var_name)
            else:
                api_key = api_key_value

        if not api_key:
            click.echo(
                "⚠️  No OpenRouter API key configured - using heuristic analysis only", err=True
            )

        llm_analyzer = LLMIdentityAnalyzer(
            api_key=api_key, confidence_threshold=confidence_threshold
        )

        # Run analysis
        result = llm_analyzer.analyze_identities(all_commits)

        click.echo("✅ Analysis complete:")
        click.echo(f"   - Found {len(result.clusters)} identity clusters")
        click.echo(f"   - {len(result.unresolved_identities)} unresolved identities")
        click.echo(f"   - Method: {result.analysis_metadata.get('analysis_method', 'unknown')}\n")

        # Create aliases manager and add clusters
        aliases_mgr = AliasesManager(output)

        # Load existing aliases if file exists
        if output.exists():
            click.echo(f"📂 Loading existing aliases from {output}...")
            aliases_mgr.load()
            existing_count = len(aliases_mgr.aliases)
            click.echo(f"   Found {existing_count} existing aliases\n")

        # Add new clusters
        new_count = 0
        updated_count = 0

        for cluster in result.clusters:
            # Check if this is a new or updated alias
            existing = aliases_mgr.get_alias(cluster.canonical_email)

            alias = DeveloperAlias(
                name=cluster.preferred_display_name or cluster.canonical_name,
                primary_email=cluster.canonical_email,
                aliases=[a.email for a in cluster.aliases],
                confidence=cluster.confidence,
                reasoning=(
                    cluster.reasoning[:200] if cluster.reasoning else ""
                ),  # Truncate for readability
            )

            if existing:
                updated_count += 1
            else:
                new_count += 1

            aliases_mgr.add_alias(alias)

        # Save aliases
        click.echo("💾 Saving aliases...\n")
        aliases_mgr.save()

        click.echo(f"✅ Saved to {output}")
        click.echo(f"   - New aliases: {new_count}")
        click.echo(f"   - Updated aliases: {updated_count}")
        click.echo(f"   - Total aliases: {len(aliases_mgr.aliases)}\n")

        # Display summary
        if aliases_mgr.aliases:
            click.echo("📋 Generated Aliases:\n")

            for alias in sorted(aliases_mgr.aliases, key=lambda a: a.primary_email):
                name_display = (
                    f"{alias.name} <{alias.primary_email}>" if alias.name else alias.primary_email
                )
                click.echo(f"  • {name_display}")

                if alias.aliases:
                    for alias_email in alias.aliases:
                        click.echo(f"    → {alias_email}")

                if alias.confidence < 1.0:
                    confidence_color = (
                        "green"
                        if alias.confidence >= 0.9
                        else "yellow"
                        if alias.confidence >= 0.8
                        else "red"
                    )
                    click.echo("    Confidence: ", nl=False)
                    click.secho(f"{alias.confidence:.0%}", fg=confidence_color)

                click.echo()  # Blank line between aliases

        # Apply to config if requested
        if apply:
            click.echo(f"🔄 Updating {config} to reference aliases file...\n")

            # Read current config
            with open(config) as f:
                config_data = yaml.safe_load(f)

            # Ensure analysis section exists
            if "analysis" not in config_data:
                config_data["analysis"] = {}

            if "identity" not in config_data["analysis"]:
                config_data["analysis"]["identity"] = {}

            # Calculate relative path from config to aliases file
            try:
                rel_path = output.relative_to(config.parent)
                config_data["analysis"]["identity"]["aliases_file"] = str(rel_path)
            except ValueError:
                # Not relative, use absolute
                config_data["analysis"]["identity"]["aliases_file"] = str(output)

            # Remove manual_mappings if present (now in aliases file)
            if "manual_identity_mappings" in config_data["analysis"].get("identity", {}):
                del config_data["analysis"]["identity"]["manual_identity_mappings"]
                click.echo("   Removed inline manual_identity_mappings (now in aliases file)")

            # Save updated config
            with open(config, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

            click.echo(f"✅ Updated {config}")
            click.echo(
                f"   Added: analysis.identity.aliases_file = "
                f"{config_data['analysis']['identity']['aliases_file']}\n"
            )

        # Summary and next steps
        click.echo("✨ Identity alias generation complete!\n")

        if not apply:
            click.echo("💡 Next steps:")
            click.echo(f"   1. Review the aliases in {output}")
            click.echo("   2. Update your config.yaml to reference the aliases file:")
            click.echo("      analysis:")
            click.echo("        identity:")
            click.echo(f"          aliases_file: {output.name}")
            click.echo("   3. Or run with --apply flag to update automatically\n")

    except Exception as e:
        click.echo(f"\n❌ Error generating aliases: {e}", err=True)

        if os.getenv("GITFLOW_DEBUG"):
            traceback.print_exc()
        sys.exit(1)


@click.command()
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
    from .core.identity import DeveloperIdentityResolver

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
        # Load configuration
        cfg = ConfigLoader.load(config)

        # Determine output path for aliases file
        if not output:
            output = config.parent / "aliases.yaml"

        # Initialize identity resolver
        identity_resolver = DeveloperIdentityResolver(cfg.cache.directory / "identities.db")

        # Initialize aliases manager
        aliases_manager = AliasesManager(output if output.exists() else None)

        click.echo("\n" + "=" * 80)
        click.echo(click.style("🔧 Interactive Alias Creator", fg="cyan", bold=True))
        click.echo("=" * 80 + "\n")

        # Main loop for creating multiple aliases
        continue_creating = True

        while continue_creating:
            # Get all developers from database
            developers = identity_resolver.get_developer_stats()

            if not developers:
                click.echo("❌ No developers found. Run analysis first.")
                sys.exit(1)

            # Display numbered list of developers
            click.echo(
                click.style(f"\n📋 Found {len(developers)} developers:\n", fg="green", bold=True)
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

            # Get user selection
            while True:
                try:
                    selection_input = click.prompt(
                        click.style(
                            "Select developers to merge (enter numbers separated by spaces, or 'q' to quit)",
                            fg="yellow",
                        ),
                        type=str,
                    ).strip()

                    # Handle quit
                    if selection_input.lower() in ["q", "quit", "exit"]:
                        click.echo("\n👋 Exiting alias creation.")
                        sys.exit(0)

                    # Parse selection
                    selected_indices = []
                    for num_str in selection_input.split():
                        try:
                            num = int(num_str)
                            if 1 <= num <= len(developers):
                                selected_indices.append(num)
                            else:
                                click.echo(
                                    click.style(
                                        f"⚠️  Number {num} is out of range (1-{len(developers)})",
                                        fg="red",
                                    )
                                )
                                raise ValueError("Invalid range")
                        except ValueError:
                            click.echo(
                                click.style(
                                    f"⚠️  Invalid input: '{num_str}' is not a valid number", fg="red"
                                )
                            )
                            raise

                    # Check minimum selection
                    if len(selected_indices) < 2:
                        click.echo(
                            click.style(
                                "⚠️  You must select at least 2 developers to merge", fg="red"
                            )
                        )
                        continue

                    # Remove duplicates and sort
                    selected_indices = sorted(set(selected_indices))
                    break

                except ValueError:
                    continue
                except click.exceptions.Abort:
                    click.echo("\n\n👋 Exiting alias creation.")
                    sys.exit(0)

            # Display selected developers
            selected_devs = [developers[idx - 1] for idx in selected_indices]

            click.echo(click.style("\n✅ Selected developers:", fg="green", bold=True))
            for idx, dev in zip(selected_indices, selected_devs):
                click.echo(
                    f"  [{idx}] {dev['primary_name']} <{dev['primary_email']}> "
                    f"({dev['total_commits']} commits)"
                )

            # Ask which one should be primary
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
                                f"⚠️  Please select one of: {', '.join(map(str, selected_indices))}",
                                fg="red",
                            )
                        )
                except ValueError:
                    click.echo(click.style("⚠️  Please enter a valid number", fg="red"))
                except click.exceptions.Abort:
                    click.echo("\n\n👋 Exiting alias creation.")
                    sys.exit(0)

            # Build alias configuration
            primary_dev = developers[primary_idx - 1]
            alias_emails = [
                dev["primary_email"]
                for idx, dev in zip(selected_indices, selected_devs)
                if idx != primary_idx
            ]

            # Create the alias
            new_alias = DeveloperAlias(
                primary_email=primary_dev["primary_email"],
                aliases=alias_emails,
                name=primary_dev["primary_name"],
                confidence=1.0,  # Manual aliases have full confidence
                reasoning="Manually created via interactive CLI",
            )

            # Display the alias configuration
            click.echo(click.style("\n📝 Alias Configuration:", fg="cyan", bold=True))
            click.echo(f"  Primary: {new_alias.name} <{new_alias.primary_email}>")
            click.echo("  Aliases:")
            for alias_email in new_alias.aliases:
                click.echo(f"    - {alias_email}")

            # Add to aliases manager
            aliases_manager.add_alias(new_alias)

            # Ask if user wants to save
            click.echo()
            if click.confirm(click.style(f"💾 Save alias to {output}?", fg="green"), default=True):
                try:
                    aliases_manager.save()
                    click.echo(click.style(f"✅ Alias saved to {output}", fg="green"))

                    # Also update the database directly by merging identities
                    # For each alias email, find its canonical_id and merge with primary
                    for alias_email in alias_emails:
                        # Find the developer entry for this alias email
                        alias_dev = next(
                            (dev for dev in developers if dev["primary_email"] == alias_email), None
                        )

                        if alias_dev:
                            # Merge using canonical IDs
                            identity_resolver.merge_identities(
                                primary_dev["canonical_id"],  # Primary's canonical_id
                                alias_dev["canonical_id"],  # Alias's canonical_id
                            )
                        else:
                            # Edge case: alias email doesn't match any developer
                            # This shouldn't happen, but log a warning
                            click.echo(
                                click.style(
                                    f"⚠️  Warning: Could not find developer entry for {alias_email}",
                                    fg="yellow",
                                )
                            )

                    click.echo(
                        click.style("✅ Database updated with merged identities", fg="green")
                    )

                except Exception as e:
                    click.echo(click.style(f"❌ Error saving alias: {e}", fg="red"), err=True)
            else:
                click.echo(click.style("⏭️  Alias not saved", fg="yellow"))

            # Ask if user wants to create more aliases
            click.echo()
            if not click.confirm(click.style("🔄 Create another alias?", fg="cyan"), default=True):
                continue_creating = False

        click.echo(click.style("\n✅ Alias creation completed!", fg="green", bold=True))
        click.echo(f"📄 Aliases file: {output}")
        click.echo("\n💡 To use these aliases, ensure your config references: {output}\n")

    except KeyboardInterrupt:
        click.echo("\n\n👋 Interrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        click.echo(click.style(f"\n❌ Error: {e}", fg="red"), err=True)

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

        # Load the YAML config file
        click.echo(f"\n📋 Loading configuration from {config}...")

        try:
            with open(config, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            click.echo(f"❌ Error loading config file: {e}", err=True)
            sys.exit(1)

        # Navigate to analysis.identity.manual_mappings
        if "analysis" not in config_data:
            click.echo("❌ Error: 'analysis' section not found in config", err=True)
            sys.exit(1)

        if "identity" not in config_data["analysis"]:
            click.echo("❌ Error: 'analysis.identity' section not found in config", err=True)
            sys.exit(1)

        if "manual_mappings" not in config_data["analysis"]["identity"]:
            click.echo(
                "❌ Error: 'analysis.identity.manual_mappings' not found in config", err=True
            )
            sys.exit(1)

        manual_mappings = config_data["analysis"]["identity"]["manual_mappings"]

        if not manual_mappings:
            click.echo("❌ Error: manual_mappings is empty", err=True)
            sys.exit(1)

        # Validate explicitly empty strings before entering interactive mode
        # (prevents infinite click.prompt() loop when "" is passed)
        if old_name is not None and not old_name.strip():
            click.echo("❌ Error: --old-name cannot be empty", err=True)
            sys.exit(1)

        if new_name is not None and not new_name.strip():
            click.echo("❌ Error: --new-name cannot be empty", err=True)
            sys.exit(1)

        # Interactive mode: display numbered list and prompt for selection
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

            # Prompt for selection
            try:
                selection = click.prompt(
                    "Select developer number to rename (or 0 to cancel)",
                    type=click.IntRange(0, len(developer_names)),
                )
            except click.Abort:
                click.echo("\n👋 Cancelled by user.")
                sys.exit(0)

            if selection == 0:
                click.echo("\n👋 Cancelled.")
                sys.exit(0)

            # Get selected developer name
            old_name = developer_names[selection - 1]
            click.echo(f"\n📝 Selected: {click.style(old_name, fg='green')}")

            # Prompt for new name if not provided
            if not new_name:
                new_name = click.prompt("Enter new canonical name", type=str)

        # Validate inputs
        if not old_name or not old_name.strip():
            click.echo("❌ Error: --old-name cannot be empty", err=True)
            sys.exit(1)

        if not new_name or not new_name.strip():
            click.echo("❌ Error: --new-name cannot be empty", err=True)
            sys.exit(1)

        old_name = old_name.strip()
        new_name = new_name.strip()

        if old_name == new_name:
            click.echo("❌ Error: old-name and new-name are identical", err=True)
            sys.exit(1)

        # Find the matching entry
        matching_entry = None
        matching_index = None

        for idx, mapping in enumerate(manual_mappings):
            if mapping.get("name") == old_name:
                matching_entry = mapping
                matching_index = idx
                break

        if not matching_entry:
            click.echo(f"❌ Error: No manual mapping found with name '{old_name}'", err=True)
            click.echo("\nAvailable names in manual_mappings:")
            for mapping in manual_mappings:
                if "name" in mapping:
                    click.echo(f"  - {mapping['name']}")
            sys.exit(1)

        # Display what will be changed
        click.echo("\n🔍 Found matching entry:")
        click.echo(f"   Current name: {old_name}")
        click.echo(f"   New name:     {new_name}")
        click.echo(f"   Email:        {matching_entry.get('primary_email', 'N/A')}")
        click.echo(f"   Aliases:      {len(matching_entry.get('aliases', []))} email(s)")

        if dry_run:
            click.echo("\n🔎 DRY RUN - No changes will be made")

        # Update the config file
        if not dry_run:
            click.echo("\n📝 Updating configuration file...")
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
                click.echo("✅ Configuration file updated")
            except Exception as e:
                click.echo(f"❌ Error writing config file: {e}", err=True)
                sys.exit(1)
        else:
            click.echo(f"   [Would update config: {config}]")

        # Update database cache if requested
        if update_cache:
            click.echo("\n💾 Checking database cache...")

            # Load config to get cache directory
            cfg = ConfigLoader.load(config)
            identity_db_path = cfg.cache.directory / "identities.db"

            if not identity_db_path.exists():
                click.echo(f"⚠️  Warning: Identity database not found at {identity_db_path}")
                click.echo("   Skipping cache update")
            else:
                # Initialize identity resolver to access database
                identity_resolver = DeveloperIdentityResolver(
                    str(identity_db_path),
                    manual_mappings=None,  # Don't apply mappings during rename
                )

                # Count affected records
                from sqlalchemy import text

                with identity_resolver.get_session() as session:
                    # Count developer_identities records
                    result = session.execute(
                        text(
                            "SELECT COUNT(*) FROM developer_identities WHERE primary_name = :old_name"
                        ),
                        {"old_name": old_name},
                    )
                    identity_count = result.scalar()

                    # Count developer_aliases records
                    result = session.execute(
                        text("SELECT COUNT(*) FROM developer_aliases WHERE name = :old_name"),
                        {"old_name": old_name},
                    )
                    alias_count = result.scalar()

                click.echo(f"   Found {identity_count} identity record(s)")
                click.echo(f"   Found {alias_count} alias record(s)")

                if identity_count == 0 and alias_count == 0:
                    click.echo("   ℹ️  No database records to update")
                elif not dry_run:
                    click.echo("   Updating database records...")

                    with identity_resolver.get_session() as session:
                        # Update developer_identities
                        if identity_count > 0:
                            session.execute(
                                text(
                                    "UPDATE developer_identities SET primary_name = :new_name WHERE primary_name = :old_name"
                                ),
                                {"new_name": new_name, "old_name": old_name},
                            )

                        # Update developer_aliases
                        if alias_count > 0:
                            session.execute(
                                text(
                                    "UPDATE developer_aliases SET name = :new_name WHERE name = :old_name"
                                ),
                                {"new_name": new_name, "old_name": old_name},
                            )

                    click.echo("   ✅ Database updated")
                else:
                    click.echo(
                        f"   [Would update {identity_count + alias_count} database record(s)]"
                    )

        # Summary
        click.echo(f"\n{'🔎 DRY RUN SUMMARY' if dry_run else '✅ RENAME COMPLETE'}")
        click.echo(f"   Old name: {old_name}")
        click.echo(f"   New name: {new_name}")
        click.echo(f"   Config:   {'Would update' if dry_run else 'Updated'}")
        if update_cache:
            click.echo(f"   Cache:    {'Would update' if dry_run else 'Updated'}")
        else:
            click.echo("   Cache:    Skipped (use --update-cache to update)")

        if dry_run:
            click.echo("\n💡 Run without --dry-run to apply changes")
        else:
            click.echo("\n💡 Next steps:")
            click.echo(f"   - Review the updated config file: {config}")
            click.echo("   - Re-run analysis to see updated reports with new name")

    except KeyboardInterrupt:
        click.echo("\n\n👋 Interrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)

        traceback.print_exc()
        sys.exit(1)
