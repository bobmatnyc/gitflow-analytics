"""Extended identity CLI commands for GitFlow Analytics.

This module contains identity analysis commands:
- identities: Analyze and manage developer identity resolution
- aliases_command: Generate developer aliases using LLM

Interactive alias ops (create_alias_interactive, alias_rename) live in
cli_identity_alias_ops.py. Simpler commands (merge_identity, list_developers)
live in cli_identity.py.
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
        current_week_start = get_week_start(current_time)
        last_complete_week_start = current_week_start - timedelta(weeks=1)
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

        click.echo("Analyzing repositories for developer identities...")

        all_commits = []
        for repo_config in cfg.repositories:
            if repo_config.path.exists():
                commits = analyzer.analyze_repository(
                    repo_config.path, start_date, repo_config.branch
                )
                all_commits.extend(commits)

        if not all_commits:
            click.echo("No commits found in the specified period!", err=True)
            return

        click.echo(f"Found {len(all_commits)} commits")

        from .identity_llm.analysis_pass import IdentityAnalysisPass

        analysis_pass = IdentityAnalysisPass(config)

        identity_report_path = (
            cfg.cache.directory / f"identity_analysis_{datetime.now().strftime('%Y%m%d')}.yaml"
        )
        identity_result = analysis_pass.run_analysis(
            all_commits, output_path=identity_report_path, apply_to_config=False
        )

        click.echo(f"\nAnalysis report saved to: {identity_report_path}")

        if identity_result.clusters:
            suggested_config = analysis_pass.generate_suggested_config(identity_result)

            click.echo(f"\nFound {len(identity_result.clusters)} potential identity clusters:")

            if suggested_config.get("analysis", {}).get("manual_identity_mappings"):
                click.echo("\nSuggested identity mappings:")
                for i, mapping in enumerate(
                    suggested_config["analysis"]["manual_identity_mappings"], 1
                ):
                    canonical = mapping["primary_email"]
                    aliases = mapping.get("aliases", [])
                    confidence = mapping.get("confidence", 0.0)
                    reasoning = mapping.get("reasoning", "")

                    if confidence >= 0.95:
                        confidence_indicator = "[high]"
                    elif confidence >= 0.90:
                        confidence_indicator = "[good]"
                    else:
                        confidence_indicator = "[low]"

                    if aliases:
                        click.echo(
                            f"\n   {confidence_indicator} Cluster {i} "
                            f"(Confidence: {confidence:.1%}):"
                        )
                        click.echo(f"      Primary: {canonical}")
                        for alias in aliases:
                            click.echo(f"      Alias:   {alias}")

                        if reasoning:
                            display_reasoning = (
                                reasoning if len(reasoning) <= 80 else reasoning[:77] + "..."
                            )
                            click.echo(f"      Reason:  {display_reasoning}")

            if suggested_config.get("exclude", {}).get("authors"):
                bot_count = len(suggested_config["exclude"]["authors"])
                click.echo(f"\nFound {bot_count} bot accounts to exclude:")
                for bot in suggested_config["exclude"]["authors"]:
                    click.echo(f"   - {bot}")

            if apply or click.confirm(
                "\nApply these identity mappings to your configuration?", default=True
            ):
                analysis_pass._apply_to_config(identity_result)
                click.echo("Applied identity mappings to configuration")

                last_prompt_file = cfg.cache.directory / ".identity_last_prompt"
                last_prompt_file.unlink(missing_ok=True)
        else:
            click.echo("No identity clusters found - all developers appear unique")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
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

        click.echo(f"\nLoading configuration from {config}...")
        cfg = ConfigLoader.load(config)

        if not output:
            output = config.parent / "aliases.yaml"

        click.echo(f"Analyzing developer identities (last {weeks} weeks)")
        click.echo(f"Confidence threshold: {confidence_threshold:.0%}")
        click.echo(f"Output: {output}\n")

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)

        click.echo("Fetching commit history...\n")
        cache = GitAnalysisCache(cfg.cache.directory)

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

        all_commits = []
        repositories = cfg.repositories if cfg.repositories else []

        if not repositories:
            click.echo("No repositories configured", err=True)
            sys.exit(1)

        with click.progressbar(
            repositories,
            label="Analyzing repositories",
            item_show_func=lambda r: r.name if r else "",
        ) as repos:
            for repo_config in repos:
                try:
                    if not repo_config.path.exists():
                        continue
                    repo_commits = analyzer.analyze_repository(
                        repo_config.path, since=start_date, branch=repo_config.branch
                    )
                    if repo_commits:
                        all_commits.extend(repo_commits)
                except Exception as e:
                    click.echo(f"\nWarning: Failed to analyze repository: {e}", err=True)
                    continue

        click.echo(f"\nCollected {len(all_commits)} commits\n")

        if not all_commits:
            click.echo("No commits found to analyze", err=True)
            sys.exit(1)

        click.echo("Running LLM identity analysis...\n")

        # ------------------------------------------------------------------
        # Determine LLM provider and build the analyzer.
        #
        # Priority order:
        #   1. Bedrock — if analysis.llm_classification.provider == "bedrock"
        #   2. OpenRouter — if cfg.chatgpt.api_key is set
        #   3. Heuristic-only fallback
        # ------------------------------------------------------------------
        llm_cfg = cfg.analysis.llm_classification

        # Resolve OpenRouter API key (support ${ENV_VAR} syntax)
        api_key: Optional[str] = None
        if cfg.chatgpt and cfg.chatgpt.api_key:
            api_key_value = cfg.chatgpt.api_key
            if api_key_value.startswith("${") and api_key_value.endswith("}"):
                var_name = api_key_value[2:-1]
                api_key = os.getenv(var_name)
            else:
                api_key = api_key_value

        # Also accept an OpenRouter key directly from llm_classification config
        if not api_key and llm_cfg.api_key:
            api_key = llm_cfg.api_key

        # Detect requested provider from llm_classification config
        requested_provider = llm_cfg.provider  # "auto" | "bedrock" | "openrouter" | "heuristic"

        # Map the llm_classification Bedrock settings for the identity analyzer
        bedrock_region = llm_cfg.aws_region or "us-east-1"
        bedrock_profile = llm_cfg.aws_profile
        bedrock_model_id = llm_cfg.bedrock_model_id

        # Collect user-configured strip suffixes (if any)
        extra_strip_suffixes = list(getattr(cfg.analysis.identity, "strip_suffixes", []))

        if requested_provider == "bedrock":
            click.echo("LLM provider: AWS Bedrock", err=True)
        elif api_key:
            click.echo("LLM provider: OpenRouter", err=True)
        else:
            click.echo(
                "Warning: No LLM provider configured - using heuristic analysis only",
                err=True,
            )

        llm_analyzer = LLMIdentityAnalyzer(
            api_key=api_key,
            confidence_threshold=confidence_threshold,
            provider=requested_provider,
            aws_region=bedrock_region,
            aws_profile=bedrock_profile,
            bedrock_model_id=bedrock_model_id,
            extra_strip_suffixes=extra_strip_suffixes,
        )

        result = llm_analyzer.analyze_identities(all_commits)

        click.echo("Analysis complete:")
        click.echo(f"   - Found {len(result.clusters)} identity clusters")
        click.echo(f"   - {len(result.unresolved_identities)} unresolved identities")
        click.echo(f"   - Method: {result.analysis_metadata.get('analysis_method', 'unknown')}\n")

        aliases_mgr = AliasesManager(output)

        if output.exists():
            click.echo(f"Loading existing aliases from {output}...")
            aliases_mgr.load()
            existing_count = len(aliases_mgr.aliases)
            click.echo(f"   Found {existing_count} existing aliases\n")

        new_count = 0
        updated_count = 0

        for cluster in result.clusters:
            existing = aliases_mgr.get_alias(cluster.canonical_email)

            alias = DeveloperAlias(
                name=cluster.preferred_display_name or cluster.canonical_name,
                primary_email=cluster.canonical_email,
                aliases=[a.email for a in cluster.aliases],
                confidence=cluster.confidence,
                reasoning=(cluster.reasoning[:200] if cluster.reasoning else ""),
            )

            if existing:
                updated_count += 1
            else:
                new_count += 1

            aliases_mgr.add_alias(alias)

        click.echo("Saving aliases...\n")
        aliases_mgr.save()

        click.echo(f"Saved to {output}")
        click.echo(f"   - New aliases: {new_count}")
        click.echo(f"   - Updated aliases: {updated_count}")
        click.echo(f"   - Total aliases: {len(aliases_mgr.aliases)}\n")

        if aliases_mgr.aliases:
            click.echo("Generated Aliases:\n")

            for alias in sorted(aliases_mgr.aliases, key=lambda a: a.primary_email):
                name_display = (
                    f"{alias.name} <{alias.primary_email}>" if alias.name else alias.primary_email
                )
                click.echo(f"  * {name_display}")

                if alias.aliases:
                    for alias_email in alias.aliases:
                        click.echo(f"    -> {alias_email}")

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

                click.echo()

        if apply:
            click.echo(f"Updating {config} to reference aliases file...\n")

            with open(config) as f:
                config_data = yaml.safe_load(f)

            if "analysis" not in config_data:
                config_data["analysis"] = {}

            if "identity" not in config_data["analysis"]:
                config_data["analysis"]["identity"] = {}

            try:
                rel_path = output.relative_to(config.parent)
                config_data["analysis"]["identity"]["aliases_file"] = str(rel_path)
            except ValueError:
                config_data["analysis"]["identity"]["aliases_file"] = str(output)

            if "manual_identity_mappings" in config_data["analysis"].get("identity", {}):
                del config_data["analysis"]["identity"]["manual_identity_mappings"]
                click.echo("   Removed inline manual_identity_mappings (now in aliases file)")

            with open(config, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

            click.echo(f"Updated {config}")
            click.echo(
                f"   Added: analysis.identity.aliases_file = "
                f"{config_data['analysis']['identity']['aliases_file']}\n"
            )

        click.echo("Identity alias generation complete!\n")

        if not apply:
            click.echo("Next steps:")
            click.echo(f"   1. Review the aliases in {output}")
            click.echo("   2. Update your config.yaml to reference the aliases file:")
            click.echo("      analysis:")
            click.echo("        identity:")
            click.echo(f"          aliases_file: {output.name}")
            click.echo("   3. Or run with --apply flag to update automatically\n")

    except Exception as e:
        click.echo(f"\nError generating aliases: {e}", err=True)

        if os.getenv("GITFLOW_DEBUG"):
            traceback.print_exc()
        sys.exit(1)
