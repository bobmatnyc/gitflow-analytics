"""Analyze command orchestration logic for GitFlow Analytics CLI.

Stages 1-9 and 11-15 live here. Stage 10 (batch and traditional modes) is
delegated to cli_analysis_modes. Security and identity helpers are in
cli_analysis_helpers.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import click

from ._version import __version__
from .cli_formatting import ImprovedErrorHandler
from .cli_utils import setup_logging
from .config.errors import ConfigurationError
from .ui.progress_display import create_progress_display

logger = logging.getLogger(__name__)


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
        discover_repositories,
        generate_all_reports,
        load_and_validate_config,
        run_qualitative_analysis,
        aggregate_pm_data,
    )
    from .core.analyze_pipeline_helpers import get_qualitative_config, is_qualitative_enabled

    try:
        from ._version import __version__
        version = __version__
    except ImportError:
        version = "1.3.11"

    # Initialize progress service with user's preference
    progress = get_progress_service(display_style=progress_style, version=version)

    # Create display — only if rich output is explicitly requested
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
            from .cli_analysis_helpers import run_security_only_analysis
            run_security_only_analysis(
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

        # ------------------------------------------------------------------
        # STAGE 10 – Fetch + classify (batch or traditional)
        # ------------------------------------------------------------------
        from .cli_analysis_modes import run_batch_mode, run_traditional_mode

        if use_batch_classification:
            mode_result = run_batch_mode(
                cfg=cfg,
                cache=cache,
                analyzer=analyzer,
                identity_resolver=identity_resolver,
                repositories_to_analyze=repositories_to_analyze,
                start_date=start_date,
                end_date=end_date,
                weeks=weeks,
                config_hash=config_hash,
                force_fetch=force_fetch,
                clear_cache=clear_cache,
                display=display,
            )
        else:
            mode_result = run_traditional_mode(
                cfg=cfg,
                cache=cache,
                analyzer=analyzer,
                identity_resolver=identity_resolver,
                repositories_to_analyze=repositories_to_analyze,
                start_date=start_date,
                end_date=end_date,
                config=config,
                cache_dir=cache_dir,
                skip_identity_analysis=skip_identity_analysis,
                display=display,
            )

        all_commits = mode_result.all_commits
        all_prs = mode_result.all_prs
        all_enrichments = mode_result.all_enrichments
        branch_health_metrics = mode_result.branch_health_metrics
        developer_stats = mode_result.developer_stats
        ticket_analysis = mode_result.ticket_analysis

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
                    click.echo(f"   ❌ Qualitative analysis dependencies missing: {e}")
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
            from .cli_analysis_helpers import show_final_summary
            show_final_summary(
                all_commits=all_commits,
                all_prs=all_prs,
                developer_stats=developer_stats,
                ticket_analysis=ticket_analysis,
                report_result=report_result,
                qualitative_result=qualitative_result,
                cache=cache,
                output=output,
                display=display,
            )
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
        raise
    except Exception as e:
        error_msg = str(e)

        if "❌ YAML configuration error" in error_msg or "❌ Configuration file" in error_msg:
            if display:
                display.show_error(error_msg, show_debug_hint=False)
            else:
                click.echo(f"\n{error_msg}", err=True)
        else:
            ImprovedErrorHandler.handle_command_error(click.get_current_context(), e)
            if display and "--debug" not in sys.argv:
                display.show_error(error_msg, show_debug_hint=True)

        if "--debug" in sys.argv:
            raise
        sys.exit(1)
