"""Helper functions for the analyze command.

Contains:
- run_security_only_analysis  — security-only analysis path
- run_identity_analysis       — optional identity-cluster analysis
- show_final_summary          — stage 15 summary output
"""

import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import click
import yaml

logger = logging.getLogger(__name__)


def show_final_summary(
    *,
    all_commits: list,
    all_prs: list,
    developer_stats: list,
    ticket_analysis: dict,
    report_result: Any,
    qualitative_result: Any,
    cache: Any,
    output: Path,
    display: Any,
) -> None:
    """Render stage 15: final analysis summary to terminal."""
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
            click.echo(f"   - Lead time: {dora_metrics['lead_time_hours']:.1f} hours")
            click.echo(
                f"   - Change failure rate: "
                f"{dora_metrics['change_failure_rate']:.1f}%"
            )
            click.echo(f"   - MTTR: {dora_metrics['mttr_hours']:.1f} hours")
            click.echo(f"   - Performance level: {dora_metrics['performance_level']}")

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
            click.echo(f"   - Total requests: {cache_stats['total_requests']}")
            click.echo(
                f"   - Cache hits: {cache_stats['cache_hits']} "
                f"({cache_stats['hit_rate_percent']:.1f}%)"
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
            if cache_stats.get("stale_commits", 0) > 0:
                click.echo(f"   - Stale commits: {cache_stats['stale_commits']}")
            click.echo(
                f"   - Database size: {cache_stats['database_size_mb']:.1f} MB"
            )
        except Exception as e:
            click.echo(f"   Warning: Could not display cache statistics: {e}")

        click.echo(f"\n✅ Analysis complete! Reports saved to {output}")


def run_security_only_analysis(
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


def run_identity_analysis(
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
    from datetime import datetime as _dt

    from .identity_llm.analysis_pass import IdentityAnalysisPass

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
