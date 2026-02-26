"""Batch and traditional repository analysis modes for the analyze command."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click

from .core.repo_cloner import clone_repository

logger = logging.getLogger(__name__)


@dataclass
class AnalysisModeResult:
    """Result returned by both batch and traditional analysis modes."""

    all_commits: list[dict] = field(default_factory=list)
    all_prs: list = field(default_factory=list)
    all_enrichments: dict = field(default_factory=dict)
    branch_health_metrics: dict = field(default_factory=dict)
    developer_stats: list[dict] = field(default_factory=list)
    ticket_analysis: dict = field(default_factory=dict)
    developer_ticket_coverage: dict = field(default_factory=dict)


def run_batch_mode(
    *,
    cfg: Any,
    cache: Any,
    analyzer: Any,
    identity_resolver: Any,
    repositories_to_analyze: list,
    start_date: datetime,
    end_date: datetime,
    weeks: int,
    config_hash: str,
    force_fetch: bool,
    clear_cache: bool,
    display: Any,
) -> AnalysisModeResult:
    """Run Stage 10: two-step batch fetch + classify pipeline.

    Returns an AnalysisModeResult with all collected data.
    Raises SystemExit(0) implicitly via click.echo when no commits found
    (caller checks result and generates empty reports).
    """
    from .core.analyze_pipeline import (
        classify_commits_batch,
        fetch_repositories_batch,
        load_commits_from_db,
        resolve_developer_identities,
        validate_batch_state,
    )

    result = AnalysisModeResult()

    if display:
        display.add_progress_task(
            "repos",
            "Checking cache and preparing analysis",
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
        result.developer_ticket_coverage = {}
        result.developer_stats = identity_resolver.get_developer_stats(
            ticket_coverage=result.developer_ticket_coverage
        )
        result.ticket_analysis = analyzer.ticket_extractor.analyze_ticket_coverage(
            [], [], display
        )
        return result

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
            "repos", "Classifying batches", existing_batches or 1
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
        display.print_status("Loading classified commits from database...", "info")
    else:
        click.echo("📊 Loading classified commits from database...")

    commit_load = load_commits_from_db(
        cache=cache,
        repositories=repositories_to_analyze,
        start_date=start_date,
        end_date=end_date,
    )
    result.all_commits = commit_load.all_commits
    result.all_prs = commit_load.all_prs
    result.all_enrichments = commit_load.all_enrichments
    result.branch_health_metrics = commit_load.branch_health_metrics

    if display and display._live:
        display.update_progress_task(
            "main",
            description=f"Loaded {len(result.all_commits)} classified commits from database",
            completed=85,
        )
    else:
        click.echo(f"✅ Loaded {len(result.all_commits)} classified commits from database")

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
        all_commits=result.all_commits,
        ticket_extractor=analyzer.ticket_extractor,
    )
    result.developer_stats = identity_result.developer_stats
    result.developer_ticket_coverage = identity_result.developer_ticket_coverage

    # Ticket analysis
    if display and display._live:
        display.update_progress_task(
            "main",
            description="Analyzing ticket references...",
            completed=95,
        )
    else:
        click.echo("🎫 Analyzing ticket references...")

    result.ticket_analysis = analyzer.ticket_extractor.analyze_ticket_coverage(
        result.all_commits, result.all_prs, display
    )
    result.developer_stats = identity_resolver.get_developer_stats(
        ticket_coverage=analyzer.ticket_extractor.calculate_developer_ticket_coverage(
            result.all_commits
        )
    )

    if display and display._live:
        display.update_progress_task(
            "main",
            description=f"Identified {len(result.developer_stats)} unique developers",
            completed=98,
        )
    else:
        click.echo(f"   ✅ Identified {len(result.developer_stats)} unique developers")

    return result


def run_traditional_mode(
    *,
    cfg: Any,
    cache: Any,
    analyzer: Any,
    identity_resolver: Any,
    repositories_to_analyze: list,
    start_date: datetime,
    end_date: datetime,
    config: Path,
    cache_dir: Path,
    skip_identity_analysis: bool,
    display: Any,
) -> AnalysisModeResult:
    """Run Stage 10b: traditional per-repository analysis (non-batch).

    Returns an AnalysisModeResult with all collected data.
    """
    from .integrations.orchestrator import IntegrationOrchestrator

    result = AnalysisModeResult()

    if display and display._live:
        display.add_progress_task(
            "repos", "Processing repositories", len(repositories_to_analyze)
        )

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
                    click.echo(f"   ❌ Repository path not found: {repo_config.path}")
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
                commit["canonical_name"] = identity_resolver.get_canonical_name(canonical_id)
            result.all_commits.extend(commits)
            if display:
                display.print_status(f"Found {len(commits)} commits", "success")
            else:
                click.echo(f"   ✅ Found {len(commits)} commits")

            from .metrics.branch_health import BranchHealthAnalyzer

            branch_metrics = BranchHealthAnalyzer().analyze_repository_branches(
                str(repo_config.path)
            )
            result.branch_health_metrics[repo_config.name] = branch_metrics

            enrichment = orchestrator.enrich_repository_data(
                repo_config, commits, start_date
            )
            result.all_enrichments[repo_config.name] = enrichment
            if enrichment["prs"]:
                result.all_prs.extend(enrichment["prs"])
                if display:
                    display.print_status(
                        f"Found {len(enrichment['prs'])} pull requests", "success"
                    )
                else:
                    click.echo(f"   ✅ Found {len(enrichment['prs'])} pull requests")
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

    if not result.all_commits:
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
        result.developer_stats = identity_resolver.get_developer_stats(ticket_coverage={})
        result.ticket_analysis = analyzer.ticket_extractor.analyze_ticket_coverage(
            [], [], display
        )
        return result

    # Identity resolution
    if display:
        display.print_status("Resolving developer identities...", "info")
    else:
        click.echo("\n👥 Resolving developer identities...")
    identity_resolver.update_commit_stats(result.all_commits)
    result.developer_stats = identity_resolver.get_developer_stats()

    if display:
        display.print_status(
            f"Identified {len(result.developer_stats)} unique developers", "success"
        )
    else:
        click.echo(f"   ✅ Identified {len(result.developer_stats)} unique developers")

    # Auto identity analysis
    should_check_identities = (
        not skip_identity_analysis
        and cfg.analysis.auto_identity_analysis
        and not cfg.analysis.manual_identity_mappings
        and len(result.developer_stats) > 1
    )
    if should_check_identities:
        from .cli_analysis_helpers import run_identity_analysis
        run_identity_analysis(
            config=config,
            cfg=cfg,
            cache_dir=cache_dir,
            identity_resolver=identity_resolver,
            all_commits=result.all_commits,
            developer_stats=result.developer_stats,
            display=display,
            logger=logger,
        )

    # Ticket analysis
    if display:
        display.print_status("Analyzing ticket references...", "info")
    else:
        click.echo("\n🎫 Analyzing ticket references...")
    result.ticket_analysis = analyzer.ticket_extractor.analyze_ticket_coverage(
        result.all_commits, result.all_prs, display
    )
    result.developer_ticket_coverage = (
        analyzer.ticket_extractor.calculate_developer_ticket_coverage(result.all_commits)
    )
    result.developer_stats = identity_resolver.get_developer_stats(
        ticket_coverage=result.developer_ticket_coverage
    )
    for platform, count in result.ticket_analysis["ticket_summary"].items():
        if display:
            display.print_status(
                f"{platform.title()}: {count} unique tickets", "success"
            )
        else:
            click.echo(f"   - {platform.title()}: {count} unique tickets")

    return result
