"""Stage 1 — collect: fetch raw commits from git repositories into the cache."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .pipeline_types import CollectResult

logger = logging.getLogger(__name__)


def run_collect(
    cfg: Any,
    weeks: int,
    force: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> CollectResult:
    """Fetch raw commit data from repositories into the weekly cache.

    This is Stage 1 of the pipeline.  It mirrors the data-fetching block
    inside ``analyze()`` but is self-contained so it can also be called by the
    standalone ``gfa collect`` command.

    Args:
        cfg: Loaded configuration object (result of ``ConfigLoader.load()``).
        weeks: Number of complete weeks to collect.
        force: When True, bypass the per-week cache and always re-fetch.
        progress_callback: Optional function called with human-readable status
            messages as work progresses.

    Returns:
        A :class:`CollectResult` with summary statistics.
    """
    from .core.cache import GitAnalysisCache
    from .core.data_fetcher import GitDataFetcher
    from .integrations.orchestrator import IntegrationOrchestrator
    from .utils.date_utils import get_week_end, get_week_start

    def _emit(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    result = CollectResult()

    current_time = datetime.now(timezone.utc)
    current_week_start = get_week_start(current_time)
    last_complete_week_start = current_week_start - timedelta(weeks=1)
    start_date = last_complete_week_start - timedelta(weeks=weeks - 1)
    end_date = get_week_end(last_complete_week_start + timedelta(days=6))

    result.start_date = start_date
    result.end_date = end_date

    _emit(f"Collect period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    cache = GitAnalysisCache(cfg.cache.directory)
    data_fetcher = GitDataFetcher(
        cache=cache,
        branch_mapping_rules=getattr(cfg.analysis, "branch_mapping_rules", {}),
        allowed_ticket_platforms=getattr(
            cfg.analysis, "ticket_platforms", ["jira", "github", "clickup", "linear"]
        ),
        exclude_paths=getattr(cfg.analysis, "exclude_paths", None),
        exclude_merge_commits=cfg.analysis.exclude_merge_commits,
    )
    orchestrator = IntegrationOrchestrator(cfg, cache)
    jira_integration = orchestrator.integrations.get("jira")

    config_hash = cache.generate_config_hash(
        branch_mapping_rules=getattr(cfg.analysis, "branch_mapping_rules", {}),
        ticket_platforms=getattr(
            cfg.analysis, "ticket_platforms", ["jira", "github", "clickup", "linear"]
        ),
        exclude_paths=getattr(cfg.analysis, "exclude_paths", None),
        ml_categorization_enabled=False,
        additional_config={"weeks": weeks},
    )

    repositories = cfg.repositories
    repos_needing_fetch = []
    repos_already_cached = []

    if not force:
        for repo_config in repositories:
            repo_path_str = str(Path(repo_config.path))
            status = cache.get_repository_analysis_status(
                repo_path=repo_path_str,
                analysis_start=start_date,
                analysis_end=end_date,
                config_hash=config_hash,
            )
            if status:
                repos_already_cached.append(repo_config)
            else:
                repos_needing_fetch.append(repo_config)
    else:
        repos_needing_fetch = list(repositories)

    result.repos_cached = len(repos_already_cached)
    if repos_already_cached:
        cached_msg = (
            f"Skipping {len(repos_already_cached)} already-cached "
            f"{'repository' if len(repos_already_cached) == 1 else 'repositories'}"
        )
        _emit(cached_msg)

    total_developers: set[str] = set()

    for idx, repo_config in enumerate(repos_needing_fetch, 1):
        repo_path = Path(repo_config.path)
        project_key = repo_config.project_key or repo_path.name

        if not repo_path.exists():
            msg = f"Repository not found, skipping: {repo_path}"
            _emit(msg)
            result.repos_failed += 1
            result.errors.append(msg)
            continue

        _emit(f"Fetching {project_key} ({idx}/{len(repos_needing_fetch)})...")

        try:
            branch_patterns: list[str] | None = None
            if hasattr(cfg.analysis, "branch_patterns"):
                branch_patterns = cfg.analysis.branch_patterns
            elif cfg.github.organization:
                branch_patterns = ["*"]

            def _progress_cb(message: str) -> None:
                _emit(f"  {message}")

            fetch_result = data_fetcher.fetch_repository_data(
                repo_path=repo_path,
                project_key=project_key,
                weeks_back=weeks,
                branch_patterns=branch_patterns,
                jira_integration=jira_integration,
                progress_callback=_progress_cb,
                start_date=start_date,
                end_date=end_date,
                force=force,
            )

            commits_count = fetch_result["stats"]["total_commits"]
            tickets_count = fetch_result["stats"]["unique_tickets"]
            result.total_commits += commits_count
            result.total_tickets += tickets_count
            result.repos_fetched += 1

            if "developers" in fetch_result["stats"]:
                total_developers.update(fetch_result["stats"]["developers"])

            if repo_config.github_repo:
                try:
                    with cache.get_session() as session:
                        from gitflow_analytics.models.database import CachedCommit

                        cached_commits_rows = (
                            session.query(CachedCommit)
                            .filter(
                                CachedCommit.repo_path == str(repo_path),
                                CachedCommit.timestamp >= start_date,
                                CachedCommit.timestamp <= end_date,
                            )
                            .all()
                        )
                        commits_for_enrichment = [
                            {
                                "hash": c.commit_hash,
                                "author_name": c.author_name,
                                "author_email": c.author_email,
                                "date": c.timestamp,
                                "message": c.message,
                            }
                            for c in cached_commits_rows
                        ]

                    enrichment = orchestrator.enrich_repository_data(
                        repo_config, commits_for_enrichment, start_date
                    )
                    if enrichment["prs"]:
                        _emit(f"  Found {len(enrichment['prs'])} pull requests")
                except Exception as pr_err:
                    logger.warning(f"PR enrichment failed for {repo_config.github_repo}: {pr_err}")
                    _emit(f"  PR fetch skipped: {pr_err}")

            cache.mark_repository_analysis_complete(
                repo_path=str(repo_path),
                repo_name=repo_config.name,
                project_key=project_key,
                analysis_start=start_date,
                analysis_end=end_date,
                weeks_analyzed=weeks,
                commit_count=commits_count,
                ticket_count=tickets_count,
                config_hash=config_hash,
            )

            _emit(f"  {project_key}: {commits_count} commits, {tickets_count} tickets")

        except Exception as exc:
            msg = f"Error fetching {project_key}: {exc}"
            logger.error(msg, exc_info=True)
            _emit(f"  Error: {exc}")
            result.repos_failed += 1
            result.errors.append(msg)
            with contextlib.suppress(Exception):
                cache.mark_repository_analysis_failed(
                    repo_path=str(repo_path),
                    repo_name=repo_config.name,
                    analysis_start=start_date,
                    analysis_end=end_date,
                    error_message=str(exc),
                    config_hash=config_hash,
                )

    # --- PR enrichment for already-cached repos that have github_repo set ---
    cached_with_github = [r for r in repos_already_cached if getattr(r, "github_repo", None)]
    if cached_with_github and orchestrator:
        _emit(f"Enriching {len(cached_with_github)} cached repos with PR data...")
        for idx, repo_config in enumerate(cached_with_github, 1):
            repo_path = Path(repo_config.path)
            project_key = repo_config.project_key or repo_path.name
            try:
                with cache.get_session() as session:
                    from gitflow_analytics.models.database import CachedCommit

                    cached_commits_rows = (
                        session.query(CachedCommit)
                        .filter(
                            CachedCommit.repo_path == str(repo_path),
                            CachedCommit.timestamp >= start_date,
                            CachedCommit.timestamp <= end_date,
                        )
                        .all()
                    )
                    commits_for_enrichment = [
                        {
                            "hash": c.commit_hash,
                            "author_name": c.author_name,
                            "author_email": c.author_email,
                            "date": c.timestamp,
                            "message": c.message,
                        }
                        for c in cached_commits_rows
                    ]

                enrichment = orchestrator.enrich_repository_data(
                    repo_config, commits_for_enrichment, start_date
                )
                if enrichment["prs"]:
                    _emit(
                        f"  {project_key}: {len(enrichment['prs'])} PRs ({idx}/{len(cached_with_github)})"
                    )
            except Exception as pr_err:
                logger.warning(f"PR enrichment failed for {repo_config.github_repo}: {pr_err}")
                _emit(f"  {project_key}: PR fetch skipped ({idx}/{len(cached_with_github)})")

    result.total_developers = len(total_developers)
    return result
