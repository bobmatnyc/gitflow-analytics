"""Pipeline stages for the analyze command.

Each function in this module represents one discrete stage of the analysis
pipeline.  The stages are called in sequence by the thin ``analyze()``
orchestrator in ``cli.py``.

Design rules:
- Every function takes explicit parameters (no closure variables).
- Every function returns a typed dataclass or plain value.
- Error handling uses exceptions; display/echo calls stay in cli.py.
- Heavy imports are lazy so CLI startup time is not affected.
"""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConfigResult:
    """Output of the config-loading stage."""

    cfg: Any  # gitflow_analytics.config.AnalysisConfig
    warnings: list[str]


@dataclass
class DateRangeResult:
    """Analysis date window."""

    start_date: datetime
    end_date: datetime


@dataclass
class FetchResult:
    """Output of the batch-fetch stage."""

    total_commits: int
    total_tickets: int
    total_developers: set[str]
    repos_fetched: int
    repos_cached: int


@dataclass
class ClassificationResult:
    """Output of the batch-classification stage."""

    processed_batches: int
    total_commits: int
    skipped: bool = False  # True when no commits in date range


@dataclass
class CommitLoadResult:
    """Commits loaded from DB after batch classification."""

    all_commits: list[dict[str, Any]]
    all_prs: list[Any]
    all_enrichments: dict[str, Any]
    branch_health_metrics: dict[str, Any]


@dataclass
class IdentityResult:
    """Output of the identity-resolution stage."""

    developer_stats: list[dict[str, Any]]
    developer_ticket_coverage: dict[str, Any]


@dataclass
class TicketResult:
    """Output of the ticket-analysis stage."""

    ticket_analysis: dict[str, Any]
    developer_stats: list[dict[str, Any]]
    developer_ticket_coverage: dict[str, Any]


@dataclass
class QualitativeResult:
    """Output of the qualitative-analysis stage."""

    results: list[Any]
    cost_stats: Optional[dict[str, Any]]
    commits_for_qual: list[dict[str, Any]]  # kept for JSON export


@dataclass
class ReportResult:
    """Output of the report-generation stage."""

    generated_reports: list[str]
    dora_metrics: Optional[dict[str, Any]]
    pr_metrics: dict[str, Any]


# ---------------------------------------------------------------------------
# Stage 1 – Config loading & validation
# ---------------------------------------------------------------------------


def load_and_validate_config(
    config: Path,
    enable_pm: bool,
    disable_pm: bool,
    pm_platform: tuple[str, ...],
    cicd_metrics: bool,
    cicd_platforms: tuple[str, ...],
) -> ConfigResult:
    """Load YAML config and apply CLI overrides.

    Returns the parsed config object together with any validation warnings.
    Raises ``ConfigurationError`` / ``FileNotFoundError`` on bad config.
    """
    from ..config import ConfigLoader
    from ..config.errors import ConfigurationError  # noqa: F401 (re-raised by caller)

    cfg = ConfigLoader.load(config)

    # Apply PM CLI overrides
    if disable_pm:
        if cfg.pm_integration:
            cfg.pm_integration.enabled = False
    elif enable_pm:
        if not cfg.pm_integration:
            from ..config import PMIntegrationConfig

            cfg.pm_integration = PMIntegrationConfig(enabled=True)
        else:
            cfg.pm_integration.enabled = True

    if pm_platform and cfg.pm_integration:
        requested_platforms = set(pm_platform)
        for platform_name in list(cfg.pm_integration.platforms.keys()):
            if platform_name not in requested_platforms:
                cfg.pm_integration.platforms[platform_name].enabled = False

    # Apply CI/CD CLI overrides
    if cicd_metrics:
        if not hasattr(cfg, "cicd"):

            class _CICDConfig:
                def __init__(self) -> None:
                    self.enabled = True
                    self.github_actions_enabled = "github-actions" in cicd_platforms

            cfg.cicd = _CICDConfig()
        else:
            cfg.cicd.enabled = True
            if "github-actions" in cicd_platforms:
                cfg.cicd.github_actions_enabled = True

    warnings = ConfigLoader.validate_config(cfg)
    return ConfigResult(cfg=cfg, warnings=warnings)


# ---------------------------------------------------------------------------
# Stage 2 – GitHub authentication pre-flight
# ---------------------------------------------------------------------------


def check_github_auth(cfg: Any) -> bool:
    """Return True when GitHub features are configured and auth succeeds.

    Returns False (not raises) when GitHub is not needed at all.
    Raises ``SystemExit`` (via ``click``) when auth is required but fails –
    the CLI layer catches this and handles display.

    Actually: returns a tuple (needed, ok) so the caller can decide output.
    """
    from ..core.git_auth import preflight_git_authentication

    needs_auth = bool(
        (cfg.repositories and any(getattr(r, "github_repo", None) for r in cfg.repositories))
        or (cfg.github and cfg.github.organization)
    )

    if not needs_auth:
        return False  # type: ignore[return-value]

    config_dict = {
        "github": {
            "token": cfg.github.token if cfg.github else None,
            "organization": cfg.github.organization if cfg.github else None,
        }
    }
    return preflight_git_authentication(config_dict)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Stage 3 – Date range calculation
# ---------------------------------------------------------------------------


def calculate_date_range(weeks: int) -> DateRangeResult:
    """Return Monday-aligned start/end datetimes for the last *weeks* complete weeks."""
    from ..utils.date_utils import get_week_end, get_week_start

    current_time = datetime.now(timezone.utc)
    current_week_start = get_week_start(current_time)
    last_complete_week_start = current_week_start - timedelta(weeks=1)
    start_date = last_complete_week_start - timedelta(weeks=weeks - 1)
    end_date = get_week_end(last_complete_week_start + timedelta(days=6))
    return DateRangeResult(start_date=start_date, end_date=end_date)


# ---------------------------------------------------------------------------
# Stage 4 – Organisation repository discovery
# ---------------------------------------------------------------------------


def discover_repositories(
    cfg: Any,
    config_path: Path,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> list[Any]:
    """Discover repositories from a GitHub organisation if configured.

    Returns the list of repository configs to analyse (may equal
    ``cfg.repositories`` when no org discovery is needed).
    """
    if not (cfg.github.organization and not cfg.repositories):
        return cfg.repositories

    config_dir = config_path.parent if config_path else Path.cwd()
    repos_dir = config_dir / "repos"

    discovered = cfg.discover_organization_repositories(
        clone_base_path=repos_dir,
        progress_callback=progress_callback,
    )
    return discovered


# ---------------------------------------------------------------------------
# Stage 5 – Batch data fetch (Step 1)
# ---------------------------------------------------------------------------


def fetch_repositories_batch(
    cfg: Any,
    cache: Any,
    repositories: list[Any],
    start_date: datetime,
    end_date: datetime,
    weeks: int,
    config_hash: str,
    force_fetch: bool,
    progress_callback: Optional[Callable[[str], None]] = None,
    repo_progress_callback: Optional[Callable[[str, str, dict[str, Any]], None]] = None,
) -> FetchResult:
    """Fetch raw git data for all repositories that need analysis.

    Skips repos whose cache is already fresh for the requested date range.
    Returns aggregate counters so the caller can display progress.
    """
    from ..core.data_fetcher import GitDataFetcher
    from ..integrations.orchestrator import IntegrationOrchestrator

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

    # Determine which repos need fetching
    repos_needing_analysis: list[Any] = []
    cached_repos: list[tuple[Any, Any]] = []

    if not force_fetch:
        for repo_config in repositories:
            repo_path = str(Path(repo_config.path))
            status = cache.get_repository_analysis_status(
                repo_path=repo_path,
                analysis_start=start_date,
                analysis_end=end_date,
                config_hash=config_hash,
            )
            if status:
                cached_repos.append((repo_config, status))
            else:
                repos_needing_analysis.append(repo_config)
    else:
        repos_needing_analysis = list(repositories)

    total_commits = 0
    total_tickets = 0
    total_developers: set[str] = set()

    for repo_config in repos_needing_analysis:
        try:
            repo_path = Path(repo_config.path)
            project_key = repo_config.project_key or repo_path.name

            # Clone if missing
            if not repo_path.exists():
                if repo_config.github_repo and cfg.github.organization:
                    from ..core.repo_cloner import clone_repository

                    clone_result = clone_repository(
                        repo_path=repo_path,
                        github_repo=repo_config.github_repo,
                        token=cfg.github.token if cfg.github else None,
                        branch=getattr(repo_config, "branch", None),
                        timeout_seconds=300,
                        max_retries=2,
                        progress_callback=progress_callback,
                    )
                    if not clone_result.success:
                        continue
                else:
                    logger.warning("Repository not found: %s", repo_path)
                    continue

            branch_patterns: Optional[list[str]] = None
            if hasattr(cfg.analysis, "branch_patterns"):
                branch_patterns = cfg.analysis.branch_patterns
            elif cfg.github.organization:
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
                force=force_fetch,
            )

            total_commits += result["stats"]["total_commits"]
            total_tickets += result["stats"]["unique_tickets"]

            # Enrich with GitHub PRs
            if repo_config.github_repo:
                try:
                    with cache.get_session() as session:
                        from gitflow_analytics.models.database import CachedCommit

                        cached_commits_db = (
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
                                "hash": cc.commit_hash,
                                "author_name": cc.author_name,
                                "author_email": cc.author_email,
                                "date": cc.timestamp,
                                "message": cc.message,
                            }
                            for cc in cached_commits_db
                        ]

                    enrichment = orchestrator.enrich_repository_data(
                        repo_config, commits_for_enrichment, start_date
                    )
                    if repo_progress_callback and enrichment.get("prs"):
                        repo_progress_callback(
                            project_key,
                            "pr_fetch",
                            {"pr_count": len(enrichment["prs"])},
                        )
                except Exception as e:
                    logger.warning("Failed to fetch PRs for %s: %s", repo_config.github_repo, e)

            if "developers" in result["stats"]:
                total_developers.update(result["stats"]["developers"])

            # Mark complete in cache
            cache.mark_repository_analysis_complete(
                repo_path=str(repo_path),
                repo_name=repo_config.name,
                project_key=project_key,
                analysis_start=start_date,
                analysis_end=end_date,
                weeks_analyzed=weeks,
                commit_count=result["stats"]["total_commits"],
                ticket_count=result["stats"]["unique_tickets"],
                config_hash=config_hash,
            )

            if repo_progress_callback:
                repo_progress_callback(
                    project_key,
                    "complete",
                    {
                        "commits": result["stats"]["total_commits"],
                        "tickets": result["stats"]["unique_tickets"],
                    },
                )

        except Exception as e:
            logger.error("Error fetching %s: %s", getattr(repo_config, "name", "?"), e)
            with contextlib.suppress(Exception):
                cache.mark_repository_analysis_failed(
                    repo_path=str(Path(repo_config.path)),
                    repo_name=repo_config.name,
                    analysis_start=start_date,
                    analysis_end=end_date,
                    error_message=str(e),
                    config_hash=config_hash,
                )

    return FetchResult(
        total_commits=total_commits,
        total_tickets=total_tickets,
        total_developers=total_developers,
        repos_fetched=len(repos_needing_analysis),
        repos_cached=len(cached_repos),
    )


# ---------------------------------------------------------------------------
# Stage 6 – Validate DB state before classification
# ---------------------------------------------------------------------------


def validate_batch_state(
    cache: Any,
    start_date: datetime,
    end_date: datetime,
    total_commits_fetched: int,
) -> tuple[bool, int, int]:
    """Verify that commits and daily batches exist in DB for the date range.

    Returns ``(validation_passed, stored_commits, existing_batches)``.
    Raises ``click.ClickException`` when Step-1 claimed success but DB is empty.
    """
    import click
    from sqlalchemy import and_

    from ..models.database import CachedCommit, DailyCommitBatch

    with cache.get_session() as session:
        stored_commits = (
            session.query(CachedCommit)
            .filter(
                and_(
                    CachedCommit.timestamp >= start_date,
                    CachedCommit.timestamp <= end_date,
                )
            )
            .count()
        )
        existing_batches = (
            session.query(DailyCommitBatch)
            .filter(
                and_(
                    DailyCommitBatch.date >= start_date.date(),
                    DailyCommitBatch.date <= end_date.date(),
                )
            )
            .count()
        )

    if stored_commits > 0 and existing_batches > 0:
        return True, stored_commits, existing_batches

    if stored_commits == 0 and total_commits_fetched > 0:
        raise click.ClickException(
            f"Data validation failed: Step 1 reported {total_commits_fetched} commits "
            f"but database contains 0 commits for date range "
            f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

    return False, stored_commits, existing_batches


# ---------------------------------------------------------------------------
# Stage 7 – Batch classification (Step 2)
# ---------------------------------------------------------------------------


def classify_commits_batch(
    cfg: Any,
    cache: Any,
    repositories: list[Any],
    start_date: datetime,
    end_date: datetime,
    force_reclassify: bool,
) -> ClassificationResult:
    """Run batch LLM classification over all daily-commit batches.

    Returns classification statistics.
    """
    from ..classification.batch_classifier import BatchCommitClassifier

    llm_cfg = cfg.analysis.llm_classification
    llm_config = {
        "enabled": llm_cfg.enabled,
        "api_key": llm_cfg.api_key,
        "model": llm_cfg.model,
        "confidence_threshold": llm_cfg.confidence_threshold,
        "max_tokens": llm_cfg.max_tokens,
        "temperature": llm_cfg.temperature,
        "timeout_seconds": llm_cfg.timeout_seconds,
        "cache_duration_days": llm_cfg.cache_duration_days,
        "enable_caching": llm_cfg.enable_caching,
        "max_daily_requests": llm_cfg.max_daily_requests,
    }

    batch_classifier = BatchCommitClassifier(
        cache_dir=cfg.cache.directory,
        llm_config=llm_config,
        batch_size=50,
        confidence_threshold=llm_cfg.confidence_threshold,
        fallback_enabled=True,
    )

    project_keys = [
        repo.project_key or repo.name for repo in repositories
    ]

    result = batch_classifier.classify_date_range(
        start_date=start_date,
        end_date=end_date,
        project_keys=project_keys,
        force_reclassify=force_reclassify,
    )

    return ClassificationResult(
        processed_batches=result["processed_batches"],
        total_commits=result["total_commits"],
    )


# ---------------------------------------------------------------------------
# Stage 8 – Load classified commits from DB
# ---------------------------------------------------------------------------


def load_commits_from_db(
    cache: Any,
    repositories: list[Any],
    start_date: datetime,
    end_date: datetime,
) -> CommitLoadResult:
    """Load classified commits from the database after batch classification."""
    from sqlalchemy import and_

    from ..models.database import CachedCommit

    with cache.get_session() as session:
        cached_commits = (
            session.query(CachedCommit)
            .filter(
                and_(
                    CachedCommit.timestamp >= start_date,
                    CachedCommit.timestamp <= end_date,
                    CachedCommit.repo_path.in_(
                        [str(Path(repo.path)) for repo in repositories]
                    ),
                )
            )
            .order_by(CachedCommit.timestamp.desc())
            .all()
        )

        all_commits: list[dict[str, Any]] = []
        for cc in cached_commits:
            commit_dict = cache._commit_to_dict(cc)
            if "project_key" not in commit_dict:
                repo_path = Path(cc.repo_path)
                for repo_config in repositories:
                    if repo_config.path == repo_path:
                        commit_dict["project_key"] = (
                            repo_config.project_key or repo_config.name
                        )
                        break
                else:
                    commit_dict["project_key"] = repo_path.name
            commit_dict["canonical_id"] = cc.author_email or "unknown"
            all_commits.append(commit_dict)

    return CommitLoadResult(
        all_commits=all_commits,
        all_prs=[],
        all_enrichments={},
        branch_health_metrics={},
    )


# ---------------------------------------------------------------------------
# Stage 9 – Identity resolution
# ---------------------------------------------------------------------------


def resolve_developer_identities(
    identity_resolver: Any,
    all_commits: list[dict[str, Any]],
    ticket_extractor: Any,
) -> IdentityResult:
    """Update identity resolver with commit stats and compute developer stats."""
    identity_resolver.update_commit_stats(all_commits)
    developer_ticket_coverage = ticket_extractor.calculate_developer_ticket_coverage(
        all_commits
    )
    developer_stats = identity_resolver.get_developer_stats(
        ticket_coverage=developer_ticket_coverage
    )
    return IdentityResult(
        developer_stats=developer_stats,
        developer_ticket_coverage=developer_ticket_coverage,
    )


# ---------------------------------------------------------------------------
# Stage 10 – Ticket analysis & daily metrics storage
# ---------------------------------------------------------------------------


def analyze_tickets_and_store_metrics(
    analyzer: Any,
    identity_resolver: Any,
    all_commits: list[dict[str, Any]],
    all_prs: list[Any],
    display: Any,
    cfg: Any,
    start_date: datetime,
    weeks: int,
) -> TicketResult:
    """Run ticket analysis and store daily metrics to DB."""
    ticket_analysis = analyzer.ticket_extractor.analyze_ticket_coverage(
        all_commits, all_prs, display
    )
    developer_ticket_coverage = analyzer.ticket_extractor.calculate_developer_ticket_coverage(
        all_commits
    )
    developer_stats = identity_resolver.get_developer_stats(
        ticket_coverage=developer_ticket_coverage
    )

    # Store daily metrics
    try:
        from ..core.metrics_storage import DailyMetricsStorage

        metrics_db_path = cfg.cache.directory / "daily_metrics.db"
        metrics_storage = DailyMetricsStorage(metrics_db_path)

        developer_identities: dict[str, Any] = {}
        for commit in all_commits:
            email = commit.get("author_email", "")
            developer_identities[email] = {
                "canonical_id": commit.get("canonical_id", email),
                "name": commit.get("author_name", "Unknown"),
                "email": email,
            }

        current_date = start_date.date() if hasattr(start_date, "date") else start_date
        end_date_obj = (
            (start_date + timedelta(weeks=weeks)).date()
            if hasattr(start_date, "date")
            else (start_date + timedelta(weeks=weeks))
        )

        daily_commits: dict[Any, list[dict[str, Any]]] = {}
        for commit in all_commits:
            commit_date = commit.get("timestamp")
            if commit_date:
                if hasattr(commit_date, "date"):
                    commit_date = commit_date.date()
                elif isinstance(commit_date, str):
                    from datetime import datetime as dt

                    commit_date = dt.fromisoformat(
                        commit_date.replace("Z", "+00:00")
                    ).date()
                daily_commits.setdefault(commit_date, []).append(commit)

        total_records = 0
        for analysis_date, day_commits in daily_commits.items():
            if current_date <= analysis_date <= end_date_obj:
                total_records += metrics_storage.store_daily_metrics(
                    analysis_date, day_commits, developer_identities
                )

        logger.debug("Stored %d daily metric records", total_records)
    except Exception as e:
        logger.error("Failed to store daily metrics: %s", e)

    return TicketResult(
        ticket_analysis=ticket_analysis,
        developer_stats=developer_stats,
        developer_ticket_coverage=developer_ticket_coverage,
    )


# ---------------------------------------------------------------------------
# Stage 11 – Qualitative analysis
# ---------------------------------------------------------------------------


def run_qualitative_analysis(
    cfg: Any,
    all_commits: list[dict[str, Any]],
    enable_qualitative: bool,
    qualitative_only: bool,
    display: Any,
) -> QualitativeResult:
    """Run qualitative NLP/LLM analysis if enabled.

    Returns empty result when qualitative analysis is not configured or
    disabled.  Raises ImportError / Exception when qualitative_only=True
    and analysis cannot proceed.
    """
    from ..core.analyze_pipeline_helpers import (
        get_qualitative_config,
        is_qualitative_enabled,
    )

    qual_config = get_qualitative_config(cfg)
    if not (enable_qualitative or qualitative_only or is_qualitative_enabled(cfg)):
        return QualitativeResult(results=[], cost_stats=None, commits_for_qual=[])

    if not qual_config:
        return QualitativeResult(results=[], cost_stats=None, commits_for_qual=[])

    from ..models.database import Database
    from ..qualitative import QualitativeProcessor

    qual_db = Database(cfg.cache.directory / "qualitative.db")
    qual_processor = QualitativeProcessor(qual_config, qual_db)

    # Normalise commit dicts
    commits_for_qual: list[dict[str, Any]] = []
    for commit in all_commits:
        if isinstance(commit, dict):
            commits_for_qual.append(
                {
                    "hash": commit.get("hash") or commit.get("commit_hash"),
                    "message": commit.get("message"),
                    "author_name": commit.get("author_name"),
                    "author_email": commit.get("author_email"),
                    "timestamp": commit.get("timestamp"),
                    "files_changed": commit.get("files_changed") or [],
                    "insertions": commit.get(
                        "filtered_insertions", commit.get("insertions", 0)
                    ),
                    "deletions": commit.get(
                        "filtered_deletions", commit.get("deletions", 0)
                    ),
                    "branch": commit.get("branch", "main"),
                }
            )
        else:
            commits_for_qual.append(
                {
                    "hash": commit.hash,
                    "message": commit.message,
                    "author_name": commit.author_name,
                    "author_email": commit.author_email,
                    "timestamp": commit.timestamp,
                    "files_changed": commit.files_changed or [],
                    "insertions": getattr(commit, "filtered_insertions", commit.insertions),
                    "deletions": getattr(commit, "filtered_deletions", commit.deletions),
                    "branch": getattr(commit, "branch", "main"),
                }
            )

    qualitative_results = qual_processor.process_commits(
        commits_for_qual, show_progress=True
    )

    qual_stats = qual_processor.get_processing_statistics()
    cost_stats = None
    if qual_stats and "llm_statistics" in qual_stats:
        llm_stats = qual_stats["llm_statistics"]
        if llm_stats.get("model_usage") == "available":
            cost_stats = llm_stats.get("cost_tracking", {})

    return QualitativeResult(
        results=qualitative_results,
        cost_stats=cost_stats,
        commits_for_qual=commits_for_qual,
    )


# ---------------------------------------------------------------------------
# Stage 12 – PM data aggregation
# ---------------------------------------------------------------------------


def aggregate_pm_data(
    cfg: Any,
    all_enrichments: dict[str, Any],
    disable_pm: bool,
) -> Optional[dict[str, Any]]:
    """Aggregate PM platform data from enrichments.

    Returns ``None`` when PM is disabled or no data is available.
    """
    if disable_pm or not cfg.pm_integration or not cfg.pm_integration.enabled:
        return None

    try:
        aggregated: dict[str, Any] = {"issues": {}, "correlations": [], "metrics": {}}
        for enrichment in all_enrichments.values():
            pm_data = enrichment.get("pm_data", {})
            if pm_data:
                for platform, issues in pm_data.get("issues", {}).items():
                    aggregated["issues"].setdefault(platform, []).extend(issues)
                aggregated["correlations"].extend(pm_data.get("correlations", []))
                if pm_data.get("metrics"):
                    aggregated["metrics"] = pm_data["metrics"]

        if not aggregated["correlations"] and not aggregated["issues"]:
            return None
        return aggregated
    except Exception as e:
        logger.error("PM data aggregation failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Stage 13 – Report generation
# ---------------------------------------------------------------------------


def generate_all_reports(
    cfg: Any,
    output: Path,
    all_commits: list[dict[str, Any]],
    all_prs: list[Any],
    all_enrichments: dict[str, Any],
    developer_stats: list[dict[str, Any]],
    ticket_analysis: dict[str, Any],
    branch_health_metrics: dict[str, Any],
    start_date: datetime,
    end_date: datetime,
    weeks: int,
    anonymize: bool,
    generate_csv: bool,
    aggregated_pm_data: Optional[dict[str, Any]],
    qualitative_result: QualitativeResult,
    analyzer: Any,
    identity_resolver: Any,
) -> ReportResult:
    """Generate all output reports.

    This is the largest stage – it produces every CSV, markdown, and JSON
    output file.  Errors in individual reports are logged but do not abort
    the whole stage unless the report is critical.
    """
    import contextlib as _ctx

    from ..metrics.dora import DORAMetricsCalculator
    from ..reports.analytics_writer import AnalyticsReportGenerator
    from ..reports.csv_writer import CSVReportGenerator
    from ..reports.json_exporter import ComprehensiveJSONExporter
    from ..reports.narrative_writer import NarrativeReportGenerator
    from ..reports.weekly_trends_writer import WeeklyTrendsWriter

    date_suffix = datetime.now().strftime("%Y%m%d")
    date_range = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    generated_reports: list[str] = []

    report_gen = CSVReportGenerator(
        anonymize=anonymize or cfg.output.anonymize_enabled,
        exclude_authors=cfg.analysis.exclude_authors,
        identity_resolver=identity_resolver,
    )
    analytics_gen = AnalyticsReportGenerator(
        anonymize=anonymize or cfg.output.anonymize_enabled,
        exclude_authors=cfg.analysis.exclude_authors,
        identity_resolver=identity_resolver,
    )

    # ---- CSV reports (optional) ----
    if generate_csv:
        _gen_csv_report(
            "weekly_metrics",
            lambda p: report_gen.generate_weekly_report(
                all_commits, developer_stats, p, weeks
            ),
            output / f"weekly_metrics_{date_suffix}.csv",
            generated_reports,
        )
        _gen_csv_report(
            "developer_activity_summary",
            lambda p: report_gen.generate_developer_activity_summary(
                all_commits, developer_stats, all_prs, p, weeks
            ),
            output / f"developer_activity_summary_{date_suffix}.csv",
            generated_reports,
            reraise=True,
        )
        _gen_csv_report(
            "summary",
            lambda p: report_gen.generate_summary_report(
                all_commits,
                all_prs,
                developer_stats,
                ticket_analysis,
                p,
                aggregated_pm_data,
                pr_metrics=None,
            ),
            output / f"summary_{date_suffix}.csv",
            generated_reports,
            reraise=True,
        )
        _gen_csv_report(
            "developers",
            lambda p: report_gen.generate_developer_report(developer_stats, p),
            output / f"developers_{date_suffix}.csv",
            generated_reports,
            reraise=True,
        )
        if all_prs:
            _gen_csv_report(
                "pr_metrics",
                lambda p: report_gen.generate_pr_metrics_report(all_prs, p),
                output / f"pr_metrics_{date_suffix}.csv",
                generated_reports,
            )
        _gen_csv_report(
            "untracked_commits",
            lambda p: report_gen.generate_untracked_commits_report(ticket_analysis, p),
            output / f"untracked_commits_{date_suffix}.csv",
            generated_reports,
            reraise=True,
        )
        _gen_csv_report(
            "weekly_categorization",
            lambda p: report_gen.generate_weekly_categorization_report(
                all_commits, analyzer.ticket_extractor, p, weeks
            ),
            output / f"weekly_categorization_{date_suffix}.csv",
            generated_reports,
        )
        if aggregated_pm_data:
            _gen_csv_report(
                "pm_correlations",
                lambda p: report_gen.generate_pm_correlations_report(
                    aggregated_pm_data, p
                ),
                output / f"pm_correlations_{date_suffix}.csv",
                generated_reports,
            )
        _gen_csv_report(
            "story_point_correlation",
            lambda p: report_gen.generate_story_point_correlation_report(
                all_commits, all_prs, aggregated_pm_data, p, weeks
            ),
            output / f"story_point_correlation_{date_suffix}.csv",
            generated_reports,
        )

    # ---- Analytics reports (always generated for narrative; CSV optional) ----
    activity_report = output / f"activity_distribution_{date_suffix}.csv"
    _gen_csv_report(
        "activity_distribution",
        lambda p: analytics_gen.generate_activity_distribution_report(
            all_commits, developer_stats, p
        ),
        activity_report,
        generated_reports if generate_csv else [],
        reraise=True,
    )

    focus_report = output / f"developer_focus_{date_suffix}.csv"
    _gen_csv_report(
        "developer_focus",
        lambda p: analytics_gen.generate_developer_focus_report(
            all_commits, developer_stats, p, weeks
        ),
        focus_report,
        generated_reports if generate_csv else [],
        reraise=True,
    )

    insights_report = output / f"qualitative_insights_{date_suffix}.csv"
    _gen_csv_report(
        "qualitative_insights",
        lambda p: analytics_gen.generate_qualitative_insights_report(
            all_commits, developer_stats, ticket_analysis, p
        ),
        insights_report,
        generated_reports if generate_csv else [],
    )

    # ---- Branch health ----
    if branch_health_metrics and generate_csv:
        from ..reports.branch_health_writer import BranchHealthReportGenerator

        bh_gen = BranchHealthReportGenerator()
        _gen_csv_report(
            "branch_health",
            lambda p: bh_gen.generate_csv_report(branch_health_metrics, p),
            output / f"branch_health_{date_suffix}.csv",
            generated_reports,
        )
        _gen_csv_report(
            "branch_details",
            lambda p: bh_gen.generate_detailed_branch_report(branch_health_metrics, p),
            output / f"branch_details_{date_suffix}.csv",
            generated_reports,
        )

    # ---- Weekly classification trends ----
    if generate_csv:
        try:
            trends_paths = WeeklyTrendsWriter().generate_weekly_trends_reports(
                all_commits,
                output,
                weeks,
                f"_{date_suffix}",
                categorize_fn=analyzer.ticket_extractor.categorize_commit,
            )
            for _, rp in trends_paths.items():
                generated_reports.append(rp.name)
        except Exception as e:
            logger.error("Weekly classification trends report failed: %s", e)

    # ---- Weekly trends (analytics) ----
    if generate_csv:
        trends_report = output / f"weekly_trends_{date_suffix}.csv"
        _gen_csv_report(
            "weekly_trends",
            lambda p: analytics_gen.generate_weekly_trends_report(
                all_commits, developer_stats, p, weeks
            ),
            trends_report,
            generated_reports,
            reraise=True,
        )
        # Check for side-car trend files
        ts = trends_report.stem.split("_")[-1]
        for extra in [
            output / f"developer_trends_{ts}.csv",
            output / f"project_trends_{ts}.csv",
        ]:
            if extra.exists():
                generated_reports.append(extra.name)

    # ---- DORA metrics ----
    dora_metrics: Optional[dict[str, Any]] = None
    try:
        dora_calculator = DORAMetricsCalculator()
        dora_metrics = dora_calculator.calculate_dora_metrics(
            all_commits, all_prs, start_date, end_date
        )
    except Exception as e:
        logger.error("DORA metrics calculation failed: %s", e)
        raise

    # ---- PR metrics aggregation ----
    pr_metrics: dict[str, Any] = {}
    for enrichment in all_enrichments.values():
        if enrichment.get("pr_metrics"):
            pr_metrics = enrichment["pr_metrics"]
            break

    # ---- Weekly velocity / DORA CSV ----
    if generate_csv:
        _gen_csv_report(
            "weekly_velocity",
            lambda p: report_gen.generate_weekly_velocity_report(
                all_commits, all_prs, p, weeks
            ),
            output / f"weekly_velocity_{date_suffix}.csv",
            generated_reports,
            reraise=True,
        )
        _gen_csv_report(
            "weekly_dora_metrics",
            lambda p: report_gen.generate_weekly_dora_report(
                all_commits, all_prs, p, weeks
            ),
            output / f"weekly_dora_metrics_{date_suffix}.csv",
            generated_reports,
            reraise=True,
        )

    # ---- Narrative markdown ----
    if "markdown" in cfg.output.formats and generate_csv:
        try:
            import pandas as pd

            activity_df = pd.read_csv(activity_report)
            focus_df = pd.read_csv(focus_report)
            insights_df = pd.read_csv(insights_report)

            narrative_report = output / f"narrative_report_{date_range}.md"
            chatgpt_summary = _try_chatgpt_summary(
                all_commits, developer_stats, ticket_analysis, weeks
            )

            NarrativeReportGenerator().generate_narrative_report(
                all_commits,
                all_prs,
                developer_stats,
                list(activity_df.to_dict("records")),
                list(focus_df.to_dict("records")),
                list(insights_df.to_dict("records")),
                ticket_analysis,
                pr_metrics,
                narrative_report,
                weeks,
                aggregated_pm_data,
                chatgpt_summary,
                branch_health_metrics,
                cfg.analysis.exclude_authors,
                analysis_start_date=start_date,
                analysis_end_date=end_date,
            )
            generated_reports.append(narrative_report.name)
        except Exception as e:
            logger.error("Narrative report generation failed: %s", e)
            raise

    # ---- Database-backed qualitative markdown ----
    if "markdown" in cfg.output.formats:
        try:
            from ..core.metrics_storage import DailyMetricsStorage
            from ..reports.database_report_generator import DatabaseReportGenerator

            metrics_db_path = cfg.cache.directory / "daily_metrics.db"
            db_report_gen = DatabaseReportGenerator(DailyMetricsStorage(metrics_db_path))
            db_qualitative_report = output / f"database_qualitative_report_{date_range}.md"
            analysis_start = start_date.date() if hasattr(start_date, "date") else start_date
            analysis_end = (
                (start_date + timedelta(weeks=weeks)).date()
                if hasattr(start_date, "date")
                else (start_date + timedelta(weeks=weeks))
            )
            db_report_gen.generate_qualitative_report(
                analysis_start, analysis_end, db_qualitative_report
            )
            generated_reports.append(db_qualitative_report.name)
        except Exception as e:
            logger.error("Database qualitative report failed: %s", e)

    # ---- Comprehensive JSON export ----
    if "json" in cfg.output.formats:
        try:
            json_report = output / f"comprehensive_export_{date_suffix}.json"
            enhanced_analysis = _try_enhanced_analysis(
                qualitative_result, developer_stats, ticket_analysis, pr_metrics,
                all_enrichments, aggregated_pm_data, weeks
            )
            ComprehensiveJSONExporter(anonymize=anonymize).export_comprehensive_data(
                commits=all_commits,
                prs=all_prs,
                developer_stats=developer_stats,
                project_metrics={
                    "ticket_analysis": ticket_analysis,
                    "pr_metrics": pr_metrics,
                    "enrichments": all_enrichments,
                },
                dora_metrics=dora_metrics,
                output_path=json_report,
                weeks=weeks,
                pm_data=aggregated_pm_data or None,
                qualitative_data=qualitative_result.results or None,
                enhanced_qualitative_analysis=enhanced_analysis,
            )
            generated_reports.append(json_report.name)
        except Exception as e:
            logger.error("Comprehensive JSON export failed: %s", e)
            raise

    return ReportResult(
        generated_reports=generated_reports,
        dora_metrics=dora_metrics,
        pr_metrics=pr_metrics,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gen_csv_report(
    label: str,
    fn: Callable[[Path], None],
    path: Path,
    report_list: list[str],
    reraise: bool = False,
) -> None:
    """Call *fn(path)* and append *path.name* to *report_list* on success."""
    try:
        fn(path)
        report_list.append(path.name)
    except Exception as e:
        logger.error("Error generating %s report: %s", label, e)
        if reraise:
            raise


def _try_chatgpt_summary(
    all_commits: list[dict[str, Any]],
    developer_stats: list[dict[str, Any]],
    ticket_analysis: dict[str, Any],
    weeks: int,
) -> Optional[Any]:
    """Attempt to generate a ChatGPT executive summary; returns None on failure."""
    openai_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return None
    try:
        from ..qualitative.chatgpt_analyzer import ChatGPTQualitativeAnalyzer

        comprehensive_data: dict[str, Any] = {
            "metadata": {
                "analysis_weeks": weeks,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "executive_summary": {
                "key_metrics": {
                    "commits": {"total": len(all_commits)},
                    "developers": {"total": len(developer_stats)},
                    "lines_changed": {
                        "total": sum(
                            c.get("filtered_insertions", c.get("insertions", 0))
                            + c.get("filtered_deletions", c.get("deletions", 0))
                            for c in all_commits
                        )
                    },
                    "story_points": {
                        "total": sum(c.get("story_points", 0) or 0 for c in all_commits)
                    },
                    "ticket_coverage": {
                        "percentage": ticket_analysis.get("commit_coverage_pct", 0)
                    },
                },
                "health_score": {"overall": 75, "rating": "good"},
                "trends": {"velocity": {"direction": "stable"}},
                "wins": [],
                "concerns": [],
            },
            "developers": {
                dev.get("canonical_id", dev.get("primary_email", "unknown")): {
                    "identity": {"name": dev.get("primary_name", "Unknown")},
                    "summary": {
                        "total_commits": dev.get("total_commits", 0),
                        "total_story_points": dev.get("total_story_points", 0),
                    },
                    "projects": {},
                }
                for dev in developer_stats
            },
            "projects": {},
        }
        return ChatGPTQualitativeAnalyzer(openai_key).generate_executive_summary(
            comprehensive_data
        )
    except Exception as e:
        logger.warning("ChatGPT summary generation failed: %s", e)
        return None


def _try_enhanced_analysis(
    qualitative_result: QualitativeResult,
    developer_stats: list[dict[str, Any]],
    ticket_analysis: dict[str, Any],
    pr_metrics: dict[str, Any],
    all_enrichments: dict[str, Any],
    aggregated_pm_data: Optional[dict[str, Any]],
    weeks: int,
) -> Optional[Any]:
    """Attempt enhanced qualitative analysis; returns None on failure."""
    if not qualitative_result.results:
        return None
    try:
        from ..qualitative.enhanced_analyzer import EnhancedQualitativeAnalyzer

        return EnhancedQualitativeAnalyzer().analyze_comprehensive(
            commits=qualitative_result.commits_for_qual,
            qualitative_data=qualitative_result.results,
            developer_stats=developer_stats,
            project_metrics={
                "ticket_analysis": ticket_analysis,
                "pr_metrics": pr_metrics,
                "enrichments": all_enrichments,
            },
            pm_data=aggregated_pm_data,
            weeks_analyzed=weeks,
        )
    except Exception as e:
        logger.warning("Enhanced qualitative analysis failed: %s", e)
        return None
