"""Stage 3 — report: generate reports from classified commit data in the cache."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .pipeline_types import ReportResult

logger = logging.getLogger(__name__)


def run_report(
    cfg: Any,
    weeks: int,
    output_dir: Path,
    generate_csv: bool = True,
    anonymize: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> ReportResult:
    """Generate reports from classified commit data in the cache.

    This is Stage 3 of the pipeline.  It reads commits from the database,
    resolves developer identities, and writes report files.  No git
    operations are performed.

    Args:
        cfg: Loaded configuration object.
        weeks: Number of weeks covered by the analysis.
        output_dir: Directory where report files will be written.
        generate_csv: When True, write CSV reports in addition to the narrative
            markdown report.
        anonymize: When True, anonymize developer names/emails in reports.
        progress_callback: Optional function called with status messages.

    Returns:
        A :class:`ReportResult` with the list of generated file names.
    """
    from .core.cache import GitAnalysisCache
    from .core.identity import DeveloperIdentityResolver
    from .metrics.dora import DORAMetricsCalculator
    from .reports.analytics_writer import AnalyticsReportGenerator
    from .reports.csv_writer import CSVReportGenerator
    from .reports.json_exporter import ComprehensiveJSONExporter
    from .reports.narrative_writer import NarrativeReportGenerator
    from .reports.weekly_trends_writer import WeeklyTrendsWriter
    from .utils.date_utils import get_week_end, get_week_start

    def _emit(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    result = ReportResult(output_dir=output_dir)

    current_time = datetime.now(timezone.utc)
    current_week_start = get_week_start(current_time)
    last_complete_week_start = current_week_start - timedelta(weeks=1)
    start_date = last_complete_week_start - timedelta(weeks=weeks - 1)
    end_date = get_week_end(last_complete_week_start + timedelta(days=6))

    _emit(f"Report period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    cache = GitAnalysisCache(cfg.cache.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    from sqlalchemy import and_  # type: ignore[import-not-found]

    from .models.database import CachedCommit
    from .models.database_metrics_models import QualitativeCommitData

    _emit("Loading classified commits from database...")

    with cache.get_session() as session:
        repo_paths = [str(Path(r.path)) for r in cfg.repositories]
        cached_commits_rows = (
            session.query(CachedCommit)
            .filter(
                and_(
                    CachedCommit.timestamp >= start_date,
                    CachedCommit.timestamp <= end_date,
                    CachedCommit.repo_path.in_(repo_paths),
                )
            )
            .order_by(CachedCommit.timestamp.desc())
            .all()
        )

        # Build commit dicts and track the DB row id so we can join qualitative data.
        all_commits: list[dict[str, Any]] = []
        # Map from cached_commit.id → index in all_commits for the merge step below.
        _id_to_index: dict[int, int] = {}

        for cached_commit in cached_commits_rows:
            commit_dict = cache._commit_to_dict(cached_commit)
            if "project_key" not in commit_dict:
                repo_path_obj = Path(cached_commit.repo_path)
                for repo_config in cfg.repositories:
                    if Path(repo_config.path) == repo_path_obj:
                        commit_dict["project_key"] = repo_config.project_key or repo_config.name
                        break
                else:
                    commit_dict["project_key"] = repo_path_obj.name
            commit_dict["canonical_id"] = cached_commit.author_email or "unknown"
            _id_to_index[cached_commit.id] = len(all_commits)
            all_commits.append(commit_dict)

        # Merge qualitative classifications (change_type, complexity, etc.) stored in
        # the qualitative_commits table.  Not all commits have qualitative data; the
        # join is left-optional so missing rows are silently skipped.
        if _id_to_index:
            qual_rows = (
                session.query(QualitativeCommitData)
                .filter(QualitativeCommitData.commit_id.in_(list(_id_to_index.keys())))
                .all()
            )
            for qual in qual_rows:
                idx = _id_to_index.get(qual.commit_id)
                if idx is None:
                    continue
                all_commits[idx]["change_type"] = qual.change_type
                all_commits[idx]["change_type_confidence"] = qual.change_type_confidence
                all_commits[idx]["complexity"] = qual.complexity
                all_commits[idx]["processing_method"] = qual.processing_method
            logger.debug(
                "Merged qualitative data: %d/%d commits have classifications",
                len(qual_rows),
                len(all_commits),
            )

    _emit(f"Loaded {len(all_commits)} commits")

    _emit("Resolving developer identities...")
    identity_db_path = cfg.cache.directory / "identities.db"
    _strip_suffixes = list(getattr(cfg.analysis.identity, "strip_suffixes", []))
    identity_resolver = DeveloperIdentityResolver(
        identity_db_path,
        similarity_threshold=cfg.analysis.similarity_threshold,
        manual_mappings=cfg.analysis.manual_identity_mappings,
        strip_suffixes=_strip_suffixes or None,
    )
    identity_resolver.update_commit_stats(all_commits)

    from .core.analyzer import GitAnalyzer

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

    llm_cfg = {
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
        llm_config=llm_cfg,
        branch_analysis_config=branch_analysis_config,
        exclude_merge_commits=cfg.analysis.exclude_merge_commits,
    )

    _emit("Loading cached PRs from database...")
    github_repo_paths = [r.github_repo for r in cfg.repositories if getattr(r, "github_repo", None)]
    if github_repo_paths:
        all_prs: list[Any] = cache.get_cached_prs_for_report(
            github_repo_paths, start_date, end_date
        )
        _emit(f"Loaded {len(all_prs)} cached PRs")
    else:
        all_prs = []
        logger.debug("No github_repo configured for any repository; skipping PR cache load")

    _emit("Analyzing ticket references...")
    ticket_analysis = analyzer.ticket_extractor.analyze_ticket_coverage(all_commits, all_prs, None)
    developer_ticket_coverage = analyzer.ticket_extractor.calculate_developer_ticket_coverage(
        all_commits
    )
    developer_stats = identity_resolver.get_developer_stats(
        ticket_coverage=developer_ticket_coverage
    )
    _emit(f"Identified {len(developer_stats)} unique developers")

    exclude_authors = cfg.analysis.exclude_authors
    _anonymize = anonymize or cfg.output.anonymize_enabled

    report_gen = CSVReportGenerator(
        anonymize=_anonymize,
        exclude_authors=exclude_authors,
        identity_resolver=identity_resolver,
    )
    analytics_gen = AnalyticsReportGenerator(
        anonymize=_anonymize,
        exclude_authors=exclude_authors,
        identity_resolver=identity_resolver,
    )
    date_suffix = datetime.now(timezone.utc).strftime("%Y%m%d")
    date_range = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

    generated: list[str] = []

    def _try_report(name: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception as exc:
            msg = f"Report '{name}' failed: {exc}"
            logger.error(msg, exc_info=True)
            _emit(f"  Warning: {msg}")
            result.errors.append(msg)

    if generate_csv:
        weekly_report = output_dir / f"weekly_metrics_{date_suffix}.csv"
        _try_report(
            "weekly_metrics",
            lambda: report_gen.generate_weekly_report(
                all_commits, developer_stats, weekly_report, weeks
            ),
        )
        if weekly_report.exists():
            generated.append(weekly_report.name)

        activity_summary = output_dir / f"developer_activity_summary_{date_suffix}.csv"
        _try_report(
            "developer_activity_summary",
            lambda: report_gen.generate_developer_activity_summary(
                all_commits, developer_stats, all_prs, activity_summary, weeks
            ),
        )
        if activity_summary.exists():
            generated.append(activity_summary.name)

        summary_report = output_dir / f"summary_{date_suffix}.csv"
        _try_report(
            "summary",
            lambda: report_gen.generate_summary_report(
                all_commits, all_prs, developer_stats, ticket_analysis, summary_report, None, None
            ),
        )
        if summary_report.exists():
            generated.append(summary_report.name)

        developer_report = output_dir / f"developers_{date_suffix}.csv"
        _try_report(
            "developer_stats",
            lambda: report_gen.generate_developer_report(developer_stats, developer_report),
        )
        if developer_report.exists():
            generated.append(developer_report.name)

        untracked_report = output_dir / f"untracked_commits_{date_suffix}.csv"
        _try_report(
            "untracked_commits",
            lambda: report_gen.generate_untracked_commits_report(ticket_analysis, untracked_report),
        )
        if untracked_report.exists():
            generated.append(untracked_report.name)

        weekly_cat = output_dir / f"weekly_categorization_{date_suffix}.csv"
        _try_report(
            "weekly_categorization",
            lambda: report_gen.generate_weekly_categorization_report(
                all_commits, analyzer.ticket_extractor, weekly_cat, weeks
            ),
        )
        if weekly_cat.exists():
            generated.append(weekly_cat.name)

    activity_report = output_dir / f"activity_distribution_{date_suffix}.csv"
    _try_report(
        "activity_distribution",
        lambda: analytics_gen.generate_activity_distribution_report(
            all_commits, developer_stats, activity_report
        ),
    )
    if activity_report.exists() and generate_csv:
        generated.append(activity_report.name)

    focus_report = output_dir / f"developer_focus_{date_suffix}.csv"
    _try_report(
        "developer_focus",
        lambda: analytics_gen.generate_developer_focus_report(
            all_commits, developer_stats, focus_report, weeks
        ),
    )
    if focus_report.exists() and generate_csv:
        generated.append(focus_report.name)

    insights_report = output_dir / f"qualitative_insights_{date_suffix}.csv"
    _try_report(
        "qualitative_insights",
        lambda: analytics_gen.generate_qualitative_insights_report(
            all_commits, developer_stats, ticket_analysis, insights_report
        ),
    )
    if insights_report.exists() and generate_csv:
        generated.append(insights_report.name)

    if generate_csv:
        weekly_trends_writer = WeeklyTrendsWriter()
        _try_report(
            "weekly_classification_trends",
            lambda: _generate_classification_trends(
                weekly_trends_writer,
                all_commits,
                output_dir,
                weeks,
                date_suffix,
                analyzer,
                generated,
            ),
        )

        weekly_trends = output_dir / f"weekly_trends_{date_suffix}.csv"
        _try_report(
            "weekly_trends",
            lambda: analytics_gen.generate_weekly_trends_report(
                all_commits, developer_stats, weekly_trends, weeks
            ),
        )
        if weekly_trends.exists():
            generated.append(weekly_trends.name)
            timestamp = weekly_trends.stem.split("_")[-1]
            for extra in [
                output_dir / f"developer_trends_{timestamp}.csv",
                output_dir / f"project_trends_{timestamp}.csv",
            ]:
                if extra.exists():
                    generated.append(extra.name)

        weekly_velocity = output_dir / f"weekly_velocity_{date_suffix}.csv"
        _try_report(
            "weekly_velocity",
            lambda: report_gen.generate_weekly_velocity_report(
                all_commits, all_prs, weekly_velocity, weeks
            ),
        )
        if weekly_velocity.exists():
            generated.append(weekly_velocity.name)

        weekly_dora = output_dir / f"weekly_dora_metrics_{date_suffix}.csv"
        _try_report(
            "weekly_dora",
            lambda: report_gen.generate_weekly_dora_report(
                all_commits, all_prs, weekly_dora, weeks
            ),
        )
        if weekly_dora.exists():
            generated.append(weekly_dora.name)

    dora_metrics: dict[str, Any] = {}
    try:
        dora_calculator = DORAMetricsCalculator()
        dora_metrics = dora_calculator.calculate_dora_metrics(
            all_commits, all_prs, start_date, end_date
        )
    except Exception as exc:
        logger.warning(f"DORA metrics calculation failed: {exc}")

    if "markdown" in cfg.output.formats and generate_csv:
        try:
            from typing import cast

            import pandas as pd  # type: ignore[import-not-found]

            narrative_gen = NarrativeReportGenerator()
            activity_df = pd.read_csv(activity_report)
            focus_df = pd.read_csv(focus_report)
            insights_df = pd.read_csv(insights_report)
            activity_data = cast(list[dict[str, Any]], activity_df.to_dict("records"))
            focus_data = cast(list[dict[str, Any]], focus_df.to_dict("records"))
            insights_data = cast(list[dict[str, Any]], insights_df.to_dict("records"))

            narrative_report = output_dir / f"narrative_report_{date_range}.md"
            narrative_gen.generate_narrative_report(
                all_commits,
                all_prs,
                developer_stats,
                activity_data,
                focus_data,
                insights_data,
                ticket_analysis,
                {},
                narrative_report,
                weeks,
                {},
                "",
                {},
                exclude_authors,
                analysis_start_date=start_date,
                analysis_end_date=end_date,
            )
            generated.append(narrative_report.name)
            _emit(f"Narrative report: {narrative_report}")
        except Exception as exc:
            msg = f"Narrative report failed: {exc}"
            logger.error(msg, exc_info=True)
            _emit(f"  Warning: {msg}")
            result.errors.append(msg)

    if "markdown" in cfg.output.formats:
        try:
            from .core.metrics_storage import DailyMetricsStorage
            from .reports.database_report_generator import DatabaseReportGenerator

            metrics_db_path = cfg.cache.directory / "daily_metrics.db"
            metrics_storage = DailyMetricsStorage(metrics_db_path)
            db_report_gen = DatabaseReportGenerator(metrics_storage)
            db_report = output_dir / f"database_qualitative_report_{date_range}.md"
            analysis_start = start_date.date() if hasattr(start_date, "date") else start_date
            analysis_end = (
                (start_date + timedelta(weeks=weeks)).date()
                if hasattr(start_date, "date")
                else (start_date + timedelta(weeks=weeks))
            )
            db_report_gen.generate_qualitative_report(analysis_start, analysis_end, db_report)
            generated.append(db_report.name)
        except Exception as exc:
            logger.warning(f"Database qualitative report failed: {exc}")

    # --- Native velocity report (Issue #25) ---
    velocity_cfg = getattr(cfg, "velocity", None)
    if velocity_cfg is None or getattr(velocity_cfg, "enabled", True):
        try:
            from .reports.velocity_report import VelocityReportGenerator

            vel_gen = VelocityReportGenerator(velocity_cfg)
            vel_gen.generate(
                all_prs,
                output_dir,
                start_date=start_date.date() if hasattr(start_date, "date") else start_date,
                end_date=end_date.date() if hasattr(end_date, "date") else end_date,
            )
            generated.append("velocity_summary.json")
        except Exception as exc:
            msg = f"Velocity report failed: {exc}"
            logger.error(msg, exc_info=True)
            result.errors.append(msg)

    # --- Team/pod aggregation report (Issue #23) ---
    teams_cfg = getattr(cfg, "teams", None)
    if teams_cfg is None or getattr(teams_cfg, "enabled", True):
        try:
            from .reports.team_aggregator import TeamAggregator

            team_aggregator = TeamAggregator(teams_cfg)
            # Build lightweight proxy dicts from all_commits so the aggregator
            # can resolve team membership via author_email / developer fields.
            # When a full daily_metrics DB integration is added in future, this
            # can be replaced with a richer data source.
            commit_proxy: list[dict[str, Any]] = [
                {
                    "developer_email": c.get("author_email") or c.get("canonical_id"),
                    "developer_id": c.get("developer_id") or c.get("canonical_id"),
                    "developer_name": c.get("author_name") or c.get("author"),
                    "author_email": c.get("author_email"),
                    "author": c.get("author"),
                    "total_commits": 1,
                    "feature_commits": 1 if c.get("commit_type") == "feature" else 0,
                    "bug_fix_commits": 1 if c.get("commit_type") == "bug_fix" else 0,
                    "refactor_commits": 1 if c.get("commit_type") == "refactor" else 0,
                    "documentation_commits": 1 if c.get("commit_type") == "documentation" else 0,
                    "maintenance_commits": 1 if c.get("commit_type") == "maintenance" else 0,
                    "test_commits": 1 if c.get("commit_type") == "test" else 0,
                    "lines_added": int(c.get("lines_added") or 0),
                    "lines_deleted": int(c.get("lines_deleted") or 0),
                    "story_points": int(c.get("story_points") or 0),
                    "tracked_commits": 1 if c.get("ticket_reference") else 0,
                    "untracked_commits": 0 if c.get("ticket_reference") else 1,
                    "ai_assisted_commits": 1 if c.get("is_ai_assisted") else 0,
                    "ai_generated_commits": 1 if c.get("is_ai_generated") else 0,
                    "timestamp": c.get("timestamp"),
                }
                for c in all_commits
            ]

            # --- Boilerplate filter (Issue #28) -----------------------------
            # Aggregate per-developer-per-week metrics, apply the filter, and
            # replace the commit proxy with the filtered detail rows before
            # the team aggregator runs.  When the filter is disabled this is
            # a no-op pass-through.
            bp_cfg = getattr(cfg, "boilerplate_filter", None)
            if bp_cfg is not None and getattr(bp_cfg, "enabled", False):
                from .reports.boilerplate_filter import (
                    BoilerplateFilter,
                    apply_boilerplate_filter_to_commits,
                )

                bp_filter = BoilerplateFilter(bp_cfg)
                commit_proxy = apply_boilerplate_filter_to_commits(bp_filter, commit_proxy)
                _emit(
                    f"Boilerplate filter action='{bp_cfg.action}' applied "
                    f"(remaining proxy rows: {len(commit_proxy)})"
                )

            summary = team_aggregator.generate(commit_proxy, output_dir)
            if summary:
                generated.append("weekly_summary.json")
        except Exception as exc:
            msg = f"Team aggregation report failed: {exc}"
            logger.error(msg, exc_info=True)
            result.errors.append(msg)

    # --- Ticketing activity report (GitHub Issues + Confluence) ---
    gh_issues_cfg = getattr(cfg, "github_issues", None)
    confluence_cfg = getattr(cfg, "confluence", None)
    if (gh_issues_cfg and getattr(gh_issues_cfg, "enabled", False)) or (
        confluence_cfg and getattr(confluence_cfg, "enabled", False)
    ):

        def _gen_ticketing_reports() -> None:
            from .reports.ticketing_activity_report import TicketingActivityReport

            ticket_gen = TicketingActivityReport(cache=cache, identity_resolver=identity_resolver)
            written = ticket_gen.write_reports(output_dir, start_date, end_date)
            for path_str in written:
                name = Path(path_str).name
                if name not in generated:
                    generated.append(name)

        _try_report("ticketing_activity", _gen_ticketing_reports)

    # --- Native quality report (Issue #26) ---
    quality_cfg = getattr(cfg, "quality_report", None)
    if quality_cfg is None or getattr(quality_cfg, "enabled", True):
        try:
            from .reports.quality_report import QualityReportGenerator

            qual_gen = QualityReportGenerator(quality_cfg)
            # qualitative_data is already merged into all_commits above; pass []
            # here because the generator accepts a separate qualitative list only
            # for the commit_hash join — the pipeline already embedded risk_level
            # and complexity directly on each commit dict via the qual_rows loop.
            # A future enhancement can extract and pass the raw qual rows instead.
            qual_gen.generate(all_commits, [], all_prs, output_dir)
            generated.append("quality_summary.json")
        except Exception as exc:
            msg = f"Quality report failed: {exc}"
            logger.error(msg, exc_info=True)
            result.errors.append(msg)

    if "json" in cfg.output.formats:
        try:
            json_report = output_dir / f"comprehensive_export_{date_suffix}.json"
            json_exporter = ComprehensiveJSONExporter(anonymize=_anonymize)
            json_exporter.export_comprehensive_data(
                commits=all_commits,
                prs=all_prs,
                developer_stats=developer_stats,
                project_metrics={
                    "ticket_analysis": ticket_analysis,
                    "pr_metrics": {},
                    "enrichments": {},
                },
                dora_metrics=dora_metrics,
                output_path=json_report,
                weeks=weeks,
                pm_data=None,
                qualitative_data=None,
                enhanced_qualitative_analysis=None,
            )
            generated.append(json_report.name)
        except Exception as exc:
            msg = f"JSON export failed: {exc}"
            logger.error(msg, exc_info=True)
            result.errors.append(msg)

    result.generated_reports = generated
    _emit(f"Generated {len(generated)} reports in {output_dir}")
    return result


def _generate_classification_trends(
    writer: Any,
    all_commits: list[dict[str, Any]],
    output_dir: Path,
    weeks: int,
    date_suffix: str,
    analyzer: Any,
    generated: list[str],
) -> None:
    """Helper to generate weekly classification trend reports."""
    trends_paths = writer.generate_weekly_trends_reports(
        all_commits,
        output_dir,
        weeks,
        f"_{date_suffix}",
        categorize_fn=analyzer.ticket_extractor.categorize_commit,
    )
    for _, report_path in trends_paths.items():
        generated.append(report_path.name)
