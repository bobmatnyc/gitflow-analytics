"""Report generation pipeline stages.

Extracted from analyze_pipeline.py. Handles qualitative analysis,
PM data aggregation, and report generation stages.
"""

from __future__ import annotations

import contextlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .analyze_pipeline import (
    QualitativeResult,
    ReportResult,
)

logger = logging.getLogger(__name__)


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
