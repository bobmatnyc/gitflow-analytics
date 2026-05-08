"""Stage 2 — classify: run batch LLM classification on cached commits."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from .constants import Thresholds
from .pipeline_types import ClassifyResult

logger = logging.getLogger(__name__)


def run_classify(
    cfg: Any,
    weeks: int,
    reclassify: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    show_jira_signals: bool = False,
    coverage_threshold: float = Thresholds.CLASSIFICATION_COVERAGE_DEFAULT,
) -> ClassifyResult:
    """Run batch LLM classification on commits that are already in the cache.

    This is Stage 2 of the pipeline.  It reads :class:`DailyCommitBatch`
    rows that were created during Stage 1 (collect) and writes classification
    results back to those rows.

    Args:
        cfg: Loaded configuration object.
        weeks: Number of weeks to classify (used to compute the date range).
        reclassify: When True, re-classify commits that were already classified.
        progress_callback: Optional function called with status messages.
        show_jira_signals: When True, log INFO-level messages for every commit
            classified via the JIRA project-key mapping (issue #62).
        coverage_threshold: Per-repo classification coverage percentage below
            which a ``logging.warning()`` is emitted (issue #65). The warning
            is informational and does not affect the return value; CLI gating
            via ``--validate-coverage`` is handled by the caller.

    Returns:
        A :class:`ClassifyResult` with summary statistics, including
        :attr:`coverage_by_repo` populated for every repository that had
        classified commits in the analysed period.
    """
    from .classification.batch_classifier import BatchCommitClassifier
    from .core.cache import GitAnalysisCache
    from .utils.date_utils import get_week_end, get_week_start

    def _emit(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    result = ClassifyResult()

    current_time = datetime.now(timezone.utc)
    current_week_start = get_week_start(current_time)
    last_complete_week_start = current_week_start - timedelta(weeks=1)
    start_date = last_complete_week_start - timedelta(weeks=weeks - 1)
    end_date = get_week_end(last_complete_week_start + timedelta(days=6))

    _emit(f"Classify period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    cache = GitAnalysisCache(cfg.cache.directory)

    from sqlalchemy import and_

    from .models.database import CachedCommit, DailyCommitBatch

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
        stored_batches = (
            session.query(DailyCommitBatch)
            .filter(
                and_(
                    DailyCommitBatch.date >= start_date.date(),
                    DailyCommitBatch.date <= end_date.date(),
                )
            )
            .count()
        )

    if stored_commits == 0:
        msg = (
            f"No commits found in the cache for the period "
            f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}. "
            "Run 'gfa collect' first."
        )
        _emit(msg)
        result.errors.append(msg)
        return result

    if stored_batches == 0:
        msg = (
            f"Found {stored_commits} commits but no daily batches. "
            "Data may be inconsistent — try running 'gfa collect -f' to re-fetch."
        )
        _emit(msg)
        result.errors.append(msg)
        return result

    _emit(f"Found {stored_commits} commits in {stored_batches} batches — starting classification")

    llm_config = {
        "enabled": cfg.analysis.llm_classification.enabled,
        "provider": cfg.analysis.llm_classification.provider,
        "api_key": cfg.analysis.llm_classification.api_key,
        "model": cfg.analysis.llm_classification.model,
        "aws_region": cfg.analysis.llm_classification.aws_region,
        "aws_profile": cfg.analysis.llm_classification.aws_profile,
        "bedrock_model_id": cfg.analysis.llm_classification.bedrock_model_id,
        "confidence_threshold": cfg.analysis.llm_classification.confidence_threshold,
        "max_tokens": cfg.analysis.llm_classification.max_tokens,
        "temperature": cfg.analysis.llm_classification.temperature,
        "timeout_seconds": cfg.analysis.llm_classification.timeout_seconds,
        "cache_duration_days": cfg.analysis.llm_classification.cache_duration_days,
        "enable_caching": cfg.analysis.llm_classification.enable_caching,
        "max_daily_requests": cfg.analysis.llm_classification.max_daily_requests,
    }

    batch_classifier = BatchCommitClassifier(
        cache_dir=cfg.cache.directory,
        llm_config=llm_config,
        batch_size=50,
        confidence_threshold=cfg.analysis.llm_classification.confidence_threshold,
        fallback_enabled=True,
        # Issue #62: JIRA project-key → work_type mapping (tier-3 signal).
        jira_project_mappings=getattr(cfg, "jira_project_mappings", None) or {},
        show_jira_signals=show_jira_signals,
        # Issue #69: native change_type → custom work_type mapping.
        taxonomy_mapping=getattr(cfg, "taxonomy_mapping", None) or {},
    )

    project_keys = [repo_config.project_key or repo_config.name for repo_config in cfg.repositories]

    try:
        classification_result = batch_classifier.classify_date_range(
            start_date=start_date,
            end_date=end_date,
            project_keys=project_keys,
            force_reclassify=reclassify,
        )

        result.processed_batches = classification_result.get("processed_batches", 0)
        result.total_commits = classification_result.get("total_commits", 0)
        result.skipped_batches = classification_result.get("skipped_batches", 0)

        _emit(
            f"Classification complete: {result.processed_batches} batches, "
            f"{result.total_commits} commits"
        )

        if result.skipped_batches:
            _emit(f"Skipped {result.skipped_batches} already-classified batches")

        # Issue #69: Fast taxonomy-only remap path. Updates work_type column
        # for any rows whose mapped value drifted from config (e.g. user added
        # a new mapping or ran --reclassify with skipped batches). Cheap — no
        # LLM calls; runs only when taxonomy_mapping is configured.
        taxonomy_cfg = getattr(cfg, "taxonomy_mapping", None) or {}
        if taxonomy_cfg:
            try:
                session = batch_classifier.database.get_session()
                try:
                    updated = batch_classifier._apply_taxonomy_remap(session)
                finally:
                    session.close()
                if updated:
                    _emit(f"Taxonomy remap: updated work_type on {updated} rows")
            except Exception as exc:
                logger.warning("Taxonomy remap failed: %s", exc)

        # Issue #65: measure per-repo classification coverage and emit
        # warnings when too many commits fell to maintenance/KTLO.  This
        # always runs (regardless of --validate-coverage) so users see the
        # warning even if they are not running in CI.
        _measure_and_warn_coverage(
            cache=cache,
            cfg=cfg,
            start_date=start_date,
            end_date=end_date,
            coverage_threshold=coverage_threshold,
            result=result,
            emit=_emit,
        )

    except Exception as exc:
        msg = f"Classification failed: {exc}"
        logger.error(msg, exc_info=True)
        _emit(msg)
        result.errors.append(msg)

    return result


def _measure_and_warn_coverage(
    cache: Any,
    cfg: Any,
    start_date: datetime,
    end_date: datetime,
    coverage_threshold: float,
    result: ClassifyResult,
    emit: Callable[[str], None],
) -> None:
    """Compute per-repo coverage, log warnings, and persist to the summary table.

    Why split out: keeps :func:`run_classify` focused on orchestration; the
    coverage logic touches three separate concerns (compute, log, persist)
    and is easier to test in isolation.

    Args:
        cache: ``GitAnalysisCache`` instance whose session is reused.
        cfg: Loaded configuration (used for repository name lookups).
        start_date: Inclusive lower bound on commit timestamps.
        end_date: Inclusive upper bound on commit timestamps.
        coverage_threshold: Warn when per-repo coverage falls below this %.
        result: Mutated to populate :attr:`coverage_by_repo`.
        emit: Progress callback for user-visible output.
    """
    from .classification.coverage import (
        compute_repo_coverage,
        format_low_coverage_warning,
    )
    from .models.database import RepositoryAnalysisStatus

    repo_paths_by_name: dict[str, str] = {str(repo.path): repo.name for repo in cfg.repositories}

    with cache.get_session() as session:
        for repo_path, repo_name in repo_paths_by_name.items():
            try:
                coverage_pct = compute_repo_coverage(
                    session=session,
                    repo_path=repo_path,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as e:
                # Coverage measurement should never break classification.
                logger.debug("Coverage computation failed for %s: %s", repo_name, e)
                continue

            if coverage_pct is None:
                # No classified commits in the period — undefined coverage.
                continue

            # Defensive: skip non-numeric returns (e.g. when callers passed
            # a MagicMock-backed session in tests; production callers always
            # supply a real SQLAlchemy session that returns float|None).
            if not isinstance(coverage_pct, (int, float)):
                continue

            result.coverage_by_repo[repo_path] = float(coverage_pct)

            if coverage_pct < coverage_threshold:
                warning_msg = format_low_coverage_warning(repo_name, coverage_pct)
                logger.warning("WARNING: %s", warning_msg)
                emit(f"WARNING: {warning_msg}")

            # Persist on RepositoryAnalysisStatus rows that match this repo.
            # We update every row for the period rather than inserting a new
            # one because RepositoryAnalysisStatus is created by the analyze
            # workflow (not by classify alone), so we just upsert the column
            # on existing rows when present.
            try:
                rows = (
                    session.query(RepositoryAnalysisStatus)
                    .filter(RepositoryAnalysisStatus.repo_path == repo_path)
                    .all()
                )
                for row in rows:
                    row.classification_coverage_pct = coverage_pct
                if rows:
                    session.commit()
            except Exception as e:
                logger.debug(
                    "Could not persist classification_coverage_pct for %s: %s",
                    repo_name,
                    e,
                )
                session.rollback()
