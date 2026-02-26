"""Stage 2 — classify: run batch LLM classification on cached commits."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from .pipeline_types import ClassifyResult

logger = logging.getLogger(__name__)


def run_classify(
    cfg: Any,
    weeks: int,
    reclassify: bool = False,
    progress_callback: Callable[[str], None] | None = None,
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

    Returns:
        A :class:`ClassifyResult` with summary statistics.
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

    except Exception as exc:
        msg = f"Classification failed: {exc}"
        logger.error(msg, exc_info=True)
        _emit(msg)
        result.errors.append(msg)

    return result
