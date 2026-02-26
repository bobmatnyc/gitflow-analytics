"""Batch LLM classifier for intelligent commit categorization with context.

This module implements the second step of the two-step fetch/analyze process,
providing intelligent batch classification of commits using LLM with ticket context.

Public API and orchestration live here. DB access, LLM calling, and storage
implementation details live in batch_classifier_impl.py (BatchClassifierImplMixin).
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from ..core.progress import get_progress_service
from ..models.database import DailyCommitBatch, Database
from ..qualitative.classifiers.llm_commit_classifier import LLMCommitClassifier, LLMConfig
from .batch_classifier_impl import BatchClassifierImplMixin

logger = logging.getLogger(__name__)


class BatchCommitClassifier(BatchClassifierImplMixin):
    """Intelligent batch classifier using LLM with ticket context.

    WHY: This class implements the second step of the two-step process by:
    - Reading cached commit data organized by day/week
    - Adding ticket context to improve classification accuracy
    - Sending batches of commits to LLM for intelligent classification
    - Falling back to rule-based classification when LLM fails
    - Storing results with confidence tracking

    DESIGN DECISION: Uses batch processing to reduce API calls and costs
    while providing better context for classification accuracy.

    PROGRESS REPORTING: Provides granular progress feedback with nested progress bars:
    - Repository level: Shows which repository is being processed (position 0)
    - Weekly level: Shows week being processed within repository (position 1)
    - API batch level: Shows LLM API batches being processed (position 2)
    Each level shows commit counts and progress indicators for user feedback.
    """

    def __init__(
        self,
        cache_dir: Path,
        llm_config: Optional[dict[str, Any]] = None,
        batch_size: int = 50,
        confidence_threshold: float = 0.7,
        fallback_enabled: bool = True,
        max_processing_time_minutes: int = 30,  # Maximum time for classification
    ):
        """Initialize the batch classifier.

        Args:
            cache_dir: Path to cache directory containing database
            llm_config: Configuration for LLM classifier
            batch_size: Number of commits per batch (max 50 for token limits)
            confidence_threshold: Minimum confidence for LLM classification
            fallback_enabled: Whether to fall back to rule-based classification
        """
        self.cache_dir = cache_dir
        self.database = Database(cache_dir / "gitflow_cache.db")
        self.batch_size = min(batch_size, 50)  # Limit for token constraints
        self.confidence_threshold = confidence_threshold
        self.fallback_enabled = fallback_enabled
        self.max_processing_time_minutes = max_processing_time_minutes
        self.classification_start_time = None

        # Initialize LLM classifier
        # Handle different config types
        if isinstance(llm_config, dict):
            # Convert dict config to LLMConfig object, forwarding all provider fields
            llm_config_obj = LLMConfig(
                provider=llm_config.get("provider", "auto"),
                api_key=llm_config.get("api_key", ""),
                model=llm_config.get("model", "mistralai/mistral-7b-instruct"),
                aws_region=llm_config.get("aws_region"),
                aws_profile=llm_config.get("aws_profile"),
                bedrock_model_id=llm_config.get(
                    "bedrock_model_id", "anthropic.claude-3-haiku-20240307-v1:0"
                ),
                max_tokens=llm_config.get("max_tokens", 50),
                temperature=llm_config.get("temperature", 0.1),
                confidence_threshold=llm_config.get("confidence_threshold", 0.7),
                timeout_seconds=llm_config.get("timeout_seconds", 30),
                cache_duration_days=llm_config.get("cache_duration_days", 7),
                enable_caching=llm_config.get("enable_caching", True),
                max_daily_requests=llm_config.get("max_daily_requests", 1000),
            )
        elif hasattr(llm_config, "api_key"):
            # Use provided config object (e.g., mock config for testing)
            llm_config_obj = llm_config
        else:
            # Use default LLMConfig
            llm_config_obj = LLMConfig()

        self.llm_classifier = LLMCommitClassifier(config=llm_config_obj, cache_dir=cache_dir)

        # Determine if LLM is operational.  With Bedrock, no api_key is needed
        # so we check whether the classifier actually has a provider set up.
        classifier_has_provider = self.llm_classifier.classifier is not None
        if not classifier_has_provider and not llm_config_obj.api_key:
            logger.warning(
                "No LLM provider configured (no AWS credentials or OpenRouter key). "
                "Will fall back to rule-based classification."
            )
            self.llm_enabled = False
        else:
            self.llm_enabled = True
            provider_name = (
                self.llm_classifier.classifier.get_provider_name()
                if self.llm_classifier.classifier
                else "unknown"
            )
            logger.info(
                "LLM Classifier initialised: provider=%s model=%s",
                provider_name,
                llm_config_obj.model,
            )

        # Circuit breaker for LLM API failures
        self.api_failure_count = 0
        self.max_consecutive_failures = 5
        self.circuit_breaker_open = False

        # Rule-based fallback patterns for when LLM fails
        self.fallback_patterns = {
            "feature": [
                r"feat(?:ure)?[\(\:]",
                r"add(?:ed|ing)?.*(?:feature|functionality|capability)",
                r"implement(?:ed|ing|s)?",
                r"introduce(?:d|s)?",
            ],
            "bug_fix": [
                r"fix(?:ed|es|ing)?[\(\:]",
                r"bug[\(\:]",
                r"resolve(?:d|s)?",
                r"repair(?:ed|ing|s)?",
                r"correct(?:ed|ing|s)?",
            ],
            "refactor": [
                r"refactor(?:ed|ing|s)?[\(\:]",
                r"restructure(?:d|ing|s)?",
                r"optimize(?:d|ing|s)?",
                r"improve(?:d|ing|s)?",
                r"clean(?:ed|ing)?\s+up",
            ],
            "documentation": [
                r"docs?[\(\:]",
                r"documentation[\(\:]",
                r"readme",
                r"update.*(?:comment|docs?|documentation)",
            ],
            "maintenance": [
                r"chore[\(\:]",
                r"maintenance[\(\:]",
                r"update.*(?:dependencies|deps)",
                r"bump.*version",
                r"cleanup",
            ],
            "test": [
                r"test(?:s|ing)?[\(\:]",
                r"spec[\(\:]",
                r"add.*(?:test|spec)",
                r"fix.*test",
            ],
            "style": [
                r"style[\(\:]",
                r"format(?:ted|ting)?[\(\:]",
                r"lint(?:ed|ing)?",
                r"prettier",
                r"whitespace",
            ],
            "build": [
                r"build[\(\:]",
                r"ci[\(\:]",
                r"deploy(?:ed|ment)?",
                r"docker",
                r"webpack",
                r"package\.json",
            ],
        }

    def classify_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        project_keys: Optional[list[str]] = None,
        force_reclassify: bool = False,
    ) -> dict[str, Any]:
        """Classify all commits in a date range using batch processing.

        Args:
            start_date: Start date for classification
            end_date: End date for classification
            project_keys: Optional list of specific projects to classify
            force_reclassify: Whether to reclassify already processed batches

        Returns:
            Dictionary containing classification results and statistics
        """
        logger.info(f"Starting batch classification from {start_date.date()} to {end_date.date()}")
        self.classification_start_time = datetime.utcnow()

        # Get daily batches to process
        batches_to_process = self._get_batches_to_process(
            start_date, end_date, project_keys, force_reclassify
        )

        if not batches_to_process:
            logger.info("No batches need classification")
            return {"processed_batches": 0, "total_commits": 0}

        # Group batches by repository first for better progress reporting
        repo_batches = self._group_batches_by_repository(batches_to_process)

        total_processed = 0
        total_commits = 0

        # Use centralized progress service
        progress = get_progress_service()

        # Add progress bar for repository processing
        with progress.progress(
            total=len(repo_batches),
            description="AI Classification",
            unit="repo",
            nested=False,
            leave=True,
        ) as repo_ctx:
            for repo_num, (repo_info, repo_batch_list) in enumerate(repo_batches.items(), 1):
                project_key, repo_path = repo_info
                repo_name = Path(repo_path).name if repo_path else project_key

                # Count commits in this repository for detailed progress
                repo_commit_count = sum(batch.commit_count for batch in repo_batch_list)

                progress.set_description(
                    repo_ctx, f"Classifying {repo_name} ({repo_commit_count} commits)"
                )
                logger.info(
                    f"Processing repository {repo_num}/{len(repo_batches)}: {repo_name} ({len(repo_batch_list)} batches, {repo_commit_count} commits)"
                )

                # Check if we've exceeded max processing time
                if self.classification_start_time:
                    elapsed_minutes = (
                        datetime.utcnow() - self.classification_start_time
                    ).total_seconds() / 60
                    if elapsed_minutes > self.max_processing_time_minutes:
                        logger.error(
                            f"Classification exceeded maximum time limit of {self.max_processing_time_minutes} minutes. "
                            f"Stopping classification to prevent hanging."
                        )
                        break

                # Process this repository's batches by week for optimal context
                weekly_batches = self._group_batches_by_week(repo_batch_list)

                repo_processed = 0
                repo_commits_processed = 0

                # Add nested progress bar for weekly processing within repository
                with progress.progress(
                    total=len(weekly_batches),
                    description="  Processing weeks",
                    unit="week",
                    nested=True,
                    leave=False,
                ) as week_ctx:
                    for week_num, (week_start, week_batches) in enumerate(
                        weekly_batches.items(), 1
                    ):
                        progress.set_description(
                            week_ctx,
                            f"  Week {week_num}/{len(weekly_batches)} ({week_start.strftime('%Y-%m-%d')})",
                        )
                        logger.info(
                            f"  Processing week starting {week_start}: {len(week_batches)} daily batches"
                        )

                        week_result = self._classify_weekly_batches(week_batches)
                        repo_processed += week_result["batches_processed"]
                        repo_commits_processed += week_result["commits_processed"]

                        progress.update(week_ctx, 1)
                        # Update description to show commits processed
                        progress.set_description(
                            week_ctx,
                            f"  Week {week_num}/{len(weekly_batches)} - {week_result['commits_processed']} commits",
                        )

                total_processed += repo_processed
                total_commits += repo_commits_processed

                progress.update(repo_ctx, 1)
                # Update description to show total progress
                progress.set_description(
                    repo_ctx,
                    f"AI Classification [{repo_num}/{len(repo_batches)} repos, {total_commits} commits]",
                )

                logger.info(
                    f"  Repository {repo_name} completed: {repo_processed} batches, {repo_commits_processed} commits"
                )

        # Store daily metrics from classification results
        self._store_daily_metrics(start_date, end_date, project_keys)

        logger.info(
            f"Batch classification completed: {total_processed} batches, {total_commits} commits"
        )

        return {
            "processed_batches": total_processed,
            "total_commits": total_commits,
            "date_range": {"start": start_date, "end": end_date},
            "project_keys": project_keys or [],
        }

    def _get_batches_to_process(
        self,
        start_date: datetime,
        end_date: datetime,
        project_keys: Optional[list[str]],
        force_reclassify: bool,
    ) -> list[DailyCommitBatch]:
        """Get daily commit batches that need classification."""
        session = self.database.get_session()

        try:
            query = session.query(DailyCommitBatch).filter(
                DailyCommitBatch.date >= start_date.date(), DailyCommitBatch.date <= end_date.date()
            )

            if project_keys:
                query = query.filter(DailyCommitBatch.project_key.in_(project_keys))

            if not force_reclassify:
                # Only get batches that haven't been classified or failed
                query = query.filter(
                    DailyCommitBatch.classification_status.in_(["pending", "failed"])
                )

            batches = query.order_by(DailyCommitBatch.date).all()
            logger.info(f"Found {len(batches)} batches needing classification")

            # Debug: Log filtering criteria
            logger.debug(
                f"Query criteria: start_date={start_date.date()}, end_date={end_date.date()}"
            )
            if project_keys:
                logger.debug(f"Project key filter: {project_keys}")
            logger.debug(f"Force reclassify: {force_reclassify}")

            return batches

        except Exception as e:
            logger.error(f"Error getting batches to process: {e}")
            return []
        finally:
            session.close()

    def _group_batches_by_repository(
        self, batches: list[DailyCommitBatch]
    ) -> dict[tuple[str, str], list[DailyCommitBatch]]:
        """Group daily batches by repository for granular progress reporting."""
        repo_batches = defaultdict(list)

        for batch in batches:
            # Use (project_key, repo_path) as the key for unique repository identification
            repo_key = (batch.project_key, batch.repo_path)
            repo_batches[repo_key].append(batch)

        # Sort each repository's batches by date
        for batches_list in repo_batches.values():
            batches_list.sort(key=lambda b: b.date)

        return dict(repo_batches)

    def _group_batches_by_week(
        self, batches: list[DailyCommitBatch]
    ) -> dict[datetime, list[DailyCommitBatch]]:
        """Group daily batches by week for optimal context window."""
        weekly_batches = defaultdict(list)

        for batch in batches:
            # Get Monday of the week
            batch_date = datetime.combine(batch.date, datetime.min.time())
            days_since_monday = batch_date.weekday()
            week_start = batch_date - timedelta(days=days_since_monday)

            weekly_batches[week_start].append(batch)

        # Sort each week's batches by date
        for week_batches in weekly_batches.values():
            week_batches.sort(key=lambda b: b.date)

        return dict(weekly_batches)
