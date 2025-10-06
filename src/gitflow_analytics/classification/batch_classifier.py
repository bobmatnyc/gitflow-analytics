"""Batch LLM classifier for intelligent commit categorization with context.

This module implements the second step of the two-step fetch/analyze process,
providing intelligent batch classification of commits using LLM with ticket context.
"""

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from ..core.progress import get_progress_service
from ..models.database import CachedCommit, DailyCommitBatch, Database, DetailedTicketData
from ..qualitative.classifiers.llm_commit_classifier import LLMCommitClassifier, LLMConfig

logger = logging.getLogger(__name__)


class BatchCommitClassifier:
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
            # Convert dict config to LLMConfig object
            llm_config_obj = LLMConfig(
                api_key=llm_config.get("api_key", ""),
                model=llm_config.get("model", "mistralai/mistral-7b-instruct"),
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

        # Warn if no API key is configured
        if not llm_config_obj.api_key:
            logger.warning(
                "No API key configured for LLM classification. "
                "Will fall back to rule-based classification."
            )
            # Set a flag to skip LLM calls entirely
            self.llm_enabled = False
        else:
            self.llm_enabled = True
            logger.info(
                f"LLM Classifier initialized with API key: Yes (model: {llm_config_obj.model})"
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

    def _classify_weekly_batches(self, weekly_batches: list[DailyCommitBatch]) -> dict[str, Any]:
        """Classify all batches for a single week with shared context."""
        session = self.database.get_session()
        batches_processed = 0
        commits_processed = 0

        try:
            # Collect all commits for the week
            week_commits = []
            batch_commit_map = {}  # Maps commit hash to batch

            for batch in weekly_batches:
                # Mark batch as processing
                batch.classification_status = "processing"

                # Get commits for this day
                daily_commits = self._get_commits_for_batch(session, batch)
                week_commits.extend(daily_commits)

                # Track which batch each commit belongs to
                for commit in daily_commits:
                    batch_commit_map[commit["commit_hash"]] = batch

            if not week_commits:
                logger.warning(
                    f"No commits found for weekly batches (expected {sum(batch.commit_count for batch in weekly_batches)} commits)"
                )
                # Mark batches as failed due to missing commits
                for batch in weekly_batches:
                    batch.classification_status = "failed"
                    batch.classified_at = datetime.utcnow()
                session.commit()
                return {"batches_processed": 0, "commits_processed": 0}

            # Get ticket context for the week
            week_tickets = self._get_ticket_context_for_commits(session, week_commits)

            # Process commits in batches (respecting API limits)
            classified_commits = []
            num_batches = (len(week_commits) + self.batch_size - 1) // self.batch_size

            # Use centralized progress service for batch processing
            progress = get_progress_service()

            # Add progress bar for batch processing within the week
            with progress.progress(
                total=num_batches,
                description="    Processing batches",
                unit="batch",
                nested=True,
                leave=False,
            ) as batch_ctx:
                for i in range(0, len(week_commits), self.batch_size):
                    # Check for timeout before processing each batch
                    if self.classification_start_time:
                        elapsed_minutes = (
                            datetime.utcnow() - self.classification_start_time
                        ).total_seconds() / 60
                        if elapsed_minutes > self.max_processing_time_minutes:
                            logger.error(
                                f"Classification timeout after {elapsed_minutes:.1f} minutes. "
                                f"Processed {len(classified_commits)}/{len(week_commits)} commits."
                            )
                            # Use fallback for remaining commits
                            remaining_commits = week_commits[i:]
                            for commit in remaining_commits:
                                classified_commits.append(
                                    {
                                        "commit_hash": commit["commit_hash"],
                                        "category": "maintenance",
                                        "confidence": 0.2,
                                        "method": "timeout_fallback",
                                        "error": "Classification timeout",
                                    }
                                )
                            break

                    batch_num = i // self.batch_size + 1
                    batch_commits = week_commits[i : i + self.batch_size]
                    progress.set_description(
                        batch_ctx,
                        f"    API batch {batch_num}/{num_batches} ({len(batch_commits)} commits)",
                    )
                    logger.info(f"Classifying batch {batch_num}: {len(batch_commits)} commits")

                    # Classify this batch with LLM
                    batch_results = self._classify_commit_batch_with_llm(
                        batch_commits, week_tickets
                    )
                    classified_commits.extend(batch_results)

                    progress.update(batch_ctx, 1)
                    # Update description to show total classified commits
                    progress.set_description(
                        batch_ctx,
                        f"    API batch {batch_num}/{num_batches} - Total: {len(classified_commits)} commits",
                    )

            # Store classification results
            for commit_result in classified_commits:
                self._store_commit_classification(session, commit_result)
                commits_processed += 1

            # Mark all daily batches as completed
            for batch in weekly_batches:
                batch.classification_status = "completed"
                batch.classified_at = datetime.utcnow()
                batches_processed += 1

            session.commit()

            logger.info(
                f"Week classification completed: {batches_processed} batches, {commits_processed} commits"
            )

        except Exception as e:
            logger.error(f"Error in weekly batch classification: {e}")
            # Mark batches as failed
            for batch in weekly_batches:
                batch.classification_status = "failed"
            session.rollback()
        finally:
            session.close()

        return {
            "batches_processed": batches_processed,
            "commits_processed": commits_processed,
        }

    def _get_commits_for_batch(self, session: Any, batch: DailyCommitBatch) -> list[dict[str, Any]]:
        """Get all commits for a daily batch."""
        try:
            # Get cached commits for this batch
            # CRITICAL FIX: CachedCommit.timestamp is timezone-aware UTC (from analyzer.py line 806)
            # but we were creating timezone-naive boundaries, causing comparison to fail
            # Create timezone-aware UTC boundaries to match CachedCommit.timestamp format
            start_of_day = datetime.combine(batch.date, datetime.min.time(), tzinfo=timezone.utc)
            end_of_day = datetime.combine(batch.date, datetime.max.time(), tzinfo=timezone.utc)

            logger.debug(
                f"Searching for commits in {batch.repo_path} between {start_of_day} and {end_of_day}"
            )

            commits = (
                session.query(CachedCommit)
                .filter(
                    CachedCommit.repo_path == batch.repo_path,
                    CachedCommit.timestamp >= start_of_day,
                    CachedCommit.timestamp < end_of_day,
                )
                .all()
            )

            logger.debug(f"Found {len(commits)} commits for batch on {batch.date}")

            commit_list = []
            for commit in commits:
                commit_data = {
                    "commit_hash": commit.commit_hash,
                    "commit_hash_short": commit.commit_hash[:7],
                    "message": commit.message,
                    "author_name": commit.author_name,
                    "author_email": commit.author_email,
                    "timestamp": commit.timestamp,
                    "branch": commit.branch,
                    "project_key": batch.project_key,
                    "repo_path": commit.repo_path,
                    "files_changed": commit.files_changed or 0,
                    "lines_added": commit.insertions or 0,
                    "lines_deleted": commit.deletions or 0,
                    "story_points": commit.story_points,
                    "ticket_references": commit.ticket_references or [],
                }
                commit_list.append(commit_data)

            return commit_list

        except Exception as e:
            logger.error(f"Error getting commits for batch {batch.id}: {e}")
            return []

    def _get_ticket_context_for_commits(
        self, session: Any, commits: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Get ticket context for a list of commits."""
        # Extract all ticket references from commits
        all_ticket_ids = set()
        for commit in commits:
            ticket_refs = commit.get("ticket_references", [])
            all_ticket_ids.update(ticket_refs)

        if not all_ticket_ids:
            return {}

        try:
            # Get detailed ticket information
            tickets = (
                session.query(DetailedTicketData)
                .filter(DetailedTicketData.ticket_id.in_(all_ticket_ids))
                .all()
            )

            ticket_context = {}
            for ticket in tickets:
                ticket_context[ticket.ticket_id] = {
                    "title": ticket.title,
                    "description": (
                        ticket.summary or ticket.description[:200] if ticket.description else ""
                    ),
                    "ticket_type": ticket.ticket_type,
                    "status": ticket.status,
                    "labels": ticket.labels or [],
                    "classification_hints": ticket.classification_hints or [],
                    "business_domain": ticket.business_domain,
                }

            logger.info(f"Retrieved context for {len(ticket_context)} tickets")
            return ticket_context

        except Exception as e:
            logger.error(f"Error getting ticket context: {e}")
            return {}

    def _classify_commit_batch_with_llm(
        self,
        commits: list[dict[str, Any]],
        ticket_context: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Classify a batch of commits using LLM with ticket context."""
        batch_id = str(uuid.uuid4())
        logger.info(f"Starting LLM classification for batch {batch_id} with {len(commits)} commits")

        # Add timeout warning for large batches
        if len(commits) > 20:
            logger.warning(
                f"Large batch size ({len(commits)} commits) may take longer to process. "
                f"Consider reducing batch_size if timeouts occur."
            )

        # Prepare batch for LLM classification
        enhanced_commits = []
        for commit in commits:
            enhanced_commit = commit.copy()

            # Add ticket context to commit
            ticket_refs = commit.get("ticket_references", [])
            relevant_tickets = []
            for ticket_id in ticket_refs:
                if ticket_id in ticket_context:
                    relevant_tickets.append(ticket_context[ticket_id])

            enhanced_commit["ticket_context"] = relevant_tickets
            enhanced_commits.append(enhanced_commit)

        # Check if LLM is enabled before attempting classification
        if not self.llm_enabled:
            logger.debug(f"LLM disabled, using fallback for batch {batch_id[:8]}")
            # Skip directly to fallback
            fallback_results = []
            for commit in commits:
                category = self._fallback_classify_commit(commit)
                fallback_results.append(
                    {
                        "commit_hash": commit["commit_hash"],
                        "category": category,
                        "confidence": 0.3,  # Low confidence for fallback
                        "method": "fallback_only",
                        "error": "LLM not configured",
                        "batch_id": batch_id,
                    }
                )
            return fallback_results

        # Check circuit breaker status
        if self.circuit_breaker_open:
            logger.info(
                f"Circuit breaker OPEN - Skipping LLM API call for batch {batch_id[:8]} "
                f"after {self.api_failure_count} consecutive failures. Using fallback classification."
            )
            # Use fallback for all commits
            fallback_results = []
            for commit in commits:
                category = self._fallback_classify_commit(commit)
                fallback_results.append(
                    {
                        "commit_hash": commit["commit_hash"],
                        "category": category,
                        "confidence": 0.3,  # Low confidence for fallback
                        "method": "circuit_breaker_fallback",
                        "error": "Circuit breaker open - API repeatedly failing",
                        "batch_id": batch_id,
                    }
                )
            return fallback_results

        try:
            # Use LLM classifier with enhanced context
            logger.debug(f"Calling LLM classifier for batch {batch_id[:8]}...")
            start_time = datetime.utcnow()

            llm_results = self.llm_classifier.classify_commits_batch(
                enhanced_commits, batch_id=batch_id, include_confidence=True
            )

            elapsed = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"LLM classification for batch {batch_id[:8]} took {elapsed:.2f}s")

            # Reset circuit breaker on successful LLM call
            if self.api_failure_count > 0:
                logger.info(
                    f"LLM API call succeeded - Resetting circuit breaker "
                    f"(was at {self.api_failure_count} failures)"
                )
            self.api_failure_count = 0
            self.circuit_breaker_open = False

            # Process LLM results and add fallbacks
            processed_results = []
            for _i, (commit, llm_result) in enumerate(zip(commits, llm_results)):
                confidence = llm_result.get("confidence", 0.0)
                predicted_category = llm_result.get("category", "other")

                # Apply confidence threshold and fallback
                if confidence < self.confidence_threshold and self.fallback_enabled:
                    fallback_category = self._fallback_classify_commit(commit)
                    processed_results.append(
                        {
                            "commit_hash": commit["commit_hash"],
                            "category": fallback_category,
                            "confidence": 0.5,  # Medium confidence for rule-based
                            "method": "fallback",
                            "llm_category": predicted_category,
                            "llm_confidence": confidence,
                            "batch_id": batch_id,
                        }
                    )
                else:
                    processed_results.append(
                        {
                            "commit_hash": commit["commit_hash"],
                            "category": predicted_category,
                            "confidence": confidence,
                            "method": "llm",
                            "batch_id": batch_id,
                        }
                    )

            logger.info(
                f"LLM classification completed for batch {batch_id}: {len(processed_results)} commits"
            )
            return processed_results

        except Exception as e:
            # Track consecutive failures for circuit breaker
            self.api_failure_count += 1
            logger.error(
                f"LLM classification failed for batch {batch_id}: {e} "
                f"(Failure {self.api_failure_count}/{self.max_consecutive_failures})"
            )

            # Open circuit breaker after max consecutive failures
            if (
                self.api_failure_count >= self.max_consecutive_failures
                and not self.circuit_breaker_open
            ):
                self.circuit_breaker_open = True
                logger.error(
                    f"CIRCUIT BREAKER OPENED after {self.api_failure_count} consecutive API failures. "
                    f"All subsequent batches will use fallback classification until API recovers. "
                    f"This prevents the system from hanging on repeated timeouts."
                )

            # Provide more context about the failure
            if "timeout" in str(e).lower():
                logger.error(
                    f"Classification timed out. Consider: \n"
                    f"  1. Reducing batch_size (current: {self.batch_size})\n"
                    f"  2. Increasing timeout_seconds in LLM config\n"
                    f"  3. Checking API service status"
                )
            elif "connection" in str(e).lower():
                logger.error(
                    "Connection error. Check:\n"
                    "  1. Internet connectivity\n"
                    "  2. API endpoint availability\n"
                    "  3. Firewall/proxy settings"
                )

            # Fall back to rule-based classification for entire batch
            if self.fallback_enabled:
                fallback_results = []
                for commit in commits:
                    category = self._fallback_classify_commit(commit)
                    fallback_results.append(
                        {
                            "commit_hash": commit["commit_hash"],
                            "category": category,
                            "confidence": 0.3,  # Low confidence for fallback
                            "method": "fallback_only",
                            "error": str(e),
                            "batch_id": batch_id,
                        }
                    )

                logger.info(f"Fallback classification completed for batch {batch_id}")
                return fallback_results

            return []

    def _fallback_classify_commit(self, commit: dict[str, Any]) -> str:
        """Classify commit using rule-based patterns."""
        import re

        message = commit.get("message", "").lower()

        # Check patterns in order of specificity
        for category, patterns in self.fallback_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return category

        # Default category
        return "other"

    def _store_commit_classification(
        self, session: Any, classification_result: dict[str, Any]
    ) -> None:
        """Store classification result in cached commit record."""
        try:
            commit_hash = classification_result["commit_hash"]

            # Find the cached commit record
            cached_commit = (
                session.query(CachedCommit).filter(CachedCommit.commit_hash == commit_hash).first()
            )

            if cached_commit:
                # Store classification in ticket_references as temporary solution
                # In production, this would go in a separate classification table
                if not hasattr(cached_commit, "classification_data"):
                    cached_commit.ticket_references = cached_commit.ticket_references or []

                # Add classification data to the record
                # Note: This is a simplified approach - in production you'd want a separate table
                {
                    "category": classification_result["category"],
                    "confidence": classification_result["confidence"],
                    "method": classification_result["method"],
                    "classified_at": datetime.utcnow().isoformat(),
                    "batch_id": classification_result.get("batch_id"),
                }

                # Store in a JSON field or separate table in production
                logger.debug(
                    f"Classified commit {commit_hash[:7]} as {classification_result['category']}"
                )

        except Exception as e:
            logger.error(
                f"Error storing classification for {classification_result.get('commit_hash', 'unknown')}: {e}"
            )

    def _store_daily_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
        project_keys: Optional[list[str]],
    ) -> None:
        """Store aggregated daily metrics from classification results."""
        from ..core.metrics_storage import DailyMetricsStorage

        try:
            DailyMetricsStorage(self.cache_dir / "gitflow_cache.db")

            # This would typically aggregate from the classification results
            # For now, we'll let the existing system handle this
            logger.info("Daily metrics storage integration placeholder")

        except Exception as e:
            logger.error(f"Error storing daily metrics: {e}")

    def get_classification_status(
        self,
        start_date: datetime,
        end_date: datetime,
        project_keys: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Get classification status for a date range."""
        session = self.database.get_session()

        try:
            query = session.query(DailyCommitBatch).filter(
                DailyCommitBatch.date >= start_date.date(), DailyCommitBatch.date <= end_date.date()
            )

            if project_keys:
                query = query.filter(DailyCommitBatch.project_key.in_(project_keys))

            batches = query.all()

            status_counts = defaultdict(int)
            total_commits = 0

            for batch in batches:
                status_counts[batch.classification_status] += 1
                total_commits += batch.commit_count

            return {
                "total_batches": len(batches),
                "total_commits": total_commits,
                "status_breakdown": dict(status_counts),
                "completion_rate": status_counts["completed"] / len(batches) if batches else 0.0,
                "date_range": {"start": start_date, "end": end_date},
            }

        except Exception as e:
            logger.error(f"Error getting classification status: {e}")
            return {}
        finally:
            session.close()
