"""Batch classifier implementation details: DB access, LLM calling, and storage.

This module provides BatchClassifierImplMixin, which adds the heavy implementation
methods to BatchCommitClassifier via multiple inheritance.

Methods here depend on instance attributes set by BatchCommitClassifier.__init__:
    self.database, self.batch_size, self.confidence_threshold,
    self.fallback_enabled, self.max_processing_time_minutes,
    self.classification_start_time, self.llm_enabled, self.llm_classifier,
    self.api_failure_count, self.max_consecutive_failures,
    self.circuit_breaker_open, self.fallback_patterns, self.cache_dir
"""

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from ..core.progress import get_progress_service
from ..models.database import (
    CachedCommit,
    DailyCommitBatch,
    DetailedTicketData,
    QualitativeCommitData,
)

logger = logging.getLogger(__name__)


class BatchClassifierImplMixin:
    """Mixin adding implementation methods to BatchCommitClassifier.

    Extracted to reduce batch_classifier.py below 800 lines while keeping
    the public API in one place.
    """

    def _classify_weekly_batches(self, weekly_batches: list[DailyCommitBatch]) -> dict[str, Any]:
        """Classify all batches for a single week with shared context.

        Bug B fix: ``weekly_batches`` are ORM objects loaded by
        ``_get_batches_to_process()`` in a *closed* session.  They are
        detached, so mutating them and calling ``session.commit()`` here
        would silently do nothing.  We therefore re-fetch fresh ORM objects
        using the IDs of the detached instances so that all mutations
        (classification_status, classified_at) are tracked by the active
        session and actually written to the database.
        """
        # Collect the primary-key IDs from the detached objects so we can
        # re-fetch them inside the new session below.
        batch_ids = [batch.id for batch in weekly_batches]

        session = self.database.get_session()
        batches_processed = 0
        commits_processed = 0

        try:
            # Re-fetch fresh (attached) ORM objects for this session so that
            # status updates are actually persisted on commit.
            live_batches: list[DailyCommitBatch] = (
                session.query(DailyCommitBatch)
                .filter(DailyCommitBatch.id.in_(batch_ids))
                .order_by(DailyCommitBatch.date)
                .all()
            )

            # Build a lookup so we can correlate detached metadata with live rows.
            live_by_id = {b.id: b for b in live_batches}

            # Collect all commits for the week
            week_commits = []
            batch_commit_map = {}  # Maps commit hash to live batch

            for detached_batch in weekly_batches:
                live_batch = live_by_id.get(detached_batch.id)
                if live_batch is None:
                    logger.warning(f"Batch id={detached_batch.id} not found in re-fetch; skipping")
                    continue

                # Mark batch as processing (on the live, attached object)
                live_batch.classification_status = "processing"

                # Get commits for this day (using live_batch for accurate metadata)
                daily_commits = self._get_commits_for_batch(session, live_batch)
                week_commits.extend(daily_commits)

                # Track which batch each commit belongs to
                for commit in daily_commits:
                    batch_commit_map[commit["commit_hash"]] = live_batch

            if not week_commits:
                logger.warning(
                    f"No commits found for weekly batches (expected {sum(b.commit_count for b in live_batches)} commits)"
                )
                # Mark batches as failed due to missing commits
                for live_batch in live_batches:
                    live_batch.classification_status = "failed"
                    live_batch.classified_at = datetime.now(timezone.utc)
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
                                        # Timeout fallback: no complexity rating
                                        "complexity": None,
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

            # Mark all daily batches as completed (using live, attached objects)
            for live_batch in live_batches:
                live_batch.classification_status = "completed"
                live_batch.classified_at = datetime.now(timezone.utc)
                batches_processed += 1

            session.commit()

            logger.info(
                f"Week classification completed: {batches_processed} batches, {commits_processed} commits"
            )

        except Exception as e:
            logger.error(f"Error in weekly batch classification: {e}")
            # Mark batches as failed.  ``live_batches`` may be unbound if the
            # re-fetch query itself raised, so guard with a try/except.
            try:
                for live_batch in live_batches:  # type: ignore[possibly-undefined]
                    live_batch.classification_status = "failed"
                session.commit()
            except Exception:
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
                        # Rule-based path: no complexity rating
                        "complexity": None,
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
                        # Rule-based path: no complexity rating
                        "complexity": None,
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
                            # Fallback path does not produce a complexity rating
                            "complexity": None,
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
                            "complexity": llm_result.get("complexity"),
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
                            # Rule-based path: no complexity rating
                            "complexity": None,
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
        """Store classification result in the qualitative_commits table.

        WHY: QualitativeCommitData uses commit_id (FK to cached_commits.id) as
        its primary key.  We look up the CachedCommit row first, then either
        INSERT a new QualitativeCommitData row or UPDATE the existing one so
        that re-classification runs are idempotent.
        """
        try:
            commit_hash = classification_result["commit_hash"]

            # Find the cached commit record
            cached_commit = (
                session.query(CachedCommit).filter(CachedCommit.commit_hash == commit_hash).first()
            )

            if not cached_commit:
                logger.warning(
                    f"Cannot store classification for {commit_hash[:7]}: commit not found in cache"
                )
                return

            # Map batch-classifier category names to QualitativeCommitData fields.
            # QualitativeCommitData.change_type stores the canonical category string.
            category = classification_result["category"]
            confidence = float(classification_result["confidence"])
            method = classification_result.get("method", "unknown")

            # Upsert: update existing row or create a new one.
            qualitative = (
                session.query(QualitativeCommitData)
                .filter(QualitativeCommitData.commit_id == cached_commit.id)
                .first()
            )

            if qualitative:
                # Update existing classification record
                qualitative.change_type = category
                qualitative.change_type_confidence = confidence
                qualitative.confidence_score = confidence
                qualitative.processing_method = method
                qualitative.analyzed_at = datetime.now(timezone.utc)
                qualitative.complexity = classification_result.get("complexity")
            else:
                # Create new classification record
                qualitative = QualitativeCommitData(
                    commit_id=cached_commit.id,
                    change_type=category,
                    change_type_confidence=confidence,
                    # business_domain and domain_confidence are required NOT NULL —
                    # use sensible defaults when the batch classifier does not provide them.
                    business_domain="unknown",
                    domain_confidence=0.0,
                    risk_level="low",
                    risk_factors=[],
                    intent_signals={"batch_id": classification_result.get("batch_id")},
                    collaboration_patterns={},
                    technical_context={
                        "llm_category": classification_result.get("llm_category"),
                        "llm_confidence": classification_result.get("llm_confidence"),
                        "error": classification_result.get("error"),
                    },
                    processing_method=method,
                    processing_time_ms=0.0,
                    confidence_score=confidence,
                    complexity=classification_result.get("complexity"),
                )
                session.add(qualitative)

            logger.debug(
                f"Stored classification for commit {commit_hash[:7]}: "
                f"category={category}, confidence={confidence:.2f}, method={method}"
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
