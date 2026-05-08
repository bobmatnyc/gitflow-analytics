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

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from ..models.database import Database
    from ..qualitative.classifiers.llm_commit_classifier import LLMCommitClassifier

from ..core.progress import get_progress_service
from ..models.database import (
    CachedCommit,
    ClassificationOverride,
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

    # Attributes are initialised by ``BatchCommitClassifier.__init__`` — declared
    # here so Pyright knows about them on ``self`` inside mixin methods.
    database: Database
    batch_size: int
    confidence_threshold: float
    fallback_enabled: bool
    max_processing_time_minutes: int
    classification_start_time: datetime | None
    llm_enabled: bool
    llm_classifier: LLMCommitClassifier
    api_failure_count: int
    max_consecutive_failures: int
    circuit_breaker_open: bool
    fallback_patterns: dict[str, list[str]]
    cache_dir: Path
    # Tier-3 classification signal: JIRA project key → work_type mapping
    # (issue #62). Populated by ``BatchCommitClassifier.__init__``.
    jira_project_mappings: dict[str, str]
    show_jira_signals: bool
    JIRA_PROJECT_KEY_CONFIDENCE: float
    LOW_CONFIDENCE_THRESHOLD: float = 0.30

    def _lookup_classification_overrides(
        self, commits: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Look up manual classification overrides for a batch of commits.

        Why (issue #63): Manual corrections persisted in
        ``classification_overrides`` must take priority over every classifier
        (LLM, JIRA project-key, fallback). This helper performs a single
        bulk SELECT keyed on (commit_hash, repo_path) and returns a map of
        commit_hash -> override result dict ready to be merged into the
        result list with method=``manual_override`` and confidence 1.0.

        Args:
            commits: Batch of commit dicts. Each must expose ``commit_hash``
                and ``repo_path``.

        Returns:
            Map of ``commit_hash`` -> classification result dict. Commits
            without an override are simply absent from the map.
        """
        if not commits:
            return {}

        # Build lookup keys; we filter on the union of hashes and verify the
        # repo_path in Python so a single IN-clause query is sufficient.
        hashes = [c["commit_hash"] for c in commits if c.get("commit_hash")]
        if not hashes:
            return {}
        commit_repo_pairs = {
            (c["commit_hash"], c.get("repo_path", "")) for c in commits if c.get("commit_hash")
        }

        # Defensive: tests sometimes instantiate the mixin via __new__ without
        # calling __init__, leaving ``database`` unset. Skip the lookup in
        # that scenario rather than crashing — those tests verify orthogonal
        # behaviour (e.g. complexity propagation) that does not depend on
        # overrides.
        database = getattr(self, "database", None)
        if database is None:
            return {}
        session = database.get_session()
        try:
            rows = (
                session.query(ClassificationOverride)
                .filter(ClassificationOverride.commit_hash.in_(hashes))
                .all()
            )
            overrides: dict[str, dict[str, Any]] = {}
            for row in rows:
                key = (str(row.commit_hash), str(row.repo_path))
                if key not in commit_repo_pairs:
                    continue
                overrides[str(row.commit_hash)] = {
                    "commit_hash": str(row.commit_hash),
                    "category": str(row.work_type),
                    "confidence": float(row.confidence) if row.confidence is not None else 1.0,
                    "method": "manual_override",
                    "override_reason": row.reason,
                    "override_created_by": row.created_by,
                    "complexity": None,
                }
            return overrides
        except Exception as e:
            logger.warning("Could not look up classification overrides: %s", e)
            return {}
        finally:
            session.close()

    def _classify_via_jira_project_key(self, commit: dict[str, Any]) -> tuple[str, str] | None:
        """Look up commit's JIRA project key in the configured mapping.

        Why: JIRA project keys are a near-perfect predictor of work_type when
        teams have disciplined ticket prefixes (issue #62). For example a
        commit referencing ``ADV-1234`` is almost certainly feature work even
        if the message text contains words like "fix" or "cleanup".

        Args:
            commit: Commit dict — expected to contain ``ticket_references``
                as either a list of dicts (with ``platform`` and ``id`` keys,
                the canonical extractor output) or a list of plain ticket
                identifier strings (e.g. ``"ADV-123"``) for back-compat with
                older cached data.

        Returns:
            Tuple of (work_type, matched_project_key) on first hit, or ``None``
            when no JIRA reference matches a configured project key. The first
            match wins to keep behaviour deterministic.

        Test: with ``jira_project_mappings={"ADV": "feature"}``, a commit whose
        ``ticket_references`` is ``[{"platform": "jira", "id": "ADV-1"}]``
        should return ``("feature", "ADV")``; an unmapped key like
        ``[{"platform": "jira", "id": "ZZZ-1"}]`` should return ``None``.
        """
        mappings = getattr(self, "jira_project_mappings", None)
        if not mappings:
            return None

        ticket_refs = commit.get("ticket_references") or []
        for ref in ticket_refs:
            ticket_id: str | None = None
            platform: str | None = None
            if isinstance(ref, dict):
                # Canonical extractor format
                platform = ref.get("platform")
                ticket_id = ref.get("id") or ref.get("full_id")
            elif isinstance(ref, str):
                # Back-compat: bare ticket id like "ADV-123"
                ticket_id = ref
                platform = "jira"

            if not ticket_id or not isinstance(ticket_id, str):
                continue

            # Only apply to JIRA-style refs. We accept ``platform == "jira"``
            # as well as ``platform`` unset (older caches), since the project
            # key prefix is itself the discriminator.
            if platform and platform.lower() not in ("jira", ""):
                continue

            # Extract the project key — everything before the first "-".
            if "-" not in ticket_id:
                continue
            project_key = ticket_id.split("-", 1)[0].strip().upper()
            if not project_key:
                continue

            work_type = mappings.get(project_key)
            if work_type:
                return work_type, project_key

        return None

    def _merge_jira_with_llm_results(
        self,
        original_commits: list[dict[str, Any]],
        jira_classified: dict[str, dict[str, Any]],
        llm_or_fallback_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge JIRA short-circuited results with LLM/fallback results.

        Why: ``_classify_commit_batch_with_llm`` splits the input batch into
        commits classified via the JIRA project key (tier 3) and commits that
        still need the LLM/fallback pipeline. Downstream code expects the
        return list to align with ``original_commits`` order, so we re-zip
        results back into the original sequence here.

        Args:
            original_commits: The full input batch in caller order.
            jira_classified: Map of commit_hash → tier-3 result for commits
                short-circuited by the JIRA project-key mapping.
            llm_or_fallback_results: Results for the remaining commits, in the
                same order they were passed to the LLM/fallback path.

        Returns:
            One result dict per ``original_commits`` entry, in caller order.
        """
        llm_by_hash = {r["commit_hash"]: r for r in llm_or_fallback_results}
        merged: list[dict[str, Any]] = []
        for commit in original_commits:
            commit_hash = commit["commit_hash"]
            if commit_hash in jira_classified:
                merged.append(jira_classified[commit_hash])
            elif commit_hash in llm_by_hash:
                merged.append(llm_by_hash[commit_hash])
            # If neither path produced a result we skip the commit rather than
            # synthesising a bogus row — the surrounding logic logs the gap.
        return merged

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
                live_batch.classification_status = "processing"  # type: ignore[assignment]

                # Get commits for this day (using live_batch for accurate metadata)
                daily_commits = self._get_commits_for_batch(session, live_batch)
                week_commits.extend(daily_commits)

                # Track which batch each commit belongs to
                for commit in daily_commits:
                    batch_commit_map[commit["commit_hash"]] = live_batch

            if not week_commits:
                expected = sum(b.commit_count for b in live_batches)
                # If every commit already has a high-confidence classification,
                # _get_commits_for_batch intentionally filters them all out —
                # that's a success, not a failure, so mark completed.
                all_already_classified = all(
                    str(b.classification_status) == "completed" for b in live_batches
                )
                if all_already_classified:
                    logger.info(
                        "Skipping weekly batches: all %d commits already "
                        "have high-confidence classifications",
                        expected,
                    )
                    for live_batch in live_batches:
                        live_batch.classified_at = datetime.now(timezone.utc)  # type: ignore[assignment]
                    session.commit()
                    return {"batches_processed": 0, "commits_processed": 0}
                logger.warning(f"No commits found for weekly batches (expected {expected} commits)")
                # Mark batches as failed due to missing commits
                for live_batch in live_batches:
                    live_batch.classification_status = "failed"  # type: ignore[assignment]
                    live_batch.classified_at = datetime.now(timezone.utc)  # type: ignore[assignment]
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
                            datetime.now(timezone.utc) - self.classification_start_time
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
                live_batch.classification_status = "completed"  # type: ignore[assignment]
                live_batch.classified_at = datetime.now(timezone.utc)  # type: ignore[assignment]
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
                    live_batch.classification_status = "failed"  # type: ignore[assignment]
                session.commit()
            except Exception:
                session.rollback()
        finally:
            session.close()

        return {
            "batches_processed": batches_processed,
            "commits_processed": commits_processed,
        }

    def _batch_has_low_confidence_commits(self, session: Any, batch: DailyCommitBatch) -> bool:
        """Return True if the batch contains at least one low-confidence commit.

        Why: ``_get_batches_to_process`` needs to decide whether a
        ``completed`` batch is actually "done" or whether it was finalised
        during an LLM outage with only fallback classifications. A single
        cheap EXISTS-style query keeps the overhead bounded.
        Test: seed one ``QualitativeCommitData`` row with confidence 0.25
        for a commit in the batch's day and assert this returns True; then
        bump it to 0.90 and assert it returns False.
        """
        threshold = getattr(self, "LOW_CONFIDENCE_THRESHOLD", 0.30)
        batch_dt: datetime = batch.date  # type: ignore[assignment]
        batch_day = batch_dt.date()
        start_of_day = datetime.combine(batch_day, datetime.min.time(), tzinfo=timezone.utc)
        end_of_day = datetime.combine(batch_day, datetime.max.time(), tzinfo=timezone.utc)

        row = (
            session.query(QualitativeCommitData.commit_id)
            .join(CachedCommit, CachedCommit.id == QualitativeCommitData.commit_id)
            .filter(
                CachedCommit.repo_path == batch.repo_path,
                CachedCommit.timestamp >= start_of_day,
                CachedCommit.timestamp < end_of_day,
                QualitativeCommitData.confidence_score <= threshold,
            )
            .first()
        )
        return row is not None

    def _get_commits_for_batch(self, session: Any, batch: DailyCommitBatch) -> list[dict[str, Any]]:
        """Get commits for a daily batch that still need (re-)classification.

        Why: When a batch is re-queued because the LLM recovered after an
        outage, we only want to spend LLM tokens on the commits that have
        no classification or a low-confidence fallback classification.
        Commits already scored above the low-confidence threshold keep
        their existing result.
        Test: seed two commits in the same day — one with confidence 0.30
        and one with 0.80 — and assert only the 0.30 commit is returned
        when the batch is re-processed.
        """
        try:
            # Get cached commits for this batch
            # CRITICAL FIX: CachedCommit.timestamp is timezone-aware UTC (from analyzer.py line 806)
            # but we were creating timezone-naive boundaries, causing comparison to fail
            # Create timezone-aware UTC boundaries to match CachedCommit.timestamp format
            batch_dt: datetime = batch.date  # type: ignore[assignment]
            batch_day = batch_dt.date()
            start_of_day = datetime.combine(batch_day, datetime.min.time(), tzinfo=timezone.utc)
            end_of_day = datetime.combine(batch_day, datetime.max.time(), tzinfo=timezone.utc)

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

            # If the batch was completed during an LLM outage we only want
            # to re-classify the low-confidence commits. Look up their
            # existing qualitative rows once and filter in Python to keep
            # the query simple.
            threshold = getattr(self, "LOW_CONFIDENCE_THRESHOLD", 0.30)
            commit_ids = [c.id for c in commits]
            existing_confidence: dict[int, float] = {}
            existing_method: dict[int, str] = {}
            if commit_ids:
                for row in (
                    session.query(
                        QualitativeCommitData.commit_id,
                        QualitativeCommitData.confidence_score,
                        QualitativeCommitData.processing_method,
                    )
                    .filter(QualitativeCommitData.commit_id.in_(commit_ids))
                    .all()
                ):
                    existing_confidence[row[0]] = float(row[1] or 0.0)
                    existing_method[row[0]] = str(row[2] or "")

            # Issue #63: Always include commits that have a manual override on
            # file but whose stored classification has not yet been replaced
            # with method=``manual_override``. This guarantees newly-added
            # overrides are applied even if the existing classification has a
            # high confidence value (e.g. 0.95 from JIRA project key).
            commit_hashes = [c.commit_hash for c in commits]
            override_hashes: set[str] = set()
            if commit_hashes:
                override_rows = (
                    session.query(ClassificationOverride.commit_hash)
                    .filter(
                        ClassificationOverride.commit_hash.in_(commit_hashes),
                        ClassificationOverride.repo_path == batch.repo_path,
                    )
                    .all()
                )
                override_hashes = {str(r[0]) for r in override_rows}

            def _needs_reclassification(c: Any) -> bool:
                # Pending override that hasn't been stamped onto qualitative_commits yet.
                if (
                    c.commit_hash in override_hashes
                    and existing_method.get(c.id) != "manual_override"
                ):
                    return True
                # Original logic: missing or low-confidence existing row.
                return c.id not in existing_confidence or existing_confidence[c.id] <= threshold

            commits = [c for c in commits if _needs_reclassification(c)]

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

        # Manual classification overrides (issue #63) take priority over
        # EVERY classifier. Apply BEFORE the JIRA project-key short-circuit and
        # the LLM call so curated work_type values survive every rerun.
        manual_overrides = self._lookup_classification_overrides(commits)
        if manual_overrides:
            for hash_, override in manual_overrides.items():
                override["batch_id"] = batch_id
                logger.info(
                    "Manual override: commit %s -> work_type=%s",
                    hash_[:7],
                    override.get("category"),
                )
            logger.info(
                "Manual overrides short-circuited %d/%d commits in batch %s",
                len(manual_overrides),
                len(commits),
                batch_id[:8],
            )

        # Commits with overrides bypass every other classifier — even the JIRA
        # project-key signal. Filter them out of the downstream pipeline.
        post_override_commits = [c for c in commits if c["commit_hash"] not in manual_overrides]

        # Tier-3 classification signal: JIRA project key → work_type mapping
        # (issue #62). Apply BEFORE building the LLM batch so commits we can
        # confidently classify by project key never burn an LLM token. We
        # split ``commits`` into:
        #   - ``jira_classified``: commits short-circuited via project key
        #   - ``commits_for_llm``: everything else, sent to the LLM as before
        # The final list returned preserves caller order so downstream code
        # (which zips results back onto the original commits) stays correct.
        jira_classified: dict[str, dict[str, Any]] = {}
        commits_for_llm: list[dict[str, Any]] = []
        for commit in post_override_commits:
            jira_match = self._classify_via_jira_project_key(commit)
            if jira_match is not None:
                work_type, matched_key = jira_match
                if getattr(self, "show_jira_signals", False):
                    logger.info(
                        "JIRA project-key signal: commit %s matched key=%s -> work_type=%s",
                        commit.get("commit_hash", "")[:7],
                        matched_key,
                        work_type,
                    )
                jira_classified[commit["commit_hash"]] = {
                    "commit_hash": commit["commit_hash"],
                    "category": work_type,
                    "confidence": getattr(self, "JIRA_PROJECT_KEY_CONFIDENCE", 0.95),
                    "method": "jira_project_key",
                    "matched_project_key": matched_key,
                    "batch_id": batch_id,
                    # Tier-3 short-circuit path: no LLM complexity rating.
                    "complexity": None,
                }
            else:
                commits_for_llm.append(commit)

        if jira_classified:
            logger.info(
                "JIRA project-key signal short-circuited %d/%d commits in batch %s",
                len(jira_classified),
                len(commits),
                batch_id[:8],
            )

        # Combine short-circuit results (manual overrides + JIRA project key)
        # so the existing merge logic returns them in caller order. Manual
        # overrides win when both keys are present (defensive — should never
        # happen since override commits are filtered out of the JIRA loop).
        short_circuit_results: dict[str, dict[str, Any]] = {
            **jira_classified,
            **manual_overrides,
        }

        # If every commit was classified via a short-circuit path we can
        # return immediately without calling the LLM at all.
        if not commits_for_llm:
            return [
                short_circuit_results[c["commit_hash"]]
                for c in commits
                if c["commit_hash"] in short_circuit_results
            ]

        # Prepare batch for LLM classification (only commits that didn't hit
        # the JIRA short-circuit path).
        enhanced_commits = []
        for commit in commits_for_llm:
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
            # Skip directly to fallback. JIRA-classified commits keep their
            # high-confidence project-key result; the rest go through the
            # rule-based fallback as before.
            return self._merge_jira_with_llm_results(
                commits,
                short_circuit_results,
                [
                    {
                        "commit_hash": commit["commit_hash"],
                        "category": self._fallback_classify_commit(commit),
                        "confidence": 0.3,  # Low confidence for fallback
                        "method": "fallback_only",
                        "error": "LLM not configured",
                        "batch_id": batch_id,
                        # Rule-based path: no complexity rating
                        "complexity": None,
                    }
                    for commit in commits_for_llm
                ],
            )

        # Check circuit breaker status
        if self.circuit_breaker_open:
            logger.info(
                f"Circuit breaker OPEN - Skipping LLM API call for batch {batch_id[:8]} "
                f"after {self.api_failure_count} consecutive failures. Using fallback classification."
            )
            # Use fallback for non-JIRA commits; JIRA-classified commits keep
            # their high-confidence project-key result.
            return self._merge_jira_with_llm_results(
                commits,
                short_circuit_results,
                [
                    {
                        "commit_hash": commit["commit_hash"],
                        "category": self._fallback_classify_commit(commit),
                        "confidence": 0.3,  # Low confidence for fallback
                        "method": "circuit_breaker_fallback",
                        "error": "Circuit breaker open - API repeatedly failing",
                        "batch_id": batch_id,
                        # Rule-based path: no complexity rating
                        "complexity": None,
                    }
                    for commit in commits_for_llm
                ],
            )

        try:
            # Use LLM classifier with enhanced context
            logger.debug(f"Calling LLM classifier for batch {batch_id[:8]}...")
            start_time = datetime.now(timezone.utc)

            llm_results = self.llm_classifier.classify_commits_batch(
                enhanced_commits, batch_id=batch_id, include_confidence=True
            )

            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"LLM classification for batch {batch_id[:8]} took {elapsed:.2f}s")

            # Reset circuit breaker on successful LLM call
            if self.api_failure_count > 0:
                logger.info(
                    f"LLM API call succeeded - Resetting circuit breaker "
                    f"(was at {self.api_failure_count} failures)"
                )
            self.api_failure_count = 0
            self.circuit_breaker_open = False

            # Process LLM results and add fallbacks. ``llm_results`` aligns
            # with ``commits_for_llm`` (NOT the original ``commits``), since we
            # only sent unmapped commits to the LLM.
            processed_llm_results: list[dict[str, Any]] = []
            for _, (commit, llm_result) in enumerate(zip(commits_for_llm, llm_results)):
                confidence = llm_result.get("confidence", 0.0)
                predicted_category = llm_result.get("category", "other")

                # Apply confidence threshold and fallback
                if confidence < self.confidence_threshold and self.fallback_enabled:
                    fallback_category = self._fallback_classify_commit(commit)
                    processed_llm_results.append(
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
                    processed_llm_results.append(
                        {
                            "commit_hash": commit["commit_hash"],
                            "category": predicted_category,
                            "confidence": confidence,
                            "method": "llm",
                            "batch_id": batch_id,
                            "complexity": llm_result.get("complexity"),
                        }
                    )

            merged = self._merge_jira_with_llm_results(
                commits, short_circuit_results, processed_llm_results
            )
            logger.info(f"LLM classification completed for batch {batch_id}: {len(merged)} commits")
            return merged

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

            # Fall back to rule-based classification for entire batch.
            # JIRA-classified commits still keep their tier-3 result; only the
            # LLM-bound commits drop to fallback patterns.
            if self.fallback_enabled:
                fallback_only = [
                    {
                        "commit_hash": commit["commit_hash"],
                        "category": self._fallback_classify_commit(commit),
                        "confidence": 0.3,  # Low confidence for fallback
                        "method": "fallback_only",
                        "error": str(e),
                        "batch_id": batch_id,
                        # Rule-based path: no complexity rating
                        "complexity": None,
                    }
                    for commit in commits_for_llm
                ]
                merged = self._merge_jira_with_llm_results(
                    commits, short_circuit_results, fallback_only
                )
                logger.info(f"Fallback classification completed for batch {batch_id}")
                return merged

            # No fallback available — still return short-circuited results
            # (manual overrides + JIRA project key) rather than dropping them.
            if short_circuit_results:
                return [
                    short_circuit_results[c["commit_hash"]]
                    for c in commits
                    if c["commit_hash"] in short_circuit_results
                ]
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
        _start_date: datetime,
        _end_date: datetime,
        _project_keys: list[str] | None,
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
        project_keys: list[str] | None = None,
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
