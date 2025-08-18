"""Batch processing logic for efficient LLM classification.

This module handles batch processing of commits for classification,
including progress tracking, error handling, and result aggregation.

WHY: Processing commits in batches improves efficiency, enables better
progress tracking, and allows for optimizations like parallel processing.

DESIGN DECISIONS:
- Support configurable batch sizes
- Provide detailed progress feedback
- Handle failures gracefully without losing progress
- Support resume from partial completion
- Track batch-level metrics
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ....core.progress import get_progress_service

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    WHY: Centralizing batch configuration makes it easy to tune
    performance characteristics for different scenarios.
    """

    batch_size: int = 50  # Number of commits per batch
    max_parallel_batches: int = 1  # Currently serial, but structured for future parallel support
    retry_failed_batches: bool = True
    continue_on_batch_failure: bool = True
    show_progress: bool = True
    progress_nested: bool = True  # Show nested progress bars

    def validate(self) -> None:
        """Validate batch configuration."""
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.max_parallel_batches < 1:
            raise ValueError(
                f"Max parallel batches must be positive, got {self.max_parallel_batches}"
            )


@dataclass
class BatchResult:
    """Result of processing a single batch.

    WHY: Structured batch results enable better error handling
    and performance analysis.
    """

    batch_id: str
    total_items: int
    successful_items: int
    failed_items: int
    results: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    processing_time_seconds: float
    retry_count: int = 0


class BatchProcessor:
    """Processes commits in batches for LLM classification.

    WHY: Batch processing improves efficiency and provides better
    user feedback for large-scale classification tasks.
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize batch processor.

        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        self.config.validate()

        # Processing statistics
        self.total_processed = 0
        self.total_successful = 0
        self.total_failed = 0
        self.batch_results: list[BatchResult] = []

    def process_commits(
        self,
        commits: list[dict[str, Any]],
        classifier_func: Callable[[dict[str, Any]], dict[str, Any]],
        job_description: str = "Processing commits",
    ) -> list[dict[str, Any]]:
        """Process commits in batches using the provided classifier.

        Args:
            commits: List of commits to process
            classifier_func: Function to classify a single commit
            job_description: Description for progress tracking

        Returns:
            List of classification results for all commits
        """
        if not commits:
            return []

        # Split into batches
        batches = self._create_batches(commits)
        logger.info(f"Processing {len(commits)} commits in {len(batches)} batches")

        # Get progress service
        progress = get_progress_service()

        all_results = []

        # Process batches
        if self.config.show_progress:
            with progress.progress(
                total=len(batches), description=job_description, unit="batch", leave=True
            ) as batch_ctx:
                for i, batch in enumerate(batches, 1):
                    batch_id = self._generate_batch_id(batch, i)
                    progress.set_description(
                        batch_ctx, f"{job_description} (batch {i}/{len(batches)})"
                    )

                    # Process single batch
                    batch_result = self._process_single_batch(
                        batch, batch_id, classifier_func, progress
                    )

                    # Collect results
                    all_results.extend(batch_result.results)
                    self.batch_results.append(batch_result)

                    # Update progress
                    progress.update(batch_ctx, 1)

                    # Update statistics
                    self.total_processed += batch_result.total_items
                    self.total_successful += batch_result.successful_items
                    self.total_failed += batch_result.failed_items

                    # Log batch summary
                    if batch_result.failed_items > 0:
                        logger.warning(
                            f"Batch {batch_id}: {batch_result.failed_items}/{batch_result.total_items} failed"
                        )
        else:
            # Process without progress bars
            for i, batch in enumerate(batches, 1):
                batch_id = self._generate_batch_id(batch, i)
                batch_result = self._process_single_batch(batch, batch_id, classifier_func, None)
                all_results.extend(batch_result.results)
                self.batch_results.append(batch_result)

                self.total_processed += batch_result.total_items
                self.total_successful += batch_result.successful_items
                self.total_failed += batch_result.failed_items

        # Log final summary
        logger.info(
            f"Batch processing complete: {self.total_successful}/{self.total_processed} successful"
        )

        return all_results

    def _create_batches(self, commits: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Split commits into batches.

        Args:
            commits: List of commits to batch

        Returns:
            List of commit batches
        """
        batches = []
        for i in range(0, len(commits), self.config.batch_size):
            batch = commits[i : i + self.config.batch_size]
            batches.append(batch)
        return batches

    def _generate_batch_id(self, batch: list[dict[str, Any]], batch_num: int) -> str:
        """Generate a unique ID for a batch.

        Args:
            batch: Batch of commits
            batch_num: Batch number

        Returns:
            Unique batch ID
        """
        # Create hash from first and last commit in batch
        if batch:
            first_msg = batch[0].get("message", "")
            last_msg = batch[-1].get("message", "")
            content = f"{batch_num}:{first_msg}:{last_msg}"
            return hashlib.md5(content.encode()).hexdigest()[:8]
        return f"batch_{batch_num}"

    def _process_single_batch(
        self,
        batch: list[dict[str, Any]],
        batch_id: str,
        classifier_func: Callable[[dict[str, Any]], dict[str, Any]],
        progress: Optional[Any],
    ) -> BatchResult:
        """Process a single batch of commits.

        Args:
            batch: Batch of commits to process
            batch_id: Unique batch identifier
            classifier_func: Classification function
            progress: Progress tracking service

        Returns:
            BatchResult with processing outcomes
        """
        start_time = time.time()
        results = []
        errors = []

        # Show nested progress for individual commits if configured
        if self.config.show_progress and self.config.progress_nested and progress:
            with progress.progress(
                total=len(batch),
                description=f"Batch {batch_id[:8]}",
                unit="commit",
                nested=True,
                leave=False,
            ) as commit_ctx:
                for j, commit in enumerate(batch, 1):
                    # Update progress description
                    message_preview = commit.get("message", "")[:30]
                    progress.set_description(
                        commit_ctx, f"Batch {batch_id[:8]} ({j}/{len(batch)}): {message_preview}..."
                    )

                    # Process commit
                    result, error = self._process_single_commit(commit, batch_id, classifier_func)

                    if result:
                        results.append(result)
                    if error:
                        errors.append(error)

                    # Update progress
                    progress.update(commit_ctx, 1)
        else:
            # Process without nested progress
            for commit in batch:
                result, error = self._process_single_commit(commit, batch_id, classifier_func)

                if result:
                    results.append(result)
                if error:
                    errors.append(error)

        # Create batch result
        processing_time = time.time() - start_time

        return BatchResult(
            batch_id=batch_id,
            total_items=len(batch),
            successful_items=len(results),
            failed_items=len(errors),
            results=results,
            errors=errors,
            processing_time_seconds=processing_time,
        )

    def _process_single_commit(
        self,
        commit: dict[str, Any],
        batch_id: str,
        classifier_func: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """Process a single commit within a batch.

        Args:
            commit: Commit to process
            batch_id: Batch identifier
            classifier_func: Classification function

        Returns:
            Tuple of (result, error) where one will be None
        """
        try:
            # Call classifier function
            result = classifier_func(commit)

            # Add batch ID to result
            result["batch_id"] = batch_id

            # Add original commit data if not present
            if "commit_hash" not in result and "hash" in commit:
                result["commit_hash"] = commit["hash"]
            if "author" not in result and "author" in commit:
                result["author"] = commit["author"]

            return result, None

        except Exception as e:
            logger.debug(f"Failed to classify commit: {e}")

            # Create error record
            error = {
                "batch_id": batch_id,
                "commit_hash": commit.get("hash", "unknown"),
                "message": commit.get("message", "")[:100],
                "error": str(e),
                "error_type": type(e).__name__,
            }

            # Return fallback result if configured to continue
            if self.config.continue_on_batch_failure:
                fallback_result = {
                    "category": "maintenance",
                    "confidence": 0.1,
                    "method": "error_fallback",
                    "reasoning": f"Classification failed: {str(e)}",
                    "batch_id": batch_id,
                    "commit_hash": commit.get("hash", "unknown"),
                    "error": True,
                }
                return fallback_result, error

            return None, error

    def get_statistics(self) -> dict[str, Any]:
        """Get batch processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        if not self.batch_results:
            return {
                "total_batches": 0,
                "total_processed": 0,
                "total_successful": 0,
                "total_failed": 0,
                "success_rate": 0.0,
                "average_batch_time": 0.0,
            }

        total_time = sum(br.processing_time_seconds for br in self.batch_results)

        return {
            "total_batches": len(self.batch_results),
            "total_processed": self.total_processed,
            "total_successful": self.total_successful,
            "total_failed": self.total_failed,
            "success_rate": (
                self.total_successful / self.total_processed if self.total_processed > 0 else 0.0
            ),
            "average_batch_time": total_time / len(self.batch_results),
            "total_processing_time": total_time,
            "batch_size": self.config.batch_size,
            "batches_with_errors": sum(1 for br in self.batch_results if br.failed_items > 0),
        }

    def get_failed_commits(self) -> list[dict[str, Any]]:
        """Get list of all failed commits.

        Returns:
            List of error records for failed commits
        """
        failed = []
        for batch_result in self.batch_results:
            failed.extend(batch_result.errors)
        return failed

    def reset_statistics(self) -> None:
        """Reset all processing statistics."""
        self.total_processed = 0
        self.total_successful = 0
        self.total_failed = 0
        self.batch_results = []
