"""Batch processing utilities for efficient commit analysis."""

import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class BatchProcessor:
    """Efficient batch processing for commit analysis.

    This class provides utilities for processing large numbers of commits
    in batches with parallel execution, progress tracking, and error handling.
    """

    def __init__(self, batch_size: int = 1000, max_workers: int = 4):
        """Initialize batch processor.

        Args:
            batch_size: Number of items to process per batch
            max_workers: Maximum number of worker threads
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        self._stats_lock = Lock()
        self._processing_stats = {
            "total_processed": 0,
            "total_errors": 0,
            "batch_times": [],
            "start_time": None,
        }

    def create_batches(self, items: list[T], batch_size: Optional[int] = None) -> Iterator[list[T]]:
        """Split items into batches for processing.

        Args:
            items: List of items to batch
            batch_size: Override default batch size

        Yields:
            Batches of items
        """
        batch_size = batch_size or self.batch_size

        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def process_batches(
        self, items: list[T], processor_func: Callable[[list[T]], list[R]], parallel: bool = True
    ) -> list[R]:
        """Process items in batches with optional parallelization.

        Args:
            items: Items to process
            processor_func: Function that processes a batch and returns results
            parallel: Whether to use parallel processing

        Returns:
            List of all processing results
        """
        if not items:
            return []

        self._reset_stats()
        self._processing_stats["start_time"] = time.time()

        batches = list(self.create_batches(items))
        self.logger.info(f"Processing {len(items)} items in {len(batches)} batches")

        all_results = []

        if parallel and len(batches) > 1:
            all_results = self._process_parallel(batches, processor_func)
        else:
            all_results = self._process_sequential(batches, processor_func)

        self._log_final_stats(len(items))
        return all_results

    def process_with_callback(
        self,
        items: list[T],
        processor_func: Callable[[list[T]], list[R]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[R]:
        """Process batches with progress callback.

        Args:
            items: Items to process
            processor_func: Function that processes a batch
            progress_callback: Callback for progress updates (processed, total)

        Returns:
            List of all processing results
        """
        if not items:
            return []

        self._reset_stats()
        batches = list(self.create_batches(items))
        all_results = []
        processed_count = 0

        for i, batch in enumerate(batches):
            batch_start = time.time()

            try:
                batch_results = processor_func(batch)
                all_results.extend(batch_results)
                processed_count += len(batch)

                with self._stats_lock:
                    self._processing_stats["total_processed"] += len(batch)
                    self._processing_stats["batch_times"].append(time.time() - batch_start)

            except Exception as e:
                self.logger.error(f"Error processing batch {i}: {e}")
                with self._stats_lock:
                    self._processing_stats["total_errors"] += len(batch)

            # Call progress callback if provided
            if progress_callback:
                progress_callback(processed_count, len(items))

        return all_results

    def _process_parallel(
        self, batches: list[list[T]], processor_func: Callable[[list[T]], list[R]]
    ) -> list[R]:
        """Process batches in parallel using ThreadPoolExecutor.

        Args:
            batches: List of batches to process
            processor_func: Function to process each batch

        Returns:
            Combined results from all batches
        """
        all_results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_batch_with_timing, batch, processor_func): i
                for i, batch in enumerate(batches)
            }

            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]

                try:
                    batch_results, batch_time = future.result()
                    all_results.extend(batch_results)

                    with self._stats_lock:
                        self._processing_stats["total_processed"] += len(batches[batch_idx])
                        self._processing_stats["batch_times"].append(batch_time)

                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_idx}: {e}")
                    with self._stats_lock:
                        self._processing_stats["total_errors"] += len(batches[batch_idx])

        return all_results

    def _process_sequential(
        self, batches: list[list[T]], processor_func: Callable[[list[T]], list[R]]
    ) -> list[R]:
        """Process batches sequentially.

        Args:
            batches: List of batches to process
            processor_func: Function to process each batch

        Returns:
            Combined results from all batches
        """
        all_results = []

        for i, batch in enumerate(batches):
            try:
                batch_results, batch_time = self._process_batch_with_timing(batch, processor_func)
                all_results.extend(batch_results)

                self._processing_stats["total_processed"] += len(batch)
                self._processing_stats["batch_times"].append(batch_time)

            except Exception as e:
                self.logger.error(f"Error processing batch {i}: {e}")
                self._processing_stats["total_errors"] += len(batch)

        return all_results

    def _process_batch_with_timing(
        self, batch: list[T], processor_func: Callable[[list[T]], list[R]]
    ) -> tuple[list[R], float]:
        """Process a single batch with timing.

        Args:
            batch: Batch to process
            processor_func: Processing function

        Returns:
            Tuple of (results, processing_time_seconds)
        """
        start_time = time.time()
        results = processor_func(batch)
        processing_time = time.time() - start_time

        return results, processing_time

    def _reset_stats(self) -> None:
        """Reset processing statistics."""
        with self._stats_lock:
            self._processing_stats = {
                "total_processed": 0,
                "total_errors": 0,
                "batch_times": [],
                "start_time": time.time(),
            }

    def _log_final_stats(self, total_items: int) -> None:
        """Log final processing statistics.

        Args:
            total_items: Total number of items processed
        """
        with self._stats_lock:
            stats = self._processing_stats.copy()

        if not stats["batch_times"]:
            return

        total_time = time.time() - stats["start_time"]
        avg_batch_time = sum(stats["batch_times"]) / len(stats["batch_times"])
        items_per_second = stats["total_processed"] / total_time if total_time > 0 else 0

        self.logger.info(
            f"Batch processing complete: {stats['total_processed']}/{total_items} items processed "
            f"in {total_time:.2f}s ({items_per_second:.1f} items/s), "
            f"{stats['total_errors']} errors, avg batch time: {avg_batch_time:.2f}s"
        )

    def get_processing_stats(self) -> dict[str, Any]:
        """Get current processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        with self._stats_lock:
            stats = self._processing_stats.copy()

        if stats["start_time"] and stats["batch_times"]:
            elapsed_time = time.time() - stats["start_time"]
            avg_batch_time = sum(stats["batch_times"]) / len(stats["batch_times"])
            items_per_second = stats["total_processed"] / elapsed_time if elapsed_time > 0 else 0

            return {
                "total_processed": stats["total_processed"],
                "total_errors": stats["total_errors"],
                "elapsed_time_seconds": elapsed_time,
                "avg_batch_time_seconds": avg_batch_time,
                "items_per_second": items_per_second,
                "batches_completed": len(stats["batch_times"]),
                "error_rate": (
                    stats["total_errors"] / (stats["total_processed"] + stats["total_errors"])
                    if (stats["total_processed"] + stats["total_errors"]) > 0
                    else 0.0
                ),
            }
        else:
            return {
                "total_processed": 0,
                "total_errors": 0,
                "elapsed_time_seconds": 0,
                "avg_batch_time_seconds": 0,
                "items_per_second": 0,
                "batches_completed": 0,
                "error_rate": 0.0,
            }


class ProgressTracker:
    """Simple progress tracking for long-running operations."""

    def __init__(self, total: int, description: str = "Processing"):
        """Initialize progress tracker.

        Args:
            total: Total number of items to process
            description: Description of the operation
        """
        self.total = total
        self.description = description
        self.processed = 0
        self.start_time = time.time()
        self.last_report = 0
        self.logger = logging.getLogger(__name__)

    def update(self, count: int = 1) -> None:
        """Update progress count.

        Args:
            count: Number of items processed since last update
        """
        self.processed += count

        # Report progress every 10% or every 1000 items, whichever is less frequent
        report_interval = max(self.total // 10, 1000)

        if self.processed - self.last_report >= report_interval or self.processed >= self.total:
            self._report_progress()
            self.last_report = self.processed

    def _report_progress(self) -> None:
        """Report current progress."""
        elapsed_time = time.time() - self.start_time
        percentage = (self.processed / self.total) * 100 if self.total > 0 else 0
        rate = self.processed / elapsed_time if elapsed_time > 0 else 0

        # Estimate time remaining
        if rate > 0 and self.processed < self.total:
            remaining_items = self.total - self.processed
            eta_seconds = remaining_items / rate
            eta_str = f", ETA: {eta_seconds:.0f}s"
        else:
            eta_str = ""

        self.logger.info(
            f"{self.description}: {self.processed}/{self.total} ({percentage:.1f}%) "
            f"at {rate:.1f} items/s{eta_str}"
        )
