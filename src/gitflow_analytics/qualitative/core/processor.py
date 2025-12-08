"""Main orchestrator for qualitative analysis of Git commits."""

import logging
import time
from pathlib import Path
from typing import Any, Optional

from ...core.schema_version import create_schema_manager
from ...models.database import Database
from ..models.schemas import QualitativeCommitData, QualitativeConfig
from ..utils.batch_processor import BatchProcessor, ProgressTracker
from ..utils.metrics import PerformanceMetrics
from .llm_fallback import LLMFallback
from .nlp_engine import NLPEngine
from .pattern_cache import PatternCache


class QualitativeProcessor:
    """Main orchestrator for qualitative analysis of Git commits.

    This processor coordinates the entire qualitative analysis pipeline:
    1. Pattern cache lookup for known commit patterns
    2. Fast NLP processing for most commits
    3. Strategic LLM fallback for uncertain cases
    4. Pattern learning and cache updates
    5. Performance monitoring and optimization

    The system is designed to process 10,000+ commits in under 60 seconds
    while maintaining high accuracy and keeping LLM costs low.
    """

    def __init__(
        self, config: QualitativeConfig, database: Database, cache_dir: Optional[Path] = None
    ):
        """Initialize qualitative processor.

        Args:
            config: Configuration for qualitative analysis
            database: Database instance for caching and storage
            cache_dir: Cache directory for schema versioning
        """
        self.config = config
        self.database = database
        self.logger = logging.getLogger(__name__)

        # Initialize schema version manager
        if cache_dir is None:
            cache_dir = Path(config.cache_config.cache_dir)
        self.schema_manager = create_schema_manager(cache_dir)

        # Initialize core components
        try:
            self.nlp_engine = NLPEngine(config.nlp_config)
        except OSError as e:
            self.logger.warning(
                f"Failed to initialize NLP engine: {e}. "
                f"Qualitative analysis will be disabled for this session."
            )
            self.nlp_engine = None

        self.pattern_cache = PatternCache(config.cache_config, database)

        # Initialize LLM fallback if enabled
        self.llm_fallback = None
        if config.llm_config.openrouter_api_key:
            try:
                self.llm_fallback = LLMFallback(config.llm_config, cache_dir)
                self.logger.info("LLM fallback system initialized")
            except Exception as e:
                self.logger.warning(f"LLM fallback initialization failed: {e}")
        else:
            self.logger.info("LLM fallback disabled (no API key configured)")

        # Initialize utilities
        self.batch_processor = BatchProcessor(
            batch_size=config.batch_size, max_workers=config.nlp_config.max_workers
        )
        self.metrics = PerformanceMetrics()

        # Processing statistics
        self.processing_stats = {
            "total_processed": 0,
            "cache_hits": 0,
            "nlp_processed": 0,
            "llm_processed": 0,
            "processing_start_time": None,
            "last_optimization": None,
        }

        self.logger.info("Qualitative processor initialized")

    def _filter_commits_for_processing(
        self, commits: list[dict[str, Any]], force_reprocess: bool = False
    ) -> list[dict[str, Any]]:
        """Filter commits to only those that need processing based on schema versioning."""
        if force_reprocess:
            return commits

        # Convert config to dict for schema comparison
        config_dict = {
            "nlp_config": self.config.nlp_config.__dict__,
            "llm_config": self.config.llm_config.__dict__,
            "cache_config": self.config.cache_config.__dict__,
            "confidence_threshold": self.config.confidence_threshold,
            "max_llm_fallback_pct": self.config.max_llm_fallback_pct,
        }

        # Check if schema has changed
        schema_changed = self.schema_manager.has_schema_changed("qualitative", config_dict)

        if schema_changed:
            self.logger.info("Qualitative analysis schema has changed, reprocessing all commits")
            return commits

        # Filter by date - only process commits after last processed date
        last_processed = self.schema_manager.get_last_processed_date("qualitative")
        if not last_processed:
            self.logger.info("No previous processing date found, processing all commits")
            return commits

        # Filter commits by date
        commits_to_process = []
        for commit in commits:
            commit_date = commit.get("timestamp")
            if commit_date and commit_date > last_processed:
                commits_to_process.append(commit)

        return commits_to_process

    def _get_existing_results(self, commits: list[dict[str, Any]]) -> list[QualitativeCommitData]:
        """Get existing qualitative results for commits from the database."""
        results = []

        # Try to load existing results from database
        # This is a simplified version - in practice you'd query the qualitative_commits table
        for commit in commits:
            # Create minimal result indicating no processing needed
            result = QualitativeCommitData(
                hash=commit.get("hash", ""),
                message=commit.get("message", ""),
                author_name=commit.get("author_name", ""),
                author_email=commit.get("author_email", ""),
                timestamp=commit.get("timestamp"),
                files_changed=commit.get("files_changed", []),
                insertions=commit.get("insertions", 0),
                deletions=commit.get("deletions", 0),
                change_type="unknown",
                change_type_confidence=0.0,
                business_domain="unknown",
                domain_confidence=0.0,
                risk_level="low",
                risk_factors=[],
                intent_signals={},
                collaboration_patterns={},
                technical_context={},
                processing_method="cached",
                processing_time_ms=0.0,
                confidence_score=0.0,
            )
            results.append(result)

        return results

    def _update_schema_tracking(self, commits: list[dict[str, Any]]):
        """Update schema version tracking after processing commits."""
        if not commits:
            return

        # Convert config to dict for schema tracking
        config_dict = {
            "nlp_config": self.config.nlp_config.__dict__,
            "llm_config": self.config.llm_config.__dict__,
            "cache_config": self.config.cache_config.__dict__,
            "confidence_threshold": self.config.confidence_threshold,
            "max_llm_fallback_pct": self.config.max_llm_fallback_pct,
        }

        # Find the latest commit date
        latest_date = max(commit.get("timestamp") for commit in commits if commit.get("timestamp"))

        # Update schema version with latest processed date
        self.schema_manager.update_schema_version("qualitative", config_dict, latest_date)
        self.schema_manager.mark_date_processed("qualitative", latest_date, config_dict)

    def process_commits(
        self,
        commits: list[dict[str, Any]],
        show_progress: bool = True,
        force_reprocess: bool = False,
    ) -> list[QualitativeCommitData]:
        """Process commits with qualitative analysis using incremental processing.

        Args:
            commits: List of commit dictionaries from GitFlow Analytics
            show_progress: Whether to show progress information
            force_reprocess: Force reprocessing even if schema hasn't changed

        Returns:
            List of QualitativeCommitData with analysis results
        """
        if not commits:
            return []

        if not self.config.enabled:
            self.logger.info("Qualitative analysis disabled in configuration")
            return self._create_disabled_results(commits)

        if self.nlp_engine is None:
            self.logger.warning(
                "Qualitative analysis skipped: NLP engine not available (spaCy model not installed)"
            )
            return self._create_disabled_results(commits)

        # Filter commits for incremental processing
        commits_to_process = self._filter_commits_for_processing(commits, force_reprocess)

        if not commits_to_process:
            self.logger.info("No commits require processing (all up-to-date)")
            # Return existing results for all commits
            return self._get_existing_results(commits)

        self.processing_stats["processing_start_time"] = time.time()
        self.logger.info(
            f"Starting qualitative analysis of {len(commits_to_process)} commits "
            f"({len(commits) - len(commits_to_process)} already processed)"
        )

        # Setup progress tracking
        progress_tracker = (
            ProgressTracker(total=len(commits), description="Qualitative Analysis")
            if show_progress
            else None
        )

        # Step 1: Check cache for known patterns
        cached_results, uncached_commits = self._check_cache(commits, progress_tracker)
        self.logger.info(
            f"Cache provided {len(cached_results)} results, processing {len(uncached_commits)} commits"
        )

        # Step 2: Process uncached commits with NLP
        nlp_results = []
        if uncached_commits:
            nlp_results = self._process_with_nlp(uncached_commits, progress_tracker)

        # Step 3: Identify uncertain cases for LLM processing
        confident_results, uncertain_commits = self._separate_by_confidence(nlp_results)
        self.logger.info(
            f"NLP confident: {len(confident_results)}, uncertain: {len(uncertain_commits)}"
        )

        # Step 4: Process uncertain cases with LLM if available
        llm_results = []
        if uncertain_commits and self.llm_fallback:
            llm_results = self._process_with_llm(uncertain_commits, progress_tracker)
        else:
            # If no LLM available, keep uncertain results with lower confidence
            llm_results = [
                self._convert_to_uncertain_result(commit) for commit in uncertain_commits
            ]

        # Step 5: Update cache with new high-confidence patterns
        all_new_results = confident_results + llm_results
        if self.config.cache_config.enable_pattern_learning:
            self.pattern_cache.learn_from_results(all_new_results)

        # Step 6: Combine all results
        all_results = cached_results + confident_results + llm_results

        # Update processing statistics
        self._update_processing_stats(
            len(commits), len(cached_results), len(confident_results), len(llm_results)
        )

        # Periodic cache optimization
        if self._should_optimize_cache():
            self._optimize_system()

        # Update schema tracking after successful processing
        self._update_schema_tracking(commits_to_process)

        self.logger.info(
            f"Qualitative analysis completed in {time.time() - self.processing_stats['processing_start_time']:.2f}s"
        )
        return all_results

    def _check_cache(
        self, commits: list[dict[str, Any]], progress_tracker: Optional[ProgressTracker]
    ) -> tuple[list[QualitativeCommitData], list[dict[str, Any]]]:
        """Check pattern cache for known commit patterns.

        Args:
            commits: List of commit dictionaries
            progress_tracker: Optional progress tracker

        Returns:
            Tuple of (cached_results, uncached_commits)
        """
        cached_results = []
        uncached_commits = []

        for commit in commits:
            cached_result = self.pattern_cache.lookup_pattern(
                commit.get("message", ""), commit.get("files_changed", [])
            )

            if cached_result:
                # Convert cached result to QualitativeCommitData
                result = self._create_result_from_cache(commit, cached_result)
                cached_results.append(result)
                self.processing_stats["cache_hits"] += 1
            else:
                uncached_commits.append(commit)

            if progress_tracker:
                progress_tracker.update(1)

        return cached_results, uncached_commits

    def _process_with_nlp(
        self, commits: list[dict[str, Any]], progress_tracker: Optional[ProgressTracker]
    ) -> list[QualitativeCommitData]:
        """Process commits using NLP engine.

        Args:
            commits: List of commit dictionaries
            progress_tracker: Optional progress tracker

        Returns:
            List of QualitativeCommitData from NLP processing
        """
        if not commits:
            return []

        if self.nlp_engine is None:
            self.logger.warning("NLP engine not available, skipping NLP processing")
            return []

        def process_batch_with_progress(batch: list[dict[str, Any]]) -> list[QualitativeCommitData]:
            results = self.nlp_engine.process_batch(batch)
            if progress_tracker:
                progress_tracker.update(len(batch))
            return results

        # Use batch processing for efficiency
        if self.config.nlp_config.enable_parallel_processing and len(commits) > 1000:
            all_results = self.batch_processor.process_batches(
                commits, process_batch_with_progress, parallel=True
            )
        else:
            all_results = self.batch_processor.process_batches(
                commits, process_batch_with_progress, parallel=False
            )

        self.processing_stats["nlp_processed"] += len(commits)
        return all_results

    def _separate_by_confidence(
        self, results: list[QualitativeCommitData]
    ) -> tuple[list[QualitativeCommitData], list[dict[str, Any]]]:
        """Separate results by confidence threshold.

        Args:
            results: List of NLP analysis results

        Returns:
            Tuple of (confident_results, uncertain_commit_dicts)
        """
        confident_results = []
        uncertain_commits = []

        for result in results:
            if result.confidence_score >= self.config.confidence_threshold:
                confident_results.append(result)
            else:
                # Convert back to commit dict for LLM processing
                commit_dict = {
                    "hash": result.hash,
                    "message": result.message,
                    "author_name": result.author_name,
                    "author_email": result.author_email,
                    "timestamp": result.timestamp,
                    "files_changed": result.files_changed,
                    "insertions": result.insertions,
                    "deletions": result.deletions,
                }
                uncertain_commits.append(commit_dict)

        return confident_results, uncertain_commits

    def _process_with_llm(
        self, commits: list[dict[str, Any]], progress_tracker: Optional[ProgressTracker]
    ) -> list[QualitativeCommitData]:
        """Process uncertain commits with LLM fallback.

        Args:
            commits: List of uncertain commit dictionaries
            progress_tracker: Optional progress tracker

        Returns:
            List of QualitativeCommitData from LLM processing
        """
        if not commits or not self.llm_fallback:
            return []

        # Check LLM usage limits
        max_llm_commits = int(len(commits) * self.config.max_llm_fallback_pct)
        if len(commits) > max_llm_commits:
            self.logger.warning(
                f"LLM limit reached: processing {max_llm_commits} of {len(commits)} uncertain commits"
            )
            commits = commits[:max_llm_commits]

        # Group similar commits for batch processing
        grouped_commits = self.llm_fallback.group_similar_commits(commits)
        self.logger.debug(
            f"Grouped {len(commits)} commits into {len(grouped_commits)} groups for LLM processing"
        )

        all_results = []

        for group in grouped_commits:
            try:
                group_results = self.llm_fallback.process_group(group)
                all_results.extend(group_results)

                if progress_tracker:
                    progress_tracker.update(len(group))

            except Exception as e:
                self.logger.error(f"LLM processing failed for group of {len(group)} commits: {e}")
                # Create fallback results for this group
                fallback_results = [self._convert_to_uncertain_result(commit) for commit in group]
                all_results.extend(fallback_results)

                if progress_tracker:
                    progress_tracker.update(len(group))

        self.processing_stats["llm_processed"] += len(commits)
        return all_results

    def _create_result_from_cache(
        self, commit: dict[str, Any], cached_data: dict[str, Any]
    ) -> QualitativeCommitData:
        """Create QualitativeCommitData from cached pattern.

        Args:
            commit: Original commit dictionary
            cached_data: Cached classification data

        Returns:
            QualitativeCommitData object
        """
        return QualitativeCommitData(
            # Copy commit fields
            hash=commit.get("hash", ""),
            message=commit.get("message", ""),
            author_name=commit.get("author_name", ""),
            author_email=commit.get("author_email", ""),
            timestamp=commit.get("timestamp", time.time()),
            files_changed=commit.get("files_changed", []),
            insertions=commit.get("insertions", 0),
            deletions=commit.get("deletions", 0),
            # Use cached classification data
            change_type=cached_data.get("change_type", "unknown"),
            change_type_confidence=cached_data.get("change_type_confidence", 0.5),
            business_domain=cached_data.get("business_domain", "unknown"),
            domain_confidence=cached_data.get("domain_confidence", 0.5),
            risk_level=cached_data.get("risk_level", "medium"),
            risk_factors=cached_data.get("risk_factors", []),
            intent_signals=cached_data.get("intent_signals", {}),
            collaboration_patterns=cached_data.get("collaboration_patterns", {}),
            technical_context={"processing_method": "cached"},
            # Processing metadata
            processing_method="cache",
            processing_time_ms=0.5,  # Very fast for cached results
            confidence_score=cached_data.get("confidence_score", 0.5),
        )

    def _convert_to_uncertain_result(self, commit: dict[str, Any]) -> QualitativeCommitData:
        """Convert commit to uncertain result when LLM is unavailable.

        Args:
            commit: Commit dictionary

        Returns:
            QualitativeCommitData with uncertain classifications
        """
        return QualitativeCommitData(
            # Copy commit fields
            hash=commit.get("hash", ""),
            message=commit.get("message", ""),
            author_name=commit.get("author_name", ""),
            author_email=commit.get("author_email", ""),
            timestamp=commit.get("timestamp", time.time()),
            files_changed=commit.get("files_changed", []),
            insertions=commit.get("insertions", 0),
            deletions=commit.get("deletions", 0),
            # Uncertain classifications
            change_type="unknown",
            change_type_confidence=0.3,
            business_domain="unknown",
            domain_confidence=0.3,
            risk_level="medium",
            risk_factors=["low_confidence_classification"],
            intent_signals={"confidence": 0.3},
            collaboration_patterns={},
            technical_context={"processing_method": "uncertain_fallback"},
            # Processing metadata
            processing_method="nlp",
            processing_time_ms=1.0,
            confidence_score=0.3,
        )

    def _create_disabled_results(
        self, commits: list[dict[str, Any]]
    ) -> list[QualitativeCommitData]:
        """Create disabled results when qualitative analysis is turned off.

        Args:
            commits: List of commit dictionaries

        Returns:
            List of QualitativeCommitData with disabled status
        """
        results = []

        for commit in commits:
            result = QualitativeCommitData(
                # Copy commit fields
                hash=commit.get("hash", ""),
                message=commit.get("message", ""),
                author_name=commit.get("author_name", ""),
                author_email=commit.get("author_email", ""),
                timestamp=commit.get("timestamp", time.time()),
                files_changed=commit.get("files_changed", []),
                insertions=commit.get("insertions", 0),
                deletions=commit.get("deletions", 0),
                # Disabled classifications
                change_type="disabled",
                change_type_confidence=0.0,
                business_domain="disabled",
                domain_confidence=0.0,
                risk_level="unknown",
                risk_factors=["qualitative_analysis_disabled"],
                intent_signals={"disabled": True},
                collaboration_patterns={},
                technical_context={"processing_method": "disabled"},
                # Processing metadata
                processing_method="disabled",
                processing_time_ms=0.0,
                confidence_score=0.0,
            )
            results.append(result)

        return results

    def _update_processing_stats(
        self, total_commits: int, cached: int, nlp_processed: int, llm_processed: int
    ) -> None:
        """Update processing statistics.

        Args:
            total_commits: Total number of commits processed
            cached: Number of cache hits
            nlp_processed: Number processed by NLP
            llm_processed: Number processed by LLM
        """
        self.processing_stats["total_processed"] += total_commits
        self.processing_stats["cache_hits"] += cached
        self.processing_stats["nlp_processed"] += nlp_processed
        self.processing_stats["llm_processed"] += llm_processed

        # Log processing breakdown
        cache_pct = (cached / total_commits) * 100 if total_commits > 0 else 0
        nlp_pct = (nlp_processed / total_commits) * 100 if total_commits > 0 else 0
        llm_pct = (llm_processed / total_commits) * 100 if total_commits > 0 else 0

        self.logger.info(
            f"Processing breakdown: {cache_pct:.1f}% cached, {nlp_pct:.1f}% NLP, {llm_pct:.1f}% LLM"
        )

    def _should_optimize_cache(self) -> bool:
        """Check if cache optimization should be performed.

        Returns:
            True if optimization should be performed
        """
        # Optimize every 10,000 commits or every hour
        return bool(
            self.processing_stats["total_processed"] % 10000 == 0
            or self.processing_stats["last_optimization"] is None
            or time.time() - self.processing_stats["last_optimization"] > 3600
        )

    def _optimize_system(self) -> None:
        """Perform system optimization."""
        self.logger.info("Performing system optimization...")

        # Optimize pattern cache
        self.pattern_cache.optimize_cache()

        # Update last optimization time
        self.processing_stats["last_optimization"] = time.time()

        self.logger.info("System optimization completed")

    def get_processing_statistics(self) -> dict[str, Any]:
        """Get comprehensive processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        # Get component statistics
        cache_stats = self.pattern_cache.get_cache_statistics()
        nlp_stats = (
            self.nlp_engine.get_performance_stats()
            if self.nlp_engine is not None
            else {"status": "unavailable", "reason": "spaCy model not installed"}
        )

        # Calculate processing rates
        total_time = time.time() - (self.processing_stats["processing_start_time"] or time.time())
        commits_per_second = (
            self.processing_stats["total_processed"] / total_time if total_time > 0 else 0
        )

        # Calculate method percentages
        total = self.processing_stats["total_processed"]
        method_percentages = {
            "cache": (self.processing_stats["cache_hits"] / total * 100) if total > 0 else 0,
            "nlp": (self.processing_stats["nlp_processed"] / total * 100) if total > 0 else 0,
            "llm": (self.processing_stats["llm_processed"] / total * 100) if total > 0 else 0,
        }

        stats = {
            "processing_summary": {
                "total_commits_processed": self.processing_stats["total_processed"],
                "commits_per_second": commits_per_second,
                "total_processing_time_seconds": total_time,
                "method_breakdown": method_percentages,
            },
            "cache_statistics": cache_stats,
            "nlp_statistics": nlp_stats,
            "configuration": {
                "enabled": self.config.enabled,
                "confidence_threshold": self.config.confidence_threshold,
                "max_llm_fallback_pct": self.config.max_llm_fallback_pct,
                "batch_size": self.config.batch_size,
            },
        }

        # Add LLM statistics if available
        if self.llm_fallback:
            stats["llm_statistics"] = {
                "cost_tracking": self.llm_fallback.cost_tracker.get_usage_stats(),
                "model_usage": "available",
            }
        else:
            stats["llm_statistics"] = {"model_usage": "disabled"}

        return stats

    def validate_setup(self) -> tuple[bool, list[str]]:
        """Validate processor setup and dependencies.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Validate NLP engine
        if self.nlp_engine is None:
            issues.append("NLP: Engine not available (spaCy model not installed)")
        else:
            nlp_valid, nlp_issues = self.nlp_engine.validate_setup()
            if not nlp_valid:
                issues.extend([f"NLP: {issue}" for issue in nlp_issues])

        # Validate LLM fallback if configured
        if self.config.llm_config.openrouter_api_key and self.llm_fallback is None:
            issues.append("LLM: API key configured but fallback system failed to initialize")

        # Validate configuration
        config_warnings = self.config.validate()
        issues.extend([f"Config: {warning}" for warning in config_warnings])

        # Test database connection
        try:
            with self.database.get_session() as session:
                from sqlalchemy import text

                session.execute(text("SELECT 1"))
        except Exception as e:
            issues.append(f"Database: Connection failed - {e}")

        return len(issues) == 0, issues
