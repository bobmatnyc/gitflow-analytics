"""Git repository analyzer with batch processing support."""

import logging
import re
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import git
from git import Repo

from ..types import FilteredCommitStats
from ..utils.commit_utils import extract_co_authors, is_merge_commit
from ..utils.debug import is_debug_mode
from ..utils.glob_matcher import match_recursive_pattern as _match_recursive_pattern_fn
from ..utils.glob_matcher import matches_glob_pattern as _matches_glob_pattern_fn
from ..utils.glob_matcher import should_exclude_file as _should_exclude_file_fn
from .analysis_components import (
    build_branch_mapper,
    build_story_point_extractor,
    build_ticket_extractor,
)
from .cache import GitAnalysisCache
from .progress import get_progress_service

# Get logger for this module
logger = logging.getLogger(__name__)



from .analyzer_git import GitAnalyzerMixin
from .analyzer_commit import CommitAnalyzerMixin


class GitAnalyzer(GitAnalyzerMixin, CommitAnalyzerMixin):
    """Analyze Git repositories with caching and batch processing."""

    def __init__(
        self,
        cache: GitAnalysisCache,
        batch_size: int = 1000,
        branch_mapping_rules: Optional[dict[str, list[str]]] = None,
        allowed_ticket_platforms: Optional[list[str]] = None,
        exclude_paths: Optional[list[str]] = None,
        story_point_patterns: Optional[list[str]] = None,
        ml_categorization_config: Optional[dict[str, Any]] = None,
        llm_config: Optional[dict[str, Any]] = None,
        classification_config: Optional[dict[str, Any]] = None,
        branch_analysis_config: Optional[dict[str, Any]] = None,
        exclude_merge_commits: bool = False,
    ):
        """Initialize analyzer with cache and optional ML categorization and commit classification.

        Args:
            cache: Git analysis cache instance
            batch_size: Number of commits to process in each batch
            branch_mapping_rules: Rules for mapping branches to projects
            allowed_ticket_platforms: List of allowed ticket platforms
            exclude_paths: List of file paths to exclude from analysis
            story_point_patterns: List of regex patterns for extracting story points
            ml_categorization_config: Configuration for ML-based categorization
            llm_config: Configuration for LLM-based commit classification
            classification_config: Configuration for commit classification
            branch_analysis_config: Configuration for branch analysis optimization
            exclude_merge_commits: Exclude merge commits from filtered line count calculations
        """
        self.cache = cache
        self.batch_size = batch_size
        self.exclude_merge_commits = exclude_merge_commits
        self.story_point_extractor = build_story_point_extractor(patterns=story_point_patterns)
        self.ticket_extractor = build_ticket_extractor(
            allowed_platforms=allowed_ticket_platforms,
            ml_config=ml_categorization_config,
            llm_config=llm_config,
            cache_dir=cache.cache_dir / "ml_predictions",
        )
        self.branch_mapper = build_branch_mapper(branch_mapping_rules)
        self.exclude_paths = exclude_paths or []

        # Initialize branch analysis configuration
        self.branch_analysis_config = branch_analysis_config or {}
        self.branch_strategy = self.branch_analysis_config.get("strategy", "all")
        self.max_branches_per_repo = self.branch_analysis_config.get("max_branches_per_repo", 50)
        self.active_days_threshold = self.branch_analysis_config.get("active_days_threshold", 90)
        self.include_main_branches = self.branch_analysis_config.get("include_main_branches", True)
        self.always_include_patterns = self.branch_analysis_config.get(
            "always_include_patterns",
            [r"^(main|master|develop|dev)$", r"^release/.*", r"^hotfix/.*"],
        )
        self.always_exclude_patterns = self.branch_analysis_config.get(
            "always_exclude_patterns",
            [r"^dependabot/.*", r"^renovate/.*", r".*-backup$", r".*-temp$"],
        )
        self.enable_progress_logging = self.branch_analysis_config.get(
            "enable_progress_logging", True
        )
        self.branch_commit_limit = self.branch_analysis_config.get(
            "branch_commit_limit", None
        )  # No limit by default

        # Initialize commit classifier if enabled
        self.classification_enabled = classification_config and classification_config.get(
            "enabled", False
        )
        self.commit_classifier = None

        if self.classification_enabled:
            try:
                from ..classification.classifier import CommitClassifier

                self.commit_classifier = CommitClassifier(
                    config=classification_config, cache_dir=cache.cache_dir / "classification"
                )
                logger.info("Commit classification enabled")
            except ImportError as e:
                logger.warning(f"Classification dependencies not available: {e}")
                self.classification_enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize commit classifier: {e}")
                self.classification_enabled = False

    def analyze_repository(
        self, repo_path: Path, since: datetime, branch: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Analyze a Git repository with batch processing and optional classification."""
        try:
            repo = Repo(repo_path)
            # Update repository from remote before analysis
            self._update_repository(repo)
        except Exception as e:
            raise ValueError(f"Failed to open repository at {repo_path}: {e}") from e

        # Get commits to analyze with optimized branch selection
        commits = self._get_commits_optimized(repo, since, branch)
        total_commits = len(commits)

        if total_commits == 0:
            return []

        analyzed_commits = []
        total_cache_hits = 0
        total_cache_misses = 0

        # Process in batches with progress bar
        processed_commits = 0
        progress_service = get_progress_service()

        # Only create progress bar if enabled
        if self.enable_progress_logging:
            progress_ctx = progress_service.create_progress(
                total=total_commits, description=f"Analyzing {repo_path.name}", unit="commits"
            )
        else:
            progress_ctx = None

        try:
            for batch in self._batch_commits(commits, self.batch_size):
                batch_results, batch_hits, batch_misses = self._process_batch(
                    repo, repo_path, batch, since
                )
                analyzed_commits.extend(batch_results)

                # Track overall cache performance
                total_cache_hits += batch_hits
                total_cache_misses += batch_misses

                # Note: Caching is now handled within _process_batch for better performance

                # Update progress tracking
                batch_size = len(batch)
                processed_commits += batch_size

                # Update progress bar with cache info if enabled
                if progress_ctx:
                    hit_rate = (batch_hits / batch_size) * 100 if batch_size > 0 else 0
                    progress_service.set_description(
                        progress_ctx,
                        f"Analyzing {repo_path.name} (cache hit: {hit_rate:.1f}%, {processed_commits}/{total_commits})",
                    )
                    progress_service.update(progress_ctx, batch_size)
        finally:
            if progress_ctx:
                progress_service.complete(progress_ctx)

                # Debug logging for progress tracking issues
                if is_debug_mode():
                    logger.debug(
                        f"Final progress: Processed: {processed_commits}/{total_commits} commits"
                    )

        # Log overall cache performance
        if total_cache_hits + total_cache_misses > 0:
            overall_hit_rate = (total_cache_hits / (total_cache_hits + total_cache_misses)) * 100
            logger.info(
                f"Repository {repo_path.name}: {total_cache_hits} cached, {total_cache_misses} analyzed ({overall_hit_rate:.1f}% cache hit rate)"
            )

        # Apply commit classification if enabled
        if self.classification_enabled and self.commit_classifier and analyzed_commits:
            logger.info(f"Applying commit classification to {len(analyzed_commits)} commits")

            try:
                # Prepare commits for classification (add file changes information)
                commits_with_files = self._prepare_commits_for_classification(
                    repo, analyzed_commits
                )

                # Get classification results
                classification_results = self.commit_classifier.classify_commits(commits_with_files)

                # Merge classification results back into analyzed commits
                for commit, classification in zip(analyzed_commits, classification_results):
                    if classification:  # Classification might be empty if disabled or failed
                        commit.update(
                            {
                                "predicted_class": classification.get("predicted_class"),
                                "classification_confidence": classification.get("confidence"),
                                "is_reliable_prediction": classification.get(
                                    "is_reliable_prediction"
                                ),
                                "class_probabilities": classification.get("class_probabilities"),
                                "file_analysis_summary": classification.get("file_analysis"),
                                "classification_metadata": classification.get(
                                    "classification_metadata"
                                ),
                            }
                        )

                logger.info(f"Successfully classified {len(classification_results)} commits")

            except Exception as e:
                logger.error(f"Commit classification failed: {e}")
                # Continue without classification rather than failing entirely

        return analyzed_commits

    def is_analysis_needed(
        self,
        repo_path: Path,
        project_key: str,
        analysis_start: datetime,
        analysis_end: datetime,
        weeks_analyzed: int,
        config_hash: Optional[str] = None,
        force_fetch: bool = False,
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """Check if repository analysis is needed or if cached data can be used.

        WHY: Implements cache-first workflow by checking if repository has been
        fully analyzed for the given period. Enables "fetch once, report many".

        Args:
            repo_path: Path to the repository
            project_key: Project key for the repository
            analysis_start: Start of the analysis period
            analysis_end: End of the analysis period
            weeks_analyzed: Number of weeks to analyze
            config_hash: Hash of relevant configuration to detect changes
            force_fetch: Force re-analysis even if cached data exists

        Returns:
            Tuple of (needs_analysis, cached_status_info)
        """
        if force_fetch:
            logger.info(f"Force fetch enabled for {project_key} - analysis needed")
            return True, None

        # Check if analysis is already complete
        status = self.cache.get_repository_analysis_status(
            repo_path=str(repo_path),
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            config_hash=config_hash,
        )

        if not status:
            logger.info(f"No cached analysis found for {project_key} - analysis needed")
            return True, None

        # Validate completeness
        if (
            status["git_analysis_complete"]
            and status["weeks_analyzed"] >= weeks_analyzed
            and status["commit_count"] > 0
        ):
            logger.info(
                f"Using cached analysis for {project_key}: "
                f"{status['commit_count']} commits, "
                f"{status.get('unique_developers', 0)} developers"
            )
            return False, status
        else:
            logger.info(f"Incomplete cached analysis for {project_key} - re-analysis needed")
            return True, None

    def mark_analysis_complete(
        self,
        repo_path: Path,
        repo_name: str,
        project_key: str,
        analysis_start: datetime,
        analysis_end: datetime,
        weeks_analyzed: int,
        commit_count: int,
        unique_developers: int = 0,
        processing_time_seconds: Optional[float] = None,
        config_hash: Optional[str] = None,
    ) -> None:
        """Mark repository analysis as complete in the cache.

        WHY: Records successful completion to enable cache-first workflow.
        Should be called after successful repository analysis.

        Args:
            repo_path: Path to the repository
            repo_name: Display name for the repository
            project_key: Project key for the repository
            analysis_start: Start of the analysis period
            analysis_end: End of the analysis period
            weeks_analyzed: Number of weeks analyzed
            commit_count: Number of commits processed
            unique_developers: Number of unique developers found
            processing_time_seconds: Time taken for analysis
            config_hash: Hash of relevant configuration
        """
        try:
            self.cache.mark_repository_analysis_complete(
                repo_path=str(repo_path),
                repo_name=repo_name,
                project_key=project_key,
                analysis_start=analysis_start,
                analysis_end=analysis_end,
                weeks_analyzed=weeks_analyzed,
                commit_count=commit_count,
                unique_developers=unique_developers,
                processing_time_seconds=processing_time_seconds,
                config_hash=config_hash,
            )
            logger.info(f"Marked {project_key} analysis as complete: {commit_count} commits")
        except Exception as e:
            logger.warning(f"Failed to mark analysis complete for {project_key}: {e}")

    def mark_analysis_failed(
        self,
        repo_path: Path,
        repo_name: str,
        analysis_start: datetime,
        analysis_end: datetime,
        error_message: str,
        config_hash: Optional[str] = None,
    ) -> None:
        """Mark repository analysis as failed in the cache.

        Args:
            repo_path: Path to the repository
            repo_name: Display name for the repository
            analysis_start: Start of the analysis period
            analysis_end: End of the analysis period
            error_message: Error message describing the failure
            config_hash: Hash of relevant configuration
        """
        try:
            self.cache.mark_repository_analysis_failed(
                repo_path=str(repo_path),
                repo_name=repo_name,
                analysis_start=analysis_start,
                analysis_end=analysis_end,
                error_message=error_message,
                config_hash=config_hash,
            )
            logger.warning(f"Marked {repo_name} analysis as failed: {error_message}")
        except Exception as e:
            logger.error(f"Failed to mark analysis failure for {repo_name}: {e}")

