"""Git repository analyzer with batch processing support."""

import fnmatch
import logging
import os
import re
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import git
from git import Repo

from ..extractors.story_points import StoryPointExtractor
from ..extractors.tickets import TicketExtractor
from ..types import FilteredCommitStats
from ..utils.commit_utils import is_merge_commit
from .branch_mapper import BranchToProjectMapper
from .cache import GitAnalysisCache
from .progress import get_progress_service

# Import ML extractor with fallback
try:
    from ..extractors.ml_tickets import MLTicketExtractor

    ML_EXTRACTOR_AVAILABLE = True
except ImportError:
    ML_EXTRACTOR_AVAILABLE = False

# Get logger for this module
logger = logging.getLogger(__name__)


class GitAnalyzer:
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
        self.story_point_extractor = StoryPointExtractor(patterns=story_point_patterns)

        # Initialize ticket extractor (ML or standard based on config and availability)
        if (
            ml_categorization_config
            and ml_categorization_config.get("enabled", True)
            and ML_EXTRACTOR_AVAILABLE
        ):
            logger.info("Initializing ML-enhanced ticket extractor")

            # Check if LLM classification is enabled
            enable_llm = llm_config and llm_config.get("enabled", False)
            if enable_llm:
                logger.info("LLM-based commit classification enabled")

            self.ticket_extractor = MLTicketExtractor(
                allowed_platforms=allowed_ticket_platforms,
                ml_config=ml_categorization_config,
                llm_config=llm_config,
                cache_dir=cache.cache_dir / "ml_predictions",
                enable_ml=True,
                enable_llm=enable_llm,
            )
        else:
            if ml_categorization_config and ml_categorization_config.get("enabled", True):
                if not ML_EXTRACTOR_AVAILABLE:
                    logger.warning(
                        "ML categorization requested but dependencies not available, using standard extractor"
                    )
                else:
                    logger.info(
                        "ML categorization disabled in configuration, using standard extractor"
                    )
            else:
                logger.debug("Using standard ticket extractor")

            self.ticket_extractor = TicketExtractor(allowed_platforms=allowed_ticket_platforms)

        self.branch_mapper = BranchToProjectMapper(branch_mapping_rules)
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
                    repo, repo_path, batch
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
                if os.getenv("GITFLOW_DEBUG", "").lower() in ("1", "true", "yes"):
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

    def _update_repository(self, repo) -> bool:
        """Update repository from remote before analysis.

        WHY: This ensures we have the latest commits from the remote repository
        before performing analysis. Critical for getting accurate data especially
        when analyzing repositories that are actively being developed.

        DESIGN DECISION: Uses fetch() for all cases, then pull() only when on a
        tracking branch that's not in detached HEAD state. This approach:
        - Handles detached HEAD states gracefully (common in CI/CD)
        - Always gets latest refs from remote via fetch
        - Only attempts pull when it's safe to do so
        - Continues analysis even if update fails (logs warning)

        Args:
            repo: GitPython Repo object

        Returns:
            bool: True if update succeeded, False if failed (but analysis continues)
        """
        try:
            if repo.remotes:
                origin = repo.remotes.origin
                logger.info("Fetching latest changes from remote")
                origin.fetch()

                # Only try to pull if not in detached HEAD state
                if not repo.head.is_detached:
                    current_branch = repo.active_branch
                    tracking = current_branch.tracking_branch()
                    if tracking:
                        # Pull latest changes
                        origin.pull()
                        logger.debug(f"Pulled latest changes for {current_branch.name}")
                    else:
                        logger.debug(
                            f"Branch {current_branch.name} has no tracking branch, skipping pull"
                        )
                else:
                    logger.debug("Repository in detached HEAD state, skipping pull")
                return True
            else:
                logger.debug("No remotes configured, skipping repository update")
                return True
        except Exception as e:
            logger.warning(f"Could not update repository: {e}")
            # Continue with analysis using local state
            return False

    def _get_commits_optimized(
        self, repo: Repo, since: datetime, branch: Optional[str] = None
    ) -> list[git.Commit]:
        """Get commits from repository with branch analysis strategy.

        WHY: Different analysis needs require different branch coverage approaches.
        The default "all" strategy ensures complete commit coverage without missing
        important development work that happens on feature branches.

        DESIGN DECISION: Three strategies available:
        1. "main_only": Only analyze main/master branch (fastest, least comprehensive)
        2. "smart": Analyze active branches with smart filtering (balanced, may miss commits)
        3. "all": Analyze all branches (comprehensive coverage, default)

        DEFAULT STRATEGY CHANGED: Now defaults to "all" to ensure complete coverage
        after reports that "smart" strategy was missing significant commits (~100+ commits
        and entire developers working on feature branches).

        The "smart" strategy filters branches based on:
        - Recent activity (commits within active_days_threshold)
        - Branch naming patterns (always include main, release, hotfix branches)
        - Exclude automation branches (dependabot, renovate, etc.)
        - Limit total branches per repository

        The "all" strategy:
        - Analyzes all local and remote branches/refs
        - No artificial branch limits
        - No commit limits per branch (unless explicitly configured)
        - Ensures complete development history capture
        """
        logger.debug(f"Getting commits since: {since} (tzinfo: {getattr(since, 'tzinfo', 'N/A')})")
        logger.debug(f"Using branch analysis strategy: {self.branch_strategy}")

        if self.branch_strategy == "main_only":
            return self._get_main_branch_commits(repo, since, branch)
        elif self.branch_strategy == "all":
            logger.info("Using 'all' branches strategy for complete commit coverage")
            return self._get_all_branch_commits(repo, since)
        else:  # smart strategy
            return self._get_smart_branch_commits(repo, since)

    def _get_main_branch_commits(
        self, repo: Repo, since: datetime, branch: Optional[str] = None
    ) -> list[git.Commit]:
        """Get commits from main branch only (fastest strategy).

        Args:
            repo: Git repository object
            since: Date to get commits since
            branch: Specific branch to analyze (overrides main branch detection)

        Returns:
            List of commits from main branch only
        """
        target_branch = branch
        if not target_branch:
            # Auto-detect main branch
            main_branch_names = ["main", "master", "develop", "dev"]
            for branch_name in main_branch_names:
                try:
                    if branch_name in [b.name for b in repo.branches]:
                        target_branch = branch_name
                        break
                except Exception:
                    continue

            if not target_branch and repo.branches:
                target_branch = repo.branches[0].name  # Fallback to first branch

        if not target_branch:
            logger.warning("No main branch found, no commits will be analyzed")
            return []

        logger.debug(f"Analyzing main branch only: {target_branch}")

        try:
            if self.branch_commit_limit:
                commits = list(
                    repo.iter_commits(
                        target_branch, since=since, max_count=self.branch_commit_limit
                    )
                )
            else:
                commits = list(repo.iter_commits(target_branch, since=since))
            logger.debug(f"Found {len(commits)} commits in main branch {target_branch}")
            return sorted(commits, key=lambda c: c.committed_datetime)
        except git.GitCommandError as e:
            logger.warning(f"Failed to get commits from branch {target_branch}: {e}")
            return []

    def _get_all_branch_commits(self, repo: Repo, since: datetime) -> list[git.Commit]:
        """Get commits from all branches (comprehensive analysis).

        WHY: This strategy captures ALL commits from ALL branches without artificial limitations.
        It's designed to ensure complete coverage even if it takes longer to run.

        DESIGN DECISION: Analyzes both local and remote branches to ensure we don't miss
        commits that exist only on remote branches. Uses no commit limits per branch
        to capture complete development history.

        Args:
            repo: Git repository object
            since: Date to get commits since

        Returns:
            List of unique commits from all branches
        """
        logger.info("Analyzing all branches for complete commit coverage")

        commits = []
        branch_count = 0
        processed_refs = set()  # Track processed refs to avoid duplicates

        # Process all refs (local branches, remote branches, tags)
        for ref in repo.refs:
            # Skip if we've already processed this ref
            ref_name = ref.name
            if ref_name in processed_refs:
                continue

            processed_refs.add(ref_name)
            branch_count += 1

            try:
                # No commit limit - get ALL commits from this branch
                if self.branch_commit_limit:
                    branch_commits = list(
                        repo.iter_commits(ref, since=since, max_count=self.branch_commit_limit)
                    )
                    logger.debug(
                        f"Branch {ref_name}: found {len(branch_commits)} commits (limited to {self.branch_commit_limit})"
                    )
                else:
                    branch_commits = list(repo.iter_commits(ref, since=since))
                    logger.debug(
                        f"Branch {ref_name}: found {len(branch_commits)} commits (no limit)"
                    )

                commits.extend(branch_commits)

                if self.enable_progress_logging and branch_count % 10 == 0:
                    logger.info(
                        f"Processed {branch_count} branches, found {len(commits)} total commits so far"
                    )

            except git.GitCommandError as e:
                logger.debug(f"Skipping branch {ref_name} due to error: {e}")
                continue

        # Remove duplicates while preserving order
        unique_commits = self._deduplicate_commits(commits)

        logger.info(
            f"Found {len(unique_commits)} unique commits across {branch_count} branches/refs"
        )
        return sorted(unique_commits, key=lambda c: c.committed_datetime)

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

    def _get_smart_branch_commits(self, repo: Repo, since: datetime) -> list[git.Commit]:
        """Get commits using smart branch filtering (balanced approach).

        This method implements intelligent branch selection that:
        1. Always includes main/important branches
        2. Includes recently active branches
        3. Excludes automation/temporary branches
        4. Limits total number of branches analyzed

        Args:
            repo: Git repository object
            since: Date to get commits since

        Returns:
            List of unique commits from selected branches
        """
        logger.debug("Using smart branch analysis strategy")

        # Get active date threshold
        active_threshold = datetime.now(timezone.utc) - timedelta(days=self.active_days_threshold)

        # Collect branch information
        branch_info = []

        for ref in repo.refs:
            if ref.name.startswith("origin/"):
                continue  # Skip remote tracking branches

            try:
                branch_name = ref.name

                # Check if branch should be excluded
                if self._should_exclude_branch(branch_name):
                    continue

                # Get latest commit date for this branch
                try:
                    latest_commit = next(repo.iter_commits(ref, max_count=1))
                    latest_date = latest_commit.committed_datetime

                    # Convert to timezone-aware if needed
                    if latest_date.tzinfo is None:
                        latest_date = latest_date.replace(tzinfo=timezone.utc)
                    elif latest_date.tzinfo != timezone.utc:
                        latest_date = latest_date.astimezone(timezone.utc)

                except StopIteration:
                    continue  # Empty branch

                # Determine branch priority
                is_important = self._is_important_branch(branch_name)
                is_active = latest_date >= active_threshold

                branch_info.append(
                    {
                        "ref": ref,
                        "name": branch_name,
                        "latest_date": latest_date,
                        "is_important": is_important,
                        "is_active": is_active,
                    }
                )

            except Exception as e:
                logger.debug(f"Skipping branch {ref.name} due to error: {e}")
                continue

        # Sort branches by importance and activity
        branch_info.sort(
            key=lambda x: (
                x["is_important"],  # Important branches first
                x["is_active"],  # Then active branches
                x["latest_date"],  # Then by recency
            ),
            reverse=True,
        )

        # Select branches to analyze
        selected_branches = branch_info[: self.max_branches_per_repo]

        if self.enable_progress_logging:
            logger.info(
                f"Selected {len(selected_branches)} branches out of {len(branch_info)} total branches"
            )
            important_count = sum(1 for b in selected_branches if b["is_important"])
            active_count = sum(1 for b in selected_branches if b["is_active"])
            logger.debug(f"Selected branches: {important_count} important, {active_count} active")

        # Get commits from selected branches
        commits = []

        # Use centralized progress service
        progress = get_progress_service()

        # Only create progress if logging is enabled
        if self.enable_progress_logging:
            with progress.progress(
                total=len(selected_branches),
                description="Analyzing branches",
                unit="branches",
                leave=False,
            ) as ctx:
                for branch_data in selected_branches:
                    try:
                        if self.branch_commit_limit:
                            branch_commits = list(
                                repo.iter_commits(
                                    branch_data["ref"],
                                    since=since,
                                    max_count=self.branch_commit_limit,
                                )
                            )
                        else:
                            branch_commits = list(
                                repo.iter_commits(branch_data["ref"], since=since)
                            )
                        commits.extend(branch_commits)

                        # Update progress description with branch info
                        branch_display = branch_data["name"][:15] + (
                            "..." if len(branch_data["name"]) > 15 else ""
                        )
                        progress.set_description(
                            ctx,
                            f"Analyzing branches [{branch_display}: {len(branch_commits)} commits]",
                        )

                    except git.GitCommandError as e:
                        logger.debug(
                            f"Failed to get commits from branch {branch_data['name']}: {e}"
                        )

                    progress.update(ctx, 1)
        else:
            # No progress bar when logging is disabled
            for branch_data in selected_branches:
                try:
                    if self.branch_commit_limit:
                        branch_commits = list(
                            repo.iter_commits(
                                branch_data["ref"], since=since, max_count=self.branch_commit_limit
                            )
                        )
                    else:
                        branch_commits = list(repo.iter_commits(branch_data["ref"], since=since))
                    commits.extend(branch_commits)

                except git.GitCommandError as e:
                    logger.debug(f"Failed to get commits from branch {branch_data['name']}: {e}")

        # Remove duplicates while preserving order
        unique_commits = self._deduplicate_commits(commits)

        logger.info(
            f"Smart analysis found {len(unique_commits)} unique commits from {len(selected_branches)} branches"
        )
        return sorted(unique_commits, key=lambda c: c.committed_datetime)

    def _should_exclude_branch(self, branch_name: str) -> bool:
        """Check if a branch should be excluded from analysis.

        Args:
            branch_name: Name of the branch to check

        Returns:
            True if the branch should be excluded, False otherwise
        """
        # Check against exclude patterns
        for pattern in self.always_exclude_patterns:
            if re.match(pattern, branch_name, re.IGNORECASE):
                return True
        return False

    def _is_important_branch(self, branch_name: str) -> bool:
        """Check if a branch is considered important and should always be included.

        Args:
            branch_name: Name of the branch to check

        Returns:
            True if the branch is important, False otherwise
        """
        # Check against important branch patterns
        for pattern in self.always_include_patterns:
            if re.match(pattern, branch_name, re.IGNORECASE):
                return True
        return False

    def _deduplicate_commits(self, commits: list[git.Commit]) -> list[git.Commit]:
        """Remove duplicate commits while preserving order.

        Args:
            commits: List of commits that may contain duplicates

        Returns:
            List of unique commits in original order
        """
        seen = set()
        unique_commits = []

        for commit in commits:
            if commit.hexsha not in seen:
                seen.add(commit.hexsha)
                unique_commits.append(commit)

        return unique_commits

    def _batch_commits(
        self, commits: list[git.Commit], batch_size: int
    ) -> Generator[list[git.Commit], None, None]:
        """Yield batches of commits."""
        for i in range(0, len(commits), batch_size):
            yield commits[i : i + batch_size]

    def _process_batch(
        self, repo: Repo, repo_path: Path, commits: list[git.Commit]
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Process a batch of commits with optimized cache lookups.

        WHY: Bulk cache lookups are much faster than individual queries.
        This optimization can reduce subsequent run times from minutes to seconds
        when most commits are already cached.

        ENHANCEMENT: Now uses enhanced bulk_get_commits for better performance
        and automatically detects when to use bulk_store_commits for new data.

        Returns:
            Tuple of (results, cache_hits, cache_misses)
        """
        results = []

        # Use enhanced bulk fetch with better performance
        commit_hashes = [commit.hexsha for commit in commits]
        cached_commits = self.cache.bulk_get_commits(str(repo_path), commit_hashes)

        cache_hits = 0
        cache_misses = 0
        new_commits = []

        for commit in commits:
            # Check bulk cache results
            if commit.hexsha in cached_commits:
                results.append(cached_commits[commit.hexsha])
                cache_hits += 1
                continue

            # Analyze commit
            commit_data = self._analyze_commit(repo, commit, repo_path)
            results.append(commit_data)
            new_commits.append(commit_data)
            cache_misses += 1

        # Use bulk_store_commits for better performance when we have many new commits
        if len(new_commits) >= 10:  # Threshold for bulk operations
            logger.debug(f"Using bulk_store_commits for {len(new_commits)} new commits")
            stats = self.cache.bulk_store_commits(str(repo_path), new_commits)
            if stats["inserted"] > 0:
                logger.debug(
                    f"Bulk stored {stats['inserted']} commits at {stats['commits_per_second']:.0f} commits/sec"
                )
        elif new_commits:
            # Fall back to regular batch caching for small numbers
            self.cache.cache_commits_batch(str(repo_path), new_commits)

        # Log cache performance for debugging
        if cache_hits + cache_misses > 0:
            cache_hit_rate = (cache_hits / (cache_hits + cache_misses)) * 100
            logger.debug(
                f"Batch cache performance: {cache_hits} hits, {cache_misses} misses ({cache_hit_rate:.1f}% hit rate)"
            )

        return results, cache_hits, cache_misses

    def _analyze_commit(self, repo: Repo, commit: git.Commit, repo_path: Path) -> dict[str, Any]:
        """Analyze a single commit."""
        # Normalize timestamp handling
        commit_timestamp = commit.committed_datetime
        logger.debug(
            f"Analyzing commit {commit.hexsha[:8]}: original timestamp={commit_timestamp} (tzinfo: {getattr(commit_timestamp, 'tzinfo', 'N/A')})"
        )

        # Ensure timezone-aware timestamp in UTC
        from datetime import timezone

        if commit_timestamp.tzinfo is None:
            # Convert naive datetime to UTC
            commit_timestamp = commit_timestamp.replace(tzinfo=timezone.utc)
            logger.debug(f"  Converted naive timestamp to UTC: {commit_timestamp}")
        elif commit_timestamp.tzinfo != timezone.utc:
            # Convert to UTC if in different timezone
            commit_timestamp = commit_timestamp.astimezone(timezone.utc)
            logger.debug(f"  Converted timestamp to UTC: {commit_timestamp}")
        else:
            logger.debug(f"  Timestamp already in UTC: {commit_timestamp}")

        # Get the local hour for the developer (before UTC conversion)
        local_hour = commit.committed_datetime.hour

        # Basic commit data
        commit_data = {
            "hash": commit.hexsha,
            "author_name": commit.author.name,
            "author_email": commit.author.email,
            "message": commit.message,
            "timestamp": commit_timestamp,  # Now guaranteed to be UTC timezone-aware
            "local_hour": local_hour,  # Hour in developer's local timezone
            "is_merge": len(commit.parents) > 1,
        }

        # Get branch name
        commit_data["branch"] = self._get_commit_branch(repo, commit)

        # Map branch to project
        commit_data["inferred_project"] = self.branch_mapper.map_branch_to_project(
            str(commit_data["branch"]), repo_path
        )

        # Calculate metrics using reliable git numstat for accurate line counts
        raw_stats = self._calculate_raw_stats(commit)
        commit_data["files_changed_count"] = raw_stats[
            "files"
        ]  # Integer count for backward compatibility
        commit_data["files_changed"] = self._get_changed_file_paths(
            commit
        )  # List of file paths for ML
        commit_data["insertions"] = raw_stats["insertions"]
        commit_data["deletions"] = raw_stats["deletions"]

        # Calculate filtered metrics (excluding boilerplate/generated files)
        filtered_stats = self._calculate_filtered_stats(commit)
        commit_data["filtered_files_changed"] = filtered_stats["files"]
        commit_data["filtered_insertions"] = filtered_stats["insertions"]
        commit_data["filtered_deletions"] = filtered_stats["deletions"]

        # Extract story points
        message_str = (
            commit.message
            if isinstance(commit.message, str)
            else commit.message.decode("utf-8", errors="ignore")
        )
        commit_data["story_points"] = self.story_point_extractor.extract_from_text(message_str)

        # Extract ticket references
        commit_data["ticket_references"] = self.ticket_extractor.extract_from_text(message_str)

        # Calculate complexity delta
        commit_data["complexity_delta"] = self._calculate_complexity_delta(commit)

        return commit_data

    def _get_commit_branch(self, repo: Repo, commit: git.Commit) -> str:
        """Get the branch name for a commit."""
        # This is a simplified approach - getting the first branch that contains the commit
        for branch in repo.branches:
            if commit in repo.iter_commits(branch):
                return branch.name
        return "unknown"

    def _get_changed_file_paths(self, commit: git.Commit) -> list[str]:
        """Extract list of changed file paths from a git commit.

        Args:
            commit: Git commit object

        Returns:
            List of file paths that were changed in the commit
        """
        file_paths = []

        # Handle initial commits (no parents) and regular commits
        parent = commit.parents[0] if commit.parents else None

        try:
            for diff in commit.diff(parent):
                # Get file path - prefer the new path (b_path) for modifications and additions,
                # fall back to old path (a_path) for deletions
                file_path = diff.b_path if diff.b_path else diff.a_path
                if file_path:
                    file_paths.append(file_path)
        except Exception as e:
            logger.warning(f"Failed to extract file paths from commit {commit.hexsha[:8]}: {e}")

        return file_paths

    def _calculate_complexity_delta(self, commit: git.Commit) -> float:
        """Calculate complexity change for a commit with graceful error handling.

        WHY: Repository corruption or missing blobs can cause SHA resolution errors.
        This method provides a fallback complexity calculation that continues
        analysis even when individual blobs are missing or corrupt.
        """
        total_delta = 0.0

        try:
            parent = commit.parents[0] if commit.parents else None
            diffs = commit.diff(parent)
        except Exception as e:
            # If we can't get diffs at all, return 0 complexity delta
            logger.debug(f"Cannot calculate complexity for commit {commit.hexsha[:8]}: {e}")
            return 0.0

        for diff in diffs:
            try:
                if not self._is_code_file(diff.b_path or diff.a_path or ""):
                    continue

                # Simple complexity estimation based on diff size
                # In a real implementation, you'd parse the code and calculate cyclomatic complexity
                if diff.new_file:
                    try:
                        if diff.b_blob and hasattr(diff.b_blob, "size"):
                            total_delta += diff.b_blob.size / 100
                    except (ValueError, AttributeError) as e:
                        logger.debug(
                            f"Cannot access b_blob for new file in {commit.hexsha[:8]}: {e}"
                        )
                        # Use a default small positive delta for new files
                        total_delta += 1.0

                elif diff.deleted_file:
                    try:
                        if diff.a_blob and hasattr(diff.a_blob, "size"):
                            total_delta -= diff.a_blob.size / 100
                    except (ValueError, AttributeError) as e:
                        logger.debug(
                            f"Cannot access a_blob for deleted file in {commit.hexsha[:8]}: {e}"
                        )
                        # Use a default small negative delta for deleted files
                        total_delta -= 1.0

                else:
                    # Modified file - estimate based on change size
                    try:
                        if diff.diff:
                            diff_content = (
                                diff.diff
                                if isinstance(diff.diff, str)
                                else diff.diff.decode("utf-8", errors="ignore")
                            )
                            added = len(diff_content.split("\n+"))
                            removed = len(diff_content.split("\n-"))
                            total_delta += (added - removed) / 10
                    except (ValueError, AttributeError, UnicodeDecodeError) as e:
                        logger.debug(f"Cannot process diff content in {commit.hexsha[:8]}: {e}")
                        # Skip this diff but continue processing
                        pass

            except Exception as e:
                logger.debug(f"Error processing diff in commit {commit.hexsha[:8]}: {e}")
                # Continue to next diff
                continue

        return total_delta

    def _is_code_file(self, filepath: str) -> bool:
        """Check if file is a code file."""
        code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".cs",
            ".vb",
            ".r",
            ".m",
            ".mm",
            ".f90",
            ".f95",
            ".lua",
        }

        return any(filepath.endswith(ext) for ext in code_extensions)

    def _should_exclude_file(self, filepath: str) -> bool:
        """Check if file should be excluded from line counting."""
        if not filepath:
            return False

        # Normalize path separators for consistent matching
        filepath = filepath.replace("\\", "/")

        # Check against exclude patterns with proper ** handling
        return any(self._matches_glob_pattern(filepath, pattern) for pattern in self.exclude_paths)

    def _matches_glob_pattern(self, filepath: str, pattern: str) -> bool:
        """Check if a file path matches a glob pattern, handling ** recursion correctly.

        This method properly handles different glob pattern types:
        - **/vendor/** : matches files inside vendor directories at any level
        - **/*.min.js : matches files with specific suffix anywhere in directory tree
        - vendor/** : matches files inside vendor directory at root level only
        - **pattern** : handles other complex patterns with pathlib.match()
        - simple patterns : uses fnmatch for basic wildcards

        Args:
            filepath: The file path to check
            pattern: The glob pattern to match against

        Returns:
            True if the file path matches the pattern, False otherwise
        """
        from pathlib import PurePath

        # Handle empty or invalid inputs
        if not filepath or not pattern:
            return False

        path = PurePath(filepath)

        # Check for multiple ** patterns first (most complex)
        if "**" in pattern and pattern.count("**") > 1:
            # Multiple ** patterns - use custom recursive matching for complex patterns
            return self._match_recursive_pattern(filepath, pattern)

        # Then handle simple ** patterns
        elif pattern.startswith("**/") and pattern.endswith("/**"):
            # Pattern like **/vendor/** - matches files inside vendor directories at any level
            dir_name = pattern[3:-3]  # Extract 'vendor' from '**/vendor/**'
            if not dir_name:  # Handle edge case of '**/**'
                return True
            return dir_name in path.parts

        elif pattern.startswith("**/"):
            # Pattern like **/*.min.js - matches files with specific suffix anywhere
            suffix_pattern = pattern[3:]
            if not suffix_pattern:  # Handle edge case of '**/'
                return True
            # Check against filename for file patterns, or any path part for directory patterns
            if suffix_pattern.endswith("/"):
                # Directory pattern like **/build/
                dir_name = suffix_pattern[:-1]
                return dir_name in path.parts
            else:
                # File pattern like *.min.js
                # Check both filename AND full path to handle patterns like **/pnpm-lock.yaml
                # matching root-level files (e.g., pnpm-lock.yaml)
                return fnmatch.fnmatch(path.name, suffix_pattern) or fnmatch.fnmatch(
                    filepath, suffix_pattern
                )

        elif pattern.endswith("/**"):
            # Pattern like vendor/** or docs/build/** - matches files inside directory at root level
            dir_name = pattern[:-3]
            if not dir_name:  # Handle edge case of '/**'
                return True

            # Handle both single directory names and nested paths
            expected_parts = PurePath(dir_name).parts
            return (
                len(path.parts) >= len(expected_parts)
                and path.parts[: len(expected_parts)] == expected_parts
            )

        elif "**" in pattern:
            # Single ** pattern - use pathlib matching with fallback
            try:
                return path.match(pattern)
            except (ValueError, TypeError):
                # Fall back to fnmatch if pathlib fails (e.g., invalid pattern)
                try:
                    return fnmatch.fnmatch(filepath, pattern)
                except re.error:
                    # Invalid regex pattern - return False to be safe
                    return False
        else:
            # Simple pattern - use fnmatch for basic wildcards
            try:
                return fnmatch.fnmatch(filepath, pattern)
            except re.error:
                # Invalid regex pattern - return False to be safe
                return False

    def _match_recursive_pattern(self, filepath: str, pattern: str) -> bool:
        """Handle complex patterns with multiple ** wildcards.

        Args:
            filepath: The file path to check
            pattern: The pattern with multiple ** wildcards

        Returns:
            True if the path matches the pattern, False otherwise
        """
        from pathlib import PurePath

        # Split pattern by ** to handle each segment
        parts = pattern.split("**")
        path = PurePath(filepath)
        path_str = str(path)

        # Handle patterns like 'src/**/components/**/*.tsx' or '**/test/**/*.spec.js'
        if len(parts) >= 2:
            # First part should match from the beginning (if not empty)
            start_pattern = parts[0].rstrip("/")
            if start_pattern and not path_str.startswith(start_pattern):
                return False

            # Last part should match the filename/end pattern
            end_pattern = parts[-1].lstrip("/")
            if end_pattern and not fnmatch.fnmatch(path.name, end_pattern):
                # Check if filename matches the end pattern
                return False

            # Middle parts should exist somewhere in the path between start and end
            for i in range(1, len(parts) - 1):
                middle_pattern = parts[i].strip("/")
                if middle_pattern and middle_pattern not in path.parts:
                    # Check if this directory exists in the path
                    return False

            return True

        return False

    def _calculate_filtered_stats(self, commit: git.Commit) -> FilteredCommitStats:
        """Calculate commit statistics excluding boilerplate/generated files using git diff --numstat.

        When exclude_merge_commits is enabled, merge commits (commits with 2+ parents) will have
        their filtered line counts set to 0 to exclude them from productivity metrics.
        """
        filtered_stats: FilteredCommitStats = {"files": 0, "insertions": 0, "deletions": 0}

        # Check if this is a merge commit and we should exclude it from filtered counts
        is_merge = is_merge_commit(commit)
        if self.exclude_merge_commits and is_merge:
            logger.debug(
                f"Excluding merge commit {commit.hexsha[:8]} from filtered line counts "
                f"(has {len(commit.parents)} parents)"
            )
            return filtered_stats  # Return zeros for merge commits

        # For initial commits or commits without parents
        parent = commit.parents[0] if commit.parents else None

        try:
            # Use git command directly for accurate line counts
            repo = commit.repo
            if parent:
                diff_output = repo.git.diff(parent.hexsha, commit.hexsha, "--numstat")
            else:
                # Initial commit - use git show with --numstat
                diff_output = repo.git.show(commit.hexsha, "--numstat", "--format=")

            # Parse the numstat output: insertions\tdeletions\tfilename
            for line in diff_output.strip().split("\n"):
                if not line.strip():
                    continue

                parts = line.split("\t")
                if len(parts) >= 3:
                    try:
                        insertions = int(parts[0]) if parts[0] != "-" else 0
                        deletions = int(parts[1]) if parts[1] != "-" else 0
                        filename = parts[2]

                        # Skip excluded files using the existing filter logic
                        if self._should_exclude_file(filename):
                            continue

                        # Count the file and its changes
                        filtered_stats["files"] += 1
                        filtered_stats["insertions"] += insertions
                        filtered_stats["deletions"] += deletions

                    except ValueError:
                        # Skip binary files or malformed lines
                        continue

        except Exception as e:
            # Log the error for debugging but don't crash
            logger.warning(f"Error calculating filtered stats for commit {commit.hexsha[:8]}: {e}")

        return filtered_stats

    def _calculate_raw_stats(self, commit: git.Commit) -> dict[str, int]:
        """Calculate commit statistics for all files (no filtering) using git diff --numstat."""
        raw_stats = {"files": 0, "insertions": 0, "deletions": 0}

        # For initial commits or commits without parents
        parent = commit.parents[0] if commit.parents else None

        try:
            # Use git command directly for accurate line counts
            repo = commit.repo
            if parent:
                diff_output = repo.git.diff(parent.hexsha, commit.hexsha, "--numstat")
            else:
                # Initial commit - use git show with --numstat
                diff_output = repo.git.show(commit.hexsha, "--numstat", "--format=")

            # Parse the numstat output: insertions\tdeletions\tfilename
            for line in diff_output.strip().split("\n"):
                if not line.strip():
                    continue

                parts = line.split("\t")
                if len(parts) >= 3:
                    try:
                        insertions = int(parts[0]) if parts[0] != "-" else 0
                        deletions = int(parts[1]) if parts[1] != "-" else 0
                        # filename = parts[2] - not used in raw stats

                        # Count all files and their changes (no filtering)
                        raw_stats["files"] += 1
                        raw_stats["insertions"] += insertions
                        raw_stats["deletions"] += deletions

                    except ValueError:
                        # Skip binary files or malformed lines
                        continue

        except Exception as e:
            # Log the error for debugging but don't crash
            logger.warning(f"Error calculating raw stats for commit {commit.hexsha[:8]}: {e}")

        return raw_stats

    def _prepare_commits_for_classification(
        self, repo: Repo, commits: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prepare commits for classification by adding file change information.

        Args:
            repo: Git repository object
            commits: List of analyzed commit dictionaries

        Returns:
            List of commits with file change information needed for classification
        """
        prepared_commits = []

        for commit_data in commits:
            commit_hash = commit_data.get("hash")
            if not commit_hash:
                prepared_commits.append(commit_data)
                continue

            try:
                # Use the file paths already extracted during analysis
                files_changed = commit_data.get("files_changed", [])

                # If files_changed is somehow not available or empty, extract it as fallback
                if not files_changed:
                    logger.warning(
                        f"No file paths found for commit {commit_hash[:8]}, extracting as fallback"
                    )
                    files_changed = self._get_changed_file_paths(repo.commit(commit_hash))

                # Create enhanced commit data for classification
                enhanced_commit = commit_data.copy()
                enhanced_commit["files_changed"] = files_changed

                # Add file details if needed by classifier
                if files_changed:
                    file_details = {}
                    # Only extract file details if we need to get commit object for other reasons
                    # or if file details are specifically required by the classifier
                    try:
                        commit = repo.commit(commit_hash)
                        parent = commit.parents[0] if commit.parents else None

                        for diff in commit.diff(parent):
                            file_path = diff.b_path if diff.b_path else diff.a_path
                            if file_path and file_path in files_changed and diff.diff:
                                # Calculate insertions and deletions per file
                                diff_text = (
                                    diff.diff
                                    if isinstance(diff.diff, str)
                                    else diff.diff.decode("utf-8", errors="ignore")
                                )
                                insertions = len(
                                    [
                                        line
                                        for line in diff_text.split("\n")
                                        if line.startswith("+") and not line.startswith("+++")
                                    ]
                                )
                                deletions = len(
                                    [
                                        line
                                        for line in diff_text.split("\n")
                                        if line.startswith("-") and not line.startswith("---")
                                    ]
                                )

                                file_details[file_path] = {
                                    "insertions": insertions,
                                    "deletions": deletions,
                                }

                        enhanced_commit["file_details"] = file_details
                    except Exception as detail_error:
                        logger.warning(
                            f"Failed to extract file details for commit {commit_hash[:8]}: {detail_error}"
                        )
                        enhanced_commit["file_details"] = {}

                prepared_commits.append(enhanced_commit)

            except Exception as e:
                logger.warning(f"Failed to prepare commit {commit_hash} for classification: {e}")
                prepared_commits.append(commit_data)

        return prepared_commits
