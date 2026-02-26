"""Git operations mixin for GitAnalyzer."""

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



class GitAnalyzerMixin:
    """Mixin providing git fetch and branch operations for GitAnalyzer."""

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

        BUG-FIX (BUG 1 + BUG 2): Build the commit-to-branch mapping here during collection
        instead of materialising full histories per commit in _get_commit_branch.  We also
        deduplicate inline with a seen-set so we never hold the full multi-branch list in
        memory simultaneously.

        Args:
            repo: Git repository object
            since: Date to get commits since

        Returns:
            List of unique commits from all branches
        """
        logger.info("Analyzing all branches for complete commit coverage")

        # BUG 2 FIX: Deduplicate inline during collection to avoid the triple-copy
        # pattern (branch list -> commits list -> unique_commits list).  A single
        # seen-set keeps peak memory proportional to the number of UNIQUE commits.
        seen: set[str] = set()
        commits: list[git.Commit] = []

        # BUG 1 FIX: Build commit->branch mapping while iterating so that
        # _get_commit_branch never needs to re-traverse branch histories.
        # First-writer-wins: earlier branches (e.g. main) take priority.
        self._commit_branch_map: dict[str, str] = {}

        branch_count = 0
        processed_refs: set[str] = set()  # Track processed refs to avoid duplicates

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
                    ref_commits = repo.iter_commits(
                        ref, since=since, max_count=self.branch_commit_limit
                    )
                else:
                    ref_commits = repo.iter_commits(ref, since=since)

                branch_commit_count = 0
                for commit in ref_commits:
                    hexsha = commit.hexsha
                    # Map commit to the first branch we encounter it on
                    if hexsha not in self._commit_branch_map:
                        self._commit_branch_map[hexsha] = ref_name
                    # Deduplicate across branches in-place
                    if hexsha not in seen:
                        seen.add(hexsha)
                        commits.append(commit)
                    branch_commit_count += 1

                logger.debug(
                    f"Branch {ref_name}: found {branch_commit_count} commits"
                    + (
                        f" (limited to {self.branch_commit_limit})"
                        if self.branch_commit_limit
                        else ""
                    )
                )

                if self.enable_progress_logging and branch_count % 10 == 0:
                    logger.info(
                        f"Processed {branch_count} branches, found {len(commits)} unique commits so far"
                    )

            except git.GitCommandError as e:
                logger.debug(f"Skipping branch {ref_name} due to error: {e}")
                continue

        logger.info(f"Found {len(commits)} unique commits across {branch_count} branches/refs")
        return sorted(commits, key=lambda c: c.committed_datetime)


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

        # BUG 2 FIX: Collect unique commits inline via a seen-set so we never hold
        # per-branch lists AND the deduplicated list in memory at the same time.
        # BUG 1 FIX: Build commit->branch mapping here so _get_commit_branch is O(1).
        seen: set[str] = set()
        commits: list[git.Commit] = []
        self._commit_branch_map: dict[str, str] = {}

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
                            ref_commits = repo.iter_commits(
                                branch_data["ref"],
                                since=since,
                                max_count=self.branch_commit_limit,
                            )
                        else:
                            ref_commits = repo.iter_commits(branch_data["ref"], since=since)

                        branch_commit_count = 0
                        for commit in ref_commits:
                            hexsha = commit.hexsha
                            if hexsha not in self._commit_branch_map:
                                self._commit_branch_map[hexsha] = branch_data["name"]
                            if hexsha not in seen:
                                seen.add(hexsha)
                                commits.append(commit)
                            branch_commit_count += 1

                        # Update progress description with branch info
                        branch_display = branch_data["name"][:15] + (
                            "..." if len(branch_data["name"]) > 15 else ""
                        )
                        progress.set_description(
                            ctx,
                            f"Analyzing branches [{branch_display}: {branch_commit_count} commits]",
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
                        ref_commits = repo.iter_commits(
                            branch_data["ref"], since=since, max_count=self.branch_commit_limit
                        )
                    else:
                        ref_commits = repo.iter_commits(branch_data["ref"], since=since)

                    for commit in ref_commits:
                        hexsha = commit.hexsha
                        if hexsha not in self._commit_branch_map:
                            self._commit_branch_map[hexsha] = branch_data["name"]
                        if hexsha not in seen:
                            seen.add(hexsha)
                            commits.append(commit)

                except git.GitCommandError as e:
                    logger.debug(f"Failed to get commits from branch {branch_data['name']}: {e}")

        logger.info(
            f"Smart analysis found {len(commits)} unique commits from {len(selected_branches)} branches"
        )
        return sorted(commits, key=lambda c: c.committed_datetime)

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
        self, repo: Repo, repo_path: Path, commits: list[git.Commit], since: datetime
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
                cached_commit_data = cached_commits[commit.hexsha]
                # Filter cached commits by date range to ensure consistency with git filtering
                cached_timestamp = cached_commit_data.get("timestamp")
                if cached_timestamp:
                    # Ensure both timestamps are timezone-aware for comparison
                    if cached_timestamp.tzinfo is None:
                        cached_timestamp = cached_timestamp.replace(tzinfo=timezone.utc)
                    since_tz = since.replace(tzinfo=timezone.utc) if since.tzinfo is None else since

                    if cached_timestamp >= since_tz:
                        results.append(cached_commit_data)
                        cache_hits += 1
                        continue
                    else:
                        # Cached commit is outside date range, treat as cache miss and re-analyze
                        logger.debug(
                            f"Cached commit {commit.hexsha[:8]} outside date range ({cached_timestamp} < {since_tz}), re-analyzing"
                        )

            # Analyze commit (for cache misses or date-filtered cached commits)
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

