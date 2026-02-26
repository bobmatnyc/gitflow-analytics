"""Parallel execution and commit stats mixin for GitDataFetcher."""

import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import git
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from ..constants import BatchSizes, Timeouts
from ..integrations.jira_integration import JIRAIntegration
from ..models.database import (
    CachedCommit,
    CommitTicketCorrelation,
    DailyCommitBatch,
    DetailedTicketData,
)
from ..types import CommitStats
from ..utils.commit_utils import is_merge_commit
from ..utils.glob_matcher import match_recursive_pattern as _match_recursive_pattern_fn
from ..utils.glob_matcher import matches_glob_pattern as _matches_glob_pattern_fn
from ..utils.glob_matcher import should_exclude_file as _should_exclude_file_fn
from .analysis_components import (
    build_branch_mapper,
    build_story_point_extractor,
    build_ticket_extractor,
)
from .cache import GitAnalysisCache
from .git_timeout_wrapper import GitOperationTimeout, GitTimeoutWrapper, HeartbeatLogger
from .identity import DeveloperIdentityResolver
from .progress import get_progress_service

logger = logging.getLogger(__name__)
_thread_local = threading.local()


class ParallelFetcherMixin:
    """Mixin providing parallel processing and commit stats methods.

    Provides: _calculate_commit_stats, _store_day_commits_incremental,
    process_repositories_parallel, _process_repository_with_timeout.
    """

    def _calculate_commit_stats(self, commit: git.Commit) -> CommitStats:
        """Calculate commit statistics using reliable git diff --numstat with exclude_paths filtering.

        When exclude_merge_commits is enabled, merge commits (commits with 2+ parents) will have
        their filtered line counts set to 0 to exclude them from productivity metrics.

        Returns:
            CommitStats dictionary with both raw and filtered statistics:
            - 'files', 'insertions', 'deletions': filtered counts (0 for merge commits if excluded)
            - 'raw_insertions', 'raw_deletions': unfiltered counts (always calculated)

        THREAD SAFETY: This method is thread-safe as it works with commit objects
        that have their own repo references.
        """
        stats = {"files": 0, "insertions": 0, "deletions": 0}

        # Track raw stats for storage
        raw_stats = {"files": 0, "insertions": 0, "deletions": 0}
        excluded_stats = {"files": 0, "insertions": 0, "deletions": 0}

        # Check if this is a merge commit and we should exclude it from filtered counts
        is_merge = is_merge_commit(commit)
        if self.exclude_merge_commits and is_merge:
            logger.debug(
                f"Excluding merge commit {commit.hexsha[:8]} from filtered line counts "
                f"(has {len(commit.parents)} parents)"
            )
            # Still need to calculate raw stats for the commit, but filtered stats will be 0
            # Continue with calculation but will return zeros for filtered stats at the end

        # For initial commits or commits without parents
        parent = commit.parents[0] if commit.parents else None

        try:
            # THREAD SAFETY: Use the repo reference from the commit object
            # Each thread has its own commit object with its own repo reference
            repo = commit.repo
            Path(repo.working_dir)

            def get_diff_output() -> str:
                """Get diff output for commit using git numstat.

                Returns:
                    Git diff output string in numstat format
                """
                if parent:
                    return repo.git.diff(parent.hexsha, commit.hexsha, "--numstat")
                else:
                    # Initial commit - use git show with --numstat
                    return repo.git.show(commit.hexsha, "--numstat", "--format=")

            # Use timeout wrapper for git diff operations
            try:
                diff_output = self.git_wrapper.run_with_timeout(
                    get_diff_output,
                    timeout=Timeouts.GIT_DIFF,
                    operation_name=f"diff_{commit.hexsha[:8]}",
                )
            except GitOperationTimeout:
                logger.warning(f"⏱️ Timeout calculating stats for commit {commit.hexsha[:8]}")
                timeout_result: CommitStats = {
                    "files": 0,
                    "insertions": 0,
                    "deletions": 0,
                    "raw_insertions": 0,
                    "raw_deletions": 0,
                }
                return timeout_result

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

                        # Always count raw stats
                        raw_stats["files"] += 1
                        raw_stats["insertions"] += insertions
                        raw_stats["deletions"] += deletions

                        # Skip excluded files based on exclude_paths patterns
                        if self._should_exclude_file(filename):
                            logger.debug(f"Excluding file from line counts: {filename}")
                            excluded_stats["files"] += 1
                            excluded_stats["insertions"] += insertions
                            excluded_stats["deletions"] += deletions
                            continue

                        # Count only non-excluded files and their changes
                        stats["files"] += 1
                        stats["insertions"] += insertions
                        stats["deletions"] += deletions

                    except ValueError:
                        # Skip binary files or malformed lines
                        continue

            # Log exclusion statistics if significant
            if excluded_stats["files"] > 0 or (
                raw_stats["insertions"] > 0 and stats["insertions"] < raw_stats["insertions"]
            ):
                reduction_pct = (
                    100 * (1 - stats["insertions"] / raw_stats["insertions"])
                    if raw_stats["insertions"] > 0
                    else 0
                )
                logger.info(
                    f"Commit {commit.hexsha[:8]}: Excluded {excluded_stats['files']} files, "
                    f"{excluded_stats['insertions']} insertions, {excluded_stats['deletions']} deletions "
                    f"({reduction_pct:.1f}% reduction)"
                )

            # Log if exclusions are configured
            if self.exclude_paths and raw_stats["files"] > 0:
                logger.debug(
                    f"Commit {commit.hexsha[:8]}: Applied {len(self.exclude_paths)} exclusion patterns. "
                    f"Raw: {raw_stats['files']} files, +{raw_stats['insertions']} -{raw_stats['deletions']}. "
                    f"Filtered: {stats['files']} files, +{stats['insertions']} -{stats['deletions']}"
                )

        except Exception as e:
            # Log the error for debugging but don't crash
            logger.warning(f"Error calculating commit stats for {commit.hexsha[:8]}: {e}")

        # If this is a merge commit and we're excluding them, return zeros for filtered stats
        # but keep the raw stats
        if self.exclude_merge_commits and is_merge:
            result: CommitStats = {
                "files": 0,
                "insertions": 0,
                "deletions": 0,
                "raw_insertions": raw_stats["insertions"],
                "raw_deletions": raw_stats["deletions"],
            }
            return result

        # Return both raw and filtered stats
        result: CommitStats = {
            "files": stats["files"],
            "insertions": stats["insertions"],
            "deletions": stats["deletions"],
            "raw_insertions": raw_stats["insertions"],
            "raw_deletions": raw_stats["deletions"],
        }
        return result

    def _store_day_commits_incremental(
        self, repo_path: Path, date_str: str, commits: list[dict[str, Any]], project_key: str
    ) -> None:
        """Store commits for a single day incrementally to enable progress tracking.

        This method stores commits immediately after fetching them for a day,
        allowing for better progress tracking and recovery from interruptions.

        Args:
            repo_path: Path to the repository
            date_str: Date string in YYYY-MM-DD format
            commits: List of commit data for the day
            project_key: Project identifier
        """
        try:
            # Collect summary statistics for INFO-level logging
            merge_count = 0
            excluded_file_count = 0
            total_excluded_insertions = 0
            total_excluded_deletions = 0

            # Transform commits to cache format
            cache_format_commits = []
            for commit in commits:
                # Track merge commits for summary logging
                if commit.get("is_merge", False):
                    merge_count += 1

                # Track excluded file statistics
                raw_insertions = commit.get("raw_insertions", commit.get("lines_added", 0))
                raw_deletions = commit.get("raw_deletions", commit.get("lines_deleted", 0))
                filtered_insertions = commit.get(
                    "filtered_insertions", commit.get("lines_added", 0)
                )
                filtered_deletions = commit.get(
                    "filtered_deletions", commit.get("lines_deleted", 0)
                )

                excluded_insertions = raw_insertions - filtered_insertions
                excluded_deletions = raw_deletions - filtered_deletions
                if excluded_insertions > 0 or excluded_deletions > 0:
                    excluded_file_count += 1
                    total_excluded_insertions += excluded_insertions
                    total_excluded_deletions += excluded_deletions

                cache_format_commit = {
                    "hash": commit["commit_hash"],
                    "author_name": commit.get("author_name", ""),
                    "author_email": commit.get("author_email", ""),
                    "message": commit.get("message", ""),
                    "timestamp": commit["timestamp"],
                    "branch": commit.get("branch", "main"),
                    "is_merge": commit.get("is_merge", False),
                    "files_changed_count": commit.get("files_changed_count", 0),
                    # Store raw unfiltered values
                    "insertions": raw_insertions,
                    "deletions": raw_deletions,
                    # Store filtered values
                    "filtered_insertions": filtered_insertions,
                    "filtered_deletions": filtered_deletions,
                    "story_points": commit.get("story_points"),
                    "ticket_references": commit.get("ticket_references", []),
                }
                cache_format_commits.append(cache_format_commit)

            # Use bulk store for efficiency
            if cache_format_commits:
                bulk_stats = self.cache.bulk_store_commits(str(repo_path), cache_format_commits)
                logger.debug(
                    f"Incrementally stored {bulk_stats['inserted']} commits for {date_str} "
                    f"({bulk_stats['skipped']} already cached)"
                )

            # Summary logging at INFO level for user-facing visibility
            if self.exclude_merge_commits and merge_count > 0:
                logger.info(
                    f"{date_str}: Excluded {merge_count} merge commits from filtered line counts "
                    f"(exclude_merge_commits enabled)"
                )

            if self.exclude_paths and excluded_file_count > 0:
                logger.info(
                    f"{date_str}: Excluded changes from {excluded_file_count} commits "
                    f"(+{total_excluded_insertions} -{total_excluded_deletions} lines) "
                    f"due to path exclusions"
                )

        except Exception as e:
            # Log error but don't fail - commits will be stored again in batch at the end
            logger.warning(f"Failed to incrementally store commits for {date_str}: {e}")

    def process_repositories_parallel(
        self,
        repositories: list[dict],
        weeks_back: int = 4,
        jira_integration: Optional[JIRAIntegration] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_workers: int = 3,
    ) -> dict[str, Any]:
        """Process multiple repositories in parallel with proper timeout protection.

        Args:
            repositories: List of repository configurations
            weeks_back: Number of weeks to analyze
            jira_integration: Optional JIRA integration for ticket data
            start_date: Optional explicit start date
            end_date: Optional explicit end date
            max_workers: Maximum number of parallel workers

        Returns:
            Dictionary containing processing results and statistics
        """
        logger.info(
            f"🚀 Starting parallel processing of {len(repositories)} repositories with {max_workers} workers"
        )

        # Initialize statistics
        self.processing_stats = {
            "total": len(repositories),
            "processed": 0,
            "success": 0,
            "failed": 0,
            "timeout": 0,
            "repositories": {},
        }

        # Get progress service for updates
        progress = get_progress_service()

        # Start heartbeat logger for monitoring
        with HeartbeatLogger(interval=5):
            results = {}

            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all repository processing tasks
                future_to_repo = {}

                for repo_config in repositories:
                    repo_path = Path(repo_config.get("path", ""))
                    project_key = repo_config.get("project_key", repo_path.name)
                    branch_patterns = repo_config.get("branch_patterns")

                    # Submit task with timeout wrapper
                    future = executor.submit(
                        self._process_repository_with_timeout,
                        repo_path,
                        project_key,
                        weeks_back,
                        branch_patterns,
                        jira_integration,
                        start_date,
                        end_date,
                    )
                    future_to_repo[future] = {
                        "path": repo_path,
                        "project_key": project_key,
                        "start_time": time.time(),
                    }

                    logger.info(f"📋 Submitted {project_key} for processing")

                # Process results as they complete
                for future in as_completed(future_to_repo):
                    repo_info = future_to_repo[future]
                    project_key = repo_info["project_key"]
                    elapsed_time = time.time() - repo_info["start_time"]

                    try:
                        result = future.result(timeout=Timeouts.SUBPROCESS_DEFAULT)

                        if result:
                            self.processing_stats["success"] += 1
                            self.processing_stats["repositories"][project_key] = {
                                "status": "success",
                                "elapsed_time": elapsed_time,
                                "commits": result.get("stats", {}).get("total_commits", 0),
                                "tickets": result.get("stats", {}).get("unique_tickets", 0),
                            }
                            results[project_key] = result

                            logger.info(
                                f"✅ {project_key}: Successfully processed "
                                f"{result['stats']['total_commits']} commits in {elapsed_time:.1f}s"
                            )

                            # Update progress
                            if hasattr(progress, "finish_repository"):
                                # Check if progress adapter supports stats parameter
                                if hasattr(progress, "update_stats"):
                                    progress.update_stats(
                                        processed=self.processing_stats["processed"],
                                        success=self.processing_stats["success"],
                                        failed=self.processing_stats["failed"],
                                        timeout=self.processing_stats["timeout"],
                                        total=self.processing_stats["total"],
                                    )
                                    progress.finish_repository(project_key, success=True)
                                else:
                                    progress.finish_repository(project_key, success=True)
                        else:
                            self.processing_stats["failed"] += 1
                            self.processing_stats["repositories"][project_key] = {
                                "status": "failed",
                                "elapsed_time": elapsed_time,
                                "error": "Processing returned no result",
                            }

                            logger.error(
                                f"❌ {project_key}: Processing failed after {elapsed_time:.1f}s"
                            )

                            # Update progress
                            if hasattr(progress, "finish_repository"):
                                # Check if progress adapter supports stats parameter
                                if hasattr(progress, "update_stats"):
                                    progress.update_stats(
                                        processed=self.processing_stats["processed"],
                                        success=self.processing_stats["success"],
                                        failed=self.processing_stats["failed"],
                                        timeout=self.processing_stats["timeout"],
                                        total=self.processing_stats["total"],
                                    )
                                    progress.finish_repository(
                                        project_key,
                                        success=False,
                                        error_message="Processing failed",
                                    )
                                else:
                                    progress.finish_repository(
                                        project_key,
                                        success=False,
                                        error_message="Processing failed",
                                    )

                    except GitOperationTimeout:
                        self.processing_stats["timeout"] += 1
                        self.processing_stats["repositories"][project_key] = {
                            "status": "timeout",
                            "elapsed_time": elapsed_time,
                            "error": "Operation timed out",
                        }

                        logger.error(f"⏱️ {project_key}: Timed out after {elapsed_time:.1f}s")

                        # Update progress
                        if hasattr(progress, "finish_repository"):
                            # Check if progress adapter supports stats parameter
                            if hasattr(progress, "update_stats"):
                                progress.update_stats(
                                    processed=self.processing_stats["processed"],
                                    success=self.processing_stats["success"],
                                    failed=self.processing_stats["failed"],
                                    timeout=self.processing_stats["timeout"],
                                    total=self.processing_stats["total"],
                                )
                                progress.finish_repository(
                                    project_key, success=False, error_message="Timeout"
                                )
                            else:
                                progress.finish_repository(
                                    project_key, success=False, error_message="Timeout"
                                )

                    except Exception as e:
                        self.processing_stats["failed"] += 1
                        self.processing_stats["repositories"][project_key] = {
                            "status": "failed",
                            "elapsed_time": elapsed_time,
                            "error": str(e),
                        }

                        logger.error(f"❌ {project_key}: Error after {elapsed_time:.1f}s - {e}")

                        # Update progress
                        if hasattr(progress, "finish_repository"):
                            # Check if progress adapter supports stats parameter
                            if hasattr(progress, "update_stats"):
                                progress.update_stats(
                                    processed=self.processing_stats["processed"],
                                    success=self.processing_stats["success"],
                                    failed=self.processing_stats["failed"],
                                    timeout=self.processing_stats["timeout"],
                                    total=self.processing_stats["total"],
                                )
                                progress.finish_repository(
                                    project_key, success=False, error_message=str(e)
                                )
                            else:
                                progress.finish_repository(
                                    project_key, success=False, error_message=str(e)
                                )

                    finally:
                        # Update processed counter BEFORE logging and progress updates
                        self.processing_stats["processed"] += 1

                        # Update progress service with actual processing stats
                        if hasattr(progress, "update_stats"):
                            progress.update_stats(
                                processed=self.processing_stats["processed"],
                                success=self.processing_stats["success"],
                                failed=self.processing_stats["failed"],
                                timeout=self.processing_stats["timeout"],
                                total=self.processing_stats["total"],
                            )

                        # Log progress
                        logger.info(
                            f"📊 Progress: {self.processing_stats['processed']}/{self.processing_stats['total']} repositories "
                            f"(✅ {self.processing_stats['success']} | ❌ {self.processing_stats['failed']} | ⏱️ {self.processing_stats['timeout']})"
                        )

        # Final summary
        logger.info("=" * 60)
        logger.info("📈 PARALLEL PROCESSING SUMMARY")
        logger.info(f"   Total repositories: {self.processing_stats['total']}")
        logger.info(f"   Successfully processed: {self.processing_stats['success']}")
        logger.info(f"   Failed: {self.processing_stats['failed']}")
        logger.info(f"   Timed out: {self.processing_stats['timeout']}")
        logger.info("=" * 60)

        return {"results": results, "statistics": self.processing_stats}

    def _process_repository_with_timeout(
        self,
        repo_path: Path,
        project_key: str,
        weeks_back: int = 4,
        branch_patterns: Optional[list[str]] = None,
        jira_integration: Optional[JIRAIntegration] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeout_per_operation: int = Timeouts.DEFAULT_GIT_OPERATION,
    ) -> Optional[dict[str, Any]]:
        """Process a single repository with comprehensive timeout protection.

        Args:
            repo_path: Path to the repository
            project_key: Project identifier
            weeks_back: Number of weeks to analyze
            branch_patterns: Branch patterns to include
            jira_integration: JIRA integration for ticket data
            start_date: Optional explicit start date
            end_date: Optional explicit end date
            timeout_per_operation: Timeout for individual git operations

        Returns:
            Repository processing results or None if failed
        """
        try:
            # Track this repository in progress
            progress = get_progress_service()
            if hasattr(progress, "start_repository"):
                progress.start_repository(project_key, 0)

            logger.info(f"🔍 Processing repository: {project_key} at {repo_path}")

            # Use the regular fetch method but with timeout wrapper active
            with self.git_wrapper.operation_tracker("fetch_repository_data", repo_path):
                result = self.fetch_repository_data(
                    repo_path=repo_path,
                    project_key=project_key,
                    weeks_back=weeks_back,
                    branch_patterns=branch_patterns,
                    jira_integration=jira_integration,
                    progress_callback=None,  # We handle progress at a higher level
                    start_date=start_date,
                    end_date=end_date,
                )

                return result

        except GitOperationTimeout as e:
            logger.error(f"⏱️ Repository {project_key} processing timed out: {e}")
            raise

        except Exception as e:
            logger.error(f"❌ Error processing repository {project_key}: {e}")
            import traceback

            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None
