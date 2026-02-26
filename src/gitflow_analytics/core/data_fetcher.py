"""Data fetcher for collecting raw git commits and ticket data without classification.

This module implements the first step of the two-step fetch/analyze process,
focusing purely on data collection from Git repositories and ticket systems
without performing any LLM-based classification.
"""

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

# THREAD SAFETY: Module-level thread-local storage for repository instances
# Each thread gets its own isolated storage to prevent thread-safety issues
# when GitDataFetcher is called from ThreadPoolExecutor
_thread_local = threading.local()


from .data_fetcher_git import GitFetcherMixin
from .data_fetcher_processing import ProcessingMixin
from .data_fetcher_parallel import ParallelFetcherMixin


class GitDataFetcher(GitFetcherMixin, ProcessingMixin, ParallelFetcherMixin):
    """Fetches raw Git commit data and organizes it by day for efficient batch processing.

    WHY: This class implements the first step of the two-step process by collecting
    all raw data (commits, tickets, correlations) without performing classification.
    This separation enables:
    - Fast data collection without LLM costs
    - Repeatable analysis runs without re-fetching
    - Better batch organization for efficient LLM classification
    """

    def __init__(
        self,
        cache: GitAnalysisCache,
        branch_mapping_rules: Optional[dict[str, list[str]]] = None,
        allowed_ticket_platforms: Optional[list[str]] = None,
        exclude_paths: Optional[list[str]] = None,
        skip_remote_fetch: bool = False,
        exclude_merge_commits: bool = False,
    ) -> None:
        """Initialize the data fetcher.

        Args:
            cache: Git analysis cache instance
            branch_mapping_rules: Rules for mapping branches to projects
            allowed_ticket_platforms: List of allowed ticket platforms
            exclude_paths: List of file paths to exclude from analysis
            skip_remote_fetch: If True, skip git fetch/pull operations
            exclude_merge_commits: Exclude merge commits from filtered line count calculations
        """
        self.cache = cache
        self.skip_remote_fetch = skip_remote_fetch
        self.exclude_merge_commits = exclude_merge_commits
        self.repository_status = {}  # Track status of each repository
        # CRITICAL FIX: Use the same database instance as the cache to avoid session conflicts
        self.database = cache.db
        self.story_point_extractor = build_story_point_extractor()
        self.ticket_extractor = build_ticket_extractor(allowed_platforms=allowed_ticket_platforms)
        self.branch_mapper = build_branch_mapper(branch_mapping_rules)
        self.exclude_paths = exclude_paths or []

        # Log exclusion configuration
        if self.exclude_paths:
            logger.info(
                f"GitDataFetcher initialized with {len(self.exclude_paths)} exclusion patterns:"
            )
            for pattern in self.exclude_paths[:5]:  # Show first 5 patterns
                logger.debug(f"  - {pattern}")
            if len(self.exclude_paths) > 5:
                logger.debug(f"  ... and {len(self.exclude_paths) - 5} more patterns")
        else:
            logger.info("GitDataFetcher initialized with no file exclusions")

        # Initialize identity resolver
        identity_db_path = cache.cache_dir / "identities.db"
        self.identity_resolver = DeveloperIdentityResolver(identity_db_path)

        # Initialize git timeout wrapper for safe operations
        self.git_wrapper = GitTimeoutWrapper(default_timeout=Timeouts.DEFAULT_GIT_OPERATION)

        # Statistics for tracking repository processing
        self.processing_stats = {
            "total": 0,
            "processed": 0,
            "success": 0,
            "failed": 0,
            "timeout": 0,
            "repositories": {},
        }

    def fetch_repository_data(
        self,
        repo_path: Path,
        project_key: str,
        weeks_back: int = 4,
        branch_patterns: Optional[list[str]] = None,
        jira_integration: Optional[JIRAIntegration] = None,
        progress_callback: Optional[callable] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Fetch all data for a repository and organize by day.

        This method collects:
        1. All commits organized by day
        2. All referenced tickets with full metadata
        3. Commit-ticket correlations
        4. Developer identity mappings

        Week-level incremental caching:
        - The date range is split into Monday-aligned ISO weeks.
        - Only weeks absent from WeeklyFetchStatus are fetched from Git.
        - When force=True every week is re-fetched regardless of cache status.

        Args:
            repo_path: Path to the Git repository
            project_key: Project identifier
            weeks_back: Number of weeks to analyze (used only if start_date/end_date not provided)
            branch_patterns: Branch patterns to include
            jira_integration: JIRA integration for ticket data
            progress_callback: Optional callback for progress updates
            start_date: Optional explicit start date (overrides weeks_back calculation)
            end_date: Optional explicit end date (overrides weeks_back calculation)
            force: If True, re-fetch all weeks even if already cached.

        Returns:
            Dictionary containing fetch results and statistics
        """
        logger.debug("🔍 DEBUG: ===== FETCH METHOD CALLED =====")
        logger.info(f"Starting data fetch for project {project_key} at {repo_path}")
        logger.debug(f"🔍 DEBUG: weeks_back={weeks_back}, repo_path={repo_path}")

        # Calculate date range - use explicit dates if provided, otherwise calculate from weeks_back
        if start_date is not None and end_date is not None:
            logger.debug(f"🔍 DEBUG: Using explicit date range: {start_date} to {end_date}")
        else:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(weeks=weeks_back)
            logger.debug(
                f"🔍 DEBUG: Calculated date range from weeks_back: {start_date} to {end_date}"
            )

        # -----------------------------------------------------------------
        # Week-level incremental fetch gate
        # -----------------------------------------------------------------
        required_weeks = self.cache.calculate_weeks(start_date, end_date)
        repo_path_str = str(repo_path)

        if force:
            weeks_to_fetch = required_weeks
            if required_weeks:
                logger.info(
                    f"Force mode: re-fetching all {len(required_weeks)} weeks for {repo_path_str}"
                )
                # Clear existing weekly status so mark_week_cached can upsert cleanly
                self.cache.clear_weekly_cache(repo_path=repo_path_str)
        else:
            weeks_to_fetch = self.cache.get_missing_weeks(repo_path_str, required_weeks)

        if not weeks_to_fetch:
            # Every required week is already cached — return a lightweight summary
            # built from commits already stored in the database.
            cached_count = self._count_cached_commits(repo_path_str, start_date, end_date)
            logger.info(
                f"All {len(required_weeks)} weeks already cached for {repo_path_str} "
                f"({cached_count} commits)"
            )
            return {
                "project_key": project_key,
                "repo_path": repo_path_str,
                "date_range": {"start": start_date, "end": end_date},
                "stats": {
                    "total_commits": cached_count,
                    "stored_commits": cached_count,
                    "storage_success": True,
                    "days_with_commits": 0,
                    "unique_tickets": 0,
                    "correlations_created": 0,
                    "batches_created": 0,
                    "weeks_cached": len(required_weeks),
                    "weeks_fetched": 0,
                    "cache_hit": True,
                },
                "exclusions": {
                    "patterns_applied": len(self.exclude_paths),
                    "enabled": bool(self.exclude_paths),
                },
                "daily_commits": {},
            }

        if len(weeks_to_fetch) < len(required_weeks):
            cached_week_count = len(required_weeks) - len(weeks_to_fetch)
            logger.info(
                f"Incremental fetch: {len(weeks_to_fetch)}/{len(required_weeks)} weeks needed "
                f"({cached_week_count} weeks already cached) for {repo_path_str}"
            )
        else:
            logger.info(f"Fetching all {len(required_weeks)} weeks for {repo_path_str}")

        # Narrow the date range to only the weeks we need to fetch so that
        # _fetch_commits_by_day doesn't re-scan weeks we already have.
        effective_start = min(ws for ws, _ in weeks_to_fetch)
        effective_end = max(we for _, we in weeks_to_fetch)

        # Get progress service for top-level progress tracking
        progress = get_progress_service()

        # Start Rich display for this repository if enabled
        if hasattr(progress, "_use_rich") and progress._use_rich:
            # Count total commits for progress estimation
            try:
                import git

                git.Repo(repo_path)
                # Check if we need to clone or pull
                if not repo_path.exists() or not (repo_path / ".git").exists():
                    logger.info(f"📥 Repository {project_key} needs cloning")
                    progress.start_repository(f"{project_key} (cloning)", 0)
                else:
                    # Rough estimate based on weeks
                    estimated_commits = weeks_back * BatchSizes.COMMITS_PER_WEEK_ESTIMATE
                    progress.start_repository(project_key, estimated_commits)
            except Exception:
                progress.start_repository(project_key, BatchSizes.DEFAULT_PROGRESS_ESTIMATE)

        # Step 1: Collect all commits organized by day with enhanced progress tracking
        logger.debug("🔍 DEBUG: About to fetch commits by day")
        logger.info(f"Fetching commits organized by day for repository: {project_key}")

        # Create top-level progress for this repository
        with progress.progress(
            total=3,  # Three main steps: fetch commits, extract tickets, store data
            description=f"📊 Processing repository: {project_key}",
            unit="steps",
        ) as repo_progress_ctx:
            # Step 1: Fetch commits — use the narrowed date range so we only
            # scan the weeks that are actually missing from the cache.
            progress.set_description(repo_progress_ctx, f"🔍 {project_key}: Fetching commits")
            daily_commits = self._fetch_commits_by_day(
                repo_path,
                project_key,
                effective_start,
                effective_end,
                branch_patterns,
                progress_callback,
            )
            logger.debug(f"🔍 DEBUG: Fetched {len(daily_commits)} days of commits")
            progress.update(repo_progress_ctx)

            # Step 2: Extract and fetch all referenced tickets
            progress.set_description(repo_progress_ctx, f"🎫 {project_key}: Processing tickets")
            logger.debug("🔍 DEBUG: About to extract ticket references")
            logger.info(f"Extracting ticket references for {project_key}...")
            ticket_ids = self._extract_all_ticket_references(daily_commits)
            logger.debug(f"🔍 DEBUG: Extracted {len(ticket_ids)} ticket IDs")

            if jira_integration and ticket_ids:
                logger.info(
                    f"Fetching {len(ticket_ids)} unique tickets from JIRA for {project_key}..."
                )
                self._fetch_detailed_tickets(
                    ticket_ids, jira_integration, project_key, progress_callback
                )

            # Build commit-ticket correlations
            logger.info(f"Building commit-ticket correlations for {project_key}...")
            correlations_created = self._build_commit_ticket_correlations(daily_commits, repo_path)
            progress.update(repo_progress_ctx)

            # Step 3: Store daily commit batches
            progress.set_description(repo_progress_ctx, f"💾 {project_key}: Storing data")
            logger.debug(
                f"🔍 DEBUG: About to store daily batches. Daily commits has {len(daily_commits)} days"
            )
            logger.info("Storing daily commit batches...")
            batches_created = self._store_daily_batches(daily_commits, repo_path, project_key)
            logger.debug(f"🔍 DEBUG: Storage complete. Batches created: {batches_created}")
            progress.update(repo_progress_ctx)

        # CRITICAL FIX: Verify actual storage before reporting success
        # BUG 3 FIX: daily_commits values are now summary dicts; use "count" key.
        session = self.database.get_session()
        try:
            expected_commits = sum(
                v["count"] if isinstance(v, dict) and "count" in v else len(v)
                for v in daily_commits.values()
            )
            verification_result = self._verify_commit_storage(
                session, daily_commits, repo_path, expected_commits
            )
            actual_stored_commits = verification_result["total_found"]
        except Exception as e:
            logger.error(f"❌ Final storage verification failed: {e}")
            # Don't let verification failure break the return, but log it clearly
            actual_stored_commits = 0
        finally:
            session.close()

        # Return summary statistics with ACTUAL stored counts
        # expected_commits already calculated above

        # Calculate exclusion impact summary if exclusions are configured
        exclusion_stats = {
            "patterns_applied": len(self.exclude_paths),
            "enabled": bool(self.exclude_paths),
        }

        if self.exclude_paths and expected_commits > 0:
            # Get aggregate stats from daily_commit_batches
            session = self.database.get_session()
            try:
                batch_stats = (
                    session.query(
                        func.sum(DailyCommitBatch.total_lines_added).label("total_added"),
                        func.sum(DailyCommitBatch.total_lines_deleted).label("total_deleted"),
                    )
                    .filter(
                        DailyCommitBatch.project_key == project_key,
                        DailyCommitBatch.repo_path == str(repo_path),
                    )
                    .first()
                )

                if batch_stats and batch_stats.total_added:
                    total_lines = (batch_stats.total_added or 0) + (batch_stats.total_deleted or 0)
                    exclusion_stats["total_lines_after_filtering"] = total_lines
                    exclusion_stats["lines_added"] = batch_stats.total_added or 0
                    exclusion_stats["lines_deleted"] = batch_stats.total_deleted or 0

                    logger.info(
                        f"📊 Exclusion Impact Summary for {project_key}:\n"
                        f"  - Exclusion patterns applied: {len(self.exclude_paths)}\n"
                        f"  - Total lines after filtering: {total_lines:,}\n"
                        f"  - Lines added: {batch_stats.total_added:,}\n"
                        f"  - Lines deleted: {batch_stats.total_deleted:,}"
                    )
            except Exception as e:
                logger.debug(f"Could not calculate exclusion impact summary: {e}")
            finally:
                session.close()

        # -----------------------------------------------------------------
        # Mark fetched weeks as cached regardless of commit count.
        # Even weeks with 0 commits must be marked so we don't re-fetch
        # them on the next run (the repo simply had no activity that week).
        # -----------------------------------------------------------------
        for week_start, week_end in weeks_to_fetch:
            # Count commits that fall within this specific week
            week_commit_count = self._count_cached_commits(
                repo_path_str,
                week_start,
                week_end,
            )
            try:
                self.cache.mark_week_cached(
                    repo_path=repo_path_str,
                    week_start=week_start,
                    week_end=week_end,
                    commit_count=week_commit_count,
                )
            except Exception as e:
                # Non-fatal: next run will just re-fetch this week
                logger.warning(f"Could not mark week {week_start.date()} as cached: {e}")

        results = {
            "project_key": project_key,
            "repo_path": str(repo_path),
            "date_range": {"start": start_date, "end": end_date},
            "stats": {
                "total_commits": expected_commits,  # What we tried to store
                "stored_commits": actual_stored_commits,  # What was actually stored
                "storage_success": actual_stored_commits == expected_commits,
                "days_with_commits": len(daily_commits),
                "unique_tickets": len(ticket_ids),
                "correlations_created": correlations_created,
                "batches_created": batches_created,
                "weeks_cached": len(required_weeks),
                "weeks_fetched": len(weeks_to_fetch),
                "cache_hit": False,
            },
            "exclusions": exclusion_stats,
            "daily_commits": daily_commits,  # For immediate use if needed
        }

        # Log with actual storage results
        if actual_stored_commits == expected_commits:
            logger.info(
                f"✅ Data fetch completed successfully for {project_key}: {actual_stored_commits}/{expected_commits} commits stored, {len(ticket_ids)} tickets"
            )
            # Finish repository in Rich display with success
            if hasattr(progress, "_use_rich") and progress._use_rich:
                progress.finish_repository(project_key, success=True)
        else:
            logger.error(
                f"⚠️ Data fetch completed with storage issues for {project_key}: {actual_stored_commits}/{expected_commits} commits stored, {len(ticket_ids)} tickets"
            )
            # Finish repository in Rich display with error
            if hasattr(progress, "_use_rich") and progress._use_rich:
                progress.finish_repository(
                    project_key,
                    success=False,
                    error_message=f"Storage issue: {actual_stored_commits}/{expected_commits} commits",
                )

        return results

    def _count_cached_commits(
        self,
        repo_path: str,
        start_date: datetime,
        end_date: datetime,
    ) -> int:
        """Count commits already stored in CachedCommit for a repo and date range.

        WHY: Used to populate the informational commit_count field in
        WeeklyFetchStatus and to produce an accurate summary when all weeks
        are already cached (avoiding a full Git log scan).

        Args:
            repo_path: Canonical string form of the repository path.
            start_date: Inclusive start of the date range (timezone-aware UTC).
            end_date: Inclusive end of the date range (timezone-aware UTC).

        Returns:
            Number of CachedCommit rows within the date range.
        """
        session = self.database.get_session()
        try:
            from sqlalchemy import and_

            from ..models.database import CachedCommit

            count = (
                session.query(CachedCommit)
                .filter(
                    and_(
                        CachedCommit.repo_path == repo_path,
                        CachedCommit.timestamp >= start_date,
                        CachedCommit.timestamp <= end_date,
                    )
                )
                .count()
            )
            return count
        except Exception as e:
            logger.debug(f"Could not count cached commits for {repo_path}: {e}")
            return 0
        finally:
            session.close()

