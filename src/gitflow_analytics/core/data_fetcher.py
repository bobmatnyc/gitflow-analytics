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
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..constants import BatchSizes, Timeouts
from ..extractors.story_points import StoryPointExtractor
from ..extractors.tickets import TicketExtractor
from ..integrations.jira_integration import JIRAIntegration
from ..models.database import (
    CachedCommit,
    CommitTicketCorrelation,
    DailyCommitBatch,
    DetailedTicketData,
)
from ..types import CommitStats
from ..utils.commit_utils import is_merge_commit
from .branch_mapper import BranchToProjectMapper
from .cache import GitAnalysisCache
from .git_timeout_wrapper import GitOperationTimeout, GitTimeoutWrapper, HeartbeatLogger
from .identity import DeveloperIdentityResolver
from .progress import get_progress_service

logger = logging.getLogger(__name__)

# THREAD SAFETY: Module-level thread-local storage for repository instances
# Each thread gets its own isolated storage to prevent thread-safety issues
# when GitDataFetcher is called from ThreadPoolExecutor
_thread_local = threading.local()


class GitDataFetcher:
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
        self.story_point_extractor = StoryPointExtractor()
        self.ticket_extractor = TicketExtractor(allowed_platforms=allowed_ticket_platforms)
        self.branch_mapper = BranchToProjectMapper(branch_mapping_rules)
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
    ) -> dict[str, Any]:
        """Fetch all data for a repository and organize by day.

        This method collects:
        1. All commits organized by day
        2. All referenced tickets with full metadata
        3. Commit-ticket correlations
        4. Developer identity mappings

        Args:
            repo_path: Path to the Git repository
            project_key: Project identifier
            weeks_back: Number of weeks to analyze (used only if start_date/end_date not provided)
            branch_patterns: Branch patterns to include
            jira_integration: JIRA integration for ticket data
            progress_callback: Optional callback for progress updates
            start_date: Optional explicit start date (overrides weeks_back calculation)
            end_date: Optional explicit end date (overrides weeks_back calculation)

        Returns:
            Dictionary containing fetch results and statistics
        """
        logger.info("🔍 DEBUG: ===== FETCH METHOD CALLED =====")
        logger.info(f"Starting data fetch for project {project_key} at {repo_path}")
        logger.info(f"🔍 DEBUG: weeks_back={weeks_back}, repo_path={repo_path}")

        # Calculate date range - use explicit dates if provided, otherwise calculate from weeks_back
        if start_date is not None and end_date is not None:
            logger.info(f"🔍 DEBUG: Using explicit date range: {start_date} to {end_date}")
        else:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(weeks=weeks_back)
            logger.info(
                f"🔍 DEBUG: Calculated date range from weeks_back: {start_date} to {end_date}"
            )

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
        logger.info("🔍 DEBUG: About to fetch commits by day")
        logger.info(f"Fetching commits organized by day for repository: {project_key}")

        # Create top-level progress for this repository
        with progress.progress(
            total=3,  # Three main steps: fetch commits, extract tickets, store data
            description=f"📊 Processing repository: {project_key}",
            unit="steps",
        ) as repo_progress_ctx:
            # Step 1: Fetch commits
            progress.set_description(repo_progress_ctx, f"🔍 {project_key}: Fetching commits")
            daily_commits = self._fetch_commits_by_day(
                repo_path, project_key, start_date, end_date, branch_patterns, progress_callback
            )
            logger.info(f"🔍 DEBUG: Fetched {len(daily_commits)} days of commits")
            progress.update(repo_progress_ctx)

            # Step 2: Extract and fetch all referenced tickets
            progress.set_description(repo_progress_ctx, f"🎫 {project_key}: Processing tickets")
            logger.info("🔍 DEBUG: About to extract ticket references")
            logger.info(f"Extracting ticket references for {project_key}...")
            ticket_ids = self._extract_all_ticket_references(daily_commits)
            logger.info(f"🔍 DEBUG: Extracted {len(ticket_ids)} ticket IDs")

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
            logger.info(
                f"🔍 DEBUG: About to store daily batches. Daily commits has {len(daily_commits)} days"
            )
            logger.info("Storing daily commit batches...")
            batches_created = self._store_daily_batches(daily_commits, repo_path, project_key)
            logger.info(f"🔍 DEBUG: Storage complete. Batches created: {batches_created}")
            progress.update(repo_progress_ctx)

        # CRITICAL FIX: Verify actual storage before reporting success
        session = self.database.get_session()
        try:
            expected_commits = sum(len(commits) for commits in daily_commits.values())
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

    def _fetch_commits_by_day(
        self,
        repo_path: Path,
        project_key: str,
        start_date: datetime,
        end_date: datetime,
        branch_patterns: Optional[list[str]],
        progress_callback: Optional[callable] = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch all commits organized by day with full metadata.

        Returns:
            Dictionary mapping date strings (YYYY-MM-DD) to lists of commit data
        """

        # THREAD SAFETY: Use the module-level thread-local storage to ensure each thread
        # gets its own Repo instance. This prevents thread-safety issues when called from ThreadPoolExecutor

        # Set environment variables to prevent ANY password prompts before opening repo
        original_env = {}
        env_vars = {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "/bin/echo",  # Use full path to echo
            "SSH_ASKPASS": "/bin/echo",
            "GCM_INTERACTIVE": "never",
            "GIT_SSH_COMMAND": "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o PasswordAuthentication=no",
            "DISPLAY": "",
            "GIT_CREDENTIAL_HELPER": "",  # Disable credential helper
            "GCM_PROVIDER": "none",  # Disable Git Credential Manager
            "GIT_CREDENTIALS": "",  # Clear any cached credentials
            "GIT_CONFIG_NOSYSTEM": "1",  # Don't use system config
        }

        # Save original environment and set our values
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            # THREAD SAFETY: Create a fresh Repo instance for this thread
            # Do NOT reuse Repo instances across threads
            from git import Repo

            # Check for security issues in repository configuration
            self._check_repository_security(repo_path, project_key)

            # When skip_remote_fetch is enabled, use a more restricted repository access
            if self.skip_remote_fetch:
                # Use a special git configuration that completely disables credential helpers
                import tempfile

                # Create a secure temporary directory for our config
                temp_dir = tempfile.mkdtemp(prefix="gitflow_")

                # Create a temporary git config that disables all authentication
                tmp_config_path = os.path.join(temp_dir, ".gitconfig")
                with open(tmp_config_path, "w") as tmp_config:
                    tmp_config.write("[credential]\n")
                    tmp_config.write("    helper = \n")
                    tmp_config.write("[core]\n")
                    tmp_config.write("    askpass = \n")

                # Set GIT_CONFIG to use our temporary config
                os.environ["GIT_CONFIG_GLOBAL"] = tmp_config_path
                os.environ["GIT_CONFIG_SYSTEM"] = "/dev/null"

                # Store temp_dir in thread-local storage for cleanup
                _thread_local.temp_dir = temp_dir

                try:
                    # Open repository with our restricted configuration
                    # THREAD SAFETY: Each thread gets its own Repo instance
                    repo = Repo(repo_path)
                finally:
                    # Clean up temporary config directory
                    try:
                        import shutil

                        if hasattr(_thread_local, "temp_dir"):
                            shutil.rmtree(_thread_local.temp_dir, ignore_errors=True)
                            delattr(_thread_local, "temp_dir")
                    except OSError as e:
                        # Log cleanup failures but don't fail the operation
                        logger.debug(f"Failed to clean up temp directory for {project_key}: {e}")
                    except Exception as e:
                        logger.warning(
                            f"Unexpected error during temp cleanup for {project_key}: {e}"
                        )

                try:
                    # Configure git to never prompt for credentials
                    with repo.config_writer() as git_config:
                        git_config.set_value("core", "askpass", "")
                        git_config.set_value("credential", "helper", "")
                except Exception as e:
                    logger.debug(f"Could not update git config: {e}")

                # Note: We can't monkey-patch remotes as it's a property without setter
                # The skip_remote_fetch flag will prevent remote operations elsewhere

                logger.debug(
                    f"Opened repository {project_key} in offline mode (skip_remote_fetch=true)"
                )
            else:
                # THREAD SAFETY: Each thread gets its own Repo instance
                repo = Repo(repo_path)
            # Track repository status
            self.repository_status[project_key] = {
                "path": str(repo_path),
                "remote_update": "skipped" if self.skip_remote_fetch else "pending",
                "authentication_issues": False,
                "error": None,
            }

            # Update repository from remote before analysis
            if not self.skip_remote_fetch:
                logger.info(f"📥 Updating repository {project_key} from remote...")

            update_success = self._update_repository(repo)
            if not self.skip_remote_fetch:
                self.repository_status[project_key]["remote_update"] = (
                    "success" if update_success else "failed"
                )
                if not update_success:
                    logger.warning(
                        f"⚠️ {project_key}: Continuing with local repository state (remote update failed)"
                    )

        except Exception as e:
            logger.error(f"Failed to open repository {project_key} at {repo_path}: {e}")
            self.repository_status[project_key] = {
                "path": str(repo_path),
                "remote_update": "error",
                "authentication_issues": "authentication" in str(e).lower()
                or "password" in str(e).lower(),
                "error": str(e),
            }
            # Restore original environment variables before returning
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            return {}

        # Get branches to analyze
        branches_to_analyze = self._get_branches_to_analyze(repo, branch_patterns)

        if not branches_to_analyze:
            logger.warning(
                f"No accessible branches found in repository {project_key} at {repo_path}"
            )
            # Restore original environment variables before returning
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            return {}

        logger.info(f"🌿 {project_key}: Analyzing branches: {branches_to_analyze}")

        # Calculate days to process
        current_date = start_date.date()
        end_date_only = end_date.date()
        days_to_process = []
        while current_date <= end_date_only:
            days_to_process.append(current_date)
            current_date += timedelta(days=1)

        logger.info(
            f"Processing {len(days_to_process)} days from {start_date.date()} to {end_date.date()}"
        )

        # Get progress service for nested progress tracking
        progress = get_progress_service()

        # Dictionary to store commits by day
        daily_commits = {}
        all_commit_hashes = set()  # Track all hashes for deduplication

        # Count total commits first for Rich display
        try:
            for branch_name in branches_to_analyze[:1]:  # Sample from first branch
                sample_commits = list(
                    repo.iter_commits(branch_name, since=start_date, until=end_date)
                )
                # Estimate based on first branch (multiply by number of branches for rough estimate)
                len(sample_commits) * len(branches_to_analyze)
                break
        except GitOperationTimeout:
            logger.warning(
                f"Timeout while sampling commits for {project_key}, using default estimate"
            )
            len(days_to_process) * BatchSizes.COMMITS_PER_WEEK_ESTIMATE  # Default estimate
        except Exception as e:
            logger.debug(f"Could not sample commits for {project_key}: {e}, using default estimate")
            len(days_to_process) * BatchSizes.COMMITS_PER_WEEK_ESTIMATE  # Default estimate

        # Update repository in Rich display with estimated commit count
        if hasattr(progress, "_use_rich") and progress._use_rich:
            progress.update_repository(project_key, 0, 0.0)

        # Create nested progress for day-by-day processing
        with progress.progress(
            total=len(days_to_process),
            description=f"📅 Fetching commits for repository: {project_key}",
            unit="days",
            nested=True,
        ) as day_progress_ctx:
            for day_date in days_to_process:
                # Update description to show current repository and day clearly
                day_str = day_date.strftime("%Y-%m-%d")
                progress.set_description(day_progress_ctx, f"🔍 {project_key}: Analyzing {day_str}")

                # Calculate day boundaries
                day_start = datetime.combine(day_date, datetime.min.time(), tzinfo=timezone.utc)
                day_end = datetime.combine(day_date, datetime.max.time(), tzinfo=timezone.utc)

                day_commits = []
                commits_found_today = 0

                # Process each branch for this specific day
                for branch_name in branches_to_analyze:
                    try:
                        # Fetch commits for this specific day and branch with timeout protection
                        def fetch_branch_commits(
                            branch: str = branch_name,
                            start: datetime = day_start,
                            end: datetime = day_end,
                        ) -> list[Any]:
                            """Fetch commits for a specific branch and day range.

                            Returns:
                                List of GitPython commit objects
                            """
                            return list(
                                repo.iter_commits(branch, since=start, until=end, reverse=False)
                            )

                        # Use timeout wrapper to prevent hanging on iter_commits
                        try:
                            branch_commits = self.git_wrapper.run_with_timeout(
                                fetch_branch_commits,
                                timeout=Timeouts.GIT_BRANCH_ITERATION,
                                operation_name=f"iter_commits_{branch_name}_{day_str}",
                            )
                        except GitOperationTimeout:
                            logger.warning(
                                f"⏱️ Timeout fetching commits for branch {branch_name} on {day_str}, skipping"
                            )
                            continue

                        for commit in branch_commits:
                            # Skip if we've already processed this commit
                            if commit.hexsha in all_commit_hashes:
                                continue

                            # Extract commit data with full metadata
                            commit_data = self._extract_commit_data(
                                commit, branch_name, project_key, repo_path
                            )
                            if commit_data:
                                day_commits.append(commit_data)
                                all_commit_hashes.add(commit.hexsha)
                                commits_found_today += 1

                    except GitOperationTimeout as e:
                        logger.warning(
                            f"⏱️ Timeout processing branch {branch_name} for day {day_str}: {e}"
                        )
                        continue
                    except Exception as e:
                        logger.warning(
                            f"Error processing branch {branch_name} for day {day_str}: {e}"
                        )
                        continue

                # Store commits for this day if any were found
                if day_commits:
                    # Sort commits by timestamp
                    day_commits.sort(key=lambda c: c["timestamp"])
                    daily_commits[day_str] = day_commits

                    # Incremental caching - store commits for this day immediately
                    self._store_day_commits_incremental(
                        repo_path, day_str, day_commits, project_key
                    )

                    logger.debug(f"Found {commits_found_today} commits on {day_str}")

                # Update progress callback if provided
                if progress_callback:
                    progress_callback(f"Processed {day_str}: {commits_found_today} commits")

                # Update progress bar
                progress.update(day_progress_ctx)

                # Update Rich display with current commit count and speed
                if hasattr(progress, "_use_rich") and progress._use_rich:
                    total_processed = len(all_commit_hashes)
                    # Calculate speed (commits per second) based on elapsed time
                    import time

                    if not hasattr(self, "_fetch_start_time"):
                        self._fetch_start_time = time.time()
                    elapsed = time.time() - self._fetch_start_time
                    speed = total_processed / elapsed if elapsed > 0 else 0
                    progress.update_repository(project_key, total_processed, speed)

        total_commits = sum(len(commits) for commits in daily_commits.values())
        logger.info(f"Collected {total_commits} unique commits across {len(daily_commits)} days")

        # Restore original environment variables
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        return daily_commits

    def _extract_commit_data(
        self, commit: git.Commit, branch_name: str, project_key: str, repo_path: Path
    ) -> Optional[dict[str, Any]]:
        """Extract comprehensive data from a Git commit.

        Returns:
            Dictionary containing all commit metadata needed for classification
        """
        try:
            # Basic commit information
            commit_data = {
                "commit_hash": commit.hexsha,
                "commit_hash_short": commit.hexsha[:7],
                "message": commit.message.strip(),
                "author_name": commit.author.name,
                "author_email": commit.author.email,
                "timestamp": datetime.fromtimestamp(commit.committed_date, tz=timezone.utc),
                "branch": branch_name,
                "project_key": project_key,
                "repo_path": str(repo_path),
                "is_merge": len(commit.parents) > 1,  # Match the original analyzer behavior
            }

            # Calculate file changes
            try:
                # Compare with first parent or empty tree for initial commit
                diff = commit.parents[0].diff(commit) if commit.parents else commit.diff(None)

                # Get file paths with filtering
                files_changed = []
                for diff_item in diff:
                    file_path = diff_item.a_path or diff_item.b_path
                    if file_path and not self._should_exclude_file(file_path):
                        files_changed.append(file_path)

                # Use reliable git numstat command for accurate line counts
                line_stats = self._calculate_commit_stats(commit)
                total_insertions = line_stats["insertions"]
                total_deletions = line_stats["deletions"]
                raw_insertions = line_stats.get("raw_insertions", total_insertions)
                raw_deletions = line_stats.get("raw_deletions", total_deletions)

                commit_data.update(
                    {
                        "files_changed": files_changed,
                        "files_changed_count": len(files_changed),
                        "lines_added": total_insertions,  # Filtered counts for backward compatibility
                        "lines_deleted": total_deletions,
                        "filtered_insertions": total_insertions,  # Explicitly filtered counts
                        "filtered_deletions": total_deletions,
                        "raw_insertions": raw_insertions,  # Raw unfiltered counts
                        "raw_deletions": raw_deletions,
                    }
                )

            except Exception as e:
                logger.debug(f"Error calculating changes for commit {commit.hexsha}: {e}")
                commit_data.update(
                    {
                        "files_changed": [],
                        "files_changed_count": 0,
                        "lines_added": 0,
                        "lines_deleted": 0,
                        "filtered_insertions": 0,
                        "filtered_deletions": 0,
                        "raw_insertions": 0,
                        "raw_deletions": 0,
                    }
                )

            # Extract story points
            story_points = self.story_point_extractor.extract_from_text(commit_data["message"])
            commit_data["story_points"] = story_points

            # Extract ticket references
            ticket_refs_data = self.ticket_extractor.extract_from_text(commit_data["message"])
            # Convert to list of ticket IDs for compatibility
            # Fix: Use 'id' field instead of 'ticket_id' field from extractor output
            ticket_refs = [ref_data["id"] for ref_data in ticket_refs_data]
            commit_data["ticket_references"] = ticket_refs

            # Resolve developer identity
            canonical_id = self.identity_resolver.resolve_developer(
                commit_data["author_name"], commit_data["author_email"]
            )
            commit_data["canonical_developer_id"] = canonical_id

            return commit_data

        except Exception as e:
            logger.error(f"Error extracting data for commit {commit.hexsha}: {e}")
            return None

    def _should_exclude_file(self, file_path: str) -> bool:
        """Check if a file should be excluded based on exclude patterns."""
        return any(self._matches_glob_pattern(file_path, pattern) for pattern in self.exclude_paths)

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
        import fnmatch
        import re
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
                return fnmatch.fnmatch(path.name, suffix_pattern)

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
                # Try matching the full path first
                if fnmatch.fnmatch(filepath, pattern):
                    return True
                # Also try matching just the filename for simple patterns
                # This allows "package-lock.json" to match "src/package-lock.json"
                return fnmatch.fnmatch(path.name, pattern)
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
        import fnmatch
        from pathlib import PurePath

        # Split pattern by ** to handle each segment
        parts = pattern.split("**")

        # Validate that we have actual segments
        if not parts:
            return False

        # Convert filepath to parts for easier matching
        path_parts = list(PurePath(filepath).parts)

        # Start matching from the beginning
        path_index = 0

        for i, part in enumerate(parts):
            if not part:
                # Empty part (e.g., from leading or trailing **)
                if i == 0 or i == len(parts) - 1:
                    # Leading or trailing ** - continue
                    continue
                # Middle empty part (consecutive **) - match any number of path components
                continue

            # Clean the part (remove leading/trailing slashes)
            part = part.strip("/")

            if not part:
                continue

            # Find where this part matches in the remaining path
            found = False
            for j in range(path_index, len(path_parts)):
                # Check if the current path part matches the pattern part
                if "/" in part:
                    # Part contains multiple path components
                    sub_parts = part.split("/")
                    if j + len(sub_parts) <= len(path_parts) and all(
                        fnmatch.fnmatch(path_parts[j + k], sub_parts[k])
                        for k in range(len(sub_parts))
                    ):
                        path_index = j + len(sub_parts)
                        found = True
                        break
                else:
                    # Single component part
                    if fnmatch.fnmatch(path_parts[j], part):
                        path_index = j + 1
                        found = True
                        break

            if not found and part:
                # Required part not found in path
                return False

        return True

    def _get_branches_to_analyze(
        self, repo: Any, branch_patterns: Optional[list[str]]
    ) -> list[str]:
        """Get list of branches to analyze based on patterns.

        WHY: Robust branch detection that handles missing remotes, missing default branches,
        and provides good fallback behavior. When no patterns specified, analyzes ALL branches
        to capture the complete development picture.

        DESIGN DECISION:
        - When no patterns: analyze ALL accessible branches (not just main)
        - When patterns specified: match against those patterns only
        - Handle missing remotes gracefully
        - Skip remote tracking branches to avoid duplicates
        - Use actual branch existence checking rather than assuming branches exist

        THREAD SAFETY: This method is thread-safe as it doesn't modify shared state
        and works with a repo instance passed as a parameter.
        """
        # Collect all available branches (local branches preferred)
        available_branches = []

        # First, try local branches
        try:
            # THREAD SAFETY: Create a new list to avoid sharing references
            local_branches = list([branch.name for branch in repo.branches])
            available_branches.extend(local_branches)
            logger.debug(f"Found local branches: {local_branches}")
        except Exception as e:
            logger.debug(f"Error getting local branches: {e}")

        # If we have remotes, also consider remote branches (keep full remote reference)
        # Skip remote branch checking if skip_remote_fetch is enabled to avoid auth prompts
        if not self.skip_remote_fetch:
            try:
                if repo.remotes and hasattr(repo.remotes, "origin"):
                    # CRITICAL FIX: Keep full remote reference (origin/branch-name) for accessibility testing
                    # Remote branches need the full reference to work with iter_commits()
                    # THREAD SAFETY: Create a new list to avoid sharing references
                    remote_branches = list(
                        [
                            ref.name  # Keep full "origin/branch-name" format
                            for ref in repo.remotes.origin.refs
                            if not ref.name.endswith("HEAD")  # Skip HEAD ref
                        ]
                    )
                    # Add remote branches with full reference (origin/branch-name)
                    # Extract short name only for duplicate checking against local branches
                    for branch_ref in remote_branches:
                        short_name = branch_ref.replace("origin/", "")
                        # Only add if we don't have this branch locally
                        if short_name not in available_branches:
                            available_branches.append(branch_ref)  # Store full reference
                    logger.debug(f"Found remote branches: {remote_branches}")
            except Exception as e:
                logger.debug(f"Error getting remote branches (may require authentication): {e}")
                # Continue with local branches only
        else:
            logger.debug("Skipping remote branch enumeration (skip_remote_fetch=true)")

        # If no branches found, fallback to trying common names directly
        if not available_branches:
            logger.warning("No branches found via normal detection, falling back to common names")
            available_branches = ["main", "master", "develop", "dev"]

        # Filter branches based on patterns if provided
        if branch_patterns:
            import fnmatch

            matching_branches = []
            for pattern in branch_patterns:
                matching = [
                    branch for branch in available_branches if fnmatch.fnmatch(branch, pattern)
                ]
                matching_branches.extend(matching)
            # Remove duplicates while preserving order
            branches_to_test = list(dict.fromkeys(matching_branches))
        else:
            # No patterns specified - analyze ALL branches for complete coverage
            branches_to_test = available_branches
            logger.info(
                f"No branch patterns specified - will analyze all {len(branches_to_test)} branches"
            )

        # Test that branches are actually accessible
        accessible_branches = []
        for branch in branches_to_test:
            try:
                # THREAD SAFETY: Use iterator without storing intermediate results
                next(iter(repo.iter_commits(branch, max_count=1)), None)
                accessible_branches.append(branch)
            except Exception as e:
                logger.debug(f"Branch {branch} not accessible: {e}")

        if not accessible_branches:
            # Last resort: try to find ANY working branch
            logger.warning("No accessible branches found from patterns/default, trying fallback")
            main_branches = ["main", "master", "develop", "dev"]
            for branch in main_branches:
                if branch in available_branches:
                    try:
                        next(iter(repo.iter_commits(branch, max_count=1)), None)
                        logger.info(f"Using fallback main branch: {branch}")
                        return [branch]
                    except Exception:
                        continue

            # Try any available branch
            for branch in available_branches:
                try:
                    next(iter(repo.iter_commits(branch, max_count=1)), None)
                    logger.info(f"Using fallback branch: {branch}")
                    return [branch]
                except Exception:
                    continue

            logger.warning("No accessible branches found")
            return []

        logger.info(f"Will analyze {len(accessible_branches)} branches: {accessible_branches}")
        return accessible_branches

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
        # Skip remote operations if configured
        if self.skip_remote_fetch:
            logger.info("🚫 Skipping remote fetch (skip_remote_fetch=true)")
            return True

        # Check for stale repository (last fetch > 1 hour ago)
        self._check_repository_staleness(repo)

        try:
            # Check if we have remotes without triggering authentication
            has_remotes = False
            try:
                has_remotes = bool(repo.remotes)
            except Exception as e:
                logger.debug(f"Could not check for remotes (may require authentication): {e}")
                return True  # Continue with local analysis

            if has_remotes:
                logger.info("Fetching latest changes from remote")

                # Use our timeout wrapper for safe git operations
                repo_path = Path(repo.working_dir)

                # Try to fetch with timeout protection
                fetch_success = self.git_wrapper.fetch_with_timeout(
                    repo_path, timeout=Timeouts.GIT_FETCH
                )

                if not fetch_success:
                    # Mark this repository as having authentication issues if applicable
                    if hasattr(self, "repository_status"):
                        for key in self.repository_status:
                            if repo.working_dir.endswith(key) or key in repo.working_dir:
                                self.repository_status[key]["remote_update"] = "failed"
                                break

                    # Explicit warning to user about stale data
                    logger.warning(
                        f"❌ Failed to fetch updates for {repo_path.name}. "
                        f"Analysis will use potentially stale local data. "
                        f"Check authentication or network connectivity."
                    )
                    return False
                else:
                    # Explicit success confirmation
                    logger.info(f"✅ Successfully fetched updates for {repo_path.name}")

                # Only try to pull if not in detached HEAD state
                if not repo.head.is_detached:
                    current_branch = repo.active_branch
                    tracking = current_branch.tracking_branch()
                    if tracking:
                        # Pull latest changes using timeout wrapper
                        pull_success = self.git_wrapper.pull_with_timeout(
                            repo_path, timeout=Timeouts.GIT_PULL
                        )
                        if pull_success:
                            logger.debug(f"Pulled latest changes for {current_branch.name}")
                        else:
                            logger.warning("Git pull failed, continuing with fetched state")
                            return False
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

    def _check_repository_staleness(self, repo) -> None:
        """Check if repository hasn't been fetched recently and warn user.

        Args:
            repo: GitPython Repo object
        """
        try:
            repo_path = Path(repo.working_dir)
            fetch_head_path = repo_path / ".git" / "FETCH_HEAD"

            if fetch_head_path.exists():
                # Get last fetch time from FETCH_HEAD modification time
                last_fetch_time = datetime.fromtimestamp(
                    fetch_head_path.stat().st_mtime, tz=timezone.utc
                )
                now = datetime.now(timezone.utc)
                hours_since_fetch = (now - last_fetch_time).total_seconds() / 3600

                if hours_since_fetch > 1:
                    logger.warning(
                        f"⏰ Repository {repo_path.name} last fetched {hours_since_fetch:.1f} hours ago. "
                        f"Data may be stale."
                    )
            else:
                logger.warning(
                    f"⚠️ Repository {repo_path.name} has never been fetched. "
                    f"Will attempt to fetch now."
                )
        except Exception as e:
            logger.debug(f"Could not check repository staleness: {e}")

    def _check_repository_security(self, repo_path: Path, project_key: str) -> None:
        """Check for security issues in repository configuration.

        Warns about:
        - Exposed tokens in remote URLs
        - Insecure credential storage
        """

        try:
            # Check for tokens in remote URLs
            result = subprocess.run(
                ["git", "remote", "-v"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=Timeouts.GIT_CONFIG,
                env={"GIT_TERMINAL_PROMPT": "0"},
            )

            if result.returncode == 0:
                output = result.stdout
                # Check for various token patterns in URLs
                token_patterns = [
                    r"https://[^@]*@",  # Any HTTPS URL with embedded credentials
                    r"ghp_[a-zA-Z0-9]+",  # GitHub Personal Access Token
                    r"ghs_[a-zA-Z0-9]+",  # GitHub Server Token
                    r"github_pat_[a-zA-Z0-9]+",  # New GitHub PAT format
                ]

                for pattern in token_patterns:
                    import re

                    if re.search(pattern, output):
                        logger.warning(
                            f"⚠️  SECURITY WARNING for {project_key}: "
                            f"Repository appears to have credentials in remote URL. "
                            f"This is a security risk! Consider using: "
                            f"1) GitHub CLI (gh auth login), "
                            f"2) SSH keys, or "
                            f"3) Git credential manager instead."
                        )
                        break
        except AttributeError as e:
            # Repository might not have remotes attribute (e.g., in tests or unusual repo structures)
            logger.debug(f"Could not check remote URLs for security scan: {e}")
        except Exception as e:
            # Don't fail analysis due to security check, but log unexpected errors
            logger.warning(f"Error during credential security check: {e}")

    def get_repository_status_summary(self) -> dict[str, Any]:
        """Get a summary of repository fetch status.

        Returns:
            Dictionary with status summary including any repositories with issues
        """
        summary = {
            "total_repositories": len(self.repository_status),
            "successful_updates": 0,
            "failed_updates": 0,
            "skipped_updates": 0,
            "authentication_issues": [],
            "errors": [],
        }

        for project_key, status in self.repository_status.items():
            if status["remote_update"] == "success":
                summary["successful_updates"] += 1
            elif status["remote_update"] == "failed":
                summary["failed_updates"] += 1
                if status.get("authentication_issues"):
                    summary["authentication_issues"].append(project_key)
            elif status["remote_update"] == "skipped":
                summary["skipped_updates"] += 1
            elif status["remote_update"] == "error":
                summary["errors"].append(
                    {"repository": project_key, "error": status.get("error", "Unknown error")}
                )

        return summary

    def _extract_all_ticket_references(
        self, daily_commits: dict[str, list[dict[str, Any]]]
    ) -> set[str]:
        """Extract all unique ticket IDs from commits."""
        ticket_ids = set()

        for day_commits in daily_commits.values():
            for commit in day_commits:
                ticket_refs = commit.get("ticket_references", [])
                ticket_ids.update(ticket_refs)

        logger.info(f"Found {len(ticket_ids)} unique ticket references")
        return ticket_ids

    def _fetch_detailed_tickets(
        self,
        ticket_ids: set[str],
        jira_integration: JIRAIntegration,
        project_key: str,
        progress_callback: Optional[callable] = None,
    ) -> None:
        """Fetch detailed ticket information and store in database."""
        session = self.database.get_session()

        try:
            # Check which tickets we already have
            existing_tickets = (
                session.query(DetailedTicketData)
                .filter(
                    DetailedTicketData.ticket_id.in_(ticket_ids),
                    DetailedTicketData.platform == "jira",
                )
                .all()
            )

            existing_ids = {ticket.ticket_id for ticket in existing_tickets}
            tickets_to_fetch = ticket_ids - existing_ids

            if not tickets_to_fetch:
                logger.info("All tickets already cached")
                return

            logger.info(f"Fetching {len(tickets_to_fetch)} new tickets")

            # Fetch tickets in batches
            batch_size = BatchSizes.TICKET_FETCH
            tickets_list = list(tickets_to_fetch)

            # Use centralized progress service
            progress = get_progress_service()

            with progress.progress(
                total=len(tickets_list), description="Fetching tickets", unit="tickets"
            ) as ctx:
                for i in range(0, len(tickets_list), batch_size):
                    batch = tickets_list[i : i + batch_size]

                    for ticket_id in batch:
                        try:
                            # Fetch ticket from JIRA
                            issue_data = jira_integration.get_issue(ticket_id)

                            if issue_data:
                                # Create detailed ticket record
                                detailed_ticket = self._create_detailed_ticket_record(
                                    issue_data, project_key, "jira"
                                )
                                session.add(detailed_ticket)

                        except Exception as e:
                            logger.warning(f"Failed to fetch ticket {ticket_id}: {e}")

                        progress.update(ctx, 1)

                        if progress_callback:
                            progress_callback(f"Fetched ticket {ticket_id}")

                    # Commit batch to database
                    session.commit()

            logger.info(f"Successfully fetched {len(tickets_to_fetch)} tickets")

        except Exception as e:
            logger.error(f"Error fetching detailed tickets: {e}")
            session.rollback()
        finally:
            session.close()

    def _create_detailed_ticket_record(
        self, issue_data: dict[str, Any], project_key: str, platform: str
    ) -> DetailedTicketData:
        """Create a detailed ticket record from JIRA issue data."""
        # Extract classification hints from issue type and labels
        classification_hints = []

        issue_type = issue_data.get("issue_type", "").lower()
        if "bug" in issue_type or "defect" in issue_type:
            classification_hints.append("bug_fix")
        elif "story" in issue_type or "feature" in issue_type:
            classification_hints.append("feature")
        elif "task" in issue_type:
            classification_hints.append("maintenance")

        # Extract business domain from labels or summary
        business_domain = None
        labels = issue_data.get("labels", [])
        for label in labels:
            if any(keyword in label.lower() for keyword in ["frontend", "backend", "ui", "api"]):
                business_domain = label.lower()
                break

        # Create the record
        return DetailedTicketData(
            platform=platform,
            ticket_id=issue_data["key"],
            project_key=project_key,
            title=issue_data.get("summary", ""),
            description=issue_data.get("description", ""),
            summary=issue_data.get("summary", "")[:500],  # Truncated summary
            ticket_type=issue_data.get("issue_type", ""),
            status=issue_data.get("status", ""),
            priority=issue_data.get("priority", ""),
            labels=labels,
            assignee=issue_data.get("assignee", ""),
            reporter=issue_data.get("reporter", ""),
            created_at=issue_data.get("created"),
            updated_at=issue_data.get("updated"),
            resolved_at=issue_data.get("resolved"),
            story_points=issue_data.get("story_points"),
            classification_hints=classification_hints,
            business_domain=business_domain,
            platform_data=issue_data,  # Store full JIRA data
        )

    def _build_commit_ticket_correlations(
        self, daily_commits: dict[str, list[dict[str, Any]]], repo_path: Path
    ) -> int:
        """Build and store commit-ticket correlations."""
        session = self.database.get_session()
        correlations_created = 0

        try:
            for day_commits in daily_commits.values():
                for commit in day_commits:
                    commit_hash = commit["commit_hash"]
                    ticket_refs = commit.get("ticket_references", [])

                    for ticket_id in ticket_refs:
                        try:
                            # Create correlation record
                            correlation = CommitTicketCorrelation(
                                commit_hash=commit_hash,
                                repo_path=str(repo_path),
                                ticket_id=ticket_id,
                                platform="jira",  # Assuming JIRA for now
                                project_key=commit["project_key"],
                                correlation_type="direct",
                                confidence=1.0,
                                extracted_from="commit_message",
                                matching_pattern=None,  # Could add pattern detection
                            )

                            # Check if correlation already exists
                            existing = (
                                session.query(CommitTicketCorrelation)
                                .filter(
                                    CommitTicketCorrelation.commit_hash == commit_hash,
                                    CommitTicketCorrelation.repo_path == str(repo_path),
                                    CommitTicketCorrelation.ticket_id == ticket_id,
                                    CommitTicketCorrelation.platform == "jira",
                                )
                                .first()
                            )

                            if not existing:
                                session.add(correlation)
                                correlations_created += 1

                        except Exception as e:
                            logger.warning(
                                f"Failed to create correlation for {commit_hash}-{ticket_id}: {e}"
                            )

            session.commit()
            logger.info(f"Created {correlations_created} commit-ticket correlations")

        except Exception as e:
            logger.error(f"Error building correlations: {e}")
            session.rollback()
        finally:
            session.close()

        return correlations_created

    def _store_daily_batches(
        self, daily_commits: dict[str, list[dict[str, Any]]], repo_path: Path, project_key: str
    ) -> int:
        """Store daily commit batches for efficient retrieval using bulk operations.

        WHY: Enhanced to use bulk operations from the cache layer for significantly
        better performance when storing large numbers of commits.
        """
        session = self.database.get_session()
        batches_created = 0
        commits_stored = 0
        expected_commits = 0

        try:
            total_commits = sum(len(commits) for commits in daily_commits.values())
            logger.info(f"🔍 DEBUG: Storing {total_commits} commits from {len(daily_commits)} days")
            logger.info(
                f"🔍 DEBUG: Daily commits keys: {list(daily_commits.keys())[:5]}"
            )  # First 5 dates

            # Track dates to process for efficient batch handling
            dates_to_process = [
                datetime.strptime(date_str, "%Y-%m-%d").date() for date_str in daily_commits
            ]
            logger.info(f"🔍 DEBUG: Processing {len(dates_to_process)} dates for daily batches")

            # Pre-load existing batches to avoid constraint violations during processing
            existing_batches_map = {}
            for date_str in daily_commits:
                # Convert to datetime instead of date to match database storage
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                existing_batch = (
                    session.query(DailyCommitBatch)
                    .filter(
                        DailyCommitBatch.date == date_obj,
                        DailyCommitBatch.project_key == project_key,
                        DailyCommitBatch.repo_path == str(repo_path),
                    )
                    .first()
                )
                if existing_batch:
                    existing_batches_map[date_str] = existing_batch
                    logger.info(
                        f"🔍 DEBUG: Found existing batch for {date_str}: ID={existing_batch.id}"
                    )
                else:
                    logger.info(f"🔍 DEBUG: No existing batch found for {date_str}")

            # Collect all commits for bulk operations
            all_commits_to_store = []
            commit_hashes_to_check = []

            # First, collect all commit hashes to check existence in bulk
            for _date_str, commits in daily_commits.items():
                for commit in commits:
                    commit_hashes_to_check.append(commit["commit_hash"])

            # Use bulk_exists to check which commits already exist
            existing_commits_map = self.cache.bulk_exists(str(repo_path), commit_hashes_to_check)

            # Disable autoflush to prevent premature batch creation during commit storage
            with session.no_autoflush:
                for date_str, commits in daily_commits.items():
                    if not commits:
                        continue

                    logger.info(f"🔍 DEBUG: Processing {len(commits)} commits for {date_str}")

                    # Prepare commits for bulk storage
                    commits_to_store_this_date = []
                    for commit in commits:
                        # Check if commit already exists using bulk check results
                        if not existing_commits_map.get(commit["commit_hash"], False):
                            # Transform commit data to cache format
                            cache_format_commit = {
                                "hash": commit["commit_hash"],
                                "author_name": commit.get("author_name", ""),
                                "author_email": commit.get("author_email", ""),
                                "message": commit.get("message", ""),
                                "timestamp": commit["timestamp"],
                                "branch": commit.get("branch", "main"),
                                "is_merge": commit.get("is_merge", False),
                                "files_changed_count": commit.get("files_changed_count", 0),
                                # Store raw unfiltered values in insertions/deletions
                                "insertions": commit.get(
                                    "raw_insertions", commit.get("lines_added", 0)
                                ),
                                "deletions": commit.get(
                                    "raw_deletions", commit.get("lines_deleted", 0)
                                ),
                                # Store filtered values separately
                                "filtered_insertions": commit.get(
                                    "filtered_insertions", commit.get("lines_added", 0)
                                ),
                                "filtered_deletions": commit.get(
                                    "filtered_deletions", commit.get("lines_deleted", 0)
                                ),
                                "story_points": commit.get("story_points"),
                                "ticket_references": commit.get("ticket_references", []),
                            }
                            commits_to_store_this_date.append(cache_format_commit)
                            all_commits_to_store.append(cache_format_commit)
                            expected_commits += 1
                        else:
                            logger.debug(
                                f"Commit {commit['commit_hash'][:7]} already exists in database"
                            )

                    logger.info(
                        f"🔍 DEBUG: Prepared {len(commits_to_store_this_date)} new commits for {date_str}"
                    )

                    # Calculate batch statistics
                    total_files = sum(commit.get("files_changed_count", 0) for commit in commits)
                    total_additions = sum(commit.get("lines_added", 0) for commit in commits)
                    total_deletions = sum(commit.get("lines_deleted", 0) for commit in commits)

                    # Get unique developers and tickets for this day
                    active_devs = list(
                        set(commit.get("canonical_developer_id", "") for commit in commits)
                    )
                    unique_tickets = []
                    for commit in commits:
                        unique_tickets.extend(commit.get("ticket_references", []))
                    unique_tickets = list(set(unique_tickets))

                    # Create context summary
                    context_summary = f"{len(commits)} commits by {len(active_devs)} developers"
                    if unique_tickets:
                        context_summary += f", {len(unique_tickets)} tickets referenced"

                    # Create or update daily batch using pre-loaded existing batches
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

                    try:
                        existing_batch = existing_batches_map.get(date_str)

                        if existing_batch:
                            # Update existing batch with new data
                            existing_batch.commit_count = len(commits)
                            existing_batch.total_files_changed = total_files
                            existing_batch.total_lines_added = total_additions
                            existing_batch.total_lines_deleted = total_deletions
                            existing_batch.active_developers = active_devs
                            existing_batch.unique_tickets = unique_tickets
                            existing_batch.context_summary = context_summary
                            existing_batch.fetched_at = datetime.utcnow()
                            existing_batch.classification_status = "pending"
                            logger.info(f"🔍 DEBUG: Updated existing batch for {date_str}")
                        else:
                            # Create new batch
                            batch = DailyCommitBatch(
                                date=date_obj,
                                project_key=project_key,
                                repo_path=str(repo_path),
                                commit_count=len(commits),
                                total_files_changed=total_files,
                                total_lines_added=total_additions,
                                total_lines_deleted=total_deletions,
                                active_developers=active_devs,
                                unique_tickets=unique_tickets,
                                context_summary=context_summary,
                                classification_status="pending",
                                fetched_at=datetime.utcnow(),
                            )
                            session.add(batch)
                            batches_created += 1
                            logger.info(f"🔍 DEBUG: Created new batch for {date_str}")
                    except Exception as batch_error:
                        # Don't let batch creation failure kill commit storage
                        logger.error(
                            f"❌ CRITICAL: Failed to create/update batch for {date_str}: {batch_error}"
                        )
                        import traceback

                        logger.error(f"❌ Full batch error trace: {traceback.format_exc()}")
                        # Important: rollback any pending transaction to restore session state
                        session.rollback()
                        # Skip this batch but continue processing

            # Use bulk store operation for all commits at once for maximum performance
            if all_commits_to_store:
                logger.info(f"Using bulk_store_commits for {len(all_commits_to_store)} commits")
                bulk_stats = self.cache.bulk_store_commits(str(repo_path), all_commits_to_store)
                commits_stored = bulk_stats["inserted"]
                logger.info(
                    f"Bulk stored {commits_stored} commits in {bulk_stats['time_seconds']:.2f}s ({bulk_stats['commits_per_second']:.0f} commits/sec)"
                )
            else:
                commits_stored = 0
                logger.info("No new commits to store")

            # Commit all changes to database (for daily batch records)
            session.commit()

            # CRITICAL FIX: Verify commits were actually stored
            logger.info("🔍 DEBUG: Verifying commit storage...")
            verification_result = self._verify_commit_storage(
                session, daily_commits, repo_path, expected_commits
            )
            actual_stored = verification_result["actual_stored"]

            # Validate storage success based on what we expected to store vs what we actually stored
            if expected_commits > 0 and actual_stored != expected_commits:
                error_msg = f"Storage verification failed: expected to store {expected_commits} new commits, actually stored {actual_stored}"
                logger.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg)

            logger.info(
                f"✅ Storage verified: {actual_stored}/{expected_commits} commits successfully stored"
            )
            logger.info(
                f"Created/updated {batches_created} daily commit batches, stored {actual_stored} commits"
            )

        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR storing daily batches: {e}")
            logger.error("❌ This error causes ALL commits to be lost!")
            import traceback

            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            session.rollback()
        finally:
            session.close()

        return batches_created

    def _verify_commit_storage(
        self,
        session: Session,
        daily_commits: dict[str, list[dict[str, Any]]],
        repo_path: Path,
        expected_new_commits: int,
    ) -> dict[str, int]:
        """Verify that commits were actually stored in the database.

        WHY: Ensures that session.commit() actually persisted the data and didn't
        silently fail. This prevents the GitDataFetcher from reporting success
        when commits weren't actually stored.

        Args:
            session: Database session to query
            daily_commits: Original commit data to verify against
            repo_path: Repository path for filtering
            expected_new_commits: Number of new commits we expected to store this session

        Returns:
            Dict containing verification results:
                - actual_stored: Number of commits from this session found in database
                - total_found: Total commits found matching our hashes
                - expected_new: Number of new commits we expected to store

        Raises:
            RuntimeError: If verification fails due to database errors
        """
        try:
            # Collect all commit hashes we tried to store
            expected_hashes = set()
            for day_commits in daily_commits.values():
                for commit in day_commits:
                    expected_hashes.add(commit["commit_hash"])

            if not expected_hashes:
                logger.info("No commits to verify")
                return {"actual_stored": 0, "total_found": 0, "expected_new": 0}

            # Query database for actual stored commits
            stored_commits = (
                session.query(CachedCommit)
                .filter(
                    CachedCommit.commit_hash.in_(expected_hashes),
                    CachedCommit.repo_path == str(repo_path),
                )
                .all()
            )

            stored_hashes = {commit.commit_hash for commit in stored_commits}
            total_found = len(stored_hashes)

            # For this verification, we assume all matching commits were stored successfully
            # Since we only attempt to store commits that don't already exist,
            # the number we "actually stored" equals what we expected to store
            actual_stored = expected_new_commits

            # Log detailed verification results
            logger.info(
                f"🔍 DEBUG: Storage verification - Expected new: {expected_new_commits}, Total matching found: {total_found}"
            )

            # Check for missing commits (this would indicate storage failure)
            missing_hashes = expected_hashes - stored_hashes
            if missing_hashes:
                missing_short = [h[:7] for h in list(missing_hashes)[:5]]  # First 5 for logging
                logger.error(
                    f"❌ Missing commits in database: {missing_short} (showing first 5 of {len(missing_hashes)})"
                )
                # If we have missing commits, we didn't store what we expected
                actual_stored = total_found

            return {
                "actual_stored": actual_stored,
                "total_found": total_found,
                "expected_new": expected_new_commits,
            }

        except Exception as e:
            logger.error(f"❌ Critical error during storage verification: {e}")
            # Re-raise as RuntimeError to indicate this is a critical failure
            raise RuntimeError(f"Storage verification failed: {e}") from e

    def get_fetch_status(self, project_key: str, repo_path: Path) -> dict[str, Any]:
        """Get status of data fetching for a project."""
        session = self.database.get_session()

        try:
            # Count daily batches
            batches = (
                session.query(DailyCommitBatch)
                .filter(
                    DailyCommitBatch.project_key == project_key,
                    DailyCommitBatch.repo_path == str(repo_path),
                )
                .all()
            )

            # Count tickets
            tickets = (
                session.query(DetailedTicketData)
                .filter(DetailedTicketData.project_key == project_key)
                .count()
            )

            # Count correlations
            correlations = (
                session.query(CommitTicketCorrelation)
                .filter(
                    CommitTicketCorrelation.project_key == project_key,
                    CommitTicketCorrelation.repo_path == str(repo_path),
                )
                .count()
            )

            # Calculate statistics
            total_commits = sum(batch.commit_count for batch in batches)
            classified_batches = sum(
                1 for batch in batches if batch.classification_status == "completed"
            )

            return {
                "project_key": project_key,
                "repo_path": str(repo_path),
                "daily_batches": len(batches),
                "total_commits": total_commits,
                "unique_tickets": tickets,
                "commit_correlations": correlations,
                "classification_status": {
                    "completed_batches": classified_batches,
                    "pending_batches": len(batches) - classified_batches,
                    "completion_rate": classified_batches / len(batches) if batches else 0.0,
                },
            }

        except Exception as e:
            logger.error(f"Error getting fetch status: {e}")
            return {}
        finally:
            session.close()

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
