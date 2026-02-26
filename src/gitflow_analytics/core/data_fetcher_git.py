"""Git operations mixin for GitDataFetcher."""

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


class GitFetcherMixin:
    """Mixin providing git operations for GitDataFetcher.

    Provides: _fetch_commits_by_day, _extract_commit_data, branch/file helpers,
    _get_branches_to_analyze, _update_repository.
    """

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

        # BUG 4 FIX: Estimate total commit count using `git rev-list --count` instead of
        # materialising a full branch history that is immediately discarded.  The old code
        # called list(repo.iter_commits(...)) for every analysis run and never used the
        # result — it was pure memory waste.
        try:
            count_str = repo.git.rev_list(
                branches_to_analyze[0],
                "--count",
                f"--after={start_date.isoformat()}",
                f"--before={end_date.isoformat()}",
            )
            int(count_str.strip()) * len(branches_to_analyze)
        except GitOperationTimeout:
            logger.warning(
                f"Timeout while sampling commits for {project_key}, using default estimate"
            )
            len(days_to_process) * BatchSizes.COMMITS_PER_WEEK_ESTIMATE
        except Exception as e:
            logger.debug(f"Could not sample commits for {project_key}: {e}, using default estimate")
            len(days_to_process) * BatchSizes.COMMITS_PER_WEEK_ESTIMATE

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

                    # Incremental caching: persist full commit data to SQLite immediately.
                    self._store_day_commits_incremental(
                        repo_path, day_str, day_commits, project_key
                    )

                    # BUG 3 FIX: After writing to SQLite we no longer need the full commit
                    # dicts in memory.  Extract the lightweight summary information that
                    # downstream callers actually need (ticket refs, commit hashes, batch
                    # aggregates) and discard the rest.  Peak RAM is now proportional to a
                    # single day's commits rather than the entire date-range.
                    #
                    # The summary dict held in daily_commits is consumed by:
                    #   _extract_all_ticket_references  → "ticket_references"
                    #   _build_commit_ticket_correlations → "commit_ticket_pairs"
                    #   _verify_commit_storage           → "commit_hashes"
                    #   _store_daily_batches             → all summary fields
                    day_summary: dict[str, Any] = {
                        "count": len(day_commits),
                        # Flat set of all ticket refs referenced on this day
                        "ticket_references": list(
                            {ref for c in day_commits for ref in c.get("ticket_references", [])}
                        ),
                        # Pairs needed to build commit-ticket correlations
                        "commit_ticket_pairs": [
                            {
                                "commit_hash": c["commit_hash"],
                                "ticket_references": c.get("ticket_references", []),
                                "project_key": c.get("project_key", project_key),
                            }
                            for c in day_commits
                            if c.get("ticket_references")
                        ],
                        # Hashes needed by _verify_commit_storage
                        "commit_hashes": [c["commit_hash"] for c in day_commits],
                        # Aggregate stats needed by _store_daily_batches
                        "total_files": sum(c.get("files_changed_count", 0) for c in day_commits),
                        "total_additions": sum(c.get("lines_added", 0) for c in day_commits),
                        "total_deletions": sum(c.get("lines_deleted", 0) for c in day_commits),
                        "active_devs": list(
                            {c.get("canonical_developer_id", "") for c in day_commits}
                        ),
                    }
                    daily_commits[day_str] = day_summary
                    del day_commits  # Full commit dicts can now be GC'd

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

        # BUG 3 FIX: daily_commits now holds summary dicts, not full commit lists.
        total_commits = sum(info["count"] for info in daily_commits.values())
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
        """Check if a file should be excluded based on exclude patterns.

        Delegates to :func:`~gitflow_analytics.utils.glob_matcher.should_exclude_file`.
        """
        return _should_exclude_file_fn(file_path, self.exclude_paths)

    def _matches_glob_pattern(self, filepath: str, pattern: str) -> bool:
        """Check if a file path matches a glob pattern, handling ** recursion correctly.

        Delegates to :func:`~gitflow_analytics.utils.glob_matcher.matches_glob_pattern`.

        Args:
            filepath: The file path to check
            pattern: The glob pattern to match against

        Returns:
            True if the file path matches the pattern, False otherwise
        """
        return _matches_glob_pattern_fn(filepath, pattern)

    def _match_recursive_pattern(self, filepath: str, pattern: str) -> bool:
        """Handle complex patterns with multiple ** wildcards.

        Delegates to :func:`~gitflow_analytics.utils.glob_matcher.match_recursive_pattern`.

        Args:
            filepath: The file path to check
            pattern: The pattern with multiple ** wildcards

        Returns:
            True if the path matches the pattern, False otherwise
        """
        return _match_recursive_pattern_fn(filepath, pattern)

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

