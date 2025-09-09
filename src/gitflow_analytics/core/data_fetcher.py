"""Data fetcher for collecting raw git commits and ticket data without classification.

This module implements the first step of the two-step fetch/analyze process,
focusing purely on data collection from Git repositories and ticket systems
without performing any LLM-based classification.
"""

import contextlib
import logging
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..extractors.story_points import StoryPointExtractor
from ..extractors.tickets import TicketExtractor
from ..integrations.jira_integration import JIRAIntegration
from ..models.database import (
    CachedCommit,
    CommitTicketCorrelation,
    DailyCommitBatch,
    DetailedTicketData,
)
from .branch_mapper import BranchToProjectMapper
from .cache import GitAnalysisCache
from .identity import DeveloperIdentityResolver
from .progress import get_progress_service

logger = logging.getLogger(__name__)


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
    ):
        """Initialize the data fetcher.

        Args:
            cache: Git analysis cache instance
            branch_mapping_rules: Rules for mapping branches to projects
            allowed_ticket_platforms: List of allowed ticket platforms
            exclude_paths: List of file paths to exclude from analysis
        """
        self.cache = cache
        # CRITICAL FIX: Use the same database instance as the cache to avoid session conflicts
        self.database = cache.db
        self.story_point_extractor = StoryPointExtractor()
        self.ticket_extractor = TicketExtractor(allowed_platforms=allowed_ticket_platforms)
        self.branch_mapper = BranchToProjectMapper(branch_mapping_rules)
        self.exclude_paths = exclude_paths or []

        # Initialize identity resolver
        identity_db_path = cache.cache_dir / "identities.db"
        self.identity_resolver = DeveloperIdentityResolver(identity_db_path)

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

        # Step 1: Collect all commits organized by day
        logger.info("🔍 DEBUG: About to fetch commits by day")
        logger.info("Fetching commits organized by day...")
        daily_commits = self._fetch_commits_by_day(
            repo_path, project_key, start_date, end_date, branch_patterns, progress_callback
        )
        logger.info(f"🔍 DEBUG: Fetched {len(daily_commits)} days of commits")

        # Step 2: Extract and fetch all referenced tickets
        logger.info("🔍 DEBUG: About to extract ticket references")
        logger.info("Extracting ticket references...")
        ticket_ids = self._extract_all_ticket_references(daily_commits)
        logger.info(f"🔍 DEBUG: Extracted {len(ticket_ids)} ticket IDs")

        if jira_integration and ticket_ids:
            logger.info(f"Fetching {len(ticket_ids)} unique tickets from JIRA...")
            self._fetch_detailed_tickets(
                ticket_ids, jira_integration, project_key, progress_callback
            )

        # Step 3: Store commit-ticket correlations
        logger.info("Building commit-ticket correlations...")
        correlations_created = self._build_commit_ticket_correlations(daily_commits, repo_path)

        # Step 4: Store daily commit batches
        logger.info(
            f"🔍 DEBUG: About to store daily batches. Daily commits has {len(daily_commits)} days"
        )
        logger.info("Storing daily commit batches...")
        batches_created = self._store_daily_batches(daily_commits, repo_path, project_key)
        logger.info(f"🔍 DEBUG: Storage complete. Batches created: {batches_created}")

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
            "daily_commits": daily_commits,  # For immediate use if needed
        }

        # Log with actual storage results
        if actual_stored_commits == expected_commits:
            logger.info(
                f"✅ Data fetch completed successfully for {project_key}: {actual_stored_commits}/{expected_commits} commits stored, {len(ticket_ids)} tickets"
            )
        else:
            logger.error(
                f"⚠️ Data fetch completed with storage issues for {project_key}: {actual_stored_commits}/{expected_commits} commits stored, {len(ticket_ids)} tickets"
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
        from git import Repo

        try:
            repo = Repo(repo_path)
            # Update repository from remote before analysis
            self._update_repository(repo)
        except Exception as e:
            logger.error(f"Failed to open repository at {repo_path}: {e}")
            return {}

        # Collect commits from all relevant branches
        all_commits = []
        branches_to_analyze = self._get_branches_to_analyze(repo, branch_patterns)

        if not branches_to_analyze:
            logger.warning(f"No accessible branches found in repository {repo_path}")
            return {}

        logger.info(f"Analyzing branches: {branches_to_analyze}")

        for branch_name in branches_to_analyze:
            try:
                branch_commits = list(
                    repo.iter_commits(branch_name, since=start_date, until=end_date, reverse=False)
                )

                logger.debug(
                    f"Found {len(branch_commits)} commits in branch {branch_name} for date range"
                )

                for commit in branch_commits:
                    # Include merge commits like the original analyzer
                    # The original analyzer marks merge commits with is_merge=True but doesn't skip them

                    # Extract commit data with full metadata
                    commit_data = self._extract_commit_data(
                        commit, branch_name, project_key, repo_path
                    )
                    if commit_data:
                        all_commits.append(commit_data)

            except Exception as e:
                logger.warning(f"Error processing branch {branch_name}: {e}")
                continue

        # Deduplicate commits (same commit may appear in multiple branches)
        seen_hashes = set()
        unique_commits = []
        for commit_data in all_commits:
            commit_hash = commit_data["commit_hash"]
            if commit_hash not in seen_hashes:
                seen_hashes.add(commit_hash)
                unique_commits.append(commit_data)

        # Organize commits by day
        daily_commits = defaultdict(list)
        for commit_data in unique_commits:
            # Convert timestamp to date key
            commit_date = commit_data["timestamp"].date()
            date_key = commit_date.strftime("%Y-%m-%d")
            daily_commits[date_key].append(commit_data)

        # Sort commits within each day by timestamp
        for date_key in daily_commits:
            daily_commits[date_key].sort(key=lambda c: c["timestamp"])

        logger.info(f"Collected {len(unique_commits)} commits across {len(daily_commits)} days")
        return dict(daily_commits)

    def _extract_commit_data(
        self, commit: Any, branch_name: str, project_key: str, repo_path: Path
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

                commit_data.update(
                    {
                        "files_changed": files_changed,
                        "files_changed_count": len(files_changed),
                        "lines_added": total_insertions,
                        "lines_deleted": total_deletions,
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
        import fnmatch

        return any(fnmatch.fnmatch(file_path, pattern) for pattern in self.exclude_paths)

    def _get_branches_to_analyze(
        self, repo: Any, branch_patterns: Optional[list[str]]
    ) -> list[str]:
        """Get list of branches to analyze based on patterns.

        WHY: Robust branch detection that handles missing remotes, missing default branches,
        and provides good fallback behavior. Based on the approach used in the existing analyzer.

        DESIGN DECISION:
        - Try default branches first, fall back to all available branches
        - Handle missing remotes gracefully
        - Skip remote tracking branches to avoid duplicates
        - Use actual branch existence checking rather than assuming branches exist
        """
        if not branch_patterns:
            # Get all available branches (local branches preferred)
            available_branches = []

            # First, try local branches
            try:
                local_branches = [branch.name for branch in repo.branches]
                available_branches.extend(local_branches)
                logger.debug(f"Found local branches: {local_branches}")
            except Exception as e:
                logger.debug(f"Error getting local branches: {e}")

            # If we have remotes, also consider remote branches (but clean the names)
            try:
                if repo.remotes and hasattr(repo.remotes, "origin"):
                    remote_branches = [
                        ref.name.replace("origin/", "")
                        for ref in repo.remotes.origin.refs
                        if not ref.name.endswith("HEAD")  # Skip HEAD ref
                    ]
                    # Only add remote branches that aren't already in local branches
                    for branch in remote_branches:
                        if branch not in available_branches:
                            available_branches.append(branch)
                    logger.debug(f"Found remote branches: {remote_branches}")
            except Exception as e:
                logger.debug(f"Error getting remote branches: {e}")

            # If no branches found, fallback to trying common names directly
            if not available_branches:
                logger.warning(
                    "No branches found via normal detection, falling back to common names"
                )
                available_branches = ["main", "master", "develop", "dev"]

            # Try default main branches first, in order of preference
            main_branches = ["main", "master", "develop", "dev"]
            for branch in main_branches:
                if branch in available_branches:
                    # Test that we can actually access this branch
                    try:
                        # Just try to get the commit object to verify branch exists and is accessible
                        next(iter(repo.iter_commits(branch, max_count=1)), None)
                        logger.info(f"Using main branch: {branch}")
                        return [branch]
                    except Exception as e:
                        logger.debug(f"Branch {branch} exists but not accessible: {e}")
                        continue

            # If no main branches work, try the first available branch that actually works
            for branch in available_branches:
                try:
                    next(iter(repo.iter_commits(branch, max_count=1)), None)
                    logger.info(f"Using fallback branch: {branch}")
                    return [branch]
                except Exception as e:
                    logger.debug(f"Branch {branch} not accessible: {e}")
                    continue

            # Last resort: return empty list (will be handled gracefully by caller)
            logger.warning("No accessible branches found")
            return []

        # Use specified patterns - match against all available branches
        import fnmatch

        available_branches = []

        # Collect all branches (local and remote)
        with contextlib.suppress(Exception):
            available_branches.extend([branch.name for branch in repo.branches])

        try:
            if repo.remotes and hasattr(repo.remotes, "origin"):
                remote_branches = [
                    ref.name.replace("origin/", "")
                    for ref in repo.remotes.origin.refs
                    if not ref.name.endswith("HEAD")
                ]
                for branch in remote_branches:
                    if branch not in available_branches:
                        available_branches.append(branch)
        except Exception:
            pass

        # Match patterns against available branches
        matching_branches = []
        for pattern in branch_patterns:
            matching = [branch for branch in available_branches if fnmatch.fnmatch(branch, pattern)]
            matching_branches.extend(matching)

        # Test that matched branches are actually accessible
        accessible_branches = []
        for branch in list(set(matching_branches)):  # Remove duplicates
            try:
                next(iter(repo.iter_commits(branch, max_count=1)), None)
                accessible_branches.append(branch)
            except Exception as e:
                logger.debug(f"Matched branch {branch} not accessible: {e}")

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
        try:
            if repo.remotes:
                logger.info("Fetching latest changes from remote")

                # Use subprocess for git fetch to prevent password prompts
                env = os.environ.copy()
                env["GIT_TERMINAL_PROMPT"] = "0"
                env["GIT_ASKPASS"] = ""
                env["GCM_INTERACTIVE"] = "never"

                # Run git fetch with timeout
                try:
                    result = subprocess.run(
                        ["git", "fetch", "--all", "--config", "credential.helper="],
                        cwd=repo.working_dir,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=30,  # 30 second timeout
                    )

                    if result.returncode != 0:
                        error_str = result.stderr.lower()
                        if any(
                            x in error_str
                            for x in ["authentication", "permission denied", "401", "403"]
                        ):
                            logger.error(
                                "GitHub authentication failed. Please check your token is valid and has appropriate permissions. "
                                "You can verify with: gh auth status"
                            )
                            # Skip fetch but continue with local analysis
                            return False
                        logger.warning(f"Git fetch failed: {result.stderr}")
                        return False

                except subprocess.TimeoutExpired:
                    logger.error(
                        "Git fetch timed out (likely authentication failure). Continuing with local repository state."
                    )
                    return False

                # Only try to pull if not in detached HEAD state
                if not repo.head.is_detached:
                    current_branch = repo.active_branch
                    tracking = current_branch.tracking_branch()
                    if tracking:
                        # Pull latest changes using subprocess
                        try:
                            result = subprocess.run(
                                ["git", "pull", "--config", "credential.helper="],
                                cwd=repo.working_dir,
                                env=env,
                                capture_output=True,
                                text=True,
                                timeout=30,
                            )

                            if result.returncode != 0:
                                error_str = result.stderr.lower()
                                if any(
                                    x in error_str
                                    for x in ["authentication", "permission denied", "401", "403"]
                                ):
                                    logger.error(
                                        "GitHub authentication failed during pull. Continuing with local repository state."
                                    )
                                    return False
                                logger.warning(f"Git pull failed: {result.stderr}")
                                return False
                            logger.debug(f"Pulled latest changes for {current_branch.name}")

                        except subprocess.TimeoutExpired:
                            logger.error(
                                "Git pull timed out. Continuing with local repository state."
                            )
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
            batch_size = 50
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
                                "insertions": commit.get("lines_added", 0),
                                "deletions": commit.get("lines_deleted", 0),
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

    def _calculate_commit_stats(self, commit: Any) -> dict[str, int]:
        """Calculate commit statistics using reliable git diff --numstat.

        Returns:
            Dictionary with 'files', 'insertions', and 'deletions' counts
        """
        stats = {"files": 0, "insertions": 0, "deletions": 0}

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
                        # filename = parts[2] - we don't filter here since files_changed list is handled separately

                        # Count all changes (raw stats, no filtering)
                        stats["files"] += 1
                        stats["insertions"] += insertions
                        stats["deletions"] += deletions

                    except ValueError:
                        # Skip binary files or malformed lines
                        continue

        except Exception as e:
            # Log the error for debugging but don't crash
            logger.warning(f"Error calculating commit stats for {commit.hexsha[:8]}: {e}")

        return stats
