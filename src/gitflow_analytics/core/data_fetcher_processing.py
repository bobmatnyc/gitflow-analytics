"""Repository management, ticket, and storage mixin for GitDataFetcher."""

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


class ProcessingMixin:
    """Mixin providing repo management, ticket extraction, and storage methods.

    Provides: _check_repository_staleness, _check_repository_security,
    get_repository_status_summary, ticket methods, _store_daily_batches,
    _verify_commit_storage, get_fetch_status.
    """

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

    def _extract_all_ticket_references(self, daily_commits: dict[str, Any]) -> set[str]:
        """Extract all unique ticket IDs from the daily-commits summary dicts.

        BUG 3 FIX: daily_commits now maps date strings to lightweight summary dicts
        (see _fetch_commits_by_day).  Each summary dict contains a pre-computed
        "ticket_references" list so we never need to re-hydrate the full commit objects.
        """
        ticket_ids: set[str] = set()

        for day_info in daily_commits.values():
            # Support both the new summary-dict format and legacy list format for
            # any callers that pass raw commit lists directly.
            if isinstance(day_info, dict) and "ticket_references" in day_info:
                ticket_ids.update(day_info["ticket_references"])
            elif isinstance(day_info, list):
                for commit in day_info:
                    ticket_ids.update(commit.get("ticket_references", []))

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
        self, daily_commits: dict[str, Any], repo_path: Path
    ) -> int:
        """Build and store commit-ticket correlations.

        BUG 3 FIX: Accepts the lightweight summary-dict format returned by
        _fetch_commits_by_day (each day entry has a "commit_ticket_pairs" list)
        as well as the legacy full-list format for backward compatibility.
        """
        session = self.database.get_session()
        correlations_created = 0

        try:
            for day_info in daily_commits.values():
                # Resolve the iterable of (commit_hash, ticket_refs, project_key) triples
                # from either the new summary-dict format or the legacy list format.
                if isinstance(day_info, dict) and "commit_ticket_pairs" in day_info:
                    pairs = day_info["commit_ticket_pairs"]
                elif isinstance(day_info, list):
                    pairs = [
                        {
                            "commit_hash": c["commit_hash"],
                            "ticket_references": c.get("ticket_references", []),
                            "project_key": c.get("project_key", ""),
                        }
                        for c in day_info
                    ]
                else:
                    pairs = []

                for commit in pairs:
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
        self, daily_commits: dict[str, Any], repo_path: Path, project_key: str
    ) -> int:
        """Store daily commit batches for efficient retrieval using bulk operations.

        WHY: Enhanced to use bulk operations from the cache layer for significantly
        better performance when storing large numbers of commits.

        BUG 3 FIX: Accepts the lightweight summary-dict format returned by
        _fetch_commits_by_day.  Commits are already in the DB (written by
        _store_day_commits_incremental); this method only creates/updates the
        DailyCommitBatch metadata records using the pre-aggregated summary fields.
        Legacy full-list format is also handled for backward compatibility.
        """
        session = self.database.get_session()
        batches_created = 0
        commits_stored = 0
        expected_commits = 0

        try:
            # Resolve total commit count regardless of format
            def _day_count(day_info: Any) -> int:
                if isinstance(day_info, dict):
                    return day_info.get("count", 0)
                return len(day_info) if isinstance(day_info, list) else 0

            total_commits = sum(_day_count(v) for v in daily_commits.values())
            logger.debug(
                f"🔍 DEBUG: Storing {total_commits} commits from {len(daily_commits)} days"
            )
            logger.debug(
                f"🔍 DEBUG: Daily commits keys: {list(daily_commits.keys())[:5]}"
            )  # First 5 dates

            # Track dates to process for efficient batch handling
            dates_to_process = [
                datetime.strptime(date_str, "%Y-%m-%d").date() for date_str in daily_commits
            ]
            logger.debug(f"🔍 DEBUG: Processing {len(dates_to_process)} dates for daily batches")

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
                    logger.debug(
                        f"🔍 DEBUG: Found existing batch for {date_str}: ID={existing_batch.id}"
                    )
                else:
                    logger.debug(f"🔍 DEBUG: No existing batch found for {date_str}")

            # BUG 3 FIX: Commits are already written to the DB by _store_day_commits_incremental.
            # We only need to create/update the DailyCommitBatch records using the pre-aggregated
            # summary fields.  For legacy full-list format we still do the bulk-store pass.
            all_commits_to_store: list[dict[str, Any]] = []

            # For legacy list format: collect commit hashes to check existence in bulk
            has_legacy_lists = any(isinstance(v, list) for v in daily_commits.values())
            if has_legacy_lists:
                commit_hashes_to_check = [
                    commit["commit_hash"]
                    for commits in daily_commits.values()
                    if isinstance(commits, list)
                    for commit in commits
                ]
                existing_commits_map = self.cache.bulk_exists(
                    str(repo_path), commit_hashes_to_check
                )
            else:
                existing_commits_map = {}

            # Disable autoflush to prevent premature batch creation during commit storage
            with session.no_autoflush:
                for date_str, day_info in daily_commits.items():
                    # --- Resolve summary fields from either format ---
                    if isinstance(day_info, dict) and "count" in day_info:
                        # New summary-dict format (BUG 3 FIX path)
                        commit_count = day_info["count"]
                        total_files = day_info.get("total_files", 0)
                        total_additions = day_info.get("total_additions", 0)
                        total_deletions = day_info.get("total_deletions", 0)
                        active_devs = day_info.get("active_devs", [])
                        unique_tickets = day_info.get("ticket_references", [])
                        commits_to_store_this_date: list[dict[str, Any]] = []
                    elif isinstance(day_info, list):
                        # Legacy full-list format — kept for backward compatibility
                        commits = day_info
                        if not commits:
                            continue
                        commit_count = len(commits)
                        commits_to_store_this_date = []
                        for commit in commits:
                            if not existing_commits_map.get(commit["commit_hash"], False):
                                cache_format_commit = {
                                    "hash": commit["commit_hash"],
                                    "author_name": commit.get("author_name", ""),
                                    "author_email": commit.get("author_email", ""),
                                    "message": commit.get("message", ""),
                                    "timestamp": commit["timestamp"],
                                    "branch": commit.get("branch", "main"),
                                    "is_merge": commit.get("is_merge", False),
                                    "files_changed_count": commit.get("files_changed_count", 0),
                                    "insertions": commit.get(
                                        "raw_insertions", commit.get("lines_added", 0)
                                    ),
                                    "deletions": commit.get(
                                        "raw_deletions", commit.get("lines_deleted", 0)
                                    ),
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
                        total_files = sum(c.get("files_changed_count", 0) for c in commits)
                        total_additions = sum(c.get("lines_added", 0) for c in commits)
                        total_deletions = sum(c.get("lines_deleted", 0) for c in commits)
                        active_devs = list({c.get("canonical_developer_id", "") for c in commits})
                        unique_tickets = list(
                            {ref for c in commits for ref in c.get("ticket_references", [])}
                        )
                    else:
                        continue

                    logger.debug(f"🔍 DEBUG: Processing {commit_count} commits for {date_str}")
                    logger.debug(
                        f"🔍 DEBUG: Prepared {len(commits_to_store_this_date)} new commits for {date_str}"
                    )

                    # Create context summary
                    context_summary = f"{commit_count} commits by {len(active_devs)} developers"
                    if unique_tickets:
                        context_summary += f", {len(unique_tickets)} tickets referenced"

                    # Create or update daily batch using pre-loaded existing batches
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

                    try:
                        existing_batch = existing_batches_map.get(date_str)

                        if existing_batch:
                            # Update existing batch with new data
                            existing_batch.commit_count = commit_count
                            existing_batch.total_files_changed = total_files
                            existing_batch.total_lines_added = total_additions
                            existing_batch.total_lines_deleted = total_deletions
                            existing_batch.active_developers = active_devs
                            existing_batch.unique_tickets = unique_tickets
                            existing_batch.context_summary = context_summary
                            existing_batch.fetched_at = datetime.utcnow()
                            existing_batch.classification_status = "pending"
                            logger.debug(f"🔍 DEBUG: Updated existing batch for {date_str}")
                        else:
                            # Create new batch
                            batch = DailyCommitBatch(
                                date=date_obj,
                                project_key=project_key,
                                repo_path=str(repo_path),
                                commit_count=commit_count,
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
                            logger.debug(f"🔍 DEBUG: Created new batch for {date_str}")
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
            logger.debug("🔍 DEBUG: Verifying commit storage...")
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
        daily_commits: dict[str, Any],
        repo_path: Path,
        expected_new_commits: int,
    ) -> dict[str, int]:
        """Verify that commits were actually stored in the database.

        WHY: Ensures that session.commit() actually persisted the data and didn't
        silently fail. This prevents the GitDataFetcher from reporting success
        when commits weren't actually stored.

        Args:
            session: Database session to query
            daily_commits: Day-keyed mapping — either the new summary-dict format
                (each value has "commit_hashes") or the legacy full-list format.
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
            # Collect all commit hashes we tried to store.
            # BUG 3 FIX: support both new summary-dict format and legacy list format.
            expected_hashes: set[str] = set()
            for day_info in daily_commits.values():
                if isinstance(day_info, dict) and "commit_hashes" in day_info:
                    expected_hashes.update(day_info["commit_hashes"])
                elif isinstance(day_info, list):
                    for commit in day_info:
                        expected_hashes.add(commit["commit_hash"])

            if not expected_hashes:
                logger.info("No commits to verify")
                return {"actual_stored": 0, "total_found": 0, "expected_new": 0}

            # BUG 6 FIX: Project only the commit_hash column instead of loading full
            # ORM objects.  The old code hydrated every column of every CachedCommit row
            # (message, stats, JSON blobs, …) just to build a set of hashes, which could
            # allocate hundreds of MB for large repositories.
            stored_hashes = set(
                row[0]
                for row in session.query(CachedCommit.commit_hash)
                .filter(
                    CachedCommit.commit_hash.in_(expected_hashes),
                    CachedCommit.repo_path == str(repo_path),
                )
                .all()
            )
            total_found = len(stored_hashes)

            # For this verification, we assume all matching commits were stored successfully
            # Since we only attempt to store commits that don't already exist,
            # the number we "actually stored" equals what we expected to store
            actual_stored = expected_new_commits

            # Log detailed verification results
            logger.debug(
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
        """Get status of data fetching for a project.

        BUG 7 FIX: Previously loaded every DailyCommitBatch row into Python objects and
        aggregated them in a loop — O(n) memory for n batches.  Now uses a single SQL
        aggregation query so only the scalar summary is transferred over the wire.
        """
        session = self.database.get_session()

        try:
            # BUG 7 FIX: Aggregate batch statistics in SQL instead of loading all rows.
            # func.sum / func.count execute in the database engine; we receive three scalars.
            batch_agg = (
                session.query(
                    func.count(DailyCommitBatch.id).label("total_batches"),
                    func.coalesce(func.sum(DailyCommitBatch.commit_count), 0).label(
                        "total_commits"
                    ),
                    func.count(
                        case(
                            (DailyCommitBatch.classification_status == "completed", 1),
                        )
                    ).label("classified_batches"),
                )
                .filter(
                    DailyCommitBatch.project_key == project_key,
                    DailyCommitBatch.repo_path == str(repo_path),
                )
                .first()
            )

            total_batches = batch_agg.total_batches if batch_agg else 0
            total_commits = batch_agg.total_commits if batch_agg else 0
            classified_batches = batch_agg.classified_batches if batch_agg else 0

            # Count tickets (scalar — already efficient)
            tickets = (
                session.query(DetailedTicketData)
                .filter(DetailedTicketData.project_key == project_key)
                .count()
            )

            # Count correlations (scalar — already efficient)
            correlations = (
                session.query(CommitTicketCorrelation)
                .filter(
                    CommitTicketCorrelation.project_key == project_key,
                    CommitTicketCorrelation.repo_path == str(repo_path),
                )
                .count()
            )

            return {
                "project_key": project_key,
                "repo_path": str(repo_path),
                "daily_batches": total_batches,
                "total_commits": total_commits,
                "unique_tickets": tickets,
                "commit_correlations": correlations,
                "classification_status": {
                    "completed_batches": classified_batches,
                    "pending_batches": total_batches - classified_batches,
                    "completion_rate": classified_batches / total_batches if total_batches else 0.0,
                },
            }

        except Exception as e:
            logger.error(f"Error getting fetch status: {e}")
            return {}
        finally:
            session.close()

