"""Commit, PR, and issue caching mixin for GitAnalysisCache."""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import git
from sqlalchemy import and_

from ..models.database import (
    CachedCommit,
    IssueCache,
    PullRequestCache,
)

logger = logging.getLogger(__name__)


class CommitCacheMixin:
    """Mixin providing commit, PR, and issue caching methods.

    Mixed into GitAnalysisCache. Uses self.db, self.ttl_hours,
    self.cache_hits, self.cache_misses, self.batch_size from the host class.
    """

    def get_cached_commit(self, repo_path: str, commit_hash: str) -> Optional[dict[str, Any]]:
        """Retrieve cached commit data if not stale."""
        with self.get_session() as session:
            cached = (
                session.query(CachedCommit)
                .filter(
                    and_(
                        CachedCommit.repo_path == repo_path, CachedCommit.commit_hash == commit_hash
                    )
                )
                .first()
            )

            if cached and not self._is_stale(cached.cached_at):
                self.cache_hits += 1
                if self.debug_mode:
                    print(f"DEBUG: Cache HIT for {commit_hash[:8]} in {repo_path}")
                return self._commit_to_dict(cached)

            self.cache_misses += 1
            if self.debug_mode:
                print(f"DEBUG: Cache MISS for {commit_hash[:8]} in {repo_path}")
            return None

    def get_cached_commits_bulk(
        self, repo_path: str, commit_hashes: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Retrieve multiple cached commits in a single query.

        WHY: Individual cache lookups are inefficient for large batches.
        This method fetches multiple commits at once, reducing database overhead
        and significantly improving performance for subsequent runs.

        Args:
            repo_path: Repository path for filtering
            commit_hashes: List of commit hashes to look up

        Returns:
            Dictionary mapping commit hash to commit data (only non-stale entries)
        """
        if not commit_hashes:
            return {}

        cached_commits = {}
        with self.get_session() as session:
            cached_results = (
                session.query(CachedCommit)
                .filter(
                    and_(
                        CachedCommit.repo_path == repo_path,
                        CachedCommit.commit_hash.in_(commit_hashes),
                    )
                )
                .all()
            )

            for cached in cached_results:
                if not self._is_stale(cached.cached_at):
                    cached_commits[cached.commit_hash] = self._commit_to_dict(cached)

        # Track cache performance
        hits = len(cached_commits)
        misses = len(commit_hashes) - hits
        self.cache_hits += hits
        self.cache_misses += misses

        if self.debug_mode:
            print(
                f"DEBUG: Bulk cache lookup - {hits} hits, {misses} misses for {len(commit_hashes)} commits"
            )

        return cached_commits

    def cache_commit(self, repo_path: str, commit_data: dict[str, Any]) -> None:
        """Cache commit analysis results."""
        with self.get_session() as session:
            # Check if already exists
            existing = (
                session.query(CachedCommit)
                .filter(
                    and_(
                        CachedCommit.repo_path == repo_path,
                        CachedCommit.commit_hash == commit_data["hash"],
                    )
                )
                .first()
            )

            if existing:
                # Update existing
                for key, value in commit_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                # Bug 1 fix: use timezone-aware UTC datetime instead of naive utcnow()
                existing.cached_at = datetime.now(timezone.utc)
            else:
                # Create new
                cached_commit = CachedCommit(
                    repo_path=repo_path,
                    commit_hash=commit_data["hash"],
                    author_name=commit_data.get("author_name"),
                    author_email=commit_data.get("author_email"),
                    message=commit_data.get("message"),
                    timestamp=commit_data.get("timestamp"),
                    branch=commit_data.get("branch"),
                    is_merge=commit_data.get("is_merge", False),
                    files_changed=commit_data.get(
                        "files_changed_count",
                        (
                            commit_data.get("files_changed", 0)
                            if isinstance(commit_data.get("files_changed"), int)
                            else len(commit_data.get("files_changed", []))
                        ),
                    ),
                    insertions=commit_data.get("insertions", 0),
                    deletions=commit_data.get("deletions", 0),
                    filtered_insertions=commit_data.get(
                        "filtered_insertions", commit_data.get("insertions", 0)
                    ),
                    filtered_deletions=commit_data.get(
                        "filtered_deletions", commit_data.get("deletions", 0)
                    ),
                    complexity_delta=commit_data.get("complexity_delta", 0.0),
                    story_points=commit_data.get("story_points"),
                    ticket_references=commit_data.get("ticket_references", []),
                )
                session.add(cached_commit)

    def cache_commits_batch(self, repo_path: str, commits: list[dict[str, Any]]) -> None:
        """Cache multiple commits in a single transaction.

        WHY: Optimized batch caching reduces database overhead by using
        bulk queries to check for existing commits instead of individual lookups.
        This significantly improves performance when caching large batches.
        """
        if not commits:
            return

        import time

        start_time = time.time()

        with self.get_session() as session:
            # Get all commit hashes in this batch
            commit_hashes = [commit_data["hash"] for commit_data in commits]

            # Bulk fetch existing commits
            existing_commits = {
                cached.commit_hash: cached
                for cached in session.query(CachedCommit)
                .filter(
                    and_(
                        CachedCommit.repo_path == repo_path,
                        CachedCommit.commit_hash.in_(commit_hashes),
                    )
                )
                .all()
            }

            # Process each commit
            for commit_data in commits:
                commit_hash = commit_data["hash"]

                if commit_hash in existing_commits:
                    # Update existing
                    existing = existing_commits[commit_hash]
                    for key, value in commit_data.items():
                        if key != "hash" and hasattr(existing, key):
                            setattr(existing, key, value)
                    # Bug 1 fix: use timezone-aware UTC datetime instead of naive utcnow()
                    existing.cached_at = datetime.now(timezone.utc)
                else:
                    # Create new
                    cached_commit = CachedCommit(
                        repo_path=repo_path,
                        commit_hash=commit_data["hash"],
                        author_name=commit_data.get("author_name"),
                        author_email=commit_data.get("author_email"),
                        message=commit_data.get("message"),
                        timestamp=commit_data.get("timestamp"),
                        branch=commit_data.get("branch"),
                        is_merge=commit_data.get("is_merge", False),
                        files_changed=commit_data.get(
                            "files_changed_count",
                            (
                                commit_data.get("files_changed", 0)
                                if isinstance(commit_data.get("files_changed"), int)
                                else len(commit_data.get("files_changed", []))
                            ),
                        ),
                        insertions=commit_data.get("insertions", 0),
                        deletions=commit_data.get("deletions", 0),
                        filtered_insertions=commit_data.get(
                            "filtered_insertions", commit_data.get("insertions", 0)
                        ),
                        filtered_deletions=commit_data.get(
                            "filtered_deletions", commit_data.get("deletions", 0)
                        ),
                        complexity_delta=commit_data.get("complexity_delta", 0.0),
                        story_points=commit_data.get("story_points"),
                        ticket_references=commit_data.get("ticket_references", []),
                    )
                    session.add(cached_commit)

            # Track performance metrics
            elapsed = time.time() - start_time
            self.bulk_operations_count += 1
            self.bulk_operations_time += elapsed

            if self.debug_mode:
                print(f"DEBUG: Bulk cached {len(commits)} commits in {elapsed:.3f}s")

    def get_cached_pr(self, repo_path: str, pr_number: int) -> Optional[dict[str, Any]]:
        """Retrieve cached pull request data."""
        with self.get_session() as session:
            cached = (
                session.query(PullRequestCache)
                .filter(
                    and_(
                        PullRequestCache.repo_path == repo_path,
                        PullRequestCache.pr_number == pr_number,
                    )
                )
                .first()
            )

            if cached and not self._is_stale(cached.cached_at):
                return self._pr_to_dict(cached)

            return None

    def cache_pr(self, repo_path: str, pr_data: dict[str, Any]) -> None:
        """Cache pull request data."""
        with self.get_session() as session:
            # Check if already exists
            existing = (
                session.query(PullRequestCache)
                .filter(
                    and_(
                        PullRequestCache.repo_path == repo_path,
                        PullRequestCache.pr_number == pr_data["number"],
                    )
                )
                .first()
            )

            if existing:
                # Update existing
                existing.title = pr_data.get("title")
                existing.description = pr_data.get("description")
                existing.author = pr_data.get("author")
                existing.created_at = pr_data.get("created_at")
                existing.merged_at = pr_data.get("merged_at")
                existing.story_points = pr_data.get("story_points")
                existing.labels = pr_data.get("labels", [])
                existing.commit_hashes = pr_data.get("commit_hashes", [])
                # PR state fields (v4.0) - only update when present in payload
                # to avoid overwriting good data with None from a partial fetch.
                if "pr_state" in pr_data:
                    existing.pr_state = pr_data.get("pr_state")
                if "closed_at" in pr_data:
                    existing.closed_at = pr_data.get("closed_at")
                if "is_merged" in pr_data:
                    existing.is_merged = pr_data.get("is_merged")
                # Enhanced PR tracking fields (v3.0) - only update when present in payload
                # to avoid accidentally zeroing out data already stored from a richer fetch.
                if "review_comments" in pr_data:
                    existing.review_comments_count = pr_data.get("review_comments", 0)
                if "pr_comments_count" in pr_data:
                    existing.pr_comments_count = pr_data.get("pr_comments_count", 0)
                if "approvals_count" in pr_data:
                    existing.approvals_count = pr_data.get("approvals_count", 0)
                if "change_requests_count" in pr_data:
                    existing.change_requests_count = pr_data.get("change_requests_count", 0)
                if "reviewers" in pr_data:
                    existing.reviewers = pr_data.get("reviewers", [])
                if "approved_by" in pr_data:
                    existing.approved_by = pr_data.get("approved_by", [])
                if "time_to_first_review_hours" in pr_data:
                    existing.time_to_first_review_hours = pr_data.get("time_to_first_review_hours")
                if "revision_count" in pr_data:
                    existing.revision_count = pr_data.get("revision_count", 0)
                if "changed_files" in pr_data:
                    existing.changed_files = pr_data.get("changed_files", 0)
                if "additions" in pr_data:
                    existing.additions = pr_data.get("additions", 0)
                if "deletions" in pr_data:
                    existing.deletions = pr_data.get("deletions", 0)
                # Bug 1 fix: use timezone-aware UTC datetime instead of naive utcnow()
                existing.cached_at = datetime.now(timezone.utc)
            else:
                # Create new
                cached_pr = PullRequestCache(
                    repo_path=repo_path,
                    pr_number=pr_data["number"],
                    title=pr_data.get("title"),
                    description=pr_data.get("description"),
                    author=pr_data.get("author"),
                    created_at=pr_data.get("created_at"),
                    merged_at=pr_data.get("merged_at"),
                    story_points=pr_data.get("story_points"),
                    labels=pr_data.get("labels", []),
                    commit_hashes=pr_data.get("commit_hashes", []),
                    # PR state fields (v4.0)
                    pr_state=pr_data.get("pr_state"),
                    closed_at=pr_data.get("closed_at"),
                    is_merged=pr_data.get("is_merged"),
                    # Enhanced PR tracking fields (v3.0)
                    review_comments_count=pr_data.get("review_comments", 0),
                    pr_comments_count=pr_data.get("pr_comments_count", 0),
                    approvals_count=pr_data.get("approvals_count", 0),
                    change_requests_count=pr_data.get("change_requests_count", 0),
                    reviewers=pr_data.get("reviewers", []),
                    approved_by=pr_data.get("approved_by", []),
                    time_to_first_review_hours=pr_data.get("time_to_first_review_hours"),
                    revision_count=pr_data.get("revision_count", 0),
                    changed_files=pr_data.get("changed_files", 0),
                    additions=pr_data.get("additions", 0),
                    deletions=pr_data.get("deletions", 0),
                )
                session.add(cached_pr)

    def cache_issue(self, platform: str, issue_data: dict[str, Any]) -> None:
        """Cache issue data from various platforms."""
        with self.get_session() as session:
            # Check if already exists
            existing = (
                session.query(IssueCache)
                .filter(
                    and_(
                        IssueCache.platform == platform,
                        IssueCache.issue_id == str(issue_data["id"]),
                    )
                )
                .first()
            )

            if existing:
                # Update existing
                existing.project_key = issue_data["project_key"]
                existing.title = issue_data.get("title")
                existing.description = issue_data.get("description")
                existing.status = issue_data.get("status")
                existing.assignee = issue_data.get("assignee")
                existing.created_at = issue_data.get("created_at")
                existing.updated_at = issue_data.get("updated_at")
                existing.resolved_at = issue_data.get("resolved_at")
                existing.story_points = issue_data.get("story_points")
                existing.labels = issue_data.get("labels", [])
                existing.platform_data = issue_data.get("platform_data", {})
                # Bug 1 fix: use timezone-aware UTC datetime instead of naive utcnow()
                existing.cached_at = datetime.now(timezone.utc)
            else:
                # Create new
                cached_issue = IssueCache(
                    platform=platform,
                    issue_id=str(issue_data["id"]),
                    project_key=issue_data["project_key"],
                    title=issue_data.get("title"),
                    description=issue_data.get("description"),
                    status=issue_data.get("status"),
                    assignee=issue_data.get("assignee"),
                    created_at=issue_data.get("created_at"),
                    updated_at=issue_data.get("updated_at"),
                    resolved_at=issue_data.get("resolved_at"),
                    story_points=issue_data.get("story_points"),
                    labels=issue_data.get("labels", []),
                    platform_data=issue_data.get("platform_data", {}),
                )
                session.add(cached_issue)

    def get_cached_issues(self, platform: str, project_key: str) -> list[dict[str, Any]]:
        """Get all cached issues for a platform and project."""
        with self.get_session() as session:
            issues = (
                session.query(IssueCache)
                .filter(
                    and_(IssueCache.platform == platform, IssueCache.project_key == project_key)
                )
                .all()
            )

            return [
                self._issue_to_dict(issue)
                for issue in issues
                if not self._is_stale(issue.cached_at)
            ]

    def bulk_store_commits(self, repo_path: str, commits: list[dict[str, Any]]) -> dict[str, Any]:
        """Store multiple commits using SQLAlchemy bulk operations for maximum performance.

        WHY: This method uses SQLAlchemy's bulk_insert_mappings which is significantly
        faster than individual inserts. It's designed for initial data loading where
        we know commits don't exist yet in the cache.

        DESIGN DECISION: Unlike cache_commits_batch which handles updates, this method
        only inserts new commits. Use this when you know commits are not in cache.

        Args:
            repo_path: Repository path
            commits: List of commit dictionaries to store

        Returns:
            Dictionary with operation statistics
        """
        if not commits:
            return {"inserted": 0, "time_seconds": 0}

        import time

        start_time = time.time()

        # Prepare mappings for bulk insert
        mappings = []
        for commit_data in commits:
            mapping = {
                "repo_path": repo_path,
                "commit_hash": commit_data["hash"],
                "author_name": commit_data.get("author_name"),
                "author_email": commit_data.get("author_email"),
                "message": commit_data.get("message"),
                "timestamp": commit_data.get("timestamp"),
                "branch": commit_data.get("branch"),
                "is_merge": commit_data.get("is_merge", False),
                "files_changed": commit_data.get(
                    "files_changed_count",
                    (
                        commit_data.get("files_changed", 0)
                        if isinstance(commit_data.get("files_changed"), int)
                        else len(commit_data.get("files_changed", []))
                    ),
                ),
                "insertions": commit_data.get("insertions", 0),
                "deletions": commit_data.get("deletions", 0),
                "filtered_insertions": commit_data.get(
                    "filtered_insertions", commit_data.get("insertions", 0)
                ),
                "filtered_deletions": commit_data.get(
                    "filtered_deletions", commit_data.get("deletions", 0)
                ),
                "complexity_delta": commit_data.get("complexity_delta", 0.0),
                "story_points": commit_data.get("story_points"),
                "ticket_references": commit_data.get("ticket_references", []),
                "cached_at": datetime.now(timezone.utc),
            }
            mappings.append(mapping)

        # Process in configurable batch sizes for memory efficiency
        inserted_count = 0
        with self.get_session() as session:
            for i in range(0, len(mappings), self.batch_size):
                batch = mappings[i : i + self.batch_size]
                try:
                    session.bulk_insert_mappings(CachedCommit, batch)
                    inserted_count += len(batch)
                except Exception as e:
                    # On error, fall back to individual inserts for this batch
                    logger.warning(f"Bulk insert failed, falling back to individual inserts: {e}")
                    session.rollback()  # Important: rollback failed transaction

                    for mapping in batch:
                        try:
                            # Create new record
                            new_commit = CachedCommit(**mapping)
                            session.add(new_commit)
                            session.flush()  # Try to save this individual record
                            inserted_count += 1
                        except Exception:
                            # Skip duplicate commits silently
                            session.rollback()  # Rollback this specific failure
                            continue

        elapsed = time.time() - start_time
        self.bulk_operations_count += 1
        self.bulk_operations_time += elapsed

        if self.debug_mode:
            rate = inserted_count / elapsed if elapsed > 0 else 0
            print(
                f"DEBUG: Bulk stored {inserted_count} commits in {elapsed:.3f}s ({rate:.0f} commits/sec)"
            )

        return {
            "inserted": inserted_count,
            "time_seconds": elapsed,
            "commits_per_second": inserted_count / elapsed if elapsed > 0 else 0,
        }

    def bulk_update_commits(self, repo_path: str, commits: list[dict[str, Any]]) -> dict[str, Any]:
        """Update multiple commits efficiently using bulk operations.

        WHY: Bulk updates are faster than individual updates when modifying many
        commits at once (e.g., after classification or enrichment).

        Args:
            repo_path: Repository path
            commits: List of commit dictionaries with updates

        Returns:
            Dictionary with operation statistics
        """
        if not commits:
            return {"updated": 0, "time_seconds": 0}

        import time

        start_time = time.time()

        with self.get_session() as session:
            # Get all commit hashes for bulk fetch
            commit_hashes = [c["hash"] for c in commits]

            # Bulk fetch existing commits to get their primary keys
            existing = {
                cached.commit_hash: cached
                for cached in session.query(CachedCommit)
                .filter(
                    and_(
                        CachedCommit.repo_path == repo_path,
                        CachedCommit.commit_hash.in_(commit_hashes),
                    )
                )
                .all()
            }

            # Prepare bulk update mappings with primary key
            update_mappings = []
            for commit_data in commits:
                if commit_data["hash"] in existing:
                    cached_record = existing[commit_data["hash"]]
                    # Must include primary key for bulk_update_mappings
                    update_mapping = {"id": cached_record.id}

                    # Map commit data fields to database columns
                    field_mapping = {
                        "author_name": commit_data.get("author_name"),
                        "author_email": commit_data.get("author_email"),
                        "message": commit_data.get("message"),
                        "timestamp": commit_data.get("timestamp"),
                        "branch": commit_data.get("branch"),
                        "is_merge": commit_data.get("is_merge"),
                        "files_changed": commit_data.get(
                            "files_changed_count",
                            (
                                commit_data.get("files_changed", 0)
                                if isinstance(commit_data.get("files_changed"), int)
                                else len(commit_data.get("files_changed", []))
                            ),
                        ),
                        "insertions": commit_data.get("insertions"),
                        "deletions": commit_data.get("deletions"),
                        "complexity_delta": commit_data.get("complexity_delta"),
                        "story_points": commit_data.get("story_points"),
                        "ticket_references": commit_data.get("ticket_references"),
                        "cached_at": datetime.now(timezone.utc),
                    }

                    # Only include non-None values in update
                    for key, value in field_mapping.items():
                        if value is not None:
                            update_mapping[key] = value

                    update_mappings.append(update_mapping)

            # Perform bulk update
            if update_mappings:
                session.bulk_update_mappings(CachedCommit, update_mappings)

        elapsed = time.time() - start_time
        self.bulk_operations_count += 1
        self.bulk_operations_time += elapsed

        if self.debug_mode:
            print(f"DEBUG: Bulk updated {len(update_mappings)} commits in {elapsed:.3f}s")

        return {
            "updated": len(update_mappings),
            "time_seconds": elapsed,
            "commits_per_second": len(update_mappings) / elapsed if elapsed > 0 else 0,
        }

    def bulk_exists(self, repo_path: str, commit_hashes: list[str]) -> dict[str, bool]:
        """Check existence of multiple commits in a single query.

        WHY: Checking existence of many commits individually is inefficient.
        This method uses a single query to check all commits at once.

        Args:
            repo_path: Repository path
            commit_hashes: List of commit hashes to check

        Returns:
            Dictionary mapping commit hash to existence boolean
        """
        if not commit_hashes:
            return {}

        with self.get_session() as session:
            # Query for existing commits
            existing = set(
                row[0]
                for row in session.query(CachedCommit.commit_hash)
                .filter(
                    and_(
                        CachedCommit.repo_path == repo_path,
                        CachedCommit.commit_hash.in_(commit_hashes),
                    )
                )
                .all()
            )

        # Build result dictionary
        return {hash: hash in existing for hash in commit_hashes}

    def bulk_get_commits(
        self, repo_path: str, commit_hashes: list[str], include_stale: bool = False
    ) -> dict[str, dict[str, Any]]:
        """Retrieve multiple commits with enhanced performance.

        WHY: Enhanced version of get_cached_commits_bulk with better performance
        characteristics and optional stale data inclusion.

        Args:
            repo_path: Repository path
            commit_hashes: List of commit hashes to retrieve
            include_stale: Whether to include stale entries (default: False)

        Returns:
            Dictionary mapping commit hash to commit data
        """
        if not commit_hashes:
            return {}

        import time

        start_time = time.time()

        # Process in batches to avoid query size limits
        all_results = {}
        for i in range(0, len(commit_hashes), self.batch_size):
            batch_hashes = commit_hashes[i : i + self.batch_size]

            with self.get_session() as session:
                cached_results = (
                    session.query(CachedCommit)
                    .filter(
                        and_(
                            CachedCommit.repo_path == repo_path,
                            CachedCommit.commit_hash.in_(batch_hashes),
                        )
                    )
                    .all()
                )

                for cached in cached_results:
                    if include_stale or not self._is_stale(cached.cached_at):
                        all_results[cached.commit_hash] = self._commit_to_dict(cached)

        # Track performance
        elapsed = time.time() - start_time
        hits = len(all_results)
        misses = len(commit_hashes) - hits

        self.cache_hits += hits
        self.cache_misses += misses
        self.bulk_operations_count += 1
        self.bulk_operations_time += elapsed

        if self.debug_mode:
            hit_rate = (hits / len(commit_hashes)) * 100 if commit_hashes else 0
            print(
                f"DEBUG: Bulk get {hits}/{len(commit_hashes)} commits in {elapsed:.3f}s ({hit_rate:.1f}% hit rate)"
            )

        return all_results

    def _get_files_changed_count(self, commit: git.Commit) -> int:
        """Get the number of files changed using reliable git command."""
        parent = commit.parents[0] if commit.parents else None

        try:
            repo = commit.repo
            if parent:
                diff_output = repo.git.diff(parent.hexsha, commit.hexsha, "--numstat")
            else:
                diff_output = repo.git.show(commit.hexsha, "--numstat", "--format=")

            file_count = 0
            for line in diff_output.strip().split("\n"):
                if line.strip() and "\t" in line:
                    file_count += 1

            return file_count
        except Exception:
            return 0

    def _get_insertions_count(self, commit: git.Commit) -> int:
        """Get the number of insertions using reliable git command."""
        parent = commit.parents[0] if commit.parents else None

        try:
            repo = commit.repo
            if parent:
                diff_output = repo.git.diff(parent.hexsha, commit.hexsha, "--numstat")
            else:
                diff_output = repo.git.show(commit.hexsha, "--numstat", "--format=")

            total_insertions = 0
            for line in diff_output.strip().split("\n"):
                if not line.strip():
                    continue

                parts = line.split("\t")
                if len(parts) >= 3:
                    try:
                        insertions = int(parts[0]) if parts[0] != "-" else 0
                        total_insertions += insertions
                    except ValueError:
                        continue

            return total_insertions
        except Exception:
            return 0

    def _get_deletions_count(self, commit: git.Commit) -> int:
        """Get the number of deletions using reliable git command."""
        parent = commit.parents[0] if commit.parents else None

        try:
            repo = commit.repo
            if parent:
                diff_output = repo.git.diff(parent.hexsha, commit.hexsha, "--numstat")
            else:
                diff_output = repo.git.show(commit.hexsha, "--numstat", "--format=")

            total_deletions = 0
            for line in diff_output.strip().split("\n"):
                if not line.strip():
                    continue

                parts = line.split("\t")
                if len(parts) >= 3:
                    try:
                        deletions = int(parts[1]) if parts[1] != "-" else 0
                        total_deletions += deletions
                    except ValueError:
                        continue

            return total_deletions
        except Exception:
            return 0
