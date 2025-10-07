"""Caching layer for Git analysis with SQLite backend."""

import hashlib
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Union

import git
from sqlalchemy import and_

from ..constants import BatchSizes, CacheTTL, Thresholds
from ..models.database import (
    CachedCommit,
    Database,
    IssueCache,
    PullRequestCache,
    RepositoryAnalysisStatus,
)

logger = logging.getLogger(__name__)


class GitAnalysisCache:
    """Cache for Git analysis results."""

    def __init__(
        self,
        cache_dir: Union[Path, str],
        ttl_hours: int = CacheTTL.ONE_WEEK_HOURS,
        batch_size: int = BatchSizes.COMMIT_STORAGE,
    ) -> None:
        """Initialize cache with SQLite backend and configurable batch size.

        WHY: Adding configurable batch size allows tuning for different repository
        sizes and system capabilities. Default of 1000 balances memory usage with
        performance gains from bulk operations.

        Args:
            cache_dir: Directory for cache database
            ttl_hours: Time-to-live for cache entries in hours (default: 168 = 1 week)
            batch_size: Default batch size for bulk operations (default: 1000)
        """
        self.cache_dir = Path(cache_dir)  # Ensure it's a Path object
        self.ttl_hours = ttl_hours
        self.batch_size = batch_size
        self.db = Database(self.cache_dir / "gitflow_cache.db")

        # Cache performance tracking with enhanced metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_start_time = datetime.now()
        self.bulk_operations_count = 0
        self.bulk_operations_time = 0.0
        self.single_operations_count = 0
        self.single_operations_time = 0.0
        self.total_bytes_cached = 0

        # Debug mode controlled by environment variable
        self.debug_mode = os.getenv("GITFLOW_DEBUG", "").lower() in ("1", "true", "yes")

    @contextmanager
    def get_session(self) -> Any:
        """Get database session context manager."""
        session = self.db.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

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
                existing.cached_at = datetime.utcnow()
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
                    existing.cached_at = datetime.utcnow()
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
                existing.cached_at = datetime.utcnow()
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
                existing.cached_at = datetime.utcnow()
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

    def clear_stale_cache(self) -> None:
        """Remove stale cache entries."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.ttl_hours)

        with self.get_session() as session:
            session.query(CachedCommit).filter(CachedCommit.cached_at < cutoff_time).delete()

            session.query(PullRequestCache).filter(
                PullRequestCache.cached_at < cutoff_time
            ).delete()

            session.query(IssueCache).filter(IssueCache.cached_at < cutoff_time).delete()

            # Also clear stale repository analysis status
            session.query(RepositoryAnalysisStatus).filter(
                RepositoryAnalysisStatus.last_updated < cutoff_time
            ).delete()

    def clear_all_cache(self) -> dict[str, int]:
        """Clear all cache entries including repository analysis status.

        WHY: Used by --clear-cache flag to force complete re-analysis.
        Returns counts of cleared entries for user feedback.

        Returns:
            Dictionary with counts of cleared entries by type
        """
        with self.get_session() as session:
            # Count before clearing
            commit_count = session.query(CachedCommit).count()
            pr_count = session.query(PullRequestCache).count()
            issue_count = session.query(IssueCache).count()
            status_count = session.query(RepositoryAnalysisStatus).count()

            # Clear all entries
            session.query(CachedCommit).delete()
            session.query(PullRequestCache).delete()
            session.query(IssueCache).delete()
            session.query(RepositoryAnalysisStatus).delete()

            return {
                "commits": commit_count,
                "pull_requests": pr_count,
                "issues": issue_count,
                "repository_status": status_count,
                "total": commit_count + pr_count + issue_count + status_count,
            }

    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics including external API cache performance."""
        with self.get_session() as session:
            # Basic counts
            total_commits = session.query(CachedCommit).count()
            total_prs = session.query(PullRequestCache).count()
            total_issues = session.query(IssueCache).count()

            # Platform-specific issue counts
            jira_issues = session.query(IssueCache).filter(IssueCache.platform == "jira").count()
            github_issues = (
                session.query(IssueCache).filter(IssueCache.platform == "github").count()
            )

            # Stale entries
            cutoff_time = datetime.utcnow() - timedelta(hours=self.ttl_hours)
            stale_commits = (
                session.query(CachedCommit).filter(CachedCommit.cached_at < cutoff_time).count()
            )
            stale_prs = (
                session.query(PullRequestCache)
                .filter(PullRequestCache.cached_at < cutoff_time)
                .count()
            )
            stale_issues = (
                session.query(IssueCache).filter(IssueCache.cached_at < cutoff_time).count()
            )

            # Performance metrics
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

            # Bulk vs Single operation performance
            avg_bulk_time = (
                self.bulk_operations_time / self.bulk_operations_count
                if self.bulk_operations_count > 0
                else 0
            )
            avg_single_time = (
                self.single_operations_time / self.single_operations_count
                if self.single_operations_count > 0
                else 0
            )
            bulk_speedup = (
                avg_single_time / avg_bulk_time if avg_bulk_time > 0 and avg_single_time > 0 else 0
            )

            # Estimated time savings (conservative estimates)
            commit_time_saved = self.cache_hits * 0.1  # 0.1 seconds per commit analysis
            api_time_saved = (total_issues * 0.5) + (total_prs * 0.3)  # API call time savings
            bulk_time_saved = (
                self.bulk_operations_count * 2.0
            )  # Estimated 2 seconds saved per bulk op
            total_time_saved = commit_time_saved + api_time_saved + bulk_time_saved

            # Database file size
            db_file = self.cache_dir / "gitflow_cache.db"
            db_size_mb = db_file.stat().st_size / (1024 * 1024) if db_file.exists() else 0

            # Session duration
            session_duration = (datetime.now() - self.cache_start_time).total_seconds()

            # Cache efficiency metrics
            fresh_commits = total_commits - stale_commits
            fresh_prs = total_prs - stale_prs
            fresh_issues = total_issues - stale_issues
            total_fresh_entries = fresh_commits + fresh_prs + fresh_issues

            stats = {
                # Counts by type
                "cached_commits": total_commits,
                "cached_prs": total_prs,
                "cached_issues": total_issues,
                "cached_jira_issues": jira_issues,
                "cached_github_issues": github_issues,
                # Freshness analysis
                "stale_commits": stale_commits,
                "stale_prs": stale_prs,
                "stale_issues": stale_issues,
                "fresh_commits": fresh_commits,
                "fresh_prs": fresh_prs,
                "fresh_issues": fresh_issues,
                "total_fresh_entries": total_fresh_entries,
                "freshness_rate_percent": (
                    total_fresh_entries / max(1, total_commits + total_prs + total_issues)
                )
                * 100,
                # Performance metrics
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "total_requests": total_requests,
                "hit_rate_percent": hit_rate,
                # Bulk operation metrics
                "bulk_operations_count": self.bulk_operations_count,
                "bulk_operations_time_seconds": self.bulk_operations_time,
                "avg_bulk_operation_time": avg_bulk_time,
                "single_operations_count": self.single_operations_count,
                "single_operations_time_seconds": self.single_operations_time,
                "avg_single_operation_time": avg_single_time,
                "bulk_speedup_factor": bulk_speedup,
                # Time savings
                "commit_analysis_time_saved_seconds": commit_time_saved,
                "api_call_time_saved_seconds": api_time_saved,
                "bulk_operations_time_saved_seconds": bulk_time_saved,
                "total_time_saved_seconds": total_time_saved,
                "total_time_saved_minutes": total_time_saved / 60,
                # Backward compatibility aliases for CLI
                "time_saved_seconds": total_time_saved,
                "time_saved_minutes": total_time_saved / 60,
                "estimated_api_calls_avoided": total_issues + total_prs,
                # Storage metrics
                "database_size_mb": db_size_mb,
                "session_duration_seconds": session_duration,
                "avg_entries_per_mb": (total_commits + total_prs + total_issues)
                / max(0.1, db_size_mb),
                "total_bytes_cached": self.total_bytes_cached,
                # Configuration
                "ttl_hours": self.ttl_hours,
                "batch_size": self.batch_size,
                "debug_mode": self.debug_mode,
            }

            return stats

    def print_cache_performance_summary(self) -> None:
        """Print a user-friendly cache performance summary.

        WHY: Users need visibility into cache performance to understand
        why repeated runs are faster and to identify any caching issues.
        This provides actionable insights into cache effectiveness.
        """
        stats = self.get_cache_stats()

        print("📊 Cache Performance Summary")
        print("─" * 50)

        # Cache contents
        print("📦 Cache Contents:")
        print(
            f"   • Commits: {stats['cached_commits']:,} ({stats['fresh_commits']:,} fresh, {stats['stale_commits']:,} stale)"
        )
        print(
            f"   • Pull Requests: {stats['cached_prs']:,} ({stats['fresh_prs']:,} fresh, {stats['stale_prs']:,} stale)"
        )
        print(
            f"   • Issues: {stats['cached_issues']:,} ({stats['fresh_issues']:,} fresh, {stats['stale_issues']:,} stale)"
        )

        if stats["cached_jira_issues"] > 0:
            print(f"     ├─ JIRA: {stats['cached_jira_issues']:,} issues")
        if stats["cached_github_issues"] > 0:
            print(f"     └─ GitHub: {stats['cached_github_issues']:,} issues")

        # Performance metrics
        if stats["total_requests"] > 0:
            print("\n⚡ Session Performance:")
            print(
                f"   • Cache Hit Rate: {stats['hit_rate_percent']:.1f}% ({stats['cache_hits']:,}/{stats['total_requests']:,})"
            )

            if stats["total_time_saved_minutes"] > 1:
                print(f"   • Time Saved: {stats['total_time_saved_minutes']:.1f} minutes")
            else:
                print(f"   • Time Saved: {stats['total_time_saved_seconds']:.1f} seconds")

            if stats["estimated_api_calls_avoided"] > 0:
                print(f"   • API Calls Avoided: {stats['estimated_api_calls_avoided']:,}")

        # Bulk operation performance
        if stats["bulk_operations_count"] > 0:
            print("\n🚀 Bulk Operations:")
            print(f"   • Bulk Operations: {stats['bulk_operations_count']:,}")
            print(f"   • Avg Bulk Time: {stats['avg_bulk_operation_time']:.3f}s")
            if stats["bulk_speedup_factor"] > 1:
                print(
                    f"   • Speedup Factor: {stats['bulk_speedup_factor']:.1f}x faster than single ops"
                )
            print(f"   • Batch Size: {stats['batch_size']:,} items")

        # Storage info
        print("\n💾 Storage:")
        print(f"   • Database Size: {stats['database_size_mb']:.1f} MB")
        print(f"   • Cache TTL: {stats['ttl_hours']} hours")
        print(f"   • Overall Freshness: {stats['freshness_rate_percent']:.1f}%")

        # Performance insights
        if stats["hit_rate_percent"] > 80:
            print("   ✅ Excellent cache performance!")
        elif stats["hit_rate_percent"] > Thresholds.CACHE_HIT_RATE_GOOD:
            print("   👍 Good cache performance")
        elif stats["total_requests"] > 0:
            print("   ⚠️  Consider clearing stale cache entries")

        print()

    def validate_cache(self) -> dict[str, Any]:
        """Validate cache consistency and integrity.

        WHY: Cache validation ensures data integrity and identifies issues
        that could cause analysis errors or inconsistent results.

        Returns:
            Dictionary with validation results and issues found
        """
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "stats": {},
        }

        with self.get_session() as session:
            try:
                # Check for missing required fields
                commits_without_hash = (
                    session.query(CachedCommit).filter(CachedCommit.commit_hash.is_(None)).count()
                )

                if commits_without_hash > 0:
                    validation_results["issues"].append(
                        f"Found {commits_without_hash} cached commits without hash"
                    )
                    validation_results["is_valid"] = False

                # Check for duplicate commits
                from sqlalchemy import func

                duplicates = (
                    session.query(
                        CachedCommit.repo_path,
                        CachedCommit.commit_hash,
                        func.count().label("count"),
                    )
                    .group_by(CachedCommit.repo_path, CachedCommit.commit_hash)
                    .having(func.count() > 1)
                    .all()
                )

                if duplicates:
                    validation_results["warnings"].append(
                        f"Found {len(duplicates)} duplicate commit entries"
                    )

                # Check for very old entries (older than 2 * TTL)
                very_old_cutoff = datetime.utcnow() - timedelta(hours=self.ttl_hours * 2)
                very_old_count = (
                    session.query(CachedCommit)
                    .filter(CachedCommit.cached_at < very_old_cutoff)
                    .count()
                )

                if very_old_count > 0:
                    validation_results["warnings"].append(
                        f"Found {very_old_count} very old cache entries (older than {self.ttl_hours * 2}h)"
                    )

                # Basic integrity checks
                commits_with_negative_changes = (
                    session.query(CachedCommit)
                    .filter(
                        (CachedCommit.files_changed < 0)
                        | (CachedCommit.insertions < 0)
                        | (CachedCommit.deletions < 0)
                    )
                    .count()
                )

                if commits_with_negative_changes > 0:
                    validation_results["issues"].append(
                        f"Found {commits_with_negative_changes} commits with negative change counts"
                    )
                    validation_results["is_valid"] = False

                # Statistics
                validation_results["stats"] = {
                    "total_commits": session.query(CachedCommit).count(),
                    "duplicates": len(duplicates),
                    "very_old_entries": very_old_count,
                    "invalid_commits": commits_without_hash + commits_with_negative_changes,
                }

            except Exception as e:
                validation_results["issues"].append(f"Validation error: {str(e)}")
                validation_results["is_valid"] = False

        return validation_results

    def warm_cache(self, repo_paths: list[str], weeks: int = 12) -> dict[str, Any]:
        """Pre-warm cache by analyzing all commits in repositories.

        WHY: Cache warming ensures all commits are pre-analyzed and cached,
        making subsequent runs much faster. This is especially useful for
        CI/CD environments or when analyzing the same repositories repeatedly.

        Args:
            repo_paths: List of repository paths to warm cache for
            weeks: Number of weeks of history to warm (default: 12)

        Returns:
            Dictionary with warming results and statistics
        """
        from datetime import datetime, timedelta

        import git

        from .progress import get_progress_service

        warming_results = {
            "repos_processed": 0,
            "total_commits_found": 0,
            "commits_cached": 0,
            "commits_already_cached": 0,
            "errors": [],
            "duration_seconds": 0,
        }

        start_time = datetime.now()
        cutoff_date = datetime.now() - timedelta(weeks=weeks)

        try:
            for repo_path in repo_paths:
                try:
                    from pathlib import Path

                    repo_path_obj = Path(repo_path)
                    repo = git.Repo(repo_path)

                    # Get commits from the specified time period
                    commits = list(
                        repo.iter_commits(all=True, since=cutoff_date.strftime("%Y-%m-%d"))
                    )

                    warming_results["total_commits_found"] += len(commits)

                    # Check which commits are already cached
                    commit_hashes = [c.hexsha for c in commits]
                    cached_commits = self.get_cached_commits_bulk(str(repo_path_obj), commit_hashes)
                    already_cached = len(cached_commits)
                    to_analyze = len(commits) - already_cached

                    warming_results["commits_already_cached"] += already_cached

                    if to_analyze > 0:
                        # Use centralized progress service
                        progress = get_progress_service()

                        # Analyze uncached commits with progress bar
                        with progress.progress(
                            total=to_analyze,
                            description=f"Warming cache for {repo_path_obj.name}",
                            unit="commits",
                            leave=False,
                        ) as ctx:
                            new_commits = []
                            for commit in commits:
                                if commit.hexsha not in cached_commits:
                                    # Basic commit analysis (minimal for cache warming)
                                    commit_data = self._analyze_commit_minimal(
                                        repo, commit, repo_path_obj
                                    )
                                    new_commits.append(commit_data)
                                    progress.update(ctx, 1)

                                    # Batch cache commits for efficiency
                                    if len(new_commits) >= 100:
                                        self.cache_commits_batch(str(repo_path_obj), new_commits)
                                        warming_results["commits_cached"] += len(new_commits)
                                        new_commits = []

                            # Cache remaining commits
                            if new_commits:
                                self.cache_commits_batch(str(repo_path_obj), new_commits)
                                warming_results["commits_cached"] += len(new_commits)

                    warming_results["repos_processed"] += 1

                except Exception as e:
                    warming_results["errors"].append(f"Error processing {repo_path}: {str(e)}")

        except Exception as e:
            warming_results["errors"].append(f"General error during cache warming: {str(e)}")

        warming_results["duration_seconds"] = (datetime.now() - start_time).total_seconds()
        return warming_results

    def _analyze_commit_minimal(
        self, repo: git.Repo, commit: git.Commit, repo_path: Path
    ) -> dict[str, Any]:
        """Minimal commit analysis for cache warming.

        WHY: Cache warming doesn't need full analysis complexity,
        just enough data to populate the cache effectively.
        """
        # Basic commit data
        commit_data = {
            "hash": commit.hexsha,
            "author_name": commit.author.name,
            "author_email": commit.author.email,
            "message": commit.message,
            "timestamp": commit.committed_datetime,
            "is_merge": len(commit.parents) > 1,
            "files_changed": self._get_files_changed_count(commit),
            "insertions": self._get_insertions_count(commit),
            "deletions": self._get_deletions_count(commit),
            "complexity_delta": 0.0,  # Skip complexity calculation for warming
            "story_points": None,  # Skip story point extraction for warming
            "ticket_references": [],  # Skip ticket analysis for warming
        }

        # Try to get branch info (if available)
        try:
            branches = repo.git.branch("--contains", commit.hexsha).split("\n")
            commit_data["branch"] = branches[0].strip("* ") if branches else "unknown"
        except Exception:
            commit_data["branch"] = "unknown"

        return commit_data

    def _is_stale(self, cached_at: datetime) -> bool:
        """Check if cache entry is stale."""
        if self.ttl_hours == 0:  # No expiration
            return False
        return cached_at < datetime.utcnow() - timedelta(hours=self.ttl_hours)

    def _commit_to_dict(self, commit: CachedCommit) -> dict[str, Any]:
        """Convert CachedCommit to dictionary."""
        return {
            "hash": commit.commit_hash,
            "author_name": commit.author_name,
            "author_email": commit.author_email,
            "message": commit.message,
            "timestamp": commit.timestamp,
            "branch": commit.branch,
            "is_merge": commit.is_merge,
            "files_changed": commit.files_changed,
            "insertions": commit.insertions,
            "deletions": commit.deletions,
            "filtered_insertions": getattr(commit, "filtered_insertions", commit.insertions),
            "filtered_deletions": getattr(commit, "filtered_deletions", commit.deletions),
            "complexity_delta": commit.complexity_delta,
            "story_points": commit.story_points,
            "ticket_references": commit.ticket_references or [],
        }

    def _pr_to_dict(self, pr: PullRequestCache) -> dict[str, Any]:
        """Convert PullRequestCache to dictionary."""
        return {
            "number": pr.pr_number,
            "title": pr.title,
            "description": pr.description,
            "author": pr.author,
            "created_at": pr.created_at,
            "merged_at": pr.merged_at,
            "story_points": pr.story_points,
            "labels": pr.labels or [],
            "commit_hashes": pr.commit_hashes or [],
        }

    def _issue_to_dict(self, issue: IssueCache) -> dict[str, Any]:
        """Convert IssueCache to dictionary."""
        return {
            "platform": issue.platform,
            "id": issue.issue_id,
            "project_key": issue.project_key,
            "title": issue.title,
            "description": issue.description,
            "status": issue.status,
            "assignee": issue.assignee,
            "created_at": issue.created_at,
            "updated_at": issue.updated_at,
            "resolved_at": issue.resolved_at,
            "story_points": issue.story_points,
            "labels": issue.labels or [],
            "platform_data": issue.platform_data or {},
        }

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

    def get_repository_analysis_status(
        self,
        repo_path: str,
        analysis_start: datetime,
        analysis_end: datetime,
        config_hash: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Check if repository analysis is complete for the given period.

        WHY: Enables "fetch once, report many" by tracking which repositories
        have been fully analyzed. Prevents re-fetching Git data when generating
        different report formats from the same cached data.

        CRITICAL FIX: Now verifies actual commits exist in cache, not just metadata.
        This prevents "Using cached data (X commits)" when commits aren't actually stored.

        Args:
            repo_path: Path to the repository
            analysis_start: Start of the analysis period
            analysis_end: End of the analysis period
            config_hash: Optional hash of relevant configuration to detect changes

        Returns:
            Dictionary with analysis status or None if not found/incomplete
        """
        with self.get_session() as session:
            status = (
                session.query(RepositoryAnalysisStatus)
                .filter(
                    and_(
                        RepositoryAnalysisStatus.repo_path == repo_path,
                        RepositoryAnalysisStatus.analysis_start == analysis_start,
                        RepositoryAnalysisStatus.analysis_end == analysis_end,
                        RepositoryAnalysisStatus.status == "completed",
                    )
                )
                .first()
            )

            if not status:
                return None

            # Check if configuration has changed (invalidates cache)
            if config_hash and status.config_hash != config_hash:
                return None

            # CRITICAL FIX: Verify actual commits exist in the database
            # Don't trust metadata if commits aren't actually stored
            actual_commit_count = (
                session.query(CachedCommit)
                .filter(
                    and_(
                        CachedCommit.repo_path == repo_path,
                        CachedCommit.timestamp >= analysis_start,
                        CachedCommit.timestamp <= analysis_end,
                        # Only count non-stale commits
                        CachedCommit.cached_at
                        >= datetime.utcnow() - timedelta(hours=self.ttl_hours),
                    )
                )
                .count()
            )

            # If metadata says we have commits but no commits are actually stored,
            # force a fresh fetch by returning None
            if status.commit_count > 0 and actual_commit_count == 0:
                if self.debug_mode:
                    print(
                        f"DEBUG: Metadata claims {status.commit_count} commits but found 0 in cache - forcing fresh fetch"
                    )
                # Log warning about inconsistent cache state
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Cache inconsistency detected for {repo_path}: "
                    f"metadata reports {status.commit_count} commits but "
                    f"actual stored commits: {actual_commit_count}. Forcing fresh analysis."
                )
                return None

            # Update the commit count to reflect actual stored commits
            # This ensures the UI shows accurate information
            status.commit_count = actual_commit_count

            # Return status information
            return {
                "repo_path": status.repo_path,
                "repo_name": status.repo_name,
                "project_key": status.project_key,
                "analysis_start": status.analysis_start,
                "analysis_end": status.analysis_end,
                "weeks_analyzed": status.weeks_analyzed,
                "git_analysis_complete": status.git_analysis_complete,
                "commit_count": actual_commit_count,  # Use verified count, not metadata
                "pr_analysis_complete": status.pr_analysis_complete,
                "pr_count": status.pr_count,
                "ticket_analysis_complete": status.ticket_analysis_complete,
                "ticket_count": status.ticket_count,
                "identity_resolution_complete": status.identity_resolution_complete,
                "unique_developers": status.unique_developers,
                "last_updated": status.last_updated,
                "processing_time_seconds": status.processing_time_seconds,
                "cache_hit_rate_percent": status.cache_hit_rate_percent,
                "config_hash": status.config_hash,
            }

    def mark_repository_analysis_complete(
        self,
        repo_path: str,
        repo_name: str,
        project_key: str,
        analysis_start: datetime,
        analysis_end: datetime,
        weeks_analyzed: int,
        commit_count: int = 0,
        pr_count: int = 0,
        ticket_count: int = 0,
        unique_developers: int = 0,
        processing_time_seconds: Optional[float] = None,
        cache_hit_rate_percent: Optional[float] = None,
        config_hash: Optional[str] = None,
    ) -> None:
        """Mark repository analysis as complete for the given period.

        WHY: Records successful completion of repository analysis to enable
        cache-first workflow. Subsequent runs can skip re-analysis and go
        directly to report generation.

        Args:
            repo_path: Path to the repository
            repo_name: Display name for the repository
            project_key: Project key for the repository
            analysis_start: Start of the analysis period
            analysis_end: End of the analysis period
            weeks_analyzed: Number of weeks analyzed
            commit_count: Number of commits analyzed
            pr_count: Number of pull requests analyzed
            ticket_count: Number of tickets analyzed
            unique_developers: Number of unique developers found
            processing_time_seconds: Time taken for analysis
            cache_hit_rate_percent: Cache hit rate during analysis
            config_hash: Hash of relevant configuration
        """
        with self.get_session() as session:
            # Check if status already exists
            existing = (
                session.query(RepositoryAnalysisStatus)
                .filter(
                    and_(
                        RepositoryAnalysisStatus.repo_path == repo_path,
                        RepositoryAnalysisStatus.analysis_start == analysis_start,
                        RepositoryAnalysisStatus.analysis_end == analysis_end,
                    )
                )
                .first()
            )

            if existing:
                # Update existing record
                existing.repo_name = repo_name
                existing.project_key = project_key
                existing.weeks_analyzed = weeks_analyzed
                existing.git_analysis_complete = True
                existing.commit_count = commit_count
                existing.pr_analysis_complete = True
                existing.pr_count = pr_count
                existing.ticket_analysis_complete = True
                existing.ticket_count = ticket_count
                existing.identity_resolution_complete = True
                existing.unique_developers = unique_developers
                existing.processing_time_seconds = processing_time_seconds
                existing.cache_hit_rate_percent = cache_hit_rate_percent
                existing.config_hash = config_hash
                existing.status = "completed"
                existing.error_message = None
                existing.last_updated = datetime.utcnow()
            else:
                # Create new record
                status = RepositoryAnalysisStatus(
                    repo_path=repo_path,
                    repo_name=repo_name,
                    project_key=project_key,
                    analysis_start=analysis_start,
                    analysis_end=analysis_end,
                    weeks_analyzed=weeks_analyzed,
                    git_analysis_complete=True,
                    commit_count=commit_count,
                    pr_analysis_complete=True,
                    pr_count=pr_count,
                    ticket_analysis_complete=True,
                    ticket_count=ticket_count,
                    identity_resolution_complete=True,
                    unique_developers=unique_developers,
                    processing_time_seconds=processing_time_seconds,
                    cache_hit_rate_percent=cache_hit_rate_percent,
                    config_hash=config_hash,
                    status="completed",
                )
                session.add(status)

    def mark_repository_analysis_failed(
        self,
        repo_path: str,
        repo_name: str,
        analysis_start: datetime,
        analysis_end: datetime,
        error_message: str,
        config_hash: Optional[str] = None,
    ) -> None:
        """Mark repository analysis as failed.

        Args:
            repo_path: Path to the repository
            repo_name: Display name for the repository
            analysis_start: Start of the analysis period
            analysis_end: End of the analysis period
            error_message: Error message describing the failure
            config_hash: Hash of relevant configuration
        """
        with self.get_session() as session:
            # Check if status already exists
            existing = (
                session.query(RepositoryAnalysisStatus)
                .filter(
                    and_(
                        RepositoryAnalysisStatus.repo_path == repo_path,
                        RepositoryAnalysisStatus.analysis_start == analysis_start,
                        RepositoryAnalysisStatus.analysis_end == analysis_end,
                    )
                )
                .first()
            )

            if existing:
                existing.repo_name = repo_name
                existing.status = "failed"
                existing.error_message = error_message
                existing.config_hash = config_hash
                existing.last_updated = datetime.utcnow()
            else:
                # Create new failed record
                status = RepositoryAnalysisStatus(
                    repo_path=repo_path,
                    repo_name=repo_name,
                    project_key="unknown",
                    analysis_start=analysis_start,
                    analysis_end=analysis_end,
                    weeks_analyzed=0,
                    status="failed",
                    error_message=error_message,
                    config_hash=config_hash,
                )
                session.add(status)

    def clear_repository_analysis_status(
        self, repo_path: Optional[str] = None, older_than_days: Optional[int] = None
    ) -> int:
        """Clear repository analysis status records.

        WHY: Allows forcing re-analysis by clearing cached status records.
        Used by --force-fetch flag and for cleanup of old status records.

        Args:
            repo_path: Specific repository path to clear (all repos if None)
            older_than_days: Clear records older than N days (all if None)

        Returns:
            Number of records cleared
        """
        with self.get_session() as session:
            query = session.query(RepositoryAnalysisStatus)

            if repo_path:
                query = query.filter(RepositoryAnalysisStatus.repo_path == repo_path)

            if older_than_days:
                cutoff = datetime.utcnow() - timedelta(days=older_than_days)
                query = query.filter(RepositoryAnalysisStatus.last_updated < cutoff)

            count = query.count()
            query.delete()

            return count

    def get_analysis_status_summary(self) -> dict[str, Any]:
        """Get summary of repository analysis status records.

        Returns:
            Dictionary with summary statistics
        """
        with self.get_session() as session:
            from sqlalchemy import func

            # Count by status
            status_counts = dict(
                session.query(RepositoryAnalysisStatus.status, func.count().label("count"))
                .group_by(RepositoryAnalysisStatus.status)
                .all()
            )

            # Total records
            total_records = session.query(RepositoryAnalysisStatus).count()

            # Recent activity (last 7 days)
            recent_cutoff = datetime.utcnow() - timedelta(days=7)
            recent_completed = (
                session.query(RepositoryAnalysisStatus)
                .filter(
                    and_(
                        RepositoryAnalysisStatus.status == "completed",
                        RepositoryAnalysisStatus.last_updated >= recent_cutoff,
                    )
                )
                .count()
            )

            # Average processing time for completed analyses
            avg_processing_time = (
                session.query(func.avg(RepositoryAnalysisStatus.processing_time_seconds))
                .filter(
                    and_(
                        RepositoryAnalysisStatus.status == "completed",
                        RepositoryAnalysisStatus.processing_time_seconds.isnot(None),
                    )
                )
                .scalar()
            )

            return {
                "total_records": total_records,
                "status_counts": status_counts,
                "recent_completed": recent_completed,
                "avg_processing_time_seconds": avg_processing_time,
                "completed_count": status_counts.get("completed", 0),
                "failed_count": status_counts.get("failed", 0),
                "pending_count": status_counts.get("pending", 0),
            }

    @staticmethod
    def generate_config_hash(
        branch_mapping_rules: Optional[dict] = None,
        ticket_platforms: Optional[list] = None,
        exclude_paths: Optional[list] = None,
        ml_categorization_enabled: bool = False,
        additional_config: Optional[dict] = None,
    ) -> str:
        """Generate MD5 hash of relevant configuration for cache invalidation.

        WHY: Configuration changes can affect analysis results, so we need to
        detect when cached analysis is no longer valid due to config changes.

        Args:
            branch_mapping_rules: Branch to project mapping rules
            ticket_platforms: Allowed ticket platforms
            exclude_paths: Paths to exclude from analysis
            ml_categorization_enabled: Whether ML categorization is enabled
            additional_config: Any additional configuration to include

        Returns:
            MD5 hash string representing the configuration
        """
        config_data = {
            "branch_mapping_rules": branch_mapping_rules or {},
            "ticket_platforms": sorted(ticket_platforms or []),
            "exclude_paths": sorted(exclude_paths or []),
            "ml_categorization_enabled": ml_categorization_enabled,
            "additional_config": additional_config or {},
        }

        # Convert to JSON string with sorted keys for consistent hashing
        config_json = json.dumps(config_data, sort_keys=True, default=str)

        # Generate MD5 hash
        return hashlib.md5(config_json.encode()).hexdigest()
