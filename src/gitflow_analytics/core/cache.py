"""Caching layer for Git analysis with SQLite backend."""

import hashlib
import json
import logging
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
    WeeklyFetchStatus,
)
from ..utils.commit_utils import extract_co_authors as _extract_co_authors_from_message
from ..utils.debug import is_debug_mode

logger = logging.getLogger(__name__)


from .cache_commits import CommitCacheMixin
from .cache_weekly import WeeklyCacheMixin

class GitAnalysisCache(CommitCacheMixin, WeeklyCacheMixin):
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
        self.debug_mode = is_debug_mode()

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

    def clear_stale_cache(self) -> None:
        """Remove stale cache entries."""
        # Bug 1 fix: use timezone-aware UTC datetime instead of naive utcnow()
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.ttl_hours)

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

        BUG FIX: Now also clears WeeklyFetchStatus so that --clear-cache forces
        a full re-fetch on the next run.  Previously, WeeklyFetchStatus rows
        survived clear_all_cache, causing the incremental fetcher to believe all
        weeks were already cached even after an explicit cache clear.

        Returns:
            Dictionary with counts of cleared entries by type
        """
        with self.get_session() as session:
            # Count before clearing
            commit_count = session.query(CachedCommit).count()
            pr_count = session.query(PullRequestCache).count()
            issue_count = session.query(IssueCache).count()
            status_count = session.query(RepositoryAnalysisStatus).count()
            weekly_count = session.query(WeeklyFetchStatus).count()

            # Clear all entries
            session.query(CachedCommit).delete()
            session.query(PullRequestCache).delete()
            session.query(IssueCache).delete()
            session.query(RepositoryAnalysisStatus).delete()
            session.query(WeeklyFetchStatus).delete()

            return {
                "commits": commit_count,
                "pull_requests": pr_count,
                "issues": issue_count,
                "repository_status": status_count,
                "weekly_fetch_status": weekly_count,
                "total": commit_count + pr_count + issue_count + status_count + weekly_count,
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
            # Bug 1 fix: use timezone-aware UTC datetime instead of naive utcnow()
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.ttl_hours)
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
                # Bug 1 fix: use timezone-aware UTC datetime instead of naive utcnow()
                very_old_cutoff = datetime.now(timezone.utc) - timedelta(hours=self.ttl_hours * 2)
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

        # Bug 5 fix: use timezone-aware UTC datetime so the cutoff is unambiguous.
        # Previously datetime.now() produced a naive local-time datetime which:
        #   1. Could differ from UTC by hours depending on the host timezone.
        #   2. Was passed to git as a date-only string ("%Y-%m-%d"), discarding
        #      the time component and potentially including/excluding an extra day
        #      at the boundary.  Full ISO-8601 format with timezone is passed now.
        start_time = datetime.now(timezone.utc)
        cutoff_date = datetime.now(timezone.utc) - timedelta(weeks=weeks)

        try:
            for repo_path in repo_paths:
                try:
                    from pathlib import Path

                    repo_path_obj = Path(repo_path)
                    repo = git.Repo(repo_path)

                    # Pass full ISO-8601 datetime string (includes timezone offset)
                    # so git respects the exact cutoff moment rather than rounding
                    # to the start of the day as it would with a date-only string.
                    commits = list(repo.iter_commits(all=True, since=cutoff_date.isoformat()))

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

        # Bug 5 fix: use timezone-aware datetime to match the start_time above
        warming_results["duration_seconds"] = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()
        return warming_results

    def _analyze_commit_minimal(
        self, repo: git.Repo, commit: git.Commit, repo_path: Path
    ) -> dict[str, Any]:
        """Minimal commit analysis for cache warming.

        WHY: Cache warming doesn't need full analysis complexity,
        just enough data to populate the cache effectively.
        """
        # Bug 6 fix: normalise commit timestamp to UTC before storing.
        # commit.committed_datetime carries the author's local timezone offset
        # (e.g. +09:00 for JST).  The main analyzer normalises to UTC before
        # storage; skipping that here caused week-boundary mismatches when the
        # cache-warming path stored raw offset-aware datetimes that compared
        # differently against UTC-based query filters.
        commit_utc = commit.committed_datetime.astimezone(timezone.utc)

        # Basic commit data
        commit_data = {
            "hash": commit.hexsha,
            "author_name": commit.author.name,
            "author_email": commit.author.email,
            "message": commit.message,
            "timestamp": commit_utc,
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
        # Bug 1 fix: use timezone-aware UTC datetime instead of naive utcnow().
        # cached_at stored by DateTime(timezone=True) columns is already tz-aware.
        now_utc = datetime.now(timezone.utc)
        # Handle the case where cached_at may still be timezone-naive (legacy rows)
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)
        return cached_at < now_utc - timedelta(hours=self.ttl_hours)

    def _commit_to_dict(self, commit: CachedCommit) -> dict[str, Any]:
        """Convert CachedCommit to dictionary.

        Gap 4: Co-author trailers are parsed from the cached commit message
        so that the identity resolver can credit co-authors even when the
        commit is served from cache rather than re-analysed from git.
        """
        message = commit.message or ""
        return {
            "hash": commit.commit_hash,
            "author_name": commit.author_name,
            "author_email": commit.author_email,
            "message": message,
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
            # Gap 4: parse Co-authored-by trailers so co-author attribution
            # works for cache hits without a full re-analysis pass.
            "co_authors": _extract_co_authors_from_message(message),
        }

    def _pr_to_dict(self, pr: PullRequestCache) -> dict[str, Any]:
        """Convert PullRequestCache to dictionary.

        WHY: Uses getattr with defaults for the v3.0 enhanced fields so that
        this method works correctly against older database rows that pre-date the
        migration (columns will be None after ALTER TABLE adds them).
        """
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
            # Enhanced PR tracking fields (v3.0)
            "review_comments": getattr(pr, "review_comments_count", None) or 0,
            "pr_comments_count": getattr(pr, "pr_comments_count", None) or 0,
            "approvals_count": getattr(pr, "approvals_count", None) or 0,
            "change_requests_count": getattr(pr, "change_requests_count", None) or 0,
            "reviewers": getattr(pr, "reviewers", None) or [],
            "approved_by": getattr(pr, "approved_by", None) or [],
            "time_to_first_review_hours": getattr(pr, "time_to_first_review_hours", None),
            "revision_count": getattr(pr, "revision_count", None) or 0,
            "changed_files": getattr(pr, "changed_files", None) or 0,
            "additions": getattr(pr, "additions", None) or 0,
            "deletions": getattr(pr, "deletions", None) or 0,
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

