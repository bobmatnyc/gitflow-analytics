"""Caching layer for Git analysis with SQLite backend."""

import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import git
from sqlalchemy import and_

from ..models.database import CachedCommit, Database, IssueCache, PullRequestCache


class GitAnalysisCache:
    """Cache for Git analysis results."""

    def __init__(self, cache_dir: Path, ttl_hours: int = 168) -> None:
        """Initialize cache with SQLite backend."""
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        self.db = Database(cache_dir / "gitflow_cache.db")
        
        # Cache performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_start_time = datetime.now()
        
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

    def get_cached_commits_bulk(self, repo_path: str, commit_hashes: list[str]) -> dict[str, dict[str, Any]]:
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
                        CachedCommit.commit_hash.in_(commit_hashes)
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
            print(f"DEBUG: Bulk cache lookup - {hits} hits, {misses} misses for {len(commit_hashes)} commits")

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
                    files_changed=commit_data.get("files_changed_count", commit_data.get("files_changed", 0) if isinstance(commit_data.get("files_changed"), int) else len(commit_data.get("files_changed", []))),
                    insertions=commit_data.get("insertions", 0),
                    deletions=commit_data.get("deletions", 0),
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
                        CachedCommit.commit_hash.in_(commit_hashes)
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
                        files_changed=commit_data.get("files_changed_count", commit_data.get("files_changed", 0) if isinstance(commit_data.get("files_changed"), int) else len(commit_data.get("files_changed", []))),
                        insertions=commit_data.get("insertions", 0),
                        deletions=commit_data.get("deletions", 0),
                        complexity_delta=commit_data.get("complexity_delta", 0.0),
                        story_points=commit_data.get("story_points"),
                        ticket_references=commit_data.get("ticket_references", []),
                    )
                    session.add(cached_commit)

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

    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics including external API cache performance."""
        with self.get_session() as session:
            # Basic counts
            total_commits = session.query(CachedCommit).count()
            total_prs = session.query(PullRequestCache).count()
            total_issues = session.query(IssueCache).count()
            
            # Platform-specific issue counts
            jira_issues = session.query(IssueCache).filter(IssueCache.platform == "jira").count()
            github_issues = session.query(IssueCache).filter(IssueCache.platform == "github").count()
            
            # Stale entries
            cutoff_time = datetime.utcnow() - timedelta(hours=self.ttl_hours)
            stale_commits = session.query(CachedCommit).filter(
                CachedCommit.cached_at < cutoff_time
            ).count()
            stale_prs = session.query(PullRequestCache).filter(
                PullRequestCache.cached_at < cutoff_time
            ).count()
            stale_issues = session.query(IssueCache).filter(
                IssueCache.cached_at < cutoff_time
            ).count()
            
            # Performance metrics
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
            
            # Estimated time savings (conservative estimates)
            commit_time_saved = self.cache_hits * 0.1  # 0.1 seconds per commit analysis
            api_time_saved = (total_issues * 0.5) + (total_prs * 0.3)  # API call time savings
            total_time_saved = commit_time_saved + api_time_saved
            
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
                "freshness_rate_percent": (total_fresh_entries / max(1, total_commits + total_prs + total_issues)) * 100,
                
                # Performance metrics
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "total_requests": total_requests,
                "hit_rate_percent": hit_rate,
                
                # Time savings
                "commit_analysis_time_saved_seconds": commit_time_saved,
                "api_call_time_saved_seconds": api_time_saved,
                "total_time_saved_seconds": total_time_saved,
                "total_time_saved_minutes": total_time_saved / 60,
                "estimated_api_calls_avoided": total_issues + total_prs,
                
                # Storage metrics
                "database_size_mb": db_size_mb,
                "session_duration_seconds": session_duration,
                "avg_entries_per_mb": (total_commits + total_prs + total_issues) / max(0.1, db_size_mb),
                
                # Configuration
                "ttl_hours": self.ttl_hours,
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
        print(f"📦 Cache Contents:")
        print(f"   • Commits: {stats['cached_commits']:,} ({stats['fresh_commits']:,} fresh, {stats['stale_commits']:,} stale)")
        print(f"   • Pull Requests: {stats['cached_prs']:,} ({stats['fresh_prs']:,} fresh, {stats['stale_prs']:,} stale)")
        print(f"   • Issues: {stats['cached_issues']:,} ({stats['fresh_issues']:,} fresh, {stats['stale_issues']:,} stale)")
        
        if stats['cached_jira_issues'] > 0:
            print(f"     ├─ JIRA: {stats['cached_jira_issues']:,} issues")
        if stats['cached_github_issues'] > 0:
            print(f"     └─ GitHub: {stats['cached_github_issues']:,} issues")
        
        # Performance metrics
        if stats['total_requests'] > 0:
            print(f"\n⚡ Session Performance:")
            print(f"   • Cache Hit Rate: {stats['hit_rate_percent']:.1f}% ({stats['cache_hits']:,}/{stats['total_requests']:,})")
            
            if stats['total_time_saved_minutes'] > 1:
                print(f"   • Time Saved: {stats['total_time_saved_minutes']:.1f} minutes")
            else:
                print(f"   • Time Saved: {stats['total_time_saved_seconds']:.1f} seconds")
            
            if stats['estimated_api_calls_avoided'] > 0:
                print(f"   • API Calls Avoided: {stats['estimated_api_calls_avoided']:,}")
        
        # Storage info
        print(f"\n💾 Storage:")
        print(f"   • Database Size: {stats['database_size_mb']:.1f} MB")
        print(f"   • Cache TTL: {stats['ttl_hours']} hours")
        print(f"   • Overall Freshness: {stats['freshness_rate_percent']:.1f}%")
        
        # Performance insights
        if stats['hit_rate_percent'] > 80:
            print(f"   ✅ Excellent cache performance!")
        elif stats['hit_rate_percent'] > 50:
            print(f"   👍 Good cache performance")
        elif stats['total_requests'] > 0:
            print(f"   ⚠️  Consider clearing stale cache entries")
            
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
                commits_without_hash = session.query(CachedCommit).filter(
                    CachedCommit.commit_hash.is_(None)
                ).count()
                
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
                        func.count().label('count')
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
                very_old_count = session.query(CachedCommit).filter(
                    CachedCommit.cached_at < very_old_cutoff
                ).count()
                
                if very_old_count > 0:
                    validation_results["warnings"].append(
                        f"Found {very_old_count} very old cache entries (older than {self.ttl_hours * 2}h)"
                    )
                
                # Basic integrity checks
                commits_with_negative_changes = session.query(CachedCommit).filter(
                    (CachedCommit.files_changed < 0) |
                    (CachedCommit.insertions < 0) |
                    (CachedCommit.deletions < 0)
                ).count()
                
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
        from tqdm import tqdm
        
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
                    commits = list(repo.iter_commits(
                        all=True,
                        since=cutoff_date.strftime('%Y-%m-%d')
                    ))
                    
                    warming_results["total_commits_found"] += len(commits)
                    
                    # Check which commits are already cached
                    commit_hashes = [c.hexsha for c in commits]
                    cached_commits = self.get_cached_commits_bulk(str(repo_path_obj), commit_hashes)
                    already_cached = len(cached_commits)
                    to_analyze = len(commits) - already_cached
                    
                    warming_results["commits_already_cached"] += already_cached
                    
                    if to_analyze > 0:
                        # Analyze uncached commits with progress bar
                        with tqdm(
                            total=to_analyze,
                            desc=f"Warming cache for {repo_path_obj.name}",
                            leave=False
                        ) as pbar:
                            
                            new_commits = []
                            for commit in commits:
                                if commit.hexsha not in cached_commits:
                                    # Basic commit analysis (minimal for cache warming)
                                    commit_data = self._analyze_commit_minimal(repo, commit, repo_path_obj)
                                    new_commits.append(commit_data)
                                    pbar.update(1)
                                    
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

    def _analyze_commit_minimal(self, repo: git.Repo, commit: git.Commit, repo_path: Path) -> dict[str, Any]:
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
            "files_changed": len(commit.stats.files),
            "insertions": commit.stats.total.get('insertions', 0),
            "deletions": commit.stats.total.get('deletions', 0),
            "complexity_delta": 0.0,  # Skip complexity calculation for warming
            "story_points": None,  # Skip story point extraction for warming
            "ticket_references": [],  # Skip ticket analysis for warming
        }
        
        # Try to get branch info (if available)
        try:
            branches = repo.git.branch('--contains', commit.hexsha).split('\n')
            commit_data["branch"] = branches[0].strip('* ') if branches else "unknown"
        except:
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
