"""Caching layer for Git analysis with SQLite backend."""

from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import and_

from ..models.database import CachedCommit, Database, IssueCache, PullRequestCache


class GitAnalysisCache:
    """Cache for Git analysis results."""

    def __init__(self, cache_dir: Path, ttl_hours: int = 168) -> None:
        """Initialize cache with SQLite backend."""
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        self.db = Database(cache_dir / "gitflow_cache.db")

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
                return self._commit_to_dict(cached)

            return None

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
                    files_changed=commit_data.get("files_changed", 0),
                    insertions=commit_data.get("insertions", 0),
                    deletions=commit_data.get("deletions", 0),
                    complexity_delta=commit_data.get("complexity_delta", 0.0),
                    story_points=commit_data.get("story_points"),
                    ticket_references=commit_data.get("ticket_references", []),
                )
                session.add(cached_commit)

    def cache_commits_batch(self, repo_path: str, commits: list[dict[str, Any]]) -> None:
        """Cache multiple commits in a single transaction."""
        with self.get_session() as session:
            for commit_data in commits:
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
                        files_changed=commit_data.get("files_changed", 0),
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

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        with self.get_session() as session:
            stats = {
                "cached_commits": session.query(CachedCommit).count(),
                "cached_prs": session.query(PullRequestCache).count(),
                "cached_issues": session.query(IssueCache).count(),
                "stale_commits": session.query(CachedCommit)
                .filter(
                    CachedCommit.cached_at < datetime.utcnow() - timedelta(hours=self.ttl_hours)
                )
                .count(),
            }
            return stats

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
