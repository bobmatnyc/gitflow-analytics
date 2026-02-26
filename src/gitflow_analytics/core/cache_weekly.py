"""Weekly fetch status and config hash cache mixin for GitAnalysisCache."""

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


class WeeklyCacheMixin:
    """Mixin providing weekly fetch status tracking and config hash methods.

    Mixed into GitAnalysisCache. Uses self.db from the host class.
    """


    # ---------------------------------------------------------------------------
    # Week-granularity incremental fetch tracking
    # ---------------------------------------------------------------------------

    @staticmethod
    def calculate_weeks(
        start_date: datetime, end_date: datetime
    ) -> list[tuple[datetime, datetime]]:
        """Calculate Monday-aligned ISO weeks covering the date range.

        WHY: Historical weeks never change once committed, so each Monday-to-Sunday
        week is a discrete cacheable unit. Aligning to Monday boundaries ensures
        consistent week definitions regardless of when the user runs the tool.

        Args:
            start_date: Inclusive start of the date range (timezone-aware UTC)
            end_date: Inclusive end of the date range (timezone-aware UTC)

        Returns:
            List of (week_start, week_end) tuples where week_start is always
            Monday 00:00:00 UTC and week_end is always Sunday 23:59:59 UTC.
            The last tuple's week_end is capped to end_date when end_date
            falls mid-week.
        """
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        weeks: list[tuple[datetime, datetime]] = []
        # Align start to Monday of the week containing start_date
        current = start_date - timedelta(days=start_date.weekday())
        current = current.replace(hour=0, minute=0, second=0, microsecond=0)

        while current < end_date:
            week_end = current + timedelta(days=6, hours=23, minutes=59, seconds=59)
            # Cap the last partial week to end_date so we don't claim we've fetched
            # future data that doesn't exist yet.
            capped_end = min(week_end, end_date)
            weeks.append((current, capped_end))
            current += timedelta(weeks=1)

        return weeks

    def get_cached_weeks(self, repo_path: str) -> list[tuple[datetime, datetime]]:
        """Return the (week_start, week_end) pairs already fetched for this repo.

        Args:
            repo_path: Canonical string form of the repository path.

        Returns:
            List of (week_start, week_end) tuples for weeks that have a
            WeeklyFetchStatus row.  Empty list if nothing is cached.
        """
        with self.get_session() as session:
            rows = (
                session.query(WeeklyFetchStatus.week_start, WeeklyFetchStatus.week_end)
                .filter(WeeklyFetchStatus.repository_path == repo_path)
                .all()
            )
            return [(row.week_start, row.week_end) for row in rows]

    def get_missing_weeks(
        self,
        repo_path: str,
        required_weeks: list[tuple[datetime, datetime]],
    ) -> list[tuple[datetime, datetime]]:
        """Return which of the required weeks are NOT yet cached for this repo.

        WHY: The caller passes the full list of weeks it needs; we return only
        the subset that must be fetched.  The comparison is done on week_start
        alone because week_start uniquely identifies a Monday-aligned week.

        NOTE on SQLite timezone handling: SQLite's DateTime column strips tzinfo
        when reading back values, so the queried week_start values are naive
        datetimes in UTC.  We normalise both sides to timezone-naive UTC before
        the set comparison to avoid false "missing" mismatches.

        Args:
            repo_path: Canonical string form of the repository path.
            required_weeks: List of (week_start, week_end) tuples the caller needs.

        Returns:
            Subset of required_weeks that have no WeeklyFetchStatus row yet.
        """
        if not required_weeks:
            return []

        def _to_naive_utc(dt: datetime) -> datetime:
            """Strip tzinfo after converting to UTC so comparisons are reliable."""
            if dt.tzinfo is not None:
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt

        with self.get_session() as session:
            # SQLite may return naive datetimes here; normalise defensively.
            cached_starts = set(
                _to_naive_utc(row[0])
                for row in session.query(WeeklyFetchStatus.week_start)
                .filter(WeeklyFetchStatus.repository_path == repo_path)
                .all()
            )

        missing = []
        for week_start, week_end in required_weeks:
            if _to_naive_utc(week_start) not in cached_starts:
                missing.append((week_start, week_end))

        return missing

    def mark_week_cached(
        self,
        repo_path: str,
        week_start: datetime,
        week_end: datetime,
        commit_count: int,
    ) -> None:
        """Record that a specific ISO week has been fetched for this repo.

        Uses an UPSERT pattern (delete-then-insert inside the same transaction)
        so that force-refetching a week simply overwrites the previous record.

        Args:
            repo_path: Canonical string form of the repository path.
            week_start: Monday 00:00:00 UTC of the week.
            week_end: Sunday 23:59:59 UTC of the week (may be capped to today).
            commit_count: Number of commits found during the fetch (informational).
        """
        if week_start.tzinfo is None:
            week_start = week_start.replace(tzinfo=timezone.utc)
        if week_end.tzinfo is None:
            week_end = week_end.replace(tzinfo=timezone.utc)

        with self.get_session() as session:
            # Delete any existing row for this (repo, week_start) pair so we can
            # re-insert cleanly without hitting the unique constraint.
            session.query(WeeklyFetchStatus).filter(
                and_(
                    WeeklyFetchStatus.repository_path == repo_path,
                    WeeklyFetchStatus.week_start == week_start,
                )
            ).delete(synchronize_session=False)

            session.add(
                WeeklyFetchStatus(
                    repository_path=repo_path,
                    week_start=week_start,
                    week_end=week_end,
                    commit_count=commit_count,
                    fetch_timestamp=datetime.now(timezone.utc),
                )
            )

    def clear_weekly_cache(
        self,
        repo_path: Optional[str] = None,
        week_start: Optional[datetime] = None,
    ) -> int:
        """Clear WeeklyFetchStatus rows.

        Args:
            repo_path: If provided, only clear rows for this repository.
            week_start: If provided (alongside repo_path), clear only the row
                for this specific week.  Ignored when repo_path is None.

        Returns:
            Number of rows deleted.
        """
        with self.get_session() as session:
            query = session.query(WeeklyFetchStatus)

            if repo_path is not None:
                query = query.filter(WeeklyFetchStatus.repository_path == repo_path)
                if week_start is not None:
                    ws = (
                        week_start if week_start.tzinfo else week_start.replace(tzinfo=timezone.utc)
                    )
                    query = query.filter(WeeklyFetchStatus.week_start == ws)

            deleted = query.delete(synchronize_session=False)
            return deleted

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
                        # Bug 1 fix: use timezone-aware UTC datetime instead of naive utcnow()
                        CachedCommit.cached_at
                        >= datetime.now(timezone.utc) - timedelta(hours=self.ttl_hours),
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
                # Bug 1 fix: use timezone-aware UTC datetime instead of naive utcnow()
                existing.last_updated = datetime.now(timezone.utc)
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
                # Bug 1 fix: use timezone-aware UTC datetime instead of naive utcnow()
                existing.last_updated = datetime.now(timezone.utc)
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
                # Bug 1 fix: use timezone-aware UTC datetime instead of naive utcnow()
                cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
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
            # Bug 1 fix: use timezone-aware UTC datetime instead of naive utcnow()
            recent_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
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
