"""Daily metrics storage system for GitFlow Analytics.

WHY: This module handles the storage and retrieval of daily classified activity
metrics for developers and projects. It provides the foundation for database-backed
reporting with trend analysis capabilities.
"""

import logging
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

from sqlalchemy import and_, func  # type: ignore[import-not-found]
from sqlalchemy.orm import Session  # type: ignore[import-not-found]

from ..models.database import DailyMetrics, Database
from ..utils.ai_detection import detect_ai_tool, is_ai_generated

logger = logging.getLogger(__name__)


class _DayStats(TypedDict):
    """Typed accumulator for one developer+project+day bucket.

    WHY: Using TypedDict instead of a plain dict lets Pyright resolve each
    field to its concrete type, eliminating the broad ``int | set | str``
    union that otherwise blocks all arithmetic and set operations.
    """

    total_commits: int
    feature_commits: int
    bug_fix_commits: int
    refactor_commits: int
    documentation_commits: int
    maintenance_commits: int
    test_commits: int
    style_commits: int
    build_commits: int
    other_commits: int
    files_changed: int
    lines_added: int
    lines_deleted: int
    # WHY float (issue #56): aggregated story points may be fractional when
    # underlying issues use modified Fibonacci scales (e.g., 0.5, 1.5, 3.5).
    story_points: float
    tracked_commits: int
    untracked_commits: int
    unique_tickets: set[str]
    merge_commits: int
    complex_commits: int
    # Issue #64: count of commits flagged as revert/rollback for this day-bucket.
    reversion_commits: int
    developer_name: str
    developer_email: str


def _make_day_stats() -> _DayStats:
    """Return a zeroed _DayStats bucket for use as defaultdict factory."""
    return _DayStats(
        total_commits=0,
        feature_commits=0,
        bug_fix_commits=0,
        refactor_commits=0,
        documentation_commits=0,
        maintenance_commits=0,
        test_commits=0,
        style_commits=0,
        build_commits=0,
        other_commits=0,
        files_changed=0,
        lines_added=0,
        lines_deleted=0,
        story_points=0.0,
        tracked_commits=0,
        untracked_commits=0,
        unique_tickets=set(),
        merge_commits=0,
        complex_commits=0,
        reversion_commits=0,
        developer_name="",
        developer_email="",
    )


class DailyMetricsStorage:
    """Storage manager for daily developer/project activity metrics.

    WHY: Centralized storage management ensures consistent data aggregation
    and enables efficient querying for reports and trend analysis.
    """

    def __init__(self, db_path: Path):
        """Initialize daily metrics storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db = Database(db_path)
        logger.info(f"Initialized daily metrics storage at {db_path}")

    @contextmanager
    def get_session(self):
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

    def store_daily_metrics(
        self,
        analysis_date: date,
        commits: list[dict[str, Any]],
        developer_identities: dict[str, dict[str, str]],
    ) -> int:
        """Store daily metrics from commit analysis.

        WHY: Aggregates and stores daily metrics per developer-project combination
        to enable fast report generation and trend analysis.

        Args:
            analysis_date: Date for the metrics
            commits: List of analyzed commits with categorization
            developer_identities: Mapping of email to canonical developer info

        Returns:
            Number of daily metric records created/updated
        """
        daily_aggregates = self._aggregate_commits_by_day(
            commits, developer_identities, analysis_date
        )

        records_processed = 0
        session = self.db.get_session()
        try:
            for (dev_id, project_key), metrics in daily_aggregates.items():
                try:
                    # Get or create daily metrics record
                    existing = (
                        session.query(DailyMetrics)
                        .filter(
                            and_(
                                DailyMetrics.date == analysis_date,
                                DailyMetrics.developer_id == dev_id,
                                DailyMetrics.project_key == project_key,
                            )
                        )
                        .first()
                    )

                    if existing:
                        # Update existing record
                        self._update_metrics_record(existing, metrics)
                        existing.updated_at = datetime.now(timezone.utc).replace(  # type: ignore[assignment]
                            tzinfo=None
                        )
                        logger.debug(
                            f"Updated existing daily metrics for {dev_id} in {project_key} on {analysis_date}"
                        )
                    else:
                        # Create new record
                        new_record = DailyMetrics(
                            date=analysis_date,
                            developer_id=dev_id,
                            project_key=project_key,
                            developer_name=metrics["developer_name"],
                            developer_email=metrics["developer_email"],
                            **{
                                k: v
                                for k, v in metrics.items()
                                if k not in ["developer_name", "developer_email"]
                            },
                        )
                        session.add(new_record)
                        logger.debug(
                            f"Created new daily metrics for {dev_id} in {project_key} on {analysis_date}"
                        )

                    # Commit this record individually to avoid constraint violations
                    session.commit()
                    records_processed += 1

                except Exception as e:
                    is_unique_violation = "UNIQUE constraint failed" in str(e)
                    if is_unique_violation:
                        logger.debug(
                            f"UNIQUE constraint conflict for {dev_id} in {project_key} on {analysis_date}: {e}"
                        )
                    else:
                        logger.warning(
                            f"Failed to store/update daily metrics for {dev_id} in {project_key} on {analysis_date}: {e}"
                        )
                    session.rollback()
                    # Try to handle UNIQUE constraint violations by doing another lookup
                    try:
                        existing = (
                            session.query(DailyMetrics)
                            .filter(
                                and_(
                                    DailyMetrics.date == analysis_date,
                                    DailyMetrics.developer_id == dev_id,
                                    DailyMetrics.project_key == project_key,
                                )
                            )
                            .first()
                        )
                        if existing:
                            # Record was created by another process, just update it
                            self._update_metrics_record(existing, metrics)
                            existing.updated_at = datetime.now(timezone.utc).replace(  # type: ignore[assignment]
                                tzinfo=None
                            )
                            session.commit()
                            records_processed += 1
                            logger.info(
                                f"Updated metrics after constraint violation for {dev_id} in {project_key} on {analysis_date}"
                            )
                        else:
                            # UNIQUE violation but record gone — expected during re-classification
                            logger.debug(
                                f"Could not resolve constraint violation for {dev_id} in {project_key} on {analysis_date}"
                            )
                    except Exception as retry_e:
                        logger.warning(
                            f"Retry failed for {dev_id} in {project_key} on {analysis_date}: {retry_e}"
                        )
                        session.rollback()
        finally:
            session.close()

        logger.info(f"Stored/updated {records_processed} daily metrics records for {analysis_date}")
        return records_processed

    def get_date_range_metrics(
        self,
        start_date: date,
        end_date: date,
        developer_ids: Optional[list[str]] = None,
        project_keys: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve daily metrics for a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            developer_ids: Optional filter by developer IDs
            project_keys: Optional filter by project keys

        Returns:
            List of daily metrics dictionaries
        """
        with self.get_session() as session:
            query = session.query(DailyMetrics).filter(
                and_(DailyMetrics.date >= start_date, DailyMetrics.date <= end_date)
            )

            if developer_ids:
                query = query.filter(DailyMetrics.developer_id.in_(developer_ids))

            if project_keys:
                query = query.filter(DailyMetrics.project_key.in_(project_keys))

            results = query.order_by(DailyMetrics.date, DailyMetrics.developer_id).all()

            return [self._metrics_record_to_dict(record) for record in results]

    def calculate_weekly_trends(
        self, start_date: date, end_date: date
    ) -> dict[tuple[str, str], dict[str, float]]:
        """Calculate week-over-week trends for developer-project combinations.

        WHY: Pre-calculated trends improve report performance and provide
        consistent trend analysis across different report types.

        Args:
            start_date: Analysis start date
            end_date: Analysis end date

        Returns:
            Dict mapping (developer_id, project_key) to trend metrics
        """
        trends = {}

        with self.get_session() as session:
            # Get all unique developer-project combinations in the date range
            combinations = (
                session.query(DailyMetrics.developer_id, DailyMetrics.project_key)
                .filter(and_(DailyMetrics.date >= start_date, DailyMetrics.date <= end_date))
                .distinct()
                .all()
            )

            for dev_id, project_key in combinations:
                trend_data = self._calculate_developer_project_trend(
                    session, dev_id, project_key, start_date, end_date
                )

                if trend_data:
                    trends[(dev_id, project_key)] = trend_data

        logger.info(f"Calculated trends for {len(trends)} developer-project combinations")
        return trends

    def get_classification_summary(
        self, start_date: date, end_date: date
    ) -> dict[str, dict[str, dict[str, int]]]:
        """Get classification summary across all developers and projects.

        Args:
            start_date: Start date for summary
            end_date: End date for summary

        Returns:
            Dict with classification counts by developer and project
        """
        with self.get_session() as session:
            # Classification totals by developer
            dev_query = (
                session.query(
                    DailyMetrics.developer_name,
                    func.sum(DailyMetrics.feature_commits).label("features"),
                    func.sum(DailyMetrics.bug_fix_commits).label("bug_fixes"),
                    func.sum(DailyMetrics.refactor_commits).label("refactors"),
                    func.sum(DailyMetrics.total_commits).label("total"),
                )
                .filter(and_(DailyMetrics.date >= start_date, DailyMetrics.date <= end_date))
                .group_by(DailyMetrics.developer_name)
                .all()
            )

            # Classification totals by project
            proj_query = (
                session.query(
                    DailyMetrics.project_key,
                    func.sum(DailyMetrics.feature_commits).label("features"),
                    func.sum(DailyMetrics.bug_fix_commits).label("bug_fixes"),
                    func.sum(DailyMetrics.refactor_commits).label("refactors"),
                    func.sum(DailyMetrics.total_commits).label("total"),
                )
                .filter(and_(DailyMetrics.date >= start_date, DailyMetrics.date <= end_date))
                .group_by(DailyMetrics.project_key)
                .all()
            )

            return {
                "by_developer": {
                    row.developer_name: {
                        "features": row.features or 0,
                        "bug_fixes": row.bug_fixes or 0,
                        "refactors": row.refactors or 0,
                        "total": row.total or 0,
                    }
                    for row in dev_query
                },
                "by_project": {
                    row.project_key: {
                        "features": row.features or 0,
                        "bug_fixes": row.bug_fixes or 0,
                        "refactors": row.refactors or 0,
                        "total": row.total or 0,
                    }
                    for row in proj_query
                },
            }

    def _aggregate_commits_by_day(
        self,
        commits: list[dict[str, Any]],
        developer_identities: dict[str, dict[str, str]],
        target_date: date,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Aggregate commits into daily metrics by developer-project.

        WHY: Groups commits by developer and project for the target date,
        calculating all relevant metrics for storage.
        """
        # Group commits by developer and project for the target date.
        # _make_day_stats() is used as the factory so Pyright can infer the
        # concrete _DayStats type for each bucket instead of a broad union.
        daily_groups: defaultdict[tuple[str, str], _DayStats] = defaultdict(_make_day_stats)

        # Separate plain dict for AI tracking — avoids TypedDict set/int union
        # complexity that Pyright cannot resolve cleanly inside _DayStats.
        ai_data: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {
                "ai_assisted_commits": 0,
                "ai_generated_commits": 0,
                "ai_assisted_lines": 0,
                "ai_generated_lines": 0,
                "_ai_tools_used": set(),
            }
        )

        for commit in commits:
            # Filter to target date
            commit_date = commit.get("timestamp")
            if not commit_date:
                continue

            # Handle both datetime and date objects
            if isinstance(commit_date, datetime) or hasattr(commit_date, "date"):
                if commit_date.date() != target_date:
                    continue
            elif str(commit_date)[:10] != str(target_date):
                continue

            # Get developer identity
            author_email = commit.get("author_email", "")
            dev_identity = developer_identities.get(
                author_email,
                {
                    "canonical_id": author_email,
                    "name": commit.get("author_name", "Unknown"),
                    "email": author_email,
                },
            )

            dev_id: str = str(dev_identity.get("canonical_id", author_email) or author_email)
            project_key: str = str(commit.get("project_key", "default") or "default")

            metrics = daily_groups[(dev_id, project_key)]

            # Set developer info (will be overwritten with same values, that's OK)
            metrics["developer_name"] = str(
                dev_identity.get("name", commit.get("author_name", "Unknown")) or "Unknown"
            )
            metrics["developer_email"] = str(
                dev_identity.get("email", author_email) or author_email
            )

            # Aggregate basic metrics
            metrics["total_commits"] += 1
            # Handle files_changed safely - could be int or list
            files_changed = commit.get("files_changed", 0)
            if isinstance(files_changed, list):
                metrics["files_changed"] += len(files_changed)
            elif isinstance(files_changed, int):
                metrics["files_changed"] += files_changed
            else:
                # Fallback for unexpected types
                metrics["files_changed"] += 0
            # Use filtered values if available, fallback to raw values
            metrics["lines_added"] += commit.get("filtered_insertions", commit.get("insertions", 0))
            metrics["lines_deleted"] += commit.get("filtered_deletions", commit.get("deletions", 0))
            metrics["story_points"] += commit.get("story_points", 0) or 0

            # Classification counts
            category = commit.get("category", "other")
            category_field = f"{category}_commits"
            if category_field in metrics:
                metrics[category_field] += 1
            else:
                metrics["other_commits"] += 1

            # Ticket tracking
            ticket_refs = commit.get("ticket_references", [])
            if ticket_refs:
                metrics["tracked_commits"] += 1
                # Extract ticket IDs from ticket reference objects
                # ticket_refs can be either [{"id": "PROJ-123", "platform": "jira"}] or ["PROJ-123"]
                ticket_ids = []
                for ref in ticket_refs:
                    if isinstance(ref, dict):
                        ticket_ids.append(ref.get("id", str(ref)))
                    else:
                        ticket_ids.append(str(ref))
                metrics["unique_tickets"].update(ticket_ids)
            else:
                metrics["untracked_commits"] += 1

            # AI tool usage detection — accumulated in ai_data (plain dict) to
            # avoid TypedDict set/int union issues that Pyright cannot resolve.
            msg = str(commit.get("message", "") or "")
            tool = detect_ai_tool(msg)
            if tool is not None:
                ai_bucket = ai_data[(dev_id, project_key)]
                ai_bucket["ai_assisted_commits"] += 1
                cast(set[str], ai_bucket["_ai_tools_used"]).add(tool)
                lines_val = int(
                    commit.get("filtered_insertions", 0) or commit.get("insertions", 0) or 0
                )
                ai_bucket["ai_assisted_lines"] += lines_val
                if is_ai_generated(msg):
                    ai_bucket["ai_generated_commits"] += 1
                    ai_bucket["ai_generated_lines"] += lines_val

            # Work patterns
            if commit.get("is_merge", False):
                metrics["merge_commits"] += 1

            if commit.get("files_changed", 0) > 5:
                metrics["complex_commits"] += 1

            # Issue #64: aggregate revert count for daily_metrics.reversion_commits.
            # Prefer the persisted is_revert flag from cached_commits; fall back to
            # re-scanning the message so commits coming from non-cache sources
            # (e.g., direct analyzer output before being cached) still count.
            is_revert_flag = commit.get("is_revert")
            if is_revert_flag is None:
                from ..utils.revert_detection import is_revert_commit as _is_revert

                is_revert_flag = _is_revert(commit.get("message"))
            if is_revert_flag:
                metrics["reversion_commits"] += 1

        # Convert sets to scalar values and merge AI accumulator data.
        # We cast to dict[str, Any] here because we're intentionally replacing
        # set fields with their int/str summaries, which TypedDict can't express.
        result: dict[tuple[str, str], dict[str, Any]] = {}
        for key, day in daily_groups.items():
            out: dict[str, Any] = dict(day)
            out["unique_tickets"] = len(day["unique_tickets"])
            # Merge AI metrics from the separate accumulator
            if key in ai_data:
                ai = ai_data[key]
                tools_set: set[str] = cast(set[str], ai["_ai_tools_used"])
                if len(tools_set) == 0:
                    out["ai_tool_primary"] = ""
                elif len(tools_set) == 1:
                    out["ai_tool_primary"] = next(iter(tools_set))
                else:
                    out["ai_tool_primary"] = "mixed"
                out["ai_assisted_commits"] = ai["ai_assisted_commits"]
                out["ai_generated_commits"] = ai["ai_generated_commits"]
                out["ai_assisted_lines"] = ai["ai_assisted_lines"]
                out["ai_generated_lines"] = ai["ai_generated_lines"]
            else:
                out["ai_tool_primary"] = ""
                out["ai_assisted_commits"] = 0
                out["ai_generated_commits"] = 0
                out["ai_assisted_lines"] = 0
                out["ai_generated_lines"] = 0
            result[key] = out

        return result

    def _update_metrics_record(self, record: DailyMetrics, metrics: dict[str, Any]) -> None:
        """Update an existing DailyMetrics record with new data."""
        for key, value in metrics.items():
            if hasattr(record, key) and key not in ["developer_name", "developer_email"]:
                setattr(record, key, value)

    def _metrics_record_to_dict(self, record: DailyMetrics) -> dict[str, Any]:
        """Convert DailyMetrics SQLAlchemy record to dictionary."""
        return {
            "date": record.date,
            "developer_id": record.developer_id,
            "developer_name": record.developer_name,
            "developer_email": record.developer_email,
            "project_key": record.project_key,
            "feature_commits": record.feature_commits,
            "bug_fix_commits": record.bug_fix_commits,
            "refactor_commits": record.refactor_commits,
            "documentation_commits": record.documentation_commits,
            "maintenance_commits": record.maintenance_commits,
            "test_commits": record.test_commits,
            "style_commits": record.style_commits,
            "build_commits": record.build_commits,
            "other_commits": record.other_commits,
            "total_commits": record.total_commits,
            "files_changed": record.files_changed,
            "lines_added": record.lines_added,
            "lines_deleted": record.lines_deleted,
            "story_points": record.story_points,
            "tracked_commits": record.tracked_commits,
            "untracked_commits": record.untracked_commits,
            "unique_tickets": record.unique_tickets,
            "merge_commits": record.merge_commits,
            "complex_commits": record.complex_commits,
            # Issue #64: surface reversion_commits from daily_metrics so reports
            # can render revert trends alongside other classification counts.
            "reversion_commits": getattr(record, "reversion_commits", 0) or 0,
            "ai_assisted_commits": record.ai_assisted_commits,
            "ai_generated_commits": record.ai_generated_commits,
            "ai_tool_primary": record.ai_tool_primary,
            "ai_assisted_lines": record.ai_assisted_lines,
            "ai_generated_lines": record.ai_generated_lines,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    def _calculate_developer_project_trend(
        self, session: Session, dev_id: str, project_key: str, start_date: date, end_date: date
    ) -> Optional[dict[str, float]]:
        """Calculate trend data for a specific developer-project combination."""
        # Get weekly aggregates
        weekly_data = self._get_weekly_aggregates(
            session, dev_id, project_key, start_date, end_date
        )

        if len(weekly_data) < 2:
            # Need at least 2 weeks for trend calculation
            return None

        # Calculate week-over-week changes for the most recent week
        current_week = weekly_data[-1]
        previous_week = weekly_data[-2]

        def calculate_change(current: int, previous: int) -> float:
            """Calculate percentage change."""
            if previous == 0:
                return 100.0 if current > 0 else 0.0
            return ((current - previous) / previous) * 100.0

        return {
            "total_commits_change": calculate_change(
                current_week["total_commits"], previous_week["total_commits"]
            ),
            "feature_commits_change": calculate_change(
                current_week["feature_commits"], previous_week["feature_commits"]
            ),
            "bug_fix_commits_change": calculate_change(
                current_week["bug_fix_commits"], previous_week["bug_fix_commits"]
            ),
            "refactor_commits_change": calculate_change(
                current_week["refactor_commits"], previous_week["refactor_commits"]
            ),
            "current_week_total": current_week["total_commits"],
            "previous_week_total": previous_week["total_commits"],
        }

    def _get_weekly_aggregates(
        self, session: Session, dev_id: str, project_key: str, start_date: date, end_date: date
    ) -> list[dict[str, Any]]:
        """Get weekly aggregated data for trend calculation."""
        # Query daily metrics and group by week
        results = (
            session.query(
                func.strftime("%Y-%W", DailyMetrics.date).label("week"),
                func.sum(DailyMetrics.total_commits).label("total_commits"),
                func.sum(DailyMetrics.feature_commits).label("feature_commits"),
                func.sum(DailyMetrics.bug_fix_commits).label("bug_fix_commits"),
                func.sum(DailyMetrics.refactor_commits).label("refactor_commits"),
            )
            .filter(
                and_(
                    DailyMetrics.developer_id == dev_id,
                    DailyMetrics.project_key == project_key,
                    DailyMetrics.date >= start_date,
                    DailyMetrics.date <= end_date,
                )
            )
            .group_by("week")
            .order_by("week")
            .all()
        )

        return [
            {
                "week": result.week,
                "total_commits": result.total_commits or 0,
                "feature_commits": result.feature_commits or 0,
                "bug_fix_commits": result.bug_fix_commits or 0,
                "refactor_commits": result.refactor_commits or 0,
            }
            for result in results
        ]
