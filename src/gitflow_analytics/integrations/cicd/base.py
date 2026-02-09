"""Base class for CI/CD platform integrations."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from ...core.cache import GitAnalysisCache
from ...core.schema_version import create_schema_manager


class BaseCICDIntegration(ABC):
    """Base class for CI/CD platform integrations.

    Provides common functionality for caching, incremental fetching,
    and metrics calculation across different CI/CD platforms.
    """

    def __init__(self, cache: GitAnalysisCache, **kwargs: Any):
        """Initialize CI/CD integration.

        Args:
            cache: GitAnalysisCache instance for caching pipeline data
            **kwargs: Additional platform-specific configuration
        """
        self.cache = cache
        self.schema_manager = create_schema_manager(cache.cache_dir)
        self.platform_name = "unknown"  # Override in subclass

    @abstractmethod
    def fetch_pipelines(self, repo_name: str, since: datetime) -> list[dict[str, Any]]:
        """Fetch pipeline runs from the platform.

        Args:
            repo_name: Repository name (e.g., "owner/repo")
            since: Fetch pipelines created after this date

        Returns:
            List of pipeline data dictionaries
        """
        pass

    @abstractmethod
    def enrich_commits_with_pipelines(
        self, commits: list[dict[str, Any]], pipelines: list[dict[str, Any]]
    ) -> None:
        """Enrich commits with pipeline status.

        Args:
            commits: List of commit dictionaries to enrich (modified in-place)
            pipelines: List of pipeline data dictionaries
        """
        pass

    def calculate_metrics(self, pipelines: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate CI/CD metrics from pipelines.

        Args:
            pipelines: List of pipeline data dictionaries

        Returns:
            Dictionary of calculated metrics
        """
        if not pipelines:
            return {
                "total_pipelines": 0,
                "successful_pipelines": 0,
                "failed_pipelines": 0,
                "success_rate": 0.0,
                "avg_duration_seconds": 0.0,
                "avg_duration_minutes": 0.0,
                "platform": self.platform_name,
            }

        successful = [p for p in pipelines if p.get("status") == "success"]
        failed = [p for p in pipelines if p.get("status") == "failure"]
        durations = [p["duration_seconds"] for p in pipelines if p.get("duration_seconds", 0) > 0]

        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "total_pipelines": len(pipelines),
            "successful_pipelines": len(successful),
            "failed_pipelines": len(failed),
            "success_rate": len(successful) / len(pipelines) * 100 if pipelines else 0,
            "avg_duration_seconds": avg_duration,
            "avg_duration_minutes": avg_duration / 60,
            "platform": self.platform_name,
        }

    def _get_cached_pipelines_bulk(self, repo_name: str, since: datetime) -> list[dict[str, Any]]:
        """Get cached pipelines for a repository from the given date onwards.

        WHY: Bulk pipeline cache lookups avoid redundant CI/CD API calls and
        significantly improve performance on repeated analysis runs.

        Args:
            repo_name: Repository name (e.g., "owner/repo")
            since: Only return pipelines created after this date

        Returns:
            List of cached pipeline data dictionaries
        """
        cached_pipelines = []
        with self.cache.get_session() as session:
            from ...models.database import CICDPipelineCache

            # Ensure since is timezone-aware for comparison
            if since.tzinfo is None:
                since = since.replace(tzinfo=timezone.utc)

            cached_results = (
                session.query(CICDPipelineCache)
                .filter(
                    CICDPipelineCache.repo_path == repo_name,
                    CICDPipelineCache.platform == self.platform_name,
                    CICDPipelineCache.created_at >= since,
                )
                .all()
            )

            for cached_pipeline in cached_results:
                if not self._is_pipeline_stale(cached_pipeline.cached_at):
                    pipeline_data = {
                        "pipeline_id": cached_pipeline.pipeline_id,
                        "workflow_name": cached_pipeline.workflow_name or "",
                        "status": cached_pipeline.status or "unknown",
                        "duration_seconds": cached_pipeline.duration_seconds or 0,
                        "trigger_type": cached_pipeline.trigger_type or "",
                        "commit_sha": cached_pipeline.commit_sha or "",
                        "branch": cached_pipeline.branch or "",
                        "created_at": cached_pipeline.created_at,
                        "author": cached_pipeline.author or "",
                        "url": cached_pipeline.url or "",
                        "platform": cached_pipeline.platform,
                        "platform_data": cached_pipeline.platform_data or {},
                    }
                    cached_pipelines.append(pipeline_data)

        return cached_pipelines

    def _cache_pipelines_bulk(self, repo_name: str, pipelines: list[dict[str, Any]]) -> None:
        """Cache multiple pipelines in bulk for better performance.

        WHY: Bulk caching is more efficient than individual cache operations,
        reducing database overhead when caching many pipelines from CI/CD API.

        Args:
            repo_name: Repository name
            pipelines: List of pipeline data dictionaries to cache
        """
        if not pipelines:
            return

        with self.cache.get_session() as session:
            from ...models.database import CICDPipelineCache

            for pipeline_data in pipelines:
                # Check if pipeline already exists
                existing = (
                    session.query(CICDPipelineCache)
                    .filter(
                        CICDPipelineCache.platform == self.platform_name,
                        CICDPipelineCache.pipeline_id == pipeline_data["pipeline_id"],
                    )
                    .first()
                )

                if existing:
                    # Update existing pipeline
                    existing.status = pipeline_data.get("status")
                    existing.duration_seconds = pipeline_data.get("duration_seconds")
                    existing.cached_at = datetime.now(timezone.utc)
                else:
                    # Create new pipeline cache entry
                    created_at = pipeline_data.get("created_at")
                    if created_at and created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)

                    cache_entry = CICDPipelineCache(
                        platform=self.platform_name,
                        pipeline_id=pipeline_data["pipeline_id"],
                        workflow_name=pipeline_data.get("workflow_name"),
                        repo_path=repo_name,
                        branch=pipeline_data.get("branch"),
                        commit_sha=pipeline_data.get("commit_sha"),
                        status=pipeline_data.get("status"),
                        duration_seconds=pipeline_data.get("duration_seconds"),
                        trigger_type=pipeline_data.get("trigger_type"),
                        created_at=created_at,
                        author=pipeline_data.get("author"),
                        url=pipeline_data.get("url"),
                        platform_data=pipeline_data.get("platform_data"),
                    )
                    session.add(cache_entry)

            session.commit()

    def _is_pipeline_stale(self, cached_at: datetime) -> bool:
        """Check if cached pipeline data is stale based on cache TTL.

        Args:
            cached_at: When the pipeline was cached

        Returns:
            True if stale and should be refreshed, False if still fresh
        """
        from datetime import timedelta

        if self.cache.ttl_hours == 0:  # No expiration
            return False

        stale_threshold = datetime.now(timezone.utc) - timedelta(hours=self.cache.ttl_hours)
        return cached_at < stale_threshold
