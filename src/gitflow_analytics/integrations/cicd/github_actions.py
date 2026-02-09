"""GitHub Actions CI/CD integration."""

import time
from datetime import datetime, timezone
from typing import Any

from github import Github
from github.GithubException import RateLimitExceededException, UnknownObjectException

from .base import BaseCICDIntegration


class GitHubActionsIntegration(BaseCICDIntegration):
    """Integrate with GitHub Actions for CI/CD data."""

    def __init__(
        self,
        token: str,
        cache: Any,
        rate_limit_retries: int = 3,
        backoff_factor: int = 2,
        **kwargs: Any,
    ):
        """Initialize GitHub Actions integration.

        Args:
            token: GitHub personal access token
            cache: GitAnalysisCache instance
            rate_limit_retries: Number of retry attempts on rate limit
            backoff_factor: Exponential backoff factor
            **kwargs: Additional configuration
        """
        super().__init__(cache, **kwargs)
        self.github = Github(token)
        self.rate_limit_retries = rate_limit_retries
        self.backoff_factor = backoff_factor
        self.platform_name = "github_actions"

    def fetch_pipelines(self, repo_name: str, since: datetime) -> list[dict[str, Any]]:
        """Fetch workflow runs from GitHub Actions.

        Args:
            repo_name: Repository name (e.g., "owner/repo")
            since: Fetch pipelines created after this date

        Returns:
            List of pipeline data dictionaries
        """
        try:
            repo = self.github.get_repo(repo_name)
        except UnknownObjectException:
            print(f"   ⚠️  GitHub repo not found: {repo_name}")
            return []

        # Check cache first
        cached_pipelines = self._get_cached_pipelines_bulk(repo_name, since)

        # Determine which pipelines are missing
        cached_pipeline_ids = {p["pipeline_id"] for p in cached_pipelines}

        # Ensure since is timezone-aware for comparison with GitHub's timezone-aware datetimes
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)

        # Fetch workflow runs with rate limiting
        new_pipelines = []
        for attempt in range(self.rate_limit_retries):
            try:
                # Get all workflow runs created since date
                for run in repo.get_workflow_runs():
                    # Stop if we've reached our date threshold
                    if run.created_at < since:
                        break

                    # Skip if already cached
                    if str(run.id) in cached_pipeline_ids:
                        continue

                    # Extract pipeline data
                    pipeline_data = self._extract_pipeline_data(run, repo_name)
                    new_pipelines.append(pipeline_data)

                break  # Success, exit retry loop

            except RateLimitExceededException:
                if attempt < self.rate_limit_retries - 1:
                    wait_time = self.backoff_factor**attempt
                    print(f"   ⏳ GitHub rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print("   ❌ GitHub rate limit exceeded, using cached pipelines only")
                    return cached_pipelines

        # Cache new pipelines
        if new_pipelines:
            self._cache_pipelines_bulk(repo_name, new_pipelines)
            print(f"   💾 Cached {len(new_pipelines)} new GitHub Actions pipelines")

        # Track cache performance
        cache_hits = len(cached_pipelines)
        cache_misses = len(new_pipelines)
        if cache_hits > 0 or cache_misses > 0:
            hit_rate = cache_hits / (cache_hits + cache_misses) * 100
            print(
                f"   📊 CI/CD cache: {cache_hits} hits, {cache_misses} misses ({hit_rate:.1f}% hit rate)"
            )

        return cached_pipelines + new_pipelines

    def _extract_pipeline_data(self, run: Any, repo_name: str) -> dict[str, Any]:
        """Extract pipeline data from workflow run.

        Args:
            run: GitHub workflow run object
            repo_name: Repository name

        Returns:
            Pipeline data dictionary
        """
        # Calculate duration
        duration = 0
        if run.updated_at and run.created_at:
            duration = int((run.updated_at - run.created_at).total_seconds())

        # Ensure created_at is timezone-aware
        created_at = run.created_at
        if created_at and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        return {
            "pipeline_id": str(run.id),
            "workflow_name": run.name,
            "repo_path": repo_name,
            "branch": run.head_branch,
            "commit_sha": run.head_sha,
            "status": run.conclusion or "pending",
            "created_at": created_at,
            "duration_seconds": duration,
            "trigger_type": run.event,
            "author": run.head_commit.author.name if run.head_commit else None,
            "url": run.html_url,
            "platform": self.platform_name,
            "platform_data": {
                "workflow_id": run.workflow_id,
                "run_number": run.run_number,
                "run_attempt": run.run_attempt,
                "workflow_name": run.name,
            },
        }

    def enrich_commits_with_pipelines(
        self, commits: list[dict[str, Any]], pipelines: list[dict[str, Any]]
    ) -> None:
        """Enrich commits with pipeline status.

        Args:
            commits: List of commit dictionaries to enrich (modified in-place)
            pipelines: List of pipeline data dictionaries
        """
        # Create commit SHA to pipelines mapping
        commit_to_pipelines = {}
        for pipeline in pipelines:
            sha = pipeline.get("commit_sha")
            if sha:
                if sha not in commit_to_pipelines:
                    commit_to_pipelines[sha] = []
                commit_to_pipelines[sha].append(pipeline)

        # Enrich commits
        for commit in commits:
            sha = commit.get("hash")
            if sha in commit_to_pipelines:
                commit_pipelines = commit_to_pipelines[sha]

                # Add pipeline information
                commit["ci_pipelines"] = commit_pipelines
                commit["ci_pipeline_count"] = len(commit_pipelines)

                # Determine overall pipeline status
                statuses = [p["status"] for p in commit_pipelines]
                if "failure" in statuses:
                    commit["ci_status"] = "failure"
                elif "success" in statuses:
                    commit["ci_status"] = "success"
                else:
                    commit["ci_status"] = "pending"
