"""GitHub API integration for PR and issue enrichment."""

import time
from datetime import datetime, timezone
from typing import Any, Optional

from github import Github
from github.GithubException import RateLimitExceededException, UnknownObjectException

from ..core.cache import GitAnalysisCache
from ..core.schema_version import create_schema_manager


class GitHubIntegration:
    """Integrate with GitHub API for PR and issue data."""

    def __init__(
        self,
        token: str,
        cache: GitAnalysisCache,
        rate_limit_retries: int = 3,
        backoff_factor: int = 2,
        allowed_ticket_platforms: Optional[list[str]] = None,
    ):
        """Initialize GitHub integration."""
        self.github = Github(token)
        self.cache = cache
        self.rate_limit_retries = rate_limit_retries
        self.backoff_factor = backoff_factor
        self.allowed_ticket_platforms = allowed_ticket_platforms

        # Initialize schema version manager for incremental API data fetching
        self.schema_manager = create_schema_manager(cache.cache_dir)

    def _get_incremental_fetch_date(
        self, component: str, requested_since: datetime, config: dict[str, Any]
    ) -> datetime:
        """Determine the actual fetch date based on schema versioning."""
        # Ensure requested_since is timezone-aware
        if requested_since.tzinfo is None:
            requested_since = requested_since.replace(tzinfo=timezone.utc)

        # Check if schema has changed
        if self.schema_manager.has_schema_changed(component, config):
            print(
                f"   🔄 {component.title()} API schema changed, fetching all data since {requested_since}"
            )
            return requested_since

        # Get last processed date
        last_processed = self.schema_manager.get_last_processed_date(component)
        if not last_processed:
            print(f"   📥 First {component} API fetch, getting data since {requested_since}")
            return requested_since

        # Ensure last_processed is timezone-aware
        if last_processed.tzinfo is None:
            last_processed = last_processed.replace(tzinfo=timezone.utc)

        # Use the later of the two dates (don't go backwards)
        fetch_since = max(last_processed, requested_since)

        if fetch_since > requested_since:
            print(f"   ⚡ {component.title()} incremental fetch since {fetch_since}")
        else:
            print(f"   📥 {component.title()} full fetch since {requested_since}")

        return fetch_since

    def enrich_repository_with_prs(
        self, repo_name: str, commits: list[dict[str, Any]], since: datetime
    ) -> list[dict[str, Any]]:
        """Enrich repository commits with PR data using incremental fetching."""
        try:
            repo = self.github.get_repo(repo_name)
        except UnknownObjectException:
            print(f"   ⚠️  GitHub repo not found: {repo_name}")
            return []

        # Check if we need to fetch new PR data
        github_config = {
            "rate_limit_retries": self.rate_limit_retries,
            "backoff_factor": self.backoff_factor,
            "allowed_ticket_platforms": self.allowed_ticket_platforms,
        }

        # Determine the actual start date for fetching
        fetch_since = self._get_incremental_fetch_date("github", since, github_config)

        # Check cache first for existing PRs in this time period
        cached_prs_data = self._get_cached_prs_bulk(repo_name, fetch_since)

        # Get PRs for the time period (may be incremental)
        prs = self._get_pull_requests(repo, fetch_since)

        # Track cache performance
        cached_pr_numbers = {pr["number"] for pr in cached_prs_data}
        new_prs = [pr for pr in prs if pr.number not in cached_pr_numbers]
        cache_hits = len(cached_prs_data)
        cache_misses = len(new_prs)

        if cache_hits > 0 or cache_misses > 0:
            print(
                f"   📊 GitHub PR cache: {cache_hits} hits, {cache_misses} misses ({cache_hits / (cache_hits + cache_misses) * 100:.1f}% hit rate)"
                if (cache_hits + cache_misses) > 0
                else ""
            )

        # Update schema tracking after successful fetch
        if prs:
            self.schema_manager.mark_date_processed("github", since, github_config)

        # Process new PRs and cache them
        new_pr_data = []
        for pr in new_prs:
            pr_data = self._extract_pr_data(pr)
            new_pr_data.append(pr_data)

        # Bulk cache new PR data
        if new_pr_data:
            self._cache_prs_bulk(repo_name, new_pr_data)
            print(f"   💾 Cached {len(new_pr_data)} new GitHub PRs")

        # Combine cached and new PR data
        all_pr_data = cached_prs_data + new_pr_data

        # Build commit to PR mapping
        commit_to_pr = {}
        for pr_data in all_pr_data:
            # Map commits to this PR (need to get commit hashes from cached data)
            for commit_hash in pr_data.get("commit_hashes", []):
                commit_to_pr[commit_hash] = pr_data

        # Enrich commits with PR data
        enriched_prs = []
        for commit in commits:
            if commit["hash"] in commit_to_pr:
                pr_data = commit_to_pr[commit["hash"]]

                # Use PR story points if commit doesn't have them
                if not commit.get("story_points") and pr_data.get("story_points"):
                    commit["story_points"] = pr_data["story_points"]

                # Add PR reference
                commit["pr_number"] = pr_data["number"]
                commit["pr_title"] = pr_data["title"]

                # Add to PR list if not already there
                if pr_data not in enriched_prs:
                    enriched_prs.append(pr_data)

        return enriched_prs

    def _get_cached_prs_bulk(self, repo_name: str, since: datetime) -> list[dict[str, Any]]:
        """Get cached PRs for a repository from the given date onwards.

        WHY: Bulk PR cache lookups avoid redundant GitHub API calls and
        significantly improve performance on repeated analysis runs.

        Args:
            repo_name: GitHub repository name (e.g., "owner/repo")
            since: Only return PRs merged after this date

        Returns:
            List of cached PR data dictionaries
        """
        cached_prs = []
        with self.cache.get_session() as session:
            from ..models.database import PullRequestCache

            # Ensure since is timezone-aware for comparison
            if since.tzinfo is None:
                since = since.replace(tzinfo=timezone.utc)

            cached_results = (
                session.query(PullRequestCache)
                .filter(
                    PullRequestCache.repo_path == repo_name,
                    PullRequestCache.merged_at >= since.replace(tzinfo=None),  # Store as naive UTC
                )
                .all()
            )

            for cached_pr in cached_results:
                if not self._is_pr_stale(cached_pr.cached_at):
                    pr_data = {
                        "number": cached_pr.pr_number,
                        "title": cached_pr.title or "",
                        "description": cached_pr.description or "",
                        "author": cached_pr.author or "",
                        "created_at": cached_pr.created_at,
                        "merged_at": cached_pr.merged_at,
                        "story_points": cached_pr.story_points or 0,
                        "labels": cached_pr.labels or [],
                        "commit_hashes": cached_pr.commit_hashes or [],
                        "ticket_references": [],  # Would need additional extraction
                        "review_comments": 0,  # Not stored in current schema
                        "changed_files": 0,  # Not stored in current schema
                        "additions": 0,  # Not stored in current schema
                        "deletions": 0,  # Not stored in current schema
                    }
                    cached_prs.append(pr_data)

        return cached_prs

    def _cache_prs_bulk(self, repo_name: str, prs: list[dict[str, Any]]) -> None:
        """Cache multiple PRs in bulk for better performance.

        WHY: Bulk caching is more efficient than individual cache operations,
        reducing database overhead when caching many PRs from GitHub API.

        Args:
            repo_name: GitHub repository name
            prs: List of PR data dictionaries to cache
        """
        if not prs:
            return

        for pr_data in prs:
            # Use existing cache_pr method which handles upserts properly
            self.cache.cache_pr(repo_name, pr_data)

    def _is_pr_stale(self, cached_at: datetime) -> bool:
        """Check if cached PR data is stale based on cache TTL.

        Args:
            cached_at: When the PR was cached

        Returns:
            True if stale and should be refreshed, False if still fresh
        """
        from datetime import timedelta

        if self.cache.ttl_hours == 0:  # No expiration
            return False

        stale_threshold = datetime.utcnow() - timedelta(hours=self.cache.ttl_hours)
        return cached_at < stale_threshold

    def _get_pull_requests(self, repo, since: datetime) -> list[Any]:
        """Get pull requests with rate limit handling."""
        prs = []

        # Ensure since is timezone-aware for comparison with GitHub's timezone-aware datetimes
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)

        for attempt in range(self.rate_limit_retries):
            try:
                # Get all PRs updated since the date
                for pr in repo.get_pulls(state="all", sort="updated", direction="desc"):
                    if pr.updated_at < since:
                        break

                    # Only include PRs that were merged in our time period
                    if pr.merged and pr.merged_at >= since:
                        prs.append(pr)

                return prs

            except RateLimitExceededException:
                if attempt < self.rate_limit_retries - 1:
                    wait_time = self.backoff_factor**attempt
                    print(f"   ⏳ GitHub rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print("   ❌ GitHub rate limit exceeded, skipping PR enrichment")
                    return []

        return prs

    def _extract_pr_data(self, pr) -> dict[str, Any]:
        """Extract relevant data from a GitHub PR object."""
        from ..extractors.story_points import StoryPointExtractor
        from ..extractors.tickets import TicketExtractor

        sp_extractor = StoryPointExtractor()
        ticket_extractor = TicketExtractor(allowed_platforms=self.allowed_ticket_platforms)

        # Extract story points from PR title and body
        pr_text = f"{pr.title} {pr.body or ''}"
        story_points = sp_extractor.extract_from_text(pr_text)

        # Extract ticket references
        tickets = ticket_extractor.extract_from_text(pr_text)

        # Get commit SHAs
        commit_hashes = [c.sha for c in pr.get_commits()]

        return {
            "number": pr.number,
            "title": pr.title,
            "description": pr.body,
            "author": pr.user.login,
            "created_at": pr.created_at,
            "merged_at": pr.merged_at,
            "story_points": story_points,
            "labels": [label.name for label in pr.labels],
            "commit_hashes": commit_hashes,
            "ticket_references": tickets,
            "review_comments": pr.review_comments,
            "changed_files": pr.changed_files,
            "additions": pr.additions,
            "deletions": pr.deletions,
        }

    def calculate_pr_metrics(self, prs: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate PR-level metrics."""
        if not prs:
            return {
                "avg_pr_size": 0,
                "avg_pr_lifetime_hours": 0,
                "avg_files_per_pr": 0,
                "total_review_comments": 0,
            }

        total_size = sum(pr["additions"] + pr["deletions"] for pr in prs)
        total_files = sum(pr.get("changed_files", 0) for pr in prs)
        total_comments = sum(pr.get("review_comments", 0) for pr in prs)

        # Calculate average PR lifetime
        lifetimes = []
        for pr in prs:
            if pr.get("merged_at") and pr.get("created_at"):
                lifetime = (pr["merged_at"] - pr["created_at"]).total_seconds() / 3600
                lifetimes.append(lifetime)

        avg_lifetime = sum(lifetimes) / len(lifetimes) if lifetimes else 0

        return {
            "total_prs": len(prs),
            "avg_pr_size": total_size / len(prs),
            "avg_pr_lifetime_hours": avg_lifetime,
            "avg_files_per_pr": total_files / len(prs),
            "total_review_comments": total_comments,
            "prs_with_story_points": sum(1 for pr in prs if pr.get("story_points")),
            "story_point_coverage": sum(1 for pr in prs if pr.get("story_points")) / len(prs) * 100,
        }
