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
        fetch_pr_reviews: bool = False,
    ):
        """Initialize GitHub integration.

        Args:
            token: GitHub personal access token.
            cache: Shared analysis cache instance.
            rate_limit_retries: Number of retry attempts on rate-limit errors.
            backoff_factor: Exponential backoff base (seconds).
            allowed_ticket_platforms: Optional whitelist of ticket platforms to extract.
            fetch_pr_reviews: When True, fetch review data (approvals, change requests,
                time-to-first-review) via additional API calls per PR.  Disabled by
                default to protect existing users from unexpected rate-limit increases.
        """
        self.github = Github(token)
        self.cache = cache
        self.rate_limit_retries = rate_limit_retries
        self.backoff_factor = backoff_factor
        self.allowed_ticket_platforms = allowed_ticket_platforms
        self.fetch_pr_reviews = fetch_pr_reviews

        # Initialize schema version manager for incremental API data fetching
        self.schema_manager = create_schema_manager(cache.cache_dir)

        # BUG 5 FIX: Instantiate extractors once at construction time instead of
        # re-creating them on every _extract_pr_data call.  Each StoryPointExtractor /
        # TicketExtractor allocates regex patterns and (optionally) ML models, so
        # recreating them per-PR caused both memory pressure and CPU overhead.
        from ..extractors.story_points import StoryPointExtractor
        from ..extractors.tickets import TicketExtractor

        self._sp_extractor = StoryPointExtractor()
        self._ticket_extractor = TicketExtractor(allowed_platforms=self.allowed_ticket_platforms)

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
        # Announce review-fetch mode so operators know extra API calls are being made
        if self.fetch_pr_reviews and new_prs:
            print(
                f"   🔍 fetch_pr_reviews enabled — fetching review data for "
                f"{len(new_prs)} new PR(s) (uses additional API quota)"
            )

        new_pr_data = []
        for pr in new_prs:
            pr_data = self._extract_pr_data(pr, fetch_reviews=self.fetch_pr_reviews)
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
                        # Enhanced PR tracking fields (v3.0) - use getattr for backward
                        # compatibility with databases that pre-date the v3.0 migration.
                        "review_comments": getattr(cached_pr, "review_comments_count", None) or 0,
                        "pr_comments_count": getattr(cached_pr, "pr_comments_count", None) or 0,
                        "approvals_count": getattr(cached_pr, "approvals_count", None) or 0,
                        "change_requests_count": getattr(cached_pr, "change_requests_count", None)
                        or 0,
                        "reviewers": getattr(cached_pr, "reviewers", None) or [],
                        "approved_by": getattr(cached_pr, "approved_by", None) or [],
                        "time_to_first_review_hours": getattr(
                            cached_pr, "time_to_first_review_hours", None
                        ),
                        "revision_count": getattr(cached_pr, "revision_count", None) or 0,
                        "changed_files": getattr(cached_pr, "changed_files", None) or 0,
                        "additions": getattr(cached_pr, "additions", None) or 0,
                        "deletions": getattr(cached_pr, "deletions", None) or 0,
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

    def _extract_review_data(self, pr) -> dict[str, Any]:
        """Fetch and extract review-level data for a pull request.

        WHY: Review data (approvals, change requests, reviewer lists, time-to-review)
        requires separate API calls beyond the base PR object.  This method is only
        invoked when the ``fetch_pr_reviews`` config flag is enabled so that existing
        users are not impacted by the additional API budget.

        PyGitHub paginates reviews automatically; we iterate the full list once and
        process each state in a single pass to keep complexity O(n).

        Args:
            pr: PyGitHub PullRequest object.

        Returns:
            Dictionary with review-specific fields ready to merge into PR data.
        """
        approvals_count = 0
        change_requests_count = 0
        reviewers: list[str] = []
        approved_by: list[str] = []
        time_to_first_review_hours: Optional[float] = None
        earliest_review_at: Optional[datetime] = None

        try:
            for review in pr.get_reviews():
                reviewer_login: str = review.user.login if review.user else "unknown"
                state: str = review.state  # APPROVED | CHANGES_REQUESTED | COMMENTED | DISMISSED

                # Track unique reviewers (any meaningful interaction)
                if reviewer_login not in reviewers and state in {
                    "APPROVED",
                    "CHANGES_REQUESTED",
                    "COMMENTED",
                }:
                    reviewers.append(reviewer_login)

                if state == "APPROVED":
                    approvals_count += 1
                    if reviewer_login not in approved_by:
                        approved_by.append(reviewer_login)

                elif state == "CHANGES_REQUESTED":
                    change_requests_count += 1

                # Track time to first non-COMMENTED review for latency metric
                if state in {"APPROVED", "CHANGES_REQUESTED"} and review.submitted_at:
                    review_time: datetime = review.submitted_at
                    if review_time.tzinfo is None:
                        review_time = review_time.replace(tzinfo=timezone.utc)
                    if earliest_review_at is None or review_time < earliest_review_at:
                        earliest_review_at = review_time

        except Exception as exc:
            print(f"   ⚠️  Could not fetch reviews for PR #{pr.number}: {exc}")

        # Calculate time-to-first-review
        if earliest_review_at and pr.created_at:
            created = pr.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            delta_seconds = (earliest_review_at - created).total_seconds()
            # Guard against negative values (clock skew / backdated reviews)
            time_to_first_review_hours = max(0.0, delta_seconds / 3600)

        # Fetch general issue/PR comments count (separate from inline review comments)
        pr_comments_count = 0
        try:
            # get_issue_comments() returns general timeline comments, not inline review comments
            for _ in pr.get_issue_comments():
                pr_comments_count += 1
        except Exception as exc:
            print(f"   ⚠️  Could not fetch issue comments for PR #{pr.number}: {exc}")

        return {
            "approvals_count": approvals_count,
            "change_requests_count": change_requests_count,
            "reviewers": reviewers,
            "approved_by": approved_by,
            "time_to_first_review_hours": time_to_first_review_hours,
            "pr_comments_count": pr_comments_count,
        }

    def _count_pr_revisions(self, pr) -> int:
        """Estimate PR revision count from commit timeline.

        WHY: GitHub does not expose a first-class "revision" concept.  The best
        proxy available without the Events API is counting commits added *after*
        the first review event.  A simpler approximation — which avoids the extra
        Events API call — is to count distinct commit "pushes": each force-push
        resets the commit list, but the PR timeline events expose ``head_sha``
        changes.  For now we use a lightweight heuristic: total commits beyond 1
        suggests at least one revision cycle.

        This is intentionally conservative; it under-counts but never over-counts.

        Args:
            pr: PyGitHub PullRequest object.

        Returns:
            Estimated revision count (>= 0).
        """
        try:
            commit_count = pr.commits  # Lightweight integer attribute — no extra API call
            # Heuristic: each commit beyond the first is counted as a potential
            # revision push.  Cap at a sane upper bound to avoid garbage data.
            return max(0, min(commit_count - 1, 50))
        except Exception:
            return 0

    def _extract_pr_data(self, pr, fetch_reviews: bool = False) -> dict[str, Any]:
        """Extract relevant data from a GitHub PR object.

        Args:
            pr: PyGitHub PullRequest object.
            fetch_reviews: When True, also call ``_extract_review_data()`` and
                ``_count_pr_revisions()`` to populate enhanced review fields.
                Controlled by the ``fetch_pr_reviews`` config flag.

        Returns:
            Dictionary of PR data ready for caching and metrics calculation.
        """
        # BUG 5 FIX: Use the shared extractor instances created in __init__ instead of
        # instantiating new objects on every call (was O(PR count) allocations).
        # Extract story points from PR title and body
        pr_text = f"{pr.title} {pr.body or ''}"
        story_points = self._sp_extractor.extract_from_text(pr_text)

        # Extract ticket references
        tickets = self._ticket_extractor.extract_from_text(pr_text)

        # Get commit SHAs — pr.get_commits() is paginated automatically by PyGitHub
        commit_hashes = [c.sha for c in pr.get_commits()]

        pr_data: dict[str, Any] = {
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
            # Inline review comment count — available on the base PR object,
            # no extra API call required.
            "review_comments": pr.review_comments,
            "changed_files": pr.changed_files,
            "additions": pr.additions,
            "deletions": pr.deletions,
        }

        if fetch_reviews:
            review_data = self._extract_review_data(pr)
            pr_data.update(review_data)
            pr_data["revision_count"] = self._count_pr_revisions(pr)

        return pr_data

    def calculate_pr_metrics(self, prs: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate PR-level metrics including review quality indicators.

        Basic metrics (size, lifetime, story-point coverage) are always computed.
        Enhanced review metrics (approval rate, time-to-first-review, change-request
        rate) are computed when the underlying data is present — i.e. when
        ``fetch_pr_reviews`` was enabled during data collection.

        Args:
            prs: List of PR data dictionaries as returned by ``_extract_pr_data()``.

        Returns:
            Dictionary of aggregated PR metrics.
        """
        if not prs:
            return {
                "total_prs": 0,
                "avg_pr_size": 0,
                "avg_pr_lifetime_hours": 0,
                "avg_files_per_pr": 0,
                "total_review_comments": 0,
                "prs_with_story_points": 0,
                "story_point_coverage": 0.0,
                # Enhanced review metrics — zeroed/null when no PRs
                "review_data_collected": False,
                "approval_rate": 0.0,
                "avg_approvals_per_pr": 0.0,
                "avg_change_requests_per_pr": 0.0,
                "review_coverage": 0.0,
                "avg_time_to_first_review_hours": None,
                "median_time_to_first_review_hours": None,
                "total_pr_comments": 0,
                "avg_pr_comments_per_pr": 0.0,
                "avg_revision_count": 0.0,
            }

        n = len(prs)

        # --- Basic size / lifetime metrics ---
        total_size = sum((pr.get("additions") or 0) + (pr.get("deletions") or 0) for pr in prs)
        total_files = sum(pr.get("changed_files", 0) or 0 for pr in prs)
        total_inline_comments = sum(pr.get("review_comments", 0) or 0 for pr in prs)

        lifetimes: list[float] = []
        for pr in prs:
            if pr.get("merged_at") and pr.get("created_at"):
                lifetime = (pr["merged_at"] - pr["created_at"]).total_seconds() / 3600
                lifetimes.append(lifetime)

        avg_lifetime = sum(lifetimes) / len(lifetimes) if lifetimes else 0.0

        # --- Story point coverage ---
        prs_with_sp = sum(1 for pr in prs if pr.get("story_points"))
        sp_coverage = prs_with_sp / n * 100

        # --- Enhanced review metrics ---
        # Only aggregate when review data is actually present (non-None approvals_count)
        prs_with_review_data = [pr for pr in prs if pr.get("approvals_count") is not None]
        review_data_available = len(prs_with_review_data)

        # Approval rate: fraction of reviewed PRs that received at least one approval
        if review_data_available:
            approved_prs = sum(
                1 for pr in prs_with_review_data if (pr.get("approvals_count") or 0) > 0
            )
            approval_rate = approved_prs / review_data_available * 100

            avg_approvals = (
                sum(pr.get("approvals_count") or 0 for pr in prs_with_review_data)
                / review_data_available
            )
            avg_change_requests = (
                sum(pr.get("change_requests_count") or 0 for pr in prs_with_review_data)
                / review_data_available
            )

            # Review coverage: fraction of PRs that received at least one review
            reviewed_prs = sum(
                1
                for pr in prs_with_review_data
                if (pr.get("approvals_count") or 0) + (pr.get("change_requests_count") or 0) > 0
            )
            review_coverage = reviewed_prs / review_data_available * 100
        else:
            approval_rate = 0.0
            avg_approvals = 0.0
            avg_change_requests = 0.0
            review_coverage = 0.0

        # Time-to-first-review statistics
        ttfr_values: list[float] = [
            pr["time_to_first_review_hours"]
            for pr in prs
            if pr.get("time_to_first_review_hours") is not None
        ]
        avg_ttfr: Optional[float] = sum(ttfr_values) / len(ttfr_values) if ttfr_values else None
        median_ttfr: Optional[float] = None
        if ttfr_values:
            sorted_ttfr = sorted(ttfr_values)
            mid = len(sorted_ttfr) // 2
            median_ttfr = (
                sorted_ttfr[mid]
                if len(sorted_ttfr) % 2 == 1
                else (sorted_ttfr[mid - 1] + sorted_ttfr[mid]) / 2
            )

        # General PR comments (timeline, not inline review comments)
        total_pr_comments = sum(pr.get("pr_comments_count", 0) or 0 for pr in prs)
        avg_pr_comments = total_pr_comments / n

        # Revision count
        avg_revisions = sum(pr.get("revision_count", 0) or 0 for pr in prs) / n

        return {
            "total_prs": n,
            "avg_pr_size": total_size / n,
            "avg_pr_lifetime_hours": avg_lifetime,
            "avg_files_per_pr": total_files / n,
            "total_review_comments": total_inline_comments,
            "prs_with_story_points": prs_with_sp,
            "story_point_coverage": sp_coverage,
            # --- Enhanced review metrics ---
            # ``review_data_collected`` is True when at least one PR in the
            # dataset has approvals_count populated (i.e. fetch_pr_reviews was
            # enabled for this batch).  Consumers should gate display of review
            # metrics on this flag rather than checking for zero values, because
            # a legitimate dataset may have 0% approval rate (all PRs unreviewed).
            "review_data_collected": review_data_available > 0,
            "approval_rate": approval_rate,
            "avg_approvals_per_pr": avg_approvals,
            "avg_change_requests_per_pr": avg_change_requests,
            "review_coverage": review_coverage,
            "avg_time_to_first_review_hours": avg_ttfr,
            "median_time_to_first_review_hours": median_ttfr,
            "total_pr_comments": total_pr_comments,
            "avg_pr_comments_per_pr": avg_pr_comments,
            "avg_revision_count": avg_revisions,
        }
