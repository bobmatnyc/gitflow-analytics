"""GitHub API integration for PR and issue enrichment."""

import time
from datetime import datetime, timezone
from typing import Any, Optional

from github import Github
from github.GithubException import RateLimitExceededException, UnknownObjectException

from ..core.cache import GitAnalysisCache


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

    def enrich_repository_with_prs(
        self, repo_name: str, commits: list[dict[str, Any]], since: datetime
    ) -> list[dict[str, Any]]:
        """Enrich repository commits with PR data."""
        try:
            repo = self.github.get_repo(repo_name)
        except UnknownObjectException:
            print(f"   ⚠️  GitHub repo not found: {repo_name}")
            return []

        # Get PRs for the time period
        prs = self._get_pull_requests(repo, since)

        # Build commit to PR mapping
        commit_to_pr = {}
        for pr in prs:
            pr_data = self._extract_pr_data(pr)

            # Cache PR data
            self.cache.cache_pr(repo_name, pr_data)

            # Map commits to this PR
            for commit in pr.get_commits():
                commit_to_pr[commit.sha] = pr_data

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
