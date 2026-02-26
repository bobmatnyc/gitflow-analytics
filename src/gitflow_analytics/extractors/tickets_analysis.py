"""Ticket coverage and untracked commit analysis mixin.

This module provides TicketAnalysisMixin, which adds coverage analysis and
untracked-commit pattern analysis to TicketExtractor via multiple inheritance.

Methods here depend on instance attributes set by TicketExtractor.__init__:
    self.allowed_platforms, self.untracked_file_threshold,
    self.compiled_category_patterns
and on instance methods:
    self.extract_from_text(), self.categorize_commit(), self._remove_ticket_references()
"""

import logging
import re
from collections import defaultdict
from datetime import timezone
from typing import Any, cast

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)


class TicketAnalysisMixin:
    """Mixin adding coverage and untracked-pattern analysis to TicketExtractor.

    Requires the host class to define:
        - self.allowed_platforms: list[str] | None
        - self.untracked_file_threshold: int
        - self.compiled_category_patterns: dict
        - self.extract_from_text(text) -> list[dict]
        - self.categorize_commit(message) -> str
        - self._remove_ticket_references(message) -> str
    """

    def analyze_ticket_coverage(
        self, commits: list[dict[str, Any]], prs: list[dict[str, Any]], progress_display=None
    ) -> dict[str, Any]:
        """Analyze ticket reference coverage across commits and PRs.

        Args:
            commits: List of commit dictionaries to analyze
            prs: List of PR dictionaries to analyze
            progress_display: Optional progress display for showing analysis progress

        Note:
            This method re-extracts tickets from commit messages rather than using cached
            'ticket_references' to ensure the analysis respects the current allowed_platforms
            configuration. Cached data may contain tickets from all platforms from previous runs.
        """
        ticket_platforms: defaultdict[str, int] = defaultdict(int)
        untracked_commits: list[dict[str, Any]] = []
        ticket_summary: defaultdict[str, set[str]] = defaultdict(set)

        results = {
            "total_commits": len(commits),
            "total_prs": len(prs),
            "commits_with_tickets": 0,
            "prs_with_tickets": 0,
            "ticket_platforms": ticket_platforms,
            "untracked_commits": untracked_commits,
            "ticket_summary": ticket_summary,
        }

        commits_analyzed = 0
        commits_with_ticket_refs = 0
        tickets_found = 0

        commit_iterator = commits
        if progress_display and hasattr(progress_display, "console"):
            commit_iterator = commits
        elif TQDM_AVAILABLE:
            commit_iterator = tqdm(
                commits, desc="Analyzing commits for tickets", unit="commits", leave=False
            )

        for commit in commit_iterator:
            if not isinstance(commit, dict):
                logger.error(f"Expected commit to be dict, got {type(commit)}: {commit}")
                continue

            commits_analyzed += 1
            commit_message = commit.get("message", "")
            ticket_refs = self.extract_from_text(commit_message)  # type: ignore[attr-defined]

            if commits_analyzed <= 5:
                logger.debug(
                    f"Commit {commits_analyzed}: hash={commit.get('hash', 'N/A')[:8]}, "
                    f"re-extracted ticket_refs={ticket_refs} "
                    f"(allowed_platforms={self.allowed_platforms})"  # type: ignore[attr-defined]
                )

            if ticket_refs:
                commits_with_ticket_refs += 1
                commits_with_tickets = cast(int, results["commits_with_tickets"])
                results["commits_with_tickets"] = commits_with_tickets + 1
                for ticket in ticket_refs:
                    if isinstance(ticket, dict):
                        platform = ticket.get("platform", "unknown")
                        ticket_id = ticket.get("id", "")
                    else:
                        platform = "jira"
                        ticket_id = ticket

                    platform_count = ticket_platforms[platform]
                    ticket_platforms[platform] = platform_count + 1
                    ticket_summary[platform].add(ticket_id)
                    tickets_found += 1
            else:
                files_changed = self._get_files_changed_count(commit)
                threshold = self.untracked_file_threshold  # type: ignore[attr-defined]
                if not commit.get("is_merge") and files_changed >= threshold:
                    category = self.categorize_commit(commit.get("message", ""))  # type: ignore[attr-defined]

                    commit_data = {
                        "hash": commit.get("hash", "")[:7],
                        "full_hash": commit.get("hash", ""),
                        "message": commit.get("message", "").split("\n")[0][:100],
                        "full_message": commit.get("message", ""),
                        "author": commit.get(
                            "canonical_name", commit.get("author_name", "Unknown")
                        ),
                        "author_email": commit.get("author_email", ""),
                        "canonical_id": commit.get("canonical_id", commit.get("author_email", "")),
                        "timestamp": commit.get("timestamp"),
                        "project_key": commit.get("project_key", "UNKNOWN"),
                        "files_changed": files_changed,
                        "lines_added": commit.get("insertions", 0),
                        "lines_removed": commit.get("deletions", 0),
                        "lines_changed": (commit.get("insertions", 0) + commit.get("deletions", 0)),
                        "category": category,
                        "is_merge": commit.get("is_merge", False),
                    }

                    untracked_commits.append(commit_data)

            if TQDM_AVAILABLE and hasattr(commit_iterator, "set_postfix"):
                commit_iterator.set_postfix(
                    {
                        "tickets": tickets_found,
                        "with_tickets": commits_with_ticket_refs,
                        "untracked": len(untracked_commits),
                    }
                )

        pr_tickets_found = 0

        pr_iterator = prs
        if (
            prs
            and TQDM_AVAILABLE
            and not (progress_display and hasattr(progress_display, "console"))
        ):
            pr_iterator = tqdm(prs, desc="Analyzing PRs for tickets", unit="PRs", leave=False)

        for pr in pr_iterator:
            pr_text = f"{pr.get('title', '')} {pr.get('description', '')}"
            tickets = self.extract_from_text(pr_text)  # type: ignore[attr-defined]

            if tickets:
                prs_with_tickets = cast(int, results["prs_with_tickets"])
                results["prs_with_tickets"] = prs_with_tickets + 1
                for ticket in tickets:
                    platform = ticket["platform"]
                    platform_count = ticket_platforms[platform]
                    ticket_platforms[platform] = platform_count + 1
                    ticket_summary[platform].add(ticket["id"])
                    pr_tickets_found += 1

            if TQDM_AVAILABLE and hasattr(pr_iterator, "set_postfix"):
                pr_iterator.set_postfix(
                    {"tickets": pr_tickets_found, "with_tickets": results["prs_with_tickets"]}
                )

        total_commits = cast(int, results["total_commits"])
        commits_with_tickets_count = cast(int, results["commits_with_tickets"])
        results["commit_coverage_pct"] = (
            commits_with_tickets_count / total_commits * 100 if total_commits > 0 else 0
        )

        total_prs = cast(int, results["total_prs"])
        prs_with_tickets_count = cast(int, results["prs_with_tickets"])
        results["pr_coverage_pct"] = (
            prs_with_tickets_count / total_prs * 100 if total_prs > 0 else 0
        )

        results["ticket_summary"] = {
            platform: len(tickets) for platform, tickets in ticket_summary.items()
        }

        def safe_timestamp_key(commit: dict[str, Any]) -> Any:
            ts = commit.get("timestamp")
            if ts is None:
                return ""
            if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts

        untracked_commits.sort(key=safe_timestamp_key, reverse=True)

        final_commits_with_tickets = cast(int, results["commits_with_tickets"])
        logger.debug(
            f"Ticket coverage analysis complete: {commits_analyzed} commits analyzed, "
            f"{commits_with_ticket_refs} had ticket_refs, "
            f"{final_commits_with_tickets} counted as with tickets"
        )
        if commits_analyzed > 0 and final_commits_with_tickets == 0:
            logger.warning(
                f"Zero commits with tickets found out of {commits_analyzed} commits analyzed"
            )

        return results

    def calculate_developer_ticket_coverage(
        self, commits: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate ticket coverage percentage per developer.

        WHY: Individual developer ticket coverage was hardcoded to 0.0, causing
        reports to show contradictory information where total coverage was >0%
        but all individual developers showed 0%.

        Args:
            commits: List of commit dictionaries with ticket_references and identity info

        Returns:
            Dictionary mapping canonical_id/author_email to coverage percentage
        """
        if not commits:
            return {}

        developer_commits: dict[str, int] = {}
        developer_with_tickets: dict[str, int] = {}

        threshold = self.untracked_file_threshold  # type: ignore[attr-defined]

        for commit in commits:
            if commit.get("is_merge"):
                continue

            files_changed = self._get_files_changed_count(commit)
            if files_changed < threshold:
                continue

            developer_id = commit.get("canonical_id") or commit.get("author_email", "unknown")

            if developer_id not in developer_commits:
                developer_commits[developer_id] = 0
                developer_with_tickets[developer_id] = 0

            developer_commits[developer_id] += 1

            ticket_refs = commit.get("ticket_references", [])
            if ticket_refs:
                developer_with_tickets[developer_id] += 1

        coverage_by_developer: dict[str, float] = {}
        for developer_id in developer_commits:
            total = developer_commits[developer_id]
            with_tickets = developer_with_tickets[developer_id]
            coverage_by_developer[developer_id] = round(
                (with_tickets / total) * 100, 1
            ) if total > 0 else 0.0

        logger.debug(f"Calculated ticket coverage for {len(coverage_by_developer)} developers")
        return coverage_by_developer

    def _get_files_changed_count(self, commit: dict[str, Any]) -> int:
        """Extract the number of files changed from commit data.

        Handles files_changed as either an integer count or a list of file paths.
        Priority: files_changed_count > files_changed (int) > files_changed (list) > 0
        """
        if "files_changed_count" in commit:
            return commit["files_changed_count"]

        files_changed = commit.get("files_changed")
        if files_changed is not None:
            if isinstance(files_changed, int):
                return files_changed
            elif isinstance(files_changed, list):
                return len(files_changed)

        return 0

    def analyze_untracked_patterns(self, untracked_commits: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze patterns in untracked commits for insights.

        WHY: Understanding patterns in untracked work helps identify:
        - Common types of work that bypass ticket tracking
        - Developers who need process guidance
        - Categories of work that should be tracked vs. allowed to be untracked

        Args:
            untracked_commits: List of untracked commit data

        Returns:
            Dictionary with pattern analysis results
        """
        if not untracked_commits:
            return {
                "total_untracked": 0,
                "categories": {},
                "top_contributors": [],
                "projects": {},
                "avg_commit_size": 0,
                "recommendations": [],
            }

        categories: dict[str, Any] = {}
        for commit in untracked_commits:
            category = commit.get("category", "other")
            if category not in categories:
                categories[category] = {"count": 0, "lines_changed": 0, "examples": []}
            categories[category]["count"] += 1
            categories[category]["lines_changed"] += commit.get("lines_changed", 0)
            if len(categories[category]["examples"]) < 3:
                categories[category]["examples"].append(
                    {
                        "hash": commit.get("hash", ""),
                        "message": commit.get("message", ""),
                        "author": commit.get("author", ""),
                    }
                )

        contributors: dict[str, Any] = {}
        for commit in untracked_commits:
            author = commit.get("canonical_id", commit.get("author_email", "Unknown"))
            if author not in contributors:
                contributors[author] = {"count": 0, "categories": set()}
            contributors[author]["count"] += 1
            contributors[author]["categories"].add(commit.get("category", "other"))

        for author_data in contributors.values():
            author_data["categories"] = list(author_data["categories"])

        top_contributors = sorted(
            [(author, data["count"]) for author, data in contributors.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        projects: dict[str, Any] = {}
        for commit in untracked_commits:
            project = commit.get("project_key", "UNKNOWN")
            if project not in projects:
                projects[project] = {"count": 0, "categories": set()}
            projects[project]["count"] += 1
            projects[project]["categories"].add(commit.get("category", "other"))

        for project_data in projects.values():
            project_data["categories"] = list(project_data["categories"])

        total_lines = sum(commit.get("lines_changed", 0) for commit in untracked_commits)
        avg_commit_size = total_lines / len(untracked_commits) if untracked_commits else 0

        recommendations = self._generate_untracked_recommendations(
            categories, contributors, projects, len(untracked_commits)
        )

        return {
            "total_untracked": len(untracked_commits),
            "categories": categories,
            "top_contributors": top_contributors,
            "projects": projects,
            "avg_commit_size": round(avg_commit_size, 1),
            "recommendations": recommendations,
        }

    def _generate_untracked_recommendations(
        self,
        categories: dict[str, Any],
        contributors: dict[str, Any],
        projects: dict[str, Any],
        total_untracked: int,
    ) -> list[dict[str, str]]:
        """Generate recommendations based on untracked commit patterns."""
        recommendations: list[dict[str, str]] = []

        if categories.get("feature", {}).get("count", 0) > total_untracked * 0.2:
            recommendations.append(
                {
                    "type": "process",
                    "title": "Track Feature Development",
                    "description": "Many feature commits lack ticket references. Consider requiring tickets for new features.",
                    "priority": "high",
                }
            )

        if categories.get("bug_fix", {}).get("count", 0) > total_untracked * 0.15:
            recommendations.append(
                {
                    "type": "process",
                    "title": "Improve Bug Tracking",
                    "description": "Bug fixes should be tracked through issue management systems.",
                    "priority": "high",
                }
            )

        low_priority_categories = ["style", "documentation", "maintenance"]
        low_priority_count = sum(
            categories.get(cat, {}).get("count", 0) for cat in low_priority_categories
        )

        if low_priority_count > total_untracked * 0.6:
            recommendations.append(
                {
                    "type": "positive",
                    "title": "Appropriate Untracked Work",
                    "description": "Most untracked commits are maintenance/style/docs - this is acceptable.",
                    "priority": "low",
                }
            )

        if len(contributors) > 1:
            max_contributor_count = max(data["count"] for data in contributors.values())
            if max_contributor_count > total_untracked * 0.5:
                recommendations.append(
                    {
                        "type": "team",
                        "title": "Provide Process Training",
                        "description": "Some developers need guidance on ticket referencing practices.",
                        "priority": "medium",
                    }
                )

        return recommendations
