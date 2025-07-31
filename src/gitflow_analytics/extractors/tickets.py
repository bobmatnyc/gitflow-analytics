"""Ticket reference extraction for multiple platforms."""

import re
from collections import defaultdict
from typing import Any, Optional, cast


class TicketExtractor:
    """Extract ticket references from various issue tracking systems."""

    def __init__(self, allowed_platforms: Optional[list[str]] = None) -> None:
        """Initialize with patterns for different platforms.

        Args:
            allowed_platforms: List of platforms to extract tickets from.
                              If None, all platforms are allowed.
        """
        self.allowed_platforms = allowed_platforms
        self.patterns = {
            "jira": [
                r"([A-Z]{2,10}-\d+)",  # Standard JIRA format: PROJ-123
            ],
            "github": [
                r"#(\d+)",  # GitHub issues: #123
                r"GH-(\d+)",  # Alternative format: GH-123
                r"(?:fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved)\s+#(\d+)",
            ],
            "clickup": [
                r"CU-([a-z0-9]+)",  # ClickUp: CU-abc123
                r"#([a-z0-9]{6,})",  # ClickUp short format
            ],
            "linear": [
                r"([A-Z]{2,5}-\d+)",  # Linear: ENG-123, similar to JIRA
                r"LIN-(\d+)",  # Alternative: LIN-123
            ],
        }

        # Compile patterns only for allowed platforms
        self.compiled_patterns = {}
        for platform, patterns in self.patterns.items():
            # Skip platforms not in allowed list
            if self.allowed_platforms and platform not in self.allowed_platforms:
                continue
            self.compiled_patterns[platform] = [
                re.compile(pattern, re.IGNORECASE if platform != "jira" else 0)
                for pattern in patterns
            ]

    def extract_from_text(self, text: str) -> list[dict[str, str]]:
        """Extract all ticket references from text."""
        if not text:
            return []

        tickets = []
        seen = set()  # Avoid duplicates

        for platform, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                for match in matches:
                    ticket_id = match if isinstance(match, str) else match[0]

                    # Normalize ticket ID
                    if platform == "jira" or platform == "linear":
                        ticket_id = ticket_id.upper()

                    # Create unique key
                    key = f"{platform}:{ticket_id}"
                    if key not in seen:
                        seen.add(key)
                        tickets.append(
                            {
                                "platform": platform,
                                "id": ticket_id,
                                "full_id": self._format_ticket_id(platform, ticket_id),
                            }
                        )

        return tickets

    def extract_by_platform(self, text: str) -> dict[str, list[str]]:
        """Extract tickets grouped by platform."""
        tickets = self.extract_from_text(text)

        by_platform = defaultdict(list)
        for ticket in tickets:
            by_platform[ticket["platform"]].append(ticket["id"])

        return dict(by_platform)

    def analyze_ticket_coverage(
        self, commits: list[dict[str, Any]], prs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze ticket reference coverage across commits and PRs."""
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

        # Analyze commits
        for commit in commits:
            ticket_refs = commit.get("ticket_references", [])
            if ticket_refs:
                commits_with_tickets = cast(int, results["commits_with_tickets"])
                results["commits_with_tickets"] = commits_with_tickets + 1
                for ticket in ticket_refs:
                    if isinstance(ticket, dict):
                        platform = ticket.get("platform", "unknown")
                        ticket_id = ticket.get("id", "")
                    else:
                        # Legacy format - assume JIRA
                        platform = "jira"
                        ticket_id = ticket

                    platform_count = ticket_platforms[platform]
                    ticket_platforms[platform] = platform_count + 1
                    ticket_summary[platform].add(ticket_id)
            else:
                # Track significant untracked commits
                if not commit.get("is_merge") and commit.get("files_changed", 0) > 3:
                    untracked_commits.append(
                        {
                            "hash": commit.get("hash", "")[:7],
                            "message": commit.get("message", "").split("\n")[0][:60],
                            "files_changed": commit.get("files_changed", 0),
                        }
                    )

        # Analyze PRs
        for pr in prs:
            # Extract tickets from PR title and description
            pr_text = f"{pr.get('title', '')} {pr.get('description', '')}"
            tickets = self.extract_from_text(pr_text)

            if tickets:
                prs_with_tickets = cast(int, results["prs_with_tickets"])
                results["prs_with_tickets"] = prs_with_tickets + 1
                for ticket in tickets:
                    platform = ticket["platform"]
                    platform_count = ticket_platforms[platform]
                    ticket_platforms[platform] = platform_count + 1
                    ticket_summary[platform].add(ticket["id"])

        # Calculate coverage percentages
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

        # Convert sets to counts for summary
        results["ticket_summary"] = {
            platform: len(tickets) for platform, tickets in ticket_summary.items()
        }

        return results

    def _format_ticket_id(self, platform: str, ticket_id: str) -> str:
        """Format ticket ID for display."""
        if platform == "github":
            return f"#{ticket_id}"
        elif platform == "clickup":
            return f"CU-{ticket_id}" if not ticket_id.startswith("CU-") else ticket_id
        else:
            return ticket_id
