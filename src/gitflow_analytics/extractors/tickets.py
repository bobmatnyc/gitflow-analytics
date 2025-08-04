"""Ticket reference extraction for multiple platforms."""

import re
from collections import defaultdict
from typing import Any, Optional, cast


class TicketExtractor:
    """Extract ticket references from various issue tracking systems.
    
    Enhanced to support detailed untracked commit analysis including:
    - Commit categorization (maintenance, bug fix, refactor, docs, etc.)
    - Configurable file change thresholds
    - Extended untracked commit metadata collection
    """

    def __init__(self, allowed_platforms: Optional[list[str]] = None, 
                 untracked_file_threshold: int = 1) -> None:
        """Initialize with patterns for different platforms.

        Args:
            allowed_platforms: List of platforms to extract tickets from.
                              If None, all platforms are allowed.
            untracked_file_threshold: Minimum number of files changed to consider
                                    a commit as 'significant' for untracked analysis.
                                    Default is 1 (all commits), previously was 3.
        """
        self.allowed_platforms = allowed_platforms
        self.untracked_file_threshold = untracked_file_threshold
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
        
        # Commit categorization patterns
        self.category_patterns = {
            "bug_fix": [
                r"\b(fix|bug|error|issue|problem|crash|exception)\b",
                r"\b(resolve|solve|repair|correct)\b",
                r"\b(hotfix|bugfix|patch)\b"
            ],
            "feature": [
                r"\b(add|new|feature|implement|create)\b",
                r"\b(introduce|enhance|extend)\b",
                r"\b(feat|feature):"
            ],
            "refactor": [
                r"\b(refactor|restructure|reorganize|cleanup|clean up)\b",
                r"\b(optimize|improve|simplify)\b",
                r"\b(rename|move|extract)\b"
            ],
            "documentation": [
                r"\b(doc|docs|documentation|readme|comment)\b",
                r"\b(javadoc|jsdoc|docstring)\b",
                r"\b(manual|guide|tutorial)\b"
            ],
            "maintenance": [
                r"\b(update|upgrade|bump|version)\b",
                r"\b(dependency|dependencies|package)\b",
                r"\b(config|configuration|setting)\b",
                r"\b(maintenance|maint|chore)\b"
            ],
            "test": [
                r"\b(test|testing|spec|unit test|integration test)\b",
                r"\b(junit|pytest|mocha|jest)\b",
                r"\b(mock|stub|fixture)\b"
            ],
            "style": [
                r"\b(format|formatting|style|lint|linting)\b",
                r"\b(prettier|eslint|black|autopep8)\b",
                r"\b(whitespace|indentation|spacing)\b"
            ],
            "build": [
                r"\b(build|compile|deploy|deployment)\b",
                r"\b(ci|cd|pipeline|workflow)\b",
                r"\b(docker|dockerfile|makefile)\b"
            ]
        }
        
        # Compile categorization patterns
        self.compiled_category_patterns = {}
        for category, patterns in self.category_patterns.items():
            self.compiled_category_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
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
            # Debug: check if commit is actually a dictionary
            if not isinstance(commit, dict):
                logger.error(f"Expected commit to be dict, got {type(commit)}: {commit}")
                continue
                
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
                # Track untracked commits with configurable threshold and enhanced data
                if not commit.get("is_merge") and commit.get("files_changed", 0) >= self.untracked_file_threshold:
                    # Categorize the commit
                    category = self.categorize_commit(commit.get("message", ""))
                    
                    # Extract enhanced commit data
                    commit_data = {
                        "hash": commit.get("hash", "")[:7],
                        "full_hash": commit.get("hash", ""),
                        "message": commit.get("message", "").split("\n")[0][:100],  # Increased from 60 to 100
                        "full_message": commit.get("message", ""),
                        "author": commit.get("author_name", "Unknown"),
                        "author_email": commit.get("author_email", ""),
                        "canonical_id": commit.get("canonical_id", commit.get("author_email", "")),
                        "timestamp": commit.get("timestamp"),
                        "project_key": commit.get("project_key", "UNKNOWN"),
                        "files_changed": commit.get("files_changed", 0),
                        "lines_added": commit.get("insertions", 0),
                        "lines_removed": commit.get("deletions", 0),
                        "lines_changed": (commit.get("insertions", 0) + commit.get("deletions", 0)),
                        "category": category,
                        "is_merge": commit.get("is_merge", False)
                    }
                    
                    untracked_commits.append(commit_data)

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

        # Sort untracked commits by timestamp (most recent first)
        untracked_commits.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return results

    def categorize_commit(self, message: str) -> str:
        """Categorize a commit based on its message.
        
        WHY: Commit categorization helps identify patterns in untracked work,
        enabling better insights into what types of work are not being tracked
        through tickets. This supports improved process recommendations.
        
        Args:
            message: The commit message to categorize
            
        Returns:
            String category (bug_fix, feature, refactor, documentation, 
            maintenance, test, style, build, or other)
        """
        if not message:
            return "other"
        
        message_lower = message.lower()
        
        # Check each category pattern
        for category, patterns in self.compiled_category_patterns.items():
            for pattern in patterns:
                if pattern.search(message_lower):
                    return category
        
        return "other"
    
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
                "recommendations": []
            }
        
        # Category analysis
        categories = {}
        for commit in untracked_commits:
            category = commit.get("category", "other")
            if category not in categories:
                categories[category] = {"count": 0, "lines_changed": 0, "examples": []}
            categories[category]["count"] += 1
            categories[category]["lines_changed"] += commit.get("lines_changed", 0)
            if len(categories[category]["examples"]) < 3:
                categories[category]["examples"].append({
                    "hash": commit.get("hash", ""),
                    "message": commit.get("message", ""),
                    "author": commit.get("author", "")
                })
        
        # Contributor analysis
        contributors = {}
        for commit in untracked_commits:
            author = commit.get("canonical_id", commit.get("author_email", "Unknown"))
            if author not in contributors:
                contributors[author] = {"count": 0, "categories": set()}
            contributors[author]["count"] += 1
            contributors[author]["categories"].add(commit.get("category", "other"))
        
        # Convert sets to lists for JSON serialization
        for author_data in contributors.values():
            author_data["categories"] = list(author_data["categories"])
        
        # Top contributors
        top_contributors = sorted(
            [(author, data["count"]) for author, data in contributors.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        # Project analysis
        projects = {}
        for commit in untracked_commits:
            project = commit.get("project_key", "UNKNOWN")
            if project not in projects:
                projects[project] = {"count": 0, "categories": set()}
            projects[project]["count"] += 1
            projects[project]["categories"].add(commit.get("category", "other"))
        
        # Convert sets to lists for JSON serialization
        for project_data in projects.values():
            project_data["categories"] = list(project_data["categories"])
        
        # Calculate average commit size
        total_lines = sum(commit.get("lines_changed", 0) for commit in untracked_commits)
        avg_commit_size = total_lines / len(untracked_commits) if untracked_commits else 0
        
        # Generate recommendations
        recommendations = self._generate_untracked_recommendations(
            categories, contributors, projects, len(untracked_commits)
        )
        
        return {
            "total_untracked": len(untracked_commits),
            "categories": categories,
            "top_contributors": top_contributors,
            "projects": projects,
            "avg_commit_size": round(avg_commit_size, 1),
            "recommendations": recommendations
        }
    
    def _generate_untracked_recommendations(
        self, 
        categories: dict[str, Any], 
        contributors: dict[str, Any],
        projects: dict[str, Any],
        total_untracked: int
    ) -> list[dict[str, str]]:
        """Generate recommendations based on untracked commit patterns."""
        recommendations = []
        
        # Category-based recommendations
        if categories.get("feature", {}).get("count", 0) > total_untracked * 0.2:
            recommendations.append({
                "type": "process",
                "title": "Track Feature Development",
                "description": "Many feature commits lack ticket references. Consider requiring tickets for new features.",
                "priority": "high"
            })
        
        if categories.get("bug_fix", {}).get("count", 0) > total_untracked * 0.15:
            recommendations.append({
                "type": "process",
                "title": "Improve Bug Tracking",
                "description": "Bug fixes should be tracked through issue management systems.",
                "priority": "high"
            })
        
        # Allow certain categories to be untracked
        low_priority_categories = ["style", "documentation", "maintenance"]
        low_priority_count = sum(
            categories.get(cat, {}).get("count", 0) for cat in low_priority_categories
        )
        
        if low_priority_count > total_untracked * 0.6:
            recommendations.append({
                "type": "positive",
                "title": "Appropriate Untracked Work",
                "description": "Most untracked commits are maintenance/style/docs - this is acceptable.",
                "priority": "low"
            })
        
        # Contributor-based recommendations
        if len(contributors) > 1:
            max_contributor_count = max(data["count"] for data in contributors.values())
            if max_contributor_count > total_untracked * 0.5:
                recommendations.append({
                    "type": "team",
                    "title": "Provide Process Training",
                    "description": "Some developers need guidance on ticket referencing practices.",
                    "priority": "medium"
                })
        
        return recommendations
    
    def _format_ticket_id(self, platform: str, ticket_id: str) -> str:
        """Format ticket ID for display."""
        if platform == "github":
            return f"#{ticket_id}"
        elif platform == "clickup":
            return f"CU-{ticket_id}" if not ticket_id.startswith("CU-") else ticket_id
        else:
            return ticket_id
