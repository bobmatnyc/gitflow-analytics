"""Story point extraction from commits and pull requests."""

import re
from typing import Any, Optional


class StoryPointExtractor:
    """Extract story points from text using configurable patterns."""

    def __init__(self, patterns: Optional[list[str]] = None):
        """Initialize with extraction patterns."""
        if patterns is None:
            patterns = [
                r"(?:story\s*points?|sp|pts?)\s*[:=]\s*(\d+)",  # SP: 5, story points = 3
                r"\[(\d+)\s*(?:sp|pts?)\]",  # [3sp], [5 pts]
                r"#(\d+)sp",  # #3sp
                r"estimate:\s*(\d+)",  # estimate: 5
                r"\bSP(\d+)\b",  # SP5, SP13
                r"points?:\s*(\d+)",  # points: 8
            ]

        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def extract_from_text(self, text: str) -> Optional[int]:
        """Extract story points from text."""
        if not text:
            return None

        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                try:
                    points = int(match.group(1))
                    # Sanity check - story points should be reasonable
                    if 0 < points <= 100:
                        return points
                except (ValueError, IndexError):
                    continue

        return None

    def extract_from_pr(
        self, pr_data: dict[str, Any], commit_messages: Optional[list[str]] = None
    ) -> Optional[int]:
        """Extract story points from PR with fallback to commits."""
        # Try PR description first (most authoritative)
        points = self.extract_from_text(pr_data.get("description", ""))
        if points:
            return points

        # Try PR title
        points = self.extract_from_text(pr_data.get("title", ""))
        if points:
            return points

        # Try PR body (if different from description)
        if "body" in pr_data:
            points = self.extract_from_text(pr_data["body"])
            if points:
                return points

        # Fallback to commit messages
        if commit_messages:
            commit_points = []
            for message in commit_messages:
                points = self.extract_from_text(message)
                if points:
                    commit_points.append(points)

            if commit_points:
                # Use the most common value or max if no consensus
                from collections import Counter

                point_counts = Counter(commit_points)
                most_common = point_counts.most_common(1)
                if most_common:
                    return most_common[0][0]

        return None

    def aggregate_story_points(
        self, prs: list[dict[str, Any]], commits: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Aggregate story points from PRs and commits."""
        # Map commits to PRs
        pr_by_commit = {}
        for pr in prs:
            for commit_hash in pr.get("commit_hashes", []):
                pr_by_commit[commit_hash] = pr

        # Track which commits are associated with PRs
        pr_commits = set(pr_by_commit.keys())

        # Aggregate results
        orphan_commits: list[dict[str, Any]] = []
        unestimated_prs: list[dict[str, Any]] = []

        results = {
            "total_story_points": 0,
            "pr_story_points": 0,
            "commit_story_points": 0,
            "orphan_commits": orphan_commits,  # Commits without PRs
            "unestimated_prs": unestimated_prs,  # PRs without story points
        }

        # Process PRs
        for pr in prs:
            pr_points = pr.get("story_points", 0)
            if pr_points:
                results["pr_story_points"] += pr_points
                results["total_story_points"] += pr_points
            else:
                unestimated_prs.append(
                    {"number": pr.get("number", 0), "title": pr.get("title", "")}
                )

        # Process commits not in PRs
        for commit in commits:
            commit_hash = commit.get("hash", "")
            if commit_hash not in pr_commits:
                commit_points = commit.get("story_points", 0)
                if commit_points:
                    results["commit_story_points"] += commit_points
                    results["total_story_points"] += commit_points

                # Track significant orphan commits
                files_changed = commit.get(
                    "files_changed_count",
                    (
                        commit.get("files_changed", 0)
                        if isinstance(commit.get("files_changed"), int)
                        else len(commit.get("files_changed", []))
                    ),
                )
                insertions = commit.get("insertions", 0)
                if files_changed > 5 or insertions > 100:
                    orphan_commits.append(
                        {
                            "hash": commit.get("hash", "")[:7],
                            "message": commit.get("message", "").split("\n")[0][:80],
                            "story_points": commit_points,
                            "files_changed": files_changed,
                        }
                    )

        return results
