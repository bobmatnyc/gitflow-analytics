"""Branch health metrics for project health assessment.

Based on 2025 software engineering best practices, this module analyzes
branch patterns to assess project health, integration practices, and
development workflow efficiency.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import git
from git import Repo

logger = logging.getLogger(__name__)


class BranchHealthAnalyzer:
    """Analyze branch patterns and health metrics for repositories."""

    # Default thresholds based on 2025 best practices
    DEFAULT_STALE_BRANCH_DAYS = 30
    DEFAULT_HEALTHY_BRANCH_COUNT = 10
    DEFAULT_LONG_LIVED_BRANCH_DAYS = 14
    DEFAULT_IDEAL_PR_SIZE_LINES = 200

    def __init__(
        self,
        stale_branch_days: int = DEFAULT_STALE_BRANCH_DAYS,
        healthy_branch_count: int = DEFAULT_HEALTHY_BRANCH_COUNT,
        long_lived_branch_days: int = DEFAULT_LONG_LIVED_BRANCH_DAYS,
    ):
        """Initialize branch health analyzer with configurable thresholds."""
        self.stale_branch_days = stale_branch_days
        self.healthy_branch_count = healthy_branch_count
        self.long_lived_branch_days = long_lived_branch_days

    def analyze_repository_branches(self, repo_path: str) -> dict[str, Any]:
        """Analyze all branches in a repository for health metrics.

        Args:
            repo_path: Path to the git repository

        Returns:
            Dictionary containing comprehensive branch health metrics
        """
        try:
            repo = Repo(repo_path)
        except Exception as e:
            logger.error(f"Failed to open repository at {repo_path}: {e}")
            return self._empty_metrics()

        now = datetime.now(timezone.utc)
        metrics = {
            "analysis_timestamp": now.isoformat(),
            "repository_path": repo_path,
            "branches": {},
            "summary": {},
            "health_indicators": {},
            "recommendations": [],
        }

        # Identify main/master branch
        main_branch = self._identify_main_branch(repo)
        if not main_branch:
            logger.warning(f"Could not identify main branch for {repo_path}")
            return metrics

        metrics["main_branch"] = main_branch.name

        # Analyze all branches
        all_branches = list(repo.heads)
        remote_branches = [ref for ref in repo.refs if ref.name.startswith("origin/")]

        # Track branch categories
        active_branches = []
        stale_branches = []
        long_lived_branches = []

        for branch in all_branches:
            branch_data = self._analyze_branch(repo, branch, main_branch, now)
            metrics["branches"][branch.name] = branch_data

            # Categorize branches
            if branch_data["is_stale"]:
                stale_branches.append(branch.name)
            elif branch_data["age_days"] > self.long_lived_branch_days:
                long_lived_branches.append(branch.name)
            else:
                active_branches.append(branch.name)

        # Calculate summary metrics
        metrics["summary"] = {
            "total_branches": len(all_branches),
            "active_branches": len(active_branches),
            "stale_branches": len(stale_branches),
            "long_lived_branches": len(long_lived_branches),
            "remote_branches": len(remote_branches),
            "branch_creation_rate_per_week": self._calculate_creation_rate(repo, all_branches),
            "average_branch_age_days": self._calculate_average_age(metrics["branches"]),
            "average_commits_per_branch": self._calculate_average_commits(metrics["branches"]),
        }

        # Calculate health indicators
        metrics["health_indicators"] = self._calculate_health_indicators(metrics["summary"])

        # Generate recommendations
        metrics["recommendations"] = self._generate_recommendations(metrics)

        return metrics

    def _identify_main_branch(self, repo: Repo) -> Optional[git.Head]:
        """Identify the main/master branch of the repository."""
        # Common main branch names
        main_branch_names = ["main", "master", "develop", "trunk"]

        for name in main_branch_names:
            try:
                return repo.heads[name]
            except IndexError:
                continue

        # If no standard names, use the branch with most commits
        if repo.heads:
            return max(repo.heads, key=lambda b: len(list(repo.iter_commits(b))))

        return None

    def _analyze_branch(
        self, repo: Repo, branch: git.Head, main_branch: git.Head, now: datetime
    ) -> dict[str, Any]:
        """Analyze a single branch for health metrics."""
        try:
            # Get branch commits
            branch_commits = list(repo.iter_commits(branch))
            if not branch_commits:
                return self._empty_branch_metrics(branch.name)

            # Get latest commit info
            latest_commit = branch_commits[0]
            latest_activity = latest_commit.committed_datetime
            if latest_activity.tzinfo is None:
                latest_activity = latest_activity.replace(tzinfo=timezone.utc)

            # Calculate age
            age_delta = now - latest_activity
            age_days = age_delta.days

            # Check if branch is merged
            is_merged = self._is_branch_merged(repo, branch, main_branch)

            # Calculate divergence from main
            ahead, behind = self._calculate_divergence(repo, branch, main_branch)

            # Analyze commit patterns
            commit_frequency = self._analyze_commit_frequency(branch_commits)

            return {
                "name": branch.name,
                "latest_activity": latest_activity.isoformat(),
                "age_days": age_days,
                "is_stale": age_days > self.stale_branch_days,
                "is_merged": is_merged,
                "total_commits": len(branch_commits),
                "unique_authors": len(set(c.author.email for c in branch_commits if c.author)),
                "ahead_of_main": ahead,
                "behind_main": behind,
                "divergence_score": ahead + behind,
                "commit_frequency": commit_frequency,
                "health_score": self._calculate_branch_health_score(
                    age_days, ahead, behind, is_merged, len(branch_commits)
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing branch {branch.name}: {e}")
            return self._empty_branch_metrics(branch.name)

    def _is_branch_merged(self, repo: Repo, branch: git.Head, main_branch: git.Head) -> bool:
        """Check if a branch has been merged into main."""
        try:
            # Get merge base
            merge_base = repo.merge_base(branch, main_branch)
            if not merge_base:
                return False

            # If branch tip is in main's history, it's merged
            branch_tip = branch.commit
            # Use commit hashes instead of commit objects for hashability
            main_commit_hashes = set(commit.hexsha for commit in repo.iter_commits(main_branch))
            return branch_tip.hexsha in main_commit_hashes

        except Exception:
            return False

    def _calculate_divergence(
        self, repo: Repo, branch: git.Head, main_branch: git.Head
    ) -> tuple[int, int]:
        """Calculate how many commits a branch is ahead/behind main."""
        try:
            # Get commits ahead (in branch but not in main)
            ahead = list(repo.iter_commits(f"{main_branch.name}..{branch.name}"))

            # Get commits behind (in main but not in branch)
            behind = list(repo.iter_commits(f"{branch.name}..{main_branch.name}"))

            return len(ahead), len(behind)

        except Exception as e:
            logger.error(f"Error calculating divergence: {e}")
            return 0, 0

    def _analyze_commit_frequency(self, commits: list[git.Commit]) -> dict[str, Any]:
        """Analyze commit frequency patterns."""
        if not commits:
            return {"daily_average": 0, "weekly_average": 0}

        # Sort commits by date
        sorted_commits = sorted(commits, key=lambda c: c.committed_datetime)

        # Calculate date range
        first_date = sorted_commits[0].committed_datetime
        last_date = sorted_commits[-1].committed_datetime

        if first_date.tzinfo is None:
            first_date = first_date.replace(tzinfo=timezone.utc)
        if last_date.tzinfo is None:
            last_date = last_date.replace(tzinfo=timezone.utc)

        duration_days = max((last_date - first_date).days, 1)

        return {
            "daily_average": len(commits) / duration_days,
            "weekly_average": (len(commits) / duration_days) * 7,
            "total_days": duration_days,
        }

    def _calculate_branch_health_score(
        self, age_days: int, ahead: int, behind: int, is_merged: bool, commit_count: int
    ) -> float:
        """Calculate a health score for a branch (0-100)."""
        if is_merged:
            return 100.0  # Merged branches are healthy

        score = 100.0

        # Penalize for age
        if age_days > self.stale_branch_days:
            score -= 40
        elif age_days > self.long_lived_branch_days:
            score -= 20
        elif age_days > 7:
            score -= 10

        # Penalize for divergence
        if behind > 100:
            score -= 30
        elif behind > 50:
            score -= 20
        elif behind > 20:
            score -= 10

        # Penalize for being too far ahead (large PRs)
        if ahead > 50:
            score -= 15
        elif ahead > 20:
            score -= 5

        # Bonus for regular activity
        if commit_count > 1 and age_days < 7:
            score += 10

        return max(0, min(100, score))

    def _calculate_creation_rate(self, repo: Repo, branches: list[git.Head]) -> float:
        """Calculate branch creation rate per week."""
        # This is an approximation based on first commit dates
        creation_dates = []

        for branch in branches:
            try:
                commits = list(repo.iter_commits(branch))
                if commits:
                    first_commit = commits[-1]
                    creation_date = first_commit.committed_datetime
                    if creation_date.tzinfo is None:
                        creation_date = creation_date.replace(tzinfo=timezone.utc)
                    creation_dates.append(creation_date)
            except Exception:
                continue

        if len(creation_dates) < 2:
            return 0.0

        # Calculate rate over the past 4 weeks
        now = datetime.now(timezone.utc)
        four_weeks_ago = now - timedelta(weeks=4)
        recent_branches = sum(1 for d in creation_dates if d > four_weeks_ago)

        return recent_branches / 4.0

    def _calculate_average_age(self, branches: dict[str, dict[str, Any]]) -> float:
        """Calculate average age of active branches."""
        active_ages = [
            b["age_days"]
            for b in branches.values()
            if not b.get("is_merged", False) and b.get("age_days", 0) > 0
        ]

        return sum(active_ages) / len(active_ages) if active_ages else 0.0

    def _calculate_average_commits(self, branches: dict[str, dict[str, Any]]) -> float:
        """Calculate average commits per branch."""
        commit_counts = [b.get("total_commits", 0) for b in branches.values()]
        return sum(commit_counts) / len(commit_counts) if commit_counts else 0.0

    def _calculate_health_indicators(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Calculate overall health indicators based on 2025 best practices."""
        total = summary["total_branches"]
        stale = summary["stale_branches"]
        active = summary["active_branches"]

        # Calculate health percentages
        stale_percentage = (stale / total * 100) if total > 0 else 0

        # Determine health status
        if stale_percentage > 50:
            branch_health = "poor"
        elif stale_percentage > 30:
            branch_health = "fair"
        elif stale_percentage > 15:
            branch_health = "good"
        else:
            branch_health = "excellent"

        # Check branch count health
        if total > self.healthy_branch_count * 2:
            count_health = "poor"
        elif total > self.healthy_branch_count:
            count_health = "fair"
        else:
            count_health = "good"

        return {
            "overall_health": branch_health,
            "branch_count_health": count_health,
            "stale_branch_percentage": round(stale_percentage, 1),
            "active_branch_percentage": round((active / total * 100) if total > 0 else 0, 1),
            "integration_frequency": (
                "daily" if summary.get("branch_creation_rate_per_week", 0) > 7 else "weekly"
            ),
        }

    def _generate_recommendations(self, metrics: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        summary = metrics["summary"]
        health = metrics["health_indicators"]

        # Check stale branches
        if summary["stale_branches"] > 0:
            recommendations.append(
                f"🧹 Clean up {summary['stale_branches']} stale branches "
                f"(inactive for >{self.stale_branch_days} days)"
            )

        # Check branch count
        if summary["total_branches"] > self.healthy_branch_count:
            recommendations.append(
                f"📊 Consider reducing active branches from {summary['total_branches']} "
                f"to under {self.healthy_branch_count} for better focus"
            )

        # Check long-lived branches
        if summary["long_lived_branches"] > 3:
            recommendations.append(
                f"⏱️ Review {summary['long_lived_branches']} long-lived branches - "
                "consider smaller, more frequent integrations"
            )

        # Check for branches far behind main
        behind_branches = [
            name
            for name, data in metrics["branches"].items()
            if data.get("behind_main", 0) > 50 and not data.get("is_merged", False)
        ]
        if behind_branches:
            recommendations.append(
                f"🔄 Update {len(behind_branches)} branches that are >50 commits behind main"
            )

        # Positive feedback
        if health["overall_health"] == "excellent":
            recommendations.append("✅ Excellent branch hygiene! Keep up the good practices")

        return recommendations

    def _empty_metrics(self) -> dict[str, Any]:
        """Return empty metrics structure."""
        return {
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "repository_path": "",
            "branches": {},
            "summary": {},
            "health_indicators": {},
            "recommendations": [],
        }

    def _empty_branch_metrics(self, branch_name: str) -> dict[str, Any]:
        """Return empty metrics for a branch."""
        return {
            "name": branch_name,
            "latest_activity": None,
            "age_days": 0,
            "is_stale": False,
            "is_merged": False,
            "total_commits": 0,
            "unique_authors": 0,
            "ahead_of_main": 0,
            "behind_main": 0,
            "divergence_score": 0,
            "commit_frequency": {"daily_average": 0, "weekly_average": 0},
            "health_score": 0.0,
        }

    def analyze_multiple_repositories(self, repo_paths: list[str]) -> dict[str, dict[str, Any]]:
        """Analyze branch health across multiple repositories.

        Args:
            repo_paths: List of repository paths to analyze

        Returns:
            Dictionary mapping repo paths to their health metrics
        """
        results = {}

        for repo_path in repo_paths:
            logger.info(f"Analyzing branch health for {repo_path}")
            results[repo_path] = self.analyze_repository_branches(repo_path)

        return results

    def generate_aggregate_metrics(
        self, multi_repo_results: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate aggregate metrics across multiple repositories."""
        if not multi_repo_results:
            return {}

        total_branches = 0
        total_stale = 0
        total_active = 0
        all_recommendations = []

        for _repo_path, metrics in multi_repo_results.items():
            summary = metrics.get("summary", {})
            total_branches += summary.get("total_branches", 0)
            total_stale += summary.get("stale_branches", 0)
            total_active += summary.get("active_branches", 0)
            all_recommendations.extend(metrics.get("recommendations", []))

        return {
            "total_repositories": len(multi_repo_results),
            "total_branches_all_repos": total_branches,
            "total_stale_branches": total_stale,
            "total_active_branches": total_active,
            "average_branches_per_repo": (
                total_branches / len(multi_repo_results) if multi_repo_results else 0
            ),
            "aggregate_recommendations": list(set(all_recommendations)),  # Unique recommendations
        }
