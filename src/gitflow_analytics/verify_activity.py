"""Project activity verification tool for GitFlow Analytics.

This module provides functionality to verify day-by-day activity for projects
without pulling code, using GitHub API or local git commands to query metadata.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import click
import git
from github import Github
from github.GithubException import GithubException, RateLimitExceededException
from tabulate import tabulate
from tqdm import tqdm

from .config import Config, ConfigLoader

logger = logging.getLogger(__name__)


class ActivityVerifier:
    """Verify project activity without pulling code."""

    def __init__(self, config: Config, weeks: int = 4, config_dir: Optional[Path] = None):
        """Initialize the activity verifier.

        Args:
            config: Configuration object
            weeks: Number of weeks to analyze
            config_dir: Directory containing the config file (for resolving relative paths)
        """
        self.config = config
        self.weeks = weeks
        self.config_dir = config_dir or Path.cwd()
        self.end_date = datetime.now(timezone.utc).replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
        self.start_date = (self.end_date - timedelta(weeks=weeks)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # Initialize GitHub client if configured
        self.github_client = None
        if config.github and config.github.token:
            self.github_client = Github(config.github.token)

    def verify_all_projects(self) -> dict[str, Any]:
        """Verify activity for all configured projects.

        Returns:
            Dictionary containing activity data for all projects
        """
        results = {
            "period": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
                "weeks": self.weeks,
            },
            "projects": {},
            "daily_matrix": self._initialize_daily_matrix(),
            "summary": {
                "total_commits": 0,
                "total_branches": 0,
                "active_days": set(),
                "inactive_projects": [],
            },
        }

        # Get list of repositories to analyze
        repositories = self._get_repositories()

        if not repositories:
            logger.warning("No repositories found to analyze")
            return results

        # Analyze each repository
        with tqdm(repositories, desc="Analyzing repositories", unit="repo") as pbar:
            for repo_info in pbar:
                pbar.set_description(f"Analyzing {repo_info['name']}")

                try:
                    project_data = self._verify_project_activity(repo_info)
                    results["projects"][repo_info["name"]] = project_data

                    # Update summary statistics
                    results["summary"]["total_commits"] += project_data["total_commits"]
                    results["summary"]["total_branches"] += len(project_data["branches"])

                    # Update daily matrix
                    self._update_daily_matrix(
                        results["daily_matrix"], repo_info["name"], project_data["daily_commits"]
                    )

                    # Track active days
                    for date_str in project_data["daily_commits"]:
                        if project_data["daily_commits"][date_str] > 0:
                            results["summary"]["active_days"].add(date_str)

                    # Check if project is inactive
                    if project_data["total_commits"] == 0:
                        results["summary"]["inactive_projects"].append(repo_info["name"])

                except Exception as e:
                    logger.error(f"Error analyzing {repo_info['name']}: {e}")
                    results["projects"][repo_info["name"]] = {
                        "error": str(e),
                        "total_commits": 0,
                        "branches": [],
                        "daily_commits": {},
                    }

        return results

    def _get_repositories(self) -> list[dict[str, Any]]:
        """Get list of repositories to analyze.

        Returns:
            List of repository information dictionaries
        """
        repositories = []

        # Add explicitly configured repositories
        if self.config.repositories:
            # Handle both list and dict formats
            if isinstance(self.config.repositories, list):
                for repo_config in self.config.repositories:
                    # Handle RepositoryConfig objects
                    if hasattr(repo_config, "name"):
                        repo_name = repo_config.name
                        repo_path = repo_config.path if hasattr(repo_config, "path") else None
                        github_repo = (
                            repo_config.github_repo if hasattr(repo_config, "github_repo") else None
                        )
                    # Handle dict format
                    else:
                        repo_name = repo_config.get("name", "")
                        repo_path = repo_config.get("path", None)
                        github_repo = repo_config.get("github_repo", None)

                    # Resolve relative paths relative to config directory
                    if repo_path:
                        path = Path(repo_path)
                        if not path.is_absolute():
                            path = self.config_dir / path
                        path = path.resolve()
                    else:
                        path = None

                    repo_info = {
                        "name": repo_name,
                        "path": path,
                        "is_local": True,
                        "github_name": github_repo,
                    }

                    # Check if it's a GitHub repo
                    if github_repo:
                        repo_info["github_name"] = github_repo
                        repo_info["is_local"] = bool(path)  # Could be both local and GitHub
                    elif self.github_client and "/" in repo_name:
                        repo_info["github_name"] = repo_name
                        repo_info["is_local"] = bool(path)

                    repositories.append(repo_info)
            elif isinstance(self.config.repositories, dict):
                for repo_key, repo_config in self.config.repositories.items():
                    # Handle RepositoryConfig objects
                    if hasattr(repo_config, "path"):
                        repo_path = repo_config.path
                        github_repo = (
                            repo_config.github_repo if hasattr(repo_config, "github_repo") else None
                        )
                    else:
                        repo_path = None
                        github_repo = None

                    # Resolve relative paths relative to config directory
                    if repo_path:
                        path = Path(repo_path)
                        if not path.is_absolute():
                            path = self.config_dir / path
                        path = path.resolve()
                    else:
                        path = None

                    repo_info = {
                        "name": repo_key,
                        "path": path,
                        "is_local": True,
                        "github_name": github_repo,
                    }

                    # Check if it's a GitHub repo
                    if github_repo:
                        repo_info["github_name"] = github_repo
                        repo_info["is_local"] = bool(path)
                    elif self.github_client and "/" in repo_key:
                        repo_info["github_name"] = repo_key
                        repo_info["is_local"] = bool(path)

                    repositories.append(repo_info)

        # Add GitHub organization repositories if configured
        if self.config.github and self.config.github.organization and self.github_client:
            try:
                org = self.github_client.get_organization(self.config.github.organization)
                for repo in org.get_repos(type="all"):
                    # Check if not archived and not already added
                    if not repo.archived and not any(
                        r["name"] == repo.full_name for r in repositories
                    ):
                        repositories.append(
                            {
                                "name": repo.full_name,
                                "path": None,
                                "is_local": False,
                                "github_name": repo.full_name,
                            }
                        )
            except GithubException as e:
                logger.error(f"Error fetching organization repos: {e}")

        return repositories

    def _verify_project_activity(self, repo_info: dict[str, Any]) -> dict[str, Any]:
        """Verify activity for a single project.

        Args:
            repo_info: Repository information dictionary

        Returns:
            Dictionary containing project activity data
        """
        if repo_info.get("github_name") and self.github_client:
            return self._verify_github_activity(repo_info["github_name"])
        elif repo_info.get("path"):
            return self._verify_local_activity(repo_info["path"])
        else:
            raise ValueError(f"No valid path or GitHub name for repository {repo_info['name']}")

    def _verify_github_activity(self, repo_name: str) -> dict[str, Any]:
        """Verify activity for a GitHub repository using API.

        Args:
            repo_name: Full repository name (owner/repo)

        Returns:
            Dictionary containing activity data
        """
        result = {
            "total_commits": 0,
            "branches": [],
            "daily_commits": defaultdict(int),
            "last_activity": None,
        }

        try:
            repo = self.github_client.get_repo(repo_name)

            # Get branches with their last activity
            branches_data = []
            for branch in repo.get_branches():
                try:
                    commit = branch.commit
                    commit_date = commit.commit.author.date.replace(tzinfo=timezone.utc)
                    branches_data.append(
                        {
                            "name": branch.name,
                            "last_activity": commit_date.isoformat(),
                            "sha": commit.sha[:8],
                        }
                    )

                    # Update last activity
                    if not result["last_activity"] or commit_date > datetime.fromisoformat(
                        result["last_activity"]
                    ):
                        result["last_activity"] = commit_date.isoformat()
                except Exception as e:
                    logger.debug(f"Error processing branch {branch.name}: {e}")

            result["branches"] = sorted(
                branches_data, key=lambda x: x["last_activity"], reverse=True
            )

            # Get commits in the date range
            # We'll fetch commits from all branches to get complete activity
            seen_shas = set()

            for branch in repo.get_branches():
                try:
                    commits = repo.get_commits(
                        sha=branch.name, since=self.start_date, until=self.end_date
                    )

                    for commit in commits:
                        if commit.sha not in seen_shas:
                            seen_shas.add(commit.sha)
                            commit_date = commit.commit.author.date.replace(tzinfo=timezone.utc)

                            # Only count commits within our date range
                            if self.start_date <= commit_date <= self.end_date:
                                date_str = commit_date.strftime("%Y-%m-%d")
                                result["daily_commits"][date_str] += 1
                                result["total_commits"] += 1

                except RateLimitExceededException:
                    logger.warning(f"Rate limit reached while fetching commits for {repo_name}")
                    break
                except Exception as e:
                    logger.debug(f"Error fetching commits from branch {branch.name}: {e}")

            # Ensure all dates are present in daily_commits
            current_date = self.start_date
            while current_date <= self.end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                if date_str not in result["daily_commits"]:
                    result["daily_commits"][date_str] = 0
                current_date += timedelta(days=1)

        except Exception as e:
            logger.error(f"Error verifying GitHub activity for {repo_name}: {e}")
            raise

        return result

    def _verify_local_activity(self, repo_path: Path) -> dict[str, Any]:
        """Verify activity for a local Git repository.

        Args:
            repo_path: Path to the local repository

        Returns:
            Dictionary containing activity data
        """
        result = {
            "total_commits": 0,
            "branches": [],
            "daily_commits": defaultdict(int),
            "last_activity": None,
        }

        try:
            repo = git.Repo(repo_path)

            # Get all branches (local and remote)
            branches_data = []

            # Local branches
            for branch in repo.heads:
                try:
                    commit = branch.commit
                    commit_date = datetime.fromtimestamp(commit.committed_date, tz=timezone.utc)
                    branches_data.append(
                        {
                            "name": branch.name,
                            "last_activity": commit_date.isoformat(),
                            "sha": commit.hexsha[:8],
                            "type": "local",
                        }
                    )

                    # Update last activity
                    if not result["last_activity"] or commit_date > datetime.fromisoformat(
                        result["last_activity"]
                    ):
                        result["last_activity"] = commit_date.isoformat()
                except Exception as e:
                    logger.debug(f"Error processing branch {branch.name}: {e}")

            # Remote branches
            for remote in repo.remotes:
                try:
                    remote.fetch(prune=True, dry_run=True)  # Update remote refs without pulling
                    for ref in remote.refs:
                        if not ref.name.endswith("/HEAD"):
                            try:
                                commit = ref.commit
                                commit_date = datetime.fromtimestamp(
                                    commit.committed_date, tz=timezone.utc
                                )
                                branches_data.append(
                                    {
                                        "name": ref.name,
                                        "last_activity": commit_date.isoformat(),
                                        "sha": commit.hexsha[:8],
                                        "type": "remote",
                                    }
                                )
                            except Exception as e:
                                logger.debug(f"Error processing remote branch {ref.name}: {e}")
                except Exception as e:
                    logger.debug(f"Error fetching remote {remote.name}: {e}")

            result["branches"] = sorted(
                branches_data, key=lambda x: x["last_activity"], reverse=True
            )

            # Get commits in the date range from all branches
            seen_shas = set()

            # Analyze all branches
            all_refs = list(repo.heads) + [
                ref
                for remote in repo.remotes
                for ref in remote.refs
                if not ref.name.endswith("/HEAD")
            ]

            for ref in all_refs:
                try:
                    # Use git log with date filtering
                    commits = list(
                        repo.iter_commits(
                            ref.name,
                            since=self.start_date.strftime("%Y-%m-%d"),
                            until=self.end_date.strftime("%Y-%m-%d"),
                        )
                    )

                    for commit in commits:
                        if commit.hexsha not in seen_shas:
                            seen_shas.add(commit.hexsha)
                            commit_date = datetime.fromtimestamp(
                                commit.committed_date, tz=timezone.utc
                            )

                            # Only count commits within our date range
                            if self.start_date <= commit_date <= self.end_date:
                                date_str = commit_date.strftime("%Y-%m-%d")
                                result["daily_commits"][date_str] += 1
                                result["total_commits"] += 1

                except Exception as e:
                    logger.debug(f"Error processing commits from {ref.name}: {e}")

            # Ensure all dates are present in daily_commits
            current_date = self.start_date
            while current_date <= self.end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                if date_str not in result["daily_commits"]:
                    result["daily_commits"][date_str] = 0
                current_date += timedelta(days=1)

        except Exception as e:
            logger.error(f"Error verifying local activity for {repo_path}: {e}")
            raise

        return result

    def _initialize_daily_matrix(self) -> dict[str, dict[str, int]]:
        """Initialize the daily activity matrix structure.

        Returns:
            Dictionary with dates as keys and empty project dictionaries
        """
        matrix = {}
        current_date = self.start_date

        while current_date <= self.end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            matrix[date_str] = {}
            current_date += timedelta(days=1)

        return matrix

    def _update_daily_matrix(
        self, matrix: dict[str, dict[str, int]], project_name: str, daily_commits: dict[str, int]
    ) -> None:
        """Update the daily matrix with project commit data.

        Args:
            matrix: Daily matrix to update
            project_name: Name of the project
            daily_commits: Daily commit counts for the project
        """
        for date_str, count in daily_commits.items():
            if date_str in matrix:
                matrix[date_str][project_name] = count

    def format_report(self, results: dict[str, Any]) -> str:
        """Format the verification results as a readable report.

        Args:
            results: Verification results dictionary

        Returns:
            Formatted report string
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("Activity Verification Report")
        lines.append("=" * 80)
        lines.append(
            f"Period: {results['period']['start'][:10]} to {results['period']['end'][:10]}"
        )
        lines.append(f"Analysis duration: {results['period']['weeks']} weeks")
        lines.append("")

        # Summary statistics
        lines.append("Summary Statistics:")
        lines.append("-" * 40)
        lines.append(f"Total commits: {results['summary']['total_commits']}")
        lines.append(f"Total branches: {results['summary']['total_branches']}")
        lines.append(f"Active days: {len(results['summary']['active_days'])}")
        lines.append(f"Inactive projects: {len(results['summary']['inactive_projects'])}")

        if results["summary"]["inactive_projects"]:
            lines.append(
                f"  Projects with no activity: {', '.join(results['summary']['inactive_projects'])}"
            )
        lines.append("")

        # Daily Activity Matrix
        lines.append("Daily Activity Matrix:")
        lines.append("-" * 40)

        # Prepare matrix data for tabulation
        matrix_data = []
        dates = sorted(results["daily_matrix"].keys())
        projects = sorted(
            set(project for date_data in results["daily_matrix"].values() for project in date_data)
        )

        if projects and dates:
            # Create condensed view - show week summaries
            week_data = defaultdict(lambda: defaultdict(int))
            week_starts = []

            for date_str in dates:
                date = datetime.strptime(date_str, "%Y-%m-%d")
                week_start = date - timedelta(days=date.weekday())
                week_key = week_start.strftime("%m/%d")

                if week_key not in week_starts:
                    week_starts.append(week_key)

                for project in projects:
                    count = results["daily_matrix"][date_str].get(project, 0)
                    week_data[project][week_key] += count

            # Build table rows
            headers = ["Project"] + week_starts + ["Total"]

            for project in projects:
                row = [project[:20]]  # Truncate long project names
                total = 0
                for week in week_starts:
                    count = week_data[project].get(week, 0)
                    total += count
                    # Use symbols for readability
                    if count == 0:
                        row.append("-")
                    elif count < 10:
                        row.append(str(count))
                    else:
                        row.append(f"{count}+")
                row.append(str(total))
                matrix_data.append(row)

            # Add summary row
            summary_row = ["TOTAL"]
            grand_total = 0
            for week in week_starts:
                week_total = sum(week_data[p].get(week, 0) for p in projects)
                grand_total += week_total
                if week_total == 0:
                    summary_row.append("-")
                else:
                    summary_row.append(str(week_total))
            summary_row.append(str(grand_total))
            matrix_data.append(summary_row)

            lines.append(tabulate(matrix_data, headers=headers, tablefmt="grid"))
        else:
            lines.append("No activity data available")

        lines.append("")

        # Branch Summary for each project
        lines.append("Branch Summary by Project:")
        lines.append("-" * 40)

        for project_name, project_data in sorted(results["projects"].items()):
            lines.append(f"\n{project_name}:")

            if "error" in project_data:
                lines.append(f"  ERROR: {project_data['error']}")
                continue

            lines.append(f"  Total commits: {project_data['total_commits']}")

            if project_data.get("last_activity"):
                lines.append(f"  Last activity: {project_data['last_activity'][:10]}")

            if project_data.get("branches"):
                lines.append(f"  Branches ({len(project_data['branches'])}):")
                # Show top 5 most recently active branches
                for branch in project_data["branches"][:5]:
                    branch_type = f" [{branch.get('type', 'unknown')}]" if "type" in branch else ""
                    lines.append(
                        f"    - {branch['name']}{branch_type}: "
                        f"last activity {branch['last_activity'][:10]} "
                        f"({branch['sha']})"
                    )
                if len(project_data["branches"]) > 5:
                    lines.append(f"    ... and {len(project_data['branches']) - 5} more branches")
            else:
                lines.append("  No branches found")

        lines.append("")

        # Days with zero activity
        lines.append("Days with Zero Activity:")
        lines.append("-" * 40)

        zero_activity_days = []
        for date_str in sorted(results["daily_matrix"].keys()):
            total_commits = (
                sum(results["daily_matrix"][date_str].get(p, 0) for p in projects)
                if projects
                else 0
            )

            if total_commits == 0:
                date = datetime.strptime(date_str, "%Y-%m-%d")
                zero_activity_days.append(date.strftime("%a %Y-%m-%d"))

        if zero_activity_days:
            # Group consecutive days
            lines.append(f"Found {len(zero_activity_days)} days with no activity:")
            for i in range(0, len(zero_activity_days), 7):
                lines.append(f"  {', '.join(zero_activity_days[i : i + 7])}")
        else:
            lines.append("No days with zero activity found!")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)


def verify_activity_command(
    config_path: Path, weeks: int, output_path: Optional[Path] = None
) -> None:
    """Run the activity verification command.

    Args:
        config_path: Path to configuration file
        weeks: Number of weeks to analyze
        output_path: Optional path to save the report
    """
    # Load configuration
    click.echo(f"Loading configuration from {config_path}...")
    config = ConfigLoader.load(config_path)

    # Store config directory for resolving relative paths
    config_dir = config_path.parent

    # Create verifier
    verifier = ActivityVerifier(config, weeks, config_dir)

    # Run verification
    click.echo(f"Verifying activity for the last {weeks} weeks...")
    results = verifier.verify_all_projects()

    # Format and display report
    report = verifier.format_report(results)

    # Output to console
    click.echo(report)

    # Save to file if requested
    if output_path:
        output_path.write_text(report)
        click.echo(f"\nReport saved to: {output_path}")

    # Highlight any issues found
    if results["summary"]["inactive_projects"]:
        click.echo("\n⚠️  WARNING: Found projects with no activity!")
        for project in results["summary"]["inactive_projects"]:
            click.echo(f"   - {project}")

    zero_days = len(
        [d for d in results["daily_matrix"] if sum(results["daily_matrix"][d].values()) == 0]
    )
    if zero_days > 0:
        click.echo(f"\n⚠️  WARNING: Found {zero_days} days with zero activity across all projects!")
