"""Git repository analyzer with batch processing support."""

import fnmatch
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import git
from git import Repo
from tqdm import tqdm

from ..extractors.story_points import StoryPointExtractor
from ..extractors.tickets import TicketExtractor
from .branch_mapper import BranchToProjectMapper
from .cache import GitAnalysisCache


class GitAnalyzer:
    """Analyze Git repositories with caching and batch processing."""

    def __init__(
        self,
        cache: GitAnalysisCache,
        batch_size: int = 1000,
        branch_mapping_rules: Optional[dict[str, list[str]]] = None,
        allowed_ticket_platforms: Optional[list[str]] = None,
        exclude_paths: Optional[list[str]] = None,
    ):
        """Initialize analyzer with cache."""
        self.cache = cache
        self.batch_size = batch_size
        self.story_point_extractor = StoryPointExtractor()
        self.ticket_extractor = TicketExtractor(allowed_platforms=allowed_ticket_platforms)
        self.branch_mapper = BranchToProjectMapper(branch_mapping_rules)
        self.exclude_paths = exclude_paths or []

    def analyze_repository(
        self, repo_path: Path, since: datetime, branch: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Analyze a Git repository with batch processing."""
        try:
            repo = Repo(repo_path)
        except Exception as e:
            raise ValueError(f"Failed to open repository at {repo_path}: {e}") from e

        # Get commits to analyze
        commits = self._get_commits(repo, since, branch)
        total_commits = len(commits)

        if total_commits == 0:
            return []

        analyzed_commits = []

        # Process in batches with progress bar
        with tqdm(total=total_commits, desc=f"Analyzing {repo_path.name}") as pbar:
            for batch in self._batch_commits(commits, self.batch_size):
                batch_results = self._process_batch(repo, repo_path, batch)
                analyzed_commits.extend(batch_results)

                # Cache the batch
                self.cache.cache_commits_batch(str(repo_path), batch_results)

                pbar.update(len(batch))

        return analyzed_commits

    def _get_commits(
        self, repo: Repo, since: datetime, branch: Optional[str] = None
    ) -> list[git.Commit]:
        """Get commits from repository."""
        if branch:
            try:
                commits = list(repo.iter_commits(branch, since=since))
            except git.GitCommandError:
                # Branch doesn't exist
                return []
        else:
            # Get commits from all branches
            commits = []
            for ref in repo.refs:
                if ref.name.startswith("origin/"):
                    continue  # Skip remote branches
                try:
                    branch_commits = list(repo.iter_commits(ref, since=since))
                    commits.extend(branch_commits)
                except git.GitCommandError:
                    continue

            # Remove duplicates while preserving order
            seen = set()
            unique_commits = []
            for commit in commits:
                if commit.hexsha not in seen:
                    seen.add(commit.hexsha)
                    unique_commits.append(commit)

            commits = unique_commits

        # Sort by date
        return sorted(commits, key=lambda c: c.committed_datetime)

    def _batch_commits(
        self, commits: list[git.Commit], batch_size: int
    ) -> Generator[list[git.Commit], None, None]:
        """Yield batches of commits."""
        for i in range(0, len(commits), batch_size):
            yield commits[i : i + batch_size]

    def _process_batch(
        self, repo: Repo, repo_path: Path, commits: list[git.Commit]
    ) -> list[dict[str, Any]]:
        """Process a batch of commits."""
        results = []

        for commit in commits:
            # Check cache first
            cached = self.cache.get_cached_commit(str(repo_path), commit.hexsha)
            if cached:
                results.append(cached)
                continue

            # Analyze commit
            commit_data = self._analyze_commit(repo, commit, repo_path)
            results.append(commit_data)

        return results

    def _analyze_commit(self, repo: Repo, commit: git.Commit, repo_path: Path) -> dict[str, Any]:
        """Analyze a single commit."""
        # Basic commit data
        commit_data = {
            "hash": commit.hexsha,
            "author_name": commit.author.name,
            "author_email": commit.author.email,
            "message": commit.message,
            "timestamp": commit.committed_datetime,
            "is_merge": len(commit.parents) > 1,
        }

        # Get branch name
        commit_data["branch"] = self._get_commit_branch(repo, commit)

        # Map branch to project
        commit_data["inferred_project"] = self.branch_mapper.map_branch_to_project(
            str(commit_data["branch"]), repo_path
        )

        # Calculate metrics - use raw stats for backward compatibility
        stats = commit.stats.total
        commit_data["files_changed"] = int(stats.get("files", 0)) if hasattr(stats, "get") else 0
        commit_data["insertions"] = int(stats.get("insertions", 0)) if hasattr(stats, "get") else 0
        commit_data["deletions"] = int(stats.get("deletions", 0)) if hasattr(stats, "get") else 0

        # Calculate filtered metrics (excluding boilerplate/generated files)
        filtered_stats = self._calculate_filtered_stats(commit)
        commit_data["filtered_files_changed"] = filtered_stats["files"]
        commit_data["filtered_insertions"] = filtered_stats["insertions"]
        commit_data["filtered_deletions"] = filtered_stats["deletions"]

        # Extract story points
        message_str = (
            commit.message
            if isinstance(commit.message, str)
            else commit.message.decode("utf-8", errors="ignore")
        )
        commit_data["story_points"] = self.story_point_extractor.extract_from_text(message_str)

        # Extract ticket references
        commit_data["ticket_references"] = self.ticket_extractor.extract_from_text(message_str)

        # Calculate complexity delta
        commit_data["complexity_delta"] = self._calculate_complexity_delta(commit)

        return commit_data

    def _get_commit_branch(self, repo: Repo, commit: git.Commit) -> str:
        """Get the branch name for a commit."""
        # This is a simplified approach - getting the first branch that contains the commit
        for branch in repo.branches:
            if commit in repo.iter_commits(branch):
                return branch.name
        return "unknown"

    def _calculate_complexity_delta(self, commit: git.Commit) -> float:
        """Calculate complexity change for a commit."""
        total_delta = 0.0

        for diff in commit.diff(commit.parents[0] if commit.parents else None):
            if not self._is_code_file(diff.b_path or diff.a_path or ""):
                continue

            # Simple complexity estimation based on diff size
            # In a real implementation, you'd parse the code and calculate cyclomatic complexity
            if diff.new_file:
                total_delta += diff.b_blob.size / 100 if diff.b_blob else 0
            elif diff.deleted_file:
                total_delta -= diff.a_blob.size / 100 if diff.a_blob else 0
            else:
                # Modified file - estimate based on change size
                if diff.diff:
                    diff_content = (
                        diff.diff
                        if isinstance(diff.diff, str)
                        else diff.diff.decode("utf-8", errors="ignore")
                    )
                    added = len(diff_content.split("\n+"))
                    removed = len(diff_content.split("\n-"))
                    total_delta += (added - removed) / 10

        return total_delta

    def _is_code_file(self, filepath: str) -> bool:
        """Check if file is a code file."""
        code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".cs",
            ".vb",
            ".r",
            ".m",
            ".mm",
            ".f90",
            ".f95",
            ".lua",
        }

        return any(filepath.endswith(ext) for ext in code_extensions)

    def _should_exclude_file(self, filepath: str) -> bool:
        """Check if file should be excluded from line counting."""
        if not filepath:
            return False

        # Normalize path separators for consistent matching
        filepath = filepath.replace("\\", "/")

        # Check against exclude patterns
        return any(fnmatch.fnmatch(filepath, pattern) for pattern in self.exclude_paths)

    def _calculate_filtered_stats(self, commit: git.Commit) -> dict[str, int]:
        """Calculate commit statistics excluding boilerplate/generated files."""
        filtered_stats = {"files": 0, "insertions": 0, "deletions": 0}

        # For initial commits or commits without parents
        parent = commit.parents[0] if commit.parents else None

        try:
            for diff in commit.diff(parent):
                # Get file path
                file_path = diff.b_path if diff.b_path else diff.a_path
                if not file_path:
                    continue

                # Skip excluded files
                if self._should_exclude_file(file_path):
                    continue

                # Count the file
                filtered_stats["files"] += 1

                # Count insertions and deletions
                if diff.diff:
                    diff_text = (
                        diff.diff
                        if isinstance(diff.diff, str)
                        else diff.diff.decode("utf-8", errors="ignore")
                    )
                    for line in diff_text.split("\n"):
                        if line.startswith("+") and not line.startswith("+++"):
                            filtered_stats["insertions"] += 1
                        elif line.startswith("-") and not line.startswith("---"):
                            filtered_stats["deletions"] += 1
        except Exception:
            # If we can't calculate filtered stats, return zeros
            pass

        return filtered_stats
