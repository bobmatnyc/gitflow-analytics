"""Git repository analyzer with batch processing support."""

import fnmatch
import logging
import re
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

# Import ML extractor with fallback
try:
    from ..extractors.ml_tickets import MLTicketExtractor
    ML_EXTRACTOR_AVAILABLE = True
except ImportError:
    ML_EXTRACTOR_AVAILABLE = False

# Get logger for this module
logger = logging.getLogger(__name__)


class GitAnalyzer:
    """Analyze Git repositories with caching and batch processing."""

    def __init__(
        self,
        cache: GitAnalysisCache,
        batch_size: int = 1000,
        branch_mapping_rules: Optional[dict[str, list[str]]] = None,
        allowed_ticket_platforms: Optional[list[str]] = None,
        exclude_paths: Optional[list[str]] = None,
        ml_categorization_config: Optional[dict[str, Any]] = None,
    ):
        """Initialize analyzer with cache and optional ML categorization.
        
        Args:
            cache: Git analysis cache instance
            batch_size: Number of commits to process in each batch
            branch_mapping_rules: Rules for mapping branches to projects
            allowed_ticket_platforms: List of allowed ticket platforms
            exclude_paths: List of file paths to exclude from analysis
            ml_categorization_config: Configuration for ML-based categorization
        """
        self.cache = cache
        self.batch_size = batch_size
        self.story_point_extractor = StoryPointExtractor()
        
        # Initialize ticket extractor (ML or standard based on config and availability)
        if (ml_categorization_config and 
            ml_categorization_config.get('enabled', True) and 
            ML_EXTRACTOR_AVAILABLE):
            
            logger.info("Initializing ML-enhanced ticket extractor")
            self.ticket_extractor = MLTicketExtractor(
                allowed_platforms=allowed_ticket_platforms,
                ml_config=ml_categorization_config,
                cache_dir=cache.cache_dir / "ml_predictions",
                enable_ml=True
            )
        else:
            if ml_categorization_config and ml_categorization_config.get('enabled', True):
                if not ML_EXTRACTOR_AVAILABLE:
                    logger.warning("ML categorization requested but dependencies not available, using standard extractor")
                else:
                    logger.info("ML categorization disabled in configuration, using standard extractor")
            else:
                logger.debug("Using standard ticket extractor")
            
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
        logger.debug(f"Getting commits since: {since} (tzinfo: {getattr(since, 'tzinfo', 'N/A')})")
        
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
        # Normalize timestamp handling
        commit_timestamp = commit.committed_datetime
        logger.debug(f"Analyzing commit {commit.hexsha[:8]}: original timestamp={commit_timestamp} (tzinfo: {getattr(commit_timestamp, 'tzinfo', 'N/A')})")
        
        # Ensure timezone-aware timestamp in UTC
        from datetime import timezone
        if commit_timestamp.tzinfo is None:
            # Convert naive datetime to UTC
            commit_timestamp = commit_timestamp.replace(tzinfo=timezone.utc)
            logger.debug(f"  Converted naive timestamp to UTC: {commit_timestamp}")
        elif commit_timestamp.tzinfo != timezone.utc:
            # Convert to UTC if in different timezone
            commit_timestamp = commit_timestamp.astimezone(timezone.utc)
            logger.debug(f"  Converted timestamp to UTC: {commit_timestamp}")
        else:
            logger.debug(f"  Timestamp already in UTC: {commit_timestamp}")
        
        # Get the local hour for the developer (before UTC conversion)
        local_hour = commit.committed_datetime.hour
        
        # Basic commit data
        commit_data = {
            "hash": commit.hexsha,
            "author_name": commit.author.name,
            "author_email": commit.author.email,
            "message": commit.message,
            "timestamp": commit_timestamp,  # Now guaranteed to be UTC timezone-aware
            "local_hour": local_hour,  # Hour in developer's local timezone
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
        commit_data["files_changed"] = stats["files"] if isinstance(stats, dict) else stats.files
        commit_data["insertions"] = stats["insertions"] if isinstance(stats, dict) else stats.insertions
        commit_data["deletions"] = stats["deletions"] if isinstance(stats, dict) else stats.deletions

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

        # Check against exclude patterns with proper ** handling
        return any(self._matches_glob_pattern(filepath, pattern) for pattern in self.exclude_paths)
    
    def _matches_glob_pattern(self, filepath: str, pattern: str) -> bool:
        """Check if a file path matches a glob pattern, handling ** recursion correctly.
        
        This method properly handles different glob pattern types:
        - **/vendor/** : matches files inside vendor directories at any level
        - **/*.min.js : matches files with specific suffix anywhere in directory tree
        - vendor/** : matches files inside vendor directory at root level only
        - **pattern** : handles other complex patterns with pathlib.match()
        - simple patterns : uses fnmatch for basic wildcards
        
        Args:
            filepath: The file path to check
            pattern: The glob pattern to match against
            
        Returns:
            True if the file path matches the pattern, False otherwise
        """
        from pathlib import PurePath
        
        # Handle empty or invalid inputs
        if not filepath or not pattern:
            return False
            
        path = PurePath(filepath)
        
        # Check for multiple ** patterns first (most complex)
        if '**' in pattern and pattern.count('**') > 1:
            # Multiple ** patterns - use custom recursive matching for complex patterns
            return self._match_recursive_pattern(filepath, pattern)
            
        # Then handle simple ** patterns
        elif pattern.startswith('**/') and pattern.endswith('/**'):
            # Pattern like **/vendor/** - matches files inside vendor directories at any level
            dir_name = pattern[3:-3]  # Extract 'vendor' from '**/vendor/**'
            if not dir_name:  # Handle edge case of '**/**'
                return True
            return dir_name in path.parts
            
        elif pattern.startswith('**/'):
            # Pattern like **/*.min.js - matches files with specific suffix anywhere
            suffix_pattern = pattern[3:]
            if not suffix_pattern:  # Handle edge case of '**/'
                return True
            # Check against filename for file patterns, or any path part for directory patterns
            if suffix_pattern.endswith('/'):
                # Directory pattern like **/build/
                dir_name = suffix_pattern[:-1]
                return dir_name in path.parts
            else:
                # File pattern like *.min.js
                return fnmatch.fnmatch(path.name, suffix_pattern)
                
        elif pattern.endswith('/**'):
            # Pattern like vendor/** or docs/build/** - matches files inside directory at root level
            dir_name = pattern[:-3]
            if not dir_name:  # Handle edge case of '/**'
                return True
            
            # Handle both single directory names and nested paths
            expected_parts = PurePath(dir_name).parts
            return (len(path.parts) >= len(expected_parts) and 
                    path.parts[:len(expected_parts)] == expected_parts)
            
        elif '**' in pattern:
            # Single ** pattern - use pathlib matching with fallback
            try:
                return path.match(pattern)
            except (ValueError, TypeError):
                # Fall back to fnmatch if pathlib fails (e.g., invalid pattern)
                try:
                    return fnmatch.fnmatch(filepath, pattern)
                except re.error:
                    # Invalid regex pattern - return False to be safe
                    return False
        else:
            # Simple pattern - use fnmatch for basic wildcards
            try:
                return fnmatch.fnmatch(filepath, pattern)
            except re.error:
                # Invalid regex pattern - return False to be safe
                return False
    
    def _match_recursive_pattern(self, filepath: str, pattern: str) -> bool:
        """Handle complex patterns with multiple ** wildcards.
        
        Args:
            filepath: The file path to check
            pattern: The pattern with multiple ** wildcards
            
        Returns:
            True if the path matches the pattern, False otherwise
        """
        from pathlib import PurePath
        
        # Split pattern by ** to handle each segment
        parts = pattern.split('**')
        path = PurePath(filepath)
        path_str = str(path)
        
        # Handle patterns like 'src/**/components/**/*.tsx' or '**/test/**/*.spec.js'
        if len(parts) >= 2:
            # First part should match from the beginning (if not empty)
            start_pattern = parts[0].rstrip('/')
            if start_pattern and not path_str.startswith(start_pattern):
                return False
            
            # Last part should match the filename/end pattern
            end_pattern = parts[-1].lstrip('/')
            if end_pattern:
                # Check if filename matches the end pattern
                if not fnmatch.fnmatch(path.name, end_pattern):
                    return False
            
            # Middle parts should exist somewhere in the path between start and end
            for i in range(1, len(parts) - 1):
                middle_pattern = parts[i].strip('/')
                if middle_pattern:
                    # Check if this directory exists in the path
                    if middle_pattern not in path.parts:
                        return False
            
            return True
        
        return False

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
