"""Commit analysis mixin for GitAnalyzer."""

"""Git repository analyzer with batch processing support."""

import logging
import re
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import git
from git import Repo

from ..types import FilteredCommitStats
from ..utils.commit_utils import extract_co_authors, is_merge_commit
from ..utils.debug import is_debug_mode
from ..utils.glob_matcher import match_recursive_pattern as _match_recursive_pattern_fn
from ..utils.glob_matcher import matches_glob_pattern as _matches_glob_pattern_fn
from ..utils.glob_matcher import should_exclude_file as _should_exclude_file_fn
from .analysis_components import (
    build_branch_mapper,
    build_story_point_extractor,
    build_ticket_extractor,
)
from .cache import GitAnalysisCache
from .progress import get_progress_service

# Get logger for this module
logger = logging.getLogger(__name__)



class CommitAnalyzerMixin:
    """Mixin providing commit analysis helpers for GitAnalyzer."""

    def _analyze_commit(self, repo: Repo, commit: git.Commit, repo_path: Path) -> dict[str, Any]:
        """Analyze a single commit."""
        # Normalize timestamp handling
        commit_timestamp = commit.committed_datetime
        logger.debug(
            f"Analyzing commit {commit.hexsha[:8]}: original timestamp={commit_timestamp} (tzinfo: {getattr(commit_timestamp, 'tzinfo', 'N/A')})"
        )

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

        # Calculate metrics using reliable git numstat for accurate line counts
        raw_stats = self._calculate_raw_stats(commit)
        commit_data["files_changed_count"] = raw_stats[
            "files"
        ]  # Integer count for backward compatibility
        commit_data["files_changed"] = self._get_changed_file_paths(
            commit
        )  # List of file paths for ML
        commit_data["insertions"] = raw_stats["insertions"]
        commit_data["deletions"] = raw_stats["deletions"]

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

        # Gap 4: Extract Co-authored-by trailers for co-author attribution.
        # WHY: GitHub, VS Code, and many tools add "Co-authored-by: Name <email>"
        # trailers when pairs/groups collaborate on a commit.  Without parsing these,
        # the co-author's work is invisible in the analytics.  We store them so the
        # identity resolver can credit each co-author with the commit.
        commit_data["co_authors"] = extract_co_authors(message_str)

        # Calculate complexity delta
        commit_data["complexity_delta"] = self._calculate_complexity_delta(commit)

        return commit_data

    def _get_commit_branch(self, repo: Repo, commit: git.Commit) -> str:
        """Get the branch name for a commit.

        BUG 1 FIX: Previously this materialised the ENTIRE commit history for every
        branch for every single commit lookup — O(commits * branches) memory.  Now we
        return from the pre-built mapping populated in _get_all_branch_commits /
        _get_smart_branch_commits so the lookup is O(1) with no extra allocation.
        """
        # Fast path: use the mapping built during commit collection
        if hasattr(self, "_commit_branch_map"):
            return self._commit_branch_map.get(commit.hexsha, "unknown")

        # Fallback for callers that bypass the normal collection path.
        # Use git name-rev which is a single cheap git command rather than a
        # Python-level iteration over full branch histories.
        try:
            result = repo.git.name_rev(commit.hexsha, "--name-only", "--no-undefined")
            # name-rev may return "branch~N" or "tags/..." — strip the suffix
            name = result.split("~")[0].split("^")[0].strip()
            # Remove remote prefix (e.g. "remotes/origin/main" -> "main")
            for prefix in ("remotes/origin/", "remotes/"):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
            return name or "unknown"
        except Exception:
            return "unknown"

    def _get_changed_file_paths(self, commit: git.Commit) -> list[str]:
        """Extract list of changed file paths from a git commit.

        Args:
            commit: Git commit object

        Returns:
            List of file paths that were changed in the commit
        """
        file_paths = []

        # Handle initial commits (no parents) and regular commits
        parent = commit.parents[0] if commit.parents else None

        try:
            for diff in commit.diff(parent):
                # Get file path - prefer the new path (b_path) for modifications and additions,
                # fall back to old path (a_path) for deletions
                file_path = diff.b_path if diff.b_path else diff.a_path
                if file_path:
                    file_paths.append(file_path)
        except Exception as e:
            logger.warning(f"Failed to extract file paths from commit {commit.hexsha[:8]}: {e}")

        return file_paths

    def _calculate_complexity_delta(self, commit: git.Commit) -> float:
        """Calculate complexity change for a commit with graceful error handling.

        WHY: Repository corruption or missing blobs can cause SHA resolution errors.
        This method provides a fallback complexity calculation that continues
        analysis even when individual blobs are missing or corrupt.
        """
        total_delta = 0.0

        try:
            parent = commit.parents[0] if commit.parents else None
            diffs = commit.diff(parent)
        except Exception as e:
            # If we can't get diffs at all, return 0 complexity delta
            logger.debug(f"Cannot calculate complexity for commit {commit.hexsha[:8]}: {e}")
            return 0.0

        for diff in diffs:
            try:
                if not self._is_code_file(diff.b_path or diff.a_path or ""):
                    continue

                # Simple complexity estimation based on diff size
                # In a real implementation, you'd parse the code and calculate cyclomatic complexity
                if diff.new_file:
                    try:
                        if diff.b_blob and hasattr(diff.b_blob, "size"):
                            total_delta += diff.b_blob.size / 100
                    except (ValueError, AttributeError) as e:
                        logger.debug(
                            f"Cannot access b_blob for new file in {commit.hexsha[:8]}: {e}"
                        )
                        # Use a default small positive delta for new files
                        total_delta += 1.0

                elif diff.deleted_file:
                    try:
                        if diff.a_blob and hasattr(diff.a_blob, "size"):
                            total_delta -= diff.a_blob.size / 100
                    except (ValueError, AttributeError) as e:
                        logger.debug(
                            f"Cannot access a_blob for deleted file in {commit.hexsha[:8]}: {e}"
                        )
                        # Use a default small negative delta for deleted files
                        total_delta -= 1.0

                else:
                    # Modified file - estimate based on change size
                    try:
                        if diff.diff:
                            diff_content = (
                                diff.diff
                                if isinstance(diff.diff, str)
                                else diff.diff.decode("utf-8", errors="ignore")
                            )
                            added = len(diff_content.split("\n+"))
                            removed = len(diff_content.split("\n-"))
                            total_delta += (added - removed) / 10
                    except (ValueError, AttributeError, UnicodeDecodeError) as e:
                        logger.debug(f"Cannot process diff content in {commit.hexsha[:8]}: {e}")
                        # Skip this diff but continue processing
                        pass

            except Exception as e:
                logger.debug(f"Error processing diff in commit {commit.hexsha[:8]}: {e}")
                # Continue to next diff
                continue

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
        """Check if file should be excluded from line counting.

        Delegates to :func:`~gitflow_analytics.utils.glob_matcher.should_exclude_file`.
        """
        return _should_exclude_file_fn(filepath, self.exclude_paths)

    def _matches_glob_pattern(self, filepath: str, pattern: str) -> bool:
        """Check if a file path matches a glob pattern, handling ** recursion correctly.

        Delegates to :func:`~gitflow_analytics.utils.glob_matcher.matches_glob_pattern`.

        Args:
            filepath: The file path to check
            pattern: The glob pattern to match against

        Returns:
            True if the file path matches the pattern, False otherwise
        """
        return _matches_glob_pattern_fn(filepath, pattern)

    def _match_recursive_pattern(self, filepath: str, pattern: str) -> bool:
        """Handle complex patterns with multiple ** wildcards.

        Delegates to :func:`~gitflow_analytics.utils.glob_matcher.match_recursive_pattern`.

        Args:
            filepath: The file path to check
            pattern: The pattern with multiple ** wildcards

        Returns:
            True if the path matches the pattern, False otherwise
        """
        return _match_recursive_pattern_fn(filepath, pattern)

    def _calculate_filtered_stats(self, commit: git.Commit) -> FilteredCommitStats:
        """Calculate commit statistics excluding boilerplate/generated files using git diff --numstat.

        When exclude_merge_commits is enabled, merge commits (commits with 2+ parents) will have
        their filtered line counts set to 0 to exclude them from productivity metrics.
        """
        filtered_stats: FilteredCommitStats = {"files": 0, "insertions": 0, "deletions": 0}

        # Check if this is a merge commit and we should exclude it from filtered counts
        is_merge = is_merge_commit(commit)
        if self.exclude_merge_commits and is_merge:
            logger.debug(
                f"Excluding merge commit {commit.hexsha[:8]} from filtered line counts "
                f"(has {len(commit.parents)} parents)"
            )
            return filtered_stats  # Return zeros for merge commits

        # For initial commits or commits without parents
        parent = commit.parents[0] if commit.parents else None

        try:
            # Use git command directly for accurate line counts
            repo = commit.repo
            if parent:
                diff_output = repo.git.diff(parent.hexsha, commit.hexsha, "--numstat")
            else:
                # Initial commit - use git show with --numstat
                diff_output = repo.git.show(commit.hexsha, "--numstat", "--format=")

            # Parse the numstat output: insertions\tdeletions\tfilename
            for line in diff_output.strip().split("\n"):
                if not line.strip():
                    continue

                parts = line.split("\t")
                if len(parts) >= 3:
                    try:
                        insertions = int(parts[0]) if parts[0] != "-" else 0
                        deletions = int(parts[1]) if parts[1] != "-" else 0
                        filename = parts[2]

                        # Skip excluded files using the existing filter logic
                        if self._should_exclude_file(filename):
                            continue

                        # Count the file and its changes
                        filtered_stats["files"] += 1
                        filtered_stats["insertions"] += insertions
                        filtered_stats["deletions"] += deletions

                    except ValueError:
                        # Skip binary files or malformed lines
                        continue

        except Exception as e:
            # Log the error for debugging but don't crash
            logger.warning(f"Error calculating filtered stats for commit {commit.hexsha[:8]}: {e}")

        return filtered_stats

    def _calculate_raw_stats(self, commit: git.Commit) -> dict[str, int]:
        """Calculate commit statistics for all files (no filtering) using git diff --numstat."""
        raw_stats = {"files": 0, "insertions": 0, "deletions": 0}

        # For initial commits or commits without parents
        parent = commit.parents[0] if commit.parents else None

        try:
            # Use git command directly for accurate line counts
            repo = commit.repo
            if parent:
                diff_output = repo.git.diff(parent.hexsha, commit.hexsha, "--numstat")
            else:
                # Initial commit - use git show with --numstat
                diff_output = repo.git.show(commit.hexsha, "--numstat", "--format=")

            # Parse the numstat output: insertions\tdeletions\tfilename
            for line in diff_output.strip().split("\n"):
                if not line.strip():
                    continue

                parts = line.split("\t")
                if len(parts) >= 3:
                    try:
                        insertions = int(parts[0]) if parts[0] != "-" else 0
                        deletions = int(parts[1]) if parts[1] != "-" else 0
                        # filename = parts[2] - not used in raw stats

                        # Count all files and their changes (no filtering)
                        raw_stats["files"] += 1
                        raw_stats["insertions"] += insertions
                        raw_stats["deletions"] += deletions

                    except ValueError:
                        # Skip binary files or malformed lines
                        continue

        except Exception as e:
            # Log the error for debugging but don't crash
            logger.warning(f"Error calculating raw stats for commit {commit.hexsha[:8]}: {e}")

        return raw_stats

    def _prepare_commits_for_classification(
        self, repo: Repo, commits: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prepare commits for classification by adding file change information.

        Args:
            repo: Git repository object
            commits: List of analyzed commit dictionaries

        Returns:
            List of commits with file change information needed for classification
        """
        prepared_commits = []

        for commit_data in commits:
            commit_hash = commit_data.get("hash")
            if not commit_hash:
                prepared_commits.append(commit_data)
                continue

            try:
                # Use the file paths already extracted during analysis
                files_changed = commit_data.get("files_changed", [])

                # If files_changed is somehow not available or empty, extract it as fallback
                if not files_changed:
                    logger.warning(
                        f"No file paths found for commit {commit_hash[:8]}, extracting as fallback"
                    )
                    files_changed = self._get_changed_file_paths(repo.commit(commit_hash))

                # Create enhanced commit data for classification
                enhanced_commit = commit_data.copy()
                enhanced_commit["files_changed"] = files_changed

                # Add file details if needed by classifier
                if files_changed:
                    file_details = {}
                    # Only extract file details if we need to get commit object for other reasons
                    # or if file details are specifically required by the classifier
                    try:
                        commit = repo.commit(commit_hash)
                        parent = commit.parents[0] if commit.parents else None

                        for diff in commit.diff(parent):
                            file_path = diff.b_path if diff.b_path else diff.a_path
                            if file_path and file_path in files_changed and diff.diff:
                                # Calculate insertions and deletions per file
                                diff_text = (
                                    diff.diff
                                    if isinstance(diff.diff, str)
                                    else diff.diff.decode("utf-8", errors="ignore")
                                )
                                insertions = len(
                                    [
                                        line
                                        for line in diff_text.split("\n")
                                        if line.startswith("+") and not line.startswith("+++")
                                    ]
                                )
                                deletions = len(
                                    [
                                        line
                                        for line in diff_text.split("\n")
                                        if line.startswith("-") and not line.startswith("---")
                                    ]
                                )

                                file_details[file_path] = {
                                    "insertions": insertions,
                                    "deletions": deletions,
                                }

                        enhanced_commit["file_details"] = file_details
                    except Exception as detail_error:
                        logger.warning(
                            f"Failed to extract file details for commit {commit_hash[:8]}: {detail_error}"
                        )
                        enhanced_commit["file_details"] = {}

                prepared_commits.append(enhanced_commit)

            except Exception as e:
                logger.warning(f"Failed to prepare commit {commit_hash} for classification: {e}")
                prepared_commits.append(commit_data)

        return prepared_commits
