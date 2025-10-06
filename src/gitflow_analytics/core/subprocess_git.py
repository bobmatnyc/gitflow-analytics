"""Subprocess-based Git operations to avoid authentication prompts."""

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SubprocessGit:
    """Git operations using subprocess to avoid GitPython authentication issues."""

    @staticmethod
    def get_commits_in_range(
        repo_path: Path, start_date: datetime, end_date: datetime, branch: str = "HEAD"
    ) -> list[dict[str, Any]]:
        """Get commits in date range using git log subprocess.

        This avoids GitPython's potential authentication triggers by using
        subprocess directly with environment variables that prevent prompts.
        """
        # Format dates for git log
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Build git log command with JSON-like format for easy parsing
        cmd = [
            "git",
            "log",
            f"--since={start_str}",
            f"--until={end_str}",
            "--all",  # All branches
            "--no-merges",  # Skip merge commits
            "--format=%H|%ae|%an|%at|%s",  # hash|email|name|timestamp|subject
            "--numstat",  # Include file changes
        ]

        # Set environment to prevent any authentication prompts
        env = {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "echo",
            "SSH_ASKPASS": "echo",
            "GCM_INTERACTIVE": "never",
            "DISPLAY": "",
        }

        try:
            result = subprocess.run(
                cmd, cwd=repo_path, capture_output=True, text=True, env=env, timeout=30
            )

            if result.returncode != 0:
                logger.warning(f"Git log failed: {result.stderr}")
                return []

            return SubprocessGit._parse_git_log(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error("Git log timed out - likely authentication issue")
            return []
        except Exception as e:
            logger.error(f"Error running git log: {e}")
            return []

    @staticmethod
    def _parse_git_log(output: str) -> list[dict[str, Any]]:
        """Parse git log output into commit dictionaries."""
        commits = []
        current_commit = None

        for line in output.split("\n"):
            if not line:
                continue

            if "|" in line and not line[0].isdigit():
                # This is a commit line
                if current_commit:
                    commits.append(current_commit)

                parts = line.split("|")
                if len(parts) >= 5:
                    current_commit = {
                        "hash": parts[0],
                        "author_email": parts[1],
                        "author_name": parts[2],
                        "timestamp": int(parts[3]),
                        "message": "|".join(parts[4:]),  # Message might contain |
                        "files": [],
                    }
            elif current_commit and line[0].isdigit():
                # This is a numstat line (additions deletions filename)
                parts = line.split("\t")
                if len(parts) >= 3:
                    current_commit["files"].append(
                        {"additions": parts[0], "deletions": parts[1], "filename": parts[2]}
                    )

        # Don't forget the last commit
        if current_commit:
            commits.append(current_commit)

        return commits

    @staticmethod
    def check_remotes_safe(repo_path: Path) -> bool:
        """Check if repository has remotes without triggering authentication.

        Args:
            repo_path: Path to the git repository

        Returns:
            True if repository has remotes, False otherwise or on error
        """
        cmd = ["git", "remote", "-v"]

        env = {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "echo",
        }

        try:
            result = subprocess.run(
                cmd, cwd=repo_path, capture_output=True, text=True, env=env, timeout=5
            )
            return bool(result.stdout.strip())
        except subprocess.TimeoutExpired:
            logger.warning(f"Git remote check timed out for {repo_path}")
            return False
        except OSError as e:
            logger.warning(f"Failed to check git remotes for {repo_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking git remotes for {repo_path}: {e}")
            return False

    @staticmethod
    def get_branches_safe(repo_path: Path) -> list[str]:
        """Get list of branches without triggering authentication.

        Args:
            repo_path: Path to the git repository

        Returns:
            List of branch names, defaults to common branch names if detection fails
        """
        branches = []

        # Get local branches
        cmd = ["git", "branch", "--format=%(refname:short)"]

        try:
            result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                branches = [b.strip() for b in result.stdout.split("\n") if b.strip()]

        except subprocess.TimeoutExpired:
            logger.warning(f"Git branch listing timed out for {repo_path}")
        except OSError as e:
            logger.warning(f"Failed to list git branches for {repo_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error listing git branches for {repo_path}: {e}")

        # Default to common branch names if none found
        if not branches:
            logger.info(f"No branches detected for {repo_path}, using default branch names")
            branches = ["main", "master", "develop"]

        return branches
