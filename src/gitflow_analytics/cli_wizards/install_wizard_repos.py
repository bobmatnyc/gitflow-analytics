"""Repository setup and git cloning methods for InstallWizard.

Extracted from install_wizard.py to keep it under 800 lines.
Methods: _clone_git_repository, _normalize_git_url, _get_git_progress,
         _setup_manual_repos, _setup_local_repositories
"""

import logging
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import Optional

import click
from git import GitCommandError, Repo
from git.exc import InvalidGitRepositoryError

logger = logging.getLogger(__name__)


class InstallWizardReposMixin:
    """Mixin adding repository setup methods to InstallWizard."""

    def _clone_git_repository(self, git_url: str) -> Optional[tuple[Path, str]]:
        """Clone a Git repository to the local repos/ directory.

        Args:
            git_url: Git URL to clone

        Returns:
            Tuple of (local_path, original_url) if successful, None if failed
        """
        try:
            # Extract repository name from URL
            # Handle both HTTPS and SSH formats
            match = re.search(r"/([^/]+?)(?:\.git)?$", git_url)
            if not match:
                click.echo("❌ Could not extract repository name from URL")
                return None

            repo_name = match.group(1)
            click.echo(f"📦 Repository: {repo_name}")

            # Create repos directory in current working directory
            repos_dir = Path.cwd() / "repos"
            repos_dir.mkdir(parents=True, exist_ok=True)
            click.echo(f"📁 Clone directory: {repos_dir}")

            # Target path for cloned repository
            target_path = repos_dir / repo_name

            # Check if repository already exists
            if target_path.exists():
                click.echo(f"⚠️  Directory already exists: {target_path}")

                # Check if it's a valid git repository
                try:
                    existing_repo = Repo(target_path)
                    if existing_repo.working_dir:
                        click.echo("✅ Found existing git repository")

                        # Check if remote URL matches
                        try:
                            origin_url = existing_repo.remotes.origin.url
                            if origin_url == git_url or self._normalize_git_url(
                                origin_url
                            ) == self._normalize_git_url(git_url):
                                click.echo(f"✅ Remote URL matches: {origin_url}")

                                # Offer to update
                                if click.confirm(
                                    "Update existing repository (git pull)?", default=True
                                ):
                                    click.echo("🔄 Updating repository...")
                                    origin = existing_repo.remotes.origin
                                    origin.pull()
                                    click.echo("✅ Repository updated")

                                return (target_path, git_url)
                            else:
                                click.echo("⚠️  Remote URL mismatch:")
                                click.echo(f"   Existing: {origin_url}")
                                click.echo(f"   Requested: {git_url}")
                                if not click.confirm(
                                    "Use existing repository anyway?", default=False
                                ):
                                    return None
                                return (target_path, git_url)
                        except Exception as e:
                            click.echo(f"⚠️  Could not check remote URL: {type(e).__name__}")
                            if click.confirm("Use existing repository anyway?", default=False):
                                return (target_path, git_url)
                            return None
                except InvalidGitRepositoryError:
                    click.echo("❌ Directory exists but is not a git repository")
                    if not click.confirm("Remove and re-clone?", default=False):
                        return None

                    # Remove existing directory
                    shutil.rmtree(target_path)
                    click.echo("🗑️  Removed existing directory")

            # Clone the repository
            click.echo(f"🔄 Cloning {git_url}...")
            click.echo("   This may take a moment depending on repository size...")

            # Clone with progress
            Repo.clone_from(git_url, target_path, progress=self._get_git_progress())

            # Verify clone succeeded
            if not (target_path / ".git").exists():
                click.echo("❌ Clone appeared to succeed but .git directory not found")
                return None

            click.echo(f"✅ Successfully cloned to: {target_path}")
            return (target_path, git_url)

        except GitCommandError as e:
            click.echo("❌ Git clone failed")

            # Parse error message for common issues
            error_str = str(e).lower()
            if "authentication failed" in error_str or "permission denied" in error_str:
                click.echo("🔐 Authentication required")
                click.echo("   For HTTPS: Configure Git credentials or use a personal access token")
                click.echo("   For SSH: Ensure your SSH key is added to your Git provider")
            elif "not found" in error_str or "does not exist" in error_str:
                click.echo("🔍 Repository not found")
                click.echo("   Check the URL and ensure you have access")
            elif "network" in error_str or "timeout" in error_str:
                click.echo("🌐 Network error")
                click.echo("   Check your internet connection and try again")
            else:
                logger.debug(f"Git clone error type: {type(e).__name__}")

            return None

        except OSError as e:
            error_type = type(e).__name__
            click.echo(f"❌ File system error: {error_type}")
            if "space" in str(e).lower():
                click.echo("💾 Insufficient disk space")
            logger.debug(f"Clone file system error: {error_type}")
            return None

        except Exception as e:
            error_type = type(e).__name__
            click.echo(f"❌ Unexpected error during clone: {error_type}")
            logger.error(f"Clone error type: {error_type}")
            return None

    def _normalize_git_url(self, url: str) -> str:
        """Normalize Git URL for comparison.

        Args:
            url: Git URL to normalize

        Returns:
            Normalized URL (lowercase, with .git extension)
        """
        url = url.lower().strip()
        if not url.endswith(".git"):
            url = url + ".git"
        return url

    def _get_git_progress(self):
        """Get a Git progress handler for clone operations.

        Returns:
            Progress handler for GitPython or None
        """
        try:
            from git import RemoteProgress

            class CloneProgress(RemoteProgress):
                """Progress handler for git clone operations."""

                def __init__(self):
                    super().__init__()
                    self.last_percent = 0

                def update(self, op_code, cur_count, max_count=None, message=""):
                    if max_count:
                        percent = int((cur_count / max_count) * 100)
                        # Only show updates every 10%
                        if percent >= self.last_percent + 10:
                            click.echo(f"   Progress: {percent}%")
                            self.last_percent = percent

            return CloneProgress()
        except Exception:
            # If progress handler fails, return None (clone will work without it)
            return None

    def _setup_manual_repos(self) -> bool:
        """Setup manual repository configuration.

        Returns:
            True if setup successful, False otherwise
        """
        click.echo("\n📦 Manual Repository Mode")
        click.echo("You can specify one or more local repository paths or Git URLs.")
        click.echo("Supported formats:")
        click.echo("  • Local path: /path/to/repo or ~/repos/myproject")
        click.echo("  • HTTPS URL: https://github.com/owner/repo.git")
        click.echo("  • SSH URL: git@github.com:owner/repo.git")
        click.echo()

        repositories = []
        while True:
            repo_input = click.prompt(
                "Enter repository path or Git URL (or press Enter to finish)",
                type=str,
                default="",
                show_default=False,
            ).strip()

            if not repo_input:
                if not repositories:
                    click.echo("❌ At least one repository is required")
                    continue
                break

            # Check if input is a Git URL
            git_url = self._detect_git_url(repo_input)
            if git_url:
                # Handle Git URL cloning
                result = self._clone_git_repository(git_url)
                if result is None:
                    # Clone failed, ask user if they want to retry or skip
                    if not click.confirm("Try a different repository?", default=True):
                        if repositories:
                            break  # User has other repos, can finish
                        continue  # User has no repos yet, must add at least one
                    continue

                # Clone successful
                local_path, original_url = result
                repositories.append({"path": str(local_path), "git_url": original_url})
                click.echo(f"Added repository #{len(repositories)}")
            else:
                # Handle local path
                path_obj = self._validate_directory_path(repo_input, "Repository path")
                if path_obj is None:
                    continue  # Re-prompt

                if not path_obj.exists():
                    click.echo(f"⚠️  Path does not exist: {path_obj}")
                    if not click.confirm("Add anyway?", default=False):
                        continue

                # Check if it's a git repository
                if (path_obj / ".git").exists():
                    click.echo(f"✅ Valid git repository: {path_obj}")
                else:
                    click.echo(f"⚠️  Not a git repository: {path_obj}")
                    if not click.confirm("Add anyway?", default=False):
                        continue

                repositories.append({"path": str(path_obj)})
                click.echo(f"Added repository #{len(repositories)}")

            if not click.confirm("Add another repository?", default=False):
                break

        self.config_data["github"]["repositories"] = repositories
        return True

    def _setup_local_repositories(self) -> bool:
        """Setup local repository paths (no GitHub).

        Returns:
            True if setup successful, False otherwise
        """
        click.echo("\n📦 Local Repository Mode")
        click.echo("Specify local Git repository paths to analyze.")
        click.echo()

        repositories = []
        while True:
            repo_path_str = click.prompt(
                "Enter repository path (or press Enter to finish)",
                type=str,
                default="",
                show_default=False,
            ).strip()

            if not repo_path_str:
                if not repositories:
                    click.echo("❌ At least one repository is required")
                    continue
                break

            # Validate path is safe
            path_obj = self._validate_directory_path(repo_path_str, "Repository path")
            if path_obj is None:
                continue  # Re-prompt

            if not path_obj.exists():
                click.echo(f"⚠️  Path does not exist: {path_obj}")
                if not click.confirm("Add anyway?", default=False):
                    continue

            # Check if it's a git repository
            if (path_obj / ".git").exists():
                click.echo(f"✅ Valid git repository: {path_obj}")
            else:
                click.echo(f"⚠️  Not a git repository: {path_obj}")
                if not click.confirm("Add anyway?", default=False):
                    continue

            repo_name = click.prompt("Repository name", default=path_obj.name).strip()

            repositories.append({"name": repo_name, "path": str(path_obj)})
            click.echo(f"Added repository #{len(repositories)}\n")

            if not click.confirm("Add another repository?", default=False):
                break

        # Store repositories directly without GitHub section
        self.config_data["repositories"] = repositories
        return True

