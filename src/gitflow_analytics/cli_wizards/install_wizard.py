"""Interactive installation wizard for GitFlow Analytics.

This module provides a user-friendly installation experience with credential validation
and comprehensive configuration generation.
"""

import getpass
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import requests
import yaml
from git import GitCommandError, Repo
from git.exc import InvalidGitRepositoryError
from github import Github
from github.GithubException import GithubException

from ..core.git_auth import verify_github_token

logger = logging.getLogger(__name__)


class InstallWizard:
    """Interactive installation wizard for GitFlow Analytics setup."""

    # Installation profiles
    PROFILES = {
        "1": {
            "name": "Standard",
            "description": "GitHub + PM Tools (JIRA/Linear/ClickUp/GitHub Issues) + AI (Full featured)",
            "github": True,
            "repositories": "manual",
            "jira": True,
            "ai": True,
            "analysis": True,
        },
        "2": {
            "name": "GitHub Only",
            "description": "GitHub integration without PM tools",
            "github": True,
            "repositories": "manual",
            "jira": False,
            "ai": False,
            "analysis": True,
        },
        "3": {
            "name": "Organization Mode",
            "description": "Auto-discover repos from GitHub org",
            "github": True,
            "repositories": "organization",
            "jira": True,
            "ai": True,
            "analysis": True,
        },
        "4": {
            "name": "Minimal",
            "description": "Local repos only, no integrations",
            "github": False,
            "repositories": "local",
            "jira": False,
            "ai": False,
            "analysis": True,
        },
        "5": {
            "name": "Custom",
            "description": "Configure everything manually",
            "github": None,  # Ask user
            "repositories": None,  # Ask user
            "jira": None,  # Ask user
            "ai": None,  # Ask user
            "analysis": True,
        },
    }

    def __init__(self, output_dir: Path, skip_validation: bool = False):
        """Initialize the installation wizard.

        Args:
            output_dir: Directory where config files will be created
            skip_validation: Skip credential validation (for testing)
        """
        self.output_dir = Path(output_dir).resolve()
        self.skip_validation = skip_validation
        self.config_data = {}
        self.env_data = {}
        self.profile = None  # Selected installation profile
        self.config_filename = "config.yaml"  # Default config filename (can be overridden)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _is_interactive(self) -> bool:
        """Check if running in interactive terminal.

        Returns:
            True if stdin and stdout are connected to a TTY
        """
        return sys.stdin.isatty() and sys.stdout.isatty()

    def _get_password(self, prompt: str, field_name: str = "password") -> str:
        """Get password input with non-interactive detection.

        Args:
            prompt: Prompt text to display
            field_name: Field name for error messages

        Returns:
            Password string
        """
        if self._is_interactive():
            return getpass.getpass(prompt)
        else:
            click.echo(f"⚠️  Non-interactive mode detected - {field_name} will be visible", err=True)
            return click.prompt(prompt, hide_input=False).strip()

    def _select_profile(self) -> dict:
        """Let user select installation profile."""
        click.echo("\n📋 Installation Profiles")
        click.echo("=" * 60 + "\n")

        for key, profile in self.PROFILES.items():
            click.echo(f"  {key}. {profile['name']}")
            click.echo(f"     {profile['description']}")
            click.echo()

        profile_choice = click.prompt(
            "Select installation profile",
            type=click.Choice(list(self.PROFILES.keys())),
            default="1",
        )

        selected = self.PROFILES[profile_choice].copy()
        click.echo(f"\n✅ Selected: {selected['name']}\n")

        return selected

    def run(self) -> bool:
        """Run the installation wizard.

        Returns:
            True if installation completed successfully, False otherwise
        """
        try:
            click.echo("🚀 GitFlow Analytics Installation Wizard")
            click.echo("=" * 50)
            click.echo()

            # Step 0: Select profile
            self.profile = self._select_profile()

            # Step 1: GitHub Setup (conditional based on profile)
            if self.profile["github"] is not False:
                if not self._setup_github():
                    return False
            else:
                # Minimal mode - no GitHub
                pass

            # Step 2: Repository Configuration (based on profile)
            if self.profile["repositories"] == "organization":
                # Organization mode - already handled in GitHub setup
                pass
            elif self.profile["repositories"] == "manual":
                if not self._setup_repositories():
                    return False
            elif self.profile["repositories"] == "local":
                if not self._setup_local_repositories():
                    return False
            elif self.profile["repositories"] is None and not self._setup_repositories():
                # Custom mode - ask user
                return False

            # Step 3: PM Platform Setup (conditional based on profile)
            if self.profile["jira"]:
                # Profile includes PM tools - let user select which ones
                selected_platforms = self._select_pm_platforms()

                # Setup each selected platform
                if "jira" in selected_platforms:
                    self._setup_jira()
                if "linear" in selected_platforms:
                    linear_config = self._setup_linear()
                    if linear_config:
                        if "pm" not in self.config_data:
                            self.config_data["pm"] = {}
                        self.config_data["pm"]["linear"] = linear_config
                if "clickup" in selected_platforms:
                    clickup_config = self._setup_clickup()
                    if clickup_config:
                        if "pm" not in self.config_data:
                            self.config_data["pm"] = {}
                        self.config_data["pm"]["clickup"] = clickup_config
                # GitHub Issues uses github.token automatically - no separate setup needed

                # Store selected platforms for analysis configuration
                self._selected_pm_platforms = selected_platforms
            elif self.profile["jira"] is None:
                # Custom mode - ask user
                selected_platforms = self._select_pm_platforms()

                # Setup each selected platform
                if "jira" in selected_platforms:
                    self._setup_jira()
                if "linear" in selected_platforms:
                    linear_config = self._setup_linear()
                    if linear_config:
                        if "pm" not in self.config_data:
                            self.config_data["pm"] = {}
                        self.config_data["pm"]["linear"] = linear_config
                if "clickup" in selected_platforms:
                    clickup_config = self._setup_clickup()
                    if clickup_config:
                        if "pm" not in self.config_data:
                            self.config_data["pm"] = {}
                        self.config_data["pm"]["clickup"] = clickup_config

                # Store selected platforms for analysis configuration
                self._selected_pm_platforms = selected_platforms
            else:
                # Profile excludes PM tools
                self._selected_pm_platforms = []

            # Step 4: OpenRouter/ChatGPT Setup (conditional based on profile)
            if self.profile["ai"]:
                self._setup_ai()
            elif self.profile["ai"] is None:
                # Custom mode - ask user
                self._setup_ai()

            # Step 5: Analysis Configuration
            if self.profile["analysis"]:
                self._setup_analysis()

            # Step 6: Generate Files
            if not self._generate_files():
                return False

            # Step 7: Validation
            if not self._validate_installation():
                return False

            # Success summary
            self._display_success_summary()

            return True

        except KeyboardInterrupt:
            click.echo("\n\n⚠️  Installation cancelled by user")
            return False
        except (EOFError, click.exceptions.Abort):
            click.echo("\n\n⚠️  Installation cancelled (non-interactive mode or user abort)")
            return False
        except requests.exceptions.RequestException as e:
            # Never expose raw exception - could contain credentials
            error_type = type(e).__name__
            click.echo(f"\n\n❌ Installation failed: Network error ({error_type})")
            logger.error(f"Installation network error: {error_type}")
            return False
        except Exception as e:
            click.echo("\n\n❌ Installation failed: Unexpected error occurred")
            logger.error(f"Installation error type: {type(e).__name__}")
            return False

    def _setup_github(self) -> bool:
        """Setup GitHub credentials with validation.

        Returns:
            True if setup successful, False otherwise
        """
        click.echo("📋 Step 1: GitHub Setup (REQUIRED)")
        click.echo("-" * 50)
        click.echo("GitHub Personal Access Token is required for repository access.")
        click.echo("Generate token at: https://github.com/settings/tokens")
        click.echo("\nRequired permissions:")
        click.echo("  • repo (Full control of private repositories)")
        click.echo("  • read:org (Read org and team membership)")
        click.echo()

        max_retries = 3
        for attempt in range(max_retries):
            if attempt > 0:
                # Add exponential backoff to prevent rate limiting
                delay = 2 ** (attempt - 1)  # 1, 2, 4 seconds
                click.echo(f"⏳ Waiting {delay} seconds before retry...")
                time.sleep(delay)

            token = self._get_password(
                "Enter GitHub Personal Access Token: ", "GitHub token"
            ).strip()

            if not token:
                click.echo("❌ Token cannot be empty")
                continue

            # Validate token
            if not self.skip_validation:
                click.echo("🔍 Validating GitHub token...")
                success, username, error_msg = verify_github_token(token)

                if success:
                    click.echo(f"✅ Token verified successfully! (user: {username})")
                    self.env_data["GITHUB_TOKEN"] = token
                    self.config_data["github"] = {"token": "${GITHUB_TOKEN}"}
                    return True
                else:
                    click.echo(f"❌ Validation failed: {error_msg}")
                    if attempt < max_retries - 1:
                        retry = click.confirm("Try again?", default=True)
                        if not retry:
                            return False
                    else:
                        click.echo("❌ Maximum retry attempts reached")
                        return False
            else:
                # Skip validation mode
                self.env_data["GITHUB_TOKEN"] = token
                self.config_data["github"] = {"token": "${GITHUB_TOKEN}"}
                return True

        return False

    def _select_pm_platforms(self) -> list:
        """Let user select which PM platforms to configure.

        Returns:
            List of selected platform names (e.g., ['jira', 'linear'])
        """
        click.echo("\n📋 Project Management Platform Selection")
        click.echo("-" * 50)
        click.echo("Select which PM platforms you want to configure:\n")
        click.echo("  1. JIRA (Atlassian)")
        click.echo("  2. Linear (linear.app)")
        click.echo("  3. ClickUp (clickup.com)")
        click.echo("  4. GitHub Issues (Auto-configured with your GitHub token)")
        click.echo()
        click.echo("Enter numbers separated by spaces or commas (e.g., '1 2 4' or '1,2,4')")
        click.echo("Press Enter to skip all PM platform setup")
        click.echo()

        selection = click.prompt(
            "Select platforms",
            type=str,
            default="",
            show_default=False,
        ).strip()

        if not selection:
            click.echo("⏭️  Skipping all PM platform setup")
            return []

        # Parse selection (handle both space and comma separated)
        selection = selection.replace(",", " ")
        choices = selection.split()

        platforms = []
        platform_map = {"1": "jira", "2": "linear", "3": "clickup", "4": "github"}

        for choice in choices:
            if choice in platform_map:
                platforms.append(platform_map[choice])

        if not platforms:
            click.echo(
                "⚠️  No valid platforms selected, defaulting to JIRA for backward compatibility"
            )
            return ["jira"]

        # Display selected platforms
        platform_names = {
            "jira": "JIRA",
            "linear": "Linear",
            "clickup": "ClickUp",
            "github": "GitHub Issues",
        }
        selected_names = [platform_names[p] for p in platforms]
        click.echo(f"\n✅ Selected platforms: {', '.join(selected_names)}\n")

        return platforms

    def _setup_repositories(self) -> bool:
        """Setup repository configuration.

        Returns:
            True if setup successful, False otherwise
        """
        click.echo("\n📋 Step 2: Repository Configuration")
        click.echo("-" * 50)
        click.echo("Choose how to configure repositories:")
        click.echo("  A) Organization mode (auto-discover all repos)")
        click.echo("  B) Manual mode (specify individual repos)")
        click.echo()

        mode = click.prompt(
            "Select mode",
            type=click.Choice(["A", "B", "a", "b"], case_sensitive=False),
            default="A",
        ).upper()

        if mode == "A":
            return self._setup_organization_mode()
        else:
            return self._setup_manual_repos()

    def _setup_organization_mode(self) -> bool:
        """Setup organization mode with validation.

        Returns:
            True if setup successful, False otherwise
        """
        click.echo("\n📦 Organization Mode")
        click.echo("All non-archived repositories will be automatically discovered.")
        click.echo()

        org_name = click.prompt("Enter GitHub organization name", type=str).strip()

        if not org_name:
            click.echo("❌ Organization name cannot be empty")
            return False

        # Validate organization exists (if not skipping validation)
        if not self.skip_validation:
            click.echo(f"🔍 Validating organization '{org_name}'...")
            try:
                github = Github(self.env_data["GITHUB_TOKEN"])
                org = github.get_organization(org_name)
                repo_count = org.public_repos + org.total_private_repos
                click.echo(f"✅ Organization found! (~{repo_count} total repositories)")
            except GithubException as e:
                # Never expose raw exception - could contain credentials
                error_type = type(e).__name__
                click.echo(f"❌ Cannot access organization: {error_type}")
                logger.debug(f"Organization validation error: {error_type}")
                retry = click.confirm("Continue anyway?", default=False)
                if not retry:
                    return False
            except Exception as e:
                error_type = type(e).__name__
                click.echo(f"❌ Unexpected error: {error_type}")
                logger.error(f"Organization validation unexpected error: {error_type}")
                retry = click.confirm("Continue anyway?", default=False)
                if not retry:
                    return False

        self.config_data["github"]["organization"] = org_name
        return True

    def _validate_directory_path(self, path: str, purpose: str) -> Optional[Path]:
        """Validate directory path is safe and within expected boundaries.

        Args:
            path: User-provided path
            purpose: Description of path purpose for error messages

        Returns:
            Validated Path object or None if invalid
        """
        try:
            # Expand and resolve path
            path_obj = Path(path).expanduser().resolve()

            # Prevent absolute paths outside user's home or current directory
            if path_obj.is_absolute():
                home = Path.home()
                cwd = Path.cwd()

                # Check if path is within safe boundaries
                try:
                    # Try relative_to for Python 3.9+
                    is_safe = path_obj.is_relative_to(home) or path_obj.is_relative_to(cwd)
                except AttributeError:
                    # Fallback for Python 3.8
                    is_safe = str(path_obj).startswith(str(home)) or str(path_obj).startswith(
                        str(cwd)
                    )

                if not is_safe:
                    click.echo(f"⚠️  {purpose} must be within home directory or current directory")
                    return None

            return path_obj

        except (ValueError, OSError) as e:
            click.echo(f"⚠️  Invalid path for {purpose}: Path validation error")
            logger.debug(f"Path validation error: {type(e).__name__}")
            return None

    def _detect_git_url(self, input_str: str) -> Optional[str]:
        """Detect if input is a Git URL and normalize it.

        Args:
            input_str: User input string

        Returns:
            Normalized Git URL if detected, None if it's a local path
        """
        import re

        # HTTPS URL patterns
        https_pattern = r"^https?://[^/]+/[^/]+/[^/]+(?:\.git)?$"
        # SSH URL pattern
        ssh_pattern = r"^git@[^:]+:[^/]+/[^/]+(?:\.git)?$"

        input_str = input_str.strip()

        if re.match(https_pattern, input_str, re.IGNORECASE) or re.match(ssh_pattern, input_str):
            # Ensure .git extension for consistency
            if not input_str.endswith(".git"):
                input_str = input_str + ".git"
            return input_str

        return None

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

    def _setup_jira(self) -> None:
        """Setup JIRA integration (optional)."""
        click.echo("\n📋 Step 3: JIRA Setup (OPTIONAL)")
        click.echo("-" * 50)

        if not click.confirm("Enable JIRA integration?", default=False):
            click.echo("⏭️  Skipping JIRA setup")
            return

        click.echo("\nJIRA Configuration:")
        click.echo("You'll need:")
        click.echo("  • JIRA instance URL (e.g., https://yourcompany.atlassian.net)")
        click.echo("  • Email address for API authentication")
        click.echo(
            "  • API token from: https://id.atlassian.com/manage-profile/security/api-tokens"
        )
        click.echo()

        max_retries = 3
        for attempt in range(max_retries):
            if attempt > 0:
                # Add exponential backoff to prevent rate limiting
                delay = 2 ** (attempt - 1)  # 1, 2, 4 seconds
                click.echo(f"⏳ Waiting {delay} seconds before retry...")
                time.sleep(delay)

            base_url = click.prompt("JIRA base URL", type=str).strip()
            access_user = click.prompt("JIRA email", type=str).strip()
            access_token = self._get_password("JIRA API token: ", "JIRA token").strip()

            if not all([base_url, access_user, access_token]):
                click.echo("❌ All JIRA fields are required")
                continue

            # Normalize base_url
            base_url = base_url.rstrip("/")

            # Validate JIRA credentials
            if not self.skip_validation:
                click.echo("🔍 Validating JIRA credentials...")
                if self._validate_jira(base_url, access_user, access_token):
                    click.echo("✅ JIRA credentials validated!")
                    self._store_jira_config(base_url, access_user, access_token)
                    self._discover_jira_fields(base_url, access_user, access_token)
                    return
                else:
                    if attempt < max_retries - 1:
                        retry = click.confirm("JIRA validation failed. Try again?", default=True)
                        if not retry:
                            click.echo("⏭️  Skipping JIRA setup")
                            return
                    else:
                        click.echo("❌ Maximum retry attempts reached")
                        click.echo("⏭️  Skipping JIRA setup")
                        return
            else:
                # Skip validation mode
                self._store_jira_config(base_url, access_user, access_token)
                return

        click.echo("⏭️  Skipping JIRA setup")

    def _setup_linear(self) -> Optional[dict]:
        """Setup Linear integration (optional).

        Returns:
            Linear configuration dict if successful, None otherwise
        """
        click.echo("\n📋 Linear Setup")
        click.echo("-" * 50)
        click.echo("\nLinear Configuration:")
        click.echo("You'll need:")
        click.echo("  • Linear API key from: https://linear.app/settings/api")
        click.echo("  • Optional: Team IDs to filter issues")
        click.echo()

        max_retries = 3
        for attempt in range(max_retries):
            if attempt > 0:
                # Add exponential backoff to prevent rate limiting
                delay = 2 ** (attempt - 1)  # 1, 2, 4 seconds
                click.echo(f"⏳ Waiting {delay} seconds before retry...")
                time.sleep(delay)

            api_key = self._get_password("Linear API key: ", "Linear API key").strip()

            if not api_key:
                click.echo("❌ API key cannot be empty")
                continue

            # Validate Linear credentials
            if not self.skip_validation:
                click.echo("🔍 Validating Linear API key...")
                if self._validate_linear(api_key):
                    click.echo("✅ Linear API key validated!")

                    # Optional: Team IDs
                    team_ids = click.prompt(
                        "Team IDs (comma-separated, press Enter to skip)",
                        type=str,
                        default="",
                        show_default=False,
                    ).strip()

                    # Store configuration
                    self.env_data["LINEAR_API_KEY"] = api_key
                    linear_config = {
                        "api_key": "${LINEAR_API_KEY}",
                    }

                    if team_ids:
                        team_list = [tid.strip() for tid in team_ids.split(",") if tid.strip()]
                        if team_list:
                            linear_config["team_ids"] = team_list
                            click.echo(f"✅ Configured {len(team_list)} team ID(s)")

                    return linear_config
                else:
                    if attempt < max_retries - 1:
                        retry = click.confirm("Linear validation failed. Try again?", default=True)
                        if not retry:
                            click.echo("⏭️  Skipping Linear setup")
                            return None
                    else:
                        click.echo("❌ Maximum retry attempts reached")
                        click.echo("⏭️  Skipping Linear setup")
                        return None
            else:
                # Skip validation mode
                self.env_data["LINEAR_API_KEY"] = api_key
                return {"api_key": "${LINEAR_API_KEY}"}

        click.echo("⏭️  Skipping Linear setup")
        return None

    def _validate_linear(self, api_key: str) -> bool:
        """Validate Linear API key.

        Args:
            api_key: Linear API key

        Returns:
            True if credentials are valid, False otherwise
        """
        # Suppress requests logging to prevent credential exposure
        urllib3_logger = logging.getLogger("urllib3")
        requests_logger = logging.getLogger("requests")
        original_urllib3 = urllib3_logger.level
        original_requests = requests_logger.level

        urllib3_logger.setLevel(logging.WARNING)
        requests_logger.setLevel(logging.WARNING)

        try:
            headers = {
                "Authorization": api_key,
                "Content-Type": "application/json",
            }

            # Simple GraphQL query to validate authentication
            query = {"query": "{ viewer { name } }"}

            response = requests.post(
                "https://api.linear.app/graphql",
                headers=headers,
                json=query,
                timeout=10,
                verify=True,
            )

            if response.status_code == 200:
                data = response.json()
                if "data" in data and "viewer" in data["data"]:
                    viewer_name = data["data"]["viewer"].get("name", "Unknown")
                    click.echo(f"   Authenticated as: {viewer_name}")
                    return True
                else:
                    click.echo("   Authentication failed: Invalid response")
                    return False
            else:
                click.echo(f"   Authentication failed (status {response.status_code})")
                return False

        except requests.exceptions.RequestException as e:
            # Never expose raw exception - could contain credentials
            error_type = type(e).__name__
            click.echo(f"   Connection error: {error_type}")
            logger.debug(f"Linear connection error type: {error_type}")
            return False
        except Exception as e:
            click.echo("   Linear validation failed")
            logger.error(f"Linear validation error type: {type(e).__name__}")
            return False
        finally:
            # Always restore logging levels
            urllib3_logger.setLevel(original_urllib3)
            requests_logger.setLevel(original_requests)

    def _setup_clickup(self) -> Optional[dict]:
        """Setup ClickUp integration (optional).

        Returns:
            ClickUp configuration dict if successful, None otherwise
        """
        click.echo("\n📋 ClickUp Setup")
        click.echo("-" * 50)
        click.echo("\nClickUp Configuration:")
        click.echo("You'll need:")
        click.echo("  • ClickUp API token from: https://app.clickup.com/settings/apps")
        click.echo("  • Workspace URL (e.g., https://app.clickup.com/12345/v/)")
        click.echo()

        max_retries = 3
        for attempt in range(max_retries):
            if attempt > 0:
                # Add exponential backoff to prevent rate limiting
                delay = 2 ** (attempt - 1)  # 1, 2, 4 seconds
                click.echo(f"⏳ Waiting {delay} seconds before retry...")
                time.sleep(delay)

            api_token = self._get_password("ClickUp API token: ", "ClickUp API token").strip()
            workspace_url = click.prompt("ClickUp workspace URL", type=str).strip()

            if not all([api_token, workspace_url]):
                click.echo("❌ All ClickUp fields are required")
                continue

            # Normalize workspace_url
            workspace_url = workspace_url.rstrip("/")

            # Validate ClickUp credentials
            if not self.skip_validation:
                click.echo("🔍 Validating ClickUp credentials...")
                if self._validate_clickup(api_token):
                    click.echo("✅ ClickUp credentials validated!")

                    # Store configuration
                    self.env_data["CLICKUP_API_TOKEN"] = api_token
                    self.env_data["CLICKUP_WORKSPACE_URL"] = workspace_url

                    clickup_config = {
                        "api_token": "${CLICKUP_API_TOKEN}",
                        "workspace_url": "${CLICKUP_WORKSPACE_URL}",
                    }

                    return clickup_config
                else:
                    if attempt < max_retries - 1:
                        retry = click.confirm("ClickUp validation failed. Try again?", default=True)
                        if not retry:
                            click.echo("⏭️  Skipping ClickUp setup")
                            return None
                    else:
                        click.echo("❌ Maximum retry attempts reached")
                        click.echo("⏭️  Skipping ClickUp setup")
                        return None
            else:
                # Skip validation mode
                self.env_data["CLICKUP_API_TOKEN"] = api_token
                self.env_data["CLICKUP_WORKSPACE_URL"] = workspace_url
                return {
                    "api_token": "${CLICKUP_API_TOKEN}",
                    "workspace_url": "${CLICKUP_WORKSPACE_URL}",
                }

        click.echo("⏭️  Skipping ClickUp setup")
        return None

    def _validate_clickup(self, api_token: str) -> bool:
        """Validate ClickUp API token.

        Args:
            api_token: ClickUp API token

        Returns:
            True if credentials are valid, False otherwise
        """
        # Suppress requests logging to prevent credential exposure
        urllib3_logger = logging.getLogger("urllib3")
        requests_logger = logging.getLogger("requests")
        original_urllib3 = urllib3_logger.level
        original_requests = requests_logger.level

        urllib3_logger.setLevel(logging.WARNING)
        requests_logger.setLevel(logging.WARNING)

        try:
            headers = {
                "Authorization": api_token,
                "Content-Type": "application/json",
            }

            response = requests.get(
                "https://api.clickup.com/api/v2/user",
                headers=headers,
                timeout=10,
                verify=True,
            )

            if response.status_code == 200:
                user_info = response.json()
                if "user" in user_info:
                    username = user_info["user"].get("username", "Unknown")
                    click.echo(f"   Authenticated as: {username}")
                    return True
                else:
                    click.echo("   Authentication failed: Invalid response")
                    return False
            else:
                click.echo(f"   Authentication failed (status {response.status_code})")
                return False

        except requests.exceptions.RequestException as e:
            # Never expose raw exception - could contain credentials
            error_type = type(e).__name__
            click.echo(f"   Connection error: {error_type}")
            logger.debug(f"ClickUp connection error type: {error_type}")
            return False
        except Exception as e:
            click.echo("   ClickUp validation failed")
            logger.error(f"ClickUp validation error type: {type(e).__name__}")
            return False
        finally:
            # Always restore logging levels
            urllib3_logger.setLevel(original_urllib3)
            requests_logger.setLevel(original_requests)

    def _validate_jira(self, base_url: str, username: str, api_token: str) -> bool:
        """Validate JIRA credentials.

        Args:
            base_url: JIRA instance URL
            username: JIRA username/email
            api_token: JIRA API token

        Returns:
            True if credentials are valid, False otherwise
        """
        # Suppress requests logging to prevent credential exposure
        urllib3_logger = logging.getLogger("urllib3")
        requests_logger = logging.getLogger("requests")
        original_urllib3 = urllib3_logger.level
        original_requests = requests_logger.level

        urllib3_logger.setLevel(logging.WARNING)
        requests_logger.setLevel(logging.WARNING)

        try:
            import base64

            # Create authentication header
            credentials = base64.b64encode(f"{username}:{api_token}".encode()).decode()
            headers = {
                "Authorization": f"Basic {credentials}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }

            # Test authentication by getting current user info
            response = requests.get(
                f"{base_url}/rest/api/3/myself",
                headers=headers,
                timeout=10,
                verify=True,  # Explicit SSL verification
            )

            if response.status_code == 200:
                user_info = response.json()
                click.echo(f"   Authenticated as: {user_info.get('displayName', username)}")
                return True
            else:
                click.echo(f"   Authentication failed (status {response.status_code})")
                return False

        except requests.exceptions.RequestException as e:
            # Never expose raw exception - could contain credentials
            error_type = type(e).__name__
            click.echo(f"   Connection error: {error_type}")
            logger.debug(f"JIRA connection error type: {error_type}")
            return False
        except Exception as e:
            click.echo("   JIRA validation failed")
            logger.error(f"JIRA validation error type: {type(e).__name__}")
            return False
        finally:
            # Always restore logging levels
            urllib3_logger.setLevel(original_urllib3)
            requests_logger.setLevel(original_requests)

    def _store_jira_config(self, base_url: str, username: str, api_token: str) -> None:
        """Store JIRA configuration.

        Args:
            base_url: JIRA instance URL
            username: JIRA username/email
            api_token: JIRA API token
        """
        self.env_data["JIRA_BASE_URL"] = base_url
        self.env_data["JIRA_ACCESS_USER"] = username
        self.env_data["JIRA_ACCESS_TOKEN"] = api_token

        if "pm" not in self.config_data:
            self.config_data["pm"] = {}

        self.config_data["pm"]["jira"] = {
            "base_url": "${JIRA_BASE_URL}",
            "username": "${JIRA_ACCESS_USER}",
            "api_token": "${JIRA_ACCESS_TOKEN}",
        }

    def _discover_jira_fields(self, base_url: str, username: str, api_token: str) -> None:
        """Discover story point fields in JIRA.

        Args:
            base_url: JIRA instance URL
            username: JIRA username/email
            api_token: JIRA API token
        """
        # Suppress requests logging to prevent credential exposure
        urllib3_logger = logging.getLogger("urllib3")
        requests_logger = logging.getLogger("requests")
        original_urllib3 = urllib3_logger.level
        original_requests = requests_logger.level

        urllib3_logger.setLevel(logging.WARNING)
        requests_logger.setLevel(logging.WARNING)

        try:
            import base64

            click.echo("🔍 Discovering story point fields...")

            credentials = base64.b64encode(f"{username}:{api_token}".encode()).decode()
            headers = {
                "Authorization": f"Basic {credentials}",
                "Accept": "application/json",
            }

            response = requests.get(
                f"{base_url}/rest/api/3/field",
                headers=headers,
                timeout=10,
                verify=True,  # Explicit SSL verification
            )

            if response.status_code != 200:
                return

            fields = response.json()
            story_point_fields = []

            # Look for fields with "story", "point", or "estimate" in name
            for field in fields:
                name = field.get("name", "").lower()
                if any(term in name for term in ["story", "point", "estimate"]):
                    story_point_fields.append(field["id"])

            if story_point_fields:
                click.echo(f"✅ Found {len(story_point_fields)} potential story point field(s)")
                self.config_data["pm"]["jira"]["story_point_fields"] = story_point_fields
            else:
                click.echo("⚠️  No story point fields detected")

        except Exception as e:
            logger.debug(f"JIRA field discovery error type: {type(e).__name__}")
        finally:
            # Always restore logging levels
            urllib3_logger.setLevel(original_urllib3)
            requests_logger.setLevel(original_requests)

    def _setup_ai(self) -> None:
        """Setup AI-powered insights (optional)."""
        click.echo("\n📋 Step 4: AI-Powered Insights (OPTIONAL)")
        click.echo("-" * 50)

        if not click.confirm("Enable AI-powered qualitative analysis?", default=False):
            click.echo("⏭️  Skipping AI setup")
            return

        click.echo("\nAI Configuration:")
        click.echo("GitFlow Analytics supports:")
        click.echo("  • OpenRouter (sk-or-...) - Recommended, supports multiple models")
        click.echo("  • OpenAI (sk-...) - Direct OpenAI API access")
        click.echo("\nGet API key from:")
        click.echo("  • OpenRouter: https://openrouter.ai/keys")
        click.echo("  • OpenAI: https://platform.openai.com/api-keys")
        click.echo()

        max_retries = 3
        for attempt in range(max_retries):
            if attempt > 0:
                # Add exponential backoff to prevent rate limiting
                delay = 2 ** (attempt - 1)  # 1, 2, 4 seconds
                click.echo(f"⏳ Waiting {delay} seconds before retry...")
                time.sleep(delay)

            api_key = self._get_password("Enter API key: ", "AI API key").strip()

            if not api_key:
                click.echo("❌ API key cannot be empty")
                continue

            # Detect provider
            is_openrouter = api_key.startswith("sk-or-")
            provider = "OpenRouter" if is_openrouter else "OpenAI"

            # Validate API key
            if not self.skip_validation:
                click.echo(f"🔍 Validating {provider} API key...")
                if self._validate_ai_key(api_key, is_openrouter):
                    click.echo(f"✅ {provider} API key validated!")
                    self._store_ai_config(api_key, is_openrouter)
                    return
                else:
                    if attempt < max_retries - 1:
                        retry = click.confirm(
                            f"{provider} validation failed. Try again?", default=True
                        )
                        if not retry:
                            click.echo("⏭️  Skipping AI setup")
                            return
                    else:
                        click.echo("❌ Maximum retry attempts reached")
                        click.echo("⏭️  Skipping AI setup")
                        return
            else:
                # Skip validation mode
                self._store_ai_config(api_key, is_openrouter)
                return

        click.echo("⏭️  Skipping AI setup")

    def _validate_ai_key(self, api_key: str, is_openrouter: bool) -> bool:
        """Validate AI API key with simple test request.

        Args:
            api_key: API key to validate
            is_openrouter: True if OpenRouter key, False if OpenAI

        Returns:
            True if key is valid, False otherwise
        """
        # Suppress requests logging to prevent credential exposure
        urllib3_logger = logging.getLogger("urllib3")
        requests_logger = logging.getLogger("requests")
        original_urllib3 = urllib3_logger.level
        original_requests = requests_logger.level

        urllib3_logger.setLevel(logging.WARNING)
        requests_logger.setLevel(logging.WARNING)

        try:
            if is_openrouter:
                # Test OpenRouter
                url = "https://openrouter.ai/api/v1/models"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                }
                response = requests.get(url, headers=headers, timeout=10, verify=True)
                return response.status_code == 200
            else:
                # Test OpenAI
                url = "https://api.openai.com/v1/models"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                }
                response = requests.get(url, headers=headers, timeout=10, verify=True)
                return response.status_code == 200

        except requests.exceptions.RequestException as e:
            # Never expose raw exception - could contain credentials
            error_type = type(e).__name__
            click.echo(f"   Connection error: {error_type}")
            logger.debug(f"AI API connection error type: {error_type}")
            return False
        except Exception as e:
            click.echo("   AI API validation failed")
            logger.error(f"AI API validation error type: {type(e).__name__}")
            return False
        finally:
            # Always restore logging levels
            urllib3_logger.setLevel(original_urllib3)
            requests_logger.setLevel(original_requests)

    def _store_ai_config(self, api_key: str, is_openrouter: bool) -> None:
        """Store AI configuration.

        Args:
            api_key: API key
            is_openrouter: True if OpenRouter key, False if OpenAI
        """
        if is_openrouter:
            self.env_data["OPENROUTER_API_KEY"] = api_key
            self.config_data["chatgpt"] = {
                "api_key": "${OPENROUTER_API_KEY}",
            }
        else:
            self.env_data["OPENAI_API_KEY"] = api_key
            self.config_data["chatgpt"] = {
                "api_key": "${OPENAI_API_KEY}",
            }

    def _setup_analysis(self) -> None:
        """Setup analysis configuration."""
        click.echo("\n📋 Step 5: Analysis Configuration")
        click.echo("-" * 50)

        period_weeks = click.prompt(
            "Analysis period (weeks)",
            type=int,
            default=4,
        )

        # Validate output directory path
        while True:
            output_dir = click.prompt(
                "Output directory for reports",
                type=str,
                default="./reports",
            ).strip()
            output_path = self._validate_directory_path(output_dir, "Output directory")
            if output_path is not None:
                output_dir = str(output_path)
                break
            click.echo("Please enter a valid directory path.")

        # Validate cache directory path
        while True:
            cache_dir = click.prompt(
                "Cache directory",
                type=str,
                default="./.gitflow-cache",
            ).strip()
            cache_path = self._validate_directory_path(cache_dir, "Cache directory")
            if cache_path is not None:
                cache_dir = str(cache_path)
                break
            click.echo("Please enter a valid directory path.")

        if "analysis" not in self.config_data:
            self.config_data["analysis"] = {}

        self.config_data["analysis"]["period_weeks"] = period_weeks
        self.config_data["analysis"]["output_directory"] = output_dir
        self.config_data["analysis"]["cache_directory"] = cache_dir

        # NEW: Aliases configuration
        click.echo("\n🔗 Developer Identity Aliases")
        click.echo("-" * 40 + "\n")

        click.echo("Aliases consolidate multiple email addresses for the same developer.")
        click.echo("You can use a shared aliases.yaml file across multiple configs.\n")

        use_aliases = click.confirm("Configure aliases file?", default=True)

        if use_aliases:
            aliases_options = [
                "1. Create new aliases.yaml in this directory",
                "2. Use existing shared aliases file",
                "3. Generate aliases using LLM (after installation)",
            ]

            click.echo("\nOptions:")
            for option in aliases_options:
                click.echo(f"  {option}")

            aliases_choice = click.prompt(
                "\nSelect option", type=click.Choice(["1", "2", "3"]), default="1"
            )

            if aliases_choice == "1":
                # Create new aliases file
                aliases_path = "aliases.yaml"

                # Ensure analysis.identity section exists
                if "identity" not in self.config_data.get("analysis", {}):
                    if "analysis" not in self.config_data:
                        self.config_data["analysis"] = {}
                    self.config_data["analysis"]["identity"] = {}

                self.config_data["analysis"]["identity"]["aliases_file"] = aliases_path

                # Create empty aliases file
                from ..config.aliases import AliasesManager

                aliases_full_path = self.output_dir / aliases_path
                aliases_mgr = AliasesManager(aliases_full_path)
                aliases_mgr.save()  # Creates empty file with comments

                click.echo(f"\n✅ Created {aliases_path}")
                click.echo("   Generate aliases after installation with:")
                click.echo("   gitflow-analytics aliases -c config.yaml --apply\n")

            elif aliases_choice == "2":
                # Use existing file
                aliases_path = click.prompt(
                    "Path to aliases.yaml (relative to config)", default="../shared/aliases.yaml"
                ).strip()

                # Ensure analysis.identity section exists
                if "identity" not in self.config_data.get("analysis", {}):
                    if "analysis" not in self.config_data:
                        self.config_data["analysis"] = {}
                    self.config_data["analysis"]["identity"] = {}

                self.config_data["analysis"]["identity"]["aliases_file"] = aliases_path

                click.echo(f"\n✅ Configured to use: {aliases_path}\n")

            else:  # choice == "3"
                # Will generate after installation
                click.echo("\n💡 After installation, run:")
                click.echo("   gitflow-analytics aliases -c config.yaml --apply")
                click.echo("   This will analyze your repos and generate aliases automatically.\n")

        # Configure ticket platforms based on selected PM tools
        if hasattr(self, "_selected_pm_platforms") and self._selected_pm_platforms:
            ticket_platforms = []

            # Add platforms in order of setup
            if "jira" in self._selected_pm_platforms:
                ticket_platforms.append("jira")
            if "linear" in self._selected_pm_platforms:
                ticket_platforms.append("linear")
            if "clickup" in self._selected_pm_platforms:
                ticket_platforms.append("clickup")
            if "github" in self._selected_pm_platforms or "github" in self.config_data:
                # GitHub Issues auto-configured with GitHub token
                ticket_platforms.append("github")

            if ticket_platforms:
                self.config_data["analysis"]["ticket_platforms"] = ticket_platforms
                click.echo(f"✅ Configured ticket platforms: {', '.join(ticket_platforms)}\n")

    def _clear_sensitive_data(self) -> None:
        """Clear sensitive data from memory after use."""
        sensitive_keys = ["TOKEN", "KEY", "PASSWORD", "SECRET"]

        for key in list(self.env_data.keys()):
            if any(sensitive in key.upper() for sensitive in sensitive_keys):
                # Overwrite with random data before deletion
                self.env_data[key] = "CLEARED_" + os.urandom(16).hex()
                del self.env_data[key]

        # Clear the dictionary
        self.env_data.clear()

    def _generate_files(self) -> bool:
        """Generate configuration and environment files.

        Returns:
            True if files generated successfully, False otherwise
        """
        click.echo("\n📋 Step 6: Generating Configuration Files")
        click.echo("-" * 50)

        try:
            # Generate config file with custom filename
            config_path = self.output_dir / self.config_filename
            if config_path.exists() and not click.confirm(
                f"⚠️  {config_path} already exists. Overwrite?", default=False
            ):
                click.echo("❌ Installation cancelled")
                return False

            with open(config_path, "w") as f:
                yaml.safe_dump(
                    self.config_data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                )
            click.echo(f"✅ Created: {config_path}")

            # Generate .env file with atomic secure permissions
            env_path = self.output_dir / ".env"
            if env_path.exists() and not click.confirm(
                f"⚠️  {env_path} already exists. Overwrite?", default=False
            ):
                click.echo("❌ Installation cancelled")
                return False

            # Atomically create file with secure permissions using umask
            old_umask = os.umask(0o077)  # Ensure only owner can read/write
            try:
                with open(env_path, "w") as f:
                    f.write("# GitFlow Analytics Environment Variables\n")
                    f.write(
                        f"# Generated by installation wizard on {datetime.now().strftime('%Y-%m-%d')}\n"
                    )
                    f.write(
                        "# WARNING: This file contains sensitive credentials - never commit to git\n\n"
                    )

                    for key, value in self.env_data.items():
                        f.write(f"{key}={value}\n")
            finally:
                # Always restore original umask
                os.umask(old_umask)

            # Verify permissions are correct (redundant but defensive)
            env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600

            # Verify actual permissions
            actual_perms = stat.S_IMODE(os.stat(env_path).st_mode)
            if actual_perms != 0o600:
                click.echo(f"⚠️  Warning: .env permissions are {oct(actual_perms)}, expected 0o600")
                return False

            click.echo(f"✅ Created: {env_path} (permissions: 0600)")

            # Update .gitignore if in git repository
            self._update_gitignore()

            return True

        except OSError as e:
            click.echo("❌ Failed to generate files: File system error")
            logger.error(f"File generation OS error: {type(e).__name__}")
            return False
        except Exception as e:
            click.echo("❌ Failed to generate files: Unexpected error occurred")
            logger.error(f"File generation error type: {type(e).__name__}")
            return False
        finally:
            # Always clear sensitive data from memory
            self._clear_sensitive_data()

    def _update_gitignore(self) -> None:
        """Update .gitignore to include .env if in a git repository."""
        try:
            # Check if we're in a git repository
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.output_dir,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                # Not a git repository
                return

            gitignore_path = self.output_dir / ".gitignore"

            # Read existing .gitignore
            existing_patterns = set()
            if gitignore_path.exists():
                with open(gitignore_path) as f:
                    existing_patterns = set(line.strip() for line in f if line.strip())

            # Add .env pattern if not present
            if ".env" not in existing_patterns:
                with open(gitignore_path, "a") as f:
                    if existing_patterns:
                        f.write("\n")
                    f.write("# GitFlow Analytics environment variables\n")
                    f.write(".env\n")
                click.echo("✅ Updated .gitignore to exclude .env")

        except Exception as e:
            logger.debug(f"Could not update .gitignore: {e}")

    def _validate_installation(self) -> bool:
        """Validate the installation by testing the configuration.

        Returns:
            True if validation successful, False otherwise
        """
        click.echo("\n📋 Step 7: Validating Installation")
        click.echo("-" * 50)

        config_path = self.output_dir / self.config_filename

        if not config_path.exists():
            click.echo("❌ Configuration file not found")
            return False

        click.echo("🔍 Testing configuration...")

        try:
            # Test configuration loading
            from ..config import ConfigLoader

            ConfigLoader.load(config_path)
            click.echo("✅ Configuration validated successfully")

            # Offer to run first analysis
            if click.confirm("\nRun initial analysis now?", default=False):
                self._run_analysis(config_path)

            return True

        except Exception as e:
            click.echo(f"❌ Configuration validation failed: {e}", err=True)
            click.echo("You may need to adjust the configuration manually.")
            logger.error(f"Configuration validation error type: {type(e).__name__}")
            return True  # Don't fail installation on validation error

    def _run_analysis(self, config_path: Path) -> None:
        """Run initial analysis.

        Args:
            config_path: Path to configuration file
        """
        try:
            import sys

            click.echo("\n🚀 Running analysis...")
            click.echo("-" * 50)

            # Use subprocess to run analysis
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gitflow_analytics.cli",
                    "analyze",
                    "--config",
                    str(config_path),
                ],
                cwd=self.output_dir,
                capture_output=False,
            )

            if result.returncode == 0:
                click.echo("\n✅ Analysis completed successfully!")
            else:
                click.echo(f"\n⚠️  Analysis exited with code {result.returncode}")

        except subprocess.SubprocessError as e:
            click.echo("\n❌ Failed to run analysis: Process error")
            logger.error(f"Analysis subprocess error type: {type(e).__name__}")
        except Exception as e:
            click.echo("\n❌ Failed to run analysis: Unexpected error occurred")
            logger.error(f"Analysis error type: {type(e).__name__}")

    def _display_success_summary(self) -> None:
        """Display installation success summary."""
        click.echo("\n" + "=" * 50)
        click.echo("✅ Installation Complete!")
        click.echo("=" * 50)

        config_path = self.output_dir / self.config_filename
        env_path = self.output_dir / ".env"

        click.echo("\n📁 Generated Files:")
        click.echo(f"   • Configuration: {config_path}")
        click.echo(f"   • Environment:   {env_path}")

        click.echo("\n🔐 Security Reminders:")
        click.echo("   • .env file contains sensitive credentials")
        click.echo("   • Permissions set to 0600 (owner read/write only)")
        click.echo("   • Never commit .env to version control")

        click.echo("\n🚀 Next Steps:")
        click.echo(f"   1. Review configuration: {config_path}")
        click.echo("   2. Run analysis:")
        click.echo(f"      gitflow-analytics analyze --config {config_path}")
        click.echo("   3. Check generated reports in: ./reports/")

        click.echo("\n📚 Documentation:")
        click.echo("   • Configuration Guide: docs/guides/configuration.md")
        click.echo("   • Getting Started: docs/getting-started/README.md")
        click.echo("   • Repository: https://github.com/EWTN-Global/gitflow-analytics")

        click.echo()
