"""Interactive installation wizard for GitFlow Analytics.

This module provides a user-friendly installation experience with credential validation
and comprehensive configuration generation.

Method groups extracted to sibling modules via mixins:
- install_wizard_repos.py: git cloning + repos setup (InstallWizardReposMixin)
- install_wizard_pm.py: JIRA/Linear/ClickUp setup (InstallWizardPMMixin)
- install_wizard_output.py: AI/analysis/file generation (InstallWizardOutputMixin)
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from github import Github
from github.GithubException import GithubException

from ..core.git_auth import verify_github_token
from .install_wizard_output import InstallWizardOutputMixin
from .install_wizard_pm import InstallWizardPMMixin
from .install_wizard_repos import InstallWizardReposMixin

logger = logging.getLogger(__name__)


class InstallWizard(InstallWizardOutputMixin, InstallWizardPMMixin, InstallWizardReposMixin):
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

