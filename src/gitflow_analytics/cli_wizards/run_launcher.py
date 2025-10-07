"""Interactive launcher for GitFlow Analytics with repository selection and preferences.

This module provides an interactive workflow for running GitFlow Analytics with:
- Configuration file selection
- Repository multi-select
- Analysis period configuration
- Cache management
- Persistent preferences
"""

import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import click
import yaml

from ..config import Config, ConfigLoader

logger = logging.getLogger(__name__)


class InteractiveLauncher:
    """Interactive launcher for gitflow-analytics with preferences."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the interactive launcher.

        Args:
            config_path: Optional path to configuration file.
                        If not provided, will search for default configs.
        """
        self.config_path = config_path or self._find_default_config()
        self.config: Optional[Config] = None
        self.preferences: dict[str, Any] = {}

    def run(self) -> bool:
        """Execute interactive launcher workflow.

        Returns:
            True if analysis completed successfully, False otherwise.
        """
        click.echo("🚀 GitFlow Analytics Interactive Launcher\n")

        # Step 1: Load configuration
        if not self._load_config():
            return False

        # Step 2: Load existing preferences
        self._load_preferences()

        # Step 3: Repository selection
        selected_repos = self._select_repositories()
        if not selected_repos:
            click.echo("❌ No repositories selected!")
            return False

        # Step 4: Analysis period
        weeks = self._select_analysis_period()

        # Step 5: Cache management
        clear_cache = self._confirm_clear_cache()

        # Step 6: Identity analysis option
        skip_identity = self._confirm_skip_identity()

        # Step 7: Save preferences
        self._save_preferences(selected_repos, weeks, clear_cache, skip_identity)

        # Step 8: Run analysis
        return self._run_analysis(selected_repos, weeks, clear_cache, skip_identity)

    def _find_default_config(self) -> Optional[Path]:
        """Search for default configuration file.

        Returns:
            Path to config file if found, None otherwise.
        """
        # Common config file names in order of preference
        config_names = [
            "config.yaml",
            "config.yml",
            "gitflow-config.yaml",
            "gitflow-config.yml",
            ".gitflow.yaml",
        ]

        cwd = Path.cwd()
        for name in config_names:
            config_path = cwd / name
            if config_path.exists():
                return config_path

        return None

    def _load_config(self) -> bool:
        """Load configuration file.

        Returns:
            True if configuration loaded successfully, False otherwise.
        """
        if not self.config_path:
            click.echo("❌ No configuration file found!")
            click.echo("\nSearched for: config.yaml, config.yml, gitflow-config.yaml")
            click.echo("\n💡 Run 'gitflow-analytics install' to create a configuration")
            return False

        if not self.config_path.exists():
            click.echo(f"❌ Configuration file not found: {self.config_path}")
            return False

        try:
            click.echo(f"📁 Loading configuration from: {self.config_path}")
            self.config = ConfigLoader.load(self.config_path)
            click.echo("✅ Configuration loaded\n")
            return True
        except Exception as e:
            click.echo(f"❌ Error loading configuration: {e}")
            logger.error(f"Config loading error: {type(e).__name__}")
            return False

    def _load_preferences(self) -> None:
        """Load existing launcher preferences from configuration."""
        if not self.config:
            return

        # Load preferences from config YAML if they exist
        try:
            with open(self.config_path) as f:
                config_data = yaml.safe_load(f)

            if "launcher" in config_data:
                self.preferences = config_data["launcher"]
                logger.info(f"Loaded preferences: {self.preferences}")
        except Exception as e:
            logger.warning(f"Could not load preferences: {e}")
            self.preferences = {}

    def _get_available_repositories(self) -> list[str]:
        """Get list of available repositories from config.

        Supports both explicit repository configuration and organization mode.

        Returns:
            List of repository names
        """
        repos = []

        # Check for organization mode
        try:
            with open(self.config_path) as f:
                config_data = yaml.safe_load(f)

            github_config = config_data.get("github", {})

            if github_config.get("organization"):
                # Organization mode - fetch repos from GitHub
                token = self._resolve_env_var(github_config.get("token", ""))
                if token:
                    from github import Github

                    try:
                        gh = Github(token)
                        org_name = github_config["organization"]
                        org = gh.get_organization(org_name)

                        # Get all non-archived repos
                        for repo in org.get_repos(type="all"):
                            if not repo.archived:
                                repos.append(repo.full_name)

                        if repos:
                            click.echo(
                                f"🔍 Discovered {len(repos)} repositories from organization '{org_name}'\n"
                            )
                    except Exception as e:
                        click.echo(
                            f"⚠️  Could not fetch organization repos: {type(e).__name__}", err=True
                        )
                        logger.error(f"Organization repo fetch error: {e}")

            # Fall back to explicit repositories
            if not repos and self.config and self.config.repositories:
                repos = [repo.path.name for repo in self.config.repositories]

        except Exception as e:
            logger.warning(f"Could not determine repositories: {e}")
            # Final fallback
            if self.config and self.config.repositories:
                repos = [repo.path.name for repo in self.config.repositories]

        return repos

    def _resolve_env_var(self, value: str) -> str:
        """Resolve environment variable in config value.

        Args:
            value: Config value that may contain ${VAR_NAME}

        Returns:
            Resolved value
        """
        import os
        import re

        # Load .env file if it exists next to config
        env_path = self.config_path.parent / ".env"
        if env_path.exists():
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, val = line.split("=", 1)
                            os.environ.setdefault(key.strip(), val.strip())
            except Exception as e:
                logger.debug(f"Could not load .env: {e}")

        # Resolve ${VAR_NAME} patterns
        pattern = r"\$\{([^}]+)\}"

        def replace_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(pattern, replace_var, value)

    def _select_repositories(self) -> list[str]:
        """Interactive repository selection with multi-select.

        Returns:
            List of selected repository names.
        """
        # Check for organization mode first
        repos = self._get_available_repositories()
        if not repos:
            click.echo("❌ No repositories configured!")
            return []

        click.echo("📂 Available Repositories:\n")

        # Get last selected repos from preferences
        last_selected = self.preferences.get("last_selected_repos", [])

        # Display repositories with numbering and selection status
        for i, repo_name in enumerate(repos, 1):
            status = "✓" if repo_name in last_selected else " "
            click.echo(f"   [{status}] {i}. {repo_name}")

        # Get user selection
        click.echo("\n📝 Select repositories:")
        click.echo("  • Enter numbers (comma-separated): 1,3,5")
        click.echo("  • Enter 'all' for all repositories")
        click.echo("  • Press Enter to use previous selection")

        selection = click.prompt("Selection", default="", show_default=False).strip()

        if selection.lower() == "all":
            selected = repos
            click.echo(f"✅ Selected all {len(selected)} repositories\n")
            return selected
        elif not selection and last_selected:
            click.echo(f"✅ Using previous selection: {len(last_selected)} repositories\n")
            return last_selected
        elif not selection:
            # Default to all repos if no previous selection
            selected = repos
            click.echo(f"✅ Selected all {len(selected)} repositories (default)\n")
            return selected
        else:
            # Parse comma-separated numbers
            try:
                indices = [int(x.strip()) for x in selection.split(",")]
                selected = []
                for i in indices:
                    if 1 <= i <= len(repos):
                        selected.append(repos[i - 1])
                    else:
                        click.echo(f"⚠️  Invalid index: {i} (ignored)")

                if not selected:
                    click.echo("❌ No valid repositories selected!")
                    return []

                click.echo(f"✅ Selected {len(selected)} repositories\n")
                return selected
            except (ValueError, IndexError) as e:
                click.echo(f"❌ Invalid selection: {e}")
                return self._select_repositories()  # Retry

    def _select_analysis_period(self) -> int:
        """Prompt for analysis period in weeks.

        Returns:
            Number of weeks to analyze.
        """
        default_weeks = self.preferences.get("default_weeks", 4)
        weeks = click.prompt(
            "📅 Number of weeks to analyze",
            type=click.IntRange(1, 52),
            default=default_weeks,
        )
        return weeks

    def _confirm_clear_cache(self) -> bool:
        """Confirm cache clearing.

        Returns:
            True if cache should be cleared, False otherwise.
        """
        default = self.preferences.get("auto_clear_cache", False)
        return click.confirm("🗑️  Clear cache before analysis?", default=default)

    def _confirm_skip_identity(self) -> bool:
        """Confirm skipping identity analysis.

        Returns:
            True if identity analysis should be skipped, False otherwise.
        """
        default = self.preferences.get("skip_identity_analysis", False)
        return click.confirm("🔍 Skip identity analysis?", default=default)

    def _save_preferences(
        self,
        repos: list[str],
        weeks: int,
        clear_cache: bool,
        skip_identity: bool,
    ) -> None:
        """Save preferences to config file.

        Args:
            repos: Selected repository names
            weeks: Analysis period in weeks
            clear_cache: Whether to clear cache
            skip_identity: Whether to skip identity analysis
        """
        try:
            click.echo("💾 Saving preferences...")

            # Load existing config YAML
            with open(self.config_path) as f:
                config_data = yaml.safe_load(f)

            # Update launcher preferences
            config_data["launcher"] = {
                "last_selected_repos": repos,
                "default_weeks": weeks,
                "auto_clear_cache": clear_cache,
                "skip_identity_analysis": skip_identity,
                "last_run": datetime.now(timezone.utc).isoformat(),
            }

            # Write back to file
            with open(self.config_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

            click.echo("✅ Preferences saved to config.yaml\n")
        except Exception as e:
            click.echo(f"⚠️  Could not save preferences: {e}")
            logger.error(f"Preference saving error: {type(e).__name__}")

    def _run_analysis(
        self,
        repos: list[str],
        weeks: int,
        clear_cache: bool,
        skip_identity: bool,
    ) -> bool:
        """Execute analysis with selected options.

        Args:
            repos: Selected repository names
            weeks: Analysis period in weeks
            clear_cache: Whether to clear cache
            skip_identity: Whether to skip identity analysis

        Returns:
            True if analysis completed successfully, False otherwise.
        """
        click.echo("🚀 Starting analysis...")
        click.echo(f"   Repositories: {', '.join(repos)}")
        click.echo(f"   Period: {weeks} weeks")
        click.echo(f"   Clear cache: {'Yes' if clear_cache else 'No'}")
        click.echo(f"   Skip identity: {'Yes' if skip_identity else 'No'}\n")

        # Build command to execute
        cmd = [
            sys.executable,
            "-m",
            "gitflow_analytics.cli",
            "analyze",
            "-c",
            str(self.config_path),
            "--weeks",
            str(weeks),
        ]

        if clear_cache:
            cmd.append("--clear-cache")

        if skip_identity:
            cmd.append("--skip-identity-analysis")

        # Execute analysis as subprocess to avoid Click context issues
        try:
            result = subprocess.run(cmd, check=True)
            click.echo("\n✅ Analysis complete!")
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            click.echo(f"\n❌ Analysis failed with exit code: {e.returncode}")
            logger.error(f"Analysis subprocess error: {e}")
            return False
        except Exception as e:
            click.echo(f"\n❌ Analysis failed: {e}")
            logger.error(f"Analysis error: {type(e).__name__}")
            return False


def run_interactive_launcher(config_path: Optional[Path] = None) -> bool:
    """Run the interactive launcher.

    Args:
        config_path: Optional path to configuration file

    Returns:
        True if launcher completed successfully, False otherwise
    """
    launcher = InteractiveLauncher(config_path=config_path)
    return launcher.run()
