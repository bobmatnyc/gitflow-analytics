"""AI config, analysis setup, file generation, and output methods for InstallWizard.

Extracted from install_wizard.py to keep it under 800 lines.
Methods: _setup_ai, _validate_ai_key, _store_ai_config, _setup_analysis,
         _clear_sensitive_data, _generate_files, _update_gitignore,
         _validate_installation, _run_analysis, _display_success_summary
"""

import logging
import os
import stat
import subprocess
import time
from datetime import datetime
from pathlib import Path

import click
import requests
import yaml

logger = logging.getLogger(__name__)


class InstallWizardOutputMixin:
    """Mixin adding AI, analysis, file generation, and output methods to InstallWizard."""

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
