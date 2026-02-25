"""Tests for the installation wizard.

Validates the installation wizard's initialization, configuration
generation, and integration with git authentication validation.
"""

import stat
import tempfile
from pathlib import Path

import pytest

from gitflow_analytics.cli_wizards.install_wizard import InstallWizard


@pytest.mark.unit
class TestInstallWizardInitialization:
    """Test InstallWizard initialization."""

    def test_wizard_initializes_with_temp_dir(self):
        """Test that the wizard can be initialized with an output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wizard = InstallWizard(output_dir=Path(tmpdir), skip_validation=True)
            assert wizard.output_dir.exists()
            assert wizard.skip_validation is True

    def test_wizard_stores_output_dir(self):
        """Test that the wizard stores the output directory path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_output"
            wizard = InstallWizard(output_dir=new_dir, skip_validation=True)
            # Compare resolved paths to handle macOS /private/tmp symlink
            assert wizard.output_dir.resolve() == new_dir.resolve()


@pytest.mark.unit
class TestInstallWizardConfigGeneration:
    """Test configuration file generation."""

    def test_generates_config_and_env_files(self):
        """Test that configuration structure files are properly created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wizard = InstallWizard(output_dir=Path(tmpdir), skip_validation=True)

            wizard.env_data = {
                "GITHUB_TOKEN": "test_token_12345",
                "JIRA_BASE_URL": "https://test.atlassian.net",
                "JIRA_ACCESS_USER": "test@example.com",
                "JIRA_ACCESS_TOKEN": "test_jira_token",
            }

            wizard.config_data = {
                "github": {
                    "token": "${GITHUB_TOKEN}",
                    "organization": "test-org",
                },
                "pm": {
                    "jira": {
                        "base_url": "${JIRA_BASE_URL}",
                        "username": "${JIRA_ACCESS_USER}",
                        "api_token": "${JIRA_ACCESS_TOKEN}",
                    }
                },
                "analysis": {
                    "period_weeks": 4,
                    "output_directory": "./reports",
                    "cache_directory": "./.gitflow-cache",
                },
            }

            success = wizard._generate_files()
            assert success is True

            config_path = Path(tmpdir) / "config.yaml"
            env_path = Path(tmpdir) / ".env"

            assert config_path.exists()
            assert env_path.exists()

    def test_config_file_contains_expected_content(self):
        """Test that generated config file has correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wizard = InstallWizard(output_dir=Path(tmpdir), skip_validation=True)
            wizard.env_data = {"GITHUB_TOKEN": "test_token_12345"}
            wizard.config_data = {
                "github": {"token": "${GITHUB_TOKEN}", "organization": "test-org"},
                "analysis": {"period_weeks": 4},
            }

            wizard._generate_files()
            config_path = Path(tmpdir) / "config.yaml"

            with open(config_path) as f:
                config_content = f.read()

            assert "${GITHUB_TOKEN}" in config_content
            assert "test-org" in config_content

    def test_env_file_contains_expected_credentials(self):
        """Test that generated .env file has correct key=value entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wizard = InstallWizard(output_dir=Path(tmpdir), skip_validation=True)
            wizard.env_data = {
                "GITHUB_TOKEN": "test_token_12345",
                "JIRA_BASE_URL": "https://test.atlassian.net",
            }
            wizard.config_data = {"github": {"token": "${GITHUB_TOKEN}"}}

            wizard._generate_files()
            env_path = Path(tmpdir) / ".env"

            with open(env_path) as f:
                env_content = f.read()

            assert "GITHUB_TOKEN=test_token_12345" in env_content
            assert "JIRA_BASE_URL=https://test.atlassian.net" in env_content

    def test_env_file_has_restricted_permissions(self):
        """Test that generated .env file has 0600 permissions (owner read/write only)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wizard = InstallWizard(output_dir=Path(tmpdir), skip_validation=True)
            wizard.env_data = {"GITHUB_TOKEN": "test_token_12345"}
            wizard.config_data = {"github": {"token": "${GITHUB_TOKEN}"}}

            wizard._generate_files()
            env_path = Path(tmpdir) / ".env"

            env_stat = env_path.stat()
            permissions = stat.filemode(env_stat.st_mode)
            assert permissions == "-rw-------", f"Expected -rw-------, got {permissions}"


@pytest.mark.external
class TestInstallWizardGitHubValidation:
    """Tests that require external GitHub API access."""

    def test_verify_github_token_rejects_invalid_token(self):
        """Test that invalid tokens are rejected by the GitHub API."""
        from gitflow_analytics.core.git_auth import verify_github_token

        success, username, error = verify_github_token("invalid_token_test")
        assert success is False
        assert username == ""
        assert "invalid or expired" in error.lower() or "api error" in error.lower()
