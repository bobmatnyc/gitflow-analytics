"""
Tests for the CLI module.

These tests verify the command-line interface functionality including argument parsing,
configuration loading, and command execution.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
import tempfile
from pathlib import Path

from gitflow_analytics.cli import cli, analyze, validate


class TestCLI:
    """Test cases for the main CLI functionality."""

    def test_cli_help(self):
        """Test that CLI help message is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "GitFlow Analytics" in result.output
        assert "analyze" in result.output
        assert "validate" in result.output

    def test_analyze_command_help(self):
        """Test that analyze command help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(analyze, ["--help"])

        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--weeks" in result.output
        assert "--clear-cache" in result.output

    @patch("gitflow_analytics.cli.AnalyticsOrchestrator")
    def test_analyze_command_basic(self, mock_orchestrator):
        """Test basic analyze command execution."""
        mock_instance = Mock()
        mock_orchestrator.return_value = mock_instance
        mock_instance.run_analysis.return_value = None

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
            repositories:
              - name: test
                path: /tmp/test
                url: https://github.com/test/test.git
            """
            )
            config_path = f.name

        result = runner.invoke(analyze, ["--config", config_path, "--weeks", "4"])

        # Clean up
        Path(config_path).unlink()

        assert result.exit_code == 0
        mock_orchestrator.assert_called_once()
        mock_instance.run_analysis.assert_called_once()

    @patch("gitflow_analytics.cli.AnalyticsOrchestrator")
    def test_analyze_with_clear_cache(self, mock_orchestrator):
        """Test analyze command with clear cache option."""
        mock_instance = Mock()
        mock_orchestrator.return_value = mock_instance
        mock_instance.run_analysis.return_value = None

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
            repositories:
              - name: test
                path: /tmp/test
                url: https://github.com/test/test.git
            """
            )
            config_path = f.name

        result = runner.invoke(analyze, ["--config", config_path, "--weeks", "4", "--clear-cache"])

        # Clean up
        Path(config_path).unlink()

        assert result.exit_code == 0
        # Verify clear_cache was passed to orchestrator
        args, kwargs = mock_orchestrator.call_args
        assert "clear_cache" in kwargs
        assert kwargs["clear_cache"] is True

    def test_analyze_missing_config(self):
        """Test analyze command with missing configuration file."""
        runner = CliRunner()
        result = runner.invoke(analyze, ["--config", "/nonexistent/config.yaml"])

        assert result.exit_code != 0
        assert "Config file not found" in result.output or "Error" in result.output

    def test_validate_command_help(self):
        """Test that validate command help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(validate, ["--help"])

        assert result.exit_code == 0
        assert "--config" in result.output

    @patch("gitflow_analytics.cli.Config")
    def test_validate_command_basic(self, mock_config_class):
        """Test basic validate command execution."""
        mock_config = Mock()
        mock_config_class.from_file.return_value = mock_config
        mock_config.validate.return_value = True

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
            repositories:
              - name: test
                path: /tmp/test
                url: https://github.com/test/test.git
            """
            )
            config_path = f.name

        result = runner.invoke(validate, ["--config", config_path])

        # Clean up
        Path(config_path).unlink()

        assert result.exit_code == 0
        assert "Configuration is valid" in result.output

    @patch("gitflow_analytics.cli.Config")
    def test_validate_command_invalid_config(self, mock_config_class):
        """Test validate command with invalid configuration."""
        mock_config_class.from_file.side_effect = ValueError("Invalid configuration")

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:")
            config_path = f.name

        result = runner.invoke(validate, ["--config", config_path])

        # Clean up
        Path(config_path).unlink()

        assert result.exit_code != 0
        assert "Invalid configuration" in result.output or "Error" in result.output


class TestVersionDisplay:
    """Test version display functionality."""

    @patch("gitflow_analytics.cli.__version__", "1.2.3")
    def test_version_display(self):
        """Test that version is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "1.2.3" in result.output
