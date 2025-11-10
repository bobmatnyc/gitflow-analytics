"""Tests for interactive CLI menu system."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest
import yaml

from gitflow_analytics.cli_wizards.menu import (
    edit_configuration,
    find_or_prompt_config,
    fix_aliases,
    get_current_weeks,
    repull_data,
    run_full_analysis,
    set_weeks,
    validate_config,
)


class TestFindOrPromptConfig:
    """Tests for find_or_prompt_config function."""

    def test_finds_config_in_cwd(self, tmp_path, monkeypatch):
        """Test that config.yaml in current directory is found."""
        # Create config.yaml in temp directory
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        with patch("click.confirm", return_value=True):
            result = find_or_prompt_config()

        assert result == config_file
        assert result.exists()

    def test_prompts_when_no_config_in_cwd(self, tmp_path):
        """Test that user is prompted when no config in current directory."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        with (
            patch("click.confirm", return_value=False),
            patch("click.prompt", return_value=str(config_file)),
        ):
            result = find_or_prompt_config()

        assert result == config_file

    def test_handles_invalid_path(self, tmp_path):
        """Test handling of invalid config path."""
        invalid_path = tmp_path / "nonexistent.yaml"

        with (
            patch("click.prompt", return_value=str(invalid_path)),
            patch("click.confirm", return_value=False),
        ):
            result = find_or_prompt_config()

        assert result is None

    def test_handles_abort(self):
        """Test handling of Ctrl+C abort."""
        from click.exceptions import Abort

        with patch("click.prompt", side_effect=Abort):
            result = find_or_prompt_config()

        assert result is None


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_config(self, tmp_path):
        """Test validation of valid config."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "github": {"token": "test_token"},
            "analysis": {"weeks_back": 12},
            "cache": {"directory": ".cache"},
            "output": {"directory": "reports"},
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch("gitflow_analytics.config.loader.ConfigLoader.load"):
            result = validate_config(config_file)

        assert result is True

    def test_invalid_config(self, tmp_path):
        """Test validation of invalid config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: {yaml")

        result = validate_config(config_file)

        assert result is False


class TestEditConfiguration:
    """Tests for edit_configuration function."""

    def test_opens_editor_successfully(self, tmp_path, monkeypatch):
        """Test that editor is opened successfully."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        monkeypatch.setenv("EDITOR", "cat")

        with (
            patch("subprocess.run") as mock_run,
            patch("gitflow_analytics.cli_wizards.menu.validate_config", return_value=True),
        ):
            mock_run.return_value = Mock(returncode=0)
            edit_configuration(config_file)

        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["cat", str(config_file)]

    def test_handles_missing_editor(self, tmp_path):
        """Test handling of missing editor."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        with patch("subprocess.run", side_effect=FileNotFoundError):
            edit_configuration(config_file)  # Should not raise

    def test_falls_back_to_vi(self, tmp_path, monkeypatch):
        """Test fallback to vi when no EDITOR set."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        # Remove EDITOR from environment
        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.delenv("VISUAL", raising=False)

        with (
            patch("subprocess.run") as mock_run,
            patch("gitflow_analytics.cli_wizards.menu.validate_config", return_value=True),
        ):
            mock_run.return_value = Mock(returncode=0)
            edit_configuration(config_file)

        assert mock_run.call_args[0][0] == ["vi", str(config_file)]


class TestFixAliases:
    """Tests for fix_aliases function."""

    def test_launches_alias_creator(self, tmp_path):
        """Test that alias creator subprocess is launched."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            fix_aliases(config_file)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "create-alias-interactive" in cmd
        assert "-c" in cmd
        assert str(config_file) in cmd


class TestGetCurrentWeeks:
    """Tests for get_current_weeks function."""

    def test_reads_weeks_from_config(self, tmp_path):
        """Test reading weeks from config file."""
        config_file = tmp_path / "config.yaml"
        config_data = {"analysis": {"weeks_back": 8}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        result = get_current_weeks(config_file)

        assert result == 8

    def test_defaults_to_12_when_not_set(self, tmp_path):
        """Test default value when weeks not set."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("other: config")

        result = get_current_weeks(config_file)

        assert result == 12

    def test_handles_missing_file(self, tmp_path):
        """Test handling of missing config file."""
        config_file = tmp_path / "nonexistent.yaml"

        result = get_current_weeks(config_file)

        assert result == 12  # Default


class TestRepullData:
    """Tests for repull_data function."""

    def test_runs_analysis_with_cache_clear(self, tmp_path):
        """Test running analysis with cache clear."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("analysis:\n  weeks_back: 4")

        with patch("click.confirm", side_effect=[True, True]), patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            repull_data(config_file)

        cmd = mock_run.call_args[0][0]
        assert "analyze" in cmd
        assert "--clear-cache" in cmd
        assert "--weeks" in cmd

    def test_runs_analysis_without_cache_clear(self, tmp_path):
        """Test running analysis without cache clear."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("analysis:\n  weeks_back: 4")

        with patch("click.confirm", side_effect=[False, True]), patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            repull_data(config_file)

        cmd = mock_run.call_args[0][0]
        assert "analyze" in cmd
        assert "--clear-cache" not in cmd

    def test_prompts_for_custom_weeks(self, tmp_path):
        """Test prompting for custom number of weeks."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("analysis:\n  weeks_back: 4")

        with (
            patch("click.confirm", side_effect=[True, False]),
            patch("click.prompt", return_value=8),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = Mock(returncode=0)
            repull_data(config_file)

        cmd = mock_run.call_args[0][0]
        weeks_idx = cmd.index("--weeks") + 1
        assert cmd[weeks_idx] == "8"


class TestSetWeeks:
    """Tests for set_weeks function."""

    def test_updates_weeks_in_config(self, tmp_path):
        """Test updating weeks in config file."""
        config_file = tmp_path / "config.yaml"
        config_data = {"analysis": {"weeks_back": 4}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with (
            patch("click.prompt", return_value=8),
            patch("gitflow_analytics.cli_wizards.menu.validate_config", return_value=True),
        ):
            set_weeks(config_file)

        # Read back and verify
        with open(config_file) as f:
            updated_config = yaml.safe_load(f)

        assert updated_config["analysis"]["weeks_back"] == 8

    def test_creates_analysis_section_if_missing(self, tmp_path):
        """Test creating analysis section if it doesn't exist."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("other: config")

        with (
            patch("click.prompt", return_value=10),
            patch("gitflow_analytics.cli_wizards.menu.validate_config", return_value=True),
        ):
            set_weeks(config_file)

        # Read back and verify
        with open(config_file) as f:
            updated_config = yaml.safe_load(f)

        assert updated_config["analysis"]["weeks_back"] == 10

    def test_handles_yaml_error(self, tmp_path):
        """Test handling of YAML error."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: {yaml")

        with patch("click.prompt", return_value=8):
            set_weeks(config_file)  # Should not raise, but will fail gracefully


class TestRunFullAnalysis:
    """Tests for run_full_analysis function."""

    def test_runs_analysis_with_current_settings(self, tmp_path):
        """Test running full analysis with current settings."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("analysis:\n  weeks_back: 6")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            run_full_analysis(config_file)

        cmd = mock_run.call_args[0][0]
        assert "analyze" in cmd
        assert "-c" in cmd
        assert str(config_file) in cmd

    def test_handles_analysis_failure(self, tmp_path):
        """Test handling of analysis failure."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("analysis:\n  weeks_back: 6")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1)
            run_full_analysis(config_file)  # Should not raise

    def test_handles_subprocess_error(self, tmp_path):
        """Test handling of subprocess error."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("analysis:\n  weeks_back: 6")

        with patch("subprocess.run", side_effect=Exception("Test error")):
            run_full_analysis(config_file)  # Should not raise


class TestShowMainMenu:
    """Tests for show_main_menu function (integration-style tests)."""

    def test_exits_on_choice_0(self, tmp_path):
        """Test that menu exits on choice 0."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        with patch("click.prompt", return_value="0"):
            from gitflow_analytics.cli_wizards.menu import show_main_menu

            show_main_menu(config_file)  # Should exit cleanly

    def test_prompts_for_config_if_not_provided(self, tmp_path):
        """Test that menu prompts for config if not provided."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        with (
            patch(
                "gitflow_analytics.cli_wizards.menu.find_or_prompt_config", return_value=config_file
            ),
            patch("click.prompt", return_value="0"),
        ):
            from gitflow_analytics.cli_wizards.menu import show_main_menu

            show_main_menu()  # Should find config and exit

    def test_exits_if_no_config_provided(self):
        """Test that menu exits if no config provided."""
        with (
            patch("gitflow_analytics.cli_wizards.menu.find_or_prompt_config", return_value=None),
            pytest.raises(SystemExit),
        ):
            from gitflow_analytics.cli_wizards.menu import show_main_menu

            show_main_menu()
