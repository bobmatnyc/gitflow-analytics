"""Tests for configuration loading and YAML error handling."""

import tempfile
from pathlib import Path

import pytest

from gitflow_analytics.config import ConfigLoader


class TestConfigLoader:
    """Test configuration loading functionality."""

    def test_load_valid_config(self):
        """Test loading a valid configuration."""
        yaml_content = """version: "1.0"
github:
  token: "${GITHUB_TOKEN}"
  owner: "test-owner"
repositories:
  - name: "test-repo"
    path: "/path/to/repo"
analysis:
  story_point_patterns:
    - "SP: (\\\\d+)"
output:
  formats: ["csv"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            # This should work without error (though it may fail later due to missing env vars)
            # We're mainly testing that YAML parsing works
            ConfigLoader.load(temp_path)
        except Exception as e:
            # Environment variable errors are expected in tests
            error_str = str(e)
            if "GITHUB_TOKEN" not in error_str and "Environment variable" not in error_str:
                pytest.fail(f"Unexpected error: {e}")
        finally:
            temp_path.unlink()

    def test_yaml_tab_character_error(self):
        """Test friendly error handling for tab characters in YAML."""
        yaml_content = """version: "1.0"
github:
\ttoken: "${GITHUB_TOKEN}"  # This line has a tab character
\towner: "test-owner"
repositories:
  - name: "test-repo"
    path: "/path/to/repo"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                ConfigLoader.load(temp_path)

            error_msg = str(exc_info.value)
            assert "❌ YAML configuration error" in error_msg
            assert "🚫 Tab characters are not allowed" in error_msg
            assert "Fix: Replace all tab characters with spaces" in error_msg
            assert "line 3" in error_msg
            assert temp_path.name in error_msg
        finally:
            temp_path.unlink()

    def test_yaml_missing_colon_error(self):
        """Test friendly error handling for missing colon."""
        yaml_content = """version: "1.0"
github
  token: "${GITHUB_TOKEN}"
  owner: "test-owner"
repositories:
  - name: "test-repo"
    path: "/path/to/repo"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                ConfigLoader.load(temp_path)

            error_msg = str(exc_info.value)
            assert "❌ YAML configuration error" in error_msg
            assert (
                "🚫 Missing colon (:) after a key name!" in error_msg
                or "🚫 Invalid YAML syntax" in error_msg
            )
            assert "Fix:" in error_msg
            assert temp_path.name in error_msg
        finally:
            temp_path.unlink()

    def test_yaml_unclosed_quote_error(self):
        """Test friendly error handling for unclosed quotes."""
        yaml_content = """version: "1.0"
github:
  token: "${GITHUB_TOKEN}
  owner: "test-owner"
repositories:
  - name: "test-repo"
    path: "/path/to/repo"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                ConfigLoader.load(temp_path)

            error_msg = str(exc_info.value)
            assert "❌ YAML configuration error" in error_msg
            assert "🚫" in error_msg  # Should have some error indicator
            assert "💡" in error_msg  # Should have some fix suggestion
            assert temp_path.name in error_msg
        finally:
            temp_path.unlink()

    def test_yaml_invalid_indentation_error(self):
        """Test friendly error handling for invalid indentation."""
        yaml_content = """version: "1.0"
github:
  token: "${GITHUB_TOKEN}"
owner: "test-owner"  # This should be indented under github
repositories:
  - name: "test-repo"
    path: "/path/to/repo"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            # This may or may not trigger a YAML error depending on the parser
            # But if it does, we should get a friendly message
            ConfigLoader.load(temp_path)
        except ValueError as e:
            error_msg = str(e)
            if "YAML configuration error" in error_msg:
                assert "❌" in error_msg
                assert "💡" in error_msg
                assert temp_path.name in error_msg
            else:
                # If it's not a YAML error, that's also valid for this test case
                pass
        finally:
            temp_path.unlink()

    def test_yaml_unexpected_character_error(self):
        """Test friendly error handling for unexpected characters."""
        yaml_content = """version: "1.0"
github:
  token: ${GITHUB_TOKEN}@invalid  # @ character might cause issues
  owner: "test-owner"
repositories:
  - name: "test-repo"
    path: "/path/to/repo"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            # This may or may not trigger a YAML error
            ConfigLoader.load(temp_path)
        except ValueError as e:
            error_msg = str(e)
            if "YAML configuration error" in error_msg:
                assert "❌" in error_msg
                assert temp_path.name in error_msg
            # Environment variable errors are also expected
        finally:
            temp_path.unlink()

    def test_file_not_found_error(self):
        """Test error handling for missing files."""
        non_existent_path = Path("/path/that/does/not/exist.yaml")

        with pytest.raises(ValueError) as exc_info:
            ConfigLoader.load(non_existent_path)

        error_msg = str(exc_info.value)
        assert "Configuration file not found" in error_msg
        assert str(non_existent_path) in error_msg

    def test_yaml_error_includes_help_resources(self):
        """Test that YAML errors include helpful resources."""
        yaml_content = """version: "1.0"
github:
\ttoken: "invalid"  # Tab character
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                ConfigLoader.load(temp_path)

            error_msg = str(exc_info.value)
            assert "🔗 For YAML syntax help, visit:" in error_msg
            assert "yaml.org" in error_msg
            assert "online YAML validator" in error_msg
        finally:
            temp_path.unlink()

    def test_yaml_error_includes_line_column_info(self):
        """Test that YAML errors include line and column information."""
        yaml_content = """version: "1.0"
github:
\ttoken: "invalid"  # Tab at column 1
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                ConfigLoader.load(temp_path)

            error_msg = str(exc_info.value)
            # Should include line and column information
            assert "line 3" in error_msg
            assert "column" in error_msg
        finally:
            temp_path.unlink()
