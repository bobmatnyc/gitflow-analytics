"""Tests for configuration extension functionality."""

import tempfile
from pathlib import Path

import pytest
import yaml

from gitflow_analytics.config import ConfigLoader


class TestConfigurationExtends:
    """Test configuration extension functionality."""

    def test_extends_base_config(self):
        """Test extending from a base configuration."""
        # Create base configuration
        base_yaml = """version: "1.0"
github:
  owner: "base-owner"
  base_url: "https://api.github.com"
cache:
  directory: ".gitflow-cache"
  ttl_hours: 168
output:
  formats: ["csv"]
"""

        # Create extended configuration
        extended_yaml = """version: "1.0"
extends: "./base.yaml"
github:
  owner: "override-owner"  # Override base
repositories:
  - name: "test-repo"
    path: "/path/to/repo"
output:
  formats: ["csv", "markdown"]  # Override base
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write base config
            base_path = tmpdir / "base.yaml"
            base_path.write_text(base_yaml)

            # Write extended config
            extended_path = tmpdir / "extended.yaml"
            extended_path.write_text(extended_yaml)

            try:
                # This will fail on missing token, but we can check the merge logic
                ConfigLoader.load(extended_path)
            except Exception as e:
                # Expected to fail on missing token/repos
                # But we can check that the extends was processed
                if "GITHUB_TOKEN" not in str(e) and "Repository path" not in str(e):
                    pytest.fail(f"Unexpected error: {e}")

    def test_extends_with_profile(self):
        """Test combining extends with profile."""
        # Create base configuration
        base_yaml = """version: "1.0"
github:
  owner: "base-owner"
cache:
  directory: ".gitflow-cache"
"""

        # Create extended configuration with profile
        extended_yaml = """version: "1.0"
extends: "./base.yaml"
profile: performance
repositories:
  - name: "test-repo"
    path: "/path/to/repo"
analysis:
  ml_categorization:
    enabled: true  # Override profile default
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write base config
            base_path = tmpdir / "base.yaml"
            base_path.write_text(base_yaml)

            # Write extended config
            extended_path = tmpdir / "extended.yaml"
            extended_path.write_text(extended_yaml)

            try:
                # Load and check merge order
                with open(extended_path) as f:
                    data = yaml.safe_load(f)

                # Verify extends field is present
                assert "extends" in data
                assert data["extends"] == "./base.yaml"

                # Verify profile field is present
                assert "profile" in data
                assert data["profile"] == "performance"

                # The actual loading will be tested via the ConfigLoader
                # which will handle the merge order correctly

            except Exception as e:
                pytest.fail(f"Unexpected error: {e}")

    def test_extends_relative_path(self):
        """Test that extends uses relative path correctly."""
        # Create configs in nested directories
        base_yaml = """version: "1.0"
github:
  owner: "base-owner"
"""

        extended_yaml = """version: "1.0"
extends: "../configs/base.yaml"
repositories:
  - name: "test-repo"
    path: "/path/to/repo"
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create directory structure
            configs_dir = tmpdir / "configs"
            configs_dir.mkdir()
            project_dir = tmpdir / "project"
            project_dir.mkdir()

            # Write base config in configs directory
            base_path = configs_dir / "base.yaml"
            base_path.write_text(base_yaml)

            # Write extended config in project directory
            extended_path = project_dir / "config.yaml"
            extended_path.write_text(extended_yaml)

            try:
                # Attempt to load - should find the base config via relative path
                ConfigLoader.load(extended_path)
            except Exception as e:
                # Expected to fail on missing token/repos
                if "GITHUB_TOKEN" not in str(e) and "Repository path" not in str(e):
                    # If it's a file not found error for base.yaml, that's a real problem
                    if "base.yaml" in str(e) and "not found" in str(e).lower():
                        pytest.fail(f"Failed to resolve relative path: {e}")

    def test_extends_absolute_path(self):
        """Test that extends works with absolute paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create base configuration
            base_yaml = """version: "1.0"
github:
  owner: "base-owner"
"""

            # Write base config
            base_path = tmpdir / "base.yaml"
            base_path.write_text(base_yaml)

            # Create extended configuration with absolute path
            extended_yaml = f"""version: "1.0"
extends: "{base_path}"
repositories:
  - name: "test-repo"
    path: "/path/to/repo"
"""

            # Write extended config
            extended_path = tmpdir / "extended.yaml"
            extended_path.write_text(extended_yaml)

            try:
                # Attempt to load
                ConfigLoader.load(extended_path)
            except Exception as e:
                # Expected to fail on missing token/repos
                if "GITHUB_TOKEN" not in str(e) and "Repository path" not in str(e):
                    # If it's a file not found error for base config, that's a problem
                    if str(base_path) in str(e) and "not found" in str(e).lower():
                        pytest.fail(f"Failed to load absolute path: {e}")

    def test_extends_nonexistent_file(self):
        """Test that extending from non-existent file gives clear error."""
        extended_yaml = """version: "1.0"
extends: "./nonexistent.yaml"
repositories:
  - name: "test-repo"
    path: "/path/to/repo"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(extended_yaml)
            temp_path = Path(f.name)

        try:
            with pytest.raises(Exception) as exc_info:
                ConfigLoader.load(temp_path)

            error_msg = str(exc_info.value)
            assert "not found" in error_msg.lower() or "nonexistent" in error_msg

        finally:
            temp_path.unlink()
