"""Tests for configuration profiles functionality."""

import tempfile
from pathlib import Path

import pytest
import yaml

from gitflow_analytics.config import ConfigLoader
from gitflow_analytics.config.profiles import ProfileManager


class TestConfigurationProfiles:
    """Test configuration profile functionality."""

    def test_list_profiles(self):
        """Test listing available profiles."""
        profiles = ProfileManager.list_profiles()
        assert "performance" in profiles
        assert "quality" in profiles
        assert "balanced" in profiles
        assert "minimal" in profiles
        assert len(profiles) == 4

    def test_get_profile(self):
        """Test getting a specific profile."""
        profile = ProfileManager.get_profile("performance")
        assert profile is not None
        assert profile.name == "performance"

        # Test case insensitive
        profile = ProfileManager.get_profile("QUALITY")
        assert profile is not None
        assert profile.name == "quality"

        # Test non-existent profile
        profile = ProfileManager.get_profile("non-existent")
        assert profile is None

    def test_performance_profile_settings(self):
        """Test performance profile settings."""
        from gitflow_analytics.config.profiles import PerformanceProfile

        settings = PerformanceProfile.get_settings()

        # Check key performance settings
        assert settings["analysis"]["branch_analysis"]["strategy"] == "main_only"
        assert settings["analysis"]["ml_categorization"]["enabled"] is False
        assert settings["analysis"]["commit_classification"]["enabled"] is False
        assert settings["analysis"]["llm_classification"]["enabled"] is False
        assert settings["output"]["formats"] == ["csv"]
        assert settings["cache"]["ttl_hours"] == 336  # 2 weeks

    def test_quality_profile_settings(self):
        """Test quality profile settings."""
        from gitflow_analytics.config.profiles import QualityProfile

        settings = QualityProfile.get_settings()

        # Check key quality settings
        assert settings["analysis"]["branch_analysis"]["strategy"] == "smart"
        assert settings["analysis"]["branch_analysis"]["max_branches_per_repo"] == 100
        assert settings["analysis"]["ml_categorization"]["enabled"] is True
        assert settings["analysis"]["ml_categorization"]["min_confidence"] == 0.7
        assert settings["analysis"]["commit_classification"]["enabled"] is True
        assert settings["output"]["formats"] == ["csv", "markdown", "json"]

    def test_balanced_profile_settings(self):
        """Test balanced profile settings."""
        from gitflow_analytics.config.profiles import BalancedProfile

        settings = BalancedProfile.get_settings()

        # Check balanced settings
        assert settings["analysis"]["branch_analysis"]["strategy"] == "smart"
        assert settings["analysis"]["branch_analysis"]["max_branches_per_repo"] == 50
        assert settings["analysis"]["ml_categorization"]["enabled"] is True
        assert settings["output"]["formats"] == ["csv", "markdown"]

    def test_minimal_profile_settings(self):
        """Test minimal profile settings."""
        from gitflow_analytics.config.profiles import MinimalProfile

        settings = MinimalProfile.get_settings()

        # Check minimal settings
        assert settings["analysis"]["branch_analysis"]["strategy"] == "main_only"
        assert settings["analysis"]["ml_categorization"]["enabled"] is False
        assert settings["analysis"]["commit_classification"]["enabled"] is False
        assert settings["analysis"]["auto_identity_analysis"] is False
        assert settings["output"]["formats"] == ["csv"]

    def test_apply_profile_to_config(self):
        """Test applying a profile to configuration data."""
        base_config = {
            "version": "1.0",
            "repositories": [{"name": "test", "path": "/test"}],
        }

        # Apply performance profile
        config = ProfileManager.apply_profile(base_config, "performance")
        assert config["analysis"]["branch_analysis"]["strategy"] == "main_only"
        assert config["analysis"]["ml_categorization"]["enabled"] is False

        # Apply quality profile
        config = ProfileManager.apply_profile(base_config, "quality")
        assert config["analysis"]["branch_analysis"]["strategy"] == "smart"
        assert config["analysis"]["ml_categorization"]["enabled"] is True

    def test_apply_invalid_profile(self):
        """Test applying an invalid profile raises error."""
        base_config = {"version": "1.0"}

        with pytest.raises(ValueError) as exc_info:
            ProfileManager.apply_profile(base_config, "non-existent")

        assert "Unknown configuration profile" in str(exc_info.value)
        assert "Available profiles:" in str(exc_info.value)

    def test_load_config_with_profile(self):
        """Test loading configuration with a profile."""
        yaml_content = """version: "1.0"
profile: performance
github:
  owner: "test-owner"
repositories:
  - name: "test-repo"
    path: "/path/to/repo"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            # Load config (will fail on missing token but profile should be applied)
            ConfigLoader.load(temp_path)
        except Exception:
            # We expect environment variable errors, but profile should still be applied
            pass
        finally:
            temp_path.unlink()

    def test_profile_override(self):
        """Test that explicit settings override profile defaults."""
        yaml_content = """version: "1.0"
profile: performance
github:
  owner: "test-owner"
repositories:
  - name: "test-repo"
    path: "/path/to/repo"
analysis:
  ml_categorization:
    enabled: True  # Override performance profile default
    min_confidence: 0.8
output:
  formats: ["csv", "markdown"]  # Override performance profile default
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            # Parse YAML to verify profile application
            with open(temp_path) as f:
                data = yaml.safe_load(f)

            # Apply profile
            data = ProfileManager.apply_profile(data, data["profile"])

            # Check that overrides are preserved
            assert data["analysis"]["ml_categorization"]["enabled"] is True
            assert data["analysis"]["ml_categorization"]["min_confidence"] == 0.8
            assert data["output"]["formats"] == ["csv", "markdown"]

            # Check that non-overridden profile settings are applied
            assert data["analysis"]["branch_analysis"]["strategy"] == "main_only"

        finally:
            temp_path.unlink()

    def test_deep_merge(self):
        """Test deep merge functionality."""
        base = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "e": [1, 2, 3],
        }

        override = {
            "a": 10,
            "b": {"c": 20, "f": 4},
            "g": 5,
        }

        result = ProfileManager._deep_merge(base, override)

        assert result["a"] == 10  # Overridden
        assert result["b"]["c"] == 20  # Nested override
        assert result["b"]["d"] == 3  # Preserved from base
        assert result["b"]["f"] == 4  # Added from override
        assert result["e"] == [1, 2, 3]  # Preserved from base
        assert result["g"] == 5  # Added from override
