"""Tests for PM integration environment variable resolution."""

import os
import tempfile
from pathlib import Path

from gitflow_analytics.config.loader import ConfigLoader


def test_pm_integration_env_resolution():
    """Test that environment variables are resolved in PM integration config."""

    # Set test environment variables
    os.environ["TEST_JIRA_USER"] = "test.user@example.com"
    os.environ["TEST_JIRA_TOKEN"] = "test-api-token-12345"
    os.environ["TEST_JIRA_URL"] = "https://test.atlassian.net"
    os.environ["TEST_CUSTOM_FIELD"] = "customfield_99999"

    # Create a test config with environment variables
    test_config = """
analysis:
  time:
    weeks: 2

repositories:
  - name: "test-repo"
    path: "/tmp/test-repo"
    include: true

pm_integration:
  enabled: true
  primary_platform: "jira"
  platforms:
    jira:
      enabled: true
      platform_type: "jira"
      config:
        base_url: "${TEST_JIRA_URL}"
        username: "${TEST_JIRA_USER}"
        api_token: "${TEST_JIRA_TOKEN}"
        story_point_fields:
          - "${TEST_CUSTOM_FIELD}"
          - "Story Points"
        nested_config:
          some_user: "${TEST_JIRA_USER}"
          some_token: "${TEST_JIRA_TOKEN}"
"""

    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(test_config)
        config_path = Path(f.name)

    try:
        # Load config
        config = ConfigLoader.load(config_path)

        # Verify PM integration exists and is enabled
        assert config.pm_integration is not None, "PM integration should exist"
        assert config.pm_integration.enabled is True, "PM integration should be enabled"
        assert config.pm_integration.primary_platform == "jira", "Primary platform should be jira"

        # Get JIRA platform config
        jira_config = config.pm_integration.platforms.get("jira")
        assert jira_config is not None, "JIRA platform should exist"
        assert jira_config.enabled is True, "JIRA should be enabled"

        # Get the resolved config dictionary
        jira_cfg = jira_config.config

        # Check base_url resolution
        assert jira_cfg.get("base_url") == "https://test.atlassian.net", (
            f"base_url not resolved correctly: {jira_cfg.get('base_url')}"
        )

        # Check username resolution
        assert jira_cfg.get("username") == "test.user@example.com", (
            f"username not resolved correctly: {jira_cfg.get('username')}"
        )

        # Check api_token resolution
        assert jira_cfg.get("api_token") == "test-api-token-12345", (
            f"api_token not resolved correctly: {jira_cfg.get('api_token')}"
        )

        # Check list with environment variable resolution
        story_point_fields = jira_cfg.get("story_point_fields", [])
        assert "customfield_99999" in story_point_fields, (
            f"Custom field not resolved in list: {story_point_fields}"
        )

        # Check nested dictionary resolution
        nested = jira_cfg.get("nested_config", {})
        assert nested.get("some_user") == "test.user@example.com", (
            f"Nested user not resolved: {nested.get('some_user')}"
        )
        assert nested.get("some_token") == "test-api-token-12345", (
            f"Nested token not resolved: {nested.get('some_token')}"
        )

    finally:
        # Clean up
        if config_path.exists():
            config_path.unlink()
        # Clean up environment variables
        for var in ["TEST_JIRA_USER", "TEST_JIRA_TOKEN", "TEST_JIRA_URL", "TEST_CUSTOM_FIELD"]:
            if var in os.environ:
                del os.environ[var]


def test_pm_integration_missing_env_var():
    """Test handling of missing environment variables."""

    # Ensure the environment variable is not set
    if "MISSING_VAR" in os.environ:
        del os.environ["MISSING_VAR"]

    # Create a test config with a missing environment variable
    test_config = """
analysis:
  time:
    weeks: 2

repositories:
  - name: "test-repo"
    path: "/tmp/test-repo"

pm_integration:
  enabled: true
  platforms:
    jira:
      config:
        username: "${MISSING_VAR}"
        api_token: "valid-token"
"""

    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(test_config)
        config_path = Path(f.name)

    try:
        # Load config - should not fail, but return None for missing var
        config = ConfigLoader.load(config_path)

        jira_config = config.pm_integration.platforms.get("jira")
        jira_cfg = jira_config.config

        # Missing environment variable should resolve to None
        assert jira_cfg.get("username") is None, (
            f"Missing env var should resolve to None, got: {jira_cfg.get('username')}"
        )

        # Valid token should still be there
        assert jira_cfg.get("api_token") == "valid-token", (
            f"Valid token should be preserved: {jira_cfg.get('api_token')}"
        )

    finally:
        # Clean up
        if config_path.exists():
            config_path.unlink()


def test_backward_compatibility_pm_config():
    """Test that old PM config still works with env resolution."""

    # Set test environment variables
    os.environ["TEST_JIRA_USER_OLD"] = "old.user@example.com"
    os.environ["TEST_JIRA_TOKEN_OLD"] = "old-api-token-54321"

    # Create a test config with old PM structure
    test_config = """
analysis:
  time:
    weeks: 2

repositories:
  - name: "test-repo"
    path: "/tmp/test-repo"

pm:
  jira:
    enabled: true
    base_url: "https://old.atlassian.net"
    username: "${TEST_JIRA_USER_OLD}"
    api_token: "${TEST_JIRA_TOKEN_OLD}"
    story_point_fields:
      - "customfield_10016"
"""

    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(test_config)
        config_path = Path(f.name)

    try:
        # Load config
        config = ConfigLoader.load(config_path)

        # Check old PM config structure
        assert hasattr(config, "pm"), "Should have pm attribute for backward compatibility"
        assert config.pm is not None, "PM config should exist"
        assert hasattr(config.pm, "jira"), "Should have jira in PM config"

        jira = config.pm.jira
        assert jira.enabled is True, "JIRA should be enabled"
        assert jira.base_url == "https://old.atlassian.net", "Base URL should be preserved"

        # Check environment variable resolution in old structure
        assert jira.username == "old.user@example.com", (
            f"Username not resolved in old structure: {jira.username}"
        )
        assert jira.api_token == "old-api-token-54321", (
            f"API token not resolved in old structure: {jira.api_token}"
        )

    finally:
        # Clean up
        if config_path.exists():
            config_path.unlink()
        # Clean up environment variables
        for var in ["TEST_JIRA_USER_OLD", "TEST_JIRA_TOKEN_OLD"]:
            if var in os.environ:
                del os.environ[var]
