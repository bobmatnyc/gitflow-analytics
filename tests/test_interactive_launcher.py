"""Tests for interactive launcher and identity detection.

Validates module imports, configuration dataclasses, and
the identity analysis pipeline for the interactive launcher.
"""

from datetime import datetime

import pytest


@pytest.mark.unit
class TestInteractiveLauncherImports:
    """Test that interactive launcher modules can be imported."""

    def test_interactive_launcher_importable(self):
        """Test that interactive launcher module imports successfully."""
        from gitflow_analytics.cli_wizards.run_launcher import (  # noqa: F401
            InteractiveLauncher,
            run_interactive_launcher,
        )

    def test_llm_identity_analyzer_importable(self):
        """Test that LLM identity analyzer imports successfully."""
        from gitflow_analytics.identity_llm.analyzer import LLMIdentityAnalyzer  # noqa: F401

    def test_launcher_preferences_schema_importable(self):
        """Test that launcher preferences schema imports successfully."""
        from gitflow_analytics.config.schema import LauncherPreferences  # noqa: F401


@pytest.mark.unit
class TestLLMIdentityAnalyzer:
    """Tests for the LLM identity analyzer configuration."""

    def test_default_confidence_threshold_is_ninety_percent(self):
        """Test that LLM analyzer uses 90% confidence threshold by default."""
        from gitflow_analytics.identity_llm.analyzer import LLMIdentityAnalyzer

        analyzer = LLMIdentityAnalyzer()
        assert analyzer.confidence_threshold == 0.9


@pytest.mark.unit
class TestLauncherPreferences:
    """Tests for the launcher preferences dataclass."""

    def test_preferences_stores_selected_repos(self):
        """Test launcher preferences stores repository selections."""
        from gitflow_analytics.config.schema import LauncherPreferences

        prefs = LauncherPreferences(
            last_selected_repos=["repo1", "repo2"],
            default_weeks=8,
            auto_clear_cache=True,
        )

        assert prefs.last_selected_repos == ["repo1", "repo2"]
        assert prefs.default_weeks == 8
        assert prefs.auto_clear_cache is True

    def test_skip_identity_analysis_defaults_to_false(self):
        """Test that skip_identity_analysis defaults to False."""
        from gitflow_analytics.config.schema import LauncherPreferences

        prefs = LauncherPreferences()
        assert prefs.skip_identity_analysis is False


@pytest.mark.unit
class TestIdentityAnalysisResult:
    """Tests for identity analysis result model."""

    def test_manual_mappings_include_confidence_and_reasoning(self):
        """Test that manual mappings include confidence score and reasoning."""
        from gitflow_analytics.identity_llm.models import (
            DeveloperAlias,
            DeveloperCluster,
            IdentityAnalysisResult,
        )

        alias = DeveloperAlias(
            name="J. Doe",
            email="j.doe@example.com",
            commit_count=5,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            repositories={"repo1"},
        )

        cluster = DeveloperCluster(
            canonical_name="John Doe",
            canonical_email="john@example.com",
            aliases=[alias],
            confidence=0.95,
            reasoning="Same person based on name similarity and commit patterns",
            total_commits=15,
            total_story_points=0,
        )

        result = IdentityAnalysisResult(
            clusters=[cluster],
            unresolved_identities=[],
        )

        mappings = result.get_manual_mappings()

        assert len(mappings) == 1
        assert "confidence" in mappings[0]
        assert mappings[0]["confidence"] == 0.95
        assert "reasoning" in mappings[0]
        assert "primary_email" in mappings[0]


@pytest.mark.unit
class TestCLICommandRegistration:
    """Tests for CLI command registration."""

    def test_run_command_is_registered(self):
        """Test that 'run' command is registered in the CLI."""
        from gitflow_analytics.cli import cli

        assert "run" in cli.commands
