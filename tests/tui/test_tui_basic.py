"""Basic TUI testing for GitFlow Analytics."""

import asyncio
import pytest
from pathlib import Path

from gitflow_analytics.tui.app import GitFlowAnalyticsApp


class TestGitFlowAnalyticsTUI:
    """Test cases for GitFlow Analytics TUI."""

    @pytest.mark.asyncio
    async def test_app_startup(self):
        """Test that the TUI app starts without errors."""
        app = GitFlowAnalyticsApp()

        async with app.run_test(size=(80, 24)) as pilot:
            # Test that the app starts
            assert app.is_running

            # Test that we can take a screenshot
            screenshot = pilot.app.export_screenshot()
            assert screenshot is not None
            assert len(screenshot) > 0

            # Test basic navigation
            await pilot.press("ctrl+q")  # Quit the app

    @pytest.mark.asyncio
    async def test_config_loading_screen(self):
        """Test the configuration loading screen."""
        app = GitFlowAnalyticsApp()

        async with app.run_test(size=(80, 24)) as pilot:
            # Test keyboard shortcuts
            await pilot.press("f1")  # Help
            await asyncio.sleep(0.1)
            await pilot.press("escape")  # Back
            await asyncio.sleep(0.1)

            await pilot.press("ctrl+q")  # Quit

    @pytest.mark.asyncio
    async def test_tui_with_config(self):
        """Test TUI with pre-loaded configuration."""
        app = GitFlowAnalyticsApp()

        # Create a minimal test config
        test_config_path = Path("test_config.yaml")
        test_config_content = """
repositories:
  - name: "test-repo"
    path: "."
    project_key: "TEST"
analysis:
  weeks: 4
"""
        test_config_path.write_text(test_config_content)

        try:
            # Load config
            from gitflow_analytics.config import ConfigLoader

            app.config = ConfigLoader.load(test_config_path)
            app.config_path = test_config_path

            async with app.run_test(size=(80, 24)) as pilot:
                # Test that config is recognized
                assert app.config_path is not None
                assert app.config is not None

                await pilot.press("ctrl+q")  # Quit
        finally:
            # Clean up test config
            if test_config_path.exists():
                test_config_path.unlink()

    @pytest.mark.asyncio
    async def test_tui_error_handling(self):
        """Test TUI error handling with invalid inputs."""
        app = GitFlowAnalyticsApp()

        async with app.run_test(size=(80, 24)) as pilot:
            # Test invalid key presses don't crash
            await pilot.press("ctrl+z")  # Invalid shortcut
            await asyncio.sleep(0.1)
            await pilot.press("alt+x")  # Invalid shortcut
            await asyncio.sleep(0.1)

            assert app.is_running, "App crashed on invalid input"

            await pilot.press("ctrl+q")

    @pytest.mark.asyncio
    async def test_tui_screenshots(self):
        """Test TUI screenshot functionality."""
        app = GitFlowAnalyticsApp()

        async with app.run_test(size=(100, 30)) as pilot:
            # Take initial screenshot
            screenshot1 = pilot.app.export_screenshot()
            assert screenshot1 is not None

            # Navigate and take another screenshot
            await pilot.press("f1")  # Help
            await asyncio.sleep(0.2)
            screenshot2 = pilot.app.export_screenshot()
            assert screenshot2 is not None

            # Screenshots should be different
            assert screenshot1 != screenshot2

            await pilot.press("ctrl+q")
