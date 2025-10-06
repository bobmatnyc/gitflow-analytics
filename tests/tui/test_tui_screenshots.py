"""TUI screenshot testing for GitFlow Analytics."""

import asyncio
import pytest
from pathlib import Path

from gitflow_analytics.tui.app import GitFlowAnalyticsApp


@pytest.mark.asyncio
async def test_tui_screenshots():
    """Test TUI by taking screenshots of different screens."""
    app = GitFlowAnalyticsApp()

    # Create screenshots directory
    screenshots_dir = Path("tests/screenshots")
    screenshots_dir.mkdir(exist_ok=True)

    async with app.run_test(size=(120, 40)) as pilot:
        # Take screenshot of initial screen
        screenshot = pilot.app.export_screenshot()
        (screenshots_dir / "01_startup.svg").write_text(screenshot)
        print("📸 Captured startup screen")

        # Simulate some navigation
        await pilot.press("f1")  # Help screen
        await asyncio.sleep(0.5)  # Wait for screen to update

        screenshot = pilot.app.export_screenshot()
        (screenshots_dir / "02_help.svg").write_text(screenshot)
        print("📸 Captured help screen")

        await pilot.press("escape")  # Back to main
        await asyncio.sleep(0.5)

        screenshot = pilot.app.export_screenshot()
        (screenshots_dir / "03_main.svg").write_text(screenshot)
        print("📸 Captured main screen")

        # Test config loading screen
        await pilot.press("ctrl+o")  # Open config (if implemented)
        await asyncio.sleep(0.5)

        screenshot = pilot.app.export_screenshot()
        (screenshots_dir / "04_config.svg").write_text(screenshot)
        print("📸 Captured config screen")

        await pilot.press("ctrl+q")  # Quit

    print("🎉 Screenshot tests completed!")


@pytest.mark.asyncio
async def test_tui_with_real_config():
    """Test TUI with a real configuration file."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("⚠️  No config.yaml found, skipping real config test")
        return

    app = GitFlowAnalyticsApp()
    app.config_path = config_path

    async with app.run_test(size=(120, 40)) as pilot:
        # Wait for config to load
        await asyncio.sleep(1.0)

        # Take screenshot with loaded config
        screenshot = pilot.app.export_screenshot()
        screenshots_dir = Path("tests/screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        (screenshots_dir / "05_with_config.svg").write_text(screenshot)
        print("📸 Captured TUI with real config")

        await pilot.press("ctrl+q")  # Quit

    print("✅ Real config test completed!")


if __name__ == "__main__":

    async def main():
        await test_tui_screenshots()
        await test_tui_with_real_config()

    asyncio.run(main())
