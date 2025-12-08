#!/usr/bin/env python3
"""
Test script to verify the updated EWTN config works for story points tracking.

This script will:
1. Load the updated config
2. Verify JIRA integration is configured
3. Test story point pattern extraction
4. Check if the configuration is properly structured
"""

import os
import sys
from pathlib import Path

# Add the gitflow-analytics package to the path
sys.path.insert(0, "/Users/masa/Projects/managed/gitflow-analytics/src")

try:
    from gitflow_analytics.config import ConfigLoader
    from gitflow_analytics.core.cache import GitAnalysisCache
    from gitflow_analytics.extractors.story_points import StoryPointExtractor
    from gitflow_analytics.integrations.jira_integration import JIRAIntegration
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_config_loading():
    """Test loading the updated EWTN config."""
    config_path = "configs/ewtn-test-config.yaml"

    # Skip test if config file doesn't exist
    if not Path(config_path).exists():
        import pytest

        pytest.skip(f"Config file {config_path} not found")

    config = ConfigLoader.load(config_path)
    assert config is not None, "Config should load successfully"

    # Check JIRA configuration
    assert config.jira is not None, "JIRA config should be present"
    assert config.jira.base_url == "https://ewtn.atlassian.net"
    assert config.jira.access_user is not None
    assert config.jira.access_token is not None

    # Check JIRA integration
    assert config.jira_integration is not None, "JIRA integration config should be present"
    assert config.jira_integration.enabled is True
    assert config.jira_integration.fetch_story_points is True
    assert len(config.jira_integration.story_point_fields) > 0

    # Check story point patterns
    assert hasattr(config.analysis, "story_point_patterns")
    assert len(config.analysis.story_point_patterns) > 0


def test_story_point_extractor():
    """Test the story point extractor with the configured patterns."""
    config_path = "configs/ewtn-test-config.yaml"

    # Skip test if config file doesn't exist
    if not Path(config_path).exists():
        import pytest

        pytest.skip(f"Config file {config_path} not found")

    config = ConfigLoader.load(config_path)
    patterns = getattr(config.analysis, "story_point_patterns", None)
    if patterns:
        extractor = StoryPointExtractor(patterns=patterns)
        print(f"✅ Story point extractor created with {len(patterns)} patterns")
    else:
        extractor = StoryPointExtractor()  # Use default patterns
        print("⚠️  Using default patterns (config patterns not found)")

    # Test extraction with various text formats
    test_texts = [
        "RMVP-1030: Fix login issue [3 points]",
        "Story Points: 5 - Update user interface",
        "SP: 8 - Refactor authentication module",
        "Points: 2 - Bug fix for payment processing",
        "RMVP-1075 (5 story points) - New feature implementation",
        "estimate: 13 - Large refactoring task",
        "SP5 - Quick bug fix",
        "#8sp - Performance optimization",
    ]

    successful_extractions = 0
    for text in test_texts:
        points = extractor.extract_from_text(text)
        if points:
            print(f"✅ '{text[:50]}...' → {points} points")
            successful_extractions += 1
        else:
            print(f"❌ '{text[:50]}...' → No points extracted")

    print(f"\n📊 Extraction success rate: {successful_extractions}/{len(test_texts)}")
    return successful_extractions > 0


def test_jira_integration_setup():
    """Test JIRA integration setup (without making actual API calls)."""
    print("\n🔍 Testing JIRA Integration Setup...")

    config_path = "configs/ewtn-test-config.yaml"

    # Skip test if config file doesn't exist
    if not Path(config_path).exists():
        import pytest

        pytest.skip(f"Config file {config_path} not found")

    config = ConfigLoader.load(config_path)

    try:
        if not config.jira or not config.jira_integration:
            print("❌ JIRA configuration missing")
            return False

        # Check environment variables
        jira_user = os.getenv("JIRA_ACCESS_USER")
        jira_token = os.getenv("JIRA_ACCESS_TOKEN")

        if not jira_user or not jira_token:
            print("⚠️  JIRA credentials not found in environment variables")
            print("   Set JIRA_ACCESS_USER and JIRA_ACCESS_TOKEN to test JIRA integration")
            return False

        # Try to create JIRA integration instance
        cache_dir = Path("./test-cache")
        cache_dir.mkdir(exist_ok=True)
        cache = GitAnalysisCache(cache_dir)

        JIRAIntegration(
            base_url=config.jira.base_url,
            username=jira_user,
            api_token=jira_token,
            cache=cache,
            story_point_fields=config.jira_integration.story_point_fields,
        )

        print("✅ JIRA integration instance created successfully")
        print(f"   - Base URL: {config.jira.base_url}")
        print(f"   - Story point fields: {config.jira_integration.story_point_fields}")

        return True

    except Exception as e:
        print(f"❌ Error setting up JIRA integration: {e}")
        return False


def main():
    """Run all configuration tests."""
    print("🚀 Testing Updated EWTN Config for Story Points Tracking")
    print("=" * 60)

    # Test 1: Config loading
    config = test_config_loading()
    if not config:
        print("\n❌ Config loading failed - cannot continue")
        return False

    # Test 2: Story point extractor
    extractor_works = test_story_point_extractor(config)

    # Test 3: JIRA integration setup
    jira_works = test_jira_integration_setup(config)

    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   ✅ Config Loading: {'PASS' if config else 'FAIL'}")
    print(f"   ✅ Story Point Extractor: {'PASS' if extractor_works else 'FAIL'}")
    print(f"   ✅ JIRA Integration Setup: {'PASS' if jira_works else 'FAIL'}")

    if config and extractor_works:
        print("\n🎉 Configuration is ready for story points tracking!")
        print("\nNext steps:")
        print("1. Set JIRA_ACCESS_USER and JIRA_ACCESS_TOKEN environment variables")
        print(
            "2. Run analysis with: gitflow-analytics analyze --config configs/ewtn-test-config.yaml"
        )
        print("3. Check reports for story points data")
        return True
    else:
        print("\n❌ Configuration needs fixes before story points tracking will work")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
