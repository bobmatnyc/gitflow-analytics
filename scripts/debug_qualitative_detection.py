#!/usr/bin/env python3
"""Test script to verify qualitative configuration detection in both locations."""

from pathlib import Path

from src.gitflow_analytics.config.loader import ConfigLoader


def test_qualitative_detection():
    """Test qualitative config detection in nested location."""
    config_path = Path("/Users/masa/Clients/EWTN/gfa/config.yaml")

    print("Loading configuration...")
    cfg = ConfigLoader.load(config_path)

    print("\n=== Configuration Structure ===")
    print(f"Top-level qualitative exists: {cfg.qualitative is not None}")
    if cfg.qualitative:
        print(f"  - Enabled: {cfg.qualitative.enabled}")
        print(f"  - Model: {cfg.qualitative.model}")

    print(f"\nAnalysis has qualitative attribute: {hasattr(cfg.analysis, 'qualitative')}")
    if hasattr(cfg.analysis, "qualitative"):
        print(f"Analysis.qualitative exists: {cfg.analysis.qualitative is not None}")
        if cfg.analysis.qualitative:
            print(f"  - Enabled: {cfg.analysis.qualitative.enabled}")
            print(f"  - Model: {cfg.analysis.qualitative.model}")
            print(f"  - API Base URL: {cfg.analysis.qualitative.api_base_url}")

    # Test helper functions (as they would be in CLI)
    def is_qualitative_enabled() -> bool:
        """Check if qualitative analysis is enabled in either location."""
        if cfg.qualitative and cfg.qualitative.enabled:
            return True
        if (
            hasattr(cfg.analysis, "qualitative")
            and cfg.analysis.qualitative
            and cfg.analysis.qualitative.enabled
        ):
            return True
        return False

    def get_qualitative_config():
        """Get qualitative config from either top-level or nested location."""
        if cfg.qualitative:
            return cfg.qualitative
        if hasattr(cfg.analysis, "qualitative") and cfg.analysis.qualitative:
            return cfg.analysis.qualitative
        return None

    print("\n=== Detection Results ===")
    print(f"is_qualitative_enabled(): {is_qualitative_enabled()}")
    qual_config = get_qualitative_config()
    print(f"get_qualitative_config(): {qual_config is not None}")
    if qual_config:
        print(f"  - Config model: {qual_config.model}")
        print(f"  - Config enabled: {qual_config.enabled}")

    print("\n=== Expected Behavior ===")
    if is_qualitative_enabled():
        print("✅ Token/cost tracking warning should NOT appear")
        print("✅ Qualitative analysis should be available")
    else:
        print("❌ Token/cost tracking warning WILL appear")
        print("❌ Qualitative analysis will NOT be available")


if __name__ == "__main__":
    test_qualitative_detection()
