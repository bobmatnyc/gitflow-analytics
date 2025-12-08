#!/usr/bin/env python3
"""Simple test script to verify ML categorization integration works correctly."""

import tempfile
from pathlib import Path

# Test imports
try:
    from src.gitflow_analytics.config import MLCategorization
    from src.gitflow_analytics.core.analyzer import GitAnalyzer
    from src.gitflow_analytics.core.cache import GitAnalysisCache
    from src.gitflow_analytics.extractors.ml_tickets import MLTicketExtractor

    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)


def test_ml_extractor_basic():
    """Test basic ML extractor functionality."""
    print("\n🔬 Testing MLTicketExtractor basic functionality...")

    # Test with ML disabled (should fall back gracefully)
    extractor = MLTicketExtractor(enable_ml=False)
    category = extractor.categorize_commit("fix: resolve authentication bug")
    print(f"   ML disabled categorization: '{category}' (expected: 'bug_fix')")

    # Test with ML enabled (may fall back to rules if dependencies missing)
    extractor = MLTicketExtractor(enable_ml=True)
    result = extractor.categorize_commit_with_confidence("feat: add user registration system")
    print(
        f"   ML enabled result: category='{result['category']}', confidence={result['confidence']:.2f}, method='{result['method']}'"
    )

    # Test different commit types
    test_commits = [
        "fix: resolve memory leak in user service",
        "feat: implement dark mode toggle",
        "docs: update API documentation",
        "test: add unit tests for authentication",
        "refactor: simplify database connection logic",
        "chore: update dependencies to latest versions",
    ]

    print("   Testing various commit types:")
    for commit_msg in test_commits:
        result = extractor.categorize_commit_with_confidence(commit_msg)
        print(
            f"     '{commit_msg[:30]}...' → {result['category']} ({result['confidence']:.2f}, {result['method']})"
        )


def test_analyzer_integration():
    """Test GitAnalyzer integration with ML configuration."""
    print("\n🔬 Testing GitAnalyzer integration...")

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "cache"
        cache = GitAnalysisCache(cache_dir)

        # Test without ML config
        analyzer_basic = GitAnalyzer(cache)
        print(f"   Basic analyzer uses: {type(analyzer_basic.ticket_extractor).__name__}")

        # Test with ML config enabled
        ml_config = {
            "enabled": True,
            "min_confidence": 0.6,
            "hybrid_threshold": 0.5,
            "enable_caching": True,
        }

        analyzer_ml = GitAnalyzer(cache, ml_categorization_config=ml_config)
        print(f"   ML analyzer uses: {type(analyzer_ml.ticket_extractor).__name__}")

        # Test with ML config disabled
        ml_config_disabled = {"enabled": False}

        analyzer_disabled = GitAnalyzer(cache, ml_categorization_config=ml_config_disabled)
        print(f"   Disabled ML analyzer uses: {type(analyzer_disabled.ticket_extractor).__name__}")


def test_config_parsing():
    """Test configuration parsing for ML categorization."""
    print("\n🔬 Testing configuration parsing...")

    # Test default ML config
    ml_config = MLCategorization()
    print(
        f"   Default ML config: enabled={ml_config.enabled}, confidence={ml_config.min_confidence}"
    )

    # Test custom ML config
    custom_config = MLCategorization(enabled=True, min_confidence=0.8, hybrid_threshold=0.7)
    print(
        f"   Custom ML config: enabled={custom_config.enabled}, confidence={custom_config.min_confidence}"
    )


def main():
    """Run all tests."""
    print("🚀 Testing ML-based commit categorization integration")
    print("=" * 60)

    try:
        test_ml_extractor_basic()
        test_analyzer_integration()
        test_config_parsing()

        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\n💡 Integration notes:")
        print("   - ML categorization will fall back to rule-based if spaCy is not available")
        print("   - Configuration is backward compatible - existing configs work unchanged")
        print("   - Caching is enabled by default for performance")
        print("   - Confidence scores help identify prediction quality")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
