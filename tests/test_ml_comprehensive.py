#!/usr/bin/env python3
"""Comprehensive test suite for ML-based commit categorization system.

This script performs extensive testing including:
- Accuracy metrics with diverse commit messages
- Performance benchmarking
- Caching effectiveness
- Backward compatibility
- Configuration testing
- Graceful fallback validation
"""

import json
import statistics
import tempfile
import time
from pathlib import Path

# Test imports
try:
    from src.gitflow_analytics.config import MLCategorization
    from src.gitflow_analytics.core.analyzer import GitAnalyzer
    from src.gitflow_analytics.core.cache import GitAnalysisCache
    from src.gitflow_analytics.extractors.ml_tickets import MLPredictionCache, MLTicketExtractor
    from src.gitflow_analytics.extractors.tickets import TicketExtractor

    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test data with expected categories for accuracy measurement
TEST_COMMIT_MESSAGES = [
    # Feature commits
    ("feat: add user authentication system", "feature"),
    ("feature: implement dark mode toggle", "feature"),
    ("add new API endpoint for payments", "feature"),
    ("implement OAuth 2.0 integration", "feature"),
    ("new: add dashboard analytics", "feature"),
    # Bug fixes
    ("fix: resolve memory leak in user service", "bug_fix"),
    ("bugfix: handle null pointer exception", "bug_fix"),
    ("resolve authentication timeout issue", "bug_fix"),
    ("hotfix: critical security vulnerability", "bug_fix"),
    ("patch memory allocation bug", "bug_fix"),
    # Documentation
    ("docs: update API documentation", "documentation"),
    ("update README with installation instructions", "documentation"),
    ("add code comments for authentication module", "documentation"),
    ("document new configuration options", "documentation"),
    ("create developer guide", "documentation"),
    # Tests
    ("test: add unit tests for authentication", "test"),
    ("add integration tests for payment API", "test"),
    ("update test fixtures", "test"),
    ("create end-to-end test suite", "test"),
    ("fix failing test cases", "test"),
    # Refactoring
    ("refactor: simplify database connection logic", "refactor"),
    ("restructure authentication modules", "refactor"),
    ("clean up legacy code", "refactor"),
    ("extract common utilities", "refactor"),
    ("optimize query performance", "refactor"),
    # Maintenance/Chores
    ("chore: update dependencies to latest versions", "maintenance"),
    ("update build configuration", "maintenance"),
    ("bump version to 2.0.0", "maintenance"),
    ("configure CI/CD pipeline", "maintenance"),
    ("update linting rules", "maintenance"),
    # Style
    ("style: fix code formatting", "style"),
    ("format code with prettier", "style"),
    ("fix linting issues", "style"),
    ("standardize naming conventions", "style"),
    ("organize imports", "style"),
    # Build/CI
    ("build: update webpack configuration", "build"),
    ("ci: add automated testing", "build"),
    ("update Docker configuration", "build"),
    ("configure deployment pipeline", "build"),
    ("optimize build process", "build"),
    # Edge cases and ambiguous commits
    ("quick fix", "other"),
    ("update stuff", "other"),
    ("misc changes", "other"),
    ("WIP: working on new feature", "feature"),
    ("revert previous commit", "other"),
]


def test_categorization_accuracy():
    """Test ML categorization accuracy against expected results."""
    print("\n🎯 Testing categorization accuracy...")

    # Test both ML-enabled and rule-based extractors
    ml_extractor = MLTicketExtractor(enable_ml=True)
    rule_extractor = TicketExtractor()

    ml_results = []
    rule_results = []

    for message, expected in TEST_COMMIT_MESSAGES:
        # ML categorization (may fall back to rules)
        ml_result = ml_extractor.categorize_commit_with_confidence(message)
        ml_category = ml_result["category"]
        ml_correct = ml_category == expected
        ml_results.append(
            {
                "message": message,
                "expected": expected,
                "predicted": ml_category,
                "correct": ml_correct,
                "confidence": ml_result["confidence"],
                "method": ml_result["method"],
            }
        )

        # Rule-based categorization
        rule_category = rule_extractor.categorize_commit(message)
        rule_correct = rule_category == expected
        rule_results.append(
            {
                "message": message,
                "expected": expected,
                "predicted": rule_category,
                "correct": rule_correct,
            }
        )

    # Calculate accuracy metrics
    ml_accuracy = sum(1 for r in ml_results if r["correct"]) / len(ml_results)
    rule_accuracy = sum(1 for r in rule_results if r["correct"]) / len(rule_results)

    print(f"   ML System Accuracy: {ml_accuracy:.2%}")
    print(f"   Rule-based Accuracy: {rule_accuracy:.2%}")
    print(f"   Improvement: {ml_accuracy - rule_accuracy:+.2%}")

    # Method breakdown
    method_counts = {}
    for result in ml_results:
        method = result["method"]
        method_counts[method] = method_counts.get(method, 0) + 1

    print(f"   Method breakdown: {method_counts}")

    # Confidence analysis
    confidences = [r["confidence"] for r in ml_results]
    print(f"   Average confidence: {statistics.mean(confidences):.2f}")
    print(f"   Confidence range: {min(confidences):.2f} - {max(confidences):.2f}")

    # Show misclassified examples
    misclassified = [r for r in ml_results if not r["correct"]]
    if misclassified:
        print(f"   Misclassified ({len(misclassified)} examples):")
        for error in misclassified[:5]:  # Show first 5
            print(
                f"     '{error['message'][:50]}...' → {error['predicted']} (expected: {error['expected']})"
            )

    return {
        "ml_accuracy": ml_accuracy,
        "rule_accuracy": rule_accuracy,
        "method_counts": method_counts,
        "avg_confidence": statistics.mean(confidences),
        "misclassified_count": len(misclassified),
    }


def test_performance_benchmarks():
    """Test performance impact of ML categorization vs rule-based."""
    print("\n⚡ Testing performance benchmarks...")

    ml_extractor = MLTicketExtractor(enable_ml=True)
    rule_extractor = TicketExtractor()

    # Sample messages for performance testing
    test_messages = [msg for msg, _ in TEST_COMMIT_MESSAGES[:20]]  # Use subset for timing

    # Benchmark rule-based categorization
    start_time = time.time()
    for _ in range(5):  # Run multiple iterations
        for message in test_messages:
            _ = rule_extractor.categorize_commit(message)
    rule_time = (time.time() - start_time) / (5 * len(test_messages))

    # Benchmark ML categorization (first run - no cache)
    start_time = time.time()
    for _ in range(5):  # Run multiple iterations
        for message in test_messages:
            _ = ml_extractor.categorize_commit_with_confidence(message)
    ml_first_run_time = (time.time() - start_time) / (5 * len(test_messages))

    # Benchmark ML categorization (second run - with cache)
    start_time = time.time()
    for _ in range(5):  # Run multiple iterations
        for message in test_messages:
            _ = ml_extractor.categorize_commit_with_confidence(message)
    ml_cached_time = (time.time() - start_time) / (5 * len(test_messages))

    print(f"   Rule-based avg time: {rule_time * 1000:.2f} ms per commit")
    print(f"   ML first run avg time: {ml_first_run_time * 1000:.2f} ms per commit")
    print(f"   ML cached avg time: {ml_cached_time * 1000:.2f} ms per commit")
    print(f"   ML overhead (first run): {(ml_first_run_time / rule_time - 1) * 100:+.1f}%")
    print(f"   ML speedup (cached): {(rule_time / ml_cached_time - 1) * 100:+.1f}%")

    return {
        "rule_time_ms": rule_time * 1000,
        "ml_first_run_time_ms": ml_first_run_time * 1000,
        "ml_cached_time_ms": ml_cached_time * 1000,
        "ml_overhead_percent": (ml_first_run_time / rule_time - 1) * 100,
        "cache_speedup_percent": (rule_time / ml_cached_time - 1) * 100,
    }


def test_caching_effectiveness():
    """Test ML prediction caching functionality."""
    print("\n💾 Testing caching effectiveness...")

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)

        # Create ML extractor with caching enabled
        extractor = MLTicketExtractor(
            enable_ml=True,
            cache_dir=cache_dir,
            ml_config={"enable_caching": True, "cache_duration_days": 1},
        )

        test_message = "feat: implement user authentication system"
        test_files = ["src/auth.py", "tests/test_auth.py"]

        # First prediction (should not be cached)
        result1 = extractor.categorize_commit_with_confidence(test_message, test_files)
        print(f"   First prediction: {result1['category']} (method: {result1['method']})")

        # Second prediction (should be cached)
        result2 = extractor.categorize_commit_with_confidence(test_message, test_files)
        print(f"   Second prediction: {result2['category']} (method: {result2['method']})")

        # Verify caching worked
        cache_hit = result2["method"] == "cached"
        print(f"   Cache hit: {'✅' if cache_hit else '❌'}")

        # Test cache statistics
        if extractor.ml_cache:
            stats = extractor.ml_cache.get_statistics()
            print(f"   Cache statistics: {stats}")

        # Test cache cleanup
        if extractor.ml_cache:
            cleaned = extractor.ml_cache.cleanup_expired()
            print(f"   Expired entries cleaned: {cleaned}")

        return {
            "cache_hit": cache_hit,
            "first_method": result1["method"],
            "second_method": result2["method"],
            "cache_stats": stats if extractor.ml_cache else {},
        }


def test_configuration_options():
    """Test different ML configuration options."""
    print("\n⚙️  Testing configuration options...")

    configs_to_test = [
        # Default configuration
        {"name": "default", "config": {}},
        # High confidence threshold
        {"name": "high_confidence", "config": {"min_confidence": 0.9, "hybrid_threshold": 0.8}},
        # Low confidence threshold (more ML usage)
        {"name": "low_confidence", "config": {"min_confidence": 0.3, "hybrid_threshold": 0.3}},
        # Caching disabled
        {"name": "no_cache", "config": {"enable_caching": False}},
        # Different semantic weights
        {"name": "file_focused", "config": {"semantic_weight": 0.3, "file_pattern_weight": 0.7}},
    ]

    results = {}
    test_message = "feat: add new user authentication system"

    for config_test in configs_to_test:
        name = config_test["name"]
        config = config_test["config"]

        with tempfile.TemporaryDirectory() as temp_dir:
            extractor = MLTicketExtractor(
                enable_ml=True, cache_dir=Path(temp_dir), ml_config=config
            )

            result = extractor.categorize_commit_with_confidence(test_message)
            results[name] = {
                "category": result["category"],
                "confidence": result["confidence"],
                "method": result["method"],
            }

            print(
                f"   {name}: {result['category']} ({result['confidence']:.2f}, {result['method']})"
            )

    return results


def test_backward_compatibility():
    """Test that the system maintains backward compatibility."""
    print("\n🔄 Testing backward compatibility...")

    # Test 1: Basic TicketExtractor should still work
    basic_extractor = TicketExtractor()
    basic_result = basic_extractor.categorize_commit("fix: resolve authentication bug")
    print(f"   Basic extractor: {basic_result}")

    # Test 2: MLTicketExtractor with ML disabled should behave like basic
    ml_disabled = MLTicketExtractor(enable_ml=False)
    ml_disabled_result = ml_disabled.categorize_commit("fix: resolve authentication bug")
    print(f"   ML disabled: {ml_disabled_result}")

    # Test 3: MLTicketExtractor should provide same interface as basic
    ml_enabled = MLTicketExtractor(enable_ml=True)
    ml_enabled_result = ml_enabled.categorize_commit("fix: resolve authentication bug")
    print(f"   ML enabled: {ml_enabled_result}")

    # Verify compatibility
    basic_compatible = basic_result == ml_disabled_result
    interface_compatible = isinstance(ml_enabled_result, str)  # Should return same type

    print(f"   Basic compatibility: {'✅' if basic_compatible else '❌'}")
    print(f"   Interface compatibility: {'✅' if interface_compatible else '❌'}")

    # Test 4: GitAnalyzer integration
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = GitAnalysisCache(Path(temp_dir))

        # Without ML config
        analyzer_basic = GitAnalyzer(cache)
        extractor_type_basic = type(analyzer_basic.ticket_extractor).__name__

        # With ML config
        ml_config = {"enabled": True}
        analyzer_ml = GitAnalyzer(cache, ml_categorization_config=ml_config)
        extractor_type_ml = type(analyzer_ml.ticket_extractor).__name__

        print(f"   Basic analyzer extractor: {extractor_type_basic}")
        print(f"   ML analyzer extractor: {extractor_type_ml}")

        analyzer_compatible = (
            extractor_type_basic == "TicketExtractor" and extractor_type_ml == "MLTicketExtractor"
        )

    return {
        "basic_compatible": basic_compatible,
        "interface_compatible": interface_compatible,
        "analyzer_compatible": analyzer_compatible,
        "basic_result": basic_result,
        "ml_disabled_result": ml_disabled_result,
        "ml_enabled_result": ml_enabled_result,
    }


def test_graceful_fallback():
    """Test graceful fallback when ML components are unavailable."""
    print("\n🛡️  Testing graceful fallback scenarios...")

    # Test 1: spaCy unavailable (already tested since spacy is not installed)
    print("   spaCy availability test:")
    try:
        import spacy

        spacy_available = True
        print("     spaCy is available")
    except ImportError:
        spacy_available = False
        print("     spaCy is NOT available - testing fallback")

    # Test 2: ML extractor should work even without spaCy
    extractor = MLTicketExtractor(enable_ml=True)
    print(f"   ML extractor initialized successfully: {'✅' if extractor else '❌'}")
    print(f"   ML actually enabled: {extractor.enable_ml}")

    # Test 3: Should fall back to rule-based categorization
    result = extractor.categorize_commit_with_confidence("feat: add new feature")
    print(f"   Fallback categorization works: {'✅' if result else '❌'}")
    print(f"   Result method: {result.get('method', 'unknown')}")

    # Test 4: Should handle errors gracefully
    try:
        # Try with complex inputs that might cause ML issues
        complex_result = extractor.categorize_commit_with_confidence(
            "🎉 feat(auth): implement OAuth 2.0 with JWT tokens and refresh logic #123",
            ["src/auth/oauth.py", "tests/auth/test_oauth.py", "docs/oauth-guide.md"],
        )
        error_handling = True
        print("   Complex input handling: ✅")
        print(f"   Complex result: {complex_result['category']} ({complex_result['method']})")
    except Exception as e:
        error_handling = False
        print(f"   Complex input handling: ❌ ({e})")

    return {
        "spacy_available": spacy_available,
        "ml_extractor_initialized": extractor is not None,
        "ml_actually_enabled": extractor.enable_ml if extractor else False,
        "fallback_works": result is not None,
        "error_handling": error_handling,
        "fallback_method": result.get("method", "unknown") if result else None,
    }


def test_edge_cases():
    """Test edge cases and unusual inputs."""
    print("\n🔍 Testing edge cases...")

    extractor = MLTicketExtractor(enable_ml=True)

    edge_cases = [
        ("", "empty string"),
        ("   ", "whitespace only"),
        ("x", "single character"),
        ("🎉🚀✨", "emojis only"),
        ("a" * 1000, "very long message"),
        ("fix: resolve issue\nwith\nmultiple\nlines", "multiline message"),
        ("feat!: breaking change", "conventional commit with breaking change"),
        ("revert: feat: add feature", "revert commit"),
    ]

    results = []
    for message, description in edge_cases:
        try:
            result = extractor.categorize_commit_with_confidence(message)
            success = True
            category = result["category"]
            confidence = result["confidence"]
            method = result["method"]
        except Exception as e:
            success = False
            category = str(e)
            confidence = 0.0
            method = "error"

        results.append(
            {
                "description": description,
                "success": success,
                "category": category,
                "confidence": confidence,
                "method": method,
            }
        )

        status = "✅" if success else "❌"
        print(f"   {description}: {status} → {category} ({confidence:.2f}, {method})")

    success_rate = sum(1 for r in results if r["success"]) / len(results)
    print(f"   Edge case success rate: {success_rate:.1%}")

    return {"success_rate": success_rate, "results": results}


def main():
    """Run comprehensive ML categorization tests."""
    print("🚀 Comprehensive ML-based Commit Categorization Test Suite")
    print("=" * 80)

    all_results = {}

    try:
        # Run all test suites
        all_results["accuracy"] = test_categorization_accuracy()
        all_results["performance"] = test_performance_benchmarks()
        all_results["caching"] = test_caching_effectiveness()
        all_results["configuration"] = test_configuration_options()
        all_results["backward_compatibility"] = test_backward_compatibility()
        all_results["graceful_fallback"] = test_graceful_fallback()
        all_results["edge_cases"] = test_edge_cases()

        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 80)

        # Accuracy Summary
        acc = all_results["accuracy"]
        print("🎯 ACCURACY METRICS:")
        print(f"   ML System Accuracy: {acc['ml_accuracy']:.1%}")
        print(f"   Rule-based Accuracy: {acc['rule_accuracy']:.1%}")
        print(f"   Improvement: {acc['ml_accuracy'] - acc['rule_accuracy']:+.1%}")
        print(f"   Average Confidence: {acc['avg_confidence']:.2f}")

        # Performance Summary
        perf = all_results["performance"]
        print("\n⚡ PERFORMANCE METRICS:")
        print(f"   Rule-based: {perf['rule_time_ms']:.1f} ms/commit")
        print(f"   ML (first run): {perf['ml_first_run_time_ms']:.1f} ms/commit")
        print(f"   ML (cached): {perf['ml_cached_time_ms']:.1f} ms/commit")
        print(f"   ML Overhead: {perf['ml_overhead_percent']:+.1f}%")

        # Compatibility Summary
        compat = all_results["backward_compatibility"]
        print("\n🔄 COMPATIBILITY STATUS:")
        print(f"   Basic Compatibility: {'✅' if compat['basic_compatible'] else '❌'}")
        print(f"   Interface Compatibility: {'✅' if compat['interface_compatible'] else '❌'}")
        print(f"   Analyzer Integration: {'✅' if compat['analyzer_compatible'] else '❌'}")

        # Fallback Summary
        fallback = all_results["graceful_fallback"]
        print("\n🛡️  FALLBACK STATUS:")
        print(f"   spaCy Available: {'✅' if fallback['spacy_available'] else '❌'}")
        print(f"   Graceful Fallback: {'✅' if fallback['fallback_works'] else '❌'}")
        print(f"   Error Handling: {'✅' if fallback['error_handling'] else '❌'}")
        print(f"   Fallback Method: {fallback['fallback_method']}")

        # Edge Cases Summary
        edges = all_results["edge_cases"]
        print("\n🔍 EDGE CASE HANDLING:")
        print(f"   Success Rate: {edges['success_rate']:.1%}")

        # Overall Assessment
        print("\n🏆 OVERALL ASSESSMENT:")
        overall_score = (
            acc["ml_accuracy"] * 0.3  # 30% weight on accuracy
            + (1 - perf["ml_overhead_percent"] / 100 * 0.1) * 0.2  # 20% weight on performance
            + (
                compat["basic_compatible"]
                + compat["interface_compatible"]
                + compat["analyzer_compatible"]
            )
            / 3
            * 0.2  # 20% weight on compatibility
            + (fallback["fallback_works"] + fallback["error_handling"])
            / 2
            * 0.2  # 20% weight on robustness
            + edges["success_rate"] * 0.1  # 10% weight on edge case handling
        )
        print(f"   Overall Score: {overall_score:.1%}")

        if overall_score >= 0.9:
            print("   Status: 🌟 EXCELLENT - Production ready")
        elif overall_score >= 0.8:
            print("   Status: ✅ GOOD - Ready with minor improvements")
        elif overall_score >= 0.7:
            print("   Status: ⚠️  FAIR - Needs improvements")
        else:
            print("   Status: ❌ POOR - Significant issues found")

        # Save detailed results
        results_file = Path("ml_test_results.json")
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {results_file}")

        return 0 if overall_score >= 0.7 else 1

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
