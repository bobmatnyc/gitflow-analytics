#!/usr/bin/env python3
"""Comprehensive test to validate ML categorization accuracy and performance."""

import time
from collections import defaultdict

from src.gitflow_analytics.extractors.ml_tickets import MLTicketExtractor
from src.gitflow_analytics.extractors.tickets import TicketExtractor

# Test dataset with expected categories
TEST_COMMITS = [
    # Features
    ("feat: add user authentication system", "feature"),
    ("feature: implement shopping cart", "feature"),
    ("Add new dashboard for analytics", "feature"),
    ("Implement OAuth2 integration", "feature"),
    ("NEW: Add export to PDF functionality", "feature"),
    # Bug fixes
    ("fix: resolve memory leak in payment processing", "bug_fix"),
    ("bugfix: correct calculation error in reports", "bug_fix"),
    ("Fix broken navigation on mobile", "bug_fix"),
    ("Hotfix: Critical security vulnerability", "bug_fix"),
    ("Fixed issue with date parsing", "bug_fix"),
    # Refactoring
    ("refactor: simplify database queries", "refactor"),
    ("Refactor authentication logic for clarity", "refactor"),
    ("Clean up legacy code in user module", "refactor"),
    ("Restructure project layout", "refactor"),
    # Documentation
    ("docs: update API documentation", "documentation"),
    ("doc: add installation guide", "documentation"),
    ("Update README with new features", "documentation"),
    ("Document deployment process", "documentation"),
    # Tests
    ("test: add unit tests for payment module", "test"),
    ("tests: improve coverage for user service", "test"),
    ("Add integration tests for API", "test"),
    ("Write e2e tests for checkout flow", "test"),
    # Maintenance
    ("chore: update dependencies", "maintenance"),
    ("build: upgrade webpack to v5", "maintenance"),
    ("ci: add GitHub Actions workflow", "maintenance"),
    ("deps: bump security vulnerabilities", "maintenance"),
    # Style
    ("style: format code with prettier", "style"),
    ("Format all Python files with black", "style"),
    ("Fix linting errors", "style"),
    # Ambiguous/difficult cases
    ("Update user interface", "feature"),  # Could be feature or refactor
    ("Improve performance", "refactor"),  # Could be refactor or feature
    ("Fix typo", "other"),  # Minor fix
    ("WIP: Work in progress", "other"),
    ("Merge branch 'develop'", "other"),
]


def test_accuracy():
    """Test categorization accuracy comparing ML vs rule-based."""
    print("🎯 Testing Categorization Accuracy")
    print("=" * 60)

    # Initialize extractors
    rule_extractor = TicketExtractor()
    ml_extractor = MLTicketExtractor(enable_ml=True)

    # Track results
    rule_correct = 0
    ml_correct = 0
    results = []

    for commit_msg, expected in TEST_COMMITS:
        # Rule-based categorization
        rule_category = rule_extractor.categorize_commit(commit_msg)
        rule_is_correct = rule_category == expected
        if rule_is_correct:
            rule_correct += 1

        # ML-based categorization
        ml_result = ml_extractor.categorize_commit_with_confidence(commit_msg)
        ml_category = ml_result["category"]
        ml_confidence = ml_result["confidence"]
        ml_method = ml_result["method"]
        ml_is_correct = ml_category == expected
        if ml_is_correct:
            ml_correct += 1

        results.append(
            {
                "message": commit_msg,
                "expected": expected,
                "rule": rule_category,
                "ml": ml_category,
                "confidence": ml_confidence,
                "method": ml_method,
                "rule_correct": rule_is_correct,
                "ml_correct": ml_is_correct,
            }
        )

    # Print results
    print("\n📊 Overall Accuracy:")
    print(
        f"   Rule-based: {rule_correct}/{len(TEST_COMMITS)} ({rule_correct / len(TEST_COMMITS) * 100:.1f}%)"
    )
    print(
        f"   ML-based:   {ml_correct}/{len(TEST_COMMITS)} ({ml_correct / len(TEST_COMMITS) * 100:.1f}%)"
    )

    # Show improvements
    print("\n🚀 Improvements:")
    improvements = [r for r in results if not r["rule_correct"] and r["ml_correct"]]
    for imp in improvements[:5]:  # Show first 5 improvements
        print(f"   ✅ '{imp['message'][:40]}...'")
        print(
            f"      Expected: {imp['expected']}, Rule: {imp['rule']}, ML: {imp['ml']} (conf: {imp['confidence']:.2f})"
        )

    # Show regressions
    regressions = [r for r in results if r["rule_correct"] and not r["ml_correct"]]
    if regressions:
        print("\n⚠️  Regressions:")
        for reg in regressions[:3]:
            print(f"   ❌ '{reg['message'][:40]}...'")
            print(
                f"      Expected: {reg['expected']}, Rule: {reg['rule']}, ML: {reg['ml']} (conf: {reg['confidence']:.2f})"
            )

    return results


def test_performance():
    """Test performance impact of ML categorization."""
    print("\n\n⏱️  Testing Performance Impact")
    print("=" * 60)

    rule_extractor = TicketExtractor()
    ml_extractor = MLTicketExtractor(enable_ml=True)

    # Warm up caches
    for msg, _ in TEST_COMMITS[:5]:
        rule_extractor.categorize_commit(msg)
        ml_extractor.categorize_commit(msg)

    # Test rule-based performance
    start = time.time()
    for _ in range(10):
        for msg, _ in TEST_COMMITS:
            rule_extractor.categorize_commit(msg)
    rule_time = time.time() - start
    rule_per_commit = (rule_time / (10 * len(TEST_COMMITS))) * 1000

    # Test ML performance (with caching)
    start = time.time()
    for _ in range(10):
        for msg, _ in TEST_COMMITS:
            ml_extractor.categorize_commit(msg)
    ml_time = time.time() - start
    ml_per_commit = (ml_time / (10 * len(TEST_COMMITS))) * 1000

    print(f"   Rule-based: {rule_per_commit:.2f}ms per commit")
    print(f"   ML-based:   {ml_per_commit:.2f}ms per commit (with caching)")
    print(
        f"   Overhead:   {ml_per_commit - rule_per_commit:.2f}ms ({(ml_per_commit / rule_per_commit - 1) * 100:.0f}% slower)"
    )

    # Note: Cache clearing would require access to internal cache implementation
    print("   \n   Note: ML categorization uses intelligent caching for performance")


def test_confidence_distribution():
    """Test confidence score distribution."""
    print("\n\n📈 Testing Confidence Distribution")
    print("=" * 60)

    ml_extractor = MLTicketExtractor(enable_ml=True)
    confidence_buckets = defaultdict(int)
    method_counts = defaultdict(int)

    for msg, _ in TEST_COMMITS:
        result = ml_extractor.categorize_commit_with_confidence(msg)
        confidence = result["confidence"]
        method = result["method"]

        # Bucket confidence scores
        bucket = int(confidence * 10) / 10
        confidence_buckets[bucket] += 1
        method_counts[method] += 1

    print("   Confidence Distribution:")
    for bucket in sorted(confidence_buckets.keys(), reverse=True):
        count = confidence_buckets[bucket]
        bar = "█" * int(count * 2)
        print(f"   {bucket:.1f}: {bar} {count}")

    print("\n   Method Usage:")
    for method, count in method_counts.items():
        pct = count / len(TEST_COMMITS) * 100
        print(f"   {method}: {count} ({pct:.1f}%)")


def main():
    """Run all tests."""
    print("🚀 Comprehensive ML Categorization Testing")
    print("=" * 60)

    try:
        test_accuracy()
        test_performance()
        test_confidence_distribution()

        print("\n" + "=" * 60)
        print("✅ All tests completed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
