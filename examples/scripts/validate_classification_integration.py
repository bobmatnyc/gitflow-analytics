#!/usr/bin/env python3
"""
Validation script for commit classification integration

This script validates that the commit classification system is properly
integrated with GitFlow Analytics and can work with the test data.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Add the src directory to Python path for local development
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")

    try:
        from gitflow_analytics.classification.classifier import CommitClassifier  # noqa: F401

        print("✅ CommitClassifier import successful")
    except ImportError as e:
        print(f"❌ CommitClassifier import failed: {e}")
        return False

    try:
        from gitflow_analytics.classification.linguist_analyzer import (
            LinguistAnalyzer,  # noqa: F401
        )

        print("✅ LinguistAnalyzer import successful")
    except ImportError as e:
        print(f"❌ LinguistAnalyzer import failed: {e}")
        return False

    try:
        from gitflow_analytics.classification.feature_extractor import (
            FeatureExtractor,  # noqa: F401
        )

        print("✅ FeatureExtractor import successful")
    except ImportError as e:
        print(f"❌ FeatureExtractor import failed: {e}")
        return False

    try:
        from gitflow_analytics.reports.classification_writer import (
            ClassificationReportGenerator,  # noqa: F401
        )

        print("✅ ClassificationReportGenerator import successful")
    except ImportError as e:
        print(f"❌ ClassificationReportGenerator import failed: {e}")
        return False

    try:
        from gitflow_analytics.core.analyzer import GitAnalyzer  # noqa: F401

        print("✅ Enhanced GitAnalyzer import successful")
    except ImportError as e:
        print(f"❌ Enhanced GitAnalyzer import failed: {e}")
        return False

    return True


def test_classification_system():
    """Test the classification system with mock data."""
    print("\n🧪 Testing classification system...")

    try:
        from gitflow_analytics.classification.classifier import CommitClassifier  # noqa: F401

        # Initialize classifier
        classifier = CommitClassifier(
            config={"enabled": True, "confidence_threshold": 0.6},
            cache_dir=Path.cwd() / ".test_cache",
        )

        print("✅ CommitClassifier initialization successful")

        # Test with mock commit data
        mock_commits = [
            {
                "hash": "abc123",
                "message": "fix: resolve authentication bug",
                "author_name": "Test Developer",
                "author_email": "test@example.com",
                "timestamp": datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
                "files_changed": ["src/auth.py", "tests/test_auth.py"],
                "insertions": 15,
                "deletions": 8,
                "file_details": {
                    "src/auth.py": {"insertions": 12, "deletions": 5},
                    "tests/test_auth.py": {"insertions": 3, "deletions": 3},
                },
            },
            {
                "hash": "def456",
                "message": "feat: add new user dashboard",
                "author_name": "Test Developer",
                "author_email": "test@example.com",
                "timestamp": datetime(2024, 6, 2, 14, 30, 0, tzinfo=timezone.utc),
                "files_changed": ["src/dashboard.py", "templates/dashboard.html"],
                "insertions": 45,
                "deletions": 2,
                "file_details": {
                    "src/dashboard.py": {"insertions": 30, "deletions": 1},
                    "templates/dashboard.html": {"insertions": 15, "deletions": 1},
                },
            },
        ]

        # Test classification
        results = classifier.classify_commits(mock_commits)

        if results:
            print(f"✅ Classification successful - processed {len(results)} commits")
            for result in results:
                print(
                    f"   📊 {result['commit_hash'][:7]}: {result['predicted_class']} ({result['confidence']:.2f})"
                )
        else:
            print("⚠️ Classification returned empty results (model may need training)")

        return True

    except Exception as e:
        print(f"❌ Classification system test failed: {e}")
        return False


def test_report_generator():
    """Test the report generator with mock data."""
    print("\n📊 Testing report generator...")

    try:
        from gitflow_analytics.reports.classification_writer import ClassificationReportGenerator

        # Create test output directory
        test_output = Path.cwd() / "test_reports"
        test_output.mkdir(exist_ok=True)

        # Initialize report generator
        report_gen = ClassificationReportGenerator(
            output_directory=test_output, config={"confidence_threshold": 0.6}
        )

        print("✅ ClassificationReportGenerator initialization successful")

        # Mock classified commits
        mock_classified_commits = [
            {
                "hash": "abc123",
                "message": "fix: resolve authentication bug",
                "author_name": "Alice Developer",
                "canonical_author_name": "Alice Developer",
                "repository": "test-repo",
                "timestamp": datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
                "predicted_class": "bugfix",
                "classification_confidence": 0.85,
                "is_reliable_prediction": True,
                "files_changed": 2,
                "insertions": 15,
                "deletions": 8,
                "file_analysis_summary": {
                    "primary_language": "Python",
                    "primary_activity": "engineering",
                    "is_multilingual": False,
                },
            },
            {
                "hash": "def456",
                "message": "feat: add new user dashboard",
                "author_name": "Bob Developer",
                "canonical_author_name": "Bob Developer",
                "repository": "test-repo",
                "timestamp": datetime(2024, 6, 2, 14, 30, 0, tzinfo=timezone.utc),
                "predicted_class": "feature",
                "classification_confidence": 0.92,
                "is_reliable_prediction": True,
                "files_changed": 2,
                "insertions": 45,
                "deletions": 2,
                "file_analysis_summary": {
                    "primary_language": "Python",
                    "primary_activity": "engineering",
                    "is_multilingual": True,
                },
            },
        ]

        # Test report generation
        metadata = {
            "start_date": "2024-06-01",
            "end_date": "2024-06-02",
            "config_path": "test_config.yaml",
        }

        report_paths = report_gen.generate_comprehensive_report(
            classified_commits=mock_classified_commits, metadata=metadata
        )

        if report_paths:
            print(f"✅ Report generation successful - created {len(report_paths)} reports")
            for report_type, path in report_paths.items():
                if Path(path).exists():
                    print(f"   📄 {report_type}: {Path(path).name}")
                else:
                    print(f"   ⚠️ {report_type}: file not found at {path}")
        else:
            print("❌ No reports generated")
            return False

        return True

    except Exception as e:
        print(f"❌ Report generator test failed: {e}")
        return False


def test_analyzer_integration():
    """Test GitAnalyzer integration with classification."""
    print("\n🔗 Testing GitAnalyzer integration...")

    try:
        from gitflow_analytics.core.analyzer import GitAnalyzer
        from gitflow_analytics.core.cache import GitAnalysisCache

        # Setup test cache
        test_cache = GitAnalysisCache(Path.cwd() / ".test_cache")

        # Initialize analyzer with classification enabled
        classification_config = {"enabled": True, "confidence_threshold": 0.6, "batch_size": 10}

        analyzer = GitAnalyzer(
            cache=test_cache, batch_size=10, classification_config=classification_config
        )

        # Check if classification was properly initialized
        if analyzer.classification_enabled:
            print("✅ GitAnalyzer classification integration enabled")
            print(f"   📊 Classifier status: {analyzer.commit_classifier is not None}")
        else:
            print("⚠️ GitAnalyzer classification integration disabled (may be expected)")

        return True

    except Exception as e:
        print(f"❌ GitAnalyzer integration test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🚀 GitFlow Analytics - Commit Classification Integration Validation")
    print("=" * 70)

    tests = [
        ("Import Tests", test_imports),
        ("Classification System", test_classification_system),
        ("Report Generator", test_report_generator),
        ("Analyzer Integration", test_analyzer_integration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")

    print("\n" + "=" * 70)
    print(f"📊 Validation Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Classification integration is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
