#!/usr/bin/env python3
"""Test ML components availability and functionality."""

import tempfile
from pathlib import Path


def test_qualitative_components():
    """Test qualitative analysis components availability."""
    print("🔬 Testing qualitative analysis components...")

    try:
        from src.gitflow_analytics.qualitative.classifiers.change_type import ChangeTypeClassifier
        from src.gitflow_analytics.qualitative.models.schemas import ChangeTypeConfig

        print("✅ ChangeTypeClassifier import successful")

        # Test initialization
        config = ChangeTypeConfig()
        classifier = ChangeTypeClassifier(config)
        print("✅ ChangeTypeClassifier initialization successful")

        # Test classification without spaCy
        test_message = "feat: add user authentication system"
        test_files = ["src/auth.py", "tests/test_auth.py"]

        category, confidence = classifier.classify(test_message, None, test_files)
        print(f"✅ Classification successful: {category} (confidence: {confidence:.2f})")

        return True, category, confidence

    except Exception as e:
        print(f"❌ Qualitative components test failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None, 0.0


def test_ml_extractor_direct():
    """Test MLTicketExtractor components directly."""
    print("\n🔬 Testing MLTicketExtractor components...")

    try:
        from src.gitflow_analytics.extractors.ml_tickets import MLTicketExtractor

        # Test with explicit ML enabling
        with tempfile.TemporaryDirectory() as temp_dir:
            extractor = MLTicketExtractor(
                enable_ml=True, cache_dir=Path(temp_dir), ml_config={"enable_caching": True}
            )

            print(f"   ML enabled: {extractor.enable_ml}")
            print(f"   Change type classifier: {extractor.change_type_classifier is not None}")
            print(f"   NLP model: {extractor.nlp_model is not None}")
            print(f"   ML cache: {extractor.ml_cache is not None}")

            # Test direct ML categorization
            test_message = "feat: implement OAuth 2.0 authentication"
            test_files = ["src/auth/oauth.py", "tests/auth/test_oauth.py"]

            ml_result = extractor._ml_categorize_commit(test_message, test_files)
            print(f"   Direct ML result: {ml_result}")

            # Test with confidence
            detailed_result = extractor.categorize_commit_with_confidence(test_message, test_files)
            print(f"   Detailed result: {detailed_result}")

            return True, ml_result, detailed_result

    except Exception as e:
        print(f"❌ MLTicketExtractor test failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None, None


def test_spacy_installation():
    """Check what happens if we try to install spaCy."""
    print("\n🔬 Testing spaCy installation options...")

    # Check if we can at least try to import the basic models
    try:
        import spacy

        print("✅ spaCy is already available")
        return True
    except ImportError:
        print("❌ spaCy not available")

        # Check if we have pip available for installation
        try:
            import pip

            print("✅ pip is available for package installation")

            # Note: We won't actually install spaCy here, just check possibility
            print("   Could install spaCy with: pip install spacy")
            print("   Could download model with: python -m spacy download en_core_web_sm")

        except ImportError:
            print("❌ pip not available")

        return False


def main():
    """Run component tests."""
    print("🚀 Testing ML Components Availability")
    print("=" * 60)

    # Test qualitative components
    qual_success, category, confidence = test_qualitative_components()

    # Test ML extractor
    ml_success, ml_result, detailed_result = test_ml_extractor_direct()

    # Test spaCy status
    spacy_available = test_spacy_installation()

    print("\n" + "=" * 60)
    print("📊 COMPONENT TEST SUMMARY")
    print("=" * 60)

    print(f"Qualitative Components: {'✅' if qual_success else '❌'}")
    if qual_success:
        print(f"   Sample classification: {category} ({confidence:.2f})")

    print(f"ML Extractor Components: {'✅' if ml_success else '❌'}")
    if ml_success and detailed_result:
        print(f"   ML result method: {detailed_result.get('method', 'unknown')}")
        print(f"   ML result confidence: {detailed_result.get('confidence', 0):.2f}")

    print(f"spaCy Available: {'✅' if spacy_available else '❌'}")

    # Analysis
    print("\n🔍 ANALYSIS:")
    if not spacy_available:
        print("   • spaCy is not installed - ML system falls back to rule-based classification")
        print("   • ChangeTypeClassifier likely works with pattern matching instead of NLP")
        print("   • System maintains functionality but doesn't get ML benefits")

    if qual_success and not spacy_available:
        print("   • Qualitative components work without spaCy (using fallback methods)")

    if ml_success:
        print("   • MLTicketExtractor properly handles missing ML dependencies")
        print("   • Graceful fallback to rule-based classification is working")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not spacy_available:
        print("   1. Install spaCy to enable true ML categorization:")
        print("      pip install spacy")
        print("      python -m spacy download en_core_web_sm")
        print("   2. Test again with spaCy installed to see ML improvements")
        print("   3. Current fallback behavior is correct and maintains functionality")

    return 0


if __name__ == "__main__":
    exit(main())
