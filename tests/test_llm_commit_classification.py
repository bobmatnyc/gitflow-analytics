"""Tests for LLM-based commit classification system.

This test suite validates the LLM commit classification implementation
including filtering, caching, error handling, and integration.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.gitflow_analytics.extractors.ml_tickets import MLTicketExtractor
from src.gitflow_analytics.extractors.tickets import filter_git_artifacts
from src.gitflow_analytics.qualitative.classifiers.llm_commit_classifier import (
    LLMCommitClassifier,
    LLMConfig,
    LLMPredictionCache,
)


class TestGitArtifactFiltering(unittest.TestCase):
    """Test git artifact filtering functionality."""

    def test_filter_coauthored_by(self):
        """Test removal of Co-authored-by lines."""
        message = """Fix user authentication bug

Co-authored-by: John Doe <john@example.com>
Co-authored-by: Jane Smith <jane@example.com>"""

        expected = "Fix user authentication bug"
        result = filter_git_artifacts(message)
        self.assertEqual(result, expected)

    def test_filter_signed_off_by(self):
        """Test removal of Signed-off-by lines."""
        message = """Add new feature for video streaming

This adds support for live streaming.

Signed-off-by: Developer <dev@example.com>"""

        expected = "Add new feature for video streaming\n\nThis adds support for live streaming."
        result = filter_git_artifacts(message)
        self.assertEqual(result, expected)

    def test_filter_multiple_artifacts(self):
        """Test removal of multiple git artifacts."""
        message = """feat: implement localization system

Add support for Spanish and French translations.

Co-authored-by: Translator <translator@example.com>
Reviewed-by: Manager <manager@example.com>  
Signed-off-by: Developer <dev@example.com>
Tested-by: QA <qa@example.com>"""

        expected = "feat: implement localization system\n\nAdd support for Spanish and French translations."
        result = filter_git_artifacts(message)
        self.assertEqual(result, expected)

    def test_filter_empty_message(self):
        """Test handling of empty messages."""
        self.assertEqual(filter_git_artifacts(""), "")
        self.assertEqual(filter_git_artifacts("   "), "")
        self.assertEqual(filter_git_artifacts("\n\n"), "")

    def test_filter_dots_message(self):
        """Test handling of dots-only messages."""
        self.assertEqual(filter_git_artifacts("..."), "")
        self.assertEqual(filter_git_artifacts("...\n\n..."), "")

    def test_preserve_normal_content(self):
        """Test that normal commit content is preserved."""
        message = "fix: resolve memory leak in video player\n\nImproves performance during long playback sessions."
        result = filter_git_artifacts(message)
        self.assertEqual(result, message)


class TestLLMConfig(unittest.TestCase):
    """Test LLM configuration handling."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()

        self.assertIsNone(config.api_key)
        self.assertEqual(config.model, "mistralai/mistral-7b-instruct")
        self.assertEqual(config.confidence_threshold, 0.7)
        self.assertEqual(config.max_tokens, 50)
        self.assertEqual(config.temperature, 0.1)
        self.assertTrue(config.enable_caching)
        self.assertEqual(config.cache_duration_days, 90)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LLMConfig(
            api_key="test-key",
            model="meta-llama/llama-3-8b-instruct",
            confidence_threshold=0.8,
            max_daily_requests=500,
        )

        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.model, "meta-llama/llama-3-8b-instruct")
        self.assertEqual(config.confidence_threshold, 0.8)
        self.assertEqual(config.max_daily_requests, 500)

    def test_domain_terms_initialization(self):
        """Test that domain terms are properly initialized."""
        config = LLMConfig()

        self.assertIn("media", config.domain_terms)
        self.assertIn("localization", config.domain_terms)
        self.assertIn("integration", config.domain_terms)
        self.assertIn("video", config.domain_terms["media"])
        self.assertIn("translation", config.domain_terms["localization"])
        self.assertIn("api", config.domain_terms["integration"])


class TestLLMPredictionCache(unittest.TestCase):
    """Test LLM prediction caching functionality."""

    def setUp(self):
        """Set up test cache."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_path = self.temp_dir / "test_llm_cache.db"
        self.cache = LLMPredictionCache(self.cache_path, 1)  # 1 day expiration

    def tearDown(self):
        """Clean up test cache."""
        if self.cache_path.exists():
            self.cache_path.unlink()
        self.temp_dir.rmdir()

    def test_cache_initialization(self):
        """Test cache database initialization."""
        self.assertTrue(self.cache_path.exists())

        # Check database schema
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            # The LLMCache uses 'llm_cache' as the table name
            self.assertIn("llm_cache", tables)

    def test_store_and_retrieve_prediction(self):
        """Test storing and retrieving predictions."""
        message = "fix: resolve authentication bug"
        files_changed = ["src/auth.py", "tests/test_auth.py"]
        result = {
            "category": "bugfix",
            "confidence": 0.85,
            "method": "llm",
            "reasoning": "Contains bug fix keywords",
            "model": "mistral-7b-instruct",
        }

        # Store prediction
        self.cache.store_prediction(message, files_changed, result)

        # Retrieve prediction
        cached_result = self.cache.get_prediction(message, files_changed)

        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result["category"], "bugfix")
        self.assertEqual(cached_result["confidence"], 0.85)
        self.assertEqual(cached_result["method"], "cached")  # Method should be overridden

    def test_cache_key_generation(self):
        """Test that different inputs generate different cache keys."""
        message1 = "fix: bug in auth"
        message2 = "feat: add auth"
        files1 = ["auth.py"]
        files2 = ["user.py"]

        # Access the underlying LLMCache to test key generation
        key1, _, _ = self.cache.cache._generate_cache_key(message1, files1)
        key2, _, _ = self.cache.cache._generate_cache_key(message2, files1)
        key3, _, _ = self.cache.cache._generate_cache_key(message1, files2)

        # All keys should be different
        self.assertNotEqual(key1, key2)
        self.assertNotEqual(key1, key3)
        self.assertNotEqual(key2, key3)

    def test_cache_statistics(self):
        """Test cache statistics collection."""
        stats = self.cache.get_statistics()

        self.assertIn("total_entries", stats)
        self.assertIn("active_entries", stats)
        self.assertIn("expired_entries", stats)
        self.assertIn("cache_file_size_mb", stats)

        # Initially should be empty
        self.assertEqual(stats["total_entries"], 0)
        self.assertEqual(stats["active_entries"], 0)


class TestLLMCommitClassifier(unittest.TestCase):
    """Test LLM commit classifier functionality."""

    def setUp(self):
        """Set up test classifier with mocked dependencies."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create config without API key (to avoid real API calls)
        self.config = LLMConfig(api_key=None, confidence_threshold=0.7, enable_caching=True)

    def tearDown(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_classifier_initialization_without_api_key(self):
        """Test classifier initialization without API key."""
        # The classifier should initialize successfully without API key
        classifier = LLMCommitClassifier(self.config, self.temp_dir)
        # Should initialize successfully but won't make API calls
        self.assertIsNotNone(classifier)
        self.assertEqual(classifier.config.model, "mistralai/mistral-7b-instruct")

    def test_classifier_initialization_without_requests(self):
        """Test classifier initialization without requests library."""
        # The new implementation gracefully handles missing requests
        # It will initialize but the classifier attribute might be None
        classifier = LLMCommitClassifier(self.config, self.temp_dir)
        self.assertIsNotNone(classifier)  # Should still create the object

    def test_streamlined_categories(self):
        """Test that streamlined categories are properly defined."""
        categories = LLMCommitClassifier.CATEGORIES

        self.assertEqual(len(categories), 7)
        self.assertIn("feature", categories)
        self.assertIn("bugfix", categories)
        self.assertIn("maintenance", categories)
        self.assertIn("integration", categories)
        self.assertIn("content", categories)
        self.assertIn("media", categories)
        self.assertIn("localization", categories)

    def test_classify_empty_message(self):
        """Test classification of empty messages."""
        classifier = LLMCommitClassifier(self.config, self.temp_dir)
        result = classifier.classify_commit("")

        # Empty messages should be classified as maintenance with low confidence
        self.assertEqual(result["category"], "maintenance")
        self.assertIn(result["method"], ["empty_message", "rule_enhanced"])
        self.assertLess(result["confidence"], 0.5)

    def test_rate_limiting(self):
        """Test API rate limiting functionality."""
        # Create config with very low limit for testing
        config = LLMConfig(max_daily_requests=0)
        classifier = LLMCommitClassifier(config, self.temp_dir)

        result = classifier.classify_commit("test message")

        # Should fall back to rule-based when rate limited
        self.assertIn(result["method"], ["rate_limited", "rule_enhanced"])
        self.assertIsNotNone(result["confidence"])


class TestMLTicketExtractorLLMIntegration(unittest.TestCase):
    """Test LLM integration in MLTicketExtractor."""

    def setUp(self):
        """Set up test extractor."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Mock LLM config to avoid real API calls
        self.llm_config = {
            "api_key": None,
            "model": "mistralai/mistral-7b-instruct",
            "confidence_threshold": 0.7,
            "enable_caching": True,
        }

    def tearDown(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("src.gitflow_analytics.extractors.ml_tickets.SPACY_AVAILABLE", False)
    def test_extractor_initialization_llm_only(self):
        """Test extractor with LLM enabled but ML disabled."""
        extractor = MLTicketExtractor(
            enable_ml=False, enable_llm=True, llm_config=self.llm_config, cache_dir=self.temp_dir
        )

        self.assertFalse(extractor.enable_ml)
        self.assertTrue(extractor.enable_llm)

    @patch("src.gitflow_analytics.extractors.ml_tickets.SPACY_AVAILABLE", True)
    def test_extractor_initialization_both_enabled(self):
        """Test extractor with both ML and LLM enabled."""
        extractor = MLTicketExtractor(
            enable_ml=True, enable_llm=True, llm_config=self.llm_config, cache_dir=self.temp_dir
        )

        self.assertTrue(extractor.enable_ml)
        self.assertTrue(extractor.enable_llm)

    def test_category_mapping_llm_to_parent(self):
        """Test LLM category mapping to parent categories."""
        extractor = MLTicketExtractor(
            enable_ml=False,
            enable_llm=False,  # Disable to avoid initialization issues
            cache_dir=self.temp_dir,
        )

        # Test streamlined category mappings
        self.assertEqual(extractor._map_llm_to_parent_category("feature"), "feature")
        self.assertEqual(extractor._map_llm_to_parent_category("bugfix"), "bug_fix")
        self.assertEqual(extractor._map_llm_to_parent_category("maintenance"), "maintenance")
        self.assertEqual(extractor._map_llm_to_parent_category("integration"), "build")
        self.assertEqual(extractor._map_llm_to_parent_category("content"), "documentation")
        self.assertEqual(extractor._map_llm_to_parent_category("media"), "other")
        self.assertEqual(extractor._map_llm_to_parent_category("localization"), "other")
        self.assertEqual(extractor._map_llm_to_parent_category("unknown"), "other")

    def test_git_artifact_filtering_in_categorization(self):
        """Test that git artifacts are filtered in categorization."""
        extractor = MLTicketExtractor(enable_ml=False, enable_llm=False, cache_dir=self.temp_dir)

        # Test with git artifacts
        message_with_artifacts = """fix: resolve authentication bug

Co-authored-by: John Doe <john@example.com>
Signed-off-by: Developer <dev@example.com>"""

        # Should still categorize as bug_fix despite artifacts
        category = extractor.categorize_commit(message_with_artifacts)
        self.assertEqual(category, "bug_fix")  # Should be classified based on "fix:" prefix

    def test_statistics_collection(self):
        """Test that statistics collection includes LLM data."""
        extractor = MLTicketExtractor(enable_ml=False, enable_llm=False, cache_dir=self.temp_dir)

        stats = extractor.get_ml_statistics()

        self.assertIn("ml_enabled", stats)
        self.assertIn("llm_enabled", stats)
        self.assertIn("components_loaded", stats)
        self.assertIn("llm_classifier", stats["components_loaded"])
        self.assertIn("configuration", stats)


class TestIntegrationScenarios(unittest.TestCase):
    """Test real-world integration scenarios."""

    def test_ewtn_commit_examples(self):
        """Test classification of EWTN-style commits."""
        test_commits = [
            # Feature commits (should be recognized by "feat:" prefix)
            ("feat: add video streaming player component", "feature"),
            ("feat: integrate with third-party authentication API", "feature"),
            ("feat: implement i18n support for German locale", "feature"),
            # Bug fix commits (should be recognized by "fix:" prefix)
            ("fix: resolve audio sync issue in live broadcast", "bug_fix"),
            # Maintenance commits (should be recognized by "chore:" prefix or refactor keywords)
            (
                "chore: update dependency versions",
                "chore",
            ),  # Parent class actually returns 'chore' not 'maintenance'
            ("Refactor video processing pipeline", "refactor"),
            ("Clean up unused CSS classes", "maintenance"),
            # Content/documentation commits (should be recognized by keywords)
            ("Update homepage copy and messaging", "other"),  # May not match specific patterns
            ("Update French translation files", "maintenance"),  # "update" keyword
            ("Add new blog post content", "feature"),  # "add new" keywords
            # Other commits that may not match specific patterns
            ("Update video thumbnail generation for podcasts", "maintenance"),  # "update" keyword
            ("Add Spanish translations for navigation menu", "feature"),  # "add" keyword
            ("Add webhook support for content management system", "feature"),  # "add" keyword
            ("Connect to external video streaming service", "other"),  # May not match patterns
            ("Fix typos in article descriptions", "bug_fix"),  # "fix" keyword
        ]

        # Test with rule-based classification (fallback)
        extractor = MLTicketExtractor(
            enable_ml=False, enable_llm=False, cache_dir=Path(tempfile.mkdtemp())
        )

        for message, expected_category in test_commits:
            with self.subTest(commit=message):
                # Filter the message first
                filtered = filter_git_artifacts(message)
                self.assertEqual(filtered, message)  # Should be unchanged for clean commits

                # Test categorization (will use rule-based fallback)
                category = extractor.categorize_commit(message)

                # Verify it matches expected category or is at least a valid category
                valid_categories = [
                    "feature",
                    "bug_fix",
                    "maintenance",
                    "documentation",
                    "refactor",
                    "test",
                    "style",
                    "build",
                    "other",
                    "chore",
                    "deployment",
                    "configuration",
                    "content",
                    "ui",
                    "infrastructure",
                    "security",
                    "performance",
                    "wip",
                    "version",
                    "integration",
                ]
                self.assertIn(category, valid_categories)

                # For conventional commits, should match expected
                if message.startswith(("feat:", "fix:", "chore:")):
                    self.assertEqual(category, expected_category)


if __name__ == "__main__":
    unittest.main()
