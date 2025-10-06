"""Test circuit breaker functionality in BatchCommitClassifier."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gitflow_analytics.classification.batch_classifier import BatchCommitClassifier
from gitflow_analytics.qualitative.classifiers.llm_commit_classifier import LLMConfig


class TestCircuitBreaker:
    """Test circuit breaker pattern in batch classification."""

    @pytest.fixture
    def mock_llm_config(self):
        """Create a mock LLM config with API key."""
        return LLMConfig(
            api_key="test-key",
            model="test-model",
            timeout_seconds=5.0,
            max_retries=1,
        )

    @pytest.fixture
    def classifier(self, tmp_path, mock_llm_config):
        """Create a batch classifier for testing."""
        return BatchCommitClassifier(
            cache_dir=tmp_path,
            llm_config=mock_llm_config,
            batch_size=10,
            confidence_threshold=0.7,
            fallback_enabled=True,
        )

    def test_circuit_breaker_initialization(self, classifier):
        """Test circuit breaker is initialized properly."""
        assert classifier.api_failure_count == 0
        assert classifier.max_consecutive_failures == 5
        assert classifier.circuit_breaker_open is False

    def test_circuit_breaker_opens_after_failures(self, classifier):
        """Test circuit breaker opens after consecutive failures."""
        # Create mock commits
        commits = [
            {
                "commit_hash": f"hash{i}",
                "message": f"Test commit {i}",
                "ticket_references": [],
            }
            for i in range(3)
        ]
        ticket_context = {}

        # Mock the LLM classifier to raise exceptions
        with patch.object(
            classifier.llm_classifier,
            "classify_commits_batch",
            side_effect=Exception("API timeout"),
        ):
            # Simulate 5 consecutive failures
            for i in range(5):
                result = classifier._classify_commit_batch_with_llm(commits, ticket_context)

                # Verify fallback was used
                assert len(result) == len(commits)
                assert all(
                    r["method"] in ["fallback_only", "circuit_breaker_fallback"] for r in result
                )

                # Check failure count increments
                assert classifier.api_failure_count == i + 1

            # Verify circuit breaker is open
            assert classifier.circuit_breaker_open is True
            assert classifier.api_failure_count == 5

    def test_circuit_breaker_skips_llm_when_open(self, classifier):
        """Test that LLM is skipped when circuit breaker is open."""
        # Manually open circuit breaker
        classifier.circuit_breaker_open = True
        classifier.api_failure_count = 5

        commits = [{"commit_hash": "hash1", "message": "Test commit", "ticket_references": []}]
        ticket_context = {}

        # Mock should not be called when circuit breaker is open
        with patch.object(
            classifier.llm_classifier,
            "classify_commits_batch",
        ) as mock_classify:
            result = classifier._classify_commit_batch_with_llm(commits, ticket_context)

            # Verify LLM was not called
            mock_classify.assert_not_called()

            # Verify fallback was used with circuit breaker method
            assert len(result) == 1
            assert result[0]["method"] == "circuit_breaker_fallback"
            assert "Circuit breaker open" in result[0]["error"]

    def test_circuit_breaker_resets_on_success(self, classifier):
        """Test circuit breaker resets after successful LLM call."""
        # Set some failures
        classifier.api_failure_count = 3

        commits = [{"commit_hash": "hash1", "message": "Test commit", "ticket_references": []}]
        ticket_context = {}

        # Mock successful LLM response
        mock_result = [{"category": "feature", "confidence": 0.8, "method": "llm"}]

        with patch.object(
            classifier.llm_classifier,
            "classify_commits_batch",
            return_value=mock_result,
        ):
            result = classifier._classify_commit_batch_with_llm(commits, ticket_context)

            # Verify circuit breaker was reset
            assert classifier.api_failure_count == 0
            assert classifier.circuit_breaker_open is False

            # Verify LLM result was used
            assert result[0]["method"] == "llm"
            assert result[0]["category"] == "feature"

    def test_fallback_classification_quality(self, classifier):
        """Test that fallback classification still provides reasonable results."""
        # Open circuit breaker
        classifier.circuit_breaker_open = True

        commits = [
            {
                "commit_hash": "hash1",
                "message": "feat: add new feature",
                "ticket_references": [],
            },
            {
                "commit_hash": "hash2",
                "message": "fix: resolve bug",
                "ticket_references": [],
            },
            {
                "commit_hash": "hash3",
                "message": "chore: update dependencies",
                "ticket_references": [],
            },
        ]
        ticket_context = {}

        result = classifier._classify_commit_batch_with_llm(commits, ticket_context)

        # Verify all commits were classified
        assert len(result) == 3

        # Verify reasonable fallback categories (rule-based matching)
        categories = [r["category"] for r in result]
        assert "feature" in categories or "maintenance" in categories
        assert all(r["method"] == "circuit_breaker_fallback" for r in result)

    def test_logging_when_circuit_breaker_opens(self, classifier, caplog):
        """Test that appropriate logging occurs when circuit breaker opens."""
        commits = [{"commit_hash": "hash1", "message": "Test commit", "ticket_references": []}]
        ticket_context = {}

        with patch.object(
            classifier.llm_classifier,
            "classify_commits_batch",
            side_effect=Exception("API timeout"),
        ):
            with caplog.at_level(logging.ERROR):
                # Trigger 5 consecutive failures
                for _ in range(5):
                    classifier._classify_commit_batch_with_llm(commits, ticket_context)

                # Verify circuit breaker open message was logged
                assert any("CIRCUIT BREAKER OPENED" in record.message for record in caplog.records)
                assert any(
                    "consecutive API failures" in record.message for record in caplog.records
                )

    def test_reduced_timeouts_config(self, mock_llm_config):
        """Test that default timeouts have been reduced."""
        # Verify timeout is reduced to 5 seconds
        assert mock_llm_config.timeout_seconds == 5.0

        # Verify retries are reduced to 1
        assert mock_llm_config.max_retries == 1
