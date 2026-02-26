"""Unit tests for BedrockClassifier batch classification.

These tests mock the boto3 Bedrock client so no AWS credentials or network
access are required.  They exercise:

- Happy path: JSON array response covers all commits in one API call
- Partial response: fewer results than commits → fallback for the rest
- Parse failure: model returns garbage → full individual fallback
- Sub-batching: more than _API_BATCH_SIZE commits → multiple API calls
- Empty input: no commits → empty list immediately
- _parse_batch_json edge cases: text wrapped around the array
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to build a mocked BedrockClassifier without real AWS credentials
# ---------------------------------------------------------------------------


def _make_bedrock_response(text: str, input_tokens: int = 10, output_tokens: int = 20) -> dict:
    """Build the boto3 invoke_model response structure for the given text."""
    body_bytes = json.dumps(
        {
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
    ).encode()
    body_mock = MagicMock()
    body_mock.read.return_value = body_bytes
    return {"body": body_mock}


def _make_classifier() -> Any:
    """Return a BedrockClassifier with its boto3 client fully mocked."""
    from gitflow_analytics.qualitative.classifiers.llm.bedrock_client import (
        BedrockClassifier,
        BedrockConfig,
    )

    config = BedrockConfig(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        bedrock_model_id="anthropic.claude-3-haiku-20240307-v1:0",
        aws_region="us-east-1",
        temperature=0.1,
        max_tokens=50,
        timeout_seconds=10.0,
        max_retries=1,
    )

    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        classifier = BedrockClassifier(config=config)
        classifier._client = mock_client  # expose for assertions
        return classifier


# ---------------------------------------------------------------------------
# Tests for _parse_batch_json
# ---------------------------------------------------------------------------


class TestParseBatchJson:
    """Tests for the internal JSON parsing helper."""

    def setup_method(self) -> None:
        self.classifier = _make_classifier()

    def test_clean_json_array(self) -> None:
        payload = json.dumps(
            [
                {"category": "feature", "confidence": 0.9, "reasoning": "adds new stuff"},
                {"category": "bugfix", "confidence": 0.85, "reasoning": "fixes crash"},
            ]
        )
        result = self.classifier._parse_batch_json(payload, 2)
        assert result is not None
        assert len(result) == 2
        assert result[0]["category"] == "feature"
        assert result[1]["category"] == "bugfix"

    def test_json_wrapped_in_prose(self) -> None:
        """Model sometimes prepends/appends explanation text."""
        array_part = json.dumps(
            [{"category": "maintenance", "confidence": 0.7, "reasoning": "updates deps"}]
        )
        payload = f"Here is the result:\n{array_part}\nHope that helps!"
        result = self.classifier._parse_batch_json(payload, 1)
        assert result is not None
        assert result[0]["category"] == "maintenance"

    def test_garbage_returns_none(self) -> None:
        result = self.classifier._parse_batch_json("I cannot classify these commits.", 3)
        assert result is None

    def test_invalid_category_coerced_to_maintenance(self) -> None:
        payload = json.dumps(
            [{"category": "unknown_category", "confidence": 0.5, "reasoning": "odd commit"}]
        )
        result = self.classifier._parse_batch_json(payload, 1)
        assert result is not None
        assert result[0]["category"] == "maintenance"

    def test_confidence_clamped(self) -> None:
        payload = json.dumps(
            [{"category": "feature", "confidence": 1.5, "reasoning": "way too confident"}]
        )
        result = self.classifier._parse_batch_json(payload, 1)
        assert result is not None
        assert result[0]["confidence"] == 1.0

    def test_non_dict_items_replaced_with_fallback(self) -> None:
        payload = json.dumps(
            ["feature", None, {"category": "bugfix", "confidence": 0.9, "reasoning": "fix"}]
        )
        result = self.classifier._parse_batch_json(payload, 3)
        assert result is not None
        assert len(result) == 3
        assert result[0]["category"] == "maintenance"  # non-dict replaced
        assert result[1]["category"] == "maintenance"  # None replaced
        assert result[2]["category"] == "bugfix"


# ---------------------------------------------------------------------------
# Tests for classify_commits_batch (integration with mocked API)
# ---------------------------------------------------------------------------


_FIVE_COMMITS = [
    {"message": "feat: add user auth", "files_changed": []},
    {"message": "fix: crash on login", "files_changed": []},
    {"message": "chore: update deps", "files_changed": []},
    {"message": "docs: update README", "files_changed": []},
    {"message": "refactor: simplify DB queries", "files_changed": []},
]

_EXPECTED_BATCH_RESPONSE = json.dumps(
    [
        {"category": "feature", "confidence": 0.9, "reasoning": "adds user authentication"},
        {"category": "bugfix", "confidence": 0.88, "reasoning": "fixes login crash"},
        {"category": "maintenance", "confidence": 0.85, "reasoning": "updates dependencies"},
        {"category": "content", "confidence": 0.92, "reasoning": "updates documentation"},
        {"category": "maintenance", "confidence": 0.80, "reasoning": "refactors db queries"},
    ]
)


class TestClassifyCommitsBatch:
    """Integration tests for the high-level batch classification method."""

    def setup_method(self) -> None:
        self.classifier = _make_classifier()

    def test_happy_path_single_api_call(self) -> None:
        """5 commits → 1 API call, 5 results with method=llm_batch."""
        self.classifier._client.invoke_model.return_value = _make_bedrock_response(
            _EXPECTED_BATCH_RESPONSE
        )

        results = self.classifier.classify_commits_batch(_FIVE_COMMITS, batch_id="test-batch")

        assert len(results) == 5
        assert self.classifier._client.invoke_model.call_count == 1

        categories = [r.category for r in results]
        assert categories == ["feature", "bugfix", "maintenance", "content", "maintenance"]

        methods = [r.method for r in results]
        assert all(m == "llm_batch" for m in methods)

        batch_ids = [r.batch_id for r in results]
        assert all(b == "test-batch" for b in batch_ids)

    def test_empty_input_returns_empty_list(self) -> None:
        results = self.classifier.classify_commits_batch([])
        assert results == []
        self.classifier._client.invoke_model.assert_not_called()

    def test_api_failure_falls_back_to_individual(self) -> None:
        """When the batch API call raises, each commit is classified individually."""
        # First call raises; subsequent individual calls return single-commit format
        single_response = "feature 0.80 adds something\n"

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Bedrock service unavailable")
            return _make_bedrock_response(single_response)

        self.classifier._client.invoke_model.side_effect = side_effect

        results = self.classifier.classify_commits_batch(_FIVE_COMMITS)

        assert len(results) == 5
        # First call was the failed batch; then 5 individual calls
        assert self.classifier._client.invoke_model.call_count == 6

    def test_parse_failure_falls_back_to_individual(self) -> None:
        """When the batch response is not valid JSON, fall back to individual calls."""
        single_response = "maintenance 0.75 minor update\n"

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_bedrock_response("I cannot classify these commits.")
            return _make_bedrock_response(single_response)

        self.classifier._client.invoke_model.side_effect = side_effect

        results = self.classifier.classify_commits_batch(_FIVE_COMMITS)

        assert len(results) == 5
        # 1 failed batch call + 5 individual calls
        assert self.classifier._client.invoke_model.call_count == 6

    def test_sub_batching_for_large_input(self) -> None:
        """30 commits should produce 2 API calls (sub-batch size is 25)."""
        commits = [{"message": f"feat: feature {i}", "files_changed": []} for i in range(30)]
        # Build a response for 25 items (first sub-batch)
        response_25 = json.dumps(
            [{"category": "feature", "confidence": 0.8, "reasoning": "adds feature"}] * 25
        )
        # Build a response for 5 items (second sub-batch)
        response_5 = json.dumps(
            [{"category": "feature", "confidence": 0.8, "reasoning": "adds feature"}] * 5
        )

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_bedrock_response(response_25)
            return _make_bedrock_response(response_5)

        self.classifier._client.invoke_model.side_effect = side_effect

        results = self.classifier.classify_commits_batch(commits, batch_id="large-batch")

        assert len(results) == 30
        assert self.classifier._client.invoke_model.call_count == 2

    def test_partial_json_response_falls_back_for_missing(self) -> None:
        """If JSON has fewer items than commits, missing ones are classified individually."""
        # 5 commits but API returns only 3
        partial_response = json.dumps(
            [
                {"category": "feature", "confidence": 0.9, "reasoning": "feature work"},
                {"category": "bugfix", "confidence": 0.85, "reasoning": "bug fix"},
                {"category": "maintenance", "confidence": 0.7, "reasoning": "maintenance"},
            ]
        )
        single_response = "content 0.75 doc update\n"

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_bedrock_response(partial_response)
            return _make_bedrock_response(single_response)

        self.classifier._client.invoke_model.side_effect = side_effect

        results = self.classifier.classify_commits_batch(_FIVE_COMMITS)

        assert len(results) == 5
        # 1 batch call + 2 individual fallback calls for the missing items
        assert self.classifier._client.invoke_model.call_count == 3
        # First 3 from batch
        assert results[0].method == "llm_batch"
        assert results[1].method == "llm_batch"
        assert results[2].method == "llm_batch"
        # Last 2 from individual classification
        assert results[3].method in ("llm", "llm_error")
        assert results[4].method in ("llm", "llm_error")

    def test_cost_tracking_updated(self) -> None:
        """Successful batch call increments api_calls_made and token counters."""
        self.classifier._client.invoke_model.return_value = _make_bedrock_response(
            _EXPECTED_BATCH_RESPONSE, input_tokens=100, output_tokens=50
        )

        self.classifier.classify_commits_batch(_FIVE_COMMITS)

        assert self.classifier.api_calls_made == 1
        assert self.classifier.total_tokens_used == 150

    def test_batch_id_propagated(self) -> None:
        """batch_id is set on every ClassificationResult."""
        self.classifier._client.invoke_model.return_value = _make_bedrock_response(
            _EXPECTED_BATCH_RESPONSE
        )

        results = self.classifier.classify_commits_batch(_FIVE_COMMITS, batch_id="my-batch")

        assert all(r.batch_id == "my-batch" for r in results)
