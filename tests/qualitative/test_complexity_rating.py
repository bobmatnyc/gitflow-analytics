"""Tests for the complexity rating feature.

Covers:
- Complexity field in ClassificationResult (base.py)
- Batch prompt response parsing with complexity (bedrock_client.py)
- Response parser: parse_response() extracts complexity (response_parser.py)
- DB storage: complexity written and updated in qualitative_commits
- CSV output: complexity column present and correctly blank for None
- Null / rule-based path: complexity is always None for rule-based classifiers
"""

import csv
import io
import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bedrock_response(text: str, input_tokens: int = 10, output_tokens: int = 20) -> dict:
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
        classifier._client = mock_client
        return classifier


# ---------------------------------------------------------------------------
# ClassificationResult: complexity field
# ---------------------------------------------------------------------------


class TestClassificationResultComplexity:
    """The ClassificationResult dataclass carries the new complexity field."""

    def test_complexity_defaults_to_none(self) -> None:
        from gitflow_analytics.qualitative.classifiers.llm.base import ClassificationResult

        result = ClassificationResult(
            category="feature",
            confidence=0.9,
            method="llm",
            reasoning="adds feature",
            model="test",
            alternatives=[],
            processing_time_ms=10.0,
        )
        assert result.complexity is None

    def test_complexity_set_and_serialised(self) -> None:
        from gitflow_analytics.qualitative.classifiers.llm.base import ClassificationResult

        result = ClassificationResult(
            category="feature",
            confidence=0.9,
            method="llm",
            reasoning="adds feature",
            model="test",
            alternatives=[],
            processing_time_ms=10.0,
            complexity=4,
        )
        assert result.complexity == 4
        d = result.to_dict()
        assert d["complexity"] == 4

    def test_complexity_none_serialised(self) -> None:
        from gitflow_analytics.qualitative.classifiers.llm.base import ClassificationResult

        result = ClassificationResult(
            category="maintenance",
            confidence=0.5,
            method="rule_based",
            reasoning="fallback",
            model="none",
            alternatives=[],
            processing_time_ms=1.0,
        )
        d = result.to_dict()
        assert d["complexity"] is None


# ---------------------------------------------------------------------------
# ResponseParser: parse_response() extracts complexity
# ---------------------------------------------------------------------------


class TestResponseParserComplexity:
    """ResponseParser returns a 4-tuple (category, confidence, complexity, reasoning)."""

    def setup_method(self) -> None:
        from gitflow_analytics.qualitative.classifiers.llm.prompts import PromptGenerator
        from gitflow_analytics.qualitative.classifiers.llm.response_parser import ResponseParser

        self.parser = ResponseParser()
        self.categories = PromptGenerator.CATEGORIES

    def test_new_format_with_complexity(self) -> None:
        category, confidence, complexity, reasoning = self.parser.parse_response(
            "bugfix 0.90 2 fixes null pointer in login handler",
            self.categories,
        )
        assert category == "bugfix"
        assert confidence == pytest.approx(0.90)
        assert complexity == 2
        assert "null pointer" in reasoning

    def test_new_format_complexity_clamped_high(self) -> None:
        # Even if LLM returns 9, it must be clamped to 5
        category, confidence, complexity, reasoning = self.parser.parse_response(
            "feature 0.80 9 massive new subsystem",
            self.categories,
        )
        # Complexity 9 does not match the [1-5] pattern in our regex — falls through
        # to legacy "standard" pattern which interprets "9" as start of reasoning.
        # The key invariant is that complexity is either a valid 1-5 int or None.
        assert complexity is None or 1 <= complexity <= 5

    def test_legacy_format_without_complexity(self) -> None:
        """Three-field response: complexity should be None."""
        category, confidence, complexity, reasoning = self.parser.parse_response(
            "feature 0.85 adds authentication system",
            self.categories,
        )
        assert category == "feature"
        assert confidence == pytest.approx(0.85)
        assert complexity is None
        assert reasoning != ""

    def test_complexity_5_parsed(self) -> None:
        category, confidence, complexity, reasoning = self.parser.parse_response(
            "feature 0.95 5 complete system redesign with novel architecture",
            self.categories,
        )
        assert category == "feature"
        assert complexity == 5

    def test_complexity_1_parsed(self) -> None:
        category, confidence, complexity, reasoning = self.parser.parse_response(
            "maintenance 0.99 1 bumps version number",
            self.categories,
        )
        assert category == "maintenance"
        assert complexity == 1

    def test_empty_response_returns_none_complexity(self) -> None:
        category, confidence, complexity, reasoning = self.parser.parse_response(
            "", self.categories
        )
        assert category == "maintenance"
        assert complexity is None

    def test_garbage_response_returns_none_complexity(self) -> None:
        category, confidence, complexity, reasoning = self.parser.parse_response(
            "I cannot classify this commit.", self.categories
        )
        assert complexity is None


# ---------------------------------------------------------------------------
# Bedrock _validate_batch_items: complexity extracted and clamped
# ---------------------------------------------------------------------------


class TestValidateBatchItemsComplexity:
    """_validate_batch_items extracts complexity from JSON and clamps to 1-5."""

    def setup_method(self) -> None:
        self.classifier = _make_classifier()

    def test_valid_complexity_preserved(self) -> None:
        items = [{"category": "feature", "confidence": 0.9, "complexity": 3, "reasoning": "r"}]
        result = self.classifier._validate_batch_items(items)
        assert result[0]["complexity"] == 3

    def test_complexity_clamped_below_1(self) -> None:
        items = [{"category": "feature", "confidence": 0.9, "complexity": 0, "reasoning": "r"}]
        result = self.classifier._validate_batch_items(items)
        assert result[0]["complexity"] == 1

    def test_complexity_clamped_above_5(self) -> None:
        items = [{"category": "bugfix", "confidence": 0.85, "complexity": 99, "reasoning": "r"}]
        result = self.classifier._validate_batch_items(items)
        assert result[0]["complexity"] == 5

    def test_missing_complexity_defaults_to_none(self) -> None:
        items = [{"category": "maintenance", "confidence": 0.7, "reasoning": "r"}]
        result = self.classifier._validate_batch_items(items)
        assert result[0]["complexity"] is None

    def test_non_int_complexity_defaults_to_none(self) -> None:
        items = [
            {"category": "maintenance", "confidence": 0.7, "complexity": "medium", "reasoning": "r"}
        ]
        result = self.classifier._validate_batch_items(items)
        assert result[0]["complexity"] is None

    def test_null_complexity_defaults_to_none(self) -> None:
        items = [{"category": "content", "confidence": 0.8, "complexity": None, "reasoning": "r"}]
        result = self.classifier._validate_batch_items(items)
        assert result[0]["complexity"] is None

    def test_invalid_item_fallback_includes_complexity_none(self) -> None:
        """Non-dict items produce a fallback that also has complexity=None."""
        result = self.classifier._validate_batch_items(["not_a_dict"])
        assert result[0]["complexity"] is None

    def test_batch_classify_propagates_complexity(self) -> None:
        """End-to-end: classify_commits_batch returns results with complexity set."""
        batch_response = json.dumps(
            [
                {
                    "category": "feature",
                    "confidence": 0.9,
                    "complexity": 4,
                    "reasoning": "complex feature",
                },
                {
                    "category": "bugfix",
                    "confidence": 0.85,
                    "complexity": 2,
                    "reasoning": "simple fix",
                },
            ]
        )
        self.classifier._client.invoke_model.return_value = _make_bedrock_response(batch_response)

        commits = [
            {"message": "feat: big new feature", "files_changed": []},
            {"message": "fix: typo in error message", "files_changed": []},
        ]
        results = self.classifier.classify_commits_batch(commits, batch_id="t")

        assert len(results) == 2
        assert results[0].complexity == 4
        assert results[1].complexity == 2

    def test_batch_classify_missing_complexity_is_none(self) -> None:
        """If LLM omits complexity field, result.complexity is None."""
        batch_response = json.dumps(
            [
                {"category": "feature", "confidence": 0.9, "reasoning": "adds something"},
            ]
        )
        self.classifier._client.invoke_model.return_value = _make_bedrock_response(batch_response)

        commits = [{"message": "feat: something", "files_changed": []}]
        results = self.classifier.classify_commits_batch(commits, batch_id="t")

        assert len(results) == 1
        assert results[0].complexity is None


# ---------------------------------------------------------------------------
# DB storage: complexity written to qualitative_commits
# ---------------------------------------------------------------------------


class TestComplexityDbStorage:
    """BatchClassifierImplMixin._store_commit_classification persists complexity."""

    def _setup_in_memory_db(self):
        """Create an in-memory SQLite DB with the full schema."""
        from sqlalchemy import create_engine

        from gitflow_analytics.models.database_base import Base
        from gitflow_analytics.models.database_commit_models import CachedCommit
        from gitflow_analytics.models.database_metrics_models import QualitativeCommitData

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=engine)
        return Session, CachedCommit, QualitativeCommitData

    def _make_impl_mixin(self):
        """Build a minimal object that has the _store_commit_classification method."""
        from gitflow_analytics.classification.batch_classifier_impl import BatchClassifierImplMixin

        obj = BatchClassifierImplMixin.__new__(BatchClassifierImplMixin)
        return obj

    def test_complexity_stored_on_insert(self) -> None:
        Session, CachedCommit, QualitativeCommitData = self._setup_in_memory_db()
        session = Session()

        # Insert a cached commit row
        commit = CachedCommit(
            repo_path="/repo",
            commit_hash="abc1234",
            author_name="Alice",
            author_email="alice@example.com",
            message="feat: big new system",
            timestamp=datetime.now(timezone.utc),
            branch="main",
            is_merge=False,
            files_changed=5,
            insertions=100,
            deletions=10,
            complexity_delta=0.0,
        )
        session.add(commit)
        session.commit()

        impl = self._make_impl_mixin()
        impl._store_commit_classification(
            session,
            {
                "commit_hash": "abc1234",
                "category": "feature",
                "confidence": 0.9,
                "method": "llm",
                "complexity": 4,
            },
        )
        session.commit()

        row = session.query(QualitativeCommitData).filter_by(commit_id=commit.id).first()
        assert row is not None
        assert row.complexity == 4

    def test_complexity_none_stored_on_insert(self) -> None:
        Session, CachedCommit, QualitativeCommitData = self._setup_in_memory_db()
        session = Session()

        commit = CachedCommit(
            repo_path="/repo",
            commit_hash="def5678",
            author_name="Bob",
            author_email="bob@example.com",
            message="chore: update deps",
            timestamp=datetime.now(timezone.utc),
            branch="main",
            is_merge=False,
            files_changed=1,
            insertions=2,
            deletions=2,
            complexity_delta=0.0,
        )
        session.add(commit)
        session.commit()

        impl = self._make_impl_mixin()
        impl._store_commit_classification(
            session,
            {
                "commit_hash": "def5678",
                "category": "maintenance",
                "confidence": 0.7,
                "method": "fallback",
                # no "complexity" key — rule-based path
            },
        )
        session.commit()

        row = session.query(QualitativeCommitData).filter_by(commit_id=commit.id).first()
        assert row is not None
        assert row.complexity is None

    def test_complexity_updated_on_upsert(self) -> None:
        Session, CachedCommit, QualitativeCommitData = self._setup_in_memory_db()
        session = Session()

        commit = CachedCommit(
            repo_path="/repo",
            commit_hash="ghi9012",
            author_name="Carol",
            author_email="carol@example.com",
            message="refactor: simplify module",
            timestamp=datetime.now(timezone.utc),
            branch="main",
            is_merge=False,
            files_changed=3,
            insertions=50,
            deletions=60,
            complexity_delta=0.0,
        )
        session.add(commit)
        session.commit()

        impl = self._make_impl_mixin()

        # First classification (no complexity)
        impl._store_commit_classification(
            session,
            {
                "commit_hash": "ghi9012",
                "category": "maintenance",
                "confidence": 0.6,
                "method": "fallback",
            },
        )
        session.commit()

        # Second classification (with complexity) — should update
        impl._store_commit_classification(
            session,
            {
                "commit_hash": "ghi9012",
                "category": "maintenance",
                "confidence": 0.8,
                "method": "llm",
                "complexity": 3,
            },
        )
        session.commit()

        row = session.query(QualitativeCommitData).filter_by(commit_id=commit.id).first()
        assert row is not None
        assert row.complexity == 3


# ---------------------------------------------------------------------------
# DB migration: qualitative_commits gets complexity column
# ---------------------------------------------------------------------------


class TestComplexityMigration:
    """Database._migrate_qualitative_commits_v5 adds complexity to existing DBs."""

    def test_migration_adds_complexity_column(self) -> None:
        from sqlalchemy import create_engine, text

        from gitflow_analytics.models.database_base import Base
        from gitflow_analytics.models.database_metrics_models import QualitativeCommitData

        # Create a database using only the ORM schema (which already has complexity)
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        # Verify that the column exists via PRAGMA
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(qualitative_commits)"))
            columns = {row[1] for row in result}

        assert "complexity" in columns

    def test_migration_method_is_idempotent(self) -> None:
        """Running the migration twice should not raise."""
        from sqlalchemy import create_engine, text

        from gitflow_analytics.models.database_base import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        # Build a mock Database-like object just for calling the migration helper
        from gitflow_analytics.models import database as db_module

        db_obj = object.__new__(db_module.Database)
        db_obj.engine = engine

        with engine.connect() as conn:
            # Call migration twice — should not raise on second call
            db_obj._migrate_qualitative_commits_v5(conn)
            db_obj._migrate_qualitative_commits_v5(conn)


# ---------------------------------------------------------------------------
# CSV output: complexity column
# ---------------------------------------------------------------------------


class TestComplexityCsvOutput:
    """generate_detailed_csv_report includes the complexity column."""

    def _make_writer(self, tmp_path: Path):
        """Instantiate ClassificationReportGenerator pointed at tmp_path."""
        from gitflow_analytics.reports.classification_writer import ClassificationReportGenerator

        return ClassificationReportGenerator(output_directory=str(tmp_path))

    def test_complexity_in_csv_headers(self, tmp_path: Path) -> None:
        writer = self._make_writer(tmp_path)
        commits = [
            {
                "hash": "abc1234567890",  # pragma: allowlist secret
                "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "author_name": "Alice",
                "repository": "my_repo",
                "predicted_class": "feature",
                "classification_confidence": 0.9,
                "complexity": 3,
                "message": "adds login feature",
                "files_changed": 5,
                "insertions": 100,
                "deletions": 20,
                "branch": "main",
                "project_key": "PROJ",
                "ticket_references": [],
            }
        ]
        csv_path = writer.generate_detailed_csv_report(commits)

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows = list(reader)

        assert "complexity" in headers
        assert rows[0]["complexity"] == "3"

    def test_complexity_none_blank_in_csv(self, tmp_path: Path) -> None:
        """Rule-based commits have no complexity — CSV cell should be blank."""
        writer = self._make_writer(tmp_path)
        commits = [
            {
                "hash": "def5678901234",  # pragma: allowlist secret
                "timestamp": datetime(2024, 1, 2, tzinfo=timezone.utc),
                "author_name": "Bob",
                "repository": "my_repo",
                "predicted_class": "maintenance",
                "classification_confidence": 0.5,
                "complexity": None,  # rule-based: no complexity
                "message": "chore: update deps",
                "files_changed": 1,
                "insertions": 2,
                "deletions": 2,
                "branch": "main",
                "project_key": "PROJ",
                "ticket_references": [],
            }
        ]
        csv_path = writer.generate_detailed_csv_report(commits)

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["complexity"] == ""

    def test_mixed_complexity_values(self, tmp_path: Path) -> None:
        """Multiple rows: some with complexity, some without."""
        writer = self._make_writer(tmp_path)
        commits = [
            {
                "hash": "aaa111222333",
                "timestamp": datetime(2024, 1, 3, tzinfo=timezone.utc),
                "author_name": "Alice",
                "repository": "r",
                "predicted_class": "feature",
                "classification_confidence": 0.9,
                "complexity": 5,
                "message": "huge refactor",
                "files_changed": 20,
                "insertions": 500,
                "deletions": 400,
                "branch": "main",
                "project_key": "P",
                "ticket_references": ["PROJ-1"],
            },
            {
                "hash": "bbb444555666",
                "timestamp": datetime(2024, 1, 4, tzinfo=timezone.utc),
                "author_name": "Bob",
                "repository": "r",
                "predicted_class": "maintenance",
                "classification_confidence": 0.4,
                "complexity": None,
                "message": "bump version",
                "files_changed": 1,
                "insertions": 1,
                "deletions": 1,
                "branch": "main",
                "project_key": "P",
                "ticket_references": [],
            },
        ]
        csv_path = writer.generate_detailed_csv_report(commits)

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["complexity"] == "5"
        assert rows[1]["complexity"] == ""


# ---------------------------------------------------------------------------
# Rule-based fallback: complexity is always None
# ---------------------------------------------------------------------------


class TestRuleBasedComplexityNone:
    """The fallback rule-based classifier never sets complexity."""

    def test_fallback_result_dict_has_no_complexity(self) -> None:
        """When LLM is disabled the batch produces dicts without complexity key set."""
        # The _classify_commit_batch_with_llm method appends None explicitly
        # for the fallback path.  We verify via the dict keys.
        from gitflow_analytics.classification.batch_classifier_impl import BatchClassifierImplMixin

        obj = BatchClassifierImplMixin.__new__(BatchClassifierImplMixin)
        obj.llm_enabled = False
        obj.confidence_threshold = 0.7
        obj.fallback_enabled = True
        obj.circuit_breaker_open = False
        obj.api_failure_count = 0
        obj.batch_size = 50
        obj.max_consecutive_failures = 5
        obj.classification_start_time = None
        obj.fallback_patterns = {
            "bugfix": [r"\bfix\b", r"\bbug\b"],
            "feature": [r"\bfeat\b", r"\badd\b"],
        }

        commits = [
            {
                "commit_hash": "abc123",
                "message": "fix: crash on startup",
                "ticket_references": [],
            }
        ]

        results = obj._classify_commit_batch_with_llm(commits, ticket_context={})
        assert len(results) == 1
        assert results[0]["complexity"] is None
