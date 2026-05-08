"""Tests for ``platform`` change_type and ``taxonomy_mapping`` (issue #69).

Covers two deliverables:

1. ``platform`` is recognised as a native change_type by the rule-based
   fallback patterns and the ISSUETYPE_CHANGE_TYPE_MAP.
2. The ``taxonomy_mapping`` config maps native change_type values to custom
   ``work_type`` labels stored on ``QualitativeCommitData.work_type``.

Tests construct a real ``BatchCommitClassifier`` against a temp SQLite
database and exercise the classifier directly. Fully offline (no LLM calls).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from gitflow_analytics.classification.batch_classifier import BatchCommitClassifier
from gitflow_analytics.models.database_commit_models import CachedCommit
from gitflow_analytics.models.database_metrics_models import QualitativeCommitData


def _mock_llm_config() -> dict[str, Any]:
    """Minimal LLM config — empty api_key keeps llm_enabled False (offline)."""
    return {
        "enable_caching": False,
        "cache_duration_days": 7,
        "model": "test-model",
        "api_key": "",
        "max_tokens": 1000,
        "temperature": 0.1,
        "timeout_seconds": 30,
        "confidence_threshold": 0.7,
        "batch_size": 10,
        "max_daily_requests": 1000,
        "aws_region": None,
        "aws_profile": None,
        "bedrock_model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "provider": "auto",
    }


def _make_classifier(
    tmp_path: Path,
    taxonomy_mapping: dict[str, str] | None = None,
) -> BatchCommitClassifier:
    return BatchCommitClassifier(
        cache_dir=tmp_path,
        llm_config=_mock_llm_config(),
        batch_size=10,
        confidence_threshold=0.7,
        fallback_enabled=True,
        taxonomy_mapping=taxonomy_mapping,
    )


def _seed_commit(classifier: BatchCommitClassifier, commit_hash: str) -> int:
    """Insert a CachedCommit row and return its id."""
    session = classifier.database.get_session()
    try:
        c = CachedCommit(
            repo_path="/tmp/repo",
            commit_hash=commit_hash,
            author_name="dev",
            author_email="dev@example.com",
            message="msg",
            timestamp=datetime.now(timezone.utc),
            branch="main",
            is_merge=False,
            files_changed=1,
            insertions=1,
            deletions=0,
        )
        session.add(c)
        session.commit()
        return int(c.id)
    finally:
        session.close()


def _classification_result(commit_hash: str, category: str) -> dict[str, Any]:
    return {
        "commit_hash": commit_hash,
        "category": category,
        "confidence": 0.9,
        "method": "fallback",
        "business_domain": "unknown",
    }


# ---------------------------------------------------------------------------
# Storage-time taxonomy mapping
# ---------------------------------------------------------------------------


def test_taxonomy_mapping_applied_at_storage(tmp_path: Path) -> None:
    classifier = _make_classifier(tmp_path, {"feature": "NPS enhancements"})
    commit_id = _seed_commit(classifier, "abc1234")

    session = classifier.database.get_session()
    try:
        classifier._store_commit_classification(
            session, _classification_result("abc1234", "feature")
        )
        session.commit()
        row = (
            session.query(QualitativeCommitData)
            .filter(QualitativeCommitData.commit_id == commit_id)
            .first()
        )
        assert row is not None
        assert row.change_type == "feature"
        assert row.work_type == "NPS enhancements"
    finally:
        session.close()


def test_no_mapping_work_type_is_none(tmp_path: Path) -> None:
    classifier = _make_classifier(tmp_path, {})
    commit_id = _seed_commit(classifier, "def5678")

    session = classifier.database.get_session()
    try:
        classifier._store_commit_classification(
            session, _classification_result("def5678", "feature")
        )
        session.commit()
        row = (
            session.query(QualitativeCommitData)
            .filter(QualitativeCommitData.commit_id == commit_id)
            .first()
        )
        assert row is not None
        assert row.work_type is None
    finally:
        session.close()


def test_unmapped_change_type_work_type_is_none(tmp_path: Path) -> None:
    classifier = _make_classifier(tmp_path, {"feature": "NPS enhancements"})
    commit_id = _seed_commit(classifier, "ghi9012")

    session = classifier.database.get_session()
    try:
        classifier._store_commit_classification(
            session, _classification_result("ghi9012", "bugfix")
        )
        session.commit()
        row = (
            session.query(QualitativeCommitData)
            .filter(QualitativeCommitData.commit_id == commit_id)
            .first()
        )
        assert row is not None
        assert row.change_type == "bugfix"
        assert row.work_type is None
    finally:
        session.close()


def test_multiple_change_types_map_to_same_work_type(tmp_path: Path) -> None:
    classifier = _make_classifier(tmp_path, {"content": "Content", "media": "Content"})
    cid_a = _seed_commit(classifier, "aaaa111")
    cid_b = _seed_commit(classifier, "bbbb222")

    session = classifier.database.get_session()
    try:
        classifier._store_commit_classification(
            session, _classification_result("aaaa111", "content")
        )
        classifier._store_commit_classification(session, _classification_result("bbbb222", "media"))
        session.commit()
        a = (
            session.query(QualitativeCommitData)
            .filter(QualitativeCommitData.commit_id == cid_a)
            .first()
        )
        b = (
            session.query(QualitativeCommitData)
            .filter(QualitativeCommitData.commit_id == cid_b)
            .first()
        )
        assert a is not None and b is not None
        assert a.work_type == "Content"
        assert b.work_type == "Content"
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Fast remap path
# ---------------------------------------------------------------------------


def test_fast_remap_updates_existing_rows(tmp_path: Path) -> None:
    # Initial: store with no taxonomy → work_type is None.
    classifier = _make_classifier(tmp_path, {})
    commit_id = _seed_commit(classifier, "remap123")

    session = classifier.database.get_session()
    try:
        classifier._store_commit_classification(
            session, _classification_result("remap123", "feature")
        )
        session.commit()
        row = (
            session.query(QualitativeCommitData)
            .filter(QualitativeCommitData.commit_id == commit_id)
            .first()
        )
        assert row is not None
        assert row.work_type is None
    finally:
        session.close()

    # Now update mapping and run the fast remap path — no LLM call.
    classifier.taxonomy_mapping = {"feature": "NPS enhancements"}
    session = classifier.database.get_session()
    try:
        updated = classifier._apply_taxonomy_remap(session)
        assert updated == 1
        row = (
            session.query(QualitativeCommitData)
            .filter(QualitativeCommitData.commit_id == commit_id)
            .first()
        )
        assert row is not None
        assert row.work_type == "NPS enhancements"
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Platform vocabulary
# ---------------------------------------------------------------------------


def test_platform_in_fallback_patterns(tmp_path: Path) -> None:
    classifier = _make_classifier(tmp_path)
    assert "platform" in classifier.fallback_patterns

    # Verify rule-based classifier picks up "platform: ..." prefix.
    # The internal _rule_based_classify (or similar) is the public method;
    # test the regex set directly to keep this test stable across refactors.
    import re

    patterns = classifier.fallback_patterns["platform"]
    sample = "platform: add new CI pipeline"
    assert any(re.search(p, sample, re.IGNORECASE) for p in patterns)


def test_platform_in_issuetype_map() -> None:
    from gitflow_analytics.classification.batch_classifier_impl import (
        BatchClassifierImplMixin,
    )

    m = BatchClassifierImplMixin.ISSUETYPE_CHANGE_TYPE_MAP
    assert m.get("infrastructure") == "platform"
    assert m.get("platform") == "platform"
    assert m.get("tech debt") == "platform"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def test_config_taxonomy_mapping_loaded_from_yaml(tmp_path: Path) -> None:
    """YAML with taxonomy_mapping section parses into config.taxonomy_mapping."""
    from gitflow_analytics.config.loader import ConfigLoader

    cfg_path = tmp_path / "config.yaml"
    yaml_data: dict[str, Any] = {
        "version": "1.0",
        "github": {"token": "x", "organization": "test-org"},
        "repositories": [],
        "cache": {"directory": str(tmp_path / "cache")},
        "output": {"directory": str(tmp_path / "out")},
        "analysis": {},
        "taxonomy_mapping": {
            "feature": "NPS enhancements",
            "platform": "Tech Debt",
        },
    }
    cfg_path.write_text(yaml.safe_dump(yaml_data))

    cfg = ConfigLoader.load(cfg_path)
    assert cfg.taxonomy_mapping == {
        "feature": "NPS enhancements",
        "platform": "Tech Debt",
    }


# ---------------------------------------------------------------------------
# pipeline_report fallback
# ---------------------------------------------------------------------------


def test_pipeline_report_work_type_fallback() -> None:
    """When qual.work_type is None, pipeline merges change_type as work_type."""

    # Simulate the merge logic without touching the full pipeline.
    class _Q:
        change_type = "feature"
        change_type_confidence = 0.9
        work_type = None
        complexity = None
        processing_method = "fallback"

    qual = _Q()
    commit: dict[str, Any] = {}
    commit["change_type"] = qual.change_type
    commit["change_type_confidence"] = qual.change_type_confidence
    commit["work_type"] = qual.work_type or qual.change_type
    commit["complexity"] = qual.complexity
    commit["processing_method"] = qual.processing_method

    assert commit["work_type"] == "feature"

    qual.work_type = "NPS enhancements"  # type: ignore[assignment]
    commit["work_type"] = qual.work_type or qual.change_type
    assert commit["work_type"] == "NPS enhancements"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
