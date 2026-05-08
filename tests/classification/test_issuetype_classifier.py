"""Tests for the JIRA/GH issuetype tier-1.5 classifier (issue #68).

The tier-1.5 signal sits between manual overrides (tier 1) and the JIRA
project-key tier (tier 3) in ``BatchClassifierImplMixin``. It routes
issue-linked commits straight to a work_type using the ticket's issuetype
field (Bug → bugfix, Story → feature, etc.) so the LLM is never called for
commits that already carry strong ticket signal.

These tests construct a real ``BatchCommitClassifier`` against a temp SQLite
database, seed ``DetailedTicketData`` / ``IssueCache`` rows, and exercise
the classifier method directly. Tests are fully offline (no LLM calls).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gitflow_analytics.classification.batch_classifier import BatchCommitClassifier
from gitflow_analytics.models.database import DetailedTicketData, IssueCache

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


@pytest.fixture()
def classifier(
    tmp_path: Path,
) -> BatchCommitClassifier:
    """Construct a classifier with a real (empty) SQLite DB in a tmp dir."""
    return BatchCommitClassifier(
        cache_dir=tmp_path,
        llm_config=_mock_llm_config(),
        batch_size=10,
        confidence_threshold=0.7,
        fallback_enabled=True,
    )


def _seed_detailed_ticket(
    classifier: BatchCommitClassifier,
    *,
    ticket_id: str,
    ticket_type: str | None,
    labels: list[str] | None = None,
    project_key: str = "ADV",
    platform: str = "jira",
) -> None:
    """Seed a DetailedTicketData row for tier-1.5 lookup."""
    session = classifier.database.get_session()
    try:
        session.add(
            DetailedTicketData(
                platform=platform,
                ticket_id=ticket_id,
                project_key=project_key,
                title=f"Ticket {ticket_id}",
                ticket_type=ticket_type,
                labels=labels or [],
            )
        )
        session.commit()
    finally:
        session.close()


def _seed_issue_cache(
    classifier: BatchCommitClassifier,
    *,
    issue_id: str,
    issue_type: str | None = None,
    labels: list[str] | None = None,
    platform_data: dict[str, Any] | None = None,
    project_key: str = "ADV",
    platform: str = "jira",
) -> None:
    """Seed an IssueCache row for tier-1.5 lookup."""
    session = classifier.database.get_session()
    try:
        session.add(
            IssueCache(
                platform=platform,
                issue_id=issue_id,
                project_key=project_key,
                title=f"Issue {issue_id}",
                issue_type=issue_type,
                labels=labels or [],
                platform_data=platform_data or {},
            )
        )
        session.commit()
    finally:
        session.close()


def _commit(commit_hash: str, ticket_refs: list[Any], message: str = "test") -> dict[str, Any]:
    return {
        "commit_hash": commit_hash,
        "message": message,
        "ticket_references": ticket_refs,
    }


def _classify(
    classifier: BatchCommitClassifier,
    commit: dict[str, Any],
) -> tuple[str, str, str] | None:
    """Run tier-1.5 classifier with a managed session."""
    session = classifier.database.get_session()
    try:
        return classifier._classify_via_issuetype(commit, session)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# 1. Happy paths — DetailedTicketData
# ---------------------------------------------------------------------------


def test_bug_issuetype_maps_to_bugfix(classifier: BatchCommitClassifier) -> None:
    """DetailedTicketData.ticket_type='Bug' → ('bugfix', 'Bug', 'unknown')."""
    _seed_detailed_ticket(classifier, ticket_id="ADV-1", ticket_type="Bug")

    result = _classify(
        classifier,
        _commit("h1", [{"platform": "jira", "id": "ADV-1"}]),
    )

    assert result is not None
    work_type, issuetype_name, business_domain = result
    assert work_type == "bugfix"
    assert issuetype_name == "Bug"
    assert business_domain == "unknown"


def test_story_issuetype_maps_to_feature(classifier: BatchCommitClassifier) -> None:
    """DetailedTicketData.ticket_type='Story' → ('feature', 'Story', ...)."""
    _seed_detailed_ticket(classifier, ticket_id="ADV-2", ticket_type="Story")

    result = _classify(
        classifier,
        _commit("h2", [{"platform": "jira", "id": "ADV-2"}]),
    )

    assert result is not None
    assert result[0] == "feature"
    assert result[1] == "Story"


# ---------------------------------------------------------------------------
# 2. Ambiguous "Task" disambiguation via labels
# ---------------------------------------------------------------------------


def test_task_with_platform_label_maps_to_maintenance(
    classifier: BatchCommitClassifier,
) -> None:
    """IssueCache issuetype='Task' + labels=['platform'] → 'maintenance'."""
    _seed_issue_cache(
        classifier,
        issue_id="ADV-3",
        issue_type="Task",
        labels=["platform"],
    )

    result = _classify(
        classifier,
        _commit("h3", [{"platform": "jira", "id": "ADV-3"}]),
    )

    assert result is not None
    assert result[0] == "maintenance"
    assert result[1] == "Task"


def test_task_without_labels_falls_through(classifier: BatchCommitClassifier) -> None:
    """IssueCache issuetype='Task' with no labels returns None (let LLM decide)."""
    _seed_issue_cache(
        classifier,
        issue_id="ADV-4",
        issue_type="Task",
        labels=[],
    )

    result = _classify(
        classifier,
        _commit("h4", [{"platform": "jira", "id": "ADV-4"}]),
    )

    assert result is None


# ---------------------------------------------------------------------------
# 3. Negative paths
# ---------------------------------------------------------------------------


def test_no_ticket_refs_falls_through(classifier: BatchCommitClassifier) -> None:
    """Commit with no ticket_references returns None."""
    result = _classify(classifier, _commit("h5", []))
    assert result is None


def test_unknown_issuetype_falls_through(classifier: BatchCommitClassifier) -> None:
    """Issuetype not in the mapping returns None so LLM can decide."""
    _seed_detailed_ticket(
        classifier,
        ticket_id="ADV-6",
        ticket_type="Weird Custom Type",
    )

    result = _classify(
        classifier,
        _commit("h6", [{"platform": "jira", "id": "ADV-6"}]),
    )

    assert result is None


def test_no_matching_ticket_in_db_falls_through(classifier: BatchCommitClassifier) -> None:
    """Commit references a ticket id that has no DB row → None."""
    result = _classify(
        classifier,
        _commit("h_missing", [{"platform": "jira", "id": "ADV-999"}]),
    )
    assert result is None


# ---------------------------------------------------------------------------
# 4. Fallback to platform_data when issue_type column is empty
# ---------------------------------------------------------------------------


def test_issuetype_from_platform_data_fallback(
    classifier: BatchCommitClassifier,
) -> None:
    """IssueCache with no issue_type but platform_data containing issuetype → match."""
    _seed_issue_cache(
        classifier,
        issue_id="ADV-7",
        issue_type=None,
        platform_data={"fields": {"issuetype": {"name": "Bug"}}},
    )

    result = _classify(
        classifier,
        _commit("h7", [{"platform": "jira", "id": "ADV-7"}]),
    )

    assert result is not None
    assert result[0] == "bugfix"
    assert result[1] == "Bug"


# ---------------------------------------------------------------------------
# 5. business_domain derivation
# ---------------------------------------------------------------------------


def test_business_domain_from_components(classifier: BatchCommitClassifier) -> None:
    """First component name becomes business_domain (lower-cased)."""
    _seed_issue_cache(
        classifier,
        issue_id="ADV-8",
        issue_type="Bug",
        platform_data={
            "fields": {
                "issuetype": {"name": "Bug"},
                "components": [{"name": "Platform"}, {"name": "Billing"}],
            }
        },
    )

    result = _classify(
        classifier,
        _commit("h8", [{"platform": "jira", "id": "ADV-8"}]),
    )

    assert result is not None
    assert result[2] == "platform"


def test_business_domain_unknown_when_no_components(
    classifier: BatchCommitClassifier,
) -> None:
    """No components and no domain-keyword labels → business_domain='unknown'."""
    _seed_detailed_ticket(
        classifier,
        ticket_id="ADV-9",
        ticket_type="Bug",
        labels=["nothing-relevant"],
    )

    result = _classify(
        classifier,
        _commit("h9", [{"platform": "jira", "id": "ADV-9"}]),
    )

    assert result is not None
    assert result[2] == "unknown"


# ---------------------------------------------------------------------------
# 6. Cascade integration: tier 1.5 precedes JIRA project-key tier
# ---------------------------------------------------------------------------


def test_tier1_5_precedes_jira_project_key(tmp_path: Path) -> None:
    """When both tier 1.5 (Bug issuetype) and tier 3 (project-key=feature) match,
    tier 1.5 wins because it fires first in the cascade.

    Setup: ADV project-key is mapped to "feature", but the linked ticket is a
    Bug. Without tier 1.5 the commit would be mis-classified as feature; with
    tier 1.5 it correctly resolves to bugfix.
    """
    classifier = BatchCommitClassifier(
        cache_dir=tmp_path,
        llm_config=_mock_llm_config(),
        batch_size=10,
        confidence_threshold=0.7,
        fallback_enabled=True,
        jira_project_mappings={"ADV": "feature"},
    )
    classifier.llm_enabled = False  # Force offline
    _seed_detailed_ticket(classifier, ticket_id="ADV-100", ticket_type="Bug")

    commits = [
        _commit("hwin", [{"platform": "jira", "id": "ADV-100"}], message="anything"),
    ]
    results = classifier._classify_commit_batch_with_llm(commits, ticket_context={})

    assert len(results) == 1
    assert results[0]["method"] == "jira_issuetype"
    assert results[0]["category"] == "bugfix"
    assert results[0]["matched_issuetype"] == "Bug"
    # Tier 1.5 confidence is 0.90 (below tier 3's 0.95) but still high.
    assert results[0]["confidence"] == pytest.approx(0.90)


def test_tier1_5_falls_through_to_project_key_when_issuetype_unmapped(
    tmp_path: Path,
) -> None:
    """When issuetype is unknown but project_key matches, tier 3 catches it."""
    classifier = BatchCommitClassifier(
        cache_dir=tmp_path,
        llm_config=_mock_llm_config(),
        batch_size=10,
        confidence_threshold=0.7,
        fallback_enabled=True,
        jira_project_mappings={"ADV": "feature"},
    )
    classifier.llm_enabled = False
    _seed_detailed_ticket(
        classifier,
        ticket_id="ADV-101",
        ticket_type="Weird Custom Type",
    )

    commits = [
        _commit("h101", [{"platform": "jira", "id": "ADV-101"}], message="anything"),
    ]
    results = classifier._classify_commit_batch_with_llm(commits, ticket_context={})

    assert len(results) == 1
    assert results[0]["method"] == "jira_project_key"
    assert results[0]["category"] == "feature"
