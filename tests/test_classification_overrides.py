"""Tests for manual classification overrides (issue #63).

Verifies that:

- ``ClassificationOverride`` rows take priority over every classifier
  (LLM, JIRA project key, fallback) inside the batch classifier.
- The override survives a reclassification rerun (the qualitative row is
  rewritten with method=``manual_override`` and confidence 1.0).
- The CLI subcommands (``set``/``list``/``remove``) round-trip correctly
  through the database.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from gitflow_analytics.classification.batch_classifier import BatchCommitClassifier
from gitflow_analytics.cli_overrides import (
    override_list,
    override_remove,
    override_set,
)
from gitflow_analytics.models.database import (
    ClassificationOverride,
    Database,
)

# ---------------------------------------------------------------------------
# Helpers (mirroring tests/test_jira_project_key_classifier.py)
# ---------------------------------------------------------------------------


def _mock_llm_config() -> dict[str, Any]:
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
    cache_dir: Path,
    mappings: dict[str, str] | None = None,
) -> BatchCommitClassifier:
    return BatchCommitClassifier(
        cache_dir=cache_dir,
        llm_config=_mock_llm_config(),
        batch_size=10,
        confidence_threshold=0.7,
        fallback_enabled=True,
        jira_project_mappings=mappings,
    )


def _commit(commit_hash: str, repo_path: str = "/repo/a", **extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "commit_hash": commit_hash,
        "repo_path": repo_path,
        "message": "anything",
        "ticket_references": [],
    }
    base.update(extra)
    return base


def _add_override(
    db: Database,
    commit_hash: str,
    repo_path: str,
    work_type: str,
    *,
    confidence: float = 1.0,
    reason: str | None = None,
) -> None:
    assert db.SessionLocal is not None
    with db.SessionLocal() as session:
        session.add(
            ClassificationOverride(
                commit_hash=commit_hash,
                repo_path=repo_path,
                work_type=work_type,
                confidence=confidence,
                reason=reason,
            )
        )
        session.commit()


# ---------------------------------------------------------------------------
# Pipeline integration: overrides take priority over every classifier
# ---------------------------------------------------------------------------


def test_override_takes_priority_over_jira_project_key(tmp_path: Path) -> None:
    """A manual override beats the JIRA project-key short-circuit."""
    classifier = _make_classifier(tmp_path, mappings={"ADV": "feature"})
    classifier.llm_enabled = False

    _add_override(
        classifier.database,
        commit_hash="h1",
        repo_path="/repo/a",
        work_type="bug_fix",
        reason="manual correction",
    )

    commits = [
        _commit(
            "h1",
            ticket_references=[{"platform": "jira", "id": "ADV-1"}],
        ),
    ]
    results = classifier._classify_commit_batch_with_llm(commits, ticket_context={})

    assert len(results) == 1
    assert results[0]["category"] == "bug_fix"
    assert results[0]["method"] == "manual_override"
    assert results[0]["confidence"] == pytest.approx(1.0)


def test_override_takes_priority_over_fallback(tmp_path: Path) -> None:
    """Override wins even when no LLM and no JIRA mapping is available."""
    classifier = _make_classifier(tmp_path)
    classifier.llm_enabled = False

    _add_override(
        classifier.database,
        commit_hash="h1",
        repo_path="/repo/a",
        work_type="documentation",
    )

    commits = [_commit("h1", message="fix: null pointer crash")]
    results = classifier._classify_commit_batch_with_llm(commits, ticket_context={})

    assert results[0]["method"] == "manual_override"
    assert results[0]["category"] == "documentation"


def test_override_repo_scoping(tmp_path: Path) -> None:
    """An override is scoped to (commit_hash, repo_path) — wrong repo doesn't match."""
    classifier = _make_classifier(tmp_path)
    classifier.llm_enabled = False

    _add_override(
        classifier.database,
        commit_hash="h1",
        repo_path="/repo/a",
        work_type="documentation",
    )

    # Same hash, different repo — must NOT match.
    commits = [_commit("h1", repo_path="/repo/b", message="fix: crash")]
    results = classifier._classify_commit_batch_with_llm(commits, ticket_context={})

    assert results[0]["method"] != "manual_override"


def test_partial_batch_with_override(tmp_path: Path) -> None:
    """Override applies to one commit, the rest go through normal pipeline."""
    classifier = _make_classifier(tmp_path, mappings={"ADV": "feature"})
    classifier.llm_enabled = False

    _add_override(
        classifier.database,
        commit_hash="h1",
        repo_path="/repo/a",
        work_type="documentation",
    )

    commits = [
        _commit("h1", message="feat: new thing"),
        _commit("h2", ticket_references=[{"platform": "jira", "id": "ADV-9"}]),
        _commit("h3", message="fix: null pointer"),
    ]
    results = classifier._classify_commit_batch_with_llm(commits, ticket_context={})

    by_hash = {r["commit_hash"]: r for r in results}
    assert by_hash["h1"]["method"] == "manual_override"
    assert by_hash["h1"]["category"] == "documentation"
    assert by_hash["h2"]["method"] == "jira_project_key"
    assert by_hash["h2"]["category"] == "feature"
    assert by_hash["h3"]["method"] == "fallback_only"
    assert by_hash["h3"]["category"] == "bug_fix"


def test_override_survives_reclassification_rerun(tmp_path: Path) -> None:
    """Re-running the classifier yields the same override result."""
    classifier = _make_classifier(tmp_path)
    classifier.llm_enabled = False

    _add_override(
        classifier.database,
        commit_hash="h1",
        repo_path="/repo/a",
        work_type="documentation",
    )

    commits = [_commit("h1", message="feat: anything")]

    first = classifier._classify_commit_batch_with_llm(commits, ticket_context={})
    second = classifier._classify_commit_batch_with_llm(commits, ticket_context={})

    assert first[0]["category"] == second[0]["category"] == "documentation"
    assert first[0]["method"] == second[0]["method"] == "manual_override"


def test_override_lookup_helper_is_repo_path_aware(tmp_path: Path) -> None:
    """``_lookup_classification_overrides`` skips rows from the wrong repo_path."""
    classifier = _make_classifier(tmp_path)

    _add_override(
        classifier.database,
        commit_hash="h1",
        repo_path="/repo/a",
        work_type="feature",
    )

    matched = classifier._lookup_classification_overrides([_commit("h1", repo_path="/repo/a")])
    not_matched = classifier._lookup_classification_overrides([_commit("h1", repo_path="/repo/b")])

    assert "h1" in matched
    assert matched["h1"]["category"] == "feature"
    assert matched["h1"]["method"] == "manual_override"
    assert not_matched == {}


# ---------------------------------------------------------------------------
# CLI: set / list / remove round-trip via Click's CliRunner
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a fake config + cache dir and patch ConfigLoader.load.

    The CLI commands call ``ConfigLoader.load(config_path)`` and then build
    ``Database(cfg.cache.directory / 'gitflow_cache.db')``. We bypass the YAML
    loader entirely by patching ``ConfigLoader.load`` to return a stub object
    exposing ``cache.directory``.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    config_file = tmp_path / "config.yaml"
    config_file.write_text("# stub config for tests\n")

    class _StubCache:
        directory = cache_dir

    class _StubConfig:
        cache = _StubCache()

    from gitflow_analytics import cli_overrides as mod

    monkeypatch.setattr(mod.ConfigLoader, "load", staticmethod(lambda _path: _StubConfig()))

    # Pre-create the database so list/remove on empty DB doesn't fail.
    Database(cache_dir / "gitflow_cache.db")
    return config_file


def test_cli_override_set_creates_row(cli_setup: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        override_set,
        [
            "fakehash01",
            "feature",
            "-c",
            str(cli_setup),
            "--repo",
            "/path/to/repo",
            "--reason",
            "manual fix",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "created" in result.output

    # Re-running with the same key updates rather than duplicating.
    result2 = runner.invoke(
        override_set,
        [
            "fakehash01",
            "bug_fix",
            "-c",
            str(cli_setup),
            "--repo",
            "/path/to/repo",
        ],
    )
    assert result2.exit_code == 0, result2.output
    assert "updated" in result2.output


def test_cli_override_list_filters_by_repo(cli_setup: Path) -> None:
    runner = CliRunner()
    runner.invoke(
        override_set,
        ["h1", "feature", "-c", str(cli_setup), "--repo", "/repo/a"],
    )
    runner.invoke(
        override_set,
        ["h2", "bug_fix", "-c", str(cli_setup), "--repo", "/repo/b"],
    )

    all_result = runner.invoke(override_list, ["-c", str(cli_setup)])
    assert all_result.exit_code == 0
    assert "h1" in all_result.output
    assert "h2" in all_result.output

    filtered = runner.invoke(override_list, ["-c", str(cli_setup), "--repo", "/repo/a"])
    assert filtered.exit_code == 0
    assert "h1" in filtered.output
    assert "h2" not in filtered.output


def test_cli_override_remove_deletes_row(cli_setup: Path) -> None:
    runner = CliRunner()
    runner.invoke(
        override_set,
        ["fakehash", "feature", "-c", str(cli_setup), "--repo", "/repo/a"],
    )

    remove = runner.invoke(
        override_remove,
        ["fakehash", "-c", str(cli_setup), "--repo", "/repo/a"],
    )
    assert remove.exit_code == 0, remove.output
    assert "removed" in remove.output

    listing = runner.invoke(override_list, ["-c", str(cli_setup)])
    assert "No overrides found" in listing.output


def test_cli_override_remove_missing_returns_error(cli_setup: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        override_remove,
        ["nope", "-c", str(cli_setup), "--repo", "/repo/a"],
    )
    assert result.exit_code != 0
    assert "No override found" in result.output
