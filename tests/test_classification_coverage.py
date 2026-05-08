"""Tests for classification coverage measurement (issue #65).

Coverage validates that the pipeline:
1. Computes the per-repo classification coverage percentage correctly.
2. Emits a logging.warning() when coverage is below the threshold.
3. Does NOT emit a warning when coverage is above the threshold.
4. The CLI --validate-coverage flag exits non-zero when any repo is below.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gitflow_analytics.classification.coverage import (
    compute_repo_coverage,
    format_low_coverage_warning,
)
from gitflow_analytics.constants import CLASSIFICATION_FALLTHROUGH_CATEGORIES
from gitflow_analytics.models.database import (
    Base,
    CachedCommit,
    QualitativeCommitData,
)

# ---------------------------------------------------------------------------
# In-memory database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_session():
    """Yield a SQLAlchemy session backed by an in-memory SQLite DB."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _make_commit(
    session: Any,
    repo_path: str,
    sha: str,
    timestamp: datetime,
    change_type: str | None,
) -> None:
    """Insert a CachedCommit and (optionally) its QualitativeCommitData row."""
    commit = CachedCommit(
        repo_path=repo_path,
        commit_hash=sha,
        author_name="Test Author",
        author_email="test@example.com",
        message=f"commit {sha}",
        timestamp=timestamp,
        branch="main",
        is_merge=False,
        files_changed=1,
        insertions=1,
        deletions=0,
    )
    session.add(commit)
    session.flush()  # populate commit.id

    if change_type is not None:
        qual = QualitativeCommitData(
            commit_id=commit.id,
            change_type=change_type,
            change_type_confidence=0.9,
            business_domain="unknown",
            domain_confidence=0.0,
            risk_level="low",
            risk_factors=[],
            intent_signals={},
            collaboration_patterns={},
            technical_context={},
            processing_method="rule_based",
            processing_time_ms=0.0,
            confidence_score=0.9,
        )
        session.add(qual)
    session.commit()


# ---------------------------------------------------------------------------
# compute_repo_coverage tests
# ---------------------------------------------------------------------------


class TestComputeRepoCoverage:
    """Verify the coverage percentage formula."""

    def test_no_commits_returns_none(self, in_memory_session: Any) -> None:
        """Empty repository yields None (undefined coverage)."""
        result = compute_repo_coverage(
            session=in_memory_session,
            repo_path="/repo/empty",
            start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 12, 31, tzinfo=timezone.utc),
        )
        assert result is None

    def test_all_meaningful_returns_100(self, in_memory_session: Any) -> None:
        """Repo with only feature/fix commits has 100% coverage."""
        repo = "/repo/good"
        ts = datetime(2025, 5, 1, tzinfo=timezone.utc)
        _make_commit(in_memory_session, repo, "a" * 40, ts, "feature")
        _make_commit(in_memory_session, repo, "b" * 40, ts, "bug_fix")

        result = compute_repo_coverage(
            session=in_memory_session,
            repo_path=repo,
            start_date=ts - timedelta(days=1),
            end_date=ts + timedelta(days=1),
        )
        assert result == 100.0

    def test_all_maintenance_returns_0(self, in_memory_session: Any) -> None:
        """Repo where every commit fell to maintenance has 0% coverage."""
        repo = "/repo/bad"
        ts = datetime(2025, 5, 1, tzinfo=timezone.utc)
        for i in range(5):
            _make_commit(in_memory_session, repo, f"{i:040d}", ts, "maintenance")

        result = compute_repo_coverage(
            session=in_memory_session,
            repo_path=repo,
            start_date=ts - timedelta(days=1),
            end_date=ts + timedelta(days=1),
        )
        assert result == 0.0

    def test_mixed_returns_correct_percentage(self, in_memory_session: Any) -> None:
        """A 1/4 meaningful split rounds to 25.0%."""
        repo = "/repo/mixed"
        ts = datetime(2025, 5, 1, tzinfo=timezone.utc)
        _make_commit(in_memory_session, repo, "1" * 40, ts, "feature")
        _make_commit(in_memory_session, repo, "2" * 40, ts, "maintenance")
        _make_commit(in_memory_session, repo, "3" * 40, ts, "other")
        _make_commit(in_memory_session, repo, "4" * 40, ts, "unknown")

        result = compute_repo_coverage(
            session=in_memory_session,
            repo_path=repo,
            start_date=ts - timedelta(days=1),
            end_date=ts + timedelta(days=1),
        )
        assert result == 25.0

    def test_unclassified_commits_excluded(self, in_memory_session: Any) -> None:
        """Commits without a QualitativeCommitData row are excluded."""
        repo = "/repo/partial"
        ts = datetime(2025, 5, 1, tzinfo=timezone.utc)
        _make_commit(in_memory_session, repo, "1" * 40, ts, "feature")
        # Unclassified commit — should not appear in denominator.
        _make_commit(in_memory_session, repo, "2" * 40, ts, None)

        result = compute_repo_coverage(
            session=in_memory_session,
            repo_path=repo,
            start_date=ts - timedelta(days=1),
            end_date=ts + timedelta(days=1),
        )
        # Only the classified commit counts → 1/1 = 100%.
        assert result == 100.0

    def test_other_repos_excluded(self, in_memory_session: Any) -> None:
        """Coverage computation is scoped to a single repo_path."""
        ts = datetime(2025, 5, 1, tzinfo=timezone.utc)
        _make_commit(in_memory_session, "/repo/a", "a" * 40, ts, "feature")
        _make_commit(in_memory_session, "/repo/b", "b" * 40, ts, "maintenance")

        result_a = compute_repo_coverage(
            session=in_memory_session,
            repo_path="/repo/a",
            start_date=ts - timedelta(days=1),
            end_date=ts + timedelta(days=1),
        )
        assert result_a == 100.0

    def test_outside_date_range_excluded(self, in_memory_session: Any) -> None:
        """Commits outside the [start, end] window are excluded."""
        repo = "/repo/dates"
        in_range = datetime(2025, 5, 15, tzinfo=timezone.utc)
        out_of_range = datetime(2025, 1, 1, tzinfo=timezone.utc)
        _make_commit(in_memory_session, repo, "1" * 40, in_range, "feature")
        _make_commit(in_memory_session, repo, "2" * 40, out_of_range, "maintenance")

        result = compute_repo_coverage(
            session=in_memory_session,
            repo_path=repo,
            start_date=datetime(2025, 5, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 5, 31, tzinfo=timezone.utc),
        )
        # Only the in-range feature commit counts.
        assert result == 100.0


# ---------------------------------------------------------------------------
# Constant sanity checks
# ---------------------------------------------------------------------------


class TestFallthroughCategories:
    """Lock down the membership of CLASSIFICATION_FALLTHROUGH_CATEGORIES."""

    def test_includes_maintenance_and_ktlo(self) -> None:
        assert "maintenance" in CLASSIFICATION_FALLTHROUGH_CATEGORIES
        assert "ktlo" in CLASSIFICATION_FALLTHROUGH_CATEGORIES

    def test_includes_unknown_and_other(self) -> None:
        assert "unknown" in CLASSIFICATION_FALLTHROUGH_CATEGORIES
        assert "other" in CLASSIFICATION_FALLTHROUGH_CATEGORIES

    def test_excludes_meaningful_categories(self) -> None:
        for meaningful in ("feature", "bug_fix", "refactor", "documentation", "test"):
            assert meaningful not in CLASSIFICATION_FALLTHROUGH_CATEGORIES


# ---------------------------------------------------------------------------
# Warning message formatter
# ---------------------------------------------------------------------------


class TestFormatLowCoverageWarning:
    """Verify the human-readable warning string."""

    def test_includes_repo_name_and_pct(self) -> None:
        msg = format_low_coverage_warning("my-repo", 5.2)
        assert "my-repo" in msg
        assert "5.2" in msg

    def test_includes_remediation_hint(self) -> None:
        msg = format_low_coverage_warning("repo", 0.0)
        assert "jira_project_mappings" in msg
        assert "conventional commit" in msg


# ---------------------------------------------------------------------------
# run_classify integration: warning fires/skips, --validate-coverage exit code
# ---------------------------------------------------------------------------


_MINIMAL_CONFIG = """
cache:
  directory: "{cache_dir}"
  retention_days: 30
  ttl_hours: 168

repositories:
  - name: "test-repo"
    path: "{repo_path}"
    project_key: "TEST"

analysis:
  exclude_authors: []
  exclude_paths: []
  branch_mapping_rules: {{}}
  auto_identity_analysis: false
  manual_identity_mappings: []
  exclude_merge_commits: false
  similarity_threshold: 0.85
  story_point_patterns: []
  llm_classification:
    enabled: false
    api_key: "test"  # pragma: allowlist secret
    model: "gpt-3.5-turbo"
    confidence_threshold: 0.7
    max_tokens: 4000
    temperature: 0.1
    timeout_seconds: 30
    cache_duration_days: 7
    enable_caching: true
    max_daily_requests: 1000
    domain_terms: []
  branch_analysis:
    strategy: "smart"
    max_branches_per_repo: 10
    active_days_threshold: 30
    include_main_branches: true
    always_include_patterns: []
    always_exclude_patterns: []
    enable_progress_logging: false
    branch_commit_limit: 500
  ticket_platforms:
    - jira
    - github
    - clickup
    - linear

github:
  token: null
  organization: null

output:
  directory: "{cache_dir}/reports"
  formats:
    - csv
"""


def _load_config(tmp_path: Path) -> Any:
    from gitflow_analytics.config import ConfigLoader

    cfg_file = tmp_path / "config.yaml"
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir()
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()

    cfg_file.write_text(_MINIMAL_CONFIG.format(cache_dir=str(cache_dir), repo_path=str(repo_path)))
    return ConfigLoader.load(cfg_file)


def _seed_classified_commits(
    cache_dir: Path,
    repo_path: str,
    feature_count: int,
    maintenance_count: int,
) -> tuple[datetime, datetime]:
    """Seed the cache with classified commits and return the period bounds."""
    from gitflow_analytics.core.cache import GitAnalysisCache
    from gitflow_analytics.models.database import DailyCommitBatch

    cache = GitAnalysisCache(cache_dir)

    # Pick a commit timestamp inside the standard 4-week classify window.
    # run_classify computes its window as
    #     [last_complete_week_start - (weeks-1), last_complete_week_end]
    # where last_complete_week_start = current_week_start - 1 week.
    # Picking "10 days ago" lands safely inside that window for weeks>=2.
    ts = datetime.now(timezone.utc) - timedelta(days=10)

    with cache.get_session() as session:
        for i in range(feature_count):
            _make_commit(session, repo_path, f"f{i:039d}", ts, "feature")
        for i in range(maintenance_count):
            _make_commit(session, repo_path, f"m{i:039d}", ts, "maintenance")

        # Seed a daily batch so run_classify does not bail out early.
        batch = DailyCommitBatch(
            date=ts.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None),
            project_key="TEST",
            repo_path=repo_path,
            commit_count=feature_count + maintenance_count,
            classification_status="completed",
        )
        session.add(batch)
        session.commit()

    return ts - timedelta(days=1), ts + timedelta(days=1)


class TestRunClassifyCoverageWarning:
    """run_classify emits low-coverage warnings via logging."""

    def test_low_coverage_emits_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When coverage < threshold, logging.warning() fires."""
        from gitflow_analytics.pipeline import run_classify

        cfg = _load_config(tmp_path)
        repo_path = str(cfg.repositories[0].path)
        _seed_classified_commits(
            cache_dir=cfg.cache.directory,
            repo_path=repo_path,
            feature_count=1,
            maintenance_count=9,  # 10% coverage — below 20% threshold
        )

        # Stub the actual classifier so we don't try to call an LLM.
        with patch(
            "gitflow_analytics.classification.batch_classifier.BatchCommitClassifier"
        ) as MockClassifier:
            mock_clf = MagicMock()
            mock_clf.classify_date_range.return_value = {
                "processed_batches": 1,
                "total_commits": 10,
                "skipped_batches": 0,
            }
            MockClassifier.return_value = mock_clf

            # Explicitly target the pipeline_classify logger so other tests
            # cannot suppress propagation by setting a higher level on the
            # root or a parent logger earlier in the run.
            with caplog.at_level(logging.WARNING, logger="gitflow_analytics.pipeline_classify"):
                result = run_classify(cfg=cfg, weeks=4, coverage_threshold=20.0)

        assert repo_path in result.coverage_by_repo
        assert result.coverage_by_repo[repo_path] == 10.0

        warning_records = [
            r for r in caplog.records if r.levelno >= logging.WARNING and "coverage" in r.message
        ]
        assert warning_records, (
            f"Expected a low-coverage warning to be logged. "
            f"Captured records: {[(r.name, r.levelname, r.message) for r in caplog.records]}"
        )

    def test_high_coverage_no_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When coverage >= threshold, no low-coverage warning fires."""
        from gitflow_analytics.pipeline import run_classify

        cfg = _load_config(tmp_path)
        repo_path = str(cfg.repositories[0].path)
        _seed_classified_commits(
            cache_dir=cfg.cache.directory,
            repo_path=repo_path,
            feature_count=9,
            maintenance_count=1,  # 90% coverage — well above 20%
        )

        with patch(
            "gitflow_analytics.classification.batch_classifier.BatchCommitClassifier"
        ) as MockClassifier:
            mock_clf = MagicMock()
            mock_clf.classify_date_range.return_value = {
                "processed_batches": 1,
                "total_commits": 10,
                "skipped_batches": 0,
            }
            MockClassifier.return_value = mock_clf

            with caplog.at_level(logging.WARNING, logger="gitflow_analytics.pipeline_classify"):
                result = run_classify(cfg=cfg, weeks=4, coverage_threshold=20.0)

        assert result.coverage_by_repo[repo_path] == 90.0

        low_coverage_warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "classification coverage" in r.message
        ]
        assert not low_coverage_warnings


class TestValidateCoverageCLIFlag:
    """Verify --validate-coverage exits non-zero when any repo is below threshold."""

    def test_validate_coverage_exits_one_when_low(self, tmp_path: Path) -> None:
        """The CLI exits 1 when --validate-coverage is set and coverage is low."""
        from click.testing import CliRunner

        from gitflow_analytics.cli import cli

        cfg = _load_config(tmp_path)
        repo_path = str(cfg.repositories[0].path)
        _seed_classified_commits(
            cache_dir=cfg.cache.directory,
            repo_path=repo_path,
            feature_count=1,
            maintenance_count=9,
        )

        with patch(
            "gitflow_analytics.classification.batch_classifier.BatchCommitClassifier"
        ) as MockClassifier:
            mock_clf = MagicMock()
            mock_clf.classify_date_range.return_value = {
                "processed_batches": 1,
                "total_commits": 10,
                "skipped_batches": 0,
            }
            MockClassifier.return_value = mock_clf

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "classify",
                    "-c",
                    str(tmp_path / "config.yaml"),
                    "--validate-coverage",
                    "--coverage-threshold",
                    "20",
                ],
            )

        assert result.exit_code == 1
        # Either stdout or stderr should mention validation failure.
        combined = (result.output or "") + (
            result.stderr_bytes.decode() if result.stderr_bytes else ""
        )
        assert "coverage" in combined.lower()

    def test_validate_coverage_exits_zero_when_high(self, tmp_path: Path) -> None:
        """When all repos are above threshold, --validate-coverage exits 0."""
        from click.testing import CliRunner

        from gitflow_analytics.cli import cli

        cfg = _load_config(tmp_path)
        repo_path = str(cfg.repositories[0].path)
        _seed_classified_commits(
            cache_dir=cfg.cache.directory,
            repo_path=repo_path,
            feature_count=9,
            maintenance_count=1,
        )

        with patch(
            "gitflow_analytics.classification.batch_classifier.BatchCommitClassifier"
        ) as MockClassifier:
            mock_clf = MagicMock()
            mock_clf.classify_date_range.return_value = {
                "processed_batches": 1,
                "total_commits": 10,
                "skipped_batches": 0,
            }
            MockClassifier.return_value = mock_clf

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "classify",
                    "-c",
                    str(tmp_path / "config.yaml"),
                    "--validate-coverage",
                    "--coverage-threshold",
                    "20",
                ],
            )

        assert result.exit_code == 0
