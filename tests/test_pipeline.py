"""Tests for the pipeline stage helpers: run_collect, run_classify, run_report.

Each test class covers one pipeline stage in isolation using mocked
infrastructure (cache, data fetcher, batch classifier) so the tests
run quickly without real git repositories or LLM API keys.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from gitflow_analytics.pipeline import (
    ClassifyResult,
    CollectResult,
    ReportResult,
    run_classify,
    run_collect,
    run_report,
)

# ---------------------------------------------------------------------------
# Config fixture helpers
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG_TEMPLATE = """
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
    api_key: "test-key"  # pragma: allowlist secret
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

jira:
  enabled: false

output:
  directory: "{output_dir}"
  formats:
    - markdown
  anonymize_developers: false

pm_integration:
  enabled: false
  platforms: {{}}
"""


def _make_config(tmp_path: Path) -> Any:
    """Create and load a minimal config object backed by tmp directories."""
    from gitflow_analytics.config import ConfigLoader

    cache_dir = tmp_path / ".gitflow-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_text = _MINIMAL_CONFIG_TEMPLATE.format(
        cache_dir=str(cache_dir),
        repo_path=str(repo_path),
        output_dir=str(output_dir),
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_text)
    return ConfigLoader.load(config_file)


def _make_fetch_result(commits: int = 5, tickets: int = 2) -> dict[str, Any]:
    """Return a minimal fetch_repository_data result dict."""
    return {
        "project_key": "TEST",
        "repo_path": "/some/repo",
        "date_range": {"start": None, "end": None},
        "stats": {
            "total_commits": commits,
            "stored_commits": commits,
            "storage_success": True,
            "days_with_commits": 1,
            "unique_tickets": tickets,
            "correlations_created": 0,
            "batches_created": 1,
            "weeks_cached": 4,
            "weeks_fetched": 4,
            "cache_hit": False,
        },
        "exclusions": {"patterns_applied": 0, "enabled": False},
        "daily_commits": {},
    }


# ---------------------------------------------------------------------------
# Stage 1 — run_collect
# ---------------------------------------------------------------------------


class TestRunCollect:
    """Unit tests for run_collect pipeline helper.

    Since run_collect uses lazy imports inside the function body, we patch
    the source modules (not pipeline-level attributes).
    """

    _COLLECT_PATCHES = (
        "gitflow_analytics.core.cache.GitAnalysisCache",
        "gitflow_analytics.core.data_fetcher.GitDataFetcher",
        "gitflow_analytics.integrations.orchestrator.IntegrationOrchestrator",
    )

    def _mock_infrastructure(
        self,
        MockCache: MagicMock,
        MockFetcher: MagicMock,
        MockOrch: MagicMock,
        fetch_result: dict[str, Any] | None = None,
        cached: bool = False,
    ) -> tuple[MagicMock, MagicMock]:
        """Configure mocks and return (mock_cache, mock_fetcher)."""
        mock_cache = MagicMock()
        mock_cache.generate_config_hash.return_value = "abc123"
        mock_cache.get_repository_analysis_status.return_value = (
            {"commit_count": 5} if cached else None
        )
        MockCache.return_value = mock_cache

        mock_fetcher = MagicMock()
        if fetch_result is not None:
            mock_fetcher.fetch_repository_data.return_value = fetch_result
        MockFetcher.return_value = mock_fetcher

        mock_orch = MagicMock()
        mock_orch.integrations = {}
        MockOrch.return_value = mock_orch

        return mock_cache, mock_fetcher

    def test_returns_collect_result_dataclass(self, tmp_path: Path) -> None:
        """run_collect always returns a CollectResult instance."""
        cfg = _make_config(tmp_path)

        with (
            patch("gitflow_analytics.core.cache.GitAnalysisCache") as MockCache,
            patch("gitflow_analytics.core.data_fetcher.GitDataFetcher") as MockFetcher,
            patch(
                "gitflow_analytics.integrations.orchestrator.IntegrationOrchestrator"
            ) as MockOrch,
        ):
            self._mock_infrastructure(MockCache, MockFetcher, MockOrch, _make_fetch_result(10, 3))
            result = run_collect(cfg=cfg, weeks=4)

        assert isinstance(result, CollectResult)

    def test_counts_fetched_repos(self, tmp_path: Path) -> None:
        """repos_fetched increments for each successfully fetched repository."""
        cfg = _make_config(tmp_path)

        with (
            patch("gitflow_analytics.core.cache.GitAnalysisCache") as MockCache,
            patch("gitflow_analytics.core.data_fetcher.GitDataFetcher") as MockFetcher,
            patch(
                "gitflow_analytics.integrations.orchestrator.IntegrationOrchestrator"
            ) as MockOrch,
        ):
            self._mock_infrastructure(MockCache, MockFetcher, MockOrch, _make_fetch_result(7, 1))
            result = run_collect(cfg=cfg, weeks=4)

        assert result.repos_fetched == 1
        assert result.total_commits == 7
        assert result.total_tickets == 1

    def test_cached_repo_skipped(self, tmp_path: Path) -> None:
        """Repos with a valid cache status are skipped when force=False."""
        cfg = _make_config(tmp_path)

        with (
            patch("gitflow_analytics.core.cache.GitAnalysisCache") as MockCache,
            patch("gitflow_analytics.core.data_fetcher.GitDataFetcher") as MockFetcher,
            patch(
                "gitflow_analytics.integrations.orchestrator.IntegrationOrchestrator"
            ) as MockOrch,
        ):
            _, mock_fetcher = self._mock_infrastructure(
                MockCache, MockFetcher, MockOrch, cached=True
            )
            result = run_collect(cfg=cfg, weeks=4, force=False)

        assert result.repos_cached == 1
        assert result.repos_fetched == 0
        mock_fetcher.fetch_repository_data.assert_not_called()

    def test_force_bypasses_cache(self, tmp_path: Path) -> None:
        """force=True fetches even repos that have a cached status."""
        cfg = _make_config(tmp_path)

        with (
            patch("gitflow_analytics.core.cache.GitAnalysisCache") as MockCache,
            patch("gitflow_analytics.core.data_fetcher.GitDataFetcher") as MockFetcher,
            patch(
                "gitflow_analytics.integrations.orchestrator.IntegrationOrchestrator"
            ) as MockOrch,
        ):
            _, mock_fetcher = self._mock_infrastructure(
                MockCache, MockFetcher, MockOrch, _make_fetch_result(10), cached=True
            )
            result = run_collect(cfg=cfg, weeks=4, force=True)

        mock_fetcher.fetch_repository_data.assert_called_once()
        assert result.repos_fetched == 1

    def test_missing_repo_recorded_as_failure(self, tmp_path: Path) -> None:
        """A repo whose path does not exist is counted as repos_failed."""
        import shutil

        cfg = _make_config(tmp_path)
        repo_path = Path(cfg.repositories[0].path)
        if repo_path.exists():
            shutil.rmtree(repo_path)

        with (
            patch("gitflow_analytics.core.cache.GitAnalysisCache") as MockCache,
            patch("gitflow_analytics.core.data_fetcher.GitDataFetcher") as MockFetcher,
            patch(
                "gitflow_analytics.integrations.orchestrator.IntegrationOrchestrator"
            ) as MockOrch,
        ):
            self._mock_infrastructure(MockCache, MockFetcher, MockOrch)
            result = run_collect(cfg=cfg, weeks=4)

        assert result.repos_failed == 1
        assert len(result.errors) == 1
        assert "not found" in result.errors[0].lower()

    def test_progress_callback_called(self, tmp_path: Path) -> None:
        """Progress callback receives status messages during collect."""
        cfg = _make_config(tmp_path)
        messages: list[str] = []

        with (
            patch("gitflow_analytics.core.cache.GitAnalysisCache") as MockCache,
            patch("gitflow_analytics.core.data_fetcher.GitDataFetcher") as MockFetcher,
            patch(
                "gitflow_analytics.integrations.orchestrator.IntegrationOrchestrator"
            ) as MockOrch,
        ):
            self._mock_infrastructure(MockCache, MockFetcher, MockOrch, _make_fetch_result(3, 0))
            run_collect(cfg=cfg, weeks=4, progress_callback=messages.append)

        assert any(messages), "Expected at least one progress message"

    def test_date_range_is_monday_aligned(self, tmp_path: Path) -> None:
        """The computed date range always starts on a Monday."""
        cfg = _make_config(tmp_path)
        captured_start: list[datetime] = []

        def capture_start_date(**kwargs: Any) -> dict[str, Any]:
            captured_start.append(kwargs["start_date"])
            return _make_fetch_result(0)

        with (
            patch("gitflow_analytics.core.cache.GitAnalysisCache") as MockCache,
            patch("gitflow_analytics.core.data_fetcher.GitDataFetcher") as MockFetcher,
            patch(
                "gitflow_analytics.integrations.orchestrator.IntegrationOrchestrator"
            ) as MockOrch,
        ):
            mock_cache = MagicMock()
            mock_cache.generate_config_hash.return_value = "abc123"
            mock_cache.get_repository_analysis_status.return_value = None
            MockCache.return_value = mock_cache

            mock_fetcher = MagicMock()
            mock_fetcher.fetch_repository_data.side_effect = capture_start_date
            MockFetcher.return_value = mock_fetcher

            mock_orch = MagicMock()
            mock_orch.integrations = {}
            MockOrch.return_value = mock_orch

            run_collect(cfg=cfg, weeks=4)

        if captured_start:
            assert captured_start[0].weekday() == 0, "start_date is not a Monday"

    def test_result_has_date_range(self, tmp_path: Path) -> None:
        """CollectResult.start_date and end_date are populated."""
        cfg = _make_config(tmp_path)

        with (
            patch("gitflow_analytics.core.cache.GitAnalysisCache") as MockCache,
            patch("gitflow_analytics.core.data_fetcher.GitDataFetcher") as MockFetcher,
            patch(
                "gitflow_analytics.integrations.orchestrator.IntegrationOrchestrator"
            ) as MockOrch,
        ):
            self._mock_infrastructure(MockCache, MockFetcher, MockOrch, _make_fetch_result(2))
            result = run_collect(cfg=cfg, weeks=2)

        assert result.start_date is not None
        assert result.end_date is not None
        assert result.start_date < result.end_date


# ---------------------------------------------------------------------------
# Stage 2 — run_classify
# ---------------------------------------------------------------------------


class TestRunClassify:
    """Unit tests for run_classify pipeline helper.

    Since run_classify uses lazy imports, we patch at the source module level.
    """

    def _make_cache_mock_with_counts(self, commits: int, batches: int) -> MagicMock:
        """Return a GitAnalysisCache mock that yields given commit/batch counts."""
        mock_cache = MagicMock()
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.side_effect = [commits, batches]
        mock_session.query.return_value = mock_query
        mock_cache.get_session.return_value = mock_session
        return mock_cache

    def test_returns_classify_result_dataclass(self, tmp_path: Path) -> None:
        """run_classify always returns a ClassifyResult instance."""
        cfg = _make_config(tmp_path)

        with (
            patch("gitflow_analytics.core.cache.GitAnalysisCache") as MockCache,
            patch(
                "gitflow_analytics.classification.batch_classifier.BatchCommitClassifier"
            ) as MockClassifier,
        ):
            MockCache.return_value = self._make_cache_mock_with_counts(10, 5)
            mock_clf = MagicMock()
            mock_clf.classify_date_range.return_value = {
                "processed_batches": 5,
                "total_commits": 10,
                "skipped_batches": 0,
            }
            MockClassifier.return_value = mock_clf

            result = run_classify(cfg=cfg, weeks=4)

        assert isinstance(result, ClassifyResult)

    def test_no_commits_returns_error(self, tmp_path: Path) -> None:
        """run_classify returns an error when the cache has no commits."""
        cfg = _make_config(tmp_path)

        with (
            patch("gitflow_analytics.core.cache.GitAnalysisCache") as MockCache,
            patch(
                "gitflow_analytics.classification.batch_classifier.BatchCommitClassifier"
            ) as MockClassifier,
        ):
            MockCache.return_value = self._make_cache_mock_with_counts(0, 0)
            mock_clf = MagicMock()
            MockClassifier.return_value = mock_clf

            result = run_classify(cfg=cfg, weeks=4)

        assert len(result.errors) >= 1
        assert any("collect" in e.lower() for e in result.errors)
        mock_clf.classify_date_range.assert_not_called()

    def test_commits_but_no_batches_returns_error(self, tmp_path: Path) -> None:
        """run_classify returns an error when commits exist but batches don't."""
        cfg = _make_config(tmp_path)

        with (
            patch("gitflow_analytics.core.cache.GitAnalysisCache") as MockCache,
            patch(
                "gitflow_analytics.classification.batch_classifier.BatchCommitClassifier"
            ) as MockClassifier,
        ):
            MockCache.return_value = self._make_cache_mock_with_counts(10, 0)
            MockClassifier.return_value = MagicMock()

            result = run_classify(cfg=cfg, weeks=4)

        assert len(result.errors) >= 1
        assert any("batches" in e.lower() for e in result.errors)

    def test_successful_classification_populates_result(self, tmp_path: Path) -> None:
        """Successful classification populates processed_batches and total_commits."""
        cfg = _make_config(tmp_path)

        with (
            patch("gitflow_analytics.core.cache.GitAnalysisCache") as MockCache,
            patch(
                "gitflow_analytics.classification.batch_classifier.BatchCommitClassifier"
            ) as MockClassifier,
        ):
            MockCache.return_value = self._make_cache_mock_with_counts(20, 7)
            mock_clf = MagicMock()
            mock_clf.classify_date_range.return_value = {
                "processed_batches": 7,
                "total_commits": 20,
                "skipped_batches": 2,
            }
            MockClassifier.return_value = mock_clf

            result = run_classify(cfg=cfg, weeks=4)

        assert result.processed_batches == 7
        assert result.total_commits == 20
        assert result.skipped_batches == 2
        assert result.errors == []

    def test_reclassify_passed_to_classifier(self, tmp_path: Path) -> None:
        """reclassify=True is forwarded to BatchCommitClassifier."""
        cfg = _make_config(tmp_path)

        with (
            patch("gitflow_analytics.core.cache.GitAnalysisCache") as MockCache,
            patch(
                "gitflow_analytics.classification.batch_classifier.BatchCommitClassifier"
            ) as MockClassifier,
        ):
            MockCache.return_value = self._make_cache_mock_with_counts(5, 2)
            mock_clf = MagicMock()
            mock_clf.classify_date_range.return_value = {
                "processed_batches": 2,
                "total_commits": 5,
                "skipped_batches": 0,
            }
            MockClassifier.return_value = mock_clf

            run_classify(cfg=cfg, weeks=4, reclassify=True)

        call_kwargs = mock_clf.classify_date_range.call_args
        assert call_kwargs.kwargs.get("force_reclassify") is True

    def test_classifier_exception_recorded_in_errors(self, tmp_path: Path) -> None:
        """A RuntimeError from the classifier is recorded in errors."""
        cfg = _make_config(tmp_path)

        with (
            patch("gitflow_analytics.core.cache.GitAnalysisCache") as MockCache,
            patch(
                "gitflow_analytics.classification.batch_classifier.BatchCommitClassifier"
            ) as MockClassifier,
        ):
            MockCache.return_value = self._make_cache_mock_with_counts(5, 2)
            mock_clf = MagicMock()
            mock_clf.classify_date_range.side_effect = RuntimeError("LLM quota exceeded")
            MockClassifier.return_value = mock_clf

            result = run_classify(cfg=cfg, weeks=4)

        assert any("quota" in e.lower() for e in result.errors)

    def test_progress_callback_called(self, tmp_path: Path) -> None:
        """Progress callback receives messages during classification."""
        cfg = _make_config(tmp_path)
        messages: list[str] = []

        with (
            patch("gitflow_analytics.core.cache.GitAnalysisCache") as MockCache,
            patch(
                "gitflow_analytics.classification.batch_classifier.BatchCommitClassifier"
            ) as MockClassifier,
        ):
            MockCache.return_value = self._make_cache_mock_with_counts(5, 2)
            mock_clf = MagicMock()
            mock_clf.classify_date_range.return_value = {
                "processed_batches": 2,
                "total_commits": 5,
                "skipped_batches": 0,
            }
            MockClassifier.return_value = mock_clf

            run_classify(cfg=cfg, weeks=4, progress_callback=messages.append)

        assert any(messages), "Expected at least one progress message"


# ---------------------------------------------------------------------------
# Stage 3 — run_report
# ---------------------------------------------------------------------------


class TestRunReport:
    """Unit tests for run_report pipeline helper.

    Since run_report uses lazy imports, patch at the source module level.
    """

    # Patch paths for all the lazy imports used inside run_report
    _REPORT_PATCH_PATHS = {
        "cache": "gitflow_analytics.core.cache.GitAnalysisCache",
        "identity": "gitflow_analytics.core.identity.DeveloperIdentityResolver",
        "analyzer": "gitflow_analytics.core.analyzer.GitAnalyzer",
        "csv_writer": "gitflow_analytics.reports.csv_writer.CSVReportGenerator",
        "analytics_writer": "gitflow_analytics.reports.analytics_writer.AnalyticsReportGenerator",
        "narrative_writer": "gitflow_analytics.reports.narrative_writer.NarrativeReportGenerator",
        "trends_writer": "gitflow_analytics.reports.weekly_trends_writer.WeeklyTrendsWriter",
        "json_exporter": "gitflow_analytics.reports.json_exporter.ComprehensiveJSONExporter",
        "dora": "gitflow_analytics.metrics.dora.DORAMetricsCalculator",
    }

    def _make_cache_mock(self) -> MagicMock:
        """Return a cache mock that yields empty commit rows."""
        mock_cache = MagicMock()
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []
        mock_session.query.return_value = mock_query
        mock_cache.get_session.return_value = mock_session
        mock_cache._commit_to_dict.return_value = {}
        return mock_cache

    def _make_identity_mock(self) -> MagicMock:
        mock_identity = MagicMock()
        mock_identity.get_developer_stats.return_value = {}
        return mock_identity

    def _make_analyzer_mock(self) -> MagicMock:
        mock_analyzer = MagicMock()
        mock_analyzer.ticket_extractor.analyze_ticket_coverage.return_value = {}
        mock_analyzer.ticket_extractor.calculate_developer_ticket_coverage.return_value = {}
        return mock_analyzer

    def _run_with_mocks(
        self,
        cfg: Any,
        output_dir: Path,
        weeks: int = 4,
        analytics_side_effect: Exception | None = None,
        progress_callback: Any = None,
    ) -> ReportResult:
        """Run run_report with all heavy dependencies mocked out."""
        with (
            patch(self._REPORT_PATCH_PATHS["cache"]) as MockCache,
            patch(self._REPORT_PATCH_PATHS["identity"]) as MockIdentity,
            patch(self._REPORT_PATCH_PATHS["analyzer"]) as MockAnalyzer,
            patch(self._REPORT_PATCH_PATHS["csv_writer"]),
            patch(self._REPORT_PATCH_PATHS["analytics_writer"]) as MockAnalytics,
            patch(self._REPORT_PATCH_PATHS["narrative_writer"]),
            patch(self._REPORT_PATCH_PATHS["trends_writer"]),
            patch(self._REPORT_PATCH_PATHS["json_exporter"]),
            patch(self._REPORT_PATCH_PATHS["dora"]),
        ):
            MockCache.return_value = self._make_cache_mock()
            MockIdentity.return_value = self._make_identity_mock()
            MockAnalyzer.return_value = self._make_analyzer_mock()

            mock_analytics = MagicMock()
            if analytics_side_effect is not None:
                mock_analytics.generate_activity_distribution_report.side_effect = (
                    analytics_side_effect
                )
            MockAnalytics.return_value = mock_analytics

            kwargs: dict[str, Any] = {"cfg": cfg, "weeks": weeks, "output_dir": output_dir}
            if progress_callback is not None:
                kwargs["progress_callback"] = progress_callback

            return run_report(**kwargs)

    def test_returns_report_result_dataclass(self, tmp_path: Path) -> None:
        """run_report always returns a ReportResult instance."""
        cfg = _make_config(tmp_path)
        output_dir = tmp_path / "reports"
        output_dir.mkdir(exist_ok=True)

        result = self._run_with_mocks(cfg, output_dir)

        assert isinstance(result, ReportResult)

    def test_output_dir_is_created(self, tmp_path: Path) -> None:
        """run_report creates the output directory if it does not exist."""
        cfg = _make_config(tmp_path)
        output_dir = tmp_path / "new_reports_dir" / "nested"

        assert not output_dir.exists()
        self._run_with_mocks(cfg, output_dir)
        assert output_dir.exists()

    def test_result_output_dir_matches_argument(self, tmp_path: Path) -> None:
        """ReportResult.output_dir matches the directory passed in."""
        cfg = _make_config(tmp_path)
        output_dir = tmp_path / "my-reports"
        output_dir.mkdir(exist_ok=True)

        result = self._run_with_mocks(cfg, output_dir)

        assert result.output_dir == output_dir

    def test_progress_callback_called(self, tmp_path: Path) -> None:
        """Progress callback receives messages during report generation."""
        cfg = _make_config(tmp_path)
        output_dir = tmp_path / "reports"
        output_dir.mkdir(exist_ok=True)
        messages: list[str] = []

        self._run_with_mocks(cfg, output_dir, progress_callback=messages.append)

        assert any(messages), "Expected at least one progress message"

    def test_report_errors_do_not_raise(self, tmp_path: Path) -> None:
        """A report generator failure is recorded in errors, not raised."""
        cfg = _make_config(tmp_path)
        output_dir = tmp_path / "reports"
        output_dir.mkdir(exist_ok=True)

        result = self._run_with_mocks(
            cfg,
            output_dir,
            analytics_side_effect=ValueError("simulated report failure"),
        )

        assert any("simulated report failure" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Stage 3 PR cache loading tests
# ---------------------------------------------------------------------------


class TestRunReportPRCacheLoading:
    """Verify that run_report loads cached PR data from PullRequestCache.

    These tests exercise the PR-loading path added to fix the gap where
    ``all_prs`` was always an empty list regardless of cached data.
    """

    _REPORT_PATCH_PATHS = {
        "cache": "gitflow_analytics.core.cache.GitAnalysisCache",
        "identity": "gitflow_analytics.core.identity.DeveloperIdentityResolver",
        "analyzer": "gitflow_analytics.core.analyzer.GitAnalyzer",
        "csv_writer": "gitflow_analytics.reports.csv_writer.CSVReportGenerator",
        "analytics_writer": "gitflow_analytics.reports.analytics_writer.AnalyticsReportGenerator",
        "narrative_writer": "gitflow_analytics.reports.narrative_writer.NarrativeReportGenerator",
        "trends_writer": "gitflow_analytics.reports.weekly_trends_writer.WeeklyTrendsWriter",
        "json_exporter": "gitflow_analytics.reports.json_exporter.ComprehensiveJSONExporter",
        "dora": "gitflow_analytics.metrics.dora.DORAMetricsCalculator",
    }

    def _make_pr_dict(self, number: int = 1) -> dict[str, Any]:
        """Return a minimal PR dict matching the _pr_to_dict() output format."""
        return {
            "number": number,
            "title": f"PR #{number}",
            "description": "desc",
            "author": "alice",
            "created_at": datetime(2025, 1, 10, tzinfo=timezone.utc),
            "merged_at": datetime(2025, 1, 11, tzinfo=timezone.utc),
            "pr_state": "merged",
            "closed_at": datetime(2025, 1, 11, tzinfo=timezone.utc),
            "is_merged": True,
            "story_points": None,
            "labels": [],
            "commit_hashes": [],
            "review_comments": 0,
            "pr_comments_count": 0,
            "approvals_count": 0,
            "change_requests_count": 0,
            "reviewers": [],
            "approved_by": [],
            "time_to_first_review_hours": None,
            "revision_count": 0,
            "changed_files": 2,
            "additions": 10,
            "deletions": 3,
        }

    def _make_cache_mock(self, prs: list[dict[str, Any]]) -> MagicMock:
        """Return a GitAnalysisCache mock pre-loaded with given PR dicts."""
        mock_cache = MagicMock()

        # Session mock used for the CachedCommit query (commit loading path)
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []
        mock_session.query.return_value = mock_query
        mock_cache.get_session.return_value = mock_session
        mock_cache._commit_to_dict.return_value = {}

        # The new PR loading method
        mock_cache.get_cached_prs_for_report.return_value = prs
        return mock_cache

    def _make_identity_mock(self) -> MagicMock:
        mock_identity = MagicMock()
        mock_identity.get_developer_stats.return_value = {}
        return mock_identity

    def _make_analyzer_mock(self) -> MagicMock:
        mock_analyzer = MagicMock()
        mock_analyzer.ticket_extractor.analyze_ticket_coverage.return_value = {}
        mock_analyzer.ticket_extractor.calculate_developer_ticket_coverage.return_value = {}
        return mock_analyzer

    def _run_with_pr_mocks(
        self,
        cfg: Any,
        output_dir: Path,
        prs: list[dict[str, Any]],
    ) -> ReportResult:
        """Run run_report with a cache mock that returns the given PRs."""
        with (
            patch(self._REPORT_PATCH_PATHS["cache"]) as MockCache,
            patch(self._REPORT_PATCH_PATHS["identity"]) as MockIdentity,
            patch(self._REPORT_PATCH_PATHS["analyzer"]) as MockAnalyzer,
            patch(self._REPORT_PATCH_PATHS["csv_writer"]),
            patch(self._REPORT_PATCH_PATHS["analytics_writer"]),
            patch(self._REPORT_PATCH_PATHS["narrative_writer"]),
            patch(self._REPORT_PATCH_PATHS["trends_writer"]),
            patch(self._REPORT_PATCH_PATHS["json_exporter"]),
            patch(self._REPORT_PATCH_PATHS["dora"]) as MockDora,
        ):
            MockCache.return_value = self._make_cache_mock(prs)
            MockIdentity.return_value = self._make_identity_mock()
            MockAnalyzer.return_value = self._make_analyzer_mock()

            mock_dora = MagicMock()
            mock_dora.calculate_dora_metrics.return_value = {}
            MockDora.return_value = mock_dora

            return run_report(cfg=cfg, weeks=4, output_dir=output_dir)

    def test_get_cached_prs_for_report_called_with_github_repos(self, tmp_path: Path) -> None:
        """run_report calls get_cached_prs_for_report with the repo's github_repo slug."""

        cfg_text = _MINIMAL_CONFIG_TEMPLATE.format(
            cache_dir=str(tmp_path / ".cache"),
            repo_path=str(tmp_path / "repo"),
            output_dir=str(tmp_path / "reports"),
        )
        # Inject a github_repo field so the PR loading path is exercised
        cfg_dict = yaml.safe_load(cfg_text)
        cfg_dict["repositories"][0]["github_repo"] = "myorg/myrepo"
        config_file = tmp_path / "config_with_github.yaml"
        config_file.write_text(yaml.dump(cfg_dict))

        from gitflow_analytics.config import ConfigLoader

        cfg = ConfigLoader.load(config_file)
        output_dir = tmp_path / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        prs = [self._make_pr_dict(1), self._make_pr_dict(2)]

        with (
            patch(self._REPORT_PATCH_PATHS["cache"]) as MockCache,
            patch(self._REPORT_PATCH_PATHS["identity"]) as MockIdentity,
            patch(self._REPORT_PATCH_PATHS["analyzer"]) as MockAnalyzer,
            patch(self._REPORT_PATCH_PATHS["csv_writer"]),
            patch(self._REPORT_PATCH_PATHS["analytics_writer"]),
            patch(self._REPORT_PATCH_PATHS["narrative_writer"]),
            patch(self._REPORT_PATCH_PATHS["trends_writer"]),
            patch(self._REPORT_PATCH_PATHS["json_exporter"]),
            patch(self._REPORT_PATCH_PATHS["dora"]) as MockDora,
        ):
            mock_cache = self._make_cache_mock(prs)
            MockCache.return_value = mock_cache
            MockIdentity.return_value = self._make_identity_mock()
            MockAnalyzer.return_value = self._make_analyzer_mock()
            MockDora.return_value = MagicMock(calculate_dora_metrics=MagicMock(return_value={}))

            run_report(cfg=cfg, weeks=4, output_dir=output_dir)

        # Confirm the new method was called with the correct slug
        mock_cache.get_cached_prs_for_report.assert_called_once()
        call_args = mock_cache.get_cached_prs_for_report.call_args
        repo_paths_arg = call_args[0][0]  # first positional argument
        assert "myorg/myrepo" in repo_paths_arg

    def test_no_github_repo_skips_pr_load(self, tmp_path: Path) -> None:
        """When no repository has github_repo set, get_cached_prs_for_report is not called."""
        cfg = _make_config(tmp_path)
        # _make_config produces a repo without github_repo
        assert all(getattr(r, "github_repo", None) is None for r in cfg.repositories)

        output_dir = tmp_path / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch(self._REPORT_PATCH_PATHS["cache"]) as MockCache,
            patch(self._REPORT_PATCH_PATHS["identity"]) as MockIdentity,
            patch(self._REPORT_PATCH_PATHS["analyzer"]) as MockAnalyzer,
            patch(self._REPORT_PATCH_PATHS["csv_writer"]),
            patch(self._REPORT_PATCH_PATHS["analytics_writer"]),
            patch(self._REPORT_PATCH_PATHS["narrative_writer"]),
            patch(self._REPORT_PATCH_PATHS["trends_writer"]),
            patch(self._REPORT_PATCH_PATHS["json_exporter"]),
            patch(self._REPORT_PATCH_PATHS["dora"]) as MockDora,
        ):
            mock_cache = self._make_cache_mock([])
            MockCache.return_value = mock_cache
            MockIdentity.return_value = self._make_identity_mock()
            MockAnalyzer.return_value = self._make_analyzer_mock()
            MockDora.return_value = MagicMock(calculate_dora_metrics=MagicMock(return_value={}))

            run_report(cfg=cfg, weeks=4, output_dir=output_dir)

        mock_cache.get_cached_prs_for_report.assert_not_called()

    def test_prs_passed_to_dora_calculator(self, tmp_path: Path) -> None:
        """Loaded PRs are forwarded to DORAMetricsCalculator.calculate_dora_metrics."""

        cfg_dict = yaml.safe_load(
            _MINIMAL_CONFIG_TEMPLATE.format(
                cache_dir=str(tmp_path / ".cache"),
                repo_path=str(tmp_path / "repo"),
                output_dir=str(tmp_path / "reports"),
            )
        )
        cfg_dict["repositories"][0]["github_repo"] = "org/repo"
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(yaml.dump(cfg_dict))

        from gitflow_analytics.config import ConfigLoader

        cfg = ConfigLoader.load(config_file)
        output_dir = tmp_path / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        prs = [self._make_pr_dict(10)]

        with (
            patch(self._REPORT_PATCH_PATHS["cache"]) as MockCache,
            patch(self._REPORT_PATCH_PATHS["identity"]) as MockIdentity,
            patch(self._REPORT_PATCH_PATHS["analyzer"]) as MockAnalyzer,
            patch(self._REPORT_PATCH_PATHS["csv_writer"]),
            patch(self._REPORT_PATCH_PATHS["analytics_writer"]),
            patch(self._REPORT_PATCH_PATHS["narrative_writer"]),
            patch(self._REPORT_PATCH_PATHS["trends_writer"]),
            patch(self._REPORT_PATCH_PATHS["json_exporter"]),
            patch(self._REPORT_PATCH_PATHS["dora"]) as MockDora,
        ):
            MockCache.return_value = self._make_cache_mock(prs)
            MockIdentity.return_value = self._make_identity_mock()
            MockAnalyzer.return_value = self._make_analyzer_mock()
            mock_dora_instance = MagicMock()
            mock_dora_instance.calculate_dora_metrics.return_value = {}
            MockDora.return_value = mock_dora_instance

            run_report(cfg=cfg, weeks=4, output_dir=output_dir)

        # Verify PRs were passed to the DORA calculator
        mock_dora_instance.calculate_dora_metrics.assert_called_once()
        call_kwargs = mock_dora_instance.calculate_dora_metrics.call_args
        # Second positional argument is all_prs
        passed_prs = call_kwargs[0][1]
        assert passed_prs == prs

    def test_empty_pr_cache_produces_no_errors(self, tmp_path: Path) -> None:
        """When PRs are cached but the result set is empty, run_report completes cleanly."""

        cfg_dict = yaml.safe_load(
            _MINIMAL_CONFIG_TEMPLATE.format(
                cache_dir=str(tmp_path / ".cache"),
                repo_path=str(tmp_path / "repo"),
                output_dir=str(tmp_path / "reports"),
            )
        )
        cfg_dict["repositories"][0]["github_repo"] = "org/repo"
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(yaml.dump(cfg_dict))

        from gitflow_analytics.config import ConfigLoader

        cfg = ConfigLoader.load(config_file)
        output_dir = tmp_path / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self._run_with_pr_mocks(cfg, output_dir, prs=[])

        # run_report should complete (return a ReportResult) without raising.
        # PR-loading itself must not inject errors — only unrelated report
        # generation issues (CSV file not found from mocked writers) may appear.
        assert isinstance(result, ReportResult)
        pr_load_errors = [
            e for e in result.errors if "pr cache" in e.lower() or "get_cached_prs" in e.lower()
        ]
        assert pr_load_errors == [], f"Unexpected PR cache load errors: {pr_load_errors}"


# ---------------------------------------------------------------------------
# Unit tests for get_cached_prs_for_report method
# ---------------------------------------------------------------------------


class TestGetCachedPrsForReport:
    """Unit tests for GitAnalysisCache.get_cached_prs_for_report.

    Tests run against a real in-memory SQLite database via GitAnalysisCache
    to validate the query logic without mocking the internals.
    """

    def _make_cache(self, tmp_path: Path) -> Any:
        from gitflow_analytics.core.cache import GitAnalysisCache

        cache_dir = tmp_path / ".gitflow-cache"
        return GitAnalysisCache(cache_dir, ttl_hours=24)

    def _seed_pr(
        self,
        cache: Any,
        repo: str,
        number: int,
        created_at: datetime,
        merged_at: datetime | None = None,
    ) -> None:
        cache.cache_pr(
            repo,
            {
                "number": number,
                "title": f"PR {number}",
                "description": "",
                "author": "dev",
                "created_at": created_at,
                "merged_at": merged_at,
                "story_points": None,
                "labels": [],
                "commit_hashes": [],
            },
        )

    def test_returns_prs_within_date_range(self, tmp_path: Path) -> None:
        """PRs whose created_at falls inside the window are returned."""
        cache = self._make_cache(tmp_path)
        repo = "org/repo"
        inside = datetime(2025, 2, 5, tzinfo=timezone.utc)
        self._seed_pr(cache, repo, 1, inside)

        start = datetime(2025, 2, 1, tzinfo=timezone.utc)
        end = datetime(2025, 2, 28, tzinfo=timezone.utc)
        result = cache.get_cached_prs_for_report([repo], start, end)

        assert len(result) == 1
        assert result[0]["number"] == 1

    def test_excludes_prs_outside_date_range(self, tmp_path: Path) -> None:
        """PRs outside the date window are not returned."""
        cache = self._make_cache(tmp_path)
        repo = "org/repo"
        outside = datetime(2025, 3, 15, tzinfo=timezone.utc)
        self._seed_pr(cache, repo, 99, outside)

        start = datetime(2025, 2, 1, tzinfo=timezone.utc)
        end = datetime(2025, 2, 28, tzinfo=timezone.utc)
        result = cache.get_cached_prs_for_report([repo], start, end)

        assert result == []

    def test_returns_prs_from_multiple_repos(self, tmp_path: Path) -> None:
        """PRs from all listed repos are returned."""
        cache = self._make_cache(tmp_path)
        ts = datetime(2025, 2, 10, tzinfo=timezone.utc)
        self._seed_pr(cache, "org/alpha", 1, ts)
        self._seed_pr(cache, "org/beta", 2, ts)

        start = datetime(2025, 2, 1, tzinfo=timezone.utc)
        end = datetime(2025, 2, 28, tzinfo=timezone.utc)
        result = cache.get_cached_prs_for_report(["org/alpha", "org/beta"], start, end)

        numbers = {pr["number"] for pr in result}
        assert numbers == {1, 2}

    def test_filters_by_repo_path(self, tmp_path: Path) -> None:
        """PRs from repos not in the list are excluded."""
        cache = self._make_cache(tmp_path)
        ts = datetime(2025, 2, 10, tzinfo=timezone.utc)
        self._seed_pr(cache, "org/wanted", 1, ts)
        self._seed_pr(cache, "org/unwanted", 2, ts)

        start = datetime(2025, 2, 1, tzinfo=timezone.utc)
        end = datetime(2025, 2, 28, tzinfo=timezone.utc)
        result = cache.get_cached_prs_for_report(["org/wanted"], start, end)

        assert len(result) == 1
        assert result[0]["number"] == 1

    def test_empty_repo_list_returns_empty(self, tmp_path: Path) -> None:
        """An empty repo_paths list returns an empty result without querying."""
        cache = self._make_cache(tmp_path)
        ts = datetime(2025, 2, 10, tzinfo=timezone.utc)
        self._seed_pr(cache, "org/repo", 1, ts)

        start = datetime(2025, 2, 1, tzinfo=timezone.utc)
        end = datetime(2025, 2, 28, tzinfo=timezone.utc)
        result = cache.get_cached_prs_for_report([], start, end)

        assert result == []

    def test_returns_correct_pr_dict_fields(self, tmp_path: Path) -> None:
        """Returned dicts include all standard _pr_to_dict fields."""
        cache = self._make_cache(tmp_path)
        ts = datetime(2025, 2, 5, tzinfo=timezone.utc)
        merged = datetime(2025, 2, 6, tzinfo=timezone.utc)
        self._seed_pr(cache, "org/repo", 7, ts, merged)

        start = datetime(2025, 2, 1, tzinfo=timezone.utc)
        end = datetime(2025, 2, 28, tzinfo=timezone.utc)
        result = cache.get_cached_prs_for_report(["org/repo"], start, end)

        assert len(result) == 1
        pr = result[0]
        for key in (
            "number",
            "title",
            "author",
            "created_at",
            "merged_at",
            "labels",
            "commit_hashes",
        ):
            assert key in pr, f"Expected key '{key}' in PR dict"

    def test_boundary_dates_are_inclusive(self, tmp_path: Path) -> None:
        """PRs created exactly on start_date or end_date are included."""
        cache = self._make_cache(tmp_path)
        start = datetime(2025, 2, 1, tzinfo=timezone.utc)
        end = datetime(2025, 2, 28, tzinfo=timezone.utc)

        self._seed_pr(cache, "org/repo", 1, start)  # on start boundary
        self._seed_pr(cache, "org/repo", 2, end)  # on end boundary

        result = cache.get_cached_prs_for_report(["org/repo"], start, end)
        numbers = {pr["number"] for pr in result}
        assert 1 in numbers
        assert 2 in numbers

    def test_no_cached_prs_returns_empty_list(self, tmp_path: Path) -> None:
        """When no PRs are cached for the repos/date range, an empty list is returned."""
        cache = self._make_cache(tmp_path)

        start = datetime(2025, 2, 1, tzinfo=timezone.utc)
        end = datetime(2025, 2, 28, tzinfo=timezone.utc)
        result = cache.get_cached_prs_for_report(["org/repo"], start, end)

        assert result == []


# ---------------------------------------------------------------------------
# Qualitative join tests — run_report merges change_type / complexity
# ---------------------------------------------------------------------------


class TestRunReportQualitativeJoin:
    """Verify that run_report merges QualitativeCommitData into commit dicts.

    The fix ensures that ``change_type``, ``change_type_confidence``,
    ``complexity``, and ``processing_method`` stored in the
    ``qualitative_commits`` table are included in the commit dicts passed to
    all downstream report generators.
    """

    _REPORT_PATCH_PATHS = {
        "cache": "gitflow_analytics.core.cache.GitAnalysisCache",
        "identity": "gitflow_analytics.core.identity.DeveloperIdentityResolver",
        "analyzer": "gitflow_analytics.core.analyzer.GitAnalyzer",
        "csv_writer": "gitflow_analytics.reports.csv_writer.CSVReportGenerator",
        "analytics_writer": "gitflow_analytics.reports.analytics_writer.AnalyticsReportGenerator",
        "narrative_writer": "gitflow_analytics.reports.narrative_writer.NarrativeReportGenerator",
        "trends_writer": "gitflow_analytics.reports.weekly_trends_writer.WeeklyTrendsWriter",
        "json_exporter": "gitflow_analytics.reports.json_exporter.ComprehensiveJSONExporter",
        "dora": "gitflow_analytics.metrics.dora.DORAMetricsCalculator",
    }

    def _make_identity_mock(self) -> MagicMock:
        mock_identity = MagicMock()
        mock_identity.get_developer_stats.return_value = {}
        return mock_identity

    def _make_analyzer_mock(self) -> MagicMock:
        mock_analyzer = MagicMock()
        mock_analyzer.ticket_extractor.analyze_ticket_coverage.return_value = {}
        mock_analyzer.ticket_extractor.calculate_developer_ticket_coverage.return_value = {}
        return mock_analyzer

    def _make_cache_mock_with_commits(
        self,
        commit_rows: list[MagicMock],
        qual_rows: list[MagicMock],
    ) -> MagicMock:
        """Build a GitAnalysisCache mock that returns given ORM row mocks.

        The session's ``query()`` must dispatch differently depending on which
        model class is queried:

        - ``CachedCommit`` → returns commit_rows
        - ``QualitativeCommitData`` → returns qual_rows
        """
        from gitflow_analytics.models.database import CachedCommit
        from gitflow_analytics.models.database_metrics_models import QualitativeCommitData

        mock_cache = MagicMock()

        # Build two independent query mocks, one per model class
        commit_query = MagicMock()
        commit_query.filter.return_value = commit_query
        commit_query.order_by.return_value = commit_query
        commit_query.all.return_value = commit_rows

        qual_query = MagicMock()
        qual_query.filter.return_value = qual_query
        qual_query.all.return_value = qual_rows

        def _dispatch_query(model_class: type) -> MagicMock:
            if model_class is CachedCommit:
                return commit_query
            if model_class is QualitativeCommitData:
                return qual_query
            return MagicMock()

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.side_effect = _dispatch_query

        mock_cache.get_session.return_value = mock_session
        mock_cache._commit_to_dict.side_effect = lambda cc: {
            "hash": cc.commit_hash,
            "author_email": cc.author_email,
        }
        mock_cache.get_cached_prs_for_report.return_value = []
        return mock_cache

    def _make_commit_row(self, row_id: int, commit_hash: str = "abc123") -> MagicMock:
        """Build a minimal CachedCommit ORM row mock."""
        row = MagicMock()
        row.id = row_id
        row.commit_hash = commit_hash
        row.author_email = "dev@example.com"
        row.repo_path = "/repos/test"
        return row

    def _make_qual_row(
        self,
        commit_id: int,
        change_type: str = "feature",
        confidence: float = 0.9,
        complexity: int = 3,
        processing_method: str = "llm",
    ) -> MagicMock:
        """Build a minimal QualitativeCommitData ORM row mock."""
        row = MagicMock()
        row.commit_id = commit_id
        row.change_type = change_type
        row.change_type_confidence = confidence
        row.complexity = complexity
        row.processing_method = processing_method
        return row

    def _captured_commit_dicts(
        self,
        cfg: Any,
        output_dir: Path,
        cache_mock: MagicMock,
    ) -> list[dict]:
        """Run run_report and capture all_commits via the DORA calculator call."""
        captured: list[list] = []

        with (
            patch(self._REPORT_PATCH_PATHS["cache"]) as MockCache,
            patch(self._REPORT_PATCH_PATHS["identity"]) as MockIdentity,
            patch(self._REPORT_PATCH_PATHS["analyzer"]) as MockAnalyzer,
            patch(self._REPORT_PATCH_PATHS["csv_writer"]),
            patch(self._REPORT_PATCH_PATHS["analytics_writer"]),
            patch(self._REPORT_PATCH_PATHS["narrative_writer"]),
            patch(self._REPORT_PATCH_PATHS["trends_writer"]),
            patch(self._REPORT_PATCH_PATHS["json_exporter"]),
            patch(self._REPORT_PATCH_PATHS["dora"]) as MockDora,
        ):
            MockCache.return_value = cache_mock
            MockIdentity.return_value = self._make_identity_mock()
            MockAnalyzer.return_value = self._make_analyzer_mock()

            mock_dora_instance = MagicMock()
            # Capture the first positional arg (all_commits) passed to DORA
            mock_dora_instance.calculate_dora_metrics.side_effect = (
                lambda commits, *args, **kwargs: captured.append(commits) or {}
            )
            MockDora.return_value = mock_dora_instance

            run_report(cfg=cfg, weeks=4, output_dir=output_dir)

        return captured[0] if captured else []

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_qualitative_fields_merged_into_commit_dicts(self, tmp_path: Path) -> None:
        """Commits with qualitative data have change_type and complexity populated."""
        cfg = _make_config(tmp_path)
        output_dir = tmp_path / "reports"
        output_dir.mkdir(exist_ok=True)

        commit_row = self._make_commit_row(row_id=1, commit_hash="deadbeef")
        qual_row = self._make_qual_row(
            commit_id=1, change_type="bugfix", confidence=0.95, complexity=2
        )
        cache_mock = self._make_cache_mock_with_commits([commit_row], [qual_row])

        commits = self._captured_commit_dicts(cfg, output_dir, cache_mock)

        assert len(commits) == 1
        commit = commits[0]
        assert commit["change_type"] == "bugfix"
        assert commit["change_type_confidence"] == 0.95
        assert commit["complexity"] == 2
        assert commit["processing_method"] == "llm"

    def test_commits_without_qualitative_data_unaffected(self, tmp_path: Path) -> None:
        """Commits with no qualitative row are returned without change_type keys."""
        cfg = _make_config(tmp_path)
        output_dir = tmp_path / "reports"
        output_dir.mkdir(exist_ok=True)

        commit_row = self._make_commit_row(row_id=42, commit_hash="cafebabe")
        # No qualitative rows
        cache_mock = self._make_cache_mock_with_commits([commit_row], [])

        commits = self._captured_commit_dicts(cfg, output_dir, cache_mock)

        assert len(commits) == 1
        # change_type must not be present when there is no qualitative row
        assert "change_type" not in commits[0]

    def test_partial_qualitative_data_merged_correctly(self, tmp_path: Path) -> None:
        """Only commits that have qualitative rows get classification fields."""
        cfg = _make_config(tmp_path)
        output_dir = tmp_path / "reports"
        output_dir.mkdir(exist_ok=True)

        row_with_qual = self._make_commit_row(row_id=1, commit_hash="aaa111")
        row_without_qual = self._make_commit_row(row_id=2, commit_hash="bbb222")
        qual_row = self._make_qual_row(commit_id=1, change_type="refactor", complexity=4)

        cache_mock = self._make_cache_mock_with_commits(
            [row_with_qual, row_without_qual], [qual_row]
        )

        commits = self._captured_commit_dicts(cfg, output_dir, cache_mock)

        assert len(commits) == 2
        # Find the enriched commit
        enriched = next(c for c in commits if c.get("hash") == "aaa111")
        bare = next(c for c in commits if c.get("hash") == "bbb222")

        assert enriched["change_type"] == "refactor"
        assert enriched["complexity"] == 4
        assert "change_type" not in bare

    def test_multiple_commits_multiple_qual_rows(self, tmp_path: Path) -> None:
        """All commits with qualitative rows receive their correct classifications."""
        cfg = _make_config(tmp_path)
        output_dir = tmp_path / "reports"
        output_dir.mkdir(exist_ok=True)

        rows = [
            self._make_commit_row(row_id=1, commit_hash="hash1"),
            self._make_commit_row(row_id=2, commit_hash="hash2"),
            self._make_commit_row(row_id=3, commit_hash="hash3"),
        ]
        qual_rows = [
            self._make_qual_row(commit_id=1, change_type="feature", complexity=5),
            self._make_qual_row(commit_id=2, change_type="bugfix", complexity=1),
            self._make_qual_row(commit_id=3, change_type="chore", complexity=2),
        ]
        cache_mock = self._make_cache_mock_with_commits(rows, qual_rows)

        commits = self._captured_commit_dicts(cfg, output_dir, cache_mock)

        by_hash = {c["hash"]: c for c in commits}
        assert by_hash["hash1"]["change_type"] == "feature"
        assert by_hash["hash1"]["complexity"] == 5
        assert by_hash["hash2"]["change_type"] == "bugfix"
        assert by_hash["hash3"]["change_type"] == "chore"

    def test_no_commits_produces_no_errors(self, tmp_path: Path) -> None:
        """An empty commit result set does not cause errors in qualitative join."""
        cfg = _make_config(tmp_path)
        output_dir = tmp_path / "reports"
        output_dir.mkdir(exist_ok=True)

        cache_mock = self._make_cache_mock_with_commits([], [])

        with (
            patch(self._REPORT_PATCH_PATHS["cache"]) as MockCache,
            patch(self._REPORT_PATCH_PATHS["identity"]) as MockIdentity,
            patch(self._REPORT_PATCH_PATHS["analyzer"]) as MockAnalyzer,
            patch(self._REPORT_PATCH_PATHS["csv_writer"]),
            patch(self._REPORT_PATCH_PATHS["analytics_writer"]),
            patch(self._REPORT_PATCH_PATHS["narrative_writer"]),
            patch(self._REPORT_PATCH_PATHS["trends_writer"]),
            patch(self._REPORT_PATCH_PATHS["json_exporter"]),
            patch(self._REPORT_PATCH_PATHS["dora"]) as MockDora,
        ):
            MockCache.return_value = cache_mock
            MockIdentity.return_value = self._make_identity_mock()
            MockAnalyzer.return_value = self._make_analyzer_mock()
            MockDora.return_value = MagicMock(calculate_dora_metrics=MagicMock(return_value={}))

            result = run_report(cfg=cfg, weeks=4, output_dir=output_dir)

        qual_errors = [e for e in result.errors if "qualitative" in e.lower()]
        assert qual_errors == []


# ---------------------------------------------------------------------------
# Tests for strip_suffixes wiring to DeveloperIdentityResolver
# ---------------------------------------------------------------------------


class TestDeveloperIdentityResolverStripSuffixes:
    """Verify that strip_suffixes is accepted and applied by DeveloperIdentityResolver.

    These tests use a real in-memory SQLite database to confirm the suffix
    normalisation actually improves fuzzy-match results.
    """

    def _make_resolver(
        self,
        tmp_path: Path,
        strip_suffixes: list[str] | None = None,
    ) -> Any:
        from gitflow_analytics.core.identity import DeveloperIdentityResolver

        db_path = tmp_path / "identities.db"
        return DeveloperIdentityResolver(
            str(db_path),
            similarity_threshold=0.85,
            strip_suffixes=strip_suffixes,
        )

    def test_strip_suffixes_stored_on_instance(self, tmp_path: Path) -> None:
        """Constructor stores strip_suffixes on the instance."""
        from gitflow_analytics.core.identity import DeveloperIdentityResolver

        resolver = DeveloperIdentityResolver(
            str(tmp_path / "id.db"),
            strip_suffixes=["-dev", "-staging"],
        )
        assert "-dev" in resolver._strip_suffixes
        assert "-staging" in resolver._strip_suffixes

    def test_no_strip_suffixes_defaults_to_empty(self, tmp_path: Path) -> None:
        """When strip_suffixes is not provided the list is empty."""
        from gitflow_analytics.core.identity import DeveloperIdentityResolver

        resolver = DeveloperIdentityResolver(str(tmp_path / "id.db"))
        assert resolver._strip_suffixes == []

    def test_normalize_email_local_strips_suffix(self, tmp_path: Path) -> None:
        """_normalize_email_local removes configured suffixes from the local-part."""
        resolver = self._make_resolver(tmp_path, strip_suffixes=["-dev", "-staging"])
        assert resolver._normalize_email_local("john-dev") == "john"
        assert resolver._normalize_email_local("alice-staging") == "alice"

    def test_normalize_email_local_no_suffix_unchanged(self, tmp_path: Path) -> None:
        """_normalize_email_local leaves local-parts without matching suffixes intact."""
        resolver = self._make_resolver(tmp_path, strip_suffixes=["-dev"])
        assert resolver._normalize_email_local("john") == "john"

    def test_strip_suffixes_improves_fuzzy_match(self, tmp_path: Path) -> None:
        """Suffix stripping allows john-dev@co.com to match john@co.com.

        Scoring breakdown (name similarity = 1.0, same domain):
          Without stripping: 0.4 (name) + 0.3 (domain) = 0.7
          With stripping:    0.4 (name) + 0.3 (domain) + 0.1 (stripped-local match) = 0.8

        Using a threshold of 0.75 means:
          - Without stripping: 0.7 < 0.75  → new identity created
          - With stripping:    0.8 >= 0.75 → existing identity reused
        """
        from gitflow_analytics.core.identity import DeveloperIdentityResolver

        threshold = 0.75
        resolver_without = DeveloperIdentityResolver(
            str(tmp_path / "without" / "id.db"),
            similarity_threshold=threshold,
            strip_suffixes=None,
        )
        resolver_with = DeveloperIdentityResolver(
            str(tmp_path / "with" / "id.db"),
            similarity_threshold=threshold,
            strip_suffixes=["-dev"],
        )

        # Seed the canonical identity into both resolvers
        canonical_email = "john@example.com"
        canonical_id_without = resolver_without.resolve_developer("John Smith", canonical_email)
        canonical_id_with = resolver_with.resolve_developer("John Smith", canonical_email)

        # Resolve the suffixed variant
        variant_email = "john-dev@example.com"
        id_without = resolver_without.resolve_developer("John Smith", variant_email)
        id_with = resolver_with.resolve_developer("John Smith", variant_email)

        # Without suffix stripping the variant misses the threshold → new identity
        assert id_without != canonical_id_without, (
            "Without strip_suffixes, john-dev@ should create a new identity "
            "(score 0.7 < threshold 0.75)"
        )
        # With suffix stripping the normalised local-parts match → reuses canonical identity
        assert id_with == canonical_id_with, (
            "With strip_suffixes=['-dev'], john-dev@example.com should resolve to the "
            "same identity as john@example.com (score 0.8 >= threshold 0.75)"
        )

    def test_strip_suffixes_wired_from_pipeline_report(self, tmp_path: Path) -> None:
        """run_report passes analysis.identity.strip_suffixes to DeveloperIdentityResolver."""
        from gitflow_analytics.config import ConfigLoader

        cfg_dict = yaml.safe_load(
            _MINIMAL_CONFIG_TEMPLATE.format(
                cache_dir=str(tmp_path / ".cache"),
                repo_path=str(tmp_path / "repo"),
                output_dir=str(tmp_path / "reports"),
            )
        )
        # Inject identity.strip_suffixes into config
        cfg_dict.setdefault("analysis", {})["identity"] = {
            "strip_suffixes": ["-dev", "-contractor"]
        }
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(yaml.dump(cfg_dict))
        cfg = ConfigLoader.load(config_file)
        output_dir = tmp_path / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        _PATCHES = {
            "cache": "gitflow_analytics.core.cache.GitAnalysisCache",
            "identity": "gitflow_analytics.core.identity.DeveloperIdentityResolver",
            "analyzer": "gitflow_analytics.core.analyzer.GitAnalyzer",
            "csv_writer": "gitflow_analytics.reports.csv_writer.CSVReportGenerator",
            "analytics_writer": "gitflow_analytics.reports.analytics_writer.AnalyticsReportGenerator",
            "narrative_writer": "gitflow_analytics.reports.narrative_writer.NarrativeReportGenerator",
            "trends_writer": "gitflow_analytics.reports.weekly_trends_writer.WeeklyTrendsWriter",
            "json_exporter": "gitflow_analytics.reports.json_exporter.ComprehensiveJSONExporter",
            "dora": "gitflow_analytics.metrics.dora.DORAMetricsCalculator",
        }

        captured_kwargs: list[dict] = []

        with (
            patch(_PATCHES["cache"]) as MockCache,
            patch(_PATCHES["identity"]) as MockIdentity,
            patch(_PATCHES["analyzer"]) as MockAnalyzer,
            patch(_PATCHES["csv_writer"]),
            patch(_PATCHES["analytics_writer"]),
            patch(_PATCHES["narrative_writer"]),
            patch(_PATCHES["trends_writer"]),
            patch(_PATCHES["json_exporter"]),
            patch(_PATCHES["dora"]) as MockDora,
        ):
            # Minimal cache mock (no commits)
            mock_cache = MagicMock()
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_query = MagicMock()
            mock_query.filter.return_value = mock_query
            mock_query.order_by.return_value = mock_query
            mock_query.all.return_value = []
            mock_session.query.return_value = mock_query
            mock_cache.get_session.return_value = mock_session
            mock_cache._commit_to_dict.return_value = {}
            mock_cache.get_cached_prs_for_report.return_value = []
            MockCache.return_value = mock_cache

            # Capture constructor kwargs
            def _capture_identity(*args: Any, **kwargs: Any) -> MagicMock:
                captured_kwargs.append(kwargs)
                mock_id = MagicMock()
                mock_id.get_developer_stats.return_value = {}
                return mock_id

            MockIdentity.side_effect = _capture_identity
            mock_analyzer = MagicMock()
            mock_analyzer.ticket_extractor.analyze_ticket_coverage.return_value = {}
            mock_analyzer.ticket_extractor.calculate_developer_ticket_coverage.return_value = {}
            MockAnalyzer.return_value = mock_analyzer
            MockDora.return_value = MagicMock(calculate_dora_metrics=MagicMock(return_value={}))

            run_report(cfg=cfg, weeks=4, output_dir=output_dir)

        assert captured_kwargs, "DeveloperIdentityResolver was not constructed"
        kwargs = captured_kwargs[0]
        passed_suffixes = kwargs.get("strip_suffixes") or []
        assert (
            "-dev" in passed_suffixes
        ), f"Expected '-dev' in strip_suffixes passed to DeveloperIdentityResolver, got: {passed_suffixes}"
        assert "-contractor" in passed_suffixes


# ---------------------------------------------------------------------------
# CLI command wiring tests (smoke-test via Click's test runner)
# ---------------------------------------------------------------------------


class TestCLIPipelineCommands:
    """Smoke tests to verify collect/classify/report CLI commands are registered."""

    def test_collect_command_help(self) -> None:
        """gfa collect --help exits 0 and shows key options."""
        from click.testing import CliRunner

        from gitflow_analytics.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["collect", "--help"])

        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--weeks" in result.output
        assert "--force" in result.output

    def test_classify_command_help(self) -> None:
        """gfa classify --help exits 0 and shows key options."""
        from click.testing import CliRunner

        from gitflow_analytics.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["classify", "--help"])

        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--reclassify" in result.output

    def test_report_command_help(self) -> None:
        """gfa report --help exits 0 and shows key options."""
        from click.testing import CliRunner

        from gitflow_analytics.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--help"])

        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--weeks" in result.output
        assert "--generate-csv" in result.output
        assert "--output" in result.output

    def test_collect_missing_config_exits_nonzero(self, tmp_path: Path) -> None:
        """gfa collect with a nonexistent config file exits with a nonzero code."""
        from click.testing import CliRunner

        from gitflow_analytics.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["collect", "-c", str(tmp_path / "nonexistent.yaml")])

        assert result.exit_code != 0

    def test_classify_missing_config_exits_nonzero(self, tmp_path: Path) -> None:
        """gfa classify with a nonexistent config file exits with a nonzero code."""
        from click.testing import CliRunner

        from gitflow_analytics.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["classify", "-c", str(tmp_path / "nonexistent.yaml")])

        assert result.exit_code != 0

    def test_report_missing_config_exits_nonzero(self, tmp_path: Path) -> None:
        """gfa report with a nonexistent config file exits with a nonzero code."""
        from click.testing import CliRunner

        from gitflow_analytics.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "-c", str(tmp_path / "nonexistent.yaml")])

        assert result.exit_code != 0

    def test_collect_reports_summary_on_success(self, tmp_path: Path) -> None:
        """gfa collect prints a 'Collect complete' summary when collect finishes."""
        from click.testing import CliRunner

        from gitflow_analytics.cli import cli

        # Write a valid config with one (missing) repository
        cfg_file = tmp_path / "config.yaml"
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        output_dir = tmp_path / "reports"
        output_dir.mkdir()
        cfg_file.write_text(
            _MINIMAL_CONFIG_TEMPLATE.format(
                cache_dir=str(cache_dir),
                repo_path=str(tmp_path / "nonexistent-repo"),
                output_dir=str(output_dir),
            )
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["collect", "-c", str(cfg_file), "--weeks", "1"])

        # The repo does not exist so it is counted as a failure, but the command
        # itself should finish with a "Collect complete" summary (exit code 0).
        assert result.exit_code == 0
        assert "collect complete" in result.output.lower()


# ---------------------------------------------------------------------------
# Result dataclass tests
# ---------------------------------------------------------------------------


class TestResultDataclasses:
    """Verify default values and structure of result dataclasses."""

    def test_collect_result_defaults(self) -> None:
        r = CollectResult()
        assert r.total_commits == 0
        assert r.repos_fetched == 0
        assert r.repos_cached == 0
        assert r.repos_failed == 0
        assert r.errors == []

    def test_classify_result_defaults(self) -> None:
        r = ClassifyResult()
        assert r.processed_batches == 0
        assert r.total_commits == 0
        assert r.skipped_batches == 0
        assert r.errors == []

    def test_report_result_defaults(self) -> None:
        r = ReportResult()
        assert r.generated_reports == []
        assert r.output_dir is None
        assert r.errors == []

    def test_collect_result_error_accumulation(self) -> None:
        r = CollectResult()
        r.errors.append("error 1")
        r.errors.append("error 2")
        assert len(r.errors) == 2

    def test_report_result_generated_reports_accumulation(self) -> None:
        r = ReportResult()
        r.generated_reports.append("report_a.csv")
        r.generated_reports.append("report_b.md")
        assert len(r.generated_reports) == 2
