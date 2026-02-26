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
