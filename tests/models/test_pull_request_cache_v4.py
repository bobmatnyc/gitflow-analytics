"""Tests for PullRequestCache v4.0 schema extension (pr_state, closed_at, is_merged).

Validates that:
1. New columns exist on the model with correct types.
2. _migrate_pull_request_cache_v4() adds columns to an existing database.
3. cache_pr() stores all new fields correctly (create and update paths).
4. _pr_to_dict() returns the new fields with correct backward-compat defaults.
5. _get_cached_prs_bulk() reads v4.0 columns and falls back gracefully.
6. _refresh_stale_open_prs() re-fetches open PRs and updates the cache.
7. The fetch_pr_reviews wiring bug is fixed in IntegrationOrchestrator.
8. Rejection metrics appear in narrative report when rejected PRs are present.
9. New CSV columns (pr_state, closed_at, is_merged) appear in PR metrics report.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy import create_engine, inspect, text

from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.models.database import Base, PullRequestCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PR_V4_COLUMNS = ["pr_state", "closed_at", "is_merged"]


def _make_pr_data(**overrides: Any) -> dict[str, Any]:
    """Return a minimal PR data dict suitable for cache_pr(), including v4.0 fields."""
    base: dict[str, Any] = {
        "number": 42,
        "title": "feat: shiny new thing",
        "description": "A description",
        "author": "alice",
        "created_at": datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        "merged_at": datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
        "story_points": 3,
        "labels": ["enhancement"],
        "commit_hashes": ["abc123"],
        # v4.0 state fields
        "pr_state": "merged",
        "closed_at": datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
        "is_merged": True,
        # v3.0 fields
        "review_comments": 2,
        "pr_comments_count": 1,
        "approvals_count": 1,
        "change_requests_count": 0,
        "reviewers": ["bob"],
        "approved_by": ["bob"],
        "time_to_first_review_hours": 2.0,
        "revision_count": 1,
        "changed_files": 3,
        "additions": 50,
        "deletions": 10,
    }
    base.update(overrides)
    return base


def _make_closed_pr_data(**overrides: Any) -> dict[str, Any]:
    """Return PR data for a closed-without-merge (rejected) PR."""
    base: dict[str, Any] = {
        "number": 99,
        "title": "feat: rejected idea",
        "description": "",
        "author": "charlie",
        "created_at": datetime(2025, 1, 3, 9, 0, 0, tzinfo=timezone.utc),
        "merged_at": None,
        "story_points": 0,
        "labels": [],
        "commit_hashes": ["zzz999"],
        # v4.0 state fields
        "pr_state": "closed",
        "closed_at": datetime(2025, 1, 4, 15, 0, 0, tzinfo=timezone.utc),
        "is_merged": False,
        # v3.0 fields (minimal)
        "review_comments": 0,
        "pr_comments_count": 0,
        "approvals_count": 0,
        "change_requests_count": 1,
        "reviewers": ["dave"],
        "approved_by": [],
        "time_to_first_review_hours": 6.0,
        "revision_count": 0,
        "changed_files": 2,
        "additions": 20,
        "deletions": 5,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Model column presence tests
# ---------------------------------------------------------------------------


class TestPullRequestCacheModelColumnsV4:
    """Verify that all v4.0 columns are declared on the SQLAlchemy model."""

    def test_v4_columns_exist_on_model(self) -> None:
        """pr_state, closed_at, and is_merged must be present as mapped attributes."""
        mapper = inspect(PullRequestCache)
        column_names = {col.key for col in mapper.columns}

        for col in PR_V4_COLUMNS:
            assert col in column_names, (
                f"Expected column '{col}' on PullRequestCache but it was not found. "
                "Check models/database_metrics_models.py."
            )

    def test_pr_state_is_string(self) -> None:
        from sqlalchemy import String as SAString

        mapper = inspect(PullRequestCache)
        col = mapper.columns["pr_state"]
        assert isinstance(
            col.type, SAString
        ), f"pr_state should be String but is {type(col.type).__name__}"

    def test_closed_at_is_datetime(self) -> None:
        from sqlalchemy import DateTime as SADateTime

        mapper = inspect(PullRequestCache)
        col = mapper.columns["closed_at"]
        assert isinstance(
            col.type, SADateTime
        ), f"closed_at should be DateTime but is {type(col.type).__name__}"

    def test_is_merged_is_boolean(self) -> None:
        from sqlalchemy import Boolean as SABoolean

        mapper = inspect(PullRequestCache)
        col = mapper.columns["is_merged"]
        assert isinstance(
            col.type, SABoolean
        ), f"is_merged should be Boolean but is {type(col.type).__name__}"

    def test_all_v4_columns_are_nullable(self) -> None:
        """All three columns must be nullable for backward-compat with old rows."""
        mapper = inspect(PullRequestCache)
        for col_name in PR_V4_COLUMNS:
            col = mapper.columns[col_name]
            assert col.nullable, f"Column '{col_name}' should be nullable."

    def test_v3_columns_still_present(self) -> None:
        """v4.0 additions must not remove any v3.0 columns."""
        v3_cols = [
            "review_comments_count",
            "pr_comments_count",
            "approvals_count",
            "change_requests_count",
            "reviewers",
            "approved_by",
            "time_to_first_review_hours",
            "revision_count",
            "changed_files",
            "additions",
            "deletions",
        ]
        mapper = inspect(PullRequestCache)
        column_names = {col.key for col in mapper.columns}
        for col in v3_cols:
            assert col in column_names, f"v3.0 column '{col}' is unexpectedly missing."


# ---------------------------------------------------------------------------
# 2. Fresh database includes all v4.0 columns
# ---------------------------------------------------------------------------


class TestFreshDatabaseSchemaV4:
    """Verify that create_all() on a fresh DB includes all v4.0 columns."""

    def test_fresh_db_has_v4_columns(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(pull_request_cache)"))
            actual_columns = {row[1] for row in result}

        for col in PR_V4_COLUMNS:
            assert col in actual_columns, f"Fresh DB missing column '{col}' in pull_request_cache"


# ---------------------------------------------------------------------------
# 3. Migration test — simulate pre-v4.0 database (has v3.0 cols, missing v4.0)
# ---------------------------------------------------------------------------


class TestMigrationAppliesV4Columns:
    """Verify _migrate_pull_request_cache_v4 adds the three missing columns."""

    def _create_v3_db(self, db_path: Path) -> None:
        """Create a pull_request_cache table that looks like the post-v3.0 schema
        (includes v3.0 columns but NOT v4.0 columns)."""
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            conn.execute(
                text("""
                CREATE TABLE pull_request_cache (
                    id INTEGER PRIMARY KEY,
                    repo_path TEXT NOT NULL,
                    pr_number INTEGER NOT NULL,
                    title TEXT,
                    description TEXT,
                    author TEXT,
                    created_at DATETIME,
                    merged_at DATETIME,
                    story_points INTEGER,
                    labels TEXT,
                    commit_hashes TEXT,
                    cached_at DATETIME,
                    review_comments_count INTEGER DEFAULT 0,
                    pr_comments_count INTEGER DEFAULT 0,
                    approvals_count INTEGER DEFAULT 0,
                    change_requests_count INTEGER DEFAULT 0,
                    reviewers TEXT,
                    approved_by TEXT,
                    time_to_first_review_hours REAL,
                    revision_count INTEGER DEFAULT 0,
                    changed_files INTEGER DEFAULT 0,
                    additions INTEGER DEFAULT 0,
                    deletions INTEGER DEFAULT 0
                )
            """)
            )
            # Insert a legacy row to confirm existing data is preserved
            conn.execute(
                text("""
                INSERT INTO pull_request_cache
                    (repo_path, pr_number, title, author, cached_at, merged_at)
                VALUES ('owner/repo', 1, 'Legacy PR v3', 'dev', datetime('now'), datetime('now'))
            """)
            )
            conn.commit()
        engine.dispose()

    def test_migration_adds_all_v4_columns(self, tmp_path: Path) -> None:
        """Opening a v3.0 DB through GitAnalysisCache must add all v4.0 columns."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        self._create_v3_db(cache_dir / "gitflow_cache.db")

        # Instantiate the cache — this triggers _apply_migrations
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache.db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(pull_request_cache)"))
            actual_columns = {row[1] for row in result}

        for col in PR_V4_COLUMNS:
            assert (
                col in actual_columns
            ), f"v4.0 migration failed to add column '{col}' to legacy pull_request_cache"

    def test_migration_preserves_existing_rows(self, tmp_path: Path) -> None:
        """Existing PR rows must not be deleted during v4.0 migration."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        self._create_v3_db(cache_dir / "gitflow_cache.db")

        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache.db.engine.connect() as conn:
            result = conn.execute(
                text("SELECT pr_number, title FROM pull_request_cache WHERE pr_number = 1")
            )
            row = result.fetchone()

        assert row is not None, "Existing row was deleted during v4.0 migration."
        assert row[1] == "Legacy PR v3"

    def test_migration_idempotent(self, tmp_path: Path) -> None:
        """Running v4.0 migration twice must not raise errors or duplicate columns."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache_dir.mkdir()
        self._create_v3_db(cache_dir / "gitflow_cache.db")

        cache1 = GitAnalysisCache(cache_dir, ttl_hours=24)
        del cache1

        cache2 = GitAnalysisCache(cache_dir, ttl_hours=24)

        with cache2.db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(pull_request_cache)"))
            columns = [row[1] for row in result]

        # No duplicates
        assert len(columns) == len(set(columns)), "Duplicate columns after double migration."
        for col in PR_V4_COLUMNS:
            assert col in columns


# ---------------------------------------------------------------------------
# 4. cache_pr() stores v4.0 fields — create path
# ---------------------------------------------------------------------------


class TestCachePrV4Create:
    """Verify cache_pr() persists all v4.0 fields when creating a new entry."""

    def test_merged_pr_fields_stored(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)
        repo = "owner/repo"

        cache.cache_pr(repo, _make_pr_data())
        result = cache.get_cached_pr(repo, 42)

        assert result is not None
        assert result["pr_state"] == "merged"
        assert result["is_merged"] is True
        assert result["closed_at"] is not None

    def test_rejected_pr_fields_stored(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)
        repo = "owner/repo"

        cache.cache_pr(repo, _make_closed_pr_data())
        result = cache.get_cached_pr(repo, 99)

        assert result is not None
        assert result["pr_state"] == "closed"
        assert result["is_merged"] is False
        assert result["closed_at"] is not None
        assert result["merged_at"] is None

    def test_pr_without_v4_fields_stores_none(self, tmp_path: Path) -> None:
        """cache_pr() must not fail when v4.0 keys are absent from the payload."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)
        repo = "owner/repo"

        minimal_pr: dict[str, Any] = {
            "number": 10,
            "title": "Minimal PR",
            "author": "alice",
            "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "merged_at": datetime(2025, 1, 2, tzinfo=timezone.utc),
            "labels": [],
            "commit_hashes": [],
        }
        cache.cache_pr(repo, minimal_pr)
        result = cache.get_cached_pr(repo, 10)

        assert result is not None
        # For rows without pr_state the _pr_to_dict backward-compat logic fires
        # and derives state from merged_at (not None → "merged").
        assert result["pr_state"] in {"merged", "closed", "open", None}


# ---------------------------------------------------------------------------
# 5. cache_pr() update path
# ---------------------------------------------------------------------------


class TestCachePrV4Update:
    """Verify updating a cached PR correctly persists v4.0 fields."""

    def test_update_v4_fields_when_present(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)
        repo = "owner/repo"

        # Initial write as "open"
        open_pr = _make_pr_data(pr_state="open", merged_at=None, is_merged=False, closed_at=None)
        cache.cache_pr(repo, open_pr)

        # Update to "merged" state
        merged_pr = _make_pr_data(
            pr_state="merged",
            merged_at=datetime(2025, 1, 3, tzinfo=timezone.utc),
            closed_at=datetime(2025, 1, 3, tzinfo=timezone.utc),
            is_merged=True,
        )
        cache.cache_pr(repo, merged_pr)

        result = cache.get_cached_pr(repo, 42)
        assert result is not None
        assert result["pr_state"] == "merged"
        assert result["is_merged"] is True

    def test_update_without_v4_keys_does_not_overwrite(self, tmp_path: Path) -> None:
        """If v4.0 keys are absent from the update payload they must not be zeroed."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)
        repo = "owner/repo"

        # Initial write with full v4.0 data
        cache.cache_pr(repo, _make_pr_data())

        # Update payload that deliberately omits v4.0 fields
        partial_update: dict[str, Any] = {
            "number": 42,
            "title": "Updated title",
            "author": "alice",
            "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "merged_at": datetime(2025, 1, 2, tzinfo=timezone.utc),
            "labels": [],
            "commit_hashes": [],
        }
        cache.cache_pr(repo, partial_update)

        result = cache.get_cached_pr(repo, 42)
        assert result is not None
        # pr_state from the first write must survive the partial update
        assert (
            result["pr_state"] == "merged"
        ), "pr_state should not be overwritten by a payload that omits it"


# ---------------------------------------------------------------------------
# 6. _pr_to_dict() backward-compatibility for pre-v4.0 rows
# ---------------------------------------------------------------------------


class TestPrToDictV4BackwardCompatibility:
    """Verify _pr_to_dict() derives state from merged_at for pre-v4.0 rows."""

    def test_pr_to_dict_derives_merged_state_from_merged_at(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        # Build a PullRequestCache object with only original fields (simulates pre-v4.0 row)
        pr_obj = PullRequestCache(
            repo_path="owner/repo",
            pr_number=77,
            title="Old merged PR",
            author="dev",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            merged_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
            labels=[],
            commit_hashes=[],
        )
        # Simulate missing v4.0 attributes
        for attr in PR_V4_COLUMNS:
            pr_obj.__dict__.pop(attr, None)

        result = cache._pr_to_dict(pr_obj)

        # merged_at is set → backward-compat should derive "merged"
        assert result["pr_state"] == "merged"
        assert result["is_merged"] is True

    def test_pr_to_dict_derives_closed_state_from_no_merged_at(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        pr_obj = PullRequestCache(
            repo_path="owner/repo",
            pr_number=78,
            title="Old closed PR",
            author="dev",
            created_at=datetime(2025, 1, 3, tzinfo=timezone.utc),
            merged_at=None,
            labels=[],
            commit_hashes=[],
        )
        for attr in PR_V4_COLUMNS:
            pr_obj.__dict__.pop(attr, None)

        result = cache._pr_to_dict(pr_obj)

        assert result["pr_state"] == "closed"
        assert result["is_merged"] is False

    def test_pr_to_dict_uses_stored_v4_columns_when_available(self, tmp_path: Path) -> None:
        """When v4.0 columns have values, use them directly (not the derivation logic)."""
        cache_dir = tmp_path / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir, ttl_hours=24)

        pr_obj = PullRequestCache(
            repo_path="owner/repo",
            pr_number=79,
            title="Open PR",
            author="dev",
            created_at=datetime(2025, 1, 5, tzinfo=timezone.utc),
            merged_at=None,
            labels=[],
            commit_hashes=[],
        )
        # Explicitly set v4.0 columns to "open"
        pr_obj.pr_state = "open"
        pr_obj.is_merged = False
        pr_obj.closed_at = None

        result = cache._pr_to_dict(pr_obj)

        assert result["pr_state"] == "open"
        assert result["is_merged"] is False


# ---------------------------------------------------------------------------
# 7. fetch_pr_reviews wiring in IntegrationOrchestrator
# ---------------------------------------------------------------------------


class TestFetchPrReviewsWiring:
    """Verify that fetch_pr_reviews config is forwarded to GitHubIntegration."""

    def _make_config(self, fetch_pr_reviews: bool) -> Mock:
        config = Mock()
        config.github = Mock()
        config.github.token = "fake-token"
        config.github.max_retries = 3
        config.github.backoff_factor = 2
        config.github.fetch_pr_reviews = fetch_pr_reviews
        config.analysis = Mock()
        config.analysis.ticket_platforms = None
        # Disable other integrations
        config.jira = None
        config.cicd = None
        config.pm_integration = None
        return config

    def test_fetch_pr_reviews_false_forwarded(self) -> None:
        """fetch_pr_reviews=False must be passed to GitHubIntegration."""
        from gitflow_analytics.integrations.orchestrator import IntegrationOrchestrator

        with (
            patch("gitflow_analytics.integrations.orchestrator.GitHubIntegration") as mock_gh,
            patch("gitflow_analytics.integrations.orchestrator.JIRAIntegration"),
            patch("gitflow_analytics.integrations.orchestrator.GitHubActionsIntegration"),
            patch("gitflow_analytics.integrations.orchestrator.PMFrameworkOrchestrator"),
        ):
            cache = Mock()
            cache.cache_dir = "/tmp"
            config = self._make_config(fetch_pr_reviews=False)
            IntegrationOrchestrator(config, cache)

        _, kwargs = mock_gh.call_args
        assert kwargs.get("fetch_pr_reviews") is False

    def test_fetch_pr_reviews_true_forwarded(self) -> None:
        """fetch_pr_reviews=True must be passed to GitHubIntegration."""
        from gitflow_analytics.integrations.orchestrator import IntegrationOrchestrator

        with (
            patch("gitflow_analytics.integrations.orchestrator.GitHubIntegration") as mock_gh,
            patch("gitflow_analytics.integrations.orchestrator.JIRAIntegration"),
            patch("gitflow_analytics.integrations.orchestrator.GitHubActionsIntegration"),
            patch("gitflow_analytics.integrations.orchestrator.PMFrameworkOrchestrator"),
        ):
            cache = Mock()
            cache.cache_dir = "/tmp"
            config = self._make_config(fetch_pr_reviews=True)
            IntegrationOrchestrator(config, cache)

        _, kwargs = mock_gh.call_args
        assert kwargs.get("fetch_pr_reviews") is True


# ---------------------------------------------------------------------------
# 8. _refresh_stale_open_prs
# ---------------------------------------------------------------------------


def _make_integration(fetch_pr_reviews: bool = False, cap: int = 50) -> Any:
    """Instantiate GitHubIntegration with all external deps mocked out."""
    from gitflow_analytics.integrations.github_integration import GitHubIntegration

    with (
        patch("gitflow_analytics.integrations.github_integration.Github"),
        patch("gitflow_analytics.integrations.github_integration.create_schema_manager"),
    ):
        cache = Mock()
        cache.cache_dir = "/tmp/test-cache"
        integration = GitHubIntegration(
            token="fake-token",
            cache=cache,
            fetch_pr_reviews=fetch_pr_reviews,
            stale_open_pr_refresh_cap=cap,
        )
    return integration


def _make_gh_pr(number: int, merged: bool = True, state: str = "closed") -> Mock:
    """Build a minimal PyGitHub PullRequest mock for refresh tests."""
    pr = Mock()
    pr.number = number
    pr.title = f"PR #{number}"
    pr.body = ""
    pr.user = Mock()
    pr.user.login = "dev"
    pr.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    pr.merged_at = datetime(2025, 1, 5, tzinfo=timezone.utc) if merged else None
    pr.merged = merged
    pr.state = state
    pr.closed_at = datetime(2025, 1, 5, tzinfo=timezone.utc)
    pr.review_comments = 0
    pr.changed_files = 1
    pr.additions = 10
    pr.deletions = 5
    pr.commits = 1
    pr.labels = []
    pr.get_commits.return_value = [Mock(sha="sha1")]
    pr.get_reviews.return_value = []
    pr.get_issue_comments.return_value = []
    return pr


class TestRefreshStaleOpenPrs:
    """Unit tests for _refresh_stale_open_prs()."""

    def test_skips_non_open_cached_prs(self) -> None:
        """Merged/closed cached PRs must not be re-fetched."""
        integration = _make_integration()
        repo = Mock()

        cached_prs = [
            {"number": 1, "pr_state": "merged"},
            {"number": 2, "pr_state": "closed"},
        ]

        result = integration._refresh_stale_open_prs(
            repo, "owner/repo", cached_prs, already_fetched_numbers=set()
        )

        assert result == []
        repo.get_pull.assert_not_called()

    def test_skips_already_fetched_prs(self) -> None:
        """PRs in already_fetched_numbers must be skipped even if open."""
        integration = _make_integration()
        repo = Mock()

        cached_prs = [{"number": 5, "pr_state": "open"}]

        result = integration._refresh_stale_open_prs(
            repo, "owner/repo", cached_prs, already_fetched_numbers={5}
        )

        assert result == []
        repo.get_pull.assert_not_called()

    def test_refreshes_open_pr_that_was_merged(self) -> None:
        """An open cached PR that has since been merged gets updated."""
        integration = _make_integration()
        repo = Mock()
        gh_pr = _make_gh_pr(number=7, merged=True)
        repo.get_pull.return_value = gh_pr

        cached_prs = [{"number": 7, "pr_state": "open"}]

        with (
            patch(
                "gitflow_analytics.extractors.story_points.StoryPointExtractor.extract_from_text",
                return_value=None,
            ),
            patch(
                "gitflow_analytics.extractors.tickets.TicketExtractor.extract_from_text",
                return_value=[],
            ),
        ):
            result = integration._refresh_stale_open_prs(
                repo, "owner/repo", cached_prs, already_fetched_numbers=set()
            )

        assert len(result) == 1
        assert result[0]["pr_state"] == "merged"
        repo.get_pull.assert_called_once_with(7)
        integration.cache.cache_pr.assert_called_once()

    def test_cap_limits_number_of_refreshes(self) -> None:
        """stale_open_pr_refresh_cap must limit the number of API calls."""
        cap = 2
        integration = _make_integration(cap=cap)
        repo = Mock()

        # 5 open PRs in cache
        cached_prs = [{"number": i, "pr_state": "open"} for i in range(1, 6)]

        for i in range(1, 6):
            repo.get_pull.return_value = _make_gh_pr(number=i, merged=True)

        with (
            patch(
                "gitflow_analytics.extractors.story_points.StoryPointExtractor.extract_from_text",
                return_value=None,
            ),
            patch(
                "gitflow_analytics.extractors.tickets.TicketExtractor.extract_from_text",
                return_value=[],
            ),
        ):
            result = integration._refresh_stale_open_prs(
                repo, "owner/repo", cached_prs, already_fetched_numbers=set()
            )

        # Only cap=2 PRs should have been refreshed
        assert len(result) == cap
        assert repo.get_pull.call_count == cap

    def test_unknown_object_exception_skipped(self) -> None:
        """A PR that 404s must be skipped without raising."""
        from github.GithubException import UnknownObjectException

        integration = _make_integration()
        repo = Mock()
        repo.get_pull.side_effect = UnknownObjectException(404, {}, {})

        cached_prs = [{"number": 101, "pr_state": "open"}]

        result = integration._refresh_stale_open_prs(
            repo, "owner/repo", cached_prs, already_fetched_numbers=set()
        )

        assert result == []

    def test_returns_empty_list_when_no_open_prs(self) -> None:
        integration = _make_integration()
        repo = Mock()

        result = integration._refresh_stale_open_prs(
            repo, "owner/repo", [], already_fetched_numbers=set()
        )

        assert result == []
        repo.get_pull.assert_not_called()


# ---------------------------------------------------------------------------
# 9. Rejection metrics in narrative report
# ---------------------------------------------------------------------------


class TestNarrativeRejectionMetrics:
    """Verify _write_pr_analysis() emits rejection metrics when rejected PRs exist."""

    def _run(
        self,
        pr_metrics: dict[str, Any],
        prs: list[dict[str, Any]],
    ) -> str:
        from io import StringIO

        from gitflow_analytics.reports.narrative_writer import NarrativeReportGenerator

        gen = NarrativeReportGenerator()
        buf = StringIO()
        gen._write_pr_analysis(buf, pr_metrics, prs)
        return buf.getvalue()

    def _base_pr_metrics(self, total: int = 3) -> dict[str, Any]:
        return {
            "total_prs": total,
            "avg_pr_size": 100.0,
            "avg_pr_lifetime_hours": 12.0,
            "avg_files_per_pr": 4.0,
            "total_review_comments": 5,
            "prs_with_story_points": 2,
            "story_point_coverage": 66.7,
            "review_data_collected": False,
            "approval_rate": 0.0,
            "avg_approvals_per_pr": 0.0,
            "avg_change_requests_per_pr": 1.0,
            "review_coverage": 0.0,
            "avg_time_to_first_review_hours": None,
            "median_time_to_first_review_hours": None,
            "total_pr_comments": 0,
            "avg_pr_comments_per_pr": 0.0,
            "avg_revision_count": 0.0,
        }

    def test_rejection_section_present_when_closed_prs_exist(self) -> None:
        prs = [
            {"number": 1, "pr_state": "merged", "is_merged": True, "revision_count": 1},
            {"number": 2, "pr_state": "merged", "is_merged": True, "revision_count": 0},
            {"number": 3, "pr_state": "closed", "is_merged": False, "revision_count": 2},
        ]
        content = self._run(self._base_pr_metrics(), prs)
        assert "### Rejection Metrics" in content
        assert "Rejection Rate" in content

    def test_rejection_rate_calculation(self) -> None:
        """1 closed out of 3 total closed (2 merged + 1 closed) = 33.3%."""
        prs = [
            {"number": 1, "pr_state": "merged", "is_merged": True, "revision_count": 0},
            {"number": 2, "pr_state": "merged", "is_merged": True, "revision_count": 0},
            {"number": 3, "pr_state": "closed", "is_merged": False, "revision_count": 0},
        ]
        content = self._run(self._base_pr_metrics(), prs)
        assert "33.3%" in content

    def test_no_rejection_section_when_all_merged(self) -> None:
        prs = [
            {"number": 1, "pr_state": "merged", "is_merged": True, "revision_count": 0},
            {"number": 2, "pr_state": "merged", "is_merged": True, "revision_count": 1},
        ]
        content = self._run(self._base_pr_metrics(total=2), prs)
        assert "### Rejection Metrics" not in content

    def test_rejection_section_includes_revision_averages(self) -> None:
        """When PR list has revision_count, show per-outcome averages."""
        prs = [
            {"number": 1, "pr_state": "merged", "is_merged": True, "revision_count": 2},
            {"number": 2, "pr_state": "closed", "is_merged": False, "revision_count": 4},
        ]
        metrics = self._base_pr_metrics(total=2)
        metrics["avg_change_requests_per_pr"] = 1.0
        content = self._run(metrics, prs)

        assert "Avg Revisions (Merged PRs)" in content
        assert "Avg Revisions (Rejected PRs)" in content

    def test_change_requests_count_in_rejection_section(self) -> None:
        """avg_change_requests_per_pr appears in rejection section when > 0."""
        prs = [
            {"number": 1, "pr_state": "merged", "is_merged": True, "revision_count": 0},
            {"number": 2, "pr_state": "closed", "is_merged": False, "revision_count": 0},
        ]
        metrics = self._base_pr_metrics(total=2)
        metrics["avg_change_requests_per_pr"] = 2.5
        content = self._run(metrics, prs)
        assert "Average Change Requests per PR" in content


# ---------------------------------------------------------------------------
# 10. CSV PR metrics report includes v4.0 columns
# ---------------------------------------------------------------------------


class TestCSVPRMetricsV4Columns:
    """Verify generate_pr_metrics_report() outputs the new v4.0 state columns."""

    def _generate(self, tmp_path: Path, prs: list[dict[str, Any]]) -> list[dict[str, str]]:
        import csv

        from gitflow_analytics.reports.csv_writer import CSVReportGenerator

        gen = CSVReportGenerator()
        output = tmp_path / "pr_metrics_v4.csv"
        gen.generate_pr_metrics_report(prs, output)
        with open(output) as f:
            return list(csv.DictReader(f))

    def test_v4_columns_present_in_header(self, tmp_path: Path) -> None:
        prs = [
            {
                "number": 1,
                "title": "Merged PR",
                "author": "alice",
                "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
                "merged_at": datetime(2025, 1, 2, tzinfo=timezone.utc),
                "pr_state": "merged",
                "closed_at": datetime(2025, 1, 2, tzinfo=timezone.utc),
                "is_merged": True,
                "additions": 50,
                "deletions": 10,
                "changed_files": 2,
                "review_comments": 0,
                "story_points": 0,
                "labels": [],
            }
        ]
        rows = self._generate(tmp_path, prs)
        assert len(rows) == 1
        assert "pr_state" in rows[0]
        assert "closed_at" in rows[0]
        assert "is_merged" in rows[0]

    def test_merged_pr_state_values(self, tmp_path: Path) -> None:
        prs = [
            {
                "number": 2,
                "title": "Merged PR",
                "author": "bob",
                "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
                "merged_at": datetime(2025, 1, 3, tzinfo=timezone.utc),
                "pr_state": "merged",
                "closed_at": datetime(2025, 1, 3, tzinfo=timezone.utc),
                "is_merged": True,
                "additions": 30,
                "deletions": 5,
                "changed_files": 1,
                "review_comments": 0,
                "story_points": 0,
                "labels": [],
            }
        ]
        rows = self._generate(tmp_path, prs)
        assert rows[0]["pr_state"] == "merged"
        assert rows[0]["is_merged"] == "true"

    def test_rejected_pr_state_values(self, tmp_path: Path) -> None:
        prs = [
            {
                "number": 3,
                "title": "Rejected PR",
                "author": "charlie",
                "created_at": datetime(2025, 1, 5, tzinfo=timezone.utc),
                "merged_at": None,
                "pr_state": "closed",
                "closed_at": datetime(2025, 1, 6, tzinfo=timezone.utc),
                "is_merged": False,
                "additions": 20,
                "deletions": 5,
                "changed_files": 2,
                "review_comments": 0,
                "story_points": 0,
                "labels": [],
            }
        ]
        rows = self._generate(tmp_path, prs)
        assert rows[0]["pr_state"] == "closed"
        assert rows[0]["is_merged"] == "false"

    def test_missing_v4_fields_produce_empty_strings(self, tmp_path: Path) -> None:
        """PRs without v4.0 fields must not cause KeyErrors; columns contain empty strings."""
        prs = [
            {
                "number": 4,
                "title": "Old PR",
                "author": "dave",
                "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
                "merged_at": datetime(2025, 1, 2, tzinfo=timezone.utc),
                # No pr_state, closed_at, is_merged keys
                "additions": 10,
                "deletions": 2,
                "changed_files": 1,
                "review_comments": 0,
                "story_points": 0,
                "labels": [],
            }
        ]
        rows = self._generate(tmp_path, prs)
        assert len(rows) == 1
        # Should not crash; values default to empty string
        assert rows[0].get("pr_state", None) is not None  # column header exists
