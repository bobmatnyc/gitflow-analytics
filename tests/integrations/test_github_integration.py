"""Tests for GitHubIntegration — review data extraction and PR metrics.

Coverage targets:
- _extract_review_data(): review state parsing, time-to-first-review, comment count
- _count_pr_revisions(): commit-count heuristic
- _extract_pr_data(): base fields + conditional review enrichment
- calculate_pr_metrics(): all aggregated metrics, empty-list edge case
- GitHubConfig.fetch_pr_reviews: schema default and loader round-trip
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_review(
    state: str,
    login: str = "reviewer1",
    submitted_at: datetime | None = None,
) -> Mock:
    """Build a minimal PyGitHub review mock."""
    review = Mock()
    review.state = state
    review.user = Mock()
    review.user.login = login
    review.submitted_at = submitted_at or datetime(2024, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
    return review


def _make_pr(
    number: int = 1,
    title: str = "Test PR",
    body: str = "",
    created_at: datetime | None = None,
    merged_at: datetime | None = None,
    review_comments: int = 3,
    changed_files: int = 5,
    additions: int = 100,
    deletions: int = 20,
    commits: int = 3,
    reviews: list[Any] | None = None,
    issue_comments: list[Any] | None = None,
) -> Mock:
    """Build a minimal PyGitHub PullRequest mock."""
    pr = Mock()
    pr.number = number
    pr.title = title
    pr.body = body
    pr.user = Mock()
    pr.user.login = "author1"
    pr.created_at = created_at or datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    pr.merged_at = merged_at or datetime(2024, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
    pr.merged = True
    pr.review_comments = review_comments
    pr.changed_files = changed_files
    pr.additions = additions
    pr.deletions = deletions
    pr.commits = commits  # integer attribute, not a callable
    pr.labels = []

    # Paginated getters return iterables
    pr.get_commits.return_value = [Mock(sha=f"sha{i}") for i in range(commits)]
    pr.get_reviews.return_value = reviews if reviews is not None else []
    pr.get_issue_comments.return_value = issue_comments if issue_comments is not None else []

    return pr


def _make_integration(fetch_pr_reviews: bool = False) -> Any:
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
        )
    return integration


# ---------------------------------------------------------------------------
# _extract_review_data
# ---------------------------------------------------------------------------


class TestExtractReviewData:
    """Unit tests for _extract_review_data()."""

    def test_single_approval(self) -> None:
        integration = _make_integration()
        # review submitted exactly 54 hours after PR creation (Jan 1 00:00 → Jan 3 06:00)
        review_time = datetime(2024, 1, 3, 6, 0, 0, tzinfo=timezone.utc)
        pr = _make_pr(
            created_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            reviews=[_make_review("APPROVED", "alice", review_time)],
            issue_comments=[],
        )

        result = integration._extract_review_data(pr)

        assert result["approvals_count"] == 1
        assert result["change_requests_count"] == 0
        assert result["reviewers"] == ["alice"]
        assert result["approved_by"] == ["alice"]
        # Jan 1 00:00 → Jan 3 06:00 = 54 hours
        assert result["time_to_first_review_hours"] == pytest.approx(54.0)
        assert result["pr_comments_count"] == 0

    def test_change_request_counted_separately(self) -> None:
        integration = _make_integration()
        review_time = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        pr = _make_pr(
            created_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            reviews=[_make_review("CHANGES_REQUESTED", "bob", review_time)],
            issue_comments=[],
        )

        result = integration._extract_review_data(pr)

        assert result["approvals_count"] == 0
        assert result["change_requests_count"] == 1
        assert result["reviewers"] == ["bob"]
        assert result["approved_by"] == []
        assert result["time_to_first_review_hours"] == pytest.approx(36.0)

    def test_multiple_reviewers_deduplication(self) -> None:
        integration = _make_integration()
        base_time = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        pr = _make_pr(
            created_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            reviews=[
                _make_review("APPROVED", "alice", base_time),
                _make_review("APPROVED", "alice", base_time),  # duplicate
                _make_review("CHANGES_REQUESTED", "bob", base_time),
            ],
            issue_comments=[],
        )

        result = integration._extract_review_data(pr)

        assert result["approvals_count"] == 2  # both reviews counted
        assert len(result["approved_by"]) == 1  # alice deduplicated
        assert set(result["reviewers"]) == {"alice", "bob"}

    def test_commented_review_adds_to_reviewers_not_approvals(self) -> None:
        integration = _make_integration()
        pr = _make_pr(
            reviews=[_make_review("COMMENTED", "charlie")],
            issue_comments=[],
        )

        result = integration._extract_review_data(pr)

        assert result["approvals_count"] == 0
        assert result["change_requests_count"] == 0
        assert "charlie" in result["reviewers"]
        # COMMENTED review should NOT contribute to time_to_first_review
        assert result["time_to_first_review_hours"] is None

    def test_issue_comments_counted(self) -> None:
        integration = _make_integration()
        pr = _make_pr(
            reviews=[],
            issue_comments=[Mock(), Mock(), Mock()],  # 3 comments
        )

        result = integration._extract_review_data(pr)

        assert result["pr_comments_count"] == 3

    def test_reviews_api_exception_gracefully_handled(self) -> None:
        integration = _make_integration()
        pr = _make_pr(issue_comments=[])
        pr.get_reviews.side_effect = Exception("API error")

        # Should not raise — returns zero-state with None for time_to_first_review
        result = integration._extract_review_data(pr)

        assert result["approvals_count"] == 0
        assert result["time_to_first_review_hours"] is None

    def test_issue_comments_api_exception_gracefully_handled(self) -> None:
        integration = _make_integration()
        pr = _make_pr(reviews=[])
        pr.get_issue_comments.side_effect = Exception("API error")

        result = integration._extract_review_data(pr)

        assert result["pr_comments_count"] == 0

    def test_negative_time_clamped_to_zero(self) -> None:
        """Clock skew / backdated reviews should produce 0, not negative hours."""
        integration = _make_integration()
        # Review submitted BEFORE PR creation
        pr = _make_pr(
            created_at=datetime(2024, 1, 5, 0, 0, 0, tzinfo=timezone.utc),
            reviews=[
                _make_review(
                    "APPROVED",
                    "alice",
                    datetime(2024, 1, 4, 0, 0, 0, tzinfo=timezone.utc),  # earlier than created_at
                )
            ],
            issue_comments=[],
        )

        result = integration._extract_review_data(pr)

        assert result["time_to_first_review_hours"] == 0.0

    def test_no_reviews_returns_none_for_time_to_first_review(self) -> None:
        integration = _make_integration()
        pr = _make_pr(reviews=[], issue_comments=[])

        result = integration._extract_review_data(pr)

        assert result["time_to_first_review_hours"] is None


# ---------------------------------------------------------------------------
# _count_pr_revisions
# ---------------------------------------------------------------------------


class TestCountPrRevisions:
    """Unit tests for _count_pr_revisions()."""

    def test_single_commit_returns_zero(self) -> None:
        integration = _make_integration()
        pr = _make_pr(commits=1)

        assert integration._count_pr_revisions(pr) == 0

    def test_multiple_commits_returns_count_minus_one(self) -> None:
        integration = _make_integration()
        pr = _make_pr(commits=5)

        assert integration._count_pr_revisions(pr) == 4

    def test_capped_at_fifty(self) -> None:
        integration = _make_integration()
        pr = _make_pr(commits=200)

        assert integration._count_pr_revisions(pr) == 50

    def test_exception_returns_zero(self) -> None:
        integration = _make_integration()
        pr = Mock()
        pr.commits = property(lambda self: (_ for _ in ()).throw(AttributeError()))
        # Simulate an attribute error when accessing pr.commits
        type(pr).commits = property(lambda self: (_ for _ in ()).throw(AttributeError()))

        assert integration._count_pr_revisions(pr) == 0


# ---------------------------------------------------------------------------
# _extract_pr_data
# ---------------------------------------------------------------------------


class TestExtractPrData:
    """Unit tests for _extract_pr_data()."""

    def test_base_fields_always_present(self) -> None:
        integration = _make_integration(fetch_pr_reviews=False)
        pr = _make_pr(number=42, title="My PR", additions=50, deletions=10)

        # StoryPointExtractor / TicketExtractor are lazy-imported inside _extract_pr_data;
        # patch at the source module level so the import-inside-function picks up the mock.
        with (
            patch(
                "gitflow_analytics.extractors.story_points.StoryPointExtractor.extract_from_text",
                return_value=3,
            ),
            patch(
                "gitflow_analytics.extractors.tickets.TicketExtractor.extract_from_text",
                return_value=[],
            ),
        ):
            result = integration._extract_pr_data(pr, fetch_reviews=False)

        assert result["number"] == 42
        assert result["title"] == "My PR"
        assert result["additions"] == 50
        assert result["deletions"] == 10
        assert result["story_points"] == 3
        # Review fields should NOT be present when fetch_reviews=False
        assert "approvals_count" not in result
        assert "approved_by" not in result

    def test_review_fields_populated_when_fetch_reviews_true(self) -> None:
        integration = _make_integration(fetch_pr_reviews=True)
        review_time = datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc)
        pr = _make_pr(
            created_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            reviews=[_make_review("APPROVED", "reviewer1", review_time)],
            issue_comments=[Mock(), Mock()],
            commits=4,
        )

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
            result = integration._extract_pr_data(pr, fetch_reviews=True)

        assert result["approvals_count"] == 1
        assert result["approved_by"] == ["reviewer1"]
        assert result["pr_comments_count"] == 2
        # Jan 1 00:00 → Jan 3 00:00 = 48 hours
        assert result["time_to_first_review_hours"] == pytest.approx(48.0)
        # 4 commits → revision_count = 3
        assert result["revision_count"] == 3

    def test_integration_uses_instance_fetch_reviews_flag(self) -> None:
        """Verify _extract_pr_data respects the fetch_reviews argument, not the instance flag."""
        integration = _make_integration(fetch_pr_reviews=True)
        pr = _make_pr(reviews=[], issue_comments=[])

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
            # Explicitly disable for this call even though instance has it enabled
            result = integration._extract_pr_data(pr, fetch_reviews=False)

        assert "approvals_count" not in result


# ---------------------------------------------------------------------------
# calculate_pr_metrics
# ---------------------------------------------------------------------------


class TestCalculatePrMetrics:
    """Unit tests for calculate_pr_metrics()."""

    def test_empty_list_returns_zero_metrics(self) -> None:
        integration = _make_integration()

        result = integration.calculate_pr_metrics([])

        assert result["total_prs"] == 0
        assert result["avg_pr_size"] == 0
        assert result["approval_rate"] == 0.0
        assert result["avg_time_to_first_review_hours"] is None

    def test_basic_metrics_without_review_data(self) -> None:
        prs = [
            {
                "number": 1,
                "additions": 100,
                "deletions": 20,
                "changed_files": 5,
                "review_comments": 3,
                "story_points": 5,
                "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "merged_at": datetime(2024, 1, 3, tzinfo=timezone.utc),
            },
            {
                "number": 2,
                "additions": 60,
                "deletions": 10,
                "changed_files": 3,
                "review_comments": 1,
                "story_points": None,
                "created_at": datetime(2024, 1, 5, tzinfo=timezone.utc),
                "merged_at": datetime(2024, 1, 6, tzinfo=timezone.utc),
            },
        ]
        integration = _make_integration()

        result = integration.calculate_pr_metrics(prs)

        assert result["total_prs"] == 2
        # (120 + 70) / 2 = 95
        assert result["avg_pr_size"] == pytest.approx(95.0)
        # avg_pr_lifetime: PR1 = 48h, PR2 = 24h → 36h
        assert result["avg_pr_lifetime_hours"] == pytest.approx(36.0)
        assert result["avg_files_per_pr"] == pytest.approx(4.0)
        assert result["total_review_comments"] == 4
        assert result["prs_with_story_points"] == 1
        assert result["story_point_coverage"] == pytest.approx(50.0)
        # No review data in dicts → approval_rate = 0
        assert result["approval_rate"] == 0.0
        assert result["avg_time_to_first_review_hours"] is None

    def test_approval_rate_with_review_data(self) -> None:
        prs = [
            {"number": 1, "additions": 10, "deletions": 2, "approvals_count": 1, "change_requests_count": 0},
            {"number": 2, "additions": 10, "deletions": 2, "approvals_count": 0, "change_requests_count": 1},
            {"number": 3, "additions": 10, "deletions": 2, "approvals_count": 2, "change_requests_count": 0},
        ]
        integration = _make_integration()

        result = integration.calculate_pr_metrics(prs)

        # 2 out of 3 PRs have approvals_count > 0
        assert result["approval_rate"] == pytest.approx(200 / 3)
        assert result["avg_approvals_per_pr"] == pytest.approx(1.0)
        assert result["avg_change_requests_per_pr"] == pytest.approx(1 / 3)

    def test_review_coverage(self) -> None:
        """PRs with neither approvals nor change requests are not 'reviewed'."""
        prs = [
            {"number": 1, "additions": 10, "deletions": 0, "approvals_count": 1, "change_requests_count": 0},
            {"number": 2, "additions": 10, "deletions": 0, "approvals_count": 0, "change_requests_count": 0},
        ]
        integration = _make_integration()

        result = integration.calculate_pr_metrics(prs)

        # Only 1 of 2 has review activity
        assert result["review_coverage"] == pytest.approx(50.0)

    def test_time_to_first_review_avg_and_median(self) -> None:
        prs = [
            {"number": 1, "additions": 5, "deletions": 1, "time_to_first_review_hours": 10.0},
            {"number": 2, "additions": 5, "deletions": 1, "time_to_first_review_hours": 20.0},
            {"number": 3, "additions": 5, "deletions": 1, "time_to_first_review_hours": 30.0},
        ]
        integration = _make_integration()

        result = integration.calculate_pr_metrics(prs)

        assert result["avg_time_to_first_review_hours"] == pytest.approx(20.0)
        assert result["median_time_to_first_review_hours"] == pytest.approx(20.0)

    def test_median_even_count(self) -> None:
        prs = [
            {"number": 1, "additions": 5, "deletions": 0, "time_to_first_review_hours": 10.0},
            {"number": 2, "additions": 5, "deletions": 0, "time_to_first_review_hours": 30.0},
        ]
        integration = _make_integration()

        result = integration.calculate_pr_metrics(prs)

        assert result["median_time_to_first_review_hours"] == pytest.approx(20.0)

    def test_pr_comments_and_revision_count(self) -> None:
        prs = [
            {"number": 1, "additions": 5, "deletions": 0, "pr_comments_count": 4, "revision_count": 2},
            {"number": 2, "additions": 5, "deletions": 0, "pr_comments_count": 6, "revision_count": 0},
        ]
        integration = _make_integration()

        result = integration.calculate_pr_metrics(prs)

        assert result["total_pr_comments"] == 10
        assert result["avg_pr_comments_per_pr"] == pytest.approx(5.0)
        assert result["avg_revision_count"] == pytest.approx(1.0)

    def test_none_values_treated_as_zero_for_size_fields(self) -> None:
        """Guard against None in additions/deletions from cached rows."""
        prs = [
            {"number": 1, "additions": None, "deletions": None, "changed_files": None},
        ]
        integration = _make_integration()

        result = integration.calculate_pr_metrics(prs)

        assert result["avg_pr_size"] == 0.0
        assert result["avg_files_per_pr"] == 0.0


# ---------------------------------------------------------------------------
# GitHubConfig.fetch_pr_reviews
# ---------------------------------------------------------------------------


class TestGitHubConfigFetchPrReviews:
    """Verify schema default and config-loader plumbing for fetch_pr_reviews."""

    def test_schema_default_is_false(self) -> None:
        from gitflow_analytics.config.schema import GitHubConfig

        cfg = GitHubConfig()
        assert cfg.fetch_pr_reviews is False

    def test_schema_can_be_enabled(self) -> None:
        from gitflow_analytics.config.schema import GitHubConfig

        cfg = GitHubConfig(fetch_pr_reviews=True)
        assert cfg.fetch_pr_reviews is True

    def test_loader_passes_flag_to_config(self) -> None:
        """ConfigLoader._process_github_config() should read fetch_pr_reviews from YAML data."""
        from gitflow_analytics.config.loader import ConfigLoader

        github_data = {
            "token": None,
            "fetch_pr_reviews": True,
        }

        config = ConfigLoader._process_github_config(github_data, config_path=None)

        assert config.fetch_pr_reviews is True

    def test_loader_defaults_to_false_when_key_absent(self) -> None:
        from gitflow_analytics.config.loader import ConfigLoader

        github_data: dict[str, Any] = {"token": None}

        config = ConfigLoader._process_github_config(github_data, config_path=None)

        assert config.fetch_pr_reviews is False


# ---------------------------------------------------------------------------
# Gap 2: Closed/rejected PR tracking (_extract_pr_data)
# ---------------------------------------------------------------------------


class TestClosedPRTracking:
    """Verify _extract_pr_data captures pr_state for closed-without-merge PRs (Gap 2)."""

    def _make_closed_pr(
        self,
        merged: bool = False,
        state: str = "closed",
    ) -> Mock:
        """Build a mock closed-without-merge PR."""
        pr = Mock()
        pr.number = 99
        pr.title = "Rejected PR"
        pr.body = ""
        pr.user = Mock()
        pr.user.login = "contributor"
        pr.created_at = datetime(2024, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
        pr.merged_at = None
        pr.merged = merged
        pr.state = state
        pr.closed_at = datetime(2024, 2, 3, 0, 0, 0, tzinfo=timezone.utc)
        pr.review_comments = 0
        pr.changed_files = 2
        pr.additions = 15
        pr.deletions = 5
        pr.commits = 1
        pr.labels = []
        pr.get_commits.return_value = [Mock(sha="abc")]
        pr.get_reviews.return_value = []
        pr.get_issue_comments.return_value = []
        return pr

    def test_merged_pr_state_is_merged(self) -> None:
        """A successfully merged PR has pr_state='merged'."""
        integration = _make_integration(fetch_pr_reviews=False)
        pr = _make_pr(merged_at=datetime(2024, 1, 10, tzinfo=timezone.utc))
        pr.merged = True
        pr.state = "closed"

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
            result = integration._extract_pr_data(pr, fetch_reviews=False)

        assert result["pr_state"] == "merged"
        assert result["is_merged"] is True

    def test_closed_without_merge_state_is_closed(self) -> None:
        """A PR closed without merging has pr_state='closed'."""
        integration = _make_integration(fetch_pr_reviews=False)
        pr = self._make_closed_pr(merged=False, state="closed")

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
            result = integration._extract_pr_data(pr, fetch_reviews=False)

        assert result["pr_state"] == "closed"
        assert result["is_merged"] is False

    def test_closed_pr_has_closed_at(self) -> None:
        """A closed PR exposes closed_at so downstream can compute lifecycle time."""
        integration = _make_integration(fetch_pr_reviews=False)
        pr = self._make_closed_pr()

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
            result = integration._extract_pr_data(pr, fetch_reviews=False)

        assert result.get("closed_at") is not None


# ---------------------------------------------------------------------------
# Gap 5: Normalise GitHub login to lowercase (_extract_pr_data)
# ---------------------------------------------------------------------------


class TestGitHubLoginNormalisation:
    """Verify that GitHub usernames are stored in lowercase for consistency (Gap 5)."""

    def test_author_login_lowercased(self) -> None:
        """PR author login is always stored as lowercase in the data dict."""
        integration = _make_integration(fetch_pr_reviews=False)
        pr = _make_pr()
        pr.user.login = "UPPERCASE_USER"

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
            result = integration._extract_pr_data(pr, fetch_reviews=False)

        assert result["author"] == "uppercase_user"

    def test_reviewer_login_lowercased(self) -> None:
        """Reviewer logins in _extract_review_data are always lowercased."""
        integration = _make_integration(fetch_pr_reviews=True)
        review_time = datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc)
        pr = _make_pr(
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            reviews=[_make_review("APPROVED", "MixedCase_Reviewer", review_time)],
            issue_comments=[],
        )

        result = integration._extract_review_data(pr)

        assert "mixedcase_reviewer" in result["reviewers"]
        assert "MixedCase_Reviewer" not in result["reviewers"]

    def test_approved_by_login_lowercased(self) -> None:
        """Approved-by logins are stored in lowercase."""
        integration = _make_integration(fetch_pr_reviews=True)
        review_time = datetime(2024, 1, 2, tzinfo=timezone.utc)
        pr = _make_pr(
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            reviews=[_make_review("APPROVED", "Bob_Approver", review_time)],
            issue_comments=[],
        )

        result = integration._extract_review_data(pr)

        assert "bob_approver" in result["approved_by"]
