"""Tests for src/gitflow_analytics/reports/quality_report.py (Issue #26)."""

import json
from pathlib import Path

import pytest

from gitflow_analytics.reports.quality_report import (
    QualityReportGenerator,
    _is_revert,
)

# ---------------------------------------------------------------------------
# _is_revert
# ---------------------------------------------------------------------------


class TestIsRevert:
    def test_revert_prefix(self):
        assert _is_revert("Revert 'add user service'") is True

    def test_reverts_prefix(self):
        assert _is_revert("Reverts commit abc123") is True

    def test_rollback_in_message(self):
        assert _is_revert("Emergency rollback of payment service") is True

    def test_undo_commit(self):
        assert _is_revert("Undo commit from last night") is True

    def test_undo_merge(self):
        assert _is_revert("undo merge that broke staging") is True

    def test_undo_pr(self):
        assert _is_revert("undo PR #123") is True

    def test_normal_feature_commit(self):
        assert _is_revert("feat: add OAuth login") is False

    def test_normal_fix_commit(self):
        assert _is_revert("fix: correct null pointer in user service") is False

    def test_empty_string(self):
        assert _is_revert("") is False

    def test_none_message(self):
        assert _is_revert(None) is False  # type: ignore[arg-type]

    def test_reverse_is_not_revert(self):
        # "reverse" must not trigger the revert pattern
        assert _is_revert("reverse the sort order in results") is False

    def test_irreversible_is_not_revert(self):
        assert _is_revert("mark action as irreversible") is False

    def test_case_insensitive(self):
        assert _is_revert("REVERT: bad deploy") is True


# ---------------------------------------------------------------------------
# _commit_quality_metrics
# ---------------------------------------------------------------------------


class TestCommitQualityMetrics:
    def setup_method(self):
        self.gen = QualityReportGenerator()

    def _make_commits(self, messages: list[str]) -> list[dict]:
        return [{"commit_hash": f"hash{i}", "message": msg} for i, msg in enumerate(messages)]

    def test_empty_commits_returns_zero_metrics(self):
        result = self.gen._commit_quality_metrics([], {})
        assert result["total_commits"] == 0
        assert result["revert_commits"] == 0
        assert result["revert_rate"] == 0.0
        assert result["avg_complexity"] is None

    def test_revert_count(self):
        commits = self._make_commits(["Revert add service", "feat: new thing", "Rollback deploy"])
        result = self.gen._commit_quality_metrics(commits, {})
        assert result["revert_commits"] == 2
        assert result["revert_rate"] == pytest.approx(2 / 3, rel=1e-3)

    def test_risk_level_counting(self):
        commits = self._make_commits(["feat A", "feat B", "feat C", "feat D"])
        qual = {
            "hash0": {"risk_level": "high"},
            "hash1": {"risk_level": "medium"},
            "hash2": {"risk_level": "low"},
            # hash3 has no qualitative data
        }
        result = self.gen._commit_quality_metrics(commits, qual)
        assert result["high_risk_commits"] == 1
        assert result["medium_risk_commits"] == 1
        assert result["low_risk_commits"] == 1
        assert result["high_risk_ratio"] == pytest.approx(1 / 4, rel=1e-3)

    def test_avg_complexity_computed(self):
        commits = self._make_commits(["A", "B"])
        qual = {
            "hash0": {"risk_level": "low", "complexity": 2},
            "hash1": {"risk_level": "low", "complexity": 4},
        }
        result = self.gen._commit_quality_metrics(commits, qual)
        assert result["avg_complexity"] == pytest.approx(3.0, rel=1e-3)

    def test_avg_complexity_none_when_no_qualitative(self):
        commits = self._make_commits(["A", "B"])
        result = self.gen._commit_quality_metrics(commits, {})
        assert result["avg_complexity"] is None

    def test_invalid_complexity_value_is_skipped(self):
        commits = self._make_commits(["A"])
        qual = {"hash0": {"risk_level": "low", "complexity": "not-a-number"}}
        result = self.gen._commit_quality_metrics(commits, qual)
        assert result["avg_complexity"] is None


# ---------------------------------------------------------------------------
# _pr_quality_metrics
# ---------------------------------------------------------------------------


class TestPRQualityMetrics:
    def setup_method(self):
        self.gen = QualityReportGenerator()

    def test_empty_prs_returns_zero_metrics(self):
        result = self.gen._pr_quality_metrics([])
        assert result["prs_total"] == 0
        assert result["avg_revision_count"] == 0.0
        assert result["change_request_rate"] == 0.0
        assert result["approval_rate"] == 0.0

    def test_change_request_rate(self):
        prs = [
            {"revision_count": 1, "change_requests_count": 2, "approvals_count": 1},
            {"revision_count": 0, "change_requests_count": 0, "approvals_count": 1},
        ]
        result = self.gen._pr_quality_metrics(prs)
        assert result["change_request_rate"] == pytest.approx(0.5, rel=1e-3)

    def test_approval_rate_first_pass(self):
        prs = [
            # approved on first pass
            {"revision_count": 0, "change_requests_count": 0, "approvals_count": 2},
            # approved but had change requests first
            {"revision_count": 1, "change_requests_count": 1, "approvals_count": 1},
            # no approvals
            {"revision_count": 0, "change_requests_count": 0, "approvals_count": 0},
        ]
        result = self.gen._pr_quality_metrics(prs)
        # Only first PR qualifies
        assert result["approval_rate"] == pytest.approx(1 / 3, rel=1e-3)

    def test_avg_revision_count(self):
        prs = [
            {"revision_count": 2, "change_requests_count": 0, "approvals_count": 1},
            {"revision_count": 4, "change_requests_count": 0, "approvals_count": 1},
        ]
        result = self.gen._pr_quality_metrics(prs)
        assert result["avg_revision_count"] == pytest.approx(3.0, rel=1e-3)

    def test_none_revision_count_treated_as_zero(self):
        prs = [{"revision_count": None, "change_requests_count": 0, "approvals_count": 0}]
        result = self.gen._pr_quality_metrics(prs)
        assert result["avg_revision_count"] == 0.0


# ---------------------------------------------------------------------------
# _quality_score
# ---------------------------------------------------------------------------


class TestQualityScore:
    def setup_method(self):
        self.gen = QualityReportGenerator()

    def test_perfect_score_is_one(self):
        cm = {
            "total_commits": 10,
            "revert_rate": 0.0,
            "high_risk_ratio": 0.0,
        }
        pm = {"avg_revision_count": 0.0}
        assert self.gen._quality_score(cm, pm) == pytest.approx(1.0, rel=1e-3)

    def test_zero_commits_returns_none(self):
        cm = {"total_commits": 0, "revert_rate": 0.0, "high_risk_ratio": 0.0}
        pm = {"avg_revision_count": 0.0}
        assert self.gen._quality_score(cm, pm) is None

    def test_high_revert_rate_lowers_score(self):
        cm = {"total_commits": 5, "revert_rate": 0.5, "high_risk_ratio": 0.0}
        pm = {"avg_revision_count": 0.0}
        score = self.gen._quality_score(cm, pm)
        assert score is not None
        assert score == pytest.approx(0.5, rel=1e-3)

    def test_avg_revision_capped_at_one(self):
        """avg_revision_count >= 5 should cap the revision factor to 0."""
        cm = {"total_commits": 5, "revert_rate": 0.0, "high_risk_ratio": 0.0}
        pm = {"avg_revision_count": 10.0}
        assert self.gen._quality_score(cm, pm) == pytest.approx(0.0, abs=1e-6)

    def test_composite_formula(self):
        cm = {"total_commits": 10, "revert_rate": 0.1, "high_risk_ratio": 0.2}
        pm = {"avg_revision_count": 2.5}
        expected = (1 - 0.1) * (1 - 0.2) * (1 - min(2.5 / 5.0, 1.0))
        score = self.gen._quality_score(cm, pm)
        assert score == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# _by_developer
# ---------------------------------------------------------------------------


class TestByDeveloper:
    def setup_method(self):
        self.gen = QualityReportGenerator()

    def test_groups_commits_by_author_email(self):
        commits = [
            {"commit_hash": "a1", "message": "feat A", "author_email": "alice@example.com"},
            {"commit_hash": "b1", "message": "feat B", "author_email": "bob@example.com"},
            {"commit_hash": "a2", "message": "Revert feat A", "author_email": "alice@example.com"},
        ]
        result = self.gen._by_developer(commits, {}, [])
        assert "alice@example.com" in result
        assert "bob@example.com" in result
        assert result["alice@example.com"]["total_commits"] == 2
        assert result["bob@example.com"]["total_commits"] == 1

    def test_groups_prs_by_author(self):
        prs = [
            {
                "author": "alice",
                "revision_count": 1,
                "change_requests_count": 0,
                "approvals_count": 1,
            },
            {
                "author": "bob",
                "revision_count": 0,
                "change_requests_count": 1,
                "approvals_count": 0,
            },
        ]
        result = self.gen._by_developer([], {}, prs)
        assert "alice" in result
        assert "bob" in result
        assert result["alice"]["prs_total"] == 1
        assert result["bob"]["change_request_rate"] == pytest.approx(1.0, rel=1e-3)

    def test_unknown_author_fallback(self):
        commits = [
            {"commit_hash": "x", "message": "feat", "author_email": None, "author_name": None}
        ]
        result = self.gen._by_developer(commits, {}, [])
        assert "unknown" in result

    def test_quality_score_present_per_dev(self):
        commits = [{"commit_hash": "h1", "message": "feat", "author_email": "dev@x.com"}]
        result = self.gen._by_developer(commits, {}, [])
        assert "quality_score" in result["dev@x.com"]


# ---------------------------------------------------------------------------
# generate (integration)
# ---------------------------------------------------------------------------


class TestGenerate:
    def setup_method(self):
        self.gen = QualityReportGenerator()

    def test_writes_quality_summary_json(self, tmp_path: Path):
        commits = [
            {"commit_hash": "c1", "message": "feat: login", "author_email": "a@b.com"},
            {"commit_hash": "c2", "message": "Revert login", "author_email": "a@b.com"},
        ]
        prs = [
            {
                "author": "a@b.com",
                "revision_count": 1,
                "change_requests_count": 0,
                "approvals_count": 1,
            },
        ]
        self.gen.generate(commits, [], prs, tmp_path)
        out = tmp_path / "quality_summary.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["total_commits_analyzed"] == 2
        assert data["total_prs_analyzed"] == 1
        assert "org_level" in data
        assert "per_developer" in data

    def test_empty_inputs_produce_valid_json(self, tmp_path: Path):
        self.gen.generate([], [], [], tmp_path)
        out = tmp_path / "quality_summary.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["total_commits_analyzed"] == 0
        assert data["org_level"]["quality_score"] is None

    def test_qualitative_data_joined_via_hash(self, tmp_path: Path):
        commits = [
            {"commit_hash": "abc", "message": "feat", "author_email": "d@e.com"},
        ]
        qual = [{"commit_hash": "abc", "risk_level": "high", "complexity": 5}]
        self.gen.generate(commits, qual, [], tmp_path)
        data = json.loads((tmp_path / "quality_summary.json").read_text())
        assert data["org_level"]["high_risk_commits"] == 1
        assert data["org_level"]["avg_complexity"] == 5.0

    def test_returns_summary_dict(self, tmp_path: Path):
        result = self.gen.generate([], [], [], tmp_path)
        assert isinstance(result, dict)
        assert "generated_at" in result

    def test_creates_output_dir_if_missing(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        self.gen.generate([], [], [], nested)
        assert (nested / "quality_summary.json").exists()
