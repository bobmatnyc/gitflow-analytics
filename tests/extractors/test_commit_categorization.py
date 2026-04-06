"""Tests for commit message categorization improvements.

Covers:
- Conventional commit scopes: fix(scope):, feat(scope):, chore(scope):, etc.
- Copilot/AI suggestion commits
- PR/code review response commits
- Revert commits
- Broader keyword matching for feature, bug_fix, configuration
- Additional conventional commit prefixes: docs:, refactor:, perf:, ci:
"""

import pytest

from gitflow_analytics.extractors.tickets import TicketExtractor


@pytest.fixture
def extractor() -> TicketExtractor:
    """Default extractor with no special config."""
    return TicketExtractor()


# ---------------------------------------------------------------------------
# 1. Conventional commit scopes (the primary fix)
# ---------------------------------------------------------------------------


class TestConventionalCommitScopes:
    """Conventional commits with scopes like fix(searchlight): must be classified."""

    def test_fix_with_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("fix(searchlight): remove BUP2 references") == "bug_fix"

    def test_feat_with_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("feat(agents): add YAML frontmatter") == "feature"

    def test_chore_with_scope(self, extractor: TicketExtractor) -> None:
        result = extractor.categorize_commit("chore(dead-code): remove stale ManageRatesPage")
        assert result == "chore"

    def test_test_with_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("test(auth): add login integration tests") == "test"

    def test_build_with_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("build(docker): update base image") == "build"

    def test_style_with_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("style(lint): fix eslint warnings") == "style"

    def test_docs_with_scope(self, extractor: TicketExtractor) -> None:
        assert (
            extractor.categorize_commit("docs(readme): update installation guide")
            == "documentation"
        )

    def test_refactor_with_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("refactor(api): simplify handler logic") == "refactor"

    def test_perf_with_scope(self, extractor: TicketExtractor) -> None:
        assert (
            extractor.categorize_commit("perf(queries): optimize N+1 in user list") == "performance"
        )

    def test_ci_with_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("ci(github): add deploy workflow") == "build"

    def test_deploy_with_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("deploy(prod): rollout v2.3.1") == "deployment"

    def test_wip_with_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("wip(dashboard): partial chart component") == "wip"

    def test_version_with_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("bump(deps): update lodash to 4.17.21") == "version"


# ---------------------------------------------------------------------------
# 2. Conventional commits WITHOUT scopes still work (backward compat)
# ---------------------------------------------------------------------------


class TestConventionalCommitsWithoutScope:
    """Ensure plain prefix: patterns still classify correctly."""

    def test_fix_no_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("fix: handle null pointer in user service") == "bug_fix"

    def test_feat_no_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("feat: add user search endpoint") == "feature"

    def test_chore_no_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("chore: update dependencies") == "chore"

    def test_docs_no_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("docs: add API reference") == "documentation"

    def test_refactor_no_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("refactor: extract validation module") == "refactor"

    def test_perf_no_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("perf: cache database queries") == "performance"

    def test_ci_no_scope(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("ci: add linting step to pipeline") == "build"


# ---------------------------------------------------------------------------
# 3. PR / code review / Copilot patterns
# ---------------------------------------------------------------------------


class TestPRAndCopilotPatterns:
    """PR comments, code review responses, and Copilot suggestions."""

    def test_pr_comments(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("PR comments") == "chore"

    def test_pr_comment_singular(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("PR comment") == "chore"

    def test_apply_suggestion_from_copilot(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("Apply suggestion from @Copilot") == "chore"

    def test_apply_suggestion_copilot_lowercase(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("apply suggestion from copilot") == "chore"

    def test_address_review_feedback(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("address review feedback") == "chore"

    def test_address_pr_comments(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("address PR comments") == "chore"

    def test_applied_review_comments(self, extractor: TicketExtractor) -> None:
        assert extractor.categorize_commit("applied review comments") == "chore"


# ---------------------------------------------------------------------------
# 4. Revert commits
# ---------------------------------------------------------------------------


class TestRevertCommits:
    """Revert commits starting with 'Revert' should classify as maintenance."""

    def test_revert_at_start(self, extractor: TicketExtractor) -> None:
        result = extractor.categorize_commit("Revert incorrect global ext variable bumps")
        assert result == "maintenance"

    def test_revert_with_quotes(self, extractor: TicketExtractor) -> None:
        result = extractor.categorize_commit('Revert "add experimental feature flag"')
        assert result == "maintenance"

    def test_revert_simple(self, extractor: TicketExtractor) -> None:
        result = extractor.categorize_commit("Revert last commit")
        assert result == "maintenance"

    def test_revert_merge(self, extractor: TicketExtractor) -> None:
        result = extractor.categorize_commit("revert merge of feature branch")
        assert result == "maintenance"

    def test_reverting_in_body_matches_maintenance(self, extractor: TicketExtractor) -> None:
        """'reverting' keyword in body (not at start) still matches via body patterns."""
        result = extractor.categorize_commit("started reverting the old helper code")
        assert result == "maintenance"


# ---------------------------------------------------------------------------
# 5. Broader feature keyword matching
# ---------------------------------------------------------------------------


class TestBroaderFeatureKeywords:
    """Feature category should catch extraction/scaffolding/wiring work."""

    def test_extracting_harness(self, extractor: TicketExtractor) -> None:
        result = extractor.categorize_commit("start extracting first stage harness")
        assert result == "feature"

    def test_scaffold_component(self, extractor: TicketExtractor) -> None:
        result = extractor.categorize_commit("scaffold the new dashboard component")
        assert result == "feature"

    def test_bootstrap_service(self, extractor: TicketExtractor) -> None:
        result = extractor.categorize_commit("bootstrap payment service skeleton")
        assert result == "feature"

    def test_wire_up_endpoint(self, extractor: TicketExtractor) -> None:
        result = extractor.categorize_commit("wire up the health check endpoint")
        assert result == "feature"

    def test_toggle_feature(self, extractor: TicketExtractor) -> None:
        result = extractor.categorize_commit("toggle dark mode in settings")
        assert result == "feature"

    def test_disable_legacy_flow(self, extractor: TicketExtractor) -> None:
        result = extractor.categorize_commit("disable legacy checkout flow")
        assert result == "feature"


# ---------------------------------------------------------------------------
# 6. Configuration patterns
# ---------------------------------------------------------------------------


class TestConfigurationPatterns:
    """Configuration category should catch ECR, npmrc, GitHub Actions."""

    def test_npmrc_config_update(self, extractor: TicketExtractor) -> None:
        """When message does not contain feature-triggering words like 'add'."""
        result = extractor.categorize_commit("update .npmrc for private registry")
        assert result == "configuration"

    def test_ecr_config(self, extractor: TicketExtractor) -> None:
        """ECR without security-triggering words like 'token'."""
        result = extractor.categorize_commit("ECR registry configuration change")
        assert result == "configuration"

    def test_github_actions_workflow(self, extractor: TicketExtractor) -> None:
        """GitHub Actions matches build (ci/pipeline patterns) due to priority."""
        result = extractor.categorize_commit("update GitHub Actions deploy workflow")
        # "workflow" and "github actions" match build category (ci/pipeline patterns)
        assert result == "build"


# ---------------------------------------------------------------------------
# 7. Bug fix broader patterns
# ---------------------------------------------------------------------------


class TestBugFixBroaderPatterns:
    """Bug fix category catches incorrect/wrong/invalid keywords."""

    def test_incorrect_variable(self, extractor: TicketExtractor) -> None:
        result = extractor.categorize_commit("incorrect global ext variable bumps")
        assert result == "bug_fix"

    def test_wrong_import(self, extractor: TicketExtractor) -> None:
        result = extractor.categorize_commit("wrong import path for utils")
        assert result == "bug_fix"

    def test_invalid_json(self, extractor: TicketExtractor) -> None:
        result = extractor.categorize_commit("invalid JSON response from API")
        assert result == "bug_fix"
