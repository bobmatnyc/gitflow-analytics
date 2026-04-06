"""Tests for configurable ticket detection (GitHub Issue #20).

Covers:
- Exclude-pattern filtering (CVE, CWE, date-like strings)
- Position anchoring ("start" vs "anywhere")
- Default config produces same behaviour as old hard-coded defaults
- TicketDetectionConfig round-trips through the YAML config loader
- should_skip_commit() for each commit_filter mode
"""

import tempfile
from pathlib import Path

import pytest  # noqa: F401

from gitflow_analytics.config.schema import TicketDetectionConfig
from gitflow_analytics.extractors.tickets import TicketExtractor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_extractor(
    *,
    commit_filter: str = "all",
    target_branches: list[str] | None = None,
    position: str = "anywhere",
    patterns: dict[str, str] | None = None,
    exclude_patterns: list[str] | None = None,
    allowed_platforms: list[str] | None = None,
) -> TicketExtractor:
    """Build a TicketExtractor wired with a TicketDetectionConfig."""
    cfg = TicketDetectionConfig(
        commit_filter=commit_filter,
        target_branches=target_branches or ["develop", "main", "master"],
        position=position,
        patterns=patterns or {"jira": r"([A-Z]{2,10}-\d+)"},
        exclude_patterns=exclude_patterns or [],
    )
    return TicketExtractor(
        allowed_platforms=allowed_platforms,
        ticket_detection_config=cfg,
    )


# ---------------------------------------------------------------------------
# 1. Default config is backward-compatible
# ---------------------------------------------------------------------------


class TestDefaultConfigBackwardCompatibility:
    """Default TicketDetectionConfig must behave identically to the old defaults."""

    def test_no_config_extracts_jira(self):
        """Extractor with no config still extracts JIRA references."""
        extractor = TicketExtractor()
        tickets = extractor.extract_from_text("Implements PROJ-123 feature")
        ids = [t["id"] for t in tickets]
        assert "PROJ-123" in ids

    def test_no_config_no_excludes(self):
        """Extractor with no config does NOT filter anything by default."""
        extractor = TicketExtractor()
        # CVE-style string that happens to look like a JIRA ticket
        # The built-in default has NO exclude patterns, so it won't be filtered.
        extractor.extract_from_text("CVE-2026")
        # With no config, built-in excludes list is empty → no filtering
        assert extractor._compiled_excludes == []

    def test_default_config_no_excludes(self):
        """TicketDetectionConfig with default exclude_patterns has the cve/cwe excludes."""
        cfg = TicketDetectionConfig()
        # Defaults include CVE-\\d+, CWE-\\d+, \\d{8,}
        assert any("CVE" in p for p in cfg.exclude_patterns)
        assert any("CWE" in p for p in cfg.exclude_patterns)

    def test_no_config_position_is_anywhere(self):
        """Default (no config) searches full message, not just start."""
        extractor = TicketExtractor()
        # Ticket is not at start of message
        tickets = extractor.extract_from_text("Fix some things, see PROJ-999 for details")
        ids = [t["id"] for t in tickets]
        assert "PROJ-999" in ids

    def test_should_skip_commit_all_always_false(self):
        """should_skip_commit returns False for commit_filter='all'."""
        extractor = TicketExtractor()  # default filter = all
        assert extractor.should_skip_commit({"parents": [], "branch": "main"}) is False
        assert extractor.should_skip_commit({"parents": ["abc"], "branch": "develop"}) is False


# ---------------------------------------------------------------------------
# 2. Exclude-pattern filtering
# ---------------------------------------------------------------------------


class TestExcludePatterns:
    """Matches hitting exclude_patterns must be removed from results."""

    def test_cve_excluded(self):
        """CVE-NNNN-NNNNN should not appear in results when excluded."""
        extractor = make_extractor(
            patterns={"jira": r"(CVE-\d+)"},
            exclude_patterns=[r"CVE-\d+"],
        )
        tickets = extractor.extract_from_text("Security patch for CVE-2026-12345")
        assert tickets == []

    def test_cwe_excluded(self):
        """CWE-NNN should not appear in results when excluded."""
        extractor = make_extractor(
            patterns={"jira": r"(CWE-\d+)"},
            exclude_patterns=[r"CWE-\d+"],
        )
        tickets = extractor.extract_from_text("Related to CWE-770 buffer overflow")
        assert tickets == []

    def test_long_digit_string_excluded(self):
        """8+-digit numeric string should not appear when exclude pattern matches.

        The extractor captures only the digit part (e.g. "20260330"); the
        fullmatch of ``\\d{8,}`` should remove it.  We restrict to the "jira"
        platform to avoid noise from the built-in "linear" pattern.
        """
        extractor = make_extractor(
            patterns={"jira": r"[A-Z]+-(\d{8,})"},  # captures only the digit part
            exclude_patterns=[r"\d{8,}"],
            allowed_platforms=["jira"],
        )
        tickets = extractor.extract_from_text("UPDATE-20260330 scheduled")
        assert tickets == []

    def test_non_excluded_ticket_passes_through(self):
        """Legitimate JIRA tickets NOT matching exclude patterns are kept."""
        extractor = make_extractor(
            patterns={"jira": r"([A-Z]{2,10}-\d+)"},
            exclude_patterns=[r"CVE-\d+", r"CWE-\d+", r"\d{8,}"],
        )
        tickets = extractor.extract_from_text("Fixes PROJ-123 and adds CVE-2026 workaround")
        ids = [t["id"] for t in tickets]
        assert "PROJ-123" in ids
        # CVE-2026 matches the jira pattern capturing "CVE-2026"; fullmatch of CVE-\d+ → excluded
        assert "CVE-2026" not in ids

    def test_multiple_excludes_independent(self):
        """Each exclude pattern is checked independently."""
        extractor = make_extractor(
            patterns={"jira": r"([A-Z]{2,10}-\d+)"},
            exclude_patterns=[r"CVE-\d+", r"SKIP-\d+"],
        )
        text = "See REAL-1, CVE-2025, SKIP-9 and KEEP-42"
        tickets = extractor.extract_from_text(text)
        ids = [t["id"] for t in tickets]
        assert "REAL-1" in ids
        assert "KEEP-42" in ids
        assert "CVE-2025" not in ids
        assert "SKIP-9" not in ids

    def test_empty_exclude_list_keeps_everything(self):
        """When exclude_patterns is empty, no tickets are filtered."""
        extractor = make_extractor(
            patterns={"jira": r"([A-Z]{2,10}-\d+)"},
            exclude_patterns=[],
        )
        tickets = extractor.extract_from_text("CVE-2026 and PROJ-1")
        ids = [t["id"] for t in tickets]
        assert "CVE-2026" in ids
        assert "PROJ-1" in ids


# ---------------------------------------------------------------------------
# 3. Position anchoring
# ---------------------------------------------------------------------------


class TestPositionAnchoring:
    """position='start' should only match at the beginning of the message."""

    def test_position_start_matches_at_beginning(self):
        extractor = make_extractor(
            patterns={"jira": r"([A-Z]{2,10}-\d+)"},
            position="start",
        )
        tickets = extractor.extract_from_text("PROJ-123 implement new feature")
        ids = [t["id"] for t in tickets]
        assert "PROJ-123" in ids

    def test_position_start_does_not_match_mid_message(self):
        extractor = make_extractor(
            patterns={"jira": r"([A-Z]{2,10}-\d+)"},
            position="start",
        )
        tickets = extractor.extract_from_text("Implement new feature for PROJ-123")
        ids = [t["id"] for t in tickets]
        assert "PROJ-123" not in ids

    def test_position_anywhere_matches_mid_message(self):
        extractor = make_extractor(
            patterns={"jira": r"([A-Z]{2,10}-\d+)"},
            position="anywhere",
        )
        tickets = extractor.extract_from_text("Implement new feature for PROJ-123")
        ids = [t["id"] for t in tickets]
        assert "PROJ-123" in ids

    def test_position_start_empty_message_returns_empty(self):
        extractor = make_extractor(position="start")
        assert extractor.extract_from_text("") == []


# ---------------------------------------------------------------------------
# 4. should_skip_commit
# ---------------------------------------------------------------------------


class TestShouldSkipCommit:
    """should_skip_commit() respects commit_filter modes."""

    # --- commit_filter = "all" ---

    def test_all_never_skips_single_parent(self):
        extractor = make_extractor(commit_filter="all")
        commit = {"parents": ["abc"], "branch": "main"}
        assert extractor.should_skip_commit(commit) is False

    def test_all_never_skips_merge_commit(self):
        extractor = make_extractor(commit_filter="all")
        commit = {"parents": ["abc", "def"], "branch": "feature/foo"}
        assert extractor.should_skip_commit(commit) is False

    def test_all_never_skips_root_commit(self):
        extractor = make_extractor(commit_filter="all")
        commit = {"parents": [], "branch": "main"}
        assert extractor.should_skip_commit(commit) is False

    # --- commit_filter = "squash_merges_only" ---

    def test_squash_single_parent_on_target_not_skipped(self):
        extractor = make_extractor(
            commit_filter="squash_merges_only",
            target_branches=["main", "develop"],
        )
        commit = {"parents": ["abc"], "branch": "main"}
        assert extractor.should_skip_commit(commit) is False

    def test_squash_single_parent_off_target_is_skipped(self):
        extractor = make_extractor(
            commit_filter="squash_merges_only",
            target_branches=["main", "develop"],
        )
        commit = {"parents": ["abc"], "branch": "feature/wip"}
        assert extractor.should_skip_commit(commit) is True

    def test_squash_merge_commit_two_parents_is_skipped(self):
        """A true merge commit (2 parents) is NOT a squash merge."""
        extractor = make_extractor(
            commit_filter="squash_merges_only",
            target_branches=["main"],
        )
        commit = {"parents": ["abc", "def"], "branch": "main"}
        assert extractor.should_skip_commit(commit) is True

    def test_squash_root_commit_is_skipped(self):
        """Root commit (0 parents) on target branch is NOT a squash merge."""
        extractor = make_extractor(
            commit_filter="squash_merges_only",
            target_branches=["main"],
        )
        commit = {"parents": [], "branch": "main"}
        assert extractor.should_skip_commit(commit) is True

    # --- commit_filter = "merge_commits" ---

    def test_merge_two_parents_not_skipped(self):
        extractor = make_extractor(commit_filter="merge_commits")
        commit = {"parents": ["abc", "def"], "branch": "main"}
        assert extractor.should_skip_commit(commit) is False

    def test_merge_single_parent_is_skipped(self):
        extractor = make_extractor(commit_filter="merge_commits")
        commit = {"parents": ["abc"], "branch": "main"}
        assert extractor.should_skip_commit(commit) is True

    def test_merge_root_commit_is_skipped(self):
        extractor = make_extractor(commit_filter="merge_commits")
        commit = {"parents": [], "branch": "main"}
        assert extractor.should_skip_commit(commit) is True

    # --- unknown filter value ---

    def test_unknown_filter_does_not_skip(self):
        extractor = make_extractor(commit_filter="unknown_value")
        commit = {"parents": ["abc"], "branch": "main"}
        assert extractor.should_skip_commit(commit) is False

    # --- object-style commit (not dict) ---

    def test_object_commit_supported(self):
        """should_skip_commit also works with object-style commits."""

        class FakeCommit:
            parents = ["abc"]
            branch = "main"

        extractor = make_extractor(
            commit_filter="squash_merges_only",
            target_branches=["main"],
        )
        assert extractor.should_skip_commit(FakeCommit()) is False


# ---------------------------------------------------------------------------
# 5. Config loader round-trip
# ---------------------------------------------------------------------------


class TestConfigLoaderRoundTrip:
    """TicketDetectionConfig must deserialise correctly from YAML."""

    def _load_analysis_config(self, yaml_body: str):
        """Helper: write a minimal YAML config and load it."""
        from gitflow_analytics.config import ConfigLoader

        full_yaml = f"""version: "1.0"
repositories:
  - name: "test-repo"
    path: "/tmp/nonexistent-repo-for-testing"
{yaml_body}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(full_yaml)
            temp_path = Path(f.name)

        try:
            cfg = ConfigLoader.load(temp_path)
            return cfg.analysis
        finally:
            temp_path.unlink()

    def test_no_ticket_detection_section_gives_defaults(self):
        """Without ticket_detection in YAML, defaults are used."""
        analysis = self._load_analysis_config("analysis: {}")
        td = analysis.ticket_detection
        assert isinstance(td, TicketDetectionConfig)
        assert td.commit_filter == "all"
        assert td.position == "anywhere"
        assert "main" in td.target_branches

    def test_commit_filter_loaded(self):
        analysis = self._load_analysis_config(
            "analysis:\n  ticket_detection:\n    commit_filter: squash_merges_only\n"
        )
        assert analysis.ticket_detection.commit_filter == "squash_merges_only"

    def test_position_loaded(self):
        analysis = self._load_analysis_config(
            "analysis:\n  ticket_detection:\n    position: start\n"
        )
        assert analysis.ticket_detection.position == "start"

    def test_target_branches_loaded(self):
        analysis = self._load_analysis_config(
            "analysis:\n  ticket_detection:\n    target_branches: [main, release]\n"
        )
        assert "main" in analysis.ticket_detection.target_branches
        assert "release" in analysis.ticket_detection.target_branches

    def test_exclude_patterns_loaded(self):
        analysis = self._load_analysis_config(
            'analysis:\n  ticket_detection:\n    exclude_patterns:\n      - "CVE-\\\\d+"\n'
        )
        assert any("CVE" in p for p in analysis.ticket_detection.exclude_patterns)

    def test_custom_patterns_loaded(self):
        analysis = self._load_analysis_config(
            'analysis:\n  ticket_detection:\n    patterns:\n      jira: "([A-Z]{2,5}-\\\\d+)"\n'
        )
        assert "jira" in analysis.ticket_detection.patterns

    def test_full_ticket_detection_block(self):
        """All sub-keys are parsed correctly when provided together."""
        yaml_body = """analysis:
  ticket_detection:
    commit_filter: "squash_merges_only"
    target_branches: ["develop", "main"]
    position: "start"
    patterns:
      jira: "([A-Z]{2,10}-\\\\d+)"
    exclude_patterns:
      - "CVE-\\\\d+"
      - "CWE-\\\\d+"
      - "\\\\d{8,}"
"""
        analysis = self._load_analysis_config(yaml_body)
        td = analysis.ticket_detection
        assert td.commit_filter == "squash_merges_only"
        assert td.position == "start"
        assert set(td.target_branches) == {"develop", "main"}
        assert "jira" in td.patterns
        assert len(td.exclude_patterns) == 3
