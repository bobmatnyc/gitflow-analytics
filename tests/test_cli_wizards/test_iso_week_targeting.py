"""Tests for ISO week targeting flags on gfa classify/collect/report (#70)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from click.testing import CliRunner

from gitflow_analytics.cli_pipeline_commands import (
    classify_command,
    collect_command,
    report_command,
)
from gitflow_analytics.utils.iso_week import iso_week_range, parse_iso_week


class TestParseIsoWeek:
    def test_parse_valid_week(self):
        start, end = parse_iso_week("2026-W07")
        assert start == date(2026, 2, 9)  # Monday of W07 2026
        assert end == date(2026, 2, 15)  # Sunday

    def test_parse_week_01(self):
        start, end = parse_iso_week("2026-W01")
        assert start.weekday() == 0  # Monday
        assert (end - start).days == 6  # 7-day span

    def test_parse_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid ISO week format"):
            parse_iso_week("2026-07")

    def test_parse_invalid_week_number(self):
        with pytest.raises(ValueError):
            parse_iso_week("2026-W54")

    def test_whitespace_stripped(self):
        start, _ = parse_iso_week("  2026-W07  ")
        assert start == date(2026, 2, 9)


class TestIsoWeekRange:
    def test_single_week_range(self):
        start, end = iso_week_range("2026-W07", "2026-W07")
        assert (end - start) == timedelta(days=6)

    def test_multi_week_range(self):
        start, end = iso_week_range("2026-W01", "2026-W04")
        # 4 weeks inclusive = 28-day span = 27 days delta from Mon to Sun
        assert (end - start).days == 27

    def test_reversed_range_raises(self):
        with pytest.raises(ValueError, match="is after"):
            iso_week_range("2026-W10", "2026-W01")


class TestClassifyCommandFlags:
    """Verify CLI-level mutual-exclusivity checks for --week/--from/--to."""

    def _invoke(self, args: list[str], tmp_path):
        # Use a real (empty) file to satisfy click.Path(exists=True). We
        # don't expect ConfigLoader.load to ever be reached because the
        # mutual-exclusivity check raises UsageError beforehand.
        cfg = tmp_path / "config.yaml"
        cfg.write_text("# placeholder\n")
        runner = CliRunner()
        return runner.invoke(classify_command, ["--config", str(cfg), *args])

    def test_week_and_weeks_mutually_exclusive(self, tmp_path):
        result = self._invoke(["--week", "2026-W07", "--weeks", "8"], tmp_path)
        assert result.exit_code != 0
        assert "cannot be combined" in result.output.lower()

    def test_from_without_to_raises(self, tmp_path):
        result = self._invoke(["--from", "2026-W01"], tmp_path)
        assert result.exit_code != 0
        assert "must be used together" in result.output.lower()

    def test_to_without_from_raises(self, tmp_path):
        result = self._invoke(["--to", "2026-W04"], tmp_path)
        assert result.exit_code != 0
        assert "must be used together" in result.output.lower()

    def test_week_and_from_mutually_exclusive(self, tmp_path):
        result = self._invoke(
            ["--week", "2026-W07", "--from", "2026-W01", "--to", "2026-W04"],
            tmp_path,
        )
        assert result.exit_code != 0
        assert "cannot be combined" in result.output.lower()


class TestCollectCommandFlags:
    """Verify CLI-level mutual-exclusivity checks for --week/--from/--to on collect."""

    def _invoke(self, args: list[str], tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("# placeholder\n")
        runner = CliRunner()
        return runner.invoke(collect_command, ["--config", str(cfg), *args])

    def test_week_and_weeks_mutually_exclusive(self, tmp_path):
        result = self._invoke(["--week", "2026-W07", "--weeks", "8"], tmp_path)
        assert result.exit_code != 0
        assert "cannot be combined" in result.output.lower()

    def test_from_without_to_raises(self, tmp_path):
        result = self._invoke(["--from", "2026-W01"], tmp_path)
        assert result.exit_code != 0
        assert "must be used together" in result.output.lower()

    def test_to_without_from_raises(self, tmp_path):
        result = self._invoke(["--to", "2026-W04"], tmp_path)
        assert result.exit_code != 0
        assert "must be used together" in result.output.lower()

    def test_week_and_from_mutually_exclusive(self, tmp_path):
        result = self._invoke(
            ["--week", "2026-W07", "--from", "2026-W01", "--to", "2026-W04"],
            tmp_path,
        )
        assert result.exit_code != 0
        assert "cannot be combined" in result.output.lower()


class TestReportCommandFlags:
    """Verify CLI-level mutual-exclusivity checks for --week/--from/--to on report."""

    def _invoke(self, args: list[str], tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("# placeholder\n")
        runner = CliRunner()
        return runner.invoke(report_command, ["--config", str(cfg), *args])

    def test_week_and_weeks_mutually_exclusive(self, tmp_path):
        result = self._invoke(["--week", "2026-W07", "--weeks", "8"], tmp_path)
        assert result.exit_code != 0
        assert "cannot be combined" in result.output.lower()

    def test_from_without_to_raises(self, tmp_path):
        result = self._invoke(["--from", "2026-W01"], tmp_path)
        assert result.exit_code != 0
        assert "must be used together" in result.output.lower()

    def test_to_without_from_raises(self, tmp_path):
        result = self._invoke(["--to", "2026-W04"], tmp_path)
        assert result.exit_code != 0
        assert "must be used together" in result.output.lower()

    def test_week_and_from_mutually_exclusive(self, tmp_path):
        result = self._invoke(
            ["--week", "2026-W07", "--from", "2026-W01", "--to", "2026-W04"],
            tmp_path,
        )
        assert result.exit_code != 0
        assert "cannot be combined" in result.output.lower()
