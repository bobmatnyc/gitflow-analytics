"""Pipeline stage helpers for the GitFlow Analytics CLI.

These functions implement the three independent stages of the analysis pipeline:

  1. collect  — fetch raw commits from git repositories into the weekly cache
  2. classify — run batch LLM classification on cached commits
  3. report   — read classified commits from the cache and write reports

Both the standalone ``gfa collect / classify / report`` commands and the
all-in-one ``gfa analyze`` command call these same helpers so the logic is
never duplicated.

This module re-exports all public symbols for backward compatibility.
The implementations live in the three stage-specific modules and pipeline_types.py.
"""

from __future__ import annotations

# Re-export dataclasses from pipeline_types (backward compat)
from .pipeline_types import ClassifyResult, CollectResult, ReportResult  # noqa: F401

# Re-export stage functions (backward compat)
from .pipeline_collect import run_collect  # noqa: F401
from .pipeline_classify import run_classify  # noqa: F401
from .pipeline_report import run_report  # noqa: F401

__all__ = [
    "CollectResult",
    "ClassifyResult",
    "ReportResult",
    "run_collect",
    "run_classify",
    "run_report",
]
