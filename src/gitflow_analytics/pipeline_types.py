"""Shared result dataclasses for the GitFlow Analytics pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class CollectResult:
    """Outcome of the collect stage."""

    total_commits: int = 0
    total_tickets: int = 0
    total_developers: int = 0
    repos_fetched: int = 0
    repos_cached: int = 0
    repos_failed: int = 0
    start_date: datetime | None = None
    end_date: datetime | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class ClassifyResult:
    """Outcome of the classify stage."""

    processed_batches: int = 0
    total_commits: int = 0
    skipped_batches: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class ReportResult:
    """Outcome of the report stage."""

    generated_reports: list[str] = field(default_factory=list)
    output_dir: Path | None = None
    errors: list[str] = field(default_factory=list)
