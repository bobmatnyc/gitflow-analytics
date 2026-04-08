"""Shared types for the progress display modules.

These are extracted here to avoid circular imports between
progress_display.py, progress_display_rich.py, and progress_display_rich_render.py.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class RepositoryStatus(Enum):
    """Status of repository processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class RepositoryInfo:
    """Information about a repository being processed."""

    name: str
    status: RepositoryStatus = RepositoryStatus.PENDING
    commits: int = 0
    total_commits: int = 0
    developers: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None

    def get_status_icon(self) -> str:
        """Get icon for current status."""
        icons = {
            RepositoryStatus.PENDING: "⏸",
            RepositoryStatus.PROCESSING: "🔄",
            RepositoryStatus.COMPLETE: "✅",
            RepositoryStatus.ERROR: "❌",
            RepositoryStatus.SKIPPED: "⊘",
        }
        return icons.get(self.status, "?")

    def get_status_color(self) -> str:
        """Get color for current status."""
        colors = {
            RepositoryStatus.PENDING: "dim white",
            RepositoryStatus.PROCESSING: "yellow",
            RepositoryStatus.COMPLETE: "green",
            RepositoryStatus.ERROR: "red",
            RepositoryStatus.SKIPPED: "dim yellow",
        }
        return colors.get(self.status, "white")


@dataclass
class ProgressStatistics:
    """Overall progress statistics."""

    total_commits: int = 0
    total_commits_processed: int = 0
    total_developers: int = 0
    total_tickets: int = 0
    total_repositories: int = 0
    processed_repositories: int = 0
    successful_repositories: int = 0
    failed_repositories: int = 0
    skipped_repositories: int = 0
    processing_speed: float = 0.0  # commits per second
    memory_usage: float = 0.0  # MB
    cpu_percent: float = 0.0
    start_time: Optional[datetime] = None
    current_phase: str = "Initializing"

    def get_elapsed_time(self) -> str:
        """Get elapsed time as string."""
        if not self.start_time:
            return "0:00:00"
        elapsed = datetime.now() - self.start_time
        return str(elapsed).split(".")[0]
