"""Progress display for GitFlow Analytics.

Contains shared types, SimpleProgressDisplay, and the create_progress_display factory.
Rich display is in progress_display_rich.py.
"""

"""
Rich-based progress display for GitFlow Analytics.

This module provides a sophisticated progress meter using the Rich library
for beautiful terminal output with live updates and statistics.
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Try to import psutil, but make it optional
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from rich import box
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False



from .progress_display_rich import RichProgressDisplay

# Shared types
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
            RepositoryStatus.PENDING: "⏸",  # More visible pending icon
            RepositoryStatus.PROCESSING: "🔄",  # Clearer processing icon
            RepositoryStatus.COMPLETE: "✅",  # Green checkmark
            RepositoryStatus.ERROR: "❌",  # Red X
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



class SimpleProgressDisplay:
    """Fallback progress display using tqdm when Rich is not available."""

    def __init__(self, version: str = "1.3.11", update_frequency: float = 0.5):
        """Initialize simple progress display."""
        from tqdm import tqdm

        self.tqdm = tqdm
        self.version = version
        self.overall_progress = None
        self.repo_progress = None
        self.repositories = {}
        self.statistics = ProgressStatistics()

    def start(self, total_items: int = 100, description: str = "Analyzing repositories"):
        """Start progress display."""
        self.overall_progress = self.tqdm(
            total=total_items,
            desc=description,
            unit="items",
        )
        self.statistics.start_time = datetime.now()

    def stop(self):
        """Stop progress display."""
        if self.overall_progress:
            self.overall_progress.close()
        if self.repo_progress:
            self.repo_progress.close()

    def update_overall(self, completed: int, description: Optional[str] = None):
        """Update overall progress."""
        if self.overall_progress:
            self.overall_progress.n = completed
            if description:
                self.overall_progress.set_description(description)
            self.overall_progress.refresh()

    def start_repository(self, repo_name: str, total_commits: int = 0):
        """Start processing a repository."""
        if self.repo_progress:
            self.repo_progress.close()

        self.repositories[repo_name] = RepositoryInfo(
            name=repo_name,
            status=RepositoryStatus.PROCESSING,
            total_commits=total_commits,
            start_time=datetime.now(),
        )

        # Enhanced description to show what's happening
        action = "Analyzing" if total_commits > 0 else "Fetching"
        desc = f"{action} repository: {repo_name}"

        self.repo_progress = self.tqdm(
            total=total_commits if total_commits > 0 else 100,
            desc=desc,
            unit="commits",
            leave=False,
        )

    def update_repository(self, repo_name: str, commits: int, speed: float = 0.0):
        """Update repository progress."""
        if self.repo_progress and repo_name in self.repositories:
            self.repo_progress.n = commits
            self.repo_progress.set_postfix(speed=f"{speed:.1f} c/s")
            self.repo_progress.refresh()
            self.repositories[repo_name].commits = commits

    def finish_repository(
        self, repo_name: str, success: bool = True, error_message: Optional[str] = None
    ):
        """Finish processing a repository."""
        if repo_name in self.repositories:
            repo_info = self.repositories[repo_name]
            repo_info.status = RepositoryStatus.COMPLETE if success else RepositoryStatus.ERROR
            repo_info.error_message = error_message
            if repo_info.start_time:
                repo_info.processing_time = (datetime.now() - repo_info.start_time).total_seconds()

        if self.repo_progress:
            self.repo_progress.close()
            self.repo_progress = None

    def update_statistics(self, **kwargs):
        """Update statistics."""
        for key, value in kwargs.items():
            if hasattr(self.statistics, key):
                setattr(self.statistics, key, value)

    def initialize_repositories(self, repository_list: list):
        """Initialize all repositories with their status.

        Args:
            repository_list: List of repositories to be processed.
        """
        # Pre-populate all repositories with their status
        for repo in repository_list:
            repo_name = repo.get("name", "Unknown")
            status_str = repo.get("status", "pending")

            # Map status string to enum
            status_map = {
                "pending": RepositoryStatus.PENDING,
                "complete": RepositoryStatus.COMPLETE,
                "processing": RepositoryStatus.PROCESSING,
                "error": RepositoryStatus.ERROR,
                "skipped": RepositoryStatus.SKIPPED,
            }
            status = status_map.get(status_str.lower(), RepositoryStatus.PENDING)

            if repo_name not in self.repositories:
                self.repositories[repo_name] = RepositoryInfo(
                    name=repo_name,
                    status=status,
                )
            else:
                # Update existing status if needed
                self.repositories[repo_name].status = status
        self.statistics.total_repositories = len(self.repositories)

    def set_phase(self, phase: str):
        """Set the current processing phase."""
        self.statistics.current_phase = phase
        if self.overall_progress:
            self.overall_progress.set_description(f"{phase}")

    @contextmanager
    def progress_context(self, total_items: int = 100, description: str = "Processing"):
        """Context manager for progress display."""
        try:
            self.start(total_items, description)
            yield self
        finally:
            self.stop()

    # Compatibility methods for CLI interface
    def show_header(self):
        """Display header - compatibility method for CLI."""
        print(f"\n{'=' * 60}")
        print(f"GitFlow Analytics v{self.version}")
        print(f"{'=' * 60}\n")

    def start_live_display(self):
        """Start live display - compatibility wrapper for start()."""
        if not self.overall_progress:
            self.start(total_items=100, description="Processing")

    def stop_live_display(self):
        """Stop live display - compatibility wrapper for stop()."""
        self.stop()

    def add_progress_task(self, task_id: str, description: str, total: int):
        """Add a progress task - compatibility method."""
        # Store task information for later use
        if not hasattr(self, "_tasks"):
            self._tasks = {}
        self._tasks[task_id] = {"description": description, "total": total, "progress": None}

        if task_id == "repos":
            # Update overall progress
            if self.overall_progress:
                self.overall_progress.total = total
                self.overall_progress.set_description(description)
        elif task_id == "qualitative":
            # For qualitative, we might create a separate progress bar
            from tqdm import tqdm

            self._tasks[task_id]["progress"] = tqdm(
                total=total, desc=description, unit="items", leave=False
            )

    def update_progress_task(
        self,
        task_id: str,
        description: Optional[str] = None,
        advance: int = 0,
        completed: Optional[int] = None,
    ):
        """Update a progress task - compatibility method."""
        if task_id == "repos" and self.overall_progress:
            if description:
                self.overall_progress.set_description(description)
            if advance:
                self.overall_progress.update(advance)
            if completed is not None:
                self.overall_progress.n = completed
                self.overall_progress.refresh()
        elif hasattr(self, "_tasks") and task_id in self._tasks:
            task = self._tasks[task_id].get("progress")
            if task:
                if description:
                    task.set_description(description)
                if advance:
                    task.update(advance)
                if completed is not None:
                    task.n = completed
                    task.refresh()

    def complete_progress_task(self, task_id: str, description: str):
        """Complete a progress task - compatibility method."""
        if task_id == "repos" and self.overall_progress:
            self.overall_progress.set_description(description)
            self.overall_progress.n = self.overall_progress.total
            self.overall_progress.refresh()
        elif hasattr(self, "_tasks") and task_id in self._tasks:
            task = self._tasks[task_id].get("progress")
            if task:
                task.set_description(description)
                task.n = task.total
                task.close()
                self._tasks[task_id]["progress"] = None

    def print_status(self, message: str, style: str = "info"):
        """Print a status message - compatibility method."""
        # Simple console print with basic styling
        prefix = {"info": "ℹ️ ", "success": "✅ ", "warning": "⚠️ ", "error": "❌ "}.get(style, "")
        print(f"{prefix}{message}")

    def show_configuration_status(
        self,
        config_file,
        github_org=None,
        github_token_valid=False,
        jira_configured=False,
        jira_valid=False,
        analysis_weeks=4,
        **kwargs,
    ):
        """Display configuration status in simple format."""
        print("\n=== Configuration ===")
        print(f"Config File: {config_file}")

        if github_org:
            print(f"GitHub Organization: {github_org}")
            status = "✓ Valid" if github_token_valid else "✗ No token"
            print(f"GitHub Token: {status}")

        if jira_configured:
            status = "✓ Valid" if jira_valid else "✗ Invalid"
            print(f"JIRA Integration: {status}")

        print(f"Analysis Period: {analysis_weeks} weeks")

        # Add any additional kwargs passed
        for key, value in kwargs.items():
            formatted_key = key.replace("_", " ").title()
            print(f"{formatted_key}: {value}")

        print("==================\n")

    def show_repository_discovery(self, repositories):
        """Display discovered repositories in simple format."""
        print("\n📚 === Discovered Repositories ===")
        for idx, repo in enumerate(repositories, 1):
            name = repo.get("name", "Unknown")
            status = repo.get("status", "Ready")
            github_repo = repo.get("github_repo", "")

            # Format the output line
            if github_repo:
                print(f"  {idx:2}. {name:30} {status:12} ({github_repo})")
            else:
                print(f"  {idx:2}. {name:30} {status}")
        print(f"\nTotal repositories: {len(repositories)}")
        print("============================\n")

    def show_error(self, message: str, show_debug_hint: bool = True):
        """Display an error message in simple format."""
        print(f"\n❌ ERROR: {message}")
        if show_debug_hint:
            print("Tip: Set GITFLOW_DEBUG=1 for more detailed output")
        print("")

    def show_warning(self, message: str):
        """Display a warning message in simple format."""
        print(f"\n⚠️  WARNING: {message}\n")

    def show_qualitative_stats(self, stats):
        """Display qualitative analysis statistics in simple format."""
        print("\n=== Qualitative Analysis Statistics ===")
        if isinstance(stats, dict):
            for key, value in stats.items():
                formatted_key = key.replace("_", " ").title()
                print(f"  {formatted_key}: {value}")
        print("=====================================\n")

    def show_analysis_summary(self, commits, developers, tickets, prs=None, untracked=None):
        """Display analysis summary in simple format."""
        print("\n=== Analysis Summary ===")
        print(f"  Total Commits: {commits}")
        print(f"  Unique Developers: {developers}")
        print(f"  Tracked Tickets: {tickets}")
        if prs is not None:
            print(f"  Pull Requests: {prs}")
        if untracked is not None:
            print(f"  Untracked Commits: {untracked}")
        print("======================\n")

    def show_dora_metrics(self, metrics):
        """Display DORA metrics in simple format."""
        if not metrics:
            return

        print("\n=== DORA Metrics ===")
        metric_names = {
            "deployment_frequency": "Deployment Frequency",
            "lead_time_for_changes": "Lead Time for Changes",
            "mean_time_to_recovery": "Mean Time to Recovery",
            "change_failure_rate": "Change Failure Rate",
        }

        for key, name in metric_names.items():
            if key in metrics:
                value = metrics[key].get("value", "N/A")
                rating = metrics[key].get("rating", "")
                print(f"  {name}: {value} {f'({rating})' if rating else ''}")
        print("==================\n")

    def show_reports_generated(self, output_dir, reports):
        """Display generated reports information in simple format."""
        print(f"\n=== Reports Generated in {output_dir} ===")
        for report in reports:
            if isinstance(report, dict):
                report_type = report.get("type", "Unknown")
                filename = report.get("filename", "N/A")
                print(f"  {report_type}: {filename}")
            else:
                print(f"  Report: {report}")
        print("=====================================\n")

    def show_llm_cost_summary(self, cost_stats):
        """Display LLM cost summary in simple format."""
        if not cost_stats:
            return

        print("\n=== LLM Usage & Cost Summary ===")
        if isinstance(cost_stats, dict):
            for model, stats in cost_stats.items():
                requests = stats.get("requests", 0)
                tokens = stats.get("tokens", 0)
                cost = stats.get("cost", 0.0)
                print(f"  {model}:")
                print(f"    Requests: {requests}")
                print(f"    Tokens: {tokens}")
                print(f"    Cost: ${cost:.4f}")
        print("==============================\n")


def create_progress_display(
    style: str = "auto", version: str = "1.3.11", update_frequency: float = 0.5
) -> Any:
    """
    Create a progress display based on configuration.

    Args:
        style: Display style ("rich", "simple", or "auto")
        version: GitFlow Analytics version
        update_frequency: Update frequency in seconds

    Returns:
        Progress display instance
    """
    if style == "rich" or (style == "auto" and RICH_AVAILABLE):
        try:
            return RichProgressDisplay(version, update_frequency)
        except Exception as e:
            logger.debug(
                f"Non-critical: Rich progress display unavailable, falling back to simple: {e}"
            )

    return SimpleProgressDisplay(version, update_frequency)
