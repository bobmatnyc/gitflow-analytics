"""Rich-based progress display - public API."""

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



from .progress_display_rich_render import RichDisplayRenderMixin


class RichProgressDisplay(RichDisplayRenderMixin):
    """Rich terminal progress display with live updates and statistics."""


    def __init__(self, version: str = "1.3.11", update_frequency: float = 0.25):
        """
        Initialize the progress display.

        Args:
            version: Version of GitFlow Analytics
            update_frequency: How often to update display in seconds (default 0.25 for smooth updates)
        """
        if not RICH_AVAILABLE:
            raise ImportError("Rich library is not available. Install with: pip install rich")

        self.version = version
        self.update_frequency = update_frequency
        # Force terminal mode to ensure Rich works even when output is piped
        self.console = Console(force_terminal=True)

        # Progress tracking with enhanced styling
        # Don't start the progress bars - they'll be rendered inside Live
        self.overall_progress = Progress(
            SpinnerColumn(style="bold cyan"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, style="cyan", complete_style="green"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            transient=False,
        )

        self.repo_progress = Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(bar_width=30, style="yellow", complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.fields[speed]:.1f} commits/s"),
            transient=False,
        )

        # Data tracking
        self.repositories: dict[str, RepositoryInfo] = {}
        self.statistics = ProgressStatistics()
        self.current_repo: Optional[str] = None

        # Task IDs
        self.overall_task_id = None
        self.repo_task_id = None

        # Thread safety
        self._lock = threading.Lock()
        self._live = None
        self._layout = None
        self._update_counter = 0  # For tracking updates

        # System monitoring (only if psutil is available)
        self._process = psutil.Process() if PSUTIL_AVAILABLE else None

    def start(self, total_items: int = 100, description: str = "Analyzing repositories"):
        """
        Start the progress display with full-screen live updates.

        Args:
            total_items: Total number of items to process
            description: Description of the overall task
        """
        # Initialize statistics and progress without holding lock
        self.statistics.start_time = datetime.now()
        self.statistics.total_repositories = total_items
        self.overall_task_id = self.overall_progress.add_task(description, total=total_items)

        # Create a simpler layout that doesn't embed Progress objects
        self._layout = self._create_simple_layout()

        # Create and start Live display without holding any locks
        try:
            self._live = Live(
                self._layout,
                console=self.console,
                refresh_per_second=2,
                screen=True,  # Full screen mode
                auto_refresh=True,
            )
            self._live.start()
            # Rich's auto_refresh will handle periodic updates
        except Exception:
            # Fallback to simple display if Live fails
            self._live = None
            self.console.print(
                "[yellow]Note: Using simple progress display (Rich Live unavailable)[/yellow]"
            )
            self.console.print(Panel(f"GitFlow Analytics - {description}", title="Progress"))

    def stop(self):
        """Stop the progress display."""
        with self._lock:
            if self._live:
                try:
                    self._live.stop()
                except Exception as e:
                    logger.debug(
                        f"Non-critical: error stopping Rich live display during cleanup: {e}"
                    )
                finally:
                    self._live = None
                    self._layout = None

    def update_overall(self, completed: int, description: Optional[str] = None):
        """Update overall progress."""
        with self._lock:
            if self.overall_task_id is not None:
                update_kwargs = {"completed": completed}
                if description:
                    update_kwargs["description"] = description
                self.overall_progress.update(self.overall_task_id, **update_kwargs)

            # Update the display with new content
            self._update_all_panels()

    def start_repository(self, repo_name: str, total_commits: int = 0):
        """Start processing a repository with immediate visual feedback."""
        with self._lock:
            self.current_repo = repo_name

            if repo_name not in self.repositories:
                self.repositories[repo_name] = RepositoryInfo(name=repo_name)

            repo_info = self.repositories[repo_name]
            repo_info.status = RepositoryStatus.PROCESSING
            repo_info.total_commits = total_commits
            repo_info.start_time = datetime.now()

            # Create or update repo progress task
            if self.repo_task_id is not None:
                self.repo_progress.remove_task(self.repo_task_id)

            self.repo_task_id = self.repo_progress.add_task(
                repo_name,
                total=total_commits if total_commits > 0 else 100,
                speed=0.0,
            )

            # Immediately update all panels to show the change
            self._update_all_panels()

    def update_repository(self, repo_name: str, commits: int, speed: float = 0.0):
        """Update repository progress with continuous visual feedback."""
        with self._lock:
            if repo_name not in self.repositories:
                return

            repo_info = self.repositories[repo_name]
            repo_info.commits = commits

            if self.repo_task_id is not None and repo_name == self.current_repo:
                self.repo_progress.update(
                    self.repo_task_id,
                    completed=commits,
                    speed=speed,
                )

            # Update overall statistics
            self.statistics.processing_speed = speed

            # Update total commits across all repos
            self.statistics.total_commits = sum(r.commits for r in self.repositories.values())

            # Force update all panels every time for continuous visual feedback
            self._update_all_panels()

    def finish_repository(
        self, repo_name: str, success: bool = True, error_message: Optional[str] = None
    ):
        """Finish processing a repository with immediate status update."""
        with self._lock:
            if repo_name not in self.repositories:
                return

            repo_info = self.repositories[repo_name]
            repo_info.status = RepositoryStatus.COMPLETE if success else RepositoryStatus.ERROR
            repo_info.error_message = error_message

            if repo_info.start_time:
                repo_info.processing_time = (datetime.now() - repo_info.start_time).total_seconds()

            self.statistics.processed_repositories += 1

            # Immediately clear current repo if it was this one
            if self.current_repo == repo_name:
                self.current_repo = None
                if self.repo_task_id is not None:
                    self.repo_progress.remove_task(self.repo_task_id)
                    self.repo_task_id = None

            # Force immediate update to show completion
            self._update_all_panels()

    def update_statistics(self, **kwargs):
        """
        Update statistics.

        Args:
            **kwargs: Statistics to update (total_commits, total_developers, etc.)
        """
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.statistics, key):
                    setattr(self.statistics, key, value)

            if self._layout:
                self._layout["stats"].update(self._create_statistics_panel())

    def initialize_repositories(self, repository_list: list):
        """Initialize all repositories with pending status and show them immediately.

        Args:
            repository_list: List of repositories to be processed.
                            Each item should have 'name' and optionally 'status' fields.
        """
        with self._lock:
            # Pre-populate all repositories with their status
            for _idx, repo in enumerate(repository_list):
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

            # Update statistics
            self.statistics.total_repositories = len(self.repositories)

            # Set initial phase
            if not self.statistics.current_phase or self.statistics.current_phase == "Initializing":
                self.statistics.current_phase = (
                    f"Ready to process {len(self.repositories)} repositories"
                )

            # Force immediate update to show all repositories
            self._update_all_panels()

    def set_phase(self, phase: str):
        """Set the current processing phase with immediate display update."""
        with self._lock:
            self.statistics.current_phase = phase
            # Force immediate update to show phase change
            self._update_all_panels()

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
        # The header is shown when start() is called, so we just need to print it
        header_panel = self._create_header_panel()
        self.console.print(header_panel)

    def add_progress_task(self, task_id: str, description: str, total: int):
        """Add a progress task - compatibility method."""
        if task_id == "repos" or task_id == "main":
            # Handle both "repos" and "main" for overall progress
            if not self._live:
                # Not in live mode, just print
                self.console.print(f"[cyan]{description}[/cyan] (0/{total})")
                return
            # If Live display not started yet, start it now
            if not self._live:
                # Don't clear console - let Rich Live handle the screen management
                self.start(total_items=total, description=description)
            else:
                # Update the existing overall progress description and total
                if self.overall_task_id is not None:
                    self.overall_progress.update(
                        self.overall_task_id, description=description, total=total
                    )
        elif task_id == "qualitative":
            # Create a new task for qualitative analysis
            with self._lock:
                # Store task IDs in a dictionary for tracking
                if not hasattr(self, "_task_ids"):
                    self._task_ids = {}
                # Only add task if overall_progress is available
                if self._live:
                    self._task_ids[task_id] = self.overall_progress.add_task(
                        description, total=total
                    )

    def update_progress_task(
        self,
        task_id: str,
        description: Optional[str] = None,
        advance: int = 0,
        completed: Optional[int] = None,
    ):
        """Update a progress task - compatibility method."""
        # Handle simple mode
        if self._live == "simple" and description:
            self.console.print(f"[cyan]→ {description}[/cyan]")
            return
        if task_id == "repos" or task_id == "main":
            # Update overall progress (handle both "repos" and "main" for compatibility)
            if description:
                self.update_overall(completed or 0, description)
            elif advance and self.overall_task_id is not None:
                self.overall_progress.advance(self.overall_task_id, advance)
        elif hasattr(self, "_task_ids") and task_id in self._task_ids:
            # Update specific task
            update_kwargs = {}
            if description:
                update_kwargs["description"] = description
            if completed is not None:
                update_kwargs["completed"] = completed
            if advance:
                self.overall_progress.advance(self._task_ids[task_id], advance)
            elif update_kwargs:
                self.overall_progress.update(self._task_ids[task_id], **update_kwargs)

    def complete_progress_task(self, task_id: str, description: str):
        """Complete a progress task - compatibility method."""
        if task_id == "repos":
            # Mark overall task as complete
            if self.overall_task_id is not None:
                total = self.overall_progress.tasks[0].total if self.overall_progress.tasks else 100
                self.overall_progress.update(
                    self.overall_task_id, description=description, completed=total
                )
        elif hasattr(self, "_task_ids") and task_id in self._task_ids:
            # Complete specific task
            task = None
            for t in self.overall_progress.tasks:
                if t.id == self._task_ids[task_id]:
                    task = t
                    break
            if task:
                self.overall_progress.update(
                    self._task_ids[task_id], description=description, completed=task.total
                )

    def print_status(self, message: str, style: str = "info"):
        """Print a status message - compatibility method."""
        styles = {"info": "cyan", "success": "green", "warning": "yellow", "error": "red"}
        self.console.print(
            f"[{styles.get(style, 'white')}]{message}[/{styles.get(style, 'white')}]"
        )

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
        """Display configuration status in a Rich format."""
        table = Table(title="Configuration", box=box.ROUNDED)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Config File", str(config_file))

        if github_org:
            table.add_row("GitHub Organization", github_org)
            status = "✓ Valid" if github_token_valid else "✗ No token"
            table.add_row("GitHub Token", status)

        if jira_configured:
            status = "✓ Valid" if jira_valid else "✗ Invalid"
            table.add_row("JIRA Integration", status)

        table.add_row("Analysis Period", f"{analysis_weeks} weeks")

        # Add any additional kwargs passed
        for key, value in kwargs.items():
            formatted_key = key.replace("_", " ").title()
            table.add_row(formatted_key, str(value))

        self.console.print(table)

    def show_repository_discovery(self, repositories):
        """Display discovered repositories in a Rich format."""
        table = Table(
            title="📚 Discovered Repositories", box=box.ROUNDED, show_lines=True, highlight=True
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Repository", style="bold cyan", no_wrap=False)
        table.add_column("Status", style="green", width=12)
        table.add_column("GitHub", style="dim white", no_wrap=False)

        for idx, repo in enumerate(repositories, 1):
            name = repo.get("name", "Unknown")
            status = repo.get("status", "Ready")
            github_repo = repo.get("github_repo", "")

            # Style the status based on its value
            if "Local" in status or "exists" in status.lower():
                status_style = "[green]" + status + "[/green]"
            elif "Remote" in status or "clone" in status.lower():
                status_style = "[yellow]" + status + "[/yellow]"
            else:
                status_style = status

            table.add_row(str(idx), name, status_style, github_repo or "")

        self.console.print(table)
        self.console.print(f"\n[dim]Total repositories: {len(repositories)}[/dim]\n")

    def show_error(self, message: str, show_debug_hint: bool = True):
        """Display an error message in Rich format."""
        error_panel = Panel(
            Text(message, style="red"), title="[red]Error[/red]", border_style="red", padding=(1, 2)
        )
        self.console.print(error_panel)

        if show_debug_hint:
            self.console.print("[dim]Tip: Set GITFLOW_DEBUG=1 for more detailed output[/dim]")

    def show_warning(self, message: str):
        """Display a warning message in Rich format."""
        warning_panel = Panel(
            Text(message, style="yellow"),
            title="[yellow]Warning[/yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
        self.console.print(warning_panel)

    def show_qualitative_stats(self, stats):
        """Display qualitative analysis statistics in Rich format."""
        table = Table(title="Qualitative Analysis Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        if isinstance(stats, dict):
            for key, value in stats.items():
                # Format the key to be more readable
                formatted_key = key.replace("_", " ").title()
                formatted_value = str(value)
                table.add_row(formatted_key, formatted_value)

        self.console.print(table)

    def show_analysis_summary(self, commits, developers, tickets, prs=None, untracked=None):
        """Display analysis summary in Rich format."""
        summary = Table(title="Analysis Summary", box=box.ROUNDED)
        summary.add_column("Metric", style="cyan", width=30)
        summary.add_column("Count", style="green", width=20)

        summary.add_row("Total Commits", str(commits))
        summary.add_row("Unique Developers", str(developers))
        summary.add_row("Tracked Tickets", str(tickets))

        if prs is not None:
            summary.add_row("Pull Requests", str(prs))

        if untracked is not None:
            summary.add_row("Untracked Commits", str(untracked))

        self.console.print(summary)

    def show_dora_metrics(self, metrics):
        """Display DORA metrics in Rich format."""
        if not metrics:
            return

        table = Table(title="DORA Metrics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Rating", style="green")

        # Format and display each DORA metric
        metric_names = {
            "deployment_frequency": "Deployment Frequency",
            "lead_time_for_changes": "Lead Time for Changes",
            "mean_time_to_recovery": "Mean Time to Recovery",
            "change_failure_rate": "Change Failure Rate",
        }

        for key, name in metric_names.items():
            if key in metrics:
                metric_data = metrics[key]
                # Handle both dict format (with 'value' key) and direct float values
                if isinstance(metric_data, dict):
                    value = metric_data.get("value", "N/A")
                    rating = metric_data.get("rating", "")
                else:
                    # Direct value (float, int, string)
                    value = metric_data
                    rating = ""
                table.add_row(name, str(value), rating)

        self.console.print(table)

    def show_reports_generated(self, output_dir, reports):
        """Display generated reports information in Rich format."""
        table = Table(title=f"Reports Generated in {output_dir}", box=box.ROUNDED)
        table.add_column("Report Type", style="cyan")
        table.add_column("Filename", style="white")

        for report in reports:
            if isinstance(report, dict):
                report_type = report.get("type", "Unknown")
                filename = report.get("filename", "N/A")
            else:
                # Handle simple string format
                report_type = "Report"
                filename = str(report)

            table.add_row(report_type, filename)

        self.console.print(table)

    def show_llm_cost_summary(self, cost_stats):
        """Display LLM cost summary in Rich format."""
        if not cost_stats:
            return

        table = Table(title="LLM Usage & Cost Summary", box=box.ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Requests", style="white")
        table.add_column("Tokens", style="white")
        table.add_column("Cost", style="green")

        if isinstance(cost_stats, dict):
            for model, stats in cost_stats.items():
                requests = stats.get("requests", 0)
                tokens = stats.get("tokens", 0)
                cost = stats.get("cost", 0.0)
                table.add_row(model, str(requests), str(tokens), f"${cost:.4f}")

        self.console.print(table)

    def start_live_display(self):
        """Start live display - compatibility wrapper for start()."""
        if not self.overall_task_id:
            self.start(total_items=100, description="Processing")

    def stop_live_display(self):
        """Stop live display - compatibility wrapper for stop()."""
        self.stop()


