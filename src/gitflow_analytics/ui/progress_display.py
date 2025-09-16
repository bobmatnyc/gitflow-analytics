"""
Rich-based progress display for GitFlow Analytics.

This module provides a sophisticated progress meter using the Rich library
for beautiful terminal output with live updates and statistics.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import psutil

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        TimeRemainingColumn,
        SpinnerColumn,
        MofNCompleteColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.columns import Columns
    from rich.align import Align

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


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
            RepositoryStatus.PENDING: "◌",
            RepositoryStatus.PROCESSING: "⟳",
            RepositoryStatus.COMPLETE: "✓",
            RepositoryStatus.ERROR: "✗",
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
    total_developers: int = 0
    total_tickets: int = 0
    total_repositories: int = 0
    processed_repositories: int = 0
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
        return str(elapsed).split('.')[0]


class RichProgressDisplay:
    """Rich-based progress display for GitFlow Analytics."""

    def __init__(self, version: str = "1.3.11", update_frequency: float = 0.5):
        """
        Initialize the progress display.

        Args:
            version: Version of GitFlow Analytics
            update_frequency: How often to update display in seconds
        """
        if not RICH_AVAILABLE:
            raise ImportError("Rich library is not available. Install with: pip install rich")

        self.version = version
        self.update_frequency = update_frequency
        self.console = Console()

        # Progress tracking
        self.overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )

        self.repo_progress = Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.fields[speed]:.1f} commits/s"),
            console=self.console,
            transient=False,
        )

        # Data tracking
        self.repositories: Dict[str, RepositoryInfo] = {}
        self.statistics = ProgressStatistics()
        self.current_repo: Optional[str] = None

        # Task IDs
        self.overall_task_id = None
        self.repo_task_id = None

        # Thread safety
        self._lock = threading.Lock()
        self._live = None
        self._layout = None

        # System monitoring
        self._process = psutil.Process()

    def _create_header_panel(self) -> Panel:
        """Create the header panel with title and version."""
        title = Text(f"GitFlow Analytics v{self.version}", style="bold cyan", justify="center")
        return Panel(
            title,
            box=box.DOUBLE,
            padding=(0, 1),
            style="bright_blue",
        )

    def _create_progress_panel(self) -> Panel:
        """Create the main progress panel."""
        content_items = []

        # Overall progress
        content_items.append(Text("Overall Progress", style="bold"))
        content_items.append(self.overall_progress)

        # Current repository progress
        if self.current_repo:
            content_items.append(Text())  # Empty line
            repo_info = self.repositories.get(self.current_repo)
            if repo_info:
                current_text = Text(f"Current: {repo_info.name}", style="cyan")
                if repo_info.total_commits > 0:
                    current_text.append(f" ({repo_info.commits}/{repo_info.total_commits})")
                content_items.append(current_text)
                content_items.append(self.repo_progress)

        # Combine all items
        content = Columns(content_items, expand=True, padding=(1, 2))

        return Panel(
            content,
            title="[bold]Progress[/bold]",
            box=box.ROUNDED,
            padding=(1, 2),
        )

    def _create_repository_table(self) -> Panel:
        """Create the repository status table."""
        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.SIMPLE_HEAD,
            expand=True,
            show_lines=False,
        )

        table.add_column("Repository", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center", width=12)
        table.add_column("Commits", justify="right", width=10)
        table.add_column("Time", justify="right", width=8)
        table.add_column("Result", width=10)

        # Sort repositories: processing first, then complete, then pending
        sorted_repos = sorted(
            self.repositories.values(),
            key=lambda r: (
                r.status != RepositoryStatus.PROCESSING,
                r.status != RepositoryStatus.COMPLETE,
                r.status != RepositoryStatus.ERROR,
                r.name
            )
        )

        # Show only the most recent/relevant repositories (limit to 10)
        for repo in sorted_repos[:10]:
            status_text = Text(
                f"{repo.get_status_icon()} {repo.status.value.capitalize()}",
                style=repo.get_status_color()
            )

            commits_text = str(repo.commits) if repo.commits > 0 else "-"

            time_text = "-"
            if repo.processing_time > 0:
                time_text = f"{repo.processing_time:.1f}s"
            elif repo.status == RepositoryStatus.PROCESSING and repo.start_time:
                elapsed = (datetime.now() - repo.start_time).total_seconds()
                time_text = f"{elapsed:.1f}s"

            result_text = ""
            if repo.status == RepositoryStatus.COMPLETE:
                result_text = "[green]OK[/green]"
            elif repo.status == RepositoryStatus.ERROR:
                result_text = "[red]Failed[/red]"
            elif repo.status == RepositoryStatus.SKIPPED:
                result_text = "[yellow]Skip[/yellow]"

            table.add_row(
                repo.name[:30],  # Truncate long names
                status_text,
                commits_text,
                time_text,
                result_text,
            )

        if len(self.repositories) > 10:
            table.add_row(
                f"... and {len(self.repositories) - 10} more",
                "",
                "",
                "",
                "",
                style="dim",
            )

        return Panel(
            table,
            title="[bold]Repository Status[/bold]",
            box=box.ROUNDED,
            padding=(1, 2),
        )

    def _create_statistics_panel(self) -> Panel:
        """Create the statistics panel."""
        # Update system statistics
        with self._lock:
            try:
                self.statistics.memory_usage = self._process.memory_info().rss / 1024 / 1024
                self.statistics.cpu_percent = self._process.cpu_percent()
            except:
                pass

        stats_items = []

        # Main statistics row
        main_stats = [
            f"[bold cyan]Commits:[/bold cyan] {self.statistics.total_commits:,}",
            f"[bold cyan]Developers:[/bold cyan] {self.statistics.total_developers}",
            f"[bold cyan]Tickets:[/bold cyan] {self.statistics.total_tickets}",
        ]
        stats_items.append(" • ".join(main_stats))

        # System statistics row
        system_stats = [
            f"[bold yellow]Memory:[/bold yellow] {self.statistics.memory_usage:.0f} MB",
            f"[bold yellow]CPU:[/bold yellow] {self.statistics.cpu_percent:.1f}%",
            f"[bold yellow]Speed:[/bold yellow] {self.statistics.processing_speed:.1f} commits/s",
        ]
        stats_items.append(" • ".join(system_stats))

        # Phase and timing
        phase_text = f"[bold green]Phase:[/bold green] {self.statistics.current_phase}"
        elapsed_text = f"[bold blue]Elapsed:[/bold blue] {self.statistics.get_elapsed_time()}"
        stats_items.append(f"{phase_text} • {elapsed_text}")

        content = "\n".join(stats_items)

        return Panel(
            content,
            title="[bold]Statistics[/bold]",
            box=box.ROUNDED,
            padding=(1, 2),
        )

    def _create_layout(self) -> Layout:
        """Create the complete layout."""
        layout = Layout()

        layout.split_column(
            Layout(self._create_header_panel(), size=3),
            Layout(name="progress", size=7),
            Layout(name="repos", size=12),
            Layout(name="stats", size=6),
        )

        layout["progress"].update(self._create_progress_panel())
        layout["repos"].update(self._create_repository_table())
        layout["stats"].update(self._create_statistics_panel())

        return layout

    def start(self, total_items: int = 100, description: str = "Analyzing repositories"):
        """
        Start the progress display.

        Args:
            total_items: Total number of items to process
            description: Description of the overall task
        """
        with self._lock:
            self.statistics.start_time = datetime.now()
            self.overall_task_id = self.overall_progress.add_task(
                description,
                total=total_items
            )

            self._layout = self._create_layout()
            self._live = Live(
                self._layout,
                console=self.console,
                refresh_per_second=1 / self.update_frequency,
                screen=True,
            )
            self._live.start()

    def stop(self):
        """Stop the progress display."""
        if self._live:
            self._live.stop()
            self._live = None

    def update_overall(self, completed: int, description: Optional[str] = None):
        """Update overall progress."""
        with self._lock:
            if self.overall_task_id is not None:
                update_kwargs = {"completed": completed}
                if description:
                    update_kwargs["description"] = description
                self.overall_progress.update(self.overall_task_id, **update_kwargs)

            if self._layout:
                self._layout["progress"].update(self._create_progress_panel())

    def start_repository(self, repo_name: str, total_commits: int = 0):
        """Start processing a repository."""
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

            if self._layout:
                self._layout["repos"].update(self._create_repository_table())

    def update_repository(self, repo_name: str, commits: int, speed: float = 0.0):
        """Update repository progress."""
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

            if self._layout:
                self._layout["progress"].update(self._create_progress_panel())
                self._layout["stats"].update(self._create_statistics_panel())

    def finish_repository(self, repo_name: str, success: bool = True,
                         error_message: Optional[str] = None):
        """Finish processing a repository."""
        with self._lock:
            if repo_name not in self.repositories:
                return

            repo_info = self.repositories[repo_name]
            repo_info.status = RepositoryStatus.COMPLETE if success else RepositoryStatus.ERROR
            repo_info.error_message = error_message

            if repo_info.start_time:
                repo_info.processing_time = (datetime.now() - repo_info.start_time).total_seconds()

            self.statistics.processed_repositories += 1

            if self._layout:
                self._layout["repos"].update(self._create_repository_table())

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

    def set_phase(self, phase: str):
        """Set the current processing phase."""
        with self._lock:
            self.statistics.current_phase = phase

            if self._layout:
                self._layout["stats"].update(self._create_statistics_panel())

    @contextmanager
    def progress_context(self, total_items: int = 100, description: str = "Processing"):
        """Context manager for progress display."""
        try:
            self.start(total_items, description)
            yield self
        finally:
            self.stop()


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

        self.repo_progress = self.tqdm(
            total=total_commits if total_commits > 0 else 100,
            desc=f"Processing {repo_name}",
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

    def finish_repository(self, repo_name: str, success: bool = True,
                         error_message: Optional[str] = None):
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


def create_progress_display(style: str = "auto", version: str = "1.3.11",
                           update_frequency: float = 0.5) -> Any:
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
        except Exception:
            # Fall back to simple if Rich fails
            pass

    return SimpleProgressDisplay(version, update_frequency)