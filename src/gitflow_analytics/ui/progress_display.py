"""
Rich-based progress display for GitFlow Analytics.

This module provides a sophisticated progress meter using the Rich library
for beautiful terminal output with live updates and statistics.
"""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

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


class RichProgressDisplay:
    """Rich-based progress display for GitFlow Analytics."""

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
        """Create the main progress panel with prominent activity display."""
        content_lines = []

        # Overall progress with enhanced display
        overall_text = Text("Overall Progress: ", style="bold cyan")
        if self.statistics.processed_repositories > 0:
            pct = (
                self.statistics.processed_repositories / self.statistics.total_repositories
            ) * 100
            overall_text.append(
                f"{self.statistics.processed_repositories}/{self.statistics.total_repositories} repositories ",
                style="white",
            )
            overall_text.append(f"({pct:.1f}%)", style="bold green" if pct > 50 else "bold yellow")
        content_lines.append(overall_text)
        content_lines.append(self.overall_progress)

        # Current repository progress - VERY prominent display
        if self.current_repo:
            content_lines.append(Text())  # Empty line for spacing
            repo_info = self.repositories.get(self.current_repo)
            if repo_info and repo_info.status == RepositoryStatus.PROCESSING:
                # Animated activity indicator
                spinner_frames = ["🔄", "🔃", "🔄", "🔃"]
                frame_idx = int(time.time() * 2) % len(spinner_frames)

                # Determine current action based on progress
                if repo_info.commits == 0:
                    action = f"{spinner_frames[frame_idx]} Fetching commits from"
                    action_style = "bold yellow blink"
                elif (
                    repo_info.total_commits > 0
                    and repo_info.commits < repo_info.total_commits * 0.3
                ):
                    action = f"{spinner_frames[frame_idx]} Starting analysis of"
                    action_style = "bold yellow"
                elif (
                    repo_info.total_commits > 0
                    and repo_info.commits < repo_info.total_commits * 0.7
                ):
                    action = f"{spinner_frames[frame_idx]} Processing commits in"
                    action_style = "bold green"
                else:
                    action = f"{spinner_frames[frame_idx]} Finalizing analysis of"
                    action_style = "bold cyan"

                # Build the current activity text
                current_text = Text(action + " ", style=action_style)
                current_text.append(f"{repo_info.name}", style="bold white on blue")

                # Add detailed progress info
                if repo_info.total_commits > 0:
                    progress_pct = (repo_info.commits / repo_info.total_commits) * 100
                    current_text.append(
                        f"\n   📊 Progress: {repo_info.commits}/{repo_info.total_commits} commits ",
                        style="white",
                    )
                    current_text.append(f"({progress_pct:.1f}%)", style="bold green")

                    # Estimate time remaining
                    if repo_info.start_time and repo_info.commits > 0:
                        elapsed = (datetime.now() - repo_info.start_time).total_seconds()
                        rate = repo_info.commits / elapsed if elapsed > 0 else 0
                        remaining = (
                            (repo_info.total_commits - repo_info.commits) / rate if rate > 0 else 0
                        )
                        if remaining > 0:
                            current_text.append(f" - ETA: {remaining:.0f}s", style="dim white")
                elif repo_info.commits > 0:
                    current_text.append(
                        f"\n   📊 Found {repo_info.commits} commits so far...", style="yellow"
                    )
                else:
                    current_text.append("\n   📥 Cloning repository...", style="yellow blink")

                content_lines.append(current_text)
                content_lines.append(self.repo_progress)

        # Create a group of all elements (Group already imported at top)
        group_items = []
        for item in content_lines:
            group_items.append(item)  # Both Text and Progress objects

        return Panel(
            Group(*group_items),
            title="[bold]🚀 Live Progress Monitor[/bold]",
            box=box.ROUNDED,
            padding=(1, 2),
            border_style="bright_blue",
        )

    def _create_repository_table(self) -> Panel:
        """Create the repository status table with scrollable view."""
        # Get terminal height to determine max visible rows
        console_height = self.console.size.height
        # Reserve space for header, progress, stats panels (approximately 18 lines)
        available_height = max(10, console_height - 18)

        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.SIMPLE_HEAD,
            expand=True,
            show_lines=False,
            row_styles=["none", "dim"],  # Alternate row colors for readability
        )

        table.add_column("#", width=4, justify="right", style="dim")
        table.add_column("Repository", style="cyan", no_wrap=True, width=25)
        table.add_column("Status", justify="center", width=15)
        table.add_column("Progress", width=20)
        table.add_column("Stats", justify="right", width=20)
        table.add_column("Time", justify="right", width=8)

        # Sort repositories: processing first, then error, then complete, then pending
        sorted_repos = sorted(
            self.repositories.values(),
            key=lambda r: (
                r.status != RepositoryStatus.PROCESSING,
                r.status != RepositoryStatus.ERROR,
                r.status != RepositoryStatus.COMPLETE,
                r.name,
            ),
        )

        # Calculate visible repositories
        total_repos = len(sorted_repos)
        visible_repos = sorted_repos[: available_height - 2]  # Leave room for summary row

        for idx, repo in enumerate(visible_repos, 1):
            # Status with icon and animation for processing
            if repo.status == RepositoryStatus.PROCESSING:
                # Animated spinner for current repo
                spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                frame_idx = int(time.time() * 10) % len(spinner_frames)
                status_icon = spinner_frames[frame_idx]
                status_text = Text(f"{status_icon} Processing", style="bold yellow")
            else:
                status_text = Text(
                    f"{repo.get_status_icon()} {repo.status.value.capitalize()}",
                    style=repo.get_status_color(),
                )

            # Progress bar for processing repos
            progress_text = ""
            if repo.status == RepositoryStatus.PROCESSING:
                if repo.total_commits > 0:
                    progress_pct = (repo.commits / repo.total_commits) * 100
                    bar_width = 10
                    filled = int(bar_width * progress_pct / 100)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    progress_text = f"[yellow]{bar}[/yellow] {progress_pct:.0f}%"
                else:
                    progress_text = "[yellow]Fetching...[/yellow]"
            elif repo.status == RepositoryStatus.COMPLETE:
                progress_text = "[green]████████████[/green] 100%"
            else:
                progress_text = "[dim]────────────[/dim]"

            # Stats column
            stats_text = ""
            if repo.commits > 0:
                if repo.developers > 0:
                    stats_text = f"{repo.commits} commits, {repo.developers} devs"
                else:
                    stats_text = f"{repo.commits} commits"
            elif repo.status == RepositoryStatus.PROCESSING:
                stats_text = "[yellow]Analyzing...[/yellow]"
            else:
                stats_text = "-"

            # Time column
            time_text = "-"
            if repo.processing_time > 0:
                time_text = f"{repo.processing_time:.1f}s"
            elif repo.status == RepositoryStatus.PROCESSING and repo.start_time:
                elapsed = (datetime.now() - repo.start_time).total_seconds()
                time_text = f"[yellow]{elapsed:.0f}s[/yellow]"

            table.add_row(
                str(idx),
                repo.name[:25],  # Truncate long names
                status_text,
                progress_text,
                stats_text,
                time_text,
            )

        # Add summary row if there are more repositories
        if total_repos > len(visible_repos):
            remaining = total_repos - len(visible_repos)
            table.add_row(
                "...",
                f"[dim italic]and {remaining} more repositories[/dim italic]",
                "",
                "",
                "",
                "",
            )

        # Add totals row
        completed = sum(
            1 for r in self.repositories.values() if r.status == RepositoryStatus.COMPLETE
        )
        processing = sum(
            1 for r in self.repositories.values() if r.status == RepositoryStatus.PROCESSING
        )
        pending = sum(1 for r in self.repositories.values() if r.status == RepositoryStatus.PENDING)

        title = f"[bold]Repository Status[/bold] (✅ {completed} | 🔄 {processing} | ⏸️ {pending})"

        return Panel(
            table,
            title=title,
            box=box.ROUNDED,
            padding=(1, 2),
        )

    def _create_statistics_panel(self) -> Panel:
        """Create the statistics panel with live updates."""
        # Update system statistics (only if psutil is available)
        with self._lock:
            if self._process:
                try:
                    self.statistics.memory_usage = self._process.memory_info().rss / 1024 / 1024
                    self.statistics.cpu_percent = self._process.cpu_percent()
                except (AttributeError, OSError):
                    # Process might have terminated or psutil unavailable
                    # This is non-critical for analysis, so just skip the update
                    pass
                except Exception as e:
                    # Log unexpected errors but don't fail progress display
                    # Only log once to avoid spam
                    if not hasattr(self, "_stats_error_logged"):
                        import logging

                        logging.getLogger(__name__).debug(
                            f"Could not update process statistics: {e}"
                        )
                        self._stats_error_logged = True

        stats_items = []

        # Calculate overall completion percentage
        if self.statistics.total_repositories > 0:
            overall_pct = (
                self.statistics.processed_repositories / self.statistics.total_repositories
            ) * 100
            completion_bar = self._create_mini_progress_bar(overall_pct, 20)
            stats_items.append(f"[bold]Overall:[/bold] {completion_bar} {overall_pct:.1f}%")

        # Main statistics row with live counters
        main_stats = []
        if self.statistics.total_commits > 0:
            main_stats.append(
                f"[bold cyan]📊 Commits:[/bold cyan] {self.statistics.total_commits:,}"
            )
        if self.statistics.total_developers > 0:
            main_stats.append(
                f"[bold cyan]👥 Developers:[/bold cyan] {self.statistics.total_developers}"
            )
        if self.statistics.total_tickets > 0:
            main_stats.append(f"[bold cyan]🎫 Tickets:[/bold cyan] {self.statistics.total_tickets}")

        if main_stats:
            stats_items.append(" • ".join(main_stats))

        # System performance with visual indicators
        system_stats = []
        if PSUTIL_AVAILABLE:
            mem_icon = (
                "🟢"
                if self.statistics.memory_usage < 500
                else "🟡"
                if self.statistics.memory_usage < 1000
                else "🔴"
            )
            cpu_icon = (
                "🟢"
                if self.statistics.cpu_percent < 50
                else "🟡"
                if self.statistics.cpu_percent < 80
                else "🔴"
            )
            system_stats.append(f"{mem_icon} Memory: {self.statistics.memory_usage:.0f} MB")
            system_stats.append(f"{cpu_icon} CPU: {self.statistics.cpu_percent:.1f}%")

        if self.statistics.processing_speed > 0:
            speed_icon = (
                "🚀"
                if self.statistics.processing_speed > 100
                else "⚡"
                if self.statistics.processing_speed > 50
                else "🐢"
            )
            system_stats.append(
                f"{speed_icon} Speed: {self.statistics.processing_speed:.1f} commits/s"
            )

        if system_stats:
            stats_items.append(" • ".join(system_stats))

        # Enhanced phase display with activity indicator
        phase_indicator = (
            "⚙️"
            if "Processing" in self.statistics.current_phase
            else "🔍"
            if "Analyzing" in self.statistics.current_phase
            else "✨"
        )
        phase_text = f"{phase_indicator} [bold green]{self.statistics.current_phase}[/bold green]"
        elapsed_text = f"⏱️ [bold blue]{self.statistics.get_elapsed_time()}[/bold blue]"

        # Estimate total time if possible
        eta_text = ""
        if (
            self.statistics.processed_repositories > 0
            and self.statistics.total_repositories > 0
            and self.statistics.processed_repositories < self.statistics.total_repositories
        ):
            elapsed_seconds = (
                (datetime.now() - self.statistics.start_time).total_seconds()
                if self.statistics.start_time
                else 0
            )
            if elapsed_seconds > 0:
                rate = self.statistics.processed_repositories / elapsed_seconds
                remaining = (
                    (self.statistics.total_repositories - self.statistics.processed_repositories)
                    / rate
                    if rate > 0
                    else 0
                )
                if remaining > 0:
                    eta_text = f" • ETA: {timedelta(seconds=int(remaining))}"

        stats_items.append(f"{phase_text} • {elapsed_text}{eta_text}")

        content = "\n".join(stats_items)

        return Panel(
            content,
            title="[bold]📈 Live Statistics[/bold]",
            box=box.ROUNDED,
            padding=(1, 2),
            border_style=(
                "green"
                if self.statistics.processed_repositories == self.statistics.total_repositories
                else "yellow"
            ),
        )

    def _create_mini_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Create a mini progress bar for inline display."""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        color = "green" if percentage >= 75 else "yellow" if percentage >= 50 else "cyan"
        return f"[{color}]{bar}[/{color}]"

    def _create_simple_layout(self) -> Panel:
        """Create a simpler layout without embedded Progress objects."""
        # Create a simple panel that we'll update dynamically
        content = self._generate_display_content()
        return Panel(
            content,
            title="[bold cyan]GitFlow Analytics Progress[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )

    def _generate_display_content(self) -> str:
        """Generate the display content as a string."""
        lines = []

        # Header
        lines.append(
            f"[bold cyan]Analyzing {self.statistics.total_repositories} repositories[/bold cyan]"
        )
        lines.append("")

        # Overall progress
        if self.overall_task_id is not None:
            task = self.overall_progress.tasks[0] if self.overall_progress.tasks else None
            if task:
                progress_pct = (task.completed / task.total * 100) if task.total > 0 else 0
                bar = self._create_mini_progress_bar(progress_pct, 40)
                lines.append(f"Overall Progress: {bar} {progress_pct:.1f}%")
                lines.append(f"Status: {task.description}")

        # Current repository
        if self.current_repo:
            lines.append("")
            lines.append(f"[yellow]Current Repository:[/yellow] {self.current_repo}")
            if self.repo_task_id is not None and self.repo_progress.tasks:
                repo_task = self.repo_progress.tasks[0] if self.repo_progress.tasks else None
                if repo_task:
                    lines.append(f"  Commits: {repo_task.completed}/{repo_task.total}")

        # Statistics
        lines.append("")
        lines.append("[bold green]Statistics:[/bold green]")
        lines.append(
            f"  Processed: {self.statistics.processed_repositories}/{self.statistics.total_repositories}"
        )
        lines.append(f"  Success: {self.statistics.successful_repositories}")
        lines.append(f"  Failed: {self.statistics.failed_repositories}")
        lines.append(f"  Skipped: {self.statistics.skipped_repositories}")

        if self.statistics.total_commits_processed > 0:
            lines.append(f"  Total Commits: {self.statistics.total_commits_processed:,}")

        # Repository list (last 5)
        if self.repositories:
            lines.append("")
            lines.append("[bold]Recent Repositories:[/bold]")
            recent = list(self.repositories.values())[-5:]
            for repo in recent:
                status_icon = {
                    RepositoryStatus.PENDING: "⏳",
                    RepositoryStatus.PROCESSING: "🔄",
                    RepositoryStatus.COMPLETE: "✅",
                    RepositoryStatus.ERROR: "❌",
                    RepositoryStatus.SKIPPED: "⏭️",
                }.get(repo.status, "❓")
                lines.append(f"  {status_icon} {repo.name}")

        return "\n".join(lines)

    def _update_all_panels(self):
        """Force update all panels in the layout."""
        if self._layout and self._live:
            # Update the simple panel with new content
            new_content = self._generate_display_content()
            self._layout.renderable = new_content
            # Rich's auto_refresh handles the updates automatically
            self._update_counter += 1

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
                except Exception:
                    pass  # Ignore errors during cleanup
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
                value = metrics[key].get("value", "N/A")
                rating = metrics[key].get("rating", "")
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
        except Exception:
            # Fall back to simple if Rich fails
            pass

    return SimpleProgressDisplay(version, update_frequency)
