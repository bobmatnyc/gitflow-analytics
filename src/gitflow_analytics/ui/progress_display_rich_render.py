"""Rich display rendering mixin - panel creation and layout rendering."""

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




class RichDisplayRenderMixin:
    """Mixin: _create_* panel methods and display rendering for RichProgressDisplay."""

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

