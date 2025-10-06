"""Progress adapter for TUI to connect with core analysis progress."""

import asyncio
import time
from typing import Optional

from gitflow_analytics.core.progress import ProgressContext, ProgressService
from gitflow_analytics.tui.widgets.progress_widget import AnalysisProgressWidget


class TUIProgressAdapter(ProgressService):
    """
    Adapter that bridges the core ProgressService with TUI progress widgets.

    This allows the GitAnalyzer's progress updates to be reflected in the TUI
    interface in real-time.
    """

    def __init__(self, widget: Optional[AnalysisProgressWidget] = None):
        """Initialize the TUI progress adapter.

        Args:
            widget: The progress widget to update
        """
        super().__init__()
        self.widget = widget
        self.current_progress = 0.0
        self.total_items = 100.0
        self._loop = None
        self.processing_stats = {
            "total": 0,
            "processed": 0,
            "success": 0,
            "failed": 0,
            "timeout": 0,
        }
        self.repositories_in_progress = {}

    def set_widget(self, widget: AnalysisProgressWidget) -> None:
        """Set or update the progress widget."""
        self.widget = widget

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop for async updates."""
        self._loop = loop

    def create_progress(
        self,
        total: int,
        description: str = "",
        unit: str = "items",
        nested: bool = False,
        leave: bool = True,
        position: Optional[int] = None,
    ) -> ProgressContext:
        """Create a progress context for tracking.

        Args:
            total: Total number of items to process
            description: Description of the task
            unit: Unit of work (e.g., "commits", "files")
            nested: Whether this is a nested progress bar
            leave: Whether to leave the progress bar on screen after completion
            position: Explicit position for the progress bar (for nested contexts)

        Returns:
            Progress context for this task
        """
        # Pass all parameters to parent class with correct signature
        context = super().create_progress(total, description, unit, nested, leave, position)
        self.total_items = float(total)
        self.current_progress = 0.0

        # Update widget if available
        if self.widget and self._loop:
            self._loop.call_soon_threadsafe(self._update_widget_sync, 0.0, description)

        return context

    def update(self, context: ProgressContext, advance: int = 1) -> None:
        """Update progress by advancing the counter.

        Args:
            context: Progress context to update
            advance: Number of items completed
        """
        super().update(context, advance)

        # Calculate percentage
        if self.total_items > 0:
            self.current_progress += advance
            percentage = (self.current_progress / self.total_items) * 100.0

            # Update widget if available
            if self.widget and self._loop:
                description = getattr(context, "description", "")
                self._loop.call_soon_threadsafe(self._update_widget_sync, percentage, description)

    def set_description(self, context: ProgressContext, description: str) -> None:
        """Update the description of a progress context.

        Args:
            context: Progress context to update
            description: New description
        """
        super().set_description(context, description)

        # Update widget description if available
        if self.widget and self._loop:
            percentage = (
                (self.current_progress / self.total_items) * 100.0 if self.total_items > 0 else 0
            )
            self._loop.call_soon_threadsafe(self._update_widget_sync, percentage, description)

    def complete(self, context: ProgressContext) -> None:
        """Mark a progress context as complete.

        Args:
            context: Progress context to complete
        """
        super().complete(context)

        # Update widget to 100% if available
        if self.widget and self._loop:
            self._loop.call_soon_threadsafe(self._update_widget_sync, 100.0, "Complete")

    def _update_widget_sync(self, percentage: float, description: str) -> None:
        """Synchronously update the widget (called from event loop).

        Args:
            percentage: Progress percentage (0-100)
            description: Status description
        """
        if self.widget:
            try:
                # Format description with processing statistics if available
                if self.processing_stats["total"] > 0:
                    stats_str = (
                        f"Processed: {self.processing_stats['processed']}/{self.processing_stats['total']}, "
                        f"Success: {self.processing_stats['success']}, "
                        f"Failed: {self.processing_stats['failed']}"
                    )
                    if self.processing_stats["timeout"] > 0:
                        stats_str += f", Timeout: {self.processing_stats['timeout']}"

                    full_description = f"{description}\n{stats_str}"
                else:
                    full_description = description

                self.widget.update_progress(percentage, full_description)
                # Force a refresh of the TUI to ensure updates are visible
                if hasattr(self.widget, "refresh"):
                    self.widget.refresh()
            except Exception as e:
                # Log error but don't crash the progress tracking
                import logging

                logging.getLogger(__name__).error(f"Failed to update TUI widget: {e}")

    def start_repository(self, repo_name: str, total_commits: int) -> None:
        """Track the start of repository processing.

        Args:
            repo_name: Name of the repository
            total_commits: Expected number of commits to process
        """
        self.repositories_in_progress[repo_name] = {
            "started_at": time.time(),
            "total_commits": total_commits,
            "status": "processing",
        }

        # Update the widget with repository info
        if self.widget and self._loop:
            # Calculate correct percentage based on actual processed count
            if self.processing_stats["total"] > 0:
                percentage = (
                    self.processing_stats["processed"] / self.processing_stats["total"]
                ) * 100.0
            else:
                percentage = 0

            description = f"Processing: {repo_name} ({len(self.repositories_in_progress)}/{self.processing_stats['total']})"
            self._loop.call_soon_threadsafe(self._update_widget_sync, percentage, description)

    def finish_repository(
        self, repo_name: str, success: bool = True, error_message: str = None, stats: dict = None
    ) -> None:
        """Mark a repository as finished processing.

        Args:
            repo_name: Name of the repository
            success: Whether processing was successful
            error_message: Error message if failed
            stats: Updated processing statistics
        """
        if repo_name in self.repositories_in_progress:
            self.repositories_in_progress[repo_name]["status"] = "success" if success else "failed"
            if error_message:
                self.repositories_in_progress[repo_name]["error"] = error_message

        # Update stats if provided
        if stats:
            self.processing_stats.update(stats)

        # Update the widget with completion info
        if self.widget and self._loop:
            # Calculate correct percentage based on processed count
            if self.processing_stats["total"] > 0:
                percentage = (
                    self.processing_stats["processed"] / self.processing_stats["total"]
                ) * 100.0
            else:
                percentage = 100.0 if self.processing_stats["processed"] > 0 else 0

            status = "✅" if success else "❌"
            description = f"{status} {repo_name} completed"
            if error_message:
                description += f" - {error_message[:30]}..."

            self._loop.call_soon_threadsafe(
                self._update_widget_sync, min(percentage, 100.0), description  # Cap at 100%
            )

    def update_stats(
        self,
        processed: int = 0,
        success: int = 0,
        failed: int = 0,
        timeout: int = 0,
        total: int = 0,
    ) -> None:
        """Update processing statistics.

        Args:
            processed: Number of repositories processed
            success: Number of successful repositories
            failed: Number of failed repositories
            timeout: Number of timed out repositories
            total: Total number of repositories
        """
        if processed > 0:
            self.processing_stats["processed"] = processed
        if success > 0:
            self.processing_stats["success"] = success
        if failed > 0:
            self.processing_stats["failed"] = failed
        if timeout > 0:
            self.processing_stats["timeout"] = timeout
        if total > 0:
            self.processing_stats["total"] = total

        # Update widget with latest stats
        if self.widget and self._loop:
            percentage = (
                (self.processing_stats["processed"] / self.processing_stats["total"]) * 100.0
                if self.processing_stats["total"] > 0
                else 0
            )

            self._loop.call_soon_threadsafe(
                self._update_widget_sync,
                min(percentage, 100.0),  # Cap at 100%
                "Processing repositories...",
            )


class TUIProgressService:
    """Service to manage TUI progress adapters for different analysis phases."""

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Initialize the TUI progress service.

        Args:
            loop: Event loop for async updates
        """
        self.loop = loop or asyncio.get_event_loop()
        self.adapters = {}

    def create_adapter(self, name: str, widget: AnalysisProgressWidget) -> TUIProgressAdapter:
        """Create a progress adapter for a specific widget.

        Args:
            name: Name/ID of the adapter
            widget: The progress widget to connect

        Returns:
            The created adapter
        """
        adapter = TUIProgressAdapter(widget)
        adapter.set_event_loop(self.loop)
        self.adapters[name] = adapter
        return adapter

    def get_adapter(self, name: str) -> Optional[TUIProgressAdapter]:
        """Get an existing adapter by name.

        Args:
            name: Name/ID of the adapter

        Returns:
            The adapter if found, None otherwise
        """
        return self.adapters.get(name)

    def remove_adapter(self, name: str) -> None:
        """Remove an adapter.

        Args:
            name: Name/ID of the adapter to remove
        """
        if name in self.adapters:
            del self.adapters[name]
