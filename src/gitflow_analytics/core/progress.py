"""Centralized progress reporting service for GitFlow Analytics.

This module provides a unified interface for progress reporting across the application,
replacing scattered tqdm usage with a centralized, testable, and configurable service.

WHY: Progress reporting was scattered across multiple modules (analyzer.py, data_fetcher.py,
batch_classifier.py, etc.), violating DRY principles and making it difficult to maintain
consistent progress UX. This service centralizes all progress management.

DESIGN DECISIONS:
- Context-based API: Each progress bar gets a context object for clean lifecycle management
- Thread-safe: Uses threading locks to ensure safe concurrent access
- Testable: Can be globally disabled for testing, with event capture capability
- Nested support: Handles nested progress contexts with proper positioning
- Consistent styling: All progress bars follow the same formatting rules
- Rich integration: Optional Rich library support for enhanced terminal UI

USAGE:
    from gitflow_analytics.core.progress import get_progress_service

    progress = get_progress_service()
    context = progress.create_progress(100, "Processing items")
    for item in items:
        # Process item
        progress.update(context)
    progress.complete(context)
"""

import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

from tqdm import tqdm

# Import UI components if available
try:
    from ..ui.progress_display import (
        RICH_AVAILABLE,
        create_progress_display,
    )

    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False
    RICH_AVAILABLE = False


@dataclass
class ProgressContext:
    """Context object for a single progress operation.

    Encapsulates all state for a progress bar, allowing clean lifecycle management
    and preventing resource leaks.
    """

    progress_bar: Optional[Any]  # tqdm instance or None if disabled
    description: str
    total: int
    unit: str
    position: int
    current: int = 0
    is_nested: bool = False
    parent_context: Optional["ProgressContext"] = None


@dataclass
class ProgressEvent:
    """Event captured during progress operations for testing.

    Allows tests to verify that progress operations occurred without
    actually displaying progress bars.
    """

    event_type: str  # 'create', 'update', 'complete'
    description: str
    total: Optional[int] = None
    increment: Optional[int] = None
    current: Optional[int] = None


class ProgressService:
    """Centralized service for managing progress reporting.

    This service provides a unified interface for creating and managing progress bars
    throughout the application. It supports nested progress contexts, global disable
    for testing, event capture for verification, and optional Rich terminal UI.
    """

    def __init__(self, display_style: str = "auto", version: str = "1.3.11"):
        """Initialize the progress service.

        Args:
            display_style: Display style ("rich", "simple", or "auto")
            version: Version string for display
        """
        self._enabled = True
        self._lock = threading.Lock()
        self._active_contexts: list[ProgressContext] = []
        self._position_counter = 0
        self._capture_events = False
        self._captured_events: list[ProgressEvent] = []
        self._display_style = display_style
        self._version = version

        # Rich display components
        self._rich_display = None
        self._repository_contexts: dict[str, Any] = {}
        self._use_rich = False

        # Initialize display based on configuration
        self._init_display()

        # Check environment for testing mode
        # Note: If user explicitly requested rich mode, don't disable it
        self._check_testing_environment()

    def _init_display(self):
        """Initialize the appropriate display based on configuration."""
        if UI_AVAILABLE and self._display_style in ("rich", "auto"):
            try:
                self._rich_display = create_progress_display(
                    style=self._display_style, version=self._version, update_frequency=0.5
                )
                self._use_rich = self._display_style == "rich" or (
                    self._display_style == "auto" and RICH_AVAILABLE
                )
            except Exception:
                # Fall back to tqdm if Rich fails
                self._use_rich = False
                self._rich_display = None

    def _check_testing_environment(self):
        """Check if running in a testing environment and disable if needed.

        WHY: Progress bars interfere with test output and can cause issues in CI/CD.
        This automatically detects common testing scenarios and disables progress.
        """
        # Don't auto-disable if user explicitly requested rich mode
        explicit_rich = self._display_style == "rich"

        # Disable in pytest
        if "pytest" in sys.modules:
            self._enabled = False
            self._use_rich = False

        # Disable if explicitly requested via environment
        if os.environ.get("GITFLOW_DISABLE_PROGRESS", "").lower() in ("1", "true", "yes"):
            self._enabled = False
            self._use_rich = False

        # Disable if not in a TTY (e.g., CI/CD, piped output)
        # BUT: Keep enabled if user explicitly requested rich mode
        if not sys.stdout.isatty() and not explicit_rich:
            self._enabled = False
            self._use_rich = False

    def create_progress(
        self,
        total: int,
        description: str,
        unit: str = "items",
        nested: bool = False,
        leave: bool = True,
        position: Optional[int] = None,
    ) -> ProgressContext:
        """Create a new progress context.

        Args:
            total: Total number of items to process
            description: Description shown next to the progress bar
            unit: Unit label for items (e.g., "commits", "repos", "files")
            nested: Whether this is a nested progress bar
            leave: Whether to leave the progress bar on screen after completion
            position: Explicit position for the progress bar (for nested contexts)

        Returns:
            ProgressContext object to use for updates

        DESIGN: Returns a context object rather than the tqdm instance directly
        to provide better lifecycle management and prevent resource leaks.
        """
        with self._lock:
            # Capture event if needed
            if self._capture_events:
                self._captured_events.append(ProgressEvent("create", description, total=total))

            # Determine position for nested progress bars
            if position is None:
                if nested:
                    self._position_counter += 1
                position = self._position_counter

            # Create context
            context = ProgressContext(
                progress_bar=None,
                description=description,
                total=total,
                unit=unit,
                position=position,
                is_nested=nested,
            )

            # Create actual progress bar if enabled
            if self._enabled:
                if self._use_rich and self._rich_display:
                    # For Rich display, we don't create individual tqdm bars
                    # Instead, we'll manage everything through the Rich display
                    context.progress_bar = None
                else:
                    context.progress_bar = tqdm(
                        total=total,
                        desc=description,
                        unit=unit,
                        position=position,
                        leave=leave,
                        # Consistent styling
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                        dynamic_ncols=True,
                    )

            self._active_contexts.append(context)
            return context

    def update(
        self, context: ProgressContext, increment: int = 1, description: Optional[str] = None
    ):
        """Update progress for a given context.

        Args:
            context: The progress context to update
            increment: Number of items completed (default: 1)
            description: Optional new description to set

        WHY: Centralizes update logic and ensures consistent behavior across
        all progress bars in the application.
        """
        with self._lock:
            context.current += increment

            # Capture event if needed
            if self._capture_events:
                self._captured_events.append(
                    ProgressEvent(
                        "update",
                        description or context.description,
                        increment=increment,
                        current=context.current,
                    )
                )

            # Update actual progress bar if it exists
            if self._use_rich and self._rich_display:
                # Update Rich display based on context type
                if hasattr(context, "repository_name"):
                    # Repository-specific progress
                    speed = increment / 0.1 if increment > 0 else 0  # Simple speed calculation
                    self._rich_display.update_repository(
                        context.repository_name, context.current, speed
                    )
                else:
                    # Overall progress
                    self._rich_display.update_overall(context.current, description)
            elif context.progress_bar:
                context.progress_bar.update(increment)
                if description:
                    context.progress_bar.set_description(description)

    def set_description(self, context: ProgressContext, description: str):
        """Update the description of a progress context.

        Args:
            context: The progress context to update
            description: New description to display
        """
        with self._lock:
            context.description = description
            if context.progress_bar:
                context.progress_bar.set_description(description)

    def complete(self, context: ProgressContext):
        """Mark a progress context as complete and clean up resources.

        Args:
            context: The progress context to complete

        IMPORTANT: Always call this method when done with a progress context
        to ensure proper resource cleanup.
        """
        with self._lock:
            # Capture event if needed
            if self._capture_events:
                self._captured_events.append(
                    ProgressEvent("complete", context.description, current=context.current)
                )

            # Remove from active contexts BEFORE modifying progress_bar
            # to avoid comparison issues with None
            if context in self._active_contexts:
                self._active_contexts.remove(context)

            # Close actual progress bar if it exists
            if context.progress_bar:
                context.progress_bar.close()
                context.progress_bar = None

            # Reset position counter if no nested contexts remain
            if context.is_nested and not any(c.is_nested for c in self._active_contexts):
                self._position_counter = 0

    @contextmanager
    def progress(
        self,
        total: int,
        description: str,
        unit: str = "items",
        nested: bool = False,
        leave: bool = True,
    ):
        """Context manager for progress operations.

        Args:
            total: Total number of items to process
            description: Description shown next to the progress bar
            unit: Unit label for items
            nested: Whether this is a nested progress bar
            leave: Whether to leave the progress bar on screen

        Yields:
            ProgressContext object for updates

        Example:
            with progress.progress(100, "Processing") as ctx:
                for item in items:
                    process(item)
                    progress.update(ctx)
        """
        context = self.create_progress(total, description, unit, nested, leave)
        try:
            yield context
        finally:
            self.complete(context)

    def disable(self):
        """Disable all progress reporting globally.

        Useful for testing or quiet mode operation.
        """
        with self._lock:
            self._enabled = False
            # Close any active progress bars
            for context in self._active_contexts[:]:
                if context.progress_bar:
                    context.progress_bar.close()
                    context.progress_bar = None

    def enable(self):
        """Enable progress reporting globally."""
        with self._lock:
            self._enabled = True

    def is_enabled(self) -> bool:
        """Check if progress reporting is enabled."""
        return self._enabled

    def start_event_capture(self):
        """Start capturing progress events for testing.

        WHY: Allows tests to verify that progress operations occurred
        without actually displaying progress bars.
        """
        with self._lock:
            self._capture_events = True
            self._captured_events = []

    def stop_event_capture(self) -> list[ProgressEvent]:
        """Stop capturing events and return captured events.

        Returns:
            List of ProgressEvent objects that were captured
        """
        with self._lock:
            self._capture_events = False
            events = self._captured_events[:]
            self._captured_events = []
            return events

    def get_captured_events(self) -> list[ProgressEvent]:
        """Get currently captured events without stopping capture.

        Returns:
            List of ProgressEvent objects captured so far
        """
        with self._lock:
            return self._captured_events[:]

    def clear_captured_events(self):
        """Clear captured events without stopping capture."""
        with self._lock:
            self._captured_events = []

    # Rich-specific methods
    def start_rich_display(
        self, total_items: int = 100, description: str = "Analyzing repositories"
    ):
        """Start the Rich display if available.

        Args:
            total_items: Total number of items to process
            description: Description of the overall task
        """
        if self._use_rich and self._rich_display and self._enabled:
            self._rich_display.start(total_items, description)

    def stop_rich_display(self):
        """Stop the Rich display if active."""
        if self._use_rich and self._rich_display:
            self._rich_display.stop()

    def start_repository(self, repo_name: str, total_commits: int = 0):
        """Start processing a repository with Rich display.

        Args:
            repo_name: Name of the repository
            total_commits: Total number of commits to process
        """
        if self._use_rich and self._rich_display and self._enabled:
            self._rich_display.start_repository(repo_name, total_commits)

    def finish_repository(
        self, repo_name: str, success: bool = True, error_message: Optional[str] = None
    ):
        """Finish processing a repository with Rich display.

        Args:
            repo_name: Name of the repository
            success: Whether processing was successful
            error_message: Error message if processing failed
        """
        if self._use_rich and self._rich_display and self._enabled:
            self._rich_display.finish_repository(repo_name, success, error_message)

    def update_statistics(self, **kwargs):
        """Update Rich display statistics.

        Args:
            **kwargs: Statistics to update (total_commits, total_developers, etc.)
        """
        if self._use_rich and self._rich_display and self._enabled:
            self._rich_display.update_statistics(**kwargs)

    def initialize_repositories(self, repository_list: list):
        """Initialize all repositories with pending status in Rich display.

        Args:
            repository_list: List of repositories to be processed.
                            Each item should have 'name' and optionally 'path' fields.
        """
        if self._use_rich and self._rich_display and self._enabled:
            self._rich_display.initialize_repositories(repository_list)

    def set_phase(self, phase: str):
        """Set the current processing phase for Rich display.

        Args:
            phase: Description of the current phase
        """
        if self._use_rich and self._rich_display and self._enabled:
            self._rich_display.set_phase(phase)

    def create_repository_progress(
        self, repo_name: str, total: int, description: str
    ) -> ProgressContext:
        """Create a progress context specifically for repository processing.

        Args:
            repo_name: Name of the repository
            total: Total number of items to process
            description: Description of the task

        Returns:
            ProgressContext with repository information
        """
        context = self.create_progress(total, description, unit="commits", nested=True)
        # Add repository name to context for Rich display handling
        # Note: We use object.__setattr__ to bypass dataclass frozen status if needed
        object.__setattr__(context, "repository_name", repo_name)

        if self._use_rich and self._rich_display and self._enabled:
            self.start_repository(repo_name, total)

        return context


# Global singleton instance
_progress_service: Optional[ProgressService] = None
_service_lock = threading.Lock()


def get_progress_service(
    display_style: Optional[str] = None, version: Optional[str] = None
) -> ProgressService:
    """Get the global progress service instance.

    Args:
        display_style: Optional display style override ("rich", "simple", or "auto")
        version: Optional version string for display

    Returns:
        The singleton ProgressService instance

    Thread-safe singleton pattern ensures only one progress service exists.
    If display_style is provided and differs from current style, the service is reconfigured.
    """
    global _progress_service

    # Check if we need to reconfigure an existing service
    if _progress_service is not None and display_style is not None:
        with _service_lock:
            # If display style changed, reconfigure the service
            if _progress_service._display_style != display_style:
                # Close any active displays
                if _progress_service._use_rich and _progress_service._rich_display:
                    _progress_service.stop_rich_display()

                # Reconfigure with new display style
                _progress_service._display_style = display_style
                _progress_service._use_rich = False
                _progress_service._rich_display = None

                # Re-enable if user explicitly requested rich mode
                if display_style == "rich":
                    _progress_service._enabled = True

                # Reinitialize display
                _progress_service._init_display()

    if _progress_service is None:
        with _service_lock:
            if _progress_service is None:
                # Get display style from environment or use default
                if display_style is None:
                    display_style = os.environ.get("GITFLOW_PROGRESS_STYLE", "auto")
                if version is None:
                    # Try to get version from package
                    try:
                        from .._version import __version__

                        version = __version__
                    except ImportError:
                        version = "1.3.11"

                _progress_service = ProgressService(display_style=display_style, version=version)

    return _progress_service


def reset_progress_service():
    """Reset the global progress service instance.

    WARNING: Only use this in tests or during application shutdown.
    This will close all active progress bars and create a new service instance.
    """
    global _progress_service

    with _service_lock:
        if _progress_service:
            _progress_service.disable()
        _progress_service = None
