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
    for testing, and event capture for verification.
    """

    def __init__(self):
        """Initialize the progress service."""
        self._enabled = True
        self._lock = threading.Lock()
        self._active_contexts: list[ProgressContext] = []
        self._position_counter = 0
        self._capture_events = False
        self._captured_events: list[ProgressEvent] = []

        # Check environment for testing mode
        self._check_testing_environment()

    def _check_testing_environment(self):
        """Check if running in a testing environment and disable if needed.

        WHY: Progress bars interfere with test output and can cause issues in CI/CD.
        This automatically detects common testing scenarios and disables progress.
        """
        # Disable in pytest
        if "pytest" in sys.modules:
            self._enabled = False

        # Disable if explicitly requested via environment
        if os.environ.get("GITFLOW_DISABLE_PROGRESS", "").lower() in ("1", "true", "yes"):
            self._enabled = False

        # Disable if not in a TTY (e.g., CI/CD, piped output)
        if not sys.stdout.isatty():
            self._enabled = False

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
            if context.progress_bar:
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


# Global singleton instance
_progress_service: Optional[ProgressService] = None
_service_lock = threading.Lock()


def get_progress_service() -> ProgressService:
    """Get the global progress service instance.

    Returns:
        The singleton ProgressService instance

    Thread-safe singleton pattern ensures only one progress service exists.
    """
    global _progress_service

    if _progress_service is None:
        with _service_lock:
            if _progress_service is None:
                _progress_service = ProgressService()

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
