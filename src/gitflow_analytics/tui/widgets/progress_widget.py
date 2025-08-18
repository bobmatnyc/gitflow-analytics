"""Progress widget with ETA for GitFlow Analytics TUI."""

import time
from typing import Optional

from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Label, ProgressBar


class AnalysisProgressWidget(Container):
    """
    Custom progress widget that shows progress with ETA calculation.

    WHY: Standard progress bars don't provide time estimates which are crucial
    for long-running analysis operations. This widget combines progress tracking
    with ETA calculations to give users better feedback.

    DESIGN DECISION: Uses reactive attributes for real-time updates and
    calculates ETA based on average processing speed rather than simple linear
    extrapolation for more accurate estimates.
    """

    DEFAULT_CSS = """
    AnalysisProgressWidget {
        height: auto;
        border: solid $primary;
        margin: 1;
        padding: 1;
    }
    
    .progress-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    .progress-status {
        color: $text;
        margin-top: 1;
    }
    
    .progress-eta {
        color: $accent;
        text-style: italic;
    }
    """

    progress = reactive(0.0)
    total = reactive(100.0)
    status_text = reactive("Initializing...")

    def __init__(
        self,
        title: str,
        total: float = 100.0,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.title = title
        self.total = total
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.progress_history = []

    def compose(self):
        """Compose the progress widget."""
        yield Label(self.title, classes="progress-title")
        yield ProgressBar(total=self.total, id="progress-bar")
        yield Label(self.status_text, classes="progress-status", id="status-label")
        yield Label("", classes="progress-eta", id="eta-label")

    def update_progress(self, value: float, status: str = "") -> None:
        """
        Update progress and status with ETA calculation.

        WHY: Provides comprehensive progress updates including time estimates
        which are essential for user experience during long operations.

        @param value: Current progress value
        @param status: Status message to display
        """
        current_time = time.time()

        # Update reactive values
        self.progress = value
        if status:
            self.status_text = status

        # Update progress bar
        progress_bar = self.query_one("#progress-bar", ProgressBar)
        progress_bar.update(progress=value)

        # Update status label
        status_label = self.query_one("#status-label", Label)
        status_label.update(status)

        # Calculate and update ETA
        eta_text = self._calculate_eta(value, current_time)
        eta_label = self.query_one("#eta-label", Label)
        eta_label.update(eta_text)

        # Store progress history for better ETA calculation
        self.progress_history.append({"time": current_time, "progress": value})

        # Keep only recent history (last 10 updates)
        if len(self.progress_history) > 10:
            self.progress_history = self.progress_history[-10:]

    def _calculate_eta(self, current_progress: float, current_time: float) -> str:
        """
        Calculate estimated time of arrival based on progress history.

        WHY: Uses historical data points to calculate a more accurate ETA
        than simple linear extrapolation, accounting for variations in
        processing speed.
        """
        if current_progress <= 0 or current_progress >= self.total:
            return ""

        # Need at least 2 data points for calculation
        if len(self.progress_history) < 2:
            return "Calculating ETA..."

        # Calculate average rate from recent history
        recent_history = self.progress_history[-5:]  # Last 5 updates
        if len(recent_history) < 2:
            return "Calculating ETA..."

        time_span = recent_history[-1]["time"] - recent_history[0]["time"]
        progress_span = recent_history[-1]["progress"] - recent_history[0]["progress"]

        if time_span <= 0 or progress_span <= 0:
            return "Calculating ETA..."

        # Calculate rate (progress per second)
        rate = progress_span / time_span

        # Calculate remaining work and time
        remaining_progress = self.total - current_progress
        estimated_seconds = remaining_progress / rate

        # Format ETA
        if estimated_seconds < 60:
            return f"ETA: {estimated_seconds:.0f}s"
        elif estimated_seconds < 3600:
            minutes = estimated_seconds / 60
            return f"ETA: {minutes:.1f}m"
        else:
            hours = estimated_seconds / 3600
            return f"ETA: {hours:.1f}h"

    def reset(self) -> None:
        """Reset the progress widget to initial state."""
        self.progress = 0.0
        self.status_text = "Initializing..."
        self.start_time = time.time()
        self.progress_history = []

        # Reset UI elements
        progress_bar = self.query_one("#progress-bar", ProgressBar)
        progress_bar.update(progress=0)

        status_label = self.query_one("#status-label", Label)
        status_label.update("Initializing...")

        eta_label = self.query_one("#eta-label", Label)
        eta_label.update("")

    def complete(self, final_message: str = "Complete!") -> None:
        """Mark progress as complete."""
        self.update_progress(self.total, final_message)

        # Calculate total elapsed time
        total_time = time.time() - self.start_time
        if total_time < 60:
            time_str = f"{total_time:.1f}s"
        elif total_time < 3600:
            time_str = f"{total_time/60:.1f}m"
        else:
            time_str = f"{total_time/3600:.1f}h"

        eta_label = self.query_one("#eta-label", Label)
        eta_label.update(f"Completed in {time_str}")
