"""Loading screen for GitFlow Analytics TUI startup."""

import asyncio
import time
from typing import Any, Optional

from textual.binding import Binding
from textual.containers import Center, Container, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, LoadingIndicator, Static

from ..widgets.progress_widget import AnalysisProgressWidget


class LoadingScreen(Screen):
    """
    Loading screen displayed during application startup and heavy initialization.

    WHY: The TUI application needs to load configurations, spaCy models, and other
    heavy resources during startup. This screen provides user feedback about the
    loading process instead of showing a black screen, improving user experience.

    DESIGN DECISION: Uses a combination of progress indicators and status messages
    to show both overall progress and specific loading steps. This keeps users
    informed about what's happening and that the application is responsive.
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel", "Cancel Loading"),
        Binding("escape", "cancel", "Cancel Loading"),
    ]

    def __init__(
        self,
        loading_message: str = "Initializing GitFlow Analytics...",
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, id=id)
        self.loading_message = loading_message
        self.loading_task: Optional[asyncio.Task] = None
        self.start_time = time.time()
        self.cancelled = False

    def compose(self):
        """Compose the loading screen with progress indicators."""
        yield Header()

        with Container(id="loading-container"):
            # Main loading title
            yield Label("GitFlow Analytics", classes="screen-title")
            yield Label("Developer Productivity Analysis", classes="help-text center")

            # Loading animation and message
            with Center(), Vertical(id="loading-content"):
                yield LoadingIndicator(id="main-spinner")
                yield Label(self.loading_message, classes="center", id="loading-message")

                # Overall progress bar
                yield AnalysisProgressWidget("Startup Progress", total=100.0, id="startup-progress")

                # Status messages
                with Container(classes="status-panel"):
                    yield Label("Status", classes="panel-title")
                    yield Static("Starting up...", id="status-message")

                # Loading steps indicators
                with Container(classes="stats-panel"):
                    yield Label("Loading Steps", classes="panel-title")
                    with Vertical(id="loading-steps"):
                        yield Static("⏳ Initializing application...", id="step-init")
                        yield Static("⏳ Loading configuration...", id="step-config")
                        yield Static("⏳ Preparing analysis engine...", id="step-engine")
                        yield Static("⏳ Loading NLP models...", id="step-nlp")
                        yield Static("⏳ Finalizing setup...", id="step-finalize")

        yield Footer()

    def on_mount(self) -> None:
        """Start loading process when screen mounts."""
        self.loading_task = asyncio.create_task(self._simulate_loading())

    async def _simulate_loading(self) -> None:
        """
        Simulate the loading process with progress updates.

        WHY: This provides visual feedback during startup initialization.
        In practice, this would be replaced by actual initialization calls.
        """
        progress_widget = self.query_one("#startup-progress", AnalysisProgressWidget)
        status_message = self.query_one("#status-message", Static)

        try:
            # Step 1: Initialize application (20%)
            await self._update_step("step-init", "✅ Application initialized", "success")
            status_message.update("Initializing core components...")
            progress_widget.update_progress(20, "Core components ready")
            await asyncio.sleep(0.3)

            if self.cancelled:
                return

            # Step 2: Load configuration (40%)
            await self._update_step("step-config", "✅ Configuration loaded", "success")
            status_message.update("Loading configuration files...")
            progress_widget.update_progress(40, "Configuration loaded")
            await asyncio.sleep(0.5)

            if self.cancelled:
                return

            # Step 3: Prepare analysis engine (60%)
            await self._update_step("step-engine", "✅ Analysis engine ready", "success")
            status_message.update("Preparing Git analysis engine...")
            progress_widget.update_progress(60, "Analysis engine initialized")
            await asyncio.sleep(0.4)

            if self.cancelled:
                return

            # Step 4: Load NLP models (85%) - This is the heavy operation
            await self._update_step("step-nlp", "⏳ Loading spaCy models...", "warning")
            status_message.update("Loading natural language processing models...")
            progress_widget.update_progress(70, "Loading spaCy models...")
            await asyncio.sleep(1.2)  # Simulate spaCy model loading time

            if self.cancelled:
                return

            await self._update_step("step-nlp", "✅ NLP models loaded", "success")
            progress_widget.update_progress(85, "NLP models ready")

            # Step 5: Finalize setup (100%)
            await self._update_step("step-finalize", "✅ Setup complete", "success")
            status_message.update("Finalizing application setup...")
            progress_widget.update_progress(100, "GitFlow Analytics ready!")
            await asyncio.sleep(0.3)

            # Show completion message briefly
            loading_message = self.query_one("#loading-message", Label)
            loading_message.update("Loading complete! Starting application...")

            elapsed_time = time.time() - self.start_time
            status_message.update(f"Ready! Loaded in {elapsed_time:.1f} seconds")

            # Wait a moment to show completion
            await asyncio.sleep(0.8)

            # Signal that loading is complete
            self.app.post_message(self.LoadingComplete())

        except asyncio.CancelledError:
            progress_widget.update_progress(0, "Loading cancelled")
            status_message.update("Loading cancelled by user")
        except Exception as e:
            progress_widget.update_progress(0, f"Error: {str(e)[:50]}...")
            status_message.update(f"Loading failed: {e}")
            self.app.notify(f"Loading failed: {e}", severity="error")

    async def _update_step(self, step_id: str, message: str, status: str) -> None:
        """
        Update a loading step with status.

        @param step_id: ID of the step element to update
        @param message: Status message to display
        @param status: Status type (success, warning, error)
        """
        step_element = self.query_one(f"#{step_id}", Static)
        step_element.update(message)

        # Apply appropriate styling based on status
        step_element.remove_class("success", "warning", "error")
        step_element.add_class(status)

    def update_loading_message(self, message: str) -> None:
        """
        Update the main loading message.

        @param message: New loading message to display
        """
        try:
            loading_message = self.query_one("#loading-message", Label)
            loading_message.update(message)
        except Exception:
            pass  # Ignore if element not found

    def update_progress(self, percentage: float, status: str) -> None:
        """
        Update the overall progress bar.

        @param percentage: Progress percentage (0-100)
        @param status: Status message to display
        """
        try:
            progress_widget = self.query_one("#startup-progress", AnalysisProgressWidget)
            progress_widget.update_progress(percentage, status)
        except Exception:
            pass  # Ignore if element not found

    def update_status(self, status: str) -> None:
        """
        Update the status message.

        @param status: Status message to display
        """
        try:
            status_message = self.query_one("#status-message", Static)
            status_message.update(status)
        except Exception:
            pass  # Ignore if element not found

    def action_cancel(self) -> None:
        """Cancel the loading process."""
        self.cancelled = True
        if self.loading_task and not self.loading_task.done():
            self.loading_task.cancel()
        self.app.post_message(self.LoadingCancelled())

    class LoadingComplete(Message):
        """Message sent when loading is complete."""

        def __init__(self) -> None:
            super().__init__()

    class LoadingCancelled(Message):
        """Message sent when loading is cancelled."""

        def __init__(self) -> None:
            super().__init__()


class InitializationLoadingScreen(LoadingScreen):
    """
    Specialized loading screen for real application initialization.

    WHY: This version of the loading screen performs actual initialization
    tasks instead of just simulating them, providing real progress feedback
    during startup.
    """

    def __init__(self, config_loader_func=None, nlp_init_func=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config_loader_func = config_loader_func
        self.nlp_init_func = nlp_init_func
        self.initialization_data = {}

    async def _simulate_loading(self) -> None:
        """
        Perform actual initialization tasks with progress updates.

        WHY: This replaces the simulation with real initialization work,
        allowing the loading screen to show progress of actual operations.
        """
        progress_widget = self.query_one("#startup-progress", AnalysisProgressWidget)
        status_message = self.query_one("#status-message", Static)

        try:
            # Step 1: Initialize application (20%)
            await self._update_step("step-init", "✅ Application initialized", "success")
            status_message.update("Initializing core components...")
            progress_widget.update_progress(20, "Core components ready")
            await asyncio.sleep(0.1)

            if self.cancelled:
                return

            # Step 2: Load configuration (40%)
            status_message.update("Discovering configuration files...")
            await self._update_step("step-config", "⏳ Loading configuration...", "warning")

            if self.config_loader_func:
                config_result = await asyncio.get_event_loop().run_in_executor(
                    None, self.config_loader_func
                )
                self.initialization_data["config"] = config_result

            await self._update_step("step-config", "✅ Configuration loaded", "success")
            progress_widget.update_progress(40, "Configuration ready")
            await asyncio.sleep(0.1)

            if self.cancelled:
                return

            # Step 3: Prepare analysis engine (60%)
            await self._update_step("step-engine", "✅ Analysis engine ready", "success")
            status_message.update("Preparing Git analysis components...")
            progress_widget.update_progress(60, "Analysis engine initialized")
            await asyncio.sleep(0.2)

            if self.cancelled:
                return

            # Step 4: Load NLP models (85%) - Heavy operation
            config = self.initialization_data.get("config")
            if config and getattr(config, "qualitative", None) and config.qualitative.enabled:
                await self._update_step("step-nlp", "⏳ Loading spaCy models...", "warning")
                status_message.update("Loading natural language processing models...")
                progress_widget.update_progress(70, "Loading spaCy models...")

                if self.nlp_init_func:
                    nlp_result = await asyncio.get_event_loop().run_in_executor(
                        None, self.nlp_init_func, config
                    )
                    self.initialization_data["nlp"] = nlp_result

                await self._update_step("step-nlp", "✅ NLP models loaded", "success")
                progress_widget.update_progress(85, "NLP models ready")
            else:
                await self._update_step("step-nlp", "⏸️ NLP models skipped", "warning")
                progress_widget.update_progress(85, "NLP models skipped (qualitative disabled)")

            if self.cancelled:
                return

            # Step 5: Finalize setup (100%)
            await self._update_step("step-finalize", "✅ Setup complete", "success")
            status_message.update("Finalizing application setup...")
            progress_widget.update_progress(100, "GitFlow Analytics ready!")

            # Show completion message briefly
            loading_message = self.query_one("#loading-message", Label)
            loading_message.update("Initialization complete! Starting application...")

            elapsed_time = time.time() - self.start_time
            status_message.update(f"Ready! Initialized in {elapsed_time:.1f} seconds")

            # Wait a moment to show completion
            await asyncio.sleep(0.5)

            # Signal that loading is complete with initialization data
            self.app.post_message(self.InitializationComplete(self.initialization_data))

        except asyncio.CancelledError:
            progress_widget.update_progress(0, "Initialization cancelled")
            status_message.update("Initialization cancelled by user")
        except Exception as e:
            progress_widget.update_progress(0, f"Error: {str(e)[:50]}...")
            status_message.update(f"Initialization failed: {e}")
            self.app.notify(f"Initialization failed: {e}", severity="error")

    class InitializationComplete(Message):
        """Message sent when initialization is complete with data."""

        def __init__(self, data: dict[str, Any]) -> None:
            super().__init__()
            self.data = data
