"""Main screen for GitFlow Analytics TUI."""

from pathlib import Path
from typing import Optional

from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, Rule, Static

from gitflow_analytics.config import Config


class MainScreen(Screen):
    """
    Main dashboard screen showing project information and navigation options.

    WHY: Serves as the primary entry point for the TUI, providing users with
    an overview of their configuration and clear navigation to all major features.

    DESIGN DECISION: Uses a dashboard layout rather than a menu-driven approach
    to provide immediate visibility into the current configuration status and
    quick access to common operations.
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+n", "new_analysis", "New Analysis"),
        Binding("ctrl+o", "open_config", "Open Config"),
        Binding("f1", "help", "Help"),
        Binding("c", "cache_status", "Cache Status"),
        Binding("i", "manage_identities", "Identities"),
    ]

    class NewAnalysisRequested(Message):
        """Message sent when new analysis is requested."""

        def __init__(self) -> None:
            super().__init__()

    class ConfigurationRequested(Message):
        """Message sent when configuration is requested."""

        def __init__(self) -> None:
            super().__init__()

    class CacheStatusRequested(Message):
        """Message sent when cache status is requested."""

        def __init__(self) -> None:
            super().__init__()

    class IdentityManagementRequested(Message):
        """Message sent when identity management is requested."""

        def __init__(self) -> None:
            super().__init__()

    class HelpRequested(Message):
        """Message sent when help is requested."""

        def __init__(self) -> None:
            super().__init__()

    def __init__(
        self,
        config: Optional[Config] = None,
        config_path: Optional[Path] = None,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, id=id)
        self.config = config
        self.config_path = config_path

    def compose(self):
        """Compose the main screen."""
        yield Header()

        with Container(id="main-container"):
            yield Label("GitFlow Analytics", classes="screen-title")
            yield Static("Developer Productivity Analysis Tool", id="subtitle")

            # Configuration status section
            with Container(classes="status-panel"):
                yield Label("Configuration Status", classes="section-title")

                if self.config:
                    config_status = (
                        f"✅ Loaded: {self.config_path}"
                        if self.config_path
                        else "✅ Configuration loaded"
                    )
                    yield Static(config_status, id="config-status")

                    # Show key configuration details
                    details = self._get_config_summary()
                    yield Static(details, id="config-details")
                else:
                    yield Static("❌ No configuration loaded", id="config-status")
                    yield Static(
                        "Load or create a configuration to begin analysis", id="config-help"
                    )

            yield Rule()

            # Main actions section
            with Container(classes="actions-panel"):
                yield Label("Available Actions", classes="section-title")

                with Vertical(id="action-buttons"):
                    if self.config:
                        yield Button("🚀 Run Analysis", variant="primary", id="run-analysis")
                        yield Button("📊 View Cache Status", id="cache-status")
                        yield Button("👥 Manage Developer Identities", id="manage-identities")
                        yield Button("⚙️ Edit Configuration", id="edit-config")
                    else:
                        yield Button("📁 Load Configuration", variant="primary", id="load-config")
                        yield Button("➕ Create New Configuration", id="new-config")

                    yield Rule()
                    yield Button("❓ Help & Documentation", id="help")

            # Quick stats section (if config loaded)
            if self.config:
                with Container(classes="stats-panel"):
                    yield Label("Quick Information", classes="section-title")
                    stats = self._get_quick_stats()
                    yield Static(stats, id="quick-stats")

        yield Footer()

    def _get_config_summary(self) -> str:
        """
        Generate configuration summary for display.

        WHY: Provides users with immediate visibility into their current
        configuration without needing to navigate to a separate screen.
        """
        if not self.config:
            return ""

        lines = []

        # Repository count
        repo_count = len(self.config.repositories) if self.config.repositories else 0
        if self.config.github.organization and not self.config.repositories:
            lines.append(f"• Organization: {self.config.github.organization} (auto-discovery)")
        else:
            lines.append(f"• Repositories: {repo_count} configured")

        # GitHub integration
        if self.config.github.token:
            lines.append("✅ GitHub API configured")
        else:
            lines.append("⚠️ GitHub API not configured")

        # Qualitative analysis
        if (
            hasattr(self.config, "qualitative")
            and self.config.qualitative
            and self.config.qualitative.enabled
        ):
            lines.append("✅ Qualitative analysis enabled")
        else:
            lines.append("⚠️ Qualitative analysis disabled")

        # JIRA integration
        if hasattr(self.config, "jira") and self.config.jira and self.config.jira.base_url:
            lines.append("✅ JIRA integration configured")
        else:
            lines.append("⚠️ JIRA integration not configured")

        # Cache configuration
        lines.append(f"• Cache TTL: {self.config.cache.ttl_hours}h")

        return "\n".join(lines)

    def _get_quick_stats(self) -> str:
        """
        Generate quick statistics if cache data is available.

        WHY: Shows users what data is already available from previous runs,
        helping them understand if they need to run a new analysis.
        """
        try:
            from ....core.cache import GitAnalysisCache

            cache = GitAnalysisCache(self.config.cache.directory)
            stats = cache.get_cache_stats()

            lines = []
            lines.append(f"• Cached commits: {stats['cached_commits']:,}")
            lines.append(f"• Cached PRs: {stats['cached_prs']:,}")
            lines.append(f"• Cached issues: {stats['cached_issues']:,}")

            if stats["stale_commits"] > 0:
                lines.append(f"• Stale entries: {stats['stale_commits']:,}")

            # Cache size
            import os

            cache_size = 0
            try:
                for root, _dirs, files in os.walk(self.config.cache.directory):
                    for f in files:
                        cache_size += os.path.getsize(os.path.join(root, f))
                cache_size_mb = cache_size / 1024 / 1024
                lines.append(f"• Cache size: {cache_size_mb:.1f} MB")
            except Exception:
                pass

            return "\n".join(lines)

        except Exception:
            return "Cache statistics unavailable"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_actions = {
            "run-analysis": self._run_analysis,
            "load-config": self._load_config,
            "new-config": self._new_config,
            "edit-config": self._edit_config,
            "cache-status": self._cache_status,
            "manage-identities": self._manage_identities,
            "help": self._show_help,
        }

        action = button_actions.get(event.button.id)
        if action:
            action()

    def _run_analysis(self) -> None:
        """Request new analysis."""
        if not self.config:
            self.notify("Please load or create a configuration first", severity="error")
            return
        self.post_message(self.NewAnalysisRequested())

    def _load_config(self) -> None:
        """Request configuration loading."""
        self.post_message(self.ConfigurationRequested())

    def _new_config(self) -> None:
        """Request new configuration creation."""
        self.post_message(self.ConfigurationRequested())

    def _edit_config(self) -> None:
        """Request configuration editing."""
        self.post_message(self.ConfigurationRequested())

    def _cache_status(self) -> None:
        """Request cache status display."""
        self.post_message(self.CacheStatusRequested())

    def _manage_identities(self) -> None:
        """Request identity management."""
        self.post_message(self.IdentityManagementRequested())

    def _show_help(self) -> None:
        """Request help display."""
        self.post_message(self.HelpRequested())

    # Action handlers for key bindings
    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()

    def action_new_analysis(self) -> None:
        """Start new analysis via keyboard shortcut."""
        self._run_analysis()

    def action_open_config(self) -> None:
        """Open configuration via keyboard shortcut."""
        self._load_config()

    def action_help(self) -> None:
        """Show help via keyboard shortcut."""
        self._show_help()

    def action_cache_status(self) -> None:
        """Show cache status via keyboard shortcut."""
        self._cache_status()

    def action_manage_identities(self) -> None:
        """Manage identities via keyboard shortcut."""
        self._manage_identities()

    def update_config(self, config: Config, config_path: Optional[Path] = None) -> None:
        """
        Update the configuration and refresh the display.

        WHY: Allows the main screen to be updated when configuration changes
        without requiring a full screen rebuild, maintaining user context.
        """
        self.config = config
        self.config_path = config_path

        # Update configuration status
        if self.config:
            config_status = (
                f"✅ Loaded: {self.config_path}" if self.config_path else "✅ Configuration loaded"
            )
            self.query_one("#config-status", Static).update(config_status)

            # Update configuration details
            details = self._get_config_summary()
            self.query_one("#config-details", Static).update(details)

            # Update quick stats if available
            try:
                stats = self._get_quick_stats()
                self.query_one("#quick-stats", Static).update(stats)
            except Exception:
                pass

        # Refresh to rebuild buttons with new state
        self.refresh(recompose=True)
