"""Main TUI application for GitFlow Analytics."""

import asyncio
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding

from .screens.main_screen import MainScreen
from .screens.configuration_screen import ConfigurationScreen
from .screens.analysis_progress_screen import AnalysisProgressScreen
from .screens.results_screen import ResultsScreen
from gitflow_analytics.config import ConfigLoader, Config


class GitFlowAnalyticsApp(App):
    """
    Main Terminal User Interface application for GitFlow Analytics.
    
    WHY: Provides a comprehensive TUI that guides users through the entire
    analytics workflow from configuration to results analysis. Designed to
    be more user-friendly than command-line interface while maintaining
    the power and flexibility of the core analysis engine.
    
    DESIGN DECISION: Uses a screen-based navigation model where each major
    workflow step (configuration, analysis, results) has its own dedicated
    screen. This provides clear context separation and allows for complex
    interactions within each workflow step.
    """
    
    TITLE = "GitFlow Analytics"
    SUB_TITLE = "Developer Productivity Analysis"
    
    CSS = """
    /* Global styles */
    .screen-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin: 1;
        padding: 1;
    }
    
    .section-title {
        text-style: bold;
        color: $secondary;
        margin: 1 0;
    }
    
    .subsection-title {
        text-style: bold;
        color: $warning;
        margin: 1 0 0 0;
    }
    
    .help-text {
        color: $text-muted;
        text-style: italic;
        margin: 0 0 1 0;
    }
    
    .info-message {
        color: $warning;
        text-style: italic;
        margin: 1 0;
    }
    
    /* Panel styles */
    .status-panel {
        border: solid $primary;
        margin: 1;
        padding: 1;
        height: auto;
    }
    
    .actions-panel {
        border: solid $secondary;
        margin: 1;
        padding: 1;
        height: auto;
    }
    
    .stats-panel {
        border: solid $accent;
        margin: 1;
        padding: 1;
        height: auto;
    }
    
    .progress-panel {
        height: 8;
        border: solid $primary;
        margin: 1;
        padding: 1;
    }
    
    .log-panel {
        border: solid $accent;
        margin: 1;
        padding: 1;
        min-height: 10;
    }
    
    .panel-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    /* Form styles */
    .form-row {
        height: 3;
        margin: 1 0;
    }
    
    .form-label {
        width: 25;
        padding: 1 0;
    }
    
    .form-input {
        width: 1fr;
    }
    
    /* Button styles */
    .button-bar {
        dock: bottom;
        height: 3;
        align: center middle;
        margin: 1;
    }
    
    .action-bar {
        height: 3;
        align: center middle;
        margin: 1 0;
    }
    
    /* Modal styles */
    #config-modal {
        background: $surface;
        border: thick $primary;
        width: 90%;
        height: 85%;
        padding: 1;
    }
    
    .modal-title {
        text-align: center;
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    /* Validation styles */
    .validation-error {
        color: $error;
        text-style: bold;
    }
    
    .validation-success {
        color: $success;
        text-style: bold;
    }
    
    /* Table styles */
    EnhancedDataTable {
        height: auto;
        min-height: 20;
    }
    
    EnhancedDataTable > .datatable--header {
        background: $primary 10%;
        color: $primary;
        text-style: bold;
    }
    
    EnhancedDataTable > .datatable--row-hover {
        background: $accent 20%;
    }
    
    EnhancedDataTable > .datatable--row-cursor {
        background: $primary 30%;
    }
    
    /* Progress widget styles */
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
    
    /* Utility classes */
    .hidden {
        display: none;
    }
    
    .center {
        text-align: center;
    }
    
    .bold {
        text-style: bold;
    }
    
    .italic {
        text-style: italic;
    }
    
    .error {
        color: $error;
    }
    
    .success {
        color: $success;
    }
    
    .warning {
        color: $warning;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("f1", "help", "Help"),
        Binding("ctrl+d", "toggle_dark", "Toggle Dark Mode"),
    ]
    
    def __init__(self) -> None:
        super().__init__()
        self.config: Optional[Config] = None
        self.config_path: Optional[Path] = None
        
    def compose(self) -> ComposeResult:
        """
        Compose the main application with the initial main screen.
        
        WHY: Starts with the main screen which serves as the primary
        navigation hub, allowing users to see configuration status
        and access all major functionality.
        """
        # Start with main screen
        yield MainScreen(self.config, self.config_path, id="main-screen")
    
    def on_mount(self) -> None:
        """
        Handle application startup and auto-load configuration.
        
        WHY: Attempts to load default configurations automatically to
        provide a seamless user experience when configurations are
        available in standard locations.
        """
        # Try to load default configuration
        self._try_load_default_config()
        
        # Set up application title
        self.title = "GitFlow Analytics"
        self.sub_title = "Developer Productivity Analysis"
    
    def _try_load_default_config(self) -> None:
        """
        Attempt to load configuration from default locations.
        
        WHY: Provides automatic configuration discovery to reduce setup
        friction for users who have configurations in standard locations.
        """
        default_config_paths = [
            Path("config.yaml"),
            Path("gitflow.yaml"),
            Path(".gitflow/config.yaml"),
            Path("~/.gitflow/config.yaml").expanduser(),
        ]
        
        for config_path in default_config_paths:
            if config_path.exists():
                try:
                    self.config = ConfigLoader.load(config_path)
                    self.config_path = config_path
                    
                    # Update main screen with loaded config
                    main_screen = self.query_one("#main-screen", MainScreen)
                    main_screen.update_config(self.config, self.config_path)
                    
                    self.notify(f"Loaded configuration from {config_path}", severity="info")
                    return
                    
                except Exception as e:
                    self.notify(f"Failed to load {config_path}: {e}", severity="warning")
                    continue
    
    def on_main_screen_new_analysis_requested(self, message: MainScreen.NewAnalysisRequested) -> None:
        """Handle new analysis request from main screen."""
        if not self.config:
            self.notify("Please load or create a configuration first", severity="error")
            return
        
        # Launch analysis progress screen
        analysis_screen = AnalysisProgressScreen(
            config=self.config,
            weeks=12,  # TODO: Get from config or user preference
            enable_qualitative=getattr(self.config, 'qualitative', None) and self.config.qualitative.enabled
        )
        
        self.push_screen(analysis_screen)
    
    def on_main_screen_configuration_requested(self, message: MainScreen.ConfigurationRequested) -> None:
        """Handle configuration request from main screen."""
        config_screen = ConfigurationScreen(
            config_path=self.config_path,
            config=self.config
        )
        
        def handle_config_result(config: Optional[Config]) -> None:
            if config:
                self.config = config
                # TODO: Set config_path if provided
                
                # Update main screen
                main_screen = self.query_one("#main-screen", MainScreen)
                main_screen.update_config(self.config, self.config_path)
                
                self.notify("Configuration updated successfully", severity="success")
        
        self.push_screen(config_screen, handle_config_result)
    
    def on_main_screen_cache_status_requested(self, message: MainScreen.CacheStatusRequested) -> None:
        """Handle cache status request from main screen."""
        if not self.config:
            self.notify("No configuration loaded", severity="error")
            return
        
        try:
            from gitflow_analytics.core.cache import GitAnalysisCache
            
            cache = GitAnalysisCache(self.config.cache.directory)
            stats = cache.get_cache_stats()
            
            # Calculate cache size
            import os
            cache_size = 0
            try:
                for root, _dirs, files in os.walk(self.config.cache.directory):
                    for f in files:
                        cache_size += os.path.getsize(os.path.join(root, f))
                cache_size_mb = cache_size / 1024 / 1024
            except:
                cache_size_mb = 0
            
            message_text = f"""Cache Statistics:
• Location: {self.config.cache.directory}
• Cached commits: {stats['cached_commits']:,}
• Cached PRs: {stats['cached_prs']:,}
• Cached issues: {stats['cached_issues']:,}
• Stale entries: {stats['stale_commits']:,}
• Cache size: {cache_size_mb:.1f} MB
• TTL: {self.config.cache.ttl_hours} hours"""
            
            self.notify(message_text, severity="info")
            
        except Exception as e:
            self.notify(f"Failed to get cache statistics: {e}", severity="error")
    
    def on_main_screen_identity_management_requested(self, message: MainScreen.IdentityManagementRequested) -> None:
        """Handle identity management request from main screen."""
        if not self.config:
            self.notify("No configuration loaded", severity="error")
            return
        
        try:
            from gitflow_analytics.core.identity import DeveloperIdentityResolver
            
            identity_resolver = DeveloperIdentityResolver(
                self.config.cache.directory / "identities.db"
            )
            
            developers = identity_resolver.get_developer_stats()
            
            if not developers:
                self.notify("No developer identities found. Run analysis first.", severity="info")
                return
            
            # Show top developers
            top_devs = sorted(developers, key=lambda d: d['total_commits'], reverse=True)[:10]
            
            dev_list = []
            for dev in top_devs:
                dev_list.append(f"• {dev['primary_name']}: {dev['total_commits']} commits, {dev['alias_count']} aliases")
            
            message_text = f"""Developer Identity Statistics:
• Total unique developers: {len(developers)}
• Manual mappings: {len(self.config.analysis.manual_identity_mappings) if self.config.analysis.manual_identity_mappings else 0}

Top Contributors:
{chr(10).join(dev_list)}

Use the CLI 'merge-identity' command to merge duplicate identities."""
            
            self.notify(message_text, severity="info")
            
        except Exception as e:
            self.notify(f"Failed to get identity information: {e}", severity="error")
    
    def on_main_screen_help_requested(self, message: MainScreen.HelpRequested) -> None:
        """Handle help request from main screen."""
        help_text = """GitFlow Analytics - Terminal UI Help

🚀 Getting Started:
1. Load or create a configuration file (API keys, repositories)
2. Configure analysis settings (time period, options)
3. Run analysis to process your repositories
4. Explore results and export reports

⌨️  Key Bindings:
• Ctrl+Q / Ctrl+C: Quit application
• F1: Show this help
• Ctrl+D: Toggle dark/light mode
• Escape: Go back/cancel current action

📁 Configuration:
• GitHub Personal Access Token required
• OpenRouter API key for qualitative analysis
• JIRA optional for enhanced ticket tracking

🔧 Analysis Features:
• Git commit analysis with developer identity resolution
• Pull request metrics and timing analysis
• Qualitative analysis using AI for commit categorization
• DORA metrics calculation
• Story point tracking from commit messages

📊 Export Options:
• CSV reports for spreadsheet analysis
• JSON export for API integration
• Markdown reports for documentation

💡 Tips:
• Use organization auto-discovery for multiple repositories
• Enable qualitative analysis for deeper insights
• Review identity mappings for accurate attribution

For more information: https://github.com/bobmatnyc/gitflow-analytics"""
        
        self.notify(help_text, severity="info")
    
    def action_quit(self) -> None:
        """
        Quit the application with confirmation if analysis is running.
        
        WHY: Provides safe exit that checks for running operations to
        prevent data loss or corruption from incomplete analysis.
        """
        # Check if analysis is running
        try:
            analysis_screen = self.query("AnalysisProgressScreen")
            if analysis_screen:
                # TODO: Show confirmation dialog
                pass
        except:
            pass
        
        self.exit()
    
    def action_help(self) -> None:
        """Show application help."""
        # Trigger help from main screen if available
        try:
            main_screen = self.query_one("#main-screen", MainScreen)
            main_screen.action_help()
        except:
            # Fallback to direct help
            self.on_main_screen_help_requested(MainScreen.HelpRequested())
    
    def action_toggle_dark(self) -> None:
        """
        Toggle between dark and light themes.
        
        WHY: Provides theme flexibility for different user preferences
        and working environments (bright vs dim lighting conditions).
        """
        self.dark = not self.dark
        theme = "dark" if self.dark else "light"
        self.notify(f"Switched to {theme} theme", severity="info")
    
    def get_current_config(self) -> Optional[Config]:
        """Get the currently loaded configuration."""
        return self.config
    
    def get_current_config_path(self) -> Optional[Path]:
        """Get the path of the currently loaded configuration."""
        return self.config_path
    
    def update_config(self, config: Config, config_path: Optional[Path] = None) -> None:
        """
        Update the application configuration and refresh relevant screens.
        
        WHY: Provides centralized configuration updates that can be called
        from any screen to ensure all parts of the application stay in sync.
        """
        self.config = config
        self.config_path = config_path
        
        # Update main screen if visible
        try:
            main_screen = self.query_one("#main-screen", MainScreen)
            main_screen.update_config(config, config_path)
        except:
            pass
    
    async def run_analysis_async(
        self, 
        weeks: int = 12, 
        enable_qualitative: bool = True
    ) -> Optional[dict]:
        """
        Run analysis asynchronously and return results.
        
        WHY: Provides programmatic access to analysis functionality
        for integration with other systems or automated workflows.
        
        @param weeks: Number of weeks to analyze
        @param enable_qualitative: Whether to enable qualitative analysis
        @return: Analysis results dictionary or None if failed
        """
        if not self.config:
            raise ValueError("No configuration loaded")
        
        # Create analysis screen
        analysis_screen = AnalysisProgressScreen(
            config=self.config,
            weeks=weeks,
            enable_qualitative=enable_qualitative
        )
        
        # Run analysis (this would need to be implemented properly)
        # For now, return None to indicate not implemented
        return None


def main() -> None:
    """
    Main entry point for the TUI application.
    
    WHY: Provides a clean entry point that can be called from the CLI
    or used as a standalone application launcher.
    """
    app = GitFlowAnalyticsApp()
    app.run()


if __name__ == "__main__":
    main()