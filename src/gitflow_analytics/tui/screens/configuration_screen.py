"""Configuration screen for GitFlow Analytics TUI."""

import os
from pathlib import Path
from typing import Any, Optional

from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.validation import Function, ValidationResult
from textual.widgets import Button, Input, Label, Rule, Static, Switch, TabbedContent, TabPane

from gitflow_analytics.config import Config


class ConfigurationScreen(ModalScreen[Optional[Config]]):
    """
    Modal screen for comprehensive configuration management.

    WHY: Configuration is complex with multiple categories (API keys, analysis
    settings, cache options) that benefit from a tabbed interface. Modal design
    ensures focused interaction without losing main screen context.

    DESIGN DECISION: Uses a tabbed interface to organize related settings and
    provides real-time validation to prevent configuration errors. Supports both
    loading existing configurations and creating new ones from scratch.
    """

    DEFAULT_CSS = """
    ConfigurationScreen {
        align: center middle;
    }
    
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
    
    .section-title {
        text-style: bold;
        color: $secondary;
        margin: 1 0;
    }
    
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
    
    .button-bar {
        dock: bottom;
        height: 3;
        align: center middle;
    }
    
    .help-text {
        color: $text-muted;
        text-style: italic;
        margin: 0 0 1 0;
    }
    
    .validation-error {
        color: $error;
        text-style: bold;
    }
    
    .validation-success {
        color: $success;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save Config"),
        Binding("ctrl+t", "test_connection", "Test Connections"),
    ]

    def __init__(self, config_path: Optional[Path] = None, config: Optional[Config] = None) -> None:
        super().__init__()
        self.config_path = config_path
        self.config = config
        self.form_data = {}
        self.validation_errors = {}

    def compose(self):
        """Compose the configuration modal."""
        with Container(id="config-modal"):
            yield Label("GitFlow Analytics Configuration", classes="modal-title")

            with TabbedContent():
                # Basic Settings Tab
                with TabPane("Basic Settings", id="basic"):
                    yield from self._compose_basic_settings()

                # API Keys Tab
                with TabPane("API Keys", id="api-keys"):
                    yield from self._compose_api_keys()

                # Analysis Settings Tab
                with TabPane("Analysis", id="analysis"):
                    yield from self._compose_analysis_settings()

                # Cache Settings Tab
                with TabPane("Cache", id="cache"):
                    yield from self._compose_cache_settings()

                # Repository Settings Tab
                with TabPane("Repositories", id="repositories"):
                    yield from self._compose_repository_settings()

            # Validation status
            with Container(id="validation-status"):
                yield Static("", id="validation-message")

            # Button bar
            with Horizontal(classes="button-bar"):
                yield Button("Test Connections", id="test-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Save Configuration", variant="primary", id="save-btn")

    def _compose_basic_settings(self):
        """Compose basic settings tab."""
        # Configuration file path
        yield Label("Configuration File:", classes="section-title")
        yield Static("Path where the configuration will be saved", classes="help-text")

        with Horizontal(classes="form-row"):
            yield Label("Config Path:", classes="form-label")
            yield Input(
                value=str(self.config_path) if self.config_path else "config.yaml",
                placeholder="Path to config.yaml file",
                classes="form-input",
                id="config-path",
            )

        # Analysis period
        yield Rule()
        yield Label("Analysis Settings:", classes="section-title")

        with Horizontal(classes="form-row"):
            yield Label("Analysis Period (weeks):", classes="form-label")
            yield Input(
                value="12",
                placeholder="12",
                classes="form-input",
                id="weeks",
                validators=[
                    Function(self._validate_positive_integer, "Must be a positive integer")
                ],
            )

        with Horizontal(classes="form-row"):
            yield Label("Enable Qualitative Analysis:", classes="form-label")
            yield Switch(value=True, id="enable-qualitative")

        yield Static(
            "Qualitative analysis provides deeper insights into commit patterns and code changes",
            classes="help-text",
        )

    def _compose_api_keys(self):
        """Compose API keys tab."""
        # GitHub
        yield Label("GitHub Configuration:", classes="section-title")
        yield Static(
            "Required for accessing GitHub repositories and pull request data", classes="help-text"
        )

        with Horizontal(classes="form-row"):
            yield Label("Personal Access Token:", classes="form-label")
            yield Input(
                placeholder="ghp_xxxxxxxxxxxxxxxxxxxx",
                password=True,
                classes="form-input",
                id="github-token",
            )

        with Horizontal(classes="form-row"):
            yield Label("Organization (optional):", classes="form-label")
            yield Input(placeholder="your-organization", classes="form-input", id="github-org")

        # OpenRouter (for qualitative analysis)
        yield Rule()
        yield Label("OpenRouter Configuration:", classes="section-title")
        yield Static(
            "Required for AI-powered qualitative analysis of commit messages", classes="help-text"
        )

        with Horizontal(classes="form-row"):
            yield Label("API Key:", classes="form-label")
            yield Input(
                placeholder="sk-or-xxxxxxxxxxxxxxxxxxxx",
                password=True,
                classes="form-input",
                id="openrouter-key",
            )

        # JIRA
        yield Rule()
        yield Label("JIRA Configuration (Optional):", classes="section-title")
        yield Static(
            "Optional: Connect to JIRA for enhanced ticket tracking and story point analysis",
            classes="help-text",
        )

        with Horizontal(classes="form-row"):
            yield Label("Base URL:", classes="form-label")
            yield Input(
                placeholder="https://company.atlassian.net", classes="form-input", id="jira-url"
            )

        with Horizontal(classes="form-row"):
            yield Label("Username/Email:", classes="form-label")
            yield Input(placeholder="your.email@company.com", classes="form-input", id="jira-user")

        with Horizontal(classes="form-row"):
            yield Label("API Token:", classes="form-label")
            yield Input(
                placeholder="ATATT3xFfGF0...", password=True, classes="form-input", id="jira-token"
            )

    def _compose_analysis_settings(self):
        """Compose analysis settings tab."""
        yield Label("Developer Identity Resolution:", classes="section-title")

        with Horizontal(classes="form-row"):
            yield Label("Similarity Threshold:", classes="form-label")
            yield Input(
                value="0.85",
                placeholder="0.85",
                classes="form-input",
                id="similarity-threshold",
                validators=[Function(self._validate_float_0_1, "Must be between 0.0 and 1.0")],
            )

        yield Static(
            "Threshold for fuzzy matching of developer names (0.0 = loose, 1.0 = exact)",
            classes="help-text",
        )

        yield Rule()
        yield Label("Ticket Platform Settings:", classes="section-title")

        with Horizontal(classes="form-row"):
            yield Label("Allowed Platforms:", classes="form-label")
            yield Input(
                placeholder="jira,github,linear (comma-separated)",
                classes="form-input",
                id="ticket-platforms",
            )

        yield Rule()
        yield Label("Path Exclusions:", classes="section-title")

        with Horizontal(classes="form-row"):
            yield Label("Exclude Paths:", classes="form-label")
            yield Input(
                placeholder="node_modules,dist,build (comma-separated)",
                classes="form-input",
                id="exclude-paths",
            )

    def _compose_cache_settings(self):
        """Compose cache settings tab."""
        yield Label("Cache Configuration:", classes="section-title")
        yield Static(
            "Caching improves performance by storing analyzed data locally", classes="help-text"
        )

        with Horizontal(classes="form-row"):
            yield Label("Cache Directory:", classes="form-label")
            yield Input(
                value=".gitflow-cache",
                placeholder=".gitflow-cache",
                classes="form-input",
                id="cache-dir",
            )

        with Horizontal(classes="form-row"):
            yield Label("Cache TTL (hours):", classes="form-label")
            yield Input(
                value="168",
                placeholder="168",
                classes="form-input",
                id="cache-ttl",
                validators=[
                    Function(self._validate_positive_integer, "Must be a positive integer")
                ],
            )

        with Horizontal(classes="form-row"):
            yield Label("Clear cache on startup:", classes="form-label")
            yield Switch(value=False, id="clear-cache")

        yield Static(
            "168 hours = 1 week. Cached data older than this will be refreshed.",
            classes="help-text",
        )

    def _compose_repository_settings(self):
        """Compose repository settings tab."""
        yield Label("Repository Discovery:", classes="section-title")
        yield Static("Configure how repositories are discovered and analyzed", classes="help-text")

        with Horizontal(classes="form-row"):
            yield Label("Auto-discover from org:", classes="form-label")
            yield Switch(value=True, id="auto-discover")

        yield Static(
            "When enabled, automatically discovers all repositories from the GitHub organization",
            classes="help-text",
        )

        yield Rule()
        yield Label("Manual Repository Configuration:", classes="section-title")
        yield Static(
            "Add individual repositories manually (overrides organization discovery)",
            classes="help-text",
        )

        # TODO: Add dynamic repository list management
        yield Static(
            "Manual repository configuration will be available in a future version",
            classes="help-text",
        )

    def _validate_positive_integer(self, value: str) -> ValidationResult:
        """Validate positive integer input."""
        try:
            int_val = int(value)
            if int_val > 0:
                return ValidationResult.success()
            else:
                return ValidationResult.error("Must be greater than 0")
        except ValueError:
            return ValidationResult.error("Must be a valid integer")

    def _validate_float_0_1(self, value: str) -> ValidationResult:
        """Validate float value between 0.0 and 1.0."""
        try:
            float_val = float(value)
            if 0.0 <= float_val <= 1.0:
                return ValidationResult.success()
            else:
                return ValidationResult.error("Must be between 0.0 and 1.0")
        except ValueError:
            return ValidationResult.error("Must be a valid number")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "save-btn":
            self.action_save()
        elif event.button.id == "test-btn":
            self.action_test_connection()

    def action_cancel(self) -> None:
        """Cancel configuration changes."""
        self.dismiss(None)

    def action_save(self) -> None:
        """
        Save configuration after validation.

        WHY: Performs comprehensive validation before saving to prevent
        runtime errors and provides clear feedback about any issues.
        """
        try:
            # Collect form data
            form_data = self._collect_form_data()

            # Validate form data
            validation_result = self._validate_form_data(form_data)
            if not validation_result["valid"]:
                self._show_validation_errors(validation_result["errors"])
                return

            # Update environment variables
            self._update_env_vars(form_data)

            # Create or load configuration
            config = self._create_config_from_form(form_data)

            # Save configuration file if path provided
            config_path = form_data.get("config_path")
            if config_path:
                config_path = Path(config_path)
                # TODO: Actually save config to file
                # ConfigLoader.save(config, config_path)

            self.dismiss(config)

        except Exception as e:
            self.notify(f"Configuration error: {e}", severity="error")

    def action_test_connection(self) -> None:
        """Test API connections with current settings."""
        self.notify("Connection testing not yet implemented", severity="info")
        # TODO: Implement connection testing

    def _collect_form_data(self) -> dict[str, Any]:
        """Collect all form data from inputs."""
        return {
            "config_path": self.query_one("#config-path", Input).value,
            "weeks": self.query_one("#weeks", Input).value,
            "enable_qualitative": self.query_one("#enable-qualitative", Switch).value,
            "github_token": self.query_one("#github-token", Input).value,
            "github_org": self.query_one("#github-org", Input).value,
            "openrouter_key": self.query_one("#openrouter-key", Input).value,
            "jira_url": self.query_one("#jira-url", Input).value,
            "jira_user": self.query_one("#jira-user", Input).value,
            "jira_token": self.query_one("#jira-token", Input).value,
            "similarity_threshold": self.query_one("#similarity-threshold", Input).value,
            "ticket_platforms": self.query_one("#ticket-platforms", Input).value,
            "exclude_paths": self.query_one("#exclude-paths", Input).value,
            "cache_dir": self.query_one("#cache-dir", Input).value,
            "cache_ttl": self.query_one("#cache-ttl", Input).value,
            "clear_cache": self.query_one("#clear-cache", Switch).value,
            "auto_discover": self.query_one("#auto-discover", Switch).value,
        }

    def _validate_form_data(self, form_data: dict[str, Any]) -> dict[str, Any]:
        """Validate collected form data."""
        errors = []

        # Required fields validation
        if not form_data["config_path"]:
            errors.append("Configuration path is required")

        # GitHub token validation (if qualitative analysis enabled)
        if form_data["enable_qualitative"] and not form_data["openrouter_key"]:
            errors.append("OpenRouter API key is required for qualitative analysis")

        # Numeric field validation
        try:
            weeks = int(form_data["weeks"])
            if weeks <= 0:
                errors.append("Analysis period must be greater than 0")
        except ValueError:
            errors.append("Analysis period must be a valid number")

        try:
            threshold = float(form_data["similarity_threshold"])
            if not 0.0 <= threshold <= 1.0:
                errors.append("Similarity threshold must be between 0.0 and 1.0")
        except ValueError:
            errors.append("Similarity threshold must be a valid number")

        try:
            ttl = int(form_data["cache_ttl"])
            if ttl <= 0:
                errors.append("Cache TTL must be greater than 0")
        except ValueError:
            errors.append("Cache TTL must be a valid number")

        return {"valid": len(errors) == 0, "errors": errors}

    def _show_validation_errors(self, errors: list) -> None:
        """Display validation errors to user."""
        error_text = "Configuration validation failed:\\n" + "\\n".join(
            f"• {error}" for error in errors
        )
        validation_msg = self.query_one("#validation-message", Static)
        validation_msg.update(error_text)
        validation_msg.add_class("validation-error")
        validation_msg.remove_class("validation-success")

    def _update_env_vars(self, form_data: dict[str, Any]) -> None:
        """Update environment variables with provided values."""
        env_updates = {}

        if form_data["github_token"]:
            env_updates["GITHUB_TOKEN"] = form_data["github_token"]
        if form_data["github_org"]:
            env_updates["GITHUB_ORG"] = form_data["github_org"]
        if form_data["openrouter_key"]:
            env_updates["OPENROUTER_API_KEY"] = form_data["openrouter_key"]
        if form_data["jira_user"]:
            env_updates["JIRA_ACCESS_USER"] = form_data["jira_user"]
        if form_data["jira_token"]:
            env_updates["JIRA_ACCESS_TOKEN"] = form_data["jira_token"]

        # Update current environment
        os.environ.update(env_updates)

    def _create_config_from_form(self, form_data: dict[str, Any]) -> Config:
        """
        Create a Config object from form data.

        WHY: Centralizes the logic for converting form inputs into a proper
        configuration object, ensuring consistency and making it easier to
        maintain as the configuration schema evolves.
        """
        # This is a simplified version - in reality, you'd create a proper Config object
        # with all the necessary structure and validation

        # For now, return None and indicate this needs implementation
        # TODO: Implement proper config creation from form data
        self.notify(
            "Configuration creation from form data not yet fully implemented", severity="warning"
        )
        return None
