"""Export modal dialog for GitFlow Analytics TUI."""

from pathlib import Path
from typing import Any, Optional

from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static, Switch


class ExportModal(ModalScreen[Optional[dict[str, Any]]]):
    """
    Modal dialog for configuring and executing data exports.

    WHY: Provides a comprehensive export interface that allows users to
    choose format, location, and export options without cluttering the
    main interface. Modal design ensures focused interaction.

    DESIGN DECISION: Returns export configuration as a dictionary rather
    than executing export directly, allowing the calling code to handle
    the actual export operation with proper error handling and progress feedback.
    """

    DEFAULT_CSS = """
    ExportModal {
        align: center middle;
    }
    
    #export-dialog {
        background: $surface;
        border: thick $primary;
        width: 80;
        height: 25;
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
    
    .button-bar {
        dock: bottom;
        height: 3;
        align: center middle;
    }
    
    .form-row {
        height: 3;
        margin: 1 0;
    }
    
    .form-label {
        width: 20;
        padding: 1 0;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "export", "Export"),
    ]

    class ExportRequested(Message):
        """Message sent when export is requested."""

        def __init__(self, config: dict[str, Any]) -> None:
            super().__init__()
            self.config = config

    def __init__(
        self,
        available_formats: Optional[list[str]] = None,
        default_path: Optional[Path] = None,
        data_info: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.available_formats = available_formats or ["CSV", "JSON", "Markdown"]
        self.default_path = default_path or Path("./reports")
        self.data_info = data_info or {}

    def compose(self):
        """Compose the export modal dialog."""
        with Container(id="export-dialog"):
            yield Label("Export Data", classes="modal-title")

            # Format selection
            yield Label("Export Format:", classes="section-title")
            format_options = [(fmt, fmt.lower()) for fmt in self.available_formats]
            yield Select(format_options, value=format_options[0][1], id="format-select")

            # File path
            yield Label("Export Location:", classes="section-title")
            with Horizontal(classes="form-row"):
                yield Label("Directory:", classes="form-label")
                yield Input(
                    value=str(self.default_path),
                    placeholder="Path to export directory",
                    id="path-input",
                )

            with Horizontal(classes="form-row"):
                yield Label("Filename:", classes="form-label")
                yield Input(
                    value=self._generate_default_filename(),
                    placeholder="Export filename",
                    id="filename-input",
                )

            # Export options
            yield Label("Export Options:", classes="section-title")

            with Horizontal(classes="form-row"):
                yield Label("Include headers:", classes="form-label")
                yield Switch(value=True, id="include-headers")

            with Horizontal(classes="form-row"):
                yield Label("Anonymize data:", classes="form-label")
                yield Switch(value=False, id="anonymize-data")

            # Data info
            if self.data_info:
                yield Label("Data Summary:", classes="section-title")
                info_text = self._format_data_info()
                yield Static(info_text, id="data-info")

            # Button bar
            with Horizontal(classes="button-bar"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Export", variant="primary", id="export-btn")

    def _generate_default_filename(self) -> str:
        """
        Generate default filename based on export format and current date.

        WHY: Provides sensible defaults to reduce user input while ensuring
        unique filenames that won't accidentally overwrite existing files.
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_type = self.data_info.get("type", "export")
        return f"{data_type}_{timestamp}"

    def _format_data_info(self) -> str:
        """Format data information for display."""
        info_lines = []

        if "row_count" in self.data_info:
            info_lines.append(f"• Rows: {self.data_info['row_count']:,}")

        if "column_count" in self.data_info:
            info_lines.append(f"• Columns: {self.data_info['column_count']}")

        if "date_range" in self.data_info:
            info_lines.append(f"• Date range: {self.data_info['date_range']}")

        if "data_types" in self.data_info:
            types_str = ", ".join(self.data_info["data_types"])
            info_lines.append(f"• Data types: {types_str}")

        return "\n".join(info_lines)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "export-btn":
            self.action_export()

    def action_cancel(self) -> None:
        """Cancel the export operation."""
        self.dismiss(None)

    def action_export(self) -> None:
        """
        Validate inputs and request export operation.

        WHY: Performs comprehensive validation before submitting export
        request to prevent errors and provide immediate feedback to users.
        """
        try:
            # Collect form data
            format_select = self.query_one("#format-select", Select)
            path_input = self.query_one("#path-input", Input)
            filename_input = self.query_one("#filename-input", Input)
            include_headers = self.query_one("#include-headers", Switch)
            anonymize_data = self.query_one("#anonymize-data", Switch)

            # Validate inputs
            export_path = Path(path_input.value.strip())
            filename = filename_input.value.strip()

            if not filename:
                self.notify("Please enter a filename", severity="error")
                return

            # Add extension if not present
            selected_format = format_select.value
            if selected_format == "csv" and not filename.lower().endswith(".csv"):
                filename += ".csv"
            elif selected_format == "json" and not filename.lower().endswith(".json"):
                filename += ".json"
            elif selected_format == "markdown" and not filename.lower().endswith(".md"):
                filename += ".md"

            full_path = export_path / filename

            # Check if file exists
            if full_path.exists():
                # In a real implementation, you'd show a confirmation dialog
                # For now, we'll just proceed with overwrite warning
                pass

            # Create export configuration
            export_config = {
                "format": selected_format,
                "path": full_path,
                "include_headers": include_headers.value,
                "anonymize": anonymize_data.value,
            }

            # Send export request message
            self.post_message(self.ExportRequested(export_config))
            self.dismiss(export_config)

        except Exception as e:
            self.notify(f"Export configuration error: {e}", severity="error")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle format selection changes."""
        if event.select.id == "format-select":
            # Update filename extension when format changes
            filename_input = self.query_one("#filename-input", Input)
            current_filename = filename_input.value

            # Remove existing extension
            name_without_ext = current_filename.rsplit(".", 1)[0]

            # Add new extension
            new_format = event.value
            if new_format == "csv":
                new_filename = f"{name_without_ext}.csv"
            elif new_format == "json":
                new_filename = f"{name_without_ext}.json"
            elif new_format == "markdown":
                new_filename = f"{name_without_ext}.md"
            else:
                new_filename = name_without_ext

            filename_input.value = new_filename

    def validate_export_path(self, path: Path) -> tuple[bool, str]:
        """
        Validate export path and return validation result.

        WHY: Prevents export failures by validating paths before attempting
        to write files, providing clear error messages to users.

        @param path: Path to validate
        @return: Tuple of (is_valid, error_message)
        """
        try:
            # Check if parent directory exists or can be created
            parent_dir = path.parent
            if not parent_dir.exists():
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    return False, f"Permission denied: Cannot create directory {parent_dir}"
                except Exception as e:
                    return False, f"Cannot create directory {parent_dir}: {e}"

            # Check write permissions
            if not parent_dir.exists():
                return False, f"Directory does not exist: {parent_dir}"

            # Try creating a test file to check permissions
            test_file = parent_dir / f".test_write_{hash(str(path))}"
            try:
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                return False, f"Permission denied: Cannot write to {parent_dir}"
            except Exception as e:
                return False, f"Cannot write to {parent_dir}: {e}"

            return True, ""

        except Exception as e:
            return False, f"Path validation error: {e}"
