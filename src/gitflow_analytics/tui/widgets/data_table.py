"""Enhanced data table widget for GitFlow Analytics TUI."""

from datetime import datetime
from typing import Any, Optional, Union

from textual.reactive import reactive
from textual.widgets import DataTable


class EnhancedDataTable(DataTable):
    """
    Enhanced data table with sorting, filtering, and formatting capabilities.

    WHY: The standard DataTable widget lacks features needed for displaying
    complex analytics data like sorting, filtering, and intelligent formatting
    of different data types (dates, numbers, etc.).

    DESIGN DECISION: Extends DataTable rather than creating from scratch to
    maintain compatibility with Textual's data table features while adding
    the necessary enhancements for analytics display.
    """

    DEFAULT_CSS = """
    EnhancedDataTable {
        height: auto;
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
    """

    # Reactive attributes for dynamic updates
    sort_column = reactive("")
    sort_reverse = reactive(False)
    filter_text = reactive("")

    def __init__(
        self,
        data: Optional[list[dict[str, Any]]] = None,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._raw_data = data or []
        self._filtered_data = []
        self._column_formatters = {}

    def set_data(self, data: list[dict[str, Any]]) -> None:
        """
        Set table data with automatic column detection and formatting.

        WHY: Automatically handles different data types and formats them
        appropriately for display, reducing the need for manual formatting
        in calling code.
        """
        self._raw_data = data
        if not data:
            return

        # Clear existing data
        self.clear()

        # Get columns from first row
        columns = list(data[0].keys())

        # Add columns with appropriate widths
        for col in columns:
            width = self._calculate_column_width(col, data)
            self.add_column(self._format_column_header(col), width=width, key=col)

        # Set up formatters based on data types
        self._setup_formatters(data)

        # Add data rows
        self._apply_filter_and_sort()

    def _calculate_column_width(self, column: str, data: list[dict[str, Any]]) -> int:
        """
        Calculate appropriate column width based on content.

        WHY: Dynamically sizes columns based on actual content to optimize
        display space while ensuring all content is visible.
        """
        # Start with header width
        max_width = len(self._format_column_header(column))

        # Check sample of data for width
        sample_size = min(50, len(data))
        for row in data[:sample_size]:
            value = row.get(column, "")
            formatted_value = self._format_cell_value(column, value)
            max_width = max(max_width, len(str(formatted_value)))

        # Set reasonable bounds
        return min(max(max_width + 2, 8), 50)

    def _format_column_header(self, column: str) -> str:
        """Format column header for display."""
        # Convert snake_case to Title Case
        return column.replace("_", " ").title()

    def _setup_formatters(self, data: list[dict[str, Any]]) -> None:
        """
        Set up column formatters based on data types.

        WHY: Automatically detects data types and applies appropriate formatting
        (dates, numbers, percentages) to improve readability.
        """
        if not data:
            return

        sample_row = data[0]

        for column, value in sample_row.items():
            if isinstance(value, datetime):
                self._column_formatters[column] = self._format_datetime
            elif isinstance(value, (int, float)):
                # Check if it looks like a percentage
                if any(
                    "pct" in column.lower()
                    or "percent" in column.lower()
                    or "rate" in column.lower()
                    for _ in [column]
                ):
                    self._column_formatters[column] = self._format_percentage
                else:
                    self._column_formatters[column] = self._format_number
            elif isinstance(value, str) and len(value) > 50:
                self._column_formatters[column] = self._format_long_text

    def _format_cell_value(self, column: str, value: Any) -> str:
        """Format individual cell value."""
        if value is None:
            return ""

        formatter = self._column_formatters.get(column, str)
        return formatter(value)

    def _format_datetime(self, value: datetime) -> str:
        """Format datetime values."""
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except Exception:
                return str(value)
        return value.strftime("%Y-%m-%d %H:%M")

    def _format_number(self, value: Union[int, float]) -> str:
        """Format numeric values."""
        if isinstance(value, float):
            if value >= 1000:
                return f"{value:,.1f}"
            else:
                return f"{value:.2f}"
        return f"{value:,}"

    def _format_percentage(self, value: Union[int, float]) -> str:
        """Format percentage values."""
        return f"{value:.1f}%"

    def _format_long_text(self, value: str) -> str:
        """Format long text values."""
        if len(value) > 47:
            return value[:44] + "..."
        return value

    def _apply_filter_and_sort(self) -> None:
        """
        Apply current filter and sort settings to data.

        WHY: Provides real-time filtering and sorting capabilities that are
        essential for exploring large datasets in the analytics results.
        """
        # Start with all data
        filtered_data = self._raw_data.copy()

        # Apply filter if set
        if self.filter_text:
            filtered_data = [
                row
                for row in filtered_data
                if any(self.filter_text.lower() in str(value).lower() for value in row.values())
            ]

        # Apply sort if set
        if self.sort_column and self.sort_column in (filtered_data[0] if filtered_data else {}):
            try:
                filtered_data.sort(
                    key=lambda x: x.get(self.sort_column, ""), reverse=self.sort_reverse
                )
            except TypeError:
                # Handle mixed types by converting to string
                filtered_data.sort(
                    key=lambda x: str(x.get(self.sort_column, "")), reverse=self.sort_reverse
                )

        self._filtered_data = filtered_data

        # Clear and repopulate table
        self.clear(columns=False)  # Keep columns, clear rows

        for row in filtered_data:
            formatted_row = [self._format_cell_value(col, row.get(col, "")) for col in row]
            self.add_row(*formatted_row)

    def sort_by_column(self, column: str, reverse: bool = False) -> None:
        """Sort table by specified column."""
        self.sort_column = column
        self.sort_reverse = reverse
        self._apply_filter_and_sort()

    def filter_data(self, filter_text: str) -> None:
        """Filter table data by text search."""
        self.filter_text = filter_text
        self._apply_filter_and_sort()

    def export_to_csv(self, filename: str) -> None:
        """
        Export current filtered/sorted data to CSV.

        WHY: Allows users to export the specific view they've filtered and
        sorted, maintaining their exploration context.
        """
        import csv

        if not self._filtered_data:
            return

        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            if self._filtered_data:
                fieldnames = list(self._filtered_data[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._filtered_data)

    def get_row_count(self) -> int:
        """Get number of rows currently displayed."""
        return len(self._filtered_data)

    def get_total_count(self) -> int:
        """Get total number of rows in dataset."""
        return len(self._raw_data)
