"""Common formatting utilities for report generation.

This module provides reusable formatting functions and classes
that can be used across different report generators.
"""

import json
import re
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union


class DateFormatter:
    """Utilities for formatting dates in reports."""
    
    @staticmethod
    def format_date(
        dt: Union[datetime, date, str],
        format_string: str = "%Y-%m-%d"
    ) -> str:
        """Format a date/datetime object or string.
        
        Args:
            dt: Date/datetime object or ISO string
            format_string: strftime format string
            
        Returns:
            Formatted date string
        """
        if isinstance(dt, str):
            # Parse ISO format
            if 'T' in dt:
                dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
            else:
                dt = date.fromisoformat(dt)
        
        if isinstance(dt, datetime):
            return dt.strftime(format_string)
        elif isinstance(dt, date):
            return dt.strftime(format_string)
        else:
            return str(dt)
    
    @staticmethod
    def format_date_range(
        start: Union[datetime, date],
        end: Union[datetime, date],
        separator: str = " to "
    ) -> str:
        """Format a date range.
        
        Args:
            start: Start date
            end: End date
            separator: Separator between dates
            
        Returns:
            Formatted date range string
        """
        start_str = DateFormatter.format_date(start)
        end_str = DateFormatter.format_date(end)
        return f"{start_str}{separator}{end_str}"
    
    @staticmethod
    def format_week_label(
        week_start: Union[datetime, date],
        include_year: bool = True
    ) -> str:
        """Format a week label.
        
        Args:
            week_start: Start date of the week
            include_year: Whether to include the year
            
        Returns:
            Week label (e.g., "Week 23, 2024" or "Week 23")
        """
        if isinstance(week_start, datetime):
            week_start = week_start.date()
        
        week_num = week_start.isocalendar()[1]
        
        if include_year:
            year = week_start.year
            return f"Week {week_num}, {year}"
        else:
            return f"Week {week_num}"
    
    @staticmethod
    def format_duration(seconds: float, precision: int = 1) -> str:
        """Format a duration in seconds to human-readable format.
        
        Args:
            seconds: Duration in seconds
            precision: Decimal places for fractional parts
            
        Returns:
            Formatted duration string
        """
        if seconds < 60:
            return f"{seconds:.{precision}f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.{precision}f} minutes"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.{precision}f} hours"
        else:
            days = seconds / 86400
            return f"{days:.{precision}f} days"


class NumberFormatter:
    """Utilities for formatting numbers in reports."""
    
    @staticmethod
    def format_integer(value: Union[int, float], thousands_sep: str = ",") -> str:
        """Format an integer with thousands separator.
        
        Args:
            value: Number to format
            thousands_sep: Thousands separator character
            
        Returns:
            Formatted number string
        """
        return f"{int(value):,}".replace(",", thousands_sep)
    
    @staticmethod
    def format_decimal(
        value: Union[float, Decimal],
        decimal_places: int = 2,
        thousands_sep: str = ","
    ) -> str:
        """Format a decimal number.
        
        Args:
            value: Number to format
            decimal_places: Number of decimal places
            thousands_sep: Thousands separator
            
        Returns:
            Formatted number string
        """
        formatted = f"{value:,.{decimal_places}f}"
        if thousands_sep != ",":
            formatted = formatted.replace(",", thousands_sep)
        return formatted
    
    @staticmethod
    def format_percentage(
        value: float,
        decimal_places: int = 1,
        include_sign: bool = True
    ) -> str:
        """Format a percentage value.
        
        Args:
            value: Percentage value (0.5 = 50%)
            decimal_places: Number of decimal places
            include_sign: Whether to include % sign
            
        Returns:
            Formatted percentage string
        """
        percentage = value * 100
        formatted = f"{percentage:.{decimal_places}f}"
        
        if include_sign:
            formatted += "%"
        
        return formatted
    
    @staticmethod
    def format_bytes(
        size_bytes: int,
        decimal_places: int = 2
    ) -> str:
        """Format byte size to human-readable format.
        
        Args:
            size_bytes: Size in bytes
            decimal_places: Number of decimal places
            
        Returns:
            Formatted size string
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                if unit == 'B':
                    return f"{size_bytes} {unit}"
                else:
                    return f"{size_bytes:.{decimal_places}f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.{decimal_places}f} PB"
    
    @staticmethod
    def format_change(
        value: float,
        decimal_places: int = 1,
        include_sign: bool = True,
        positive_prefix: str = "+"
    ) -> str:
        """Format a change value with sign.
        
        Args:
            value: Change value
            decimal_places: Number of decimal places
            include_sign: Whether to include +/- sign
            positive_prefix: Prefix for positive values
            
        Returns:
            Formatted change string
        """
        formatted = f"{abs(value):.{decimal_places}f}"
        
        if include_sign:
            if value > 0:
                formatted = f"{positive_prefix}{formatted}"
            elif value < 0:
                formatted = f"-{formatted}"
        
        return formatted


class TextFormatter:
    """Utilities for formatting text in reports."""
    
    @staticmethod
    def truncate(
        text: str,
        max_length: int,
        suffix: str = "..."
    ) -> str:
        """Truncate text to maximum length.
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add when truncated
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        truncate_at = max_length - len(suffix)
        return text[:truncate_at] + suffix
    
    @staticmethod
    def wrap_text(
        text: str,
        width: int = 80,
        indent: str = ""
    ) -> str:
        """Wrap text to specified width.
        
        Args:
            text: Text to wrap
            width: Maximum line width
            indent: Indentation for wrapped lines
            
        Returns:
            Wrapped text
        """
        import textwrap
        return textwrap.fill(text, width=width, subsequent_indent=indent)
    
    @staticmethod
    def sanitize_filename(
        filename: str,
        replacement: str = "_"
    ) -> str:
        """Sanitize a filename by removing invalid characters.
        
        Args:
            filename: Filename to sanitize
            replacement: Replacement for invalid characters
            
        Returns:
            Sanitized filename
        """
        # Remove invalid characters
        invalid_chars = r'[<>:"/\\|?*]'
        sanitized = re.sub(invalid_chars, replacement, filename)
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip(". ")
        
        # Limit length
        max_length = 255
        if len(sanitized) > max_length:
            name, ext = os.path.splitext(sanitized)
            name = name[:max_length - len(ext) - 1]
            sanitized = name + ext
        
        return sanitized
    
    @staticmethod
    def anonymize_email(email: str) -> str:
        """Anonymize an email address.
        
        Args:
            email: Email address to anonymize
            
        Returns:
            Anonymized email
        """
        if '@' not in email:
            return email
        
        local, domain = email.split('@', 1)
        
        if len(local) <= 3:
            anonymized_local = '*' * len(local)
        else:
            anonymized_local = local[0] + '*' * (len(local) - 2) + local[-1]
        
        return f"{anonymized_local}@{domain}"
    
    @staticmethod
    def format_list(
        items: List[str],
        separator: str = ", ",
        last_separator: str = " and ",
        max_items: Optional[int] = None
    ) -> str:
        """Format a list of items as text.
        
        Args:
            items: List of items
            separator: Separator between items
            last_separator: Separator before last item
            max_items: Maximum items to show
            
        Returns:
            Formatted list string
        """
        if not items:
            return ""
        
        if max_items and len(items) > max_items:
            shown = items[:max_items]
            remaining = len(items) - max_items
            shown.append(f"and {remaining} more")
            items = shown
        
        if len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]}{last_separator}{items[1]}"
        else:
            return separator.join(items[:-1]) + last_separator + items[-1]


class MarkdownFormatter:
    """Utilities for formatting Markdown content."""
    
    @staticmethod
    def header(text: str, level: int = 1) -> str:
        """Create a Markdown header.
        
        Args:
            text: Header text
            level: Header level (1-6)
            
        Returns:
            Markdown header
        """
        if level < 1 or level > 6:
            level = 1
        return f"{'#' * level} {text}"
    
    @staticmethod
    def bold(text: str) -> str:
        """Format text as bold.
        
        Args:
            text: Text to format
            
        Returns:
            Bold Markdown text
        """
        return f"**{text}**"
    
    @staticmethod
    def italic(text: str) -> str:
        """Format text as italic.
        
        Args:
            text: Text to format
            
        Returns:
            Italic Markdown text
        """
        return f"*{text}*"
    
    @staticmethod
    def code(text: str, language: Optional[str] = None) -> str:
        """Format text as code.
        
        Args:
            text: Code text
            language: Optional language for syntax highlighting
            
        Returns:
            Code Markdown text
        """
        if '\n' in text:
            # Code block
            if language:
                return f"```{language}\n{text}\n```"
            else:
                return f"```\n{text}\n```"
        else:
            # Inline code
            return f"`{text}`"
    
    @staticmethod
    def link(text: str, url: str, title: Optional[str] = None) -> str:
        """Create a Markdown link.
        
        Args:
            text: Link text
            url: Link URL
            title: Optional link title
            
        Returns:
            Markdown link
        """
        if title:
            return f'[{text}]({url} "{title}")'
        else:
            return f'[{text}]({url})'
    
    @staticmethod
    def list_item(text: str, level: int = 0, ordered: bool = False) -> str:
        """Create a list item.
        
        Args:
            text: Item text
            level: Indentation level
            ordered: Whether this is an ordered list
            
        Returns:
            Markdown list item
        """
        indent = "  " * level
        marker = "1." if ordered else "-"
        return f"{indent}{marker} {text}"
    
    @staticmethod
    def table(
        headers: List[str],
        rows: List[List[str]],
        alignment: Optional[List[str]] = None
    ) -> str:
        """Create a Markdown table.
        
        Args:
            headers: Table headers
            rows: Table rows
            alignment: Column alignment ('left', 'center', 'right')
            
        Returns:
            Markdown table
        """
        if not headers or not rows:
            return ""
        
        # Build header row
        header_row = "| " + " | ".join(headers) + " |"
        
        # Build separator row
        if alignment:
            separators = []
            for align in alignment:
                if align == 'center':
                    separators.append(':---:')
                elif align == 'right':
                    separators.append('---:')
                else:
                    separators.append('---')
        else:
            separators = ['---'] * len(headers)
        
        separator_row = "| " + " | ".join(separators) + " |"
        
        # Build data rows
        data_rows = []
        for row in rows:
            # Ensure row has correct number of columns
            while len(row) < len(headers):
                row.append("")
            row = row[:len(headers)]
            
            data_rows.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        # Combine all parts
        table_parts = [header_row, separator_row] + data_rows
        return "\n".join(table_parts)


class CSVFormatter:
    """Utilities for formatting CSV content."""
    
    @staticmethod
    def escape_value(value: Any) -> str:
        """Escape a value for CSV output.
        
        Args:
            value: Value to escape
            
        Returns:
            Escaped string
        """
        if value is None:
            return ""
        
        str_value = str(value)
        
        # Check if escaping is needed
        if any(char in str_value for char in [',', '"', '\n', '\r']):
            # Escape quotes
            str_value = str_value.replace('"', '""')
            # Wrap in quotes
            str_value = f'"{str_value}"'
        
        return str_value
    
    @staticmethod
    def format_row(values: List[Any], delimiter: str = ",") -> str:
        """Format a row of values as CSV.
        
        Args:
            values: Row values
            delimiter: CSV delimiter
            
        Returns:
            CSV row string
        """
        escaped = [CSVFormatter.escape_value(v) for v in values]
        return delimiter.join(escaped)


class JSONFormatter:
    """Utilities for formatting JSON content."""
    
    @staticmethod
    def format_json(
        data: Any,
        indent: int = 2,
        sort_keys: bool = False,
        ensure_ascii: bool = False
    ) -> str:
        """Format data as JSON.
        
        Args:
            data: Data to format
            indent: Indentation level
            sort_keys: Whether to sort object keys
            ensure_ascii: Whether to escape non-ASCII characters
            
        Returns:
            JSON string
        """
        def json_serializer(obj):
            """Custom JSON serializer for special types."""
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif isinstance(obj, Decimal):
                return float(obj)
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)
        
        return json.dumps(
            data,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            default=json_serializer
        )
    
    @staticmethod
    def minify_json(json_str: str) -> str:
        """Minify JSON string.
        
        Args:
            json_str: JSON string to minify
            
        Returns:
            Minified JSON string
        """
        data = json.loads(json_str)
        return json.dumps(data, separators=(',', ':'))


class MetricFormatter:
    """Utilities for formatting metrics in reports."""
    
    @staticmethod
    def format_commit_count(count: int) -> str:
        """Format commit count.
        
        Args:
            count: Number of commits
            
        Returns:
            Formatted string
        """
        if count == 0:
            return "No commits"
        elif count == 1:
            return "1 commit"
        else:
            return f"{NumberFormatter.format_integer(count)} commits"
    
    @staticmethod
    def format_line_changes(
        additions: int,
        deletions: int,
        net: bool = False
    ) -> str:
        """Format line change statistics.
        
        Args:
            additions: Number of additions
            deletions: Number of deletions
            net: Whether to show net change
            
        Returns:
            Formatted string
        """
        add_str = f"+{NumberFormatter.format_integer(additions)}"
        del_str = f"-{NumberFormatter.format_integer(deletions)}"
        
        if net:
            net_change = additions - deletions
            net_str = NumberFormatter.format_change(net_change, decimal_places=0)
            return f"{add_str} / {del_str} (net: {net_str})"
        else:
            return f"{add_str} / {del_str}"
    
    @staticmethod
    def format_velocity(
        value: float,
        unit: str = "commits/week"
    ) -> str:
        """Format velocity metric.
        
        Args:
            value: Velocity value
            unit: Unit of measurement
            
        Returns:
            Formatted string
        """
        return f"{NumberFormatter.format_decimal(value, 1)} {unit}"
    
    @staticmethod
    def format_score(
        score: float,
        max_score: float = 100,
        include_grade: bool = True
    ) -> str:
        """Format a score value.
        
        Args:
            score: Score value
            max_score: Maximum possible score
            include_grade: Whether to include letter grade
            
        Returns:
            Formatted string
        """
        percentage = (score / max_score) * 100
        formatted = f"{NumberFormatter.format_decimal(score, 1)}/{max_score}"
        
        if include_grade:
            if percentage >= 90:
                grade = "A"
            elif percentage >= 80:
                grade = "B"
            elif percentage >= 70:
                grade = "C"
            elif percentage >= 60:
                grade = "D"
            else:
                grade = "F"
            
            formatted += f" ({grade})"
        
        return formatted