"""Configuration error handling and user-friendly error messages."""

from pathlib import Path
from typing import Any, Optional

import yaml


class ConfigurationError(ValueError):
    """Base exception for configuration-related errors."""

    def __init__(
        self, message: str, config_path: Optional[Path] = None, suggestion: Optional[str] = None
    ):
        """Initialize configuration error.

        Args:
            message: The error message
            config_path: Path to the configuration file
            suggestion: Helpful suggestion for fixing the error
        """
        self.message = message
        self.config_path = config_path
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with context and suggestions."""
        msg = f"❌ {self.message}"
        if self.suggestion:
            msg += f"\n\n💡 {self.suggestion}"
        if self.config_path:
            msg += f"\n\n📁 File: {self.config_path}"
        return msg


class YAMLParseError(ConfigurationError):
    """YAML parsing error with user-friendly messages."""

    @classmethod
    def from_yaml_error(cls, error: yaml.YAMLError, config_path: Path) -> "YAMLParseError":
        """Create YAMLParseError from yaml.YAMLError with helpful guidance.

        Args:
            error: The original YAML error
            config_path: Path to the configuration file

        Returns:
            YAMLParseError with user-friendly message
        """
        file_name = config_path.name

        # Extract error details if available
        line_number = getattr(error, "problem_mark", None)
        context_mark = getattr(error, "context_mark", None)
        problem = getattr(error, "problem", str(error))
        context = getattr(error, "context", None)

        # Build error message parts
        location_info = ""
        if line_number:
            location_info = f" at line {line_number.line + 1}, column {line_number.column + 1}"
        elif context_mark:
            location_info = f" at line {context_mark.line + 1}, column {context_mark.column + 1}"

        # Create user-friendly error message
        base_msg = f"YAML configuration error in {file_name}{location_info}"

        # Detect common YAML issues and provide specific guidance
        problem_lower = problem.lower()
        suggestion = cls._get_suggestion_for_problem(problem_lower, context)

        # Add context information if available
        if context and context != problem:
            base_msg += f"\n📍 Context: {context}"

        # Add helpful resources
        base_msg += "\n\n🔗 For YAML syntax help, visit: https://yaml.org/spec/1.2/spec.html"
        base_msg += "\n   Or use an online YAML validator to check your syntax."

        return cls(base_msg, config_path, suggestion)

    @staticmethod
    def _get_suggestion_for_problem(problem_lower: str, context: Optional[str] = None) -> str:
        """Get specific suggestion based on the YAML problem.

        Args:
            problem_lower: Lowercase problem description
            context: Optional context string

        Returns:
            Helpful suggestion for fixing the problem
        """
        if "found character '\\t'" in problem_lower.replace("'", "'"):
            return (
                "🚫 Tab characters are not allowed in YAML files!\n\n"
                "Fix: Replace all tab characters with spaces (usually 2 or 4 spaces).\n"
                "   Most editors can show whitespace characters and convert tabs to spaces.\n"
                "   In VS Code: View → Render Whitespace, then Edit → Convert Indentation to Spaces"
            )

        elif "mapping values are not allowed here" in problem_lower:
            return (
                "🚫 Invalid YAML syntax - missing colon or incorrect indentation!\n\n"
                "Common fixes:\n"
                "   • Add a colon (:) after the key name\n"
                "   • Check that all lines are properly indented with spaces\n"
                "   • Ensure nested items are indented consistently"
            )

        elif "could not find expected" in problem_lower and ":" in problem_lower:
            return (
                "🚫 Missing colon (:) after a key name!\n\n"
                "Fix: Add a colon and space after the key name.\n"
                "   Example: 'key_name: value' not 'key_name value'"
            )

        elif "found undefined alias" in problem_lower:
            return (
                "🚫 YAML alias reference not found!\n\n"
                "Fix: Check that the referenced alias (&name) is defined before using it (*name)"
            )

        elif "expected <block end>" in problem_lower:
            return (
                "🚫 Incorrect indentation or missing content!\n\n"
                "Common fixes:\n"
                "   • Check that all nested items are properly indented\n"
                "   • Ensure list items start with '- ' (dash and space)\n"
                "   • Make sure there's content after colons"
            )

        elif "while scanning a quoted scalar" in problem_lower:
            return (
                "🚫 Unclosed or incorrectly quoted string!\n\n"
                "Fix: Check that all quotes are properly closed.\n"
                "   • Use matching quotes: 'text' or \"text\"\n"
                '   • Escape quotes inside strings: \'don\\\'t\' or "say \\"hello\\""'
            )

        elif "found unexpected end of stream" in problem_lower:
            return (
                "🚫 Incomplete YAML structure!\n\n"
                "Fix: The file appears to end unexpectedly.\n"
                "   • Check that all sections are complete\n"
                "   • Ensure there are no missing closing brackets or braces"
            )

        elif "found unknown escape character" in problem_lower:
            return (
                "🚫 Invalid escape sequence in quoted string!\n\n"
                "Fix: Use proper YAML escape sequences or raw strings.\n"
                '   • For regex patterns: Use double quotes and double backslashes ("\\\\d+")\n'
                "   • For file paths: Use forward slashes or double backslashes\n"
                "   • Or use single quotes for literal strings: 'C:\\path\\to\\file'"
            )

        elif "scanner" in problem_lower and "character" in problem_lower:
            return (
                "🚫 Invalid character in YAML file!\n\n"
                "Fix: Check for special characters that need to be quoted.\n"
                "   • Wrap values containing special characters in quotes\n"
                "   • Common problematic characters: @, `, |, >, [, ], {, }"
            )

        else:
            return (
                f"🚫 YAML parsing error: {problem_lower}\n\n"
                "Common YAML issues to check:\n"
                "   • Use spaces for indentation, not tabs\n"
                "   • Add colons (:) after key names\n"
                "   • Ensure consistent indentation (usually 2 or 4 spaces)\n"
                "   • Quote strings containing special characters\n"
                "   • Use '- ' (dash and space) for list items"
            )


class MissingFieldError(ConfigurationError):
    """Error for missing required configuration fields."""

    def __init__(
        self,
        field_name: str,
        section: str,
        config_path: Optional[Path] = None,
        example: Optional[str] = None,
    ):
        """Initialize missing field error.

        Args:
            field_name: Name of the missing field
            section: Configuration section containing the field
            config_path: Path to the configuration file
            example: Example of correct usage
        """
        message = f"Missing required field '{field_name}' in {section}"
        suggestion = f"Add the '{field_name}' field to your configuration"
        if example:
            suggestion += f":\n   {example}"
        super().__init__(message, config_path, suggestion)


class InvalidValueError(ConfigurationError):
    """Error for invalid configuration values."""

    def __init__(
        self,
        field_name: str,
        value: Any,
        reason: str,
        config_path: Optional[Path] = None,
        valid_values: Optional[list] = None,
    ):
        """Initialize invalid value error.

        Args:
            field_name: Name of the field with invalid value
            value: The invalid value
            reason: Reason why the value is invalid
            config_path: Path to the configuration file
            valid_values: List of valid values (if applicable)
        """
        message = f"Invalid value for '{field_name}': {value} - {reason}"
        suggestion = f"Check the value of '{field_name}'"
        if valid_values:
            suggestion += f"\n   Valid values: {', '.join(str(v) for v in valid_values)}"
        super().__init__(message, config_path, suggestion)


class EnvironmentVariableError(ConfigurationError):
    """Error for missing environment variables."""

    def __init__(self, var_name: str, field_context: str, config_path: Optional[Path] = None):
        """Initialize environment variable error.

        Args:
            var_name: Name of the missing environment variable
            field_context: Context where the variable is used
            config_path: Path to the configuration file
        """
        message = f"{field_context} is configured but {var_name} environment variable is not set"
        suggestion = (
            f"Set the {var_name} environment variable:\n"
            f"   • Export directly: export {var_name}='your_value'\n"
            f"   • Or create a .env file in the same directory as your config:\n"
            f"     {var_name}=your_value"
        )
        super().__init__(message, config_path, suggestion)


def handle_yaml_error(error: yaml.YAMLError, config_path: Path) -> None:
    """Handle YAML parsing errors with user-friendly messages.

    Args:
        error: The YAML error to handle
        config_path: Path to the configuration file

    Raises:
        YAMLParseError: With user-friendly error message
    """
    raise YAMLParseError.from_yaml_error(error, config_path)
