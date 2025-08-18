"""Base platform adapter interface and capabilities definition.

This module provides the abstract base class and capability definitions that all
PM platform adapters must implement. It includes common utility methods for
data normalization and error handling.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from .models import IssueType, Priority, UnifiedIssue, UnifiedProject, UnifiedSprint, UnifiedUser

# Configure logger for PM framework
logger = logging.getLogger(__name__)


class PlatformCapabilities:
    """Defines what capabilities a platform adapter supports.

    WHY: Different PM platforms have varying feature sets. This class allows
    adapters to declare their capabilities so the orchestrator can gracefully
    handle missing features and inform users about platform limitations.

    DESIGN DECISION: Use explicit capability flags rather than trying operations
    and catching exceptions. This approach is more predictable and provides
    better user feedback about what features are available.
    """

    def __init__(self) -> None:
        # Core capabilities - most platforms support these
        self.supports_projects = True
        self.supports_issues = True

        # Advanced capabilities - vary by platform
        self.supports_sprints = False
        self.supports_time_tracking = False
        self.supports_story_points = False
        self.supports_custom_fields = False
        self.supports_issue_linking = False
        self.supports_comments = False
        self.supports_attachments = False
        self.supports_workflows = False
        self.supports_bulk_operations = False

        # Rate limiting info - critical for API management
        self.rate_limit_requests_per_hour = 1000
        self.rate_limit_burst_size = 100

        # Pagination info - for efficient data retrieval
        self.max_results_per_page = 100
        self.supports_cursor_pagination = False


class BasePlatformAdapter(ABC):
    """Abstract base class for all platform adapters.

    This class defines the standard interface that all PM platform adapters
    must implement. It provides common utility methods for data normalization
    and includes comprehensive error handling patterns.

    WHY: Standardized interface ensures consistent behavior across all platforms
    while allowing platform-specific implementations. Common utility methods
    reduce code duplication and ensure consistent data transformation.

    DESIGN DECISION: Use abstract methods for required operations and default
    implementations for optional features. This approach ensures essential
    functionality while providing flexibility for platform differences.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the platform adapter with configuration.

        Args:
            config: Platform-specific configuration including authentication
                   credentials, API endpoints, and feature settings.
        """
        self.config = config
        self.platform_name = self._get_platform_name()
        self.capabilities = self._get_capabilities()
        self._client = None

        # Set up logging for this adapter
        self.logger = logging.getLogger(f"{__name__}.{self.platform_name}")
        self.logger.info(f"Initializing {self.platform_name} adapter")

    @abstractmethod
    def _get_platform_name(self) -> str:
        """Return the platform name (e.g., 'jira', 'azure_devops').

        Returns:
            String identifier for this platform, used in logging and data tagging.
        """
        pass

    @abstractmethod
    def _get_capabilities(self) -> PlatformCapabilities:
        """Return the capabilities supported by this platform.

        Returns:
            PlatformCapabilities object describing what features this adapter supports.
        """
        pass

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the platform.

        WHY: Authentication is platform-specific but required for all adapters.
        This method establishes the connection and validates credentials.

        Returns:
            True if authentication successful, False otherwise.

        Raises:
            ConnectionError: If authentication fails due to network issues.
            ValueError: If credentials are invalid or missing.
        """
        pass

    @abstractmethod
    def test_connection(self) -> dict[str, Any]:
        """Test connection and return platform status information.

        WHY: Provides diagnostic information for troubleshooting connection
        issues and validating configuration before full data collection.

        Returns:
            Dictionary containing connection status, platform version info,
            and any diagnostic details.
        """
        pass

    @abstractmethod
    def get_projects(self) -> list[UnifiedProject]:
        """Retrieve all accessible projects from the platform.

        WHY: Projects are the primary organizational unit for most PM platforms.
        This method discovers available projects for subsequent issue retrieval.

        Returns:
            List of UnifiedProject objects representing all accessible projects.

        Raises:
            ConnectionError: If API request fails due to network issues.
            PermissionError: If user lacks permission to access projects.
        """
        pass

    @abstractmethod
    def get_issues(
        self,
        project_id: str,
        since: Optional[datetime] = None,
        issue_types: Optional[list[IssueType]] = None,
    ) -> list[UnifiedIssue]:
        """Retrieve issues for a specific project.

        WHY: Issues are the core work items that need to be correlated with
        Git commits. This method fetches issue data with optional filtering
        to optimize performance and focus on relevant timeframes.

        Args:
            project_id: Project identifier to retrieve issues from.
            since: Optional datetime to filter issues updated after this date.
            issue_types: Optional list of issue types to filter by.

        Returns:
            List of UnifiedIssue objects for the specified project.

        Raises:
            ConnectionError: If API request fails due to network issues.
            ValueError: If project_id is invalid or not accessible.
        """
        pass

    # Optional methods with default implementations
    def get_sprints(self, project_id: str) -> list[UnifiedSprint]:
        """Retrieve sprints for a project.

        Default implementation returns empty list for platforms that don't
        support sprints. Override this method if the platform supports sprints.

        Args:
            project_id: Project identifier to retrieve sprints from.

        Returns:
            List of UnifiedSprint objects, empty if not supported.
        """
        if not self.capabilities.supports_sprints:
            self.logger.debug(f"Sprints not supported by {self.platform_name}")
            return []
        raise NotImplementedError(f"get_sprints not implemented for {self.platform_name}")

    def get_users(self, project_id: str) -> list[UnifiedUser]:
        """Retrieve users for a project.

        Default implementation returns empty list. Override if the platform
        provides user enumeration capabilities.

        Args:
            project_id: Project identifier to retrieve users from.

        Returns:
            List of UnifiedUser objects, empty by default.
        """
        self.logger.debug(f"User enumeration not implemented for {self.platform_name}")
        return []

    def get_issue_comments(self, issue_key: str) -> list[dict[str, Any]]:
        """Retrieve comments for an issue.

        Default implementation returns empty list for platforms that don't
        support comments. Override this method if comments are available.

        Args:
            issue_key: Issue identifier to retrieve comments from.

        Returns:
            List of comment dictionaries, empty if not supported.
        """
        if not self.capabilities.supports_comments:
            self.logger.debug(f"Comments not supported by {self.platform_name}")
            return []
        raise NotImplementedError(f"get_issue_comments not implemented for {self.platform_name}")

    def get_custom_fields(self, project_id: str) -> dict[str, Any]:
        """Retrieve custom field definitions for a project.

        Default implementation returns empty dict for platforms that don't
        support custom fields. Override if custom fields are available.

        Args:
            project_id: Project identifier to retrieve custom fields from.

        Returns:
            Dictionary of custom field definitions, empty if not supported.
        """
        if not self.capabilities.supports_custom_fields:
            self.logger.debug(f"Custom fields not supported by {self.platform_name}")
            return {}
        raise NotImplementedError(f"get_custom_fields not implemented for {self.platform_name}")

    # Utility methods for data normalization
    def _normalize_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Normalize date string to datetime object.

        WHY: Different platforms use different date formats. This utility
        method handles common formats to ensure consistent datetime objects
        throughout the system.

        Args:
            date_str: Date string in various formats, or None.

        Returns:
            Normalized datetime object, or None if parsing fails.
        """
        if not date_str:
            return None

        # Handle common date formats from different platforms
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO with microseconds (GitHub, Linear)
            "%Y-%m-%dT%H:%M:%SZ",  # ISO without microseconds (JIRA)
            "%Y-%m-%dT%H:%M:%S%z",  # ISO with timezone (Azure DevOps)
            "%Y-%m-%dT%H:%M:%S.%f%z",  # ISO with microseconds and timezone
            "%Y-%m-%d %H:%M:%S",  # Common SQL format
            "%Y-%m-%d",  # Date only
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        self.logger.warning(f"Could not parse date: {date_str}")
        return None

    def _map_priority(self, platform_priority: str) -> Priority:
        """Map platform-specific priority to unified priority.

        WHY: Different platforms use different priority schemes (numeric,
        named, etc.). This method normalizes priorities to enable consistent
        priority-based analysis across platforms.

        Args:
            platform_priority: Priority value from the platform.

        Returns:
            Unified Priority enum value.
        """
        if not platform_priority:
            return Priority.UNKNOWN

        priority_lower = platform_priority.lower()

        # Common priority mappings across platforms
        priority_mapping = {
            # Critical/Urgent priorities
            "highest": Priority.CRITICAL,
            "critical": Priority.CRITICAL,
            "urgent": Priority.CRITICAL,
            "1": Priority.CRITICAL,
            "blocker": Priority.CRITICAL,
            # High priorities
            "high": Priority.HIGH,
            "important": Priority.HIGH,
            "2": Priority.HIGH,
            "major": Priority.HIGH,
            # Medium/Normal priorities
            "medium": Priority.MEDIUM,
            "normal": Priority.MEDIUM,
            "3": Priority.MEDIUM,
            "moderate": Priority.MEDIUM,
            # Low priorities
            "low": Priority.LOW,
            "minor": Priority.LOW,
            "4": Priority.LOW,
            # Trivial priorities
            "trivial": Priority.TRIVIAL,
            "lowest": Priority.TRIVIAL,
            "5": Priority.TRIVIAL,
        }

        mapped_priority = priority_mapping.get(priority_lower, Priority.UNKNOWN)
        if mapped_priority == Priority.UNKNOWN:
            self.logger.debug(f"Unknown priority '{platform_priority}' mapped to UNKNOWN")

        return mapped_priority

    def _extract_story_points(self, custom_fields: dict[str, Any]) -> Optional[int]:
        """Extract story points from custom fields.

        WHY: Story points are critical for velocity tracking but stored
        differently across platforms (custom fields, dedicated fields, etc.).
        This method attempts to find story points using common field names.

        Args:
            custom_fields: Dictionary of custom field values from the platform.

        Returns:
            Story points as integer, or None if not found.
        """
        if not custom_fields:
            return None

        # Common story point field names across platforms
        story_point_fields = [
            "story_points",
            "storypoints",
            "story_point_estimate",
            "customfield_10016",
            "customfield_10021",  # Common JIRA fields
            "Microsoft.VSTS.Scheduling.StoryPoints",  # Azure DevOps
            "effort",
            "size",
            "complexity",
            "points",
        ]

        for field_name in story_point_fields:
            if field_name in custom_fields:
                value = custom_fields[field_name]
                if isinstance(value, (int, float)):
                    return int(value)
                elif isinstance(value, str) and value.strip().isdigit():
                    return int(value.strip())
                elif isinstance(value, str):
                    # Handle decimal story points
                    try:
                        return int(float(value.strip()))
                    except (ValueError, AttributeError):
                        continue

        return None

    def _handle_api_error(self, error: Exception, operation: str) -> None:
        """Handle API errors with consistent logging and error reporting.

        WHY: API errors are common when integrating with external platforms.
        This method provides consistent error handling and logging to help
        diagnose issues and provide meaningful user feedback.

        Args:
            error: The exception that occurred.
            operation: Description of the operation that failed.

        Raises:
            The original exception after logging appropriate details.
        """
        error_msg = f"{operation} failed for {self.platform_name}: {str(error)}"

        # Log different error types at appropriate levels
        if "rate limit" in str(error).lower():
            self.logger.warning(f"Rate limit exceeded: {error_msg}")
        elif "unauthorized" in str(error).lower() or "403" in str(error):
            self.logger.error(f"Authentication error: {error_msg}")
        elif "not found" in str(error).lower() or "404" in str(error):
            self.logger.warning(f"Resource not found: {error_msg}")
        else:
            self.logger.error(f"API error: {error_msg}")

        # Re-raise the original exception for caller handling
        raise error
