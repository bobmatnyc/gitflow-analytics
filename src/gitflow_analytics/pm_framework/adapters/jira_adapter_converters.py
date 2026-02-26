"""JIRA adapter converter and mapping methods mixin.

Extracted from jira_adapter.py to keep that file under 800 lines.

Provides JIRAAdapterConvertersMixin which adds issue/sprint conversion,
field extraction, type/status mapping, and cache management methods to
JIRAAdapter via multiple inheritance.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from ..models import (
    IssueStatus,
    IssueType,
    UnifiedIssue,
    UnifiedSprint,
    UnifiedUser,
)

logger = logging.getLogger(__name__)


class JIRAAdapterConvertersMixin:
    """Mixin adding conversion and mapping methods to JIRAAdapter."""

    def _convert_jira_issue(self, issue_data: dict[str, Any]) -> UnifiedIssue:
        """Convert JIRA issue data to unified issue format.

        WHY: JIRA issues have complex nested structures with custom fields.
        This method normalizes JIRA data to the unified format while preserving
        important metadata in platform_data.

        Args:
            issue_data: Raw JIRA issue data from API.

        Returns:
            UnifiedIssue object with normalized data.
        """
        fields = issue_data.get("fields", {})

        # Extract basic issue information
        issue_key = issue_data.get("key", "")
        summary = fields.get("summary", "")
        description = fields.get("description", "")
        if isinstance(description, dict):
            # Handle JIRA's Atlassian Document Format
            description = self._extract_text_from_adf(description)

        # Parse dates
        created_date = self._normalize_date(fields.get("created"))
        updated_date = self._normalize_date(fields.get("updated"))
        resolved_date = self._normalize_date(fields.get("resolutiondate"))
        due_date = self._normalize_date(fields.get("duedate"))

        # Map issue type
        issue_type_data = fields.get("issuetype", {})
        issue_type = self._map_jira_issue_type(issue_type_data.get("name", ""))

        # Map status
        status_data = fields.get("status", {})
        status = self._map_jira_status(status_data.get("name", ""))

        # Map priority
        priority_data = fields.get("priority", {})
        priority = self._map_priority(priority_data.get("name", "") if priority_data else "")

        # Extract users
        assignee = self._extract_jira_user(fields.get("assignee"))
        reporter = self._extract_jira_user(fields.get("reporter"))

        # Extract story points from custom fields
        story_points = self._extract_story_points(fields)

        # Extract sprint information
        sprint_id, sprint_name = self._extract_sprint_info(fields)

        # Extract time tracking
        time_tracking = fields.get("timetracking", {})
        original_estimate_hours = self._seconds_to_hours(
            time_tracking.get("originalEstimateSeconds")
        )
        remaining_estimate_hours = self._seconds_to_hours(
            time_tracking.get("remainingEstimateSeconds")
        )
        time_spent_hours = self._seconds_to_hours(time_tracking.get("timeSpentSeconds"))

        # Extract relationships
        parent_key = None
        if fields.get("parent"):
            parent_key = fields["parent"].get("key")

        subtasks = [subtask.get("key", "") for subtask in fields.get("subtasks", [])]

        # Extract issue links
        linked_issues = []
        for link in fields.get("issuelinks", []):
            if "outwardIssue" in link:
                linked_issues.append(
                    {
                        "key": link["outwardIssue"].get("key", ""),
                        "type": link.get("type", {}).get("outward", "links"),
                    }
                )
            if "inwardIssue" in link:
                linked_issues.append(
                    {
                        "key": link["inwardIssue"].get("key", ""),
                        "type": link.get("type", {}).get("inward", "links"),
                    }
                )

        # Extract labels and components
        labels = [label for label in fields.get("labels", [])]
        components = [comp.get("name", "") for comp in fields.get("components", [])]

        # Create unified issue
        unified_issue = UnifiedIssue(
            id=issue_data.get("id", ""),
            key=issue_key,
            platform=self.platform_name,
            project_id=fields.get("project", {}).get("key", ""),
            title=summary,
            description=description,
            created_date=created_date or datetime.now(timezone.utc),
            updated_date=updated_date or datetime.now(timezone.utc),
            issue_type=issue_type,
            status=status,
            priority=priority,
            assignee=assignee,
            reporter=reporter,
            resolved_date=resolved_date,
            due_date=due_date,
            story_points=story_points,
            original_estimate_hours=original_estimate_hours,
            remaining_estimate_hours=remaining_estimate_hours,
            time_spent_hours=time_spent_hours,
            parent_issue_key=parent_key,
            subtasks=subtasks,
            linked_issues=linked_issues,
            sprint_id=sprint_id,
            sprint_name=sprint_name,
            labels=labels,
            components=components,
            platform_data={
                "issue_type_id": issue_type_data.get("id", ""),
                "status_id": status_data.get("id", ""),
                "status_category": status_data.get("statusCategory", {}).get("name", ""),
                "priority_id": priority_data.get("id", "") if priority_data else "",
                "resolution": (
                    fields.get("resolution", {}).get("name", "") if fields.get("resolution") else ""
                ),
                "environment": fields.get("environment", ""),
                "security_level": (
                    fields.get("security", {}).get("name", "") if fields.get("security") else ""
                ),
                "votes": fields.get("votes", {}).get("votes", 0),
                "watches": fields.get("watches", {}).get("watchCount", 0),
                "custom_fields": self._extract_custom_fields(fields),
                "jira_url": f"{self.base_url}/browse/{issue_key}",
            },
        )

        return unified_issue

    def _convert_jira_sprint(self, sprint_data: dict[str, Any], project_id: str) -> UnifiedSprint:
        """Convert JIRA sprint data to unified sprint format.

        Args:
            sprint_data: Raw JIRA sprint data from Agile API.
            project_id: Project ID the sprint belongs to.

        Returns:
            UnifiedSprint object with normalized data.
        """
        start_date = self._normalize_date(sprint_data.get("startDate"))
        end_date = self._normalize_date(sprint_data.get("endDate"))
        complete_date = self._normalize_date(sprint_data.get("completeDate"))

        # Determine sprint state
        state = sprint_data.get("state", "").lower()
        is_active = state == "active"
        is_completed = state == "closed" or complete_date is not None

        return UnifiedSprint(
            id=str(sprint_data.get("id", "")),
            name=sprint_data.get("name", ""),
            project_id=project_id,
            platform=self.platform_name,
            start_date=start_date,
            end_date=end_date,
            is_active=is_active,
            is_completed=is_completed,
            planned_story_points=None,  # Not directly available from JIRA API
            completed_story_points=None,  # Would need to calculate from issues
            issue_keys=[],  # Would need separate API call to get sprint issues
            platform_data={
                "state": sprint_data.get("state", ""),
                "goal": sprint_data.get("goal", ""),
                "complete_date": complete_date,
                "board_id": sprint_data.get("originBoardId"),
                "jira_url": sprint_data.get("self", ""),
            },
        )

    def _extract_jira_user(self, user_data: Optional[dict[str, Any]]) -> Optional[UnifiedUser]:
        """Extract user information from JIRA user data.

        Args:
            user_data: JIRA user object from API.

        Returns:
            UnifiedUser object or None if user_data is empty.
        """
        if not user_data:
            return None

        return UnifiedUser(
            id=user_data.get("accountId", user_data.get("name", "")),
            email=user_data.get("emailAddress"),
            display_name=user_data.get("displayName", ""),
            username=user_data.get("name"),  # Deprecated in JIRA Cloud but may exist
            platform=self.platform_name,
            is_active=user_data.get("active", True),
            platform_data={
                "avatar_urls": user_data.get("avatarUrls", {}),
                "timezone": user_data.get("timeZone", ""),
                "locale": user_data.get("locale", ""),
            },
        )

    def _extract_story_points(self, fields: dict[str, Any]) -> Optional[int]:
        """Extract story points from JIRA custom fields.

        WHY: Story points can be stored in various custom fields depending
        on JIRA configuration. This method tries multiple common field IDs
        and field names to find story point values.

        Args:
            fields: JIRA issue fields dictionary.

        Returns:
            Story points as integer, or None if not found.
        """
        # Track which fields were tried for debugging
        tried_fields = []
        found_values = {}

        # Try configured story point fields first
        for field_id in self.story_point_fields:
            tried_fields.append(field_id)
            if field_id in fields and fields[field_id] is not None:
                value = fields[field_id]
                found_values[field_id] = value
                try:
                    if isinstance(value, (int, float)):
                        logger.debug(f"Found story points in field '{field_id}': {value}")
                        return int(value)
                    elif isinstance(value, str) and value.strip():
                        points = int(float(value.strip()))
                        logger.debug(f"Found story points in field '{field_id}': {points}")
                        return points
                except (ValueError, TypeError) as e:
                    logger.debug(f"Field '{field_id}' has value {value} but failed to parse: {e}")
                    continue

        # Log diagnostic information if no story points found
        logger.debug(f"Story points not found. Tried fields: {tried_fields}")
        if found_values:
            logger.debug(f"Fields with non-null values (but unparseable): {found_values}")

        # Log all available custom fields for debugging
        custom_fields = {k: v for k, v in fields.items() if k.startswith("customfield_")}
        if custom_fields:
            logger.debug(f"Available custom fields in this issue: {list(custom_fields.keys())}")
            logger.info(
                "Story points not found. Use 'discover-storypoint-fields' command "
                "to identify the correct custom field ID for your JIRA instance."
            )

        # Use base class method as fallback
        return super()._extract_story_points(fields)

    def _extract_sprint_info(self, fields: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        """Extract sprint information from JIRA fields.

        Args:
            fields: JIRA issue fields dictionary.

        Returns:
            Tuple of (sprint_id, sprint_name) or (None, None) if not found.
        """
        # Try configured sprint fields
        for field_id in self.sprint_fields:
            sprint_data = fields.get(field_id)
            if not sprint_data:
                continue

            # Sprint field can be an array of sprints (issue in multiple sprints)
            if isinstance(sprint_data, list) and sprint_data:
                sprint_data = sprint_data[-1]  # Use the latest sprint

            if isinstance(sprint_data, dict):
                return str(sprint_data.get("id", "")), sprint_data.get("name", "")
            elif isinstance(sprint_data, str) and "id=" in sprint_data:
                # Handle legacy sprint string format: "com.atlassian.greenhopper.service.sprint.Sprint@abc[id=123,name=Sprint 1,...]"
                try:
                    import re

                    id_match = re.search(r"id=(\d+)", sprint_data)
                    name_match = re.search(r"name=([^,\]]+)", sprint_data)
                    if id_match and name_match:
                        return id_match.group(1), name_match.group(1)
                except (TypeError, AttributeError) as e:
                    logger.debug(f"Non-critical: could not parse legacy JIRA sprint string: {e}")

        return None, None

    def _extract_custom_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        """Extract custom field values from JIRA fields.

        Args:
            fields: JIRA issue fields dictionary.

        Returns:
            Dictionary of custom field values.
        """
        custom_fields = {}

        for field_id, value in fields.items():
            if field_id.startswith("customfield_") and value is not None:
                # Get field metadata if available
                field_info = self._field_mapping.get(field_id, {}) if self._field_mapping else {}
                field_name = field_info.get("name", field_id)

                # Simplify complex field values
                if isinstance(value, dict):
                    if "value" in value:
                        custom_fields[field_name] = value["value"]
                    elif "displayName" in value:
                        custom_fields[field_name] = value["displayName"]
                    elif "name" in value:
                        custom_fields[field_name] = value["name"]
                    else:
                        custom_fields[field_name] = str(value)
                elif isinstance(value, list):
                    if value and isinstance(value[0], dict):
                        # Extract display values from option lists
                        display_values = []
                        for item in value:
                            if "value" in item:
                                display_values.append(item["value"])
                            elif "name" in item:
                                display_values.append(item["name"])
                            else:
                                display_values.append(str(item))
                        custom_fields[field_name] = display_values
                    else:
                        custom_fields[field_name] = value
                else:
                    custom_fields[field_name] = value

        return custom_fields

    def _map_jira_issue_type(self, jira_type: str) -> IssueType:
        """Map JIRA issue type to unified issue type.

        Args:
            jira_type: JIRA issue type name.

        Returns:
            Unified IssueType enum value.
        """
        if not jira_type:
            return IssueType.UNKNOWN

        type_lower = jira_type.lower()

        # Common JIRA issue type mappings
        if type_lower in ["epic"]:
            return IssueType.EPIC
        elif type_lower in ["story", "user story"]:
            return IssueType.STORY
        elif type_lower in ["task"]:
            return IssueType.TASK
        elif type_lower in ["bug", "defect"]:
            return IssueType.BUG
        elif type_lower in ["new feature", "feature"]:
            return IssueType.FEATURE
        elif type_lower in ["improvement", "enhancement"]:
            return IssueType.IMPROVEMENT
        elif type_lower in ["sub-task", "subtask"]:
            return IssueType.SUBTASK
        elif type_lower in ["incident", "outage"]:
            return IssueType.INCIDENT
        else:
            return IssueType.UNKNOWN

    def _map_jira_status(self, jira_status: str) -> IssueStatus:
        """Map JIRA status to unified issue status.

        Args:
            jira_status: JIRA status name.

        Returns:
            Unified IssueStatus enum value.
        """
        if not jira_status:
            return IssueStatus.UNKNOWN

        status_lower = jira_status.lower()

        # Common JIRA status mappings
        if status_lower in ["open", "to do", "todo", "new", "created", "backlog"]:
            return IssueStatus.TODO
        elif status_lower in ["in progress", "in-progress", "in development", "active", "assigned"]:
            return IssueStatus.IN_PROGRESS
        elif status_lower in ["in review", "in-review", "review", "code review", "peer review"]:
            return IssueStatus.IN_REVIEW
        elif status_lower in ["testing", "in testing", "in-testing", "qa", "verification"]:
            return IssueStatus.TESTING
        elif status_lower in ["done", "closed", "resolved", "completed", "fixed", "verified"]:
            return IssueStatus.DONE
        elif status_lower in ["cancelled", "canceled", "rejected", "wont do", "won't do"]:
            return IssueStatus.CANCELLED
        elif status_lower in ["blocked", "on hold", "waiting", "impediment"]:
            return IssueStatus.BLOCKED
        else:
            return IssueStatus.UNKNOWN

    def _map_issue_type_to_jira(self, issue_type: IssueType) -> list[str]:
        """Map unified issue type to JIRA issue type names.

        Args:
            issue_type: Unified IssueType enum value.

        Returns:
            List of possible JIRA issue type names.
        """
        mapping = {
            IssueType.EPIC: ["Epic"],
            IssueType.STORY: ["Story", "User Story"],
            IssueType.TASK: ["Task"],
            IssueType.BUG: ["Bug", "Defect"],
            IssueType.FEATURE: ["New Feature", "Feature"],
            IssueType.IMPROVEMENT: ["Improvement", "Enhancement"],
            IssueType.SUBTASK: ["Sub-task", "Subtask"],
            IssueType.INCIDENT: ["Incident", "Outage"],
        }

        return mapping.get(issue_type, [])

    def _extract_text_from_adf(self, adf_doc: dict[str, Any]) -> str:
        """Extract plain text from JIRA's Atlassian Document Format.

        WHY: JIRA Cloud uses ADF (Atlassian Document Format) for rich text.
        This method extracts plain text for consistent processing.

        Args:
            adf_doc: ADF document structure.

        Returns:
            Plain text extracted from ADF.
        """

        def extract_text_recursive(node: Any) -> str:
            if isinstance(node, dict):
                if node.get("type") == "text":
                    text_value = node.get("text", "")
                    return str(text_value) if text_value else ""
                elif "content" in node:
                    return "".join(extract_text_recursive(child) for child in node["content"])
            elif isinstance(node, list):
                return "".join(extract_text_recursive(child) for child in node)
            return ""

        try:
            return extract_text_recursive(adf_doc)
        except Exception:
            return str(adf_doc)

    def _seconds_to_hours(self, seconds: Optional[int]) -> Optional[float]:
        """Convert seconds to hours for time tracking fields.

        Args:
            seconds: Time in seconds.

        Returns:
            Time in hours, or None if seconds is None.
        """
        return seconds / 3600.0 if seconds is not None else None

    def _format_network_error(self, error: Exception) -> str:
        """Format network errors with helpful context.

        Args:
            error: The network exception that occurred.

        Returns:
            Formatted error message with troubleshooting context.
        """
        error_str = str(error)

        if "nodename nor servname provided" in error_str or "[Errno 8]" in error_str:
            return f"DNS resolution failed - hostname not found ({error_str})"
        elif "Name or service not known" in error_str or "[Errno -2]" in error_str:
            return f"DNS resolution failed - service not known ({error_str})"
        elif "Connection refused" in error_str or "[Errno 111]" in error_str:
            return f"Connection refused - service not running ({error_str})"
        elif "Network is unreachable" in error_str or "[Errno 101]" in error_str:
            return f"Network unreachable - check internet connection ({error_str})"
        elif "timeout" in error_str.lower():
            return f"Network timeout - slow connection or high latency ({error_str})"
        else:
            return f"Network error ({error_str})"

    def _unified_issue_to_dict(self, issue: UnifiedIssue) -> dict[str, Any]:
        """Convert UnifiedIssue to dictionary for caching.

        WHY: Cache storage requires serializable data structures.
        This method converts the UnifiedIssue object to a dictionary
        that preserves all data needed for reconstruction.

        Args:
            issue: UnifiedIssue object to convert

        Returns:
            Dictionary representation suitable for caching
        """
        return {
            "id": issue.id,
            "key": issue.key,
            "platform": issue.platform,
            "project_id": issue.project_id,
            "title": issue.title,
            "description": issue.description,
            "created_date": issue.created_date.isoformat() if issue.created_date else None,
            "updated_date": issue.updated_date.isoformat() if issue.updated_date else None,
            "issue_type": issue.issue_type.value if issue.issue_type else None,
            "status": issue.status.value if issue.status else None,
            "priority": issue.priority.value if issue.priority else None,
            "assignee": (
                {
                    "id": issue.assignee.id,
                    "email": issue.assignee.email,
                    "display_name": issue.assignee.display_name,
                    "username": issue.assignee.username,
                    "platform": issue.assignee.platform,
                    "is_active": issue.assignee.is_active,
                    "platform_data": issue.assignee.platform_data,
                }
                if issue.assignee
                else None
            ),
            "reporter": (
                {
                    "id": issue.reporter.id,
                    "email": issue.reporter.email,
                    "display_name": issue.reporter.display_name,
                    "username": issue.reporter.username,
                    "platform": issue.reporter.platform,
                    "is_active": issue.reporter.is_active,
                    "platform_data": issue.reporter.platform_data,
                }
                if issue.reporter
                else None
            ),
            "resolved_date": issue.resolved_date.isoformat() if issue.resolved_date else None,
            "due_date": issue.due_date.isoformat() if issue.due_date else None,
            "story_points": issue.story_points,
            "original_estimate_hours": issue.original_estimate_hours,
            "remaining_estimate_hours": issue.remaining_estimate_hours,
            "time_spent_hours": issue.time_spent_hours,
            "parent_issue_key": issue.parent_issue_key,
            "subtasks": issue.subtasks or [],
            "linked_issues": issue.linked_issues or [],
            "sprint_id": issue.sprint_id,
            "sprint_name": issue.sprint_name,
            "labels": issue.labels or [],
            "components": issue.components or [],
            "platform_data": issue.platform_data or {},
        }

    def _dict_to_unified_issue(self, data: dict[str, Any]) -> UnifiedIssue:
        """Convert dictionary back to UnifiedIssue object.

        WHY: Cache retrieval needs to reconstruct UnifiedIssue objects
        from stored dictionary data. This method handles the conversion
        including proper enum and datetime parsing.

        Args:
            data: Dictionary representation from cache

        Returns:
            UnifiedIssue object reconstructed from cached data
        """
        from datetime import datetime, timezone

        # Helper function to parse ISO datetime strings
        def parse_datetime(date_str: Optional[str]) -> Optional[datetime]:
            if not date_str:
                return None
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                # Ensure timezone awareness
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, TypeError):
                return None

        # Convert string enums back to enum values
        def safe_enum_conversion(enum_class, value):
            if not value:
                return None
            try:
                return enum_class(value)
            except (ValueError, TypeError):
                return None

        # Reconstruct user objects
        def dict_to_user(user_data: Optional[dict[str, Any]]) -> Optional[UnifiedUser]:
            if not user_data:
                return None
            return UnifiedUser(
                id=user_data.get("id", ""),
                email=user_data.get("email"),
                display_name=user_data.get("display_name", ""),
                username=user_data.get("username"),
                platform=user_data.get("platform", self.platform_name),
                is_active=user_data.get("is_active", True),
                platform_data=user_data.get("platform_data", {}),
            )

        return UnifiedIssue(
            id=data.get("id", ""),
            key=data.get("key", ""),
            platform=data.get("platform", self.platform_name),
            project_id=data.get("project_id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            created_date=parse_datetime(data.get("created_date")) or datetime.now(timezone.utc),
            updated_date=parse_datetime(data.get("updated_date")) or datetime.now(timezone.utc),
            issue_type=safe_enum_conversion(IssueType, data.get("issue_type")),
            status=safe_enum_conversion(IssueStatus, data.get("status")),
            priority=safe_enum_conversion(self._get_priority_enum(), data.get("priority")),
            assignee=dict_to_user(data.get("assignee")),
            reporter=dict_to_user(data.get("reporter")),
            resolved_date=parse_datetime(data.get("resolved_date")),
            due_date=parse_datetime(data.get("due_date")),
            story_points=data.get("story_points"),
            original_estimate_hours=data.get("original_estimate_hours"),
            remaining_estimate_hours=data.get("remaining_estimate_hours"),
            time_spent_hours=data.get("time_spent_hours"),
            parent_issue_key=data.get("parent_issue_key"),
            subtasks=data.get("subtasks", []),
            linked_issues=data.get("linked_issues", []),
            sprint_id=data.get("sprint_id"),
            sprint_name=data.get("sprint_name"),
            labels=data.get("labels", []),
            components=data.get("components", []),
            platform_data=data.get("platform_data", {}),
        )

    def _get_priority_enum(self):
        """Get priority enum class for safe conversion."""
        # Import here to avoid circular imports
        from ..models import Priority

        return Priority

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get comprehensive cache statistics for monitoring and debugging.

        WHY: Cache performance monitoring is essential for optimization
        and troubleshooting. This method provides detailed metrics about
        cache usage, effectiveness, and storage patterns.

        Returns:
            Dictionary with detailed cache statistics
        """
        return self.ticket_cache.get_cache_stats()

    def print_cache_summary(self) -> None:
        """Print user-friendly cache performance summary."""
        self.ticket_cache.print_cache_summary()

    def clear_ticket_cache(self) -> int:
        """Clear all cached tickets.

        Returns:
            Number of entries removed
        """
        return self.ticket_cache.clear_cache()

    def cleanup_expired_cache(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of expired entries removed
        """
        return self.ticket_cache.cleanup_expired()

    def invalidate_ticket_cache(self, ticket_key: str) -> bool:
        """Invalidate cache for specific ticket.

        Args:
            ticket_key: JIRA ticket key to invalidate

        Returns:
            True if ticket was found and invalidated
        """
        return self.ticket_cache.invalidate_ticket(ticket_key)
