"""JIRA platform adapter for PM framework integration.

This module provides comprehensive JIRA integration for the GitFlow Analytics PM framework,
supporting JIRA Cloud and Server instances with advanced features like custom fields,
sprint tracking, and optimized batch operations.

JiraTicketCache lives in jira_cache.py.
Conversion/mapping methods live in jira_adapter_converters.py (JIRAAdapterConvertersMixin).
"""

import base64
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, RequestException, Timeout
from urllib3.util.retry import Retry

from ...utils.debug import is_debug_mode
from ..base import BasePlatformAdapter, PlatformCapabilities
from ..models import (
    IssueStatus,
    IssueType,
    UnifiedIssue,
    UnifiedProject,
    UnifiedSprint,
    UnifiedUser,
)
from .jira_adapter_converters import JIRAAdapterConvertersMixin
from .jira_cache import JiraTicketCache  # noqa: F401 (re-exported for backward compat)

# Configure logger for JIRA adapter
logger = logging.getLogger(__name__)



class JIRAAdapter(JIRAAdapterConvertersMixin, BasePlatformAdapter):
    """JIRA platform adapter implementation.

    WHY: JIRA is one of the most widely used project management platforms,
    requiring comprehensive support for story points, sprints, custom fields,
    and advanced workflow management.

    DESIGN DECISION: Implement full JIRA API v3 support with optimized batch
    operations, rate limiting, and comprehensive error handling. Use session
    reuse and intelligent pagination for performance.

    Key Features:
    - JIRA Cloud and Server API v3 support
    - Advanced authentication with API tokens
    - Custom field discovery and mapping
    - Sprint and agile board integration
    - Optimized batch fetching with JQL
    - Comprehensive error handling and retry logic
    - Rate limiting with exponential backoff
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize JIRA adapter with configuration.

        Args:
            config: JIRA configuration including:
                - base_url: JIRA instance URL (required)
                - username: JIRA username/email (required)
                - api_token: JIRA API token (required)
                - story_point_fields: Custom field IDs for story points (optional)
                - sprint_fields: Custom field IDs for sprint data (optional)
                - batch_size: Number of issues to fetch per request (default: 50)
                - rate_limit_delay: Delay between requests in seconds (default: 0.1)
                - verify_ssl: Whether to verify SSL certificates (default: True)
                - cache_dir: Directory for ticket cache (optional, defaults to current directory)
                - cache_ttl_hours: Cache TTL in hours (optional, default: 168 = 7 days)
        """
        # Check debug mode
        debug_mode = is_debug_mode()
        if debug_mode:
            print(f"   🔍 JIRA adapter __init__ called with config keys: {list(config.keys())}")

        super().__init__(config)

        # Required configuration (use defaults for capability checking)
        self.base_url = config.get("base_url", "https://example.atlassian.net").rstrip("/")
        self.username = config.get("username", "user@example.com")
        self.api_token = config.get("api_token", "dummy-token")  # nosec B105 - placeholder default

        # Debug output
        logger.info(
            f"JIRA adapter init: base_url={self.base_url}, username={self.username}, has_token={bool(self.api_token and self.api_token != 'dummy-token')}"  # nosec B105
        )
        if debug_mode:
            print(
                f"   🔍 JIRA adapter received: username={self.username}, has_token={bool(self.api_token and self.api_token != 'dummy-token')}, base_url={self.base_url}"  # nosec B105
            )

        # Optional configuration with defaults
        self.story_point_fields = config.get(
            "story_point_fields",
            [
                "customfield_10016",  # Common JIRA Cloud story points field
                "customfield_10021",  # Alternative field
                "customfield_10002",  # Another common ID
                "Story Points",  # Field name fallback
                "storypoints",  # Alternative name
            ],
        )
        self.sprint_fields = config.get(
            "sprint_fields",
            [
                "customfield_10020",  # Common JIRA Cloud sprint field
                "customfield_10010",  # Alternative field
                "Sprint",  # Field name fallback
            ],
        )
        self.batch_size = min(config.get("batch_size", 50), 100)  # JIRA API limit
        self.rate_limit_delay = config.get("rate_limit_delay", 0.1)
        self.verify_ssl = config.get("verify_ssl", True)

        # Initialize ticket cache
        cache_dir = Path(config.get("cache_dir", Path.cwd()))
        cache_ttl_hours = config.get("cache_ttl_hours", 168)  # 7 days default
        self.ticket_cache = JiraTicketCache(cache_dir, cache_ttl_hours)
        logger.info(f"Initialized JIRA ticket cache: {self.ticket_cache.cache_path}")

        # Initialize HTTP session with retry strategy (only if we have real config)
        self._session: Optional[requests.Session] = None
        if config.get("base_url") and config.get("username") and config.get("api_token"):
            self._session = self._create_session()

        # Cache for field mappings and metadata
        self._field_mapping: Optional[dict[str, Any]] = None
        self._project_cache: Optional[list[UnifiedProject]] = None
        self._authenticated = False

        logger.info(f"Initialized JIRA adapter for {self.base_url}")

    def _ensure_session(self) -> requests.Session:
        """Ensure session is available for API calls.

        WHY: Some methods may be called before authentication, but still need
        a session. This helper ensures the session is properly initialized.

        Returns:
            Active requests session.

        Raises:
            ConnectionError: If session cannot be created.
        """
        if self._session is None:
            self._session = self._create_session()
        return self._session

    def _get_platform_name(self) -> str:
        """Return the platform name."""
        return "jira"

    def _get_capabilities(self) -> PlatformCapabilities:
        """Return JIRA platform capabilities."""
        capabilities = PlatformCapabilities()

        # JIRA supports most advanced features
        capabilities.supports_projects = True
        capabilities.supports_issues = True
        capabilities.supports_sprints = True
        capabilities.supports_time_tracking = True
        capabilities.supports_story_points = True
        capabilities.supports_custom_fields = True
        capabilities.supports_issue_linking = True
        capabilities.supports_comments = True
        capabilities.supports_attachments = True
        capabilities.supports_workflows = True
        capabilities.supports_bulk_operations = True

        # JIRA API rate limits (conservative estimates)
        capabilities.rate_limit_requests_per_hour = 3000  # JIRA Cloud typical limit
        capabilities.rate_limit_burst_size = 100
        capabilities.max_results_per_page = 100  # JIRA API maximum
        capabilities.supports_cursor_pagination = False  # JIRA uses offset pagination

        return capabilities

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy and authentication.

        WHY: JIRA APIs can be unstable under load. This session configuration
        provides resilient connections with exponential backoff retry logic
        and persistent authentication headers.

        Returns:
            Configured requests session with retry strategy.
        """
        session = requests.Session()

        # Configure retry strategy for resilient connections
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set authentication headers
        credentials = base64.b64encode(f"{self.username}:{self.api_token}".encode()).decode()
        session.headers.update(
            {
                "Authorization": f"Basic {credentials}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "GitFlow-Analytics/1.0",
            }
        )

        # SSL verification setting
        session.verify = self.verify_ssl

        return session

    def authenticate(self) -> bool:
        """Authenticate with JIRA API.

        WHY: JIRA authentication validation ensures credentials are correct
        and the instance is accessible before attempting data collection.
        This prevents later failures during analysis.

        Returns:
            True if authentication successful, False otherwise.
        """
        try:
            self.logger.info("Authenticating with JIRA API...")

            # Test authentication by getting current user info
            session = self._ensure_session()
            response = session.get(f"{self.base_url}/rest/api/3/myself")
            response.raise_for_status()

            user_info = response.json()
            self._authenticated = True

            self.logger.info(
                f"Successfully authenticated as: {user_info.get('displayName', 'Unknown')}"
            )
            return True

        except ConnectionError as e:
            self.logger.error(f"JIRA DNS/connection error: {self._format_network_error(e)}")
            self.logger.error("Troubleshooting: Check network connectivity and DNS resolution")
            self._authenticated = False
            return False
        except Timeout as e:
            self.logger.error(f"JIRA authentication timeout: {e}")
            self.logger.error("Consider increasing timeout settings or checking network latency")
            self._authenticated = False
            return False
        except RequestException as e:
            self.logger.error(f"JIRA authentication failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 401:
                    self.logger.error("Invalid JIRA credentials - check username/API token")
                elif e.response.status_code == 403:
                    self.logger.error("JIRA access forbidden - check permissions")
                else:
                    self.logger.error(
                        f"JIRA API error: {e.response.status_code} - {e.response.text}"
                    )
            self._authenticated = False
            return False
        except Exception as e:
            self.logger.error(f"Unexpected authentication error: {e}")
            self._authenticated = False
            return False

    def test_connection(self) -> dict[str, Any]:
        """Test JIRA connection and return diagnostic information.

        WHY: Provides comprehensive diagnostic information for troubleshooting
        JIRA configuration issues, including server info, permissions, and
        available features.

        Returns:
            Dictionary with connection status and diagnostic details.
        """
        result = {
            "status": "disconnected",
            "platform": "jira",
            "base_url": self.base_url,
            "authenticated_user": None,
            "server_info": {},
            "permissions": {},
            "available_projects": 0,
            "custom_fields_discovered": 0,
            "error": None,
        }

        try:
            # Test basic connectivity
            if not self._authenticated and not self.authenticate():
                result["error"] = "Authentication failed"
                return result

            # Get server information
            session = self._ensure_session()
            server_response = session.get(f"{self.base_url}/rest/api/3/serverInfo")
            if server_response.status_code == 200:
                result["server_info"] = server_response.json()

            # Get current user info
            user_response = session.get(f"{self.base_url}/rest/api/3/myself")
            user_response.raise_for_status()
            user_info = user_response.json()
            result["authenticated_user"] = user_info.get("displayName", "Unknown")

            # Test project access
            projects_response = session.get(
                f"{self.base_url}/rest/api/3/project", params={"maxResults": 1}
            )
            if projects_response.status_code == 200:
                result["available_projects"] = len(projects_response.json())

            # Discover custom fields
            fields_response = session.get(f"{self.base_url}/rest/api/3/field")
            if fields_response.status_code == 200:
                result["custom_fields_discovered"] = len(
                    [f for f in fields_response.json() if f.get("custom", False)]
                )

            result["status"] = "connected"
            self.logger.info("JIRA connection test successful")

        except ConnectionError as e:
            error_msg = f"DNS/connection error: {self._format_network_error(e)}"
            result["error"] = error_msg
            self.logger.error(error_msg)
            self.logger.error("Troubleshooting: Check network connectivity and DNS resolution")
        except Timeout as e:
            error_msg = f"Connection timeout: {e}"
            result["error"] = error_msg
            self.logger.error(error_msg)
            self.logger.error("Consider increasing timeout settings or checking network latency")
        except RequestException as e:
            error_msg = f"Connection test failed: {e}"
            if hasattr(e, "response") and e.response is not None:
                error_msg += f" (HTTP {e.response.status_code})"
            result["error"] = error_msg
            self.logger.error(error_msg)
        except Exception as e:
            result["error"] = f"Unexpected error: {e}"
            self.logger.error(f"Unexpected connection test error: {e}")

        return result

    def get_projects(self) -> list[UnifiedProject]:
        """Retrieve all accessible projects from JIRA.

        WHY: JIRA projects are the primary organizational unit for issues.
        This method discovers all accessible projects for subsequent issue
        retrieval, with caching for performance optimization.

        Returns:
            List of UnifiedProject objects representing JIRA projects.
        """
        if self._project_cache is not None:
            self.logger.debug("Returning cached JIRA projects")
            return self._project_cache

        if not self._authenticated and not self.authenticate():
            raise ConnectionError("Not authenticated with JIRA")

        try:
            self.logger.info("Fetching JIRA projects...")

            # Fetch all projects with details
            session = self._ensure_session()
            response = session.get(
                f"{self.base_url}/rest/api/3/project",
                params={
                    "expand": "description,lead,url,projectKeys",
                    "properties": "key,name,description,projectTypeKey",
                },
            )
            response.raise_for_status()

            projects_data = response.json()
            projects = []

            for project_data in projects_data:
                # Map JIRA project to unified model
                project = UnifiedProject(
                    id=project_data["id"],
                    key=project_data["key"],
                    name=project_data["name"],
                    description=project_data.get("description", ""),
                    platform=self.platform_name,
                    is_active=True,  # JIRA doesn't provide explicit active status
                    created_date=None,  # Not available in basic project info
                    platform_data={
                        "project_type": project_data.get("projectTypeKey", "unknown"),
                        "lead": project_data.get("lead", {}).get("displayName", ""),
                        "url": project_data.get("self", ""),
                        "avatar_urls": project_data.get("avatarUrls", {}),
                        "category": project_data.get("projectCategory", {}).get("name", ""),
                    },
                )
                projects.append(project)

                self.logger.debug(f"Found project: {project.key} - {project.name}")

            self._project_cache = projects
            self.logger.info(f"Successfully retrieved {len(projects)} JIRA projects")

            return projects

        except RequestException as e:
            self._handle_api_error(e, "get_projects")
            raise

    def get_issues(
        self,
        project_id: str,
        since: Optional[datetime] = None,
        issue_types: Optional[list[IssueType]] = None,
    ) -> list[UnifiedIssue]:
        """Retrieve issues for a JIRA project with advanced filtering.

        WHY: JIRA issues contain rich metadata including story points, sprints,
        and custom fields. This method uses optimized JQL queries with pagination
        to efficiently retrieve large datasets while respecting API limits.

        Args:
            project_id: JIRA project key or ID to retrieve issues from.
            since: Optional datetime to filter issues updated after this date.
            issue_types: Optional list of issue types to filter by.

        Returns:
            List of UnifiedIssue objects for the specified project.
        """
        if not self._authenticated and not self.authenticate():
            raise ConnectionError("Not authenticated with JIRA")

        try:
            # Ensure field mapping is available
            if self._field_mapping is None:
                self._discover_fields()

            # Build JQL query
            jql_conditions = [f"project = {project_id}"]

            if since:
                # Format datetime for JIRA JQL (JIRA expects specific format)
                since_str = since.strftime("%Y-%m-%d %H:%M")
                jql_conditions.append(f"updated >= '{since_str}'")

            if issue_types:
                # Map unified issue types to JIRA issue types
                jira_types = []
                for issue_type in issue_types:
                    jira_types.extend(self._map_issue_type_to_jira(issue_type))

                if jira_types:
                    types_str = "', '".join(jira_types)
                    jql_conditions.append(f"issuetype in ('{types_str}')")

            jql = " AND ".join(jql_conditions)

            self.logger.info(f"Fetching JIRA issues with JQL: {jql}")

            # Fetch issues with pagination
            issues = []
            start_at = 0

            while True:
                # Add rate limiting delay
                time.sleep(self.rate_limit_delay)

                session = self._ensure_session()
                response = session.get(
                    f"{self.base_url}/rest/api/3/search/jql",
                    params={
                        "jql": jql,
                        "startAt": start_at,
                        "maxResults": self.batch_size,
                        "fields": "*all",  # Get all fields including custom fields
                        "expand": "changelog,renderedFields",
                    },
                )
                response.raise_for_status()

                data = response.json()
                batch_issues = data.get("issues", [])

                if not batch_issues:
                    break

                # Convert JIRA issues to unified format and cache them
                for issue_data in batch_issues:
                    unified_issue = self._convert_jira_issue(issue_data)
                    issues.append(unified_issue)

                    # Cache each issue individually for future lookups
                    if unified_issue and unified_issue.key:
                        try:
                            cache_data = self._unified_issue_to_dict(unified_issue)
                            self.ticket_cache.store_ticket(unified_issue.key, cache_data)
                        except Exception as e:
                            logger.warning(f"Failed to cache issue {unified_issue.key}: {e}")

                logger.debug(f"Processed {len(batch_issues)} issues (total: {len(issues)})")

                # Check if we've retrieved all issues
                if len(batch_issues) < self.batch_size:
                    break

                start_at += self.batch_size

                # Safety check to prevent infinite loops
                if start_at > data.get("total", 0):
                    break

            self.logger.info(
                f"Successfully retrieved {len(issues)} JIRA issues for project {project_id}"
            )
            return issues

        except RequestException as e:
            self._handle_api_error(e, f"get_issues for project {project_id}")
            raise

    def get_issue_by_key(self, issue_key: str) -> Optional[UnifiedIssue]:
        """Retrieve a single issue by its key with caching.

        WHY: Training pipeline needs to fetch specific issues to determine
        their types for classification labeling. Caching dramatically speeds
        up repeated access to the same tickets.

        Args:
            issue_key: JIRA issue key (e.g., 'PROJ-123')

        Returns:
            UnifiedIssue object if found, None otherwise.
        """
        if not self._authenticated and not self.authenticate():
            raise ConnectionError("Not authenticated with JIRA")

        try:
            # Check cache first
            cached_data = self.ticket_cache.get_ticket(issue_key)
            if cached_data:
                logger.debug(f"Using cached data for issue {issue_key}")
                # Convert cached data back to UnifiedIssue
                # The cached data is already in unified format
                return self._dict_to_unified_issue(cached_data)

            # Cache miss - fetch from API
            logger.debug(f"Fetching JIRA issue {issue_key} from API")

            session = self._ensure_session()
            response = session.get(
                f"{self.base_url}/rest/api/3/issue/{issue_key}",
                params={"expand": "names,renderedFields", "fields": "*all"},
            )

            if response.status_code == 404:
                logger.warning(f"Issue {issue_key} not found")
                return None

            response.raise_for_status()
            issue_data = response.json()

            # Convert to unified format
            unified_issue = self._convert_jira_issue(issue_data)

            # Cache the unified issue data
            if unified_issue:
                cache_data = self._unified_issue_to_dict(unified_issue)
                self.ticket_cache.store_ticket(issue_key, cache_data)

            return unified_issue

        except RequestException as e:
            self._handle_api_error(e, f"get_issue_by_key for {issue_key}")
            return None

    def get_sprints(self, project_id: str) -> list[UnifiedSprint]:
        """Retrieve sprints for a JIRA project.

        WHY: Sprint data is essential for agile metrics and velocity tracking.
        JIRA provides comprehensive sprint information through board APIs.

        Args:
            project_id: JIRA project key or ID to retrieve sprints from.

        Returns:
            List of UnifiedSprint objects for the project's agile boards.
        """
        if not self._authenticated and not self.authenticate():
            raise ConnectionError("Not authenticated with JIRA")

        try:
            self.logger.info(f"Fetching JIRA sprints for project {project_id}")

            # First, find agile boards for the project
            session = self._ensure_session()
            boards_response = session.get(
                f"{self.base_url}/rest/agile/1.0/board",
                params={
                    "projectKeyOrId": project_id,
                    "type": "scrum",  # Focus on scrum boards which have sprints
                },
            )
            boards_response.raise_for_status()

            boards = boards_response.json().get("values", [])
            all_sprints = []

            # Get sprints from each board
            for board in boards:
                board_id = board["id"]
                start_at = 0

                while True:
                    time.sleep(self.rate_limit_delay)

                    sprints_response = session.get(
                        f"{self.base_url}/rest/agile/1.0/board/{board_id}/sprint",
                        params={"startAt": start_at, "maxResults": 50},  # JIRA Agile API limit
                    )
                    sprints_response.raise_for_status()

                    sprint_data = sprints_response.json()
                    batch_sprints = sprint_data.get("values", [])

                    if not batch_sprints:
                        break

                    # Convert JIRA sprints to unified format
                    for sprint_info in batch_sprints:
                        unified_sprint = self._convert_jira_sprint(sprint_info, project_id)
                        all_sprints.append(unified_sprint)

                    # Check pagination
                    if len(batch_sprints) < 50:
                        break

                    start_at += 50

            self.logger.info(f"Retrieved {len(all_sprints)} sprints for project {project_id}")
            return all_sprints

        except RequestException as e:
            # Sprints might not be available for all project types
            if hasattr(e, "response") and e.response is not None and e.response.status_code == 404:
                self.logger.warning(f"No agile boards found for project {project_id}")
                return []
            self._handle_api_error(e, f"get_sprints for project {project_id}")
            raise

    def _discover_fields(self) -> None:
        """Discover and cache JIRA field mappings.

        WHY: JIRA custom fields use cryptic IDs (e.g., customfield_10016).
        This method discovers field mappings to enable story point extraction
        and other custom field processing.
        """
        try:
            self.logger.info("Discovering JIRA field mappings...")

            session = self._ensure_session()
            response = session.get(f"{self.base_url}/rest/api/3/field")
            response.raise_for_status()

            fields = response.json()
            self._field_mapping = {}

            story_point_candidates = []
            sprint_field_candidates = []

            for field in fields:
                field_id = field.get("id", "")
                field_name = field.get("name", "").lower()
                field_type = field.get("schema", {}).get("type", "")

                self._field_mapping[field_id] = {
                    "name": field.get("name", ""),
                    "type": field_type,
                    "custom": field.get("custom", False),
                }

                # Identify potential story point fields
                if any(term in field_name for term in ["story", "point", "estimate", "size"]):
                    story_point_candidates.append((field_id, field.get("name", "")))

                # Identify potential sprint fields
                if any(term in field_name for term in ["sprint", "iteration"]):
                    sprint_field_candidates.append((field_id, field.get("name", "")))

            self.logger.info(f"Discovered {len(fields)} JIRA fields")

            if story_point_candidates:
                self.logger.info("Potential story point fields found:")
                for field_id, field_name in story_point_candidates[:5]:  # Show top 5
                    self.logger.info(f"  {field_id}: {field_name}")

            if sprint_field_candidates:
                self.logger.info("Potential sprint fields found:")
                for field_id, field_name in sprint_field_candidates[:3]:  # Show top 3
                    self.logger.info(f"  {field_id}: {field_name}")

        except RequestException as e:
            self.logger.warning(f"Failed to discover JIRA fields: {e}")
            self._field_mapping = {}
