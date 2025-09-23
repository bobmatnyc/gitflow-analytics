"""JIRA platform adapter for PM framework integration.

This module provides comprehensive JIRA integration for the GitFlow Analytics PM framework,
supporting JIRA Cloud and Server instances with advanced features like custom fields,
sprint tracking, and optimized batch operations.
"""

import base64
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, RequestException, Timeout
from urllib3.util.retry import Retry

from ..base import BasePlatformAdapter, PlatformCapabilities
from ..models import (
    IssueStatus,
    IssueType,
    UnifiedIssue,
    UnifiedProject,
    UnifiedSprint,
    UnifiedUser,
)

# Configure logger for JIRA adapter
logger = logging.getLogger(__name__)


class JiraTicketCache:
    """SQLite-based cache for JIRA ticket responses.

    WHY: JIRA API calls are expensive and can be slow, especially for large
    organizations. This cache stores ticket responses with configurable TTL
    to dramatically speed up repeated runs while maintaining data freshness.

    DESIGN DECISION: Store cache in config directory (not .gitflow-cache)
    as requested, use SQLite for efficient querying and storage, include
    comprehensive metadata for cache management and performance tracking.

    Cache Strategy:
    - Individual ticket responses cached with full JSON data
    - Configurable TTL with default 7 days (168 hours)
    - Cache hit/miss metrics for performance monitoring
    - Automatic cleanup of expired entries
    - Size management with configurable limits
    """

    def __init__(self, config_dir: Path, ttl_hours: int = 168) -> None:
        """Initialize JIRA ticket cache.

        Args:
            config_dir: Directory to store cache database (config file directory)
            ttl_hours: Time to live for cached tickets in hours (default: 7 days)
        """
        self.config_dir = Path(config_dir)
        self.ttl_hours = ttl_hours
        self.cache_path = self.config_dir / "jira_tickets.db"

        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_stores = 0
        self.session_start = datetime.now()

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info(f"Initialized JIRA ticket cache: {self.cache_path}")

    def _init_database(self) -> None:
        """Initialize SQLite database with ticket cache tables.

        WHY: Comprehensive schema design captures all ticket metadata
        needed for analytics while enabling efficient querying and
        cache management operations.
        """
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jira_tickets (
                    ticket_key TEXT PRIMARY KEY,
                    project_key TEXT NOT NULL,
                    ticket_data JSON NOT NULL,
                    story_points INTEGER,
                    status TEXT,
                    issue_type TEXT,
                    assignee TEXT,
                    reporter TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Indexes for efficient querying
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_project_key 
                ON jira_tickets(project_key)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_expires_at 
                ON jira_tickets(expires_at)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_status 
                ON jira_tickets(status)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_updated_at 
                ON jira_tickets(updated_at)
            """
            )

            # Cache metadata table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()

    def get_ticket(self, ticket_key: str) -> Optional[dict[str, Any]]:
        """Retrieve cached ticket data if not expired.

        Args:
            ticket_key: JIRA ticket key (e.g., 'PROJ-123')

        Returns:
            Cached ticket data as dictionary, or None if not found/expired
        """
        with sqlite3.connect(self.cache_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT ticket_data, expires_at, access_count
                FROM jira_tickets 
                WHERE ticket_key = ? AND expires_at > CURRENT_TIMESTAMP
            """,
                (ticket_key,),
            )

            row = cursor.fetchone()
            if row:
                # Update access statistics
                cursor.execute(
                    """
                    UPDATE jira_tickets 
                    SET access_count = ?, last_accessed = CURRENT_TIMESTAMP
                    WHERE ticket_key = ?
                """,
                    (row["access_count"] + 1, ticket_key),
                )
                conn.commit()

                self.cache_hits += 1
                logger.debug(f"Cache HIT for ticket {ticket_key}")

                import json

                return json.loads(row["ticket_data"])

            self.cache_misses += 1
            logger.debug(f"Cache MISS for ticket {ticket_key}")
            return None

    def store_ticket(self, ticket_key: str, ticket_data: dict[str, Any]) -> None:
        """Store ticket data in cache with TTL.

        Args:
            ticket_key: JIRA ticket key (e.g., 'PROJ-123')
            ticket_data: Complete ticket data from JIRA API
        """
        import json

        # Calculate expiry time
        expires_at = datetime.now() + timedelta(hours=self.ttl_hours)

        # Extract key fields for efficient querying
        project_key = ticket_data.get("project_id", ticket_key.split("-")[0])
        story_points = ticket_data.get("story_points")
        status = ticket_data.get("status")
        issue_type = ticket_data.get("issue_type")
        assignee = (
            ticket_data.get("assignee", {}).get("display_name")
            if ticket_data.get("assignee")
            else None
        )
        reporter = (
            ticket_data.get("reporter", {}).get("display_name")
            if ticket_data.get("reporter")
            else None
        )
        created_at = ticket_data.get("created_date")
        updated_at = ticket_data.get("updated_date")
        resolved_at = ticket_data.get("resolved_date")

        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO jira_tickets (
                    ticket_key, project_key, ticket_data, story_points, status,
                    issue_type, assignee, reporter, created_at, updated_at,
                    resolved_at, expires_at, access_count, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
            """,
                (
                    ticket_key,
                    project_key,
                    json.dumps(ticket_data),
                    story_points,
                    status,
                    issue_type,
                    assignee,
                    reporter,
                    created_at,
                    updated_at,
                    resolved_at,
                    expires_at,
                ),
            )
            conn.commit()

        self.cache_stores += 1
        logger.debug(f"Cached ticket {ticket_key} (expires: {expires_at})")

    def get_project_tickets(
        self, project_key: str, include_expired: bool = False
    ) -> list[dict[str, Any]]:
        """Get all cached tickets for a project.

        Args:
            project_key: JIRA project key
            include_expired: Whether to include expired entries

        Returns:
            List of cached ticket data dictionaries
        """
        import json

        where_clause = "WHERE project_key = ?"
        params = [project_key]

        if not include_expired:
            where_clause += " AND expires_at > CURRENT_TIMESTAMP"

        with sqlite3.connect(self.cache_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT ticket_data FROM jira_tickets {where_clause}
                ORDER BY updated_at DESC
            """,
                params,
            )

            tickets = []
            for row in cursor.fetchall():
                tickets.append(json.loads(row["ticket_data"]))

            return tickets

    def invalidate_ticket(self, ticket_key: str) -> bool:
        """Mark a specific ticket as expired/invalid.

        Args:
            ticket_key: JIRA ticket key to invalidate

        Returns:
            True if ticket was found and invalidated, False otherwise
        """
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE jira_tickets 
                SET expires_at = DATETIME('now', '-1 hour')
                WHERE ticket_key = ?
            """,
                (ticket_key,),
            )
            conn.commit()

            return cursor.rowcount > 0

    def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of expired entries removed
        """
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM jira_tickets WHERE expires_at <= CURRENT_TIMESTAMP
            """
            )
            removed = cursor.rowcount
            conn.commit()

            if removed > 0:
                logger.info(f"Cleaned up {removed} expired cache entries")

            return removed

    def clear_cache(self) -> int:
        """Clear all cached tickets.

        Returns:
            Number of entries removed
        """
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM jira_tickets")
            count = cursor.fetchone()[0]

            cursor.execute("DELETE FROM jira_tickets")
            conn.commit()

            logger.info(f"Cleared all {count} cached tickets")
            return count

    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary with cache performance and storage metrics
        """
        with sqlite3.connect(self.cache_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Basic counts
            cursor.execute("SELECT COUNT(*) as total FROM jira_tickets")
            total_tickets = cursor.fetchone()["total"]

            cursor.execute(
                """
                SELECT COUNT(*) as fresh FROM jira_tickets 
                WHERE expires_at > CURRENT_TIMESTAMP
            """
            )
            fresh_tickets = cursor.fetchone()["fresh"]

            cursor.execute(
                """
                SELECT COUNT(*) as expired FROM jira_tickets 
                WHERE expires_at <= CURRENT_TIMESTAMP
            """
            )
            expired_tickets = cursor.fetchone()["expired"]

            # Project distribution
            cursor.execute(
                """
                SELECT project_key, COUNT(*) as count 
                FROM jira_tickets 
                WHERE expires_at > CURRENT_TIMESTAMP
                GROUP BY project_key 
                ORDER BY count DESC
                LIMIT 10
            """
            )
            project_distribution = {row["project_key"]: row["count"] for row in cursor.fetchall()}

            # Access patterns
            cursor.execute(
                """
                SELECT AVG(access_count) as avg_access, 
                       MAX(access_count) as max_access,
                       COUNT(*) as accessed_tickets
                FROM jira_tickets 
                WHERE access_count > 1 AND expires_at > CURRENT_TIMESTAMP
            """
            )
            access_stats = cursor.fetchone()

            # Recent activity
            cursor.execute(
                """
                SELECT COUNT(*) as recent FROM jira_tickets 
                WHERE cached_at > DATETIME('now', '-24 hours')
            """
            )
            recent_cached = cursor.fetchone()["recent"]

            # Database size
            try:
                db_size_mb = self.cache_path.stat().st_size / (1024 * 1024)
            except FileNotFoundError:
                db_size_mb = 0

            # Session performance
            session_duration = (datetime.now() - self.session_start).total_seconds()
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

            # Time savings estimation
            api_calls_avoided = self.cache_hits
            estimated_time_saved = api_calls_avoided * 0.5  # 0.5 seconds per API call

            return {
                # Storage metrics
                "total_tickets": total_tickets,
                "fresh_tickets": fresh_tickets,
                "expired_tickets": expired_tickets,
                "database_size_mb": db_size_mb,
                "recent_cached_24h": recent_cached,
                # Performance metrics
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_stores": self.cache_stores,
                "hit_rate_percent": hit_rate,
                "total_requests": total_requests,
                # Time savings
                "api_calls_avoided": api_calls_avoided,
                "estimated_time_saved_seconds": estimated_time_saved,
                "session_duration_seconds": session_duration,
                # Access patterns
                "project_distribution": project_distribution,
                "avg_access_count": float(access_stats["avg_access"] or 0),
                "max_access_count": access_stats["max_access"] or 0,
                "frequently_accessed_tickets": access_stats["accessed_tickets"] or 0,
                # Configuration
                "ttl_hours": self.ttl_hours,
                "cache_path": str(self.cache_path),
            }

    def print_cache_summary(self) -> None:
        """Print user-friendly cache performance summary."""
        stats = self.get_cache_stats()

        print("🎫 JIRA Ticket Cache Summary")
        print("─" * 40)

        # Cache contents
        print("📦 Cache Contents:")
        print(
            f"   • Total Tickets: {stats['total_tickets']:,} ({stats['fresh_tickets']:,} fresh, {stats['expired_tickets']:,} expired)"
        )
        print(f"   • Database Size: {stats['database_size_mb']:.1f} MB")
        print(f"   • Recent Activity: {stats['recent_cached_24h']:,} tickets cached in last 24h")

        # Project distribution
        if stats["project_distribution"]:
            print("\n📊 Top Projects:")
            for project, count in list(stats["project_distribution"].items())[:5]:
                print(f"   • {project}: {count:,} tickets")

        # Performance metrics
        if stats["total_requests"] > 0:
            print("\n⚡ Session Performance:")
            print(
                f"   • Hit Rate: {stats['hit_rate_percent']:.1f}% ({stats['cache_hits']:,}/{stats['total_requests']:,})"
            )
            print(f"   • API Calls Avoided: {stats['api_calls_avoided']:,}")

            if stats["estimated_time_saved_seconds"] > 60:
                print(f"   • Time Saved: {stats['estimated_time_saved_seconds'] / 60:.1f} minutes")
            else:
                print(f"   • Time Saved: {stats['estimated_time_saved_seconds']:.1f} seconds")

        # Access patterns
        if stats["frequently_accessed_tickets"] > 0:
            print("\n🔄 Access Patterns:")
            print(f"   • Frequently Accessed: {stats['frequently_accessed_tickets']:,} tickets")
            print(f"   • Average Access Count: {stats['avg_access_count']:.1f}")
            print(f"   • Most Accessed: {stats['max_access_count']} times")

        # Performance insights
        if stats["hit_rate_percent"] > 80:
            print("   ✅ Excellent cache performance!")
        elif stats["hit_rate_percent"] > 50:
            print("   👍 Good cache performance")
        elif stats["total_requests"] > 0:
            print("   ⚠️  Consider adjusting TTL or clearing stale entries")

        print()


class JIRAAdapter(BasePlatformAdapter):
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
        import os

        # Check debug mode
        debug_mode = os.getenv("GITFLOW_DEBUG", "").lower() in ("1", "true", "yes")
        if debug_mode:
            print(f"   🔍 JIRA adapter __init__ called with config keys: {list(config.keys())}")

        super().__init__(config)

        # Required configuration (use defaults for capability checking)
        self.base_url = config.get("base_url", "https://example.atlassian.net").rstrip("/")
        self.username = config.get("username", "user@example.com")
        self.api_token = config.get("api_token", "dummy-token")

        # Debug output
        logger.info(
            f"JIRA adapter init: base_url={self.base_url}, username={self.username}, has_token={bool(self.api_token and self.api_token != 'dummy-token')}"
        )
        if debug_mode:
            print(
                f"   🔍 JIRA adapter received: username={self.username}, has_token={bool(self.api_token and self.api_token != 'dummy-token')}, base_url={self.base_url}"
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
        # Try configured story point fields first
        for field_id in self.story_point_fields:
            if field_id in fields and fields[field_id] is not None:
                value = fields[field_id]
                try:
                    if isinstance(value, (int, float)):
                        return int(value)
                    elif isinstance(value, str) and value.strip():
                        return int(float(value.strip()))
                except (ValueError, TypeError):
                    continue

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
                except Exception:
                    pass

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
