"""JIRA API integration for story point and ticket enrichment."""

import base64
import socket
import time
from datetime import datetime
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, RequestException, Timeout
from urllib3.util.retry import Retry

from ..core.cache import GitAnalysisCache


class JIRAIntegration:
    """Integrate with JIRA API for ticket and story point data."""

    def __init__(
        self,
        base_url: str,
        username: str,
        api_token: str,
        cache: GitAnalysisCache,
        story_point_fields: Optional[list[str]] = None,
        dns_timeout: int = 10,
        connection_timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        enable_proxy: bool = False,
        proxy_url: Optional[str] = None,
    ):
        """Initialize JIRA integration.

        Args:
            base_url: JIRA instance base URL (e.g., https://company.atlassian.net)
            username: JIRA username/email
            api_token: JIRA API token
            cache: Git analysis cache for storing JIRA data
            story_point_fields: List of custom field IDs for story points
            dns_timeout: DNS resolution timeout in seconds (default: 10)
            connection_timeout: HTTP connection timeout in seconds (default: 30)
            max_retries: Maximum number of retry attempts (default: 3)
            backoff_factor: Exponential backoff factor for retries (default: 1.0)
            enable_proxy: Whether to use proxy settings (default: False)
            proxy_url: Proxy URL if proxy is enabled (default: None)
        """
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.dns_timeout = dns_timeout
        self.connection_timeout = connection_timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.enable_proxy = enable_proxy
        self.proxy_url = proxy_url

        # Network connectivity status
        self._connection_validated = False
        self._last_dns_check = 0
        self._dns_check_interval = 300  # 5 minutes

        # Set up authentication
        credentials = base64.b64encode(f"{username}:{api_token}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {credentials}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "GitFlow-Analytics-JIRA/1.0",
        }

        # Default story point field names/IDs
        self.story_point_fields = story_point_fields or [
            "customfield_10016",  # Common story points field
            "customfield_10021",  # Alternative field
            "Story Points",  # Field name
            "storypoints",  # Alternative name
            "customfield_10002",  # Another common ID
        ]

        # Cache for field mapping
        self._field_mapping = None

        # Initialize HTTP session with enhanced error handling
        self._session = self._create_resilient_session()

    def enrich_commits_with_jira_data(self, commits: list[dict[str, Any]]) -> None:
        """Enrich commits with JIRA story points by looking up ticket references.

        Args:
            commits: List of commit dictionaries to enrich
        """
        # Validate network connectivity before attempting JIRA operations
        if not self._validate_network_connectivity():
            print("   ⚠️  JIRA network connectivity issues detected, skipping commit enrichment")
            return

        # Collect all unique JIRA tickets from commits
        jira_tickets = set()
        for commit in commits:
            ticket_refs = commit.get("ticket_references", [])
            for ref in ticket_refs:
                if isinstance(ref, dict) and ref.get("platform") == "jira":
                    jira_tickets.add(ref["id"])
                elif isinstance(ref, str) and self._is_jira_ticket(ref):
                    jira_tickets.add(ref)

        if not jira_tickets:
            return

        # Fetch ticket data from JIRA with enhanced error handling
        ticket_data = self._fetch_tickets_batch(list(jira_tickets))

        # Enrich commits with story points
        for commit in commits:
            commit_story_points = 0
            ticket_refs = commit.get("ticket_references", [])

            for ref in ticket_refs:
                ticket_id = None
                if isinstance(ref, dict) and ref.get("platform") == "jira":
                    ticket_id = ref["id"]
                elif isinstance(ref, str) and self._is_jira_ticket(ref):
                    ticket_id = ref

                if ticket_id and ticket_id in ticket_data:
                    points = ticket_data[ticket_id].get("story_points", 0)
                    if points:
                        commit_story_points = max(commit_story_points, points)

            if commit_story_points > 0:
                commit["story_points"] = commit_story_points

    def enrich_prs_with_jira_data(self, prs: list[dict[str, Any]]) -> None:
        """Enrich PRs with JIRA story points.

        Args:
            prs: List of PR dictionaries to enrich
        """
        # Validate network connectivity before attempting JIRA operations
        if not self._validate_network_connectivity():
            print("   ⚠️  JIRA network connectivity issues detected, skipping PR enrichment")
            return

        # Similar to commits, extract JIRA tickets from PR titles/descriptions
        for pr in prs:
            pr_text = f"{pr.get('title', '')} {pr.get('description', '')}"
            jira_tickets = self._extract_jira_tickets(pr_text)

            if jira_tickets:
                ticket_data = self._fetch_tickets_batch(list(jira_tickets))

                # Use the highest story point value found
                max_points = 0
                for ticket_id in jira_tickets:
                    if ticket_id in ticket_data:
                        points = ticket_data[ticket_id].get("story_points", 0)
                        max_points = max(max_points, points)

                if max_points > 0:
                    pr["story_points"] = max_points

    def _fetch_tickets_batch(self, ticket_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch multiple tickets from JIRA API with optimized caching.

        WHY: This method implements comprehensive caching to minimize JIRA API calls,
        which are often the slowest part of the analysis. It uses bulk cache lookups
        and provides detailed cache hit/miss metrics.

        Args:
            ticket_ids: List of JIRA ticket IDs

        Returns:
            Dictionary mapping ticket ID to ticket data
        """
        if not ticket_ids:
            return {}

        # Bulk cache lookup for better performance
        cached_tickets = self._get_cached_tickets_bulk(ticket_ids)
        tickets_to_fetch = [tid for tid in ticket_ids if tid not in cached_tickets]

        # Track cache performance
        cache_hits = len(cached_tickets)
        cache_misses = len(tickets_to_fetch)

        if cache_hits > 0 or cache_misses > 0:
            print(
                f"   📊 JIRA cache: {cache_hits} hits, {cache_misses} misses ({cache_hits / (cache_hits + cache_misses) * 100:.1f}% hit rate)"
            )

        # Fetch missing tickets from JIRA
        if tickets_to_fetch:
            # JIRA JQL has a limit, so batch the requests
            batch_size = 50
            new_tickets = []  # Collect new tickets for bulk caching

            for i in range(0, len(tickets_to_fetch), batch_size):
                batch = tickets_to_fetch[i : i + batch_size]
                jql = f"key in ({','.join(batch)})"

                try:
                    print(f"   🔍 Fetching {len(batch)} JIRA tickets from API...")
                    response = self._session.get(
                        f"{self.base_url}/rest/api/3/search/jql",
                        params={
                            "jql": jql,
                            "fields": "*all",  # Get all fields to find story points
                            "maxResults": batch_size,
                        },
                        timeout=self.connection_timeout,
                    )
                    response.raise_for_status()

                    data = response.json()
                    for issue in data.get("issues", []):
                        ticket_data = self._extract_ticket_data(issue)
                        cached_tickets[ticket_data["id"]] = ticket_data
                        new_tickets.append(ticket_data)

                except ConnectionError as e:
                    print(f"   ❌ JIRA DNS/connection error: {self._format_network_error(e)}")
                    print(
                        f"      Troubleshooting: Check network connectivity and DNS resolution for {self.base_url}"
                    )
                    break  # Stop processing batches on network errors
                except Timeout as e:
                    print(f"   ⏱️  JIRA request timeout: {e}")
                    print("      Consider increasing timeout settings or checking network latency")
                except RequestException as e:
                    print(f"   ⚠️  Failed to fetch JIRA tickets: {e}")

            # Bulk cache all new tickets
            if new_tickets:
                self._cache_tickets_bulk(new_tickets)
                print(f"   💾 Cached {len(new_tickets)} new JIRA tickets")

        return cached_tickets

    def _extract_ticket_data(self, issue: dict[str, Any]) -> dict[str, Any]:
        """Extract relevant data from JIRA issue.

        Args:
            issue: JIRA issue data from API

        Returns:
            Dictionary with extracted ticket data
        """
        fields = issue.get("fields", {})

        # Extract story points from various possible fields
        story_points = 0
        for field_id in self.story_point_fields:
            if field_id in fields and fields[field_id] is not None:
                try:
                    story_points = float(fields[field_id])
                    break
                except (ValueError, TypeError):
                    continue

        return {
            "id": issue["key"],
            "summary": fields.get("summary", ""),
            "status": fields.get("status", {}).get("name", ""),
            "story_points": int(story_points) if story_points else 0,
            "assignee": (
                fields.get("assignee", {}).get("displayName", "") if fields.get("assignee") else ""
            ),
            "created": fields.get("created", ""),
            "updated": fields.get("updated", ""),
        }

    def _is_jira_ticket(self, text: str) -> bool:
        """Check if text matches JIRA ticket pattern."""
        import re

        return bool(re.match(r"^[A-Z]{2,10}-\d+$", text))

    def _extract_jira_tickets(self, text: str) -> set[str]:
        """Extract JIRA ticket IDs from text."""
        import re

        pattern = r"([A-Z]{2,10}-\d+)"
        matches = re.findall(pattern, text)
        return set(matches)

    def _get_cached_ticket(self, ticket_id: str) -> Optional[dict[str, Any]]:
        """Get ticket data from cache.

        WHY: JIRA API calls are expensive and slow. Caching ticket data
        significantly improves performance on repeated runs over the same
        time period, especially when analyzing multiple repositories.

        Args:
            ticket_id: JIRA ticket ID (e.g., "PROJ-123")

        Returns:
            Cached ticket data or None if not found/stale
        """
        with self.cache.get_session() as session:
            from ..models.database import IssueCache

            cached_ticket = (
                session.query(IssueCache)
                .filter(IssueCache.platform == "jira", IssueCache.issue_id == ticket_id)
                .first()
            )

            if cached_ticket and not self._is_ticket_stale(cached_ticket.cached_at):
                self.cache.cache_hits += 1
                if self.cache.debug_mode:
                    print(f"DEBUG: JIRA cache HIT for ticket {ticket_id}")

                return {
                    "id": cached_ticket.issue_id,
                    "summary": cached_ticket.title or "",
                    "status": cached_ticket.status or "",
                    "story_points": cached_ticket.story_points or 0,
                    "assignee": cached_ticket.assignee or "",
                    "created": (
                        cached_ticket.created_at.isoformat() if cached_ticket.created_at else ""
                    ),
                    "updated": (
                        cached_ticket.updated_at.isoformat() if cached_ticket.updated_at else ""
                    ),
                    "platform_data": cached_ticket.platform_data or {},
                }

            self.cache.cache_misses += 1
            if self.cache.debug_mode:
                print(f"DEBUG: JIRA cache MISS for ticket {ticket_id}")
            return None

    def _cache_ticket(self, ticket_id: str, ticket_data: dict[str, Any]) -> None:
        """Cache ticket data.

        WHY: Caching JIRA ticket data prevents redundant API calls and
        significantly improves performance on subsequent runs. The cache
        respects TTL settings to ensure data freshness.

        Args:
            ticket_id: JIRA ticket ID
            ticket_data: Ticket data from JIRA API
        """
        # Use the existing cache_issue method which handles JIRA tickets
        cache_data = {
            "id": ticket_id,
            "project_key": self._extract_project_key(ticket_id),
            "title": ticket_data.get("summary", ""),
            "description": "",  # Not typically needed for analytics
            "status": ticket_data.get("status", ""),
            "assignee": ticket_data.get("assignee", ""),
            "created_at": self._parse_jira_date(ticket_data.get("created")),
            "updated_at": self._parse_jira_date(ticket_data.get("updated")),
            "story_points": ticket_data.get("story_points", 0),
            "labels": [],  # Could extract from JIRA data if needed
            "platform_data": ticket_data,  # Store full JIRA response for future use
        }

        self.cache.cache_issue("jira", cache_data)

    def _is_ticket_stale(self, cached_at: datetime) -> bool:
        """Check if cached ticket data is stale based on cache TTL.

        Args:
            cached_at: When the ticket was cached

        Returns:
            True if stale and should be refreshed, False if still fresh
        """
        from datetime import timedelta

        if self.cache.ttl_hours == 0:  # No expiration
            return False

        stale_threshold = datetime.utcnow() - timedelta(hours=self.cache.ttl_hours)
        return cached_at < stale_threshold

    def _extract_project_key(self, ticket_id: str) -> str:
        """Extract project key from JIRA ticket ID.

        Args:
            ticket_id: JIRA ticket ID (e.g., "PROJ-123")

        Returns:
            Project key (e.g., "PROJ")
        """
        return ticket_id.split("-")[0] if "-" in ticket_id else ticket_id

    def _parse_jira_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse JIRA date string to datetime object.

        Args:
            date_str: JIRA date string or None

        Returns:
            Parsed datetime object or None
        """
        if not date_str:
            return None

        try:
            # JIRA typically returns ISO format dates
            from dateutil import parser

            return parser.parse(date_str).replace(tzinfo=None)  # Store as naive UTC
        except (ValueError, ImportError):
            # Fallback for basic ISO format
            try:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                return None

    def _get_cached_tickets_bulk(self, ticket_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Get multiple tickets from cache in a single query.

        WHY: Bulk cache lookups are much more efficient than individual lookups
        when checking many tickets, reducing database overhead significantly.

        Args:
            ticket_ids: List of JIRA ticket IDs to look up

        Returns:
            Dictionary mapping ticket ID to cached data (only non-stale entries)
        """
        if not ticket_ids:
            return {}

        cached_tickets = {}
        with self.cache.get_session() as session:
            from ..models.database import IssueCache

            cached_results = (
                session.query(IssueCache)
                .filter(IssueCache.platform == "jira", IssueCache.issue_id.in_(ticket_ids))
                .all()
            )

            for cached in cached_results:
                if not self._is_ticket_stale(cached.cached_at):
                    ticket_data = {
                        "id": cached.issue_id,
                        "summary": cached.title or "",
                        "status": cached.status or "",
                        "story_points": cached.story_points or 0,
                        "assignee": cached.assignee or "",
                        "created": cached.created_at.isoformat() if cached.created_at else "",
                        "updated": cached.updated_at.isoformat() if cached.updated_at else "",
                        "platform_data": cached.platform_data or {},
                    }
                    cached_tickets[cached.issue_id] = ticket_data

        return cached_tickets

    def _cache_tickets_bulk(self, tickets: list[dict[str, Any]]) -> None:
        """Cache multiple tickets in a single transaction.

        WHY: Bulk caching is more efficient than individual cache operations,
        reducing database overhead and improving performance when caching
        many tickets from JIRA API responses.

        Args:
            tickets: List of ticket data dictionaries to cache
        """
        if not tickets:
            return

        for ticket_data in tickets:
            # Use individual cache method which handles upserts properly
            self._cache_ticket(ticket_data["id"], ticket_data)

    def validate_connection(self) -> bool:
        """Validate JIRA connection and credentials.

        Returns:
            True if connection is valid
        """
        try:
            # First validate network connectivity
            if not self._validate_network_connectivity():
                return False

            response = self._session.get(
                f"{self.base_url}/rest/api/3/myself", timeout=self.connection_timeout
            )
            response.raise_for_status()
            self._connection_validated = True
            return True
        except ConnectionError as e:
            print(f"   ❌ JIRA DNS/connection error: {self._format_network_error(e)}")
            print(
                f"      Troubleshooting: Check network connectivity and DNS resolution for {self.base_url}"
            )
            return False
        except Timeout as e:
            print(f"   ⏱️  JIRA connection timeout: {e}")
            print("      Consider increasing timeout settings or checking network latency")
            return False
        except RequestException as e:
            print(f"   ❌ JIRA connection failed: {e}")
            return False

    def discover_fields(self) -> dict[str, dict[str, str]]:
        """Discover all available fields in JIRA instance.

        Returns:
            Dictionary mapping field IDs to their names and types
        """
        try:
            # Validate network connectivity first
            if not self._validate_network_connectivity():
                return {}

            response = self._session.get(
                f"{self.base_url}/rest/api/3/field", timeout=self.connection_timeout
            )
            response.raise_for_status()

            fields = {}
            for field in response.json():
                field_id = field.get("id", "")
                field_name = field.get("name", "")
                field_type = (
                    field.get("schema", {}).get("type", "unknown")
                    if field.get("schema")
                    else "unknown"
                )

                # Look for potential story point fields
                if any(
                    term in field_name.lower() for term in ["story", "point", "estimate", "size"]
                ):
                    fields[field_id] = {
                        "name": field_name,
                        "type": field_type,
                        "is_custom": field.get("custom", False),
                    }
                    print(
                        f"   📊 Potential story point field: {field_id} = '{field_name}' (type: {field_type})"
                    )

            return fields

        except ConnectionError as e:
            print(
                f"   ❌ JIRA DNS/connection error during field discovery: {self._format_network_error(e)}"
            )
            print(
                f"      Troubleshooting: Check network connectivity and DNS resolution for {self.base_url}"
            )
            return {}
        except Timeout as e:
            print(f"   ⏱️  JIRA field discovery timeout: {e}")
            print("      Consider increasing timeout settings or checking network latency")
            return {}
        except RequestException as e:
            print(f"   ⚠️  Failed to discover JIRA fields: {e}")
            return {}

    def _create_resilient_session(self) -> requests.Session:
        """Create HTTP session with enhanced retry logic and DNS error handling.

        WHY: DNS resolution failures and network issues are common when connecting
        to external JIRA instances. This session provides resilient connections
        with exponential backoff and comprehensive error handling.

        Returns:
            Configured requests session with retry strategy and network resilience.
        """
        session = requests.Session()

        # Configure retry strategy for network resilience
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            raise_on_status=False,  # Let us handle status codes
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers
        session.headers.update(self.headers)

        # Configure proxy if enabled
        if self.enable_proxy and self.proxy_url:
            session.proxies = {
                "http": self.proxy_url,
                "https": self.proxy_url,
            }
            print(f"   🌐 Using proxy: {self.proxy_url}")

        # Set default timeout
        session.timeout = self.connection_timeout

        return session

    def _validate_network_connectivity(self) -> bool:
        """Validate network connectivity to JIRA instance.

        WHY: DNS resolution errors are a common cause of JIRA integration failures.
        This method performs proactive network validation to detect issues early
        and provide better error messages for troubleshooting.

        Returns:
            True if network connectivity is available, False otherwise.
        """
        current_time = time.time()

        # Skip check if recently validated (within interval)
        if (
            self._connection_validated
            and current_time - self._last_dns_check < self._dns_check_interval
        ):
            return True

        try:
            # Extract hostname from base URL
            from urllib.parse import urlparse

            parsed_url = urlparse(self.base_url)
            hostname = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)

            if not hostname:
                print(f"   ❌ Invalid JIRA URL format: {self.base_url}")
                return False

            # Test DNS resolution
            print(f"   🔍 Validating DNS resolution for {hostname}...")
            socket.setdefaulttimeout(self.dns_timeout)

            # Attempt to resolve hostname
            addr_info = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
            if not addr_info:
                print(f"   ❌ DNS resolution failed: No addresses found for {hostname}")
                return False

            # Test basic connectivity
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.dns_timeout)
            try:
                result = sock.connect_ex((addr_info[0][4][0], port))
                if result == 0:
                    print(f"   ✅ Network connectivity confirmed to {hostname}:{port}")
                    self._connection_validated = True
                    self._last_dns_check = current_time
                    return True
                else:
                    print(f"   ❌ Connection failed to {hostname}:{port} (error code: {result})")
                    return False
            finally:
                sock.close()

        except socket.gaierror as e:
            print(f"   ❌ DNS resolution error: {self._format_dns_error(e)}")
            print(f"      Hostname: {hostname}")
            print("      Troubleshooting:")
            print(f"        1. Verify the hostname is correct: {hostname}")
            print("        2. Check your internet connection")
            print(f"        3. Verify DNS settings (try: nslookup {hostname})")
            print("        4. Check if behind corporate firewall/proxy")
            print("        5. Verify JIRA instance is accessible externally")
            return False
        except socket.timeout:
            print(f"   ⏱️  DNS resolution timeout for {hostname} (>{self.dns_timeout}s)")
            print("      Consider increasing dns_timeout or checking network latency")
            return False
        except Exception as e:
            print(f"   ❌ Network validation error: {e}")
            return False
        finally:
            socket.setdefaulttimeout(None)  # Reset to default

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

    def _format_dns_error(self, error: socket.gaierror) -> str:
        """Format DNS resolution errors with specific guidance.

        Args:
            error: The DNS resolution error that occurred.

        Returns:
            Formatted DNS error message with troubleshooting guidance.
        """
        error_code = error.errno if hasattr(error, "errno") else "unknown"
        error_msg = str(error)

        if error_code == 8 or "nodename nor servname provided" in error_msg:
            return f"Hostname not found in DNS (error code: {error_code})"
        elif error_code == -2 or "Name or service not known" in error_msg:
            return f"DNS name resolution failed (error code: {error_code})"
        elif error_code == -3 or "Temporary failure in name resolution" in error_msg:
            return f"Temporary DNS failure - try again later (error code: {error_code})"
        else:
            return f"DNS error (code: {error_code}, message: {error_msg})"
