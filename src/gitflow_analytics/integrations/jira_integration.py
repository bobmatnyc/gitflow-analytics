"""JIRA API integration for story point and ticket enrichment."""

import base64
from typing import Any, Optional

import requests
from requests.exceptions import RequestException

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
    ):
        """Initialize JIRA integration.

        Args:
            base_url: JIRA instance base URL (e.g., https://company.atlassian.net)
            username: JIRA username/email
            api_token: JIRA API token
            cache: Git analysis cache for storing JIRA data
            story_point_fields: List of custom field IDs for story points
        """
        self.base_url = base_url.rstrip("/")
        self.cache = cache

        # Set up authentication
        credentials = base64.b64encode(f"{username}:{api_token}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {credentials}",
            "Accept": "application/json",
            "Content-Type": "application/json",
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

    def enrich_commits_with_jira_data(self, commits: list[dict[str, Any]]) -> None:
        """Enrich commits with JIRA story points by looking up ticket references.

        Args:
            commits: List of commit dictionaries to enrich
        """
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

        # Fetch ticket data from JIRA
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
        """Fetch multiple tickets from JIRA API.

        Args:
            ticket_ids: List of JIRA ticket IDs

        Returns:
            Dictionary mapping ticket ID to ticket data
        """
        if not ticket_ids:
            return {}

        # Check cache first
        cached_tickets = {}
        tickets_to_fetch = []

        for ticket_id in ticket_ids:
            cached = self._get_cached_ticket(ticket_id)
            if cached:
                cached_tickets[ticket_id] = cached
            else:
                tickets_to_fetch.append(ticket_id)

        # Fetch missing tickets from JIRA
        if tickets_to_fetch:
            # JIRA JQL has a limit, so batch the requests
            batch_size = 50
            for i in range(0, len(tickets_to_fetch), batch_size):
                batch = tickets_to_fetch[i : i + batch_size]
                jql = f"key in ({','.join(batch)})"

                try:
                    response = requests.get(
                        f"{self.base_url}/rest/api/3/search",
                        headers=self.headers,
                        params={
                            "jql": jql,
                            "fields": "*all",  # Get all fields to find story points
                            "maxResults": batch_size,
                        },
                    )
                    response.raise_for_status()

                    data = response.json()
                    for issue in data.get("issues", []):
                        ticket_data = self._extract_ticket_data(issue)
                        cached_tickets[ticket_data["id"]] = ticket_data
                        self._cache_ticket(ticket_data["id"], ticket_data)

                except RequestException as e:
                    print(f"   ⚠️  Failed to fetch JIRA tickets: {e}")

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
        """Get ticket data from cache."""
        # TODO: Implement cache lookup using self.cache
        # For now, return None to always fetch from API
        return None

    def _cache_ticket(self, ticket_id: str, ticket_data: dict[str, Any]) -> None:
        """Cache ticket data."""
        # TODO: Implement cache storage using self.cache
        pass

    def validate_connection(self) -> bool:
        """Validate JIRA connection and credentials.

        Returns:
            True if connection is valid
        """
        try:
            response = requests.get(f"{self.base_url}/rest/api/3/myself", headers=self.headers)
            response.raise_for_status()
            return True
        except RequestException as e:
            print(f"   ❌ JIRA connection failed: {e}")
            return False

    def discover_fields(self) -> dict[str, dict[str, str]]:
        """Discover all available fields in JIRA instance.

        Returns:
            Dictionary mapping field IDs to their names and types
        """
        try:
            response = requests.get(f"{self.base_url}/rest/api/3/field", headers=self.headers)
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

        except RequestException as e:
            print(f"   ⚠️  Failed to discover JIRA fields: {e}")
            return {}
