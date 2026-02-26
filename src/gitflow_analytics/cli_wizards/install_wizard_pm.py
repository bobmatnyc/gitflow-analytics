"""PM tool (JIRA, Linear, ClickUp) setup methods for InstallWizard.

Extracted from install_wizard.py to keep it under 800 lines.
Methods: _setup_jira, _setup_linear, _validate_linear, _setup_clickup,
         _validate_clickup, _validate_jira, _store_jira_config,
         _discover_jira_fields
"""

import logging
import os
from typing import Optional

import click
import requests
import yaml

logger = logging.getLogger(__name__)


class InstallWizardPMMixin:
    """Mixin adding PM platform setup methods to InstallWizard."""

    def _setup_jira(self) -> None:
        """Setup JIRA integration (optional)."""
        click.echo("\n📋 Step 3: JIRA Setup (OPTIONAL)")
        click.echo("-" * 50)

        if not click.confirm("Enable JIRA integration?", default=False):
            click.echo("⏭️  Skipping JIRA setup")
            return

        click.echo("\nJIRA Configuration:")
        click.echo("You'll need:")
        click.echo("  • JIRA instance URL (e.g., https://yourcompany.atlassian.net)")
        click.echo("  • Email address for API authentication")
        click.echo(
            "  • API token from: https://id.atlassian.com/manage-profile/security/api-tokens"
        )
        click.echo()

        max_retries = 3
        for attempt in range(max_retries):
            if attempt > 0:
                # Add exponential backoff to prevent rate limiting
                delay = 2 ** (attempt - 1)  # 1, 2, 4 seconds
                click.echo(f"⏳ Waiting {delay} seconds before retry...")
                time.sleep(delay)

            base_url = click.prompt("JIRA base URL", type=str).strip()
            access_user = click.prompt("JIRA email", type=str).strip()
            access_token = self._get_password("JIRA API token: ", "JIRA token").strip()

            if not all([base_url, access_user, access_token]):
                click.echo("❌ All JIRA fields are required")
                continue

            # Normalize base_url
            base_url = base_url.rstrip("/")

            # Validate JIRA credentials
            if not self.skip_validation:
                click.echo("🔍 Validating JIRA credentials...")
                if self._validate_jira(base_url, access_user, access_token):
                    click.echo("✅ JIRA credentials validated!")
                    self._store_jira_config(base_url, access_user, access_token)
                    self._discover_jira_fields(base_url, access_user, access_token)
                    return
                else:
                    if attempt < max_retries - 1:
                        retry = click.confirm("JIRA validation failed. Try again?", default=True)
                        if not retry:
                            click.echo("⏭️  Skipping JIRA setup")
                            return
                    else:
                        click.echo("❌ Maximum retry attempts reached")
                        click.echo("⏭️  Skipping JIRA setup")
                        return
            else:
                # Skip validation mode
                self._store_jira_config(base_url, access_user, access_token)
                return

        click.echo("⏭️  Skipping JIRA setup")

    def _setup_linear(self) -> Optional[dict]:
        """Setup Linear integration (optional).

        Returns:
            Linear configuration dict if successful, None otherwise
        """
        click.echo("\n📋 Linear Setup")
        click.echo("-" * 50)
        click.echo("\nLinear Configuration:")
        click.echo("You'll need:")
        click.echo("  • Linear API key from: https://linear.app/settings/api")
        click.echo("  • Optional: Team IDs to filter issues")
        click.echo()

        max_retries = 3
        for attempt in range(max_retries):
            if attempt > 0:
                # Add exponential backoff to prevent rate limiting
                delay = 2 ** (attempt - 1)  # 1, 2, 4 seconds
                click.echo(f"⏳ Waiting {delay} seconds before retry...")
                time.sleep(delay)

            api_key = self._get_password("Linear API key: ", "Linear API key").strip()

            if not api_key:
                click.echo("❌ API key cannot be empty")
                continue

            # Validate Linear credentials
            if not self.skip_validation:
                click.echo("🔍 Validating Linear API key...")
                if self._validate_linear(api_key):
                    click.echo("✅ Linear API key validated!")

                    # Optional: Team IDs
                    team_ids = click.prompt(
                        "Team IDs (comma-separated, press Enter to skip)",
                        type=str,
                        default="",
                        show_default=False,
                    ).strip()

                    # Store configuration
                    self.env_data["LINEAR_API_KEY"] = api_key
                    linear_config = {
                        "api_key": "${LINEAR_API_KEY}",
                    }

                    if team_ids:
                        team_list = [tid.strip() for tid in team_ids.split(",") if tid.strip()]
                        if team_list:
                            linear_config["team_ids"] = team_list
                            click.echo(f"✅ Configured {len(team_list)} team ID(s)")

                    return linear_config
                else:
                    if attempt < max_retries - 1:
                        retry = click.confirm("Linear validation failed. Try again?", default=True)
                        if not retry:
                            click.echo("⏭️  Skipping Linear setup")
                            return None
                    else:
                        click.echo("❌ Maximum retry attempts reached")
                        click.echo("⏭️  Skipping Linear setup")
                        return None
            else:
                # Skip validation mode
                self.env_data["LINEAR_API_KEY"] = api_key
                return {"api_key": "${LINEAR_API_KEY}"}

        click.echo("⏭️  Skipping Linear setup")
        return None

    def _validate_linear(self, api_key: str) -> bool:
        """Validate Linear API key.

        Args:
            api_key: Linear API key

        Returns:
            True if credentials are valid, False otherwise
        """
        # Suppress requests logging to prevent credential exposure
        urllib3_logger = logging.getLogger("urllib3")
        requests_logger = logging.getLogger("requests")
        original_urllib3 = urllib3_logger.level
        original_requests = requests_logger.level

        urllib3_logger.setLevel(logging.WARNING)
        requests_logger.setLevel(logging.WARNING)

        try:
            headers = {
                "Authorization": api_key,
                "Content-Type": "application/json",
            }

            # Simple GraphQL query to validate authentication
            query = {"query": "{ viewer { name } }"}

            response = requests.post(
                "https://api.linear.app/graphql",
                headers=headers,
                json=query,
                timeout=10,
                verify=True,
            )

            if response.status_code == 200:
                data = response.json()
                if "data" in data and "viewer" in data["data"]:
                    viewer_name = data["data"]["viewer"].get("name", "Unknown")
                    click.echo(f"   Authenticated as: {viewer_name}")
                    return True
                else:
                    click.echo("   Authentication failed: Invalid response")
                    return False
            else:
                click.echo(f"   Authentication failed (status {response.status_code})")
                return False

        except requests.exceptions.RequestException as e:
            # Never expose raw exception - could contain credentials
            error_type = type(e).__name__
            click.echo(f"   Connection error: {error_type}")
            logger.debug(f"Linear connection error type: {error_type}")
            return False
        except Exception as e:
            click.echo("   Linear validation failed")
            logger.error(f"Linear validation error type: {type(e).__name__}")
            return False
        finally:
            # Always restore logging levels
            urllib3_logger.setLevel(original_urllib3)
            requests_logger.setLevel(original_requests)

    def _setup_clickup(self) -> Optional[dict]:
        """Setup ClickUp integration (optional).

        Returns:
            ClickUp configuration dict if successful, None otherwise
        """
        click.echo("\n📋 ClickUp Setup")
        click.echo("-" * 50)
        click.echo("\nClickUp Configuration:")
        click.echo("You'll need:")
        click.echo("  • ClickUp API token from: https://app.clickup.com/settings/apps")
        click.echo("  • Workspace URL (e.g., https://app.clickup.com/12345/v/)")
        click.echo()

        max_retries = 3
        for attempt in range(max_retries):
            if attempt > 0:
                # Add exponential backoff to prevent rate limiting
                delay = 2 ** (attempt - 1)  # 1, 2, 4 seconds
                click.echo(f"⏳ Waiting {delay} seconds before retry...")
                time.sleep(delay)

            api_token = self._get_password("ClickUp API token: ", "ClickUp API token").strip()
            workspace_url = click.prompt("ClickUp workspace URL", type=str).strip()

            if not all([api_token, workspace_url]):
                click.echo("❌ All ClickUp fields are required")
                continue

            # Normalize workspace_url
            workspace_url = workspace_url.rstrip("/")

            # Validate ClickUp credentials
            if not self.skip_validation:
                click.echo("🔍 Validating ClickUp credentials...")
                if self._validate_clickup(api_token):
                    click.echo("✅ ClickUp credentials validated!")

                    # Store configuration
                    self.env_data["CLICKUP_API_TOKEN"] = api_token
                    self.env_data["CLICKUP_WORKSPACE_URL"] = workspace_url

                    clickup_config = {
                        "api_token": "${CLICKUP_API_TOKEN}",
                        "workspace_url": "${CLICKUP_WORKSPACE_URL}",
                    }

                    return clickup_config
                else:
                    if attempt < max_retries - 1:
                        retry = click.confirm("ClickUp validation failed. Try again?", default=True)
                        if not retry:
                            click.echo("⏭️  Skipping ClickUp setup")
                            return None
                    else:
                        click.echo("❌ Maximum retry attempts reached")
                        click.echo("⏭️  Skipping ClickUp setup")
                        return None
            else:
                # Skip validation mode
                self.env_data["CLICKUP_API_TOKEN"] = api_token
                self.env_data["CLICKUP_WORKSPACE_URL"] = workspace_url
                return {
                    "api_token": "${CLICKUP_API_TOKEN}",
                    "workspace_url": "${CLICKUP_WORKSPACE_URL}",
                }

        click.echo("⏭️  Skipping ClickUp setup")
        return None

    def _validate_clickup(self, api_token: str) -> bool:
        """Validate ClickUp API token.

        Args:
            api_token: ClickUp API token

        Returns:
            True if credentials are valid, False otherwise
        """
        # Suppress requests logging to prevent credential exposure
        urllib3_logger = logging.getLogger("urllib3")
        requests_logger = logging.getLogger("requests")
        original_urllib3 = urllib3_logger.level
        original_requests = requests_logger.level

        urllib3_logger.setLevel(logging.WARNING)
        requests_logger.setLevel(logging.WARNING)

        try:
            headers = {
                "Authorization": api_token,
                "Content-Type": "application/json",
            }

            response = requests.get(
                "https://api.clickup.com/api/v2/user",
                headers=headers,
                timeout=10,
                verify=True,
            )

            if response.status_code == 200:
                user_info = response.json()
                if "user" in user_info:
                    username = user_info["user"].get("username", "Unknown")
                    click.echo(f"   Authenticated as: {username}")
                    return True
                else:
                    click.echo("   Authentication failed: Invalid response")
                    return False
            else:
                click.echo(f"   Authentication failed (status {response.status_code})")
                return False

        except requests.exceptions.RequestException as e:
            # Never expose raw exception - could contain credentials
            error_type = type(e).__name__
            click.echo(f"   Connection error: {error_type}")
            logger.debug(f"ClickUp connection error type: {error_type}")
            return False
        except Exception as e:
            click.echo("   ClickUp validation failed")
            logger.error(f"ClickUp validation error type: {type(e).__name__}")
            return False
        finally:
            # Always restore logging levels
            urllib3_logger.setLevel(original_urllib3)
            requests_logger.setLevel(original_requests)

    def _validate_jira(self, base_url: str, username: str, api_token: str) -> bool:
        """Validate JIRA credentials.

        Args:
            base_url: JIRA instance URL
            username: JIRA username/email
            api_token: JIRA API token

        Returns:
            True if credentials are valid, False otherwise
        """
        # Suppress requests logging to prevent credential exposure
        urllib3_logger = logging.getLogger("urllib3")
        requests_logger = logging.getLogger("requests")
        original_urllib3 = urllib3_logger.level
        original_requests = requests_logger.level

        urllib3_logger.setLevel(logging.WARNING)
        requests_logger.setLevel(logging.WARNING)

        try:
            import base64

            # Create authentication header
            credentials = base64.b64encode(f"{username}:{api_token}".encode()).decode()
            headers = {
                "Authorization": f"Basic {credentials}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }

            # Test authentication by getting current user info
            response = requests.get(
                f"{base_url}/rest/api/3/myself",
                headers=headers,
                timeout=10,
                verify=True,  # Explicit SSL verification
            )

            if response.status_code == 200:
                user_info = response.json()
                click.echo(f"   Authenticated as: {user_info.get('displayName', username)}")
                return True
            else:
                click.echo(f"   Authentication failed (status {response.status_code})")
                return False

        except requests.exceptions.RequestException as e:
            # Never expose raw exception - could contain credentials
            error_type = type(e).__name__
            click.echo(f"   Connection error: {error_type}")
            logger.debug(f"JIRA connection error type: {error_type}")
            return False
        except Exception as e:
            click.echo("   JIRA validation failed")
            logger.error(f"JIRA validation error type: {type(e).__name__}")
            return False
        finally:
            # Always restore logging levels
            urllib3_logger.setLevel(original_urllib3)
            requests_logger.setLevel(original_requests)

    def _store_jira_config(self, base_url: str, username: str, api_token: str) -> None:
        """Store JIRA configuration.

        Args:
            base_url: JIRA instance URL
            username: JIRA username/email
            api_token: JIRA API token
        """
        self.env_data["JIRA_BASE_URL"] = base_url
        self.env_data["JIRA_ACCESS_USER"] = username
        self.env_data["JIRA_ACCESS_TOKEN"] = api_token

        if "pm" not in self.config_data:
            self.config_data["pm"] = {}

        self.config_data["pm"]["jira"] = {
            "base_url": "${JIRA_BASE_URL}",
            "username": "${JIRA_ACCESS_USER}",
            "api_token": "${JIRA_ACCESS_TOKEN}",
        }

    def _discover_jira_fields(self, base_url: str, username: str, api_token: str) -> None:
        """Discover story point fields in JIRA.

        Args:
            base_url: JIRA instance URL
            username: JIRA username/email
            api_token: JIRA API token
        """
        # Suppress requests logging to prevent credential exposure
        urllib3_logger = logging.getLogger("urllib3")
        requests_logger = logging.getLogger("requests")
        original_urllib3 = urllib3_logger.level
        original_requests = requests_logger.level

        urllib3_logger.setLevel(logging.WARNING)
        requests_logger.setLevel(logging.WARNING)

        try:
            import base64

            click.echo("🔍 Discovering story point fields...")

            credentials = base64.b64encode(f"{username}:{api_token}".encode()).decode()
            headers = {
                "Authorization": f"Basic {credentials}",
                "Accept": "application/json",
            }

            response = requests.get(
                f"{base_url}/rest/api/3/field",
                headers=headers,
                timeout=10,
                verify=True,  # Explicit SSL verification
            )

            if response.status_code != 200:
                return

            fields = response.json()
            story_point_fields = []

            # Look for fields with "story", "point", or "estimate" in name
            for field in fields:
                name = field.get("name", "").lower()
                if any(term in name for term in ["story", "point", "estimate"]):
                    story_point_fields.append(field["id"])

            if story_point_fields:
                click.echo(f"✅ Found {len(story_point_fields)} potential story point field(s)")
                self.config_data["pm"]["jira"]["story_point_fields"] = story_point_fields
            else:
                click.echo("⚠️  No story point fields detected")

        except Exception as e:
            logger.debug(f"JIRA field discovery error type: {type(e).__name__}")
        finally:
            # Always restore logging levels
            urllib3_logger.setLevel(original_urllib3)
            requests_logger.setLevel(original_requests)

