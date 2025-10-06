"""Git authentication setup and validation for GitHub operations."""

import logging
import subprocess
from pathlib import Path

from github import Github
from github.GithubException import BadCredentialsException, GithubException

logger = logging.getLogger(__name__)


def verify_github_token(token: str, timeout: int = 10) -> tuple[bool, str, str]:
    """Verify GitHub token is valid and return authenticated username.

    Args:
        token: GitHub personal access token
        timeout: API request timeout in seconds (default: 10)

    Returns:
        Tuple of (success, username, error_message)
        - success: True if token is valid
        - username: GitHub username if successful, empty string otherwise
        - error_message: Error description if failed, empty string otherwise
    """
    if not token:
        return False, "", "GitHub token is empty"

    try:
        github = Github(token, timeout=timeout)
        user = github.get_user()
        username = user.login
        logger.info(f"GitHub token verified successfully for user: {username}")
        return True, username, ""
    except BadCredentialsException:
        error_msg = "GitHub token is invalid or expired"
        logger.error(error_msg)
        return False, "", error_msg
    except GithubException as e:
        error_msg = (
            f"GitHub API error: {e.data.get('message', str(e)) if hasattr(e, 'data') else str(e)}"
        )
        logger.error(error_msg)
        return False, "", error_msg
    except Exception as e:
        error_msg = f"Unexpected error verifying GitHub token: {str(e)}"
        logger.error(error_msg)
        return False, "", error_msg


def setup_git_credentials(token: str, username: str = "git") -> bool:
    """Configure git to use GitHub token for HTTPS authentication.

    This function sets up the git credential helper to store credentials
    and adds the GitHub token to ~/.git-credentials.

    Args:
        token: GitHub personal access token
        username: Username for git authentication (default: "git")

    Returns:
        True if setup successful, False otherwise
    """
    try:
        # Configure git to use credential helper store
        subprocess.run(
            ["git", "config", "--global", "credential.helper", "store"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.debug("Configured git credential helper to 'store'")

        # Add credentials to ~/.git-credentials
        credentials_file = Path.home() / ".git-credentials"
        credential_line = f"https://{username}:{token}@github.com\n"

        # Read existing credentials
        existing_credentials = []
        if credentials_file.exists():
            with open(credentials_file) as f:
                existing_credentials = f.readlines()

        # Check if GitHub credential already exists
        github_creds = [line for line in existing_credentials if "github.com" in line]
        if github_creds:
            # Remove old GitHub credentials
            existing_credentials = [
                line for line in existing_credentials if "github.com" not in line
            ]
            logger.debug("Replaced existing GitHub credentials")

        # Add new credential
        existing_credentials.append(credential_line)

        # Write back to file with proper permissions
        credentials_file.touch(mode=0o600, exist_ok=True)
        with open(credentials_file, "w") as f:
            f.writelines(existing_credentials)

        logger.info("Git credentials configured successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to configure git credential helper: {e.stderr}")
        return False
    except OSError as e:
        logger.error(f"Failed to write git credentials file: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error setting up git credentials: {e}")
        return False


def preflight_git_authentication(config: dict) -> bool:
    """Run pre-flight checks for git authentication and setup credentials.

    This function verifies the GitHub token and configures git credentials
    before any git operations are performed.

    Args:
        config: Configuration dictionary containing github.token

    Returns:
        True if authentication is ready, False if setup failed
    """
    # Extract GitHub token from config
    github_config = config.get("github", {})
    token = github_config.get("token")

    if not token:
        logger.error("❌ GITHUB_TOKEN not found in configuration")
        print("❌ GITHUB_TOKEN not found in config. Add to .env file or config.yaml")
        print("   Example .env file:")
        print("   GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx")
        print("\n   Or in config.yaml:")
        print("   github:")
        print("     token: ${GITHUB_TOKEN}")
        return False

    # Verify token is valid
    success, username, error_msg = verify_github_token(token)
    if not success:
        logger.error(f"❌ GitHub token validation failed: {error_msg}")
        if "invalid or expired" in error_msg.lower():
            print("❌ GitHub token invalid or expired. Generate new token at:")
            print("   https://github.com/settings/tokens")
            print("\n   Required permissions:")
            print("   - repo (Full control of private repositories)")
            print("   - read:org (Read org and team membership)")
        elif "api error" in error_msg.lower():
            print(f"❌ Cannot access GitHub API: {error_msg}")
            print("   Check your network connection and GitHub API status:")
            print("   https://www.githubstatus.com/")
        else:
            print(f"❌ GitHub authentication failed: {error_msg}")
        return False

    # Setup git credentials
    if not setup_git_credentials(token, username="git"):
        logger.error("❌ Failed to setup git credentials")
        print("❌ Failed to configure git credentials")
        print("   Try manually running:")
        print("   git config --global credential.helper store")
        return False

    logger.info(f"✅ GitHub authentication configured successfully (user: {username})")
    print(f"✅ GitHub authentication configured successfully (user: {username})")
    return True
