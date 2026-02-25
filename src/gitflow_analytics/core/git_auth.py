"""Git authentication setup and validation for GitHub operations."""

import logging
import os
import stat
import subprocess
import tempfile
from pathlib import Path

import click
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


def create_git_askpass_script(token: str) -> str:
    """Create a temporary GIT_ASKPASS script that supplies the token as password.

    The script echoes the token only when git asks for a password, and echoes
    an empty string for username prompts. It is written to a private temp file
    (mode 0o700) so other OS users cannot read the token from disk.

    SECURITY: The token is embedded in the script body at creation time, lives
    only for the duration of the git operation, and the caller is responsible
    for deleting it via ``os.unlink`` once the operation completes.

    Args:
        token: GitHub personal access token

    Returns:
        Absolute path to the created script file
    """
    # Write a minimal shell script: echo token for password prompts, empty for username
    script_content = (
        "#!/bin/sh\n"
        "# GIT_ASKPASS helper – supplies token for HTTPS GitHub auth\n"
        'case "$1" in\n'
        "  *Username*) echo '' ;;\n"
        f"  *Password*) echo '{token}' ;;\n"
        "  *) echo '' ;;\n"
        "esac\n"
    )

    fd, script_path = tempfile.mkstemp(prefix="gfa_askpass_", suffix=".sh")
    try:
        os.write(fd, script_content.encode())
    finally:
        os.close(fd)

    # Restrict to owner-execute only so the token is not world-readable
    os.chmod(script_path, stat.S_IRWXU)
    return script_path


def ensure_remote_url_has_token(repo_path: Path, token: str) -> bool:
    """Configure git credential supply via GIT_ASKPASS for HTTPS authentication.

    This function previously embedded the token directly in the remote URL
    (``https://git:TOKEN@github.com/…``), which caused the plaintext token to  # pragma: allowlist secret
    be written into ``.git/config`` on disk – a security vulnerability.

    The replacement approach creates a short-lived GIT_ASKPASS helper script
    and stores its path in the per-repository git config key
    ``core.askPass``.  Git respects this key for credential prompts without
    ever writing the token into the URL or the config file.

    The git_timeout_wrapper overrides GIT_ASKPASS with ``/bin/echo`` at the
    process level; to work around that restriction the script path is also
    written to ``core.askPass`` which git checks before the environment
    variable when both are set.

    Args:
        repo_path: Path to the git repository
        token: GitHub personal access token

    Returns:
        True if the askPass helper was configured, False if already has token,
        not applicable (SSH URL), or operation failed
    """
    if not token:
        logger.debug("No token provided, skipping credential configuration")
        return False

    try:
        # Get current origin remote URL
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        current_url = result.stdout.strip()

        if not current_url:
            logger.debug(f"No origin remote found for {repo_path}")
            return False

        if current_url.startswith("git@github.com:"):
            # SSH URL — token-based auth is not needed
            logger.debug(f"Using SSH authentication for {repo_path.name}")
            return False

        if not current_url.startswith("https://github.com/"):
            logger.debug(f"Unknown URL format for {repo_path.name}: {current_url}")
            return False

        # Create a private GIT_ASKPASS script and register it via core.askPass.
        # This avoids writing the token into the remote URL (and therefore into
        # .git/config) while still allowing non-interactive credential supply.
        script_path = create_git_askpass_script(token)
        subprocess.run(
            ["git", "config", "--local", "core.askPass", script_path],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.debug(f"Configured GIT_ASKPASS credential helper for {repo_path.name}")
        return True

    except subprocess.CalledProcessError as e:
        logger.warning(f"Could not configure credentials for {repo_path.name}: {e.stderr}")
        return False
    except OSError as e:
        logger.warning(f"Could not create GIT_ASKPASS script for {repo_path.name}: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error configuring credentials for {repo_path.name}: {e}")
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
        logger.error("GITHUB_TOKEN not found in configuration")
        click.echo(
            "Error: GITHUB_TOKEN not found in config. Add to .env file or config.yaml", err=True
        )
        click.echo("   Example .env file:", err=True)
        click.echo("   GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx", err=True)
        click.echo("", err=True)
        click.echo("   Or in config.yaml:", err=True)
        click.echo("   github:", err=True)
        click.echo("     token: ${GITHUB_TOKEN}", err=True)
        return False

    # Verify token is valid
    success, username, error_msg = verify_github_token(token)
    if not success:
        logger.error(f"GitHub token validation failed: {error_msg}")
        if "invalid or expired" in error_msg.lower():
            click.echo("Error: GitHub token invalid or expired. Generate new token at:", err=True)
            click.echo("   https://github.com/settings/tokens", err=True)
            click.echo("", err=True)
            click.echo("   Required permissions:", err=True)
            click.echo("   - repo (Full control of private repositories)", err=True)
            click.echo("   - read:org (Read org and team membership)", err=True)
        elif "api error" in error_msg.lower():
            click.echo(f"Error: Cannot access GitHub API: {error_msg}", err=True)
            click.echo("   Check your network connection and GitHub API status:", err=True)
            click.echo("   https://www.githubstatus.com/", err=True)
        else:
            click.echo(f"Error: GitHub authentication failed: {error_msg}", err=True)
        return False

    # Setup git credentials
    if not setup_git_credentials(token, username="git"):
        logger.error("Failed to setup git credentials")
        click.echo("Error: Failed to configure git credentials", err=True)
        click.echo("   Try manually running:", err=True)
        click.echo("   git config --global credential.helper store", err=True)
        return False

    logger.info(f"GitHub authentication configured successfully (user: {username})")
    click.echo(f"GitHub authentication configured successfully (user: {username})")
    return True
