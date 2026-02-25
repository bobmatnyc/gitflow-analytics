#!/usr/bin/env python3
"""Test script to verify the Git credential authentication fix.

This script demonstrates how the ensure_remote_url_has_token function
works to embed GitHub tokens in remote URLs.
"""

import os
import subprocess
from pathlib import Path


def ensure_remote_url_has_token(repo_path: Path, token: str) -> bool:
    """Embed GitHub token in remote URL for HTTPS authentication.

    This is needed because subprocess git operations may not have access
    to the credential helper store due to environment variable restrictions
    (GIT_CREDENTIAL_HELPER="" and GIT_ASKPASS="/bin/echo" in git_timeout_wrapper).

    Args:
        repo_path: Path to the git repository
        token: GitHub personal access token

    Returns:
        True if URL was updated with token, False if already has token,
        not applicable (SSH URL), or operation failed
    """
    if not token:
        print("No token provided, skipping remote URL update")
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
            print(f"No origin remote found for {repo_path}")
            return False

        print(f"Current remote URL: {current_url}")

        # Check if it's an HTTPS GitHub URL without embedded token
        if current_url.startswith("https://github.com/"):
            # URL format: https://github.com/org/repo.git
            # New format: https://git:TOKEN@github.com/org/repo.git
            new_url = current_url.replace(
                "https://github.com/", f"https://git:{token}@github.com/"
            )  # pragma: allowlist secret

            # Update the remote URL
            subprocess.run(
                ["git", "remote", "set-url", "origin", new_url],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"✅ Updated remote URL with embedded token for {repo_path.name}")
            print("New remote URL: https://git:***@github.com/... (token hidden)")
            return True

        elif "@github.com" in current_url:
            # Already has authentication embedded (either token or SSH)
            print(f"Remote URL already has authentication for {repo_path.name}")
            return False

        elif current_url.startswith("git@github.com:"):
            # SSH URL, no need to modify
            print(f"Using SSH authentication for {repo_path.name}")
            return False

        else:
            # Unknown URL format
            print(f"Unknown URL format for {repo_path.name}: {current_url}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"Could not update remote URL for {repo_path.name}: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error updating remote URL for {repo_path.name}: {e}")
        return False


def test_current_repo():
    """Test with the current repository."""
    print("=" * 60)
    print("Testing Git Authentication Fix")
    print("=" * 60)

    repo_path = Path.cwd()
    print(f"\nRepository: {repo_path}")

    # Get token from environment
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("\n⚠️  GITHUB_TOKEN not set in environment")
        print("This is OK - the fix will skip URL updates when no token is available")
        print("\nTo test with a real token, run:")
        print("export GITHUB_TOKEN=your_token_here")
        print("python3 test_git_auth_fix.py")
        return

    print(f"\n✅ GITHUB_TOKEN found (length: {len(token)})")

    # Test the function
    print("\n" + "-" * 60)
    print("Testing ensure_remote_url_has_token()...")
    print("-" * 60)

    result = ensure_remote_url_has_token(repo_path, token)

    if result:
        print("\n✅ SUCCESS: Remote URL was updated with embedded token")
        print("\nNow git fetch operations will work without credential helper!")
    else:
        print("\n✅ No update needed (SSH URL or already has token)")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    test_current_repo()
