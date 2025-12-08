#!/usr/bin/env python3
"""Version management for gitflow-analytics.

This script manages version bumping and validation for the project.
It reads and updates the version in src/gitflow_analytics/_version.py.

Usage:
    python scripts/manage_version.py get
    python scripts/manage_version.py set --version 3.14.0
    python scripts/manage_version.py bump --type patch
    python scripts/manage_version.py bump --type minor
    python scripts/manage_version.py bump --type major
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    # Script is in scripts/, so go up one level
    return Path(__file__).parent.parent


VERSION_FILE = get_project_root() / "src" / "gitflow_analytics" / "_version.py"


def get_current_version() -> str:
    """Read current version from _version.py.

    Returns:
        Current version string (e.g., "3.13.1")

    Raises:
        ValueError: If version cannot be found in file
        FileNotFoundError: If version file doesn't exist
    """
    if not VERSION_FILE.exists():
        raise FileNotFoundError(f"Version file not found: {VERSION_FILE}")

    content = VERSION_FILE.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError(f"Version string not found in {VERSION_FILE}")
    return match.group(1)


def set_version(new_version: str) -> None:
    """Update version in _version.py.

    Args:
        new_version: New version string (e.g., "3.14.0")

    Raises:
        ValueError: If version format is invalid
        FileNotFoundError: If version file doesn't exist
    """
    # Validate version format
    if not re.match(r'^\d+\.\d+\.\d+$', new_version):
        raise ValueError(f"Invalid version format: {new_version} (expected: X.Y.Z)")

    if not VERSION_FILE.exists():
        raise FileNotFoundError(f"Version file not found: {VERSION_FILE}")

    content = VERSION_FILE.read_text()

    # Update __version__
    new_content = re.sub(
        r'(__version__\s*=\s*["\'])[^"\']+(["\'])',
        f'\\g<1>{new_version}\\g<2>',
        content
    )

    VERSION_FILE.write_text(new_content)
    print(f"✅ Updated version to {new_version} in {VERSION_FILE.relative_to(get_project_root())}")


def bump_version(bump_type: str) -> str:
    """Bump version based on type (patch/minor/major).

    Args:
        bump_type: One of 'patch', 'minor', or 'major'

    Returns:
        New version string

    Raises:
        ValueError: If bump_type is invalid or current version is malformed
    """
    current = get_current_version()

    try:
        major, minor, patch = map(int, current.split('.'))
    except ValueError:
        raise ValueError(f"Current version has invalid format: {current}")

    if bump_type == 'patch':
        patch += 1
    elif bump_type == 'minor':
        minor += 1
        patch = 0
    elif bump_type == 'major':
        major += 1
        minor = 0
        patch = 0
    else:
        raise ValueError(f"Invalid bump type: {bump_type} (expected: patch, minor, or major)")

    return f"{major}.{minor}.{patch}"


def validate_version_format(version: str) -> bool:
    """Validate semantic version format.

    Args:
        version: Version string to validate

    Returns:
        True if valid, False otherwise
    """
    return bool(re.match(r'^\d+\.\d+\.\d+$', version))


def get_git_tag_version() -> str | None:
    """Get the latest git tag version.

    Returns:
        Latest tag version string (without 'v' prefix) or None if no tags exist
    """
    try:
        result = subprocess.run(
            ['git', 'describe', '--tags', '--abbrev=0'],
            capture_output=True,
            text=True,
            check=True,
            cwd=get_project_root()
        )
        tag = result.stdout.strip()
        # Remove 'v' prefix if present
        return tag[1:] if tag.startswith('v') else tag
    except subprocess.CalledProcessError:
        return None


def check_git_status() -> bool:
    """Check if git working directory is clean.

    Returns:
        True if working directory is clean, False otherwise
    """
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True,
            cwd=get_project_root()
        )
        return len(result.stdout.strip()) == 0
    except subprocess.CalledProcessError:
        return False


def main():
    """Main entry point for version management script."""
    parser = argparse.ArgumentParser(
        description="Manage package version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Get current version:
    python scripts/manage_version.py get

  Set specific version:
    python scripts/manage_version.py set --version 3.14.0

  Bump version:
    python scripts/manage_version.py bump --type patch  # 3.13.1 → 3.13.2
    python scripts/manage_version.py bump --type minor  # 3.13.1 → 3.14.0
    python scripts/manage_version.py bump --type major  # 3.13.1 → 4.0.0
        """
    )

    parser.add_argument(
        'action',
        choices=['get', 'set', 'bump', 'validate'],
        help='Action to perform'
    )
    parser.add_argument(
        '--type',
        choices=['patch', 'minor', 'major'],
        help='Bump type (required for bump action)'
    )
    parser.add_argument(
        '--version',
        help='New version (required for set action)'
    )
    parser.add_argument(
        '--check-git',
        action='store_true',
        help='Check git status before version operations'
    )

    args = parser.parse_args()

    try:
        if args.action == 'get':
            version = get_current_version()
            print(version)

        elif args.action == 'set':
            if not args.version:
                parser.error("--version required for set action")
            set_version(args.version)

        elif args.action == 'bump':
            if not args.type:
                parser.error("--type required for bump action")

            current = get_current_version()
            new_version = bump_version(args.type)
            print(f"Current version: {current}")
            print(f"New version: {new_version} (bumped {args.type})")
            set_version(new_version)

        elif args.action == 'validate':
            current = get_current_version()
            git_tag = get_git_tag_version()

            print(f"Current version: {current}")
            if git_tag:
                print(f"Latest git tag: v{git_tag}")
                if current == git_tag:
                    print("✅ Version matches latest git tag")
                else:
                    print("⚠️  Version does not match latest git tag")
                    sys.exit(1)
            else:
                print("ℹ️  No git tags found")

            if args.check_git and not check_git_status():
                print("❌ Git working directory is not clean")
                sys.exit(1)
            else:
                print("✅ Git working directory is clean")

    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
