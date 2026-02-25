"""Glob pattern matching utilities for file path filtering.

Provides consistent glob matching logic used across the codebase for
excluding files from analysis based on configurable patterns.
"""

import fnmatch
import re
from pathlib import PurePath


def matches_glob_pattern(filepath: str, pattern: str) -> bool:
    """Check if a file path matches a glob pattern, handling ** recursion correctly.

    This function properly handles different glob pattern types:
    - **/vendor/** : matches files inside vendor directories at any level
    - **/*.min.js : matches files with specific suffix anywhere in directory tree
    - vendor/** : matches files inside vendor directory at root level only
    - **pattern** : handles other complex patterns with pathlib.match()
    - simple patterns : uses fnmatch for basic wildcards

    Args:
        filepath: The file path to check
        pattern: The glob pattern to match against

    Returns:
        True if the file path matches the pattern, False otherwise
    """
    # Handle empty or invalid inputs
    if not filepath or not pattern:
        return False

    path = PurePath(filepath)

    # Check for multiple ** patterns first (most complex)
    if "**" in pattern and pattern.count("**") > 1:
        # Multiple ** patterns - use custom recursive matching for complex patterns
        return match_recursive_pattern(filepath, pattern)

    # Then handle simple ** patterns
    elif pattern.startswith("**/") and pattern.endswith("/**"):
        # Pattern like **/vendor/** - matches files inside vendor directories at any level
        dir_name = pattern[3:-3]  # Extract 'vendor' from '**/vendor/**'
        if not dir_name:  # Handle edge case of '**/**'
            return True
        return dir_name in path.parts

    elif pattern.startswith("**/"):
        # Pattern like **/*.min.js - matches files with specific suffix anywhere
        suffix_pattern = pattern[3:]
        if not suffix_pattern:  # Handle edge case of '**/'
            return True
        # Check against filename for file patterns, or any path part for directory patterns
        if suffix_pattern.endswith("/"):
            # Directory pattern like **/build/
            dir_name = suffix_pattern[:-1]
            return dir_name in path.parts
        else:
            # File pattern like *.min.js
            # Check both filename AND full path to handle patterns like **/pnpm-lock.yaml
            # matching root-level files (e.g., pnpm-lock.yaml)
            return fnmatch.fnmatch(path.name, suffix_pattern) or fnmatch.fnmatch(
                filepath, suffix_pattern
            )

    elif pattern.endswith("/**"):
        # Pattern like vendor/** or docs/build/** - matches files inside directory at root level
        dir_name = pattern[:-3]
        if not dir_name:  # Handle edge case of '/**'
            return True

        # Handle both single directory names and nested paths
        expected_parts = PurePath(dir_name).parts
        return (
            len(path.parts) >= len(expected_parts)
            and path.parts[: len(expected_parts)] == expected_parts
        )

    elif "**" in pattern:
        # Single ** pattern - use pathlib matching with fallback
        try:
            return path.match(pattern)
        except (ValueError, TypeError):
            # Fall back to fnmatch if pathlib fails (e.g., invalid pattern)
            try:
                return fnmatch.fnmatch(filepath, pattern)
            except re.error:
                # Invalid regex pattern - return False to be safe
                return False
    else:
        # Simple pattern - use fnmatch for basic wildcards
        try:
            # Try matching the full path first
            if fnmatch.fnmatch(filepath, pattern):
                return True
            # Also try matching just the filename for simple patterns
            # This allows "package-lock.json" to match "src/package-lock.json"
            return fnmatch.fnmatch(path.name, pattern)
        except re.error:
            # Invalid regex pattern - return False to be safe
            return False


def match_recursive_pattern(filepath: str, pattern: str) -> bool:
    """Handle complex patterns with multiple ** wildcards.

    Uses a position-tracking algorithm to match each non-wildcard segment
    of the pattern against path components, allowing ** to match any number
    of intermediate components.

    Args:
        filepath: The file path to check
        pattern: The pattern with multiple ** wildcards

    Returns:
        True if the path matches the pattern, False otherwise
    """
    # Split pattern by ** to handle each segment
    parts = pattern.split("**")

    # Validate that we have actual segments
    if not parts:
        return False

    # Convert filepath to parts for easier matching
    path_parts = list(PurePath(filepath).parts)

    # Start matching from the beginning
    path_index = 0

    for i, part in enumerate(parts):
        if not part:
            # Empty part (e.g., from leading or trailing **)
            if i == 0 or i == len(parts) - 1:
                # Leading or trailing ** - continue
                continue
            # Middle empty part (consecutive **) - match any number of path components
            continue

        # Clean the part (remove leading/trailing slashes)
        part = part.strip("/")

        if not part:
            continue

        # Find where this part matches in the remaining path
        found = False
        for j in range(path_index, len(path_parts)):
            # Check if the current path part matches the pattern part
            if "/" in part:
                # Part contains multiple path components
                sub_parts = part.split("/")
                if j + len(sub_parts) <= len(path_parts) and all(
                    fnmatch.fnmatch(path_parts[j + k], sub_parts[k]) for k in range(len(sub_parts))
                ):
                    path_index = j + len(sub_parts)
                    found = True
                    break
            else:
                # Single component part
                if fnmatch.fnmatch(path_parts[j], part):
                    path_index = j + 1
                    found = True
                    break

        if not found and part:
            # Required part not found in path
            return False

    return True


def should_exclude_file(filepath: str, exclude_patterns: list[str]) -> bool:
    """Check if a file should be excluded based on a list of glob patterns.

    Normalizes path separators before matching to ensure consistent behaviour
    across operating systems.

    Args:
        filepath: The file path to evaluate
        exclude_patterns: List of glob patterns; any match causes exclusion

    Returns:
        True if the file matches any exclusion pattern, False otherwise
    """
    if not filepath:
        return False

    # Normalize path separators for consistent matching
    filepath = filepath.replace("\\", "/")

    return any(matches_glob_pattern(filepath, pattern) for pattern in exclude_patterns)
