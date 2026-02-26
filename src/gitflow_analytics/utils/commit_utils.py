"""Utilities for working with Git commit objects."""

import re

import git

# Compiled pattern for Co-authored-by trailers (case-insensitive per Git convention).
# Format: "Co-authored-by: Name <email@example.com>"
_CO_AUTHOR_RE = re.compile(
    r"^Co-authored-by:\s*(.+?)\s*<([^>]+)>",
    re.MULTILINE | re.IGNORECASE,
)


def extract_co_authors(message: str) -> list[dict[str, str]]:
    """Extract Co-authored-by trailer entries from a commit message.

    WHY (Gap 4): GitHub, VS Code, and collaborative tools add
    ``Co-authored-by: Name <email>`` trailers when multiple developers
    work together on a single commit.  Parsing these trailers lets the
    identity system attribute the commit to every contributor, not only
    the primary git author.

    Args:
        message: Raw commit message text (may be multi-line).

    Returns:
        List of dicts with ``name`` and ``email`` keys for each
        co-author found.  Empty list when no trailers are present.

    Example::

        >>> extract_co_authors(
        ...     "Fix bug\\n\\nCo-authored-by: Alice <alice@example.com>"
        ... )
        [{'name': 'Alice', 'email': 'alice@example.com'}]
    """
    co_authors: list[dict[str, str]] = []
    for match in _CO_AUTHOR_RE.finditer(message):
        name = match.group(1).strip()
        email = match.group(2).strip().lower()
        if name and email:
            co_authors.append({"name": name, "email": email})
    return co_authors


def is_merge_commit(commit: git.Commit) -> bool:
    """Determine if a commit is a merge commit.

    A merge commit is one with 2 or more parent commits. This includes:
    - Standard merges (2 parents)
    - Octopus merges (3+ parents)

    Args:
        commit: GitPython Commit object to check

    Returns:
        True if commit has 2 or more parents, False otherwise

    Examples:
        >>> is_merge_commit(regular_commit)  # 1 parent
        False
        >>> is_merge_commit(merge_commit)    # 2 parents
        True
        >>> is_merge_commit(octopus_merge)   # 3+ parents
        True
        >>> is_merge_commit(initial_commit)  # 0 parents
        False
    """
    return len(commit.parents) > 1


def get_parent_count(commit: git.Commit) -> int:
    """Get the number of parent commits.

    Args:
        commit: GitPython Commit object

    Returns:
        Number of parent commits (0 for initial commit, 1 for regular, 2+ for merge)
    """
    return len(commit.parents)


def is_initial_commit(commit: git.Commit) -> bool:
    """Determine if a commit is an initial commit (has no parents).

    Args:
        commit: GitPython Commit object to check

    Returns:
        True if commit has no parents, False otherwise
    """
    return len(commit.parents) == 0
