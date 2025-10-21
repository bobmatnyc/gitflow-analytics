"""Utilities for working with Git commit objects."""

import git


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
