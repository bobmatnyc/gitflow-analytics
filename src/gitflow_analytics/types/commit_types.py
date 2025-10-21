"""Type definitions for commit-related data structures."""

from typing import TypedDict


class CommitStats(TypedDict):
    """Statistics for a single commit.

    This type is used by DataFetcher._calculate_commit_stats() which provides
    both filtered and raw (unfiltered) statistics. The filtered statistics
    exclude files matching exclude_paths patterns, while raw statistics include
    all changed files.

    When exclude_merge_commits is enabled, merge commits will have filtered
    counts set to 0 while raw counts reflect actual changes.

    Attributes:
        files: Number of files changed (filtered by exclude_paths)
        insertions: Lines added (filtered by exclude_paths)
        deletions: Lines removed (filtered by exclude_paths)
        raw_insertions: Lines added (unfiltered, all files)
        raw_deletions: Lines removed (unfiltered, all files)
    """

    files: int
    insertions: int
    deletions: int
    raw_insertions: int
    raw_deletions: int


class FilteredCommitStats(TypedDict):
    """Filtered statistics for a single commit.

    This type is used by GitAnalyzer._calculate_filtered_stats() which only
    provides filtered statistics (no raw counts). The filtered statistics
    exclude files matching exclude_paths patterns.

    When exclude_merge_commits is enabled, merge commits will have all
    counts set to 0 to exclude them from productivity metrics.

    Attributes:
        files: Number of files changed (filtered by exclude_paths)
        insertions: Lines added (filtered by exclude_paths)
        deletions: Lines removed (filtered by exclude_paths)
    """

    files: int
    insertions: int
    deletions: int
