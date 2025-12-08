"""Shared fixtures for integration tests.

This module provides fixtures for creating realistic test repositories,
configurations, and helper functions for verifying end-to-end workflows.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
from git import Actor, Repo

from gitflow_analytics.core.cache import GitAnalysisCache


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for test files and repositories.

    Yields:
        Path: Temporary workspace directory path

    Cleanup:
        Removes the entire workspace after test completes
    """
    workspace = Path(tempfile.mkdtemp(prefix="gitflow_test_"))
    yield workspace
    shutil.rmtree(workspace, ignore_errors=True)


@pytest.fixture
def test_author():
    """Create a consistent git author for test commits.

    Returns:
        Actor: GitPython Actor object with test author details
    """
    return Actor("Test Developer", "test@example.com")


@pytest.fixture
def test_repo_with_merges(temp_workspace, test_author):
    """Create a test Git repository with regular commits and merge commits.

    This fixture creates a realistic repository structure:
    - Main branch with 5 regular commits (10 lines each)
    - Feature branch with 5 commits (10 lines each)
    - Merge commit from feature → main (creates large diff)
    - Another feature branch with 3 commits
    - Second merge commit

    Args:
        temp_workspace: Temporary directory for repository
        test_author: Git author for commits

    Returns:
        dict: Repository information containing:
            - repo: GitPython Repo object
            - repo_path: Path to repository
            - regular_commits: List of regular commit hashes
            - merge_commits: List of merge commit hashes
            - expected_lines_regular: Expected line count from regular commits
            - expected_lines_with_merges: Expected line count including merges
    """
    repo_path = temp_workspace / "test_repo"
    repo_path.mkdir()

    # Initialize repository
    repo = Repo.init(repo_path)

    # Configure git user
    with repo.config_writer() as config:
        config.set_value("user", "name", test_author.name)
        config.set_value("user", "email", test_author.email)

    file_path = repo_path / "test.txt"
    regular_commits = []
    merge_commits = []

    # Track expected line counts
    expected_lines_regular = 0
    expected_lines_with_merges = 0

    # Create 5 regular commits on main
    for i in range(5):
        content = f"Main content {i}\n" * 10  # 10 lines per commit
        file_path.write_text(content)
        # Use relative path to avoid macOS /var vs /private/var symlink issues
        repo.index.add(["test.txt"])
        commit = repo.index.commit(f"Main commit {i}", author=test_author, committer=test_author)
        regular_commits.append(commit.hexsha)
        expected_lines_regular += 10
        expected_lines_with_merges += 10

    # Capture default branch name before creating feature branch
    default_branch = repo.active_branch

    # Create feature branch
    feature_branch = repo.create_head("feature")
    feature_branch.checkout()

    # Create 5 commits on feature branch
    for i in range(5):
        content = f"Feature content {i}\n" * 10  # 10 lines per commit
        file_path.write_text(content)
        repo.index.add(["test.txt"])
        commit = repo.index.commit(f"Feature commit {i}", author=test_author, committer=test_author)
        regular_commits.append(commit.hexsha)
        expected_lines_regular += 10
        expected_lines_with_merges += 10

    # Switch back to default branch (main or master) and create merge commit
    default_branch.checkout()

    # Merge feature branch (no-ff to ensure merge commit is created)
    repo.git.merge("feature", no_ff=True, m="Merge feature branch")
    merge_commit = repo.head.commit
    merge_commits.append(merge_commit.hexsha)

    # Merge commits typically have large diffs (all feature changes)
    # In this case: 5 commits * 10 lines = 50 lines from merge
    expected_lines_with_merges += 50  # Approximate merge diff

    # Create second feature branch with more commits
    feature2_branch = repo.create_head("feature2")
    feature2_branch.checkout()

    for i in range(3):
        content = f"Feature2 content {i}\n" * 10  # 10 lines per commit
        file_path.write_text(content)
        repo.index.add(["test.txt"])
        commit = repo.index.commit(
            f"Feature2 commit {i}", author=test_author, committer=test_author
        )
        regular_commits.append(commit.hexsha)
        expected_lines_regular += 10
        expected_lines_with_merges += 10

    # Merge second feature branch
    default_branch.checkout()
    repo.git.merge("feature2", no_ff=True, m="Merge feature2 branch")
    merge_commit2 = repo.head.commit
    merge_commits.append(merge_commit2.hexsha)

    # Second merge adds ~30 lines from feature2
    expected_lines_with_merges += 30

    return {
        "repo": repo,
        "repo_path": repo_path,
        "regular_commits": regular_commits,
        "merge_commits": merge_commits,
        "expected_lines_regular": expected_lines_regular,
        "expected_lines_with_merges": expected_lines_with_merges,
        "total_commits": len(regular_commits) + len(merge_commits),
        "regular_commit_count": len(regular_commits),
        "merge_commit_count": len(merge_commits),
    }


@pytest.fixture
def test_config_dict(temp_workspace, test_repo_with_merges):
    """Create a test configuration dictionary for GitFlow Analytics.

    This fixture provides a configuration dictionary that can be used
    to configure the system for testing without requiring a full Config object.

    Args:
        temp_workspace: Temporary workspace directory
        test_repo_with_merges: Test repository with merge commits

    Returns:
        dict: Configuration dictionary
    """
    cache_dir = temp_workspace / ".gitflow-cache"
    cache_dir.mkdir(exist_ok=True)

    reports_dir = temp_workspace / "reports"
    reports_dir.mkdir(exist_ok=True)

    config_data = {
        "repositories": [
            {
                "name": "test-repo",
                "path": str(test_repo_with_merges["repo_path"]),
                "project_key": "TEST",
            }
        ],
        "cache_dir": str(cache_dir),
        "reports_dir": str(reports_dir),
        "analysis": {
            "exclude_merge_commits": True,  # Enable merge exclusion by default
        },
        "developers": {
            "identity_threshold": 0.85,
            "manual_mappings": [],
        },
        "tickets": {
            "default_platform": "github",
        },
    }

    return config_data


@pytest.fixture
def test_cache(temp_workspace):
    """Create a test cache instance.

    Args:
        temp_workspace: Temporary workspace directory

    Returns:
        GitAnalysisCache: Cache instance for testing
    """
    cache_dir = temp_workspace / ".gitflow-cache"
    cache_dir.mkdir(exist_ok=True)

    cache = GitAnalysisCache(
        cache_dir=cache_dir,
        ttl_hours=168,  # 1 week default
    )

    return cache


@pytest.fixture
def integration_db_session(test_cache):
    """Create a database session for integration tests.

    Args:
        test_cache: Test cache instance

    Returns:
        Session: SQLAlchemy database session

    Cleanup:
        Closes the session after test completes
    """
    session = test_cache.db.get_session()
    yield session
    session.close()


def verify_commit_in_database(session, commit_hash: str, repo_path: Path) -> dict[str, Any]:
    """Verify that a commit exists in the database and return its data.

    Args:
        session: Database session
        commit_hash: Commit hash to verify
        repo_path: Repository path for filtering

    Returns:
        dict: Commit data including filtered stats

    Raises:
        AssertionError: If commit is not found
    """
    from gitflow_analytics.models.database import CachedCommit

    commit = (
        session.query(CachedCommit)
        .filter(
            CachedCommit.commit_hash == commit_hash,
            CachedCommit.repo_path == str(repo_path),
        )
        .first()
    )

    assert commit is not None, f"Commit {commit_hash[:8]} not found in database"

    return {
        "hash": commit.commit_hash,
        "is_merge": commit.is_merge,
        "insertions": commit.insertions,
        "deletions": commit.deletions,
        "filtered_insertions": commit.filtered_insertions,
        "filtered_deletions": commit.filtered_deletions,
    }


def calculate_total_lines_from_commits(
    session, repo_path: Path, use_filtered: bool = True
) -> dict[str, int]:
    """Calculate total line counts from commits in the database.

    Args:
        session: Database session
        repo_path: Repository path for filtering
        use_filtered: If True, use filtered stats; otherwise use raw stats

    Returns:
        dict: Total line counts with keys:
            - total_insertions: Total lines added
            - total_deletions: Total lines deleted
            - total_lines: Total lines changed
            - commit_count: Number of commits
    """
    from sqlalchemy import func

    from gitflow_analytics.models.database import CachedCommit

    if use_filtered:
        result = (
            session.query(
                func.sum(CachedCommit.filtered_insertions).label("insertions"),
                func.sum(CachedCommit.filtered_deletions).label("deletions"),
                func.count(CachedCommit.id).label("count"),
            )
            .filter(CachedCommit.repo_path == str(repo_path))
            .first()
        )
    else:
        result = (
            session.query(
                func.sum(CachedCommit.insertions).label("insertions"),
                func.sum(CachedCommit.deletions).label("deletions"),
                func.count(CachedCommit.id).label("count"),
            )
            .filter(CachedCommit.repo_path == str(repo_path))
            .first()
        )

    insertions = result.insertions or 0
    deletions = result.deletions or 0

    return {
        "total_insertions": insertions,
        "total_deletions": deletions,
        "total_lines": insertions + deletions,
        "commit_count": result.count or 0,
    }
