#!/usr/bin/env python3

import sys

sys.path.insert(0, "src")

from gitflow_analytics.core.cache import GitAnalysisCache


def test_bulk_exists():
    """Test the bulk_exists method to see why it's returning incorrect results."""

    # Initialize cache
    cache_dir = "configs/.gitflow-cache"
    cache = GitAnalysisCache(cache_dir)

    repo_path = "/Users/masa/Projects/managed/gitflow-analytics"

    # Get some commit hashes from git
    import subprocess

    result = subprocess.run(
        ["git", "log", "--since=2025-07-21", "--until=2025-08-18", "--pretty=format:%H"],
        capture_output=True,
        text=True,
        cwd=repo_path,
    )

    all_commit_hashes = result.stdout.strip().split("\n")[:10]  # First 10 commits
    print(f"Testing with {len(all_commit_hashes)} commit hashes:")
    for i, hash in enumerate(all_commit_hashes):
        print(f"  {i + 1}. {hash}")

    # Test bulk_exists
    print(f"\nTesting bulk_exists for repo_path: {repo_path}")
    exists_map = cache.bulk_exists(repo_path, all_commit_hashes)

    print("\nResults from bulk_exists:")
    for hash, exists in exists_map.items():
        print(f"  {hash}: {exists}")

    # Check what's actually in the database
    print("\nChecking database directly:")
    with cache.get_session() as session:
        from gitflow_analytics.models.database import CachedCommit

        # Get all commits in database for this repo
        all_cached = session.query(CachedCommit).filter(CachedCommit.repo_path == repo_path).all()

        print(f"Found {len(all_cached)} commits in database:")
        for commit in all_cached:
            print(f"  {commit.commit_hash} - {commit.timestamp}")

        # Check specific commits
        print("\nChecking specific commits:")
        for hash in all_commit_hashes[:5]:
            exists = (
                session.query(CachedCommit)
                .filter(CachedCommit.repo_path == repo_path, CachedCommit.commit_hash == hash)
                .first()
            )
            print(f"  {hash}: {'EXISTS' if exists else 'NOT FOUND'}")


if __name__ == "__main__":
    test_bulk_exists()
