#!/usr/bin/env python3
"""
Debug script to identify the database storage issue.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the gitflow-analytics package to the path
sys.path.insert(0, "/Users/masa/Projects/managed/gitflow-analytics/src")

try:
    import git

    from gitflow_analytics.config import ConfigLoader
    from gitflow_analytics.core.cache import GitAnalysisCache
    from gitflow_analytics.core.data_fetcher import GitDataFetcher
    from gitflow_analytics.models.database import CachedCommit, Database
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_database_basic_operations():
    """Test basic database operations."""
    print("🔍 Testing Basic Database Operations...")

    try:
        # Create a test database
        cache_dir = Path("./debug-cache")
        cache_dir.mkdir(exist_ok=True)

        db = Database(cache_dir / "test.db")
        print("✅ Database created successfully")

        # Test session creation
        session = db.get_session()
        print("✅ Session created successfully")

        # Test simple query
        result = session.execute("SELECT 1").fetchone()
        print(f"✅ Simple query works: {result}")

        session.close()
        return True

    except Exception as e:
        print(f"❌ Database basic operations failed: {e}")
        import traceback

        print(f"Full traceback: {traceback.format_exc()}")
        return False


def test_bulk_commit_storage():
    """Test bulk commit storage directly."""
    print("\n🔍 Testing Bulk Commit Storage...")

    try:
        # Create cache
        cache_dir = Path("./debug-cache")
        cache_dir.mkdir(exist_ok=True)
        cache = GitAnalysisCache(cache_dir)

        # Create test commits
        test_commits = [
            {
                "hash": "abc123",
                "author_name": "Test Author",
                "author_email": "test@example.com",
                "message": "Test commit 1",
                "timestamp": datetime.utcnow(),
                "branch": "main",
                "is_merge": False,
                "files_changed": 1,
                "insertions": 10,
                "deletions": 5,
                "complexity_delta": 0.1,
                "story_points": None,
                "ticket_references": [],
            },
            {
                "hash": "def456",
                "author_name": "Test Author 2",
                "author_email": "test2@example.com",
                "message": "Test commit 2",
                "timestamp": datetime.utcnow(),
                "branch": "main",
                "is_merge": False,
                "files_changed": 2,
                "insertions": 20,
                "deletions": 10,
                "complexity_delta": 0.2,
                "story_points": 3,
                "ticket_references": ["TEST-123"],
            },
        ]

        # Test bulk storage
        print(f"Attempting to store {len(test_commits)} test commits...")
        stats = cache.bulk_store_commits("test-repo", test_commits)
        print(f"✅ Bulk storage result: {stats}")

        # Verify storage
        stored_commits = cache.bulk_get_commits("test-repo", ["abc123", "def456"])
        print(f"✅ Verification: {len(stored_commits)} commits retrieved")

        for hash, commit_data in stored_commits.items():
            print(f"   - {hash}: {commit_data.get('message', 'No message')}")
            if commit_data.get("story_points"):
                print(f"     Story points: {commit_data['story_points']}")

        return len(stored_commits) == len(test_commits)

    except Exception as e:
        print(f"❌ Bulk commit storage failed: {e}")
        import traceback

        print(f"Full traceback: {traceback.format_exc()}")
        return False


def test_git_data_fetcher():
    """Test the GitDataFetcher with a small date range."""
    print("\n🔍 Testing GitDataFetcher...")

    try:
        # Load config
        config = ConfigLoader.load("configs/ewtn-test-config.yaml")
        print("✅ Config loaded")

        # Create cache
        cache_dir = Path("./debug-cache")
        cache_dir.mkdir(exist_ok=True)
        cache = GitAnalysisCache(cache_dir)

        # Create data fetcher (only needs cache)
        fetcher = GitDataFetcher(
            cache=cache,
            branch_mapping_rules=getattr(config.analysis, "branch_mapping_rules", {}),
            allowed_ticket_platforms=getattr(
                config.analysis, "ticket_platforms", ["jira", "github"]
            ),
            exclude_paths=getattr(config.analysis, "exclude_paths", None),
        )
        print("✅ Data fetcher created")

        # Test with a very small date range (last 3 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)

        repo_path = Path(".")
        project_key = "TEST_PROJECT"

        print(f"Testing fetch for date range: {start_date.date()} to {end_date.date()}")

        # Get git repo
        repo = git.Repo(".")

        # Get commits in date range
        commits = list(
            repo.iter_commits(
                since=start_date.strftime("%Y-%m-%d"), until=end_date.strftime("%Y-%m-%d")
            )
        )

        print(f"Found {len(commits)} commits in git log for this range")

        if len(commits) == 0:
            print("⚠️  No commits in date range - trying last 7 days")
            start_date = end_date - timedelta(days=7)
            commits = list(
                repo.iter_commits(
                    since=start_date.strftime("%Y-%m-%d"), until=end_date.strftime("%Y-%m-%d")
                )
            )
            print(f"Found {len(commits)} commits in 7-day range")

        if len(commits) > 0:
            print("Sample commits:")
            for i, commit in enumerate(commits[:3]):
                print(
                    f"  {i+1}. {commit.hexsha[:8]} - {commit.message.split()[0] if commit.message else 'No message'}"
                )

        # Try to fetch data
        result = fetcher.fetch_repository_data(
            repo_path=repo_path, project_key=project_key, start_date=start_date, end_date=end_date
        )

        print(f"✅ Fetch result: {result}")
        return True

    except Exception as e:
        print(f"❌ GitDataFetcher test failed: {e}")
        import traceback

        print(f"Full traceback: {traceback.format_exc()}")
        return False


def main():
    """Run all database diagnostic tests."""
    print("🚀 Database Storage Diagnostic")
    print("=" * 50)

    # Test 1: Basic database operations
    basic_works = test_database_basic_operations()

    # Test 2: Bulk commit storage
    bulk_works = test_bulk_commit_storage()

    # Test 3: GitDataFetcher
    fetcher_works = test_git_data_fetcher()

    # Summary
    print("\n" + "=" * 50)
    print("📋 Diagnostic Summary:")
    print(f"   ✅ Basic Database Operations: {'PASS' if basic_works else 'FAIL'}")
    print(f"   ✅ Bulk Commit Storage: {'PASS' if bulk_works else 'FAIL'}")
    print(f"   ✅ GitDataFetcher: {'PASS' if fetcher_works else 'FAIL'}")

    if all([basic_works, bulk_works, fetcher_works]):
        print("\n🎉 All tests passed! Database storage should be working.")
        return True
    else:
        print("\n❌ Some tests failed. Database storage needs fixes.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
