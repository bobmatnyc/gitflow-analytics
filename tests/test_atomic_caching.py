#!/usr/bin/env python3
"""
Test script to verify atomic day-based caching behavior.

This script tests whether the caching system is properly atomic by day:
- Week 1 analysis should cache days 1-7
- Week 2 analysis should reuse cached days 1-7, only fetch days 8-14
- Each day's data should be independently cached
"""

import logging
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_commit_caching_atomic_behavior():
    """Test that commit caching is atomic by day."""
    from sqlalchemy import func

    from src.gitflow_analytics.core.cache import GitAnalysisCache
    from src.gitflow_analytics.models.database import CachedCommit

    print("🧪 Testing Commit Caching Atomic Behavior")
    print("=" * 50)

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        cache = GitAnalysisCache(cache_dir)

        # Create test commits for different days
        base_date = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        test_repo_path = "/test/repo"

        # Generate commits for 14 days
        all_commits = []
        for day in range(14):
            commit_date = base_date + timedelta(days=day)
            commit_data = {
                "hash": f"commit{day:02d}",
                "author_name": "Test Author",
                "author_email": "test@example.com",
                "message": f"Day {day + 1} commit",
                "timestamp": commit_date,
                "branch": "main",
                "is_merge": False,
                "files_changed": 2,
                "insertions": 10,
                "deletions": 5,
                "complexity_delta": 1.0,
                "story_points": None,
                "ticket_references": [],
            }
            all_commits.append(commit_data)

        # Scenario 1: Cache first 7 days (week 1)
        print("\n📅 Scenario 1: Caching Week 1 (Days 1-7)")
        week1_commits = all_commits[:7]
        cache.cache_commits_batch(test_repo_path, week1_commits)

        # Verify week 1 commits are cached
        with cache.get_session() as session:
            week1_cached_count = (
                session.query(CachedCommit).filter(CachedCommit.repo_path == test_repo_path).count()
            )
            print(f"   ✅ Week 1: {week1_cached_count} commits cached")

        # Test retrieval efficiency for week 1
        week1_hashes = [c["hash"] for c in week1_commits]
        cached_week1 = cache.get_cached_commits_bulk(test_repo_path, week1_hashes)
        print(
            f"   ✅ Week 1 retrieval: {len(cached_week1)}/{len(week1_hashes)} commits found in cache"
        )

        # Scenario 2: Cache second week (should reuse week 1, add week 2)
        print("\n📅 Scenario 2: Caching Week 2 (Days 1-14, should reuse Days 1-7)")

        # Simulate what happens during week 2 analysis
        week2_hashes = [c["hash"] for c in all_commits]  # All 14 days requested

        # Check cache hits for overlapping period
        cached_commits_week2 = cache.get_cached_commits_bulk(test_repo_path, week2_hashes)
        cache_hits = len(cached_commits_week2)
        cache_misses = len(week2_hashes) - cache_hits

        print(f"   ✅ Week 2 cache analysis: {cache_hits} hits, {cache_misses} misses")
        print("   ✅ Expected: 7 hits (week 1), 7 misses (week 2)")

        # Cache the missing commits (days 8-14)
        week2_only_commits = all_commits[7:]  # Days 8-14
        cache.cache_commits_batch(test_repo_path, week2_only_commits)

        # Verify total commits cached
        with cache.get_session() as session:
            total_cached_count = (
                session.query(CachedCommit).filter(CachedCommit.repo_path == test_repo_path).count()
            )
            print(f"   ✅ Total after week 2: {total_cached_count} commits cached")

        # Scenario 3: Test day-by-day granularity
        print("\n📊 Scenario 3: Verify Day-by-Day Granularity")

        with cache.get_session() as session:
            # Group commits by date to verify atomic daily storage
            daily_counts = (
                session.query(
                    func.date(CachedCommit.timestamp).label("date"), func.count().label("count")
                )
                .filter(CachedCommit.repo_path == test_repo_path)
                .group_by(func.date(CachedCommit.timestamp))
                .all()
            )

            print(f"   ✅ Days with cached commits: {len(daily_counts)}")
            for date_obj, count in daily_counts:
                print(f"      {date_obj}: {count} commits")

        # Scenario 4: Test partial overlap (days 5-12)
        print("\n🔄 Scenario 4: Test Partial Overlap (Days 5-12)")

        # Request commits for days 5-12 (should hit cache for 5-7, miss for 8-12)
        partial_commits = all_commits[4:12]  # Days 5-12 (0-indexed)
        partial_hashes = [c["hash"] for c in partial_commits]

        cached_partial = cache.get_cached_commits_bulk(test_repo_path, partial_hashes)
        partial_hits = len(cached_partial)
        partial_misses = len(partial_hashes) - partial_hits

        print(f"   ✅ Partial overlap: {partial_hits} hits, {partial_misses} misses")
        print("   ✅ Expected: 8 hits (all days 5-12 should be cached)")

        # Results Analysis
        print("\n📈 Caching Behavior Analysis")
        print("=" * 30)

        success = True

        # Check if we got the expected cache behavior
        if cache_hits == 7 and cache_misses == 7:
            print("   ✅ PASS: Week 1-2 overlap behaves atomically")
        else:
            print(
                f"   ❌ FAIL: Expected 7 hits/7 misses, got {cache_hits} hits/{cache_misses} misses"
            )
            success = False

        if total_cached_count == 14:
            print("   ✅ PASS: All 14 days cached correctly")
        else:
            print(f"   ❌ FAIL: Expected 14 total commits, got {total_cached_count}")
            success = False

        if len(daily_counts) == 14:
            print("   ✅ PASS: Each day stored as separate cached entry")
        else:
            print(f"   ❌ FAIL: Expected 14 daily entries, got {len(daily_counts)}")
            success = False

        if partial_hits == 8 and partial_misses == 0:
            print("   ✅ PASS: Partial overlaps work correctly")
        else:
            print(
                f"   ❌ FAIL: Partial overlap should be 8 hits/0 misses, got {partial_hits}/{partial_misses}"
            )
            success = False

        return success


def test_jira_ticket_caching_atomic_behavior():
    """Test that JIRA ticket caching is atomic by ticket ID."""
    from src.gitflow_analytics.pm_framework.adapters.jira_adapter import JiraTicketCache

    print("\n🎫 Testing JIRA Ticket Caching Atomic Behavior")
    print("=" * 50)

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        ticket_cache = JiraTicketCache(cache_dir)

        # Test tickets from different projects
        test_tickets = [
            {
                "key": "PROJ1-100",
                "data": {
                    "project_id": "PROJ1",
                    "summary": "First ticket",
                    "status": "Done",
                    "story_points": 3,
                    "created_date": "2024-01-01T10:00:00Z",
                    "updated_date": "2024-01-02T10:00:00Z",
                },
            },
            {
                "key": "PROJ1-101",
                "data": {
                    "project_id": "PROJ1",
                    "summary": "Second ticket",
                    "status": "In Progress",
                    "story_points": 5,
                    "created_date": "2024-01-02T10:00:00Z",
                    "updated_date": "2024-01-03T10:00:00Z",
                },
            },
            {
                "key": "PROJ2-200",
                "data": {
                    "project_id": "PROJ2",
                    "summary": "Different project ticket",
                    "status": "To Do",
                    "story_points": 2,
                    "created_date": "2024-01-03T10:00:00Z",
                    "updated_date": "2024-01-03T10:00:00Z",
                },
            },
        ]

        # Scenario 1: Cache individual tickets
        print("\n🏷️ Scenario 1: Individual Ticket Caching")
        for ticket in test_tickets:
            ticket_cache.store_ticket(ticket["key"], ticket["data"])
            print(f"   ✅ Cached ticket: {ticket['key']}")

        # Scenario 2: Retrieve individual tickets (should hit cache)
        print("\n🔍 Scenario 2: Individual Ticket Retrieval")
        cache_hits = 0
        cache_misses = 0

        for ticket in test_tickets:
            cached_data = ticket_cache.get_ticket(ticket["key"])
            if cached_data:
                cache_hits += 1
                print(f"   ✅ Cache HIT: {ticket['key']}")
            else:
                cache_misses += 1
                print(f"   ❌ Cache MISS: {ticket['key']}")

        # Scenario 3: Project-based retrieval
        print("\n📁 Scenario 3: Project-Based Retrieval")
        proj1_tickets = ticket_cache.get_project_tickets("PROJ1")
        proj2_tickets = ticket_cache.get_project_tickets("PROJ2")

        print(f"   ✅ PROJ1 tickets: {len(proj1_tickets)}")
        print(f"   ✅ PROJ2 tickets: {len(proj2_tickets)}")

        # Scenario 4: Cache statistics
        print("\n📊 Scenario 4: Cache Statistics")
        stats = ticket_cache.get_cache_stats()
        print(f"   ✅ Total tickets: {stats['total_tickets']}")
        print(f"   ✅ Fresh tickets: {stats['fresh_tickets']}")
        print(f"   ✅ Cache hits: {stats['cache_hits']}")
        print(f"   ✅ Cache misses: {stats['cache_misses']}")
        print(f"   ✅ Hit rate: {stats['hit_rate_percent']:.1f}%")

        # Results Analysis
        print("\n📈 JIRA Caching Behavior Analysis")
        print("=" * 35)

        success = True

        if cache_hits == 3 and cache_misses == 0:
            print("   ✅ PASS: All tickets cached and retrieved successfully")
        else:
            print(f"   ❌ FAIL: Expected 3 hits/0 misses, got {cache_hits}/{cache_misses}")
            success = False

        if len(proj1_tickets) == 2:
            print("   ✅ PASS: PROJ1 project filtering works")
        else:
            print(f"   ❌ FAIL: Expected 2 PROJ1 tickets, got {len(proj1_tickets)}")
            success = False

        if len(proj2_tickets) == 1:
            print("   ✅ PASS: PROJ2 project filtering works")
        else:
            print(f"   ❌ FAIL: Expected 1 PROJ2 ticket, got {len(proj2_tickets)}")
            success = False

        return success


def test_daily_metrics_atomic_behavior():
    """Test that daily metrics are stored atomically by day."""
    from datetime import date

    from src.gitflow_analytics.core.metrics_storage import DailyMetricsStorage

    print("\n📊 Testing Daily Metrics Atomic Behavior")
    print("=" * 50)

    # Create temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_metrics.db"
        metrics_storage = DailyMetricsStorage(db_path)

        # Create test commits for multiple days
        base_date = date(2024, 1, 1)
        test_commits = []

        # Generate commits across 5 days
        for day in range(5):
            commit_date = base_date + timedelta(days=day)
            for commit_num in range(3):  # 3 commits per day
                test_commits.append(
                    {
                        "hash": f"day{day}_commit{commit_num}",
                        "author_email": "dev@example.com",
                        "author_name": "Test Developer",
                        "timestamp": datetime.combine(
                            commit_date, datetime.min.time(), timezone.utc
                        ),
                        "project_key": "TEST_PROJECT",
                        "category": "feature" if commit_num == 0 else "bug_fix",
                        "files_changed": 2,  # Use integer instead of list for test
                        "insertions": 10,
                        "deletions": 5,
                        "ticket_references": (
                            [f"TICKET-{day}{commit_num}"] if commit_num < 2 else []
                        ),
                    }
                )

        developer_identities = {
            "dev@example.com": {
                "canonical_id": "dev123",
                "name": "Test Developer",
                "email": "dev@example.com",
            }
        }

        # Scenario 1: Store metrics for individual days
        print("\n📅 Scenario 1: Daily Metrics Storage")
        for day in range(5):
            target_date = base_date + timedelta(days=day)
            day_commits = [c for c in test_commits if c["timestamp"].date() == target_date]

            records_stored = metrics_storage.store_daily_metrics(
                target_date, day_commits, developer_identities
            )
            print(f"   ✅ Day {day + 1} ({target_date}): {records_stored} records stored")

        # Scenario 2: Retrieve metrics for date ranges
        print("\n🔍 Scenario 2: Date Range Retrieval")

        # Test 1: Single day
        day1_metrics = metrics_storage.get_date_range_metrics(base_date, base_date)
        print(f"   ✅ Single day (Day 1): {len(day1_metrics)} records")

        # Test 2: Week range (5 days)
        week_metrics = metrics_storage.get_date_range_metrics(
            base_date, base_date + timedelta(days=4)
        )
        print(f"   ✅ Week range (Days 1-5): {len(week_metrics)} records")

        # Test 3: Partial range (Days 2-4)
        partial_metrics = metrics_storage.get_date_range_metrics(
            base_date + timedelta(days=1), base_date + timedelta(days=3)
        )
        print(f"   ✅ Partial range (Days 2-4): {len(partial_metrics)} records")

        # Scenario 3: Verify daily granularity
        print("\n📊 Scenario 3: Daily Granularity Verification")

        for day in range(5):
            target_date = base_date + timedelta(days=day)
            day_metrics = metrics_storage.get_date_range_metrics(target_date, target_date)

            if day_metrics:
                metric = day_metrics[0]
                print(
                    f"   ✅ Day {day + 1}: {metric['total_commits']} commits, {metric['feature_commits']} features, {metric['bug_fix_commits']} bugs"
                )

        # Results Analysis
        print("\n📈 Daily Metrics Behavior Analysis")
        print("=" * 35)

        success = True

        if len(day1_metrics) == 1:
            print("   ✅ PASS: Single day retrieval works")
        else:
            print(f"   ❌ FAIL: Expected 1 day 1 record, got {len(day1_metrics)}")
            success = False

        if len(week_metrics) == 5:
            print("   ✅ PASS: Week range retrieval works")
        else:
            print(f"   ❌ FAIL: Expected 5 week records, got {len(week_metrics)}")
            success = False

        if len(partial_metrics) == 3:
            print("   ✅ PASS: Partial range retrieval works")
        else:
            print(f"   ❌ FAIL: Expected 3 partial records, got {len(partial_metrics)}")
            success = False

        return success


def main():
    """Main test function."""
    print("🚀 GitFlow Analytics - Atomic Caching Verification")
    print("=" * 60)
    print("Testing whether caching is properly atomic by day...")
    print("Expected behavior:")
    print("  • Week 1 analysis caches days 1-7")
    print("  • Week 2 analysis reuses days 1-7, only fetches days 8-14")
    print("  • Each day's data is independently cached")
    print("  • JIRA tickets are cached by ticket ID")
    print("  • Daily metrics are stored per day")

    try:
        results = []

        # Test 1: Commit caching
        commit_result = test_commit_caching_atomic_behavior()
        results.append(("Commit Caching", commit_result))

        # Test 2: JIRA ticket caching
        jira_result = test_jira_ticket_caching_atomic_behavior()
        results.append(("JIRA Ticket Caching", jira_result))

        # Test 3: Daily metrics
        metrics_result = test_daily_metrics_atomic_behavior()
        results.append(("Daily Metrics Storage", metrics_result))

        # Final Results
        print("\n" + "=" * 60)
        print("🏁 FINAL RESULTS")
        print("=" * 60)

        all_passed = True
        for test_name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {test_name}")
            if not passed:
                all_passed = False

        print("\n" + "=" * 60)
        if all_passed:
            print("🎉 ALL TESTS PASSED - Caching is properly atomic by day!")
            print("✅ Week 1 → Week 2 analysis will reuse overlapping cached data")
            print("✅ Each day's commits are cached independently")
            print("✅ JIRA tickets are cached by ticket ID (good)")
            print("✅ Daily metrics are stored per day")
        else:
            print("❌ SOME TESTS FAILED - Caching may not be properly atomic")
            print("⚠️  Week 1 → Week 2 analysis may not reuse cached data efficiently")

        return all_passed

    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
