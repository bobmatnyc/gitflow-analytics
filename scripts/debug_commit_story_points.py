#!/usr/bin/env python3
"""Debug script to check story points extraction in commits."""

import sqlite3
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, "src")


def check_database_story_points():
    """Check story points in the database."""
    cache_dir = Path("./configs/test-ewtn-cache")
    db_path = cache_dir / "gitflow_cache.db"

    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        return

    print(f"🔍 Checking database at {db_path}")

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check commits with story points
        cursor.execute("""
            SELECT commit_hash, message, story_points, ticket_references
            FROM cached_commits
            WHERE story_points > 0
            ORDER BY story_points DESC
            LIMIT 10
        """)

        commits_with_points = cursor.fetchall()

        print("\n📊 Commits with Story Points:")
        if commits_with_points:
            for commit_hash, message, story_points, ticket_refs in commits_with_points:
                print(f"   ✅ {commit_hash[:8]}: {story_points} points")
                print(f"      Message: {message[:60]}...")
                print(f"      Tickets: {ticket_refs}")
                print()
        else:
            print("   ❌ No commits found with story points > 0")

        # Check all commits with ticket references
        cursor.execute("""
            SELECT commit_hash, message, story_points, ticket_references
            FROM cached_commits
            WHERE ticket_references IS NOT NULL AND ticket_references != ''
            ORDER BY timestamp DESC
            LIMIT 10
        """)

        commits_with_tickets = cursor.fetchall()

        print("\n🎫 Commits with Ticket References:")
        if commits_with_tickets:
            for commit_hash, message, story_points, ticket_refs in commits_with_tickets:
                print(f"   📋 {commit_hash[:8]}: {story_points} points")
                print(f"      Message: {message[:60]}...")
                print(f"      Tickets: {ticket_refs}")
                print()
        else:
            print("   ❌ No commits found with ticket references")

        # Check total story points
        cursor.execute("SELECT SUM(story_points) FROM cached_commits WHERE story_points > 0")
        total_points = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM cached_commits")
        total_commits = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM cached_commits WHERE story_points > 0")
        commits_with_points_count = cursor.fetchone()[0]

        print("\n📈 Summary:")
        print(f"   Total commits: {total_commits}")
        print(f"   Commits with story points: {commits_with_points_count}")
        print(f"   Total story points: {total_points}")
        print(
            f"   Coverage: {(commits_with_points_count / total_commits * 100):.1f}%"
            if total_commits > 0
            else "   Coverage: 0%"
        )

        conn.close()

    except Exception as e:
        print(f"❌ Error checking database: {e}")


def check_jira_cache():
    """Check JIRA ticket cache."""
    cache_dir = Path("./configs/test-ewtn-cache")
    db_path = cache_dir / "gitflow_cache.db"

    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        return

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check if there's a JIRA cache table
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name LIKE '%jira%' OR name LIKE '%ticket%' OR name LIKE '%issue%'
        """)

        tables = cursor.fetchall()
        print(f"\n🗃️ JIRA/Ticket related tables: {[t[0] for t in tables]}")

        # Try to find cached JIRA tickets
        try:
            cursor.execute("SELECT * FROM cached_issues LIMIT 5")
            issues = cursor.fetchall()

            if issues:
                print("\n🎫 Sample cached JIRA tickets:")
                cursor.execute("PRAGMA table_info(cached_issues)")
                columns = [col[1] for col in cursor.fetchall()]
                print(f"   Columns: {columns}")

                for issue in issues[:3]:
                    print(f"   📋 {issue}")
            else:
                print("\n❌ No cached JIRA tickets found")

        except sqlite3.OperationalError as e:
            print(f"\n❌ No cached_issues table found: {e}")

        conn.close()

    except Exception as e:
        print(f"❌ Error checking JIRA cache: {e}")


def main():
    """Debug story points in commits and JIRA cache."""
    print("🔍 Debug Story Points in Commits and Database")
    print("=" * 60)

    check_database_story_points()
    check_jira_cache()


if __name__ == "__main__":
    main()
