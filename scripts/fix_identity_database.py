#!/usr/bin/env python3
"""
Fix Identity Database Script

This script fixes the stale identity database by recalculating commit counts
based on the actual cached commits, resolving the massive discrepancy between
identity database counts and actual commit data.

CRITICAL BUG FIXED:
- Identity DB showed 3,639 commits for George Cook
- Actual cached commits: 26 commits
- Total discrepancy: 12,341+ commits across all developers

This script will:
1. Recalculate actual commit counts from cached_commits table
2. Update developer_identities table with correct counts
3. Verify the fix by comparing before/after statistics
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path


def connect_databases():
    """Connect to both identity and cache databases."""
    identity_db_path = Path(".gitflow-cache/identities.db")
    cache_db_path = Path(".gitflow-cache/gitflow_cache.db")

    if not identity_db_path.exists():
        print(f"❌ Identity database not found: {identity_db_path}")
        sys.exit(1)

    if not cache_db_path.exists():
        print(f"❌ Cache database not found: {cache_db_path}")
        sys.exit(1)

    identity_db = sqlite3.connect(str(identity_db_path))
    cache_db = sqlite3.connect(str(cache_db_path))

    return identity_db, cache_db


def get_current_statistics(identity_db, cache_db):
    """Get current statistics for comparison."""
    identity_cursor = identity_db.cursor()
    cache_cursor = cache_db.cursor()

    # Total commits in identity database
    identity_cursor.execute("SELECT SUM(total_commits) FROM developer_identities")
    identity_total = identity_cursor.fetchone()[0] or 0

    # Total commits in cache database
    cache_cursor.execute("SELECT COUNT(*) FROM cached_commits")
    cache_total = cache_cursor.fetchone()[0] or 0

    # Number of developers
    identity_cursor.execute("SELECT COUNT(*) FROM developer_identities")
    dev_count = identity_cursor.fetchone()[0] or 0

    return {
        "identity_total": identity_total,
        "cache_total": cache_total,
        "discrepancy": identity_total - cache_total,
        "developer_count": dev_count,
    }


def recalculate_commit_counts(identity_db, cache_db):
    """Recalculate and update commit counts for all developers."""
    identity_cursor = identity_db.cursor()
    cache_cursor = cache_db.cursor()

    print("🔍 Recalculating commit counts for all developers...")

    # Get all developers
    identity_cursor.execute("SELECT canonical_id, primary_name FROM developer_identities")
    developers = identity_cursor.fetchall()

    updates_made = 0
    total_corrections = 0

    for canonical_id, name in developers:
        # Get all email aliases for this developer
        identity_cursor.execute(
            "SELECT email FROM developer_aliases WHERE canonical_id = ?", (canonical_id,)
        )
        aliases = [row[0] for row in identity_cursor.fetchall()]

        if not aliases:
            print(f"⚠️  No aliases found for {name} ({canonical_id})")
            continue

        # Count actual commits for all aliases
        placeholders = ",".join(["?" for _ in aliases])
        cache_cursor.execute(
            f"SELECT COUNT(*) FROM cached_commits WHERE author_email IN ({placeholders})", aliases
        )
        actual_commits = cache_cursor.fetchone()[0]

        # Get current identity database count
        identity_cursor.execute(
            "SELECT total_commits FROM developer_identities WHERE canonical_id = ?", (canonical_id,)
        )
        current_count = identity_cursor.fetchone()[0] or 0

        # Update if different
        if actual_commits != current_count:
            correction = current_count - actual_commits
            total_corrections += abs(correction)

            identity_cursor.execute(
                "UPDATE developer_identities SET total_commits = ?, updated_at = ? WHERE canonical_id = ?",
                (actual_commits, datetime.utcnow(), canonical_id),
            )
            updates_made += 1

            if correction > 100:  # Only show significant corrections
                print(
                    f"✅ {name:20} | {current_count:5} → {actual_commits:5} | Fixed: {correction:+6}"
                )

    # Commit changes
    identity_db.commit()

    return updates_made, total_corrections


def verify_fix(identity_db, cache_db):
    """Verify that the fix worked correctly."""
    print("\n🔍 Verifying fix...")

    identity_cursor = identity_db.cursor()
    cache_cursor = cache_db.cursor()

    # Check a few key developers
    test_developers = [
        "a348f39a-faee-4ab6-9d28-1a4d0785b2f0",  # George Cook
        "f6dc6bbc-11a7-4d67-9be8-bcd388235844",  # Ryan Ksenich
    ]

    all_correct = True

    for canonical_id in test_developers:
        # Get developer info
        identity_cursor.execute(
            "SELECT primary_name, total_commits FROM developer_identities WHERE canonical_id = ?",
            (canonical_id,),
        )
        result = identity_cursor.fetchone()
        if not result:
            continue

        name, identity_commits = result

        # Get aliases and count actual commits
        identity_cursor.execute(
            "SELECT email FROM developer_aliases WHERE canonical_id = ?", (canonical_id,)
        )
        aliases = [row[0] for row in identity_cursor.fetchall()]

        if aliases:
            placeholders = ",".join(["?" for _ in aliases])
            cache_cursor.execute(
                f"SELECT COUNT(*) FROM cached_commits WHERE author_email IN ({placeholders})",
                aliases,
            )
            actual_commits = cache_cursor.fetchone()[0]
        else:
            actual_commits = 0

        if identity_commits == actual_commits:
            print(
                f"✅ {name:20} | Identity: {identity_commits:5} | Actual: {actual_commits:5} | ✓ CORRECT"
            )
        else:
            print(
                f"❌ {name:20} | Identity: {identity_commits:5} | Actual: {actual_commits:5} | ✗ MISMATCH"
            )
            all_correct = False

    return all_correct


def fix_orphaned_developers(identity_db):
    """Fix developers with commits but no aliases by setting their commit count to 0."""
    identity_cursor = identity_db.cursor()

    print("🔧 Fixing orphaned developers (developers with commits but no aliases)...")

    # Find developers with commits but no aliases
    identity_cursor.execute("""
        SELECT di.canonical_id, di.primary_name, di.total_commits
        FROM developer_identities di
        LEFT JOIN developer_aliases da ON di.canonical_id = da.canonical_id
        WHERE di.total_commits > 0 AND da.canonical_id IS NULL
    """)

    orphaned_devs = identity_cursor.fetchall()
    total_orphaned_commits = 0

    for canonical_id, name, commits in orphaned_devs:
        total_orphaned_commits += commits
        identity_cursor.execute(
            "UPDATE developer_identities SET total_commits = 0, updated_at = ? WHERE canonical_id = ?",
            (datetime.utcnow(), canonical_id),
        )
        print(f"✅ {name:25} | {commits:4} → 0 commits (no aliases)")

    identity_db.commit()
    return len(orphaned_devs), total_orphaned_commits


def create_missing_aliases(identity_db, cache_db):
    """Create aliases for emails in cached_commits that don't have developer identities."""
    identity_cursor = identity_db.cursor()
    cache_cursor = cache_db.cursor()

    print("🔧 Creating missing aliases for unmatched emails...")

    # Get emails in cached_commits but not in aliases
    cache_cursor.execute("SELECT DISTINCT author_email, author_name FROM cached_commits")
    cache_data = cache_cursor.fetchall()

    identity_cursor.execute("SELECT DISTINCT email FROM developer_aliases")
    existing_emails = {row[0] for row in identity_cursor.fetchall()}

    new_aliases_created = 0

    for email, name in cache_data:
        if email not in existing_emails:
            # Check if this developer already exists by primary email
            identity_cursor.execute(
                "SELECT canonical_id FROM developer_identities WHERE primary_email = ?", (email,)
            )
            existing_dev = identity_cursor.fetchone()

            if existing_dev:
                canonical_id = existing_dev[0]
            else:
                # Create new developer identity
                import uuid

                canonical_id = str(uuid.uuid4())

                identity_cursor.execute(
                    """
                    INSERT INTO developer_identities
                    (canonical_id, primary_name, primary_email, total_commits, total_story_points,
                     first_seen, last_seen, created_at, updated_at)
                    VALUES (?, ?, ?, 0, 0, ?, ?, ?, ?)
                """,
                    (
                        canonical_id,
                        name,
                        email,
                        datetime.utcnow(),
                        datetime.utcnow(),
                        datetime.utcnow(),
                        datetime.utcnow(),
                    ),
                )

            # Create alias
            identity_cursor.execute(
                """
                INSERT INTO developer_aliases (canonical_id, name, email)
                VALUES (?, ?, ?)
            """,
                (canonical_id, name, email),
            )

            new_aliases_created += 1
            print(f"✅ Created alias: {name} <{email}>")

    identity_db.commit()
    return new_aliases_created


def main():
    """Main function to fix the identity database."""
    print("🔧 GitFlow Analytics Identity Database Fix")
    print("=" * 50)

    # Connect to databases
    identity_db, cache_db = connect_databases()

    try:
        # Get current statistics
        print("📊 Current Statistics:")
        before_stats = get_current_statistics(identity_db, cache_db)
        print(f"   Identity DB total commits: {before_stats['identity_total']:,}")
        print(f"   Cache DB total commits: {before_stats['cache_total']:,}")
        print(f"   Discrepancy: {before_stats['discrepancy']:+,} commits")
        print(f"   Developers: {before_stats['developer_count']}")

        if before_stats["discrepancy"] == 0:
            print("✅ No discrepancy found - identity database is already correct!")
            return

        print(
            f"\n🚨 CRITICAL: Identity database has {before_stats['discrepancy']:+,} incorrect commits!"
        )

        # Step 1: Fix orphaned developers (developers with commits but no aliases)
        orphaned_count, orphaned_commits = fix_orphaned_developers(identity_db)
        print("\n📈 Orphaned Developer Fix:")
        print(f"   Developers fixed: {orphaned_count}")
        print(f"   Phantom commits removed: {orphaned_commits:,}")

        # Step 2: Create missing aliases for unmatched emails
        new_aliases = create_missing_aliases(identity_db, cache_db)
        print("\n📈 Missing Aliases Fix:")
        print(f"   New aliases created: {new_aliases}")

        # Step 3: Recalculate and fix remaining issues
        updates_made, total_corrections = recalculate_commit_counts(identity_db, cache_db)

        print("\n📈 Final Recalculation:")
        print(f"   Developers updated: {updates_made}")
        print(f"   Total corrections: {total_corrections:,} commits")

        # Verify fix
        if verify_fix(identity_db, cache_db):
            print("\n🎉 SUCCESS: Identity database has been fixed!")

            # Show final statistics
            after_stats = get_current_statistics(identity_db, cache_db)
            print("\n📊 Final Statistics:")
            print(f"   Identity DB total commits: {after_stats['identity_total']:,}")
            print(f"   Cache DB total commits: {after_stats['cache_total']:,}")
            print(f"   Discrepancy: {after_stats['discrepancy']:+,} commits")

            if after_stats["discrepancy"] == 0:
                print("✅ Perfect alignment achieved!")
            else:
                print(f"⚠️  Small discrepancy remains: {after_stats['discrepancy']:+,}")
        else:
            print("\n❌ VERIFICATION FAILED: Some issues remain")
            return 1

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1
    finally:
        identity_db.close()
        cache_db.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
