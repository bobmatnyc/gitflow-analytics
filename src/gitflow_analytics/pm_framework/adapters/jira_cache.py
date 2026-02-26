"""JIRA ticket cache (SQLite-based).

Extracted from jira_adapter.py for size management.
"""

import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class JiraTicketCache:
    """SQLite-based cache for JIRA ticket responses.

    WHY: JIRA API calls are expensive and can be slow, especially for large
    organizations. This cache stores ticket responses with configurable TTL
    to dramatically speed up repeated runs while maintaining data freshness.

    DESIGN DECISION: Store cache in config directory (not .gitflow-cache)
    as requested, use SQLite for efficient querying and storage, include
    comprehensive metadata for cache management and performance tracking.

    Cache Strategy:
    - Individual ticket responses cached with full JSON data
    - Configurable TTL with default 7 days (168 hours)
    - Cache hit/miss metrics for performance monitoring
    - Automatic cleanup of expired entries
    - Size management with configurable limits
    """

    def __init__(self, config_dir: Path, ttl_hours: int = 168) -> None:
        """Initialize JIRA ticket cache.

        Args:
            config_dir: Directory to store cache database (config file directory)
            ttl_hours: Time to live for cached tickets in hours (default: 7 days)
        """
        self.config_dir = Path(config_dir)
        self.ttl_hours = ttl_hours
        self.cache_path = self.config_dir / "jira_tickets.db"

        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_stores = 0
        self.session_start = datetime.now()

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info(f"Initialized JIRA ticket cache: {self.cache_path}")

    def _init_database(self) -> None:
        """Initialize SQLite database with ticket cache tables.

        WHY: Comprehensive schema design captures all ticket metadata
        needed for analytics while enabling efficient querying and
        cache management operations.
        """
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jira_tickets (
                    ticket_key TEXT PRIMARY KEY,
                    project_key TEXT NOT NULL,
                    ticket_data JSON NOT NULL,
                    story_points INTEGER,
                    status TEXT,
                    issue_type TEXT,
                    assignee TEXT,
                    reporter TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Indexes for efficient querying
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_project_key
                ON jira_tickets(project_key)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON jira_tickets(expires_at)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_status
                ON jira_tickets(status)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_updated_at
                ON jira_tickets(updated_at)
            """
            )

            # Cache metadata table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()

    def get_ticket(self, ticket_key: str) -> Optional[dict[str, Any]]:
        """Retrieve cached ticket data if not expired.

        Args:
            ticket_key: JIRA ticket key (e.g., 'PROJ-123')

        Returns:
            Cached ticket data as dictionary, or None if not found/expired
        """
        with sqlite3.connect(self.cache_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT ticket_data, expires_at, access_count
                FROM jira_tickets
                WHERE ticket_key = ? AND expires_at > CURRENT_TIMESTAMP
            """,
                (ticket_key,),
            )

            row = cursor.fetchone()
            if row:
                # Update access statistics
                cursor.execute(
                    """
                    UPDATE jira_tickets
                    SET access_count = ?, last_accessed = CURRENT_TIMESTAMP
                    WHERE ticket_key = ?
                """,
                    (row["access_count"] + 1, ticket_key),
                )
                conn.commit()

                self.cache_hits += 1
                logger.debug(f"Cache HIT for ticket {ticket_key}")

                import json

                return json.loads(row["ticket_data"])

            self.cache_misses += 1
            logger.debug(f"Cache MISS for ticket {ticket_key}")
            return None

    def store_ticket(self, ticket_key: str, ticket_data: dict[str, Any]) -> None:
        """Store ticket data in cache with TTL.

        Args:
            ticket_key: JIRA ticket key (e.g., 'PROJ-123')
            ticket_data: Complete ticket data from JIRA API
        """
        import json

        # Calculate expiry time
        expires_at = datetime.now() + timedelta(hours=self.ttl_hours)

        # Extract key fields for efficient querying
        project_key = ticket_data.get("project_id", ticket_key.split("-")[0])
        story_points = ticket_data.get("story_points")
        status = ticket_data.get("status")
        issue_type = ticket_data.get("issue_type")
        assignee = (
            ticket_data.get("assignee", {}).get("display_name")
            if ticket_data.get("assignee")
            else None
        )
        reporter = (
            ticket_data.get("reporter", {}).get("display_name")
            if ticket_data.get("reporter")
            else None
        )
        created_at = ticket_data.get("created_date")
        updated_at = ticket_data.get("updated_date")
        resolved_at = ticket_data.get("resolved_date")

        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO jira_tickets (
                    ticket_key, project_key, ticket_data, story_points, status,
                    issue_type, assignee, reporter, created_at, updated_at,
                    resolved_at, expires_at, access_count, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
            """,
                (
                    ticket_key,
                    project_key,
                    json.dumps(ticket_data),
                    story_points,
                    status,
                    issue_type,
                    assignee,
                    reporter,
                    created_at,
                    updated_at,
                    resolved_at,
                    expires_at,
                ),
            )
            conn.commit()

        self.cache_stores += 1
        logger.debug(f"Cached ticket {ticket_key} (expires: {expires_at})")

    def get_project_tickets(
        self, project_key: str, include_expired: bool = False
    ) -> list[dict[str, Any]]:
        """Get all cached tickets for a project.

        Args:
            project_key: JIRA project key
            include_expired: Whether to include expired entries

        Returns:
            List of cached ticket data dictionaries
        """
        import json

        with sqlite3.connect(self.cache_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if include_expired:
                # Fully parameterized — no dynamic SQL fragments
                cursor.execute(
                    """
                    SELECT ticket_data FROM jira_tickets
                    WHERE project_key = ?
                    ORDER BY updated_at DESC
                    """,
                    (project_key,),
                )
            else:
                cursor.execute(
                    """
                    SELECT ticket_data FROM jira_tickets
                    WHERE project_key = ?
                      AND expires_at > CURRENT_TIMESTAMP
                    ORDER BY updated_at DESC
                    """,
                    (project_key,),
                )

            tickets = []
            for row in cursor.fetchall():
                tickets.append(json.loads(row["ticket_data"]))

            return tickets

    def invalidate_ticket(self, ticket_key: str) -> bool:
        """Mark a specific ticket as expired/invalid.

        Args:
            ticket_key: JIRA ticket key to invalidate

        Returns:
            True if ticket was found and invalidated, False otherwise
        """
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE jira_tickets
                SET expires_at = DATETIME('now', '-1 hour')
                WHERE ticket_key = ?
            """,
                (ticket_key,),
            )
            conn.commit()

            return cursor.rowcount > 0

    def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of expired entries removed
        """
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM jira_tickets WHERE expires_at <= CURRENT_TIMESTAMP
            """
            )
            removed = cursor.rowcount
            conn.commit()

            if removed > 0:
                logger.info(f"Cleaned up {removed} expired cache entries")

            return removed

    def clear_cache(self) -> int:
        """Clear all cached tickets.

        Returns:
            Number of entries removed
        """
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM jira_tickets")
            count = cursor.fetchone()[0]

            cursor.execute("DELETE FROM jira_tickets")
            conn.commit()

            logger.info(f"Cleared all {count} cached tickets")
            return count

    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary with cache performance and storage metrics
        """
        with sqlite3.connect(self.cache_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Basic counts
            cursor.execute("SELECT COUNT(*) as total FROM jira_tickets")
            total_tickets = cursor.fetchone()["total"]

            cursor.execute(
                """
                SELECT COUNT(*) as fresh FROM jira_tickets
                WHERE expires_at > CURRENT_TIMESTAMP
            """
            )
            fresh_tickets = cursor.fetchone()["fresh"]

            cursor.execute(
                """
                SELECT COUNT(*) as expired FROM jira_tickets
                WHERE expires_at <= CURRENT_TIMESTAMP
            """
            )
            expired_tickets = cursor.fetchone()["expired"]

            # Project distribution
            cursor.execute(
                """
                SELECT project_key, COUNT(*) as count
                FROM jira_tickets
                WHERE expires_at > CURRENT_TIMESTAMP
                GROUP BY project_key
                ORDER BY count DESC
                LIMIT 10
            """
            )
            project_distribution = {row["project_key"]: row["count"] for row in cursor.fetchall()}

            # Access patterns
            cursor.execute(
                """
                SELECT AVG(access_count) as avg_access,
                       MAX(access_count) as max_access,
                       COUNT(*) as accessed_tickets
                FROM jira_tickets
                WHERE access_count > 1 AND expires_at > CURRENT_TIMESTAMP
            """
            )
            access_stats = cursor.fetchone()

            # Recent activity
            cursor.execute(
                """
                SELECT COUNT(*) as recent FROM jira_tickets
                WHERE cached_at > DATETIME('now', '-24 hours')
            """
            )
            recent_cached = cursor.fetchone()["recent"]

            # Database size
            try:
                db_size_mb = self.cache_path.stat().st_size / (1024 * 1024)
            except FileNotFoundError:
                db_size_mb = 0

            # Session performance
            session_duration = (datetime.now() - self.session_start).total_seconds()
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

            # Time savings estimation
            api_calls_avoided = self.cache_hits
            estimated_time_saved = api_calls_avoided * 0.5  # 0.5 seconds per API call

            return {
                # Storage metrics
                "total_tickets": total_tickets,
                "fresh_tickets": fresh_tickets,
                "expired_tickets": expired_tickets,
                "database_size_mb": db_size_mb,
                "recent_cached_24h": recent_cached,
                # Performance metrics
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_stores": self.cache_stores,
                "hit_rate_percent": hit_rate,
                "total_requests": total_requests,
                # Time savings
                "api_calls_avoided": api_calls_avoided,
                "estimated_time_saved_seconds": estimated_time_saved,
                "session_duration_seconds": session_duration,
                # Access patterns
                "project_distribution": project_distribution,
                "avg_access_count": float(access_stats["avg_access"] or 0),
                "max_access_count": access_stats["max_access"] or 0,
                "frequently_accessed_tickets": access_stats["accessed_tickets"] or 0,
                # Configuration
                "ttl_hours": self.ttl_hours,
                "cache_path": str(self.cache_path),
            }

    def print_cache_summary(self) -> None:
        """Print user-friendly cache performance summary."""
        stats = self.get_cache_stats()

        print("🎫 JIRA Ticket Cache Summary")
        print("─" * 40)

        # Cache contents
        print("📦 Cache Contents:")
        print(
            f"   • Total Tickets: {stats['total_tickets']:,} ({stats['fresh_tickets']:,} fresh, {stats['expired_tickets']:,} expired)"
        )
        print(f"   • Database Size: {stats['database_size_mb']:.1f} MB")
        print(f"   • Recent Activity: {stats['recent_cached_24h']:,} tickets cached in last 24h")

        # Project distribution
        if stats["project_distribution"]:
            print("\n📊 Top Projects:")
            for project, count in list(stats["project_distribution"].items())[:5]:
                print(f"   • {project}: {count:,} tickets")

        # Performance metrics
        if stats["total_requests"] > 0:
            print("\n⚡ Session Performance:")
            print(
                f"   • Hit Rate: {stats['hit_rate_percent']:.1f}% ({stats['cache_hits']:,}/{stats['total_requests']:,})"
            )
            print(f"   • API Calls Avoided: {stats['api_calls_avoided']:,}")

            if stats["estimated_time_saved_seconds"] > 60:
                print(f"   • Time Saved: {stats['estimated_time_saved_seconds'] / 60:.1f} minutes")
            else:
                print(f"   • Time Saved: {stats['estimated_time_saved_seconds']:.1f} seconds")

        # Access patterns
        if stats["frequently_accessed_tickets"] > 0:
            print("\n🔄 Access Patterns:")
            print(f"   • Frequently Accessed: {stats['frequently_accessed_tickets']:,} tickets")
            print(f"   • Average Access Count: {stats['avg_access_count']:.1f}")
            print(f"   • Most Accessed: {stats['max_access_count']} times")

        # Performance insights
        if stats["hit_rate_percent"] > 80:
            print("   ✅ Excellent cache performance!")
        elif stats["hit_rate_percent"] > 50:
            print("   👍 Good cache performance")
        elif stats["total_requests"] > 0:
            print("   ⚠️  Consider adjusting TTL or clearing stale entries")

        print()

