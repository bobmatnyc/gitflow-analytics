"""LLM-specific caching layer for classification results.

This module provides persistent caching of LLM classification results
to minimize API calls and reduce costs.

WHY: LLM API calls are expensive and slow. Caching results for identical
inputs dramatically reduces costs and improves performance.

DESIGN DECISIONS:
- Use SQLite for persistence and efficient lookups
- Hash-based keys for fast matching
- Configurable expiration for cache freshness
- Statistics tracking for cache effectiveness
- Support for cache warming and export
"""

import contextlib
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LLMCache:
    """SQLite-based cache for LLM classification results.

    WHY: Persistent caching reduces API costs by 90%+ for repeated
    classifications while maintaining result consistency.
    """

    def __init__(self, cache_path: Path, expiration_days: int = 90, max_cache_size_mb: int = 500):
        """Initialize LLM cache.

        Args:
            cache_path: Path to SQLite cache database
            expiration_days: Days before cache entries expire
            max_cache_size_mb: Maximum cache size in megabytes
        """
        self.cache_path = cache_path
        self.expiration_days = expiration_days
        self.max_cache_size_mb = max_cache_size_mb

        # Ensure cache directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # Track cache statistics
        self.hits = 0
        self.misses = 0
        self.stores = 0

    def _init_database(self) -> None:
        """Initialize SQLite database with cache tables.

        WHY: Structured database enables efficient lookups and
        management of cached results.
        """
        with sqlite3.connect(self.cache_path) as conn:
            # Main cache table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    message_hash TEXT NOT NULL,
                    files_hash TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    method TEXT NOT NULL,
                    reasoning TEXT,
                    model TEXT,
                    alternatives TEXT,  -- JSON array
                    processing_time_ms REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Indices for efficient operations
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON llm_cache(expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_message_hash ON llm_cache(message_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON llm_cache(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_access_count ON llm_cache(access_count)")

            # Metadata table for cache management
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

    def get(
        self, message: str, files_changed: Optional[list[str]] = None
    ) -> Optional[dict[str, Any]]:
        """Get cached classification if available.

        Args:
            message: Commit message
            files_changed: Optional list of changed files

        Returns:
            Cached classification result or None
        """
        cache_key, _, _ = self._generate_cache_key(message, files_changed or [])

        try:
            with sqlite3.connect(self.cache_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT category, confidence, reasoning, model, alternatives,
                           method, processing_time_ms
                    FROM llm_cache
                    WHERE cache_key = ? AND expires_at > datetime('now')
                """,
                    (cache_key,),
                )

                row = cursor.fetchone()
                if row:
                    # Update access statistics
                    conn.execute(
                        """
                        UPDATE llm_cache
                        SET access_count = access_count + 1,
                            last_accessed = CURRENT_TIMESTAMP
                        WHERE cache_key = ?
                    """,
                        (cache_key,),
                    )
                    conn.commit()

                    self.hits += 1

                    # Parse alternatives from JSON
                    alternatives = []
                    if row["alternatives"]:
                        with contextlib.suppress(json.JSONDecodeError):
                            alternatives = json.loads(row["alternatives"])

                    return {
                        "category": row["category"],
                        "confidence": row["confidence"],
                        "method": "cached",
                        "reasoning": row["reasoning"] or "Cached result",
                        "model": row["model"] or "unknown",
                        "alternatives": alternatives,
                        "processing_time_ms": row["processing_time_ms"] or 0.0,
                        "cache_hit": True,
                    }

                self.misses += 1

        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
            self.misses += 1

        return None

    def store(
        self, message: str, files_changed: Optional[list[str]], result: dict[str, Any]
    ) -> bool:
        """Store classification result in cache.

        Args:
            message: Commit message
            files_changed: Optional list of changed files
            result: Classification result to cache

        Returns:
            True if stored successfully
        """
        cache_key, message_hash, files_hash = self._generate_cache_key(message, files_changed or [])

        try:
            expires_at = datetime.now() + timedelta(days=self.expiration_days)

            # Serialize alternatives
            alternatives_json = json.dumps(result.get("alternatives", []))

            with sqlite3.connect(self.cache_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO llm_cache
                    (cache_key, message_hash, files_hash, category, confidence,
                     method, reasoning, model, alternatives, processing_time_ms, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        cache_key,
                        message_hash,
                        files_hash,
                        result.get("category", "maintenance"),
                        result.get("confidence", 0.5),
                        result.get("method", "llm"),
                        result.get("reasoning", ""),
                        result.get("model", ""),
                        alternatives_json,
                        result.get("processing_time_ms", 0.0),
                        expires_at,
                    ),
                )
                conn.commit()

                self.stores += 1

                # Check cache size and cleanup if needed
                self._check_cache_size(conn)

                return True

        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
            return False

    def _generate_cache_key(self, message: str, files_changed: list[str]) -> tuple[str, str, str]:
        """Generate cache key components.

        Args:
            message: Commit message
            files_changed: List of changed files

        Returns:
            Tuple of (cache_key, message_hash, files_hash)
        """
        # Normalize message
        normalized_message = message.strip().lower()
        message_hash = hashlib.md5(normalized_message.encode("utf-8")).hexdigest()

        # Normalize and hash files
        normalized_files = "|".join(sorted(f.lower() for f in files_changed))
        files_hash = hashlib.md5(normalized_files.encode("utf-8")).hexdigest()

        # Combined cache key
        cache_key = f"{message_hash}:{files_hash}"

        return cache_key, message_hash, files_hash

    def _check_cache_size(self, conn: sqlite3.Connection) -> None:
        """Check cache size and cleanup if needed.

        WHY: Prevents cache from growing unbounded and consuming
        excessive disk space.

        Args:
            conn: SQLite connection
        """
        # Get current database size
        db_size_bytes = self.cache_path.stat().st_size if self.cache_path.exists() else 0
        db_size_mb = db_size_bytes / (1024 * 1024)

        if db_size_mb > self.max_cache_size_mb:
            logger.info(
                f"Cache size {db_size_mb:.1f}MB exceeds limit {self.max_cache_size_mb}MB, cleaning up"
            )

            # Remove expired entries first
            deleted = self.cleanup_expired()
            logger.info(f"Removed {deleted} expired entries")

            # If still too large, remove least recently accessed
            db_size_bytes = self.cache_path.stat().st_size
            db_size_mb = db_size_bytes / (1024 * 1024)

            if db_size_mb > self.max_cache_size_mb * 0.9:  # Keep 10% buffer
                # Delete 20% of least recently accessed entries
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM llm_cache
                """
                )
                total_entries = cursor.fetchone()[0]

                if total_entries > 0:
                    to_delete = int(total_entries * 0.2)
                    conn.execute(
                        """
                        DELETE FROM llm_cache
                        WHERE cache_key IN (
                            SELECT cache_key FROM llm_cache
                            ORDER BY last_accessed ASC, access_count ASC
                            LIMIT ?
                        )
                    """,
                        (to_delete,),
                    )
                    conn.commit()
                    logger.info(f"Removed {to_delete} least recently used entries")

    def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM llm_cache
                    WHERE expires_at <= datetime('now')
                """
                )
                conn.commit()
                return cursor.rowcount

        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
            return 0

    def get_statistics(self) -> dict[str, Any]:
        """Get cache usage statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT 
                        COUNT(*) as total_entries,
                        COUNT(CASE WHEN expires_at > datetime('now') THEN 1 END) as active_entries,
                        COUNT(CASE WHEN expires_at <= datetime('now') THEN 1 END) as expired_entries,
                        AVG(access_count) as avg_access_count,
                        MAX(access_count) as max_access_count,
                        COUNT(DISTINCT model) as unique_models
                    FROM llm_cache
                """
                )

                row = cursor.fetchone()
                if row:
                    # Calculate hit rate
                    total_requests = self.hits + self.misses
                    hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

                    # Get cache file size
                    cache_size_mb = (
                        self.cache_path.stat().st_size / (1024 * 1024)
                        if self.cache_path.exists()
                        else 0
                    )

                    return {
                        "total_entries": row[0],
                        "active_entries": row[1],
                        "expired_entries": row[2],
                        "avg_access_count": row[3] or 0,
                        "max_access_count": row[4] or 0,
                        "unique_models": row[5],
                        "cache_hits": self.hits,
                        "cache_misses": self.misses,
                        "cache_stores": self.stores,
                        "hit_rate": hit_rate,
                        "cache_file_size_mb": cache_size_mb,
                        "max_cache_size_mb": self.max_cache_size_mb,
                    }

        except Exception as e:
            logger.warning(f"Failed to get cache statistics: {e}")

        return {
            "error": "Failed to retrieve statistics",
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_stores": self.stores,
        }

    def warm_cache(
        self, classifications: list[tuple[str, Optional[list[str]], dict[str, Any]]]
    ) -> int:
        """Warm cache with pre-computed classifications.

        WHY: Cache warming allows bulk import of classifications,
        useful for migrations or pre-processing.

        Args:
            classifications: List of (message, files, result) tuples

        Returns:
            Number of entries added
        """
        added = 0
        for message, files, result in classifications:
            if self.store(message, files, result):
                added += 1

        logger.info(f"Warmed cache with {added} entries")
        return added

    def export_cache(self, output_file: Path) -> int:
        """Export cache contents to JSON file.

        Args:
            output_file: Path to export file

        Returns:
            Number of entries exported
        """
        try:
            with sqlite3.connect(self.cache_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM llm_cache
                    WHERE expires_at > datetime('now')
                    ORDER BY access_count DESC
                """
                )

                entries = []
                for row in cursor:
                    entry = dict(row)
                    # Parse JSON fields
                    if entry["alternatives"]:
                        try:
                            entry["alternatives"] = json.loads(entry["alternatives"])
                        except json.JSONDecodeError:
                            entry["alternatives"] = []
                    entries.append(entry)

                with open(output_file, "w") as f:
                    json.dump(
                        {
                            "cache_entries": entries,
                            "statistics": self.get_statistics(),
                            "exported_at": datetime.now().isoformat(),
                        },
                        f,
                        indent=2,
                        default=str,
                    )

                logger.info(f"Exported {len(entries)} cache entries to {output_file}")
                return len(entries)

        except Exception as e:
            logger.error(f"Cache export failed: {e}")
            return 0

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.execute("DELETE FROM llm_cache")
                conn.commit()
                cleared = cursor.rowcount

                # Reset statistics
                self.hits = 0
                self.misses = 0
                self.stores = 0

                logger.info(f"Cleared {cleared} cache entries")
                return cleared

        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return 0
