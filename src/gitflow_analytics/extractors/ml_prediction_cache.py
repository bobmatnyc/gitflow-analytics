"""SQLite-based ML prediction cache for ml_tickets.py.

Extracted from ml_tickets.py to keep that file under 800 lines.
"""

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MLPredictionCache:
    """SQLite-based cache for ML predictions with expiration support.

    WHY: ML predictions can be expensive, especially for large repositories.
    This cache stores predictions with metadata to avoid re-processing identical
    commit messages and file patterns.

    DESIGN: Uses SQLite for persistence across runs with:
    - Expiration based on configurable time periods
    - Hash-based keys for efficient lookup
    - Metadata storage for cache invalidation
    """

    def __init__(self, cache_path: Path, expiration_days: int = 30):
        """Initialize ML prediction cache.

        Args:
            cache_path: Path to SQLite cache database
            expiration_days: Number of days to keep predictions
        """
        self.cache_path = cache_path
        self.expiration_days = expiration_days
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for caching."""
        try:
            with sqlite3.connect(str(self.cache_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ml_predictions (
                        cache_key TEXT PRIMARY KEY,
                        message_hash TEXT NOT NULL,
                        files_hash TEXT NOT NULL,
                        prediction JSON NOT NULL,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        hit_count INTEGER DEFAULT 0
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ml_predictions_expires
                    ON ml_predictions(expires_at)
                """)
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to initialize ML prediction cache: {e}")

    def _generate_cache_key(self, message: str, files_changed: list[str]) -> tuple[str, str, str]:
        """Generate cache key components for a prediction.

        Args:
            message: Commit message
            files_changed: List of changed file paths

        Returns:
            Tuple of (cache_key, message_hash, files_hash)
        """
        # Hash message
        message_hash = hashlib.md5(message.encode(), usedforsecurity=False).hexdigest()

        # Hash sorted file list for consistency
        files_str = "|".join(sorted(files_changed)) if files_changed else ""
        files_hash = hashlib.md5(files_str.encode(), usedforsecurity=False).hexdigest()

        # Combined cache key
        cache_key = f"{message_hash}:{files_hash}"

        return cache_key, message_hash, files_hash

    def get_prediction(self, message: str, files_changed: list[str]) -> Optional[dict[str, Any]]:
        """Get cached prediction for a commit.

        Args:
            message: Commit message
            files_changed: List of changed file paths

        Returns:
            Cached prediction dict or None if not found/expired
        """
        try:
            cache_key, _, _ = self._generate_cache_key(message, files_changed)
            current_time = time.time()

            with sqlite3.connect(str(self.cache_path)) as conn:
                cursor = conn.execute(
                    """
                    SELECT prediction, expires_at FROM ml_predictions
                    WHERE cache_key = ? AND expires_at > ?
                    """,
                    (cache_key, current_time),
                )

                row = cursor.fetchone()
                if row:
                    prediction_json, _ = row
                    prediction = json.loads(prediction_json)

                    # Update hit count
                    conn.execute(
                        "UPDATE ml_predictions SET hit_count = hit_count + 1 WHERE cache_key = ?",
                        (cache_key,),
                    )
                    conn.commit()

                    return prediction

        except Exception as e:
            logger.debug(f"Cache lookup failed: {e}")

        return None

    def store_prediction(
        self,
        message: str,
        files_changed: list[str],
        prediction: dict[str, Any],
    ) -> bool:
        """Store ML prediction in cache.

        Args:
            message: Commit message
            files_changed: List of changed file paths
            prediction: Prediction dict to cache

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            cache_key, message_hash, files_hash = self._generate_cache_key(message, files_changed)
            current_time = time.time()
            expires_at = current_time + (self.expiration_days * 24 * 3600)
            prediction_json = json.dumps(prediction)

            with sqlite3.connect(str(self.cache_path)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ml_predictions
                    (cache_key, message_hash, files_hash, prediction, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        message_hash,
                        files_hash,
                        prediction_json,
                        current_time,
                        expires_at,
                    ),
                )
                conn.commit()
            return True

        except Exception as e:
            logger.debug(f"Cache store failed: {e}")
            return False

    def cleanup_expired(self) -> int:
        """Remove expired predictions from cache.

        Returns:
            Number of entries removed
        """
        try:
            with sqlite3.connect(str(self.cache_path)) as conn:
                cursor = conn.execute(
                    "DELETE FROM ml_predictions WHERE expires_at <= ?", (time.time(),)
                )
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
            return 0

    def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            with sqlite3.connect(str(self.cache_path)) as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_entries,
                        SUM(hit_count) as total_hits,
                        AVG(hit_count) as avg_hits,
                        MIN(created_at) as oldest_entry,
                        MAX(created_at) as newest_entry
                    FROM ml_predictions
                    WHERE expires_at > ?
                    """,
                    (time.time(),),
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "total_entries": row[0] or 0,
                        "total_hits": row[1] or 0,
                        "avg_hits_per_entry": round(row[2] or 0, 2),
                        "oldest_entry": row[3],
                        "newest_entry": row[4],
                        "cache_path": str(self.cache_path),
                    }
        except Exception as e:
            logger.warning(f"Failed to get cache statistics: {e}")

        return {"error": "Failed to retrieve statistics", "cache_path": str(self.cache_path)}
