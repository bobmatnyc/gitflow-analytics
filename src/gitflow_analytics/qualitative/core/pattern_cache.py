"""Pattern caching system for qualitative analysis optimization."""

import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import and_, desc

from ...models.database import Database
from ...models.database import PatternCache as PatternCacheModel
from ..models.schemas import CacheConfig, QualitativeCommitData
from ..utils.text_processing import TextProcessor


class PatternCache:
    """Intelligent caching system for qualitative analysis patterns.

    This system learns from successful classifications to speed up future
    processing and improve accuracy through pattern recognition.

    Features:
    - Semantic fingerprinting for pattern matching
    - Hit count tracking for popular patterns
    - Automatic cache cleanup and optimization
    - Pattern learning from successful classifications
    """

    def __init__(self, config: CacheConfig, database: Database):
        """Initialize pattern cache.

        Args:
            config: Cache configuration
            database: Database instance for persistence
        """
        self.config = config
        self.database = database
        self.logger = logging.getLogger(__name__)

        # Initialize text processor for fingerprinting
        self.text_processor = TextProcessor()

        # In-memory cache for frequently accessed patterns
        self._memory_cache: dict[str, dict[str, Any]] = {}
        self._memory_cache_hits = defaultdict(int)

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.pattern_learning_count = 0

        # Initialize cache directory if using file-based caching
        cache_dir = Path(config.cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)

        self.logger.info(f"Pattern cache initialized with TTL: {config.pattern_cache_ttl_hours}h")

    def lookup_pattern(self, message: str, files: list[str]) -> Optional[dict[str, Any]]:
        """Look up cached classification for a commit pattern.

        Args:
            message: Commit message
            files: List of changed files

        Returns:
            Cached classification result or None if not found
        """
        # Generate semantic fingerprint
        fingerprint = self.text_processor.create_semantic_fingerprint(message, files)

        # Check in-memory cache first
        if fingerprint in self._memory_cache:
            self._memory_cache_hits[fingerprint] += 1
            self.cache_hits += 1
            self.logger.debug(f"Memory cache hit for pattern: {fingerprint[:8]}")
            return self._memory_cache[fingerprint]

        # Check database cache
        with self.database.get_session() as session:
            cached_pattern = (
                session.query(PatternCacheModel)
                .filter(PatternCacheModel.semantic_fingerprint == fingerprint)
                .first()
            )

            if cached_pattern:
                # Check if pattern is still valid (not expired)
                cutoff_time = datetime.utcnow() - timedelta(
                    hours=self.config.pattern_cache_ttl_hours
                )

                if cached_pattern.created_at > cutoff_time:
                    # Update hit count and last used
                    cached_pattern.hit_count += 1
                    cached_pattern.last_used = datetime.utcnow()
                    session.commit()

                    # Add to memory cache for faster future access
                    result = cached_pattern.classification_result
                    self._add_to_memory_cache(fingerprint, result)

                    self.cache_hits += 1
                    self.logger.debug(f"Database cache hit for pattern: {fingerprint[:8]}")
                    return result
                else:
                    # Pattern is expired, remove it
                    session.delete(cached_pattern)
                    session.commit()
                    self.logger.debug(f"Expired pattern removed: {fingerprint[:8]}")

        self.cache_misses += 1
        return None

    def store_pattern(
        self,
        message: str,
        files: list[str],
        classification_result: dict[str, Any],
        confidence_score: float,
        source_method: str,
        processing_time_ms: float = 0.0,
    ) -> None:
        """Store a new pattern in the cache.

        Args:
            message: Commit message
            files: List of changed files
            classification_result: Classification results to cache
            confidence_score: Confidence in the classification
            source_method: Method that produced this result ('nlp' or 'llm')
            processing_time_ms: Time taken to process
        """
        # Only cache high-confidence results
        if confidence_score < 0.6:
            return

        fingerprint = self.text_processor.create_semantic_fingerprint(message, files)
        message_hash = hashlib.md5(message.encode()).hexdigest()

        # Add to memory cache
        self._add_to_memory_cache(fingerprint, classification_result)

        # Store in database
        with self.database.get_session() as session:
            # Check if pattern already exists
            existing_pattern = (
                session.query(PatternCacheModel)
                .filter(PatternCacheModel.semantic_fingerprint == fingerprint)
                .first()
            )

            if existing_pattern:
                # Update existing pattern with new data
                existing_pattern.hit_count += 1
                existing_pattern.last_used = datetime.utcnow()

                # Update confidence if new result is more confident
                if confidence_score > existing_pattern.confidence_score:
                    existing_pattern.classification_result = classification_result
                    existing_pattern.confidence_score = confidence_score
                    existing_pattern.source_method = source_method

                # Update average processing time
                if processing_time_ms > 0:
                    if existing_pattern.avg_processing_time_ms:
                        existing_pattern.avg_processing_time_ms = (
                            existing_pattern.avg_processing_time_ms + processing_time_ms
                        ) / 2
                    else:
                        existing_pattern.avg_processing_time_ms = processing_time_ms
            else:
                # Create new pattern
                new_pattern = PatternCacheModel(
                    message_hash=message_hash,
                    semantic_fingerprint=fingerprint,
                    classification_result=classification_result,
                    confidence_score=confidence_score,
                    source_method=source_method,
                    avg_processing_time_ms=processing_time_ms,
                )
                session.add(new_pattern)
                self.pattern_learning_count += 1

            session.commit()

        self.logger.debug(
            f"Stored pattern: {fingerprint[:8]} "
            f"(confidence: {confidence_score:.2f}, method: {source_method})"
        )

    def learn_from_results(self, results: list[QualitativeCommitData]) -> None:
        """Learn patterns from successful classification results.

        Args:
            results: List of classification results to learn from
        """
        learned_patterns = 0

        for result in results:
            if result.confidence_score >= 0.7:  # Only learn from high-confidence results
                classification_data = {
                    "change_type": result.change_type,
                    "change_type_confidence": result.change_type_confidence,
                    "business_domain": result.business_domain,
                    "domain_confidence": result.domain_confidence,
                    "risk_level": result.risk_level,
                    "confidence_score": result.confidence_score,
                }

                self.store_pattern(
                    message=result.message,
                    files=result.files_changed,
                    classification_result=classification_data,
                    confidence_score=result.confidence_score,
                    source_method=result.processing_method,
                    processing_time_ms=result.processing_time_ms,
                )
                learned_patterns += 1

        if learned_patterns > 0:
            self.logger.info(f"Learned {learned_patterns} new patterns from results")

    def _add_to_memory_cache(self, fingerprint: str, result: dict[str, Any]) -> None:
        """Add result to in-memory cache with size management.

        Args:
            fingerprint: Pattern fingerprint
            result: Classification result
        """
        # Manage memory cache size
        if len(self._memory_cache) >= self.config.semantic_cache_size:
            # Remove least recently used items
            sorted_items = sorted(self._memory_cache_hits.items(), key=lambda x: x[1])

            # Remove bottom 20% of items
            items_to_remove = len(sorted_items) // 5
            for fingerprint_to_remove, _ in sorted_items[:items_to_remove]:
                self._memory_cache.pop(fingerprint_to_remove, None)
                self._memory_cache_hits.pop(fingerprint_to_remove, None)

        self._memory_cache[fingerprint] = result
        self._memory_cache_hits[fingerprint] = 1

    def cleanup_cache(self) -> dict[str, int]:
        """Clean up expired and low-quality cache entries.

        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            "expired_removed": 0,
            "low_confidence_removed": 0,
            "low_usage_removed": 0,
            "total_remaining": 0,
        }

        cutoff_time = datetime.utcnow() - timedelta(hours=self.config.pattern_cache_ttl_hours)

        with self.database.get_session() as session:
            # Remove expired patterns
            expired_patterns = session.query(PatternCacheModel).filter(
                PatternCacheModel.created_at < cutoff_time
            )
            stats["expired_removed"] = expired_patterns.count()
            expired_patterns.delete()

            # Remove very low confidence patterns (< 0.4)
            low_confidence_patterns = session.query(PatternCacheModel).filter(
                PatternCacheModel.confidence_score < 0.4
            )
            stats["low_confidence_removed"] = low_confidence_patterns.count()
            low_confidence_patterns.delete()

            # Remove patterns with very low usage (hit_count = 1 and older than 7 days)
            week_ago = datetime.utcnow() - timedelta(days=7)
            low_usage_patterns = session.query(PatternCacheModel).filter(
                and_(PatternCacheModel.hit_count == 1, PatternCacheModel.created_at < week_ago)
            )
            stats["low_usage_removed"] = low_usage_patterns.count()
            low_usage_patterns.delete()

            # Count remaining patterns
            stats["total_remaining"] = session.query(PatternCacheModel).count()

            session.commit()

        # Clear memory cache to force refresh
        self._memory_cache.clear()
        self._memory_cache_hits.clear()

        self.logger.info(
            f"Cache cleanup completed: {stats['expired_removed']} expired, "
            f"{stats['low_confidence_removed']} low-confidence, "
            f"{stats['low_usage_removed']} low-usage removed. "
            f"{stats['total_remaining']} patterns remaining."
        )

        return stats

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self.database.get_session() as session:
            # Basic counts
            total_patterns = session.query(PatternCacheModel).count()

            # Method breakdown
            nlp_patterns = (
                session.query(PatternCacheModel)
                .filter(PatternCacheModel.source_method == "nlp")
                .count()
            )

            llm_patterns = (
                session.query(PatternCacheModel)
                .filter(PatternCacheModel.source_method == "llm")
                .count()
            )

            # Confidence distribution
            high_confidence = (
                session.query(PatternCacheModel)
                .filter(PatternCacheModel.confidence_score > 0.8)
                .count()
            )

            medium_confidence = (
                session.query(PatternCacheModel)
                .filter(
                    and_(
                        PatternCacheModel.confidence_score >= 0.6,
                        PatternCacheModel.confidence_score <= 0.8,
                    )
                )
                .count()
            )

            # Usage statistics
            top_patterns = (
                session.query(PatternCacheModel)
                .order_by(desc(PatternCacheModel.hit_count))
                .limit(10)
                .all()
            )

            # Age statistics
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_patterns = (
                session.query(PatternCacheModel)
                .filter(PatternCacheModel.created_at > week_ago)
                .count()
            )

        # Calculate hit rate
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests) if total_requests > 0 else 0.0

        return {
            "total_patterns": total_patterns,
            "method_breakdown": {"nlp_patterns": nlp_patterns, "llm_patterns": llm_patterns},
            "confidence_distribution": {
                "high_confidence": high_confidence,
                "medium_confidence": medium_confidence,
                "low_confidence": total_patterns - high_confidence - medium_confidence,
            },
            "usage_stats": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": hit_rate,
                "memory_cache_size": len(self._memory_cache),
            },
            "top_patterns": [
                {
                    "fingerprint": p.semantic_fingerprint[:8],
                    "hit_count": p.hit_count,
                    "confidence": p.confidence_score,
                    "method": p.source_method,
                }
                for p in top_patterns
            ],
            "recent_patterns": recent_patterns,
            "patterns_learned": self.pattern_learning_count,
            "config": {
                "ttl_hours": self.config.pattern_cache_ttl_hours,
                "max_memory_size": self.config.semantic_cache_size,
                "learning_enabled": self.config.enable_pattern_learning,
            },
        }

    def optimize_cache(self) -> dict[str, Any]:
        """Optimize cache for better performance.

        Returns:
            Dictionary with optimization results
        """
        optimization_stats = {}

        # Step 1: Cleanup expired and low-quality entries
        cleanup_stats = self.cleanup_cache()
        optimization_stats["cleanup"] = cleanup_stats

        # Step 2: Promote high-usage patterns to memory cache
        with self.database.get_session() as session:
            # Get top patterns by hit count
            popular_patterns = (
                session.query(PatternCacheModel)
                .filter(PatternCacheModel.hit_count >= 5)
                .order_by(desc(PatternCacheModel.hit_count))
                .limit(self.config.semantic_cache_size // 2)  # Fill half of memory cache
                .all()
            )

            promoted_count = 0
            for pattern in popular_patterns:
                if pattern.semantic_fingerprint not in self._memory_cache:
                    self._add_to_memory_cache(
                        pattern.semantic_fingerprint, pattern.classification_result
                    )
                    promoted_count += 1

            optimization_stats["promoted_to_memory"] = promoted_count

        # Step 3: Update learning threshold based on cache size
        total_patterns = cleanup_stats["total_remaining"]
        if total_patterns > 1000:
            # Increase learning threshold for large caches
            self.config.learning_threshold = min(20, self.config.learning_threshold + 2)
        elif total_patterns < 100:
            # Decrease learning threshold for small caches
            self.config.learning_threshold = max(5, self.config.learning_threshold - 1)

        optimization_stats["learning_threshold"] = self.config.learning_threshold

        self.logger.info(f"Cache optimization completed: {optimization_stats}")
        return optimization_stats

    def export_patterns(self, output_path: Path, min_confidence: float = 0.8) -> int:
        """Export high-quality patterns for analysis or backup.

        Args:
            output_path: Path to export file
            min_confidence: Minimum confidence threshold for export

        Returns:
            Number of patterns exported
        """
        with self.database.get_session() as session:
            patterns = (
                session.query(PatternCacheModel)
                .filter(PatternCacheModel.confidence_score >= min_confidence)
                .order_by(desc(PatternCacheModel.hit_count))
                .all()
            )

            export_data = []
            for pattern in patterns:
                export_data.append(
                    {
                        "semantic_fingerprint": pattern.semantic_fingerprint,
                        "classification_result": pattern.classification_result,
                        "confidence_score": pattern.confidence_score,
                        "hit_count": pattern.hit_count,
                        "source_method": pattern.source_method,
                        "created_at": pattern.created_at.isoformat(),
                        "last_used": pattern.last_used.isoformat(),
                    }
                )

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)

        self.logger.info(f"Exported {len(patterns)} patterns to {output_path}")
        return len(patterns)
