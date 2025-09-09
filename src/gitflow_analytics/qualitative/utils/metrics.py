"""Performance and accuracy metrics for qualitative analysis."""

import logging
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ProcessingMetrics:
    """Metrics for a single processing operation."""

    operation: str
    processing_time_ms: float
    items_processed: int
    confidence_score: float
    method_used: str  # 'nlp' or 'llm'
    timestamp: datetime

    @property
    def items_per_second(self) -> float:
        """Calculate processing rate."""
        if self.processing_time_ms <= 0:
            return 0.0
        return (self.items_processed * 1000) / self.processing_time_ms


class PerformanceMetrics:
    """Track and analyze performance metrics for qualitative analysis.

    This class provides comprehensive performance monitoring including
    processing times, accuracy metrics, cost tracking, and system health
    indicators for the qualitative analysis pipeline.
    """

    def __init__(self, max_history: int = 10000):
        """Initialize performance metrics tracker.

        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        self.logger = logging.getLogger(__name__)

        # Processing metrics
        self.processing_metrics: deque[ProcessingMetrics] = deque(maxlen=max_history)

        # Method usage tracking
        self.method_usage = defaultdict(int)
        self.method_performance = defaultdict(list)

        # Confidence tracking
        self.confidence_history = deque(maxlen=max_history)

        # Error tracking
        self.error_counts = defaultdict(int)
        self.error_history = deque(maxlen=1000)

        # Cache performance
        self.cache_hits = 0
        self.cache_misses = 0

        # Quality metrics
        self.classification_accuracy = deque(maxlen=1000)

    def record_processing(
        self,
        operation: str,
        processing_time_ms: float,
        items_processed: int,
        confidence_score: float,
        method_used: str,
    ) -> None:
        """Record a processing operation.

        Args:
            operation: Type of operation (e.g., 'classification', 'analysis')
            processing_time_ms: Processing time in milliseconds
            items_processed: Number of items processed
            confidence_score: Average confidence score
            method_used: Method used ('nlp' or 'llm')
        """
        metric = ProcessingMetrics(
            operation=operation,
            processing_time_ms=processing_time_ms,
            items_processed=items_processed,
            confidence_score=confidence_score,
            method_used=method_used,
            timestamp=datetime.utcnow(),
        )

        self.processing_metrics.append(metric)
        self.method_usage[method_used] += items_processed
        self.method_performance[method_used].append(processing_time_ms / items_processed)
        self.confidence_history.append(confidence_score)

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.cache_misses += 1

    def record_error(self, error_type: str, error_message: str) -> None:
        """Record an error occurrence.

        Args:
            error_type: Type of error
            error_message: Error message
        """
        self.error_counts[error_type] += 1
        self.error_history.append(
            {"type": error_type, "message": error_message, "timestamp": datetime.utcnow()}
        )

    def record_classification_accuracy(self, accuracy: float) -> None:
        """Record classification accuracy measurement.

        Args:
            accuracy: Accuracy score (0.0 to 1.0)
        """
        self.classification_accuracy.append(accuracy)

    def get_processing_stats(self, hours: int = 24) -> dict[str, any]:
        """Get processing statistics for the last N hours.

        Args:
            hours: Number of hours to analyze

        Returns:
            Dictionary with processing statistics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [m for m in self.processing_metrics if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {
                "total_operations": 0,
                "total_items_processed": 0,
                "avg_processing_time_ms": 0.0,
                "avg_items_per_second": 0.0,
                "avg_confidence": 0.0,
                "method_breakdown": {},
                "cache_hit_rate": 0.0,
            }

        # Calculate statistics
        total_items = sum(m.items_processed for m in recent_metrics)
        total_time = sum(m.processing_time_ms for m in recent_metrics)

        avg_processing_time = total_time / len(recent_metrics)
        avg_items_per_second = statistics.mean([m.items_per_second for m in recent_metrics])
        avg_confidence = statistics.mean([m.confidence_score for m in recent_metrics])

        # Method breakdown
        method_breakdown = {}
        for method in ["nlp", "llm"]:
            method_metrics = [m for m in recent_metrics if m.method_used == method]
            if method_metrics:
                method_items = sum(m.items_processed for m in method_metrics)
                method_breakdown[method] = {
                    "items_processed": method_items,
                    "percentage": (method_items / total_items) * 100 if total_items > 0 else 0,
                    "avg_confidence": statistics.mean([m.confidence_score for m in method_metrics]),
                    "avg_processing_time_ms": statistics.mean(
                        [m.processing_time_ms for m in method_metrics]
                    ),
                }

        # Cache hit rate
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (
            (self.cache_hits / total_cache_requests) if total_cache_requests > 0 else 0.0
        )

        return {
            "total_operations": len(recent_metrics),
            "total_items_processed": total_items,
            "avg_processing_time_ms": avg_processing_time,
            "avg_items_per_second": avg_items_per_second,
            "avg_confidence": avg_confidence,
            "method_breakdown": method_breakdown,
            "cache_hit_rate": cache_hit_rate,
            "time_period_hours": hours,
        }

    def get_quality_metrics(self) -> dict[str, any]:
        """Get quality and accuracy metrics.

        Returns:
            Dictionary with quality metrics
        """
        if not self.confidence_history:
            return {
                "avg_confidence": 0.0,
                "confidence_distribution": {},
                "classification_accuracy": 0.0,
                "quality_trend": "stable",
            }

        # Confidence statistics
        confidences = list(self.confidence_history)
        avg_confidence = statistics.mean(confidences)

        # Confidence distribution
        confidence_buckets = {
            "high (>0.8)": sum(1 for c in confidences if c > 0.8),
            "medium (0.6-0.8)": sum(1 for c in confidences if 0.6 <= c <= 0.8),
            "low (<0.6)": sum(1 for c in confidences if c < 0.6),
        }

        # Quality trend (comparing recent vs. older metrics)
        if len(confidences) >= 100:
            recent_confidence = statistics.mean(confidences[-50:])
            older_confidence = statistics.mean(confidences[-100:-50])

            if recent_confidence > older_confidence + 0.05:
                quality_trend = "improving"
            elif recent_confidence < older_confidence - 0.05:
                quality_trend = "declining"
            else:
                quality_trend = "stable"
        else:
            quality_trend = "insufficient_data"

        # Classification accuracy
        avg_accuracy = (
            statistics.mean(self.classification_accuracy) if self.classification_accuracy else 0.0
        )

        return {
            "avg_confidence": avg_confidence,
            "confidence_distribution": confidence_buckets,
            "classification_accuracy": avg_accuracy,
            "quality_trend": quality_trend,
            "total_samples": len(confidences),
        }

    def get_error_analysis(self) -> dict[str, any]:
        """Get error analysis and system health metrics.

        Returns:
            Dictionary with error analysis
        """
        # Recent errors (last 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        recent_errors = [error for error in self.error_history if error["timestamp"] >= cutoff_time]

        # Error type breakdown
        error_type_counts = defaultdict(int)
        for error in recent_errors:
            error_type_counts[error["type"]] += 1

        # Total operations for error rate calculation
        total_operations = len([m for m in self.processing_metrics if m.timestamp >= cutoff_time])

        error_rate = len(recent_errors) / total_operations if total_operations > 0 else 0.0

        return {
            "total_errors_24h": len(recent_errors),
            "error_rate": error_rate,
            "error_types": dict(error_type_counts),
            "most_common_error": (
                max(error_type_counts.keys(), key=error_type_counts.get)
                if error_type_counts
                else None
            ),
            "system_health": (
                "healthy" if error_rate < 0.01 else "degraded" if error_rate < 0.05 else "unhealthy"
            ),
        }

    def get_performance_alerts(self) -> list[str]:
        """Get performance alerts and recommendations.

        Returns:
            List of alert messages
        """
        alerts = []

        # Check recent performance
        stats = self.get_processing_stats(hours=1)
        quality = self.get_quality_metrics()
        errors = self.get_error_analysis()

        # Processing speed alerts
        if stats["avg_items_per_second"] < 50:  # Less than 50 items/second
            alerts.append("Processing speed below target (< 50 items/second)")

        # Confidence alerts
        if quality["avg_confidence"] < 0.6:
            alerts.append("Average confidence below threshold (< 0.6)")

        # Method balance alerts
        if "llm" in stats["method_breakdown"]:
            llm_percentage = stats["method_breakdown"]["llm"]["percentage"]
            if llm_percentage > 20:  # More than 20% using LLM
                alerts.append(
                    f"High LLM usage ({llm_percentage:.1f}%) - consider tuning NLP thresholds"
                )

        # Error rate alerts
        if errors["error_rate"] > 0.05:  # More than 5% error rate
            alerts.append(f"High error rate ({errors['error_rate']:.1%})")

        # Cache performance alerts
        if stats["cache_hit_rate"] < 0.3:  # Less than 30% cache hit rate
            alerts.append("Low cache hit rate - pattern learning may be ineffective")

        # Quality trend alerts
        if quality["quality_trend"] == "declining":
            alerts.append("Quality trend declining - review recent changes")

        return alerts

    def get_optimization_suggestions(self) -> list[str]:
        """Get optimization suggestions based on metrics.

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        stats = self.get_processing_stats(hours=24)
        quality = self.get_quality_metrics()

        # Performance optimizations
        if stats["avg_items_per_second"] < 100:
            suggestions.append("Consider increasing batch size or enabling parallel processing")

        # Method optimization
        method_breakdown = stats["method_breakdown"]
        if "llm" in method_breakdown and method_breakdown["llm"]["percentage"] > 15:
            suggestions.append(
                "High LLM usage - consider lowering confidence threshold or improving NLP patterns"
            )

        if "nlp" in method_breakdown and method_breakdown["nlp"]["avg_confidence"] < 0.7:
            suggestions.append("NLP confidence low - consider updating classification patterns")

        # Quality optimizations
        if quality["avg_confidence"] < 0.7:
            suggestions.append(
                "Overall confidence low - review classification accuracy and update models"
            )

        confidence_dist = quality["confidence_distribution"]
        if confidence_dist.get("low (<0.6)", 0) > confidence_dist.get("high (>0.8)", 0):
            suggestions.append(
                "Many low-confidence predictions - consider additional training data"
            )

        # Cache optimizations
        if stats["cache_hit_rate"] < 0.5:
            suggestions.append(
                "Low cache hit rate - increase cache size or improve pattern matching"
            )

        return suggestions
