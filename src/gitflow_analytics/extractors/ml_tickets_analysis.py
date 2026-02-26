"""ML-enhanced coverage analysis mixin for MLTicketExtractor.

Provides analyze_ticket_coverage override, ML quality analysis, and
get_ml_statistics methods. Extracted from ml_tickets.py to keep it
under 800 lines.

Methods here depend on instance attributes set by MLTicketExtractor.__init__:
    self.enable_ml, self.enable_llm, self.untracked_file_threshold,
    self.ml_cache, self.trained_model_loader, self.llm_classifier,
    self.ml_config, self.llm_config_dict, self.change_type_classifier,
    self.nlp_model
and on instance methods:
    self.categorize_commit_with_confidence()
"""

import logging
from collections import defaultdict
from typing import Any

try:
    import spacy  # noqa: F401

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from ..training.model_loader import TrainingModelLoader  # noqa: F401

    TRAINING_LOADER_AVAILABLE = True
except ImportError:
    TRAINING_LOADER_AVAILABLE = False

logger = logging.getLogger(__name__)


class MLTicketAnalysisMixin:
    """Mixin adding ML-enhanced coverage analysis to MLTicketExtractor."""

    def analyze_ticket_coverage(
        self, commits: list[dict[str, Any]], prs: list[dict[str, Any]], progress_display=None
    ) -> dict[str, Any]:
        """Enhanced ticket coverage analysis with ML categorization insights.

        This method extends the parent's analysis with ML-specific insights including
        confidence distributions, method breakdowns, and prediction quality metrics.

        Args:
            commits: List of commit data
            prs: List of PR data
            progress_display: Optional progress display for showing analysis progress

        Returns:
            Enhanced analysis results with ML insights
        """
        base_analysis = super().analyze_ticket_coverage(commits, prs, progress_display)  # type: ignore[misc]

        if not self.enable_ml:  # type: ignore[attr-defined]
            base_analysis["ml_analysis"] = {
                "enabled": False,
                "reason": "ML components not available or disabled",
            }
            return base_analysis

        ml_analysis = self._analyze_ml_categorization_quality(commits)
        base_analysis["ml_analysis"] = ml_analysis

        if "untracked_commits" in base_analysis:
            self._enhance_untracked_commits(base_analysis["untracked_commits"])

        return base_analysis

    def _analyze_ml_categorization_quality(self, commits: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze the quality and distribution of ML categorizations.

        Args:
            commits: List of commit data

        Returns:
            ML analysis results including confidence distributions and method usage
        """
        ml_stats: dict[str, Any] = {
            "enabled": True,
            "total_ml_predictions": 0,
            "total_rule_predictions": 0,
            "total_cached_predictions": 0,
            "avg_confidence": 0.0,
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "method_breakdown": defaultdict(int),
            "category_confidence": defaultdict(list),
            "processing_time_stats": {"total_ms": 0.0, "avg_ms": 0.0},
        }

        total_confidence = 0.0
        total_processing_time = 0.0
        processed_commits = 0

        threshold = self.untracked_file_threshold  # type: ignore[attr-defined]

        for commit in commits:
            files_count = commit.get("files_changed_count")
            if files_count is None:
                files_changed = commit.get("files_changed", 0)
                if isinstance(files_changed, int):
                    files_count = files_changed
                elif isinstance(files_changed, list):
                    files_count = len(files_changed)
                else:
                    logger.warning(
                        f"Unexpected files_changed type: {type(files_changed)}, defaulting to 0"
                    )
                    files_count = 0

            if commit.get("is_merge") or files_count < threshold:
                continue

            message = commit.get("message", "")
            files_changed_raw = commit.get("files_changed", [])
            if isinstance(files_changed_raw, int):
                files_changed_list: list[str] = []
            elif isinstance(files_changed_raw, list):
                files_changed_list = files_changed_raw
            else:
                files_changed_list = []

            result = self.categorize_commit_with_confidence(message, files_changed_list)  # type: ignore[attr-defined]

            confidence = result["confidence"]
            method = result["method"]
            category = result["category"]
            processing_time = result.get("processing_time_ms", 0.0)

            total_confidence += confidence
            total_processing_time += processing_time
            processed_commits += 1

            ml_stats["method_breakdown"][method] += 1
            if method == "ml":
                ml_stats["total_ml_predictions"] += 1
            elif method == "rules":
                ml_stats["total_rule_predictions"] += 1
            elif method == "cached":
                ml_stats["total_cached_predictions"] += 1

            if confidence >= 0.8:
                ml_stats["confidence_distribution"]["high"] += 1
            elif confidence >= 0.6:
                ml_stats["confidence_distribution"]["medium"] += 1
            else:
                ml_stats["confidence_distribution"]["low"] += 1

            ml_stats["category_confidence"][category].append(confidence)

        if processed_commits > 0:
            ml_stats["avg_confidence"] = total_confidence / processed_commits
            ml_stats["processing_time_stats"] = {
                "total_ms": total_processing_time,
                "avg_ms": total_processing_time / processed_commits,
            }

        ml_stats["method_breakdown"] = dict(ml_stats["method_breakdown"])
        ml_stats["category_confidence"] = {
            cat: {"avg": sum(confidences) / len(confidences), "count": len(confidences)}
            for cat, confidences in ml_stats["category_confidence"].items()
        }

        return ml_stats

    def _enhance_untracked_commits(self, untracked_commits: list[dict[str, Any]]) -> None:
        """Enhance untracked commits with ML confidence scores and metadata.

        Args:
            untracked_commits: List of untracked commit data to enhance in-place
        """
        for commit in untracked_commits:
            message = commit.get("full_message", commit.get("message", ""))
            files_changed: list[str] = []

            result = self.categorize_commit_with_confidence(message, files_changed)  # type: ignore[attr-defined]

            commit["ml_confidence"] = result["confidence"]
            commit["ml_method"] = result["method"]
            commit["ml_alternatives"] = result.get("alternatives", [])
            commit["ml_processing_time_ms"] = result.get("processing_time_ms", 0.0)

    def get_ml_statistics(self) -> dict[str, Any]:
        """Get comprehensive ML and LLM usage and performance statistics.

        Returns:
            Dictionary with ML/LLM performance metrics and usage statistics
        """
        stats: dict[str, Any] = {
            "ml_enabled": self.enable_ml,  # type: ignore[attr-defined]
            "llm_enabled": self.enable_llm,  # type: ignore[attr-defined]
            "spacy_available": SPACY_AVAILABLE,
            "training_loader_available": TRAINING_LOADER_AVAILABLE,
            "components_loaded": {
                "change_type_classifier": self.change_type_classifier is not None,  # type: ignore[attr-defined]
                "nlp_model": self.nlp_model is not None,  # type: ignore[attr-defined]
                "ml_cache": self.ml_cache is not None,  # type: ignore[attr-defined]
                "trained_model_loader": self.trained_model_loader is not None,  # type: ignore[attr-defined]
                "llm_classifier": self.llm_classifier is not None,  # type: ignore[attr-defined]
            },
            "configuration": {
                "ml_config": self.ml_config.copy(),  # type: ignore[attr-defined]
                "llm_config": self.llm_config_dict.copy(),  # type: ignore[attr-defined]
            },
        }

        if self.ml_cache:  # type: ignore[attr-defined]
            stats["cache_statistics"] = self.ml_cache.get_statistics()  # type: ignore[attr-defined]

        if self.trained_model_loader:  # type: ignore[attr-defined]
            try:
                stats["trained_model_statistics"] = self.trained_model_loader.get_model_statistics()  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"Failed to get trained model statistics: {e}")
                stats["trained_model_statistics"] = {"error": str(e)}

        if self.llm_classifier:  # type: ignore[attr-defined]
            try:
                stats["llm_statistics"] = self.llm_classifier.get_statistics()  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"Failed to get LLM statistics: {e}")
                stats["llm_statistics"] = {"error": str(e)}

        return stats
