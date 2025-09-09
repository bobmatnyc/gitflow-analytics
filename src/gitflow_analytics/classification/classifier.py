"""Main commit classification orchestrator.

This module provides the primary interface for commit classification,
orchestrating feature extraction, model training, and prediction.
It integrates with GitFlow Analytics' existing infrastructure and
provides both training and inference capabilities.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .feature_extractor import FeatureExtractor
from .linguist_analyzer import LinguistAnalyzer
from .model import CommitClassificationModel

logger = logging.getLogger(__name__)


class CommitClassifier:
    """Main interface for commit classification.

    This class provides a high-level interface for commit classification,
    handling the entire pipeline from feature extraction to prediction.
    It's designed to integrate seamlessly with GitFlow Analytics while
    providing standalone functionality for other use cases.

    Key capabilities:
    - Automated feature extraction from git commits
    - Model training with cross-validation
    - Batch and single commit prediction
    - Performance monitoring and metrics
    - Integration with existing GitFlow Analytics caching
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, cache_dir: Optional[Path] = None):
        """Initialize the commit classifier.

        Args:
            config: Configuration dictionary for classification parameters
            cache_dir: Directory for caching models and intermediate results
        """
        self.config = config or {}

        # Setup paths
        self.cache_dir = cache_dir or Path(".gitflow-cache")
        self.model_path = self.cache_dir / "classification"
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.linguist_analyzer = LinguistAnalyzer()
        self.model = CommitClassificationModel(
            model_path=self.model_path, config=self.config.get("model", {})
        )

        # Classification configuration
        self.enabled = self.config.get("enabled", True)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.batch_size = self.config.get("batch_size", 100)
        self.auto_retrain = self.config.get("auto_retrain", True)
        self.retrain_threshold_days = self.config.get("retrain_threshold_days", 30)

        # Supported classification categories
        self.classification_categories = {
            "feature": "New functionality or capabilities",
            "bugfix": "Bug fixes and error corrections",
            "refactor": "Code restructuring and optimization",
            "docs": "Documentation changes and updates",
            "test": "Testing-related changes",
            "config": "Configuration and settings changes",
            "chore": "Maintenance and housekeeping tasks",
            "security": "Security-related changes",
            "hotfix": "Emergency production fixes",
            "style": "Code style and formatting changes",
            "build": "Build system and dependency changes",
            "ci": "Continuous integration changes",
            "revert": "Reverts of previous changes",
            "merge": "Merge commits and integration",
            "wip": "Work in progress commits",
        }

        logger.info(
            f"CommitClassifier initialized with {len(self.classification_categories)} categories"
        )

    def train_model(
        self, training_data: list[tuple[dict[str, Any], str]], validation_split: float = 0.2
    ) -> dict[str, Any]:
        """Train the classification model on labeled data.

        Args:
            training_data: List of (commit_data, label) tuples
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary containing training results and metrics
        """
        if not self.enabled:
            raise RuntimeError("Classification is disabled in configuration")

        if len(training_data) < 20:
            raise ValueError("Need at least 20 labeled examples for reliable training")

        logger.info(f"Training commit classifier on {len(training_data)} examples")

        # Separate commits and labels
        commits = [item[0] for item in training_data]
        labels = [item[1] for item in training_data]

        # Validate labels
        valid_labels = set(self.classification_categories.keys())
        invalid_labels = set(labels) - valid_labels
        if invalid_labels:
            logger.warning(f"Found invalid labels: {invalid_labels}. Using fallback mapping.")
            labels = [self._map_fallback_label(label) for label in labels]

        # Train the model
        training_results = self.model.train(commits, labels, validation_split)

        # Log training summary
        accuracy = training_results.get("accuracy", 0.0)
        logger.info(f"Model training completed with accuracy: {accuracy:.3f}")

        return training_results

    def classify_commits(self, commits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Classify a batch of commits.

        Args:
            commits: List of commit data dictionaries

        Returns:
            List of classification results with predictions and metadata
        """
        if not self.enabled:
            logger.info("Classification disabled, returning empty results")
            return []

        if not commits:
            return []

        logger.info(f"Classifying {len(commits)} commits")

        # Check if model needs retraining
        if self.auto_retrain and self.model.retrain_needed(self.retrain_threshold_days):
            logger.warning("Model may need retraining - consider updating with recent data")

        # Process commits in batches for memory efficiency
        results = []
        for i in range(0, len(commits), self.batch_size):
            batch = commits[i : i + self.batch_size]
            batch_results = self._classify_batch(batch)
            results.extend(batch_results)

        logger.info(f"Classification completed for {len(results)} commits")
        return results

    def _classify_batch(self, commit_batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Classify a single batch of commits.

        Args:
            commit_batch: Batch of commit data dictionaries

        Returns:
            List of classification results for the batch
        """
        # Get model predictions
        predictions = self.model.predict(commit_batch)

        # Enhance results with additional analysis
        enhanced_results = []
        for _i, (commit, prediction) in enumerate(zip(commit_batch, predictions)):
            # Add file analysis context
            file_analysis = self.linguist_analyzer.analyze_commit_files(
                commit.get("files_changed", [])
            )

            # Determine if prediction is reliable
            confidence = prediction["confidence"]
            is_reliable = confidence >= self.confidence_threshold

            # Create enhanced result
            result = {
                "commit_hash": commit.get("hash", ""),
                "commit_message": commit.get("message", ""),
                "predicted_class": prediction["predicted_class"],
                "confidence": confidence,
                "is_reliable_prediction": is_reliable,
                "class_probabilities": prediction["class_probabilities"],
                "file_analysis": {
                    "primary_language": file_analysis["primary_language"],
                    "primary_activity": file_analysis["primary_activity"],
                    "file_count": file_analysis["file_count"],
                    "is_multilingual": file_analysis["is_multilingual"],
                    "is_cross_functional": file_analysis["is_cross_functional"],
                },
                "classification_metadata": {
                    "model_timestamp": self.model.training_timestamp,
                    "feature_count": 68,
                    "categories_available": len(self.classification_categories),
                },
            }

            enhanced_results.append(result)

        return enhanced_results

    def classify_single_commit(self, commit: dict[str, Any]) -> dict[str, Any]:
        """Classify a single commit.

        Args:
            commit: Commit data dictionary

        Returns:
            Classification result dictionary
        """
        results = self.classify_commits([commit])
        return results[0] if results else {}

    def get_feature_importance(self, top_n: int = 20) -> list[tuple[str, float]]:
        """Get feature importance rankings from the trained model.

        Args:
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance_score) tuples
        """
        return self.model.get_feature_importance(top_n)

    def analyze_commit_patterns(self, commits: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze patterns in a collection of commits.

        Args:
            commits: List of commit data dictionaries

        Returns:
            Dictionary with pattern analysis results
        """
        if not commits:
            return {}

        # Classify all commits
        classifications = self.classify_commits(commits)

        # Aggregate pattern statistics
        class_counts = {}
        language_usage = {}
        activity_patterns = {}
        confidence_distribution = []

        for result in classifications:
            # Count classifications
            predicted_class = result["predicted_class"]
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1

            # Track confidence scores
            confidence_distribution.append(result["confidence"])

            # Aggregate language usage
            primary_lang = result["file_analysis"]["primary_language"]
            if primary_lang:
                if primary_lang not in language_usage:
                    language_usage[primary_lang] = {}
                if predicted_class not in language_usage[primary_lang]:
                    language_usage[primary_lang][predicted_class] = 0
                language_usage[primary_lang][predicted_class] += 1

            # Aggregate activity patterns
            primary_activity = result["file_analysis"]["primary_activity"]
            if primary_activity:
                if primary_activity not in activity_patterns:
                    activity_patterns[primary_activity] = {}
                if predicted_class not in activity_patterns[primary_activity]:
                    activity_patterns[primary_activity][predicted_class] = 0
                activity_patterns[primary_activity][predicted_class] += 1

        # Calculate statistics
        total_commits = len(classifications)
        avg_confidence = (
            sum(confidence_distribution) / len(confidence_distribution)
            if confidence_distribution
            else 0.0
        )

        return {
            "total_commits_analyzed": total_commits,
            "classification_distribution": class_counts,
            "average_confidence": avg_confidence,
            "high_confidence_ratio": sum(
                1 for c in confidence_distribution if c >= self.confidence_threshold
            )
            / total_commits,
            "language_usage_patterns": language_usage,
            "activity_patterns": activity_patterns,
            "most_common_class": (
                max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else None
            ),
            "classification_diversity": len(class_counts),
            "supported_categories": list(self.classification_categories.keys()),
        }

    def _map_fallback_label(self, label: str) -> str:
        """Map unknown labels to supported categories.

        Args:
            label: Original label

        Returns:
            Mapped label from supported categories
        """
        label_lower = label.lower()

        # Common mappings
        mappings = {
            "feat": "feature",
            "fix": "bugfix",
            "bug_fix": "bugfix",  # From training pipeline
            "doc": "docs",
            "documentation": "docs",
            "testing": "test",
            "tests": "test",
            "maintenance": "chore",  # From training pipeline
            "cleanup": "chore",
            "optimization": "refactor",
            "optimize": "refactor",
            "enhancement": "feature",
            "improvement": "refactor",
            "styling": "style",
            "format": "style",
        }

        return mappings.get(label_lower, "chore")  # Default to chore

    def get_model_status(self) -> dict[str, Any]:
        """Get comprehensive status of the classification system.

        Returns:
            Dictionary with system status and capabilities
        """
        model_info = self.model.get_model_info()

        return {
            "enabled": self.enabled,
            "model_trained": model_info["is_trained"],
            "sklearn_available": model_info["sklearn_available"],
            "training_timestamp": model_info["training_timestamp"],
            "supported_categories": list(self.classification_categories.keys()),
            "confidence_threshold": self.confidence_threshold,
            "batch_size": self.batch_size,
            "model_path": str(self.model_path),
            "auto_retrain_enabled": self.auto_retrain,
            "needs_retraining": self.model.retrain_needed(self.retrain_threshold_days),
            "training_metrics": model_info.get("training_metrics", {}),
            "cache_directory": str(self.cache_dir),
        }

    def export_training_data(self, commits: list[dict[str, Any]], output_path: Path) -> None:
        """Export commits in a format suitable for manual labeling.

        Args:
            commits: List of commit data dictionaries
            output_path: Path to save the training data CSV
        """
        import csv

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "hash",
                    "message",
                    "author",
                    "timestamp",
                    "files_changed",
                    "insertions",
                    "deletions",
                    "primary_language",
                    "primary_activity",
                    "suggested_class",
                    "manual_label",
                ]
            )

            # Analyze commits for suggestions
            for commit in commits:
                file_analysis = self.linguist_analyzer.analyze_commit_files(
                    commit.get("files_changed", [])
                )

                # Get a prediction for suggestion
                if self.model.is_trained:
                    prediction = self.classify_single_commit(commit)
                    suggested_class = prediction.get("predicted_class", "unknown")
                else:
                    suggested_class = "unknown"

                # Write row
                writer.writerow(
                    [
                        commit.get("hash", ""),
                        commit.get("message", ""),
                        commit.get("author_name", ""),
                        commit.get("timestamp", ""),
                        len(commit.get("files_changed", [])),
                        commit.get("insertions", 0),
                        commit.get("deletions", 0),
                        file_analysis["primary_language"] or "unknown",
                        file_analysis["primary_activity"] or "unknown",
                        suggested_class,
                        "",  # Empty column for manual labeling
                    ]
                )

        logger.info(f"Training data exported to {output_path}")

    def load_training_data(self, csv_path: Path) -> list[tuple[dict[str, Any], str]]:
        """Load manually labeled training data from CSV.

        Args:
            csv_path: Path to CSV file with labeled data

        Returns:
            List of (commit_data, label) tuples
        """
        import csv

        training_data = []

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Skip rows without manual labels
                if not row.get("manual_label", "").strip():
                    continue

                # Parse timestamp
                timestamp_str = row.get("timestamp", "")
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    timestamp = datetime.now()

                # Create commit data structure
                commit_data = {
                    "hash": row.get("hash", ""),
                    "message": row.get("message", ""),
                    "author_name": row.get("author", ""),
                    "timestamp": timestamp,
                    "files_changed": [],  # Would need to be reconstructed from git
                    "insertions": int(row.get("insertions", 0) or 0),
                    "deletions": int(row.get("deletions", 0) or 0),
                }

                label = row["manual_label"].strip()
                training_data.append((commit_data, label))

        logger.info(f"Loaded {len(training_data)} labeled examples from {csv_path}")
        return training_data
