"""Machine learning model for commit classification.

This module implements a Random Forest-based commit classification model with
comprehensive training, validation, and prediction capabilities. The model is
designed for production use with robust error handling, model persistence,
and performance monitoring.
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import LabelEncoder

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None
    LabelEncoder = None

from .feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class CommitClassificationModel:
    """Random Forest-based commit classification model.

    This model provides comprehensive commit classification using a Random Forest
    classifier trained on 68-dimensional feature vectors. It includes:

    - Robust training pipeline with cross-validation
    - Model persistence and versioning
    - Batch prediction capabilities
    - Performance monitoring and metrics
    - Graceful fallback when scikit-learn is unavailable

    The model is designed to classify commits into categories such as:
    - feature: New functionality
    - bugfix: Bug fixes and corrections
    - refactor: Code restructuring
    - docs: Documentation changes
    - test: Testing-related changes
    - config: Configuration changes
    - chore: Maintenance tasks
    - security: Security-related changes
    - hotfix: Emergency fixes
    """

    def __init__(self, model_path: Optional[Path] = None, config: Optional[dict[str, Any]] = None):
        """Initialize the commit classification model.

        Args:
            model_path: Path to save/load model files
            config: Configuration dictionary with model parameters
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Model functionality will be limited.")
            self.model = None
            self.label_encoder = None
            self.feature_extractor = None
            return

        self.model_path = model_path or Path(".gitflow-cache/classification")
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Configuration with defaults
        self.config = config or {}
        self.n_estimators = self.config.get("n_estimators", 100)
        self.max_depth = self.config.get("max_depth", 20)
        self.min_samples_split = self.config.get("min_samples_split", 5)
        self.min_samples_leaf = self.config.get("min_samples_leaf", 2)
        self.random_state = self.config.get("random_state", 42)
        self.n_jobs = self.config.get("n_jobs", -1)  # Use all available cores

        # Initialize components
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight="balanced",  # Handle class imbalance
        )
        self.label_encoder = LabelEncoder()
        self.feature_extractor = FeatureExtractor()

        # Model metadata
        self.is_trained = False
        self.training_timestamp = None
        self.feature_importance = None
        self.class_names = None
        self.training_metrics = {}

        # Load existing model if available
        self._load_model()

    def train(
        self, commits: list[dict[str, Any]], labels: list[str], validation_split: float = 0.2
    ) -> dict[str, Any]:
        """Train the classification model on labeled commit data.

        Args:
            commits: List of commit data dictionaries
            labels: List of corresponding classification labels
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary containing training metrics and results
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for model training")

        if len(commits) != len(labels):
            raise ValueError("Number of commits must match number of labels")

        if len(commits) < 10:
            raise ValueError("Need at least 10 samples for training")

        logger.info(f"Training classification model on {len(commits)} commits")

        # Extract features from commits
        logger.info("Extracting features from commits...")
        features = self.feature_extractor.extract_batch_features(commits)

        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.class_names = self.label_encoder.classes_.tolist()

        # Split data for validation
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                features,
                encoded_labels,
                test_size=validation_split,
                random_state=self.random_state,
                stratify=encoded_labels,
            )
        else:
            X_train, y_train = features, encoded_labels
            X_val, y_val = None, None

        # Train the model
        logger.info("Training Random Forest classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.training_timestamp = datetime.now()

        # Calculate feature importance
        self.feature_importance = self.model.feature_importances_

        # Evaluate the model
        training_metrics = self._evaluate_model(X_train, y_train, X_val, y_val)
        self.training_metrics = training_metrics

        # Save the trained model
        self._save_model()

        logger.info(f"Model training completed. Accuracy: {training_metrics['accuracy']:.3f}")
        return training_metrics

    def predict(self, commits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Predict classifications for a batch of commits.

        Args:
            commits: List of commit data dictionaries

        Returns:
            List of prediction dictionaries containing:
            - predicted_class: Predicted classification
            - confidence: Prediction confidence (0-1)
            - class_probabilities: Probabilities for all classes
        """
        if not SKLEARN_AVAILABLE or not self.is_trained:
            logger.warning("Model not available or not trained. Using fallback classification.")
            return self._fallback_predictions(commits)

        if not commits:
            return []

        # Extract features
        features = self.feature_extractor.extract_batch_features(commits)

        # Make predictions
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)

        # Format results
        results = []
        for i, commit in enumerate(commits):
            predicted_label = self.label_encoder.inverse_transform([predictions[i]])[0]
            max_prob = np.max(probabilities[i])

            # Create probability dictionary for all classes
            class_probs = dict(zip(self.class_names, probabilities[i]))

            results.append(
                {
                    "commit_hash": commit.get("hash", ""),
                    "predicted_class": predicted_label,
                    "confidence": float(max_prob),
                    "class_probabilities": class_probs,
                }
            )

        return results

    def predict_single(self, commit: dict[str, Any]) -> dict[str, Any]:
        """Predict classification for a single commit.

        Args:
            commit: Commit data dictionary

        Returns:
            Prediction dictionary with class and confidence
        """
        results = self.predict([commit])
        return results[0] if results else {"predicted_class": "unknown", "confidence": 0.0}

    def _evaluate_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Evaluate model performance with comprehensive metrics.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}

        # Cross-validation on training data
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring="accuracy")
        metrics["cv_accuracy_mean"] = float(np.mean(cv_scores))
        metrics["cv_accuracy_std"] = float(np.std(cv_scores))

        # Training accuracy
        train_pred = self.model.predict(X_train)
        metrics["train_accuracy"] = float(accuracy_score(y_train, train_pred))

        # Validation metrics if validation data provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            metrics["val_accuracy"] = float(accuracy_score(y_val, val_pred))

            # Detailed classification report
            class_names = [
                self.label_encoder.inverse_transform([i])[0]
                for i in range(len(self.label_encoder.classes_))
            ]

            val_report = classification_report(
                y_val, val_pred, target_names=class_names, output_dict=True
            )
            metrics["classification_report"] = val_report

            # Confusion matrix
            conf_matrix = confusion_matrix(y_val, val_pred)
            metrics["confusion_matrix"] = conf_matrix.tolist()

        # Overall accuracy for reporting
        metrics["accuracy"] = metrics.get("val_accuracy", metrics["train_accuracy"])

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> list[tuple[str, float]]:
        """Get top feature importances from the trained model.

        Args:
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples, sorted by importance
        """
        if not self.is_trained or self.feature_importance is None:
            return []

        feature_names = self.feature_extractor.get_feature_names()
        importance_pairs = list(zip(feature_names, self.feature_importance))

        # Sort by importance descending
        importance_pairs.sort(key=lambda x: x[1], reverse=True)

        return importance_pairs[:top_n]

    def _save_model(self) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            return

        model_file = self.model_path / "commit_classifier.joblib"
        metadata_file = self.model_path / "model_metadata.pkl"

        try:
            # Save the scikit-learn model
            joblib.dump(self.model, model_file)

            # Save metadata
            metadata = {
                "label_encoder": self.label_encoder,
                "is_trained": self.is_trained,
                "training_timestamp": self.training_timestamp,
                "feature_importance": self.feature_importance,
                "class_names": self.class_names,
                "training_metrics": self.training_metrics,
                "config": self.config,
            }

            with open(metadata_file, "wb") as f:
                pickle.dump(metadata, f)

            logger.info(f"Model saved to {model_file}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def _load_model(self) -> bool:
        """Load a previously trained model from disk.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if not SKLEARN_AVAILABLE:
            return False

        model_file = self.model_path / "commit_classifier.joblib"
        metadata_file = self.model_path / "model_metadata.pkl"

        if not (model_file.exists() and metadata_file.exists()):
            return False

        try:
            # Load the scikit-learn model
            self.model = joblib.load(model_file)

            # Load metadata
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)

            self.label_encoder = metadata["label_encoder"]
            self.is_trained = metadata["is_trained"]
            self.training_timestamp = metadata["training_timestamp"]
            self.feature_importance = metadata["feature_importance"]
            self.class_names = metadata["class_names"]
            self.training_metrics = metadata["training_metrics"]

            # Check if model is too old (older than 30 days)
            if self.training_timestamp:
                age = datetime.now() - self.training_timestamp
                if age > timedelta(days=30):
                    logger.warning(f"Loaded model is {age.days} days old. Consider retraining.")

            logger.info(f"Model loaded from {model_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _fallback_predictions(self, commits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Provide fallback predictions when ML model is not available.

        Args:
            commits: List of commit data dictionaries

        Returns:
            List of basic prediction dictionaries
        """
        results = []

        for commit in commits:
            message = commit.get("message", "").lower()

            # Simple rule-based fallback classification
            predicted_class = "chore"  # Default
            confidence = 0.3  # Low confidence for rule-based

            if any(word in message for word in ["fix", "bug", "error", "issue"]):
                predicted_class = "bugfix"
                confidence = 0.6
            elif any(word in message for word in ["feat", "add", "implement", "new"]):
                predicted_class = "feature"
                confidence = 0.6
            elif any(word in message for word in ["doc", "readme", "comment"]):
                predicted_class = "docs"
                confidence = 0.7
            elif any(word in message for word in ["test", "spec", "coverage"]):
                predicted_class = "test"
                confidence = 0.7
            elif any(word in message for word in ["refactor", "cleanup", "optimize"]):
                predicted_class = "refactor"
                confidence = 0.6
            elif any(word in message for word in ["config", "setting", "env"]):
                predicted_class = "config"
                confidence = 0.6

            results.append(
                {
                    "commit_hash": commit.get("hash", ""),
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "class_probabilities": {predicted_class: confidence},
                }
            )

        return results

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model state.

        Returns:
            Dictionary with model information
        """
        return {
            "is_trained": self.is_trained,
            "sklearn_available": SKLEARN_AVAILABLE,
            "training_timestamp": self.training_timestamp,
            "class_names": self.class_names,
            "n_classes": len(self.class_names) if self.class_names else 0,
            "training_metrics": self.training_metrics,
            "model_path": str(self.model_path),
        }

    def retrain_needed(self, days_old: int = 30) -> bool:
        """Check if model retraining is recommended.

        Args:
            days_old: Age threshold in days

        Returns:
            True if retraining is recommended
        """
        if not self.is_trained or not self.training_timestamp:
            return True

        age = datetime.now() - self.training_timestamp
        return age.days > days_old
