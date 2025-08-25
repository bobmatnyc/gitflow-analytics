"""Model loader for integrating trained classification models.

This module provides functionality to load and use trained classification models
within the existing GitFlow Analytics ML pipeline. It bridges the gap between
the training pipeline and the production classification system.

WHY: Trained models need to be seamlessly integrated into the existing ML
categorization workflow without breaking backward compatibility. This loader
provides a unified interface for both rule-based and trained model classification.

DESIGN DECISIONS:
- Backward compatibility: Falls back to rule-based classification if model unavailable
- Model versioning: Supports loading specific model versions
- Performance: Caches loaded models in memory for efficiency
- Integration: Works with existing MLTicketExtractor infrastructure
"""

import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..models.database import ClassificationModel, Database

logger = logging.getLogger(__name__)


class TrainingModelLoader:
    """Load and manage trained classification models.

    This class provides functionality to load trained models from the database
    and storage, integrate them with the existing classification pipeline, and
    manage model lifecycle (versioning, caching, fallback).
    """

    def __init__(self, cache_dir: Path) -> None:
        """Initialize model loader.

        Args:
            cache_dir: Directory containing training database and models
        """
        self.cache_dir = cache_dir
        self.db = Database(cache_dir / "training.db")
        self.loaded_models: dict[str, Any] = {}  # Model cache
        self.model_metadata: dict[str, dict[str, Any]] = {}  # Metadata cache

        logger.info("TrainingModelLoader initialized")

    def get_best_model(self) -> Optional[dict[str, Any]]:
        """Get the best performing active model.

        Returns:
            Dictionary with model metadata or None if no models available
        """
        with self.db.get_session() as session:
            best_model = (
                session.query(ClassificationModel)
                .filter_by(active=True)
                .order_by(ClassificationModel.validation_accuracy.desc())
                .first()
            )

            if best_model:
                return {
                    "model_id": best_model.model_id,
                    "version": best_model.version,
                    "accuracy": best_model.validation_accuracy,
                    "categories": best_model.categories,
                    "model_path": best_model.model_file_path,
                    "model_type": best_model.model_type,
                    "created_at": best_model.created_at,
                }

        return None

    def load_model(self, model_id: Optional[str] = None) -> tuple[Any, dict[str, Any]]:
        """Load a trained model by ID or get the best available model.

        Args:
            model_id: Specific model ID to load, or None for best model

        Returns:
            Tuple of (loaded_model, model_metadata)

        Raises:
            FileNotFoundError: If model file not found
            ValueError: If model_id not found or invalid
        """
        # Check cache first
        cache_key = model_id or "best"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key], self.model_metadata[cache_key]

        # Get model metadata
        model_info = self._get_model_by_id(model_id) if model_id else self.get_best_model()

        if not model_info:
            raise ValueError(
                f"No model found with ID: {model_id}" if model_id else "No trained models available"
            )

        # Load model from file
        model_path = Path(model_info["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Cache loaded model
            self.loaded_models[cache_key] = model
            self.model_metadata[cache_key] = model_info

            # Update usage statistics
            self._update_model_usage(model_info["model_id"])

            logger.info(
                f"Loaded model {model_info['model_id']} v{model_info['version']} ({model_info['accuracy']:.3f} accuracy)"
            )
            return model, model_info

        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}") from e

    def _get_model_by_id(self, model_id: str) -> Optional[dict[str, Any]]:
        """Get model metadata by ID.

        Args:
            model_id: Model identifier

        Returns:
            Model metadata dictionary or None if not found
        """
        with self.db.get_session() as session:
            model = (
                session.query(ClassificationModel).filter_by(model_id=model_id, active=True).first()
            )

            if model:
                return {
                    "model_id": model.model_id,
                    "version": model.version,
                    "accuracy": model.validation_accuracy,
                    "categories": model.categories,
                    "model_path": model.model_file_path,
                    "model_type": model.model_type,
                    "created_at": model.created_at,
                }

        return None

    def _update_model_usage(self, model_id: str) -> None:
        """Update model usage statistics.

        Args:
            model_id: Model identifier
        """
        try:
            with self.db.get_session() as session:
                model = session.query(ClassificationModel).filter_by(model_id=model_id).first()
                if model:
                    model.usage_count = (model.usage_count or 0) + 1
                    model.last_used = datetime.utcnow()
                    session.commit()
        except Exception as e:
            logger.warning(f"Failed to update model usage for {model_id}: {e}")

    def list_available_models(self) -> list[dict[str, Any]]:
        """List all available trained models.

        Returns:
            List of model metadata dictionaries
        """
        models = []

        with self.db.get_session() as session:
            db_models = (
                session.query(ClassificationModel)
                .filter_by(active=True)
                .order_by(ClassificationModel.validation_accuracy.desc())
                .all()
            )

            for model in db_models:
                models.append(
                    {
                        "model_id": model.model_id,
                        "version": model.version,
                        "accuracy": model.validation_accuracy,
                        "categories": model.categories,
                        "model_type": model.model_type,
                        "created_at": model.created_at,
                        "usage_count": model.usage_count or 0,
                        "model_size_mb": self._get_model_file_size(model.model_file_path),
                    }
                )

        return models

    def _get_model_file_size(self, model_path: str) -> float:
        """Get model file size in MB.

        Args:
            model_path: Path to model file

        Returns:
            File size in MB
        """
        try:
            path = Path(model_path)
            if path.exists():
                return path.stat().st_size / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def predict_commit_category(
        self,
        message: str,
        files_changed: Optional[list[str]] = None,
        model_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Predict commit category using a trained model.

        This method provides a unified interface for commit classification
        that can be integrated into the existing ML pipeline.

        Args:
            message: Commit message
            files_changed: List of changed files (optional)
            model_id: Specific model to use (optional, uses best model if None)

        Returns:
            Dictionary with prediction results:
            {
                'category': str,
                'confidence': float,
                'method': 'trained_model',
                'model_info': dict,
                'alternatives': list,
                'processing_time_ms': float
            }
        """
        start_time = time.time()

        try:
            # Load model
            model, model_info = self.load_model(model_id)

            # Prepare features (simplified - in production would use same vectorizer as training)
            # This is a basic implementation - real implementation would need the training vectorizer
            prediction_scores = model.predict_proba([message])
            prediction = model.predict([message])[0]

            # Get confidence from prediction probabilities
            max_confidence = float(prediction_scores[0].max())

            # Map model prediction to standard categories
            mapped_category = self._map_model_category(prediction)

            processing_time = (time.time() - start_time) * 1000

            result = {
                "category": mapped_category,
                "confidence": max_confidence,
                "method": "trained_model",
                "model_info": {
                    "model_id": model_info["model_id"],
                    "version": model_info["version"],
                    "accuracy": model_info["accuracy"],
                },
                "alternatives": self._get_alternative_predictions(
                    prediction_scores[0], model.classes_
                ),
                "processing_time_ms": processing_time,
            }

            return result

        except Exception as e:
            logger.warning(f"Trained model prediction failed: {e}")
            # Return error indicator
            return {
                "category": "other",
                "confidence": 0.0,
                "method": "failed",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

    def _map_model_category(self, prediction: str) -> str:
        """Map model prediction to standard category names.

        Args:
            prediction: Raw model prediction

        Returns:
            Standardized category name
        """
        # This mapping should match the training category mapping
        mapping = {
            "bug_fix": "bug_fix",
            "feature": "feature",
            "refactor": "refactor",
            "documentation": "documentation",
            "test": "test",
            "maintenance": "maintenance",
            "style": "style",
            "build": "build",
        }

        return mapping.get(prediction, "other")

    def _get_alternative_predictions(
        self, prediction_scores: Any, classes: list[str]
    ) -> list[dict[str, Any]]:
        """Get alternative predictions with confidence scores.

        Args:
            prediction_scores: Model prediction probability scores
            classes: Model class names

        Returns:
            List of alternative predictions sorted by confidence
        """
        alternatives = []

        # Get top 3 alternatives (excluding the primary prediction)
        score_indices = prediction_scores.argsort()[::-1]  # Sort descending

        for i, idx in enumerate(score_indices[1:4]):  # Skip first (primary), take next 3
            alternatives.append(
                {
                    "category": self._map_model_category(classes[idx]),
                    "confidence": float(prediction_scores[idx]),
                    "rank": i + 2,
                }
            )

        return alternatives

    def get_model_statistics(self) -> dict[str, Any]:
        """Get comprehensive model loading and usage statistics.

        Returns:
            Dictionary with model statistics
        """
        stats = {
            "loaded_models_count": len(self.loaded_models),
            "available_models_count": 0,
            "total_usage_count": 0,
            "best_model_accuracy": 0.0,
            "model_types": {},
            "memory_usage_mb": 0.0,
        }

        # Get database statistics
        with self.db.get_session() as session:
            models = session.query(ClassificationModel).filter_by(active=True).all()
            stats["available_models_count"] = len(models)

            if models:
                stats["total_usage_count"] = sum(m.usage_count or 0 for m in models)
                stats["best_model_accuracy"] = max(m.validation_accuracy or 0 for m in models)

                # Count model types
                for model in models:
                    model_type = model.model_type
                    stats["model_types"][model_type] = stats["model_types"].get(model_type, 0) + 1

        # Estimate memory usage (rough approximation)
        stats["memory_usage_mb"] = len(self.loaded_models) * 5.0  # Rough estimate

        return stats
