"""ML-enhanced ticket reference extraction with sophisticated commit categorization.

This module extends the basic TicketExtractor with machine learning capabilities for
better commit categorization. It integrates with the existing qualitative analysis
infrastructure to provide hybrid rule-based + ML classification.

WHY: Traditional regex-based categorization has limitations in understanding context
and nuanced commit messages. This ML-enhanced version provides better accuracy while
maintaining backward compatibility and performance through intelligent caching.

DESIGN DECISIONS:
- Hybrid approach: Falls back to rule-based when ML confidence is low
- Confidence scoring: All classifications include confidence scores for reliability
- Caching strategy: ML predictions are cached to maintain performance
- Feature extraction: Uses both message content and file patterns for better accuracy
- Integration: Leverages existing ChangeTypeClassifier from qualitative analysis

PERFORMANCE: Designed to handle large repositories efficiently with:
- Batch processing for ML predictions
- Intelligent caching of ML results
- Fallback to fast rule-based classification when appropriate
"""

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from ..qualitative.classifiers.change_type import ChangeTypeClassifier
from ..qualitative.classifiers.llm_commit_classifier import LLMCommitClassifier, LLMConfig
from ..qualitative.models.schemas import ChangeTypeConfig
from .ml_prediction_cache import MLPredictionCache  # noqa: F401
from .ml_tickets_analysis import MLTicketAnalysisMixin
from .tickets import TicketExtractor, filter_git_artifacts

# Import training model loader with fallback
try:
    from ..training.model_loader import TrainingModelLoader

    TRAINING_LOADER_AVAILABLE = True
except ImportError:
    TRAINING_LOADER_AVAILABLE = False

try:
    import spacy
    from spacy.tokens import Doc

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    Doc = Any

logger = logging.getLogger(__name__)


class MLTicketExtractor(MLTicketAnalysisMixin, TicketExtractor):
    """ML-enhanced ticket extractor with sophisticated commit categorization.

    This extractor extends the basic TicketExtractor with machine learning capabilities
    while maintaining full backward compatibility. It uses a hybrid approach combining
    rule-based patterns with ML-based semantic analysis for improved accuracy.

    Key features:
    - Hybrid categorization (ML + rule-based fallback)
    - Confidence scoring for all predictions
    - Intelligent caching for performance
    - Feature extraction from commit message and file patterns
    - Integration with existing qualitative analysis infrastructure
    """

    def __init__(
        self,
        allowed_platforms: Optional[list[str]] = None,
        untracked_file_threshold: int = 1,
        ml_config: Optional[dict[str, Any]] = None,
        llm_config: Optional[dict[str, Any]] = None,
        cache_dir: Optional[Path] = None,
        enable_ml: bool = True,
        enable_llm: bool = False,
    ) -> None:
        """Initialize ML-enhanced ticket extractor.

        Args:
            allowed_platforms: List of platforms to extract tickets from
            untracked_file_threshold: Minimum files changed for significant commits
            ml_config: Configuration for ML categorization (optional)
            llm_config: Configuration for LLM classification (optional)
            cache_dir: Directory for caching ML predictions
            enable_ml: Whether to enable ML features (fallback to rule-based if False)
            enable_llm: Whether to enable LLM classification (fallback to ML/rules if False)
        """
        # Initialize parent class
        super().__init__(allowed_platforms, untracked_file_threshold)

        self.enable_ml = enable_ml and SPACY_AVAILABLE
        self.enable_llm = enable_llm
        self.cache_dir = cache_dir or Path(".gitflow-cache")
        self.cache_dir.mkdir(exist_ok=True)

        # ML configuration with sensible defaults
        default_ml_config = {
            "min_confidence": 0.6,
            "semantic_weight": 0.7,
            "file_pattern_weight": 0.3,
            "hybrid_threshold": 0.5,  # Confidence threshold for using ML vs rule-based
            "cache_duration_days": 30,
            "batch_size": 100,
            "enable_caching": True,
        }

        self.ml_config = {**default_ml_config, **(ml_config or {})}

        # LLM configuration with sensible defaults
        default_llm_config = {
            "api_key": None,
            "model": "mistralai/mistral-7b-instruct",
            "confidence_threshold": 0.7,
            "max_tokens": 50,
            "temperature": 0.1,
            "timeout_seconds": 30.0,
            "cache_duration_days": 90,
            "enable_caching": True,
            "max_daily_requests": 1000,
            "domain_terms": {},
        }

        self.llm_config_dict = {**default_llm_config, **(llm_config or {})}

        # Initialize ML components
        self.change_type_classifier = None
        self.nlp_model = None
        self.ml_cache = None
        self.trained_model_loader = None
        self.llm_classifier = None

        if self.enable_ml:
            self._initialize_ml_components()

        # Initialize LLM classifier if enabled
        if self.enable_llm:
            self._initialize_llm_classifier()

        # Initialize trained model loader if available
        if TRAINING_LOADER_AVAILABLE and self.enable_ml:
            try:
                self.trained_model_loader = TrainingModelLoader(self.cache_dir)
                logger.info("Trained model loader initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize trained model loader: {e}")
                self.trained_model_loader = None

        logger.info(
            f"MLTicketExtractor initialized with ML {'enabled' if self.enable_ml else 'disabled'}, LLM {'enabled' if self.enable_llm else 'disabled'}"
        )

    def _initialize_ml_components(self) -> None:
        """Initialize ML components (ChangeTypeClassifier and spaCy model).

        WHY: Separate initialization allows for graceful degradation if ML components
        fail to load. The extractor will fall back to rule-based classification.
        """
        try:
            # Initialize ChangeTypeClassifier
            change_type_config = ChangeTypeConfig(
                min_confidence=self.ml_config["min_confidence"],
                semantic_weight=self.ml_config["semantic_weight"],
                file_pattern_weight=self.ml_config["file_pattern_weight"],
            )
            self.change_type_classifier = ChangeTypeClassifier(change_type_config)

            # Initialize spaCy model (try English first, then basic)
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("spaCy model 'en_core_web_sm' loaded successfully")
            except OSError:
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found. Trying alternative model..."
                )
                try:
                    self.nlp_model = spacy.load("en_core_web_md")
                    logger.info("spaCy model 'en_core_web_md' loaded successfully")
                except OSError:
                    logger.warning(
                        "No spaCy models found. ML categorization will gracefully fall back to rule-based classification. "
                        "To enable ML features, install a spaCy model: python -m spacy download en_core_web_sm"
                    )
                    self.nlp_model = None

            # Initialize ML cache
            if self.ml_config["enable_caching"]:
                self._initialize_ml_cache()

            logger.info("ML components initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize ML components: {e}")
            logger.info("Analysis will continue with rule-based classification only")
            self.enable_ml = False

    def _initialize_llm_classifier(self) -> None:
        """Initialize LLM classifier for commit categorization.

        WHY: LLM-based classification can provide more nuanced understanding
        of commit messages compared to rule-based or traditional ML approaches.
        This method handles graceful degradation if LLM setup fails.
        """
        try:
            # Create LLM configuration object
            llm_config = LLMConfig(
                api_key=self.llm_config_dict.get("api_key"),
                model=self.llm_config_dict.get("model", "mistralai/mistral-7b-instruct"),
                confidence_threshold=self.llm_config_dict.get("confidence_threshold", 0.7),
                max_tokens=self.llm_config_dict.get("max_tokens", 50),
                temperature=self.llm_config_dict.get("temperature", 0.1),
                timeout_seconds=self.llm_config_dict.get("timeout_seconds", 30.0),
                cache_duration_days=self.llm_config_dict.get("cache_duration_days", 90),
                enable_caching=self.llm_config_dict.get("enable_caching", True),
                max_daily_requests=self.llm_config_dict.get("max_daily_requests", 1000),
                domain_terms=self.llm_config_dict.get("domain_terms", {}),
            )

            # Initialize LLM classifier
            self.llm_classifier = LLMCommitClassifier(llm_config, self.cache_dir)
            logger.info(f"LLM classifier initialized with model: {llm_config.model}")

        except Exception as e:
            logger.warning(f"Failed to initialize LLM classifier: {e}")
            logger.info("Analysis will continue without LLM classification")
            self.enable_llm = False
            self.llm_classifier = None

    def _initialize_ml_cache(self) -> None:
        """Initialize SQLite cache for ML predictions.

        WHY: ML predictions can be expensive, so we cache results to improve performance
        on subsequent runs. The cache includes expiration and invalidation logic.
        """
        try:
            cache_path = self.cache_dir / "ml_predictions.db"
            self.ml_cache = MLPredictionCache(cache_path, self.ml_config["cache_duration_days"])
            logger.debug("ML prediction cache initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize ML cache: {e}")
            self.ml_cache = None

    def categorize_commit(self, message: str, files_changed: Optional[list[str]] = None) -> str:
        """Categorize a commit using LLM -> ML -> rule-based fallback approach.

        This method extends the parent's categorize_commit with LLM and ML capabilities
        while maintaining backward compatibility. It returns the same category strings as
        the parent class.

        Classification priority:
        1. LLM-based classification (if enabled and confident)
        2. ML-based classification (if enabled and confident)
        3. Rule-based classification (always available)

        Args:
            message: The commit message to categorize
            files_changed: Optional list of changed files for additional context

        Returns:
            String category (bug_fix, feature, refactor, documentation,
            maintenance, test, style, build, or other)
        """
        if not message:
            return "other"

        # Filter git artifacts for cleaner classification
        cleaned_message = filter_git_artifacts(message)
        if not cleaned_message:
            return "other"

        # Try LLM classification first if enabled
        if self.enable_llm and self.llm_classifier:
            llm_result = self._llm_categorize_commit(cleaned_message, files_changed or [])
            if llm_result and llm_result["confidence"] >= self.llm_config_dict.get(
                "confidence_threshold", 0.7
            ):
                # Map LLM categories to parent class categories
                mapped_category = self._map_llm_to_parent_category(llm_result["category"])
                return mapped_category

        # Fall back to ML categorization if enabled
        if self.enable_ml:
            ml_result = self._ml_categorize_commit(cleaned_message, files_changed or [])
            if ml_result and ml_result["confidence"] >= self.ml_config["hybrid_threshold"]:
                # Map ML categories to parent class categories
                mapped_category = self._map_ml_to_parent_category(ml_result["category"])
                return mapped_category

        # Final fallback to parent's rule-based categorization
        return super().categorize_commit(cleaned_message)

    def categorize_commit_with_confidence(
        self, message: str, files_changed: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Categorize commit with detailed confidence information.

        This is the main entry point for getting detailed categorization results
        including confidence scores, alternative predictions, and processing metadata.

        Args:
            message: The commit message to categorize
            files_changed: Optional list of changed files for additional context

        Returns:
            Dictionary with categorization results:
            {
                'category': str,
                'confidence': float,
                'method': str ('ml', 'rules', 'cached'),
                'alternatives': List[Dict],
                'features': Dict,
                'processing_time_ms': float
            }
        """
        start_time = time.time()

        if not message:
            return {
                "category": "other",
                "confidence": 1.0,
                "method": "default",
                "alternatives": [],
                "features": {},
                "processing_time_ms": 0.0,
            }

        # Filter git artifacts for cleaner classification
        cleaned_message = filter_git_artifacts(message)
        if not cleaned_message:
            return {
                "category": "other",
                "confidence": 0.3,
                "method": "filtered_empty",
                "alternatives": [],
                "features": {},
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

        files_changed = files_changed or []

        # Check cache first
        if self.ml_cache and self.ml_config["enable_caching"]:
            cached_result = self.ml_cache.get_prediction(cleaned_message, files_changed)
            if cached_result:
                cached_result["processing_time_ms"] = (time.time() - start_time) * 1000
                return cached_result

        # Try LLM categorization first if enabled
        if self.enable_llm and self.llm_classifier:
            llm_result = self._llm_categorize_commit_detailed(cleaned_message, files_changed)
            if llm_result and llm_result["confidence"] >= self.llm_config_dict.get(
                "confidence_threshold", 0.7
            ):
                # Map to parent categories and cache result
                llm_result["category"] = self._map_llm_to_parent_category(llm_result["category"])
                llm_result["processing_time_ms"] = (time.time() - start_time) * 1000

                if self.ml_cache and self.ml_config["enable_caching"]:
                    self.ml_cache.store_prediction(cleaned_message, files_changed, llm_result)

                return llm_result

        # Fall back to ML categorization
        if self.enable_ml:
            ml_result = self._ml_categorize_commit_detailed(cleaned_message, files_changed)
            if ml_result and ml_result["confidence"] >= self.ml_config["hybrid_threshold"]:
                # Map to parent categories and cache result
                ml_result["category"] = self._map_ml_to_parent_category(ml_result["category"])
                ml_result["processing_time_ms"] = (time.time() - start_time) * 1000

                if self.ml_cache and self.ml_config["enable_caching"]:
                    self.ml_cache.store_prediction(cleaned_message, files_changed, ml_result)

                return ml_result

        # Fall back to rule-based categorization
        rule_category = super().categorize_commit(cleaned_message)
        rule_result = {
            "category": rule_category,
            "confidence": 0.8 if rule_category != "other" else 0.3,
            "method": "rules",
            "alternatives": [],
            "features": {"rule_based": True},
            "processing_time_ms": (time.time() - start_time) * 1000,
        }

        if self.ml_cache and self.ml_config["enable_caching"]:
            self.ml_cache.store_prediction(message, files_changed, rule_result)

        return rule_result

    def _ml_categorize_commit(
        self, message: str, files_changed: list[str]
    ) -> Optional[dict[str, Any]]:
        """Internal ML categorization method (simplified version).

        Args:
            message: Commit message
            files_changed: List of changed files

        Returns:
            Dictionary with category and confidence, or None if ML unavailable
        """
        if not self.change_type_classifier or not message:
            return None

        try:
            # Process message with spaCy if available
            doc = None
            if self.nlp_model:
                doc = self.nlp_model(message)

            # Get ML classification
            ml_category, confidence = self.change_type_classifier.classify(
                message, doc, files_changed
            )

            if ml_category and ml_category != "unknown":
                return {"category": ml_category, "confidence": confidence}

        except Exception as e:
            logger.warning(f"ML categorization failed: {e}")

        return None

    def _ml_categorize_commit_detailed(
        self, message: str, files_changed: list[str]
    ) -> Optional[dict[str, Any]]:
        """Detailed ML categorization with comprehensive metadata.

        Tries trained models first, then falls back to built-in ML classification.

        Args:
            message: Commit message
            files_changed: List of changed files

        Returns:
            Detailed categorization result dictionary or None if ML unavailable
        """
        if not message:
            return None

        # Try trained model first if available
        if self.trained_model_loader:
            try:
                trained_result = self.trained_model_loader.predict_commit_category(
                    message, files_changed
                )
                if (
                    trained_result["method"] != "failed"
                    and trained_result["confidence"] >= self.ml_config["hybrid_threshold"]
                ):
                    return trained_result
            except Exception as e:
                logger.debug(f"Trained model prediction failed, falling back to built-in ML: {e}")

        # Fall back to built-in ML classification
        if not self.change_type_classifier:
            return None

        try:
            # Process message with spaCy
            doc = None
            features = {}
            if self.nlp_model:
                doc = self.nlp_model(message)
                features = self._extract_features(message, doc, files_changed)

            # Get ML classification
            ml_category, confidence = self.change_type_classifier.classify(
                message, doc, files_changed
            )

            if ml_category and ml_category != "unknown":
                return {
                    "category": ml_category,
                    "confidence": confidence,
                    "method": "builtin_ml",
                    "alternatives": self._get_alternative_predictions(message, doc, files_changed),
                    "features": features,
                }

        except Exception as e:
            logger.warning(f"Built-in ML categorization failed: {e}")

        return None

    def _extract_features(
        self, message: str, doc: Optional[Doc], files_changed: list[str]
    ) -> dict[str, Any]:
        """Extract features used for ML classification.

        Args:
            message: Commit message
            doc: spaCy processed document
            files_changed: List of changed files

        Returns:
            Dictionary of extracted features
        """
        features = {
            "message_length": len(message),
            "word_count": len(message.split()),
            "files_count": len(files_changed),
            "file_extensions": list(
                set(Path(f).suffix.lower() for f in files_changed if Path(f).suffix)
            ),
        }

        if doc:
            features.update(
                {
                    "has_verbs": any(token.pos_ == "VERB" for token in doc),
                    "has_entities": len(doc.ents) > 0,
                    "sentiment_polarity": 0.0,  # Placeholder - could add sentiment analysis
                }
            )

        return features

    def _get_alternative_predictions(
        self, message: str, doc: Optional[Doc], files_changed: list[str]
    ) -> list[dict[str, Any]]:
        """Get alternative predictions with lower confidence scores.

        This is a simplified version - in a full implementation, you would
        get all classification scores and return top N alternatives.

        Args:
            message: Commit message
            doc: spaCy processed document
            files_changed: List of changed files

        Returns:
            List of alternative predictions
        """
        # Placeholder implementation - could be enhanced to return actual alternatives
        alternatives = []

        # Add rule-based prediction as alternative
        rule_category = super().categorize_commit(message)
        if rule_category != "other":
            alternatives.append({"category": rule_category, "confidence": 0.6, "method": "rules"})

        return alternatives[:3]  # Top 3 alternatives

    def _llm_categorize_commit(
        self, message: str, files_changed: list[str]
    ) -> Optional[dict[str, Any]]:
        """Internal LLM categorization method (simplified version).

        Args:
            message: Cleaned commit message (git artifacts already filtered)
            files_changed: List of changed files

        Returns:
            Dictionary with category and confidence, or None if LLM unavailable
        """
        if not self.llm_classifier or not message:
            return None

        try:
            # Get LLM classification
            llm_result = self.llm_classifier.classify_commit(message, files_changed)

            if (
                llm_result
                and llm_result.get("category")
                and llm_result["category"] != "maintenance"
            ):
                return {"category": llm_result["category"], "confidence": llm_result["confidence"]}
            elif (
                llm_result
                and llm_result.get("category") == "maintenance"
                and llm_result["confidence"] >= 0.8
            ):
                # Accept maintenance category only if high confidence
                return {"category": llm_result["category"], "confidence": llm_result["confidence"]}

        except Exception as e:
            logger.warning(f"LLM categorization failed: {e}")

        return None

    def _llm_categorize_commit_detailed(
        self, message: str, files_changed: list[str]
    ) -> Optional[dict[str, Any]]:
        """Detailed LLM categorization with comprehensive metadata.

        Args:
            message: Cleaned commit message (git artifacts already filtered)
            files_changed: List of changed files

        Returns:
            Detailed categorization result dictionary or None if LLM unavailable
        """
        if not self.llm_classifier or not message:
            return None

        try:
            # Get detailed LLM classification
            llm_result = self.llm_classifier.classify_commit(message, files_changed)

            if llm_result and llm_result.get("category"):
                return {
                    "category": llm_result["category"],
                    "confidence": llm_result["confidence"],
                    "method": "llm",
                    "reasoning": llm_result.get("reasoning", "LLM-based classification"),
                    "model": llm_result.get("model", "unknown"),
                    "alternatives": llm_result.get("alternatives", []),
                    "features": {"llm_classification": True},
                }

        except Exception as e:
            logger.warning(f"Detailed LLM categorization failed: {e}")

        return None

    def _map_llm_to_parent_category(self, llm_category: str) -> str:
        """Map LLM categories to parent class categories.

        WHY: The LLM classifier uses streamlined 7-category system while the parent
        TicketExtractor uses different category names. This mapping ensures
        backward compatibility with existing reports and analysis.

        Args:
            llm_category: Category from LLM classifier

        Returns:
            Category compatible with parent class
        """
        # Map from LLM's 7 streamlined categories to parent categories
        mapping = {
            "feature": "feature",  # New functionality -> feature
            "bugfix": "bug_fix",  # Bug fixes -> bug_fix (parent uses underscore)
            "maintenance": "maintenance",  # Maintenance -> maintenance
            "integration": "build",  # Integration -> build (closest parent category)
            "content": "documentation",  # Content -> documentation
            "media": "other",  # Media -> other (no direct parent equivalent)
            "localization": "other",  # Localization -> other (no direct parent equivalent)
        }

        return mapping.get(llm_category, "other")

    def _map_ml_to_parent_category(self, ml_category: str) -> str:
        """Map ML categories to parent class categories.

        WHY: The ChangeTypeClassifier uses different category names than the parent
        TicketExtractor. This mapping ensures backward compatibility.

        Args:
            ml_category: Category from ML classifier

        Returns:
            Category compatible with parent class
        """
        mapping = {
            "feature": "feature",
            "bugfix": "bug_fix",
            "refactor": "refactor",
            "docs": "documentation",
            "test": "test",
            "chore": "maintenance",
            "security": "bug_fix",  # Security fixes are a type of bug fix
            "hotfix": "bug_fix",  # Hotfixes are urgent bug fixes
            "config": "maintenance",  # Configuration changes are maintenance
        }

        return mapping.get(ml_category, "other")

