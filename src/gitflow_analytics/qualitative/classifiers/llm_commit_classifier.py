"""LLM-based commit classification orchestrator.

This module provides the main interface for LLM-based commit classification,
orchestrating the various components for a complete classification solution.

WHY: This refactored version separates concerns into focused modules while
maintaining backward compatibility with the existing interface.

DESIGN DECISIONS:
- Main orchestrator delegates to specialized components
- Maintains backward compatibility with existing code
- Supports multiple LLM providers through abstraction
- Provides enhanced rule-based fallback
- Comprehensive error handling and graceful degradation
"""

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .llm.batch_processor import BatchConfig, BatchProcessor
from .llm.cache import LLMCache
from .llm.openai_client import OpenAIClassifier, OpenAIConfig
from .llm.prompts import PromptVersion

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM-based commit classification.

    Maintains backward compatibility with existing configuration structure.
    """

    # OpenRouter API configuration
    api_key: Optional[str] = None
    api_base_url: str = "https://openrouter.ai/api/v1"
    model: str = "mistralai/mistral-7b-instruct"  # Fast, affordable model

    # Classification parameters
    confidence_threshold: float = 0.7  # Minimum confidence for LLM predictions
    max_tokens: int = 50  # Keep responses short
    temperature: float = 0.1  # Low temperature for consistent results
    timeout_seconds: float = 5.0  # API timeout - reduced to fail fast on unresponsive APIs

    # Caching configuration
    cache_duration_days: int = 90  # Long cache duration for cost optimization
    enable_caching: bool = True

    # Cost optimization
    batch_size: int = 1  # Process one at a time for simplicity
    max_daily_requests: int = 1000  # Rate limiting
    max_retries: int = 1  # Reduce retries to fail faster on unresponsive APIs

    # Domain-specific terms for organization
    domain_terms: dict[str, list[str]] = None

    def __post_init__(self):
        """Initialize default domain terms if not provided."""
        if self.domain_terms is None:
            self.domain_terms = {
                "media": [
                    "video",
                    "audio",
                    "streaming",
                    "player",
                    "media",
                    "content",
                    "broadcast",
                    "live",
                    "recording",
                    "episode",
                    "program",
                ],
                "localization": [
                    "translation",
                    "i18n",
                    "l10n",
                    "locale",
                    "language",
                    "spanish",
                    "french",
                    "german",
                    "italian",
                    "portuguese",
                    "multilingual",
                ],
                "integration": [
                    "api",
                    "webhook",
                    "third-party",
                    "external",
                    "service",
                    "integration",
                    "sync",
                    "import",
                    "export",
                    "connector",
                ],
            }


class LLMCommitClassifier:
    """LLM-based commit classifier with modular architecture.

    This refactored version delegates to specialized components for better
    maintainability while preserving the original interface.
    """

    # Streamlined category definitions (same as original)
    CATEGORIES = {
        "feature": "New functionality, capabilities, enhancements, additions",
        "bugfix": "Fixes, errors, issues, crashes, bugs, corrections",
        "maintenance": "Configuration, chores, dependencies, cleanup, refactoring, updates",
        "integration": "Third-party services, APIs, webhooks, external systems",
        "content": "Text, copy, documentation, README updates, comments",
        "media": "Video, audio, streaming, players, visual assets, images",
        "localization": "Translations, i18n, l10n, regional adaptations",
    }

    def __init__(self, config: LLMConfig, cache_dir: Optional[Path] = None):
        """Initialize LLM commit classifier with modular components.

        Args:
            config: LLM configuration
            cache_dir: Directory for caching predictions
        """
        self.config = config
        self.cache_dir = cache_dir or Path(".gitflow-cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize components
        self._init_classifier()
        self._init_cache()
        self._init_batch_processor()
        self._init_rule_patterns()

        # Request tracking for rate limiting (backward compatibility)
        self._daily_requests = 0
        self._last_reset_date = None

        # Cost tracking (backward compatibility)
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.api_calls_made = 0

        logger.info(f"LLMCommitClassifier initialized with model: {self.config.model}")

    def _init_classifier(self) -> None:
        """Initialize the LLM classifier component.

        WHY: Modular initialization allows easy switching between providers.
        """
        # Convert config to OpenAI config
        openai_config = OpenAIConfig(
            api_key=self.config.api_key,
            api_base_url=self.config.api_base_url,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout_seconds=self.config.timeout_seconds,
            max_daily_requests=self.config.max_daily_requests,
            max_retries=getattr(self.config, "max_retries", 2),  # Use config or default to 2
            use_openrouter=True,  # Default to OpenRouter
        )

        # Initialize classifier
        try:
            self.classifier = OpenAIClassifier(
                config=openai_config,
                cache_dir=self.cache_dir,
                prompt_version=PromptVersion.V3_CONTEXTUAL,
            )

            # Set domain terms in prompt generator
            self.classifier.prompt_generator.domain_terms = self.config.domain_terms

        except ImportError as e:
            logger.warning(f"Failed to initialize LLM classifier: {e}")
            self.classifier = None

    def _init_cache(self) -> None:
        """Initialize the caching component.

        WHY: Separate cache initialization for better error handling.
        """
        self.cache: Optional[LLMCache] = None
        if self.config.enable_caching:
            try:
                cache_path = self.cache_dir / "llm_predictions.db"
                self.cache = LLMCache(
                    cache_path=cache_path, expiration_days=self.config.cache_duration_days
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LLM cache: {e}")
                self.cache = None

    def _init_batch_processor(self) -> None:
        """Initialize the batch processing component.

        WHY: Batch processing improves efficiency for large-scale classification.
        """
        batch_config = BatchConfig(
            batch_size=self.config.batch_size, show_progress=True, continue_on_batch_failure=True
        )
        self.batch_processor = BatchProcessor(batch_config)

    def _init_rule_patterns(self) -> None:
        """Initialize rule-based patterns for fallback classification.

        WHY: Rule-based fallback ensures classification works even
        when LLM is unavailable.
        """
        self.rule_patterns = {
            "feature": [
                r"^(feat|feature)[\(\:]",
                r"^add[\(\:]",
                r"^implement[\(\:]",
                r"^create[\(\:]",
                r"add.*feature",
                r"implement.*feature",
                r"create.*feature",
                r"new.*feature",
                r"introduce.*feature",
                r"^enhancement[\(\:]",
            ],
            "bugfix": [
                r"^(fix|bug|hotfix|patch)[\(\:]",
                r"fix.*bug(?!.*format)",
                r"fix.*issue(?!.*format)",
                r"resolve.*bug",
                r"correct.*bug",
                r"repair.*",
                r"^hotfix[\(\:]",
                r"patch.*bug",
                r"debug.*",
            ],
            "maintenance": [
                r"^(chore|refactor|style|deps|build|ci|test)[\(\:]",
                r"^update[\(\:]",
                r"^bump[\(\:]",
                r"^upgrade[\(\:]",
                r"refactor.*",
                r"cleanup",
                r"update.*depend",
                r"bump.*version",
                r"configure.*",
                r"maintenance",
                r"organize.*",
                r"format.*",
                r"style.*",
                r"lint.*",
                r"improve.*performance",
                r"optimize.*",
            ],
            "content": [
                r"^(docs|doc|readme)[\(\:]",
                r"update.*readme",
                r"documentation",
                r"^comment[\(\:]",
                r"doc.*update",
                r"add.*comment",
                r"update.*doc",
                r"add.*documentation",
            ],
        }

    def classify_commit(
        self, message: str, files_changed: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Classify a commit message using LLM or fallback methods.

        Args:
            message: Cleaned commit message
            files_changed: Optional list of changed files

        Returns:
            Classification result dictionary (backward compatible format)
        """
        start_time = time.time()

        # Check for empty message
        if not message or not message.strip():
            return self._create_result("maintenance", 0.3, "empty_message", start_time)

        # Try cache first
        if self.cache:
            cached_result = self.cache.get(message, files_changed)
            if cached_result:
                cached_result["processing_time_ms"] = (time.time() - start_time) * 1000
                return cached_result

        # Try LLM classification if available and configured
        if self.classifier and self.config.api_key:
            try:
                # Check rate limits
                if self._check_rate_limits():
                    result = self.classifier.classify_commit(message, files_changed)

                    # Check if LLM actually succeeded
                    if result.method == "llm":
                        # Update statistics for backward compatibility
                        self.api_calls_made += 1
                        self._daily_requests += 1

                        # Get cost information from classifier
                        stats = self.classifier.get_statistics()
                        self.total_tokens_used = stats.get("total_tokens_used", 0)
                        self.total_cost = stats.get("total_cost", 0.0)

                        # Convert to backward compatible format
                        result_dict = result.to_dict()

                        # Cache successful result
                        if self.cache:
                            self.cache.store(message, files_changed, result_dict)

                        return result_dict
                    # If method is not 'llm', fall through to rule-based
                else:
                    logger.debug("Rate limit exceeded, using rule-based fallback")
            except Exception as e:
                logger.debug(f"LLM classification not available: {e}")

        # Fall back to enhanced rule-based classification
        return self._enhanced_rule_based_classification(message, files_changed or [])

    def classify_commits_batch(
        self,
        commits: list[dict[str, Any]],
        batch_id: Optional[str] = None,
        include_confidence: bool = True,
    ) -> list[dict[str, Any]]:
        """Classify a batch of commits.

        Args:
            commits: List of commit dictionaries
            batch_id: Optional batch identifier
            include_confidence: Whether to include confidence scores

        Returns:
            List of classification results (backward compatible format)
        """

        def classify_func(commit: dict[str, Any]) -> dict[str, Any]:
            """Classification function for batch processor."""
            message = commit.get("message", "")
            files_changed = []

            # Extract files from commit
            if "files_changed" in commit:
                fc = commit["files_changed"]
                if isinstance(fc, list):
                    files_changed = fc

            return self.classify_commit(message, files_changed)

        # Use batch processor
        results = self.batch_processor.process_commits(
            commits, classify_func, f"Classifying {len(commits)} commits"
        )

        # Add batch_id if provided
        if batch_id:
            for result in results:
                result["batch_id"] = batch_id

        logger.info(f"Batch {batch_id}: Classified {len(results)} commits")
        return results

    def _enhanced_rule_based_classification(
        self, message: str, files_changed: list[str]
    ) -> dict[str, Any]:
        """Enhanced rule-based classification as fallback.

        Args:
            message: Commit message
            files_changed: List of changed files

        Returns:
            Classification result dictionary
        """
        message_lower = message.lower()

        # Check style/formatting first
        if re.search(r"^(style|format)[\(\:]", message_lower):
            return self._create_result(
                "maintenance", 0.8, "rule_enhanced", 0.0, "Style/formatting commit"
            )

        # Check other patterns
        for category, patterns in self.rule_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return self._create_result(
                        category, 0.8, "rule_enhanced", 0.0, f"Matched pattern: {pattern}"
                    )

        # File-based analysis
        if files_changed:
            category = self._analyze_files(files_changed)
            if category:
                return self._create_result(
                    category, 0.7, "rule_enhanced", 0.0, "File-based classification"
                )

        # Semantic analysis
        category = self._semantic_analysis(message_lower)
        if category:
            return self._create_result(
                category, 0.6, "rule_enhanced", 0.0, f"Semantic indicator for {category}"
            )

        # Default fallback
        if len(message.split()) >= 5:
            return self._create_result(
                "feature", 0.4, "rule_enhanced", 0.0, "Detailed commit suggests feature"
            )
        elif any(term in message_lower for term in ["urgent", "critical", "!"]):
            return self._create_result(
                "bugfix", 0.5, "rule_enhanced", 0.0, "Urgent language suggests bug fix"
            )
        else:
            return self._create_result(
                "maintenance", 0.3, "rule_enhanced", 0.0, "General maintenance work"
            )

    def _analyze_files(self, files_changed: list[str]) -> Optional[str]:
        """Analyze files to determine category.

        Args:
            files_changed: List of changed files

        Returns:
            Category or None
        """
        file_patterns = []

        for file_path in files_changed:
            file_lower = file_path.lower()
            ext = Path(file_path).suffix.lower()

            if any(term in file_lower for term in ["readme", "doc", "changelog", ".md"]):
                file_patterns.append("documentation")
            elif any(term in file_lower for term in ["test", "spec", "__test__"]):
                file_patterns.append("test")
            elif any(term in file_lower for term in ["config", "package.json", ".yml"]):
                file_patterns.append("configuration")
            elif ext in [".jpg", ".png", ".gif", ".mp4", ".mp3", ".svg"]:
                file_patterns.append("media")

        # Determine category from patterns
        if "documentation" in file_patterns:
            return "content"
        elif "test" in file_patterns or "configuration" in file_patterns:
            return "maintenance"
        elif "media" in file_patterns:
            return "media"

        return None

    def _semantic_analysis(self, message_lower: str) -> Optional[str]:
        """Perform semantic analysis on message.

        Args:
            message_lower: Lowercase commit message

        Returns:
            Category or None
        """
        semantic_indicators = {
            "feature": ["implement new", "create new", "introduce new", "develop", "build new"],
            "bugfix": [
                "resolve error",
                "correct issue",
                "repair bug",
                "solve problem",
                "address bug",
            ],
            "maintenance": [
                "update config",
                "upgrade",
                "modify existing",
                "change setting",
                "improve performance",
            ],
            "content": ["document", "explain", "describe", "clarify", "write documentation"],
        }

        for category, indicators in semantic_indicators.items():
            if any(indicator in message_lower for indicator in indicators):
                return category

        return None

    def _check_rate_limits(self) -> bool:
        """Check if we're within daily rate limits.

        Returns:
            True if request is allowed
        """
        from datetime import datetime

        current_date = datetime.now().date()

        # Reset counter if new day
        if current_date != self._last_reset_date:
            self._daily_requests = 0
            self._last_reset_date = current_date

        return self._daily_requests < self.config.max_daily_requests

    def _create_result(
        self,
        category: str,
        confidence: float,
        method: str,
        start_time: float,
        reasoning: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a standardized result dictionary.

        Args:
            category: Classification category
            confidence: Confidence score
            method: Classification method
            start_time: Processing start time
            reasoning: Optional reasoning text

        Returns:
            Result dictionary (backward compatible format)
        """
        return {
            "category": category,
            "confidence": confidence,
            "method": method,
            "reasoning": reasoning or f"Classified using {method}",
            "model": self.config.model if method == "llm" else "rule-based",
            "alternatives": [],
            "processing_time_ms": (time.time() - start_time) * 1000,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get classifier usage statistics.

        Returns:
            Dictionary with usage statistics (backward compatible)
        """
        stats = {
            "daily_requests": self._daily_requests,
            "max_daily_requests": self.config.max_daily_requests,
            "model": self.config.model,
            "cache_enabled": self.config.enable_caching,
            "api_configured": bool(self.config.api_key),
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "api_calls_made": self.api_calls_made,
            "average_tokens_per_call": (
                self.total_tokens_used / self.api_calls_made if self.api_calls_made > 0 else 0
            ),
        }

        # Add cache statistics
        if self.cache:
            stats["cache_statistics"] = self.cache.get_statistics()

        # Add batch processor statistics
        if self.batch_processor:
            stats["batch_statistics"] = self.batch_processor.get_statistics()

        # Add classifier statistics if available
        if self.classifier:
            stats["classifier_statistics"] = self.classifier.get_statistics()

        return stats


# Legacy class for backward compatibility
class LLMPredictionCache:
    """Legacy cache class for backward compatibility.

    This wraps the new LLMCache to maintain the old interface.
    """

    def __init__(self, cache_path: Path, expiration_days: int = 90):
        """Initialize legacy cache wrapper."""
        self.cache = LLMCache(cache_path, expiration_days)

    def get_prediction(self, message: str, files_changed: list[str]) -> Optional[dict[str, Any]]:
        """Get cached prediction (legacy interface)."""
        return self.cache.get(message, files_changed)

    def store_prediction(
        self, message: str, files_changed: list[str], result: dict[str, Any]
    ) -> None:
        """Store prediction (legacy interface)."""
        self.cache.store(message, files_changed, result)

    def cleanup_expired(self) -> int:
        """Remove expired predictions (legacy interface)."""
        return self.cache.cleanup_expired()

    def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics (legacy interface)."""
        return self.cache.get_statistics()
