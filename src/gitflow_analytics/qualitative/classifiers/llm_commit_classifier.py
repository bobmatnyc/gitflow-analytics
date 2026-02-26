"""LLM-based commit classification orchestrator.

This module provides the main interface for LLM-based commit classification,
orchestrating the various components for a complete classification solution.

WHY: This refactored version separates concerns into focused modules while
maintaining backward compatibility with the existing interface.

DESIGN DECISIONS:
- Main orchestrator delegates to specialized components
- Maintains backward compatibility with existing code
- Supports multiple LLM providers (OpenRouter, AWS Bedrock) through abstraction
- Provider auto-detection checks AWS credentials first, then OpenRouter API key
- Provides enhanced rule-based fallback when no LLM provider is available
- Comprehensive error handling and graceful degradation
"""

import logging
import os
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

    Maintains backward compatibility with existing configuration structure
    while adding support for multiple LLM providers.
    """

    # Provider selection: "auto", "openrouter", "bedrock"
    # "auto" will pick Bedrock when AWS creds are available, else OpenRouter.
    provider: str = "auto"

    # OpenRouter API configuration
    api_key: Optional[str] = None
    api_base_url: str = "https://openrouter.ai/api/v1"
    model: str = "mistralai/mistral-7b-instruct"  # Fast, affordable model

    # AWS Bedrock configuration
    aws_region: Optional[str] = None  # Defaults to AWS_REGION env or us-east-1
    aws_profile: Optional[str] = None  # Named AWS profile (None = default chain)
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"  # Fast, cheap

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


def _detect_aws_credentials() -> bool:
    """Check whether AWS credentials are available for Bedrock.

    WHY: We probe credentials before attempting to initialise the Bedrock
    client so we can emit a clear log message about which provider was chosen
    and avoid confusing ImportError / NoCredentialsError messages later.

    The check mirrors the boto3 credential chain order:
      1. AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY environment variables
      2. boto3 Session credential resolution (profiles, instance metadata, etc.)

    Returns:
        True if at least one credential source was found.
    """
    # Fast path: explicit env-var credentials
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return True

    # Slow path: ask boto3 (covers profiles, ECS task roles, EC2 instance metadata)
    try:
        import boto3  # type: ignore[import]

        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials is not None:
            # Resolve refreshable credentials; returns None if they've expired
            resolved = credentials.get_frozen_credentials()
            return resolved is not None and bool(resolved.access_key)
    except Exception:  # noqa: B110 — intentional: boto3 may raise various errors
        return False

    return False


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
        The provider selection follows this priority when set to "auto":
          1. AWS Bedrock (if credentials are available via boto3 / env vars)
          2. OpenRouter (if OPENROUTER_API_KEY or api_key is configured)
          3. None — falls back to rule-based classification
        """
        provider = getattr(self.config, "provider", "auto")

        # Resolve the effective provider when set to auto
        if provider == "auto":
            if _detect_aws_credentials():
                provider = "bedrock"
                logger.info("LLM provider auto-detected: AWS Bedrock (credentials found)")
            elif self.config.api_key or os.environ.get("OPENROUTER_API_KEY"):
                provider = "openrouter"
                logger.info("LLM provider auto-detected: OpenRouter (API key found)")
            else:
                logger.info(
                    "LLM provider auto-detected: none (no AWS credentials or OpenRouter key)"
                )
                self.classifier = None
                return

        if provider == "bedrock":
            self._init_bedrock_classifier()
        else:
            self._init_openrouter_classifier()

    def _init_openrouter_classifier(self) -> None:
        """Initialise the OpenRouter/OpenAI-compatible classifier."""
        openai_config = OpenAIConfig(
            api_key=self.config.api_key,
            api_base_url=self.config.api_base_url,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout_seconds=self.config.timeout_seconds,
            max_daily_requests=self.config.max_daily_requests,
            max_retries=getattr(self.config, "max_retries", 2),
            use_openrouter=True,
        )

        try:
            self.classifier = OpenAIClassifier(
                config=openai_config,
                cache_dir=self.cache_dir,
                prompt_version=PromptVersion.V3_CONTEXTUAL,
            )
            self.classifier.prompt_generator.domain_terms = self.config.domain_terms
            logger.info("OpenRouter classifier initialised with model: %s", self.config.model)
        except ImportError as exc:
            logger.warning("Failed to initialise OpenRouter classifier: %s", exc)
            self.classifier = None

    def _init_bedrock_classifier(self) -> None:
        """Initialise the AWS Bedrock classifier.

        WHY: Bedrock uses IAM credentials rather than an API key.
        The effective model displayed in stats/reports is updated to the
        Bedrock model ID so users can see which model was actually used.
        """
        from .llm.bedrock_client import BedrockClassifier, BedrockConfig

        bedrock_model_id = getattr(
            self.config, "bedrock_model_id", "anthropic.claude-3-haiku-20240307-v1:0"
        )
        aws_region = (
            getattr(self.config, "aws_region", None)
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        aws_profile = getattr(self.config, "aws_profile", None)

        bedrock_config = BedrockConfig(
            model=bedrock_model_id,
            bedrock_model_id=bedrock_model_id,
            aws_region=aws_region,
            aws_profile=aws_profile,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout_seconds=self.config.timeout_seconds,
            max_daily_requests=self.config.max_daily_requests,
            max_retries=getattr(self.config, "max_retries", 2),
        )

        try:
            self.classifier = BedrockClassifier(
                config=bedrock_config,
                cache_dir=self.cache_dir,
                prompt_version=PromptVersion.V3_CONTEXTUAL,
            )
            self.classifier.prompt_generator.domain_terms = self.config.domain_terms

            # Update the model field so stats and reports show the Bedrock model ID
            self.config.model = bedrock_model_id

            logger.info(
                "Bedrock classifier initialised: model=%s region=%s",
                bedrock_model_id,
                aws_region,
            )
        except (ImportError, Exception) as exc:
            logger.warning("Failed to initialise Bedrock classifier: %s", exc)
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

        # Try LLM classification if available and configured.
        # For Bedrock the classifier is present but api_key is empty — check
        # classifier presence alone to support both providers.
        classifier_ready = self.classifier is not None and (
            self.config.api_key
            or (
                hasattr(self.classifier, "get_provider_name")
                and self.classifier.get_provider_name() == "bedrock"
            )
        )
        if classifier_ready:
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

        WHY: When the underlying classifier supports native batching (e.g.
        BedrockClassifier sends N commits in one API call and gets a JSON
        array back), we delegate directly to it for dramatically fewer API
        round-trips.  Otherwise we fall back to the one-by-one batch
        processor.

        Args:
            commits: List of commit dictionaries
            batch_id: Optional batch identifier
            include_confidence: Whether to include confidence scores

        Returns:
            List of classification results (backward compatible format)
        """
        # Fast path: delegate to the classifier's native batch method if the
        # provider supports multi-commit-per-call batching (currently Bedrock).
        classifier_ready = self.classifier is not None and (
            self.config.api_key
            or (
                hasattr(self.classifier, "get_provider_name")
                and self.classifier.get_provider_name() == "bedrock"
            )
        )

        if classifier_ready and hasattr(self.classifier, "classify_commits_batch"):
            try:
                batch_results = self.classifier.classify_commits_batch(commits, batch_id=batch_id)

                # Convert ClassificationResult objects to backward-compatible dicts
                results: list[dict[str, Any]] = []
                for cr in batch_results:
                    result_dict = cr.to_dict()

                    # Cache each successful LLM result
                    if self.cache and cr.method in ("llm", "llm_batch"):
                        msg = ""
                        fc: list[str] = []
                        # Find the original commit to get the message for caching
                        idx = len(results)
                        if idx < len(commits):
                            msg = commits[idx].get("message", "")
                            raw_fc = commits[idx].get("files_changed")
                            fc = raw_fc if isinstance(raw_fc, list) else []
                        if msg:
                            self.cache.store(msg, fc, result_dict)

                    results.append(result_dict)

                # Sync cost stats from underlying classifier
                stats = self.classifier.get_statistics()
                self.total_tokens_used = stats.get("total_tokens_used", 0)
                self.total_cost = stats.get("total_cost", 0.0)
                self.api_calls_made = stats.get("api_calls_made", 0)

                logger.info(f"Batch {batch_id}: Classified {len(results)} commits via native batch")
                return results

            except Exception as e:
                logger.warning(
                    f"Native batch classification failed ({e}), falling back to one-by-one"
                )

        # Fallback: one-by-one classification via batch processor
        def classify_func(commit: dict[str, Any]) -> dict[str, Any]:
            """Classification function for batch processor."""
            message = commit.get("message", "")
            files_changed = []
            if "files_changed" in commit:
                fc_val = commit["files_changed"]
                if isinstance(fc_val, list):
                    files_changed = fc_val
            return self.classify_commit(message, files_changed)

        results = self.batch_processor.process_commits(
            commits, classify_func, f"Classifying {len(commits)} commits"
        )

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
            "provider": getattr(self.config, "provider", "auto"),
            "api_configured": bool(self.config.api_key)
            or (
                self.classifier is not None
                and hasattr(self.classifier, "get_provider_name")
                and self.classifier.get_provider_name() == "bedrock"
            ),
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
