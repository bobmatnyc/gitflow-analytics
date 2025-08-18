"""Base interface for LLM classifiers.

This module defines the abstract base class for all LLM-based classifiers,
establishing a consistent interface for different LLM providers.

WHY: Different LLM providers (OpenAI, Anthropic, OpenRouter, etc.) have different
APIs but should provide the same classification interface. This abstraction allows
easy switching between providers without changing the rest of the codebase.

DESIGN DECISIONS:
- Use ABC for enforcing interface implementation
- Define standard result format for all providers
- Include confidence scores and reasoning in results
- Support batch processing for efficiency
- Provide cost tracking interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ClassificationResult:
    """Standard result format for LLM classification.

    WHY: Consistent result format across all LLM providers ensures
    downstream code doesn't need provider-specific handling.
    """

    category: str
    confidence: float
    method: str  # 'llm', 'cached', 'rule_fallback', etc.
    reasoning: str
    model: str
    alternatives: list[dict[str, Any]]  # Alternative classifications with scores
    processing_time_ms: float
    batch_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        result = {
            "category": self.category,
            "confidence": self.confidence,
            "method": self.method,
            "reasoning": self.reasoning,
            "model": self.model,
            "alternatives": self.alternatives,
            "processing_time_ms": self.processing_time_ms,
        }
        if self.batch_id:
            result["batch_id"] = self.batch_id
        return result


@dataclass
class LLMProviderConfig:
    """Base configuration for LLM providers.

    WHY: Common configuration options that all providers need,
    with ability to extend for provider-specific settings.
    """

    api_key: Optional[str] = None
    model: str = "default"
    temperature: float = 0.1
    max_tokens: int = 50
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Rate limiting
    max_daily_requests: int = 1000
    max_requests_per_minute: int = 60

    # Cost tracking
    enable_cost_tracking: bool = True
    cost_warning_threshold: float = 10.0  # USD

    def validate(self) -> None:
        """Validate configuration settings.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(f"Temperature must be between 0 and 2, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")


class BaseLLMClassifier(ABC):
    """Abstract base class for LLM-based classifiers.

    WHY: Defines the interface that all LLM providers must implement,
    ensuring consistency and allowing provider switching.
    """

    def __init__(self, config: LLMProviderConfig, cache_dir: Optional[Path] = None):
        """Initialize LLM classifier.

        Args:
            config: Provider-specific configuration
            cache_dir: Directory for caching predictions
        """
        config.validate()
        self.config = config
        self.cache_dir = cache_dir or Path(".gitflow-cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Cost tracking
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.api_calls_made = 0

    @abstractmethod
    def classify_commit(
        self, message: str, files_changed: Optional[list[str]] = None
    ) -> ClassificationResult:
        """Classify a single commit message.

        Args:
            message: Commit message to classify
            files_changed: Optional list of changed files for context

        Returns:
            Classification result with category and metadata
        """
        pass

    @abstractmethod
    def classify_commits_batch(
        self, commits: list[dict[str, Any]], batch_id: Optional[str] = None
    ) -> list[ClassificationResult]:
        """Classify a batch of commits.

        Args:
            commits: List of commit dictionaries
            batch_id: Optional batch identifier for tracking

        Returns:
            List of classification results
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of the LLM provider.

        Returns:
            Provider name (e.g., 'openai', 'anthropic', 'openrouter')
        """
        pass

    @abstractmethod
    def estimate_cost(self, text: str) -> float:
        """Estimate the cost of classifying the given text.

        Args:
            text: Text to be classified

        Returns:
            Estimated cost in USD
        """
        pass

    def get_statistics(self) -> dict[str, Any]:
        """Get usage statistics for this classifier.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "provider": self.get_provider_name(),
            "model": self.config.model,
            "api_calls_made": self.api_calls_made,
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "average_tokens_per_call": (
                self.total_tokens_used / self.api_calls_made if self.api_calls_made > 0 else 0
            ),
            "cost_warning_threshold": self.config.cost_warning_threshold,
            "approaching_cost_limit": self.total_cost > self.config.cost_warning_threshold * 0.8,
        }

    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.api_calls_made = 0
