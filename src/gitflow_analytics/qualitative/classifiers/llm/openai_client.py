"""OpenAI and OpenRouter API client for LLM classification.

This module handles all OpenAI-compatible API interactions, including
OpenRouter which provides access to multiple models through a unified API.

WHY: Separating API interaction logic from classification logic makes the
system more maintainable and allows easy addition of new providers.

DESIGN DECISIONS:
- Support both OpenAI direct and OpenRouter endpoints
- Implement exponential backoff for retries
- Handle rate limiting gracefully
- Track token usage and costs accurately
- Support different pricing models
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

from .base import BaseLLMClassifier, ClassificationResult, LLMProviderConfig
from .cost_tracker import CostTracker, ModelPricing
from .prompts import PromptGenerator, PromptVersion
from .response_parser import ResponseParser

logger = logging.getLogger(__name__)


@dataclass
class OpenAIConfig(LLMProviderConfig):
    """Configuration specific to OpenAI/OpenRouter providers.

    WHY: OpenAI-compatible APIs have specific configuration needs
    beyond the base configuration.
    """

    api_base_url: str = "https://openrouter.ai/api/v1"  # Default to OpenRouter
    organization: Optional[str] = None  # OpenAI organization ID

    # OpenRouter specific
    site_url: str = "https://github.com/gitflow-analytics"
    app_name: str = "GitFlow Analytics"

    # Model selection
    use_openrouter: bool = True  # If False, use direct OpenAI API

    def validate(self) -> None:
        """Validate OpenAI-specific configuration."""
        super().validate()

        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required for OpenAI/OpenRouter")

        # API key is optional - classifier will gracefully degrade without it
        # if not self.api_key:
        #     raise ValueError("API key is required for OpenAI/OpenRouter")

        # Set appropriate base URL based on provider
        if not self.use_openrouter and "openrouter" in self.api_base_url:
            self.api_base_url = "https://api.openai.com/v1"


class OpenAIClassifier(BaseLLMClassifier):
    """OpenAI/OpenRouter-based commit classifier.

    WHY: OpenAI and OpenRouter provide high-quality language models
    for classification. This implementation supports both providers
    through their compatible APIs.
    """

    def __init__(
        self,
        config: OpenAIConfig,
        cache_dir: Optional[Path] = None,
        prompt_version: PromptVersion = PromptVersion.V3_CONTEXTUAL,
    ):
        """Initialize OpenAI classifier.

        Args:
            config: OpenAI-specific configuration
            cache_dir: Directory for caching predictions
            prompt_version: Version of prompts to use
        """
        super().__init__(config, cache_dir)
        self.config: OpenAIConfig = config

        # Initialize components
        self.prompt_generator = PromptGenerator(prompt_version)
        self.response_parser = ResponseParser()
        self.cost_tracker = CostTracker()

        # Set up model pricing
        self._setup_pricing()

        # Rate limiting state
        self._last_request_time = 0
        self._request_count = 0
        self._minute_start = time.time()

        logger.info(f"OpenAIClassifier initialized with model: {config.model}")

    def _setup_pricing(self) -> None:
        """Set up pricing information for the configured model.

        WHY: Accurate cost tracking helps users monitor and control
        their LLM usage expenses.
        """
        # Common model pricing (per 1M tokens)
        pricing_map = {
            "gpt-4": ModelPricing("gpt-4", 30.0, 60.0),
            "gpt-4-turbo": ModelPricing("gpt-4-turbo", 10.0, 30.0),
            "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", 0.5, 1.5),
            "mistralai/mistral-7b-instruct": ModelPricing("mistral-7b", 0.25, 0.25),
            "meta-llama/llama-2-70b-chat": ModelPricing("llama-2-70b", 0.7, 0.9),
            "anthropic/claude-2": ModelPricing("claude-2", 8.0, 24.0),
        }

        # Find matching pricing or use default
        model_lower = self.config.model.lower()
        for model_key, pricing in pricing_map.items():
            if model_key in model_lower:
                self.cost_tracker.set_model_pricing(pricing)
                return

        # Default pricing for unknown models
        self.cost_tracker.set_model_pricing(ModelPricing(self.config.model, 1.0, 1.0))

    def get_provider_name(self) -> str:
        """Get the name of the LLM provider."""
        if self.config.use_openrouter:
            return "openrouter"
        return "openai"

    def classify_commit(
        self, message: str, files_changed: Optional[list[str]] = None
    ) -> ClassificationResult:
        """Classify a single commit message.

        Args:
            message: Commit message to classify
            files_changed: Optional list of changed files

        Returns:
            Classification result
        """
        start_time = time.time()

        # Validate input
        if not message or not message.strip():
            return ClassificationResult(
                category="maintenance",
                confidence=0.3,
                method="empty_message",
                reasoning="Empty commit message",
                model="none",
                alternatives=[],
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Apply rate limiting
        self._apply_rate_limiting()

        # Generate prompt
        system_prompt, user_prompt = self.prompt_generator.generate_prompt(message, files_changed)

        # Make API request with retries
        for attempt in range(self.config.max_retries):
            try:
                response_text, tokens_used = self._make_api_request(system_prompt, user_prompt)

                # Parse response
                category, confidence, reasoning = self.response_parser.parse_response(
                    response_text, self.prompt_generator.CATEGORIES
                )

                # Track costs
                prompt_tokens = self._estimate_tokens(system_prompt + user_prompt)
                completion_tokens = tokens_used - prompt_tokens if tokens_used else 50
                cost = self.cost_tracker.track_usage(prompt_tokens, completion_tokens)

                # Update statistics
                self.total_tokens_used += (
                    tokens_used if tokens_used else prompt_tokens + completion_tokens
                )
                self.total_cost += cost
                self.api_calls_made += 1

                return ClassificationResult(
                    category=category,
                    confidence=confidence,
                    method="llm",
                    reasoning=reasoning,
                    model=self.config.model,
                    alternatives=[],
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            except Exception as e:
                logger.warning(f"API request attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_seconds * (2**attempt))
                else:
                    # Final attempt failed, return fallback
                    return ClassificationResult(
                        category="maintenance",
                        confidence=0.1,
                        method="llm_error",
                        reasoning=f"LLM classification failed: {str(e)}",
                        model="fallback",
                        alternatives=[],
                        processing_time_ms=(time.time() - start_time) * 1000,
                    )

        # Should never reach here
        return ClassificationResult(
            category="maintenance",
            confidence=0.1,
            method="llm_error",
            reasoning="Unexpected error in classification",
            model="fallback",
            alternatives=[],
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    def classify_commits_batch(
        self, commits: list[dict[str, Any]], batch_id: Optional[str] = None
    ) -> list[ClassificationResult]:
        """Classify a batch of commits.

        WHY: Batch processing can be more efficient for large numbers
        of commits, though this implementation processes them serially
        to respect rate limits.

        Args:
            commits: List of commit dictionaries
            batch_id: Optional batch identifier

        Returns:
            List of classification results
        """
        results = []

        for commit in commits:
            message = commit.get("message", "")
            files_changed = []

            # Extract files from commit data
            if "files_changed" in commit:
                fc = commit["files_changed"]
                if isinstance(fc, list):
                    files_changed = fc

            # Classify individual commit
            result = self.classify_commit(message, files_changed)

            # Add batch ID if provided
            if batch_id:
                result.batch_id = batch_id

            results.append(result)

        return results

    def _make_api_request(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        """Make API request to OpenAI/OpenRouter.

        Args:
            system_prompt: System prompt for the model
            user_prompt: User prompt with the classification task

        Returns:
            Tuple of (response_text, tokens_used)

        Raises:
            Exception: If API request fails
        """
        if not self.config.api_key:
            raise ValueError("API key not configured - cannot make LLM requests")

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        # Add OpenRouter-specific headers
        if self.config.use_openrouter:
            headers["HTTP-Referer"] = self.config.site_url
            headers["X-Title"] = self.config.app_name

        # Add OpenAI organization if specified
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization

        # Prepare request payload
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        # Make request with proper timeout handling
        url = f"{self.config.api_base_url}/chat/completions"

        # Log request details for debugging
        logger.debug(f"Making API request to {url} with model {self.config.model}")
        logger.debug(f"Timeout set to {self.config.timeout_seconds} seconds")

        try:
            # Use a more conservative timeout and handle both connection and read timeouts
            # connection timeout = 10s, read timeout = config timeout
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(10.0, self.config.timeout_seconds),  # (connection, read) timeouts
            )
        except requests.exceptions.Timeout as e:
            logger.error(f"API request timed out after {self.config.timeout_seconds}s: {e}")
            raise Exception(
                f"API request timed out after {self.config.timeout_seconds} seconds"
            ) from e
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error during API request: {e}")
            raise Exception(f"Connection error: Unable to reach API at {url}") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise Exception(f"Request failed: {str(e)}") from e

        # Check response
        if response.status_code != 200:
            error_msg = f"API request failed with status {response.status_code}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg += f": {error_data['error'].get('message', 'Unknown error')}"
            except Exception:
                error_msg += f": {response.text}"
            raise Exception(error_msg)

        # Parse response
        data = response.json()

        if "choices" not in data or not data["choices"]:
            raise Exception("No response choices in API response")

        response_text = data["choices"][0]["message"]["content"].strip()

        # Extract token usage if available
        tokens_used = 0
        if "usage" in data:
            tokens_used = data["usage"].get("total_tokens", 0)

        return response_text, tokens_used

    def _apply_rate_limiting(self) -> None:
        """Apply rate limiting to respect API limits.

        WHY: Prevents hitting API rate limits which would cause
        errors and potential account suspension.
        """
        current_time = time.time()

        # Check if we're in a new minute
        if current_time - self._minute_start >= 60:
            self._request_count = 0
            self._minute_start = current_time

        # If we've hit the per-minute limit, wait
        if self._request_count >= self.config.max_requests_per_minute:
            sleep_time = 60 - (current_time - self._minute_start)
            if sleep_time > 0:
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
                self._request_count = 0
                self._minute_start = time.time()

        # Increment request count
        self._request_count += 1
        self._last_request_time = time.time()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        WHY: Token estimation helps track costs even when the API
        doesn't return exact token counts.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token on average
        # This is a rough approximation; actual tokenization varies
        return len(text) // 4

    def estimate_cost(self, text: str) -> float:
        """Estimate the cost of classifying the given text.

        Args:
            text: Text to be classified

        Returns:
            Estimated cost in USD
        """
        # Estimate tokens for the full prompt
        system_prompt = "You are a commit classification expert."  # Simplified
        prompt_tokens = self._estimate_tokens(system_prompt + text) + 100  # Add buffer
        completion_tokens = self.config.max_tokens

        return self.cost_tracker.calculate_cost(prompt_tokens, completion_tokens)
