"""AWS Bedrock client for LLM-based commit classification.

This module implements commit classification using AWS Bedrock's managed model
hosting service, providing access to Claude models via AWS IAM credentials.

WHY: AWS Bedrock is the preferred LLM provider for teams already operating
in AWS. IAM-based authentication eliminates the need for separate API keys,
billing is consolidated into the AWS account, and VPC endpoints can be used
to keep traffic off the public internet.

DESIGN DECISIONS:
- Use the bedrock-runtime client (not bedrock) for inference calls.
- Claude on Bedrock requires the 'anthropic_version' field and uses the
  Messages API format wrapped in a plain JSON body.
- boto3 is treated as an optional dependency; an ImportError is raised at
  instantiation time (not import time) so the rest of the codebase can
  import this module safely without boto3 installed.
- Reuse PromptGenerator and ResponseParser from the shared llm module so
  prompt logic stays in one place.
- CostTracker is initialised with Claude 3 Haiku pricing (the default
  Bedrock model): $0.25/1M input, $1.25/1M output.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .base import BaseLLMClassifier, ClassificationResult, LLMProviderConfig
from .cost_tracker import CostTracker, ModelPricing
from .prompts import PromptGenerator, PromptVersion
from .response_parser import ResponseParser

logger = logging.getLogger(__name__)

# Claude 3 Haiku pricing on Bedrock (USD per 1M tokens, as of 2024)
_BEDROCK_HAIKU_PRICING = ModelPricing(
    model_name="anthropic.claude-3-haiku-20240307-v1:0",
    input_cost_per_million=0.25,
    output_cost_per_million=1.25,
)

# Claude 3 Sonnet pricing on Bedrock
_BEDROCK_SONNET_PRICING = ModelPricing(
    model_name="anthropic.claude-3-sonnet-20240229-v1:0",
    input_cost_per_million=3.0,
    output_cost_per_million=15.0,
)

# Claude 3 Opus pricing on Bedrock
_BEDROCK_OPUS_PRICING = ModelPricing(
    model_name="anthropic.claude-3-opus-20240229-v1:0",
    input_cost_per_million=15.0,
    output_cost_per_million=75.0,
)

_BEDROCK_PRICING_MAP: dict[str, ModelPricing] = {
    "haiku": _BEDROCK_HAIKU_PRICING,
    "sonnet": _BEDROCK_SONNET_PRICING,
    "opus": _BEDROCK_OPUS_PRICING,
}


@dataclass
class BedrockConfig(LLMProviderConfig):
    """Configuration specific to the AWS Bedrock provider.

    WHY: Bedrock uses IAM credentials rather than an API key and needs
    region and profile settings that OpenAI-compatible providers don't.
    """

    # AWS configuration
    aws_region: str = "us-east-1"
    aws_profile: Optional[str] = None  # None = use the default credential chain

    # The Bedrock model identifier (not the friendly model name used by OpenRouter)
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"

    def validate(self) -> None:
        """Validate Bedrock-specific configuration."""
        super().validate()
        if not self.aws_region:
            raise ValueError("aws_region must be set for Bedrock provider")
        if not self.bedrock_model_id:
            raise ValueError("bedrock_model_id must be set for Bedrock provider")


class BedrockClassifier(BaseLLMClassifier):
    """AWS Bedrock-based commit classifier.

    Uses the Bedrock Runtime API to call Claude models. Authentication is
    handled entirely by boto3's credential chain (env vars → ~/.aws/credentials
    → IAM role), so no API key configuration is required.
    """

    def __init__(
        self,
        config: BedrockConfig,
        cache_dir: Optional[Path] = None,
        prompt_version: PromptVersion = PromptVersion.V3_CONTEXTUAL,
    ):
        """Initialize Bedrock classifier.

        Args:
            config: Bedrock-specific configuration
            cache_dir: Directory for caching predictions
            prompt_version: Version of prompts to use

        Raises:
            ImportError: If boto3 is not installed
        """
        super().__init__(config, cache_dir)
        self.config: BedrockConfig = config

        # Validate boto3 availability at init time so callers get a clear error
        try:
            import boto3  # noqa: F401 — existence check only
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for AWS Bedrock support. " "Install it with: pip install boto3"
            ) from exc

        # Build the boto3 session with optional named profile
        import boto3  # type: ignore[import]
        from botocore.config import Config as BotoConfig

        if config.aws_profile:
            session = boto3.Session(
                profile_name=config.aws_profile,
                region_name=config.aws_region,
            )
        else:
            session = boto3.Session(region_name=config.aws_region)

        # WHY: Set explicit read timeout to match configured timeout_seconds.
        # boto3 defaults are very long (60s connect, 60s read) and can cause
        # the pipeline to appear stuck on slow API responses.
        boto_config = BotoConfig(
            read_timeout=int(config.timeout_seconds) + 5,
            connect_timeout=10,
            retries={"max_attempts": 0},  # We handle retries ourselves
        )
        self._client = session.client("bedrock-runtime", config=boto_config)

        # Shared classification components
        self.prompt_generator = PromptGenerator(prompt_version)
        self.response_parser = ResponseParser()
        self.cost_tracker = CostTracker()
        self._setup_pricing()

        # Rate limiting counters (mirrors OpenAIClassifier interface)
        self._last_request_time = 0.0
        self._request_count = 0
        self._minute_start = time.time()

        logger.info(
            "BedrockClassifier initialised: model=%s region=%s profile=%s",
            config.bedrock_model_id,
            config.aws_region,
            config.aws_profile or "default",
        )

    # ------------------------------------------------------------------
    # Provider name / cost interface
    # ------------------------------------------------------------------

    def get_provider_name(self) -> str:
        """Return the provider identifier string."""
        return "bedrock"

    def estimate_cost(self, text: str) -> float:
        """Estimate the cost of classifying the given text.

        Args:
            text: Text to be classified

        Returns:
            Estimated cost in USD
        """
        # Rough estimate: simplified system prompt + text, plus buffer
        prompt_tokens = self._estimate_tokens("classification system prompt" + text) + 100
        completion_tokens = self.config.max_tokens
        return self.cost_tracker.calculate_cost(prompt_tokens, completion_tokens)

    # ------------------------------------------------------------------
    # Classification interface
    # ------------------------------------------------------------------

    def classify_commit(
        self, message: str, files_changed: Optional[list[str]] = None
    ) -> ClassificationResult:
        """Classify a single commit message via Bedrock.

        Args:
            message: Commit message to classify
            files_changed: Optional list of changed file paths for context

        Returns:
            ClassificationResult with category and metadata
        """
        start_time = time.time()

        if not message or not message.strip():
            return ClassificationResult(
                category="maintenance",
                confidence=0.3,
                method="empty_message",
                reasoning="Empty commit message",
                model=self.config.bedrock_model_id,
                alternatives=[],
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        self._apply_rate_limiting()
        system_prompt, user_prompt = self.prompt_generator.generate_prompt(message, files_changed)

        for attempt in range(self.config.max_retries):
            try:
                response_text, input_tokens, output_tokens = self._invoke_bedrock(
                    user_prompt=user_prompt,
                )

                category, confidence, reasoning = self.response_parser.parse_response(
                    response_text, self.prompt_generator.CATEGORIES
                )

                cost = self.cost_tracker.track_usage(input_tokens, output_tokens)
                total_tokens = input_tokens + output_tokens
                self.total_tokens_used += total_tokens
                self.total_cost += cost
                self.api_calls_made += 1

                return ClassificationResult(
                    category=category,
                    confidence=confidence,
                    method="llm",
                    reasoning=reasoning,
                    model=self.config.bedrock_model_id,
                    alternatives=[],
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            except Exception as exc:
                logger.warning(
                    "Bedrock attempt %d/%d failed: %s", attempt + 1, self.config.max_retries, exc
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_seconds * (2**attempt))
                else:
                    return ClassificationResult(
                        category="maintenance",
                        confidence=0.1,
                        method="llm_error",
                        reasoning=f"Bedrock classification failed: {exc}",
                        model="fallback",
                        alternatives=[],
                        processing_time_ms=(time.time() - start_time) * 1000,
                    )

        # Should never reach here
        return ClassificationResult(
            category="maintenance",
            confidence=0.1,
            method="llm_error",
            reasoning="Unexpected error in Bedrock classification",
            model="fallback",
            alternatives=[],
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    def classify_commits_batch(
        self,
        commits: list[dict[str, Any]],
        batch_id: Optional[str] = None,
    ) -> list[ClassificationResult]:
        """Classify a batch of commits via Bedrock using a single API call per sub-batch.

        WHY: Sending all commit messages in one prompt and asking for a JSON array
        response reduces API round-trips from N (one per commit) to ceil(N/25),
        which is dramatically faster for large batches (e.g. 918 commits → ~37 calls
        instead of 918).

        The incoming batch is split into sub-batches of at most _API_BATCH_SIZE
        commits so that the prompt+response fits comfortably within context limits.
        If the JSON response cannot be parsed, or the array length doesn't match,
        we fall back to individual classification for any missing results.

        Args:
            commits: List of commit dictionaries (must include 'message' key)
            batch_id: Optional batch identifier for tracking

        Returns:
            List of ClassificationResult objects in the same order as commits
        """
        if not commits:
            return []

        start_time = time.time()
        results: list[ClassificationResult] = []

        # Split into sub-batches that fit in one API call
        sub_batches = [
            commits[i : i + self._API_BATCH_SIZE]
            for i in range(0, len(commits), self._API_BATCH_SIZE)
        ]

        logger.info(
            "classify_commits_batch: %d commits → %d API calls (sub-batch size %d)",
            len(commits),
            len(sub_batches),
            self._API_BATCH_SIZE,
        )

        for sub_idx, sub_batch in enumerate(sub_batches):
            sub_results = self._classify_api_batch(sub_batch, batch_id, sub_idx)
            results.extend(sub_results)

        elapsed = time.time() - start_time
        logger.info(
            "classify_commits_batch: %d commits classified in %.2fs (%.0f ms/commit)",
            len(results),
            elapsed,
            (elapsed / len(results) * 1000) if results else 0,
        )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # Maximum commits sent in a single batch API call.
    # 25 commits × ~60 tokens/result ≈ 1 500 output tokens, well within limits.
    _API_BATCH_SIZE: int = 25

    # Batch system prompt — instructs the model to return a JSON array.
    _BATCH_SYSTEM_PROMPT: str = (
        "You are a git commit classifier. For each commit, classify it into exactly one category.\n\n"
        "Categories: feature, bugfix, maintenance, integration, content, media, localization\n\n"
        "Respond with a JSON array where each element has: "
        "category, confidence (0.0-1.0), reasoning (max 10 words).\n\n"
        "Example response for 3 commits:\n"
        '[{"category":"bugfix","confidence":0.95,"reasoning":"fixes crash in login flow"},'
        '{"category":"feature","confidence":0.90,"reasoning":"adds new authentication system"},'
        '{"category":"maintenance","confidence":0.85,"reasoning":"updates dependency versions"}]\n\n'
        "IMPORTANT: Return ONLY the JSON array. No other text."
    )

    def _classify_api_batch(
        self,
        commits: list[dict[str, Any]],
        batch_id: Optional[str],
        sub_idx: int,
    ) -> list[ClassificationResult]:
        """Send one API call for up to _API_BATCH_SIZE commits.

        Builds a numbered list prompt, calls Bedrock once, parses the JSON array
        response, and maps results back to commits.  Falls back to individual
        classification for any commit whose result is missing or unparseable.

        Args:
            commits: Sub-batch of commit dicts (len ≤ _API_BATCH_SIZE)
            batch_id: Optional batch identifier for result tagging
            sub_idx: Sub-batch index used in log messages

        Returns:
            List of ClassificationResult objects (same length as commits)
        """
        n = len(commits)
        messages_text = "\n".join(
            f'{i + 1}. "{commits[i].get("message", "").strip()}"' for i in range(n)
        )
        user_prompt = f"Classify these {n} commits:\n{messages_text}"

        # max_tokens scales with batch size; each JSON element is ~60-80 tokens
        batch_max_tokens = min(4096, n * 80)

        self._apply_rate_limiting()

        try:
            response_text, input_tokens, output_tokens = self._invoke_bedrock(
                user_prompt=user_prompt,
                max_tokens_override=batch_max_tokens,
                system_override=self._BATCH_SYSTEM_PROMPT,
            )

            cost = self.cost_tracker.track_usage(input_tokens, output_tokens)
            self.total_tokens_used += input_tokens + output_tokens
            self.total_cost += cost
            self.api_calls_made += 1

            parsed = self._parse_batch_json(response_text, n)

            if parsed is not None:
                logger.debug(
                    "Sub-batch %d: parsed %d/%d results from JSON",
                    sub_idx,
                    len(parsed),
                    n,
                )
                results = self._map_parsed_to_results(commits, parsed, batch_id)
                return results

            logger.warning(
                "Sub-batch %d: JSON parse failed — falling back to individual calls",
                sub_idx,
            )

        except Exception as exc:
            logger.warning(
                "Sub-batch %d API call failed (%s) — falling back to individual calls",
                sub_idx,
                exc,
            )

        # Fallback: classify each commit individually
        return self._individual_fallback(commits, batch_id)

    def _parse_batch_json(
        self, response_text: str, expected_count: int
    ) -> Optional[list[dict[str, Any]]]:
        """Parse the JSON array returned by a batch API call.

        Attempts several recovery strategies for common JSON issues:
        - Extracts the first [...] block if the model prepends/appends text
        - Falls back to None when the array cannot be recovered at all

        Args:
            response_text: Raw text returned by Bedrock
            expected_count: Expected number of elements

        Returns:
            Parsed list of dicts, or None if parsing failed
        """
        text = response_text.strip()

        # Fast path: the whole response is the JSON array
        if text.startswith("["):
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return self._validate_batch_items(data)
            except json.JSONDecodeError:
                pass

        # Recovery: extract the first [...] substring
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    return self._validate_batch_items(data)
            except json.JSONDecodeError:
                pass

        logger.debug("_parse_batch_json: could not extract JSON array from: %r", text[:200])
        return None

    def _validate_batch_items(self, items: list[Any]) -> list[dict[str, Any]]:
        """Validate and normalise parsed JSON items.

        Ensures each element has the required fields (category, confidence,
        reasoning).  Invalid elements are replaced with a maintenance fallback
        so downstream code always receives a complete list.

        Args:
            items: Raw parsed list from JSON

        Returns:
            Normalised list of classification dicts
        """
        valid_categories = set(self.prompt_generator.CATEGORIES.keys())
        normalised: list[dict[str, Any]] = []

        for item in items:
            if not isinstance(item, dict):
                normalised.append(
                    {"category": "maintenance", "confidence": 0.1, "reasoning": "invalid json item"}
                )
                continue

            category = str(item.get("category", "maintenance")).lower()
            if category not in valid_categories:
                category = "maintenance"

            try:
                confidence = float(item.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = 0.5

            reasoning = str(item.get("reasoning", ""))[:100]

            normalised.append(
                {"category": category, "confidence": confidence, "reasoning": reasoning}
            )

        return normalised

    def _map_parsed_to_results(
        self,
        commits: list[dict[str, Any]],
        parsed: list[dict[str, Any]],
        batch_id: Optional[str],
    ) -> list[ClassificationResult]:
        """Map parsed JSON items back to ClassificationResult objects.

        If the parsed list is shorter than the commits list, the remaining
        commits are classified individually as a fallback.

        Args:
            commits: Original commits sub-batch
            parsed: Validated parsed items (may be shorter than commits)
            batch_id: Optional batch identifier

        Returns:
            ClassificationResult list aligned with commits
        """
        import time as _time

        results: list[ClassificationResult] = []
        ts = _time.time()

        for i, commit in enumerate(commits):
            if i < len(parsed):
                item = parsed[i]
                result = ClassificationResult(
                    category=item["category"],
                    confidence=item["confidence"],
                    method="llm_batch",
                    reasoning=item["reasoning"],
                    model=self.config.bedrock_model_id,
                    alternatives=[],
                    processing_time_ms=(ts - ts) * 1000,  # negligible per-commit time
                    batch_id=batch_id,
                )
            else:
                # Parsed array shorter than expected — fallback for this commit
                logger.debug(
                    "Batch result missing for commit %d, falling back to individual call",
                    i,
                )
                message = commit.get("message", "")
                fc = commit.get("files_changed")
                files_changed: list[str] = fc if isinstance(fc, list) else []
                result = self.classify_commit(message, files_changed)
                if batch_id:
                    result.batch_id = batch_id

            results.append(result)

        return results

    def _individual_fallback(
        self,
        commits: list[dict[str, Any]],
        batch_id: Optional[str],
    ) -> list[ClassificationResult]:
        """Classify commits one-by-one as a fallback.

        Used when the batch API call fails or cannot be parsed.

        Args:
            commits: Commits to classify individually
            batch_id: Optional batch identifier

        Returns:
            List of ClassificationResult objects
        """
        results: list[ClassificationResult] = []
        for commit in commits:
            message = commit.get("message", "")
            fc = commit.get("files_changed")
            files_changed: list[str] = fc if isinstance(fc, list) else []
            result = self.classify_commit(message, files_changed)
            if batch_id:
                result.batch_id = batch_id
            results.append(result)
        return results

    def _invoke_bedrock(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        system_override: Optional[str] = None,
        max_tokens_override: Optional[int] = None,
    ) -> tuple[str, int, int]:
        """Call the Bedrock Runtime invoke_model endpoint.

        WHY: Claude on Bedrock uses the Messages API format wrapped in a JSON
        body. The system prompt is sent via the 'system' field (supported for
        Claude 3 models) rather than prepending it to the user message, which
        keeps the conversation structure clean and accurate for token counting.

        When called from single-commit mode the directive system prompt is used
        (hardcoded below).  When called from batch mode, the caller passes
        ``system_override`` to substitute the batch system prompt, and
        ``max_tokens_override`` to scale the token budget for the larger response.

        Args:
            user_prompt: User message containing the classification task
            system_prompt: Ignored — kept for signature compatibility
            system_override: If provided, use this system prompt instead of the
                             default directive single-commit prompt
            max_tokens_override: If provided, override config.max_tokens for
                                  this call (used by batch mode)

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)

        Raises:
            Exception: If the Bedrock API call fails
        """
        # WHY: Claude models return verbose natural language by default.
        # We override the system prompt to be extremely directive about
        # the output format, and add a stop sequence to cut off after the
        # first structured line.  This dramatically improves parse accuracy
        # compared to the generic prompt templates.
        directive_system = (
            "You are a git commit classifier. You MUST respond with EXACTLY "
            "one line in this format:\n"
            "CATEGORY confidence reasoning\n\n"
            "Where CATEGORY is one of: feature, bugfix, maintenance, "
            "integration, content, media, localization\n"
            "confidence is a float between 0.0 and 1.0\n"
            "reasoning is a brief explanation (max 10 words)\n\n"
            "Example responses:\n"
            "bugfix 0.95 fixes crash in login flow\n"
            "feature 0.90 adds new authentication system\n"
            "maintenance 0.85 updates dependency versions\n\n"
            "Do NOT include any other text, explanation, or formatting."
        )

        effective_system = system_override if system_override is not None else directive_system
        effective_max_tokens = (
            max_tokens_override if max_tokens_override is not None else self.config.max_tokens
        )

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": effective_max_tokens,
                "temperature": self.config.temperature,
                "system": effective_system,
                "messages": [
                    {"role": "user", "content": user_prompt},
                ],
            }
        )

        logger.debug(
            "Invoking Bedrock model %s (max_tokens=%d)",
            self.config.bedrock_model_id,
            effective_max_tokens,
        )

        response = self._client.invoke_model(
            modelId=self.config.bedrock_model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())

        if not result.get("content"):
            raise ValueError(f"Unexpected Bedrock response structure: {result}")

        raw_text = result["content"][0]["text"].strip()

        # For single-commit mode, take only the first line to avoid trailing text.
        # For batch mode the caller receives the full response for JSON parsing.
        # For single-commit mode, take only the first line to avoid trailing text.
        text = raw_text if system_override is not None else raw_text.split("\n")[0].strip()

        # Extract token usage from the Bedrock response (Claude 3 always provides it)
        usage = result.get("usage", {})
        input_tokens = usage.get(
            "input_tokens", self._estimate_tokens(effective_system + user_prompt)
        )
        output_tokens = usage.get("output_tokens", self._estimate_tokens(text))

        return text, input_tokens, output_tokens

    def _setup_pricing(self) -> None:
        """Configure cost tracker pricing based on the selected Bedrock model.

        WHY: Accurate cost tracking helps teams monitor LLM spend. We match
        on a substring of the model ID (haiku/sonnet/opus) since Bedrock model
        IDs include version dates that would break an exact-match lookup.
        """
        model_lower = self.config.bedrock_model_id.lower()

        for key, pricing in _BEDROCK_PRICING_MAP.items():
            if key in model_lower:
                self.cost_tracker.set_model_pricing(pricing)
                return

        # Unknown Bedrock model: default to Haiku pricing (most conservative estimate)
        logger.warning(
            "Unknown Bedrock model %s — using Haiku pricing for cost estimates",
            self.config.bedrock_model_id,
        )
        self.cost_tracker.set_model_pricing(_BEDROCK_HAIKU_PRICING)

    def _apply_rate_limiting(self) -> None:
        """Apply per-minute rate limiting to avoid throttling.

        WHY: Bedrock enforces per-minute request quotas per model/region.
        Sleeping briefly when the limit is approached avoids ThrottlingExceptions
        that would otherwise consume a retry attempt.
        """
        current_time = time.time()

        if current_time - self._minute_start >= 60:
            self._request_count = 0
            self._minute_start = current_time

        if self._request_count >= self.config.max_requests_per_minute:
            sleep_time = 60 - (current_time - self._minute_start)
            if sleep_time > 0:
                logger.debug("Bedrock rate limit: sleeping %.1fs", sleep_time)
                time.sleep(sleep_time)
                self._request_count = 0
                self._minute_start = time.time()

        self._request_count += 1
        self._last_request_time = time.time()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using a simple character-based heuristic.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count (~4 chars per token on average)
        """
        return len(text) // 4
