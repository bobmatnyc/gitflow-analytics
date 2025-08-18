"""Cost tracking utilities for LLM usage monitoring."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


@dataclass
class LLMCall:
    """Record of a single LLM API call."""

    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    processing_time_ms: float
    estimated_cost: float
    batch_size: int = 1
    success: bool = True
    error_message: Optional[str] = None


class CostTracker:
    """Track and manage LLM API usage costs.

    This class provides cost monitoring, budgeting, and optimization
    features to keep LLM usage within acceptable limits while
    maintaining analysis quality.
    """

    # OpenRouter pricing (approximate, in USD per 1M tokens)
    MODEL_PRICING = {
        # Anthropic models
        "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
        "anthropic/claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "anthropic/claude-3-opus": {"input": 15.0, "output": 75.0},
        # OpenAI models
        "openai/gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "openai/gpt-4": {"input": 30.0, "output": 60.0},
        "openai/gpt-4-turbo": {"input": 10.0, "output": 30.0},
        # Free models (Llama)
        "meta-llama/llama-3.1-8b-instruct:free": {"input": 0.0, "output": 0.0},
        "meta-llama/llama-3.1-70b-instruct:free": {"input": 0.0, "output": 0.0},
        # Other popular models
        "google/gemini-pro": {"input": 0.5, "output": 1.5},
        "mistralai/mixtral-8x7b-instruct": {"input": 0.27, "output": 0.27},
    }

    def __init__(self, cache_dir: Optional[Path] = None, daily_budget: float = 5.0):
        """Initialize cost tracker.

        Args:
            cache_dir: Directory to store cost tracking data
            daily_budget: Maximum daily spending in USD
        """
        self.daily_budget = daily_budget
        self.cache_dir = cache_dir or Path(".qualitative_cache")
        self.cache_dir.mkdir(exist_ok=True)

        self.cost_file = self.cache_dir / "llm_costs.json"
        self.calls: list[LLMCall] = []
        self.logger = logging.getLogger(__name__)

        # Load existing cost data
        self._load_cost_data()

    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        processing_time: float,
        batch_size: int = 1,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> float:
        """Record an LLM API call and return estimated cost.

        Args:
            model: Model name used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            processing_time: Processing time in seconds
            batch_size: Number of commits processed in this call
            success: Whether the call was successful
            error_message: Error message if call failed

        Returns:
            Estimated cost in USD
        """
        estimated_cost = self._calculate_cost(model, input_tokens, output_tokens)

        call = LLMCall(
            timestamp=datetime.utcnow(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            processing_time_ms=processing_time * 1000,
            estimated_cost=estimated_cost,
            batch_size=batch_size,
            success=success,
            error_message=error_message,
        )

        self.calls.append(call)
        self._save_cost_data()

        # Log cost information
        self.logger.info(
            f"LLM call: {model} | tokens: {input_tokens}+{output_tokens} | "
            f"cost: ${estimated_cost:.4f} | batch: {batch_size}"
        )

        return estimated_cost

    def get_daily_spend(self, date: Optional[datetime] = None) -> float:
        """Get total spending for a specific date.

        Args:
            date: Date to check (defaults to today)

        Returns:
            Total spending in USD for the date
        """
        if date is None:
            date = datetime.utcnow()

        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        daily_spend = sum(
            call.estimated_cost
            for call in self.calls
            if start_of_day <= call.timestamp < end_of_day and call.success
        )

        return daily_spend

    def check_budget_remaining(self) -> float:
        """Check remaining budget for today.

        Returns:
            Remaining budget in USD (negative if over budget)
        """
        daily_spend = self.get_daily_spend()
        return self.daily_budget - daily_spend

    def can_afford_call(self, model: str, estimated_tokens: int) -> bool:
        """Check if we can afford an API call within budget.

        Args:
            model: Model to use
            estimated_tokens: Estimated total tokens (input + output)

        Returns:
            True if call is within budget
        """
        estimated_cost = self._calculate_cost(model, estimated_tokens // 2, estimated_tokens // 2)
        remaining_budget = self.check_budget_remaining()

        return remaining_budget >= estimated_cost

    def get_usage_stats(self, days: int = 7) -> dict[str, any]:
        """Get usage statistics for the last N days.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with usage statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_calls = [call for call in self.calls if call.timestamp >= cutoff_date]

        if not recent_calls:
            return {
                "total_calls": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "avg_cost_per_call": 0.0,
                "model_usage": {},
                "success_rate": 1.0,
            }

        successful_calls = [call for call in recent_calls if call.success]

        # Calculate statistics
        total_cost = sum(call.estimated_cost for call in successful_calls)
        total_tokens = sum(call.input_tokens + call.output_tokens for call in recent_calls)

        # Model usage breakdown
        model_usage = {}
        for call in recent_calls:
            if call.model not in model_usage:
                model_usage[call.model] = {"calls": 0, "cost": 0.0, "tokens": 0}
            model_usage[call.model]["calls"] += 1
            model_usage[call.model]["cost"] += call.estimated_cost
            model_usage[call.model]["tokens"] += call.input_tokens + call.output_tokens

        return {
            "total_calls": len(recent_calls),
            "successful_calls": len(successful_calls),
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "avg_cost_per_call": total_cost / len(successful_calls) if successful_calls else 0.0,
            "model_usage": model_usage,
            "success_rate": len(successful_calls) / len(recent_calls) if recent_calls else 1.0,
            "daily_average_cost": total_cost / days,
        }

    def suggest_cost_optimizations(self) -> list[str]:
        """Suggest ways to optimize costs based on usage patterns.

        Returns:
            List of optimization suggestions
        """
        suggestions = []
        stats = self.get_usage_stats(days=7)

        if stats["total_calls"] == 0:
            return suggestions

        # Check if expensive models are overused
        model_usage = stats["model_usage"]
        total_cost = stats["total_cost"]

        expensive_models = ["anthropic/claude-3-opus", "openai/gpt-4"]
        expensive_usage = sum(
            model_usage.get(model, {}).get("cost", 0) for model in expensive_models
        )

        if expensive_usage > total_cost * 0.3:
            suggestions.append(
                "Consider using cheaper models (Claude Haiku, GPT-3.5) for routine classification"
            )

        # Check for free model opportunities
        free_usage = model_usage.get("meta-llama/llama-3.1-8b-instruct:free", {}).get("calls", 0)
        if free_usage < stats["total_calls"] * 0.5:
            suggestions.append(
                "Increase usage of free Llama models for simple classification tasks"
            )

        # Check daily spend
        if self.get_daily_spend() > self.daily_budget * 0.8:
            suggestions.append(
                "Approaching daily budget limit - consider increasing NLP confidence threshold"
            )

        # Check batch efficiency
        avg_batch_size = sum(call.batch_size for call in self.calls[-50:]) / min(  # Last 50 calls
            50, len(self.calls)
        )

        if avg_batch_size < 3:
            suggestions.append("Increase batch size for LLM calls to improve cost efficiency")

        return suggestions

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost for an API call.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        if model not in self.MODEL_PRICING:
            # Default to moderate pricing for unknown models
            input_price = 1.0
            output_price = 3.0
            self.logger.warning(f"Unknown model pricing for {model}, using default rates")
        else:
            pricing = self.MODEL_PRICING[model]
            input_price = pricing["input"]
            output_price = pricing["output"]

        # Calculate cost (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price

        return input_cost + output_cost

    def _load_cost_data(self) -> None:
        """Load cost tracking data from file."""
        if not self.cost_file.exists():
            return

        try:
            with open(self.cost_file) as f:
                data = json.load(f)

            self.calls = []
            for call_data in data.get("calls", []):
                call = LLMCall(
                    timestamp=datetime.fromisoformat(call_data["timestamp"]),
                    model=call_data["model"],
                    input_tokens=call_data["input_tokens"],
                    output_tokens=call_data["output_tokens"],
                    processing_time_ms=call_data["processing_time_ms"],
                    estimated_cost=call_data["estimated_cost"],
                    batch_size=call_data.get("batch_size", 1),
                    success=call_data.get("success", True),
                    error_message=call_data.get("error_message"),
                )
                self.calls.append(call)

        except Exception as e:
            self.logger.error(f"Failed to load cost data: {e}")
            self.calls = []

    def _save_cost_data(self) -> None:
        """Save cost tracking data to file."""
        try:
            # Keep only last 1000 calls to prevent file from growing too large
            recent_calls = self.calls[-1000:]

            data = {
                "calls": [
                    {
                        "timestamp": call.timestamp.isoformat(),
                        "model": call.model,
                        "input_tokens": call.input_tokens,
                        "output_tokens": call.output_tokens,
                        "processing_time_ms": call.processing_time_ms,
                        "estimated_cost": call.estimated_cost,
                        "batch_size": call.batch_size,
                        "success": call.success,
                        "error_message": call.error_message,
                    }
                    for call in recent_calls
                ]
            }

            with open(self.cost_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save cost data: {e}")
