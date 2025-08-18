"""Cost tracking and management for LLM API usage.

This module tracks API usage costs and provides warnings when
approaching or exceeding cost thresholds.

WHY: LLM API calls can be expensive. Tracking costs helps users
monitor expenses and make informed decisions about usage.

DESIGN DECISIONS:
- Support multiple pricing models for different providers
- Track costs at token level for accuracy
- Provide cost warnings and limits
- Support cost budgets and alerts
- Export cost data for analysis
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelPricing:
    """Pricing information for a specific model.

    WHY: Different models have different pricing structures.
    This allows accurate cost calculation per model.
    """

    model_name: str
    input_cost_per_million: float  # Cost per 1M input tokens in USD
    output_cost_per_million: float  # Cost per 1M output tokens in USD

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts.

        Args:
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens

        Returns:
            Total cost in USD
        """
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_million
        return input_cost + output_cost


@dataclass
class CostRecord:
    """Record of a single API call's cost.

    WHY: Detailed cost records enable analysis of spending patterns
    and identification of optimization opportunities.
    """

    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    endpoint: str = "unknown"
    batch_id: Optional[str] = None


class CostTracker:
    """Tracks and manages LLM API usage costs.

    WHY: Cost management is critical for production LLM usage.
    This provides detailed tracking, warnings, and budgeting.
    """

    # Default pricing for common models (as of 2024)
    DEFAULT_PRICING = {
        "gpt-4": ModelPricing("gpt-4", 30.0, 60.0),
        "gpt-4-turbo": ModelPricing("gpt-4-turbo", 10.0, 30.0),
        "gpt-4-turbo-preview": ModelPricing("gpt-4-turbo-preview", 10.0, 30.0),
        "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", 0.5, 1.5),
        "gpt-3.5-turbo-16k": ModelPricing("gpt-3.5-turbo-16k", 1.0, 2.0),
        "claude-3-opus": ModelPricing("claude-3-opus", 15.0, 75.0),
        "claude-3-sonnet": ModelPricing("claude-3-sonnet", 3.0, 15.0),
        "claude-3-haiku": ModelPricing("claude-3-haiku", 0.25, 1.25),
        "claude-2.1": ModelPricing("claude-2.1", 8.0, 24.0),
        "claude-2": ModelPricing("claude-2", 8.0, 24.0),
        "mistral-7b": ModelPricing("mistral-7b", 0.25, 0.25),
        "mistral-8x7b": ModelPricing("mistral-8x7b", 0.7, 0.7),
        "llama-2-70b": ModelPricing("llama-2-70b", 0.7, 0.9),
        "llama-2-13b": ModelPricing("llama-2-13b", 0.2, 0.25),
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        daily_budget: Optional[float] = None,
        monthly_budget: Optional[float] = None,
    ):
        """Initialize cost tracker.

        Args:
            cache_dir: Directory for storing cost records
            daily_budget: Optional daily spending limit in USD
            monthly_budget: Optional monthly spending limit in USD
        """
        self.cache_dir = cache_dir or Path(".gitflow-cache")
        self.cache_dir.mkdir(exist_ok=True)

        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget

        # Current session costs
        self.session_costs: list[CostRecord] = []
        self.session_total = 0.0

        # Current model pricing
        self.current_pricing: Optional[ModelPricing] = None

        # Load historical costs
        self._load_cost_history()

    def set_model_pricing(self, pricing: ModelPricing) -> None:
        """Set the pricing for the current model.

        Args:
            pricing: Model pricing information
        """
        self.current_pricing = pricing
        logger.debug(
            f"Set pricing for {pricing.model_name}: "
            f"${pricing.input_cost_per_million}/1M input, "
            f"${pricing.output_cost_per_million}/1M output"
        )

    def track_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> float:
        """Track token usage and calculate cost.

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            model: Optional model name override
            batch_id: Optional batch identifier

        Returns:
            Cost of this usage in USD
        """
        # Use current pricing or try to find from model name
        pricing = self.current_pricing
        if not pricing and model:
            pricing = self._find_pricing_for_model(model)
        if not pricing:
            # Use a default conservative estimate
            pricing = ModelPricing("unknown", 1.0, 1.0)

        # Calculate cost
        cost = pricing.calculate_cost(input_tokens, output_tokens)

        # Create cost record
        record = CostRecord(
            timestamp=datetime.now(),
            model=model or pricing.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            batch_id=batch_id,
        )

        # Track in session
        self.session_costs.append(record)
        self.session_total += cost

        # Check budgets
        self._check_budgets(cost)

        # Log if significant cost
        if cost > 0.01:  # Log costs over 1 cent
            logger.info(f"API call cost: ${cost:.4f} ({input_tokens} in, {output_tokens} out)")

        return cost

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model: Optional[str] = None
    ) -> float:
        """Calculate cost without tracking (for estimates).

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Optional model name

        Returns:
            Estimated cost in USD
        """
        pricing = self.current_pricing
        if not pricing and model:
            pricing = self._find_pricing_for_model(model)
        if not pricing:
            pricing = ModelPricing("unknown", 1.0, 1.0)

        return pricing.calculate_cost(input_tokens, output_tokens)

    def get_session_summary(self) -> dict:
        """Get summary of current session costs.

        Returns:
            Dictionary with session cost information
        """
        if not self.session_costs:
            return {
                "total_cost": 0.0,
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "average_cost_per_call": 0.0,
            }

        total_input = sum(r.input_tokens for r in self.session_costs)
        total_output = sum(r.output_tokens for r in self.session_costs)

        return {
            "total_cost": self.session_total,
            "total_calls": len(self.session_costs),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "average_cost_per_call": self.session_total / len(self.session_costs),
            "models_used": list(set(r.model for r in self.session_costs)),
        }

    def get_daily_costs(self) -> float:
        """Get total costs for today.

        Returns:
            Total cost in USD for the current day
        """
        today = datetime.now().date()
        daily_total = sum(r.cost_usd for r in self.session_costs if r.timestamp.date() == today)

        # Also check historical costs
        history_file = self._get_history_file()
        if history_file.exists():
            try:
                with open(history_file) as f:
                    for line in f:
                        record_dict = json.loads(line)
                        timestamp = datetime.fromisoformat(record_dict["timestamp"])
                        if timestamp.date() == today:
                            daily_total += record_dict["cost_usd"]
            except Exception as e:
                logger.warning(f"Error reading cost history: {e}")

        return daily_total

    def get_monthly_costs(self) -> float:
        """Get total costs for the current month.

        Returns:
            Total cost in USD for the current month
        """
        now = datetime.now()
        month_start = datetime(now.year, now.month, 1)

        monthly_total = sum(r.cost_usd for r in self.session_costs if r.timestamp >= month_start)

        # Also check historical costs
        history_file = self._get_history_file()
        if history_file.exists():
            try:
                with open(history_file) as f:
                    for line in f:
                        record_dict = json.loads(line)
                        timestamp = datetime.fromisoformat(record_dict["timestamp"])
                        if timestamp >= month_start:
                            monthly_total += record_dict["cost_usd"]
            except Exception as e:
                logger.warning(f"Error reading cost history: {e}")

        return monthly_total

    def save_session(self) -> None:
        """Save current session costs to history file.

        WHY: Persisting cost data enables long-term tracking
        and analysis of LLM usage patterns.
        """
        if not self.session_costs:
            return

        history_file = self._get_history_file()

        try:
            with open(history_file, "a") as f:
                for record in self.session_costs:
                    # Convert to dict and handle datetime
                    record_dict = asdict(record)
                    record_dict["timestamp"] = record.timestamp.isoformat()
                    f.write(json.dumps(record_dict) + "\n")

            logger.info(f"Saved {len(self.session_costs)} cost records to history")

            # Clear session costs after saving
            self.session_costs = []
            self.session_total = 0.0

        except Exception as e:
            logger.error(f"Failed to save cost history: {e}")

    def export_costs(self, output_file: Path) -> None:
        """Export all cost data to a JSON file.

        Args:
            output_file: Path to export file
        """
        all_records = []

        # Add current session
        for record in self.session_costs:
            record_dict = asdict(record)
            record_dict["timestamp"] = record.timestamp.isoformat()
            all_records.append(record_dict)

        # Add historical
        history_file = self._get_history_file()
        if history_file.exists():
            try:
                with open(history_file) as f:
                    for line in f:
                        all_records.append(json.loads(line))
            except Exception as e:
                logger.warning(f"Error reading cost history: {e}")

        # Write export file
        with open(output_file, "w") as f:
            json.dump(
                {
                    "records": all_records,
                    "summary": self.get_session_summary(),
                    "daily_total": self.get_daily_costs(),
                    "monthly_total": self.get_monthly_costs(),
                },
                f,
                indent=2,
            )

        logger.info(f"Exported {len(all_records)} cost records to {output_file}")

    def _find_pricing_for_model(self, model: str) -> Optional[ModelPricing]:
        """Find pricing information for a model name.

        Args:
            model: Model name to find pricing for

        Returns:
            ModelPricing or None if not found
        """
        model_lower = model.lower()

        # Check exact matches first
        if model_lower in self.DEFAULT_PRICING:
            return self.DEFAULT_PRICING[model_lower]

        # Check partial matches
        for key, pricing in self.DEFAULT_PRICING.items():
            if key in model_lower or model_lower in key:
                return pricing

        # Check for common prefixes
        if "gpt-4" in model_lower:
            return self.DEFAULT_PRICING.get("gpt-4-turbo", self.DEFAULT_PRICING["gpt-4"])
        elif "gpt-3" in model_lower:
            return self.DEFAULT_PRICING["gpt-3.5-turbo"]
        elif "claude" in model_lower:
            return self.DEFAULT_PRICING.get("claude-2", ModelPricing("claude", 8.0, 24.0))
        elif "mistral" in model_lower:
            return self.DEFAULT_PRICING.get("mistral-7b", ModelPricing("mistral", 0.25, 0.25))
        elif "llama" in model_lower:
            return self.DEFAULT_PRICING.get("llama-2-70b", ModelPricing("llama", 0.7, 0.9))

        return None

    def _check_budgets(self, new_cost: float) -> None:
        """Check if budgets are being exceeded.

        Args:
            new_cost: Cost of the latest API call
        """
        # Check daily budget
        if self.daily_budget:
            daily_total = self.get_daily_costs()
            if daily_total > self.daily_budget:
                logger.warning(
                    f"DAILY BUDGET EXCEEDED: ${daily_total:.2f} > ${self.daily_budget:.2f}"
                )
            elif daily_total > self.daily_budget * 0.8:
                logger.warning(
                    f"Approaching daily budget: ${daily_total:.2f} of ${self.daily_budget:.2f}"
                )

        # Check monthly budget
        if self.monthly_budget:
            monthly_total = self.get_monthly_costs()
            if monthly_total > self.monthly_budget:
                logger.warning(
                    f"MONTHLY BUDGET EXCEEDED: ${monthly_total:.2f} > ${self.monthly_budget:.2f}"
                )
            elif monthly_total > self.monthly_budget * 0.8:
                logger.warning(
                    f"Approaching monthly budget: ${monthly_total:.2f} of ${self.monthly_budget:.2f}"
                )

    def _get_history_file(self) -> Path:
        """Get path to cost history file.

        Returns:
            Path to history file
        """
        return self.cache_dir / "llm_costs.jsonl"

    def _load_cost_history(self) -> None:
        """Load cost history from file.

        WHY: Loading historical costs enables budget tracking
        across multiple sessions.
        """
        # For now, we don't load into memory to avoid memory issues
        # History is queried when needed for daily/monthly totals
        pass
