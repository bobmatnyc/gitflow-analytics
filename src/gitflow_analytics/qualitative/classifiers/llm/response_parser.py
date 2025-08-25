"""Response parsing and validation for LLM outputs.

This module handles parsing of LLM responses into structured classification
results, including validation and error handling.

WHY: LLM responses can be unpredictable. Robust parsing with fallbacks
ensures the system remains stable even with unexpected outputs.

DESIGN DECISIONS:
- Support multiple response formats for flexibility
- Validate categories against known categories
- Extract confidence scores with bounds checking
- Parse reasoning text safely
- Provide detailed error messages for debugging
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parses and validates LLM classification responses.

    WHY: Centralizing response parsing logic makes it easier to handle
    different response formats and add new parsing strategies.
    """

    def __init__(self):
        """Initialize response parser."""
        # Regex patterns for different response formats
        self.patterns = {
            "standard": re.compile(r"^(\w+)\s+([\d.]+)\s+(.*)$", re.IGNORECASE),
            "colon_separated": re.compile(r"^(\w+):\s*([\d.]+)[,\s]+(.*)$", re.IGNORECASE),
            "json_like": re.compile(
                r'["\']?category["\']?\s*:\s*["\']?(\w+)["\']?.*?["\']?confidence["\']?\s*:\s*([\d.]+)',
                re.IGNORECASE | re.DOTALL,
            ),
            "simple": re.compile(r"^(\w+)\s+([\d.]+)$", re.IGNORECASE),
        }

    def parse_response(
        self, response: str, valid_categories: dict[str, str]
    ) -> tuple[str, float, str]:
        """Parse LLM response to extract classification components.

        Args:
            response: Raw LLM response text
            valid_categories: Dictionary of valid category names

        Returns:
            Tuple of (category, confidence, reasoning)
        """
        if not response:
            logger.warning("Empty response from LLM")
            return self._fallback_result("Empty response")

        # Clean the response
        response = response.strip()

        # Try each parsing pattern
        for pattern_name, pattern in self.patterns.items():
            match = pattern.match(response)
            if match:
                return self._process_match(match, pattern_name, valid_categories)

        # Try to extract just the category if nothing else works
        category = self._extract_category_fuzzy(response, valid_categories)
        if category:
            logger.debug(f"Fuzzy matched category: {category} from response: {response}")
            return category, 0.5, "Fuzzy match from response"

        # Complete fallback
        logger.warning(f"Could not parse response: {response}")
        return self._fallback_result(f"Parse failed: {response[:50]}")

    def _process_match(
        self, match: re.Match, pattern_name: str, valid_categories: dict[str, str]
    ) -> tuple[str, float, str]:
        """Process a regex match to extract classification components.

        Args:
            match: Regex match object
            pattern_name: Name of the pattern that matched
            valid_categories: Dictionary of valid categories

        Returns:
            Tuple of (category, confidence, reasoning)
        """
        groups = match.groups()

        # Extract category
        category = groups[0].lower().strip()

        # Validate category
        if category not in valid_categories:
            # Try to find closest match
            category = self._find_closest_category(category, valid_categories)
            if not category:
                return self._fallback_result(f"Invalid category: {groups[0]}")

        # Extract confidence
        confidence = 0.5  # Default
        if len(groups) > 1:
            try:
                confidence = float(groups[1])
                # Clamp to valid range
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                logger.debug(f"Could not parse confidence: {groups[1]}")

        # Extract reasoning
        reasoning = "No reasoning provided"
        if len(groups) > 2 and groups[2]:
            reasoning = groups[2].strip()
            # Clean up reasoning
            reasoning = self._clean_reasoning(reasoning)
        elif pattern_name == "simple":
            reasoning = f"Classified as {category}"

        return category, confidence, reasoning

    def _extract_category_fuzzy(
        self, response: str, valid_categories: dict[str, str]
    ) -> Optional[str]:
        """Try to extract a category using fuzzy matching.

        WHY: Sometimes LLMs include extra text or formatting that
        breaks strict parsing but the category is still identifiable.

        Args:
            response: Response text to search
            valid_categories: Dictionary of valid categories

        Returns:
            Matched category or None
        """
        response_lower = response.lower()

        # Look for exact category names in the response
        for category in valid_categories:
            if category in response_lower:
                # Check it's not part of another word
                pattern = r"\b" + re.escape(category) + r"\b"
                if re.search(pattern, response_lower):
                    return category

        # Look for category descriptions
        for category, description in valid_categories.items():
            # Check if key terms from description appear
            key_terms = description.lower().split(",")[0].split()
            if len(key_terms) > 0 and key_terms[0] in response_lower:
                return category

        return None

    def _find_closest_category(
        self, candidate: str, valid_categories: dict[str, str]
    ) -> Optional[str]:
        """Find the closest matching category for a candidate.

        WHY: Handle minor typos or variations in category names
        to improve robustness.

        Args:
            candidate: Candidate category name
            valid_categories: Dictionary of valid categories

        Returns:
            Closest matching category or None
        """
        candidate_lower = candidate.lower()

        # Check for common variations
        variations = {
            "bug": "bugfix",
            "fix": "bugfix",
            "bugs": "bugfix",
            "feat": "feature",
            "features": "feature",
            "maint": "maintenance",
            "maintain": "maintenance",
            "chore": "maintenance",
            "docs": "content",
            "documentation": "content",
            "doc": "content",
            "i18n": "localization",
            "l10n": "localization",
            "translation": "localization",
            "integrate": "integration",
            "api": "integration",
            "video": "media",
            "audio": "media",
        }

        if candidate_lower in variations:
            matched = variations[candidate_lower]
            if matched in valid_categories:
                return matched

        # Check for partial matches
        for category in valid_categories:
            if candidate_lower.startswith(category[:3]):
                return category
            if category.startswith(candidate_lower[:3]):
                return category

        return None

    def _clean_reasoning(self, reasoning: str) -> str:
        """Clean up reasoning text.

        Args:
            reasoning: Raw reasoning text

        Returns:
            Cleaned reasoning text
        """
        # Remove extra whitespace
        reasoning = " ".join(reasoning.split())

        # Remove quotes if present
        if reasoning.startswith('"') and reasoning.endswith('"'):
            reasoning = reasoning[1:-1]
        if reasoning.startswith("'") and reasoning.endswith("'"):
            reasoning = reasoning[1:-1]

        # Truncate if too long
        max_length = 200
        if len(reasoning) > max_length:
            reasoning = reasoning[:max_length] + "..."

        # Ensure it's not empty
        if not reasoning:
            reasoning = "No reasoning provided"

        return reasoning

    def _fallback_result(self, error_context: str) -> tuple[str, float, str]:
        """Generate a fallback result when parsing fails.

        Args:
            error_context: Context about the parsing failure

        Returns:
            Tuple of (category, confidence, reasoning)
        """
        return "maintenance", 0.1, f"Parse error: {error_context}"

    def validate_classification(
        self, category: str, confidence: float, valid_categories: dict[str, str]
    ) -> tuple[str, float, bool]:
        """Validate and potentially correct a classification.

        Args:
            category: Classified category
            confidence: Confidence score
            valid_categories: Dictionary of valid categories

        Returns:
            Tuple of (category, confidence, is_valid)
        """
        is_valid = True

        # Check category validity
        if category not in valid_categories:
            # Try to correct
            corrected = self._find_closest_category(category, valid_categories)
            if corrected:
                logger.debug(f"Corrected category {category} to {corrected}")
                category = corrected
                confidence *= 0.8  # Reduce confidence for correction
            else:
                logger.warning(f"Invalid category {category}, defaulting to maintenance")
                category = "maintenance"
                confidence = 0.1
                is_valid = False

        # Validate confidence bounds
        if confidence < 0 or confidence > 1:
            logger.warning(f"Invalid confidence {confidence}, clamping to [0, 1]")
            confidence = max(0.0, min(1.0, confidence))
            is_valid = False

        return category, confidence, is_valid
