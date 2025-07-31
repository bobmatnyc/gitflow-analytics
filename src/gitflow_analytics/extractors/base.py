"""Base classes for pluggable extractors."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class ExtractorBase(ABC):
    """Base class for all extractors."""

    @abstractmethod
    def extract_from_text(self, text: str) -> Any:
        """Extract information from text."""
        pass


class StoryPointExtractorBase(ExtractorBase):
    """Base class for story point extractors."""

    @abstractmethod
    def extract_from_text(self, text: str) -> Optional[int]:
        """Extract story points from text."""
        pass

    @abstractmethod
    def extract_from_pr(
        self, pr_data: dict[str, Any], commit_messages: Optional[list[str]] = None
    ) -> Optional[int]:
        """Extract story points from pull request."""
        pass


class TicketExtractorBase(ExtractorBase):
    """Base class for ticket extractors."""

    @abstractmethod
    def extract_from_text(self, text: str) -> list[dict[str, str]]:
        """Extract ticket references from text."""
        pass

    @abstractmethod
    def extract_by_platform(self, text: str) -> dict[str, list[str]]:
        """Extract tickets grouped by platform."""
        pass
