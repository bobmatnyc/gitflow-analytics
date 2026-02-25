"""Shared component factory for Git analysis classes.

Both GitDataFetcher and GitAnalyzer need the same trio of domain objects:
  - StoryPointExtractor
  - TicketExtractor (standard or ML-enhanced)
  - BranchToProjectMapper

This module centralises their construction so neither class duplicates the
instantiation logic, while keeping the DI pattern intact: callers still
receive fully-constructed objects and can substitute mocks or subclasses.
"""

from __future__ import annotations

import logging
from typing import Any

from ..extractors.story_points import StoryPointExtractor
from ..extractors.tickets import TicketExtractor
from .branch_mapper import BranchToProjectMapper

# Import ML extractor with graceful fallback
try:
    from ..extractors.ml_tickets import MLTicketExtractor

    _ML_EXTRACTOR_AVAILABLE = True
except ImportError:
    _ML_EXTRACTOR_AVAILABLE = False

logger = logging.getLogger(__name__)


def build_story_point_extractor(
    patterns: list[str] | None = None,
) -> StoryPointExtractor:
    """Create a StoryPointExtractor with optional custom patterns.

    Args:
        patterns: Optional list of regex patterns for story-point extraction.
                  When None the extractor uses its built-in defaults.

    Returns:
        Configured StoryPointExtractor instance.
    """
    return StoryPointExtractor(patterns=patterns)


def build_ticket_extractor(
    allowed_platforms: list[str] | None = None,
    ml_config: dict[str, Any] | None = None,
    llm_config: dict[str, Any] | None = None,
    cache_dir: Any | None = None,
) -> TicketExtractor:
    """Create the appropriate TicketExtractor based on configuration.

    Selects between ML-enhanced and standard extraction automatically.
    Falls back to the standard extractor when ML dependencies are absent
    or the feature is disabled in configuration.

    Args:
        allowed_platforms: Restrict ticket extraction to these platform keys.
        ml_config: ML categorisation configuration dict.  When *enabled* is
                   True and ML dependencies are available, the ML extractor
                   is returned.
        llm_config: LLM classification configuration dict.
        cache_dir: Cache directory for ML prediction persistence.

    Returns:
        TicketExtractor (or MLTicketExtractor subclass) instance.
    """
    if ml_config and ml_config.get("enabled", True) and _ML_EXTRACTOR_AVAILABLE:
        logger.info("Initializing ML-enhanced ticket extractor")
        enable_llm = bool(llm_config and llm_config.get("enabled", False))
        if enable_llm:
            logger.info("LLM-based commit classification enabled")
        return MLTicketExtractor(  # type: ignore[return-value]
            allowed_platforms=allowed_platforms,
            ml_config=ml_config,
            llm_config=llm_config,
            cache_dir=cache_dir,
            enable_ml=True,
            enable_llm=enable_llm,
        )

    # Log why ML is not used
    if ml_config and ml_config.get("enabled", True):
        if not _ML_EXTRACTOR_AVAILABLE:
            logger.warning(
                "ML categorization requested but dependencies not available, "
                "using standard extractor"
            )
        else:
            logger.info("ML categorization disabled in configuration, using standard extractor")
    else:
        logger.debug("Using standard ticket extractor")

    return TicketExtractor(allowed_platforms=allowed_platforms)


def build_branch_mapper(
    branch_mapping_rules: dict[str, list[str]] | None = None,
) -> BranchToProjectMapper:
    """Create a BranchToProjectMapper from optional mapping rules.

    Args:
        branch_mapping_rules: Mapping of project key to list of branch patterns.

    Returns:
        Configured BranchToProjectMapper instance.
    """
    return BranchToProjectMapper(branch_mapping_rules)
