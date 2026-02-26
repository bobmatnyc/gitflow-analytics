"""Small helper functions shared between analyze_pipeline.py and cli.py."""

from __future__ import annotations

from typing import Any, Optional


def is_qualitative_enabled(cfg: Any) -> bool:
    """Return True when qualitative analysis is enabled in config."""
    if cfg.qualitative and cfg.qualitative.enabled:
        return True
    return (
        hasattr(cfg.analysis, "qualitative")
        and cfg.analysis.qualitative
        and cfg.analysis.qualitative.enabled
    )


def get_qualitative_config(cfg: Any) -> Optional[Any]:
    """Return the qualitative config object from whichever location it lives."""
    if cfg.qualitative:
        return cfg.qualitative
    if hasattr(cfg.analysis, "qualitative") and cfg.analysis.qualitative:
        return cfg.analysis.qualitative
    return None
