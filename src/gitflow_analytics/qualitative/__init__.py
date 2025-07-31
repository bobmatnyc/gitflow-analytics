"""Qualitative data extraction module for GitFlow Analytics.

This module provides NLP-based analysis of Git commits to extract semantic meaning,
change types, domain classification, and risk assessment from commit messages and
file changes.

Key Components:
- QualitativeProcessor: Main orchestrator for qualitative analysis
- NLPEngine: spaCy-based fast processing for most commits  
- LLMFallback: Strategic use of LLMs for uncertain cases
- Various classifiers for change type, domain, risk, and intent analysis
"""

from .core.processor import QualitativeProcessor
from .models.schemas import (
    QualitativeCommitData,
    QualitativeConfig,
    NLPConfig,
    LLMConfig,
    CacheConfig as QualitativeCacheConfig,
)

__all__ = [
    "QualitativeProcessor",
    "QualitativeCommitData", 
    "QualitativeConfig",
    "NLPConfig",
    "LLMConfig",
    "QualitativeCacheConfig",
]