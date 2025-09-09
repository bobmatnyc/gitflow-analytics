"""Qualitative data extraction module for GitFlow Analytics.

This module provides NLP-based analysis of Git commits to extract semantic meaning,
change types, domain classification, and risk assessment from commit messages and
file changes.

Key Components:
- QualitativeProcessor: Main orchestrator for qualitative analysis
- EnhancedQualitativeAnalyzer: Advanced multi-dimensional analysis for executives, projects, developers, and workflows
- NLPEngine: spaCy-based fast processing for most commits
- LLMFallback: Strategic use of LLMs for uncertain cases
- Various classifiers for change type, domain, risk, and intent analysis
"""

from .core.processor import QualitativeProcessor

# from .enhanced_analyzer import EnhancedQualitativeAnalyzer  # Commented out - missing dependencies
from .models.schemas import CacheConfig as QualitativeCacheConfig
from .models.schemas import LLMConfig, NLPConfig, QualitativeCommitData, QualitativeConfig

__all__ = [
    "QualitativeProcessor",
    # "EnhancedQualitativeAnalyzer",  # Commented out - missing dependencies
    "QualitativeCommitData",
    "QualitativeConfig",
    "NLPConfig",
    "LLMConfig",
    "QualitativeCacheConfig",
]
