"""Security analysis module for GitFlow Analytics.

This module provides comprehensive security analysis of git commits using a hybrid approach:
1. Specialized security tools (Semgrep, Bandit, etc.) for known patterns
2. LLM analysis for novel vulnerabilities and context-aware security review
"""

from .config import SecurityConfig
from .security_analyzer import SecurityAnalyzer

__all__ = ["SecurityAnalyzer", "SecurityConfig"]
