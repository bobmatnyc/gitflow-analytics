"""Configuration management for GitFlow Analytics.

This module provides configuration loading, validation, and management
for the GitFlow Analytics tool. It has been refactored into focused
sub-modules while maintaining backward compatibility.
"""

# Re-export main interfaces for backward compatibility
from .aliases import AliasesManager, DeveloperAlias
from .loader import ConfigLoader
from .schema import (
    AnalysisConfig,
    BranchAnalysisConfig,
    CacheConfig,
    CommitClassificationConfig,
    Config,
    GitHubConfig,
    JIRAConfig,
    JIRAIntegrationConfig,
    LLMClassificationConfig,
    MLCategorization,
    OutputConfig,
    PMIntegrationConfig,
    PMPlatformConfig,
    RepositoryConfig,
)

__all__ = [
    "ConfigLoader",
    "Config",
    "RepositoryConfig",
    "GitHubConfig",
    "AnalysisConfig",
    "OutputConfig",
    "CacheConfig",
    "JIRAConfig",
    "JIRAIntegrationConfig",
    "PMPlatformConfig",
    "PMIntegrationConfig",
    "MLCategorization",
    "LLMClassificationConfig",
    "CommitClassificationConfig",
    "BranchAnalysisConfig",
    "AliasesManager",
    "DeveloperAlias",
]
