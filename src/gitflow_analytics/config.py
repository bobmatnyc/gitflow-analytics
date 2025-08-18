"""Configuration management for GitFlow Analytics.

This module maintains backward compatibility by re-exporting all
configuration classes and the ConfigLoader from the refactored
config submodules.
"""

# Re-export everything from the new modular structure for backward compatibility
from .config import (
    AnalysisConfig,
    BranchAnalysisConfig,
    CacheConfig,
    CommitClassificationConfig,
    Config,
    ConfigLoader,
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

# Export all public interfaces
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
]
