"""Platform-agnostic project management framework for GitFlow Analytics.

This module provides a unified interface for integrating with multiple PM platforms
(JIRA, Azure DevOps, Linear, Asana, etc.) to collect work item data and correlate
it with Git commits for comprehensive development analytics.

Key Components:
- UnifiedIssue, UnifiedProject, UnifiedSprint: Standard data models
- BasePlatformAdapter: Abstract base class for platform adapters
- PlatformRegistry: Manages adapter registration and instantiation
- PMFrameworkOrchestrator: Coordinates multi-platform data collection

Example Usage:
    from gitflow_analytics.pm_framework import PMFrameworkOrchestrator

    config = {
        'pm_platforms': {
            'jira': {
                'enabled': True,
                'base_url': 'https://company.atlassian.net',
                'username': 'user@company.com',
                'api_token': 'token'
            }
        },
        'analysis': {
            'pm_integration': {
                'enabled': True,
                'primary_platform': 'jira'
            }
        }
    }

    orchestrator = PMFrameworkOrchestrator(config)
    if orchestrator.is_enabled():
        issues = orchestrator.get_all_issues(since=datetime.now() - timedelta(weeks=12))
        correlations = orchestrator.correlate_issues_with_commits(issues, commits)
        metrics = orchestrator.calculate_enhanced_metrics(commits, prs, issues, correlations)
"""

# Import available adapters
from .adapters import JIRAAdapter
from .base import BasePlatformAdapter, PlatformCapabilities
from .models import (
    IssueStatus,
    IssueType,
    Priority,
    UnifiedIssue,
    UnifiedProject,
    UnifiedSprint,
    UnifiedUser,
)
from .orchestrator import PMFrameworkOrchestrator
from .registry import PlatformRegistry

# Lazy initialization - registry created on first access
_default_registry = None


def get_default_registry() -> PlatformRegistry:
    """Get the default platform registry with built-in adapters registered.

    WHY: Provides a convenient way to access a pre-configured registry with
    all available adapters already registered. Uses lazy initialization to
    avoid creating registry instances at import time before credentials are
    available.

    DESIGN DECISION: Lazy initialization prevents authentication issues in
    training pipeline where imports happen before configuration is loaded.
    The registry is created when first accessed, ensuring credentials are
    available from the orchestrator configuration.

    Returns:
        PlatformRegistry instance with built-in adapters registered.
    """
    global _default_registry

    if _default_registry is None:
        import logging

        logger = logging.getLogger(__name__)
        logger.debug("Initializing default PM platform registry (lazy initialization)")

        _default_registry = PlatformRegistry()
        _default_registry.register_adapter("jira", JIRAAdapter)

        logger.debug("Default registry initialized with built-in adapters")

    return _default_registry


__all__ = [
    # Core orchestration
    "PMFrameworkOrchestrator",
    "PlatformRegistry",
    "get_default_registry",
    # Base classes for adapter development
    "BasePlatformAdapter",
    "PlatformCapabilities",
    # Available adapters
    "JIRAAdapter",
    # Unified data models
    "UnifiedIssue",
    "UnifiedProject",
    "UnifiedSprint",
    "UnifiedUser",
    # Enums for standardized values
    "IssueType",
    "IssueStatus",
    "Priority",
]

# Version information
__version__ = "1.0.0"
__author__ = "GitFlow Analytics Team"
__description__ = "Platform-agnostic project management integration framework"
