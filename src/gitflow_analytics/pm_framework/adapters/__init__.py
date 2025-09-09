"""Platform-specific adapters for PM framework.

This package contains concrete implementations of BasePlatformAdapter
for different project management platforms. Each adapter translates between
platform-specific APIs and the unified data model.

Available Adapters:
- JIRAAdapter: JIRA Cloud and Server integration

Planned Adapters:
- AzureDevOpsAdapter: Azure DevOps Services and Server integration
- LinearAdapter: Linear workspace integration
- AsanaAdapter: Asana project integration
- GitHubIssuesAdapter: GitHub Issues integration
- ClickUpAdapter: ClickUp workspace integration

Example Adapter Registration:
    from gitflow_analytics.pm_framework import PlatformRegistry
    from .jira_adapter import JIRAAdapter

    registry = PlatformRegistry()
    registry.register_adapter('jira', JIRAAdapter)

The framework architecture is designed to support easy addition of new
platform adapters without modifying core framework code.
"""

# Import available adapters
from .jira_adapter import JIRAAdapter

# Placeholder for future adapter imports
# from .azure_devops_adapter import AzureDevOpsAdapter
# from .linear_adapter import LinearAdapter
# from .asana_adapter import AsanaAdapter
# from .github_issues_adapter import GitHubIssuesAdapter
# from .clickup_adapter import ClickUpAdapter

__all__: list[str] = [
    "JIRAAdapter",
    # Platform adapters will be added here as they are implemented
]

# Adapter development guidelines:
# 1. Inherit from BasePlatformAdapter
# 2. Implement all abstract methods
# 3. Handle platform-specific authentication
# 4. Map platform data to unified models
# 5. Include comprehensive error handling
# 6. Follow the logging patterns established in base classes
# 7. Add unit tests for all adapter functionality
