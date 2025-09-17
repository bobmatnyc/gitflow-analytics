"""PM Framework Orchestrator for multi-platform data collection and correlation.

This module provides the main orchestration layer that coordinates data collection
across multiple PM platforms and correlates issues with Git commits for unified
analytics.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from .base import BasePlatformAdapter
from .models import UnifiedIssue, UnifiedProject
from .registry import PlatformRegistry

# Configure logger for orchestrator
logger = logging.getLogger(__name__)


class PMFrameworkOrchestrator:
    """Orchestrates data collection across multiple PM platforms.

    WHY: Different organizations use different combinations of PM tools.
    The orchestrator provides a unified interface for collecting data from
    multiple platforms simultaneously and correlating that data with Git
    commits for comprehensive analytics.

    DESIGN DECISION: Use orchestrator pattern to coordinate multiple adapters
    rather than requiring callers to manage individual adapters. This provides
    a clean API and handles cross-platform data correlation logic.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the PM framework orchestrator.

        Args:
            config: Configuration dictionary containing PM platform settings.
                   Expected format:
                   {
                       'pm_platforms': {
                           'platform_name': {
                               'enabled': bool,
                               'config_key': 'config_value',
                               ...
                           }
                       },
                       'analysis': {
                           'pm_integration': {
                               'enabled': bool,
                               'primary_platform': str,
                               'correlation': {...}
                           }
                       }
                   }
        """
        logger.info("Initializing PM Framework Orchestrator...")
        logger.debug(f"PM platforms in config: {list(config.get('pm_platforms', {}).keys())}")

        self.config = config
        # Create a new registry instance for this orchestrator
        # This ensures we don't use the default registry which may lack credentials
        self.registry = PlatformRegistry()
        self.adapters: dict[str, BasePlatformAdapter] = {}

        # Register built-in adapters (to be implemented in future)
        self._register_builtin_adapters()

        # Configuration for PM integration
        pm_config = config.get("analysis", {}).get("pm_integration", {})
        self.pm_integration_enabled = pm_config.get("enabled", False)
        self.primary_platform = pm_config.get("primary_platform", None)

        logger.info(f"PM integration enabled: {self.pm_integration_enabled}")
        if self.primary_platform:
            logger.info(f"Primary platform: {self.primary_platform}")

        # Correlation settings
        correlation_config = pm_config.get("correlation", {})
        self.fuzzy_matching_enabled = correlation_config.get("fuzzy_matching", True)
        self.temporal_window_hours = correlation_config.get("temporal_window_hours", 72)
        self.confidence_threshold = correlation_config.get("confidence_threshold", 0.8)

        import os
        import traceback

        logger.info("PM Framework Orchestrator initialized")

        # Only show debug messages when GITFLOW_DEBUG is set
        debug_mode = os.getenv("GITFLOW_DEBUG", "").lower() in ("1", "true", "yes")
        if debug_mode:
            print("   🔍 PM Framework init stack trace:")
            for line in traceback.format_stack()[-5:-1]:
                print("   " + line.strip())

        # Initialize configured platforms if PM integration is enabled
        if self.pm_integration_enabled:
            self._initialize_platforms()
        else:
            logger.info("PM integration disabled, skipping platform initialization")

    def _register_builtin_adapters(self) -> None:
        """Register built-in platform adapters.

        WHY: Built-in adapters should be automatically available without
        requiring manual registration. Future implementations will add
        JIRA, Azure DevOps, Linear, and other adapters here.
        """
        logger.debug("Registering built-in platform adapters...")

        # Register available adapters
        from .adapters import JIRAAdapter

        self.registry.register_adapter("jira", JIRAAdapter)
        logger.debug("Registered JIRA adapter")

        # self.registry.register_adapter('azure_devops', AzureDevOpsAdapter)
        # self.registry.register_adapter('linear', LinearAdapter)
        # self.registry.register_adapter('asana', AsanaAdapter)

        available_platforms = self.registry.get_available_platforms()
        logger.info(f"Built-in adapters registered: {available_platforms}")

    def _initialize_platforms(self) -> None:
        """Initialize platform adapters based on configuration.

        WHY: Automated initialization reduces setup complexity and ensures
        all configured platforms are ready for data collection. Failed
        initializations are logged but don't prevent other platforms from
        working (graceful degradation).
        """
        platforms_config = self.config.get("pm_platforms", {})

        if not platforms_config:
            logger.warning("No PM platforms configured")
            return

        initialization_results = []

        for platform_name, platform_config in platforms_config.items():
            if not platform_config.get("enabled", False):
                logger.info(f"Platform {platform_name} disabled, skipping initialization")
                continue

            # Log configuration details for debugging (without credentials)
            config_keys = list(platform_config.keys())
            has_credentials = (
                "username" in platform_config and "api_token" in platform_config
            ) or ("access_user" in platform_config and "access_token" in platform_config)
            logger.debug(
                f"Platform {platform_name} config keys: {config_keys}, has credentials: {has_credentials}"
            )

            try:
                logger.info(f"Initializing {platform_name} adapter...")
                adapter = self.registry.create_adapter(platform_name, platform_config)
                self.adapters[platform_name] = adapter

                # Test adapter capabilities
                connection_info = adapter.test_connection()
                logger.info(
                    f"✅ {platform_name} initialized: {connection_info.get('status', 'unknown')}"
                )

                initialization_results.append(
                    {
                        "platform": platform_name,
                        "status": "success",
                        "connection_info": connection_info,
                    }
                )

            except Exception as e:
                error_msg = f"Failed to initialize {platform_name}: {e}"
                logger.error(f"❌ {error_msg}")
                logger.debug(f"Full error details for {platform_name}: {e}", exc_info=True)

                initialization_results.append(
                    {"platform": platform_name, "status": "error", "error": str(e)}
                )

        # Log initialization summary
        successful = sum(1 for r in initialization_results if r["status"] == "success")
        total = len(initialization_results)

        if successful > 0:
            logger.info(f"Successfully initialized {successful}/{total} PM platforms")
        else:
            logger.warning("No PM platforms successfully initialized")

    def is_enabled(self) -> bool:
        """Check if PM integration is enabled and has active adapters.

        Returns:
            True if PM integration is enabled and at least one adapter is active.
        """
        return self.pm_integration_enabled and len(self.adapters) > 0

    def get_active_platforms(self) -> list[str]:
        """Get list of active platform names.

        Returns:
            List of platform identifiers that are successfully initialized.
        """
        return list(self.adapters.keys())

    def get_platform_status(self) -> dict[str, Any]:
        """Get status information for all platforms.

        WHY: Provides diagnostic information for monitoring and troubleshooting
        platform connections and configuration issues.

        Returns:
            Dictionary containing status for each platform and overall summary.
        """
        status = {
            "pm_integration_enabled": self.pm_integration_enabled,
            "primary_platform": self.primary_platform,
            "active_platforms": len(self.adapters),
            "platforms": {},
        }

        for platform_name, adapter in self.adapters.items():
            try:
                connection_info = adapter.test_connection()
                capabilities = adapter.capabilities

                status["platforms"][platform_name] = {
                    "status": connection_info.get("status", "unknown"),
                    "platform_type": adapter.platform_name,
                    "capabilities": {
                        "supports_sprints": capabilities.supports_sprints,
                        "supports_story_points": capabilities.supports_story_points,
                        "supports_time_tracking": capabilities.supports_time_tracking,
                        "rate_limit_per_hour": capabilities.rate_limit_requests_per_hour,
                    },
                    "connection_info": connection_info,
                }
            except Exception as e:
                status["platforms"][platform_name] = {"status": "error", "error": str(e)}

        return status

    def get_all_projects(self) -> dict[str, list[UnifiedProject]]:
        """Get projects from all configured platforms.

        WHY: Projects are the primary organizational unit in PM platforms.
        This method discovers all accessible projects across platforms for
        subsequent issue retrieval and project-level analytics.

        Returns:
            Dictionary mapping platform names to lists of UnifiedProject objects.
        """
        all_projects = {}

        for platform_name, adapter in self.adapters.items():
            try:
                logger.info(f"Fetching projects from {platform_name}...")
                projects = adapter.get_projects()
                all_projects[platform_name] = projects
                logger.info(f"📁 Found {len(projects)} projects in {platform_name}")

            except Exception as e:
                logger.error(f"⚠️ Failed to get projects from {platform_name}: {e}")
                all_projects[platform_name] = []

        total_projects = sum(len(projects) for projects in all_projects.values())
        logger.info(f"Total projects discovered: {total_projects}")

        return all_projects

    def get_all_issues(
        self,
        since: Optional[datetime] = None,
        project_filter: Optional[dict[str, list[str]]] = None,
    ) -> dict[str, list[UnifiedIssue]]:
        """Get issues from all configured platforms.

        WHY: Issues are the core work items that need to be correlated with
        Git commits. This method collects issues from all platforms with
        optional filtering to optimize performance and focus on relevant data.

        Args:
            since: Optional datetime to filter issues updated after this date.
            project_filter: Optional dict mapping platform names to lists of
                          project keys to filter by. Format:
                          {'jira': ['PROJ1', 'PROJ2'], 'azure': ['Project1']}

        Returns:
            Dictionary mapping platform names to lists of UnifiedIssue objects.
        """
        all_issues = {}

        for platform_name, adapter in self.adapters.items():
            try:
                logger.info(f"Fetching issues from {platform_name}...")
                platform_issues = []

                # Get projects for this platform
                projects = adapter.get_projects()

                # Apply project filter if specified
                if project_filter and platform_name in project_filter:
                    project_keys = project_filter[platform_name]
                    projects = [p for p in projects if p.key in project_keys]
                    logger.info(f"Filtered to {len(projects)} projects for {platform_name}")

                # Get issues for each project
                for project in projects:
                    try:
                        logger.debug(f"Fetching issues from {platform_name}/{project.key}")
                        issues = adapter.get_issues(project.key, since)
                        platform_issues.extend(issues)
                        logger.info(
                            f"🎫 Found {len(issues)} issues in {platform_name}/{project.key}"
                        )

                    except Exception as e:
                        logger.error(
                            f"⚠️ Failed to get issues from {platform_name}/{project.key}: {e}"
                        )

                all_issues[platform_name] = platform_issues
                logger.info(f"Total issues from {platform_name}: {len(platform_issues)}")

            except Exception as e:
                logger.error(f"⚠️ Failed to get issues from {platform_name}: {e}")
                all_issues[platform_name] = []

        total_issues = sum(len(issues) for issues in all_issues.values())
        logger.info(f"Total issues collected: {total_issues}")

        return all_issues

    def get_issues_by_keys(self, platform: str, issue_keys: list[str]) -> dict[str, UnifiedIssue]:
        """Get specific issues by their keys from a platform.

        WHY: Training pipeline needs to fetch specific issues referenced in commits
        to determine their types for classification labeling.

        Args:
            platform: Platform name (e.g., 'jira')
            issue_keys: List of issue keys to fetch

        Returns:
            Dictionary mapping issue keys to UnifiedIssue objects.
        """
        if platform not in self.adapters:
            # Don't log errors for non-configured platforms - this is expected
            logger.debug(f"Platform {platform} not configured, skipping")
            return {}

        adapter = self.adapters[platform]
        issues_dict = {}

        # For JIRA, we can fetch issues directly by key
        if platform == "jira" and hasattr(adapter, "get_issue_by_key"):
            for key in issue_keys:
                try:
                    issue = adapter.get_issue_by_key(key)
                    if issue:
                        issues_dict[key] = issue
                except Exception as e:
                    logger.warning(f"Failed to fetch {key} from {platform}: {e}")
        else:
            # For other platforms, we may need to use search or other methods
            logger.warning(f"Batch fetch by keys not implemented for {platform}")

        return issues_dict

    def correlate_issues_with_commits(
        self, issues: dict[str, list[UnifiedIssue]], commits: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Correlate PM platform issues with Git commits.

        WHY: The core value of PM platform integration is correlating work
        items with actual code changes. This enables tracking story point
        accuracy, development velocity, and work item completion metrics.

        DESIGN DECISION: Use multiple correlation strategies (ticket references,
        fuzzy matching, temporal correlation) to maximize correlation accuracy
        while maintaining confidence scoring for quality assessment.

        Args:
            issues: Dictionary mapping platform names to lists of issues.
            commits: List of Git commit dictionaries with metadata.

        Returns:
            List of correlation dictionaries containing matched issues and commits.
        """
        if not issues or not commits:
            logger.warning("No issues or commits provided for correlation")
            return []

        logger.info(
            f"Correlating {sum(len(i) for i in issues.values())} issues with {len(commits)} commits"
        )

        correlations = []

        # Build a lookup of all issues by key for efficient searching
        issue_lookup = {}
        for platform_issues in issues.values():
            for issue in platform_issues:
                issue_lookup[issue.key] = issue

        logger.debug(f"Built issue lookup with {len(issue_lookup)} issues")

        # Strategy 1: Direct ticket reference correlation
        direct_correlations = self._correlate_by_ticket_references(issue_lookup, commits)
        correlations.extend(direct_correlations)

        # Strategy 2: Fuzzy matching correlation (if enabled)
        if self.fuzzy_matching_enabled:
            fuzzy_correlations = self._correlate_by_fuzzy_matching(
                issue_lookup, commits, direct_correlations
            )
            correlations.extend(fuzzy_correlations)

        # Strategy 3: Temporal correlation for bug fixes (future enhancement)
        # temporal_correlations = self._correlate_by_temporal_proximity(issue_lookup, commits)
        # correlations.extend(temporal_correlations)

        # Remove duplicates while preserving highest confidence matches
        unique_correlations = self._deduplicate_correlations(correlations)

        logger.info(f"Found {len(unique_correlations)} issue-commit correlations")

        return unique_correlations

    def _correlate_by_ticket_references(
        self, issue_lookup: dict[str, UnifiedIssue], commits: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Correlate issues with commits based on explicit ticket references.

        WHY: Explicit ticket references in commit messages are the most reliable
        correlation method. This strategy matches issues using existing ticket
        extraction from GitFlow Analytics.

        Args:
            issue_lookup: Dictionary mapping issue keys to UnifiedIssue objects.
            commits: List of commit dictionaries.

        Returns:
            List of correlation dictionaries for ticket reference matches.
        """
        correlations = []

        for commit in commits:
            # Check existing ticket references (from GitFlow Analytics ticket extractor)
            ticket_refs = commit.get("ticket_references", [])

            for ref in ticket_refs:
                # Handle both dict and string formats from ticket extractor
                if isinstance(ref, dict):
                    ticket_key = ref.get("id", "")
                    full_id = ref.get("full_id", ticket_key)
                else:
                    ticket_key = str(ref)
                    full_id = ticket_key

                # Try to find issue in our collected data
                issue = issue_lookup.get(ticket_key)
                if not issue and full_id:
                    issue = issue_lookup.get(full_id)

                if issue:
                    correlation = {
                        "commit_hash": commit["hash"],
                        "commit_message": commit.get("message", "").split("\n")[0][:100],
                        "commit_author": commit.get("author", ""),
                        "commit_date": commit.get("date"),
                        "issue_key": issue.key,
                        "issue_title": issue.title,
                        "issue_type": issue.issue_type.value,
                        "issue_status": issue.status.value,
                        "issue_platform": issue.platform,
                        "story_points": issue.story_points,
                        "correlation_method": "ticket_reference",
                        "confidence": 1.0,  # Highest confidence for explicit references
                        "matched_text": full_id,
                    }
                    correlations.append(correlation)

                    logger.debug(f"Direct correlation: {commit['hash'][:8]} → {issue.key}")

        logger.info(f"Found {len(correlations)} direct ticket reference correlations")
        return correlations

    def _correlate_by_fuzzy_matching(
        self,
        issue_lookup: dict[str, UnifiedIssue],
        commits: list[dict[str, Any]],
        existing_correlations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Correlate issues with commits using fuzzy text matching.

        WHY: Not all commits have explicit ticket references, but may mention
        issue titles or keywords. Fuzzy matching can find additional correlations
        while maintaining confidence scoring to filter low-quality matches.

        Args:
            issue_lookup: Dictionary mapping issue keys to UnifiedIssue objects.
            commits: List of commit dictionaries.
            existing_correlations: Already found correlations to avoid duplicates.

        Returns:
            List of correlation dictionaries for fuzzy matches.
        """
        # TODO: Implement fuzzy matching correlation
        # This would use techniques like:
        # 1. TF-IDF similarity between commit messages and issue titles
        # 2. Keyword extraction and matching
        # 3. Semantic similarity using embeddings (optional)

        logger.debug("Fuzzy matching correlation not yet implemented")
        return []

    def _deduplicate_correlations(self, correlations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate correlations while preserving highest confidence matches.

        WHY: Multiple correlation strategies may find the same issue-commit
        pairs. We need to deduplicate while preserving the highest confidence
        match for each unique pair.

        Args:
            correlations: List of correlation dictionaries potentially containing duplicates.

        Returns:
            List of unique correlations with highest confidence matches preserved.
        """
        # Group correlations by (commit_hash, issue_key) pair
        correlation_groups: dict[tuple[str, str], dict[str, Any]] = {}

        for correlation in correlations:
            key = (correlation["commit_hash"], correlation["issue_key"])
            confidence = correlation.get("confidence", 0.0)

            if key not in correlation_groups or confidence > correlation_groups[key]["confidence"]:
                correlation_groups[key] = correlation

        unique_correlations = list(correlation_groups.values())

        if len(unique_correlations) < len(correlations):
            removed = len(correlations) - len(unique_correlations)
            logger.debug(f"Removed {removed} duplicate correlations")

        return unique_correlations

    def calculate_enhanced_metrics(
        self,
        commits: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        pm_issues: dict[str, list[UnifiedIssue]],
        correlations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate metrics enhanced with PM platform data.

        WHY: PM platform integration enables new metrics that combine Git data
        with work item information. These metrics provide insights into story
        point accuracy, cross-platform coverage, and development efficiency.

        Args:
            commits: List of Git commit dictionaries.
            prs: List of pull request dictionaries.
            pm_issues: Dictionary mapping platforms to lists of issues.
            correlations: List of issue-commit correlations.

        Returns:
            Dictionary containing enhanced metrics with PM platform data.
        """
        # Initialize metrics dictionary
        metrics: dict[str, Any] = {}

        # Cross-platform issue metrics
        total_issues = sum(len(issues) for issues in pm_issues.values())
        metrics["total_pm_issues"] = total_issues

        # Story point analysis
        pm_story_points = 0
        issues_with_story_points = 0

        for platform_issues in pm_issues.values():
            for issue in platform_issues:
                if issue.story_points:
                    pm_story_points += issue.story_points
                    issues_with_story_points += 1

        git_story_points = sum(commit.get("story_points", 0) or 0 for commit in commits)

        metrics["story_point_analysis"] = {
            "pm_total_story_points": pm_story_points,
            "git_total_story_points": git_story_points,
            "issues_with_story_points": issues_with_story_points,
            "story_point_coverage_pct": (
                (issues_with_story_points / total_issues * 100) if total_issues > 0 else 0
            ),
            "correlation_accuracy": (
                min(git_story_points / pm_story_points, 1.0) if pm_story_points > 0 else 0
            ),
        }

        # Issue type distribution
        issue_types: dict[str, int] = {}
        for platform_issues in pm_issues.values():
            for issue in platform_issues:
                issue_type = issue.issue_type.value
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1

        metrics["issue_type_distribution"] = issue_types

        # Platform coverage analysis
        platform_coverage: dict[str, dict[str, Any]] = {}
        for platform, issues in pm_issues.items():
            linked_issues = [
                c["issue_key"] for c in correlations if c.get("issue_platform") == platform
            ]
            unique_linked = set(linked_issues)

            coverage_pct = len(unique_linked) / len(issues) * 100 if issues else 0

            platform_coverage[platform] = {
                "total_issues": len(issues),
                "linked_issues": len(unique_linked),
                "coverage_percentage": coverage_pct,
                "correlation_rate": (
                    len(linked_issues) / len(correlations) * 100 if correlations else 0
                ),
            }

        metrics["platform_coverage"] = platform_coverage

        # Correlation quality metrics
        if correlations:
            confidence_scores = [c.get("confidence", 0) for c in correlations]
            correlation_methods: dict[str, int] = {}

            for correlation in correlations:
                method = correlation.get("correlation_method", "unknown")
                correlation_methods[method] = correlation_methods.get(method, 0) + 1

            metrics["correlation_quality"] = {
                "total_correlations": len(correlations),
                "average_confidence": sum(confidence_scores) / len(confidence_scores),
                "high_confidence_correlations": sum(
                    1 for score in confidence_scores if score >= self.confidence_threshold
                ),
                "correlation_methods": correlation_methods,
            }
        else:
            metrics["correlation_quality"] = {
                "total_correlations": 0,
                "average_confidence": 0,
                "high_confidence_correlations": 0,
                "correlation_methods": {},
            }

        return metrics
