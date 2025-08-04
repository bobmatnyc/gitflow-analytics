"""Integration orchestrator for multiple platforms."""

import json
from datetime import datetime
from typing import Any, Union

from ..core.cache import GitAnalysisCache
from ..pm_framework.orchestrator import PMFrameworkOrchestrator
from .github_integration import GitHubIntegration
from .jira_integration import JIRAIntegration


class IntegrationOrchestrator:
    """Orchestrate integrations with multiple platforms."""

    def __init__(self, config: Any, cache: GitAnalysisCache):
        """Initialize integration orchestrator."""
        self.config = config
        self.cache = cache
        self.integrations: dict[str, Union[GitHubIntegration, JIRAIntegration]] = {}

        # Initialize available integrations
        if config.github and config.github.token:
            self.integrations["github"] = GitHubIntegration(
                config.github.token,
                cache,
                config.github.max_retries,
                config.github.backoff_factor,
                allowed_ticket_platforms=getattr(config.analysis, "ticket_platforms", None),
            )

        # Initialize JIRA integration if configured
        if config.jira and config.jira.access_user and config.jira.access_token:
            # Get JIRA specific settings if available
            jira_settings = getattr(config, "jira_integration", {})
            if hasattr(jira_settings, "enabled") and jira_settings.enabled:
                base_url = getattr(config.jira, "base_url", None)
                if base_url:
                    self.integrations["jira"] = JIRAIntegration(
                        base_url,
                        config.jira.access_user,
                        config.jira.access_token,
                        cache,
                        story_point_fields=getattr(jira_settings, "story_point_fields", None),
                    )

        # Initialize PM framework orchestrator
        self.pm_orchestrator = None
        if hasattr(config, 'pm_integration') and config.pm_integration and config.pm_integration.enabled:
            try:
                # Create PM platform configuration for the orchestrator
                pm_config = {
                    'pm_platforms': {},
                    'analysis': {
                        'pm_integration': {
                            'enabled': config.pm_integration.enabled,
                            'primary_platform': config.pm_integration.primary_platform,
                            'correlation': config.pm_integration.correlation
                        }
                    }
                }
                
                # Convert PM platform configs to expected format
                for platform_name, platform_config in config.pm_integration.platforms.items():
                    if platform_config.enabled:
                        pm_config['pm_platforms'][platform_name] = {
                            'enabled': True,
                            **platform_config.config
                        }
                
                self.pm_orchestrator = PMFrameworkOrchestrator(pm_config)
                print(f"📋 PM Framework initialized with {len(self.pm_orchestrator.get_active_platforms())} platforms")
                
            except Exception as e:
                print(f"⚠️  Failed to initialize PM framework: {e}")
                self.pm_orchestrator = None

    def enrich_repository_data(
        self, repo_config: Any, commits: list[dict[str, Any]], since: datetime
    ) -> dict[str, Any]:
        """Enrich repository data from all available integrations."""
        enrichment: dict[str, Any] = {"prs": [], "issues": [], "pr_metrics": {}, "pm_data": {}}

        # GitHub enrichment
        if "github" in self.integrations and repo_config.github_repo:
            github_integration = self.integrations["github"]
            if isinstance(github_integration, GitHubIntegration):
                try:
                    # Get PR data
                    prs = github_integration.enrich_repository_with_prs(
                        repo_config.github_repo, commits, since
                    )
                    enrichment["prs"] = prs

                    # Calculate PR metrics
                    if prs:
                        enrichment["pr_metrics"] = github_integration.calculate_pr_metrics(prs)

                except Exception as e:
                    print(f"   ⚠️  GitHub enrichment failed: {e}")

        # JIRA enrichment for story points
        if "jira" in self.integrations:
            jira_integration = self.integrations["jira"]
            if isinstance(jira_integration, JIRAIntegration):
                try:
                    # Enrich commits with JIRA story points
                    jira_integration.enrich_commits_with_jira_data(commits)

                    # Enrich PRs with JIRA story points
                    if enrichment["prs"]:
                        jira_integration.enrich_prs_with_jira_data(enrichment["prs"])

                except Exception as e:
                    print(f"   ⚠️  JIRA enrichment failed: {e}")

        # PM Framework enrichment
        if self.pm_orchestrator and self.pm_orchestrator.is_enabled():
            try:
                print(f"   📋 Collecting PM platform data...")
                
                # Get all issues from PM platforms
                pm_issues = self.pm_orchestrator.get_all_issues(since=since)
                enrichment["pm_data"]["issues"] = pm_issues
                
                # Correlate issues with commits
                correlations = self.pm_orchestrator.correlate_issues_with_commits(pm_issues, commits)
                enrichment["pm_data"]["correlations"] = correlations
                
                # Calculate enhanced metrics
                enhanced_metrics = self.pm_orchestrator.calculate_enhanced_metrics(
                    commits, enrichment["prs"], pm_issues, correlations
                )
                enrichment["pm_data"]["metrics"] = enhanced_metrics
                
                print(f"   ✅ PM data collected: {enhanced_metrics.get('total_pm_issues', 0)} issues, {len(correlations)} correlations")
                
            except Exception as e:
                print(f"   ⚠️  PM framework enrichment failed: {e}")
                enrichment["pm_data"] = {"error": str(e)}

        return enrichment

    def get_platform_issues(self, project_key: str, since: datetime) -> list[dict[str, Any]]:
        """Get issues from all configured platforms."""
        all_issues: list[dict[str, Any]] = []

        # Check cache first
        cached_issues = []
        for platform in ["github", "jira", "clickup", "linear"]:
            cached = self.cache.get_cached_issues(platform, project_key)
            cached_issues.extend(cached)

        if cached_issues:
            return cached_issues

        # Future: Fetch from APIs if not cached
        # This is where we'd add actual API calls to each platform

        return all_issues

    def export_to_json(
        self,
        commits: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        developer_stats: list[dict[str, Any]],
        project_metrics: dict[str, Any],
        dora_metrics: dict[str, Any],
        output_path: str,
    ) -> str:
        """Export all data to JSON format for API consumption."""

        # Prepare data for JSON serialization
        def serialize_dates(obj: Any) -> Any:
            """Convert datetime objects to ISO format strings."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: serialize_dates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_dates(item) for item in obj]
            return obj

        export_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0",
                "total_commits": len(commits),
                "total_prs": len(prs),
                "total_developers": len(developer_stats),
            },
            "commits": serialize_dates(commits),
            "pull_requests": serialize_dates(prs),
            "developers": serialize_dates(developer_stats),
            "project_metrics": serialize_dates(project_metrics),
            "dora_metrics": serialize_dates(dora_metrics),
        }

        # Write JSON file
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        return output_path
