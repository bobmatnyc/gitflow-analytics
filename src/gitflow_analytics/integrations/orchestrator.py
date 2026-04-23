"""Integration orchestrator for multiple platforms."""

import json
import logging
from datetime import datetime
from typing import Any, Union

from ..core.cache import GitAnalysisCache
from ..pm_framework.orchestrator import PMFrameworkOrchestrator
from ..utils.debug import is_debug_mode
from .cicd.github_actions import GitHubActionsIntegration
from .confluence_integration import ConfluenceIntegration  # type: ignore[import-untyped]
from .github_integration import GitHubIntegration
from .github_issues_integration import GitHubIssuesIntegration
from .jira_activity_integration import JIRAActivityIntegration  # type: ignore[import-untyped]
from .jira_integration import JIRAIntegration

logger = logging.getLogger(__name__)


class IntegrationOrchestrator:
    """Orchestrate integrations with multiple platforms."""

    def __init__(self, config: Any, cache: GitAnalysisCache):
        """Initialize integration orchestrator."""
        self.debug_mode = is_debug_mode()
        if self.debug_mode:
            print("   🔍 IntegrationOrchestrator.__init__ called")
        self.config = config
        self.cache = cache
        self.integrations: dict[
            str,
            Union[
                GitHubIntegration,
                JIRAIntegration,
                GitHubIssuesIntegration,
                ConfluenceIntegration,
                JIRAActivityIntegration,
            ],
        ] = {}
        self._jira_activity_fetched = False

        # Initialize available integrations
        if config.github and config.github.token:
            self.integrations["github"] = GitHubIntegration(
                config.github.token,
                cache,
                config.github.max_retries,
                config.github.backoff_factor,
                allowed_ticket_platforms=getattr(config.analysis, "ticket_platforms", None),
                fetch_pr_reviews=getattr(config.github, "fetch_pr_reviews", False),
            )

        # Initialize JIRA integration if configured
        if config.jira and config.jira.access_user and config.jira.access_token:
            # Get JIRA specific settings if available
            jira_settings = getattr(config, "jira_integration", {})
            jira_enabled = (
                jira_settings.get("enabled", False)
                if isinstance(jira_settings, dict)
                else getattr(jira_settings, "enabled", False)
            )
            if jira_enabled:
                base_url = getattr(config.jira, "base_url", None)
                if base_url:
                    # Extract network and proxy settings from jira_settings
                    self.integrations["jira"] = JIRAIntegration(
                        base_url,
                        config.jira.access_user,
                        config.jira.access_token,
                        cache,
                        story_point_fields=getattr(jira_settings, "story_point_fields", None),
                        dns_timeout=getattr(jira_settings, "dns_timeout", 10),
                        connection_timeout=getattr(jira_settings, "connection_timeout", 30),
                        max_retries=getattr(jira_settings, "max_retries", 3),
                        backoff_factor=getattr(jira_settings, "backoff_factor", 1.0),
                        enable_proxy=getattr(jira_settings, "enable_proxy", False),
                        proxy_url=getattr(jira_settings, "proxy_url", None),
                    )

        # Initialize JIRA activity integration (separate from ticket enrichment).
        # Activated whenever valid JIRA credentials + base_url exist, regardless
        # of jira_integration.enabled — activity tracking is additive.
        if (
            config.jira
            and getattr(config.jira, "access_user", None)
            and getattr(config.jira, "access_token", None)
            and getattr(config.jira, "base_url", None)
        ):
            try:
                jira_settings = getattr(config, "jira_integration", None)
                self.integrations["jira_activity"] = JIRAActivityIntegration(
                    base_url=config.jira.base_url,
                    username=config.jira.access_user,
                    api_token=config.jira.access_token,
                    cache=cache,
                    max_retries=getattr(jira_settings, "max_retries", 3) if jira_settings else 3,
                    backoff_factor=getattr(jira_settings, "backoff_factor", 1.0)
                    if jira_settings
                    else 1.0,
                )
                if self.debug_mode:
                    print("   ✅ JIRA Activity integration initialized")
            except Exception as e:  # noqa: BLE001
                if self.debug_mode:
                    print(f"   ⚠️  Failed to initialize JIRA Activity: {e}")

        # Initialize GitHub Issues integration (reuses GitHub token)
        github_issues_cfg = getattr(config, "github_issues", None)
        if (
            github_issues_cfg is not None
            and getattr(github_issues_cfg, "enabled", False)
            and config.github
            and config.github.token
        ):
            try:
                self.integrations["github_issues"] = GitHubIssuesIntegration(
                    token=config.github.token,
                    cache=cache,
                    fetch_comments=github_issues_cfg.fetch_comments,
                    allowed_repos=github_issues_cfg.allowed_repos,
                    issue_state=github_issues_cfg.issue_state,
                    max_issues_per_repo=github_issues_cfg.max_issues_per_repo,
                    rate_limit_retries=getattr(config.github, "max_retries", 3),
                    backoff_factor=float(getattr(config.github, "backoff_factor", 2)),
                )
                if self.debug_mode:
                    print("   ✅ GitHub Issues integration initialized")
            except Exception as e:
                if self.debug_mode:
                    print(f"   ⚠️  Failed to initialize GitHub Issues: {e}")

        # Initialize Confluence integration
        confluence_cfg = getattr(config, "confluence", None)
        self._confluence_fetched = False
        if (
            confluence_cfg is not None
            and getattr(confluence_cfg, "enabled", False)
            and getattr(confluence_cfg, "base_url", "")
        ):
            try:
                confluence_integration = ConfluenceIntegration(
                    base_url=confluence_cfg.base_url,
                    username=confluence_cfg.username,
                    api_token=confluence_cfg.api_token,
                    cache=cache,
                    spaces=confluence_cfg.spaces,
                    fetch_page_history=confluence_cfg.fetch_page_history,
                    dns_timeout=confluence_cfg.dns_timeout,
                    connection_timeout=confluence_cfg.connection_timeout,
                    max_retries=confluence_cfg.max_retries,
                    backoff_factor=confluence_cfg.backoff_factor,
                )

                # Pre-flight credential verification (issue #33): catch 401s
                # at init time so we can surface a clear error and disable
                # the integration rather than silently producing empty
                # reports later.
                try:
                    confluence_integration.verify_credentials()
                except RuntimeError as auth_err:
                    logger.warning("Disabling Confluence integration: %s", auth_err)
                    if self.debug_mode:
                        print(f"   ⚠️  Confluence disabled: {auth_err}")
                else:
                    self.integrations["confluence"] = confluence_integration
                    if self.debug_mode:
                        print("   ✅ Confluence integration initialized")
            except Exception as e:
                logger.warning("Failed to initialize Confluence: %s", e)
                if self.debug_mode:
                    print(f"   ⚠️  Failed to initialize Confluence: {e}")

        # Initialize CI/CD integrations
        self.cicd_integrations: dict[str, Any] = {}
        if hasattr(config, "cicd") and config.cicd and getattr(config.cicd, "enabled", False):
            if self.debug_mode:
                print("   🔍 CI/CD Integration detected - initializing platforms...")

            # GitHub Actions integration (reuses GitHub token)
            if (
                config.github
                and config.github.token
                and getattr(config.cicd, "github_actions_enabled", True)
            ):
                try:
                    self.cicd_integrations["github_actions"] = GitHubActionsIntegration(
                        token=config.github.token,
                        cache=cache,
                        rate_limit_retries=getattr(config.github, "max_retries", 3),
                        backoff_factor=getattr(config.github, "backoff_factor", 2),
                    )
                    if self.debug_mode:
                        print("   ✅ GitHub Actions CI/CD integration initialized")
                except Exception as e:
                    if self.debug_mode:
                        print(f"   ⚠️  Failed to initialize GitHub Actions: {e}")

        # Initialize PM framework orchestrator
        self.pm_orchestrator = None
        if (
            hasattr(config, "pm_integration")
            and config.pm_integration
            and config.pm_integration.enabled
        ):
            if self.debug_mode:
                print("   🔍 PM Integration detected - building configuration...")
            try:
                # Create PM platform configuration for the orchestrator
                pm_config = {
                    "pm_platforms": {},
                    "analysis": {
                        "pm_integration": {
                            "enabled": config.pm_integration.enabled,
                            "primary_platform": config.pm_integration.primary_platform,
                            "correlation": config.pm_integration.correlation,
                        }
                    },
                }

                # Convert PM platform configs to expected format
                platforms_dict = config.pm_integration.platforms
                if hasattr(platforms_dict, "__dict__"):
                    # It's an AttrDict, convert to regular dict
                    platforms_dict = dict(platforms_dict)

                for platform_name, platform_config in platforms_dict.items():
                    if hasattr(platform_config, "enabled") and platform_config.enabled:
                        # Convert AttrDict to regular dict
                        platform_config_dict = (
                            dict(platform_config.config)
                            if hasattr(platform_config.config, "__dict__")
                            else platform_config.config
                        )
                        platform_settings = {
                            "enabled": True,
                            **platform_config_dict,
                        }

                        # Special handling for JIRA - use credentials from top-level JIRA config
                        if platform_name == "jira" and hasattr(config, "jira") and config.jira:
                            # Safely access JIRA config attributes
                            if hasattr(config.jira, "access_user") and config.jira.access_user:
                                platform_settings["username"] = config.jira.access_user
                            if hasattr(config.jira, "access_token") and config.jira.access_token:
                                platform_settings["api_token"] = config.jira.access_token
                            # Also ensure base_url matches if not set
                            if (
                                not platform_settings.get("base_url")
                                or platform_settings["base_url"] == "will_be_set_at_runtime"
                            ) and hasattr(config.jira, "base_url"):
                                platform_settings["base_url"] = config.jira.base_url
                            # Add cache directory for ticket caching (config file directory)
                            if hasattr(config, "cache") and hasattr(config.cache, "directory"):
                                platform_settings["cache_dir"] = config.cache.directory
                            # Debug output to check credentials
                            if self.debug_mode:
                                print(
                                    f"   🔍 JIRA config: username={platform_settings['username']}, has_token={bool(platform_settings['api_token'])}, base_url={platform_settings['base_url']}, cache_dir={platform_settings.get('cache_dir', 'not_set')}"
                                )

                        pm_config["pm_platforms"][platform_name] = platform_settings

                # Debug output - show final PM config
                if self.debug_mode:
                    print(
                        f"   🔍 Final PM config platforms: {list(pm_config.get('pm_platforms', {}).keys())}"
                    )
                    for plat_name, plat_config in pm_config.get("pm_platforms", {}).items():
                        print(
                            f"   🔍 {plat_name}: enabled={plat_config.get('enabled')}, has_username={bool(plat_config.get('username'))}, has_token={bool(plat_config.get('api_token'))}"
                        )

                self.pm_orchestrator = PMFrameworkOrchestrator(pm_config)
                if self.debug_mode:
                    print(
                        f"📋 PM Framework initialized with {len(self.pm_orchestrator.get_active_platforms())} platforms"
                    )

            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️  Failed to initialize PM framework: {e}")
                self.pm_orchestrator = None

    def enrich_repository_data(
        self, repo_config: Any, commits: list[dict[str, Any]], since: datetime
    ) -> dict[str, Any]:
        """Enrich repository data from all available integrations."""
        enrichment: dict[str, Any] = {
            "prs": [],
            "issues": [],
            "pr_metrics": {},
            "pm_data": {},
            "cicd_data": {"pipelines": [], "metrics": {}},
        }

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
                    import traceback

                    if self.debug_mode:
                        print(f"   ⚠️  GitHub enrichment failed: {e}")
                        import traceback

                        print(f"   Debug traceback: {traceback.format_exc()}")

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
                    if self.debug_mode:
                        print(f"   ⚠️  JIRA enrichment failed: {e}")

        # GitHub Issues activity tracking (per-repo)
        if "github_issues" in self.integrations and repo_config.github_repo:
            gh_issues = self.integrations["github_issues"]
            if isinstance(gh_issues, GitHubIssuesIntegration):
                try:
                    gh_issues.fetch_issues_activity([repo_config.github_repo], since)
                except Exception as e:
                    if self.debug_mode:
                        print(f"   ⚠️  GitHub Issues enrichment failed: {e}")

        # JIRA activity tracking (once per run, scoped to configured project_keys)
        if "jira_activity" in self.integrations and not self._jira_activity_fetched:
            jira_act = self.integrations["jira_activity"]
            if isinstance(jira_act, JIRAActivityIntegration):
                try:
                    jira_settings = getattr(self.config, "jira_integration", None)
                    project_keys: list[str] = []
                    if jira_settings is not None:
                        project_keys = list(getattr(jira_settings, "project_keys", []) or [])
                    if project_keys:
                        jira_act.fetch_project_activity(  # type: ignore
                            project_keys, since, datetime.now(tz=None)
                        )
                    self._jira_activity_fetched = True
                except Exception as e:  # noqa: BLE001
                    if self.debug_mode:
                        print(f"   ⚠️  JIRA Activity enrichment failed: {e}")
                    self._jira_activity_fetched = True

        # Confluence activity tracking (once per run, not per-repo)
        if "confluence" in self.integrations and not self._confluence_fetched:
            conf = self.integrations["confluence"]
            if isinstance(conf, ConfluenceIntegration):
                try:
                    conf.fetch_all_spaces(since)  # type: ignore[union-attr]
                    self._confluence_fetched = True
                except Exception as e:
                    if self.debug_mode:
                        print(f"   ⚠️  Confluence enrichment failed: {e}")
                    # Even on failure, mark as attempted to avoid retrying per-repo
                    self._confluence_fetched = True

        # CI/CD enrichment
        if self.cicd_integrations and repo_config.github_repo:
            all_pipelines = []
            for platform_name, cicd_integration in self.cicd_integrations.items():
                try:
                    if self.debug_mode:
                        print(f"   🔄 Fetching {platform_name} pipelines...")

                    # Fetch pipelines from platform
                    pipelines = cicd_integration.fetch_pipelines(repo_config.github_repo, since)

                    # Enrich commits with pipeline status
                    if pipelines:
                        cicd_integration.enrich_commits_with_pipelines(commits, pipelines)
                        all_pipelines.extend(pipelines)

                        if self.debug_mode:
                            print(f"   ✅ Fetched {len(pipelines)} pipelines from {platform_name}")

                except Exception as e:
                    if self.debug_mode:
                        print(f"   ⚠️  {platform_name} enrichment failed: {e}")

            # Store pipelines and calculate aggregate metrics
            if all_pipelines:
                enrichment["cicd_data"]["pipelines"] = all_pipelines

                # Calculate aggregate metrics across all platforms
                all_metrics = {}
                for platform_name, cicd_integration in self.cicd_integrations.items():
                    platform_pipelines = [
                        p for p in all_pipelines if p.get("platform") == platform_name
                    ]
                    if platform_pipelines:
                        all_metrics[platform_name] = cicd_integration.calculate_metrics(
                            platform_pipelines
                        )

                enrichment["cicd_data"]["metrics"] = all_metrics

                if self.debug_mode:
                    total_pipelines = len(all_pipelines)
                    successful = len([p for p in all_pipelines if p.get("status") == "success"])
                    success_rate = (successful / total_pipelines * 100) if total_pipelines else 0
                    print(
                        f"   📊 CI/CD summary: {total_pipelines} pipelines, {success_rate:.1f}% success rate"
                    )

        # PM Framework enrichment
        if self.pm_orchestrator and self.pm_orchestrator.is_enabled():
            try:
                if self.debug_mode:
                    print("   📋 Collecting PM platform data...")

                # Get all issues from PM platforms
                pm_issues = self.pm_orchestrator.get_all_issues(since=since)
                enrichment["pm_data"]["issues"] = pm_issues

                # Correlate issues with commits
                correlations = self.pm_orchestrator.correlate_issues_with_commits(
                    pm_issues, commits
                )
                enrichment["pm_data"]["correlations"] = correlations

                # Calculate enhanced metrics
                enhanced_metrics = self.pm_orchestrator.calculate_enhanced_metrics(
                    commits, enrichment["prs"], pm_issues, correlations
                )
                enrichment["pm_data"]["metrics"] = enhanced_metrics

                # Only show correlations message if there are any correlations found
                if self.debug_mode:
                    if correlations:
                        print(
                            f"   ✅ PM correlations found: {len(correlations)} commits linked to issues"
                        )
                    else:
                        print("   📋 PM data processed (no correlations found)")

            except Exception as e:
                if self.debug_mode:
                    print(f"   ⚠️  PM framework enrichment failed: {e}")
                enrichment["pm_data"] = {"error": str(e)}

        return enrichment

    def get_platform_issues(self, project_key: str, since: datetime) -> list[dict[str, Any]]:
        """Get issues from all configured platforms."""
        del since  # Currently unused; retained for API compatibility / future use
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
