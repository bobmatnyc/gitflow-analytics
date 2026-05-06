"""Azure DevOps PM platform adapter (Phase 1 stub).

This module provides the registration surface for Azure DevOps inside the
PM framework. The adapter is intentionally a stub at this phase: it parses
its configuration, advertises a no-capabilities profile, and raises a
phase-tagged :class:`NotImplementedError` from every data method so the
orchestrator can register it without enabling network access.

Real behaviour is delivered by later phases of the integration plan
(``docs/design/azure-devops-integration-plan.md``):

- Phase 2: ``authenticate`` / ``test_connection`` / ``get_projects``.
- Phase 3: ``get_issues`` (WIQL + batch fetch) and the cache.
- Phase 4: ``get_sprints`` and ``get_users``.
- Phase 5: ``get_issue_comments``, ``get_custom_fields`` and native
  commit-link correlation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..base import BasePlatformAdapter, PlatformCapabilities
from ..models import IssueType, UnifiedIssue, UnifiedProject, UnifiedSprint, UnifiedUser


class AzureDevOpsAdapter(BasePlatformAdapter):
    """Azure DevOps Services adapter (Phase 1 stub).

    The full adapter ships incrementally across Phases 2–5. Phase 1 only
    registers the platform key with the orchestrator; every data method
    raises :class:`NotImplementedError` with the originating phase tag so
    test failures and runtime errors are self-describing.

    Attributes inherited from :class:`BasePlatformAdapter`:
        config: Configuration mapping the orchestrator handed to the
            adapter (typically the resolved ``pm_integration.platforms.
            azure_devops.config`` block).
        platform_name: Always ``"azure_devops"`` for this adapter.
        capabilities: A :class:`PlatformCapabilities` instance with all
            ``supports_*`` flags set to ``False`` until later phases land.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the adapter with its config block.

        Args:
            config: Resolved configuration mapping. Keys mirror the fields
                of :class:`gitflow_analytics.config.schema.AzureDevOpsConfig`
                (``organization_url``, ``personal_access_token``,
                ``project``, ``api_version``, ``story_point_fields`` …).
                Missing keys fall back to safe defaults for the stub.
        """
        super().__init__(config)
        self.organization_url: str = str(config.get("organization_url", "") or "")
        self.personal_access_token: str = str(config.get("personal_access_token", "") or "")
        self.project: str | None = config.get("project")
        self.api_version: str = str(config.get("api_version", "7.1"))
        self.story_point_fields: list[str] = list(
            config.get(
                "story_point_fields",
                [
                    "Microsoft.VSTS.Scheduling.StoryPoints",
                    "Microsoft.VSTS.Scheduling.Effort",
                    "Microsoft.VSTS.Scheduling.Size",
                ],
            )
        )

    def _get_platform_name(self) -> str:
        """Return the canonical platform identifier.

        Returns:
            The literal string ``"azure_devops"``.
        """
        return "azure_devops"

    def _get_capabilities(self) -> PlatformCapabilities:
        """Return capability flags for the Phase 1 stub.

        All ``supports_*`` flags are forced to ``False`` until the
        corresponding behaviours land in later phases.

        Returns:
            A :class:`PlatformCapabilities` with every feature flag
            disabled and conservative rate-limit defaults.
        """
        caps = PlatformCapabilities()
        # Phase 1: explicitly disable every capability so callers do not
        # accidentally exercise unimplemented paths via base-class default
        # implementations.
        caps.supports_projects = False
        caps.supports_issues = False
        caps.supports_sprints = False
        caps.supports_time_tracking = False
        caps.supports_story_points = False
        caps.supports_custom_fields = False
        caps.supports_issue_linking = False
        caps.supports_comments = False
        caps.supports_attachments = False
        caps.supports_workflows = False
        caps.supports_bulk_operations = False
        caps.supports_cursor_pagination = False
        return caps

    # ------------------------------------------------------------------
    # Stub data methods. Each raises NotImplementedError tagged with the
    # phase that will deliver the real implementation.
    # ------------------------------------------------------------------

    def authenticate(self) -> bool:
        """Report a successful Phase-1 stub authentication.

        Phase 1 ships only the registration surface. Returning ``True`` (with
        a one-time advisory log line) lets the orchestrator's
        ``_initialize_platforms`` flow complete without logging an error
        every run for users who have already configured ``pm.azure_devops``.

        Returns:
            ``True`` — the stub is always "authenticated" since no network
            call occurs. Real PAT validation arrives in Phase 2.
        """
        self.logger.info(
            "Azure DevOps adapter is a Phase 1 stub; configuration is parsed "
            "and registered but no work items will be collected until Phase 2."
        )
        return True

    def test_connection(self) -> dict[str, Any]:
        """Return a stub-status diagnostic dictionary.

        Returns:
            A dictionary with ``status="connected"`` (so the orchestrator's
            connection-test gate passes) plus stub markers (``stub=True``,
            ``phase=2``) so callers that want to distinguish stub from
            real adapters can. Real diagnostics arrive in Phase 2.
        """
        return {
            "status": "connected",
            "stub": True,
            "phase": 2,
            "platform": "azure_devops",
            "message": (
                "Azure DevOps adapter is a Phase 1 stub. "
                "Real authentication and project listing arrive in Phase 2."
            ),
        }

    def get_projects(self) -> list[UnifiedProject]:
        """List accessible Azure DevOps projects.

        Raises:
            NotImplementedError: Always; the real implementation arrives
                in Phase 2.
        """
        raise NotImplementedError("Azure DevOps adapter: get_projects — implemented in Phase 2")

    def get_issues(
        self,
        project_id: str,
        since: datetime | None = None,
        issue_types: list[IssueType] | None = None,
    ) -> list[UnifiedIssue]:
        """Fetch work items for a project.

        Args:
            project_id: Azure DevOps project identifier (id or name).
            since: Optional lower-bound for ``System.ChangedDate``.
            issue_types: Optional unified issue-type filter.

        Raises:
            NotImplementedError: Always; the WIQL + batch fetch
                implementation arrives in Phase 3.
        """
        raise NotImplementedError("Azure DevOps adapter: get_issues — implemented in Phase 3")

    def get_sprints(self, project_id: str) -> list[UnifiedSprint]:
        """Enumerate iterations (sprints) for a project.

        Args:
            project_id: Azure DevOps project identifier.

        Raises:
            NotImplementedError: Always; iteration enumeration arrives
                in Phase 4.
        """
        raise NotImplementedError("Azure DevOps adapter: get_sprints — implemented in Phase 4")

    def get_users(self, project_id: str) -> list[UnifiedUser]:
        """Enumerate users for a project.

        Args:
            project_id: Azure DevOps project identifier.

        Raises:
            NotImplementedError: Always; Graph-API enumeration arrives
                in Phase 4.
        """
        raise NotImplementedError("Azure DevOps adapter: get_users — implemented in Phase 4")

    def get_issue_comments(self, issue_key: str) -> list[dict[str, Any]]:
        """Fetch the comment history for a work item.

        Args:
            issue_key: Azure DevOps work-item identifier.

        Raises:
            NotImplementedError: Always; comments support arrives in
                Phase 5.
        """
        raise NotImplementedError(
            "Azure DevOps adapter: get_issue_comments — implemented in Phase 5"
        )

    def get_custom_fields(self, project_id: str) -> dict[str, Any]:
        """Retrieve custom field definitions for a project.

        Args:
            project_id: Azure DevOps project identifier.

        Raises:
            NotImplementedError: Always; custom-field discovery arrives
                in Phase 5.
        """
        raise NotImplementedError(
            "Azure DevOps adapter: get_custom_fields — implemented in Phase 5"
        )
