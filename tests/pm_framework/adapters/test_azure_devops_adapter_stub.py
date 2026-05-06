"""Tests for the Phase 1 Azure DevOps adapter stub.

The adapter is intentionally a registration stub at this phase: it must
register with the orchestrator, advertise an all-False capability set,
and raise phase-tagged ``NotImplementedError`` from every data method.
"""

from __future__ import annotations

import pytest

from gitflow_analytics.pm_framework.adapters import AzureDevOpsAdapter
from gitflow_analytics.pm_framework.orchestrator import PMFrameworkOrchestrator


def _make_adapter() -> AzureDevOpsAdapter:
    """Construct an adapter instance with a minimal stub config."""
    return AzureDevOpsAdapter(
        {
            "organization_url": "https://dev.azure.com/myorg",
            "personal_access_token": "fake-pat",
            "project": "MyProject",
        }
    )


class TestAzureDevOpsAdapterStub:
    """Behaviour expected from the Phase 1 stub adapter."""

    def test_platform_name(self) -> None:
        adapter = _make_adapter()
        assert adapter.platform_name == "azure_devops"

    def test_capabilities_all_false(self) -> None:
        """Every supports_* flag must be False until later phases land."""
        adapter = _make_adapter()
        caps = adapter.capabilities
        assert caps.supports_projects is False
        assert caps.supports_issues is False
        assert caps.supports_sprints is False
        assert caps.supports_time_tracking is False
        assert caps.supports_story_points is False
        assert caps.supports_custom_fields is False
        assert caps.supports_issue_linking is False
        assert caps.supports_comments is False
        assert caps.supports_attachments is False
        assert caps.supports_workflows is False
        assert caps.supports_bulk_operations is False
        assert caps.supports_cursor_pagination is False

    @pytest.mark.parametrize(
        ("method_name", "args", "phase"),
        [
            ("get_projects", (), "Phase 2"),
            ("get_issues", ("MyProject",), "Phase 3"),
            ("get_sprints", ("MyProject",), "Phase 4"),
            ("get_users", ("MyProject",), "Phase 4"),
            ("get_issue_comments", ("AB#1",), "Phase 5"),
            ("get_custom_fields", ("MyProject",), "Phase 5"),
        ],
    )
    def test_data_methods_raise_not_implemented(
        self, method_name: str, args: tuple, phase: str
    ) -> None:
        """Each stub *data* method must raise NotImplementedError with phase tag.

        Note: ``authenticate`` and ``test_connection`` deliberately do
        NOT raise — they return success-with-stub-status so the
        orchestrator's ``_initialize_platforms`` flow does not log an
        error on every ADO-configured run. That contract is verified by
        :meth:`test_authenticate_returns_true_for_stub` and
        :meth:`test_test_connection_returns_stub_status` below.
        """
        adapter = _make_adapter()
        method = getattr(adapter, method_name)
        with pytest.raises(NotImplementedError) as exc:
            method(*args)
        assert "Azure DevOps adapter" in str(exc.value)
        assert phase in str(exc.value)

    def test_authenticate_returns_true_for_stub(self) -> None:
        """Stub ``authenticate()`` must return ``True`` (not raise).

        Architecture review B1: raising ``NotImplementedError`` from
        ``authenticate`` causes the orchestrator's
        ``_initialize_platforms`` to log an ERROR with stack trace on
        every ADO-configured run. The Phase 1 stub returns ``True`` (and
        logs an advisory) so configured deployments stay quiet until the
        real authentication arrives in Phase 2.
        """
        adapter = _make_adapter()
        assert adapter.authenticate() is True

    def test_test_connection_returns_stub_status(self) -> None:
        """Stub ``test_connection()`` must return a connected-stub diagnostic.

        The dict shape is the contract the orchestrator inspects:
        ``status="connected"`` lets the connection-test gate pass;
        ``stub=True`` and ``phase=2`` mark the adapter as a Phase-1
        placeholder for any caller that wants to distinguish it from
        a real adapter.
        """
        adapter = _make_adapter()
        result = adapter.test_connection()
        assert result["status"] == "connected"
        assert result["stub"] is True
        assert result["phase"] == 2
        assert result["platform"] == "azure_devops"

    def test_adapter_registered_in_orchestrator(self) -> None:
        """Orchestrator must register 'azure_devops' even when disabled."""
        orchestrator = PMFrameworkOrchestrator(
            {
                "pm_platforms": {},
                "analysis": {"pm_integration": {"enabled": False}},
            }
        )
        available = orchestrator.registry.get_available_platforms()
        assert "azure_devops" in available
        assert "jira" in available

    def test_registry_resolves_adapter_class(self) -> None:
        """Registered class lookup should resolve to AzureDevOpsAdapter.

        ``PlatformRegistry.create_adapter`` triggers ``authenticate`` to
        fail-fast on credential issues; the Phase 1 stub deliberately
        raises there, so we verify the class lookup directly.
        """
        orchestrator = PMFrameworkOrchestrator(
            {
                "pm_platforms": {},
                "analysis": {"pm_integration": {"enabled": False}},
            }
        )
        # The registry stores the class; verify by instantiating directly.
        adapter_class = orchestrator.registry._adapters["azure_devops"]
        assert adapter_class is AzureDevOpsAdapter

        adapter = adapter_class(
            {
                "organization_url": "https://dev.azure.com/myorg",
                "personal_access_token": "fake-pat",
            }
        )
        assert isinstance(adapter, AzureDevOpsAdapter)
        assert adapter.platform_name == "azure_devops"
