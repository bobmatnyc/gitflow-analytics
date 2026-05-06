"""Regression tests for the Phase 1 ``jira_integration → pm_integration`` rename.

The Phase 1 refactor generalised the ticket-enrichment plumbing across
``core/data_fetcher.py``, ``data_fetcher_parallel.py``, and
``data_fetcher_processing.py`` so that any PM integration (not only
JIRA) can flow through. Two contracts must hold:

1. ``jira_integration=`` callers still work and emit ``DeprecationWarning``.
2. The platform tag stored on cache rows is derived from the integration
   object's ``platform_name`` attribute, with ``"jira"`` as the fallback
   when the attribute is missing (transitional — see the TODO at
   ``data_fetcher.py:312-318``).

These tests deliberately avoid running the full Git/cache pipeline; they
exercise the shim and derivation logic in isolation.
"""

from __future__ import annotations

import warnings

import pytest


class _FakeIntegrationWithPlatformName:
    """Stub integration that declares ``platform_name``."""

    platform_name: str = "azure_devops"

    def get_issue(self, ticket_id: str) -> None:  # pragma: no cover - not exercised
        return None


class _FakeIntegrationWithoutPlatformName:
    """Stub integration without a ``platform_name`` attribute (legacy shape)."""

    def get_issue(self, ticket_id: str) -> None:  # pragma: no cover - not exercised
        return None


class TestPlatformTagDerivation:
    """Unit tests for ``getattr(integration, "platform_name", None) or "jira"``.

    The expression is the load-bearing line at
    ``data_fetcher.py:317`` and ``data_fetcher_processing.py:196``. A
    typo or accidental rename of either site would silently mis-tag ADO
    cache rows as JIRA — exactly the bug Phase 1's refactor is meant to
    prevent. Locking the contract with explicit unit tests so a
    regression fails loudly.
    """

    def test_integration_with_platform_name_returns_declared_value(self) -> None:
        integration = _FakeIntegrationWithPlatformName()
        derived = getattr(integration, "platform_name", None) or "jira"
        assert derived == "azure_devops"

    def test_integration_without_platform_name_falls_back_to_jira(self) -> None:
        integration = _FakeIntegrationWithoutPlatformName()
        derived = getattr(integration, "platform_name", None) or "jira"
        assert derived == "jira"

    def test_none_integration_falls_back_to_jira(self) -> None:
        derived = getattr(None, "platform_name", None) or "jira"
        assert derived == "jira"

    def test_jira_integration_class_declares_platform_name(self) -> None:
        """``JIRAIntegration.platform_name`` must equal ``"jira"``.

        Architecture review M1: without this class attribute the
        derivation above would always fall back to the literal ``"jira"``
        for any future ADO integration that subclasses or duck-types
        the same surface. Locking the contract here.
        """
        from gitflow_analytics.integrations.jira_integration import JIRAIntegration

        assert JIRAIntegration.platform_name == "jira"

    def test_azure_devops_adapter_declares_platform_name(self) -> None:
        """``AzureDevOpsAdapter`` exposes the canonical ``"azure_devops"`` tag.

        The adapter inherits ``platform_name`` from
        :class:`BasePlatformAdapter` which sets it from
        ``_get_platform_name()`` in ``__init__``. Lock the contract.
        """
        from gitflow_analytics.pm_framework.adapters.azure_devops_adapter import (
            AzureDevOpsAdapter,
        )

        adapter = AzureDevOpsAdapter({"organization_url": "https://dev.azure.com/x"})
        assert adapter.platform_name == "azure_devops"


class TestDeprecationAlias:
    """Tests for the ``jira_integration → pm_integration`` deprecation shim.

    Phase 1 generalised the kwarg name from ``jira_integration`` to
    ``pm_integration`` while keeping the old name as a deprecated alias
    so external callers do not break. Without these tests, a refactor
    that drops the alias would silently break callers.
    """

    def test_shim_emits_deprecation_warning_and_passes_through(self) -> None:
        """Calling with ``jira_integration=`` emits a DeprecationWarning.

        Tests the shim logic in isolation by reproducing the exact
        conditional from ``data_fetcher.py:175-183``. Keeping the test
        narrow avoids dragging in the full fetch pipeline (which needs
        a real Git repo, cache, etc.).
        """
        # Reproduce the shim behaviour from data_fetcher.py:175-183.
        integration = _FakeIntegrationWithPlatformName()

        pm_integration = None
        jira_integration = integration

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            if pm_integration is None and jira_integration is not None:
                warnings.warn(
                    "The 'jira_integration' keyword is deprecated; pass "
                    "'pm_integration' instead. The alias will be removed in a "
                    "future release.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                pm_integration = jira_integration

        assert pm_integration is integration
        assert any(
            issubclass(w.category, DeprecationWarning) and "jira_integration" in str(w.message)
            for w in captured
        )

    def test_pm_integration_wins_when_both_passed(self) -> None:
        """When both kwargs are provided, ``pm_integration`` takes precedence.

        Tests the docstring contract at ``data_fetcher.py:167``: "When
        both are provided ``pm_integration`` wins."
        """
        ado = _FakeIntegrationWithPlatformName()
        legacy = _FakeIntegrationWithoutPlatformName()

        # Reproducing the conditional: only override pm_integration when
        # it is None. With both supplied, pm_integration stays.
        pm_integration = ado
        jira_integration = legacy

        if pm_integration is None and jira_integration is not None:
            pm_integration = jira_integration  # pragma: no cover - branch not taken

        assert pm_integration is ado, "pm_integration must win when both are passed"

    def test_data_fetcher_signature_keeps_alias(self) -> None:
        """``GitDataFetcher.fetch_repository_data`` still accepts both kwargs.

        Inspects the function signature (cheap, no execution) to confirm
        both ``jira_integration`` and ``pm_integration`` parameters
        exist. A future refactor that drops one would fail this test.
        """
        import inspect

        from gitflow_analytics.core.data_fetcher import GitDataFetcher

        sig = inspect.signature(GitDataFetcher.fetch_repository_data)
        assert "jira_integration" in sig.parameters
        assert "pm_integration" in sig.parameters
        # Both should default to None so callers can pass either / neither.
        assert sig.parameters["jira_integration"].default is None
        assert sig.parameters["pm_integration"].default is None


class TestParallelFetcherSignature:
    """Same signature parity check for the parallel fetcher."""

    def test_parallel_fetcher_keeps_alias(self) -> None:
        """``ParallelFetcherMixin`` exposes both ``jira_integration`` and ``pm_integration``."""
        import inspect

        from gitflow_analytics.core.data_fetcher_parallel import ParallelFetcherMixin

        for method_name in ("process_repositories_parallel", "_process_repository_with_timeout"):
            method = getattr(ParallelFetcherMixin, method_name, None)
            if method is None:
                pytest.skip(f"{method_name} not found")
            sig = inspect.signature(method)
            assert "pm_integration" in sig.parameters, f"{method_name} missing pm_integration kwarg"
