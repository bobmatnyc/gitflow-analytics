"""Tests for Azure DevOps Phase 1 configuration plumbing.

Covers:
- Round-tripping a YAML fixture with a ``pm.azure_devops`` block.
- Effective ticket-platform inference picking up ``"azure_devops"``.
- Environment-variable resolution for ``${AZURE_DEVOPS_PAT}``.
- URL validator rejection of on-prem patterns and ``is_on_premise: true``.
- URL validator acceptance of cloud hosts.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from gitflow_analytics.config import ConfigLoader
from gitflow_analytics.config.errors import (
    ConfigurationError,
    EnvironmentVariableError,
)
from gitflow_analytics.config.loader_sections import ConfigLoaderSectionsMixin
from gitflow_analytics.config.schema import AzureDevOpsConfig

_BASE_YAML = """\
version: "1.0"
github:
  token: "ghp_dummy_token_value"
  owner: "octocat"
repositories:
  - name: "demo"
    path: "/tmp/demo"
"""


def _write_yaml(content: str) -> Path:
    """Write ``content`` to a temp YAML file and return the path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
    return Path(tmp.name)


@pytest.fixture
def ado_pat_env() -> Iterator[None]:
    """Set the AZURE_DEVOPS_PAT env var for the duration of a test."""
    original = os.environ.get("AZURE_DEVOPS_PAT")
    os.environ["AZURE_DEVOPS_PAT"] = "test-pat-value"
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("AZURE_DEVOPS_PAT", None)
        else:
            os.environ["AZURE_DEVOPS_PAT"] = original


class TestAzureDevOpsConfigLoading:
    """Loader-level tests for the Azure DevOps Phase 1 wiring."""

    def test_loads_pm_azure_devops_block(self, ado_pat_env: None) -> None:
        """A YAML config with pm.azure_devops should round-trip cleanly."""
        config_yaml = _BASE_YAML + (
            "pm:\n"
            "  azure_devops:\n"
            "    enabled: true\n"
            '    organization_url: "https://dev.azure.com/myorg"\n'
            '    project: "MyProject"\n'
            '    personal_access_token: "${AZURE_DEVOPS_PAT}"\n'
            '    work_item_types: ["User Story", "Bug"]\n'
        )
        path = _write_yaml(config_yaml)
        try:
            cfg = ConfigLoader.load(path)
        finally:
            path.unlink()

        # Per plan §3.3, ``cfg.pm.azure_devops`` is the only canonical
        # home for ADO configuration. There is no top-level ``cfg.azure_devops``.
        assert cfg.pm is not None
        ado = getattr(cfg.pm, "azure_devops", None)
        assert isinstance(ado, AzureDevOpsConfig)
        assert ado.enabled is True
        assert ado.organization_url == "https://dev.azure.com/myorg"
        assert ado.project == "MyProject"
        assert ado.personal_access_token == "test-pat-value"
        assert ado.work_item_types == ["User Story", "Bug"]
        assert ado.api_version == "7.1"

    def test_top_level_azure_devops_block_is_ignored(self, ado_pat_env: None) -> None:
        """Per plan §3.3, top-level ``azure_devops:`` is NOT a supported key.

        Only ``pm.azure_devops:`` is parsed. A YAML doc with the top-level
        key (and no ``pm.azure_devops``) loads cleanly but produces no ADO
        config — guarding against re-introducing the ``jira:`` /
        ``jira_integration:`` dual-stack mistake.
        """
        config_yaml = _BASE_YAML + (
            "azure_devops:\n"
            '  organization_url: "https://myorg.visualstudio.com"\n'
            '  personal_access_token: "${AZURE_DEVOPS_PAT}"\n'
        )
        path = _write_yaml(config_yaml)
        try:
            cfg = ConfigLoader.load(path)
        finally:
            path.unlink()

        # No ADO config should be picked up; the top-level key is intentionally not a parse target.
        assert cfg.pm is None or getattr(cfg.pm, "azure_devops", None) is None

    def test_get_effective_ticket_platforms_includes_azure_devops(self, ado_pat_env: None) -> None:
        """``get_effective_ticket_platforms`` should include 'azure_devops'."""
        config_yaml = _BASE_YAML + (
            "pm:\n"
            "  azure_devops:\n"
            '    organization_url: "https://dev.azure.com/myorg"\n'
            '    personal_access_token: "${AZURE_DEVOPS_PAT}"\n'
        )
        path = _write_yaml(config_yaml)
        try:
            cfg = ConfigLoader.load(path)
        finally:
            path.unlink()

        platforms = cfg.get_effective_ticket_platforms()
        assert "azure_devops" in platforms

    def test_get_effective_ticket_platforms_fallback_excludes_ado(self) -> None:
        """Fallback (no PM/JIRA configured) does NOT include ``azure_devops``.

        The fallback ``["jira", "github", "clickup", "linear"]`` is applied
        when no platform is configured. Adding ``azure_devops`` to that
        fallback would silently turn ``AB#NNN`` references in commits into
        ADO-tagged tickets for users who never configured ADO. ADO opt-in
        only — see schema.py ``get_effective_ticket_platforms`` comment.
        """
        from gitflow_analytics.config.schema import (
            AnalysisConfig,
            CacheConfig,
            Config,
            GitHubConfig,
            OutputConfig,
        )

        cfg = Config(
            repositories=[],
            github=GitHubConfig(token=None),  # type: ignore[arg-type]
            analysis=AnalysisConfig(),
            output=OutputConfig(),
            cache=CacheConfig(),
        )
        platforms = cfg.get_effective_ticket_platforms()
        assert "azure_devops" not in platforms
        assert platforms == ["jira", "github", "clickup", "linear"]

    def test_missing_pat_env_raises(self) -> None:
        """Unresolved ${AZURE_DEVOPS_PAT} should raise EnvironmentVariableError."""
        # Make sure the env var is *not* set.
        original = os.environ.pop("AZURE_DEVOPS_PAT", None)
        try:
            config_yaml = _BASE_YAML + (
                "pm:\n"
                "  azure_devops:\n"
                '    organization_url: "https://dev.azure.com/myorg"\n'
                '    personal_access_token: "${AZURE_DEVOPS_PAT}"\n'
            )
            path = _write_yaml(config_yaml)
            try:
                with pytest.raises(EnvironmentVariableError):
                    ConfigLoader.load(path)
            finally:
                path.unlink()
        finally:
            if original is not None:
                os.environ["AZURE_DEVOPS_PAT"] = original


class TestAzureDevOpsUrlValidator:
    """Direct unit tests for the on-prem URL validator."""

    def test_rejects_tfs_collection_url(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            ConfigLoaderSectionsMixin._validate_azure_devops_url(
                "https://tfs.example.com/tfs/MyCollection",
                is_on_premise=False,
            )
        assert "on-premises" in str(exc.value).lower()
        assert "v1.2" in str(exc.value)

    def test_rejects_default_collection_url(self) -> None:
        with pytest.raises(ConfigurationError):
            ConfigLoaderSectionsMixin._validate_azure_devops_url(
                "https://server.example.com/DefaultCollection/proj",
                is_on_premise=False,
            )

    def test_rejects_arbitrary_host(self) -> None:
        with pytest.raises(ConfigurationError):
            ConfigLoaderSectionsMixin._validate_azure_devops_url(
                "https://example.com/myorg",
                is_on_premise=False,
            )

    def test_rejects_is_on_premise_true(self) -> None:
        """is_on_premise=True must always be rejected with the v1.2 message."""
        with pytest.raises(ConfigurationError) as exc:
            ConfigLoaderSectionsMixin._validate_azure_devops_url(
                "https://dev.azure.com/myorg",
                is_on_premise=True,
            )
        assert "on-premises" in str(exc.value).lower()

    def test_accepts_dev_azure_com(self) -> None:
        # No exception means the URL passes the allowlist.
        ConfigLoaderSectionsMixin._validate_azure_devops_url(
            "https://dev.azure.com/myorg",
            is_on_premise=False,
        )

    def test_accepts_visualstudio_com(self) -> None:
        ConfigLoaderSectionsMixin._validate_azure_devops_url(
            "https://myorg.visualstudio.com",
            is_on_premise=False,
        )

    def test_rejects_empty_url(self) -> None:
        with pytest.raises(ConfigurationError):
            ConfigLoaderSectionsMixin._validate_azure_devops_url("", is_on_premise=False)

    def test_loader_rejects_on_prem_yaml(self, ado_pat_env: None) -> None:
        """Loader should surface the on-prem rejection when YAML is loaded.

        Verbatim ADR-decision-2 message check (not just substring): a
        future refactor that softens the message must update the ADR
        and this test together.
        """
        from gitflow_analytics.config.loader_sections import (
            _AZURE_DEVOPS_ONPREM_REJECTION_MESSAGE,
        )

        config_yaml = _BASE_YAML + (
            "pm:\n"
            "  azure_devops:\n"
            '    organization_url: "https://tfs.example.com/tfs/MyCollection"\n'
            '    personal_access_token: "${AZURE_DEVOPS_PAT}"\n'
        )
        path = _write_yaml(config_yaml)
        try:
            with pytest.raises(ConfigurationError) as exc:
                ConfigLoader.load(path)
            # Verbatim assertion: the rejection message must match the ADR text.
            assert _AZURE_DEVOPS_ONPREM_REJECTION_MESSAGE in str(exc.value)
        finally:
            path.unlink()

    def test_loader_rejects_is_on_premise_yaml(self, ado_pat_env: None) -> None:
        """Loader should reject is_on_premise=true with the verbatim ADR message."""
        from gitflow_analytics.config.loader_sections import (
            _AZURE_DEVOPS_ONPREM_REJECTION_MESSAGE,
        )

        config_yaml = _BASE_YAML + (
            "pm:\n"
            "  azure_devops:\n"
            '    organization_url: "https://dev.azure.com/myorg"\n'
            '    personal_access_token: "${AZURE_DEVOPS_PAT}"\n'
            "    is_on_premise: true\n"
        )
        path = _write_yaml(config_yaml)
        try:
            with pytest.raises(ConfigurationError) as exc:
                ConfigLoader.load(path)
            assert _AZURE_DEVOPS_ONPREM_REJECTION_MESSAGE in str(exc.value)
        finally:
            path.unlink()

    def test_accepts_dev_azure_com_with_trailing_slash(self) -> None:
        """Trailing-slash URLs should pass the allowlist check."""
        ConfigLoaderSectionsMixin._validate_azure_devops_url(
            "https://dev.azure.com/myorg/",
            is_on_premise=False,
        )

    def test_accepts_dev_azure_com_uppercase(self) -> None:
        """Host comparison should be case-insensitive."""
        ConfigLoaderSectionsMixin._validate_azure_devops_url(
            "https://DEV.AZURE.COM/myorg",
            is_on_premise=False,
        )


class TestAzureDevOpsPATValidation:
    """Tests for the silent-empty PAT trap and EnvironmentVariableError contract."""

    def test_empty_pat_env_raises(self) -> None:
        """``AZURE_DEVOPS_PAT=""`` (empty string) must raise, not silently pass.

        ``${AZURE_DEVOPS_PAT}`` is a non-empty literal so the early
        env-var-error guard does not fire. The dedicated empty-string
        check after env-var resolution must catch it.
        """
        original = os.environ.get("AZURE_DEVOPS_PAT")
        os.environ["AZURE_DEVOPS_PAT"] = ""
        try:
            config_yaml = _BASE_YAML + (
                "pm:\n"
                "  azure_devops:\n"
                '    organization_url: "https://dev.azure.com/myorg"\n'
                '    personal_access_token: "${AZURE_DEVOPS_PAT}"\n'
            )
            path = _write_yaml(config_yaml)
            try:
                with pytest.raises(EnvironmentVariableError):
                    ConfigLoader.load(path)
            finally:
                path.unlink()
        finally:
            if original is None:
                os.environ.pop("AZURE_DEVOPS_PAT", None)
            else:
                os.environ["AZURE_DEVOPS_PAT"] = original

    def test_whitespace_pat_env_raises(self) -> None:
        """Whitespace-only PAT must also be rejected."""
        original = os.environ.get("AZURE_DEVOPS_PAT")
        os.environ["AZURE_DEVOPS_PAT"] = "   "
        try:
            config_yaml = _BASE_YAML + (
                "pm:\n"
                "  azure_devops:\n"
                '    organization_url: "https://dev.azure.com/myorg"\n'
                '    personal_access_token: "${AZURE_DEVOPS_PAT}"\n'
            )
            path = _write_yaml(config_yaml)
            try:
                with pytest.raises(EnvironmentVariableError):
                    ConfigLoader.load(path)
            finally:
                path.unlink()
        finally:
            if original is None:
                os.environ.pop("AZURE_DEVOPS_PAT", None)
            else:
                os.environ["AZURE_DEVOPS_PAT"] = original

    def test_environment_variable_error_carries_var_name_and_platform(self) -> None:
        """The raised ``EnvironmentVariableError`` must name the var + platform.

        ADR Decision 1 names ``AZURE_DEVOPS_PAT`` and the platform string
        ``"AzureDevOps"`` as part of the error contract. A refactor that
        renames either must update this assertion alongside.
        """
        original = os.environ.pop("AZURE_DEVOPS_PAT", None)
        try:
            config_yaml = _BASE_YAML + (
                "pm:\n"
                "  azure_devops:\n"
                '    organization_url: "https://dev.azure.com/myorg"\n'
                '    personal_access_token: "${AZURE_DEVOPS_PAT}"\n'
            )
            path = _write_yaml(config_yaml)
            try:
                with pytest.raises(EnvironmentVariableError) as exc:
                    ConfigLoader.load(path)
                # The error renders the var name and platform as part of the
                # message produced by ``errors.py:EnvironmentVariableError``.
                assert "AZURE_DEVOPS_PAT" in str(exc.value)
                assert "AzureDevOps" in str(exc.value)
            finally:
                path.unlink()
        finally:
            if original is not None:
                os.environ["AZURE_DEVOPS_PAT"] = original
