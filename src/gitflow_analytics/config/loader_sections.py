"""Secondary config section processors for ConfigLoader.

This module provides ConfigLoaderSectionsMixin which adds output, cache, JIRA,
qualitative, PM config processing, env var resolution, and validation to
ConfigLoader via multiple inheritance.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import click

from .errors import EnvironmentVariableError
from .schema import (
    ActivityScoringConfig,
    AIDetectionConfig,
    BoilerplateFilterConfig,
    CacheConfig,
    Config,
    ConfluenceConfig,
    GitHubIssuesConfig,
    JIRAConfig,
    JIRAIntegrationConfig,
    OutputConfig,
    PMIntegrationConfig,
    PMPlatformConfig,
    PodConfig,
    QualityReportConfig,
    TeamConfig,
    TeamMemberConfig,
    TeamsConfig,
    VelocityConfig,
)
from .validator import ConfigValidator

# Regex for embedded env-var references in config strings.
# Matches both ``${VAR}`` and ``$VAR`` (latter only for ASCII identifiers).
# WHY: Some users write ``api_token: "${CONFLUENCE_API_TOKEN}"`` but others may
# inadvertently write ``$CONFLUENCE_API_TOKEN`` or embed the reference in a
# larger string (e.g. ``"Bearer ${TOKEN}"``); handling all three keeps behaviour
# predictable and backward compatible.
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")

logger = logging.getLogger(__name__)


class ConfigLoaderSectionsMixin:
    """Mixin adding secondary config section processors to ConfigLoader."""

    @classmethod
    def _process_output_config(cls, output_data: dict[str, Any], config_path: Path) -> OutputConfig:
        """Process output configuration section.

        Args:
            output_data: Output configuration data
            config_path: Path to configuration file

        Returns:
            OutputConfig instance
        """
        # Validate settings
        ConfigValidator.validate_output_config(output_data, config_path)

        # Process output directory
        output_dir = output_data.get("directory")
        if output_dir:
            output_dir = Path(output_dir).expanduser()
            # If relative path, make it relative to config file directory
            if not output_dir.is_absolute():
                output_dir = config_path.parent / output_dir
            output_dir = output_dir.resolve()
        else:
            # Default to config file directory if not specified
            output_dir = config_path.parent

        return OutputConfig(
            directory=output_dir,
            formats=output_data.get("formats", ["csv", "markdown"]),
            csv_delimiter=output_data.get("csv", {}).get("delimiter", ","),
            csv_encoding=output_data.get("csv", {}).get("encoding", "utf-8"),
            anonymize_enabled=output_data.get("anonymization", {}).get("enabled", False),
            anonymize_fields=output_data.get("anonymization", {}).get("fields", []),
            anonymize_method=output_data.get("anonymization", {}).get("method", "hash"),
        )

    @classmethod
    def _process_cache_config(cls, cache_data: dict[str, Any], config_path: Path) -> CacheConfig:
        """Process cache configuration section.

        Args:
            cache_data: Cache configuration data
            config_path: Path to configuration file

        Returns:
            CacheConfig instance
        """
        cache_dir = cache_data.get("directory", ".gitflow-cache")
        cache_path = Path(cache_dir)
        # If relative path, make it relative to config file directory
        if not cache_path.is_absolute():
            cache_path = config_path.parent / cache_path

        return CacheConfig(
            directory=cache_path.resolve(),
            ttl_hours=cache_data.get("ttl_hours", 168),
            max_size_mb=cache_data.get("max_size_mb", 500),
        )

    @classmethod
    def _process_jira_config(
        cls, jira_data: dict[str, Any], config_path: Path
    ) -> Optional[JIRAConfig]:
        """Process JIRA configuration section.

        Args:
            jira_data: JIRA configuration data
            config_path: Path to configuration file

        Returns:
            JIRAConfig instance or None
        """
        if not jira_data:
            return None

        access_user = cls._resolve_env_var(jira_data.get("access_user", ""))
        access_token = cls._resolve_env_var(jira_data.get("access_token", ""))

        # Validate JIRA credentials if JIRA is configured
        if jira_data.get("access_user") and jira_data.get("access_token"):
            if not access_user:
                raise EnvironmentVariableError("JIRA_ACCESS_USER", "JIRA", config_path)
            if not access_token:
                raise EnvironmentVariableError("JIRA_ACCESS_TOKEN", "JIRA", config_path)

        return JIRAConfig(
            access_user=access_user or "",
            access_token=access_token or "",
            base_url=jira_data.get("base_url"),
        )

    @classmethod
    def _process_jira_integration_config(
        cls, jira_integration_data: dict[str, Any]
    ) -> Optional[JIRAIntegrationConfig]:
        """Process JIRA integration configuration section.

        Args:
            jira_integration_data: JIRA integration configuration data

        Returns:
            JIRAIntegrationConfig instance or None
        """
        if not jira_integration_data:
            return None

        return JIRAIntegrationConfig(
            enabled=jira_integration_data.get("enabled", True),
            fetch_story_points=jira_integration_data.get("fetch_story_points", True),
            project_keys=jira_integration_data.get("project_keys", []),
            story_point_fields=jira_integration_data.get(
                "story_point_fields", ["customfield_10016", "customfield_10021", "Story Points"]
            ),
        )

    @classmethod
    def _process_github_issues_config(
        cls, github_issues_data: dict[str, Any]
    ) -> Optional[GitHubIssuesConfig]:
        """Process github_issues configuration section.

        Args:
            github_issues_data: GitHub Issues configuration data from YAML.

        Returns:
            GitHubIssuesConfig instance or None when no data supplied.
        """
        if not github_issues_data:
            return None

        issue_state = github_issues_data.get("issue_state", "all")
        valid_states = {"open", "closed", "all"}
        if issue_state not in valid_states:
            click.echo(
                f"Warning: github_issues.issue_state '{issue_state}' is invalid; "
                f"must be one of {sorted(valid_states)}. Defaulting to 'all'.",
                err=True,
            )
            issue_state = "all"

        allowed_repos = github_issues_data.get("allowed_repos")
        if allowed_repos is not None:
            allowed_repos = list(allowed_repos)

        return GitHubIssuesConfig(
            enabled=bool(github_issues_data.get("enabled", False)),
            fetch_comments=bool(github_issues_data.get("fetch_comments", False)),
            allowed_repos=allowed_repos,
            issue_state=issue_state,
            max_issues_per_repo=int(github_issues_data.get("max_issues_per_repo", 500)),
        )

    @classmethod
    def _process_confluence_config(
        cls, confluence_data: dict[str, Any]
    ) -> Optional[ConfluenceConfig]:
        """Process confluence configuration section.

        Args:
            confluence_data: Confluence configuration data from YAML.

        Returns:
            ConfluenceConfig instance or None when no data supplied.
        """
        if not confluence_data:
            return None

        username = cls._resolve_env_var(confluence_data.get("username", "")) or ""
        api_token = cls._resolve_env_var(confluence_data.get("api_token", "")) or ""
        base_url = confluence_data.get("base_url", "")

        return ConfluenceConfig(
            enabled=bool(confluence_data.get("enabled", False)),
            base_url=base_url.rstrip("/") if base_url else "",
            username=username,
            api_token=api_token,
            spaces=list(confluence_data.get("spaces", [])),
            fetch_page_history=bool(confluence_data.get("fetch_page_history", False)),
            dns_timeout=int(confluence_data.get("dns_timeout", 10)),
            connection_timeout=int(confluence_data.get("connection_timeout", 30)),
            max_retries=int(confluence_data.get("max_retries", 3)),
            backoff_factor=float(confluence_data.get("backoff_factor", 1.0)),
        )

    @classmethod
    def _process_qualitative_config(cls, qualitative_data: dict[str, Any]) -> Optional[Any]:
        """Process qualitative analysis configuration section.

        Args:
            qualitative_data: Qualitative configuration data

        Returns:
            QualitativeConfig instance or None
        """
        if not qualitative_data:
            return None

        # Import here to avoid circular imports
        try:
            from ..qualitative.models.schemas import CacheConfig as QualitativeCacheConfig
            from ..qualitative.models.schemas import (
                ChangeTypeConfig,
                DomainConfig,
                IntentConfig,
                LLMConfig,
                NLPConfig,
                QualitativeConfig,
                RiskConfig,
            )

            # Parse NLP configuration
            nlp_data = qualitative_data.get("nlp", {})
            nlp_config = NLPConfig(
                spacy_model=nlp_data.get("spacy_model", "en_core_web_sm"),
                spacy_batch_size=nlp_data.get("spacy_batch_size", 1000),
                fast_mode=nlp_data.get("fast_mode", True),
                enable_parallel_processing=nlp_data.get("enable_parallel_processing", True),
                max_workers=nlp_data.get("max_workers", 4),
                change_type_config=ChangeTypeConfig(**nlp_data.get("change_type", {})),
                intent_config=IntentConfig(**nlp_data.get("intent", {})),
                domain_config=DomainConfig(**nlp_data.get("domain", {})),
                risk_config=RiskConfig(**nlp_data.get("risk", {})),
            )

            # Parse LLM configuration
            llm_data = qualitative_data.get("llm", {})
            cost_tracking_data = qualitative_data.get("cost_tracking", {})
            llm_config = LLMConfig(
                openrouter_api_key=cls._resolve_env_var(
                    llm_data.get("openrouter_api_key")
                    or llm_data.get("api_key", "${OPENROUTER_API_KEY}")
                )
                or "",
                base_url=llm_data.get("base_url", "https://openrouter.ai/api/v1"),
                primary_model=llm_data.get("primary_model")
                or llm_data.get("model", "anthropic/claude-3-haiku"),
                fallback_model=llm_data.get(
                    "fallback_model", "meta-llama/llama-3.1-8b-instruct:free"
                ),
                complex_model=llm_data.get("complex_model", "anthropic/claude-3-sonnet"),
                complexity_threshold=llm_data.get("complexity_threshold", 0.5),
                cost_threshold_per_1k=llm_data.get("cost_threshold_per_1k", 0.01),
                max_tokens=llm_data.get("max_tokens", 1000),
                temperature=llm_data.get("temperature", 0.1),
                max_group_size=llm_data.get("max_group_size", 10),
                similarity_threshold=llm_data.get("similarity_threshold", 0.8),
                requests_per_minute=llm_data.get("requests_per_minute", 200),
                max_retries=llm_data.get("max_retries", 3),
                max_daily_cost=cost_tracking_data.get("daily_budget_usd")
                or llm_data.get("max_daily_cost", 5.0),
                enable_cost_tracking=(
                    cost_tracking_data.get("enabled")
                    if cost_tracking_data.get("enabled") is not None
                    else llm_data.get("enable_cost_tracking", True)
                ),
            )

            # Parse cache configuration
            cache_data = qualitative_data.get("cache", {})
            qualitative_cache_config = QualitativeCacheConfig(
                cache_dir=cache_data.get("cache_dir", ".qualitative_cache"),
                semantic_cache_size=cache_data.get("semantic_cache_size", 10000),
                pattern_cache_ttl_hours=cache_data.get("pattern_cache_ttl_hours", 168),
                enable_pattern_learning=cache_data.get("enable_pattern_learning", True),
                learning_threshold=cache_data.get("learning_threshold", 10),
                confidence_boost_factor=cache_data.get("confidence_boost_factor", 0.1),
                enable_compression=cache_data.get("enable_compression", True),
                max_cache_size_mb=cache_data.get("max_cache_size_mb", 100),
            )

            # Create main qualitative configuration
            return QualitativeConfig(
                enabled=qualitative_data.get("enabled", True),
                batch_size=qualitative_data.get("batch_size", 1000),
                max_llm_fallback_pct=qualitative_data.get("max_llm_fallback_pct", 0.15),
                confidence_threshold=qualitative_data.get("confidence_threshold", 0.7),
                nlp_config=nlp_config,
                llm_config=llm_config,
                cache_config=qualitative_cache_config,
                enable_performance_tracking=qualitative_data.get(
                    "enable_performance_tracking", True
                ),
                target_processing_time_ms=qualitative_data.get("target_processing_time_ms", 2.0),
                min_overall_confidence=qualitative_data.get("min_overall_confidence", 0.6),
                enable_quality_feedback=qualitative_data.get("enable_quality_feedback", True),
            )

        except ImportError as e:
            click.echo(f"Warning: Qualitative analysis dependencies missing: {e}", err=True)
            click.echo("   Install with: pip install spacy scikit-learn openai tiktoken", err=True)
            return None
        except Exception as e:
            click.echo(f"Warning: Error parsing qualitative configuration: {e}", err=True)
            return None

    @classmethod
    def _process_velocity_config(cls, velocity_data: dict[str, Any]) -> VelocityConfig:
        """Process velocity report configuration section.

        Args:
            velocity_data: Velocity configuration data from YAML.

        Returns:
            VelocityConfig instance with defaults for any missing keys.
        """
        if not velocity_data:
            return VelocityConfig()
        return VelocityConfig(
            enabled=velocity_data.get("enabled", True),
            cycle_time_outlier_min_hrs=float(velocity_data.get("cycle_time_outlier_min_hrs", 0.5)),
            cycle_time_outlier_max_hrs=float(
                velocity_data.get("cycle_time_outlier_max_hrs", 720.0)
            ),
            top_n=int(velocity_data.get("top_n", 5)),
        )

    @classmethod
    def _process_activity_scoring_config(
        cls, scoring_data: dict[str, Any]
    ) -> ActivityScoringConfig:
        """Process activity scoring configuration section.

        Args:
            scoring_data: Activity scoring configuration data from YAML.

        Returns:
            ActivityScoringConfig instance with defaults for any missing keys.
        """
        if not scoring_data:
            return ActivityScoringConfig()
        return ActivityScoringConfig(
            commits_weight=float(scoring_data.get("commits_weight", 0.22)),
            prs_weight=float(scoring_data.get("prs_weight", 0.26)),
            code_impact_weight=float(scoring_data.get("code_impact_weight", 0.26)),
            complexity_weight=float(scoring_data.get("complexity_weight", 0.11)),
            ticketing_weight=float(scoring_data.get("ticketing_weight", 0.15)),
        )

    @classmethod
    def _process_quality_report_config(cls, quality_data: dict[str, Any]) -> QualityReportConfig:
        """Process quality report configuration section.

        Args:
            quality_data: Quality report configuration data from YAML.

        Returns:
            QualityReportConfig instance with defaults for any missing keys.
        """
        if not quality_data:
            return QualityReportConfig()
        return QualityReportConfig(
            enabled=quality_data.get("enabled", True),
            revert_detection_patterns=quality_data.get("revert_detection_patterns", True),
            risk_profile=quality_data.get("risk_profile", True),
            code_review_signals=quality_data.get("code_review_signals", True),
            quality_score=quality_data.get("quality_score", True),
        )

    @classmethod
    def _process_boilerplate_filter_config(cls, bp_data: dict[str, Any]) -> BoilerplateFilterConfig:
        """Process boilerplate filter configuration section (Issue #28).

        Args:
            bp_data: Boilerplate filter configuration data from YAML.

        Returns:
            BoilerplateFilterConfig instance with defaults for any missing keys.
        """
        if not bp_data:
            return BoilerplateFilterConfig()

        action = bp_data.get("action", "flag")
        valid_actions = {"flag", "exclude_from_averages", "exclude"}
        if action not in valid_actions:
            click.echo(
                f"Warning: boilerplate_filter.action '{action}' is invalid; "
                f"must be one of {sorted(valid_actions)}. Defaulting to 'flag'.",
                err=True,
            )
            action = "flag"

        return BoilerplateFilterConfig(
            enabled=bool(bp_data.get("enabled", False)),
            avg_lines_per_commit_threshold=int(bp_data.get("avg_lines_per_commit_threshold", 500)),
            total_lines_threshold=int(bp_data.get("total_lines_threshold", 10000)),
            action=action,
            flag_label=str(bp_data.get("flag_label", "boilerplate")),
        )

    @classmethod
    def _process_ai_detection_config(cls, ai_data: dict[str, Any]) -> AIDetectionConfig:
        """Process AI detection configuration section.

        Args:
            ai_data: AI detection configuration data from YAML.

        Returns:
            AIDetectionConfig instance with defaults for any missing keys.
        """
        if not ai_data:
            return AIDetectionConfig()
        return AIDetectionConfig(
            pattern_matching=ai_data.get("pattern_matching", True),
            nlp_message_scoring=ai_data.get("nlp_message_scoring", True),
            confidence_threshold=float(ai_data.get("confidence_threshold", 0.7)),
        )

    @classmethod
    def _process_teams_config(cls, teams_data: dict | list | None) -> TeamsConfig:
        """Process teams/pods configuration section.

        Args:
            teams_data: Teams configuration data — either a dict with a ``teams``
                key, a bare list of team dicts, or None.

        Returns:
            TeamsConfig instance with defaults for any missing keys.
        """
        if not teams_data:
            return TeamsConfig()

        enabled = True
        if isinstance(teams_data, list):
            teams_list = teams_data
        else:
            teams_list = teams_data.get("teams", [])
            enabled = teams_data.get("enabled", True)

        def _parse_member(m: Any) -> TeamMemberConfig:
            if isinstance(m, dict):
                return TeamMemberConfig(
                    email=m.get("email"),
                    github=m.get("github"),
                    name=m.get("name"),
                )
            # Plain string — treat as email/identifier
            return TeamMemberConfig(email=str(m))

        teams: list[TeamConfig] = []
        for t in teams_list:
            if not isinstance(t, dict):
                continue
            members = [_parse_member(m) for m in t.get("members", [])]
            pods = [
                PodConfig(
                    name=p.get("name", ""),
                    members=[_parse_member(m) for m in p.get("members", [])],
                )
                for p in t.get("pods", [])
                if isinstance(p, dict)
            ]
            teams.append(
                TeamConfig(
                    name=t.get("name", ""),
                    lead=t.get("lead"),
                    members=members,
                    pods=pods,
                )
            )

        return TeamsConfig(teams=teams, enabled=enabled)

    @classmethod
    def _process_pm_config(cls, pm_data: dict[str, Any]) -> Optional[Any]:
        """Process PM configuration section.

        Args:
            pm_data: PM configuration data

        Returns:
            PM configuration object or None
        """
        if not pm_data:
            return None

        pm_config = type("PMConfig", (), {})()  # Dynamic class

        # Parse JIRA section within PM
        if "jira" in pm_data:
            jira_pm_data = pm_data["jira"]
            jira_sub_config = type(
                "PMJIRAConfig",
                (),
                {
                    "enabled": jira_pm_data.get("enabled", True),
                    "base_url": jira_pm_data.get("base_url"),
                    "username": cls._resolve_env_var(jira_pm_data.get("username")),
                    "api_token": cls._resolve_env_var(jira_pm_data.get("api_token")),
                    "story_point_fields": jira_pm_data.get(
                        "story_point_fields",
                        ["customfield_10016", "customfield_10021", "Story Points"],
                    ),
                },
            )()
            pm_config.jira = jira_sub_config  # type: ignore[misc]

        return pm_config

    @classmethod
    def _process_pm_integration_config(
        cls, pm_integration_data: dict[str, Any]
    ) -> Optional[PMIntegrationConfig]:
        """Process PM integration configuration section.

        Args:
            pm_integration_data: PM integration configuration data

        Returns:
            PMIntegrationConfig instance or None
        """
        if not pm_integration_data:
            return None

        # Parse platform configurations
        platforms_config = {}
        platforms_data = pm_integration_data.get("platforms", {})

        for platform_name, platform_data in platforms_data.items():
            # Recursively resolve environment variables in config dictionary
            config_data = platform_data.get("config", {})
            resolved_config = cls._resolve_config_dict(config_data)

            platforms_config[platform_name] = PMPlatformConfig(
                enabled=platform_data.get("enabled", True),
                platform_type=platform_data.get("platform_type", platform_name),
                config=resolved_config,
            )

        # Parse correlation settings with defaults
        correlation_defaults = {
            "fuzzy_matching": True,
            "temporal_window_hours": 72,
            "confidence_threshold": 0.8,
        }
        correlation_config = {**correlation_defaults, **pm_integration_data.get("correlation", {})}

        return PMIntegrationConfig(
            enabled=pm_integration_data.get("enabled", False),
            primary_platform=pm_integration_data.get("primary_platform"),
            correlation=correlation_config,
            platforms=platforms_config,
        )

    @staticmethod
    def _resolve_env_var(value: Optional[str]) -> Optional[str]:
        """Resolve environment variable references in a string.

        Supports three syntaxes (all backward-compatible with prior
        exact-match behaviour):

        * ``${VAR}`` — braced reference (preferred, works anywhere in the
          string).
        * ``$VAR`` — unbraced reference (must be a valid identifier).
        * Embedded references (e.g. ``"Bearer ${TOKEN}"``).

        If a referenced variable is not set in ``os.environ``, the reference
        is replaced with an empty string and a warning is logged — this
        mirrors shell behaviour and lets callers decide whether the resulting
        value is acceptable.  For the legacy whole-string ``${VAR}`` pattern
        the function preserves the prior contract of returning ``None`` when
        the variable is unset, so downstream ``or ""`` guards continue to
        work.

        Args:
            value: Value that may contain environment variable reference(s).

        Returns:
            Resolved string, or ``None`` if the input was empty / the legacy
            whole-string reference resolved to an unset variable.
        """
        if value is None:
            return None
        if not isinstance(value, str):
            return value  # type: ignore[return-value]
        if not value:
            return None

        # Legacy whole-string ``${VAR}`` path — keep exact prior semantics so
        # callers that rely on ``None`` (rather than "") continue to work.
        if value.startswith("${") and value.endswith("}") and value.count("${") == 1:
            env_var = value[2:-1]
            resolved = os.environ.get(env_var)
            if not resolved:
                logger.warning(
                    "Environment variable %r referenced in config is not set or empty; "
                    "the resulting credential/value will be blank. Check your .env / "
                    ".env.local files and shell environment.",
                    env_var,
                )
                return None
            return resolved

        # Otherwise, perform substitution for every ``${VAR}`` / ``$VAR`` match.
        if "$" not in value:
            return value

        missing: list[str] = []

        def _sub(match: "re.Match[str]") -> str:
            name = match.group(1) or match.group(2)
            resolved = os.environ.get(name, "")
            if not resolved:
                missing.append(name)
            return resolved

        substituted = _ENV_VAR_PATTERN.sub(_sub, value)
        if missing:
            logger.warning(
                "Environment variable(s) %s referenced in config are not set or empty; "
                "substituted with empty strings. Check your .env / .env.local files.",
                ", ".join(sorted(set(missing))),
            )
        return substituted

    @classmethod
    def _resolve_config_dict(cls, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Recursively resolve environment variables in a configuration dictionary.

        Args:
            config_dict: Dictionary that may contain environment variable references

        Returns:
            Dictionary with resolved environment variables
        """
        resolved = {}
        for key, value in config_dict.items():
            if isinstance(value, str):
                # Resolve string values that might be environment variables
                resolved[key] = cls._resolve_env_var(value)
            elif isinstance(value, dict):
                # Recursively resolve nested dictionaries
                resolved[key] = cls._resolve_config_dict(value)
            elif isinstance(value, list):
                # Handle lists that might contain strings or nested dicts
                resolved[key] = [
                    (
                        cls._resolve_env_var(item)
                        if isinstance(item, str)
                        else cls._resolve_config_dict(item)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value
                ]
            else:
                # Keep other types as-is (numbers, booleans, None, etc.)
                resolved[key] = value
        return resolved

    @staticmethod
    def validate_config(config: Config) -> list[str]:
        """Validate configuration and return list of warnings.

        This method is kept for backward compatibility.

        Args:
            config: Configuration to validate

        Returns:
            List of warning messages
        """
        return ConfigValidator.validate_config(config)
