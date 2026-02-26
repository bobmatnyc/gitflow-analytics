"""Secondary config section processors for ConfigLoader.

This module provides ConfigLoaderSectionsMixin which adds output, cache, JIRA,
qualitative, PM config processing, env var resolution, and validation to
ConfigLoader via multiple inheritance.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

from .schema import (
    CacheConfig,
    Config,
    JIRAConfig,
    JIRAIntegrationConfig,
    OutputConfig,
    PMIntegrationConfig,
    PMPlatformConfig,
)
from .validator import ConfigValidator

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
            access_user=access_user,
            access_token=access_token,
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
                ),
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
            pm_config.jira = type(
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
        """Resolve environment variable references.

        Args:
            value: Value that may contain environment variable reference

        Returns:
            Resolved value or None

        Raises:
            EnvironmentVariableError: If environment variable is not set
        """
        if not value:
            return None

        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            resolved = os.environ.get(env_var)
            if not resolved:
                # Note: We don't raise here directly, let the caller handle it
                # based on whether the field is required
                return None
            return resolved

        return value

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
