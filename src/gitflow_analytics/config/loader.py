"""YAML configuration loading and environment variable expansion."""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from dotenv import load_dotenv

from .errors import (
    ConfigurationError,
    EnvironmentVariableError,
    InvalidValueError,
    handle_yaml_error,
)
from .profiles import ProfileManager
from .repository import RepositoryManager
from .schema import (
    AnalysisConfig,
    BranchAnalysisConfig,
    CacheConfig,
    CommitClassificationConfig,
    Config,
    GitHubConfig,
    JIRAConfig,
    JIRAIntegrationConfig,
    LLMClassificationConfig,
    MLCategorization,
    OutputConfig,
    PMIntegrationConfig,
    PMPlatformConfig,
    RepositoryConfig,
)
from .validator import ConfigValidator

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate configuration from YAML files."""

    # Default exclude paths for common boilerplate/generated files
    DEFAULT_EXCLUDE_PATHS = [
        "**/node_modules/**",
        "**/vendor/**",
        "**/dist/**",
        "**/build/**",
        "**/.next/**",
        "**/__pycache__/**",
        "**/*.min.js",
        "**/*.min.css",
        "**/*.bundle.js",
        "**/*.bundle.css",
        "**/package-lock.json",
        "**/yarn.lock",
        "**/poetry.lock",
        "**/Pipfile.lock",
        "**/composer.lock",
        "**/Gemfile.lock",
        "**/Cargo.lock",
        "**/go.sum",
        "**/*.generated.*",
        "**/generated/**",
        "**/coverage/**",
        "**/.coverage/**",
        "**/htmlcov/**",
        "**/*.map",
        # Additional framework/boilerplate patterns
        "**/public/assets/**",
        "**/public/css/**",
        "**/public/js/**",
        "**/public/fonts/**",
        "**/public/build/**",
        "**/storage/framework/**",
        "**/bootstrap/cache/**",
        "**/.nuxt/**",
        "**/.cache/**",
        "**/cache/**",
        "**/*.lock",
        "**/*.log",
        "**/logs/**",
        "**/tmp/**",
        "**/temp/**",
        "**/.sass-cache/**",
        "**/bower_components/**",
        # Database migrations and seeds (often auto-generated)
        "**/migrations/*.php",
        "**/database/migrations/**",
        "**/db/migrate/**",
        # Compiled assets
        "**/public/mix-manifest.json",
        "**/public/hot",
        "**/*.map.js",
        "**/webpack.mix.js",
        # IDE and OS files
        "**/.idea/**",
        "**/.vscode/**",
        "**/.DS_Store",
        "**/Thumbs.db",
        # Generated documentation (but not source docs)
        "**/docs/build/**",
        "**/docs/_build/**",
        "**/documentation/build/**",
        "**/site/**",  # For mkdocs generated sites
        # Test coverage
        "**/test-results/**",
        "**/.nyc_output/**",
        # Framework-specific
        "**/artisan",
        "**/spark",
        "**/.env",
        "**/.env.*",
        "**/storage/logs/**",
        "**/storage/debugbar/**",
        # CMS-specific patterns
        "**/wp-content/uploads/**",
        "**/wp-content/cache/**",
        "**/uploads/**",
        "**/media/**",
        "**/static/**",
        "**/staticfiles/**",
        # More aggressive filtering for generated content
        "**/*.sql",
        "**/*.dump",
        "**/backups/**",
        "**/backup/**",
        "**/*.bak",
        # Compiled/concatenated files (only in build/dist directories)
        "**/dist/**/all.js",
        "**/dist/**/all.css",
        "**/build/**/all.js",
        "**/build/**/all.css",
        "**/public/**/app.js",
        "**/public/**/app.css",
        "**/dist/**/app.js",
        "**/dist/**/app.css",
        "**/build/**/app.js",
        "**/build/**/app.css",
        "**/public/**/main.js",
        "**/public/**/main.css",
        "**/dist/**/main.js",
        "**/dist/**/main.css",
        "**/build/**/main.js",
        "**/build/**/main.css",
        "**/bundle.*",
        "**/chunk.*",
        "**/*-chunk-*",
        "**/*.chunk.*",
        # Framework scaffolding
        "**/scaffolding/**",
        "**/stubs/**",
        "**/templates/**",
        "**/views/vendor/**",
        "**/resources/views/vendor/**",
        # Package managers
        "**/packages/**",
        "**/node_modules/**",
        "**/.pnpm/**",
        "**/.yarn/**",
        # Build artifacts
        "**/out/**",
        "**/output/**",
        "**/.parcel-cache/**",
        "**/parcel-cache/**",
        # Large data files (only in specific directories)
        "**/data/*.csv",
        "**/data/*.json",
        "**/fixtures/*.json",
        "**/seeds/*.json",
        "**/*.geojson",
        "**/package.json.bak",
        "**/composer.json.bak",
        # Exclude large framework upgrades
        "**/upgrade/**",
        "**/upgrades/**",
        # Common CMS patterns (specific to avoid excluding legitimate source)
        "**/wordpress/wp-core/**",
        "**/drupal/core/**",
        "**/joomla/libraries/cms/**",
        "**/modules/**/tests/**",
        "**/plugins/**/vendor/**",
        "**/themes/**/vendor/**",
        "**/themes/**/node_modules/**",
        # Framework-specific third-party directories (not generic lib/libs)
        "**/vendor/**",
        "**/vendors/**",
        "**/bower_components/**",
        # Only exclude specific known third-party package directories
        "**/third-party/packages/**",
        "**/third_party/packages/**",
        "**/external/vendor/**",
        "**/external/packages/**",
        # Generated assets
        "**/*.min.js",
        "**/*.min.css",
        "**/dist/**",
        "**/build/**",
        "**/compiled/**",
        # Package lock files
        "**/composer.lock",
        "**/package-lock.json",
        "**/yarn.lock",
        "**/pnpm-lock.yaml",
        # Documentation/assets
        "**/*.pdf",
        "**/*.doc",
        "**/*.docx",
        "**/fonts/**",
        "**/font/**",
        # Database/migrations (auto-generated files)
        "**/migrations/*.php",
        "**/database/migrations/**",
    ]

    @classmethod
    def load(cls, config_path: Union[Path, str]) -> Config:
        """Load configuration from YAML file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Loaded and validated configuration

        Raises:
            ConfigurationError: If configuration is invalid
            YAMLParseError: If YAML parsing fails
        """
        # Ensure config_path is a Path object
        config_path = Path(config_path)

        # Load environment variables
        cls._load_environment(config_path)

        # Load and parse YAML
        data = cls._load_yaml(config_path)

        # Check for configuration profile
        if "profile" in data:
            data = ProfileManager.apply_profile(data, data["profile"])

        # Check for base configuration extension
        if "extends" in data:
            base_data = cls._load_base_config(data["extends"], config_path)
            data = ProfileManager._deep_merge(base_data, data)

        # Validate version
        cls._validate_version(data)

        # Process configuration sections
        github_config = cls._process_github_config(data.get("github", {}), config_path)
        repositories = cls._process_repositories(data, github_config, config_path)
        analysis_config = cls._process_analysis_config(data.get("analysis", {}), config_path)
        output_config = cls._process_output_config(data.get("output", {}), config_path)
        cache_config = cls._process_cache_config(data.get("cache", {}), config_path)
        jira_config = cls._process_jira_config(data.get("jira", {}), config_path)
        jira_integration_config = cls._process_jira_integration_config(
            data.get("jira_integration", {})
        )

        # Check for qualitative config in both top-level and nested under analysis
        # Prioritize top-level for backward compatibility, but support nested location
        qualitative_data = data.get("qualitative", {})
        if not qualitative_data and "analysis" in data:
            qualitative_data = data["analysis"].get("qualitative", {})
        qualitative_config = cls._process_qualitative_config(qualitative_data)

        pm_config = cls._process_pm_config(data.get("pm", {}))
        pm_integration_config = cls._process_pm_integration_config(data.get("pm_integration", {}))

        # Create configuration object
        config = Config(
            repositories=repositories,
            github=github_config,
            analysis=analysis_config,
            output=output_config,
            cache=cache_config,
            jira=jira_config,
            jira_integration=jira_integration_config,
            pm=pm_config,
            pm_integration=pm_integration_config,
            qualitative=qualitative_config,
        )

        # Validate configuration
        warnings = ConfigValidator.validate_config(config)
        if warnings:
            for warning in warnings:
                print(f"⚠️  {warning}")

        return config

    @classmethod
    def _load_environment(cls, config_path: Path) -> None:
        """Load environment variables from .env file if present.

        Args:
            config_path: Path to configuration file
        """
        config_dir = config_path.parent
        env_file = config_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=True)
            print(f"📋 Loaded environment variables from {env_file}")

    @classmethod
    def _load_yaml(cls, config_path: Path) -> dict[str, Any]:
        """Load and parse YAML file.

        Args:
            config_path: Path to YAML file

        Returns:
            Parsed YAML data

        Raises:
            YAMLParseError: If YAML parsing fails
            ConfigurationError: If file is invalid
        """
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            handle_yaml_error(e, config_path)
        except FileNotFoundError as e:
            raise ConfigurationError(
                f"Configuration file not found: {config_path}", config_path
            ) from e
        except PermissionError as e:
            raise ConfigurationError(
                f"Permission denied reading configuration file: {config_path}", config_path
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Failed to read configuration file: {e}", config_path) from e

        # Handle empty or null YAML files
        if data is None:
            raise ConfigurationError(
                "Configuration file is empty or contains only null values",
                config_path,
                suggestion=(
                    "Add proper YAML configuration content to the file.\n"
                    "   Example minimal configuration:\n"
                    "   ```yaml\n"
                    '   version: "1.0"\n'
                    "   github:\n"
                    '     token: "${GITHUB_TOKEN}"\n'
                    '     owner: "your-username"\n'
                    "   repositories:\n"
                    '     - name: "your-repo"\n'
                    '       path: "/path/to/repo"\n'
                    "   ```"
                ),
            )

        # Validate that data is a dictionary
        if not isinstance(data, dict):
            raise InvalidValueError(
                "root",
                type(data).__name__,
                "Configuration file must contain a YAML object (key-value pairs)",
                config_path,
            )

        return data

    @classmethod
    def _load_base_config(cls, base_path: str, config_path: Path) -> dict[str, Any]:
        """Load base configuration to extend from.

        Args:
            base_path: Path to base configuration (relative or absolute)
            config_path: Path to current configuration file

        Returns:
            Base configuration data
        """
        # Resolve base path relative to current config
        if not Path(base_path).is_absolute():
            base_path = config_path.parent / base_path
        else:
            base_path = Path(base_path)

        return cls._load_yaml(base_path)

    @classmethod
    def _validate_version(cls, data: dict[str, Any]) -> None:
        """Validate configuration version.

        Args:
            data: Configuration data

        Raises:
            InvalidValueError: If version is not supported
        """
        version = data.get("version", "1.0")
        if version not in ["1.0"]:
            raise InvalidValueError(
                "version", version, "Unsupported configuration version", None, valid_values=["1.0"]
            )

    @classmethod
    def _process_github_config(cls, github_data: dict[str, Any], config_path: Path) -> GitHubConfig:
        """Process GitHub configuration section.

        Args:
            github_data: GitHub configuration data
            config_path: Path to configuration file

        Returns:
            GitHubConfig instance
        """
        # Resolve GitHub token
        github_token = cls._resolve_env_var(github_data.get("token"))
        if github_data.get("token") and not github_token:
            raise EnvironmentVariableError("GITHUB_TOKEN", "GitHub", config_path)

        return GitHubConfig(
            token=github_token,
            owner=cls._resolve_env_var(github_data.get("owner")),
            organization=cls._resolve_env_var(github_data.get("organization")),
            base_url=github_data.get("base_url", "https://api.github.com"),
            max_retries=github_data.get("rate_limit", {}).get("max_retries", 3),
            backoff_factor=github_data.get("rate_limit", {}).get("backoff_factor", 2),
        )

    @classmethod
    def _process_repositories(
        cls, data: dict[str, Any], github_config: GitHubConfig, config_path: Path
    ) -> list[RepositoryConfig]:
        """Process repositories configuration.

        Args:
            data: Configuration data
            github_config: GitHub configuration
            config_path: Path to configuration file

        Returns:
            List of RepositoryConfig instances
        """
        repositories = []
        repo_manager = RepositoryManager(github_config)

        # Handle organization-based repository discovery
        if github_config.organization and not data.get("repositories"):
            # Organization specified but no explicit repositories - will be discovered at runtime
            pass
        else:
            # Process explicitly defined repositories
            for i, repo_data in enumerate(data.get("repositories", [])):
                repo_config = repo_manager.process_repository_config(repo_data, i, config_path)
                repositories.append(repo_config)

        # Allow empty repositories list if organization is specified
        if not repositories and not github_config.organization:
            raise ConfigurationError(
                "No repositories defined and no organization specified for discovery",
                config_path,
                suggestion=(
                    "Either define repositories explicitly or specify a GitHub organization:\n"
                    "   repositories:\n"
                    '     - name: "repo-name"\n'
                    '       path: "/path/to/repo"\n'
                    "   OR\n"
                    "   github:\n"
                    '     organization: "your-org"'
                ),
            )

        return repositories

    @classmethod
    def _process_analysis_config(
        cls, analysis_data: dict[str, Any], config_path: Path
    ) -> AnalysisConfig:
        """Process analysis configuration section.

        Args:
            analysis_data: Analysis configuration data
            config_path: Path to configuration file

        Returns:
            AnalysisConfig instance
        """
        # Validate settings
        ConfigValidator.validate_analysis_config(analysis_data, config_path)

        # Process exclude paths
        user_exclude_paths = analysis_data.get("exclude", {}).get("paths", [])
        exclude_paths = user_exclude_paths if user_exclude_paths else cls.DEFAULT_EXCLUDE_PATHS

        # Process ML categorization settings
        ml_data = analysis_data.get("ml_categorization", {})
        ml_categorization_config = MLCategorization(
            enabled=ml_data.get("enabled", True),
            min_confidence=ml_data.get("min_confidence", 0.6),
            semantic_weight=ml_data.get("semantic_weight", 0.7),
            file_pattern_weight=ml_data.get("file_pattern_weight", 0.3),
            hybrid_threshold=ml_data.get("hybrid_threshold", 0.5),
            cache_duration_days=ml_data.get("cache_duration_days", 30),
            batch_size=ml_data.get("batch_size", 100),
            enable_caching=ml_data.get("enable_caching", True),
            spacy_model=ml_data.get("spacy_model", "en_core_web_sm"),
        )

        # Process commit classification settings
        classification_data = analysis_data.get("commit_classification", {})
        commit_classification_config = CommitClassificationConfig(
            enabled=classification_data.get("enabled", True),
            confidence_threshold=classification_data.get("confidence_threshold", 0.5),
            batch_size=classification_data.get("batch_size", 100),
            auto_retrain=classification_data.get("auto_retrain", True),
            retrain_threshold_days=classification_data.get("retrain_threshold_days", 30),
            model=classification_data.get("model", {}),
            feature_extraction=classification_data.get("feature_extraction", {}),
            training=classification_data.get("training", {}),
            categories=classification_data.get("categories", {}),
        )

        # Process LLM classification configuration
        llm_classification_data = analysis_data.get("llm_classification", {})
        llm_classification_config = LLMClassificationConfig(
            enabled=llm_classification_data.get("enabled", False),
            api_key=cls._resolve_env_var(llm_classification_data.get("api_key")),
            api_base_url=llm_classification_data.get(
                "api_base_url", "https://openrouter.ai/api/v1"
            ),
            model=llm_classification_data.get("model", "mistralai/mistral-7b-instruct"),
            confidence_threshold=llm_classification_data.get("confidence_threshold", 0.7),
            max_tokens=llm_classification_data.get("max_tokens", 50),
            temperature=llm_classification_data.get("temperature", 0.1),
            timeout_seconds=llm_classification_data.get("timeout_seconds", 30.0),
            cache_duration_days=llm_classification_data.get("cache_duration_days", 90),
            enable_caching=llm_classification_data.get("enable_caching", True),
            max_daily_requests=llm_classification_data.get("max_daily_requests", 1000),
            domain_terms=llm_classification_data.get("domain_terms", {}),
        )

        # Process branch analysis settings
        branch_data = analysis_data.get("branch_analysis", {})
        branch_analysis_config = (
            BranchAnalysisConfig(**branch_data) if branch_data else BranchAnalysisConfig()
        )

        # Process qualitative configuration (support nested under analysis)
        qualitative_data = analysis_data.get("qualitative", {})
        qualitative_config = (
            cls._process_qualitative_config(qualitative_data) if qualitative_data else None
        )

        # Process aliases file and manual identity mappings
        manual_mappings = list(analysis_data.get("identity", {}).get("manual_mappings", []))
        aliases_file_path = None

        # Load aliases from external file if specified
        aliases_file = analysis_data.get("identity", {}).get("aliases_file")
        if aliases_file:
            aliases_path = Path(aliases_file).expanduser()
            # Make relative paths relative to config file directory
            if not aliases_path.is_absolute():
                aliases_path = config_path.parent / aliases_path

            aliases_file_path = aliases_path

            # Load and merge aliases if file exists
            if aliases_path.exists():
                try:
                    from .aliases import AliasesManager

                    aliases_mgr = AliasesManager(aliases_path)
                    # Merge aliases with existing manual mappings
                    manual_mappings.extend(aliases_mgr.to_manual_mappings())
                    logger.info(
                        f"Loaded {len(aliases_mgr.aliases)} identity aliases from {aliases_path}"
                    )
                except Exception as e:
                    logger.warning(f"Could not load aliases file {aliases_path}: {e}")
            else:
                logger.warning(f"Aliases file not found: {aliases_path}")

        return AnalysisConfig(
            story_point_patterns=analysis_data.get(
                "story_point_patterns",
                [
                    r"(?:story\s*points?|sp|pts?)\s*[:=]\s*(\d+)",
                    r"\[(\d+)\s*(?:sp|pts?)\]",
                    r"#(\d+)sp",
                ],
            ),
            exclude_authors=analysis_data.get("exclude", {}).get(
                "authors", ["dependabot[bot]", "renovate[bot]"]
            ),
            exclude_message_patterns=analysis_data.get("exclude", {}).get("message_patterns", []),
            exclude_paths=exclude_paths,
            exclude_merge_commits=analysis_data.get("exclude_merge_commits", False),
            similarity_threshold=analysis_data.get("identity", {}).get(
                "similarity_threshold", 0.85
            ),
            manual_identity_mappings=manual_mappings,
            aliases_file=aliases_file_path,
            default_ticket_platform=analysis_data.get("default_ticket_platform"),
            branch_mapping_rules=analysis_data.get("branch_mapping_rules", {}),
            ticket_platforms=analysis_data.get("ticket_platforms"),
            auto_identity_analysis=analysis_data.get("identity", {}).get("auto_analysis", True),
            branch_patterns=analysis_data.get("branch_patterns"),
            branch_analysis=branch_analysis_config,
            ml_categorization=ml_categorization_config,
            commit_classification=commit_classification_config,
            llm_classification=llm_classification_config,
            security=analysis_data.get("security", {}),
            qualitative=qualitative_config,
        )

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
            print(f"⚠️  Qualitative analysis dependencies missing: {e}")
            print("   Install with: pip install spacy scikit-learn openai tiktoken")
            return None
        except Exception as e:
            print(f"⚠️  Error parsing qualitative configuration: {e}")
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
