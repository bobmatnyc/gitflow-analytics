"""YAML configuration loading and environment variable expansion.

Core loading + analysis-section processing live here.  Secondary section
processors (output, cache, JIRA, qualitative, PM, env-var resolution,
validation) live in loader_sections.py (ConfigLoaderSectionsMixin).
"""

import logging
from pathlib import Path
from typing import Any, Union

import click
import yaml
from dotenv import load_dotenv

from .errors import (
    ConfigurationError,
    EnvironmentVariableError,
    InvalidValueError,
    handle_yaml_error,
)
from .loader_sections import ConfigLoaderSectionsMixin  # noqa: F401
from .profiles import ProfileManager
from .repository import RepositoryManager
from .schema import (
    AnalysisConfig,
    BranchAnalysisConfig,
    CommitClassificationConfig,
    Config,
    GitHubConfig,
    LLMClassificationConfig,
    MLCategorization,
    RepositoryConfig,
)
from .validator import ConfigValidator

logger = logging.getLogger(__name__)


class ConfigLoader(ConfigLoaderSectionsMixin):
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

        # BACKWARD COMPATIBILITY: Convert legacy 'developer_aliases' to modern format
        # Handle top-level developer_aliases and move to analysis.identity.manual_mappings
        if "developer_aliases" in data:
            logger.info(
                "Found legacy 'developer_aliases' configuration, converting to modern format"
            )

            # Ensure analysis section exists
            if "analysis" not in data:
                data["analysis"] = {}
            if "identity" not in data["analysis"]:
                data["analysis"]["identity"] = {}
            if "manual_mappings" not in data["analysis"]["identity"]:
                data["analysis"]["identity"]["manual_mappings"] = []

            # Convert legacy format to modern format
            for canonical_name, emails in data["developer_aliases"].items():
                if isinstance(emails, list) and emails:
                    # Use first email as primary, all emails as aliases
                    # Note: aliases list includes primary_email to ensure all emails map to canonical identity
                    primary_email = emails[0]
                    data["analysis"]["identity"]["manual_mappings"].append(
                        {"name": canonical_name, "primary_email": primary_email, "aliases": emails}
                    )
                    logger.info(
                        f"Converted legacy alias: {canonical_name} → {primary_email} with aliases: {emails}"
                    )

            # Remove the old format and warn user
            del data["developer_aliases"]
            logger.warning(
                "DEPRECATED: 'developer_aliases' is deprecated. Please use 'analysis.identity.manual_mappings' instead. "
                "See documentation for the new format."
            )

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
                click.echo(f"Warning: {warning}", err=True)

        return config

    @classmethod
    def _load_environment(cls, config_path: Path) -> None:
        """Load environment variables from .env and .env.local files if present.

        Searches for env files in standard locations, loading base .env files
        first then .env.local files as overrides (following standard convention).

        WHY: .env.local is the conventional way to provide machine-local overrides
        that should not be committed to version control. Loading all found files
        (base first, then local overrides) allows layered configuration while
        still respecting the override priority order.

        Args:
            config_path: Path to configuration file
        """
        env_files_to_check = cls._find_env_files(config_path)

        loaded_any = False
        for env_file in env_files_to_check:
            if env_file.exists():
                load_dotenv(env_file, override=True)
                logger.debug(f"Loaded environment variables from {env_file}")
                loaded_any = True

        if not loaded_any:
            logger.debug("No .env file found in any of the standard locations")

    @classmethod
    def _find_env_files(cls, config_path: Path) -> list[Path]:
        """Find potential .env file locations in priority order.

        WHY: Returns both .env and .env.local at each location so that .env.local
        can override .env values on a per-machine basis without changing committed
        config. The order is: base files first (low priority), local overrides last
        (high priority). All found files are loaded in sequence so later files win.

        Args:
            config_path: Path to configuration file

        Returns:
            List of potential .env file paths in load order (base before local)
        """
        # Collect unique search directories in discovery order
        search_dirs: list[Path] = []

        # 1. Same directory as config file
        config_dir = config_path.parent
        search_dirs.append(config_dir)

        # 2. Current working directory (if different from config dir)
        cwd = Path.cwd()
        if cwd not in search_dirs:
            search_dirs.append(cwd)

        # 3. Project root detected via .git (if different from above)
        git_root = cls._find_git_root(config_path)
        if git_root and git_root not in search_dirs:
            search_dirs.append(git_root)

        # Build the final ordered list: base .env files first, then .env.local
        # overrides. This ensures .env.local values always win.
        base_files: list[Path] = []
        local_files: list[Path] = []
        seen: set[Path] = set()

        for directory in search_dirs:
            base = directory / ".env"
            local = directory / ".env.local"
            if base not in seen:
                base_files.append(base)
                seen.add(base)
            if local not in seen:
                local_files.append(local)
                seen.add(local)

        # 4. User home directory special file (highest priority, appended last)
        home_env = Path.home() / ".gitflow-analytics.env"
        if home_env not in seen:
            local_files.append(home_env)

        return base_files + local_files

    @classmethod
    def _find_git_root(cls, start_path: Path) -> Path | None:
        """Find Git root directory by looking for .git directory.

        Args:
            start_path: File or directory path to start searching from

        Returns:
            Path to Git root directory, or None if not found
        """
        # Start from the directory containing the file, or the directory itself
        if start_path.is_file() or (not start_path.exists() and start_path.suffix):
            # It's a file path (existing or not), use parent directory
            current_path = start_path.parent
        else:
            # It's a directory path
            current_path = start_path

        while current_path != current_path.parent:  # Stop at filesystem root
            git_dir = current_path / ".git"
            if git_dir.exists():
                return current_path
            current_path = current_path.parent

        return None

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
            fetch_pr_reviews=bool(github_data.get("fetch_pr_reviews", False)),
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
            provider=llm_classification_data.get("provider", "auto"),
            api_key=cls._resolve_env_var(llm_classification_data.get("api_key")),
            api_base_url=llm_classification_data.get(
                "api_base_url", "https://openrouter.ai/api/v1"
            ),
            model=llm_classification_data.get("model", "mistralai/mistral-7b-instruct"),
            aws_region=llm_classification_data.get("aws_region"),
            aws_profile=llm_classification_data.get("aws_profile"),
            bedrock_model_id=llm_classification_data.get(
                "bedrock_model_id", "anthropic.claude-3-haiku-20240307-v1:0"
            ),
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
