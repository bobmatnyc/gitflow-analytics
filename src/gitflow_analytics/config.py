"""Configuration management for GitFlow Analytics."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import yaml
from dotenv import load_dotenv

if TYPE_CHECKING:
    from .qualitative.models.schemas import QualitativeConfig


@dataclass
class RepositoryConfig:
    """Configuration for a single repository."""

    name: str
    path: Path
    github_repo: Optional[str] = None
    project_key: Optional[str] = None
    branch: Optional[str] = None

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser().resolve()
        if not self.project_key:
            self.project_key = self.name.upper().replace("-", "_")


@dataclass
class GitHubConfig:
    """GitHub API configuration."""

    token: Optional[str] = None
    owner: Optional[str] = None
    organization: Optional[str] = None
    base_url: str = "https://api.github.com"
    max_retries: int = 3
    backoff_factor: int = 2

    def get_repo_full_name(self, repo_name: str) -> str:
        """Get full repository name including owner."""
        if "/" in repo_name:
            return repo_name
        if self.owner:
            return f"{self.owner}/{repo_name}"
        raise ValueError(f"Repository {repo_name} needs owner specified")


@dataclass
class MLCategorization:
    """ML-based commit categorization configuration."""
    
    enabled: bool = True
    min_confidence: float = 0.6
    semantic_weight: float = 0.7
    file_pattern_weight: float = 0.3
    hybrid_threshold: float = 0.5  # Confidence threshold for using ML vs rule-based
    cache_duration_days: int = 30
    batch_size: int = 100
    enable_caching: bool = True
    spacy_model: str = "en_core_web_sm"  # Preferred spaCy model


@dataclass
class LLMClassificationConfig:
    """LLM-based commit classification configuration.
    
    This configuration enables Large Language Model-based commit classification
    via OpenRouter API for more accurate and context-aware categorization.
    """
    
    # Enable/disable LLM classification
    enabled: bool = False  # Disabled by default to avoid unexpected API costs
    
    # OpenRouter API configuration
    api_key: Optional[str] = None  # Set via environment variable or config
    api_base_url: str = "https://openrouter.ai/api/v1"
    model: str = "mistralai/mistral-7b-instruct"  # Fast, affordable model
    
    # Alternative models for different use cases:
    # - "meta-llama/llama-3-8b-instruct" (Higher accuracy, slightly more expensive)
    # - "openai/gpt-3.5-turbo" (Good balance, more expensive)
    
    # Classification parameters
    confidence_threshold: float = 0.7  # Minimum confidence for LLM predictions
    max_tokens: int = 50  # Keep responses short for cost optimization
    temperature: float = 0.1  # Low temperature for consistent results
    timeout_seconds: float = 30.0  # API request timeout
    
    # Caching configuration (aggressive caching for cost optimization)
    cache_duration_days: int = 90  # Long cache duration
    enable_caching: bool = True
    
    # Cost and rate limiting
    max_daily_requests: int = 1000  # Daily API request limit
    
    # Domain-specific terms for better classification accuracy
    domain_terms: dict[str, list[str]] = field(default_factory=lambda: {
        "media": [
            "video", "audio", "streaming", "player", "media", "content",
            "broadcast", "live", "recording", "episode", "program", "tv",
            "radio", "podcast", "channel", "playlist"
        ],
        "localization": [
            "translation", "i18n", "l10n", "locale", "language", "spanish",
            "french", "german", "italian", "portuguese", "multilingual",
            "translate", "localize", "regional"
        ],
        "integration": [
            "api", "webhook", "third-party", "external", "service",
            "integration", "sync", "import", "export", "connector",
            "oauth", "auth", "authentication", "sso"
        ],
        "content": [
            "copy", "text", "wording", "messaging", "editorial", "article",
            "blog", "news", "story", "caption", "title", "headline",
            "description", "summary", "metadata"
        ]
    })
    
    # Fallback behavior when LLM is unavailable
    fallback_to_rules: bool = True  # Fall back to rule-based classification
    fallback_to_ml: bool = True     # Fall back to existing ML classification


@dataclass
class CommitClassificationConfig:
    """Configuration for commit classification system.
    
    This configuration controls the Random Forest-based commit classification
    system that analyzes commits to categorize them into types like feature,
    bugfix, refactor, docs, test, etc.
    """
    
    enabled: bool = True
    confidence_threshold: float = 0.5  # Minimum confidence for reliable predictions
    batch_size: int = 100  # Commits processed per batch
    auto_retrain: bool = True  # Automatically check if model needs retraining
    retrain_threshold_days: int = 30  # Days after which to suggest retraining
    
    # Model hyperparameters
    model: dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,  # Number of trees in random forest
        'max_depth': 20,  # Maximum depth of trees
        'min_samples_split': 5,  # Minimum samples to split a node
        'min_samples_leaf': 2,  # Minimum samples at leaf node
        'random_state': 42,  # For reproducible results
        'n_jobs': -1  # Use all available CPU cores
    })
    
    # Feature extraction settings
    feature_extraction: dict[str, Any] = field(default_factory=lambda: {
        'enable_temporal_features': True,
        'enable_author_features': True,
        'enable_file_analysis': True,
        'keyword_categories': [
            'feature', 'bugfix', 'refactor', 'docs', 'test', 'config',
            'security', 'performance', 'ui', 'api', 'database', 'deployment'
        ]
    })
    
    # Training settings
    training: dict[str, Any] = field(default_factory=lambda: {
        'validation_split': 0.2,  # Fraction for validation
        'min_training_samples': 20,  # Minimum samples needed for training
        'cross_validation_folds': 5,  # K-fold cross validation
        'class_weight': 'balanced'  # Handle class imbalance
    })
    
    # Supported classification categories
    categories: dict[str, str] = field(default_factory=lambda: {
        'feature': 'New functionality or capabilities',
        'bugfix': 'Bug fixes and error corrections',
        'refactor': 'Code restructuring and optimization',
        'docs': 'Documentation changes and updates',
        'test': 'Testing-related changes',
        'config': 'Configuration and settings changes',
        'chore': 'Maintenance and housekeeping tasks',
        'security': 'Security-related changes',
        'hotfix': 'Emergency production fixes',
        'style': 'Code style and formatting changes',
        'build': 'Build system and dependency changes',
        'ci': 'Continuous integration changes',
        'revert': 'Reverts of previous changes',
        'merge': 'Merge commits and integration',
        'wip': 'Work in progress commits'
    })


@dataclass
class BranchAnalysisConfig:
    """Configuration for branch analysis optimization.
    
    This configuration controls how branches are analyzed to prevent performance
    issues on large organizations with many repositories and branches.
    """
    
    # Branch analysis strategy
    strategy: str = "smart"  # Options: "all", "smart", "main_only"
    
    # Smart analysis parameters
    max_branches_per_repo: int = 50  # Maximum branches to analyze per repository
    active_days_threshold: int = 90  # Days to consider a branch "active"
    include_main_branches: bool = True  # Always include main/master branches
    
    # Branch name patterns to always include/exclude
    always_include_patterns: list[str] = field(default_factory=lambda: [
        r"^(main|master|develop|dev)$",  # Main development branches
        r"^release/.*",  # Release branches
        r"^hotfix/.*"   # Hotfix branches
    ])
    
    always_exclude_patterns: list[str] = field(default_factory=lambda: [
        r"^dependabot/.*",  # Dependabot branches
        r"^renovate/.*",   # Renovate branches
        r".*-backup$",     # Backup branches
        r".*-temp$"        # Temporary branches
    ])
    
    # Performance limits
    enable_progress_logging: bool = True  # Log branch analysis progress
    branch_commit_limit: int = 1000  # Max commits to analyze per branch
    

@dataclass
class AnalysisConfig:
    """Analysis-specific configuration."""

    story_point_patterns: list[str] = field(default_factory=list)
    exclude_authors: list[str] = field(default_factory=list)
    exclude_message_patterns: list[str] = field(default_factory=list)
    exclude_paths: list[str] = field(default_factory=list)
    similarity_threshold: float = 0.85
    manual_identity_mappings: list[dict[str, Any]] = field(default_factory=list)
    default_ticket_platform: Optional[str] = None
    branch_mapping_rules: dict[str, list[str]] = field(default_factory=dict)
    ticket_platforms: Optional[list[str]] = None
    auto_identity_analysis: bool = True  # Enable automatic identity analysis by default
    branch_analysis: BranchAnalysisConfig = field(default_factory=BranchAnalysisConfig)
    ml_categorization: MLCategorization = field(default_factory=MLCategorization)
    commit_classification: CommitClassificationConfig = field(default_factory=CommitClassificationConfig)
    llm_classification: LLMClassificationConfig = field(default_factory=LLMClassificationConfig)


@dataclass
class OutputConfig:
    """Output configuration."""

    directory: Optional[Path] = None
    formats: list[str] = field(default_factory=lambda: ["csv", "markdown"])
    csv_delimiter: str = ","
    csv_encoding: str = "utf-8"
    anonymize_enabled: bool = False
    anonymize_fields: list[str] = field(default_factory=list)
    anonymize_method: str = "hash"


@dataclass
class CacheConfig:
    """Cache configuration."""

    directory: Path = Path(".gitflow-cache")
    ttl_hours: int = 168
    max_size_mb: int = 500


@dataclass
class JIRAConfig:
    """JIRA configuration."""

    access_user: str
    access_token: str
    base_url: Optional[str] = None


@dataclass
class JIRAIntegrationConfig:
    """JIRA integration specific configuration."""

    enabled: bool = True
    fetch_story_points: bool = True
    project_keys: list[str] = field(default_factory=list)
    story_point_fields: list[str] = field(
        default_factory=lambda: ["customfield_10016", "customfield_10021", "Story Points"]
    )


@dataclass
class PMPlatformConfig:
    """Base PM platform configuration."""

    enabled: bool = True
    platform_type: str = ""
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class PMIntegrationConfig:
    """PM framework integration configuration."""

    enabled: bool = False
    primary_platform: Optional[str] = None
    correlation: dict[str, Any] = field(default_factory=dict)
    platforms: dict[str, PMPlatformConfig] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration container."""

    repositories: list[RepositoryConfig]
    github: GitHubConfig
    analysis: AnalysisConfig
    output: OutputConfig
    cache: CacheConfig
    jira: Optional[JIRAConfig] = None
    jira_integration: Optional[JIRAIntegrationConfig] = None
    pm_integration: Optional[PMIntegrationConfig] = None
    qualitative: Optional["QualitativeConfig"] = None

    def discover_organization_repositories(
        self, clone_base_path: Optional[Path] = None
    ) -> list[RepositoryConfig]:
        """Discover repositories from GitHub organization.

        Args:
            clone_base_path: Base directory where repos should be cloned/found.
                           If None, uses output directory.

        Returns:
            List of discovered repository configurations.
        """
        if not self.github.organization or not self.github.token:
            return []

        from github import Github

        github_client = Github(self.github.token, base_url=self.github.base_url)

        try:
            org = github_client.get_organization(self.github.organization)
            discovered_repos = []

            base_path = clone_base_path or self.output.directory
            if base_path is None:
                raise ValueError("No base path available for repository cloning")

            for repo in org.get_repos():
                # Skip archived repositories
                if repo.archived:
                    continue

                # Create repository configuration
                repo_path = base_path / repo.name
                repo_config = RepositoryConfig(
                    name=repo.name,
                    path=repo_path,
                    github_repo=repo.full_name,
                    project_key=repo.name.upper().replace("-", "_"),
                    branch=repo.default_branch,
                )
                discovered_repos.append(repo_config)

            return discovered_repos

        except Exception as e:
            raise ValueError(
                f"Failed to discover repositories from organization {self.github.organization}: {e}"
            ) from e


class ConfigLoader:
    """Load and validate configuration from YAML files."""

    @classmethod
    def load(cls, config_path: Path) -> Config:
        """Load configuration from YAML file."""
        # Load .env file from the same directory as the config file if it exists
        config_dir = config_path.parent
        env_file = config_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=True)
            print(f"📋 Loaded environment variables from {env_file}")

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            cls._handle_yaml_error(e, config_path)
        except FileNotFoundError as e:
            raise ValueError(f"Configuration file not found: {config_path}") from e
        except PermissionError as e:
            raise ValueError(f"Permission denied reading configuration file: {config_path}") from e
        except Exception as e:
            raise ValueError(f"Failed to read configuration file {config_path}: {e}") from e

        # Handle empty or null YAML files
        if data is None:
            raise ValueError(
                f"❌ Configuration file is empty or contains only null values: {config_path.name}\n\n"
                f"💡 Fix: Add proper YAML configuration content to the file.\n"
                f"   Example minimal configuration:\n"
                f"   ```yaml\n"
                f"   version: \"1.0\"\n"
                f"   github:\n"
                f"     token: \"${{GITHUB_TOKEN}}\"\n"
                f"     owner: \"your-username\"\n"
                f"   repositories:\n"
                f"     - name: \"your-repo\"\n"
                f"       path: \"/path/to/repo\"\n"
                f"   ```\n\n"
                f"📁 File: {config_path}"
            )

        # Validate that data is a dictionary
        if not isinstance(data, dict):
            raise ValueError(
                f"❌ Configuration file must contain a YAML object (key-value pairs): {config_path.name}\n\n"
                f"💡 Fix: Ensure the file contains proper YAML structure with keys and values.\n"
                f"   Current content type: {type(data).__name__}\n\n"
                f"📁 File: {config_path}"
            )

        # Validate version
        version = data.get("version", "1.0")
        if version not in ["1.0"]:
            raise ValueError(f"Unsupported config version: {version}")

        # Process GitHub config
        github_data = data.get("github", {})

        # Resolve GitHub token
        github_token = cls._resolve_env_var(github_data.get("token"))
        if github_data.get("token") and not github_token:
            raise ValueError(
                "GitHub is configured but GITHUB_TOKEN environment variable is not set"
            )

        github_config = GitHubConfig(
            token=github_token,
            owner=cls._resolve_env_var(github_data.get("owner")),
            organization=cls._resolve_env_var(github_data.get("organization")),
            base_url=github_data.get("base_url", "https://api.github.com"),
            max_retries=github_data.get("rate_limit", {}).get("max_retries", 3),
            backoff_factor=github_data.get("rate_limit", {}).get("backoff_factor", 2),
        )

        # Process repositories
        repositories = []

        # Handle organization-based repository discovery
        if github_config.organization and not data.get("repositories"):
            # Organization specified but no explicit repositories - will be discovered at runtime
            pass
        else:
            # Process explicitly defined repositories
            for i, repo_data in enumerate(data.get("repositories", [])):
                # Validate required repository fields
                if not isinstance(repo_data, dict):
                    raise ValueError(
                        f"❌ Repository entry {i+1} must be a YAML object with name and path: {config_path.name}\n\n"
                        f"💡 Fix: Ensure each repository entry has proper structure:\n"
                        f"   repositories:\n"
                        f"     - name: \"repo-name\"\n"
                        f"       path: \"/path/to/repo\"\n\n"
                        f"📁 File: {config_path}"
                    )
                
                if "name" not in repo_data or repo_data["name"] is None:
                    raise ValueError(
                        f"❌ Repository entry {i+1} missing required 'name' field: {config_path.name}\n\n"
                        f"💡 Fix: Add a name field to the repository entry:\n"
                        f"   - name: \"your-repo-name\"\n"
                        f"     path: \"/path/to/repo\"\n\n"
                        f"📁 File: {config_path}"
                    )
                
                if "path" not in repo_data or repo_data["path"] is None:
                    raise ValueError(
                        f"❌ Repository entry {i+1} ('{repo_data['name']}') missing required 'path' field: {config_path.name}\n\n"
                        f"💡 Fix: Add a path field to the repository entry:\n"
                        f"   - name: \"{repo_data['name']}\"\n"
                        f"     path: \"/path/to/repo\"\n\n"
                        f"📁 File: {config_path}"
                    )

                # Handle github_repo with owner/organization fallback
                github_repo = repo_data.get("github_repo")
                if github_repo and "/" not in github_repo:
                    if github_config.organization:
                        github_repo = f"{github_config.organization}/{github_repo}"
                    elif github_config.owner:
                        github_repo = f"{github_config.owner}/{github_repo}"

                repo_config = RepositoryConfig(
                    name=repo_data["name"],
                    path=repo_data["path"],
                    github_repo=github_repo,
                    project_key=repo_data.get("project_key"),
                    branch=repo_data.get("branch"),
                )
                repositories.append(repo_config)

        # Allow empty repositories list if organization is specified
        if not repositories and not github_config.organization:
            raise ValueError("No repositories defined and no organization specified for discovery")

        # Process analysis settings
        analysis_data = data.get("analysis", {})

        # Default exclude paths for common boilerplate/generated files
        default_exclude_paths = [
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

        # Merge user-provided paths with defaults (user paths take precedence)
        user_exclude_paths = analysis_data.get("exclude", {}).get("paths", [])
        exclude_paths = user_exclude_paths if user_exclude_paths else default_exclude_paths

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
            categories=classification_data.get("categories", {})
        )
        
        # Process LLM classification configuration with environment variable resolution
        llm_classification_data = analysis_data.get("llm_classification", {})
        llm_classification_config = LLMClassificationConfig(
            enabled=llm_classification_data.get("enabled", False),
            api_key=cls._resolve_env_var(llm_classification_data.get("api_key")),
            api_base_url=llm_classification_data.get("api_base_url", "https://openrouter.ai/api/v1"),
            model=llm_classification_data.get("model", "mistralai/mistral-7b-instruct"),
            confidence_threshold=llm_classification_data.get("confidence_threshold", 0.7),
            max_tokens=llm_classification_data.get("max_tokens", 50),
            temperature=llm_classification_data.get("temperature", 0.1),
            timeout_seconds=llm_classification_data.get("timeout_seconds", 30.0),
            cache_duration_days=llm_classification_data.get("cache_duration_days", 90),
            enable_caching=llm_classification_data.get("enable_caching", True),
            max_daily_requests=llm_classification_data.get("max_daily_requests", 1000),
            domain_terms=llm_classification_data.get("domain_terms", {})
        )

        analysis_config = AnalysisConfig(
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
            similarity_threshold=analysis_data.get("identity", {}).get(
                "similarity_threshold", 0.85
            ),
            manual_identity_mappings=analysis_data.get("identity", {}).get("manual_mappings", []),
            default_ticket_platform=analysis_data.get("default_ticket_platform"),
            branch_mapping_rules=analysis_data.get("branch_mapping_rules", {}),
            ticket_platforms=analysis_data.get("ticket_platforms"),
            auto_identity_analysis=analysis_data.get("identity", {}).get(
                "auto_analysis", True
            ),
            ml_categorization=ml_categorization_config,
            commit_classification=commit_classification_config,
            llm_classification=llm_classification_config,
        )

        # Process output settings
        output_data = data.get("output", {})
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

        output_config = OutputConfig(
            directory=output_dir,
            formats=output_data.get("formats", ["csv", "markdown"]),
            csv_delimiter=output_data.get("csv", {}).get("delimiter", ","),
            csv_encoding=output_data.get("csv", {}).get("encoding", "utf-8"),
            anonymize_enabled=output_data.get("anonymization", {}).get("enabled", False),
            anonymize_fields=output_data.get("anonymization", {}).get("fields", []),
            anonymize_method=output_data.get("anonymization", {}).get("method", "hash"),
        )

        # Process cache settings
        cache_data = data.get("cache", {})
        cache_dir = cache_data.get("directory", ".gitflow-cache")
        cache_path = Path(cache_dir)
        # If relative path, make it relative to config file directory
        if not cache_path.is_absolute():
            cache_path = config_path.parent / cache_path

        cache_config = CacheConfig(
            directory=cache_path.resolve(),
            ttl_hours=cache_data.get("ttl_hours", 168),
            max_size_mb=cache_data.get("max_size_mb", 500),
        )

        # Process JIRA settings
        jira_config = None
        jira_data = data.get("jira", {})
        if jira_data:
            access_user = cls._resolve_env_var(jira_data.get("access_user", ""))
            access_token = cls._resolve_env_var(jira_data.get("access_token", ""))

            # Validate JIRA credentials if JIRA is configured
            if jira_data.get("access_user") and jira_data.get("access_token"):
                if not access_user:
                    raise ValueError(
                        "JIRA is configured but JIRA_ACCESS_USER environment variable is not set"
                    )
                if not access_token:
                    raise ValueError(
                        "JIRA is configured but JIRA_ACCESS_TOKEN environment variable is not set"
                    )

            jira_config = JIRAConfig(
                access_user=access_user,
                access_token=access_token,
                base_url=jira_data.get("base_url"),
            )

        # Process JIRA integration settings
        jira_integration_config = None
        jira_integration_data = data.get("jira_integration", {})
        if jira_integration_data:
            jira_integration_config = JIRAIntegrationConfig(
                enabled=jira_integration_data.get("enabled", True),
                fetch_story_points=jira_integration_data.get("fetch_story_points", True),
                project_keys=jira_integration_data.get("project_keys", []),
                story_point_fields=jira_integration_data.get(
                    "story_point_fields", ["customfield_10016", "customfield_10021", "Story Points"]
                ),
            )

        # Process qualitative analysis settings
        qualitative_config = None
        qualitative_data = data.get("qualitative", {})
        if qualitative_data:
            # Import here to avoid circular imports
            try:
                from .qualitative.models.schemas import (
                    CacheConfig as QualitativeCacheConfig,
                )
                from .qualitative.models.schemas import (
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
                    risk_config=RiskConfig(**nlp_data.get("risk", {}))
                )
                
                # Parse LLM configuration
                llm_data = qualitative_data.get("llm", {})
                cost_tracking_data = qualitative_data.get("cost_tracking", {})
                llm_config = LLMConfig(
                    openrouter_api_key=cls._resolve_env_var(
                        llm_data.get("openrouter_api_key") or llm_data.get("api_key", "${OPENROUTER_API_KEY}")
                    ),
                    base_url=llm_data.get("base_url", "https://openrouter.ai/api/v1"),
                    primary_model=llm_data.get("primary_model") or llm_data.get("model", "anthropic/claude-3-haiku"),
                    fallback_model=llm_data.get("fallback_model", "meta-llama/llama-3.1-8b-instruct:free"),
                    complex_model=llm_data.get("complex_model", "anthropic/claude-3-sonnet"),
                    complexity_threshold=llm_data.get("complexity_threshold", 0.5),
                    cost_threshold_per_1k=llm_data.get("cost_threshold_per_1k", 0.01),
                    max_tokens=llm_data.get("max_tokens", 1000),
                    temperature=llm_data.get("temperature", 0.1),
                    max_group_size=llm_data.get("max_group_size", 10),
                    similarity_threshold=llm_data.get("similarity_threshold", 0.8),
                    requests_per_minute=llm_data.get("requests_per_minute", 200),
                    max_retries=llm_data.get("max_retries", 3),
                    max_daily_cost=cost_tracking_data.get("daily_budget_usd") or llm_data.get("max_daily_cost", 5.0),
                    enable_cost_tracking=cost_tracking_data.get("enabled") if cost_tracking_data.get("enabled") is not None else llm_data.get("enable_cost_tracking", True)
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
                    max_cache_size_mb=cache_data.get("max_cache_size_mb", 100)
                )
                
                # Create main qualitative configuration
                qualitative_config = QualitativeConfig(
                    enabled=qualitative_data.get("enabled", True),
                    batch_size=qualitative_data.get("batch_size", 1000),
                    max_llm_fallback_pct=qualitative_data.get("max_llm_fallback_pct", 0.15),
                    confidence_threshold=qualitative_data.get("confidence_threshold", 0.7),
                    nlp_config=nlp_config,
                    llm_config=llm_config,
                    cache_config=qualitative_cache_config,
                    enable_performance_tracking=qualitative_data.get("enable_performance_tracking", True),
                    target_processing_time_ms=qualitative_data.get("target_processing_time_ms", 2.0),
                    min_overall_confidence=qualitative_data.get("min_overall_confidence", 0.6),
                    enable_quality_feedback=qualitative_data.get("enable_quality_feedback", True)
                )
                
            except ImportError as e:
                print(f"⚠️  Qualitative analysis dependencies missing: {e}")
                print("   Install with: pip install spacy scikit-learn openai tiktoken")
                qualitative_config = None
            except Exception as e:
                print(f"⚠️  Error parsing qualitative configuration: {e}")
                qualitative_config = None

        # Process PM integration settings
        pm_integration_config = None
        pm_integration_data = data.get("pm_integration", {})
        if pm_integration_data:
            # Parse platform configurations
            platforms_config = {}
            platforms_data = pm_integration_data.get("platforms", {})
            
            for platform_name, platform_data in platforms_data.items():
                platforms_config[platform_name] = PMPlatformConfig(
                    enabled=platform_data.get("enabled", True),
                    platform_type=platform_data.get("platform_type", platform_name),
                    config=platform_data.get("config", {})
                )
            
            # Parse correlation settings with defaults
            correlation_defaults = {
                "fuzzy_matching": True,
                "temporal_window_hours": 72,
                "confidence_threshold": 0.8
            }
            correlation_config = {**correlation_defaults, **pm_integration_data.get("correlation", {})}
            
            pm_integration_config = PMIntegrationConfig(
                enabled=pm_integration_data.get("enabled", False),
                primary_platform=pm_integration_data.get("primary_platform"),
                correlation=correlation_config,
                platforms=platforms_config
            )

        return Config(
            repositories=repositories,
            github=github_config,
            analysis=analysis_config,
            output=output_config,
            cache=cache_config,
            jira=jira_config,
            jira_integration=jira_integration_config,
            pm_integration=pm_integration_config,
            qualitative=qualitative_config,
        )

    @staticmethod
    def _resolve_env_var(value: Optional[str]) -> Optional[str]:
        """Resolve environment variable references."""
        if not value:
            return None

        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            resolved = os.environ.get(env_var)
            if not resolved:
                raise ValueError(f"Environment variable {env_var} not set")
            return resolved

        return value

    @staticmethod
    def _handle_yaml_error(error: yaml.YAMLError, config_path: Path) -> None:
        """Handle YAML parsing errors with user-friendly messages."""
        file_name = config_path.name
        
        # Extract error details if available
        line_number = getattr(error, 'problem_mark', None)
        context_mark = getattr(error, 'context_mark', None)
        problem = getattr(error, 'problem', str(error))
        context = getattr(error, 'context', None)
        
        # Build error message parts
        location_info = ""
        if line_number:
            location_info = f" at line {line_number.line + 1}, column {line_number.column + 1}"
        elif context_mark:
            location_info = f" at line {context_mark.line + 1}, column {context_mark.column + 1}"
        
        # Create user-friendly error message
        error_msg = f"❌ YAML configuration error in {file_name}{location_info}:\n\n"
        
        # Detect common YAML issues and provide specific guidance
        problem_lower = problem.lower()
        
        if "found character '\\t'" in problem_lower.replace("'", "'"):
            error_msg += "🚫 Tab characters are not allowed in YAML files!\n\n"
            error_msg += "💡 Fix: Replace all tab characters with spaces (usually 2 or 4 spaces).\n"
            error_msg += "   Most editors can show whitespace characters and convert tabs to spaces.\n"
            error_msg += "   In VS Code: View → Render Whitespace, then Edit → Convert Indentation to Spaces"
            
        elif "mapping values are not allowed here" in problem_lower:
            error_msg += "🚫 Invalid YAML syntax - missing colon or incorrect indentation!\n\n"
            error_msg += "💡 Common fixes:\n"
            error_msg += "   • Add a colon (:) after the key name\n"
            error_msg += "   • Check that all lines are properly indented with spaces\n"
            error_msg += "   • Ensure nested items are indented consistently"
            
        elif "could not find expected" in problem_lower and ":" in problem_lower:
            error_msg += "🚫 Missing colon (:) after a key name!\n\n"
            error_msg += "💡 Fix: Add a colon and space after the key name.\n"
            error_msg += "   Example: 'key_name: value' not 'key_name value'"
            
        elif "found undefined alias" in problem_lower:
            error_msg += "🚫 YAML alias reference not found!\n\n"
            error_msg += "💡 Fix: Check that the referenced alias (&name) is defined before using it (*name)"
            
        elif "expected <block end>" in problem_lower:
            error_msg += "🚫 Incorrect indentation or missing content!\n\n"
            error_msg += "💡 Common fixes:\n"
            error_msg += "   • Check that all nested items are properly indented\n"
            error_msg += "   • Ensure list items start with '- ' (dash and space)\n"
            error_msg += "   • Make sure there's content after colons"
            
        elif "while scanning a quoted scalar" in problem_lower:
            error_msg += "🚫 Unclosed or incorrectly quoted string!\n\n"
            error_msg += "💡 Fix: Check that all quotes are properly closed.\n"
            error_msg += "   • Use matching quotes: 'text' or \"text\"\n"
            error_msg += "   • Escape quotes inside strings: 'don\\'t' or \"say \\\"hello\\\"\""
            
        elif "found unexpected end of stream" in problem_lower:
            error_msg += "🚫 Incomplete YAML structure!\n\n"
            error_msg += "💡 Fix: The file appears to end unexpectedly.\n"
            error_msg += "   • Check that all sections are complete\n"
            error_msg += "   • Ensure there are no missing closing brackets or braces"
            
        elif "found unknown escape character" in problem_lower:
            error_msg += "🚫 Invalid escape sequence in quoted string!\n\n"
            error_msg += "💡 Fix: Use proper YAML escape sequences or raw strings.\n"
            error_msg += "   • For regex patterns: Use double quotes and double backslashes (\"\\\\d+\")\n"
            error_msg += "   • For file paths: Use forward slashes or double backslashes\n"
            error_msg += "   • Or use single quotes for literal strings: 'C:\\path\\to\\file'"
            
        elif "scanner" in problem_lower and "character" in problem_lower:
            error_msg += "🚫 Invalid character in YAML file!\n\n"
            error_msg += "💡 Fix: Check for special characters that need to be quoted.\n"
            error_msg += "   • Wrap values containing special characters in quotes\n"
            error_msg += "   • Common problematic characters: @, `, |, >, [, ], {, }"
            
        else:
            error_msg += f"🚫 YAML parsing error: {problem}\n\n"
            error_msg += "💡 Common YAML issues to check:\n"
            error_msg += "   • Use spaces for indentation, not tabs\n"
            error_msg += "   • Add colons (:) after key names\n"
            error_msg += "   • Ensure consistent indentation (usually 2 or 4 spaces)\n"
            error_msg += "   • Quote strings containing special characters\n"
            error_msg += "   • Use '- ' (dash and space) for list items"
        
        # Add context information if available
        if context and context != problem:
            error_msg += f"\n\n📍 Context: {context}"
        
        # Add file location reminder
        error_msg += f"\n\n📁 File: {config_path}"
        
        # Add helpful resources
        error_msg += "\n\n🔗 For YAML syntax help, visit: https://yaml.org/spec/1.2/spec.html"
        error_msg += "\n   Or use an online YAML validator to check your syntax."
        
        raise ValueError(error_msg)

    @staticmethod
    def validate_config(config: Config) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        # Check repository paths exist
        for repo in config.repositories:
            if not repo.path.exists():
                warnings.append(f"Repository path does not exist: {repo.path}")
            elif not (repo.path / ".git").exists():
                warnings.append(f"Path is not a git repository: {repo.path}")

        # Check GitHub token if GitHub repos are specified
        has_github_repos = any(r.github_repo for r in config.repositories)
        if has_github_repos and not config.github.token:
            warnings.append("GitHub repositories specified but no GitHub token provided")

        # Check if owner is needed
        for repo in config.repositories:
            if repo.github_repo and "/" not in repo.github_repo and not config.github.owner:
                warnings.append(f"Repository {repo.github_repo} needs owner specified")

        # Check cache directory permissions
        try:
            config.cache.directory.mkdir(exist_ok=True, parents=True)
        except PermissionError:
            warnings.append(f"Cannot create cache directory: {config.cache.directory}")

        return warnings
