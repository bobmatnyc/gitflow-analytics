"""Configuration schema definitions and defaults for GitFlow Analytics."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..qualitative.models.schemas import QualitativeConfig


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
    domain_terms: dict[str, list[str]] = field(
        default_factory=lambda: {
            "media": [
                "video",
                "audio",
                "streaming",
                "player",
                "media",
                "content",
                "broadcast",
                "live",
                "recording",
                "episode",
                "program",
                "tv",
                "radio",
                "podcast",
                "channel",
                "playlist",
            ],
            "localization": [
                "translation",
                "i18n",
                "l10n",
                "locale",
                "language",
                "spanish",
                "french",
                "german",
                "italian",
                "portuguese",
                "multilingual",
                "translate",
                "localize",
                "regional",
            ],
            "integration": [
                "api",
                "webhook",
                "third-party",
                "external",
                "service",
                "integration",
                "sync",
                "import",
                "export",
                "connector",
                "oauth",
                "auth",
                "authentication",
                "sso",
            ],
            "content": [
                "copy",
                "text",
                "wording",
                "messaging",
                "editorial",
                "article",
                "blog",
                "news",
                "story",
                "caption",
                "title",
                "headline",
                "description",
                "summary",
                "metadata",
            ],
        }
    )

    # Fallback behavior when LLM is unavailable
    fallback_to_rules: bool = True  # Fall back to rule-based classification
    fallback_to_ml: bool = True  # Fall back to existing ML classification


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
    model: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,  # Number of trees in random forest
            "max_depth": 20,  # Maximum depth of trees
            "min_samples_split": 5,  # Minimum samples to split a node
            "min_samples_leaf": 2,  # Minimum samples at leaf node
            "random_state": 42,  # For reproducible results
            "n_jobs": -1,  # Use all available CPU cores
        }
    )

    # Feature extraction settings
    feature_extraction: dict[str, Any] = field(
        default_factory=lambda: {
            "enable_temporal_features": True,
            "enable_author_features": True,
            "enable_file_analysis": True,
            "keyword_categories": [
                "feature",
                "bugfix",
                "refactor",
                "docs",
                "test",
                "config",
                "security",
                "performance",
                "ui",
                "api",
                "database",
                "deployment",
            ],
        }
    )

    # Training settings
    training: dict[str, Any] = field(
        default_factory=lambda: {
            "validation_split": 0.2,  # Fraction for validation
            "min_training_samples": 20,  # Minimum samples needed for training
            "cross_validation_folds": 5,  # K-fold cross validation
            "class_weight": "balanced",  # Handle class imbalance
        }
    )

    # Supported classification categories
    categories: dict[str, str] = field(
        default_factory=lambda: {
            "feature": "New functionality or capabilities",
            "bugfix": "Bug fixes and error corrections",
            "refactor": "Code restructuring and optimization",
            "docs": "Documentation changes and updates",
            "test": "Testing-related changes",
            "config": "Configuration and settings changes",
            "chore": "Maintenance and housekeeping tasks",
            "security": "Security-related changes",
            "hotfix": "Emergency production fixes",
            "style": "Code style and formatting changes",
            "build": "Build system and dependency changes",
            "ci": "Continuous integration changes",
            "revert": "Reverts of previous changes",
            "merge": "Merge commits and integration",
            "wip": "Work in progress commits",
        }
    )


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
    always_include_patterns: list[str] = field(
        default_factory=lambda: [
            r"^(main|master|develop|dev)$",  # Main development branches
            r"^release/.*",  # Release branches
            r"^hotfix/.*",  # Hotfix branches
        ]
    )

    always_exclude_patterns: list[str] = field(
        default_factory=lambda: [
            r"^dependabot/.*",  # Dependabot branches
            r"^renovate/.*",  # Renovate branches
            r".*-backup$",  # Backup branches
            r".*-temp$",  # Temporary branches
        ]
    )

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
    exclude_merge_commits: bool = False  # Exclude merge commits from filtered line counts
    similarity_threshold: float = 0.85
    manual_identity_mappings: list[dict[str, Any]] = field(default_factory=list)
    aliases_file: Optional[Path] = None  # Path to shared aliases.yaml file
    default_ticket_platform: Optional[str] = None
    branch_mapping_rules: dict[str, list[str]] = field(default_factory=dict)
    ticket_platforms: Optional[list[str]] = None
    auto_identity_analysis: bool = True  # Enable automatic identity analysis by default
    branch_patterns: Optional[list[str]] = (
        None  # Branch patterns to analyze (e.g., ["*"] for all branches)
    )
    branch_analysis: BranchAnalysisConfig = field(default_factory=BranchAnalysisConfig)
    ml_categorization: MLCategorization = field(default_factory=MLCategorization)
    commit_classification: CommitClassificationConfig = field(
        default_factory=CommitClassificationConfig
    )
    llm_classification: LLMClassificationConfig = field(default_factory=LLMClassificationConfig)
    security: Optional[dict[str, Any]] = field(default_factory=dict)  # Security configuration
    qualitative: Optional["QualitativeConfig"] = None  # Nested qualitative config support


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
class LauncherPreferences:
    """Interactive launcher preferences."""

    last_selected_repos: list[str] = field(default_factory=list)
    default_weeks: int = 4
    auto_clear_cache: bool = False
    skip_identity_analysis: bool = False
    last_run: Optional[str] = None


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
    pm: Optional[Any] = None  # Modern PM framework config
    pm_integration: Optional[PMIntegrationConfig] = None
    qualitative: Optional["QualitativeConfig"] = None
    launcher: Optional[LauncherPreferences] = None

    def discover_organization_repositories(
        self, clone_base_path: Optional[Path] = None, progress_callback=None
    ) -> list[RepositoryConfig]:
        """Discover repositories from GitHub organization.

        Args:
            clone_base_path: Base directory where repos should be cloned/found.
                           If None, uses output directory.
            progress_callback: Optional callback function(repo_name, count) for progress updates.

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

            repo_count = 0
            for repo in org.get_repos():
                repo_count += 1

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(repo.name, repo_count)

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

    def get_effective_ticket_platforms(self) -> list[str]:
        """Get the effective list of ticket platforms to extract.

        If ticket_platforms is explicitly configured in analysis config, use that.
        Otherwise, infer from which PM platforms are actually configured.

        Returns:
            List of ticket platform names to extract (e.g., ['jira', 'github'])
        """
        # If explicitly configured, use that
        if self.analysis.ticket_platforms is not None:
            return self.analysis.ticket_platforms

        # Otherwise, infer from configured PM platforms
        platforms = []

        # Check modern PM framework config
        if self.pm:
            if hasattr(self.pm, "jira") and self.pm.jira:
                platforms.append("jira")
            if hasattr(self.pm, "linear") and self.pm.linear:
                platforms.append("linear")
            if hasattr(self.pm, "clickup") and self.pm.clickup:
                platforms.append("clickup")

        # Check legacy JIRA config
        if (self.jira or self.jira_integration) and "jira" not in platforms:
            platforms.append("jira")

        # Always include GitHub if we have GitHub configured (for issue tracking)
        if self.github.token:
            platforms.append("github")

        # If nothing configured, fall back to common platforms
        if not platforms:
            platforms = ["jira", "github", "clickup", "linear"]

        return platforms
