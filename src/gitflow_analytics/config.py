"""Configuration management for GitFlow Analytics."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv


@dataclass
class RepositoryConfig:
    """Configuration for a single repository."""

    name: str
    path: Path
    github_repo: Optional[str] = None
    project_key: Optional[str] = None
    branch: Optional[str] = None

    def __post_init__(self):
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
class AnalysisConfig:
    """Analysis-specific configuration."""

    story_point_patterns: List[str] = field(default_factory=list)
    exclude_authors: List[str] = field(default_factory=list)
    exclude_message_patterns: List[str] = field(default_factory=list)
    exclude_paths: List[str] = field(default_factory=list)
    similarity_threshold: float = 0.85
    manual_identity_mappings: List[Dict[str, Any]] = field(default_factory=list)
    default_ticket_platform: Optional[str] = None
    branch_mapping_rules: Dict[str, List[str]] = field(default_factory=dict)
    ticket_platforms: Optional[List[str]] = None


@dataclass
class OutputConfig:
    """Output configuration."""

    directory: Optional[Path] = None
    formats: List[str] = field(default_factory=lambda: ["csv", "markdown"])
    csv_delimiter: str = ","
    csv_encoding: str = "utf-8"
    anonymize_enabled: bool = False
    anonymize_fields: List[str] = field(default_factory=list)
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
    project_keys: List[str] = field(default_factory=list)
    story_point_fields: List[str] = field(
        default_factory=lambda: ["customfield_10016", "customfield_10021", "Story Points"]
    )


@dataclass
class Config:
    """Main configuration container."""

    repositories: List[RepositoryConfig]
    github: GitHubConfig
    analysis: AnalysisConfig
    output: OutputConfig
    cache: CacheConfig
    jira: Optional[JIRAConfig] = None
    jira_integration: Optional[JIRAIntegrationConfig] = None

    def discover_organization_repositories(
        self, clone_base_path: Optional[Path] = None
    ) -> List[RepositoryConfig]:
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

        with open(config_path) as f:
            data = yaml.safe_load(f)

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
            for repo_data in data.get("repositories", []):
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
        ]

        # Merge user-provided paths with defaults (user paths take precedence)
        user_exclude_paths = analysis_data.get("exclude", {}).get("paths", [])
        exclude_paths = user_exclude_paths if user_exclude_paths else default_exclude_paths

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

        return Config(
            repositories=repositories,
            github=github_config,
            analysis=analysis_config,
            output=output_config,
            cache=cache_config,
            jira=jira_config,
            jira_integration=jira_integration_config,
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
    def validate_config(config: Config) -> List[str]:
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
