"""Configuration management for GitFlow Analytics."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

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
            self.project_key = self.name.upper().replace('-', '_')

@dataclass
class GitHubConfig:
    """GitHub API configuration."""
    token: Optional[str] = None
    owner: Optional[str] = None
    base_url: str = "https://api.github.com"
    max_retries: int = 3
    backoff_factor: int = 2
    
    def get_repo_full_name(self, repo_name: str) -> str:
        """Get full repository name including owner."""
        if '/' in repo_name:
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
    similarity_threshold: float = 0.85
    manual_identity_mappings: List[Dict[str, Any]] = field(default_factory=list)
    default_ticket_platform: Optional[str] = None
    branch_mapping_rules: Dict[str, List[str]] = field(default_factory=dict)

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
class Config:
    """Main configuration container."""
    repositories: List[RepositoryConfig]
    github: GitHubConfig
    analysis: AnalysisConfig
    output: OutputConfig
    cache: CacheConfig

class ConfigLoader:
    """Load and validate configuration from YAML files."""
    
    @staticmethod
    def load(config_path: Path) -> Config:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Validate version
        version = data.get('version', '1.0')
        if version not in ['1.0']:
            raise ValueError(f"Unsupported config version: {version}")
        
        # Process GitHub config
        github_data = data.get('github', {})
        github_config = GitHubConfig(
            token=ConfigLoader._resolve_env_var(github_data.get('token')),
            owner=ConfigLoader._resolve_env_var(github_data.get('owner')),
            base_url=github_data.get('base_url', 'https://api.github.com'),
            max_retries=github_data.get('rate_limit', {}).get('max_retries', 3),
            backoff_factor=github_data.get('rate_limit', {}).get('backoff_factor', 2)
        )
        
        # Process repositories
        repositories = []
        for repo_data in data.get('repositories', []):
            # Handle github_repo with owner fallback
            github_repo = repo_data.get('github_repo')
            if github_repo and github_config.owner and '/' not in github_repo:
                github_repo = f"{github_config.owner}/{github_repo}"
            
            repo_config = RepositoryConfig(
                name=repo_data['name'],
                path=repo_data['path'],
                github_repo=github_repo,
                project_key=repo_data.get('project_key'),
                branch=repo_data.get('branch')
            )
            repositories.append(repo_config)
        
        if not repositories:
            raise ValueError("No repositories defined in configuration")
        
        # Process analysis settings
        analysis_data = data.get('analysis', {})
        analysis_config = AnalysisConfig(
            story_point_patterns=analysis_data.get('story_point_patterns', [
                r"(?:story\s*points?|sp|pts?)\s*[:=]\s*(\d+)",
                r"\[(\d+)\s*(?:sp|pts?)\]",
                r"#(\d+)sp"
            ]),
            exclude_authors=analysis_data.get('exclude', {}).get('authors', [
                "dependabot[bot]",
                "renovate[bot]"
            ]),
            exclude_message_patterns=analysis_data.get('exclude', {}).get('message_patterns', []),
            similarity_threshold=analysis_data.get('identity', {}).get('similarity_threshold', 0.85),
            manual_identity_mappings=analysis_data.get('identity', {}).get('manual_mappings', []),
            default_ticket_platform=analysis_data.get('default_ticket_platform'),
            branch_mapping_rules=analysis_data.get('branch_mapping_rules', {})
        )
        
        # Process output settings
        output_data = data.get('output', {})
        output_dir = output_data.get('directory')
        if output_dir:
            output_dir = Path(output_dir).expanduser().resolve()
        
        output_config = OutputConfig(
            directory=output_dir,
            formats=output_data.get('formats', ['csv', 'markdown']),
            csv_delimiter=output_data.get('csv', {}).get('delimiter', ','),
            csv_encoding=output_data.get('csv', {}).get('encoding', 'utf-8'),
            anonymize_enabled=output_data.get('anonymization', {}).get('enabled', False),
            anonymize_fields=output_data.get('anonymization', {}).get('fields', []),
            anonymize_method=output_data.get('anonymization', {}).get('method', 'hash')
        )
        
        # Process cache settings
        cache_data = data.get('cache', {})
        cache_config = CacheConfig(
            directory=Path(cache_data.get('directory', '.gitflow-cache')),
            ttl_hours=cache_data.get('ttl_hours', 168),
            max_size_mb=cache_data.get('max_size_mb', 500)
        )
        
        return Config(
            repositories=repositories,
            github=github_config,
            analysis=analysis_config,
            output=output_config,
            cache=cache_config
        )
    
    @staticmethod
    def _resolve_env_var(value: Optional[str]) -> Optional[str]:
        """Resolve environment variable references."""
        if not value:
            return None
            
        if value.startswith('${') and value.endswith('}'):
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
            elif not (repo.path / '.git').exists():
                warnings.append(f"Path is not a git repository: {repo.path}")
        
        # Check GitHub token if GitHub repos are specified
        has_github_repos = any(r.github_repo for r in config.repositories)
        if has_github_repos and not config.github.token:
            warnings.append("GitHub repositories specified but no GitHub token provided")
        
        # Check if owner is needed
        for repo in config.repositories:
            if repo.github_repo and '/' not in repo.github_repo and not config.github.owner:
                warnings.append(f"Repository {repo.github_repo} needs owner specified")
        
        # Check cache directory permissions
        try:
            config.cache.directory.mkdir(exist_ok=True, parents=True)
        except PermissionError:
            warnings.append(f"Cannot create cache directory: {config.cache.directory}")
        
        return warnings