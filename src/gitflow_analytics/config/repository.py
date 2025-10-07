"""Repository discovery and management for GitFlow Analytics."""

from pathlib import Path
from typing import Optional

from github import Github

from .schema import GitHubConfig, RepositoryConfig


class RepositoryManager:
    """Manages repository discovery and configuration."""

    def __init__(self, github_config: GitHubConfig):
        """Initialize repository manager.

        Args:
            github_config: GitHub configuration
        """
        self.github_config = github_config

    def discover_organization_repositories(
        self, clone_base_path: Optional[Path] = None, progress_callback=None
    ) -> list[RepositoryConfig]:
        """Discover repositories from GitHub organization.

        Args:
            clone_base_path: Base directory where repos should be cloned/found.
            progress_callback: Optional callback function(repo_name, count) for progress updates.

        Returns:
            List of discovered repository configurations.
        """
        if not self.github_config.organization or not self.github_config.token:
            return []

        github_client = Github(self.github_config.token, base_url=self.github_config.base_url)

        try:
            org = github_client.get_organization(self.github_config.organization)
            discovered_repos = []

            if clone_base_path is None:
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
                repo_path = clone_base_path / repo.name
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
                f"Failed to discover repositories from organization "
                f"{self.github_config.organization}: {e}"
            ) from e

    def process_repository_config(
        self, repo_data: dict, index: int, config_path: Path
    ) -> RepositoryConfig:
        """Process a single repository configuration entry.

        Args:
            repo_data: Repository configuration data
            index: Repository index in the list (for error messages)
            config_path: Path to configuration file (for error messages)

        Returns:
            RepositoryConfig instance

        Raises:
            ValueError: If repository configuration is invalid
        """
        from .errors import InvalidValueError, MissingFieldError

        # Validate required repository fields
        if not isinstance(repo_data, dict):
            raise InvalidValueError(
                f"repositories[{index}]",
                type(repo_data).__name__,
                "must be a YAML object with name and path",
                config_path,
                valid_values=["object with 'name' and 'path' fields"],
            )

        if "name" not in repo_data or repo_data["name"] is None:
            raise MissingFieldError(
                "name", f"repositories[{index}]", config_path, example='name: "your-repo-name"'
            )

        if "path" not in repo_data or repo_data["path"] is None:
            raise MissingFieldError(
                "path",
                f"repositories[{index}] ('{repo_data.get('name', 'unknown')}')",
                config_path,
                example='path: "/path/to/repo"',
            )

        # Handle github_repo with owner/organization fallback
        github_repo = repo_data.get("github_repo")
        if github_repo and "/" not in github_repo:
            if self.github_config.organization:
                github_repo = f"{self.github_config.organization}/{github_repo}"
            elif self.github_config.owner:
                github_repo = f"{self.github_config.owner}/{github_repo}"

        return RepositoryConfig(
            name=repo_data["name"],
            path=repo_data["path"],
            github_repo=github_repo,
            project_key=repo_data.get("project_key"),
            branch=repo_data.get("branch"),
        )
