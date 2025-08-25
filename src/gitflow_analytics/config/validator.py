"""Configuration validation logic for GitFlow Analytics."""

from pathlib import Path

from .schema import Config


class ConfigValidator:
    """Validates configuration settings."""

    @staticmethod
    def validate_config(config: Config) -> list[str]:
        """Validate configuration and return list of warnings.

        Args:
            config: Configuration to validate

        Returns:
            List of warning messages
        """
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

    @staticmethod
    def validate_analysis_config(analysis_config: dict, config_path: Path) -> None:
        """Validate analysis configuration section.

        Args:
            analysis_config: Analysis configuration dictionary
            config_path: Path to configuration file (for error messages)

        Raises:
            InvalidValueError: If configuration values are invalid
        """
        from .errors import InvalidValueError

        # Validate similarity threshold
        if "identity" in analysis_config:
            threshold = analysis_config["identity"].get("similarity_threshold")
            if threshold is not None and not (0.0 <= threshold <= 1.0):
                raise InvalidValueError(
                    "similarity_threshold",
                    threshold,
                    "must be between 0.0 and 1.0",
                    config_path,
                    valid_values=["0.0 to 1.0"],
                )

        # Validate ML categorization settings
        if "ml_categorization" in analysis_config:
            ml_config = analysis_config["ml_categorization"]

            if "min_confidence" in ml_config:
                conf = ml_config["min_confidence"]
                if not (0.0 <= conf <= 1.0):
                    raise InvalidValueError(
                        "ml_categorization.min_confidence",
                        conf,
                        "must be between 0.0 and 1.0",
                        config_path,
                        valid_values=["0.0 to 1.0"],
                    )

            if "semantic_weight" in ml_config:
                weight = ml_config["semantic_weight"]
                if not (0.0 <= weight <= 1.0):
                    raise InvalidValueError(
                        "ml_categorization.semantic_weight",
                        weight,
                        "must be between 0.0 and 1.0",
                        config_path,
                        valid_values=["0.0 to 1.0"],
                    )

        # Validate branch analysis strategy
        if "branch_analysis" in analysis_config:
            branch_config = analysis_config["branch_analysis"]
            if "strategy" in branch_config:
                strategy = branch_config["strategy"]
                valid_strategies = ["all", "smart", "main_only"]
                if strategy not in valid_strategies:
                    raise InvalidValueError(
                        "branch_analysis.strategy",
                        strategy,
                        "invalid branch analysis strategy",
                        config_path,
                        valid_values=valid_strategies,
                    )

    @staticmethod
    def validate_output_config(output_config: dict, config_path: Path) -> None:
        """Validate output configuration section.

        Args:
            output_config: Output configuration dictionary
            config_path: Path to configuration file (for error messages)

        Raises:
            InvalidValueError: If configuration values are invalid
        """
        from .errors import InvalidValueError

        # Validate output formats
        if "formats" in output_config:
            formats = output_config["formats"]
            valid_formats = ["csv", "markdown", "json"]
            for fmt in formats:
                if fmt not in valid_formats:
                    raise InvalidValueError(
                        "output.formats",
                        fmt,
                        "invalid output format",
                        config_path,
                        valid_values=valid_formats,
                    )

        # Validate anonymization method
        if "anonymization" in output_config:
            anon_config = output_config["anonymization"]
            if "method" in anon_config:
                method = anon_config["method"]
                valid_methods = ["hash", "random", "sequential"]
                if method not in valid_methods:
                    raise InvalidValueError(
                        "output.anonymization.method",
                        method,
                        "invalid anonymization method",
                        config_path,
                        valid_values=valid_methods,
                    )
