"""Configuration profiles and presets for GitFlow Analytics."""

from typing import Any, Optional


class ConfigurationProfile:
    """Base class for configuration profiles."""

    name: str = ""
    description: str = ""

    @classmethod
    def get_settings(cls) -> dict[str, Any]:
        """Get the profile settings.

        Returns:
            Dictionary of configuration settings for this profile
        """
        raise NotImplementedError


class PerformanceProfile(ConfigurationProfile):
    """Performance-optimized configuration profile.

    This profile prioritizes speed over accuracy and completeness.
    Suitable for large codebases or quick analysis runs.
    """

    name = "performance"
    description = "Optimized for speed with large repositories"

    @classmethod
    def get_settings(cls) -> dict[str, Any]:
        """Get performance-optimized settings."""
        return {
            "analysis": {
                "branch_analysis": {
                    "strategy": "main_only",
                    "max_branches_per_repo": 10,
                    "branch_commit_limit": 500,
                },
                "ml_categorization": {
                    "enabled": False,  # Disable ML for speed
                    "batch_size": 500,
                },
                "commit_classification": {
                    "enabled": False,  # Disable classification for speed
                },
                "llm_classification": {
                    "enabled": False,  # Disable LLM for speed
                },
            },
            "cache": {
                "ttl_hours": 336,  # 2 weeks cache
                "max_size_mb": 1000,
            },
            "output": {
                "formats": ["csv"],  # CSV only for speed
            },
        }


class QualityProfile(ConfigurationProfile):
    """Quality-focused configuration profile.

    This profile enables all analysis features for maximum insight.
    Suitable for detailed analysis and reporting.
    """

    name = "quality"
    description = "Maximum analysis depth and accuracy"

    @classmethod
    def get_settings(cls) -> dict[str, Any]:
        """Get quality-focused settings."""
        return {
            "analysis": {
                "branch_analysis": {
                    "strategy": "smart",
                    "max_branches_per_repo": 100,
                    "active_days_threshold": 180,
                    "branch_commit_limit": 2000,
                },
                "ml_categorization": {
                    "enabled": True,
                    "min_confidence": 0.7,
                    "semantic_weight": 0.8,
                    "batch_size": 50,
                },
                "commit_classification": {
                    "enabled": True,
                    "confidence_threshold": 0.6,
                    "auto_retrain": True,
                },
                "auto_identity_analysis": True,
            },
            "cache": {
                "ttl_hours": 72,  # Shorter cache for freshness
                "max_size_mb": 500,
            },
            "output": {
                "formats": ["csv", "markdown", "json"],
            },
        }


class BalancedProfile(ConfigurationProfile):
    """Balanced configuration profile.

    This profile provides a good balance between performance and quality.
    Suitable for most use cases.
    """

    name = "balanced"
    description = "Balanced performance and quality (default)"

    @classmethod
    def get_settings(cls) -> dict[str, Any]:
        """Get balanced settings."""
        return {
            "analysis": {
                "branch_analysis": {
                    "strategy": "smart",
                    "max_branches_per_repo": 50,
                    "active_days_threshold": 90,
                    "branch_commit_limit": 1000,
                },
                "ml_categorization": {
                    "enabled": True,
                    "min_confidence": 0.6,
                    "semantic_weight": 0.7,
                    "batch_size": 100,
                },
                "commit_classification": {
                    "enabled": True,
                    "confidence_threshold": 0.5,
                },
            },
            "cache": {
                "ttl_hours": 168,  # 1 week
                "max_size_mb": 500,
            },
            "output": {
                "formats": ["csv", "markdown"],
            },
        }


class MinimalProfile(ConfigurationProfile):
    """Minimal configuration profile.

    This profile runs only essential analysis features.
    Suitable for basic metrics and quick overview.
    """

    name = "minimal"
    description = "Essential features only"

    @classmethod
    def get_settings(cls) -> dict[str, Any]:
        """Get minimal settings."""
        return {
            "analysis": {
                "branch_analysis": {
                    "strategy": "main_only",
                },
                "ml_categorization": {
                    "enabled": False,
                },
                "commit_classification": {
                    "enabled": False,
                },
                "llm_classification": {
                    "enabled": False,
                },
                "auto_identity_analysis": False,
            },
            "output": {
                "formats": ["csv"],
            },
            "cache": {
                "ttl_hours": 720,  # 30 days
            },
        }


class ProfileManager:
    """Manages configuration profiles."""

    # Registry of available profiles
    _profiles: dict[str, type[ConfigurationProfile]] = {
        "performance": PerformanceProfile,
        "quality": QualityProfile,
        "balanced": BalancedProfile,
        "minimal": MinimalProfile,
    }

    @classmethod
    def get_profile(cls, name: str) -> Optional[type[ConfigurationProfile]]:
        """Get a configuration profile by name.

        Args:
            name: Profile name

        Returns:
            Profile class or None if not found
        """
        return cls._profiles.get(name.lower())

    @classmethod
    def list_profiles(cls) -> dict[str, str]:
        """List available profiles.

        Returns:
            Dictionary of profile names to descriptions
        """
        return {name: profile.description for name, profile in cls._profiles.items()}

    @classmethod
    def apply_profile(cls, config_data: dict[str, Any], profile_name: str) -> dict[str, Any]:
        """Apply a profile to configuration data.

        Args:
            config_data: Base configuration data
            profile_name: Name of profile to apply

        Returns:
            Updated configuration data with profile settings

        Raises:
            ValueError: If profile not found
        """
        profile_class = cls.get_profile(profile_name)
        if not profile_class:
            available = ", ".join(cls._profiles.keys())
            raise ValueError(
                f"Unknown configuration profile: {profile_name}. Available profiles: {available}"
            )

        profile_settings = profile_class.get_settings()
        # Profile settings are the base, config_data overrides them
        return cls._deep_merge(profile_settings, config_data)

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Dictionary with values to override

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ProfileManager._deep_merge(result[key], value)
            else:
                result[key] = value

        return result
