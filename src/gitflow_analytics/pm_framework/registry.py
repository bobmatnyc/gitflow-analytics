"""Platform registry for managing and instantiating PM platform adapters.

This module provides centralized registration and management of platform adapters,
including instance creation, caching, and capability discovery.
"""

import logging
from typing import Any, Optional

from .base import BasePlatformAdapter

# Configure logger for registry
logger = logging.getLogger(__name__)


class PlatformRegistry:
    """Registry for managing platform adapters.

    WHY: Centralized registry provides a clean way to discover available
    platforms, manage adapter instances, and ensure consistent initialization
    patterns across all adapters.

    DESIGN DECISION: Use registry pattern rather than direct imports to enable
    dynamic adapter discovery and plugin-style architecture. This allows
    future extensions without modifying core framework code.
    """

    def __init__(self) -> None:
        """Initialize the platform registry.

        Creates empty registry for adapter classes and instances.
        Built-in adapters should be registered during module initialization.
        """
        self._adapters: dict[str, type[BasePlatformAdapter]] = {}
        self._instances: dict[str, BasePlatformAdapter] = {}

        logger.info("Platform registry initialized")

    def register_adapter(
        self, platform_name: str, adapter_class: type[BasePlatformAdapter]
    ) -> None:
        """Register a platform adapter class.

        WHY: Registration allows the framework to discover and use platform
        adapters without hardcoding imports. This enables a plugin architecture
        where new adapters can be added by simply registering them.

        Args:
            platform_name: Unique identifier for the platform (e.g., 'jira').
            adapter_class: Class that implements BasePlatformAdapter interface.

        Raises:
            ValueError: If platform_name is already registered or invalid.
            TypeError: If adapter_class doesn't inherit from BasePlatformAdapter.
        """
        if not platform_name or not isinstance(platform_name, str):
            raise ValueError("Platform name must be a non-empty string")

        if not issubclass(adapter_class, BasePlatformAdapter):
            raise TypeError(f"Adapter class {adapter_class} must inherit from BasePlatformAdapter")

        if platform_name in self._adapters:
            logger.warning(f"Overriding existing adapter for platform: {platform_name}")

        self._adapters[platform_name] = adapter_class
        logger.info(f"Registered adapter for platform: {platform_name}")

    def unregister_adapter(self, platform_name: str) -> None:
        """Unregister a platform adapter.

        WHY: Allows dynamic removal of adapters, useful for testing or
        when adapters need to be replaced at runtime.

        Args:
            platform_name: Platform identifier to unregister.
        """
        if platform_name in self._adapters:
            del self._adapters[platform_name]
            logger.info(f"Unregistered adapter for platform: {platform_name}")

        # Also remove any existing instance
        if platform_name in self._instances:
            del self._instances[platform_name]
            logger.info(f"Removed cached instance for platform: {platform_name}")

    def get_available_platforms(self) -> list[str]:
        """Get list of available platform names.

        WHY: Allows discovery of available platforms for configuration
        validation and user interface generation.

        Returns:
            List of registered platform identifiers.
        """
        return list(self._adapters.keys())

    def is_platform_available(self, platform_name: str) -> bool:
        """Check if a platform adapter is available.

        Args:
            platform_name: Platform identifier to check.

        Returns:
            True if platform adapter is registered, False otherwise.
        """
        return platform_name in self._adapters

    def get_platform_capabilities(self, platform_name: str) -> Optional[dict[str, Any]]:
        """Get capabilities for a platform without instantiating it.

        WHY: Allows capability discovery for configuration validation and
        feature planning without the overhead of authentication and connection.

        Args:
            platform_name: Platform identifier to check capabilities for.

        Returns:
            Dictionary of capabilities, or None if platform not found.
        """
        if platform_name not in self._adapters:
            logger.warning(f"Platform not found: {platform_name}")
            return None

        try:
            # Create temporary instance to get capabilities
            adapter_class = self._adapters[platform_name]
            temp_adapter = adapter_class({})  # Empty config for capability check only
            capabilities = temp_adapter._get_capabilities()

            return {
                "supports_projects": capabilities.supports_projects,
                "supports_issues": capabilities.supports_issues,
                "supports_sprints": capabilities.supports_sprints,
                "supports_time_tracking": capabilities.supports_time_tracking,
                "supports_story_points": capabilities.supports_story_points,
                "supports_custom_fields": capabilities.supports_custom_fields,
                "supports_issue_linking": capabilities.supports_issue_linking,
                "supports_comments": capabilities.supports_comments,
                "supports_attachments": capabilities.supports_attachments,
                "supports_workflows": capabilities.supports_workflows,
                "supports_bulk_operations": capabilities.supports_bulk_operations,
                "rate_limit_requests_per_hour": capabilities.rate_limit_requests_per_hour,
                "rate_limit_burst_size": capabilities.rate_limit_burst_size,
                "max_results_per_page": capabilities.max_results_per_page,
                "supports_cursor_pagination": capabilities.supports_cursor_pagination,
            }
        except Exception as e:
            logger.error(f"Failed to get capabilities for {platform_name}: {e}")
            return None

    def create_adapter(self, platform_name: str, config: dict[str, Any]) -> BasePlatformAdapter:
        """Create and configure a platform adapter instance.

        WHY: Centralized instance creation ensures consistent initialization
        patterns and enables caching for performance. Authentication is
        performed during creation to fail fast on configuration issues.

        Args:
            platform_name: Platform identifier to create adapter for.
            config: Platform-specific configuration including credentials.

        Returns:
            Configured and authenticated adapter instance.

        Raises:
            ValueError: If platform is not registered.
            ConnectionError: If authentication fails.
            Exception: If adapter initialization fails.
        """
        if platform_name not in self._adapters:
            available = ", ".join(self.get_available_platforms())
            raise ValueError(f"Unknown platform: {platform_name}. Available platforms: {available}")

        logger.info(f"Creating adapter for platform: {platform_name}")

        try:
            adapter_class = self._adapters[platform_name]
            adapter = adapter_class(config)

            # Test authentication during creation
            logger.info(f"Authenticating with {platform_name}...")
            if not adapter.authenticate():
                raise ConnectionError(f"Failed to authenticate with {platform_name}")

            # Test connection to validate configuration
            connection_info = adapter.test_connection()
            if connection_info.get("status") != "connected":
                error_msg = connection_info.get("error", "Unknown connection error")
                raise ConnectionError(f"Connection test failed for {platform_name}: {error_msg}")

            # Cache the instance for reuse
            self._instances[platform_name] = adapter
            logger.info(f"Successfully created and cached adapter for {platform_name}")

            return adapter

        except Exception as e:
            logger.error(f"Failed to create adapter for {platform_name}: {e}")
            # Remove failed instance from cache
            if platform_name in self._instances:
                del self._instances[platform_name]
            raise

    def get_adapter(self, platform_name: str) -> Optional[BasePlatformAdapter]:
        """Get existing adapter instance from cache.

        WHY: Reusing adapter instances avoids repeated authentication and
        connection setup, improving performance for multiple operations.

        Args:
            platform_name: Platform identifier to retrieve adapter for.

        Returns:
            Cached adapter instance, or None if not found or not cached.
        """
        return self._instances.get(platform_name)

    def remove_adapter_instance(self, platform_name: str) -> None:
        """Remove adapter instance from cache.

        WHY: Allows forcing recreation of adapter instances, useful when
        configuration changes or connection issues require fresh authentication.

        Args:
            platform_name: Platform identifier to remove from cache.
        """
        if platform_name in self._instances:
            del self._instances[platform_name]
            logger.info(f"Removed cached adapter instance for {platform_name}")

    def clear_all_instances(self) -> None:
        """Clear all cached adapter instances.

        WHY: Useful for cleanup operations, testing, or when configuration
        changes require fresh adapter creation.
        """
        instance_count = len(self._instances)
        self._instances.clear()
        logger.info(f"Cleared {instance_count} cached adapter instances")

    def validate_config(self, platform_name: str, config: dict[str, Any]) -> dict[str, Any]:
        """Validate configuration for a platform without creating full adapter.

        WHY: Allows configuration validation during setup or testing without
        the overhead of full adapter creation and authentication.

        Args:
            platform_name: Platform identifier to validate config for.
            config: Configuration to validate.

        Returns:
            Dictionary with validation results including status and any errors.
        """
        if platform_name not in self._adapters:
            return {
                "valid": False,
                "error": f"Unknown platform: {platform_name}",
                "missing_fields": [],
                "invalid_fields": [],
            }

        try:
            # Get adapter class to check required config fields
            adapter_class = self._adapters[platform_name]

            # Create instance with config for basic validation
            # Note: This doesn't call authenticate() to avoid side effects
            temp_adapter = adapter_class(config)

            # Basic validation passed if we got here
            return {
                "valid": True,
                "platform_name": temp_adapter.platform_name,
                "capabilities": self.get_platform_capabilities(platform_name),
                "missing_fields": [],
                "invalid_fields": [],
            }

        except KeyError as e:
            return {
                "valid": False,
                "error": f"Missing required configuration field: {str(e)}",
                "missing_fields": [str(e)],
                "invalid_fields": [],
            }
        except ValueError as e:
            return {
                "valid": False,
                "error": f"Invalid configuration value: {str(e)}",
                "missing_fields": [],
                "invalid_fields": [str(e)],
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Configuration validation failed: {str(e)}",
                "missing_fields": [],
                "invalid_fields": [],
            }

    def get_registry_status(self) -> dict[str, Any]:
        """Get current registry status and statistics.

        WHY: Provides diagnostic information for monitoring and debugging
        the registry state, useful for administration and troubleshooting.

        Returns:
            Dictionary containing registry statistics and status information.
        """
        active_instances = {}
        for platform_name, adapter in self._instances.items():
            try:
                # Test if cached instance is still valid
                connection_info = adapter.test_connection()
                active_instances[platform_name] = {
                    "status": connection_info.get("status", "unknown"),
                    "platform": adapter.platform_name,
                    "capabilities_count": sum(
                        1
                        for k, v in vars(adapter.capabilities).items()
                        if k.startswith("supports_") and v
                    ),
                }
            except Exception as e:
                active_instances[platform_name] = {"status": "error", "error": str(e)}

        return {
            "registered_platforms": len(self._adapters),
            "available_platforms": self.get_available_platforms(),
            "cached_instances": len(self._instances),
            "active_instances": active_instances,
            "registry_initialized": True,
        }
