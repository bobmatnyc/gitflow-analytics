"""Developer identity aliases management.

This module provides functionality for managing developer identity aliases
across multiple configuration files. Aliases can be shared to maintain
consistent identity resolution across different analysis configurations.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DeveloperAlias:
    """A developer alias configuration.

    Represents a single developer with their primary email and all known aliases.
    Supports both manual and LLM-generated alias configurations with confidence scores.
    """

    primary_email: str
    aliases: list[str] = field(default_factory=list)
    name: Optional[str] = None
    confidence: float = 1.0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for YAML serialization.

        Returns:
            Dictionary representation with optional fields omitted if not set
        """
        result: dict[str, Any] = {
            "primary_email": self.primary_email,
            "aliases": self.aliases,
        }

        if self.name:
            result["name"] = self.name

        # Only include confidence and reasoning for LLM-generated aliases
        if self.confidence < 1.0:
            result["confidence"] = round(self.confidence, 2)
            if self.reasoning:
                result["reasoning"] = self.reasoning

        return result


class AliasesManager:
    """Manages developer identity aliases.

    Provides functionality to load, save, and manipulate developer identity aliases.
    Supports both manual aliases (confidence=1.0) and LLM-generated aliases with
    confidence scores and reasoning.

    Example:
        >>> manager = AliasesManager(Path("aliases.yaml"))
        >>> manager.load()
        >>> manager.add_alias(DeveloperAlias(
        ...     primary_email="john@company.com",
        ...     aliases=["jdoe@gmail.com"],
        ...     name="John Doe"
        ... ))
        >>> manager.save()
    """

    def __init__(self, aliases_path: Optional[Path] = None):
        """Initialize aliases manager.

        Args:
            aliases_path: Path to aliases.yaml file. If None, aliases must be
                         added programmatically or loaded from another source.
        """
        self.aliases_path = aliases_path
        self.aliases: list[DeveloperAlias] = []

        if aliases_path and aliases_path.exists():
            self.load()

    def load(self) -> None:
        """Load aliases from file.

        Loads developer aliases from the configured YAML file. If the file
        doesn't exist or is empty, initializes with an empty alias list.

        Raises:
            yaml.YAMLError: If the YAML file is malformed
        """
        if not self.aliases_path or not self.aliases_path.exists():
            logger.debug("No aliases file found or path not set")
            return

        try:
            with open(self.aliases_path) as f:
                data = yaml.safe_load(f) or {}

            self.aliases = []
            for alias_data in data.get("developer_aliases", []):
                # Support both 'primary_email' (new) and 'canonical_email' (old)
                primary_email = alias_data.get("primary_email") or alias_data.get("canonical_email")

                if not primary_email:
                    logger.warning(f"Skipping alias entry without primary_email: {alias_data}")
                    continue

                self.aliases.append(
                    DeveloperAlias(
                        primary_email=primary_email,
                        aliases=alias_data.get("aliases", []),
                        name=alias_data.get("name"),
                        confidence=alias_data.get("confidence", 1.0),
                        reasoning=alias_data.get("reasoning", ""),
                    )
                )

            logger.info(f"Loaded {len(self.aliases)} developer aliases from {self.aliases_path}")

        except yaml.YAMLError as e:
            logger.error(f"Error parsing aliases file {self.aliases_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading aliases file {self.aliases_path}: {e}")
            raise

    def save(self) -> None:
        """Save aliases to file.

        Writes all developer aliases to the configured YAML file with proper
        formatting and comments. Creates the parent directory if it doesn't exist.

        Raises:
            OSError: If file cannot be written
        """
        if not self.aliases_path:
            logger.warning("No aliases path configured, cannot save")
            return

        # Ensure directory exists
        self.aliases_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Build data structure with comments
            data = {
                "# Developer Identity Aliases": None,
                "# Generated by GitFlow Analytics": None,
                "# Share this file across multiple config files": None,
                "# Each alias maps multiple email addresses to a single developer": None,
                "developer_aliases": [alias.to_dict() for alias in self.aliases],
            }

            with open(self.aliases_path, "w") as f:
                # Custom YAML dump to preserve comments
                f.write("# Developer Identity Aliases\n")
                f.write("# Generated by GitFlow Analytics\n")
                f.write("# Share this file across multiple config files\n")
                f.write("# Each alias maps multiple email addresses to a single developer\n\n")

                # Write the aliases list
                yaml.dump(
                    {"developer_aliases": data["developer_aliases"]},
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )

            logger.info(f"Saved {len(self.aliases)} developer aliases to {self.aliases_path}")

        except Exception as e:
            logger.error(f"Error saving aliases file {self.aliases_path}: {e}")
            raise

    def add_alias(self, alias: DeveloperAlias) -> None:
        """Add or update a developer alias.

        If an alias with the same primary email already exists, it will be replaced.
        This ensures there is only one alias configuration per developer.

        Args:
            alias: The developer alias to add or update
        """
        # Remove existing alias for same primary email
        self.aliases = [a for a in self.aliases if a.primary_email != alias.primary_email]
        self.aliases.append(alias)
        logger.debug(f"Added/updated alias for {alias.primary_email}")

    def remove_alias(self, primary_email: str) -> bool:
        """Remove a developer alias by primary email.

        Args:
            primary_email: The primary email of the alias to remove

        Returns:
            True if an alias was removed, False if not found
        """
        original_count = len(self.aliases)
        self.aliases = [a for a in self.aliases if a.primary_email != primary_email]
        removed = len(self.aliases) < original_count
        if removed:
            logger.debug(f"Removed alias for {primary_email}")
        return removed

    def get_alias(self, primary_email: str) -> Optional[DeveloperAlias]:
        """Get a developer alias by primary email.

        Args:
            primary_email: The primary email to look up

        Returns:
            The developer alias if found, None otherwise
        """
        for alias in self.aliases:
            if alias.primary_email == primary_email:
                return alias
        return None

    def to_manual_mappings(self) -> list[dict[str, Any]]:
        """Convert aliases to config manual_identity_mappings format.

        Converts the internal alias representation to the format expected
        by the GitFlow Analytics configuration's manual_identity_mappings field.

        Returns:
            List of manual identity mapping dictionaries
        """
        mappings = []
        for alias in self.aliases:
            mapping: dict[str, Any] = {"primary_email": alias.primary_email}

            if alias.name:
                mapping["name"] = alias.name

            mapping["aliases"] = alias.aliases

            # Include confidence and reasoning for LLM-generated mappings
            if alias.confidence < 1.0:
                mapping["confidence"] = alias.confidence
                if alias.reasoning:
                    mapping["reasoning"] = alias.reasoning

            mappings.append(mapping)

        return mappings

    def merge_from_mappings(self, mappings: list[dict[str, Any]]) -> None:
        """Merge aliases from manual identity mappings.

        Takes manual identity mappings from a config file and merges them
        into the current alias set. Existing aliases are preserved unless
        they conflict with the new mappings.

        Args:
            mappings: List of manual identity mapping dictionaries
        """
        for mapping in mappings:
            # Support both field name variants
            primary_email = mapping.get("primary_email") or mapping.get("canonical_email")

            if not primary_email:
                logger.warning(f"Skipping mapping without primary_email: {mapping}")
                continue

            alias = DeveloperAlias(
                primary_email=primary_email,
                aliases=mapping.get("aliases", []),
                name=mapping.get("name"),
                confidence=mapping.get("confidence", 1.0),
                reasoning=mapping.get("reasoning", ""),
            )

            self.add_alias(alias)

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the aliases.

        Returns:
            Dictionary with statistics including total aliases, manual vs LLM-generated,
            average confidence, etc.
        """
        if not self.aliases:
            return {
                "total_aliases": 0,
                "manual_aliases": 0,
                "llm_aliases": 0,
                "avg_confidence": 0.0,
                "total_email_addresses": 0,
            }

        manual_count = sum(1 for a in self.aliases if a.confidence == 1.0)
        llm_count = len(self.aliases) - manual_count
        avg_confidence = sum(a.confidence for a in self.aliases) / len(self.aliases)
        total_emails = sum(len(a.aliases) + 1 for a in self.aliases)  # +1 for primary

        return {
            "total_aliases": len(self.aliases),
            "manual_aliases": manual_count,
            "llm_aliases": llm_count,
            "avg_confidence": round(avg_confidence, 3),
            "total_email_addresses": total_emails,
        }
