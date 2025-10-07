"""Analysis pass for auto-aliasing developer identities."""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from .analyzer import LLMIdentityAnalyzer
from .models import IdentityAnalysisResult

logger = logging.getLogger(__name__)


class IdentityAnalysisPass:
    """Performs an analysis pass to auto-alias developer identities."""

    def __init__(self, config_path: Path):
        """Initialize with configuration."""
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file."""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Handle environment variables
        config_dir = self.config_path.parent
        env_file = config_dir / ".env"

        if env_file.exists():
            from dotenv import load_dotenv

            load_dotenv(env_file)

        return config

    def run_analysis(
        self,
        commits: list[dict[str, Any]],
        output_path: Optional[Path] = None,
        apply_to_config: bool = False,
    ) -> IdentityAnalysisResult:
        """Run identity analysis pass on commits."""
        logger.info("Starting identity analysis pass...")

        # Get OpenRouter API key from config or env
        api_key = None
        if "qualitative" in self.config and self.config["qualitative"].get("enabled"):
            api_key = self._resolve_env_var(
                self.config["qualitative"].get("openrouter_api_key", "")
            )

        # Initialize analyzer
        analyzer = LLMIdentityAnalyzer(
            api_key=api_key,
            model=self.config.get("qualitative", {}).get("model", "openai/gpt-4o-mini"),
            confidence_threshold=self.config.get("analysis", {}).get("similarity_threshold", 0.85),
        )

        # Run analysis
        result = analyzer.analyze_identities(commits)

        logger.info(f"Analysis complete: {len(result.clusters)} identity clusters found")

        # Save analysis report
        if output_path:
            self._save_analysis_report(result, output_path)

        # Apply to configuration if requested
        if apply_to_config:
            self._apply_to_config(result)

        return result

    def _resolve_env_var(self, value: str) -> str:
        """Resolve environment variable references."""
        if value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1]
            return os.getenv(var_name, value)
        return value

    def _save_analysis_report(self, result: IdentityAnalysisResult, output_path: Path):
        """Save analysis report to file."""
        report = {"analysis_metadata": result.analysis_metadata, "identity_clusters": []}

        for cluster in result.clusters:
            cluster_data = {
                "canonical_name": cluster.canonical_name,
                "canonical_email": cluster.canonical_email,
                "confidence": cluster.confidence,
                "reasoning": cluster.reasoning,
                "total_commits": cluster.total_commits,
                "aliases": [],
            }

            for alias in cluster.aliases:
                cluster_data["aliases"].append(
                    {
                        "name": alias.name,
                        "email": alias.email,
                        "commit_count": alias.commit_count,
                        "repositories": list(alias.repositories),
                    }
                )

            report["identity_clusters"].append(cluster_data)

        # Add unresolved identities
        if result.unresolved_identities:
            report["unresolved_identities"] = []
            for identity in result.unresolved_identities:
                report["unresolved_identities"].append(
                    {
                        "name": identity.name,
                        "email": identity.email,
                        "commit_count": identity.commit_count,
                        "repositories": list(identity.repositories),
                    }
                )

        # Write YAML report
        with open(output_path, "w") as f:
            yaml.dump(report, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Analysis report saved to: {output_path}")

    def _apply_to_config(self, result: IdentityAnalysisResult):
        """Apply analysis results to configuration file."""
        # Get manual mappings from analysis
        new_mappings = result.get_manual_mappings()

        if not new_mappings:
            logger.info("No new identity mappings to apply")
            return

        # Load current config
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Ensure analysis section exists
        if "analysis" not in config:
            config["analysis"] = {}

        # Get existing manual mappings
        existing_mappings = config["analysis"].get("manual_identity_mappings", [])

        # Merge new mappings
        existing_emails = set()
        for mapping in existing_mappings:
            # Support both canonical_email and primary_email for backward compatibility
            email = mapping.get("canonical_email") or mapping.get("primary_email", "")
            existing_emails.add(email.lower())

        for new_mapping in new_mappings:
            # New mappings use primary_email, but support canonical_email for backward compat
            canonical_email = (
                new_mapping.get("primary_email") or new_mapping.get("canonical_email", "")
            ).lower()
            if not canonical_email:
                logger.warning(f"Skipping mapping with no email: {new_mapping}")
                continue
            if canonical_email not in existing_emails:
                existing_mappings.append(new_mapping)
                logger.info(f"Added identity mapping for: {canonical_email}")
            else:
                # Update existing mapping with new aliases
                for existing in existing_mappings:
                    existing_email = existing.get("canonical_email") or existing.get(
                        "primary_email", ""
                    )
                    if existing_email.lower() == canonical_email:
                        existing_aliases = set(
                            alias.lower() for alias in existing.get("aliases", [])
                        )
                        new_aliases = set(alias.lower() for alias in new_mapping["aliases"])
                        combined_aliases = existing_aliases | new_aliases
                        existing["aliases"] = list(combined_aliases)
                        # Update confidence and reasoning if new mapping has higher confidence
                        if new_mapping.get("confidence", 0) > existing.get("confidence", 0):
                            existing["confidence"] = new_mapping.get("confidence")
                            existing["reasoning"] = new_mapping.get("reasoning")
                        if new_aliases - existing_aliases:
                            logger.info(f"Updated aliases for: {canonical_email}")
                        break

        # Update config
        config["analysis"]["manual_identity_mappings"] = existing_mappings

        # Write updated config
        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Updated configuration with {len(new_mappings)} identity mappings")

    def generate_suggested_config(self, result: IdentityAnalysisResult) -> dict[str, Any]:
        """Generate suggested configuration snippet for manual review."""
        manual_mappings = result.get_manual_mappings()

        # Also generate exclusions for bots
        bot_patterns = [
            r".*\[bot\]$",  # Matches "ewtn-version-bumper[bot]", "dependabot[bot]"
            r".*-bot$",  # Matches names ending in "-bot"
            r"^bot-.*",  # Matches names starting with "bot-"
            r".*\sBot$",  # Matches "CNA Studio Bot", "GitHub Bot", etc.
            r".*\sbot$",  # Matches "studio bot", "merge bot", etc.
            r"^Bot\s.*",  # Matches "Bot User", "Bot Account", etc.
            r"^bot\s.*",  # Matches "bot user", "bot account", etc.
        ]

        suggested_exclusions = []
        logger.debug(
            f"Bot detection: checking {len(result.unresolved_identities)} unresolved identities"
        )

        for identity in result.unresolved_identities:
            logger.debug(
                f"Bot detection: checking identity '{identity.name}' against {len(bot_patterns)} patterns"
            )
            for pattern in bot_patterns:
                import re

                if re.match(pattern, identity.name, re.IGNORECASE):
                    logger.debug(
                        f"Bot detection: MATCH - '{identity.name}' matches pattern '{pattern}'"
                    )
                    suggested_exclusions.append(identity.name)
                    break
            else:
                logger.debug(
                    f"Bot detection: NO MATCH - '{identity.name}' doesn't match any bot patterns"
                )

        if suggested_exclusions:
            logger.debug(
                f"Bot detection: found {len(suggested_exclusions)} bots to exclude: {suggested_exclusions}"
            )
        else:
            logger.debug("Bot detection: no bots detected for exclusion")

        return {
            "analysis": {"manual_identity_mappings": manual_mappings},
            "exclude": {"authors": suggested_exclusions} if suggested_exclusions else {},
        }
