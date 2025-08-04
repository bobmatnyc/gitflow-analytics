"""Schema versioning for tracking data structure changes."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import Column, DateTime, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class SchemaVersion(Base):
    """Track schema versions for incremental data processing."""

    __tablename__ = "schema_versions"

    component = Column(String, primary_key=True)  # e.g., 'qualitative', 'identity', 'core'
    version_hash = Column(String, nullable=False)  # Hash of schema definition
    schema_definition = Column(Text, nullable=False)  # JSON schema definition
    created_at = Column(DateTime, default=datetime.utcnow)
    last_processed_date = Column(
        DateTime, nullable=True
    )  # Last date we processed data with this schema


class SchemaVersionManager:
    """Manages schema versions and determines if incremental processing is possible."""

    # Define current schema versions for each component
    CURRENT_SCHEMAS = {
        "qualitative": {
            "version": "2.0",
            "fields": [
                "change_type",
                "change_type_confidence",
                "business_domain",
                "domain_confidence",
                "risk_level",
                "risk_factors",
                "intent_signals",
                "collaboration_patterns",
                "technical_context",
                "processing_method",
                "processing_time_ms",
                "confidence_score",
            ],
            "config_fields": [
                "nlp_config",
                "llm_config",
                "cache_config",
                "confidence_threshold",
                "max_llm_fallback_pct",
            ],
        },
        "identity": {
            "version": "1.3",
            "fields": [
                "canonical_id",
                "primary_name",
                "primary_email",
                "manual_mappings",
                "similarity_threshold",
                "auto_analysis",
                "display_names",
                "preferred_name_field",
            ],
        },
        "core": {
            "version": "1.0",
            "fields": [
                "story_points",
                "ticket_references",
                "files_changed",
                "insertions",
                "deletions",
                "complexity_delta",
                "branch_mapping_rules",
            ],
        },
        "github": {
            "version": "1.0",
            "fields": [
                "pr_data",
                "pr_metrics",
                "issue_data",
                "rate_limit_retries",
                "backoff_factor",
                "allowed_ticket_platforms",
            ],
        },
        "jira": {
            "version": "1.0",
            "fields": ["story_point_fields", "project_keys", "base_url", "issue_data"],
        },
    }

    def __init__(self, cache_dir: Path):
        """Initialize schema version manager."""
        self.cache_dir = cache_dir
        self.db_path = cache_dir / "schema_versions.db"
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)

    def get_schema_hash(self, component: str, config: Optional[dict[str, Any]] = None) -> str:
        """Generate hash for a component's schema including configuration."""
        if component not in self.CURRENT_SCHEMAS:
            raise ValueError(f"Unknown component: {component}")

        schema_def = self.CURRENT_SCHEMAS[component].copy()

        # Include relevant configuration in the hash
        if config and "config_fields" in schema_def:
            relevant_config = {}
            for field in schema_def["config_fields"]:
                if field in config:
                    relevant_config[field] = self._normalize_config_value(config[field])
            schema_def["config"] = relevant_config

        # Create deterministic hash
        schema_json = json.dumps(schema_def, sort_keys=True)
        return hashlib.sha256(schema_json.encode()).hexdigest()[:16]

    def _normalize_config_value(self, value: Any) -> Any:
        """Normalize config values for consistent hashing."""
        if isinstance(value, dict):
            return {k: self._normalize_config_value(v) for k, v in sorted(value.items())}
        elif isinstance(value, list):
            return sorted([self._normalize_config_value(v) for v in value])
        elif isinstance(value, (int, float, str, bool, type(None))):
            return value
        else:
            # Convert complex objects to string representation
            return str(value)

    def has_schema_changed(self, component: str, config: Optional[dict[str, Any]] = None) -> bool:
        """Check if schema has changed since last processing."""
        current_hash = self.get_schema_hash(component, config)

        with self.session_factory() as session:
            stored_version = session.query(SchemaVersion).filter_by(component=component).first()

            if not stored_version:
                return True  # No previous schema, consider changed

            return stored_version.version_hash != current_hash

    def update_schema_version(
        self,
        component: str,
        config: Optional[dict[str, Any]] = None,
        last_processed_date: Optional[datetime] = None,
    ):
        """Update stored schema version."""
        current_hash = self.get_schema_hash(component, config)
        schema_def = json.dumps(self.CURRENT_SCHEMAS[component], sort_keys=True)

        # Ensure date is timezone-aware before storing
        if last_processed_date and last_processed_date.tzinfo is None:
            last_processed_date = last_processed_date.replace(tzinfo=timezone.utc)

        with self.session_factory() as session:
            stored_version = session.query(SchemaVersion).filter_by(component=component).first()

            if stored_version:
                stored_version.version_hash = current_hash
                stored_version.schema_definition = schema_def
                if last_processed_date:
                    stored_version.last_processed_date = last_processed_date
            else:
                stored_version = SchemaVersion(
                    component=component,
                    version_hash=current_hash,
                    schema_definition=schema_def,
                    last_processed_date=last_processed_date,
                )
                session.add(stored_version)

            session.commit()

    def get_last_processed_date(self, component: str) -> Optional[datetime]:
        """Get the last date data was processed for this component."""
        with self.session_factory() as session:
            stored_version = session.query(SchemaVersion).filter_by(component=component).first()
            return stored_version.last_processed_date if stored_version else None

    def should_process_date(
        self, component: str, date: datetime, config: Optional[dict[str, Any]] = None
    ) -> bool:
        """Determine if we should process data for a given date."""
        # Always process if schema has changed
        if self.has_schema_changed(component, config):
            return True

        # Check if we've already processed this date
        last_processed = self.get_last_processed_date(component)
        if not last_processed:
            return True

        # Ensure both dates are timezone-aware for comparison
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        if last_processed.tzinfo is None:
            last_processed = last_processed.replace(tzinfo=timezone.utc)

        # Process if date is after last processed date
        return date > last_processed

    def mark_date_processed(
        self, component: str, date: datetime, config: Optional[dict[str, Any]] = None
    ):
        """Mark a date as processed for incremental tracking."""
        # Ensure date is timezone-aware before storing
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)

        with self.session_factory() as session:
            stored_version = session.query(SchemaVersion).filter_by(component=component).first()

            if stored_version:
                # Update to the latest processed date
                if not stored_version.last_processed_date:
                    stored_version.last_processed_date = date
                    session.commit()
                else:
                    # Ensure stored date is timezone-aware for comparison
                    stored_date = stored_version.last_processed_date
                    if stored_date.tzinfo is None:
                        stored_date = stored_date.replace(tzinfo=timezone.utc)

                    if date > stored_date:
                        stored_version.last_processed_date = date
                        session.commit()
            else:
                # Create new entry
                self.update_schema_version(component, config, date)

    def get_schema_info(self, component: str) -> dict[str, Any]:
        """Get detailed schema information for debugging."""
        with self.session_factory() as session:
            stored_version = session.query(SchemaVersion).filter_by(component=component).first()

            current_hash = self.get_schema_hash(component)

            return {
                "component": component,
                "current_schema_hash": current_hash,
                "stored_schema_hash": stored_version.version_hash if stored_version else None,
                "schema_changed": self.has_schema_changed(component),
                "last_processed": stored_version.last_processed_date if stored_version else None,
                "created_at": stored_version.created_at if stored_version else None,
            }

    def reset_component(self, component: str):
        """Reset schema version for a component (forces full reprocessing)."""
        with self.session_factory() as session:
            stored_version = session.query(SchemaVersion).filter_by(component=component).first()
            if stored_version:
                session.delete(stored_version)
                session.commit()


def create_schema_manager(cache_dir: Path) -> SchemaVersionManager:
    """Factory function to create a schema version manager."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return SchemaVersionManager(cache_dir)
