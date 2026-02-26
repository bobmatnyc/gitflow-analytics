"""Identity-related database models for GitFlow Analytics."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
)

from .database_base import Base, utcnow_tz_aware

class DeveloperIdentity(Base):
    """Developer identity mappings."""

    __tablename__ = "developer_identities"

    id = Column(Integer, primary_key=True)
    canonical_id = Column(String, unique=True, nullable=False)
    primary_name = Column(String, nullable=False)
    primary_email = Column(String, nullable=False)
    github_username = Column(String, nullable=True)

    # Statistics
    total_commits = Column(Integer, default=0)
    total_story_points = Column(Integer, default=0)
    first_seen = Column(DateTime(timezone=True), default=utcnow_tz_aware)
    last_seen = Column(DateTime(timezone=True), default=utcnow_tz_aware)

    # Metadata
    created_at = Column(DateTime(timezone=True), default=utcnow_tz_aware)
    updated_at = Column(DateTime(timezone=True), default=utcnow_tz_aware, onupdate=utcnow_tz_aware)

    __table_args__ = (
        Index("idx_primary_email", "primary_email"),
        Index("idx_canonical_id", "canonical_id"),
    )



class DeveloperAlias(Base):
    """Alternative names/emails for developers."""

    __tablename__ = "developer_aliases"

    id = Column(Integer, primary_key=True)
    canonical_id = Column(String, nullable=False)  # Foreign key to DeveloperIdentity
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)

    __table_args__ = (
        Index("idx_alias_email", "email"),
        Index("idx_alias_canonical_id", "canonical_id"),
        Index("idx_name_email", "name", "email", unique=True),
    )



class PatternCache(Base):
    """Cache for learned patterns and classifications.

    This table stores frequently occurring patterns to avoid reprocessing
    similar commits and to improve classification accuracy over time.
    """

    __tablename__ = "pattern_cache"

    id = Column(Integer, primary_key=True)

    # Pattern identification
    message_hash = Column(String, nullable=False, unique=True)
    semantic_fingerprint = Column(String, nullable=False)

    # Cached classification results
    classification_result = Column(JSON, nullable=False)
    confidence_score = Column(Float, nullable=False)

    # Usage tracking for cache management
    hit_count = Column(Integer, default=1)
    last_used = Column(DateTime(timezone=True), default=utcnow_tz_aware)
    created_at = Column(DateTime(timezone=True), default=utcnow_tz_aware)

    # Source tracking
    source_method = Column(String, nullable=False)  # 'nlp' or 'llm'
    source_model = Column(String)  # Model/method that created this pattern

    # Performance tracking
    avg_processing_time_ms = Column(Float)

    # Indexes for pattern matching and cleanup
    __table_args__ = (
        Index("idx_semantic_fingerprint", "semantic_fingerprint"),
        Index("idx_pattern_confidence", "confidence_score"),
        Index("idx_hit_count", "hit_count"),
        Index("idx_last_used", "last_used"),
        Index("idx_source_method", "source_method"),
    )



