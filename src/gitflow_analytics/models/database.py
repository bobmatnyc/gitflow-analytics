"""Database models for GitFlow Analytics using SQLAlchemy."""

from datetime import datetime
from pathlib import Path

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, create_engine
from sqlalchemy.orm import Session, sessionmaker, declarative_base
from typing import Any

Base: Any = declarative_base()


class CachedCommit(Base):
    """Cached commit analysis results."""

    __tablename__ = "cached_commits"

    # Primary key
    id = Column(Integer, primary_key=True)

    # Commit identification
    repo_path = Column(String, nullable=False)
    commit_hash = Column(String, nullable=False)

    # Commit data
    author_name = Column(String)
    author_email = Column(String)
    message = Column(String)
    timestamp = Column(DateTime)
    branch = Column(String)
    is_merge = Column(Boolean, default=False)

    # Metrics
    files_changed = Column(Integer)
    insertions = Column(Integer)
    deletions = Column(Integer)
    complexity_delta = Column(Float)

    # Extracted data
    story_points = Column(Integer, nullable=True)
    ticket_references = Column(JSON)  # List of ticket IDs

    # Cache metadata
    cached_at = Column(DateTime, default=datetime.utcnow)
    cache_version = Column(String, default="1.0")

    # Indexes for performance
    __table_args__ = (
        Index("idx_repo_commit", "repo_path", "commit_hash", unique=True),
        Index("idx_timestamp", "timestamp"),
        Index("idx_cached_at", "cached_at"),
    )


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
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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


class PullRequestCache(Base):
    """Cached pull request data."""

    __tablename__ = "pull_request_cache"

    id = Column(Integer, primary_key=True)
    repo_path = Column(String, nullable=False)
    pr_number = Column(Integer, nullable=False)

    # PR data
    title = Column(String)
    description = Column(String)
    author = Column(String)
    created_at = Column(DateTime)
    merged_at = Column(DateTime, nullable=True)

    # Extracted data
    story_points = Column(Integer, nullable=True)
    labels = Column(JSON)  # List of labels

    # Associated commits
    commit_hashes = Column(JSON)  # List of commit hashes

    # Cache metadata
    cached_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("idx_repo_pr", "repo_path", "pr_number", unique=True),)


class IssueCache(Base):
    """Cached issue data from various platforms."""

    __tablename__ = "issue_cache"

    id = Column(Integer, primary_key=True)

    # Issue identification
    platform = Column(String, nullable=False)  # 'jira', 'github', 'clickup', 'linear'
    issue_id = Column(String, nullable=False)
    project_key = Column(String, nullable=False)

    # Issue data
    title = Column(String)
    description = Column(String)
    status = Column(String)
    assignee = Column(String, nullable=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    resolved_at = Column(DateTime, nullable=True)

    # Extracted data
    story_points = Column(Integer, nullable=True)
    labels = Column(JSON)

    # Platform-specific data
    platform_data = Column(JSON)  # Additional platform-specific fields

    # Cache metadata
    cached_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_platform_issue", "platform", "issue_id", unique=True),
        Index("idx_project_key", "project_key"),
    )


class QualitativeCommitData(Base):
    """Extended commit data with qualitative analysis results.
    
    This table stores the results of qualitative analysis performed on commits,
    including change type classification, domain analysis, risk assessment,
    and processing metadata.
    """
    __tablename__ = 'qualitative_commits'
    
    # Link to existing commit
    commit_id = Column(Integer, ForeignKey('cached_commits.id'), primary_key=True)
    
    # Classification results
    change_type = Column(String, nullable=False)
    change_type_confidence = Column(Float, nullable=False)
    business_domain = Column(String, nullable=False)  
    domain_confidence = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    risk_factors = Column(JSON)  # List of risk factors
    
    # Intent and context analysis
    intent_signals = Column(JSON)  # Intent analysis results
    collaboration_patterns = Column(JSON)  # Team interaction patterns
    technical_context = Column(JSON)  # Technical context information
    
    # Processing metadata
    processing_method = Column(String, nullable=False)  # 'nlp' or 'llm'
    processing_time_ms = Column(Float)
    confidence_score = Column(Float, nullable=False)
    
    # Timestamps
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    analysis_version = Column(String, default="1.0")
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_change_type', 'change_type'),
        Index('idx_business_domain', 'business_domain'),
        Index('idx_risk_level', 'risk_level'),
        Index('idx_qualitative_confidence', 'confidence_score'),
        Index('idx_processing_method', 'processing_method'),
        Index('idx_analyzed_at', 'analyzed_at'),
    )


class PatternCache(Base):
    """Cache for learned patterns and classifications.
    
    This table stores frequently occurring patterns to avoid reprocessing
    similar commits and to improve classification accuracy over time.
    """
    __tablename__ = 'pattern_cache'
    
    id = Column(Integer, primary_key=True)
    
    # Pattern identification
    message_hash = Column(String, nullable=False, unique=True)
    semantic_fingerprint = Column(String, nullable=False)
    
    # Cached classification results
    classification_result = Column(JSON, nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Usage tracking for cache management
    hit_count = Column(Integer, default=1)
    last_used = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Source tracking
    source_method = Column(String, nullable=False)  # 'nlp' or 'llm'
    source_model = Column(String)  # Model/method that created this pattern
    
    # Performance tracking
    avg_processing_time_ms = Column(Float)
    
    # Indexes for pattern matching and cleanup
    __table_args__ = (
        Index('idx_semantic_fingerprint', 'semantic_fingerprint'),
        Index('idx_pattern_confidence', 'confidence_score'),
        Index('idx_hit_count', 'hit_count'),
        Index('idx_last_used', 'last_used'),
        Index('idx_source_method', 'source_method'),
    )


class LLMUsageStats(Base):
    """Track LLM usage statistics for cost monitoring and optimization.
    
    This table helps monitor LLM API usage, costs, and performance to
    optimize the balance between speed, accuracy, and cost.
    """
    __tablename__ = 'llm_usage_stats'
    
    id = Column(Integer, primary_key=True)
    
    # API call metadata
    model_name = Column(String, nullable=False)
    api_provider = Column(String, default='openrouter')
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Usage metrics
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    
    # Cost tracking
    estimated_cost_usd = Column(Float)
    cost_per_token = Column(Float)
    
    # Batch information
    batch_size = Column(Integer, default=1)  # Number of commits processed
    batch_id = Column(String)  # Group related calls
    
    # Quality metrics
    avg_confidence_score = Column(Float)
    success = Column(Boolean, default=True)
    error_message = Column(String)
    
    # Indexes for analysis and monitoring
    __table_args__ = (
        Index('idx_model_timestamp', 'model_name', 'timestamp'),
        Index('idx_llm_timestamp', 'timestamp'),
        Index('idx_batch_id', 'batch_id'),
        Index('idx_success', 'success'),
    )


class Database:
    """Database connection manager."""

    def __init__(self, db_path: Path):
        """Initialize database connection."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def init_db(self) -> None:
        """Initialize database tables."""
        Base.metadata.create_all(self.engine)
