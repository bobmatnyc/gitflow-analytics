"""Database models for GitFlow Analytics using SQLAlchemy."""

from datetime import datetime
from pathlib import Path

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Index, Integer, String, create_engine
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
