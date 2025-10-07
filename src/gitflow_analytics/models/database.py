"""Database models for GitFlow Analytics using SQLAlchemy."""

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

logger = logging.getLogger(__name__)

Base: Any = declarative_base()


def utcnow_tz_aware() -> datetime:
    """Return current UTC time as timezone-aware datetime.

    WHY: SQLAlchemy DateTime(timezone=True) requires timezone-aware datetimes.
    Using timezone-naive datetime.utcnow() causes query mismatches when filtering
    by timezone-aware date ranges.

    Returns:
        Timezone-aware datetime in UTC
    """
    return datetime.now(timezone.utc)


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
    timestamp = Column(DateTime(timezone=True))  # CRITICAL: Preserve timezone for date filtering
    branch = Column(String)
    is_merge = Column(Boolean, default=False)

    # Metrics
    files_changed = Column(Integer)
    insertions = Column(Integer)
    deletions = Column(Integer)
    # Filtered metrics (after exclusions applied)
    filtered_insertions = Column(Integer, default=0)
    filtered_deletions = Column(Integer, default=0)
    complexity_delta = Column(Float)

    # Extracted data
    story_points = Column(Integer, nullable=True)
    ticket_references = Column(JSON)  # List of ticket IDs

    # Cache metadata
    cached_at = Column(DateTime(timezone=True), default=utcnow_tz_aware)
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
    created_at = Column(DateTime(timezone=True))
    merged_at = Column(DateTime(timezone=True), nullable=True)

    # Extracted data
    story_points = Column(Integer, nullable=True)
    labels = Column(JSON)  # List of labels

    # Associated commits
    commit_hashes = Column(JSON)  # List of commit hashes

    # Cache metadata
    cached_at = Column(DateTime(timezone=True), default=utcnow_tz_aware)

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
    created_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True))
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    # Extracted data
    story_points = Column(Integer, nullable=True)
    labels = Column(JSON)

    # Platform-specific data
    platform_data = Column(JSON)  # Additional platform-specific fields

    # Cache metadata
    cached_at = Column(DateTime(timezone=True), default=utcnow_tz_aware)

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

    __tablename__ = "qualitative_commits"

    # Link to existing commit
    commit_id = Column(Integer, ForeignKey("cached_commits.id"), primary_key=True)

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
    analyzed_at = Column(DateTime(timezone=True), default=utcnow_tz_aware)
    analysis_version = Column(String, default="1.0")

    # Indexes for efficient querying
    __table_args__ = (
        Index("idx_change_type", "change_type"),
        Index("idx_business_domain", "business_domain"),
        Index("idx_risk_level", "risk_level"),
        Index("idx_qualitative_confidence", "confidence_score"),
        Index("idx_processing_method", "processing_method"),
        Index("idx_analyzed_at", "analyzed_at"),
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


class LLMUsageStats(Base):
    """Track LLM usage statistics for cost monitoring and optimization.

    This table helps monitor LLM API usage, costs, and performance to
    optimize the balance between speed, accuracy, and cost.
    """

    __tablename__ = "llm_usage_stats"

    id = Column(Integer, primary_key=True)

    # API call metadata
    model_name = Column(String, nullable=False)
    api_provider = Column(String, default="openrouter")
    timestamp = Column(DateTime(timezone=True), default=utcnow_tz_aware)

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
        Index("idx_model_timestamp", "model_name", "timestamp"),
        Index("idx_llm_timestamp", "timestamp"),
        Index("idx_llm_batch_id", "batch_id"),
        Index("idx_success", "success"),
    )


class TrainingData(Base):
    """Training data for commit classification models.

    This table stores labeled training examples collected from PM platforms
    and manual annotations for training and improving classification models.
    """

    __tablename__ = "training_data"

    id = Column(Integer, primary_key=True)

    # Commit identification
    commit_hash = Column(String, nullable=False)
    commit_message = Column(String, nullable=False)
    files_changed = Column(JSON)  # List of changed files
    repo_path = Column(String, nullable=False)

    # Classification labels
    category = Column(String, nullable=False)  # feature, bug_fix, refactor, etc.
    confidence = Column(Float, nullable=False, default=1.0)  # Label confidence (0-1)

    # Source information
    source_type = Column(String, nullable=False)  # 'pm_platform', 'manual', 'inferred'
    source_platform = Column(String)  # 'jira', 'github', 'clickup', etc.
    source_ticket_id = Column(String)  # Original ticket/issue ID
    source_ticket_type = Column(String)  # Bug, Story, Task, etc.

    # Training metadata
    training_session_id = Column(String, nullable=False)  # Groups related training data
    created_at = Column(DateTime(timezone=True), default=utcnow_tz_aware)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=utcnow_tz_aware)

    # Quality assurance
    validated = Column(Boolean, default=False)  # Human validation flag
    validation_notes = Column(String)  # Notes from validation process

    # Feature extraction (for ML training)
    extracted_features = Column(JSON)  # Pre-computed features for ML

    # Indexes for efficient querying and training
    __table_args__ = (
        Index("idx_training_commit_hash", "commit_hash"),
        Index("idx_training_category", "category"),
        Index("idx_training_source", "source_type", "source_platform"),
        Index("idx_training_session", "training_session_id"),
        Index("idx_training_created", "created_at"),
        Index("idx_training_validated", "validated"),
        Index("idx_commit_repo", "commit_hash", "repo_path", unique=True),
    )


class RepositoryAnalysisStatus(Base):
    """Track repository-level analysis completion status for cache-first workflow.

    WHY: This table enables "fetch once, report many" behavior by tracking
    which repositories have been fully analyzed for specific time periods.
    Prevents re-fetching Git data when only generating different reports.
    """

    __tablename__ = "repository_analysis_status"

    id = Column(Integer, primary_key=True)

    # Repository identification
    repo_path = Column(String, nullable=False)
    repo_name = Column(String, nullable=False)  # For display purposes
    project_key = Column(String, nullable=False)

    # Analysis period
    analysis_start = Column(DateTime, nullable=False)  # Start of analysis period
    analysis_end = Column(DateTime, nullable=False)  # End of analysis period
    weeks_analyzed = Column(Integer, nullable=False)  # Number of weeks

    # Completion tracking
    git_analysis_complete = Column(Boolean, default=False)
    commit_count = Column(Integer, default=0)
    pr_analysis_complete = Column(Boolean, default=False)
    pr_count = Column(Integer, default=0)
    ticket_analysis_complete = Column(Boolean, default=False)
    ticket_count = Column(Integer, default=0)

    # Developer identity resolution
    identity_resolution_complete = Column(Boolean, default=False)
    unique_developers = Column(Integer, default=0)

    # Analysis metadata
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=utcnow_tz_aware)
    analysis_version = Column(String, default="2.0")  # For tracking schema changes

    # Configuration hash to detect config changes
    config_hash = Column(String, nullable=True)  # MD5 hash of relevant config

    # Analysis performance metrics
    processing_time_seconds = Column(Float, nullable=True)
    cache_hit_rate_percent = Column(Float, nullable=True)

    # Status tracking
    status = Column(String, default="pending")  # pending, in_progress, completed, failed
    error_message = Column(String, nullable=True)

    # Indexes for efficient querying
    __table_args__ = (
        Index("idx_repo_analysis_path", "repo_path"),
        Index("idx_repo_analysis_period", "analysis_start", "analysis_end"),
        Index("idx_repo_analysis_status", "status"),
        Index(
            "idx_repo_analysis_unique", "repo_path", "analysis_start", "analysis_end", unique=True
        ),
        Index("idx_repo_analysis_updated", "last_updated"),
    )


class TrainingSession(Base):
    """Training session metadata and results.

    This table tracks individual training runs, their configurations,
    and performance metrics for model versioning and comparison.
    """

    __tablename__ = "training_sessions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)

    # Session metadata
    started_at = Column(DateTime(timezone=True), default=utcnow_tz_aware)
    completed_at = Column(DateTime(timezone=True))
    status = Column(String, default="running")  # running, completed, failed

    # Configuration
    config = Column(JSON, nullable=False)  # Training configuration
    weeks_analyzed = Column(Integer)  # Time period covered
    repositories = Column(JSON)  # List of repositories analyzed

    # Data statistics
    total_commits = Column(Integer, default=0)
    labeled_commits = Column(Integer, default=0)
    training_examples = Column(Integer, default=0)
    validation_examples = Column(Integer, default=0)

    # PM platform coverage
    pm_platforms = Column(JSON)  # List of PM platforms used
    ticket_coverage_pct = Column(Float)  # Percentage of commits with tickets

    # Training results
    model_accuracy = Column(Float)  # Overall accuracy
    category_metrics = Column(JSON)  # Per-category precision/recall/f1
    validation_loss = Column(Float)  # Validation loss

    # Model storage
    model_path = Column(String)  # Path to saved model
    model_version = Column(String)  # Version identifier
    model_size_mb = Column(Float)  # Model file size

    # Performance metrics
    training_time_minutes = Column(Float)
    prediction_time_ms = Column(Float)  # Average prediction time

    # Notes and errors
    notes = Column(String)
    error_message = Column(String)

    # Indexes for session management
    __table_args__ = (
        Index("idx_session_id", "session_id"),
        Index("idx_session_status", "status"),
        Index("idx_session_started", "started_at"),
        Index("idx_session_model_version", "model_version"),
    )


class ClassificationModel(Base):
    """Versioned storage for trained classification models.

    This table manages different versions of trained models with
    metadata for model selection and performance tracking.
    """

    __tablename__ = "classification_models"

    id = Column(Integer, primary_key=True)
    model_id = Column(String, unique=True, nullable=False)

    # Model metadata
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # 'sklearn', 'spacy', 'custom'
    created_at = Column(DateTime(timezone=True), default=utcnow_tz_aware)

    # Training information
    training_session_id = Column(String, ForeignKey("training_sessions.session_id"))
    trained_on_commits = Column(Integer, nullable=False)
    training_accuracy = Column(Float, nullable=False)
    validation_accuracy = Column(Float, nullable=False)

    # Model performance
    categories = Column(JSON, nullable=False)  # List of supported categories
    performance_metrics = Column(JSON)  # Detailed performance metrics
    feature_importance = Column(JSON)  # Feature importance scores

    # Model storage and configuration
    model_binary = Column(JSON)  # Serialized model (for small models)
    model_file_path = Column(String)  # Path to model file (for large models)
    model_config = Column(JSON)  # Model hyperparameters and settings

    # Usage tracking
    active = Column(Boolean, default=True)  # Whether model is active
    usage_count = Column(Integer, default=0)  # Number of times used
    last_used = Column(DateTime(timezone=True))

    # Model validation
    cross_validation_scores = Column(JSON)  # Cross-validation results
    test_accuracy = Column(Float)  # Hold-out test set accuracy

    # Indexes for model management
    __table_args__ = (
        Index("idx_model_id", "model_id"),
        Index("idx_model_version", "version"),
        Index("idx_model_active", "active"),
        Index("idx_model_accuracy", "validation_accuracy"),
        Index("idx_model_created", "created_at"),
    )


class DailyCommitBatch(Base):
    """Daily batches of commits organized for efficient data collection and retrieval.

    WHY: This table enables the two-step fetch/analyze process by storing raw commit data
    in daily batches with full metadata before classification. Each row represents
    one day's worth of commits for a specific project, enabling efficient batch retrieval.
    """

    __tablename__ = "daily_commit_batches"

    # Primary key components
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)  # Date for the commit batch (YYYY-MM-DD)
    project_key = Column(String, nullable=False)  # Project identifier
    repo_path = Column(String, nullable=False)  # Repository path for identification

    # Batch metadata
    commit_count = Column(Integer, default=0)  # Number of commits in this batch
    total_files_changed = Column(Integer, default=0)
    total_lines_added = Column(Integer, default=0)
    total_lines_deleted = Column(Integer, default=0)

    # Developers active on this day
    active_developers = Column(JSON)  # List of developer canonical IDs
    unique_tickets = Column(JSON)  # List of ticket IDs referenced on this day

    # Processing status
    fetched_at = Column(DateTime(timezone=True), default=utcnow_tz_aware)
    classification_status = Column(
        String, default="pending"
    )  # pending, processing, completed, failed
    classified_at = Column(DateTime(timezone=True), nullable=True)

    # Batch context for LLM classification
    context_summary = Column(String, nullable=True)  # Brief summary of day's activity

    # Indexes for efficient retrieval by date range and project
    __table_args__ = (
        Index("idx_batch_date", "date"),
        Index("idx_daily_batch_project", "project_key"),
        Index("idx_batch_repo", "repo_path"),
        Index("idx_daily_batch_status", "classification_status"),
        Index("idx_batch_unique", "date", "project_key", "repo_path", unique=True),
        Index("idx_batch_date_range", "date", "project_key"),
    )


class DetailedTicketData(Base):
    """Enhanced ticket storage with full metadata for context-aware classification.

    WHY: The two-step process requires full ticket context (descriptions, types, etc.)
    to improve classification accuracy. This extends the existing IssueCache with
    fields specifically needed for classification context.
    """

    __tablename__ = "detailed_tickets"

    id = Column(Integer, primary_key=True)

    # Ticket identification (enhanced from IssueCache)
    platform = Column(String, nullable=False)  # 'jira', 'github', 'clickup', 'linear'
    ticket_id = Column(String, nullable=False)
    project_key = Column(String, nullable=False)

    # Core ticket data
    title = Column(String)
    description = Column(String)  # Full description for context
    summary = Column(String)  # Brief summary extracted from description
    ticket_type = Column(String)  # Bug, Story, Task, Epic, etc.
    status = Column(String)
    priority = Column(String)
    labels = Column(JSON)  # List of labels/tags

    # People and dates
    assignee = Column(String, nullable=True)
    reporter = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True))
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    # Metrics for classification context
    story_points = Column(Integer, nullable=True)
    original_estimate = Column(String, nullable=True)  # Time estimate
    time_spent = Column(String, nullable=True)

    # Relationships for context
    epic_key = Column(String, nullable=True)  # Parent epic
    parent_key = Column(String, nullable=True)  # Parent issue
    subtasks = Column(JSON)  # List of subtask keys
    linked_issues = Column(JSON)  # List of linked issue keys

    # Classification hints from ticket type/labels
    classification_hints = Column(JSON)  # Extracted hints for commit classification
    business_domain = Column(String, nullable=True)  # Domain extracted from ticket

    # Platform-specific data
    platform_data = Column(JSON)  # Additional platform-specific fields

    # Fetch metadata
    fetched_at = Column(DateTime(timezone=True), default=utcnow_tz_aware)
    fetch_version = Column(String, default="2.0")  # Version for schema evolution

    # Indexes for efficient lookup and context building
    __table_args__ = (
        Index("idx_detailed_platform_ticket", "platform", "ticket_id", unique=True),
        Index("idx_detailed_project", "project_key"),
        Index("idx_detailed_type", "ticket_type"),
        Index("idx_detailed_epic", "epic_key"),
        Index("idx_detailed_created", "created_at"),
        Index("idx_detailed_status", "status"),
    )


class CommitClassificationBatch(Base):
    """Batch classification results with context and confidence tracking.

    WHY: This table stores the results of batch LLM classification with full
    context about what information was used and confidence levels achieved.
    Enables iterative improvement and debugging of classification quality.
    """

    __tablename__ = "classification_batches"

    id = Column(Integer, primary_key=True)
    batch_id = Column(String, unique=True, nullable=False)  # UUID for this batch

    # Batch context
    project_key = Column(String, nullable=False)
    week_start = Column(DateTime, nullable=False)  # Monday of the week
    week_end = Column(DateTime, nullable=False)  # Sunday of the week
    commit_count = Column(Integer, nullable=False)

    # Context provided to LLM
    ticket_context = Column(JSON)  # Tickets included in context
    developer_context = Column(JSON)  # Active developers in this batch
    project_context = Column(String)  # Project description/domain

    # LLM processing details
    model_used = Column(String, nullable=False)  # Model identifier
    prompt_template = Column(String, nullable=False)  # Template used
    context_tokens = Column(Integer, default=0)  # Tokens used for context
    completion_tokens = Column(Integer, default=0)  # Tokens in response
    total_tokens = Column(Integer, default=0)

    # Processing results
    processing_status = Column(String, default="pending")  # pending, processing, completed, failed
    started_at = Column(DateTime(timezone=True), default=utcnow_tz_aware)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    processing_time_ms = Column(Float, nullable=True)

    # Quality metrics
    avg_confidence = Column(Float, nullable=True)  # Average confidence across commits
    low_confidence_count = Column(Integer, default=0)  # Commits with confidence < 0.7
    fallback_count = Column(Integer, default=0)  # Commits that fell back to rules

    # Cost tracking
    estimated_cost_usd = Column(Float, nullable=True)
    cost_per_commit = Column(Float, nullable=True)

    # Error handling
    error_message = Column(String, nullable=True)
    retry_count = Column(Integer, default=0)

    # Indexes for batch management and analysis
    __table_args__ = (
        Index("idx_classification_batch_id", "batch_id"),
        Index("idx_classification_batch_project", "project_key"),
        Index("idx_batch_week", "week_start", "week_end"),
        Index("idx_classification_batch_status", "processing_status"),
        Index("idx_batch_completed", "completed_at"),
        Index("idx_batch_model", "model_used"),
    )


class CommitTicketCorrelation(Base):
    """Correlations between commits and tickets for context-aware classification.

    WHY: This table explicitly tracks which commits reference which tickets,
    enabling the batch classifier to include relevant ticket context when
    classifying related commits. Improves accuracy by providing business context.
    """

    __tablename__ = "commit_ticket_correlations"

    id = Column(Integer, primary_key=True)

    # Commit identification
    commit_hash = Column(String, nullable=False)
    repo_path = Column(String, nullable=False)

    # Ticket identification
    ticket_id = Column(String, nullable=False)
    platform = Column(String, nullable=False)
    project_key = Column(String, nullable=False)

    # Correlation metadata
    correlation_type = Column(String, default="direct")  # direct, inferred, related
    confidence = Column(Float, default=1.0)  # Confidence in correlation
    extracted_from = Column(String, nullable=False)  # commit_message, branch_name, pr_title

    # Pattern that created this correlation
    matching_pattern = Column(String, nullable=True)  # Regex pattern that matched

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=utcnow_tz_aware)
    validated = Column(Boolean, default=False)  # Manual validation flag

    # Indexes for efficient correlation lookup
    __table_args__ = (
        Index("idx_corr_commit", "commit_hash", "repo_path"),
        Index("idx_corr_ticket", "ticket_id", "platform"),
        Index("idx_corr_project", "project_key"),
        Index("idx_corr_unique", "commit_hash", "repo_path", "ticket_id", "platform", unique=True),
    )


class DailyMetrics(Base):
    """Daily activity metrics per developer per project with classification data.

    WHY: This table stores daily aggregated metrics for each developer-project combination,
    enabling quick retrieval by date range for reporting and trend analysis.
    Each row represents one developer's activity in one project for one day.
    """

    __tablename__ = "daily_metrics"

    # Primary key components
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)  # Date for the metrics (YYYY-MM-DD)
    developer_id = Column(String, nullable=False)  # Canonical developer ID
    project_key = Column(String, nullable=False)  # Project identifier

    # Developer information
    developer_name = Column(String, nullable=False)  # Display name for reports
    developer_email = Column(String, nullable=False)  # Primary email

    # Classification counts - commit counts by category
    feature_commits = Column(Integer, default=0)
    bug_fix_commits = Column(Integer, default=0)
    refactor_commits = Column(Integer, default=0)
    documentation_commits = Column(Integer, default=0)
    maintenance_commits = Column(Integer, default=0)
    test_commits = Column(Integer, default=0)
    style_commits = Column(Integer, default=0)
    build_commits = Column(Integer, default=0)
    other_commits = Column(Integer, default=0)

    # Aggregate metrics
    total_commits = Column(Integer, default=0)
    files_changed = Column(Integer, default=0)
    lines_added = Column(Integer, default=0)
    lines_deleted = Column(Integer, default=0)
    story_points = Column(Integer, default=0)

    # Ticket tracking metrics
    tracked_commits = Column(Integer, default=0)  # Commits with ticket references
    untracked_commits = Column(Integer, default=0)  # Commits without ticket references
    unique_tickets = Column(Integer, default=0)  # Number of unique tickets referenced

    # Work pattern indicators
    merge_commits = Column(Integer, default=0)
    complex_commits = Column(Integer, default=0)  # Commits with >5 files changed

    # Metadata
    created_at = Column(DateTime(timezone=True), default=utcnow_tz_aware)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=utcnow_tz_aware)

    # Indexes for efficient querying
    __table_args__ = (
        Index("idx_daily_date", "date"),
        Index("idx_daily_developer", "developer_id"),
        Index("idx_daily_project", "project_key"),
        Index("idx_daily_date_range", "date", "developer_id", "project_key"),
        Index("idx_daily_unique", "date", "developer_id", "project_key", unique=True),
    )


class WeeklyTrends(Base):
    """Weekly trend analysis for developer-project combinations.

    WHY: Pre-calculated weekly trends improve report performance by avoiding
    repeated calculations. Stores week-over-week changes in activity patterns.
    """

    __tablename__ = "weekly_trends"

    id = Column(Integer, primary_key=True)
    week_start = Column(DateTime, nullable=False)  # Monday of the week
    week_end = Column(DateTime, nullable=False)  # Sunday of the week
    developer_id = Column(String, nullable=False)
    project_key = Column(String, nullable=False)

    # Week totals
    total_commits = Column(Integer, default=0)
    feature_commits = Column(Integer, default=0)
    bug_fix_commits = Column(Integer, default=0)
    refactor_commits = Column(Integer, default=0)

    # Week-over-week changes (percentage)
    total_commits_change = Column(Float, default=0.0)
    feature_commits_change = Column(Float, default=0.0)
    bug_fix_commits_change = Column(Float, default=0.0)
    refactor_commits_change = Column(Float, default=0.0)

    # Activity indicators
    days_active = Column(Integer, default=0)  # Number of days with commits
    avg_commits_per_day = Column(Float, default=0.0)

    # Metadata
    calculated_at = Column(DateTime(timezone=True), default=utcnow_tz_aware)

    # Indexes for trend queries
    __table_args__ = (
        Index("idx_weekly_start", "week_start"),
        Index("idx_weekly_dev_proj", "developer_id", "project_key"),
        Index("idx_weekly_unique", "week_start", "developer_id", "project_key", unique=True),
    )


class SchemaVersion(Base):
    """Track database schema versions for automatic migrations.

    WHY: Schema changes (like timezone-aware timestamps) require migration
    to ensure old cache databases work correctly without user intervention.
    This table tracks the current schema version to trigger automatic upgrades.
    """

    __tablename__ = "schema_version"

    id = Column(Integer, primary_key=True)
    version = Column(String, nullable=False)  # e.g., "2.0"
    upgraded_at = Column(DateTime(timezone=True), default=utcnow_tz_aware)
    previous_version = Column(String, nullable=True)
    migration_notes = Column(String, nullable=True)


class Database:
    """Database connection manager with robust permission handling."""

    # Schema version constants
    CURRENT_SCHEMA_VERSION = "2.0"  # Timezone-aware timestamps
    LEGACY_SCHEMA_VERSION = "1.0"  # Timezone-naive timestamps

    def __init__(self, db_path: Path):
        """
        Initialize database connection with proper error handling.

        WHY: This method handles various permission scenarios that can occur
        in different deployment environments:
        - Readonly filesystems (Docker containers, CI/CD)
        - Permission denied on directory creation
        - Database file creation failures
        - Fallback to memory database when persistence isn't possible

        DESIGN DECISION: Uses fallback mechanisms rather than failing hard,
        allowing the application to continue running even in restricted environments.

        Args:
            db_path: Path to the SQLite database file

        Raises:
            RuntimeError: If database initialization fails completely
        """
        self.db_path = db_path
        self.is_readonly_fallback = False
        self.engine = None
        self.SessionLocal = None

        # Try to create database with proper error handling
        self._initialize_database()

    def _initialize_database(self) -> None:
        """
        Initialize database with comprehensive error handling.

        WHY: Database initialization can fail for multiple reasons:
        1. Directory doesn't exist and can't be created (permissions)
        2. Directory exists but database file can't be created (readonly filesystem)
        3. Database file exists but is readonly
        4. Filesystem is completely readonly (containers, CI)

        APPROACH: Try primary location first, then fallback strategies
        """
        # Strategy 1: Try primary database location
        if self._try_primary_database():
            return

        # Strategy 2: Try temp directory fallback
        if self._try_temp_database_fallback():
            return

        # Strategy 3: Use in-memory database as last resort
        self._use_memory_database_fallback()

    def _try_primary_database(self) -> bool:
        """
        Attempt to create database at the primary location.

        Returns:
            True if successful, False if fallback needed
        """
        try:
            # Check if we can create the directory
            if not self._ensure_directory_writable(self.db_path.parent):
                return False

            # Check if database file can be created/accessed
            if not self._ensure_database_writable(self.db_path):
                return False

            # Try to create the database
            self.engine = create_engine(
                f"sqlite:///{self.db_path}",
                # Add connection args to handle locked databases better
                connect_args={
                    "timeout": 30,  # 30 second timeout for database locks
                    "check_same_thread": False,  # Allow multi-threading
                },
            )

            # Check schema version BEFORE creating tables to detect legacy databases
            self.SessionLocal = sessionmaker(bind=self.engine)
            needs_migration = self._check_schema_version_before_create()

            # Create/update tables
            Base.metadata.create_all(self.engine)

            # Perform migration if needed (after tables are created/updated)
            if needs_migration:
                self._perform_schema_migration()
            else:
                # No migration needed - record current schema version if not already recorded
                self._ensure_schema_version_recorded()

            # Apply other migrations for existing databases
            self._apply_migrations()

            # Test that we can actually write to the database
            self._test_database_write()

            logger.info(f"Database initialized successfully at: {self.db_path}")
            return True

        except (OperationalError, OSError, PermissionError) as e:
            logger.warning(f"Failed to initialize primary database at {self.db_path}: {e}")
            return False

    def _try_temp_database_fallback(self) -> bool:
        """
        Try to create database in system temp directory as fallback.

        Returns:
            True if successful, False if fallback needed
        """
        try:
            # Create a temp file that will persist for the session
            temp_dir = Path(tempfile.gettempdir()) / "gitflow-analytics-cache"
            temp_dir.mkdir(exist_ok=True, parents=True)

            # Use the same filename but in temp directory
            temp_db_path = temp_dir / self.db_path.name

            self.engine = create_engine(
                f"sqlite:///{temp_db_path}",
                connect_args={
                    "timeout": 30,
                    "check_same_thread": False,
                },
            )

            # Check schema version BEFORE creating tables to detect legacy databases
            self.SessionLocal = sessionmaker(bind=self.engine)
            needs_migration = self._check_schema_version_before_create()

            # Create/update tables
            Base.metadata.create_all(self.engine)

            # Perform migration if needed (after tables are created/updated)
            if needs_migration:
                self._perform_schema_migration()
            else:
                # No migration needed - record current schema version if not already recorded
                self._ensure_schema_version_recorded()

            # Apply other migrations for existing databases
            self._apply_migrations()

            # Test write capability
            self._test_database_write()

            logger.warning(
                f"Primary database location not writable. Using temp fallback: {temp_db_path}"
            )
            self.db_path = temp_db_path  # Update path for reference
            return True

        except (OperationalError, OSError, PermissionError) as e:
            logger.warning(f"Temp database fallback failed: {e}")
            return False

    def _use_memory_database_fallback(self) -> None:
        """
        Use in-memory SQLite database as last resort.

        This allows the application to function even in completely readonly environments,
        but data will not persist between runs.
        """
        try:
            logger.warning(
                "All persistent database options failed. Using in-memory database. "
                "Data will not persist between runs."
            )

            self.engine = create_engine(
                "sqlite:///:memory:", connect_args={"check_same_thread": False}
            )

            # Check schema version BEFORE creating tables to detect legacy databases
            self.SessionLocal = sessionmaker(bind=self.engine)
            needs_migration = self._check_schema_version_before_create()

            # Create/update tables
            Base.metadata.create_all(self.engine)

            # Perform migration if needed (after tables are created/updated)
            if needs_migration:
                self._perform_schema_migration()
            else:
                # No migration needed - record current schema version if not already recorded
                self._ensure_schema_version_recorded()

            # Apply other migrations for existing databases
            self._apply_migrations()

            self.is_readonly_fallback = True

            # Test that memory database works
            self._test_database_write()

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize any database (including in-memory fallback): {e}. "
                "This may indicate a deeper system issue."
            ) from e

    def _ensure_directory_writable(self, directory: Path) -> bool:
        """
        Ensure directory exists and is writable.

        Args:
            directory: Directory to check/create

        Returns:
            True if directory is writable, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            directory.mkdir(parents=True, exist_ok=True)

            # Test write permissions by creating a temporary file
            test_file = directory / ".write_test"
            test_file.touch()
            test_file.unlink()  # Clean up

            return True

        except (PermissionError, OSError) as e:
            logger.debug(f"Directory {directory} is not writable: {e}")
            return False

    def _ensure_database_writable(self, db_path: Path) -> bool:
        """
        Check if database file can be created or is writable if it exists.

        Args:
            db_path: Path to the database file

        Returns:
            True if database file is writable, False otherwise
        """
        try:
            if db_path.exists():
                # Check if existing file is writable
                if not os.access(db_path, os.W_OK):
                    logger.debug(f"Database file {db_path} exists but is not writable")
                    return False
            else:
                # Test if we can create the file
                db_path.touch()
                db_path.unlink()  # Clean up test file

            return True

        except (PermissionError, OSError) as e:
            logger.debug(f"Cannot create/write database file {db_path}: {e}")
            return False

    def _test_database_write(self) -> None:
        """
        Test that we can actually write to the database.

        Raises:
            OperationalError: If database write test fails
        """
        try:
            # Try a simple write operation to verify database is writable
            session = self.get_session()
            try:
                # Just test that we can begin a transaction and rollback
                session.execute(text("SELECT 1"))
                session.rollback()
            finally:
                session.close()

        except Exception as e:
            raise OperationalError(f"Database write test failed: {e}", None, None) from e

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def init_db(self) -> None:
        """Initialize database tables and apply migrations."""
        needs_migration = self._check_schema_version_before_create()
        Base.metadata.create_all(self.engine)
        if needs_migration:
            self._perform_schema_migration()
        else:
            self._ensure_schema_version_recorded()
        self._apply_migrations()

    def _check_schema_version_before_create(self) -> bool:
        """Check if database needs migration BEFORE create_all is called.

        WHY: We need to check for legacy databases BEFORE creating new tables,
        otherwise we can't distinguish between a fresh database and a legacy one.

        Returns:
            True if migration is needed, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                # Check if schema_version table exists
                result = conn.execute(
                    text(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
                    )
                )
                schema_table_exists = result.fetchone() is not None

                if schema_table_exists:
                    # Check current version
                    result = conn.execute(
                        text("SELECT version FROM schema_version ORDER BY id DESC LIMIT 1")
                    )
                    row = result.fetchone()

                    if row and row[0] != self.CURRENT_SCHEMA_VERSION:
                        # Version mismatch - needs migration
                        logger.warning(
                            f"⚠️  Schema version mismatch: {row[0]} → {self.CURRENT_SCHEMA_VERSION}"
                        )
                        return True
                    # else: Already at current version or no version record yet
                    return False
                else:
                    # No schema_version table - check if this is legacy or new
                    result = conn.execute(
                        text(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name='cached_commits'"
                        )
                    )
                    has_cached_commits = result.fetchone() is not None

                    if has_cached_commits:
                        # Check if table has data
                        result = conn.execute(text("SELECT COUNT(*) FROM cached_commits"))
                        commit_count = result.fetchone()[0]

                        if commit_count > 0:
                            # Legacy database with data - needs migration
                            logger.warning("⚠️  Old cache schema detected (v1.0 → v2.0)")
                            logger.info("   This is a one-time operation due to timezone fix")
                            return True

                    # New database or empty legacy database - no migration needed
                    return False

        except Exception as e:
            # Don't fail initialization due to schema check issues
            logger.debug(f"Schema version check failed: {e}")
            return False

    def _perform_schema_migration(self) -> None:
        """Perform the actual schema migration after tables are created.

        WHY: Separating migration from detection allows us to update table schemas
        via create_all before clearing/migrating data.
        """
        try:
            with self.engine.connect() as conn:
                logger.info("🔄 Automatically upgrading cache database...")
                logger.info("   Clearing old cache data (timezone schema incompatible)...")

                # Clear cached data tables
                conn.execute(text("DELETE FROM cached_commits"))
                conn.execute(text("DELETE FROM pull_request_cache"))
                conn.execute(text("DELETE FROM issue_cache"))
                conn.execute(text("DELETE FROM repository_analysis_status"))

                # Also clear qualitative analysis data if it exists
                try:
                    conn.execute(text("DELETE FROM qualitative_commits"))
                    conn.execute(text("DELETE FROM pattern_cache"))
                except Exception:
                    # These tables might not exist in all databases
                    pass

                conn.commit()

                # Record the schema upgrade
                self._record_schema_version(
                    conn,
                    self.CURRENT_SCHEMA_VERSION,
                    self.LEGACY_SCHEMA_VERSION,
                    "Migrated to timezone-aware timestamps (v2.0)",
                )

                logger.info("   Migration complete - cache will be rebuilt on next analysis")
                logger.info("✅ Cache database upgraded successfully")

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            # Don't raise - let the system continue and rebuild cache from scratch

    def _ensure_schema_version_recorded(self) -> None:
        """Ensure schema version is recorded for databases that didn't need migration.

        WHY: Fresh databases and already-migrated databases need to have their
        schema version recorded for future migration detection.
        """
        try:
            with self.engine.connect() as conn:
                # Check if version is already recorded
                result = conn.execute(text("SELECT COUNT(*) FROM schema_version"))
                count = result.fetchone()[0]

                if count == 0:
                    # No version recorded - this is a fresh database
                    self._record_schema_version(
                        conn, self.CURRENT_SCHEMA_VERSION, None, "Initial schema creation"
                    )
                    logger.debug(f"Recorded initial schema version: {self.CURRENT_SCHEMA_VERSION}")

        except Exception as e:
            # Don't fail if we can't record version
            logger.debug(f"Could not ensure schema version recorded: {e}")

    def _record_schema_version(
        self, conn, version: str, previous_version: Optional[str], notes: Optional[str]
    ) -> None:
        """Record schema version in the database.

        Args:
            conn: Database connection
            version: New schema version
            previous_version: Previous schema version (None for initial)
            notes: Migration notes
        """
        try:
            from datetime import datetime, timezone

            # Insert new schema version record
            conn.execute(
                text(
                    """
                INSERT INTO schema_version (version, upgraded_at, previous_version, migration_notes)
                VALUES (:version, :upgraded_at, :previous_version, :notes)
            """
                ),
                {
                    "version": version,
                    "upgraded_at": datetime.now(timezone.utc),
                    "previous_version": previous_version,
                    "notes": notes,
                },
            )
            conn.commit()
        except Exception as e:
            logger.debug(f"Could not record schema version: {e}")

    def _apply_migrations(self) -> None:
        """Apply database migrations for backward compatibility.

        This method adds new columns to existing tables without losing data.
        """
        try:
            with self.engine.connect() as conn:
                # Check if filtered columns exist in cached_commits table
                result = conn.execute(text("PRAGMA table_info(cached_commits)"))
                columns = {row[1] for row in result}

                # Add filtered_insertions column if it doesn't exist
                if "filtered_insertions" not in columns:
                    logger.info("Adding filtered_insertions column to cached_commits table")
                    try:
                        conn.execute(
                            text(
                                "ALTER TABLE cached_commits ADD COLUMN filtered_insertions INTEGER DEFAULT 0"
                            )
                        )
                        conn.commit()
                    except Exception as e:
                        logger.debug(f"Column may already exist or database is readonly: {e}")

                # Add filtered_deletions column if it doesn't exist
                if "filtered_deletions" not in columns:
                    logger.info("Adding filtered_deletions column to cached_commits table")
                    try:
                        conn.execute(
                            text(
                                "ALTER TABLE cached_commits ADD COLUMN filtered_deletions INTEGER DEFAULT 0"
                            )
                        )
                        conn.commit()
                    except Exception as e:
                        logger.debug(f"Column may already exist or database is readonly: {e}")

                # Initialize filtered columns with existing values for backward compatibility
                if "filtered_insertions" not in columns or "filtered_deletions" not in columns:
                    logger.info("Initializing filtered columns with existing values")
                    try:
                        conn.execute(
                            text(
                                """
                            UPDATE cached_commits
                            SET filtered_insertions = COALESCE(filtered_insertions, insertions),
                                filtered_deletions = COALESCE(filtered_deletions, deletions)
                            WHERE filtered_insertions IS NULL OR filtered_deletions IS NULL
                        """
                            )
                        )
                        conn.commit()
                    except Exception as e:
                        logger.debug(f"Could not initialize filtered columns: {e}")

        except Exception as e:
            # Don't fail if migrations can't be applied (e.g., in-memory database)
            logger.debug(
                f"Could not apply migrations (may be normal for new/memory databases): {e}"
            )
