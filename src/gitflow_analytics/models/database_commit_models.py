"""Commit-related database models for GitFlow Analytics."""

from datetime import datetime, timezone
from typing import Any

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
    UniqueConstraint,
)

from .database_base import Base, utcnow_tz_aware

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
    # Bug 2 fix: use DateTime(timezone=True) so SQLAlchemy stores/retrieves tz-aware
    # datetimes correctly. Without timezone=True, naive and aware comparisons mismatch.
    analysis_start = Column(DateTime(timezone=True), nullable=False)  # Start of analysis period
    analysis_end = Column(DateTime(timezone=True), nullable=False)  # End of analysis period
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
    # Bug 2 fix: use DateTime(timezone=True) and a tz-aware default callable.
    # The old default=datetime.utcnow produced naive datetimes that compare incorrectly
    # against timezone-aware query filters.
    last_updated = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=utcnow_tz_aware,
    )
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



class WeeklyFetchStatus(Base):
    """Track which Monday-aligned ISO weeks have been fetched per repository.

    WHY: Historical weeks never change once committed, so fetching them once is
    sufficient. This table enables week-granularity incremental fetching: on each
    run, only weeks absent from this table are fetched from Git, cutting repeat
    run times from minutes to seconds for large repository sets.

    Design decisions:
    - week_start is always Monday 00:00:00 UTC (ISO-week alignment)
    - week_end is always Sunday 23:59:59 UTC
    - fetch_timestamp records when the row was written for audit/debugging
    - commit_count is informational — it does NOT gate cache validity

    Uniqueness is on (repository_path, week_start) so a second fetch of the
    same week UPSERTS rather than duplicates.
    """

    __tablename__ = "weekly_fetch_status"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Repository identification
    repository_path = Column(String, nullable=False)

    # Week boundaries (always Monday-aligned UTC)
    week_start = Column(DateTime(timezone=True), nullable=False)  # Monday 00:00:00 UTC
    week_end = Column(DateTime(timezone=True), nullable=False)  # Sunday 23:59:59 UTC

    # Metadata recorded at fetch time
    commit_count = Column(Integer, default=0)  # Informational — commits found in this week
    fetch_timestamp = Column(
        DateTime(timezone=True), nullable=False, default=utcnow_tz_aware
    )  # When this week was fetched

    __table_args__ = (
        UniqueConstraint("repository_path", "week_start", name="uq_repo_week"),
        Index("idx_weekly_fetch_repo", "repository_path"),
        Index("idx_weekly_fetch_week_start", "week_start"),
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



