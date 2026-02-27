"""Metrics, PR, issue, training, and CI/CD database models for GitFlow Analytics."""

from datetime import datetime

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
)

from .database_base import Base, utcnow_tz_aware


class PullRequestCache(Base):
    """Cached pull request data.

    Schema version history:
    - v1.0: Initial schema (title, description, author, dates, story_points, labels, commit_hashes)
    - v2.0: Timezone-aware timestamps
    - v3.0: Enhanced PR tracking (review counts, approvals, file stats, revision tracking)
    - v4.0: PR state tracking (pr_state, closed_at, is_merged)
    """

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

    # --- PR state fields (v4.0) ---

    # WHY: Storing the explicit state avoids recomputing it from merged_at on every
    # read.  "merged" / "closed" (rejected) / "open" maps directly to GitHub's PR
    # lifecycle, making rejection-rate reporting straightforward.
    pr_state = Column(String, nullable=True)  # "open", "closed", "merged"

    # WHY: closed_at is distinct from merged_at — it is populated for both merged
    # and closed-without-merge PRs so that we can compute PR lifetime accurately
    # for rejected PRs too, not just merged ones.
    closed_at = Column(DateTime(timezone=True), nullable=True)

    # WHY: An explicit boolean is cleaner than `merged_at IS NOT NULL` checks
    # scattered across reporting code and avoids ambiguity when merged_at is absent.
    is_merged = Column(Boolean, nullable=True)

    # --- Enhanced PR tracking fields (v3.0) ---

    # Comment counts
    # WHY: Separate review comments (inline code comments) from general PR comments
    # to allow distinct analysis of code review depth vs. general discussion volume.
    review_comments_count = Column(Integer, nullable=True, default=0)  # Inline review comments
    pr_comments_count = Column(Integer, nullable=True, default=0)  # General issue/PR comments

    # Approval tracking
    # WHY: Storing counts and reviewer lists allows computing approval rate metrics
    # and identifying which reviewers are most active without re-fetching the API.
    approvals_count = Column(Integer, nullable=True, default=0)
    change_requests_count = Column(Integer, nullable=True, default=0)
    reviewers = Column(JSON, nullable=True)  # List[str] of all reviewer logins
    approved_by = Column(JSON, nullable=True)  # List[str] of approving reviewer logins

    # Time-to-review
    # WHY: First-review latency is a key engineering health indicator. Storing it
    # avoids expensive recalculation from raw event timestamps on every report run.
    time_to_first_review_hours = Column(Float, nullable=True)

    # Revision tracking
    # WHY: Revision count (force-push / new commits after review) measures rework
    # which correlates with PR quality and review effectiveness.
    revision_count = Column(Integer, nullable=True, default=0)

    # File change stats
    # WHY: These are fetched from the API already (_extract_pr_data) but discarded
    # after the run. Persisting them enables PR-size analysis from cache alone.
    changed_files = Column(Integer, nullable=True, default=0)
    additions = Column(Integer, nullable=True, default=0)
    deletions = Column(Integer, nullable=True, default=0)

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

    # Complexity rating (1-5 scale, LLM-only — None for rule-based classifications)
    # 1: Trivial  2: Simple  3: Moderate  4: Complex  5: Highly complex
    complexity = Column(Integer, nullable=True)  # 1-5 sophistication rating

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


class CICDPipelineCache(Base):
    """Cache for CI/CD pipeline run data.

    This table stores build/pipeline execution data from various CI/CD platforms
    (GitHub Actions, GitLab CI, Jenkins, etc.) to minimize API calls and improve
    performance on subsequent analysis runs.

    WHY: CI/CD API calls can be slow and rate-limited. Caching build data with
    appropriate TTL allows for fast report generation while keeping data fresh.
    """

    __tablename__ = "cicd_pipelines"

    # Primary key
    id = Column(Integer, primary_key=True)

    # Pipeline identification
    platform = Column(String, nullable=False, index=True)  # "github_actions", "gitlab_ci", etc.
    pipeline_id = Column(String, nullable=False, index=True)  # Platform-specific pipeline ID
    workflow_name = Column(String)  # Workflow/pipeline name

    # Repository information
    repo_path = Column(String, nullable=False, index=True)  # "owner/repo"
    branch = Column(String, index=True)
    commit_sha = Column(String, index=True)

    # Pipeline results
    status = Column(String, index=True)  # "success", "failure", "cancelled", "pending"
    duration_seconds = Column(Integer)
    trigger_type = Column(String)  # "push", "pull_request", "schedule", "manual"
    created_at = Column(DateTime(timezone=True), nullable=False, index=True)

    # Additional metadata
    author = Column(String)  # Who triggered the pipeline
    url = Column(String)  # Link to pipeline in CI/CD platform

    # Platform-specific data (stored as JSON)
    platform_data = Column(JSON)  # Additional platform-specific fields

    # Cache metadata
    cached_at = Column(DateTime(timezone=True), default=utcnow_tz_aware, nullable=False)

    # Composite indexes for common query patterns
    __table_args__ = (
        # Most common: lookup by repo + date range
        Index("idx_cicd_repo_date", "repo_path", "created_at"),
        # Ensure uniqueness: platform + pipeline_id
        Index("idx_cicd_platform_pipeline", "platform", "pipeline_id", unique=True),
        # Lookup pipelines for specific commit
        Index("idx_cicd_commit", "commit_sha"),
        # Filter by status
        Index("idx_cicd_status", "status"),
        # Filter by branch
        Index("idx_cicd_branch", "branch"),
    )

    def __repr__(self) -> str:
        return (
            f"<CICDPipelineCache(platform='{self.platform}', "
            f"pipeline_id='{self.pipeline_id}', "
            f"status='{self.status}', "
            f"repo='{self.repo_path}')>"
        )
