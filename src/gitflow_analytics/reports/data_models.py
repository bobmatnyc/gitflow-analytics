"""Standardized data models for reports.

This module defines the data structures used throughout the report
generation system, ensuring consistency and type safety.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class CommitType(Enum):
    """Types of commits."""
    
    FEATURE = "feature"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    DOCUMENTATION = "documentation"
    TEST = "test"
    MAINTENANCE = "maintenance"
    STYLE = "style"
    BUILD = "build"
    OTHER = "other"


class WorkStyle(Enum):
    """Developer work style categories."""
    
    CONSISTENT = "consistent"
    BURST = "burst"
    IRREGULAR = "irregular"
    DECLINING = "declining"
    GROWING = "growing"


@dataclass
class DeveloperIdentity:
    """Developer identity information."""
    
    canonical_id: str
    primary_email: str
    primary_name: str
    aliases: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    is_bot: bool = False
    display_name: Optional[str] = None
    
    def get_display_name(self) -> str:
        """Get the display name for the developer."""
        return self.display_name or self.primary_name


@dataclass
class CommitData:
    """Standardized commit data structure."""
    
    hash: str
    author_email: str
    author_name: str
    timestamp: datetime
    message: str
    
    # Identity
    canonical_id: Optional[str] = None
    
    # Repository info
    project_key: Optional[str] = None
    repository: Optional[str] = None
    branch: Optional[str] = None
    
    # Metrics
    insertions: int = 0
    deletions: int = 0
    files_changed: int = 0
    
    # Filtered metrics (excluding certain files)
    filtered_insertions: Optional[int] = None
    filtered_deletions: Optional[int] = None
    
    # Classification
    commit_type: Optional[CommitType] = None
    is_merge: bool = False
    
    # Ticket references
    ticket_ids: List[str] = field(default_factory=list)
    has_ticket: bool = False
    
    # Story points
    story_points: Optional[float] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_line_changes(self, use_filtered: bool = True) -> tuple[int, int]:
        """Get line changes (insertions, deletions).
        
        Args:
            use_filtered: Whether to use filtered metrics if available
            
        Returns:
            Tuple of (insertions, deletions)
        """
        if use_filtered and self.filtered_insertions is not None:
            return (self.filtered_insertions, self.filtered_deletions or 0)
        return (self.insertions, self.deletions)
    
    def get_total_lines(self, use_filtered: bool = True) -> int:
        """Get total lines changed.
        
        Args:
            use_filtered: Whether to use filtered metrics if available
            
        Returns:
            Total lines changed
        """
        ins, dels = self.get_line_changes(use_filtered)
        return ins + dels


@dataclass
class PullRequestData:
    """Standardized pull request data structure."""
    
    id: Union[int, str]
    title: str
    author: str
    created_at: datetime
    merged_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    
    # State
    state: str = "open"  # open, closed, merged
    is_merged: bool = False
    
    # Repository info
    project_key: Optional[str] = None
    repository: Optional[str] = None
    base_branch: Optional[str] = None
    head_branch: Optional[str] = None
    
    # Metrics
    commits_count: int = 0
    additions: int = 0
    deletions: int = 0
    files_changed: int = 0
    comments_count: int = 0
    review_comments_count: int = 0
    
    # Review info
    reviewers: List[str] = field(default_factory=list)
    approved_by: List[str] = field(default_factory=list)
    
    # Time metrics
    time_to_merge_hours: Optional[float] = None
    time_to_first_review_hours: Optional[float] = None
    
    # Labels and metadata
    labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_cycle_time(self) -> Optional[float]:
        """Get cycle time in hours.
        
        Returns:
            Cycle time in hours if PR is merged, None otherwise
        """
        if self.is_merged and self.merged_at and self.created_at:
            delta = self.merged_at - self.created_at
            return delta.total_seconds() / 3600
        return None


@dataclass
class DeveloperMetrics:
    """Developer-level metrics."""
    
    developer: DeveloperIdentity
    
    # Activity metrics
    total_commits: int = 0
    total_prs: int = 0
    lines_added: int = 0
    lines_deleted: int = 0
    files_changed: int = 0
    
    # Time-based metrics
    active_days: int = 0
    first_commit: Optional[datetime] = None
    last_commit: Optional[datetime] = None
    
    # Quality metrics
    ticket_coverage_pct: float = 0.0
    review_participation_count: int = 0
    
    # Story points
    total_story_points: float = 0.0
    
    # Work patterns
    work_style: Optional[WorkStyle] = None
    primary_project: Optional[str] = None
    projects: Dict[str, float] = field(default_factory=dict)  # project -> percentage
    
    # Commit categorization
    commit_types: Dict[CommitType, int] = field(default_factory=dict)
    
    # Additional metrics
    velocity: float = 0.0  # commits per week
    impact_score: float = 0.0
    collaboration_score: float = 0.0
    
    def get_productivity_score(self) -> float:
        """Calculate overall productivity score.
        
        Returns:
            Productivity score (0-100)
        """
        # Simple scoring algorithm
        score = 0.0
        
        # Activity component (40%)
        activity_score = min(40, (self.total_commits / 10) * 4)
        score += activity_score
        
        # Quality component (30%)
        quality_score = self.ticket_coverage_pct * 0.3
        score += quality_score
        
        # Collaboration component (20%)
        collab_score = min(20, (self.review_participation_count / 5) * 20)
        score += collab_score
        
        # Consistency component (10%)
        if self.work_style == WorkStyle.CONSISTENT:
            score += 10
        elif self.work_style == WorkStyle.GROWING:
            score += 8
        elif self.work_style == WorkStyle.BURST:
            score += 5
        
        return min(100, score)


@dataclass
class ProjectMetrics:
    """Project-level metrics."""
    
    project_key: str
    project_name: Optional[str] = None
    
    # Activity metrics
    total_commits: int = 0
    total_prs: int = 0
    active_developers: int = 0
    lines_added: int = 0
    lines_deleted: int = 0
    
    # Time-based metrics
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    active_days: int = 0
    
    # Quality metrics
    ticket_coverage_pct: float = 0.0
    pr_merge_rate: float = 0.0
    avg_pr_cycle_time_hours: float = 0.0
    
    # Story points
    total_story_points: float = 0.0
    velocity: float = 0.0  # story points per week
    
    # Commit categorization
    commit_types: Dict[CommitType, int] = field(default_factory=dict)
    
    # Developer breakdown
    developer_contributions: Dict[str, float] = field(default_factory=dict)  # developer -> percentage
    
    # Health indicators
    health_score: float = 0.0
    risk_level: str = "low"  # low, medium, high
    
    def calculate_health_score(self) -> float:
        """Calculate project health score.
        
        Returns:
            Health score (0-100)
        """
        score = 0.0
        
        # Activity health (30%)
        if self.active_developers > 0:
            activity_score = min(30, (self.total_commits / self.active_developers / 10) * 3)
            score += activity_score
        
        # Quality health (40%)
        quality_score = self.ticket_coverage_pct * 0.4
        score += quality_score
        
        # Velocity health (30%)
        if self.velocity > 0:
            velocity_score = min(30, (self.velocity / 10) * 30)
            score += velocity_score
        
        self.health_score = min(100, score)
        
        # Determine risk level
        if score >= 80:
            self.risk_level = "low"
        elif score >= 50:
            self.risk_level = "medium"
        else:
            self.risk_level = "high"
        
        return self.health_score


@dataclass
class WeeklyMetrics:
    """Weekly aggregated metrics."""
    
    week_start: date
    week_end: date
    week_number: int
    year: int
    
    # Activity metrics
    commits: int = 0
    pull_requests: int = 0
    active_developers: int = 0
    lines_added: int = 0
    lines_deleted: int = 0
    
    # Quality metrics
    ticket_coverage_pct: float = 0.0
    bug_fix_ratio: float = 0.0
    
    # Story points
    story_points_completed: float = 0.0
    
    # DORA metrics
    deployment_frequency: int = 0
    lead_time_hours: Optional[float] = None
    mttr_hours: Optional[float] = None  # Mean time to recovery
    change_failure_rate: float = 0.0
    
    # Trends
    commit_trend: float = 0.0  # % change from previous week
    developer_trend: float = 0.0  # % change from previous week
    
    # Project breakdown
    project_activity: Dict[str, int] = field(default_factory=dict)  # project -> commits
    
    # Developer breakdown
    developer_activity: Dict[str, int] = field(default_factory=dict)  # developer -> commits


@dataclass
class TicketMetrics:
    """Ticket/issue tracking metrics."""
    
    platform: str  # jira, github, gitlab, etc.
    total_tickets: int = 0
    unique_tickets: int = 0
    
    # Coverage
    commits_with_tickets: int = 0
    commits_without_tickets: int = 0
    coverage_percentage: float = 0.0
    
    # Ticket types
    ticket_types: Dict[str, int] = field(default_factory=dict)
    
    # Per-developer coverage
    developer_coverage: Dict[str, float] = field(default_factory=dict)
    
    # Per-project coverage
    project_coverage: Dict[str, float] = field(default_factory=dict)
    
    # Untracked work analysis
    untracked_categories: Dict[str, int] = field(default_factory=dict)
    untracked_developers: Dict[str, int] = field(default_factory=dict)


@dataclass
class DORAMetrics:
    """DORA (DevOps Research and Assessment) metrics."""
    
    # Elite performance thresholds
    deployment_frequency: float = 0.0  # deployments per day
    lead_time_for_changes: float = 0.0  # hours
    time_to_restore_service: float = 0.0  # hours
    change_failure_rate: float = 0.0  # percentage
    
    # Performance level
    performance_level: str = "low"  # low, medium, high, elite
    
    # Breakdown by period
    weekly_metrics: List[WeeklyMetrics] = field(default_factory=list)
    
    def calculate_performance_level(self) -> str:
        """Calculate DORA performance level.
        
        Returns:
            Performance level (low, medium, high, elite)
        """
        score = 0
        
        # Deployment frequency scoring
        if self.deployment_frequency >= 1:  # Daily
            score += 4
        elif self.deployment_frequency >= 0.14:  # Weekly
            score += 3
        elif self.deployment_frequency >= 0.03:  # Monthly
            score += 2
        else:
            score += 1
        
        # Lead time scoring
        if self.lead_time_for_changes <= 24:  # Less than a day
            score += 4
        elif self.lead_time_for_changes <= 168:  # Less than a week
            score += 3
        elif self.lead_time_for_changes <= 720:  # Less than a month
            score += 2
        else:
            score += 1
        
        # MTTR scoring
        if self.time_to_restore_service <= 1:  # Less than an hour
            score += 4
        elif self.time_to_restore_service <= 24:  # Less than a day
            score += 3
        elif self.time_to_restore_service <= 168:  # Less than a week
            score += 2
        else:
            score += 1
        
        # Change failure rate scoring
        if self.change_failure_rate <= 5:  # 0-5%
            score += 4
        elif self.change_failure_rate <= 10:  # 6-10%
            score += 3
        elif self.change_failure_rate <= 15:  # 11-15%
            score += 2
        else:
            score += 1
        
        # Calculate performance level
        avg_score = score / 4
        if avg_score >= 3.5:
            self.performance_level = "elite"
        elif avg_score >= 2.5:
            self.performance_level = "high"
        elif avg_score >= 1.5:
            self.performance_level = "medium"
        else:
            self.performance_level = "low"
        
        return self.performance_level


@dataclass
class ReportSummary:
    """Summary data for comprehensive reports."""
    
    # Period information
    start_date: date
    end_date: date
    analysis_weeks: int
    
    # High-level metrics
    total_commits: int = 0
    total_pull_requests: int = 0
    total_developers: int = 0
    total_projects: int = 0
    total_lines_changed: int = 0
    
    # Quality metrics
    overall_ticket_coverage: float = 0.0
    overall_pr_merge_rate: float = 0.0
    
    # Story points
    total_story_points: float = 0.0
    average_velocity: float = 0.0
    
    # Top performers
    top_contributors: List[DeveloperIdentity] = field(default_factory=list)
    most_active_projects: List[str] = field(default_factory=list)
    
    # Health indicators
    overall_health_score: float = 0.0
    risk_projects: List[str] = field(default_factory=list)
    
    # DORA metrics
    dora_metrics: Optional[DORAMetrics] = None
    
    # Trends
    commit_trend: str = "stable"  # declining, stable, growing
    developer_trend: str = "stable"
    velocity_trend: str = "stable"
    
    # Key findings
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)