"""Unified data models for platform-agnostic project management integration.

This module defines standardized data structures that normalize data from
different PM platforms (JIRA, Azure DevOps, Linear, etc.) into a common format
for consistent analytics across GitFlow Analytics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class IssueType(Enum):
    """Standardized issue types across platforms.

    Maps platform-specific issue types to unified categories for consistent
    analytics and reporting across different PM tools.
    """

    EPIC = "epic"
    STORY = "story"
    TASK = "task"
    BUG = "bug"
    DEFECT = "defect"
    FEATURE = "feature"
    IMPROVEMENT = "improvement"
    SUBTASK = "subtask"
    INCIDENT = "incident"
    UNKNOWN = "unknown"


class IssueStatus(Enum):
    """Standardized issue statuses across platforms.

    Maps platform-specific workflow states to unified status categories
    for consistent progress tracking and DORA metrics calculation.
    """

    BACKLOG = "backlog"
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    TESTING = "testing"
    DONE = "done"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class Priority(Enum):
    """Standardized priority levels across platforms.

    Maps platform-specific priority schemes to unified levels for
    consistent prioritization analysis and workload assessment.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TRIVIAL = "trivial"
    UNKNOWN = "unknown"


@dataclass
class UnifiedUser:
    """Platform-agnostic user representation.

    Normalizes user information from different PM platforms to enable
    consistent identity resolution and correlation with Git commit authors.

    WHY: Different platforms have varying user data structures, but we need
    consistent user identification for accurate attribution analytics.
    """

    id: str  # Platform-specific unique identifier
    email: Optional[str] = None
    display_name: Optional[str] = None
    username: Optional[str] = None
    platform: str = ""
    is_active: bool = True
    platform_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedProject:
    """Platform-agnostic project representation.

    Standardizes project information across PM platforms for consistent
    project-level analytics and repository correlation.

    WHY: Projects are the primary organizational unit in most PM tools,
    and we need consistent project identification for cross-platform analytics.
    """

    id: str  # Platform-specific unique identifier
    key: str  # Short identifier (e.g., "PROJ", used in issue keys)
    name: str
    description: Optional[str] = None
    platform: str = ""
    is_active: bool = True
    created_date: Optional[datetime] = None
    platform_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedIssue:
    """Platform-agnostic issue representation.

    The core data structure for PM platform integration. Normalizes issues,
    tickets, work items, and tasks from different platforms into a unified
    format for consistent analytics and Git correlation.

    DESIGN DECISION: Extensive field set to support most PM platforms while
    maintaining compatibility. Platform-specific data preserved in platform_data
    for advanced use cases without breaking the unified interface.

    WHY: Issues are the primary work tracking unit across all PM platforms.
    Unified representation enables consistent story point tracking, velocity
    calculation, and Git commit correlation regardless of underlying platform.
    """

    # Core identification - required for all issues
    id: str  # Platform-specific unique identifier
    key: str  # Human-readable key (e.g., "PROJ-123", "GH-456")
    platform: str  # Source platform identifier
    project_id: str  # Parent project identifier

    # Basic properties - common across most platforms
    title: str
    created_date: datetime  # Required field, moved before optional ones
    updated_date: datetime  # Required field, moved before optional ones

    # Optional basic properties
    description: Optional[str] = None
    issue_type: IssueType = IssueType.UNKNOWN
    status: IssueStatus = IssueStatus.UNKNOWN
    priority: Priority = Priority.UNKNOWN

    # People - for identity resolution and ownership tracking
    assignee: Optional[UnifiedUser] = None
    reporter: Optional[UnifiedUser] = None

    # Optional dates - critical for timeline analysis and DORA metrics
    resolved_date: Optional[datetime] = None
    due_date: Optional[datetime] = None

    # Estimation and tracking - for velocity and capacity planning
    story_points: Optional[int] = None
    original_estimate_hours: Optional[float] = None
    remaining_estimate_hours: Optional[float] = None
    time_spent_hours: Optional[float] = None

    # Relationships - for dependency analysis and epic breakdown
    parent_issue_key: Optional[str] = None
    subtasks: list[str] = field(default_factory=list)
    linked_issues: list[dict[str, str]] = field(
        default_factory=list
    )  # [{"key": "PROJ-456", "type": "blocks"}]

    # Sprint/iteration info - for agile metrics and sprint analysis
    sprint_id: Optional[str] = None
    sprint_name: Optional[str] = None

    # Labels and components - for categorization and filtering
    labels: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)

    # Platform-specific data - preserves original platform information
    platform_data: dict[str, Any] = field(default_factory=dict)

    # GitFlow Analytics integration - correlates with Git data
    linked_commits: list[str] = field(default_factory=list)
    linked_prs: list[str] = field(default_factory=list)


@dataclass
class UnifiedSprint:
    """Platform-agnostic sprint/iteration representation.

    Normalizes sprint, iteration, and milestone data from different platforms
    for consistent agile metrics and velocity tracking.

    WHY: Sprint data is essential for calculating velocity, planning accuracy,
    and team capacity metrics. Different platforms use different terminology
    (sprints, iterations, milestones) but represent similar concepts.
    """

    id: str  # Platform-specific unique identifier
    name: str
    project_id: str
    platform: str

    # Dates - for sprint timeline analysis
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # State - for active sprint identification
    is_active: bool = False
    is_completed: bool = False

    # Metrics - for velocity and planning analysis
    planned_story_points: Optional[int] = None
    completed_story_points: Optional[int] = None

    # Issues - for sprint content analysis
    issue_keys: list[str] = field(default_factory=list)

    # Platform-specific data - preserves original platform information
    platform_data: dict[str, Any] = field(default_factory=dict)
