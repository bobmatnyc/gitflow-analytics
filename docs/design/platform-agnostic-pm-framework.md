# Platform-Agnostic Project Management Framework Design

**Document Version:** 1.0  
**Date:** August 1, 2025  
**Author:** Technical Product Owner  
**Target Implementation:** GitFlow Analytics v1.2+

## 1. Executive Summary

### 1.1 Purpose
Design a platform-agnostic project management integration framework that standardizes data collection across JIRA, Azure DevOps, Linear, Asana, GitHub Issues, ClickUp, and other PM platforms while maintaining extensibility for future integrations.

### 1.2 Design Principles
- **Adapter Pattern**: Each platform implements a standard interface
- **Normalized Data Model**: Common schema regardless of source platform
- **Plugin Architecture**: Easy addition of new platforms
- **Graceful Degradation**: Partial data collection when features unavailable
- **Configuration-Driven**: Platform selection and feature mapping via config

### 1.3 Strategic Benefits
- **Market Expansion**: Support organizations using any PM platform
- **Competitive Advantage**: Most tools are platform-specific
- **Future-Proof**: Easy addition of new platforms as they emerge
- **Data Consistency**: Unified analytics regardless of underlying tools

## 2. Architecture Overview

### 2.1 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GitFlow       │───▶│  PM Framework    │───▶│  Unified Data   │
│   Analytics     │    │  Orchestrator    │    │  Model          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        ▲
                                ▼                        │
                    ┌──────────────────┐                 │
                    │  Platform        │                 │
                    │  Registry        │                 │
                    └──────────────────┘                 │
                                │                        │
                ┌───────────────┼───────────────┐        │
                ▼               ▼               ▼        │
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │  JIRA Adapter   │ │ Azure Adapter   │ │ Linear Adapter  │
    │                 │ │                 │ │                 │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
                │               │               │
                ▼               ▼               ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │   JIRA API      │ │ Azure DevOps    │ │   Linear API    │
    │                 │ │     API         │ │                 │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 2.2 Component Responsibilities

#### **PM Framework Orchestrator**
- Manages platform adapter lifecycle
- Coordinates data collection across multiple platforms
- Handles data normalization and validation
- Provides unified API to GitFlow Analytics

#### **Platform Registry**
- Discovers available platform adapters
- Manages adapter configuration and capabilities
- Handles adapter versioning and compatibility

#### **Platform Adapters**
- Implement standardized interface
- Handle platform-specific authentication and API calls
- Map platform data to unified schema
- Manage platform-specific error handling and rate limiting

## 3. Unified Data Model

### 3.1 Core Entities

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union

class IssueType(Enum):
    """Standardized issue types across platforms."""
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
    """Standardized issue statuses."""
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
    """Standardized priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TRIVIAL = "trivial"
    UNKNOWN = "unknown"

@dataclass
class UnifiedUser:
    """Platform-agnostic user representation."""
    id: str  # Platform-specific ID
    email: Optional[str] = None
    display_name: Optional[str] = None
    username: Optional[str] = None
    platform: str = ""
    is_active: bool = True
    platform_data: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class UnifiedProject:
    """Platform-agnostic project representation."""
    id: str
    key: str  # Short identifier (e.g., "PROJ")
    name: str
    description: Optional[str] = None
    platform: str = ""
    is_active: bool = True
    created_date: Optional[datetime] = None
    platform_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UnifiedIssue:
    """Platform-agnostic issue representation."""
    # Core identification
    id: str
    key: str  # Human-readable key (e.g., "PROJ-123")
    platform: str
    project_id: str
    
    # Basic properties
    title: str
    description: Optional[str] = None
    issue_type: IssueType = IssueType.UNKNOWN
    status: IssueStatus = IssueStatus.UNKNOWN
    priority: Priority = Priority.UNKNOWN
    
    # People
    assignee: Optional[UnifiedUser] = None
    reporter: Optional[UnifiedUser] = None
    
    # Dates
    created_date: datetime
    updated_date: datetime
    resolved_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    
    # Estimation and tracking
    story_points: Optional[int] = None
    original_estimate_hours: Optional[float] = None
    remaining_estimate_hours: Optional[float] = None
    time_spent_hours: Optional[float] = None
    
    # Relationships
    parent_issue_key: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    linked_issues: List[Dict[str, str]] = field(default_factory=list)  # [{"key": "PROJ-456", "type": "blocks"}]
    
    # Sprint/iteration info
    sprint_id: Optional[str] = None
    sprint_name: Optional[str] = None
    
    # Labels and components
    labels: List[str] = field(default_factory=list)
    components: List[str] = field(default_factory=list)
    
    # Platform-specific data
    platform_data: Dict[str, Any] = field(default_factory=dict)
    
    # GitFlow Analytics integration
    linked_commits: List[str] = field(default_factory=list)
    linked_prs: List[str] = field(default_factory=list)

@dataclass
class UnifiedSprint:
    """Platform-agnostic sprint/iteration representation."""
    id: str
    name: str
    project_id: str
    platform: str
    
    # Dates
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # State
    is_active: bool = False
    is_completed: bool = False
    
    # Metrics
    planned_story_points: Optional[int] = None
    completed_story_points: Optional[int] = None
    
    # Issues
    issue_keys: List[str] = field(default_factory=list)
    
    # Platform-specific data
    platform_data: Dict[str, Any] = field(default_factory=dict)
```

## 4. Platform Adapter Interface

### 4.1 Base Adapter Interface

```python
class PlatformCapabilities:
    """Defines what capabilities a platform adapter supports."""
    
    def __init__(self):
        self.supports_projects = True
        self.supports_issues = True
        self.supports_sprints = False
        self.supports_time_tracking = False
        self.supports_story_points = False
        self.supports_custom_fields = False
        self.supports_issue_linking = False
        self.supports_comments = False
        self.supports_attachments = False
        self.supports_workflows = False
        self.supports_bulk_operations = False
        
        # Rate limiting info
        self.rate_limit_requests_per_hour = 1000
        self.rate_limit_burst_size = 100
        
        # Pagination info
        self.max_results_per_page = 100
        self.supports_cursor_pagination = False

class BasePlatformAdapter(ABC):
    """Abstract base class for all platform adapters."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.platform_name = self._get_platform_name()
        self.capabilities = self._get_capabilities()
        self._client = None
    
    @abstractmethod
    def _get_platform_name(self) -> str:
        """Return the platform name (e.g., 'jira', 'azure_devops')."""
        pass
    
    @abstractmethod
    def _get_capabilities(self) -> PlatformCapabilities:
        """Return the capabilities supported by this platform."""
        pass
    
    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the platform. Returns True if successful."""
        pass
    
    @abstractmethod
    def test_connection(self) -> Dict[str, Any]:
        """Test connection and return status info."""
        pass
    
    # Core data retrieval methods
    @abstractmethod
    def get_projects(self) -> List[UnifiedProject]:
        """Retrieve all accessible projects."""
        pass
    
    @abstractmethod
    def get_issues(self, project_id: str, since: Optional[datetime] = None,
                   issue_types: Optional[List[IssueType]] = None) -> List[UnifiedIssue]:
        """Retrieve issues for a project."""
        pass
    
    # Optional methods (default implementations that can be overridden)
    def get_sprints(self, project_id: str) -> List[UnifiedSprint]:
        """Retrieve sprints for a project. Default: empty list."""
        if not self.capabilities.supports_sprints:
            return []
        raise NotImplementedError
    
    def get_users(self, project_id: str) -> List[UnifiedUser]:
        """Retrieve users for a project. Default: empty list."""
        return []
    
    def get_issue_comments(self, issue_key: str) -> List[Dict[str, Any]]:
        """Retrieve comments for an issue. Default: empty list."""
        if not self.capabilities.supports_comments:
            return []
        raise NotImplementedError
    
    def get_custom_fields(self, project_id: str) -> Dict[str, Any]:
        """Retrieve custom field definitions. Default: empty dict."""
        if not self.capabilities.supports_custom_fields:
            return {}
        raise NotImplementedError
    
    # Utility methods
    def _normalize_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Normalize date string to datetime object."""
        if not date_str:
            return None
        
        # Handle common date formats
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO with microseconds
            "%Y-%m-%dT%H:%M:%SZ",     # ISO without microseconds
            "%Y-%m-%dT%H:%M:%S%z",    # ISO with timezone
            "%Y-%m-%d %H:%M:%S",      # Common SQL format
            "%Y-%m-%d",               # Date only
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        print(f"Warning: Could not parse date: {date_str}")
        return None
    
    def _map_priority(self, platform_priority: str) -> Priority:
        """Map platform-specific priority to unified priority."""
        priority_lower = platform_priority.lower() if platform_priority else ""
        
        priority_mapping = {
            'highest': Priority.CRITICAL,
            'critical': Priority.CRITICAL,
            'urgent': Priority.CRITICAL,
            'high': Priority.HIGH,
            'important': Priority.HIGH,
            'medium': Priority.MEDIUM,
            'normal': Priority.MEDIUM,
            'low': Priority.LOW,
            'minor': Priority.LOW,
            'trivial': Priority.TRIVIAL,
            'lowest': Priority.TRIVIAL,
        }
        
        return priority_mapping.get(priority_lower, Priority.UNKNOWN)
    
    def _extract_story_points(self, custom_fields: Dict[str, Any]) -> Optional[int]:
        """Extract story points from custom fields."""
        story_point_fields = [
            'story_points', 'storypoints', 'story_point_estimate',
            'customfield_10016', 'customfield_10021',  # Common JIRA fields
            'effort', 'size', 'complexity'
        ]
        
        for field_name in story_point_fields:
            if field_name in custom_fields:
                value = custom_fields[field_name]
                if isinstance(value, (int, float)):
                    return int(value)
                elif isinstance(value, str) and value.isdigit():
                    return int(value)
        
        return None
```

### 4.2 JIRA Adapter Implementation

```python
class JIRAAdapter(BasePlatformAdapter):
    """JIRA platform adapter implementation."""
    
    def _get_platform_name(self) -> str:
        return "jira"
    
    def _get_capabilities(self) -> PlatformCapabilities:
        capabilities = PlatformCapabilities()
        capabilities.supports_sprints = True
        capabilities.supports_time_tracking = True
        capabilities.supports_story_points = True
        capabilities.supports_custom_fields = True
        capabilities.supports_issue_linking = True
        capabilities.supports_comments = True
        capabilities.supports_attachments = True
        capabilities.supports_workflows = True
        capabilities.rate_limit_requests_per_hour = 3000  # Varies by plan
        capabilities.max_results_per_page = 1000
        return capabilities
    
    def authenticate(self) -> bool:
        """Authenticate with JIRA using API token."""
        try:
            import requests
            import base64
            
            self.base_url = self.config['base_url'].rstrip('/')
            credentials = base64.b64encode(
                f"{self.config['username']}:{self.config['api_token']}".encode()
            ).decode()
            
            self.headers = {
                "Authorization": f"Basic {credentials}",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            # Test authentication
            response = requests.get(
                f"{self.base_url}/rest/api/3/myself",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            print(f"JIRA authentication failed: {e}")
            return False
    
    def test_connection(self) -> Dict[str, Any]:
        """Test JIRA connection and return server info."""
        try:
            import requests
            
            response = requests.get(
                f"{self.base_url}/rest/api/3/serverInfo",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            server_info = response.json()
            
            return {
                'status': 'connected',
                'platform': 'jira',
                'server_version': server_info.get('version'),
                'server_title': server_info.get('serverTitle'),
                'base_url': self.base_url
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'platform': 'jira',
                'error': str(e)
            }
    
    def get_projects(self) -> List[UnifiedProject]:
        """Retrieve JIRA projects."""
        try:
            import requests
            
            response = requests.get(
                f"{self.base_url}/rest/api/3/project",
                headers=self.headers,
                params={'expand': 'description,lead,url'}
            )
            response.raise_for_status()
            jira_projects = response.json()
            
            projects = []
            for jira_project in jira_projects:
                project = UnifiedProject(
                    id=jira_project['id'],
                    key=jira_project['key'],
                    name=jira_project['name'],
                    description=jira_project.get('description'),
                    platform='jira',
                    platform_data={
                        'project_type': jira_project.get('projectTypeKey'),
                        'lead': jira_project.get('lead', {}).get('displayName'),
                        'url': jira_project.get('self')
                    }
                )
                projects.append(project)
            
            return projects
            
        except Exception as e:
            print(f"Failed to retrieve JIRA projects: {e}")
            return []
    
    def get_issues(self, project_id: str, since: Optional[datetime] = None,
                   issue_types: Optional[List[IssueType]] = None) -> List[UnifiedIssue]:
        """Retrieve JIRA issues for a project."""
        try:
            import requests
            
            # Build JQL query
            jql_parts = [f"project = {project_id}"]
            
            if since:
                since_str = since.strftime('%Y-%m-%d')
                jql_parts.append(f"updated >= '{since_str}'")
            
            if issue_types:
                # Map unified types to JIRA types
                jira_types = []
                type_mapping = {
                    IssueType.EPIC: ['Epic'],
                    IssueType.STORY: ['Story', 'User Story'],
                    IssueType.TASK: ['Task'],
                    IssueType.BUG: ['Bug', 'Defect'],
                    IssueType.IMPROVEMENT: ['Improvement', 'Enhancement'],
                    IssueType.SUBTASK: ['Sub-task', 'Subtask']
                }
                
                for issue_type in issue_types:
                    jira_types.extend(type_mapping.get(issue_type, []))
                
                if jira_types:
                    types_str = "', '".join(jira_types)
                    jql_parts.append(f"issueType in ('{types_str}')")
            
            jql = " AND ".join(jql_parts) + " ORDER BY updated DESC"
            
            # Get issues with pagination
            issues = []
            start_at = 0
            max_results = 100
            
            while True:
                response = requests.get(
                    f"{self.base_url}/rest/api/3/search",
                    headers=self.headers,
                    params={
                        'jql': jql,
                        'startAt': start_at,
                        'maxResults': max_results,
                        'fields': '*all',
                        'expand': 'changelog'
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                for jira_issue in data['issues']:
                    unified_issue = self._convert_jira_issue(jira_issue)
                    issues.append(unified_issue)
                
                # Check if we have more results
                if start_at + max_results >= data['total']:
                    break
                start_at += max_results
            
            return issues
            
        except Exception as e:
            print(f"Failed to retrieve JIRA issues: {e}")
            return []
    
    def _convert_jira_issue(self, jira_issue: Dict[str, Any]) -> UnifiedIssue:
        """Convert JIRA issue to unified format."""
        fields = jira_issue['fields']
        
        # Map issue type
        issue_type_name = fields.get('issuetype', {}).get('name', '').lower()
        issue_type_mapping = {
            'epic': IssueType.EPIC,
            'story': IssueType.STORY,
            'user story': IssueType.STORY,
            'task': IssueType.TASK,
            'bug': IssueType.BUG,
            'defect': IssueType.BUG,
            'improvement': IssueType.IMPROVEMENT,
            'enhancement': IssueType.IMPROVEMENT,
            'sub-task': IssueType.SUBTASK,
            'subtask': IssueType.SUBTASK
        }
        issue_type = issue_type_mapping.get(issue_type_name, IssueType.UNKNOWN)
        
        # Map status
        status_name = fields.get('status', {}).get('name', '').lower()
        status_mapping = {
            'to do': IssueStatus.TODO,
            'backlog': IssueStatus.BACKLOG,
            'selected for development': IssueStatus.TODO,
            'in progress': IssueStatus.IN_PROGRESS,
            'in review': IssueStatus.IN_REVIEW,
            'code review': IssueStatus.IN_REVIEW,
            'testing': IssueStatus.TESTING,
            'qa': IssueStatus.TESTING,
            'done': IssueStatus.DONE,
            'resolved': IssueStatus.DONE,
            'closed': IssueStatus.DONE,
            'cancelled': IssueStatus.CANCELLED,
            'blocked': IssueStatus.BLOCKED
        }
        status = status_mapping.get(status_name, IssueStatus.UNKNOWN)
        
        # Extract users
        assignee = None
        if fields.get('assignee'):
            assignee = UnifiedUser(
                id=fields['assignee']['accountId'],
                email=fields['assignee'].get('emailAddress'),
                display_name=fields['assignee']['displayName'],
                username=fields['assignee'].get('name'),
                platform='jira'
            )
        
        reporter = None
        if fields.get('reporter'):
            reporter = UnifiedUser(
                id=fields['reporter']['accountId'],
                email=fields['reporter'].get('emailAddress'),
                display_name=fields['reporter']['displayName'],
                username=fields['reporter'].get('name'),
                platform='jira'
            )
        
        # Extract story points
        story_points = self._extract_story_points(fields)
        
        # Extract sprint info
        sprint_id = None
        sprint_name = None
        if fields.get('customfield_10020'):  # Common sprint field
            sprints = fields['customfield_10020']
            if sprints and isinstance(sprints, list) and len(sprints) > 0:
                current_sprint = sprints[-1]  # Get latest sprint
                if isinstance(current_sprint, dict):
                    sprint_id = str(current_sprint.get('id', ''))
                    sprint_name = current_sprint.get('name', '')
        
        return UnifiedIssue(
            id=jira_issue['id'],
            key=jira_issue['key'],
            platform='jira',
            project_id=fields['project']['id'],
            title=fields['summary'],
            description=fields.get('description'),
            issue_type=issue_type,
            status=status,
            priority=self._map_priority(fields.get('priority', {}).get('name', '')),
            assignee=assignee,
            reporter=reporter,
            created_date=self._normalize_date(fields['created']),
            updated_date=self._normalize_date(fields['updated']),
            resolved_date=self._normalize_date(fields.get('resolved')),
            due_date=self._normalize_date(fields.get('duedate')),
            story_points=story_points,
            sprint_id=sprint_id,
            sprint_name=sprint_name,
            labels=[label for label in fields.get('labels', [])],
            components=[comp['name'] for comp in fields.get('components', [])],
            platform_data={
                'jira_issue_type': fields.get('issuetype', {}).get('name'),
                'jira_status': fields.get('status', {}).get('name'),
                'jira_priority': fields.get('priority', {}).get('name'),
                'environment': fields.get('environment'),
                'fix_versions': [v['name'] for v in fields.get('fixVersions', [])],
                'affects_versions': [v['name'] for v in fields.get('versions', [])],
                'custom_fields': {k: v for k, v in fields.items() if k.startswith('customfield_')}
            }
        )
```

### 4.3 Azure DevOps Adapter

```python
class AzureDevOpsAdapter(BasePlatformAdapter):
    """Azure DevOps platform adapter implementation."""
    
    def _get_platform_name(self) -> str:
        return "azure_devops"
    
    def _get_capabilities(self) -> PlatformCapabilities:
        capabilities = PlatformCapabilities()
        capabilities.supports_sprints = True
        capabilities.supports_time_tracking = True
        capabilities.supports_story_points = True
        capabilities.supports_custom_fields = True
        capabilities.supports_issue_linking = True
        capabilities.supports_comments = True
        capabilities.rate_limit_requests_per_hour = 6000
        capabilities.max_results_per_page = 200
        return capabilities
    
    def authenticate(self) -> bool:
        """Authenticate with Azure DevOps using PAT."""
        try:
            import requests
            import base64
            
            self.organization = self.config['organization']
            self.pat = self.config['personal_access_token']
            self.base_url = f"https://dev.azure.com/{self.organization}"
            
            # Azure DevOps uses PAT with empty username
            credentials = base64.b64encode(f":{self.pat}".encode()).decode()
            self.headers = {
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/json"
            }
            
            # Test authentication
            response = requests.get(
                f"{self.base_url}/_apis/projects?api-version=6.0",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            print(f"Azure DevOps authentication failed: {e}")
            return False
    
    def get_projects(self) -> List[UnifiedProject]:
        """Retrieve Azure DevOps projects."""
        try:
            import requests
            
            response = requests.get(
                f"{self.base_url}/_apis/projects?api-version=6.0",
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
            
            projects = []
            for ado_project in data['value']:
                project = UnifiedProject(
                    id=ado_project['id'],
                    key=ado_project['name'],  # Azure doesn't have separate keys
                    name=ado_project['name'],
                    description=ado_project.get('description'),
                    platform='azure_devops',
                    platform_data={
                        'state': ado_project.get('state'),
                        'visibility': ado_project.get('visibility'),
                        'url': ado_project.get('url')
                    }
                )
                projects.append(project)
            
            return projects
            
        except Exception as e:
            print(f"Failed to retrieve Azure DevOps projects: {e}")
            return []
    
    def get_issues(self, project_id: str, since: Optional[datetime] = None,
                   issue_types: Optional[List[IssueType]] = None) -> List[UnifiedIssue]:
        """Retrieve Azure DevOps work items."""
        try:
            import requests
            
            # Build WIQL query
            wiql_parts = [f"[System.TeamProject] = '{project_id}'"]
            
            if since:
                since_str = since.strftime('%Y-%m-%d')
                wiql_parts.append(f"[System.ChangedDate] >= '{since_str}'")
            
            if issue_types:
                # Map unified types to Azure work item types
                ado_types = []
                type_mapping = {
                    IssueType.EPIC: ['Epic'],
                    IssueType.STORY: ['User Story'],
                    IssueType.TASK: ['Task'],
                    IssueType.BUG: ['Bug'],
                    IssueType.FEATURE: ['Feature'],
                    IssueType.IMPROVEMENT: ['Product Backlog Item']
                }
                
                for issue_type in issue_types:
                    ado_types.extend(type_mapping.get(issue_type, []))
                
                if ado_types:
                    types_str = "', '".join(ado_types)
                    wiql_parts.append(f"[System.WorkItemType] IN ('{types_str}')")
            
            wiql = f"SELECT [System.Id] FROM WorkItems WHERE {' AND '.join(wiql_parts)} ORDER BY [System.ChangedDate] DESC"
            
            # Execute WIQL query
            wiql_response = requests.post(
                f"{self.base_url}/_apis/wit/wiql?api-version=6.0",
                headers=self.headers,
                json={"query": wiql}
            )
            wiql_response.raise_for_status()
            wiql_data = wiql_response.json()
            
            # Get work item IDs
            work_item_ids = [str(item['id']) for item in wiql_data['workItems']]
            
            if not work_item_ids:
                return []
            
            # Get full work item details in batches
            issues = []
            batch_size = 200  # Azure DevOps limit
            
            for i in range(0, len(work_item_ids), batch_size):
                batch_ids = work_item_ids[i:i + batch_size]
                ids_str = ','.join(batch_ids)
                
                response = requests.get(
                    f"{self.base_url}/_apis/wit/workitems?ids={ids_str}&$expand=all&api-version=6.0",
                    headers=self.headers
                )
                response.raise_for_status()
                data = response.json()
                
                for work_item in data['value']:
                    unified_issue = self._convert_ado_work_item(work_item)
                    issues.append(unified_issue)
            
            return issues
            
        except Exception as e:
            print(f"Failed to retrieve Azure DevOps work items: {e}")
            return []
    
    def _convert_ado_work_item(self, work_item: Dict[str, Any]) -> UnifiedIssue:
        """Convert Azure DevOps work item to unified format."""
        fields = work_item['fields']
        
        # Map work item type
        work_item_type = fields.get('System.WorkItemType', '').lower()
        type_mapping = {
            'epic': IssueType.EPIC,
            'user story': IssueType.STORY,
            'product backlog item': IssueType.STORY,
            'task': IssueType.TASK,
            'bug': IssueType.BUG,
            'feature': IssueType.FEATURE,
            'issue': IssueType.BUG
        }
        issue_type = type_mapping.get(work_item_type, IssueType.UNKNOWN)
        
        # Map state to status
        state = fields.get('System.State', '').lower()
        status_mapping = {
            'new': IssueStatus.TODO,
            'active': IssueStatus.IN_PROGRESS,
            'resolved': IssueStatus.DONE,
            'closed': IssueStatus.DONE,
            'removed': IssueStatus.CANCELLED,
            'committed': IssueStatus.TODO,
            'done': IssueStatus.DONE
        }
        status = status_mapping.get(state, IssueStatus.UNKNOWN)
        
        # Extract users
        assignee = None
        if fields.get('System.AssignedTo'):
            assignee_data = fields['System.AssignedTo']
            assignee = UnifiedUser(
                id=assignee_data.get('id', ''),
                email=assignee_data.get('uniqueName'),
                display_name=assignee_data.get('displayName'),
                platform='azure_devops'
            )
        
        reporter = None
        if fields.get('System.CreatedBy'):
            reporter_data = fields['System.CreatedBy']
            reporter = UnifiedUser(
                id=reporter_data.get('id', ''),
                email=reporter_data.get('uniqueName'),
                display_name=reporter_data.get('displayName'),
                platform='azure_devops'
            )
        
        # Extract story points (different field names in Azure)
        story_points = None
        story_point_fields = [
            'Microsoft.VSTS.Scheduling.StoryPoints',
            'Microsoft.VSTS.Scheduling.Effort',
            'Custom.StoryPoints'
        ]
        
        for field_name in story_point_fields:
            if field_name in fields and fields[field_name]:
                try:
                    story_points = int(float(fields[field_name]))
                    break
                except (ValueError, TypeError):
                    continue
        
        return UnifiedIssue(
            id=str(work_item['id']),
            key=f"{fields.get('System.TeamProject', 'PROJ')}-{work_item['id']}",
            platform='azure_devops',
            project_id=fields.get('System.TeamProject', ''),
            title=fields.get('System.Title', ''),
            description=fields.get('System.Description'),
            issue_type=issue_type,
            status=status,
            priority=self._map_priority(fields.get('Microsoft.VSTS.Common.Priority', '')),
            assignee=assignee,
            reporter=reporter,
            created_date=self._normalize_date(fields.get('System.CreatedDate')),
            updated_date=self._normalize_date(fields.get('System.ChangedDate')),
            resolved_date=self._normalize_date(fields.get('Microsoft.VSTS.Common.ResolvedDate')),
            story_points=story_points,
            labels=fields.get('System.Tags', '').split(';') if fields.get('System.Tags') else [],
            platform_data={
                'work_item_type': fields.get('System.WorkItemType'),
                'state': fields.get('System.State'),
                'reason': fields.get('System.Reason'),
                'area_path': fields.get('System.AreaPath'),
                'iteration_path': fields.get('System.IterationPath'),
                'custom_fields': {k: v for k, v in fields.items() if k.startswith('Custom.')}
            }
        )
```

## 5. Platform Registry and Orchestrator

### 5.1 Platform Registry

```python
class PlatformRegistry:
    """Registry for managing platform adapters."""
    
    def __init__(self):
        self._adapters: Dict[str, type] = {}
        self._instances: Dict[str, BasePlatformAdapter] = {}
        
        # Register built-in adapters
        self.register_adapter('jira', JIRAAdapter)
        self.register_adapter('azure_devops', AzureDevOpsAdapter)
        # Additional adapters will be registered here
    
    def register_adapter(self, platform_name: str, adapter_class: type):
        """Register a platform adapter class."""
        self._adapters[platform_name] = adapter_class
    
    def get_available_platforms(self) -> List[str]:
        """Get list of available platform names."""
        return list(self._adapters.keys())
    
    def create_adapter(self, platform_name: str, config: Dict[str, Any]) -> BasePlatformAdapter:
        """Create and configure a platform adapter instance."""
        if platform_name not in self._adapters:
            raise ValueError(f"Unknown platform: {platform_name}")
        
        adapter_class = self._adapters[platform_name]
        adapter = adapter_class(config)
        
        # Test authentication
        if not adapter.authenticate():
            raise ConnectionError(f"Failed to authenticate with {platform_name}")
        
        self._instances[platform_name] = adapter
        return adapter
    
    def get_adapter(self, platform_name: str) -> Optional[BasePlatformAdapter]:
        """Get existing adapter instance."""
        return self._instances.get(platform_name)

class PMFrameworkOrchestrator:
    """Orchestrates data collection across multiple PM platforms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = PlatformRegistry()
        self.adapters: Dict[str, BasePlatformAdapter] = {}
        
        # Initialize configured platforms
        self._initialize_platforms()
    
    def _initialize_platforms(self):
        """Initialize platform adapters based on configuration."""
        platforms_config = self.config.get('pm_platforms', {})
        
        for platform_name, platform_config in platforms_config.items():
            if platform_config.get('enabled', False):
                try:
                    adapter = self.registry.create_adapter(platform_name, platform_config)
                    self.adapters[platform_name] = adapter
                    print(f"✅ Initialized {platform_name} adapter")
                except Exception as e:
                    print(f"❌ Failed to initialize {platform_name}: {e}")
    
    def get_all_projects(self) -> Dict[str, List[UnifiedProject]]:
        """Get projects from all configured platforms."""
        all_projects = {}
        
        for platform_name, adapter in self.adapters.items():
            try:
                projects = adapter.get_projects()
                all_projects[platform_name] = projects
                print(f"📁 Found {len(projects)} projects in {platform_name}")
            except Exception as e:
                print(f"⚠️ Failed to get projects from {platform_name}: {e}")
                all_projects[platform_name] = []
        
        return all_projects
    
    def get_all_issues(self, since: Optional[datetime] = None,
                      project_filter: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[UnifiedIssue]]:
        """Get issues from all configured platforms."""
        all_issues = {}
        
        for platform_name, adapter in self.adapters.items():
            try:
                platform_issues = []
                
                # Get projects for this platform
                projects = adapter.get_projects()
                
                # Apply project filter if specified
                if project_filter and platform_name in project_filter:
                    project_keys = project_filter[platform_name]
                    projects = [p for p in projects if p.key in project_keys]
                
                # Get issues for each project
                for project in projects:
                    try:
                        issues = adapter.get_issues(project.key, since)
                        platform_issues.extend(issues)
                        print(f"🎫 Found {len(issues)} issues in {platform_name}/{project.key}")
                    except Exception as e:
                        print(f"⚠️ Failed to get issues from {platform_name}/{project.key}: {e}")
                
                all_issues[platform_name] = platform_issues
                
            except Exception as e:
                print(f"⚠️ Failed to get issues from {platform_name}: {e}")
                all_issues[platform_name] = []
        
        return all_issues
    
    def correlate_issues_with_commits(self, issues: Dict[str, List[UnifiedIssue]], 
                                    commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Correlate PM platform issues with Git commits."""
        correlations = []
        
        # Build a lookup of all issues by key
        issue_lookup = {}
        for platform_issues in issues.values():
            for issue in platform_issues:
                issue_lookup[issue.key] = issue
        
        # Find correlations
        for commit in commits:
            commit_correlations = []
            
            # Check existing ticket references
            ticket_refs = commit.get('ticket_references', [])
            for ref in ticket_refs:
                if isinstance(ref, dict):
                    ticket_key = ref.get('id', '')
                else:
                    ticket_key = str(ref)
                
                if ticket_key in issue_lookup:
                    issue = issue_lookup[ticket_key]
                    correlation = {
                        'commit_hash': commit['hash'],
                        'issue_key': issue.key,
                        'issue_title': issue.title,
                        'issue_type': issue.issue_type.value,
                        'platform': issue.platform,
                        'story_points': issue.story_points,
                        'correlation_method': 'ticket_reference'
                    }
                    commit_correlations.append(correlation)
            
            # TODO: Add fuzzy matching for issues mentioned in commit messages
            # TODO: Add temporal correlation for bug fixes
            
            if commit_correlations:
                correlations.extend(commit_correlations)
        
        return correlations
```

## 6. Configuration Schema

### 6.1 Enhanced Configuration

```yaml
# Enhanced GitFlow Analytics configuration with PM platform support
version: "1.2"

# GitHub configuration (unchanged)
github:
  token: "${GITHUB_TOKEN}"
  organization: "${GITHUB_ORG}"

# PM Platform configurations
pm_platforms:
  jira:
    enabled: true
    base_url: "${JIRA_BASE_URL}"
    username: "${JIRA_USERNAME}"
    api_token: "${JIRA_API_TOKEN}"
    
    # Platform-specific settings
    projects:
      - "PROJ"
      - "DEV"
    
    # Custom field mappings
    custom_field_mappings:
      story_points: "customfield_10016"
      sprint: "customfield_10020"
      epic_link: "customfield_10014"
    
    # Rate limiting
    rate_limit:
      requests_per_hour: 3000
      batch_size: 100
  
  azure_devops:
    enabled: false
    organization: "${AZURE_ORG}"
    personal_access_token: "${AZURE_PAT}"
    
    projects:
      - "MyProject"
    
    # Work item type mappings
    work_item_mappings:
      epic: ["Epic"]
      story: ["User Story", "Product Backlog Item"]
      task: ["Task"]
      bug: ["Bug", "Issue"]
  
  linear:
    enabled: false
    api_token: "${LINEAR_API_TOKEN}"
    team_ids:
      - "team_abc123"
  
  asana:
    enabled: false
    access_token: "${ASANA_ACCESS_TOKEN}"
    workspace_id: "${ASANA_WORKSPACE_ID}"

# Analysis configuration with PM integration
analysis:
  # Existing settings
  story_point_patterns:
    - "(?:story\\s*points?|sp|pts?)\\s*[:=]\\s*(\\d+)"
  
  # PM platform settings
  pm_integration:
    enabled: true
    primary_platform: "jira"  # Platform to prefer for conflicts
    
    # Issue correlation settings
    correlation:
      fuzzy_matching: true
      temporal_window_hours: 72  # Match issues to commits within 72 hours
      confidence_threshold: 0.8
    
    # Platform priorities (for conflict resolution)
    platform_priority:
      - "jira"
      - "azure_devops"
      - "linear"
      - "asana"
```

## 7. Integration with GitFlow Analytics

### 7.1 Enhanced Main Analysis Pipeline

```python
# Updated main analysis function
def analyze_with_pm_integration(config: Config, weeks: int = 12) -> Dict[str, Any]:
    """Enhanced analysis with PM platform integration."""
    
    # Existing Git analysis
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=weeks)
    
    # Initialize PM framework
    pm_orchestrator = PMFrameworkOrchestrator(config.dict())
    
    # Get Git data (existing flow)
    git_analyzer = GitAnalyzer(...)
    all_commits = []
    all_prs = []
    
    for repo_config in repositories:
        commits = git_analyzer.analyze_repository(repo_config.path, start_date)
        # ... existing logic
        all_commits.extend(commits)
    
    # Get PM platform data
    print("🎫 Collecting PM platform data...")
    pm_issues = pm_orchestrator.get_all_issues(since=start_date)
    
    # Correlate PM issues with Git data
    issue_correlations = pm_orchestrator.correlate_issues_with_commits(pm_issues, all_commits)
    
    # Enhanced analytics with PM data
    enhanced_metrics = calculate_enhanced_metrics(all_commits, all_prs, pm_issues, issue_correlations)
    
    return {
        'commits': all_commits,
        'prs': all_prs,
        'pm_issues': pm_issues,
        'correlations': issue_correlations,
        'enhanced_metrics': enhanced_metrics
    }

def calculate_enhanced_metrics(commits, prs, pm_issues, correlations):
    """Calculate metrics enhanced with PM platform data."""
    
    metrics = {}
    
    # Cross-platform issue metrics
    total_issues = sum(len(issues) for issues in pm_issues.values())
    metrics['total_pm_issues'] = total_issues
    
    # Story point accuracy across platforms
    pm_story_points = sum(
        issue.story_points or 0 
        for platform_issues in pm_issues.values() 
        for issue in platform_issues
    )
    git_story_points = sum(commit.get('story_points', 0) or 0 for commit in commits)
    
    metrics['story_point_correlation'] = {
        'pm_total': pm_story_points,
        'git_total': git_story_points,
        'correlation_accuracy': min(git_story_points / pm_story_points, 1.0) if pm_story_points > 0 else 0
    }
    
    # Issue type distribution
    issue_types = {}
    for platform_issues in pm_issues.values():
        for issue in platform_issues:
            issue_type = issue.issue_type.value
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
    
    metrics['issue_type_distribution'] = issue_types
    
    # Platform coverage
    platform_coverage = {}
    for platform, issues in pm_issues.items():
        linked_issues = [c['issue_key'] for c in correlations if c.get('platform') == platform]
        coverage = len(set(linked_issues)) / len(issues) if issues else 0
        platform_coverage[platform] = {
            'total_issues': len(issues),
            'linked_issues': len(set(linked_issues)),
            'coverage_percentage': coverage * 100
        }
    
    metrics['platform_coverage'] = platform_coverage
    
    return metrics
```

This platform-agnostic framework provides a solid foundation for integrating any PM platform while maintaining data consistency and extensibility. The adapter pattern ensures easy addition of new platforms, while the unified data model provides consistent analytics regardless of the underlying tools.

The key benefits are:
1. **Market Expansion**: Support any organization's PM toolchain
2. **Competitive Advantage**: Most analytics tools are platform-specific
3. **Data Consistency**: Unified metrics across different platforms
4. **Future-Proof**: Easy to add new platforms as they emerge
5. **Graceful Degradation**: Works with partial platform availability