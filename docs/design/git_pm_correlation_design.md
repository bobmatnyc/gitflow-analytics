# GitFlow Analytics: Python Package Design

## Executive Summary

GitFlow Analytics is a Python package that analyzes GitHub repositories to generate comprehensive developer productivity reports. It extracts all necessary data directly from Git history and GitHub APIs, providing weekly summaries, productivity insights, and gap analysis without requiring external project management tools.

**Core Capabilities:**
- Multi-repository analysis with project grouping
- Developer identity resolution and normalization
- Work volume analysis (absolute vs relative effort)
- Story point extraction from commit messages and PR descriptions
- Gap detection for unassigned work
- DORA-inspired metrics with productivity insights
- Weekly CSV reports with narrative analysis

## Package Architecture

```
gitflow_analytics/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── git_analyzer.py      # Git repository analysis
│   ├── github_client.py     # GitHub API integration
│   ├── identity_resolver.py # Developer identity management
│   └── metrics_calculator.py # DORA and productivity metrics
├── analyzers/
│   ├── __init__.py
│   ├── commit_analyzer.py   # Commit pattern analysis
│   ├── story_point_extractor.py # SP from commit messages
│   ├── complexity_analyzer.py   # Code complexity changes
│   └── gap_detector.py      # Unassigned work detection
├── reports/
│   ├── __init__.py
│   ├── csv_generator.py     # Weekly CSV reports
│   ├── narrative_generator.py # Human-readable insights
│   └── templates/           # Report templates
├── utils/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── github_utils.py     # GitHub API utilities
│   └── date_utils.py       # Date/period utilities
└── cli.py                  # Command-line interface
```

## Core Components

### 1. Repository Analysis Engine

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
import re
from github import Github
from pydriller import Repository

@dataclass
class GitCommit:
    hash: str
    author_name: str
    author_email: str
    message: str
    timestamp: datetime
    files_changed: int
    insertions: int
    deletions: int
    complexity_delta: float
    story_points: Optional[int]
    ticket_references: List[str]
    repository: str
    project_key: str
    branch: str
    is_merge: bool

@dataclass
class PullRequest:
    number: int
    title: str
    description: str
    author: str
    created_at: datetime
    merged_at: Optional[datetime]
    story_points: Optional[int]
    commits: List[str]
    repository: str
    labels: List[str]

class GitAnalyzer:
    def __init__(self, github_token: str):
        self.github = Github(github_token)
        self.story_point_patterns = [
            r'(?:story\s*points?|sp|pts?)[\s:]*(\d+)',
            r'\[(\d+)\s*pts?\]',
            r'#(\d+)sp',
        ]
    
    def analyze_repositories(self, repo_configs: List[Dict], 
                           since: datetime) -> Dict[str, List[GitCommit]]:
        """Analyze multiple repositories and return commits grouped by project"""
        
        all_commits = {}
        
        for config in repo_configs:
            repo_path = config['path']
            project_key = config.get('project_key', self._extract_project_key(repo_path))
            
            # Analyze local repository
            local_commits = self._analyze_local_repo(repo_path, project_key, since)
            
            # Enrich with GitHub data
            if config.get('github_repo'):
                github_commits = self._enrich_with_github_data(
                    config['github_repo'], local_commits, since
                )
                all_commits[project_key] = github_commits
            else:
                all_commits[project_key] = local_commits
        
        return all_commits
    
    def _analyze_local_repo(self, repo_path: str, project_key: str, 
                           since: datetime) -> List[GitCommit]:
        """Analyze local Git repository"""
        commits = []
        
        for commit in Repository(repo_path, since=since).traverse_commits():
            # Extract story points from commit message
            story_points = self._extract_story_points(commit.msg)
            
            # Extract ticket references
            ticket_refs = self._extract_ticket_references(commit.msg)
            
            # Calculate complexity delta (simplified)
            complexity_delta = self._calculate_complexity_delta(commit)
            
            commits.append(GitCommit(
                hash=commit.hash,
                author_name=commit.author.name,
                author_email=commit.author.email,
                message=commit.msg,
                timestamp=commit.author_date,
                files_changed=len(commit.modified_files),
                insertions=commit.insertions,
                deletions=commit.deletions,
                complexity_delta=complexity_delta,
                story_points=story_points,
                ticket_references=ticket_refs,
                repository=repo_path,
                project_key=project_key,
                branch=self._get_branch_name(commit),
                is_merge=len(commit.parents) > 1
            ))
        
        return commits
    
    def _enrich_with_github_data(self, github_repo: str, commits: List[GitCommit], 
                                since: datetime) -> List[GitCommit]:
        """Enrich commits with GitHub PR and issue data"""
        
        repo = self.github.get_repo(github_repo)
        
        # Get pull requests for the period
        prs = repo.get_pulls(state='all', sort='updated', direction='desc')
        pr_commits = {}
        
        for pr in prs:
            if pr.updated_at < since:
                break
                
            # Extract story points from PR
            pr_story_points = self._extract_story_points(
                f"{pr.title} {pr.body or ''}"
            )
            
            # Map PR commits
            for pr_commit in pr.get_commits():
                pr_commits[pr_commit.sha] = {
                    'pr_number': pr.number,
                    'pr_story_points': pr_story_points,
                    'pr_title': pr.title,
                    'pr_labels': [label.name for label in pr.labels]
                }
        
        # Enrich commits with PR data
        enriched_commits = []
        for commit in commits:
            if commit.hash in pr_commits:
                pr_data = pr_commits[commit.hash]
                # Use PR story points if commit doesn't have them
                if not commit.story_points and pr_data['pr_story_points']:
                    commit.story_points = pr_data['pr_story_points']
            
            enriched_commits.append(commit)
        
        return enriched_commits
    
    def _extract_story_points(self, text: str) -> Optional[int]:
        """Extract story points from text using patterns"""
        for pattern in self.story_point_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        return None
    
    def _extract_ticket_references(self, message: str) -> List[str]:
        """Extract ticket references from commit message"""
        patterns = [
            r'([A-Z]+-\d+)',          # JIRA: PROJ-123
            r'#(\d+)',                # GitHub: #123
            r'(?:fix|close|resolve)s?\s+#(\d+)',  # GitHub closing
        ]
        
        references = []
        for pattern in patterns:
            references.extend(re.findall(pattern, message, re.IGNORECASE))
        
        return list(set(references))
    
    def _calculate_complexity_delta(self, commit) -> float:
        """Calculate complexity change (simplified cyclomatic complexity)"""
        total_delta = 0.0
        
        for modified_file in commit.modified_files:
            if not modified_file.filename.endswith(('.py', '.js', '.java', '.ts')):
                continue
                
            # Simple complexity estimation based on control structures
            before_complexity = self._estimate_complexity(
                modified_file.source_code_before or ""
            )
            after_complexity = self._estimate_complexity(
                modified_file.source_code or ""
            )
            
            total_delta += after_complexity - before_complexity
        
        return total_delta
    
    def _estimate_complexity(self, code: str) -> float:
        """Estimate code complexity by counting control structures"""
        if not code:
            return 0.0
            
        # Count control flow keywords
        control_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 
                           'except', 'switch', 'case', 'catch']
        
        complexity = 1  # Base complexity
        for keyword in control_keywords:
            complexity += code.lower().count(keyword)
        
        return float(complexity)
```

### 2. Developer Identity Resolution

```python
from collections import defaultdict
import difflib
from typing import Dict, List, Set, Tuple

@dataclass
class DeveloperProfile:
    canonical_id: str
    primary_name: str
    primary_email: str
    aliases: Set[Tuple[str, str]]  # (name, email) pairs
    github_username: Optional[str]
    total_commits: int = 0
    total_story_points: int = 0
    repositories: Set[str] = None
    
    def __post_init__(self):
        if self.repositories is None:
            self.repositories = set()

class DeveloperIdentityResolver:
    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold
        self.profiles: Dict[str, DeveloperProfile] = {}
        self.email_to_id: Dict[str, str] = {}
        self.name_variations: Dict[str, Set[str]] = defaultdict(set)
    
    def resolve_developer(self, name: str, email: str, 
                         github_username: Optional[str] = None) -> str:
        """Resolve developer identity and return canonical ID"""
        
        # Check exact email match first
        if email in self.email_to_id:
            profile_id = self.email_to_id[email]
            self._update_profile(profile_id, name, email, github_username)
            return profile_id
        
        # Find similar profiles
        best_match = self._find_best_match(name, email)
        
        if best_match and best_match[1] >= self.threshold:
            profile_id = best_match[0]
            self._add_alias(profile_id, name, email)
            return profile_id
        
        # Create new profile
        return self._create_profile(name, email, github_username)
    
    def _find_best_match(self, name: str, email: str) -> Optional[Tuple[str, float]]:
        """Find the best matching existing profile"""
        best_score = 0.0
        best_profile_id = None
        
        name_lower = name.lower().strip()
        email_domain = email.split('@')[1] if '@' in email else ''
        
        for profile_id, profile in self.profiles.items():
            score = 0.0
            
            # Name similarity
            primary_name_sim = difflib.SequenceMatcher(
                None, name_lower, profile.primary_name.lower()
            ).ratio()
            score += primary_name_sim * 0.4
            
            # Email domain similarity
            profile_domain = (profile.primary_email.split('@')[1] 
                            if '@' in profile.primary_email else '')
            if email_domain and email_domain == profile_domain:
                score += 0.3
            
            # Alias similarity
            best_alias_sim = 0.0
            for alias_name, alias_email in profile.aliases:
                alias_sim = difflib.SequenceMatcher(
                    None, name_lower, alias_name.lower()
                ).ratio()
                best_alias_sim = max(best_alias_sim, alias_sim)
            score += best_alias_sim * 0.3
            
            if score > best_score:
                best_score = score
                best_profile_id = profile_id
        
        return (best_profile_id, best_score) if best_profile_id else None
    
    def _create_profile(self, name: str, email: str, 
                       github_username: Optional[str] = None) -> str:
        """Create new developer profile"""
        import uuid
        profile_id = str(uuid.uuid4())
        
        profile = DeveloperProfile(
            canonical_id=profile_id,
            primary_name=name.strip(),
            primary_email=email.lower().strip(),
            aliases=set(),
            github_username=github_username,
            repositories=set()
        )
        
        self.profiles[profile_id] = profile
        self.email_to_id[email.lower().strip()] = profile_id
        self.name_variations[name.lower().strip()].add(profile_id)
        
        return profile_id
    
    def get_developer_summary(self) -> List[Dict]:
        """Get summary of all developers for reporting"""
        summaries = []
        
        for profile in self.profiles.values():
            summaries.append({
                'canonical_id': profile.canonical_id,
                'primary_name': profile.primary_name,
                'primary_email': profile.primary_email,
                'github_username': profile.github_username,
                'total_commits': profile.total_commits,
                'total_story_points': profile.total_story_points,
                'repositories': list(profile.repositories),
                'alias_count': len(profile.aliases)
            })
        
        return sorted(summaries, key=lambda x: x['total_commits'], reverse=True)
```

### 3. Productivity Metrics Calculator

```python
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

@dataclass
class WeeklyMetrics:
    week_start: datetime
    developer_id: str
    project_key: str
    commits: int
    story_points: int
    lines_added: int
    lines_removed: int
    files_changed: int
    complexity_added: float
    prs_created: int
    prs_merged: int
    ticket_coverage: float  # % of commits with ticket refs
    avg_commit_size: float
    work_distribution: Dict[str, int]  # file types worked on

@dataclass
class DORAMetrics:
    week_start: datetime
    project_key: str
    deployment_frequency: float
    lead_time_hours: float
    mttr_hours: float
    change_failure_rate: float

class ProductivityCalculator:
    def __init__(self):
        self.complexity_weights = {
            '.py': 1.0,
            '.js': 0.8,
            '.ts': 0.8,
            '.java': 1.2,
            '.cpp': 1.5,
            '.c': 1.5,
            '.go': 0.9,
            '.rs': 1.1,
            '.md': 0.2,
            '.json': 0.1,
            '.yaml': 0.3,
            '.yml': 0.3
        }
    
    def calculate_weekly_metrics(self, commits: Dict[str, List[GitCommit]], 
                               identity_resolver: DeveloperIdentityResolver,
                               weeks_back: int = 12) -> List[WeeklyMetrics]:
        """Calculate weekly productivity metrics for all developers"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=weeks_back)
        
        # Group commits by developer and week
        weekly_data = defaultdict(lambda: defaultdict(list))
        
        for project_key, project_commits in commits.items():
            for commit in project_commits:
                if commit.timestamp < start_date:
                    continue
                
                # Resolve developer identity
                dev_id = identity_resolver.resolve_developer(
                    commit.author_name, commit.author_email
                )
                
                # Calculate week start (Monday)
                week_start = self._get_week_start(commit.timestamp)
                
                weekly_data[(dev_id, project_key, week_start)].append(commit)
        
        # Calculate metrics for each developer/project/week
        metrics = []
        for (dev_id, project_key, week_start), week_commits in weekly_data.items():
            weekly_metric = self._calculate_week_metrics(
                dev_id, project_key, week_start, week_commits
            )
            metrics.append(weekly_metric)
        
        return sorted(metrics, key=lambda x: (x.week_start, x.developer_id))
    
    def _calculate_week_metrics(self, dev_id: str, project_key: str, 
                               week_start: datetime, 
                               commits: List[GitCommit]) -> WeeklyMetrics:
        """Calculate metrics for a specific developer/project/week"""
        
        total_story_points = sum(c.story_points or 0 for c in commits)
        total_lines_added = sum(c.insertions for c in commits)
        total_lines_removed = sum(c.deletions for c in commits)
        total_files_changed = sum(c.files_changed for c in commits)
        total_complexity = sum(c.complexity_delta for c in commits)
        
        # Ticket coverage
        commits_with_tickets = sum(1 for c in commits if c.ticket_references)
        ticket_coverage = commits_with_tickets / len(commits) if commits else 0
        
        # Average commit size
        avg_commit_size = (total_lines_added + total_lines_removed) / len(commits) if commits else 0
        
        # Work distribution by file type
        work_distribution = defaultdict(int)
        for commit in commits:
            # Estimate file types from changed files count
            # This is simplified - in reality you'd analyze file extensions
            work_distribution['code'] += commit.files_changed
        
        return WeeklyMetrics(
            week_start=week_start,
            developer_id=dev_id,
            project_key=project_key,
            commits=len(commits),
            story_points=total_story_points,
            lines_added=total_lines_added,
            lines_removed=total_lines_removed,
            files_changed=total_files_changed,
            complexity_added=total_complexity,
            prs_created=0,  # Would be calculated from GitHub data
            prs_merged=0,   # Would be calculated from GitHub data
            ticket_coverage=ticket_coverage,
            avg_commit_size=avg_commit_size,
            work_distribution=dict(work_distribution)
        )
    
    def calculate_normalized_productivity(self, metrics: List[WeeklyMetrics]) -> List[WeeklyMetrics]:
        """Normalize productivity metrics across developers and projects"""
        
        # Group by project for normalization
        project_groups = defaultdict(list)
        for metric in metrics:
            project_groups[metric.project_key].append(metric)
        
        normalized_metrics = []
        
        for project_key, project_metrics in project_groups.items():
            # Calculate baseline statistics for the project
            story_points = [m.story_points for m in project_metrics if m.story_points > 0]
            commit_sizes = [m.avg_commit_size for m in project_metrics]
            complexity_values = [m.complexity_added for m in project_metrics]
            
            # Calculate z-scores for normalization
            sp_mean, sp_std = np.mean(story_points), np.std(story_points) if story_points else (0, 1)
            size_mean, size_std = np.mean(commit_sizes), np.std(commit_sizes)
            complex_mean, complex_std = np.mean(complexity_values), np.std(complexity_values)
            
            # Apply normalization (stored in place - in real implementation, 
            # you'd create normalized fields)
            for metric in project_metrics:
                # Create normalized versions of metrics
                # This is simplified - full implementation would add normalized fields
                normalized_metrics.append(metric)
        
        return normalized_metrics
    
    def _get_week_start(self, date: datetime) -> datetime:
        """Get Monday of the week for a given date"""
        days_since_monday = date.weekday()
        monday = date - timedelta(days=days_since_monday)
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)
```

### 4. Gap Detection and Analysis

```python
class GapDetector:
    def __init__(self):
        self.issue_patterns = [
            r'(?:fix|bug|issue|problem|error)',
            r'(?:todo|fixme|hack)',
            r'(?:refactor|cleanup|debt)',
        ]
    
    def detect_gaps(self, commits: Dict[str, List[GitCommit]], 
                   weekly_metrics: List[WeeklyMetrics]) -> List[Dict]:
        """Detect various gaps in development process"""
        
        gaps = []
        
        # 1. Commits without ticket references
        gaps.extend(self._find_untracked_work(commits))
        
        # 2. High-complexity commits without story points
        gaps.extend(self._find_unestimated_complex_work(commits))
        
        # 3. Productivity outliers
        gaps.extend(self._find_productivity_outliers(weekly_metrics))
        
        # 4. Missing story point assignments
        gaps.extend(self._find_missing_story_points(commits))
        
        return gaps
    
    def _find_untracked_work(self, commits: Dict[str, List[GitCommit]]) -> List[Dict]:
        """Find commits without ticket references"""
        gaps = []
        
        for project_key, project_commits in commits.items():
            untracked_commits = [
                c for c in project_commits 
                if not c.ticket_references and not c.is_merge
            ]
            
            if untracked_commits:
                total_lines = sum(c.insertions + c.deletions for c in untracked_commits)
                
                gaps.append({
                    'type': 'untracked_work',
                    'project': project_key,
                    'severity': 'high' if total_lines > 1000 else 'medium',
                    'count': len(untracked_commits),
                    'total_lines_changed': total_lines,
                    'description': f"{len(untracked_commits)} commits without ticket references",
                    'recommendation': "Add ticket references to commit messages or link PRs to issues"
                })
        
        return gaps
    
    def _find_unestimated_complex_work(self, commits: Dict[str, List[GitCommit]]) -> List[Dict]:
        """Find high-complexity commits without story points"""
        gaps = []
        
        for project_key, project_commits in commits.items():
            complex_unestimated = [
                c for c in project_commits 
                if c.complexity_delta > 10 and not c.story_points
            ]
            
            if complex_unestimated:
                avg_complexity = np.mean([c.complexity_delta for c in complex_unestimated])
                
                gaps.append({
                    'type': 'unestimated_complex_work',
                    'project': project_key,
                    'severity': 'medium',
                    'count': len(complex_unestimated),
                    'avg_complexity': avg_complexity,
                    'description': f"{len(complex_unestimated)} high-complexity commits without story points",
                    'recommendation': "Ensure complex work is properly estimated before development"
                })
        
        return gaps
    
    def _find_productivity_outliers(self, weekly_metrics: List[WeeklyMetrics]) -> List[Dict]:
        """Find weeks with unusual productivity patterns"""
        gaps = []
        
        # Group by developer
        dev_metrics = defaultdict(list)
        for metric in weekly_metrics:
            dev_metrics[metric.developer_id].append(metric)
        
        for dev_id, metrics in dev_metrics.items():
            if len(metrics) < 4:  # Need at least 4 weeks for comparison
                continue
            
            # Calculate outliers in commit volume
            commit_counts = [m.commits for m in metrics]
            mean_commits = np.mean(commit_counts)
            std_commits = np.std(commit_counts)
            
            if std_commits > 0:
                outlier_weeks = [
                    m for m in metrics
                    if abs(m.commits - mean_commits) > 2 * std_commits
                ]
                
                if outlier_weeks:
                    gaps.append({
                        'type': 'productivity_outlier',
                        'developer_id': dev_id,
                        'severity': 'low',
                        'outlier_weeks': len(outlier_weeks),
                        'description': f"Developer has {len(outlier_weeks)} weeks with unusual commit patterns",
                        'recommendation': "Review workload distribution and potential blockers"
                    })
        
        return gaps
```

### 5. Report Generation

```python
import csv
from io import StringIO
from datetime import datetime
import pandas as pd

class ReportGenerator:
    def __init__(self):
        self.narrative_templates = {
            'summary': "Analysis of {total_developers} developers across {total_projects} projects for {weeks} weeks",
            'high_performer': "{developer} showed consistently high productivity with {avg_story_points} story points/week",
            'low_coverage': "Project {project} has {coverage}% ticket coverage - consider improving tracking",
            'unestimated_work': "{count} commits lack story point estimates, representing potential planning gaps"
        }
    
    def generate_weekly_csv(self, weekly_metrics: List[WeeklyMetrics], 
                           developer_resolver: DeveloperIdentityResolver,
                           output_path: str) -> str:
        """Generate CSV report with weekly metrics"""
        
        # Prepare data for CSV
        csv_data = []
        
        for metric in weekly_metrics:
            developer_profile = developer_resolver.profiles.get(metric.developer_id)
            developer_name = developer_profile.primary_name if developer_profile else "Unknown"
            
            csv_data.append({
                'week_start': metric.week_start.strftime('%Y-%m-%d'),
                'developer_id': metric.developer_id,
                'developer_name': developer_name,
                'project': metric.project_key,
                'commits': metric.commits,
                'story_points': metric.story_points,
                'lines_added': metric.lines_added,
                'lines_removed': metric.lines_removed,
                'files_changed': metric.files_changed,
                'complexity_added': round(metric.complexity_added, 2),
                'ticket_coverage': round(metric.ticket_coverage * 100, 1),
                'avg_commit_size': round(metric.avg_commit_size, 1),
                'efficiency_ratio': round(
                    metric.story_points / max(metric.commits, 1), 2
                ) if metric.story_points > 0 else 0
            })
        
        # Write to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def generate_narrative_report(self, weekly_metrics: List[WeeklyMetrics],
                                gaps: List[Dict],
                                developer_resolver: DeveloperIdentityResolver) -> str:
        """Generate human-readable narrative report"""
        
        narrative = StringIO()
        narrative.write("# GitFlow Analytics Report\n\n")
        narrative.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary statistics
        total_developers = len(set(m.developer_id for m in weekly_metrics))
        total_projects = len(set(m.project_key for m in weekly_metrics))
        total_weeks = len(set(m.week_start for m in weekly_metrics))
        
        narrative.write("## Executive Summary\n\n")
        narrative.write(self.narrative_templates['summary'].format(
            total_developers=total_developers,
            total_projects=total_projects,
            weeks=total_weeks
        ))
        narrative.write("\n\n")
        
        # Top performers
        developer_totals = defaultdict(lambda: {'story_points': 0, 'commits': 0, 'weeks': 0})
        for metric in weekly_metrics:
            developer_totals[metric.developer_id]['story_points'] += metric.story_points
            developer_totals[metric.developer_id]['commits'] += metric.commits
            developer_totals[metric.developer_id]['weeks'] += 1
        
        # Calculate averages and identify top performers
        top_performers = []
        for dev_id, totals in developer_totals.items():
            if totals['weeks'] >= 4:  # At least 4 weeks of data
                avg_sp = totals['story_points'] / totals['weeks']
                if avg_sp >= 10:  # Threshold for "high performer"
                    profile = developer_resolver.profiles.get(dev_id)
                    name = profile.primary_name if profile else "Unknown"
                    top_performers.append((name, avg_sp))
        
        if top_performers:
            narrative.write("## Top Performers\n\n")
            for name, avg_sp in sorted(top_performers, key=lambda x: x[1], reverse=True)[:5]:
                narrative.write(f"- {name}: {avg_sp:.1f} story points/week average\n")
            narrative.write("\n")
        
        # Gap analysis
        if gaps:
            narrative.write("## Areas for Improvement\n\n")
            
            # Group gaps by type
            gap_types = defaultdict(list)
            for gap in gaps:
                gap_types[gap['type']].append(gap)
            
            for gap_type, gap_list in gap_types.items():
                narrative.write(f"### {gap_type.replace('_', ' ').title()}\n\n")
                for gap in gap_list:
                    narrative.write(f"- {gap['description']}\n")
                    narrative.write(f"  *Recommendation: {gap['recommendation']}*\n\n")
        
        # Project-level insights
        narrative.write("## Project Analysis\n\n")
        project_metrics = defaultdict(lambda: {'commits': 0, 'coverage': [], 'story_points': 0})
        
        for metric in weekly_metrics:
            project_metrics[metric.project_key]['commits'] += metric.commits
            project_metrics[metric.project_key]['coverage'].append(metric.ticket_coverage)
            project_metrics[metric.project_key]['story_points'] += metric.story_points
        
        for project, metrics in project_metrics.items():
            avg_coverage = np.mean(metrics['coverage']) * 100
            narrative.write(f"**{project}:**\n")
            narrative.write(f"- Total commits: {metrics['commits']}\n")
            narrative.write(f"- Total story points: {metrics['story_points']}\n")
            narrative.write(f"- Average ticket coverage: {avg_coverage:.1f}%\n")
            
            if avg_coverage < 70:
                narrative.write(f"  ⚠️  Low ticket coverage suggests tracking gaps\n")
            narrative.write("\n")
        
        return narrative.getvalue()
    
    def generate_summary_stats(self, weekly_metrics: List[WeeklyMetrics]) -> Dict:
        """Generate summary statistics for the analysis period"""
        
        if not weekly_metrics:
            return {}
        
        total_commits = sum(m.commits for m in weekly_metrics)
        total_story_points = sum(m.story_points for m in weekly_metrics)
        total_lines_changed = sum(m.lines_added + m.lines_removed for m in weekly_metrics)
        
        developers = set(m.developer_id for m in weekly_metrics)
        projects = set(m.project_key for m in weekly_metrics)
        weeks = set(m.week_start for m in weekly_metrics)
        
        avg_ticket_coverage = np.mean([m.ticket_coverage for m in weekly_metrics]) * 100
        
        return {
            'total_commits': total_commits,
            'total_story_points': total_story_points,
            'total_lines_changed': total_lines_changed,
            'unique_developers': len(developers),
            'unique_projects': len(projects),
            'weeks_analyzed': len(weeks),
            'avg_ticket_coverage': round(avg_ticket_coverage, 1),
            'avg_commits_per_week': round(total_commits / len(weeks) if weeks else 0, 1),
            'avg_story_points_per_week': round(total_story_points / len(weeks) if weeks else 0, 1)
        }
```

### 6. Command Line Interface

```python
import click
import yaml
from pathlib import Path
from datetime import datetime, timedelta

@click.command()
@click.option('--config', '-c', required=True, help='Configuration file path')
@click.option('--output-dir', '-o', default='./reports', help='Output directory for reports')
@click.option('--weeks', '-w', default=12, help='Number of weeks to analyze')
@click.option('--github-token', envvar='GITHUB_TOKEN', help='GitHub API token')
def analyze(config, output_dir, weeks, github_token):
    """Analyze repositories and generate productivity reports"""
    
    # Load configuration
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Initialize components
    git_analyzer = GitAnalyzer(github_token)
    identity_resolver = DeveloperIdentityResolver()
    productivity_calculator = ProductivityCalculator()
    gap_detector = GapDetector()
    report_generator = ReportGenerator()
    
    # Set analysis period
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=weeks)
    
    click.echo(f"Analyzing {len(config_data['repositories'])} repositories for {weeks} weeks...")
    
    # Analyze repositories
    all_commits = git_analyzer.analyze_repositories(
        config_data['repositories'], 
        start_date
    )
    
    click.echo(f"Found {sum(len(commits) for commits in all_commits.values())} commits")
    
    # Calculate weekly metrics
    weekly_metrics = productivity_calculator.calculate_weekly_metrics(
        all_commits, identity_resolver, weeks
    )
    
    # Normalize productivity metrics
    normalized_metrics = productivity_calculator.calculate_normalized_productivity(weekly_metrics)
    
    # Detect gaps
    gaps = gap_detector.detect_gaps(all_commits, weekly_metrics)
    
    # Generate reports
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # CSV report
    csv_path = output_path / f'weekly_metrics_{datetime.now().strftime("%Y%m%d")}.csv'
    report_generator.generate_weekly_csv(weekly_metrics, identity_resolver, str(csv_path))
    click.echo(f"CSV report saved to: {csv_path}")
    
    # Narrative report
    narrative = report_generator.generate_narrative_report(
        weekly_metrics, gaps, identity_resolver
    )
    narrative_path = output_path / f'narrative_report_{datetime.now().strftime("%Y%m%d")}.md'
    with open(narrative_path, 'w') as f:
        f.write(narrative)
    click.echo(f"Narrative report saved to: {narrative_path}")
    
    # Summary statistics
    stats = report_generator.generate_summary_stats(weekly_metrics)
    click.echo("\n## Summary Statistics:")
    for key, value in stats.items():
        click.echo(f"{key.replace('_', ' ').title()}: {value}")

if __name__ == '__main__':
    analyze()
```

### 7. Configuration File Example

```yaml
# config.yaml
repositories:
  - path: "/path/to/repo1"
    project_key: "FRONTEND"
    github_repo: "company/frontend-app"
  - path: "/path/to/repo2" 
    project_key: "BACKEND"
    github_repo: "company/backend-api"
  - path: "/path/to/repo3"
    project_key: "MOBILE"
    github_repo: "company/mobile-app"

analysis:
  story_point_patterns:
    - "(?:story\\s*points?|sp|pts?)[\\s:]*([0-9]+)"
    - "\\[([0-9]+)\\s*pts?\\]"
    - "#([0-9]+)sp"
  
  complexity_weights:
    ".py": 1.0
    ".js": 0.8
    ".ts": 0.8
    ".java": 1.2
  
  identity_resolution:
    similarity_threshold: 0.85
    
github:
  api_base_url: "https://api.github.com"
  # Token should be provided via GITHUB_TOKEN environment variable

output:
  csv_filename_pattern: "weekly_metrics_{date}.csv"
  narrative_filename_pattern: "narrative_report_{date}.md"
```

## Usage Example

```bash
# Install the package
pip install gitflow-analytics

# Set GitHub token
export GITHUB_TOKEN=your_github_token_here

# Run analysis
gitflow-analytics analyze --config config.yaml --output-dir ./reports --weeks 12

# View results
ls ./reports/
# weekly_metrics_20240729.csv
# narrative_report_20240729.md
```

## Package Installation Setup

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name='gitflow-analytics',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pydriller>=2.5',
        'pygithub>=1.58',
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'click>=8.0.0',
        'pyyaml>=6.0',
        'scikit-learn>=1.0.0',
        'python-dateutil>=2.8.0'
    ],
    entry_points={
        'console_scripts': [
            'gitflow-analytics=gitflow_analytics.cli:analyze',
        ],
    },
    author='Your Name',
    description='Git repository productivity analysis tool',
    python_requires='>=3.8',
)
```

This package design provides a focused solution for analyzing Git repositories directly without requiring PM system integration initially. It extracts story points from commit messages and PR descriptions, provides comprehensive developer productivity metrics, and generates both CSV and narrative reports suitable for management review.

The architecture supports future AI integration through the gap detection system and narrative generation components, while maintaining the core capability to operate independently from Git data alone.