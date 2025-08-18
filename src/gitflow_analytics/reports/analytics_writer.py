"""Advanced analytics report generation with percentage and qualitative metrics."""
import csv
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Get logger for this module
logger = logging.getLogger(__name__)


class AnalyticsReportGenerator:
    """Generate advanced analytics reports with percentage breakdowns and qualitative insights."""
    
    def __init__(self, anonymize: bool = False, exclude_authors: list[str] = None, identity_resolver=None):
        """Initialize analytics report generator."""
        self.anonymize = anonymize
        self._anonymization_map = {}
        self._anonymous_counter = 0
        self.exclude_authors = exclude_authors or []
        self.identity_resolver = identity_resolver

    def _filter_excluded_authors(self, data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Filter out excluded authors from any data list using canonical_id.
        
        WHY: Bot exclusion happens in Phase 2 (reporting) instead of Phase 1 (data collection)
        to ensure manual identity mappings work correctly. This allows the system to see 
        consolidated bot identities via canonical_id instead of just original author_email/author_name.
        
        Args:
            data_list: List of data dictionaries containing canonical_id field
            
        Returns:
            Filtered list with excluded authors removed
        """
        if not self.exclude_authors:
            return data_list
            
        logger.debug(f"DEBUG EXCLUSION: Starting filter with {len(self.exclude_authors)} excluded authors: {self.exclude_authors}")
        logger.debug(f"DEBUG EXCLUSION: Filtering {len(data_list)} items from data list")
        
        excluded_lower = [author.lower() for author in self.exclude_authors]
        logger.debug(f"DEBUG EXCLUSION: Excluded authors (lowercase): {excluded_lower}")
        
        filtered_data = []
        excluded_count = 0
        
        # Sample first 5 items to see data structure
        for i, item in enumerate(data_list[:5]):
            logger.debug(f"DEBUG EXCLUSION: Sample item {i}: canonical_id='{item.get('canonical_id', '')}', "
                        f"author_email='{item.get('author_email', '')}', author_name='{item.get('author_name', '')}', "
                        f"author='{item.get('author', '')}', primary_name='{item.get('primary_name', '')}', "
                        f"name='{item.get('name', '')}', developer='{item.get('developer', '')}', "
                        f"display_name='{item.get('display_name', '')}'")  
        
        for item in data_list:
            canonical_id = item.get("canonical_id", "")
            # Also check original author fields as fallback for data without canonical_id
            author_email = item.get("author_email", "")
            author_name = item.get("author_name", "")
            
            # Check all possible author fields to ensure we catch every variation
            author = item.get("author", "")
            primary_name = item.get("primary_name", "")
            name = item.get("name", "")
            developer = item.get("developer", "")  # Common in analytics data
            display_name = item.get("display_name", "")  # Common in some data structures
            
            # Check canonical_id FIRST - this is the primary exclusion check
            should_exclude = False
            if canonical_id and canonical_id.lower() in excluded_lower:
                should_exclude = True
            # CRITICAL: Also check primary_email for manual mappings (e.g. bots mapped to bot@excluded.local)
            elif item.get("primary_email", "") and item.get("primary_email", "").lower() in excluded_lower:
                should_exclude = True
            # Fall back to checking other fields only if canonical_id and primary_email don't match
            elif not should_exclude:
                should_exclude = (
                    (author_email and author_email.lower() in excluded_lower) or
                    (author_name and author_name.lower() in excluded_lower) or
                    (author and author.lower() in excluded_lower) or
                    (primary_name and primary_name.lower() in excluded_lower) or
                    (name and name.lower() in excluded_lower) or
                    (developer and developer.lower() in excluded_lower) or
                    (display_name and display_name.lower() in excluded_lower)
                )
            
            if should_exclude:
                excluded_count += 1
                logger.debug(f"DEBUG EXCLUSION: EXCLUDING item - canonical_id='{canonical_id}', "
                           f"primary_email='{item.get('primary_email', '')}', "
                           f"author_email='{author_email}', author_name='{author_name}', author='{author}', "
                           f"primary_name='{primary_name}', name='{name}', developer='{developer}', "
                           f"display_name='{display_name}'")
            else:
                filtered_data.append(item)
                
        logger.debug(f"DEBUG EXCLUSION: Excluded {excluded_count} items, kept {len(filtered_data)} items")
        return filtered_data

    def _get_canonical_display_name(self, canonical_id: str, fallback_name: str) -> str:
        """
        Get the canonical display name for a developer.
        
        WHY: Manual identity mappings may have updated display names that aren't
        reflected in the developer_stats data passed to report generators. This
        method ensures we get the most current display name from the identity resolver.
        
        Args:
            canonical_id: The canonical ID to get the display name for
            fallback_name: The fallback name to use if identity resolver is not available
            
        Returns:
            The canonical display name or fallback name
        """
        if self.identity_resolver and canonical_id:
            try:
                canonical_name = self.identity_resolver.get_canonical_name(canonical_id)
                if canonical_name and canonical_name != "Unknown":
                    return canonical_name
            except Exception as e:
                logger.debug(f"Error getting canonical name for {canonical_id}: {e}")
        
        return fallback_name
    
    def _get_files_changed_count(self, commit: Dict[str, Any]) -> int:
        """Safely extract files_changed count from commit data.
        
        WHY: The files_changed field can be either an int (count) or list (file names).
        This helper ensures we always get an integer count for calculations.
        
        Args:
            commit: Commit dictionary with files_changed field
            
        Returns:
            Integer count of files changed
        """
        files_changed = commit.get('files_changed', 0)
        
        if isinstance(files_changed, int):
            return files_changed
        elif isinstance(files_changed, list):
            return len(files_changed)
        else:
            # Fallback for unexpected types
            logger.warning(f"Unexpected files_changed type: {type(files_changed)}, defaulting to 0")
            return 0
    
    def _log_datetime_comparison(self, dt1: datetime, dt2: datetime, operation: str, location: str) -> None:
        """Log datetime comparison details for debugging timezone issues."""
        logger.debug(f"Comparing dates in {location} ({operation}):")
        logger.debug(f"  dt1: {dt1} (tzinfo: {dt1.tzinfo}, aware: {dt1.tzinfo is not None})")
        logger.debug(f"  dt2: {dt2} (tzinfo: {dt2.tzinfo}, aware: {dt2.tzinfo is not None})")
    
    def _safe_datetime_compare(self, dt1: datetime, dt2: datetime, operation: str, location: str) -> bool:
        """Safely compare datetimes with logging and error handling."""
        try:
            self._log_datetime_comparison(dt1, dt2, operation, location)
            
            if operation == 'lt':
                result = dt1 < dt2
            elif operation == 'gt':
                result = dt1 > dt2
            elif operation == 'le':
                result = dt1 <= dt2
            elif operation == 'ge':
                result = dt1 >= dt2
            elif operation == 'eq':
                result = dt1 == dt2
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
            logger.debug(f"  Result: {result}")
            return result
            
        except TypeError as e:
            logger.error(f"Timezone comparison error in {location}:")
            logger.error(f"  dt1: {dt1} (type: {type(dt1)}, tzinfo: {getattr(dt1, 'tzinfo', 'N/A')})")
            logger.error(f"  dt2: {dt2} (type: {type(dt2)}, tzinfo: {getattr(dt2, 'tzinfo', 'N/A')})")
            logger.error(f"  Operation: {operation}")
            logger.error(f"  Error: {e}")
            
            # Import traceback for detailed error info
            import traceback
            logger.error(f"  Full traceback:\n{traceback.format_exc()}")
            
            # Try to fix by making both timezone-aware in UTC
            try:
                if dt1.tzinfo is None:
                    dt1 = dt1.replace(tzinfo=timezone.utc)
                    logger.debug(f"  Fixed dt1 to UTC: {dt1}")
                if dt2.tzinfo is None:
                    dt2 = dt2.replace(tzinfo=timezone.utc)
                    logger.debug(f"  Fixed dt2 to UTC: {dt2}")
                
                # Retry comparison
                if operation == 'lt':
                    result = dt1 < dt2
                elif operation == 'gt':
                    result = dt1 > dt2
                elif operation == 'le':
                    result = dt1 <= dt2
                elif operation == 'ge':
                    result = dt1 >= dt2
                elif operation == 'eq':
                    result = dt1 == dt2
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                    
                logger.info(f"  Fixed comparison result: {result}")
                return result
                
            except Exception as fix_error:
                logger.error(f"  Failed to fix timezone issue: {fix_error}")
                raise
    
    def generate_activity_distribution_report(self, commits: List[Dict[str, Any]], 
                                            developer_stats: List[Dict[str, Any]],
                                            output_path: Path) -> Path:
        """Generate activity distribution report with percentage breakdowns."""
        # Apply exclusion filtering in Phase 2
        commits = self._filter_excluded_authors(commits)
        developer_stats = self._filter_excluded_authors(developer_stats)
        
        # Build lookup maps
        dev_lookup = {dev['canonical_id']: dev for dev in developer_stats}
        
        # Calculate totals
        total_commits = len(commits)
        total_lines = sum(
            c.get('filtered_insertions', c.get('insertions', 0)) + 
            c.get('filtered_deletions', c.get('deletions', 0)) 
            for c in commits
        )
        total_files = sum(self._get_files_changed_count(c) for c in commits)
        
        # Group by developer and project
        dev_project_activity = defaultdict(lambda: defaultdict(lambda: {
            'commits': 0, 'lines': 0, 'files': 0, 'story_points': 0
        }))
        
        for commit in commits:
            dev_id = commit.get('canonical_id', commit.get('author_email'))
            project = commit.get('project_key', 'UNKNOWN')
            
            dev_project_activity[dev_id][project]['commits'] += 1
            dev_project_activity[dev_id][project]['lines'] += (
                commit.get('filtered_insertions', commit.get('insertions', 0)) + 
                commit.get('filtered_deletions', commit.get('deletions', 0))
            )
            # Handle files_changed safely - could be int or list
            files_changed = commit.get('filtered_files_changed')
            if files_changed is None:
                files_changed = self._get_files_changed_count(commit)
            elif isinstance(files_changed, list):
                files_changed = len(files_changed)
            elif not isinstance(files_changed, int):
                files_changed = 0
            
            dev_project_activity[dev_id][project]['files'] += files_changed
            dev_project_activity[dev_id][project]['story_points'] += commit.get('story_points', 0) or 0
        
        # Build report data
        rows = []
        
        for dev_id, projects in dev_project_activity.items():
            developer = dev_lookup.get(dev_id, {})
            dev_name = self._anonymize_value(
                self._get_canonical_display_name(
                    dev_id, 
                    developer.get('primary_name', 'Unknown')
                ), 'name'
            )
            
            # Calculate developer totals
            dev_total_commits = sum(p['commits'] for p in projects.values())
            dev_total_lines = sum(p['lines'] for p in projects.values())
            dev_total_files = sum(p['files'] for p in projects.values())
            
            for project, activity in projects.items():
                row = {
                    'developer': dev_name,
                    'project': project,
                    # Raw numbers
                    'commits': activity['commits'],
                    'lines_changed': activity['lines'],
                    'files_changed': activity['files'],
                    'story_points': activity['story_points'],
                    # Developer perspective (% of developer's time on this project)
                    'dev_commit_pct': round(activity['commits'] / dev_total_commits * 100, 1),
                    'dev_lines_pct': round(activity['lines'] / dev_total_lines * 100, 1) if dev_total_lines > 0 else 0,
                    'dev_files_pct': round(activity['files'] / dev_total_files * 100, 1) if dev_total_files > 0 else 0,
                    # Project perspective (% of project work by this developer)
                    'proj_commit_pct': round(activity['commits'] / total_commits * 100, 1),
                    'proj_lines_pct': round(activity['lines'] / total_lines * 100, 1) if total_lines > 0 else 0,
                    'proj_files_pct': round(activity['files'] / total_files * 100, 1) if total_files > 0 else 0,
                    # Overall perspective (% of total activity)
                    'total_activity_pct': round(activity['commits'] / total_commits * 100, 1)
                }
                rows.append(row)
        
        # Sort by total activity
        rows.sort(key=lambda x: x['total_activity_pct'], reverse=True)
        
        # Write CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def generate_qualitative_insights_report(self, commits: List[Dict[str, Any]],
                                           developer_stats: List[Dict[str, Any]],
                                           ticket_analysis: Dict[str, Any],
                                           output_path: Path) -> Path:
        """Generate qualitative insights and patterns report."""
        # Apply exclusion filtering in Phase 2
        commits = self._filter_excluded_authors(commits)
        developer_stats = self._filter_excluded_authors(developer_stats)
        insights = []
        
        # Analyze commit patterns
        commit_insights = self._analyze_commit_patterns(commits)
        insights.extend(commit_insights)
        
        # Analyze developer patterns
        dev_insights = self._analyze_developer_patterns(commits, developer_stats)
        insights.extend(dev_insights)
        
        # Analyze collaboration patterns
        collab_insights = self._analyze_collaboration_patterns(commits)
        insights.extend(collab_insights)
        
        # Analyze work distribution
        dist_insights = self._analyze_work_distribution(commits)
        insights.extend(dist_insights)
        
        # Write insights to CSV
        df = pd.DataFrame(insights)
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def generate_developer_focus_report(self, commits: List[Dict[str, Any]],
                                      developer_stats: List[Dict[str, Any]],
                                      output_path: Path,
                                      weeks: int = 12) -> Path:
        """Generate developer focus analysis showing concentration patterns and activity across all projects."""
        # Apply exclusion filtering in Phase 2
        commits = self._filter_excluded_authors(commits)
        developer_stats = self._filter_excluded_authors(developer_stats)
        
        # Calculate week boundaries (timezone-aware to match commit timestamps)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)
        
        logger.debug(f"Developer focus report date range:")
        logger.debug(f"  start_date: {start_date} (tzinfo: {start_date.tzinfo})")
        logger.debug(f"  end_date: {end_date} (tzinfo: {end_date.tzinfo})")
        
        # Build developer lookup
        dev_lookup = {dev['canonical_id']: dev for dev in developer_stats}
        
        # Get all unique projects
        all_projects = sorted(list(set(c.get('project_key', 'UNKNOWN') for c in commits)))
        
        # Analyze focus patterns
        focus_data = []
        
        # Calculate total commits per project for percentage calculations
        project_totals = defaultdict(int)
        for commit in commits:
            project_totals[commit.get('project_key', 'UNKNOWN')] += 1
        
        total_commits = len(commits)
        
        for dev in developer_stats:
            dev_id = dev['canonical_id']
            dev_name = self._anonymize_value(
                self._get_canonical_display_name(
                    dev_id, 
                    dev['primary_name']
                ), 'name'
            )
            
            # Get developer's commits
            dev_commits = [c for c in commits if c.get('canonical_id') == dev_id]
            if not dev_commits:
                continue
            
            # Calculate focus metrics
            projects = defaultdict(int)
            project_lines = defaultdict(int)
            weekly_activity = defaultdict(int)
            commit_sizes = []
            commit_hours = []
            
            for commit in dev_commits:
                # Log commit processing
                logger.debug(f"Processing commit for developer {dev_name}: {commit.get('hash', 'unknown')[:8]}")
                logger.debug(f"  timestamp: {commit['timestamp']} (tzinfo: {getattr(commit['timestamp'], 'tzinfo', 'N/A')})")
                
                # Project distribution
                project_key = commit.get('project_key', 'UNKNOWN')
                projects[project_key] += 1
                
                # Lines changed per project
                lines_changed = (
                    commit.get('filtered_insertions', commit.get('insertions', 0)) + 
                    commit.get('filtered_deletions', commit.get('deletions', 0))
                )
                project_lines[project_key] += lines_changed
                
                # Weekly distribution
                week_start = self._get_week_start(commit['timestamp'])
                weekly_activity[week_start] += 1
                
                # Commit size
                commit_sizes.append(lines_changed)
                
                # Time of day (use local hour if available, fallback to UTC)
                if 'local_hour' in commit:
                    commit_hours.append(commit['local_hour'])
                elif hasattr(commit['timestamp'], 'hour'):
                    commit_hours.append(commit['timestamp'].hour)
            
            # Calculate metrics
            num_projects = len(projects)
            primary_project = max(projects, key=projects.get) if projects else 'UNKNOWN'
            primary_project_pct = round(projects[primary_project] / len(dev_commits) * 100, 1)
            
            # Focus score (100% = single project, lower = more scattered)
            focus_score = round(100 / num_projects if num_projects > 0 else 0, 1)
            
            # Consistency score (active weeks / total weeks)
            active_weeks = len(weekly_activity)
            consistency_score = round(active_weeks / weeks * 100, 1)
            
            # Work pattern
            avg_commit_size = np.mean(commit_sizes) if commit_sizes else 0
            if avg_commit_size < 50:
                work_style = "Small, frequent changes"
            elif avg_commit_size < 200:
                work_style = "Moderate batch changes"
            else:
                work_style = "Large batch changes"
            
            # Time pattern
            if commit_hours:
                avg_hour = np.mean(commit_hours)
                if avg_hour < 10:
                    time_pattern = "Morning developer"
                elif avg_hour < 14:
                    time_pattern = "Midday developer"
                elif avg_hour < 18:
                    time_pattern = "Afternoon developer"
                else:
                    time_pattern = "Evening developer"
            else:
                time_pattern = "Unknown"
            
            # Build the row with basic metrics
            row = {
                'developer': dev_name,
                'total_commits': len(dev_commits),
                'num_projects': num_projects,
                'primary_project': primary_project,
                'primary_project_pct': primary_project_pct,
                'focus_score': focus_score,
                'active_weeks': active_weeks,
                'consistency_score': consistency_score,
                'avg_commit_size': round(avg_commit_size, 1),
                'work_style': work_style,
                'time_pattern': time_pattern
            }
            
            # Add project-specific metrics
            for project in all_projects:
                # Gross commits
                gross_commits = projects.get(project, 0)
                row[f'{project}_gross_commits'] = gross_commits
                
                # Adjusted commits (weighted by lines changed)
                if gross_commits > 0 and project_lines[project] > 0:
                    # Adjustment factor based on average lines per commit
                    project_avg_lines = project_lines[project] / gross_commits
                    overall_avg_lines = sum(commit_sizes) / len(commit_sizes) if commit_sizes else 1
                    adjustment_factor = project_avg_lines / overall_avg_lines if overall_avg_lines > 0 else 1
                    adjusted_commits = round(gross_commits * adjustment_factor, 1)
                else:
                    adjusted_commits = 0
                row[f'{project}_adjusted_commits'] = adjusted_commits
                
                # Percent of developer's activity
                dev_pct = round(gross_commits / len(dev_commits) * 100, 1) if len(dev_commits) > 0 else 0
                row[f'{project}_dev_pct'] = dev_pct
                
                # Percent of project's total activity
                proj_pct = round(gross_commits / project_totals[project] * 100, 1) if project_totals[project] > 0 else 0
                row[f'{project}_proj_pct'] = proj_pct
                
                # Percent of total repository activity
                total_pct = round(gross_commits / total_commits * 100, 1) if total_commits > 0 else 0
                row[f'{project}_total_pct'] = total_pct
            
            focus_data.append(row)
        
        # Sort by focus score
        focus_data.sort(key=lambda x: x['focus_score'], reverse=True)
        
        # Write CSV
        df = pd.DataFrame(focus_data)
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def generate_weekly_trends_report(self, commits: List[Dict[str, Any]], 
                                    developer_stats: List[Dict[str, Any]],
                                    output_path: Path,
                                    weeks: int = 12) -> Path:
        """Generate weekly trends analysis showing changes in activity patterns."""
        # Apply exclusion filtering in Phase 2
        commits = self._filter_excluded_authors(commits)
        developer_stats = self._filter_excluded_authors(developer_stats)
        
        # Calculate week boundaries
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)
        
        # Build developer lookup
        dev_lookup = {dev['canonical_id']: dev for dev in developer_stats}
        
        # Initialize data structures
        weekly_data = defaultdict(lambda: {
            'commits': 0,
            'developers': set(),
            'projects': defaultdict(int),
            'lines_changed': 0,
            'story_points': 0
        })
        
        developer_weekly = defaultdict(lambda: defaultdict(lambda: {
            'commits': 0, 'lines': 0, 'story_points': 0
        }))
        project_weekly = defaultdict(lambda: defaultdict(lambda: {
            'commits': 0, 'lines': 0, 'developers': set(), 'story_points': 0
        }))
        
        # Process commits
        for commit in commits:
            week_start = self._get_week_start(commit['timestamp'])
            week_key = week_start.strftime('%Y-%m-%d')
            
            # Overall weekly metrics
            weekly_data[week_key]['commits'] += 1
            weekly_data[week_key]['developers'].add(commit.get('canonical_id'))
            weekly_data[week_key]['projects'][commit.get('project_key', 'UNKNOWN')] += 1
            lines = (
                commit.get('filtered_insertions', commit.get('insertions', 0)) + 
                commit.get('filtered_deletions', commit.get('deletions', 0))
            )
            weekly_data[week_key]['lines_changed'] += lines
            weekly_data[week_key]['story_points'] += commit.get('story_points', 0) or 0
            
            # Developer-specific weekly data
            dev_id = commit.get('canonical_id')
            developer_weekly[dev_id][week_key]['commits'] += 1
            developer_weekly[dev_id][week_key]['lines'] += lines
            developer_weekly[dev_id][week_key]['story_points'] += commit.get('story_points', 0) or 0
            
            # Project-specific weekly data
            project = commit.get('project_key', 'UNKNOWN')
            project_weekly[project][week_key]['commits'] += 1
            project_weekly[project][week_key]['lines'] += lines
            project_weekly[project][week_key]['developers'].add(dev_id)
            project_weekly[project][week_key]['story_points'] += commit.get('story_points', 0) or 0
        
        # Convert to rows for CSV
        rows = []
        sorted_weeks = sorted(weekly_data.keys())
        
        # Track developer and project trends
        dev_activity_changes = defaultdict(list)  # dev_id -> list of weekly changes
        project_activity_changes = defaultdict(list)  # project -> list of weekly changes
        
        for i, week in enumerate(sorted_weeks):
            data = weekly_data[week]
            
            # Calculate week-over-week changes
            prev_week = sorted_weeks[i-1] if i > 0 else None
            
            commits_change = 0
            developers_change = 0
            if prev_week:
                prev_data = weekly_data[prev_week]
                commits_change = data['commits'] - prev_data['commits']
                developers_change = len(data['developers']) - len(prev_data['developers'])
            
            # Top project and developer this week
            top_project = max(data['projects'].items(), key=lambda x: x[1])[0] if data['projects'] else 'NONE'
            
            # Find top developer this week
            top_dev_id = None
            top_dev_commits = 0
            for dev_id in data['developers']:
                dev_commits = developer_weekly[dev_id][week]['commits']
                if dev_commits > top_dev_commits:
                    top_dev_commits = dev_commits
                    top_dev_id = dev_id
            
            top_dev_name = self._anonymize_value(
                self._get_canonical_display_name(
                    top_dev_id, 
                    dev_lookup.get(top_dev_id, {}).get('primary_name', 'Unknown')
                ), 'name'
            ) if top_dev_id else 'None'
            
            # Calculate developer trends for active developers this week
            dev_trend_summary = []
            for dev_id in data['developers']:
                dev_data = developer_weekly[dev_id][week]
                prev_dev_data = developer_weekly[dev_id].get(prev_week, {'commits': 0}) if prev_week else {'commits': 0}
                change = dev_data['commits'] - prev_dev_data['commits']
                if change != 0:
                    dev_name = self._anonymize_value(
                        self._get_canonical_display_name(
                            dev_id, 
                            dev_lookup.get(dev_id, {}).get('primary_name', 'Unknown')
                        ), 'name'
                    )
                    dev_activity_changes[dev_name].append(change)
                    if abs(change) >= 3:  # Significant changes only
                        dev_trend_summary.append(f"{dev_name}({'+' if change > 0 else ''}{change})")
            
            # Calculate project trends
            project_trend_summary = []
            for project, count in data['projects'].items():
                prev_count = weekly_data[prev_week]['projects'].get(project, 0) if prev_week else 0
                change = count - prev_count
                if change != 0:
                    project_activity_changes[project].append(change)
                    if abs(change) >= 3:  # Significant changes only
                        project_trend_summary.append(f"{project}({'+' if change > 0 else ''}{change})")
            
            row = {
                'week_start': week,
                'commits': data['commits'],
                'active_developers': len(data['developers']),
                'active_projects': len(data['projects']),
                'lines_changed': data['lines_changed'],
                'story_points': data['story_points'],
                'commits_change': commits_change,
                'developers_change': developers_change,
                'top_project': top_project,
                'top_developer': top_dev_name,
                'avg_commits_per_dev': round(data['commits'] / max(len(data['developers']), 1), 1),
                'avg_lines_per_commit': round(data['lines_changed'] / max(data['commits'], 1), 1),
                'developer_trends': '; '.join(dev_trend_summary[:5]) if dev_trend_summary else 'stable',
                'project_trends': '; '.join(project_trend_summary[:5]) if project_trend_summary else 'stable'
            }
            rows.append(row)
        
        # Write main CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        # Also generate detailed developer trends CSV with weekly columns
        dev_trends_path = output_path.parent / f'developer_trends_{output_path.stem.split("_")[-1]}.csv'
        dev_trend_rows = []
        
        # Build developer activity by week
        for dev_id, weekly_commits in developer_weekly.items():
            dev_info = dev_lookup.get(dev_id, {})
            dev_name = self._anonymize_value(
                self._get_canonical_display_name(
                    dev_id, 
                    dev_info.get('primary_name', 'Unknown')
                ), 'name'
            )
            
            # Calculate summary statistics
            weekly_values = []
            for week in sorted_weeks:
                commits = weekly_commits.get(week, {}).get('commits', 0)
                weekly_values.append(commits)
            
            # Only include developers with any activity
            if sum(weekly_values) > 0:
                # Calculate trend metrics
                changes = []
                for i in range(1, len(weekly_values)):
                    changes.append(weekly_values[i] - weekly_values[i-1])
                
                avg_change = sum(changes) / len(changes) if changes else 0
                volatility = np.std(changes) if len(changes) > 1 else 0
                trend = 'increasing' if avg_change > 1 else 'decreasing' if avg_change < -1 else 'stable'
                
                row = {
                    'developer': dev_name,
                    'total_commits': sum(weekly_values),
                    'avg_weekly_commits': round(sum(weekly_values) / len(weekly_values), 1),
                    'avg_weekly_change': round(avg_change, 1),
                    'volatility': round(volatility, 1),
                    'trend': trend,
                    'total_weeks_active': len([v for v in weekly_values if v > 0]),
                    'max_week': max(weekly_values),
                    'min_week': min([v for v in weekly_values if v > 0]) if any(v > 0 for v in weekly_values) else 0
                }
                
                # Add weekly columns
                for i, week in enumerate(sorted_weeks):
                    week_label = f'week_{i+1}_{week}'
                    row[week_label] = weekly_values[i]
                
                dev_trend_rows.append(row)
        
        if dev_trend_rows:
            dev_trends_df = pd.DataFrame(dev_trend_rows)
            # Sort by total commits to show most active developers first
            dev_trends_df.sort_values('total_commits', ascending=False, inplace=True)
            dev_trends_df.to_csv(dev_trends_path, index=False)
        
        # Also generate detailed project trends CSV with weekly columns
        proj_trends_path = output_path.parent / f'project_trends_{output_path.stem.split("_")[-1]}.csv'
        proj_trend_rows = []
        
        # Build project activity by week
        for project, weekly_commits in project_weekly.items():
            # Calculate summary statistics
            weekly_values = []
            weekly_developers = []
            for week in sorted_weeks:
                commits = weekly_commits.get(week, {}).get('commits', 0)
                weekly_values.append(commits)
                # Count unique developers for this project this week
                devs = weekly_commits.get(week, {}).get('developers', set())
                weekly_developers.append(len(devs))
            
            # Only include projects with any activity
            if sum(weekly_values) > 0:
                # Calculate trend metrics
                changes = []
                for i in range(1, len(weekly_values)):
                    changes.append(weekly_values[i] - weekly_values[i-1])
                
                avg_change = sum(changes) / len(changes) if changes else 0
                volatility = np.std(changes) if len(changes) > 1 else 0
                trend = 'growing' if avg_change > 2 else 'shrinking' if avg_change < -2 else 'stable'
                
                row = {
                    'project': project,
                    'total_commits': sum(weekly_values),
                    'avg_weekly_commits': round(sum(weekly_values) / len(weekly_values), 1),
                    'avg_weekly_developers': round(sum(weekly_developers) / len(weekly_developers), 1),
                    'avg_weekly_change': round(avg_change, 1),
                    'volatility': round(volatility, 1),
                    'trend': trend,
                    'total_weeks_active': len([v for v in weekly_values if v > 0]),
                    'max_week': max(weekly_values),
                    'min_week': min([v for v in weekly_values if v > 0]) if any(v > 0 for v in weekly_values) else 0
                }
                
                # Add weekly columns for commits
                for i, week in enumerate(sorted_weeks):
                    week_label = f'week_{i+1}_{week}'
                    row[week_label] = weekly_values[i]
                
                # Add weekly columns for developer count
                for i, week in enumerate(sorted_weeks):
                    week_label = f'devs_week_{i+1}'
                    row[week_label] = weekly_developers[i]
                
                proj_trend_rows.append(row)
        
        if proj_trend_rows:
            proj_trends_df = pd.DataFrame(proj_trend_rows)
            # Sort by total commits to show most active projects first
            proj_trends_df.sort_values('total_commits', ascending=False, inplace=True)
            proj_trends_df.to_csv(proj_trends_path, index=False)
        
        return output_path
    
    def _analyze_commit_patterns(self, commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze patterns in commit data."""
        insights = []
        
        # Time-based patterns (use local hour if available)
        commit_hours = []
        for c in commits:
            if 'local_hour' in c:
                commit_hours.append(c['local_hour'])
            elif hasattr(c['timestamp'], 'hour'):
                commit_hours.append(c['timestamp'].hour)
        
        if commit_hours:
            peak_hour = max(set(commit_hours), key=commit_hours.count)
            insights.append({
                'category': 'Timing',
                'insight': 'Peak commit hour',
                'value': f"{peak_hour}:00",
                'impact': 'Indicates team working hours'
            })
        
        # Commit message patterns
        message_lengths = [len(c['message'].split()) for c in commits]
        avg_message_length = np.mean(message_lengths)
        
        if avg_message_length < 5:
            quality = "Very brief"
        elif avg_message_length < 10:
            quality = "Concise"
        elif avg_message_length < 20:
            quality = "Detailed"
        else:
            quality = "Very detailed"
        
        insights.append({
            'category': 'Quality',
            'insight': 'Commit message quality',
            'value': quality,
            'impact': f"Average {avg_message_length:.1f} words per message"
        })
        
        # Ticket coverage insights
        commits_with_tickets = sum(1 for c in commits if c.get('ticket_references'))
        coverage_pct = commits_with_tickets / len(commits) * 100 if commits else 0
        
        if coverage_pct < 30:
            tracking = "Poor tracking"
        elif coverage_pct < 60:
            tracking = "Moderate tracking"
        elif coverage_pct < 80:
            tracking = "Good tracking"
        else:
            tracking = "Excellent tracking"
        
        insights.append({
            'category': 'Process',
            'insight': 'Ticket tracking adherence',
            'value': tracking,
            'impact': f"{coverage_pct:.1f}% commits have ticket references"
        })
        
        return insights
    
    def _analyze_developer_patterns(self, commits: List[Dict[str, Any]], 
                                  developer_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze developer behavior patterns."""
        insights = []
        
        # Team size insights
        team_size = len(developer_stats)
        if team_size < 3:
            team_type = "Very small team"
        elif team_size < 6:
            team_type = "Small team"
        elif team_size < 12:
            team_type = "Medium team"
        else:
            team_type = "Large team"
        
        insights.append({
            'category': 'Team',
            'insight': 'Team size',
            'value': team_type,
            'impact': f"{team_size} active developers"
        })
        
        # Contribution distribution
        commit_counts = [dev['total_commits'] for dev in developer_stats]
        if commit_counts:
            gini_coef = self._calculate_gini_coefficient(commit_counts)
            
            if gini_coef < 0.3:
                distribution = "Very balanced"
            elif gini_coef < 0.5:
                distribution = "Moderately balanced"
            elif gini_coef < 0.7:
                distribution = "Somewhat unbalanced"
            else:
                distribution = "Highly concentrated"
            
            insights.append({
                'category': 'Team',
                'insight': 'Work distribution',
                'value': distribution,
                'impact': f"Gini coefficient: {gini_coef:.2f}"
            })
        
        return insights
    
    def _analyze_collaboration_patterns(self, commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze collaboration patterns."""
        insights = []
        
        # Merge commit analysis
        merge_commits = [c for c in commits if c.get('is_merge')]
        merge_pct = len(merge_commits) / len(commits) * 100 if commits else 0
        
        if merge_pct < 5:
            branching = "Minimal branching"
        elif merge_pct < 15:
            branching = "Moderate branching"
        elif merge_pct < 25:
            branching = "Active branching"
        else:
            branching = "Heavy branching"
        
        insights.append({
            'category': 'Workflow',
            'insight': 'Branching strategy',
            'value': branching,
            'impact': f"{merge_pct:.1f}% merge commits"
        })
        
        return insights
    
    def _analyze_work_distribution(self, commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze work distribution patterns."""
        insights = []
        
        # File change patterns
        file_changes = []
        for c in commits:
            files_count = self._get_files_changed_count(c)
            if files_count > 0:
                file_changes.append(files_count)
        if file_changes:
            avg_files = np.mean(file_changes)
            
            if avg_files < 3:
                pattern = "Focused changes"
            elif avg_files < 8:
                pattern = "Moderate scope changes"
            else:
                pattern = "Broad scope changes"
            
            insights.append({
                'category': 'Workflow',
                'insight': 'Change scope pattern',
                'value': pattern,
                'impact': f"Average {avg_files:.1f} files per commit"
            })
        
        return insights
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for distribution analysis."""
        if not values or len(values) == 1:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((i + 1) * sorted_values[i] for i in range(n))) / (n * cumsum[-1]) - (n + 1) / n
    
    def _get_week_start(self, date: datetime) -> datetime:
        """Get Monday of the week for a given date."""
        logger.debug(f"Getting week start for date: {date} (tzinfo: {getattr(date, 'tzinfo', 'N/A')})")
        
        # Ensure consistent timezone handling - keep timezone info
        if hasattr(date, 'tzinfo') and date.tzinfo is not None:
            # Keep timezone-aware but ensure it's UTC
            if date.tzinfo != timezone.utc:
                date = date.astimezone(timezone.utc)
                logger.debug(f"  Converted to UTC: {date}")
        else:
            # Convert naive datetime to UTC timezone-aware
            date = date.replace(tzinfo=timezone.utc)
            logger.debug(f"  Made timezone-aware: {date}")
        
        days_since_monday = date.weekday()
        monday = date - timedelta(days=days_since_monday)
        result = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        
        logger.debug(f"  Week start result: {result} (tzinfo: {result.tzinfo})")
        return result
    
    def _anonymize_value(self, value: str, field_type: str) -> str:
        """Anonymize a value if anonymization is enabled."""
        if not self.anonymize or not value:
            return value
        
        if value not in self._anonymization_map:
            self._anonymous_counter += 1
            if field_type == 'name':
                anonymous = f"Developer{self._anonymous_counter}"
            else:
                anonymous = f"anon{self._anonymous_counter}"
            self._anonymization_map[value] = anonymous
        
        return self._anonymization_map[value]