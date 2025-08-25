"""Weekly classification trends CSV report generation."""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class WeeklyTrendsWriter:
    """Generate weekly classification trends CSV reports.
    
    WHY: Week-over-week classification trends provide insights into changing
    development patterns, helping identify evolving team practices, seasonal
    patterns, and the impact of process changes on development work types.
    
    DESIGN DECISION: Generate separate developer and project trend reports
    to allow analysis at different granularities. Include percentage changes
    to highlight velocity and pattern shifts.
    """
    
    def __init__(self) -> None:
        """Initialize weekly trends writer."""
        self.classification_categories = [
            'feature', 'bug_fix', 'refactor', 'documentation',
            'maintenance', 'test', 'style', 'build', 'other'
        ]
    
    def generate_weekly_trends_reports(
        self,
        commits: List[Dict[str, Any]],
        output_dir: Path,
        weeks: int = 12,
        date_suffix: str = ""
    ) -> Dict[str, Path]:
        """Generate both developer and project weekly trends reports.
        
        WHY: Providing both perspectives allows analysis of individual developer
        patterns as well as project-level trend analysis. This enables both
        personal development tracking and project health monitoring.
        
        Args:
            commits: List of commit data with classifications and timestamps
            output_dir: Directory to write CSV reports to
            weeks: Number of weeks to analyze (for validation)
            date_suffix: Date suffix for output filenames
            
        Returns:
            Dictionary mapping report type to output file paths
        """
        output_paths = {}
        
        # Generate developer trends report
        developer_trends_path = output_dir / f"developer_weekly_trends{date_suffix}.csv"
        self._generate_developer_weekly_trends(commits, developer_trends_path, weeks)
        output_paths['developer_trends'] = developer_trends_path
        
        # Generate project trends report  
        project_trends_path = output_dir / f"project_weekly_trends{date_suffix}.csv"
        self._generate_project_weekly_trends(commits, project_trends_path, weeks)
        output_paths['project_trends'] = project_trends_path
        
        logger.info(f"Generated weekly trends reports: {len(output_paths)} files")
        return output_paths
    
    def _generate_developer_weekly_trends(
        self,
        commits: List[Dict[str, Any]],
        output_path: Path,
        weeks: int
    ) -> None:
        """Generate developer weekly classification trends CSV.
        
        WHY: Developer-level trends help identify individual development patterns,
        skill progression, and changing work focus over time. This enables
        targeted coaching and recognition of evolving expertise.
        
        Args:
            commits: List of commit data with developer and classification info
            output_path: Path to write the CSV file
            weeks: Number of weeks for trend analysis
        """
        # Group commits by developer and week
        developer_weeks = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Find the date range for analysis
        if not commits:
            logger.warning("No commits provided for developer weekly trends analysis")
            self._write_empty_developer_trends_csv(output_path)
            return
        
        # Sort commits by timestamp for consistent week calculation
        sorted_commits = sorted(
            [c for c in commits if c.get('timestamp')],
            key=lambda x: x['timestamp'],
            reverse=True  # Most recent first
        )
        
        if not sorted_commits:
            logger.warning("No commits with timestamps for developer weekly trends analysis")
            self._write_empty_developer_trends_csv(output_path)
            return
        
        # Calculate week boundaries
        latest_date = sorted_commits[0]['timestamp']
        if hasattr(latest_date, 'date'):
            latest_date = latest_date.date()
        
        # Group commits by developer, week, and classification
        for commit in sorted_commits:
            timestamp = commit.get('timestamp')
            if not timestamp:
                continue
            
            # Get week number (0 = current week, 1 = last week, etc.)
            if hasattr(timestamp, 'date'):
                commit_date = timestamp.date()
            else:
                commit_date = timestamp
            
            days_diff = (latest_date - commit_date).days
            week_num = days_diff // 7
            
            # Only include commits within the analysis period
            if week_num >= weeks:
                continue
            
            # Extract developer info
            developer = (
                commit.get('canonical_id') or 
                commit.get('author_email') or 
                commit.get('author_name', 'Unknown')
            )
            
            # Get classification - try multiple possible fields
            classification = self._get_commit_classification(commit)
            
            # Increment count
            developer_weeks[developer][week_num][classification] += 1
        
        # Build DataFrame
        rows = []
        for developer, weeks_data in developer_weeks.items():
            # Sort weeks in chronological order (most recent = week 0)
            sorted_weeks = sorted(weeks_data.keys())
            
            for i, week_num in enumerate(sorted_weeks):
                week_data = weeks_data[week_num]
                
                # Calculate week start date
                week_start = latest_date - timedelta(days=(week_num * 7))
                
                # Base row data
                row = {
                    'week_start': week_start.strftime('%Y-%m-%d'),
                    'developer': developer,
                    'week_number': week_num,
                }
                
                # Add counts for each classification category
                total_commits = sum(week_data.values())
                row['total_commits'] = total_commits
                
                for category in self.classification_categories:
                    count = week_data.get(category, 0)
                    row[f'{category}_count'] = count
                    
                    # Calculate percentage change from previous week
                    if i < len(sorted_weeks) - 1:  # Not the oldest week
                        prev_week_num = sorted_weeks[i + 1]
                        prev_week_data = weeks_data[prev_week_num]
                        prev_count = prev_week_data.get(category, 0)
                        
                        if prev_count > 0:
                            pct_change = ((count - prev_count) / prev_count) * 100
                        elif count > 0:
                            pct_change = 100.0  # New activity
                        else:
                            pct_change = 0.0
                    else:
                        pct_change = 0.0  # No previous data
                    
                    row[f'{category}_pct_change'] = round(pct_change, 1)
                
                rows.append(row)
        
        # Create DataFrame and sort by developer and week
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(['developer', 'week_number'])
        
        # Write to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Generated developer weekly trends CSV: {output_path} ({len(df)} rows)")
    
    def _generate_project_weekly_trends(
        self,
        commits: List[Dict[str, Any]],
        output_path: Path,
        weeks: int
    ) -> None:
        """Generate project weekly classification trends CSV.
        
        WHY: Project-level trends reveal changing development patterns within
        specific codebases, helping identify technical debt accumulation,
        feature development cycles, and maintenance patterns.
        
        Args:
            commits: List of commit data with project and classification info
            output_path: Path to write the CSV file  
            weeks: Number of weeks for trend analysis
        """
        # Group commits by project and week
        project_weeks = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Find the date range for analysis
        if not commits:
            logger.warning("No commits provided for project weekly trends analysis")
            self._write_empty_project_trends_csv(output_path)
            return
        
        # Sort commits by timestamp for consistent week calculation
        sorted_commits = sorted(
            [c for c in commits if c.get('timestamp')],
            key=lambda x: x['timestamp'],
            reverse=True  # Most recent first
        )
        
        if not sorted_commits:
            logger.warning("No commits with timestamps for project weekly trends analysis")
            self._write_empty_project_trends_csv(output_path)
            return
        
        # Calculate week boundaries
        latest_date = sorted_commits[0]['timestamp']
        if hasattr(latest_date, 'date'):
            latest_date = latest_date.date()
        
        # Group commits by project, week, and classification
        for commit in sorted_commits:
            timestamp = commit.get('timestamp')
            if not timestamp:
                continue
            
            # Get week number (0 = current week, 1 = last week, etc.)
            if hasattr(timestamp, 'date'):
                commit_date = timestamp.date()
            else:
                commit_date = timestamp
            
            days_diff = (latest_date - commit_date).days
            week_num = days_diff // 7
            
            # Only include commits within the analysis period
            if week_num >= weeks:
                continue
            
            # Extract project info
            project = commit.get('project_key', 'UNKNOWN')
            
            # Get classification
            classification = self._get_commit_classification(commit)
            
            # Increment count
            project_weeks[project][week_num][classification] += 1
        
        # Build DataFrame
        rows = []
        for project, weeks_data in project_weeks.items():
            # Sort weeks in chronological order (most recent = week 0)
            sorted_weeks = sorted(weeks_data.keys())
            
            for i, week_num in enumerate(sorted_weeks):
                week_data = weeks_data[week_num]
                
                # Calculate week start date
                week_start = latest_date - timedelta(days=(week_num * 7))
                
                # Base row data
                row = {
                    'week_start': week_start.strftime('%Y-%m-%d'),
                    'project': project,
                    'week_number': week_num,
                }
                
                # Add counts for each classification category
                total_commits = sum(week_data.values())
                row['total_commits'] = total_commits
                
                for category in self.classification_categories:
                    count = week_data.get(category, 0)
                    row[f'{category}_count'] = count
                    
                    # Calculate percentage change from previous week
                    if i < len(sorted_weeks) - 1:  # Not the oldest week
                        prev_week_num = sorted_weeks[i + 1]
                        prev_week_data = weeks_data[prev_week_num]
                        prev_count = prev_week_data.get(category, 0)
                        
                        if prev_count > 0:
                            pct_change = ((count - prev_count) / prev_count) * 100
                        elif count > 0:
                            pct_change = 100.0  # New activity
                        else:
                            pct_change = 0.0
                    else:
                        pct_change = 0.0  # No previous data
                    
                    row[f'{category}_pct_change'] = round(pct_change, 1)
                
                rows.append(row)
        
        # Create DataFrame and sort by project and week
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(['project', 'week_number'])
        
        # Write to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Generated project weekly trends CSV: {output_path} ({len(df)} rows)")
    
    def _get_commit_classification(self, commit: Dict[str, Any]) -> str:
        """Extract commit classification from commit data.
        
        WHY: Commits may have classification data in different fields depending
        on the extraction method used (ML vs rule-based vs cached). This method
        provides a consistent way to extract the classification.
        
        DESIGN DECISION: Priority order for classification sources:
        1. predicted_class (from ML classification)
        2. category (from rule-based classification)
        3. 'other' (fallback for unclassified commits)
        
        Args:
            commit: Commit data dictionary
            
        Returns:
            Classification category string
        """
        # Try ML classification first
        if commit.get('predicted_class'):
            return commit['predicted_class']
        
        # Try rule-based classification
        if commit.get('category'):
            return commit['category']
        
        # Try to extract from ticket extractor categorization
        if 'classification' in commit:
            return commit['classification']
        
        # Fallback to 'other'
        return 'other'
    
    def _write_empty_developer_trends_csv(self, output_path: Path) -> None:
        """Write an empty developer trends CSV with proper headers.
        
        Args:
            output_path: Path to write the empty CSV file
        """
        columns = ['week_start', 'developer', 'week_number', 'total_commits']
        
        # Add count and percentage change columns for each category
        for category in self.classification_categories:
            columns.extend([f'{category}_count', f'{category}_pct_change'])
        
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(output_path, index=False)
        logger.info(f"Generated empty developer weekly trends CSV: {output_path}")
    
    def _write_empty_project_trends_csv(self, output_path: Path) -> None:
        """Write an empty project trends CSV with proper headers.
        
        Args:
            output_path: Path to write the empty CSV file
        """
        columns = ['week_start', 'project', 'week_number', 'total_commits']
        
        # Add count and percentage change columns for each category
        for category in self.classification_categories:
            columns.extend([f'{category}_count', f'{category}_pct_change'])
        
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(output_path, index=False)
        logger.info(f"Generated empty project weekly trends CSV: {output_path}")