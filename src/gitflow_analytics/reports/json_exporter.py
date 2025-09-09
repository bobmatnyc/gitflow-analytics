"""Comprehensive JSON export system for GitFlow Analytics.

This module provides a comprehensive JSON export system that consolidates all report data
into a single structured JSON format optimized for web consumption and API integration.

WHY: Traditional CSV reports are excellent for analysis tools but lack the structure needed
for modern web applications and dashboards. This JSON exporter creates a self-contained,
hierarchical data structure that includes:
- Time series data for charts
- Cross-references between entities  
- Anomaly detection and trend analysis
- Health scores and insights
- All existing report data in a unified format

DESIGN DECISIONS:
- Self-contained: All data needed for visualization is included
- Hierarchical: Supports drill-down from executive summary to detailed metrics
- Web-optimized: Compatible with common charting libraries (Chart.js, D3, etc.)
- Extensible: Easy to add new metrics and dimensions
- Consistent: Follows established patterns from existing report generators
"""

import json
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .base import BaseReportGenerator, ReportData, ReportOutput
from .interfaces import ReportFormat

# Get logger for this module
logger = logging.getLogger(__name__)


class ComprehensiveJSONExporter(BaseReportGenerator):
    """Generate comprehensive JSON exports with advanced analytics and insights.
    
    This exporter consolidates all GitFlow Analytics data into a single, structured
    JSON format that's optimized for web consumption and includes:
    
    - Executive summary with key metrics and trends
    - Project-level data with health scores
    - Developer profiles with contribution patterns
    - Time series data for visualization
    - Anomaly detection and alerting
    - Cross-references between entities
    """
    
    def __init__(self, anonymize: bool = False, **kwargs):
        """Initialize the comprehensive JSON exporter.
        
        Args:
            anonymize: Whether to anonymize developer information
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(anonymize=anonymize, **kwargs)
        # Note: anonymization map and counter are now in base class
        
        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            'spike_multiplier': 2.0,  # 2x normal activity = spike
            'drop_threshold': 0.3,    # 30% of normal activity = drop
            'volatility_threshold': 1.5,  # Standard deviation threshold
            'trend_threshold': 0.2    # 20% change = significant trend
        }
        
        # Health score weights
        self.health_weights = {
            'activity_consistency': 0.3,
            'ticket_coverage': 0.25,
            'collaboration': 0.2,
            'code_quality': 0.15,
            'velocity': 0.1
        }
    
    def export_comprehensive_data(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        project_metrics: Dict[str, Any],
        dora_metrics: Dict[str, Any],
        output_path: Path,
        weeks: int = 12,
        pm_data: Optional[Dict[str, Any]] = None,
        qualitative_data: Optional[List[Dict[str, Any]]] = None,
        enhanced_qualitative_analysis: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Export comprehensive analytics data to JSON format.
        
        Args:
            commits: List of commit data
            prs: List of pull request data  
            developer_stats: Developer statistics
            project_metrics: Project-level metrics
            dora_metrics: DORA metrics data
            output_path: Path to write JSON file
            weeks: Number of weeks analyzed
            pm_data: PM platform integration data
            qualitative_data: Qualitative analysis results
            enhanced_qualitative_analysis: Enhanced multi-dimensional qualitative analysis
            
        Returns:
            Path to the generated JSON file
        """
        logger.info(f"Starting comprehensive JSON export with {len(commits)} commits")
        
        # Calculate analysis period
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)
        
        # Build comprehensive data structure
        export_data = {
            "metadata": self._build_metadata(commits, prs, developer_stats, start_date, end_date),
            "executive_summary": self._build_executive_summary(commits, prs, developer_stats, project_metrics, dora_metrics),
            "projects": self._build_project_data(commits, prs, developer_stats, project_metrics),
            "developers": self._build_developer_profiles(commits, developer_stats),
            "workflow_analysis": self._build_workflow_analysis(commits, prs, project_metrics, pm_data),
            "time_series": self._build_time_series_data(commits, prs, weeks),
            "insights": self._build_insights_data(commits, developer_stats, qualitative_data),
            "untracked_analysis": self._build_untracked_analysis(commits, project_metrics),
            "raw_data": self._build_raw_data_summary(commits, prs, developer_stats, dora_metrics)
        }
        
        # Add enhanced qualitative analysis if available
        if enhanced_qualitative_analysis:
            export_data["enhanced_qualitative_analysis"] = enhanced_qualitative_analysis
        
        # Add PM platform data if available
        if pm_data:
            export_data["pm_integration"] = self._build_pm_integration_data(pm_data)
        
        # Serialize and write JSON
        serialized_data = self._serialize_for_json(export_data)
        
        with open(output_path, 'w') as f:
            json.dump(serialized_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comprehensive JSON export written to {output_path}")
        return output_path
    
    def create_enhanced_qualitative_analysis(
        self,
        commits: List[Dict[str, Any]],
        qualitative_data: Optional[List[Any]] = None,
        developer_stats: Optional[List[Dict[str, Any]]] = None,
        project_metrics: Optional[Dict[str, Any]] = None,
        pm_data: Optional[Dict[str, Any]] = None,
        weeks_analyzed: int = 12
    ) -> Optional[Dict[str, Any]]:
        """Create enhanced qualitative analysis using the EnhancedQualitativeAnalyzer.
        
        This method integrates with the enhanced analyzer to generate comprehensive
        qualitative insights across executive, project, developer, and workflow dimensions.
        
        Args:
            commits: List of commit data
            qualitative_data: Optional qualitative commit analysis results
            developer_stats: Optional developer statistics
            project_metrics: Optional project-level metrics
            pm_data: Optional PM platform integration data
            weeks_analyzed: Number of weeks in analysis period
            
        Returns:
            Enhanced qualitative analysis results or None if analyzer unavailable
        """
        try:
            # Import here to avoid circular dependencies
            from ..qualitative.enhanced_analyzer import EnhancedQualitativeAnalyzer

            # Initialize analyzer
            analyzer = EnhancedQualitativeAnalyzer()
            
            # Perform comprehensive analysis
            enhanced_analysis = analyzer.analyze_comprehensive(
                commits=commits,
                qualitative_data=qualitative_data,
                developer_stats=developer_stats,
                project_metrics=project_metrics,
                pm_data=pm_data,
                weeks_analyzed=weeks_analyzed
            )
            
            logger.info("Enhanced qualitative analysis completed successfully")
            return enhanced_analysis
            
        except ImportError as e:
            logger.warning(f"Enhanced qualitative analyzer not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Enhanced qualitative analysis failed: {e}")
            return None
    
    def _build_metadata(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Build metadata section with generation info and data summary."""
        
        # Get unique repositories and projects
        repositories = set()
        projects = set()
        
        for commit in commits:
            if commit.get('repository'):
                repositories.add(commit['repository'])
            if commit.get('project_key'):
                projects.add(commit['project_key'])
        
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "format_version": "2.0.0",
            "generator": "GitFlow Analytics Comprehensive JSON Exporter",
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "weeks_analyzed": (end_date - start_date).days // 7,
                "total_days": (end_date - start_date).days
            },
            "data_summary": {
                "total_commits": len(commits),
                "total_prs": len(prs),
                "total_developers": len(developer_stats),
                "repositories_analyzed": len(repositories),
                "projects_identified": len(projects),
                "repositories": sorted(list(repositories)),
                "projects": sorted(list(projects))
            },
            "export_settings": {
                "anonymized": self.anonymize,
                "timezone": "UTC"
            }
        }
    
    def _build_executive_summary(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        project_metrics: Dict[str, Any],
        dora_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build executive summary with key metrics, trends, and insights."""
        
        # Core metrics
        total_commits = len(commits)
        total_prs = len(prs)
        total_developers = len(developer_stats)
        
        # Calculate lines changed
        total_lines = sum(
            commit.get('filtered_insertions', commit.get('insertions', 0)) +
            commit.get('filtered_deletions', commit.get('deletions', 0))
            for commit in commits
        )
        
        # Story points
        total_story_points = sum(
            commit.get('story_points', 0) or 0 for commit in commits
        )
        
        # Ticket coverage
        ticket_analysis = project_metrics.get('ticket_analysis', {})
        ticket_coverage = ticket_analysis.get('commit_coverage_pct', 0)
        
        # Calculate trends (compare first half vs second half)
        trends = self._calculate_executive_trends(commits, prs)
        
        # Detect anomalies
        anomalies = self._detect_executive_anomalies(commits, developer_stats)
        
        # Identify wins and concerns
        wins, concerns = self._identify_wins_and_concerns(
            commits, developer_stats, project_metrics, dora_metrics
        )
        
        return {
            "key_metrics": {
                "commits": {
                    "total": total_commits,
                    "trend_percent": trends.get('commits_trend', 0),
                    "trend_direction": self._get_trend_direction(trends.get('commits_trend', 0))
                },
                "lines_changed": {
                    "total": total_lines,
                    "trend_percent": trends.get('lines_trend', 0),
                    "trend_direction": self._get_trend_direction(trends.get('lines_trend', 0))
                },
                "story_points": {
                    "total": total_story_points,
                    "trend_percent": trends.get('story_points_trend', 0),
                    "trend_direction": self._get_trend_direction(trends.get('story_points_trend', 0))
                },
                "developers": {
                    "total": total_developers,
                    "active_percentage": self._calculate_active_developer_percentage(developer_stats)
                },
                "pull_requests": {
                    "total": total_prs,
                    "trend_percent": trends.get('prs_trend', 0),
                    "trend_direction": self._get_trend_direction(trends.get('prs_trend', 0))
                },
                "ticket_coverage": {
                    "percentage": round(ticket_coverage, 1),
                    "quality_rating": self._get_coverage_quality_rating(ticket_coverage)
                }
            },
            "performance_indicators": {
                "velocity": {
                    "commits_per_week": round(total_commits / max((len(set(self._get_week_start(c['timestamp']) for c in commits))), 1), 1),
                    "story_points_per_week": round(total_story_points / max((len(set(self._get_week_start(c['timestamp']) for c in commits))), 1), 1)
                },
                "quality": {
                    "avg_commit_size": round(total_lines / max(total_commits, 1), 1),
                    "ticket_coverage_pct": round(ticket_coverage, 1)
                },
                "collaboration": {
                    "developers_per_project": self._calculate_avg_developers_per_project(commits),
                    "cross_project_contributors": self._count_cross_project_contributors(commits, developer_stats)
                }
            },
            "trends": trends,
            "anomalies": anomalies,
            "wins": wins,
            "concerns": concerns,
            "health_score": self._calculate_overall_health_score(commits, developer_stats, project_metrics, dora_metrics)
        }
    
    def _build_project_data(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        project_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build project-level data with health scores and contributor details."""
        
        # Group data by project
        project_data = defaultdict(lambda: {
            'commits': [],
            'prs': [],
            'contributors': set(),
            'lines_changed': 0,
            'story_points': 0,
            'files_changed': set()
        })
        
        # Process commits by project
        for commit in commits:
            project_key = commit.get('project_key', 'UNKNOWN')
            project_data[project_key]['commits'].append(commit)
            project_data[project_key]['contributors'].add(commit.get('canonical_id', commit.get('author_email')))
            
            lines = (
                commit.get('filtered_insertions', commit.get('insertions', 0)) +
                commit.get('filtered_deletions', commit.get('deletions', 0))
            )
            project_data[project_key]['lines_changed'] += lines
            project_data[project_key]['story_points'] += commit.get('story_points', 0) or 0
            
            # Track files (simplified - just count)
            files_changed = commit.get('filtered_files_changed', commit.get('files_changed', 0))
            if files_changed:
                # Add placeholder file references
                for i in range(files_changed):
                    project_data[project_key]['files_changed'].add(f"file_{i}")
        
        # Process PRs by project (if available)
        for pr in prs:
            # Try to determine project from PR data
            project_key = pr.get('project_key', 'UNKNOWN')
            project_data[project_key]['prs'].append(pr)
        
        # Build structured project data
        projects = {}
        
        for project_key, data in project_data.items():
            commits_list = data['commits']
            contributors = data['contributors']
            
            # Calculate project health score
            health_score = self._calculate_project_health_score(commits_list, contributors)
            
            # Get contributor details
            contributor_details = self._get_project_contributor_details(commits_list, developer_stats)
            
            # Calculate project trends
            project_trends = self._calculate_project_trends(commits_list)
            
            # Detect project anomalies
            project_anomalies = self._detect_project_anomalies(commits_list)
            
            projects[project_key] = {
                "summary": {
                    "total_commits": len(commits_list),
                    "total_contributors": len(contributors),
                    "lines_changed": data['lines_changed'],
                    "story_points": data['story_points'],
                    "files_touched": len(data['files_changed']),
                    "pull_requests": len(data['prs'])
                },
                "health_score": health_score,
                "contributors": contributor_details,
                "activity_patterns": {
                    "commits_per_week": self._calculate_weekly_commits(commits_list),
                    "peak_activity_day": self._find_peak_activity_day(commits_list),
                    "commit_size_distribution": self._analyze_commit_size_distribution(commits_list)
                },
                "trends": project_trends,
                "anomalies": project_anomalies,
                "focus_metrics": {
                    "primary_contributors": self._identify_primary_contributors(commits_list, contributor_details),
                    "contribution_distribution": self._calculate_contribution_distribution(commits_list)
                }
            }
        
        return projects
    
    def _build_developer_profiles(
        self,
        commits: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build comprehensive developer profiles with contribution patterns."""
        
        profiles = {}
        
        for dev in developer_stats:
            dev_id = dev['canonical_id']
            dev_name = self._anonymize_value(dev['primary_name'], 'name')
            
            # Get developer's commits
            dev_commits = [c for c in commits if c.get('canonical_id') == dev_id]
            
            # Calculate various metrics
            projects_worked = self._get_developer_projects(dev_commits)
            contribution_patterns = self._analyze_developer_contribution_patterns(dev_commits)
            collaboration_metrics = self._calculate_developer_collaboration_metrics(dev_commits, developer_stats)
            
            # Calculate developer health score
            health_score = self._calculate_developer_health_score(dev_commits, dev)
            
            # Identify achievements and areas for improvement
            achievements = self._identify_developer_achievements(dev_commits, dev)
            improvement_areas = self._identify_improvement_areas(dev_commits, dev)
            
            profiles[dev_id] = {
                "identity": {
                    "name": dev_name,
                    "canonical_id": dev_id,
                    "primary_email": self._anonymize_value(dev['primary_email'], 'email'),
                    "github_username": self._anonymize_value(dev.get('github_username', ''), 'username') if dev.get('github_username') else None,
                    "aliases_count": dev.get('alias_count', 1)
                },
                "summary": {
                    "total_commits": dev['total_commits'],
                    "total_story_points": dev['total_story_points'],
                    "projects_contributed": len(projects_worked),
                    "first_seen": dev.get('first_seen').isoformat() if dev.get('first_seen') else None,
                    "last_seen": dev.get('last_seen').isoformat() if dev.get('last_seen') else None,
                    "days_active": (dev.get('last_seen') - dev.get('first_seen')).days if dev.get('first_seen') and dev.get('last_seen') else 0
                },
                "health_score": health_score,
                "projects": projects_worked,
                "contribution_patterns": contribution_patterns,
                "collaboration": collaboration_metrics,
                "achievements": achievements,
                "improvement_areas": improvement_areas,
                "activity_timeline": self._build_developer_activity_timeline(dev_commits)
            }
        
        return profiles
    
    def _build_workflow_analysis(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        project_metrics: Dict[str, Any],
        pm_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build workflow analysis including Git-PM correlation."""
        
        # Analyze branching patterns
        branching_analysis = self._analyze_branching_patterns(commits)
        
        # Analyze commit patterns
        commit_patterns = self._analyze_commit_timing_patterns(commits)
        
        # Analyze PR workflow if available
        pr_workflow = self._analyze_pr_workflow(prs) if prs else {}
        
        # Git-PM correlation analysis
        git_pm_correlation = {}
        if pm_data:
            git_pm_correlation = self._analyze_git_pm_correlation(commits, pm_data)
        
        return {
            "branching_strategy": branching_analysis,
            "commit_patterns": commit_patterns,
            "pr_workflow": pr_workflow,
            "git_pm_correlation": git_pm_correlation,
            "process_health": {
                "ticket_linking_rate": project_metrics.get('ticket_analysis', {}).get('commit_coverage_pct', 0),
                "merge_commit_rate": self._calculate_merge_commit_rate(commits),
                "commit_message_quality": self._analyze_commit_message_quality(commits)
            }
        }
    
    def _build_time_series_data(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        weeks: int
    ) -> Dict[str, Any]:
        """Build time series data optimized for charting libraries."""
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)
        
        # Generate weekly data points
        weekly_data = self._generate_weekly_time_series(commits, prs, start_date, end_date)
        daily_data = self._generate_daily_time_series(commits, prs, start_date, end_date)
        
        return {
            "weekly": {
                "labels": [d["date"] for d in weekly_data],
                "datasets": {
                    "commits": {
                        "label": "Commits",
                        "data": [d["commits"] for d in weekly_data],
                        "backgroundColor": "rgba(54, 162, 235, 0.2)",
                        "borderColor": "rgba(54, 162, 235, 1)"
                    },
                    "lines_changed": {
                        "label": "Lines Changed", 
                        "data": [d["lines_changed"] for d in weekly_data],
                        "backgroundColor": "rgba(255, 99, 132, 0.2)",
                        "borderColor": "rgba(255, 99, 132, 1)"
                    },
                    "story_points": {
                        "label": "Story Points",
                        "data": [d["story_points"] for d in weekly_data],
                        "backgroundColor": "rgba(75, 192, 192, 0.2)",
                        "borderColor": "rgba(75, 192, 192, 1)"
                    },
                    "active_developers": {
                        "label": "Active Developers",
                        "data": [d["active_developers"] for d in weekly_data],
                        "backgroundColor": "rgba(153, 102, 255, 0.2)",
                        "borderColor": "rgba(153, 102, 255, 1)"
                    }
                }
            },
            "daily": {
                "labels": [d["date"] for d in daily_data],
                "datasets": {
                    "commits": {
                        "label": "Daily Commits",
                        "data": [d["commits"] for d in daily_data],
                        "backgroundColor": "rgba(54, 162, 235, 0.1)",
                        "borderColor": "rgba(54, 162, 235, 1)"
                    }
                }
            }
        }
    
    def _build_insights_data(
        self,
        commits: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        qualitative_data: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Build insights data with qualitative and quantitative analysis."""
        
        # Generate quantitative insights
        quantitative_insights = self._generate_quantitative_insights(commits, developer_stats)
        
        # Process qualitative insights if available
        qualitative_insights = []
        if qualitative_data:
            qualitative_insights = self._process_qualitative_insights(qualitative_data)
        
        # Combine and prioritize insights
        all_insights = quantitative_insights + qualitative_insights
        prioritized_insights = self._prioritize_insights(all_insights)
        
        return {
            "quantitative": quantitative_insights,
            "qualitative": qualitative_insights,
            "prioritized": prioritized_insights,
            "insight_categories": self._categorize_insights(all_insights),
            "actionable_recommendations": self._generate_actionable_recommendations(all_insights)
        }
    
    def _build_raw_data_summary(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        dora_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build summary of raw data for reference and validation."""
        
        return {
            "commits_sample": commits[:5] if commits else [],  # First 5 commits as sample
            "prs_sample": prs[:3] if prs else [],  # First 3 PRs as sample
            "developer_stats_schema": {
                "fields": list(developer_stats[0].keys()) if developer_stats else [],
                "sample_record": developer_stats[0] if developer_stats else {}
            },
            "dora_metrics": dora_metrics,
            "data_quality": {
                "commits_with_timestamps": sum(1 for c in commits if c.get('timestamp')),
                "commits_with_projects": sum(1 for c in commits if c.get('project_key')),
                "commits_with_tickets": sum(1 for c in commits if c.get('ticket_references')),
                "developers_with_github": sum(1 for d in developer_stats if d.get('github_username'))
            }
        }
    
    def _build_pm_integration_data(self, pm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build PM platform integration data summary."""
        
        metrics = pm_data.get('metrics', {})
        correlations = pm_data.get('correlations', [])
        
        return {
            "platforms": list(metrics.get('platform_coverage', {}).keys()),
            "total_issues": metrics.get('total_pm_issues', 0),
            "story_point_coverage": metrics.get('story_point_analysis', {}).get('story_point_coverage_pct', 0),
            "correlations_count": len(correlations),
            "correlation_quality": metrics.get('correlation_quality', {}),
            "issue_types": metrics.get('issue_type_distribution', {}),
            "platform_summary": {
                platform: {
                    "total_issues": data.get('total_issues', 0),
                    "linked_issues": data.get('linked_issues', 0),
                    "coverage_percentage": data.get('coverage_percentage', 0)
                }
                for platform, data in metrics.get('platform_coverage', {}).items()
            }
        }
    
    # Helper methods for calculations and analysis
    
    def _calculate_executive_trends(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate trends by comparing first half vs second half of data."""
        
        if not commits:
            return {}
        
        # Sort commits by timestamp
        sorted_commits = sorted(commits, key=lambda x: x['timestamp'])
        midpoint = len(sorted_commits) // 2
        
        first_half = sorted_commits[:midpoint]
        second_half = sorted_commits[midpoint:]
        
        # Calculate metrics for each half
        def get_half_metrics(commit_list):
            return {
                'commits': len(commit_list),
                'lines': sum(
                    c.get('filtered_insertions', c.get('insertions', 0)) +
                    c.get('filtered_deletions', c.get('deletions', 0))
                    for c in commit_list
                ),
                'story_points': sum(c.get('story_points', 0) or 0 for c in commit_list)
            }
        
        first_metrics = get_half_metrics(first_half)
        second_metrics = get_half_metrics(second_half)
        
        # Calculate percentage changes
        trends = {}
        for metric in ['commits', 'lines', 'story_points']:
            if first_metrics[metric] > 0:
                change = ((second_metrics[metric] - first_metrics[metric]) / first_metrics[metric]) * 100
                trends[f'{metric}_trend'] = round(change, 1)
            else:
                trends[f'{metric}_trend'] = 0
        
        # PR trends if available
        if prs:
            sorted_prs = sorted(prs, key=lambda x: x.get('merged_at', x.get('created_at', datetime.now())))
            pr_midpoint = len(sorted_prs) // 2
            
            first_pr_count = pr_midpoint
            second_pr_count = len(sorted_prs) - pr_midpoint
            
            if first_pr_count > 0:
                pr_change = ((second_pr_count - first_pr_count) / first_pr_count) * 100
                trends['prs_trend'] = round(pr_change, 1)
            else:
                trends['prs_trend'] = 0
        
        return trends
    
    def _detect_executive_anomalies(
        self,
        commits: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in executive-level data."""
        
        anomalies = []
        
        # Check for commit spikes/drops by week
        weekly_commits = self._get_weekly_commit_counts(commits)
        if len(weekly_commits) >= 3:
            mean_commits = statistics.mean(weekly_commits)
            std_commits = statistics.pstdev(weekly_commits) if len(weekly_commits) > 1 else 0
            
            for i, count in enumerate(weekly_commits):
                if std_commits > 0:
                    if count > mean_commits + (std_commits * self.anomaly_thresholds['spike_multiplier']):
                        anomalies.append({
                            "type": "spike",
                            "metric": "weekly_commits",
                            "value": count,
                            "expected": round(mean_commits, 1),
                            "severity": "high" if count > mean_commits + (std_commits * 3) else "medium",
                            "week_index": i
                        })
                    elif count < mean_commits * self.anomaly_thresholds['drop_threshold']:
                        anomalies.append({
                            "type": "drop",
                            "metric": "weekly_commits", 
                            "value": count,
                            "expected": round(mean_commits, 1),
                            "severity": "high" if count < mean_commits * 0.1 else "medium",
                            "week_index": i
                        })
        
        # Check for contributor anomalies
        commit_counts = [dev['total_commits'] for dev in developer_stats]
        if len(commit_counts) > 1:
            gini_coefficient = self._calculate_gini_coefficient(commit_counts)
            if gini_coefficient > 0.8:
                anomalies.append({
                    "type": "concentration",
                    "metric": "contribution_distribution",
                    "value": round(gini_coefficient, 2),
                    "threshold": 0.8,
                    "severity": "medium",
                    "description": "Highly concentrated contribution pattern"
                })
        
        return anomalies
    
    def _identify_wins_and_concerns(
        self,
        commits: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        project_metrics: Dict[str, Any],
        dora_metrics: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Identify key wins and concerns from the data."""
        
        wins = []
        concerns = []
        
        # Ticket coverage analysis
        ticket_coverage = project_metrics.get('ticket_analysis', {}).get('commit_coverage_pct', 0)
        if ticket_coverage > 80:
            wins.append({
                "category": "process",
                "title": "Excellent Ticket Coverage",
                "description": f"{ticket_coverage:.1f}% of commits linked to tickets",
                "impact": "high"
            })
        elif ticket_coverage < 30:
            concerns.append({
                "category": "process",
                "title": "Low Ticket Coverage",
                "description": f"Only {ticket_coverage:.1f}% of commits linked to tickets",
                "impact": "high",
                "recommendation": "Improve ticket referencing in commit messages"
            })
        
        # Team activity analysis
        if len(developer_stats) > 1:
            commit_counts = [dev['total_commits'] for dev in developer_stats]
            avg_commits = sum(commit_counts) / len(commit_counts)
            
            if min(commit_counts) > avg_commits * 0.5:
                wins.append({
                    "category": "team",
                    "title": "Balanced Team Contributions",
                    "description": "All team members are actively contributing",
                    "impact": "medium"
                })
            elif max(commit_counts) > avg_commits * 3:
                concerns.append({
                    "category": "team",
                    "title": "Unbalanced Contributions",
                    "description": "Work is heavily concentrated among few developers",
                    "impact": "medium",
                    "recommendation": "Consider distributing work more evenly"
                })
        
        # Code quality indicators
        total_lines = sum(
            c.get('filtered_insertions', c.get('insertions', 0)) +
            c.get('filtered_deletions', c.get('deletions', 0))
            for c in commits
        )
        avg_commit_size = total_lines / max(len(commits), 1)
        
        if 20 <= avg_commit_size <= 200:
            wins.append({
                "category": "quality",
                "title": "Optimal Commit Size",
                "description": f"Average commit size of {avg_commit_size:.0f} lines indicates good change management",
                "impact": "low"
            })
        elif avg_commit_size > 500:
            concerns.append({
                "category": "quality",
                "title": "Large Commit Sizes",
                "description": f"Average commit size of {avg_commit_size:.0f} lines may indicate batched changes",
                "impact": "low",
                "recommendation": "Consider breaking down changes into smaller commits"
            })
        
        return wins, concerns
    
    def _calculate_overall_health_score(
        self,
        commits: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        project_metrics: Dict[str, Any],
        dora_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall project health score."""
        
        scores = {}
        
        # Activity consistency score (0-100)
        weekly_commits = self._get_weekly_commit_counts(commits)
        if weekly_commits:
            consistency = max(0, 100 - (statistics.pstdev(weekly_commits) / max(statistics.mean(weekly_commits), 1) * 100))
            scores['activity_consistency'] = min(100, consistency)
        else:
            scores['activity_consistency'] = 0
        
        # Ticket coverage score
        ticket_coverage = project_metrics.get('ticket_analysis', {}).get('commit_coverage_pct', 0)
        scores['ticket_coverage'] = min(100, ticket_coverage)
        
        # Collaboration score (based on multi-project work and team balance)
        if len(developer_stats) > 1:
            commit_counts = [dev['total_commits'] for dev in developer_stats]
            gini = self._calculate_gini_coefficient(commit_counts)
            collaboration_score = max(0, 100 - (gini * 100))
            scores['collaboration'] = collaboration_score
        else:
            scores['collaboration'] = 50  # Neutral for single developer
        
        # Code quality score (based on commit size and patterns)
        total_lines = sum(
            c.get('filtered_insertions', c.get('insertions', 0)) +
            c.get('filtered_deletions', c.get('deletions', 0))
            for c in commits
        )
        avg_commit_size = total_lines / max(len(commits), 1)
        
        # Optimal range is 20-200 lines per commit
        if 20 <= avg_commit_size <= 200:
            quality_score = 100
        elif avg_commit_size < 20:
            quality_score = max(0, (avg_commit_size / 20) * 100)
        else:
            quality_score = max(0, 100 - ((avg_commit_size - 200) / 500 * 100))
        
        scores['code_quality'] = min(100, quality_score)
        
        # Velocity score (commits per week vs. baseline)
        weeks_with_activity = len([w for w in weekly_commits if w > 0])
        velocity_score = min(100, (weeks_with_activity / max(len(weekly_commits), 1)) * 100)
        scores['velocity'] = velocity_score
        
        # Calculate weighted overall score
        overall_score = sum(
            scores.get(metric, 0) * weight
            for metric, weight in self.health_weights.items()
        )
        
        return {
            "overall": round(overall_score, 1),
            "components": {k: round(v, 1) for k, v in scores.items()},
            "weights": self.health_weights,
            "rating": self._get_health_rating(overall_score)
        }
    
    def _get_health_rating(self, score: float) -> str:
        """Get health rating based on score."""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        else:
            return "needs_improvement"
    
    def _get_trend_direction(self, trend_percent: float) -> str:
        """Get trend direction from percentage change."""
        if abs(trend_percent) < self.anomaly_thresholds['trend_threshold'] * 100:
            return "stable"
        elif trend_percent > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _get_coverage_quality_rating(self, coverage: float) -> str:
        """Get quality rating for ticket coverage."""
        if coverage >= 80:
            return "excellent"
        elif coverage >= 60:
            return "good"
        elif coverage >= 40:
            return "fair"
        else:
            return "poor"
    
    def _calculate_active_developer_percentage(self, developer_stats: List[Dict[str, Any]]) -> float:
        """Calculate percentage of developers with meaningful activity."""
        if not developer_stats:
            return 0
        
        total_commits = sum(dev['total_commits'] for dev in developer_stats)
        avg_commits = total_commits / len(developer_stats)
        threshold = max(1, avg_commits * 0.1)  # 10% of average
        
        active_developers = sum(1 for dev in developer_stats if dev['total_commits'] >= threshold)
        return round((active_developers / len(developer_stats)) * 100, 1)
    
    def _calculate_avg_developers_per_project(self, commits: List[Dict[str, Any]]) -> float:
        """Calculate average number of developers per project."""
        project_developers = defaultdict(set)
        
        for commit in commits:
            project_key = commit.get('project_key', 'UNKNOWN')
            dev_id = commit.get('canonical_id', commit.get('author_email'))
            project_developers[project_key].add(dev_id)
        
        if not project_developers:
            return 0
        
        avg = sum(len(devs) for devs in project_developers.values()) / len(project_developers)
        return round(avg, 1)
    
    def _count_cross_project_contributors(
        self,
        commits: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]]
    ) -> int:
        """Count developers who contribute to multiple projects."""
        developer_projects = defaultdict(set)
        
        for commit in commits:
            project_key = commit.get('project_key', 'UNKNOWN')
            dev_id = commit.get('canonical_id', commit.get('author_email'))
            developer_projects[dev_id].add(project_key)
        
        return sum(1 for projects in developer_projects.values() if len(projects) > 1)
    
    def _calculate_project_health_score(
        self,
        commits: List[Dict[str, Any]],
        contributors: Set[str]
    ) -> Dict[str, Any]:
        """Calculate health score for a specific project."""
        
        if not commits:
            return {"overall": 0, "components": {}, "rating": "no_data"}
        
        scores = {}
        
        # Activity score (commits per week)
        weekly_commits = self._get_weekly_commit_counts(commits)
        if weekly_commits:
            avg_weekly = statistics.mean(weekly_commits)
            activity_score = min(100, avg_weekly * 10)  # Scale appropriately
            scores['activity'] = activity_score
        else:
            scores['activity'] = 0
        
        # Contributor diversity score
        if len(contributors) == 1:
            diversity_score = 30  # Single contributor is risky
        elif len(contributors) <= 3:
            diversity_score = 60
        else:
            diversity_score = 100
        scores['contributor_diversity'] = diversity_score
        
        # Consistency score
        if len(weekly_commits) > 1:
            consistency = max(0, 100 - (statistics.pstdev(weekly_commits) / max(statistics.mean(weekly_commits), 1) * 50))
            scores['consistency'] = consistency
        else:
            scores['consistency'] = 50
        
        # Overall score (equal weights for now)
        overall_score = sum(scores.values()) / len(scores)
        
        return {
            "overall": round(overall_score, 1),
            "components": {k: round(v, 1) for k, v in scores.items()},
            "rating": self._get_health_rating(overall_score)
        }
    
    def _get_project_contributor_details(
        self,
        commits: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get detailed contributor information for a project."""
        
        # Create developer lookup
        dev_lookup = {dev['canonical_id']: dev for dev in developer_stats}
        
        # Count contributions per developer
        contributor_commits = defaultdict(int)
        contributor_lines = defaultdict(int)
        
        for commit in commits:
            dev_id = commit.get('canonical_id', commit.get('author_email'))
            contributor_commits[dev_id] += 1
            
            lines = (
                commit.get('filtered_insertions', commit.get('insertions', 0)) +
                commit.get('filtered_deletions', commit.get('deletions', 0))
            )
            contributor_lines[dev_id] += lines
        
        # Build contributor details
        contributors = []
        total_commits = len(commits)
        
        for dev_id, commit_count in contributor_commits.items():
            dev = dev_lookup.get(dev_id, {})
            
            contributors.append({
                "id": dev_id,
                "name": self._anonymize_value(dev.get('primary_name', 'Unknown'), 'name'),
                "commits": commit_count,
                "commits_percentage": round((commit_count / total_commits) * 100, 1),
                "lines_changed": contributor_lines[dev_id],
                "role": self._determine_contributor_role(commit_count, total_commits)
            })
        
        # Sort by commits descending
        contributors.sort(key=lambda x: x['commits'], reverse=True)
        
        return contributors
    
    def _determine_contributor_role(self, commits: int, total_commits: int) -> str:
        """Determine contributor role based on contribution percentage."""
        percentage = (commits / total_commits) * 100
        
        if percentage >= 50:
            return "primary"
        elif percentage >= 25:
            return "major"
        elif percentage >= 10:
            return "regular"
        else:
            return "occasional"
    
    def _calculate_project_trends(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trends for a specific project."""
        
        if len(commits) < 4:  # Need sufficient data for trends
            return {"insufficient_data": True}
        
        # Sort by timestamp
        sorted_commits = sorted(commits, key=lambda x: x['timestamp'])
        
        # Split into quarters for trend analysis
        quarter_size = len(sorted_commits) // 4
        quarters = [
            sorted_commits[i*quarter_size:(i+1)*quarter_size]
            for i in range(4)
        ]
        
        # Handle remainder commits
        if len(sorted_commits) % 4:
            quarters[-1].extend(sorted_commits[4*quarter_size:])
        
        # Calculate metrics per quarter
        quarter_metrics = []
        for quarter in quarters:
            metrics = {
                'commits': len(quarter),
                'lines': sum(
                    c.get('filtered_insertions', c.get('insertions', 0)) +
                    c.get('filtered_deletions', c.get('deletions', 0))
                    for c in quarter
                ),
                'contributors': len(set(c.get('canonical_id', c.get('author_email')) for c in quarter))
            }
            quarter_metrics.append(metrics)
        
        # Calculate trends (compare Q1 vs Q4)
        trends = {}
        for metric in ['commits', 'lines', 'contributors']:
            q1_value = quarter_metrics[0][metric]
            q4_value = quarter_metrics[-1][metric]
            
            if q1_value > 0:
                change = ((q4_value - q1_value) / q1_value) * 100
                trends[f'{metric}_trend'] = round(change, 1)
            else:
                trends[f'{metric}_trend'] = 0
        
        return trends
    
    def _detect_project_anomalies(self, commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in project-specific data."""
        
        if len(commits) < 7:  # Need sufficient data
            return []
        
        anomalies = []
        
        # Get daily commit counts
        daily_commits = self._get_daily_commit_counts(commits)
        
        if len(daily_commits) >= 7:
            mean_daily = statistics.mean(daily_commits)
            std_daily = statistics.pstdev(daily_commits) if len(daily_commits) > 1 else 0
            
            # Find days with unusual activity
            for i, count in enumerate(daily_commits):
                if std_daily > 0 and count > mean_daily + (std_daily * 2):
                    anomalies.append({
                        "type": "activity_spike",
                        "value": count,
                        "expected": round(mean_daily, 1),
                        "day_index": i,
                        "severity": "medium"
                    })
        
        return anomalies
    
    def _identify_primary_contributors(
        self,
        commits: List[Dict[str, Any]],
        contributor_details: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify primary contributors (top 80% of activity)."""
        
        sorted_contributors = sorted(contributor_details, key=lambda x: x['commits'], reverse=True)
        total_commits = sum(c['commits'] for c in contributor_details)
        
        primary_contributors = []
        cumulative_commits = 0
        
        for contributor in sorted_contributors:
            cumulative_commits += contributor['commits']
            primary_contributors.append(contributor['name'])
            
            if cumulative_commits >= total_commits * 0.8:
                break
        
        return primary_contributors
    
    def _calculate_contribution_distribution(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate distribution metrics for contributions."""
        
        contributor_commits = defaultdict(int)
        for commit in commits:
            dev_id = commit.get('canonical_id', commit.get('author_email'))
            contributor_commits[dev_id] += 1
        
        commit_counts = list(contributor_commits.values())
        
        if not commit_counts:
            return {}
        
        gini = self._calculate_gini_coefficient(commit_counts)
        
        return {
            "gini_coefficient": round(gini, 3),
            "concentration_level": "high" if gini > 0.7 else "medium" if gini > 0.4 else "low",
            "top_contributor_percentage": round((max(commit_counts) / sum(commit_counts)) * 100, 1),
            "contributor_count": len(commit_counts)
        }
    
    def _get_developer_projects(self, commits: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Get projects a developer has worked on with contribution details."""
        
        project_contributions = defaultdict(lambda: {
            'commits': 0,
            'lines_changed': 0,
            'story_points': 0,
            'first_commit': None,
            'last_commit': None
        })
        
        for commit in commits:
            project_key = commit.get('project_key', 'UNKNOWN')
            project_data = project_contributions[project_key]
            
            project_data['commits'] += 1
            
            lines = (
                commit.get('filtered_insertions', commit.get('insertions', 0)) +
                commit.get('filtered_deletions', commit.get('deletions', 0))
            )
            project_data['lines_changed'] += lines
            project_data['story_points'] += commit.get('story_points', 0) or 0
            
            # Track first and last commits
            commit_date = commit['timestamp']
            if not project_data['first_commit'] or commit_date < project_data['first_commit']:
                project_data['first_commit'] = commit_date
            if not project_data['last_commit'] or commit_date > project_data['last_commit']:
                project_data['last_commit'] = commit_date
        
        # Convert to regular dict and add percentages
        total_commits = len(commits)
        projects = {}
        
        for project_key, data in project_contributions.items():
            projects[project_key] = {
                'commits': data['commits'],
                'commits_percentage': round((data['commits'] / total_commits) * 100, 1),
                'lines_changed': data['lines_changed'],
                'story_points': data['story_points'],
                'first_commit': data['first_commit'].isoformat() if data['first_commit'] else None,
                'last_commit': data['last_commit'].isoformat() if data['last_commit'] else None,
                'days_active': (data['last_commit'] - data['first_commit']).days if data['first_commit'] and data['last_commit'] else 0
            }
        
        return projects
    
    def _analyze_developer_contribution_patterns(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a developer's contribution patterns."""
        
        if not commits:
            return {}
        
        # Time-based patterns (use local hour if available)
        commit_hours = []
        for c in commits:
            if 'local_hour' in c:
                commit_hours.append(c['local_hour'])
            elif hasattr(c['timestamp'], 'hour'):
                commit_hours.append(c['timestamp'].hour)
        
        commit_days = [c['timestamp'].weekday() for c in commits if hasattr(c['timestamp'], 'weekday')]
        
        # Size patterns
        commit_sizes = []
        for commit in commits:
            lines = (
                commit.get('filtered_insertions', commit.get('insertions', 0)) +
                commit.get('filtered_deletions', commit.get('deletions', 0))
            )
            commit_sizes.append(lines)
        
        patterns = {
            'total_commits': len(commits),
            'avg_commit_size': round(statistics.mean(commit_sizes), 1) if commit_sizes else 0,
            'commit_size_stddev': round(statistics.pstdev(commit_sizes), 1) if len(commit_sizes) > 1 else 0
        }
        
        if commit_hours:
            patterns['peak_hour'] = max(set(commit_hours), key=commit_hours.count)
            patterns['time_distribution'] = self._get_time_distribution_pattern(commit_hours)
        
        if commit_days:
            patterns['peak_day'] = self._get_day_name(max(set(commit_days), key=commit_days.count))
            patterns['work_pattern'] = self._get_work_pattern(commit_days)
        
        # Consistency patterns
        weekly_commits = self._get_weekly_commit_counts(commits)
        if len(weekly_commits) > 1:
            patterns['consistency_score'] = round(100 - (statistics.pstdev(weekly_commits) / max(statistics.mean(weekly_commits), 1) * 100), 1)
        else:
            patterns['consistency_score'] = 50
        
        return patterns
    
    def _get_time_distribution_pattern(self, hours: List[int]) -> str:
        """Determine time distribution pattern from commit hours."""
        avg_hour = statistics.mean(hours)
        
        if avg_hour < 10:
            return "early_bird"
        elif avg_hour < 14:
            return "morning_focused"
        elif avg_hour < 18:
            return "afternoon_focused"
        else:
            return "night_owl"
    
    def _get_day_name(self, day_index: int) -> str:
        """Convert day index to day name."""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return days[day_index] if 0 <= day_index < 7 else 'Unknown'
    
    def _get_work_pattern(self, days: List[int]) -> str:
        """Determine work pattern from commit days."""
        weekday_commits = sum(1 for day in days if day < 5)  # Mon-Fri
        weekend_commits = sum(1 for day in days if day >= 5)  # Sat-Sun
        
        total = len(days)
        weekday_pct = (weekday_commits / total) * 100 if total > 0 else 0
        
        if weekday_pct > 90:
            return "strictly_weekdays"
        elif weekday_pct > 75:
            return "mostly_weekdays"
        elif weekday_pct > 50:
            return "mixed_schedule"
        else:
            return "weekend_warrior"
    
    def _calculate_developer_collaboration_metrics(
        self,
        commits: List[Dict[str, Any]],
        all_developer_stats: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate collaboration metrics for a developer."""
        
        # Get projects this developer worked on
        dev_projects = set(c.get('project_key', 'UNKNOWN') for c in commits)
        
        # Find other developers on same projects
        collaborators = set()
        for dev in all_developer_stats:
            dev_id = dev['canonical_id']
            # Simple check - assumes we can identify overlapping work
            # In real implementation, would need more sophisticated analysis
            if len(dev_projects) > 0:  # Placeholder logic
                collaborators.add(dev_id)
        
        # Remove self from collaborators
        dev_id = commits[0].get('canonical_id') if commits else None
        collaborators.discard(dev_id)
        
        return {
            'projects_count': len(dev_projects),
            'potential_collaborators': len(collaborators),
            'cross_project_work': len(dev_projects) > 1,
            'collaboration_score': min(100, len(collaborators) * 10)  # Simple scoring
        }
    
    def _calculate_developer_health_score(
        self,
        commits: List[Dict[str, Any]],
        dev_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate health score for a specific developer."""
        
        if not commits:
            return {"overall": 0, "components": {}, "rating": "no_data"}
        
        scores = {}
        
        # Activity score based on commits per week
        weekly_commits = self._get_weekly_commit_counts(commits)
        if weekly_commits:
            avg_weekly = statistics.mean(weekly_commits)
            activity_score = min(100, avg_weekly * 20)  # Scale appropriately
            scores['activity'] = activity_score
        else:
            scores['activity'] = 0
        
        # Consistency score
        if len(weekly_commits) > 1:
            consistency = max(0, 100 - (statistics.pstdev(weekly_commits) / max(statistics.mean(weekly_commits), 1) * 50))
            scores['consistency'] = consistency
        else:
            scores['consistency'] = 50
        
        # Engagement score (based on projects and commit sizes)
        project_count = len(set(c.get('project_key', 'UNKNOWN') for c in commits))
        engagement_score = min(100, project_count * 25 + 25)  # Bonus for multi-project work
        scores['engagement'] = engagement_score
        
        # Overall score
        overall_score = sum(scores.values()) / len(scores)
        
        return {
            "overall": round(overall_score, 1),
            "components": {k: round(v, 1) for k, v in scores.items()},
            "rating": self._get_health_rating(overall_score)
        }
    
    def _identify_developer_achievements(
        self,
        commits: List[Dict[str, Any]],
        dev_stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify achievements for a developer."""
        
        achievements = []
        
        # High commit count
        if dev_stats['total_commits'] > 50:
            achievements.append({
                "type": "productivity",
                "title": "High Productivity",
                "description": f"{dev_stats['total_commits']} commits in analysis period",
                "badge": "prolific_contributor"
            })
        
        # Multi-project contributor
        projects = set(c.get('project_key', 'UNKNOWN') for c in commits)
        if len(projects) > 3:
            achievements.append({
                "type": "versatility",
                "title": "Multi-Project Contributor",
                "description": f"Contributed to {len(projects)} projects",
                "badge": "versatile_developer"
            })
        
        # Consistent contributor
        weekly_commits = self._get_weekly_commit_counts(commits)
        if len(weekly_commits) > 4:
            active_weeks = sum(1 for w in weekly_commits if w > 0)
            consistency_rate = active_weeks / len(weekly_commits)
            
            if consistency_rate > 0.8:
                achievements.append({
                    "type": "consistency",
                    "title": "Consistent Contributor",
                    "description": f"Active in {active_weeks} out of {len(weekly_commits)} weeks",
                    "badge": "reliable_contributor"
                })
        
        return achievements
    
    def _identify_improvement_areas(
        self,
        commits: List[Dict[str, Any]],
        dev_stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify areas for improvement for a developer."""
        
        improvements = []
        
        # Check ticket linking
        commits_with_tickets = sum(1 for c in commits if c.get('ticket_references'))
        ticket_rate = (commits_with_tickets / len(commits)) * 100 if commits else 0
        
        if ticket_rate < 50:
            improvements.append({
                "category": "process",
                "title": "Improve Ticket Linking",
                "description": f"Only {ticket_rate:.1f}% of commits reference tickets",
                "priority": "medium",
                "suggestion": "Include ticket references in commit messages"
            })
        
        # Check commit size consistency
        commit_sizes = []
        for commit in commits:
            lines = (
                commit.get('filtered_insertions', commit.get('insertions', 0)) +
                commit.get('filtered_deletions', commit.get('deletions', 0))
            )
            commit_sizes.append(lines)
        
        if commit_sizes and len(commit_sizes) > 5:
            avg_size = statistics.mean(commit_sizes)
            if avg_size > 300:
                improvements.append({
                    "category": "quality",
                    "title": "Consider Smaller Commits",
                    "description": f"Average commit size is {avg_size:.0f} lines",
                    "priority": "low",
                    "suggestion": "Break down large changes into smaller, focused commits"
                })
        
        return improvements
    
    def _build_developer_activity_timeline(self, commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build activity timeline for a developer."""
        
        if not commits:
            return []
        
        # Group commits by week
        weekly_activity = defaultdict(lambda: {
            'commits': 0,
            'lines_changed': 0,
            'projects': set()
        })
        
        for commit in commits:
            week_start = self._get_week_start(commit['timestamp'])
            week_key = week_start.strftime('%Y-%m-%d')
            
            weekly_activity[week_key]['commits'] += 1
            
            lines = (
                commit.get('filtered_insertions', commit.get('insertions', 0)) +
                commit.get('filtered_deletions', commit.get('deletions', 0))
            )
            weekly_activity[week_key]['lines_changed'] += lines
            weekly_activity[week_key]['projects'].add(commit.get('project_key', 'UNKNOWN'))
        
        # Convert to timeline format
        timeline = []
        for week_key in sorted(weekly_activity.keys()):
            data = weekly_activity[week_key]
            timeline.append({
                'week': week_key,
                'commits': data['commits'],
                'lines_changed': data['lines_changed'],
                'projects': len(data['projects']),
                'project_list': sorted(list(data['projects']))
            })
        
        return timeline
    
    def _analyze_branching_patterns(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze branching and merge patterns."""
        
        merge_commits = sum(1 for c in commits if c.get('is_merge'))
        total_commits = len(commits)
        
        merge_rate = (merge_commits / total_commits) * 100 if total_commits > 0 else 0
        
        # Determine branching strategy
        if merge_rate < 5:
            strategy = "linear"
        elif merge_rate < 15:
            strategy = "feature_branches"
        elif merge_rate < 30:
            strategy = "git_flow"
        else:
            strategy = "complex_branching"
        
        return {
            "merge_commits": merge_commits,
            "merge_rate_percent": round(merge_rate, 1),
            "strategy": strategy,
            "complexity_rating": "low" if merge_rate < 15 else "medium" if merge_rate < 30 else "high"
        }
    
    def _analyze_commit_timing_patterns(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze when commits typically happen."""
        
        if not commits:
            return {}
        
        # Extract timing data
        hours = []
        days = []
        
        for commit in commits:
            timestamp = commit['timestamp']
            # Use local hour if available
            if 'local_hour' in commit:
                hours.append(commit['local_hour'])
            elif hasattr(timestamp, 'hour'):
                hours.append(timestamp.hour)
            if hasattr(timestamp, 'weekday'):
                days.append(timestamp.weekday())
        
        patterns = {}
        
        if hours:
            # Hour distribution
            hour_counts = defaultdict(int)
            for hour in hours:
                hour_counts[hour] += 1
            
            peak_hour = max(hour_counts, key=hour_counts.get)
            patterns['peak_hour'] = peak_hour
            patterns['peak_hour_commits'] = hour_counts[peak_hour]
            
            # Time periods
            morning = sum(1 for h in hours if 6 <= h < 12)
            afternoon = sum(1 for h in hours if 12 <= h < 18)
            evening = sum(1 for h in hours if 18 <= h < 24)
            night = sum(1 for h in hours if 0 <= h < 6)
            
            total = len(hours)
            patterns['time_distribution'] = {
                'morning_pct': round((morning / total) * 100, 1),
                'afternoon_pct': round((afternoon / total) * 100, 1),
                'evening_pct': round((evening / total) * 100, 1),
                'night_pct': round((night / total) * 100, 1)
            }
        
        if days:
            # Day distribution
            day_counts = defaultdict(int)
            for day in days:
                day_counts[day] += 1
            
            peak_day = max(day_counts, key=day_counts.get)
            patterns['peak_day'] = self._get_day_name(peak_day)
            patterns['peak_day_commits'] = day_counts[peak_day]
            
            # Weekday vs weekend
            weekday_commits = sum(1 for d in days if d < 5)
            weekend_commits = sum(1 for d in days if d >= 5)
            
            total = len(days)
            patterns['weekday_pct'] = round((weekday_commits / total) * 100, 1)
            patterns['weekend_pct'] = round((weekend_commits / total) * 100, 1)
        
        return patterns
    
    def _analyze_pr_workflow(self, prs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze pull request workflow patterns."""
        
        if not prs:
            return {}
        
        # PR lifecycle analysis
        lifetimes = []
        sizes = []
        review_counts = []
        
        for pr in prs:
            # Calculate PR lifetime
            created = pr.get('created_at')
            merged = pr.get('merged_at')
            
            if created and merged:
                if isinstance(created, str):
                    created = datetime.fromisoformat(created.replace('Z', '+00:00'))
                if isinstance(merged, str):
                    merged = datetime.fromisoformat(merged.replace('Z', '+00:00'))
                
                lifetime_hours = (merged - created).total_seconds() / 3600
                lifetimes.append(lifetime_hours)
            
            # PR size (additions + deletions)
            additions = pr.get('additions', 0)
            deletions = pr.get('deletions', 0)
            sizes.append(additions + deletions)
            
            # Review comments
            review_comments = pr.get('review_comments', 0)
            review_counts.append(review_comments)
        
        workflow = {}
        
        if lifetimes:
            workflow['avg_lifetime_hours'] = round(statistics.mean(lifetimes), 1)
            workflow['median_lifetime_hours'] = round(statistics.median(lifetimes), 1)
        
        if sizes:
            workflow['avg_pr_size'] = round(statistics.mean(sizes), 1)
            workflow['median_pr_size'] = round(statistics.median(sizes), 1)
        
        if review_counts:
            workflow['avg_review_comments'] = round(statistics.mean(review_counts), 1)
            workflow['prs_with_reviews'] = sum(1 for r in review_counts if r > 0)
            workflow['review_rate_pct'] = round((workflow['prs_with_reviews'] / len(prs)) * 100, 1)
        
        return workflow
    
    def _analyze_git_pm_correlation(
        self,
        commits: List[Dict[str, Any]],
        pm_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze correlation between Git activity and PM platform data."""
        
        correlations = pm_data.get('correlations', [])
        metrics = pm_data.get('metrics', {})
        
        if not correlations:
            return {"status": "no_correlations"}
        
        # Analyze correlation quality
        high_confidence = sum(1 for c in correlations if c.get('confidence', 0) > 0.8)
        medium_confidence = sum(1 for c in correlations if 0.5 <= c.get('confidence', 0) <= 0.8)
        low_confidence = sum(1 for c in correlations if c.get('confidence', 0) < 0.5)
        
        total_correlations = len(correlations)
        
        # Analyze correlation methods
        methods = defaultdict(int)
        for c in correlations:
            method = c.get('correlation_method', 'unknown')
            methods[method] += 1
        
        # Story point accuracy analysis
        story_analysis = metrics.get('story_point_analysis', {})
        
        return {
            "total_correlations": total_correlations,
            "confidence_distribution": {
                "high": high_confidence,
                "medium": medium_confidence,
                "low": low_confidence
            },
            "confidence_rates": {
                "high_pct": round((high_confidence / total_correlations) * 100, 1),
                "medium_pct": round((medium_confidence / total_correlations) * 100, 1),
                "low_pct": round((low_confidence / total_correlations) * 100, 1)
            },
            "correlation_methods": dict(methods),
            "story_point_analysis": story_analysis,
            "platforms": list(metrics.get('platform_coverage', {}).keys())
        }
    
    def _calculate_merge_commit_rate(self, commits: List[Dict[str, Any]]) -> float:
        """Calculate percentage of merge commits."""
        if not commits:
            return 0
        
        merge_commits = sum(1 for c in commits if c.get('is_merge'))
        return round((merge_commits / len(commits)) * 100, 1)
    
    def _analyze_commit_message_quality(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze commit message quality patterns."""
        
        if not commits:
            return {}
        
        message_lengths = []
        has_ticket_ref = 0
        conventional_commits = 0
        
        # Conventional commit prefixes
        conventional_prefixes = ['feat:', 'fix:', 'docs:', 'style:', 'refactor:', 'test:', 'chore:']
        
        for commit in commits:
            message = commit.get('message', '')
            
            # Message length (in words)
            word_count = len(message.split())
            message_lengths.append(word_count)
            
            # Ticket reference check
            if commit.get('ticket_references'):
                has_ticket_ref += 1
            
            # Conventional commit check
            if any(message.lower().startswith(prefix) for prefix in conventional_prefixes):
                conventional_commits += 1
        
        total_commits = len(commits)
        
        quality = {}
        
        if message_lengths:
            quality['avg_message_length_words'] = round(statistics.mean(message_lengths), 1)
            quality['median_message_length_words'] = round(statistics.median(message_lengths), 1)
        
        quality['ticket_reference_rate_pct'] = round((has_ticket_ref / total_commits) * 100, 1)
        quality['conventional_commit_rate_pct'] = round((conventional_commits / total_commits) * 100, 1)
        
        # Quality rating
        score = 0
        if quality.get('avg_message_length_words', 0) >= 5:
            score += 25
        if quality.get('ticket_reference_rate_pct', 0) >= 50:
            score += 35
        if quality.get('conventional_commit_rate_pct', 0) >= 30:
            score += 40
        
        if score >= 80:
            quality['overall_rating'] = 'excellent'
        elif score >= 60:
            quality['overall_rating'] = 'good'
        elif score >= 40:
            quality['overall_rating'] = 'fair'
        else:
            quality['overall_rating'] = 'needs_improvement'
        
        return quality
    
    def _generate_weekly_time_series(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Generate weekly time series data for charts."""
        
        weekly_data = []
        current_date = start_date
        
        while current_date <= end_date:
            week_end = current_date + timedelta(days=7)
            
            # Filter commits for this week
            week_commits = []
            for c in commits:
                # Ensure both timestamps are timezone-aware for comparison
                commit_ts = c['timestamp']
                if hasattr(commit_ts, 'tzinfo') and commit_ts.tzinfo is None:
                    # Make timezone-aware if needed
                    commit_ts = commit_ts.replace(tzinfo=timezone.utc)
                elif not hasattr(commit_ts, 'tzinfo'):
                    # Convert to datetime if needed
                    commit_ts = datetime.fromisoformat(str(commit_ts))
                    if commit_ts.tzinfo is None:
                        commit_ts = commit_ts.replace(tzinfo=timezone.utc)
                
                if current_date <= commit_ts < week_end:
                    week_commits.append(c)
            
            # Filter PRs for this week (by merge date)
            week_prs = []
            for pr in prs:
                merged_at = pr.get('merged_at')
                if merged_at:
                    if isinstance(merged_at, str):
                        merged_at = datetime.fromisoformat(merged_at.replace('Z', '+00:00'))
                    # Ensure timezone-aware for comparison
                    if hasattr(merged_at, 'tzinfo') and merged_at.tzinfo is None:
                        merged_at = merged_at.replace(tzinfo=timezone.utc)
                    if current_date <= merged_at < week_end:
                        week_prs.append(pr)
            
            # Calculate metrics
            lines_changed = sum(
                c.get('filtered_insertions', c.get('insertions', 0)) +
                c.get('filtered_deletions', c.get('deletions', 0))
                for c in week_commits
            )
            
            story_points = sum(c.get('story_points', 0) or 0 for c in week_commits)
            
            active_developers = len(set(
                c.get('canonical_id', c.get('author_email'))
                for c in week_commits
            ))
            
            weekly_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'commits': len(week_commits),
                'lines_changed': lines_changed,
                'story_points': story_points,
                'active_developers': active_developers,
                'pull_requests': len(week_prs)
            })
            
            current_date = week_end
        
        return weekly_data
    
    def _generate_daily_time_series(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Generate daily time series data for detailed analysis."""
        
        daily_data = []
        current_date = start_date
        
        while current_date <= end_date:
            day_end = current_date + timedelta(days=1)
            
            # Filter commits for this day
            day_commits = []
            for c in commits:
                # Ensure both timestamps are timezone-aware for comparison
                commit_ts = c['timestamp']
                if hasattr(commit_ts, 'tzinfo') and commit_ts.tzinfo is None:
                    # Make timezone-aware if needed
                    commit_ts = commit_ts.replace(tzinfo=timezone.utc)
                elif not hasattr(commit_ts, 'tzinfo'):
                    # Convert to datetime if needed
                    commit_ts = datetime.fromisoformat(str(commit_ts))
                    if commit_ts.tzinfo is None:
                        commit_ts = commit_ts.replace(tzinfo=timezone.utc)
                
                if current_date <= commit_ts < day_end:
                    day_commits.append(c)
            
            daily_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'commits': len(day_commits)
            })
            
            current_date = day_end
        
        return daily_data
    
    def _generate_quantitative_insights(
        self,
        commits: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate quantitative insights from data analysis."""
        
        insights = []
        
        # Team productivity insights
        total_commits = len(commits)
        if total_commits > 0:
            weekly_commits = self._get_weekly_commit_counts(commits)
            if weekly_commits:
                avg_weekly = statistics.mean(weekly_commits)
                insights.append({
                    "category": "productivity",
                    "type": "metric",
                    "title": "Weekly Commit Rate",
                    "description": f"Team averages {avg_weekly:.1f} commits per week",
                    "value": avg_weekly,
                    "trend": self._calculate_simple_trend(weekly_commits),
                    "priority": "medium"
                })
        
        # Developer distribution insights
        if len(developer_stats) > 1:
            commit_counts = [dev['total_commits'] for dev in developer_stats]
            gini = self._calculate_gini_coefficient(commit_counts)
            
            if gini > 0.7:
                insights.append({
                    "category": "team",
                    "type": "concern",
                    "title": "Unbalanced Contributions",
                    "description": f"Work is concentrated among few developers (Gini: {gini:.2f})",
                    "value": gini,
                    "priority": "high",
                    "recommendation": "Consider distributing work more evenly"
                })
            elif gini < 0.3:
                insights.append({
                    "category": "team",
                    "type": "positive",
                    "title": "Balanced Team Contributions",
                    "description": f"Work is well-distributed across the team (Gini: {gini:.2f})",
                    "value": gini,
                    "priority": "low"
                })
        
        # Code quality insights
        commit_sizes = []
        for commit in commits:
            lines = (
                commit.get('filtered_insertions', commit.get('insertions', 0)) +
                commit.get('filtered_deletions', commit.get('deletions', 0))
            )
            commit_sizes.append(lines)
        
        if commit_sizes:
            avg_size = statistics.mean(commit_sizes)
            if avg_size > 300:
                insights.append({
                    "category": "quality",
                    "type": "concern",
                    "title": "Large Commit Sizes",
                    "description": f"Average commit size is {avg_size:.0f} lines",
                    "value": avg_size,
                    "priority": "medium",
                    "recommendation": "Consider breaking down changes into smaller commits"
                })
            elif 20 <= avg_size <= 200:
                insights.append({
                    "category": "quality",
                    "type": "positive",
                    "title": "Optimal Commit Sizes",
                    "description": f"Average commit size of {avg_size:.0f} lines indicates good change management",
                    "value": avg_size,
                    "priority": "low"
                })
        
        return insights
    
    def _process_qualitative_insights(self, qualitative_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process qualitative analysis results into insights."""
        
        insights = []
        
        for item in qualitative_data:
            # Transform qualitative data into insight format
            insight = {
                "category": item.get('category', 'general'),
                "type": "qualitative",
                "title": item.get('insight', 'Qualitative Insight'),
                "description": item.get('description', ''),
                "priority": item.get('priority', 'medium'),
                "confidence": item.get('confidence', 0.5)
            }
            
            if 'recommendation' in item:
                insight['recommendation'] = item['recommendation']
            
            insights.append(insight)
        
        return insights
    
    def _prioritize_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize insights by importance and impact."""
        
        def get_priority_score(insight):
            priority_scores = {'high': 3, 'medium': 2, 'low': 1}
            type_scores = {'concern': 3, 'positive': 1, 'metric': 2, 'qualitative': 2}
            
            priority_score = priority_scores.get(insight.get('priority', 'medium'), 2)
            type_score = type_scores.get(insight.get('type', 'metric'), 2)
            
            return priority_score + type_score
        
        # Sort by priority score (descending)
        prioritized = sorted(insights, key=get_priority_score, reverse=True)
        
        return prioritized[:10]  # Return top 10 insights
    
    def _categorize_insights(self, insights: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize insights by category."""
        
        categories = defaultdict(list)
        
        for insight in insights:
            category = insight.get('category', 'general')
            categories[category].append(insight)
        
        return dict(categories)
    
    def _build_untracked_analysis(
        self,
        commits: List[Dict[str, Any]],
        project_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build comprehensive untracked commit analysis for JSON export.
        
        WHY: Untracked work analysis is critical for understanding what development
        activities are happening outside the formal process. This data enables
        process improvements, training identification, and better project visibility.
        
        Args:
            commits: List of all commits
            project_metrics: Project metrics including ticket analysis
            
        Returns:
            Dictionary with comprehensive untracked analysis
        """
        ticket_analysis = project_metrics.get('ticket_analysis', {})
        untracked_commits = ticket_analysis.get('untracked_commits', [])
        
        if not untracked_commits:
            return {
                "summary": {
                    "total_untracked": 0,
                    "untracked_percentage": 0,
                    "analysis_status": "no_untracked_commits"
                },
                "categories": {},
                "contributors": {},
                "projects": {},
                "trends": {},
                "recommendations": []
            }
        
        # Initialize analysis structures
        categories = {}
        contributors = {}
        projects = {}
        monthly_trends = {}
        
        total_commits = ticket_analysis.get('total_commits', len(commits))
        total_untracked = len(untracked_commits)
        
        # Process each untracked commit
        for commit in untracked_commits:
            # Category analysis
            category = commit.get('category', 'other')
            if category not in categories:
                categories[category] = {
                    'count': 0,
                    'lines_changed': 0,
                    'files_changed': 0,
                    'examples': [],
                    'authors': set()
                }
            
            categories[category]['count'] += 1
            categories[category]['lines_changed'] += commit.get('lines_changed', 0)
            categories[category]['files_changed'] += commit.get('files_changed', 0)
            categories[category]['authors'].add(commit.get('canonical_id', commit.get('author_email', 'Unknown')))
            
            if len(categories[category]['examples']) < 3:
                categories[category]['examples'].append({
                    'hash': commit.get('hash', ''),
                    'message': commit.get('message', '')[:200],
                    'author': self._anonymize_value(commit.get('author', 'Unknown'), 'name'),
                    'timestamp': commit.get('timestamp'),
                    'lines_changed': commit.get('lines_changed', 0),
                    'files_changed': commit.get('files_changed', 0)
                })
            
            # Contributor analysis
            author_id = commit.get('canonical_id', commit.get('author_email', 'Unknown'))
            author_name = self._anonymize_value(commit.get('author', 'Unknown'), 'name')
            
            if author_id not in contributors:
                contributors[author_id] = {
                    'name': author_name,
                    'count': 0,
                    'lines_changed': 0,
                    'categories': set(),
                    'projects': set(),
                    'recent_commits': []
                }
            
            contributors[author_id]['count'] += 1
            contributors[author_id]['lines_changed'] += commit.get('lines_changed', 0)
            contributors[author_id]['categories'].add(category)
            contributors[author_id]['projects'].add(commit.get('project_key', 'UNKNOWN'))
            
            if len(contributors[author_id]['recent_commits']) < 5:
                contributors[author_id]['recent_commits'].append({
                    'hash': commit.get('hash', ''),
                    'message': commit.get('message', '')[:100],
                    'category': category,
                    'timestamp': commit.get('timestamp'),
                    'lines_changed': commit.get('lines_changed', 0)
                })
            
            # Project analysis
            project = commit.get('project_key', 'UNKNOWN')
            if project not in projects:
                projects[project] = {
                    'count': 0,
                    'lines_changed': 0,
                    'categories': set(),
                    'contributors': set(),
                    'avg_commit_size': 0
                }
            
            projects[project]['count'] += 1
            projects[project]['lines_changed'] += commit.get('lines_changed', 0)
            projects[project]['categories'].add(category)
            projects[project]['contributors'].add(author_id)
            
            # Monthly trend analysis
            timestamp = commit.get('timestamp')
            if timestamp and hasattr(timestamp, 'strftime'):
                month_key = timestamp.strftime('%Y-%m')
                if month_key not in monthly_trends:
                    monthly_trends[month_key] = {
                        'count': 0,
                        'categories': {},
                        'contributors': set()
                    }
                monthly_trends[month_key]['count'] += 1
                monthly_trends[month_key]['contributors'].add(author_id)
                
                if category not in monthly_trends[month_key]['categories']:
                    monthly_trends[month_key]['categories'][category] = 0
                monthly_trends[month_key]['categories'][category] += 1
        
        # Convert sets to lists and calculate derived metrics
        for category_data in categories.values():
            category_data['authors'] = len(category_data['authors'])
            category_data['avg_lines_per_commit'] = (
                category_data['lines_changed'] / category_data['count'] 
                if category_data['count'] > 0 else 0
            )
        
        for contributor_data in contributors.values():
            contributor_data['categories'] = list(contributor_data['categories'])
            contributor_data['projects'] = list(contributor_data['projects'])
            contributor_data['avg_lines_per_commit'] = (
                contributor_data['lines_changed'] / contributor_data['count']
                if contributor_data['count'] > 0 else 0
            )
        
        for project_data in projects.values():
            project_data['categories'] = list(project_data['categories'])
            project_data['contributors'] = len(project_data['contributors'])
            project_data['avg_commit_size'] = (
                project_data['lines_changed'] / project_data['count']
                if project_data['count'] > 0 else 0
            )
        
        # Convert sets to counts in trends
        for trend_data in monthly_trends.values():
            trend_data['contributors'] = len(trend_data['contributors'])
        
        # Generate insights and recommendations
        insights = self._generate_untracked_insights(categories, contributors, projects, total_untracked, total_commits)
        recommendations = self._generate_untracked_recommendations_json(categories, contributors, total_untracked, total_commits)
        
        # Calculate quality scores
        quality_scores = self._calculate_untracked_quality_scores(categories, total_untracked, total_commits)
        
        return {
            "summary": {
                "total_untracked": total_untracked,
                "total_commits": total_commits,
                "untracked_percentage": round((total_untracked / total_commits * 100), 2) if total_commits > 0 else 0,
                "avg_lines_per_untracked_commit": round(
                    sum(commit.get('lines_changed', 0) for commit in untracked_commits) / total_untracked, 1
                ) if total_untracked > 0 else 0,
                "analysis_status": "complete"
            },
            "categories": categories,
            "contributors": contributors,
            "projects": projects,
            "monthly_trends": monthly_trends,
            "insights": insights,
            "recommendations": recommendations,
            "quality_scores": quality_scores
        }
    
    def _generate_untracked_insights(
        self,
        categories: Dict[str, Any],
        contributors: Dict[str, Any],
        projects: Dict[str, Any],
        total_untracked: int,
        total_commits: int
    ) -> List[Dict[str, Any]]:
        """Generate insights from untracked commit analysis."""
        insights = []
        
        # Category insights
        if categories:
            top_category = max(categories.items(), key=lambda x: x[1]['count'])
            category_name, category_data = top_category
            category_pct = (category_data['count'] / total_untracked * 100)
            
            if category_name in ['feature', 'bug_fix']:
                insights.append({
                    'type': 'concern',
                    'category': 'process',
                    'title': f'High {category_name.replace("_", " ").title()} Untracked Rate',
                    'description': f'{category_pct:.1f}% of untracked work is {category_name.replace("_", " ")} development',
                    'impact': 'high',
                    'value': category_pct
                })
            elif category_name in ['maintenance', 'style', 'documentation']:
                insights.append({
                    'type': 'positive',
                    'category': 'process',
                    'title': f'Appropriate Untracked Work',
                    'description': f'{category_pct:.1f}% of untracked work is {category_name} - this is acceptable',
                    'impact': 'low',
                    'value': category_pct
                })
        
        # Contributor concentration insights
        if len(contributors) > 1:
            contributor_counts = [data['count'] for data in contributors.values()]
            max_contributor_count = max(contributor_counts)
            contributor_concentration = (max_contributor_count / total_untracked * 100)
            
            if contributor_concentration > 50:
                insights.append({
                    'type': 'concern',
                    'category': 'team',
                    'title': 'Concentrated Untracked Work',
                    'description': f'One developer accounts for {contributor_concentration:.1f}% of untracked commits',
                    'impact': 'medium',
                    'value': contributor_concentration
                })
        
        # Overall coverage insight
        untracked_pct = (total_untracked / total_commits * 100) if total_commits > 0 else 0
        if untracked_pct > 40:
            insights.append({
                'type': 'concern',
                'category': 'coverage',
                'title': 'High Untracked Rate',
                'description': f'{untracked_pct:.1f}% of all commits lack ticket references',
                'impact': 'high',
                'value': untracked_pct
            })
        elif untracked_pct < 15:
            insights.append({
                'type': 'positive',
                'category': 'coverage',
                'title': 'Excellent Tracking Coverage',
                'description': f'Only {untracked_pct:.1f}% of commits are untracked',
                'impact': 'low',
                'value': untracked_pct
            })
        
        return insights
    
    def _generate_untracked_recommendations_json(
        self,
        categories: Dict[str, Any],
        contributors: Dict[str, Any],
        total_untracked: int,
        total_commits: int
    ) -> List[Dict[str, Any]]:
        """Generate JSON-formatted recommendations for untracked work."""
        recommendations = []
        
        # Category-based recommendations
        feature_count = categories.get('feature', {}).get('count', 0)
        bug_fix_count = categories.get('bug_fix', {}).get('count', 0)
        
        if feature_count > total_untracked * 0.25:
            recommendations.append({
                'type': 'process_improvement',
                'priority': 'high',
                'title': 'Enforce Feature Ticket Requirements',
                'description': 'Many feature developments lack ticket references',
                'action': 'Require ticket creation and referencing for all new features',
                'expected_impact': 'Improved project visibility and planning',
                'effort': 'low'
            })
        
        if bug_fix_count > total_untracked * 0.20:
            recommendations.append({
                'type': 'process_improvement',
                'priority': 'high',
                'title': 'Link Bug Fixes to Issues',
                'description': 'Bug fixes should be tracked through issue management',
                'action': 'Create issues for bugs and reference them in fix commits',
                'expected_impact': 'Better bug tracking and resolution visibility',
                'effort': 'low'
            })
        
        # Coverage-based recommendations
        untracked_pct = (total_untracked / total_commits * 100) if total_commits > 0 else 0
        if untracked_pct > 40:
            recommendations.append({
                'type': 'team_training',
                'priority': 'medium',
                'title': 'Team Process Training',
                'description': 'High percentage of untracked commits indicates process gaps',
                'action': 'Provide training on ticket referencing and commit best practices',
                'expected_impact': 'Improved process adherence and visibility',
                'effort': 'medium'
            })
        
        # Developer-specific recommendations
        if len(contributors) > 1:
            max_contributor_pct = max(
                (data['count'] / total_untracked * 100) for data in contributors.values()
            )
            if max_contributor_pct > 40:
                recommendations.append({
                    'type': 'individual_coaching',
                    'priority': 'medium',
                    'title': 'Targeted Developer Coaching',
                    'description': 'Some developers need additional guidance on process',
                    'action': 'Provide one-on-one coaching for developers with high untracked rates',
                    'expected_impact': 'More consistent process adherence across the team',
                    'effort': 'low'
                })
        
        return recommendations
    
    def _calculate_untracked_quality_scores(
        self,
        categories: Dict[str, Any],
        total_untracked: int,
        total_commits: int
    ) -> Dict[str, Any]:
        """Calculate quality scores for untracked work patterns."""
        scores = {}
        
        # Process adherence score (lower untracked % = higher score)
        untracked_pct = (total_untracked / total_commits * 100) if total_commits > 0 else 0
        process_score = max(0, 100 - untracked_pct * 2)  # Scale so 50% untracked = 0 score
        scores['process_adherence'] = round(min(100, process_score), 1)
        
        # Appropriate untracked score (higher % of maintenance/docs/style = higher score)
        appropriate_categories = ['maintenance', 'documentation', 'style', 'test']
        appropriate_count = sum(
            categories.get(cat, {}).get('count', 0) for cat in appropriate_categories
        )
        appropriate_pct = (appropriate_count / total_untracked * 100) if total_untracked > 0 else 0
        scores['appropriate_untracked'] = round(appropriate_pct, 1)
        
        # Work type balance score
        if categories:
            category_counts = [data['count'] for data in categories.values()]
            # Calculate distribution balance (lower Gini = more balanced)
            gini = self._calculate_gini_coefficient(category_counts)
            balance_score = max(0, 100 - (gini * 100))
            scores['work_type_balance'] = round(balance_score, 1)
        else:
            scores['work_type_balance'] = 100
        
        # Overall untracked quality score
        overall_score = (
            scores['process_adherence'] * 0.5 +
            scores['appropriate_untracked'] * 0.3 +
            scores['work_type_balance'] * 0.2
        )
        scores['overall'] = round(overall_score, 1)
        
        # Quality rating
        if overall_score >= 80:
            rating = 'excellent'
        elif overall_score >= 60:
            rating = 'good'
        elif overall_score >= 40:
            rating = 'fair'
        else:
            rating = 'needs_improvement'
        
        scores['rating'] = rating
        
        return scores
    
    def _generate_actionable_recommendations(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from insights."""
        
        recommendations = []
        
        # Extract recommendations from insights
        for insight in insights:
            if 'recommendation' in insight and insight.get('type') == 'concern':
                recommendations.append({
                    "title": insight['title'],
                    "action": insight['recommendation'],
                    "priority": insight.get('priority', 'medium'),
                    "category": insight.get('category', 'general'),
                    "expected_impact": self._estimate_recommendation_impact(insight)
                })
        
        # Add general recommendations based on patterns
        self._add_general_recommendations(recommendations, insights)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _estimate_recommendation_impact(self, insight: Dict[str, Any]) -> str:
        """Estimate the impact of implementing a recommendation."""
        
        category = insight.get('category', '')
        priority = insight.get('priority', 'medium')
        
        if priority == 'high':
            return 'high'
        elif category in ['team', 'productivity']:
            return 'medium'
        else:
            return 'low'
    
    def _add_general_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        insights: List[Dict[str, Any]]
    ) -> None:
        """Add general recommendations based on insight patterns."""
        
        # Check for lack of ticket coverage insights
        ticket_insights = [i for i in insights if 'ticket' in i.get('description', '').lower()]
        if not ticket_insights:
            recommendations.append({
                "title": "Improve Development Process Tracking",
                "action": "Implement consistent ticket referencing in commits and PRs",
                "priority": "medium",
                "category": "process",
                "expected_impact": "medium"
            })
    
    def _calculate_simple_trend(self, values: List[float]) -> str:
        """Calculate simple trend direction from a list of values."""
        
        if len(values) < 2:
            return "stable"
        
        # Compare first half vs second half
        midpoint = len(values) // 2
        first_half = values[:midpoint]
        second_half = values[midpoint:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if first_avg == 0:
            return "stable"
        
        change_pct = ((second_avg - first_avg) / first_avg) * 100
        
        if abs(change_pct) < 10:
            return "stable"
        elif change_pct > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _get_weekly_commit_counts(self, commits: List[Dict[str, Any]]) -> List[int]:
        """Get commit counts grouped by week."""
        
        if not commits:
            return []
        
        # Group commits by week
        weekly_counts = defaultdict(int)
        
        for commit in commits:
            week_start = self._get_week_start(commit['timestamp'])
            week_key = week_start.strftime('%Y-%m-%d')
            weekly_counts[week_key] += 1
        
        # Return counts in chronological order
        sorted_weeks = sorted(weekly_counts.keys())
        return [weekly_counts[week] for week in sorted_weeks]
    
    def _get_daily_commit_counts(self, commits: List[Dict[str, Any]]) -> List[int]:
        """Get commit counts grouped by day."""
        
        if not commits:
            return []
        
        # Group commits by day
        daily_counts = defaultdict(int)
        
        for commit in commits:
            day_key = commit['timestamp'].strftime('%Y-%m-%d')
            daily_counts[day_key] += 1
        
        # Return counts in chronological order
        sorted_days = sorted(daily_counts.keys())
        return [daily_counts[day] for day in sorted_days]
    
    def _calculate_weekly_commits(self, commits: List[Dict[str, Any]]) -> float:
        """Calculate average commits per week."""
        
        weekly_counts = self._get_weekly_commit_counts(commits)
        if not weekly_counts:
            return 0
        
        return round(statistics.mean(weekly_counts), 1)
    
    def _find_peak_activity_day(self, commits: List[Dict[str, Any]]) -> str:
        """Find the day of week with most commits."""
        
        if not commits:
            return "Unknown"
        
        day_counts = defaultdict(int)
        
        for commit in commits:
            if hasattr(commit['timestamp'], 'weekday'):
                day_index = commit['timestamp'].weekday()
                day_counts[day_index] += 1
        
        if not day_counts:
            return "Unknown"
        
        peak_day_index = max(day_counts, key=day_counts.get)
        return self._get_day_name(peak_day_index)
    
    def _analyze_commit_size_distribution(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of commit sizes."""
        
        if not commits:
            return {}
        
        sizes = []
        for commit in commits:
            lines = (
                commit.get('filtered_insertions', commit.get('insertions', 0)) +
                commit.get('filtered_deletions', commit.get('deletions', 0))
            )
            sizes.append(lines)
        
        if not sizes:
            return {}
        
        return {
            'mean': round(statistics.mean(sizes), 1),
            'median': round(statistics.median(sizes), 1),
            'std_dev': round(statistics.pstdev(sizes), 1) if len(sizes) > 1 else 0,
            'min': min(sizes),
            'max': max(sizes),
            'small_commits': sum(1 for s in sizes if s < 50),  # < 50 lines
            'medium_commits': sum(1 for s in sizes if 50 <= s <= 200),  # 50-200 lines
            'large_commits': sum(1 for s in sizes if s > 200)  # > 200 lines
        }
    
    def _get_week_start(self, date: datetime) -> datetime:
        """Get Monday of the week for a given date."""
        
        # Ensure timezone consistency
        if hasattr(date, 'tzinfo') and date.tzinfo is not None:
            if date.tzinfo != timezone.utc:
                date = date.astimezone(timezone.utc)
        else:
            date = date.replace(tzinfo=timezone.utc)
        
        days_since_monday = date.weekday()
        monday = date - timedelta(days=days_since_monday)
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for measuring inequality."""
        
        if not values or len(values) == 1:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        
        return (2 * np.sum((i + 1) * sorted_values[i] for i in range(n))) / (n * cumsum[-1]) - (n + 1) / n
    
    def _anonymize_value(self, value: str, field_type: str) -> str:
        """Anonymize a value if anonymization is enabled."""
        
        if not self.anonymize or not value:
            return value
        
        if field_type == 'email' and '@' in value:
            # Keep domain for email
            local, domain = value.split('@', 1)
            value = local  # Anonymize only local part
            suffix = f"@{domain}"
        else:
            suffix = ""
        
        if value not in self._anonymization_map:
            self._anonymous_counter += 1
            if field_type == 'name':
                anonymous = f"Developer{self._anonymous_counter}"
            elif field_type == 'email':
                anonymous = f"dev{self._anonymous_counter}"
            elif field_type == 'id':
                anonymous = f"ID{self._anonymous_counter:04d}"
            elif field_type == 'username':
                anonymous = f"user{self._anonymous_counter}"
            else:
                anonymous = f"anon{self._anonymous_counter}"
            
            self._anonymization_map[value] = anonymous
        
        return self._anonymization_map[value] + suffix
    
    def _serialize_for_json(self, data: Any) -> Any:
        """Serialize data for JSON output, handling datetime objects."""
        
        if isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, dict):
            return {k: self._serialize_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_for_json(item) for item in data]
        elif isinstance(data, set):
            return list(data)  # Convert sets to lists
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)  # Convert numpy types to Python types
        else:
            return data
    
    # Implementation of abstract methods from BaseReportGenerator
    
    def generate(self, data: ReportData, output_path: Optional[Path] = None) -> ReportOutput:
        """Generate comprehensive JSON export from standardized data.
        
        Args:
            data: Standardized report data
            output_path: Optional path to write the JSON to
            
        Returns:
            ReportOutput containing the results
        """
        try:
            # Validate data
            if not self.validate_data(data):
                return ReportOutput(
                    success=False,
                    errors=["Invalid or incomplete data provided"]
                )
            
            # Pre-process data
            data = self.pre_process(data)
            
            # Use the main export method with ReportData fields
            if output_path:
                self.export_comprehensive_data(
                    commits=data.commits or [],
                    prs=data.pull_requests or [],
                    developer_stats=data.developer_stats or [],
                    project_metrics=data.config.get("project_metrics", {}),
                    dora_metrics=data.dora_metrics or {},
                    output_path=output_path,
                    weeks=data.metadata.analysis_period_weeks or 12,
                    pm_data=data.pm_data,
                    qualitative_data=data.qualitative_results,
                    enhanced_qualitative_analysis=data.config.get("enhanced_qualitative_analysis")
                )
                
                return ReportOutput(
                    success=True,
                    file_path=output_path,
                    format=self.get_format_type(),
                    size_bytes=output_path.stat().st_size if output_path.exists() else 0
                )
            else:
                # Generate in-memory JSON
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(weeks=data.metadata.analysis_period_weeks or 12)
                
                export_data = {
                    "metadata": self._build_metadata(
                        data.commits or [], 
                        data.pull_requests or [], 
                        data.developer_stats or [], 
                        start_date, 
                        end_date
                    ),
                    "executive_summary": self._build_executive_summary(
                        data.commits or [],
                        data.pull_requests or [],
                        data.developer_stats or [],
                        data.config.get("project_metrics", {}),
                        data.dora_metrics or {}
                    ),
                    "raw_data": self._build_raw_data_summary(
                        data.commits or [],
                        data.pull_requests or [],
                        data.developer_stats or [],
                        data.dora_metrics or {}
                    )
                }
                
                serialized_data = self._serialize_for_json(export_data)
                json_content = json.dumps(serialized_data, indent=2, ensure_ascii=False)
                
                return ReportOutput(
                    success=True,
                    content=json_content,
                    format=self.get_format_type(),
                    size_bytes=len(json_content)
                )
                
        except Exception as e:
            logger.error(f"Error generating comprehensive JSON export: {e}")
            return ReportOutput(
                success=False,
                errors=[str(e)]
            )
    
    def get_required_fields(self) -> List[str]:
        """Get the list of required data fields for JSON export.
        
        Returns:
            List of required field names
        """
        # Comprehensive JSON export can work with any combination of data
        # but works best with commits and developer_stats
        return []  # No strict requirements, flexible export
    
    def get_format_type(self) -> str:
        """Get the format type this generator produces.
        
        Returns:
            Format identifier
        """
        return ReportFormat.JSON.value