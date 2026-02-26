"""JSON exporter mixin: data structure builders (metadata, executive summary, projects, developers, workflow, time series).

Extracted from json_exporter.py to keep file sizes manageable.
"""

import json
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class JSONExportBuildersMixin:
    """Mixin providing data structure builders (metadata, executive summary, projects, developers, workflow, time series) for ComprehensiveJSONExporter."""

    def _build_metadata(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Build metadata section with generation info and data summary."""

        # Get unique repositories and projects
        repositories = set()
        projects = set()

        for commit in commits:
            if commit.get("repository"):
                repositories.add(commit["repository"])
            if commit.get("project_key"):
                projects.add(commit["project_key"])

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "format_version": "2.0.0",
            "generator": "GitFlow Analytics Comprehensive JSON Exporter",
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "weeks_analyzed": (end_date - start_date).days // 7,
                "total_days": (end_date - start_date).days,
            },
            "data_summary": {
                "total_commits": len(commits),
                "total_prs": len(prs),
                "total_developers": len(developer_stats),
                "repositories_analyzed": len(repositories),
                "projects_identified": len(projects),
                "repositories": sorted(list(repositories)),
                "projects": sorted(list(projects)),
            },
            "export_settings": {"anonymized": self.anonymize, "timezone": "UTC"},
        }

    def _build_executive_summary(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        project_metrics: Dict[str, Any],
        dora_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build executive summary with key metrics, trends, and insights."""

        # Core metrics
        total_commits = len(commits)
        total_prs = len(prs)
        total_developers = len(developer_stats)

        # Calculate lines changed
        total_lines = sum(
            commit.get("filtered_insertions", commit.get("insertions", 0))
            + commit.get("filtered_deletions", commit.get("deletions", 0))
            for commit in commits
        )

        # Story points
        total_story_points = sum(commit.get("story_points", 0) or 0 for commit in commits)

        # Ticket coverage
        ticket_analysis = project_metrics.get("ticket_analysis", {})
        ticket_coverage = ticket_analysis.get("commit_coverage_pct", 0)

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
                    "trend_percent": trends.get("commits_trend", 0),
                    "trend_direction": self._get_trend_direction(trends.get("commits_trend", 0)),
                },
                "lines_changed": {
                    "total": total_lines,
                    "trend_percent": trends.get("lines_trend", 0),
                    "trend_direction": self._get_trend_direction(trends.get("lines_trend", 0)),
                },
                "story_points": {
                    "total": total_story_points,
                    "trend_percent": trends.get("story_points_trend", 0),
                    "trend_direction": self._get_trend_direction(
                        trends.get("story_points_trend", 0)
                    ),
                },
                "developers": {
                    "total": total_developers,
                    "active_percentage": self._calculate_active_developer_percentage(
                        developer_stats
                    ),
                },
                "pull_requests": {
                    "total": total_prs,
                    "trend_percent": trends.get("prs_trend", 0),
                    "trend_direction": self._get_trend_direction(trends.get("prs_trend", 0)),
                },
                "ticket_coverage": {
                    "percentage": round(ticket_coverage, 1),
                    "quality_rating": self._get_coverage_quality_rating(ticket_coverage),
                },
            },
            "performance_indicators": {
                "velocity": {
                    "commits_per_week": round(
                        total_commits
                        / max((len(set(self._get_week_start(c["timestamp"]) for c in commits))), 1),
                        1,
                    ),
                    "story_points_per_week": round(
                        total_story_points
                        / max((len(set(self._get_week_start(c["timestamp"]) for c in commits))), 1),
                        1,
                    ),
                },
                "quality": {
                    "avg_commit_size": round(total_lines / max(total_commits, 1), 1),
                    "ticket_coverage_pct": round(ticket_coverage, 1),
                },
                "collaboration": {
                    "developers_per_project": self._calculate_avg_developers_per_project(commits),
                    "cross_project_contributors": self._count_cross_project_contributors(
                        commits, developer_stats
                    ),
                },
            },
            "trends": trends,
            "anomalies": anomalies,
            "wins": wins,
            "concerns": concerns,
            "health_score": self._calculate_overall_health_score(
                commits, developer_stats, project_metrics, dora_metrics
            ),
        }

    def _build_project_data(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        project_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build project-level data with health scores and contributor details."""

        # Group data by project
        project_data = defaultdict(
            lambda: {
                "commits": [],
                "prs": [],
                "contributors": set(),
                "lines_changed": 0,
                "story_points": 0,
                "files_changed": set(),
            }
        )

        # Process commits by project
        for commit in commits:
            project_key = commit.get("project_key", "UNKNOWN")
            project_data[project_key]["commits"].append(commit)
            project_data[project_key]["contributors"].add(
                commit.get("canonical_id", commit.get("author_email"))
            )

            lines = commit.get("filtered_insertions", commit.get("insertions", 0)) + commit.get(
                "filtered_deletions", commit.get("deletions", 0)
            )
            project_data[project_key]["lines_changed"] += lines
            project_data[project_key]["story_points"] += commit.get("story_points", 0) or 0

            # Track files (simplified - just count)
            files_changed = commit.get("filtered_files_changed", commit.get("files_changed", 0))
            if files_changed:
                # Add placeholder file references
                for i in range(files_changed):
                    project_data[project_key]["files_changed"].add(f"file_{i}")

        # Process PRs by project (if available)
        for pr in prs:
            # Try to determine project from PR data
            project_key = pr.get("project_key", "UNKNOWN")
            project_data[project_key]["prs"].append(pr)

        # Build structured project data
        projects = {}

        for project_key, data in project_data.items():
            commits_list = data["commits"]
            contributors = data["contributors"]

            # Calculate project health score
            health_score = self._calculate_project_health_score(commits_list, contributors)

            # Get contributor details
            contributor_details = self._get_project_contributor_details(
                commits_list, developer_stats
            )

            # Calculate project trends
            project_trends = self._calculate_project_trends(commits_list)

            # Detect project anomalies
            project_anomalies = self._detect_project_anomalies(commits_list)

            projects[project_key] = {
                "summary": {
                    "total_commits": len(commits_list),
                    "total_contributors": len(contributors),
                    "lines_changed": data["lines_changed"],
                    "story_points": data["story_points"],
                    "files_touched": len(data["files_changed"]),
                    "pull_requests": len(data["prs"]),
                },
                "health_score": health_score,
                "contributors": contributor_details,
                "activity_patterns": {
                    "commits_per_week": self._calculate_weekly_commits(commits_list),
                    "peak_activity_day": self._find_peak_activity_day(commits_list),
                    "commit_size_distribution": self._analyze_commit_size_distribution(
                        commits_list
                    ),
                },
                "trends": project_trends,
                "anomalies": project_anomalies,
                "focus_metrics": {
                    "primary_contributors": self._identify_primary_contributors(
                        commits_list, contributor_details
                    ),
                    "contribution_distribution": self._calculate_contribution_distribution(
                        commits_list
                    ),
                },
            }

        return projects

    def _build_developer_profiles(
        self, commits: List[Dict[str, Any]], developer_stats: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build comprehensive developer profiles with contribution patterns."""

        profiles = {}

        for dev in developer_stats:
            dev_id = dev["canonical_id"]
            dev_name = self._anonymize_value(dev["primary_name"], "name")

            # Get developer's commits
            dev_commits = [c for c in commits if c.get("canonical_id") == dev_id]

            # Calculate various metrics
            projects_worked = self._get_developer_projects(dev_commits)
            contribution_patterns = self._analyze_developer_contribution_patterns(dev_commits)
            collaboration_metrics = self._calculate_developer_collaboration_metrics(
                dev_commits, developer_stats
            )

            # Calculate developer health score
            health_score = self._calculate_developer_health_score(dev_commits, dev)

            # Identify achievements and areas for improvement
            achievements = self._identify_developer_achievements(dev_commits, dev)
            improvement_areas = self._identify_improvement_areas(dev_commits, dev)

            profiles[dev_id] = {
                "identity": {
                    "name": dev_name,
                    "canonical_id": dev_id,
                    "primary_email": self._anonymize_value(dev["primary_email"], "email"),
                    "github_username": self._anonymize_value(
                        dev.get("github_username", ""), "username"
                    )
                    if dev.get("github_username")
                    else None,
                    "aliases_count": dev.get("alias_count", 1),
                },
                "summary": {
                    "total_commits": dev["total_commits"],
                    "total_story_points": dev["total_story_points"],
                    "projects_contributed": len(projects_worked),
                    "first_seen": dev.get("first_seen").isoformat()
                    if dev.get("first_seen")
                    else None,
                    "last_seen": dev.get("last_seen").isoformat() if dev.get("last_seen") else None,
                    "days_active": (dev.get("last_seen") - dev.get("first_seen")).days
                    if dev.get("first_seen") and dev.get("last_seen")
                    else 0,
                },
                "health_score": health_score,
                "projects": projects_worked,
                "contribution_patterns": contribution_patterns,
                "collaboration": collaboration_metrics,
                "achievements": achievements,
                "improvement_areas": improvement_areas,
                "activity_timeline": self._build_developer_activity_timeline(dev_commits),
            }

        return profiles

    def _build_workflow_analysis(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        project_metrics: Dict[str, Any],
        pm_data: Optional[Dict[str, Any]],
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
                "ticket_linking_rate": project_metrics.get("ticket_analysis", {}).get(
                    "commit_coverage_pct", 0
                ),
                "merge_commit_rate": self._calculate_merge_commit_rate(commits),
                "commit_message_quality": self._analyze_commit_message_quality(commits),
            },
        }

    def _build_time_series_data(
        self, commits: List[Dict[str, Any]], prs: List[Dict[str, Any]], weeks: int
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
                        "borderColor": "rgba(54, 162, 235, 1)",
                    },
                    "lines_changed": {
                        "label": "Lines Changed",
                        "data": [d["lines_changed"] for d in weekly_data],
                        "backgroundColor": "rgba(255, 99, 132, 0.2)",
                        "borderColor": "rgba(255, 99, 132, 1)",
                    },
                    "story_points": {
                        "label": "Story Points",
                        "data": [d["story_points"] for d in weekly_data],
                        "backgroundColor": "rgba(75, 192, 192, 0.2)",
                        "borderColor": "rgba(75, 192, 192, 1)",
                    },
                    "active_developers": {
                        "label": "Active Developers",
                        "data": [d["active_developers"] for d in weekly_data],
                        "backgroundColor": "rgba(153, 102, 255, 0.2)",
                        "borderColor": "rgba(153, 102, 255, 1)",
                    },
                },
            },
            "daily": {
                "labels": [d["date"] for d in daily_data],
                "datasets": {
                    "commits": {
                        "label": "Daily Commits",
                        "data": [d["commits"] for d in daily_data],
                        "backgroundColor": "rgba(54, 162, 235, 0.1)",
                        "borderColor": "rgba(54, 162, 235, 1)",
                    }
                },
            },
        }

    def _build_insights_data(
        self,
        commits: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        qualitative_data: Optional[List[Dict[str, Any]]],
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
            "actionable_recommendations": self._generate_actionable_recommendations(all_insights),
        }

    def _build_raw_data_summary(
        self,
        commits: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
        developer_stats: List[Dict[str, Any]],
        dora_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build summary of raw data for reference and validation."""

        return {
            "commits_sample": commits[:5] if commits else [],  # First 5 commits as sample
            "prs_sample": prs[:3] if prs else [],  # First 3 PRs as sample
            "developer_stats_schema": {
                "fields": list(developer_stats[0].keys()) if developer_stats else [],
                "sample_record": developer_stats[0] if developer_stats else {},
            },
            "dora_metrics": dora_metrics,
            "data_quality": {
                "commits_with_timestamps": sum(1 for c in commits if c.get("timestamp")),
                "commits_with_projects": sum(1 for c in commits if c.get("project_key")),
                "commits_with_tickets": sum(1 for c in commits if c.get("ticket_references")),
                "developers_with_github": sum(
                    1 for d in developer_stats if d.get("github_username")
                ),
            },
        }

    def _build_pm_integration_data(self, pm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build PM platform integration data summary."""

        metrics = pm_data.get("metrics", {})
        correlations = pm_data.get("correlations", [])

        return {
            "platforms": list(metrics.get("platform_coverage", {}).keys()),
            "total_issues": metrics.get("total_pm_issues", 0),
            "story_point_coverage": metrics.get("story_point_analysis", {}).get(
                "story_point_coverage_pct", 0
            ),
            "correlations_count": len(correlations),
            "correlation_quality": metrics.get("correlation_quality", {}),
            "issue_types": metrics.get("issue_type_distribution", {}),
            "platform_summary": {
                platform: {
                    "total_issues": data.get("total_issues", 0),
                    "linked_issues": data.get("linked_issues", 0),
                    "coverage_percentage": data.get("coverage_percentage", 0),
                }
                for platform, data in metrics.get("platform_coverage", {}).items()
            },
        }

    def _build_cicd_data(self, cicd_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build CI/CD pipeline metrics data summary.

        WHY: CI/CD metrics provide visibility into build health, deployment velocity,
        and infrastructure stability. This data enables correlation with DORA metrics
        and developer productivity trends.

        Args:
            cicd_data: CI/CD data from orchestrator with pipelines and metrics

        Returns:
            Structured CI/CD metrics summary for JSON export
        """
        pipelines = cicd_data.get("pipelines", [])
        platform_metrics = cicd_data.get("metrics", {})

        # Calculate overall aggregates
        total_pipelines = len(pipelines)
        successful_pipelines = len([p for p in pipelines if p.get("status") == "success"])
        failed_pipelines = len([p for p in pipelines if p.get("status") in ["failure", "failed"]])
        overall_success_rate = (
            (successful_pipelines / total_pipelines * 100) if total_pipelines > 0 else 0
        )

        # Calculate duration statistics
        durations = [
            p.get("duration_seconds", 0) / 60.0  # Convert to minutes
            for p in pipelines
            if p.get("duration_seconds")
        ]
        avg_duration = statistics.mean(durations) if durations else 0
        median_duration = statistics.median(durations) if durations else 0

        # Extract unique workflows
        workflows = set(
            p.get("workflow_name") or p.get("name", "unknown")
            for p in pipelines
            if p.get("workflow_name") or p.get("name")
        )

        # Build platform-specific summaries
        platform_summary = {}
        for platform_name, metrics in platform_metrics.items():
            platform_summary[platform_name] = {
                "total_pipelines": metrics.get("total_pipelines", 0),
                "successful_pipelines": metrics.get("successful_pipelines", 0),
                "failed_pipelines": metrics.get("failed_pipelines", 0),
                "success_rate": metrics.get("success_rate", 0),
                "avg_duration_minutes": metrics.get("avg_duration_minutes", 0),
            }

        return {
            "platforms": list(platform_metrics.keys()),
            "total_pipelines": total_pipelines,
            "successful_pipelines": successful_pipelines,
            "failed_pipelines": failed_pipelines,
            "overall_success_rate": round(overall_success_rate, 1),
            "avg_duration_minutes": round(avg_duration, 2),
            "median_duration_minutes": round(median_duration, 2),
            "total_workflows": len(workflows),
            "platform_summary": platform_summary,
        }

    # Helper methods for calculations and analysis

