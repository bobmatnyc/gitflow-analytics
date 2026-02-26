"""Story point correlation analysis for GitFlow Analytics.

This module provides comprehensive analysis of story point estimation accuracy and 
correlation with actual development work metrics including commits, lines of code,
and time spent. It tracks velocity trends and generates actionable insights for
process improvement and team calibration.

WHY: Story point estimation is a critical part of agile development, but accuracy
varies significantly across teams and individuals. This analysis helps identify
which teams/developers have accurate estimates vs which need calibration training.

DESIGN DECISION: Week-based aggregation using Monday-Sunday boundaries to align
with sprint planning cycles and provide consistent reporting periods. All metrics
are calculated both at individual and team levels for targeted improvements.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .story_point_helpers import StoryPointHelpersMixin

# Get logger for this module
logger = logging.getLogger(__name__)


class StoryPointCorrelationAnalyzer(StoryPointHelpersMixin):
    """Analyzes story point estimation accuracy and correlations with actual work."""

    def __init__(self, anonymize: bool = False, identity_resolver=None):
        """Initialize the correlation analyzer.
        
        Args:
            anonymize: Whether to anonymize developer names in reports
            identity_resolver: Identity resolver for canonical developer names
        """
        self.anonymize = anonymize
        self.identity_resolver = identity_resolver
        self._anonymization_map: dict[str, str] = {}
        self._anonymous_counter = 0

    def calculate_weekly_correlations(
        self,
        commits: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        pm_data: Optional[dict[str, Any]] = None,
        weeks: int = 12
    ) -> dict[str, Any]:
        """Calculate weekly story point correlations with actual work metrics.
        
        WHY: Weekly aggregation provides sprint-aligned analysis periods that match
        typical development cycles, enabling actionable insights for sprint planning
        and retrospectives.
        
        Args:
            commits: List of commit data with story points and metrics
            prs: List of pull request data with story points
            pm_data: PM platform data with issue correlations
            weeks: Number of weeks to analyze
            
        Returns:
            Dictionary containing weekly correlation metrics and analysis
        """
        logger.debug(f"Starting weekly correlation analysis for {weeks} weeks")
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)
        
        logger.debug(f"Analysis period: {start_date} to {end_date}")
        
        # Aggregate data by week and developer
        weekly_metrics = self._aggregate_weekly_metrics(commits, prs, pm_data, start_date, end_date)
        
        # Calculate correlations for each week
        correlation_results = {}
        
        for week_start, week_data in weekly_metrics.items():
            week_correlations = self._calculate_week_correlations(week_data)
            correlation_results[week_start] = week_correlations
            
        logger.debug(f"Calculated correlations for {len(correlation_results)} weeks")
        
        return {
            "weekly_correlations": correlation_results,
            "summary_stats": self._calculate_correlation_summary(correlation_results),
            "trend_analysis": self._analyze_correlation_trends(correlation_results),
            "developer_accuracy": self._analyze_developer_accuracy(weekly_metrics),
            "recommendations": self._generate_correlation_recommendations(correlation_results, weekly_metrics)
        }

    def analyze_estimation_accuracy(
        self,
        commits: list[dict[str, Any]],
        pm_data: Optional[dict[str, Any]] = None,
        weeks: int = 12
    ) -> dict[str, Any]:
        """Analyze story point estimation accuracy by comparing estimated vs actual.
        
        WHY: Estimation accuracy analysis helps identify systematic over/under-estimation
        patterns and provides targeted feedback for improving planning accuracy.
        
        DESIGN DECISION: Uses multiple accuracy metrics (absolute error, relative error,
        accuracy percentage) to provide comprehensive view of estimation quality.
        
        Args:
            commits: List of commit data with story points
            pm_data: PM platform data with original story point estimates
            weeks: Number of weeks to analyze
            
        Returns:
            Dictionary containing estimation accuracy analysis
        """
        logger.debug("Starting estimation accuracy analysis")
        
        if not pm_data or "correlations" not in pm_data:
            logger.warning("No PM data available for estimation accuracy analysis")
            return self._empty_accuracy_analysis()
        
        # Extract estimation vs actual pairs
        estimation_pairs = self._extract_estimation_pairs(commits, pm_data, weeks)
        
        if not estimation_pairs:
            logger.warning("No estimation pairs found for accuracy analysis")
            return self._empty_accuracy_analysis()
        
        # Calculate accuracy metrics
        accuracy_metrics = self._calculate_accuracy_metrics(estimation_pairs)
        
        # Analyze by developer
        developer_accuracy = self._analyze_developer_estimation_accuracy(estimation_pairs)
        
        # Analyze by story point size
        size_accuracy = self._analyze_size_based_accuracy(estimation_pairs)
        
        return {
            "overall_accuracy": accuracy_metrics,
            "developer_accuracy": developer_accuracy,
            "size_based_accuracy": size_accuracy,
            "improvement_suggestions": self._generate_accuracy_recommendations(
                accuracy_metrics, developer_accuracy, size_accuracy
            )
        }

    def calculate_velocity_metrics(
        self,
        commits: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        pm_data: Optional[dict[str, Any]] = None,
        weeks: int = 12
    ) -> dict[str, Any]:
        """Calculate velocity trends and patterns over time.
        
        WHY: Velocity analysis helps track team productivity over time and identify
        factors that impact delivery speed, enabling better sprint planning and
        capacity management.
        
        Args:
            commits: List of commit data with story points
            prs: List of pull request data
            pm_data: PM platform data for additional context
            weeks: Number of weeks to analyze
            
        Returns:
            Dictionary containing velocity metrics and trends
        """
        logger.debug(f"Calculating velocity metrics for {weeks} weeks")
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)
        
        # Aggregate velocity data by week
        weekly_velocity = self._aggregate_weekly_velocity(commits, prs, start_date, end_date)
        
        # Calculate velocity trends
        velocity_trends = self._calculate_velocity_trends(weekly_velocity)
        
        # Analyze velocity by developer
        developer_velocity = self._analyze_developer_velocity(commits, start_date, end_date)
        
        # Calculate predictability metrics
        predictability = self._calculate_velocity_predictability(weekly_velocity)
        
        return {
            "weekly_velocity": weekly_velocity,
            "trends": velocity_trends,
            "developer_velocity": developer_velocity,
            "predictability": predictability,
            "capacity_analysis": self._analyze_team_capacity(weekly_velocity, developer_velocity)
        }

    def generate_correlation_report(
        self,
        commits: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        pm_data: Optional[dict[str, Any]],
        output_path: Path,
        weeks: int = 12
    ) -> Path:
        """Generate comprehensive CSV report with story point correlation metrics.
        
        WHY: CSV format enables easy import into spreadsheet tools for additional
        analysis and sharing with stakeholders who need detailed correlation data.
        
        Args:
            commits: List of commit data with story points
            prs: List of pull request data
            pm_data: PM platform data with correlations
            output_path: Path for the output CSV file
            weeks: Number of weeks to analyze
            
        Returns:
            Path to the generated CSV report
        """
        logger.debug(f"Generating story point correlation report: {output_path}")
        
        try:
            # Calculate all correlation metrics
            weekly_correlations = self.calculate_weekly_correlations(commits, prs, pm_data, weeks)
            estimation_accuracy = self.analyze_estimation_accuracy(commits, pm_data, weeks)
            velocity_metrics = self.calculate_velocity_metrics(commits, prs, pm_data, weeks)
            
            # Build CSV rows
            rows = self._build_correlation_csv_rows(
                weekly_correlations, estimation_accuracy, velocity_metrics
            )
            
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(output_path, index=False)
                logger.debug(f"Generated correlation report with {len(rows)} rows")
            else:
                # Write empty CSV with headers
                self._write_empty_correlation_csv(output_path)
                logger.debug("Generated empty correlation report (no data)")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating story point correlation report: {e}")
            # Still create empty report file
            self._write_empty_correlation_csv(output_path)
            raise

    def _aggregate_weekly_metrics(
        self,
        commits: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        pm_data: Optional[dict[str, Any]],
        start_date: datetime,
        end_date: datetime
    ) -> dict[datetime, dict[str, dict[str, Any]]]:
        """Aggregate metrics by week and developer for correlation analysis."""
        weekly_metrics = defaultdict(lambda: defaultdict(lambda: {
            "story_points": 0,
            "commits": 0,
            "lines_added": 0,
            "lines_removed": 0,
            "files_changed": 0,
            "prs": 0,
            "complexity_delta": 0.0,
            "time_spent_hours": 0.0,  # Estimated from commit frequency
            "estimated_story_points": 0,  # From PM platform
            "actual_story_points": 0      # From commits
        }))
        
        # Process commits
        for commit in commits:
            timestamp = self._ensure_timezone_aware(commit.get("timestamp"))
            if not timestamp or timestamp < start_date or timestamp > end_date:
                continue
                
            week_start = self._get_week_start(timestamp)
            developer_id = commit.get("canonical_id", commit.get("author_email", "unknown"))
            
            metrics = weekly_metrics[week_start][developer_id]
            
            # Aggregate commit metrics
            metrics["commits"] += 1
            metrics["story_points"] += commit.get("story_points", 0) or 0
            metrics["actual_story_points"] += commit.get("story_points", 0) or 0
            metrics["lines_added"] += commit.get("insertions", 0) or 0
            metrics["lines_removed"] += commit.get("deletions", 0) or 0
            metrics["files_changed"] += commit.get("files_changed", 0) or 0
            metrics["complexity_delta"] += commit.get("complexity_delta", 0.0) or 0.0
        
        # Process PRs
        for pr in prs:
            created_at = self._ensure_timezone_aware(pr.get("created_at"))
            if not created_at or created_at < start_date or created_at > end_date:
                continue
                
            week_start = self._get_week_start(created_at)
            developer_id = pr.get("canonical_id", pr.get("author", "unknown"))
            
            if developer_id in weekly_metrics[week_start]:
                weekly_metrics[week_start][developer_id]["prs"] += 1
        
        # Add PM platform data if available
        if pm_data and "correlations" in pm_data:
            for correlation in pm_data["correlations"]:
                commit_date = correlation.get("commit_date")
                if not commit_date:
                    continue
                    
                timestamp = self._ensure_timezone_aware(
                    datetime.fromisoformat(commit_date.replace("Z", "+00:00"))
                    if isinstance(commit_date, str) else commit_date
                )
                
                if timestamp < start_date or timestamp > end_date:
                    continue
                    
                week_start = self._get_week_start(timestamp)
                developer_id = correlation.get("commit_author", "unknown")
                
                if developer_id in weekly_metrics[week_start]:
                    estimated_sp = correlation.get("story_points", 0) or 0
                    weekly_metrics[week_start][developer_id]["estimated_story_points"] += estimated_sp
        
        # Convert defaultdicts to regular dicts for JSON serialization
        return {week: dict(developers) for week, developers in weekly_metrics.items()}

    def _calculate_week_correlations(self, week_data: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Calculate correlations for a single week's data."""
        if len(week_data) < 2:
            return self._empty_week_correlations()
        
        # Extract parallel arrays for correlation calculation
        developers = []
        story_points = []
        commits = []
        lines_changed = []
        files_changed = []
        prs = []
        complexity = []
        
        for dev_id, metrics in week_data.items():
            developers.append(dev_id)
            story_points.append(metrics["story_points"])
            commits.append(metrics["commits"])
            lines_changed.append(metrics["lines_added"] + metrics["lines_removed"])
            files_changed.append(metrics["files_changed"])
            prs.append(metrics["prs"])
            complexity.append(metrics["complexity_delta"])
        
        # Calculate correlations using scipy.stats
        correlations = {}

        try:
            # Check if we have enough data points and variance for meaningful correlations
            if len(story_points) < 2:
                logger.debug(f"Insufficient data for correlation: only {len(story_points)} data points")
                correlations = {k: 0.0 for k in ["sp_commits", "sp_lines", "sp_files", "sp_prs", "sp_complexity"]}
            elif np.std(story_points) == 0:
                logger.debug("All story points are the same value - no variance for correlation")
                correlations = {k: 0.0 for k in ["sp_commits", "sp_lines", "sp_files", "sp_prs", "sp_complexity"]}
            else:
                # Calculate correlations only when we have sufficient variance
                correlations["sp_commits"] = float(stats.pearsonr(story_points, commits)[0])
                correlations["sp_lines"] = float(stats.pearsonr(story_points, lines_changed)[0])
                correlations["sp_files"] = float(stats.pearsonr(story_points, files_changed)[0])
                correlations["sp_prs"] = float(stats.pearsonr(story_points, prs)[0])
                correlations["sp_complexity"] = float(stats.pearsonr(story_points, complexity)[0])
                logger.debug(f"Calculated correlations with {len(story_points)} data points")

        except Exception as e:
            logger.warning(f"Error calculating correlations: {e}")
            correlations = {k: 0.0 for k in ["sp_commits", "sp_lines", "sp_files", "sp_prs", "sp_complexity"]}
        
        return {
            "correlations": correlations,
            "sample_size": len(developers),
            "total_story_points": sum(story_points),
            "total_commits": sum(commits),
            "total_lines_changed": sum(lines_changed)
        }

    def _calculate_correlation_summary(self, correlation_results: dict[datetime, dict[str, Any]]) -> dict[str, Any]:
        """Calculate summary statistics across all weeks."""
        if not correlation_results:
            return {"avg_correlations": {}, "trend_direction": "stable", "strength": "weak"}
        
        # Aggregate correlations across weeks
        all_correlations = defaultdict(list)
        
        for week_data in correlation_results.values():
            correlations = week_data.get("correlations", {})
            for metric, value in correlations.items():
                if not np.isnan(value):  # Filter out NaN values
                    all_correlations[metric].append(value)
        
        # Calculate averages
        avg_correlations = {}
        for metric, values in all_correlations.items():
            if values:
                avg_correlations[metric] = float(np.mean(values))
            else:
                avg_correlations[metric] = 0.0
        
        # Determine overall correlation strength
        avg_strength = np.mean(list(avg_correlations.values()))
        if avg_strength > 0.7:
            strength = "strong"
        elif avg_strength > 0.4:
            strength = "moderate"
        else:
            strength = "weak"
        
        return {
            "avg_correlations": avg_correlations,
            "strength": strength,
            "weeks_analyzed": len(correlation_results),
            "max_correlation": max(avg_correlations.values()) if avg_correlations else 0.0,
            "min_correlation": min(avg_correlations.values()) if avg_correlations else 0.0
        }

    def _analyze_correlation_trends(self, correlation_results: dict[datetime, dict[str, Any]]) -> dict[str, Any]:
        """Analyze trends in correlations over time."""
        if len(correlation_results) < 3:
            return {"trend_direction": "insufficient_data", "trend_strength": 0.0}
        
        # Sort by week for trend analysis
        sorted_weeks = sorted(correlation_results.keys())
        
        # Calculate trend for each correlation metric
        trends = {}
        
        for metric in ["sp_commits", "sp_lines", "sp_files", "sp_prs", "sp_complexity"]:
            values = []
            weeks = []
            
            for week in sorted_weeks:
                week_correlations = correlation_results[week].get("correlations", {})
                if metric in week_correlations and not np.isnan(week_correlations[metric]):
                    values.append(week_correlations[metric])
                    weeks.append(len(weeks))  # Use index as x-value
            
            if len(values) >= 3:  # Need at least 3 points for trend
                slope, intercept, r_value, p_value, std_err = stats.linregress(weeks, values)
                trends[metric] = {
                    "slope": float(slope),
                    "r_squared": float(r_value ** 2),
                    "p_value": float(p_value),
                    "direction": "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable"
                }
            else:
                trends[metric] = {"slope": 0.0, "direction": "insufficient_data"}
        
        return trends

    def _analyze_developer_accuracy(self, weekly_metrics: dict[datetime, dict[str, dict[str, Any]]]) -> dict[str, Any]:
        """Analyze story point estimation accuracy by developer."""
        developer_totals = defaultdict(lambda: {
            "estimated_total": 0,
            "actual_total": 0,
            "weeks_active": 0,
            "accuracy_scores": []
        })
        
        for week_data in weekly_metrics.values():
            for dev_id, metrics in week_data.items():
                estimated = metrics.get("estimated_story_points", 0)
                actual = metrics.get("actual_story_points", 0)
                
                if estimated > 0 or actual > 0:  # Developer was active
                    dev_stats = developer_totals[dev_id]
                    dev_stats["estimated_total"] += estimated
                    dev_stats["actual_total"] += actual
                    dev_stats["weeks_active"] += 1
                    
                    # Calculate weekly accuracy if both values exist
                    if estimated > 0 and actual > 0:
                        accuracy = 1.0 - abs(estimated - actual) / max(estimated, actual)
                        dev_stats["accuracy_scores"].append(accuracy)
        
        # Calculate final accuracy metrics for each developer
        developer_accuracy = {}
        
        for dev_id, dev_stats in developer_totals.items():
            if dev_stats["weeks_active"] > 0:
                # Overall accuracy based on totals
                if dev_stats["estimated_total"] > 0 and dev_stats["actual_total"] > 0:
                    overall_accuracy = 1.0 - abs(dev_stats["estimated_total"] - dev_stats["actual_total"]) / max(dev_stats["estimated_total"], dev_stats["actual_total"])
                else:
                    overall_accuracy = 0.0
                
                # Average weekly accuracy
                if dev_stats["accuracy_scores"]:
                    avg_weekly_accuracy = float(np.mean(dev_stats["accuracy_scores"]))
                    consistency = 1.0 - float(np.std(dev_stats["accuracy_scores"]))
                else:
                    avg_weekly_accuracy = 0.0
                    consistency = 0.0
                
                developer_accuracy[self._anonymize_value(dev_id, "name")] = {
                    "overall_accuracy": float(overall_accuracy),
                    "avg_weekly_accuracy": avg_weekly_accuracy,
                    "consistency": consistency,
                    "weeks_active": dev_stats["weeks_active"],
                    "total_estimated": dev_stats["estimated_total"],
                    "total_actual": dev_stats["actual_total"],
                    "estimation_ratio": dev_stats["actual_total"] / max(dev_stats["estimated_total"], 1)
                }
        
        return developer_accuracy

    def _generate_correlation_recommendations(
        self, correlation_results: dict[datetime, dict[str, Any]], weekly_metrics: dict[datetime, dict[str, dict[str, Any]]]
    ) -> list[dict[str, str]]:
        """Generate actionable recommendations based on correlation analysis."""
        recommendations = []
        
        summary = self._calculate_correlation_summary(correlation_results)
        avg_correlations = summary.get("avg_correlations", {})
        
        # Check story points to commits correlation
        sp_commits_corr = avg_correlations.get("sp_commits", 0)
        if sp_commits_corr < 0.3:
            recommendations.append({
                "type": "process_improvement",
                "priority": "high",
                "title": "Weak Story Points to Commits Correlation",
                "description": f"Story points show weak correlation with commit count ({sp_commits_corr:.2f}). Consider story point training or breaking down large stories.",
                "action": "Review story point estimation guidelines and provide team training"
            })
        
        # Check story points to lines of code correlation
        sp_lines_corr = avg_correlations.get("sp_lines", 0)
        if sp_lines_corr < 0.4:
            recommendations.append({
                "type": "estimation_calibration",
                "priority": "medium",
                "title": "Story Points Don't Correlate with Code Changes",
                "description": f"Story points show weak correlation with lines of code changed ({sp_lines_corr:.2f}). This may indicate estimation inconsistency.",
                "action": "Analyze whether story points reflect complexity vs. effort, and align team understanding"
            })
        
        # Analyze developer accuracy
        developer_accuracy = self._analyze_developer_accuracy(weekly_metrics)
        low_accuracy_devs = [
            dev for dev, stats in developer_accuracy.items() 
            if stats["overall_accuracy"] < 0.5 and stats["weeks_active"] >= 2
        ]
        
        if low_accuracy_devs:
            recommendations.append({
                "type": "individual_coaching",
                "priority": "medium",
                "title": "Developers Need Estimation Training",
                "description": f"{len(low_accuracy_devs)} developers have low estimation accuracy. Consider individual coaching sessions.",
                "action": f"Provide estimation training for: {', '.join(low_accuracy_devs[:3])}"
            })
        
        # Check overall correlation strength
        if summary.get("strength") == "weak":
            recommendations.append({
                "type": "process_review",
                "priority": "high",
                "title": "Overall Weak Correlations",
                "description": "Story points show weak correlations across all work metrics. The estimation process may need fundamental review.",
                "action": "Conduct team retrospective on story point estimation process and consider alternative estimation methods"
            })
        
        return recommendations

