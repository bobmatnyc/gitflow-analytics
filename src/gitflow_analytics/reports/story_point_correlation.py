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

# Get logger for this module
logger = logging.getLogger(__name__)


class StoryPointCorrelationAnalyzer:
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

    def _extract_estimation_pairs(
        self, commits: list[dict[str, Any]], pm_data: dict[str, Any], weeks: int
    ) -> list[tuple[int, int, str]]:
        """Extract (estimated, actual, developer) pairs for accuracy analysis."""
        pairs = []
        
        if not pm_data or "correlations" not in pm_data:
            return pairs
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)
        
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
            
            estimated_sp = correlation.get("story_points", 0) or 0
            commit_hash = correlation.get("commit_hash", "")
            developer = correlation.get("commit_author", "unknown")
            
            # Find matching commit for actual story points
            matching_commit = next(
                (c for c in commits if c.get("hash", "") == commit_hash), None
            )
            
            if matching_commit:
                actual_sp = matching_commit.get("story_points", 0) or 0
                if estimated_sp > 0 and actual_sp > 0:  # Valid pair
                    pairs.append((estimated_sp, actual_sp, developer))
        
        return pairs

    def _calculate_accuracy_metrics(self, estimation_pairs: list[tuple[int, int, str]]) -> dict[str, Any]:
        """Calculate overall estimation accuracy metrics."""
        if not estimation_pairs:
            return {"mean_absolute_error": 0, "mean_relative_error": 0, "accuracy_percentage": 0}
        
        estimated_values = [pair[0] for pair in estimation_pairs]
        actual_values = [pair[1] for pair in estimation_pairs]
        
        # Mean Absolute Error
        mae = float(np.mean([abs(est - act) for est, act in zip(estimated_values, actual_values)]))
        
        # Mean Relative Error (as percentage)
        relative_errors = [
            abs(est - act) / max(est, act) * 100 
            for est, act in zip(estimated_values, actual_values)
            if max(est, act) > 0
        ]
        mre = float(np.mean(relative_errors)) if relative_errors else 0
        
        # Accuracy percentage (within 20% tolerance)
        accurate_estimates = sum(
            1 for est, act in zip(estimated_values, actual_values)
            if abs(est - act) / max(est, act) <= 0.2
        )
        accuracy_percentage = (accurate_estimates / len(estimation_pairs)) * 100 if estimation_pairs else 0
        
        return {
            "mean_absolute_error": mae,
            "mean_relative_error": mre,
            "accuracy_percentage": float(accuracy_percentage),
            "total_comparisons": len(estimation_pairs),
            "correlation_coefficient": float(stats.pearsonr(estimated_values, actual_values)[0]) if len(estimation_pairs) > 1 else 0
        }

    def _analyze_developer_estimation_accuracy(self, estimation_pairs: list[tuple[int, int, str]]) -> dict[str, dict[str, Any]]:
        """Analyze estimation accuracy by individual developer."""
        developer_pairs = defaultdict(list)
        
        for estimated, actual, developer in estimation_pairs:
            developer_pairs[developer].append((estimated, actual))
        
        developer_accuracy = {}
        
        for developer, pairs in developer_pairs.items():
            if len(pairs) >= 2:  # Need multiple estimates for meaningful analysis
                estimated_values = [pair[0] for pair in pairs]
                actual_values = [pair[1] for pair in pairs]
                
                # Calculate metrics for this developer
                mae = float(np.mean([abs(est - act) for est, act in zip(estimated_values, actual_values)]))
                
                relative_errors = [
                    abs(est - act) / max(est, act) * 100 
                    for est, act in zip(estimated_values, actual_values)
                ]
                mre = float(np.mean(relative_errors))
                
                accurate_count = sum(
                    1 for est, act in zip(estimated_values, actual_values)
                    if abs(est - act) / max(est, act) <= 0.2
                )
                accuracy_pct = (accurate_count / len(pairs)) * 100
                
                developer_accuracy[self._anonymize_value(developer, "name")] = {
                    "mean_absolute_error": mae,
                    "mean_relative_error": mre,
                    "accuracy_percentage": float(accuracy_pct),
                    "estimates_count": len(pairs),
                    "tends_to_overestimate": sum(estimated_values) > sum(actual_values),
                    "consistency": 1.0 - float(np.std(relative_errors) / 100) if relative_errors else 0
                }
        
        return developer_accuracy

    def _analyze_size_based_accuracy(self, estimation_pairs: list[tuple[int, int, str]]) -> dict[str, dict[str, Any]]:
        """Analyze estimation accuracy by story point size ranges."""
        size_ranges = {
            "small": (1, 3),
            "medium": (4, 8), 
            "large": (9, 21),
            "extra_large": (22, 100)
        }
        
        size_accuracy = {}
        
        for size_name, (min_sp, max_sp) in size_ranges.items():
            size_pairs = [
                (est, act) for est, act, _ in estimation_pairs
                if min_sp <= est <= max_sp
            ]
            
            if size_pairs:
                estimated_values = [pair[0] for pair in size_pairs]
                actual_values = [pair[1] for pair in size_pairs]
                
                mae = float(np.mean([abs(est - act) for est, act in zip(estimated_values, actual_values)]))
                
                relative_errors = [
                    abs(est - act) / max(est, act) * 100 
                    for est, act in zip(estimated_values, actual_values)
                ]
                mre = float(np.mean(relative_errors))
                
                size_accuracy[size_name] = {
                    "mean_absolute_error": mae,
                    "mean_relative_error": mre,
                    "sample_size": len(size_pairs),
                    "avg_estimated": float(np.mean(estimated_values)),
                    "avg_actual": float(np.mean(actual_values))
                }
            else:
                size_accuracy[size_name] = {
                    "mean_absolute_error": 0,
                    "mean_relative_error": 0,
                    "sample_size": 0,
                    "avg_estimated": 0,
                    "avg_actual": 0
                }
        
        return size_accuracy

    def _aggregate_weekly_velocity(
        self, commits: list[dict[str, Any]], prs: list[dict[str, Any]], start_date: datetime, end_date: datetime
    ) -> dict[str, dict[str, Any]]:
        """Aggregate velocity metrics by week."""
        weekly_velocity = defaultdict(lambda: {
            "story_points_completed": 0,
            "commits": 0,
            "prs_merged": 0,
            "developers_active": set()
        })
        
        # Process commits
        for commit in commits:
            timestamp = self._ensure_timezone_aware(commit.get("timestamp"))
            if not timestamp or timestamp < start_date or timestamp > end_date:
                continue
                
            week_start = self._get_week_start(timestamp)
            week_key = week_start.strftime("%Y-%m-%d")
            
            weekly_velocity[week_key]["story_points_completed"] += commit.get("story_points", 0) or 0
            weekly_velocity[week_key]["commits"] += 1
            weekly_velocity[week_key]["developers_active"].add(
                commit.get("canonical_id", commit.get("author_email", "unknown"))
            )
        
        # Process PRs
        for pr in prs:
            merged_at = self._ensure_timezone_aware(pr.get("merged_at"))
            if not merged_at or merged_at < start_date or merged_at > end_date:
                continue
                
            week_start = self._get_week_start(merged_at)
            week_key = week_start.strftime("%Y-%m-%d")
            
            weekly_velocity[week_key]["prs_merged"] += 1
        
        # Convert sets to counts
        result = {}
        for week_key, metrics in weekly_velocity.items():
            metrics["developers_active"] = len(metrics["developers_active"])
            result[week_key] = dict(metrics)
        
        return result

    def _calculate_velocity_trends(self, weekly_velocity: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Calculate velocity trend analysis."""
        if len(weekly_velocity) < 3:
            return {"trend": "insufficient_data", "velocity_change": 0}
        
        weeks = sorted(weekly_velocity.keys())
        story_points = [weekly_velocity[week]["story_points_completed"] for week in weeks]
        
        if not any(sp > 0 for sp in story_points):
            return {"trend": "no_story_points", "velocity_change": 0}
        
        # Calculate trend using linear regression
        x_values = list(range(len(weeks)))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, story_points)
        
        # Determine trend direction
        if slope > 0.5:
            trend = "improving"
        elif slope < -0.5:
            trend = "declining"
        else:
            trend = "stable"
        
        # Calculate velocity change (percentage)
        if len(story_points) >= 2:
            recent_avg = np.mean(story_points[-3:]) if len(story_points) >= 3 else story_points[-1]
            early_avg = np.mean(story_points[:3]) if len(story_points) >= 3 else story_points[0]
            velocity_change = ((recent_avg - early_avg) / max(early_avg, 1)) * 100
        else:
            velocity_change = 0
        
        return {
            "trend": trend,
            "velocity_change": float(velocity_change),
            "trend_strength": float(abs(r_value)),
            "slope": float(slope),
            "weeks_analyzed": len(weeks),
            "avg_velocity": float(np.mean(story_points)),
            "velocity_stability": 1.0 - float(np.std(story_points) / max(np.mean(story_points), 1))
        }

    def _analyze_developer_velocity(
        self, commits: list[dict[str, Any]], start_date: datetime, end_date: datetime
    ) -> dict[str, dict[str, Any]]:
        """Analyze individual developer velocity patterns."""
        developer_metrics = defaultdict(lambda: {
            "total_story_points": 0,
            "total_commits": 0,
            "weeks_active": set(),
            "weekly_velocity": []
        })
        
        # Aggregate by developer and week
        for commit in commits:
            timestamp = self._ensure_timezone_aware(commit.get("timestamp"))
            if not timestamp or timestamp < start_date or timestamp > end_date:
                continue
                
            developer_id = commit.get("canonical_id", commit.get("author_email", "unknown"))
            week_start = self._get_week_start(timestamp)
            
            metrics = developer_metrics[developer_id]
            metrics["total_story_points"] += commit.get("story_points", 0) or 0
            metrics["total_commits"] += 1
            metrics["weeks_active"].add(week_start)
        
        # Calculate velocity metrics for each developer
        developer_velocity = {}
        
        for dev_id, metrics in developer_metrics.items():
            if metrics["total_commits"] > 0:
                weeks_active = len(metrics["weeks_active"])
                avg_velocity = metrics["total_story_points"] / max(weeks_active, 1)
                
                developer_velocity[self._anonymize_value(dev_id, "name")] = {
                    "total_story_points": metrics["total_story_points"],
                    "total_commits": metrics["total_commits"],
                    "weeks_active": weeks_active,
                    "avg_weekly_velocity": float(avg_velocity),
                    "story_points_per_commit": metrics["total_story_points"] / metrics["total_commits"]
                }
        
        return developer_velocity

    def _calculate_velocity_predictability(self, weekly_velocity: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Calculate how predictable the team's velocity is."""
        if len(weekly_velocity) < 4:
            return {"predictability": "insufficient_data", "confidence_interval": [0, 0]}
        
        story_points = [metrics["story_points_completed"] for metrics in weekly_velocity.values()]
        
        if not any(sp > 0 for sp in story_points):
            return {"predictability": "no_velocity_data", "confidence_interval": [0, 0]}
        
        mean_velocity = np.mean(story_points)
        std_velocity = np.std(story_points)
        coefficient_variation = std_velocity / max(mean_velocity, 1)
        
        # Classify predictability
        if coefficient_variation < 0.2:
            predictability = "high"
        elif coefficient_variation < 0.4:
            predictability = "moderate"
        else:
            predictability = "low"
        
        # Calculate 80% confidence interval
        confidence_interval = [
            float(max(0, mean_velocity - 1.28 * std_velocity)),
            float(mean_velocity + 1.28 * std_velocity)
        ]
        
        return {
            "predictability": predictability,
            "coefficient_of_variation": float(coefficient_variation),
            "confidence_interval": confidence_interval,
            "mean_velocity": float(mean_velocity),
            "std_deviation": float(std_velocity)
        }

    def _analyze_team_capacity(
        self, weekly_velocity: dict[str, dict[str, Any]], developer_velocity: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze team capacity and workload distribution."""
        if not weekly_velocity or not developer_velocity:
            return {"analysis": "insufficient_data"}
        
        # Calculate team metrics
        total_developers = len(developer_velocity)
        weeks_analyzed = len(weekly_velocity)
        
        # Calculate capacity utilization
        developer_contributions = [dev["total_story_points"] for dev in developer_velocity.values()]
        total_story_points = sum(developer_contributions)
        
        if total_story_points == 0:
            return {"analysis": "no_story_points"}
        
        # Analyze workload distribution
        [
            (contrib / total_story_points) * 100 for contrib in developer_contributions
        ]
        
        # Calculate Gini coefficient for workload inequality
        sorted_contributions = sorted(developer_contributions)
        n = len(sorted_contributions)
        np.cumsum(sorted_contributions)
        gini = (n + 1 - 2 * sum((n + 1 - i) * x for i, x in enumerate(sorted_contributions, 1))) / (n * sum(sorted_contributions))
        
        # Capacity recommendations
        recommendations = []
        
        # Check for workload imbalance
        if gini > 0.4:  # High inequality
            recommendations.append("Consider redistributing workload - significant imbalance detected")
        
        # Check for low contributors
        low_contributors = [
            dev for dev, metrics in developer_velocity.items()
            if metrics["avg_weekly_velocity"] < np.mean([m["avg_weekly_velocity"] for m in developer_velocity.values()]) * 0.5
        ]
        
        if low_contributors:
            recommendations.append(f"Support developers with low velocity: {', '.join(low_contributors[:3])}")
        
        return {
            "total_developers": total_developers,
            "weeks_analyzed": weeks_analyzed,
            "total_story_points": total_story_points,
            "avg_weekly_team_velocity": float(np.mean([w["story_points_completed"] for w in weekly_velocity.values()])),
            "workload_distribution_gini": float(gini),
            "workload_balance": "balanced" if gini < 0.3 else "imbalanced",
            "capacity_recommendations": recommendations,
            "top_contributors": sorted(
                [(dev, metrics["total_story_points"]) for dev, metrics in developer_velocity.items()],
                key=lambda x: x[1], reverse=True
            )[:5]
        }

    def _build_correlation_csv_rows(
        self,
        weekly_correlations: dict[str, Any],
        estimation_accuracy: dict[str, Any],
        velocity_metrics: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Build CSV rows from correlation analysis results."""
        rows = []
        
        # Add weekly correlation data
        correlation_results = weekly_correlations.get("weekly_correlations", {})
        
        for week_start, week_data in correlation_results.items():
            correlations = week_data.get("correlations", {})
            
            row = {
                "week_start": week_start.strftime("%Y-%m-%d"),
                "metric_type": "weekly_correlations",
                "sp_commits_correlation": round(correlations.get("sp_commits", 0), 3),
                "sp_lines_correlation": round(correlations.get("sp_lines", 0), 3),
                "sp_files_correlation": round(correlations.get("sp_files", 0), 3),
                "sp_prs_correlation": round(correlations.get("sp_prs", 0), 3),
                "sp_complexity_correlation": round(correlations.get("sp_complexity", 0), 3),
                "sample_size": week_data.get("sample_size", 0),
                "total_story_points": week_data.get("total_story_points", 0),
                "total_commits": week_data.get("total_commits", 0)
            }
            rows.append(row)
        
        # Add velocity data
        weekly_velocity = velocity_metrics.get("weekly_velocity", {})
        for week_key, velocity_data in weekly_velocity.items():
            row = {
                "week_start": week_key,
                "metric_type": "velocity",
                "story_points_completed": velocity_data.get("story_points_completed", 0),
                "commits_count": velocity_data.get("commits", 0),
                "prs_merged": velocity_data.get("prs_merged", 0),
                "developers_active": velocity_data.get("developers_active", 0),
                "velocity_trend": velocity_metrics.get("trends", {}).get("trend", "unknown")
            }
            rows.append(row)
        
        # Add developer accuracy summary
        developer_accuracy = estimation_accuracy.get("developer_accuracy", {})
        for developer, accuracy_data in developer_accuracy.items():
            row = {
                "developer_name": developer,
                "metric_type": "developer_accuracy",
                "overall_accuracy": round(accuracy_data.get("overall_accuracy", 0), 3),
                "avg_weekly_accuracy": round(accuracy_data.get("avg_weekly_accuracy", 0), 3),
                "consistency": round(accuracy_data.get("consistency", 0), 3),
                "weeks_active": accuracy_data.get("weeks_active", 0),
                "total_estimated_sp": accuracy_data.get("total_estimated", 0),
                "total_actual_sp": accuracy_data.get("total_actual", 0),
                "estimation_ratio": round(accuracy_data.get("estimation_ratio", 0), 3)
            }
            rows.append(row)
        
        return rows

    def _write_empty_correlation_csv(self, output_path: Path) -> None:
        """Write empty CSV file with proper headers."""
        headers = [
            "week_start", "metric_type", "developer_name",
            "sp_commits_correlation", "sp_lines_correlation", "sp_files_correlation",
            "sp_prs_correlation", "sp_complexity_correlation", "sample_size",
            "total_story_points", "total_commits", "story_points_completed",
            "commits_count", "prs_merged", "developers_active", "velocity_trend",
            "overall_accuracy", "avg_weekly_accuracy", "consistency",
            "weeks_active", "total_estimated_sp", "total_actual_sp", "estimation_ratio"
        ]
        
        df = pd.DataFrame(columns=headers)
        df.to_csv(output_path, index=False)

    def _empty_accuracy_analysis(self) -> dict[str, Any]:
        """Return empty accuracy analysis structure."""
        return {
            "overall_accuracy": {"mean_absolute_error": 0, "mean_relative_error": 0, "accuracy_percentage": 0},
            "developer_accuracy": {},
            "size_based_accuracy": {},
            "improvement_suggestions": []
        }

    def _empty_week_correlations(self) -> dict[str, Any]:
        """Return empty week correlations structure."""
        return {
            "correlations": {k: 0.0 for k in ["sp_commits", "sp_lines", "sp_files", "sp_prs", "sp_complexity"]},
            "sample_size": 0,
            "total_story_points": 0,
            "total_commits": 0,
            "total_lines_changed": 0
        }

    def _generate_accuracy_recommendations(
        self, accuracy_metrics: dict[str, Any], developer_accuracy: dict[str, dict[str, Any]], size_accuracy: dict[str, dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Generate recommendations for improving estimation accuracy."""
        recommendations = []
        
        overall_accuracy = accuracy_metrics.get("accuracy_percentage", 0)
        
        if overall_accuracy < 50:
            recommendations.append({
                "priority": "high",
                "title": "Low Overall Estimation Accuracy",
                "description": f"Only {overall_accuracy:.1f}% of estimates are within 20% tolerance",
                "action": "Conduct team workshop on story point estimation techniques"
            })
        
        # Check for developers with low accuracy
        low_accuracy_devs = [
            dev for dev, stats in developer_accuracy.items()
            if stats.get("overall_accuracy", 0) < 0.4
        ]
        
        if low_accuracy_devs:
            recommendations.append({
                "priority": "medium", 
                "title": "Individual Estimation Training Needed",
                "description": f"{len(low_accuracy_devs)} developers need estimation improvement",
                "action": f"Provide 1-on-1 training for: {', '.join(low_accuracy_devs[:3])}"
            })
        
        # Check size-based accuracy patterns
        large_story_accuracy = size_accuracy.get("large", {}).get("mean_relative_error", 0)
        if large_story_accuracy > 40:  # High error rate for large stories
            recommendations.append({
                "priority": "medium",
                "title": "Large Stories Are Poorly Estimated", 
                "description": f"Large stories (9-21 pts) have {large_story_accuracy:.1f}% average error",
                "action": "Encourage breaking down large stories into smaller, more estimable pieces"
            })
        
        return recommendations

    def _ensure_timezone_aware(self, dt: Any) -> Optional[datetime]:
        """Ensure datetime is timezone-aware UTC."""
        if not dt:
            return None
            
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                return None
        
        if not isinstance(dt, datetime):
            return None
            
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        elif dt.tzinfo != timezone.utc:
            return dt.astimezone(timezone.utc)
        else:
            return dt

    def _get_week_start(self, date: datetime) -> datetime:
        """Get Monday of the week for consistent week boundaries."""
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        elif date.tzinfo != timezone.utc:
            date = date.astimezone(timezone.utc)
        
        days_since_monday = date.weekday()
        monday = date - timedelta(days=days_since_monday)
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)

    def _anonymize_value(self, value: str, field_type: str) -> str:
        """Anonymize values if anonymization is enabled."""
        if not self.anonymize or not value:
            return value
        
        if value not in self._anonymization_map:
            self._anonymous_counter += 1
            if field_type == "name":
                anonymous = f"Developer{self._anonymous_counter}"
            elif field_type == "id": 
                anonymous = f"ID{self._anonymous_counter:04d}"
            else:
                anonymous = f"anon{self._anonymous_counter}"
            
            self._anonymization_map[value] = anonymous
        
        return self._anonymization_map[value]