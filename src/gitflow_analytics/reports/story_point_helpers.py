"""Helper methods for StoryPointCorrelationAnalyzer.

Extracted from story_point_correlation.py to keep file sizes manageable.
Contains estimation pair extraction, accuracy calculation, velocity analysis,
and CSV building helpers.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class StoryPointHelpersMixin:
    """Mixin providing helper methods for StoryPointCorrelationAnalyzer.

    Attributes expected from host class:
        anonymize: bool
        _anonymization_map: dict
        _anonymous_counter: int
    """

    # Stub attributes for IDE support
    anonymize: bool
    _anonymization_map: dict
    _anonymous_counter: int

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