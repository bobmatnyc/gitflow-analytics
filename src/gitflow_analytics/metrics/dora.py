"""DORA (DevOps Research and Assessment) metrics calculation."""

from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pytz


class DORAMetricsCalculator:
    """Calculate DORA metrics for software delivery performance."""

    def __init__(self) -> None:
        """Initialize DORA metrics calculator."""
        self.deployment_patterns = ["deploy", "release", "ship", "live", "production", "prod"]
        self.failure_patterns = ["revert", "rollback", "hotfix", "emergency", "incident", "outage"]

    def _normalize_timestamp_to_utc(self, timestamp: Optional[datetime]) -> Optional[datetime]:
        """Normalize any timestamp to UTC timezone-aware datetime.

        WHY: Ensures all timestamps are timezone-aware UTC to prevent
        comparison errors when sorting mixed timezone objects.

        Args:
            timestamp: DateTime object that may be timezone-naive, timezone-aware, or None

        Returns:
            Timezone-aware datetime in UTC, or None if input is None
        """
        if timestamp is None:
            return None

        if timestamp.tzinfo is None:
            # Assume naive timestamps are UTC
            return timestamp.replace(tzinfo=pytz.UTC)
        else:
            # Convert timezone-aware timestamps to UTC
            return timestamp.astimezone(pytz.UTC)

    def calculate_dora_metrics(
        self,
        commits: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Calculate the four key DORA metrics."""

        # Identify deployments and failures
        deployments = self._identify_deployments(commits, prs)
        failures = self._identify_failures(commits, prs)

        # Calculate metrics
        deployment_frequency = self._calculate_deployment_frequency(
            deployments, start_date, end_date
        )

        lead_time = self._calculate_lead_time(prs, deployments)

        change_failure_rate = self._calculate_change_failure_rate(deployments, failures)

        mttr = self._calculate_mttr(failures, commits)

        # Determine performance level
        performance_level = self._determine_performance_level(
            deployment_frequency, lead_time, change_failure_rate, mttr
        )

        return {
            "deployment_frequency": deployment_frequency,
            "lead_time_hours": lead_time,
            "change_failure_rate": change_failure_rate,
            "mttr_hours": mttr,
            "performance_level": performance_level,
            "total_deployments": len(deployments),
            "total_failures": len(failures),
            "metrics_period_weeks": (end_date - start_date).days / 7,
        }

    def _identify_deployments(
        self, commits: list[dict[str, Any]], prs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Identify deployment events from commits and PRs."""
        deployments = []

        # Check commits for deployment patterns
        for commit in commits:
            message_lower = commit["message"].lower()
            if any(pattern in message_lower for pattern in self.deployment_patterns):
                deployments.append(
                    {
                        "type": "commit",
                        "timestamp": self._normalize_timestamp_to_utc(commit["timestamp"]),
                        "identifier": commit["hash"],
                        "message": commit["message"],
                    }
                )

        # Check PR titles and labels for deployments
        for pr in prs:
            # Check title
            title_lower = pr.get("title", "").lower()
            if any(pattern in title_lower for pattern in self.deployment_patterns):
                raw_timestamp = pr.get("merged_at", pr.get("created_at"))
                deployments.append(
                    {
                        "type": "pr",
                        "timestamp": self._normalize_timestamp_to_utc(raw_timestamp),
                        "identifier": f"PR#{pr.get('number', 'unknown')}",
                        "message": pr["title"],
                    }
                )
                continue

            # Check labels
            labels_lower = [label.lower() for label in pr.get("labels", [])]
            if any(
                any(pattern in label for pattern in self.deployment_patterns)
                for label in labels_lower
            ):
                raw_timestamp = pr.get("merged_at", pr.get("created_at"))
                deployments.append(
                    {
                        "type": "pr",
                        "timestamp": self._normalize_timestamp_to_utc(raw_timestamp),
                        "identifier": f"PR#{pr.get('number', 'unknown')}",
                        "message": pr["title"],
                    }
                )

        # Filter out deployments with None timestamps
        deployments = [d for d in deployments if d["timestamp"] is not None]

        # Remove duplicates and sort by timestamp (now all are timezone-aware UTC)
        seen = set()
        unique_deployments = []
        for dep in sorted(deployments, key=lambda x: x["timestamp"]):
            key = f"{dep['type']}:{dep['identifier']}"
            if key not in seen:
                seen.add(key)
                unique_deployments.append(dep)

        return unique_deployments

    def _identify_failures(
        self, commits: list[dict[str, Any]], prs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Identify failure events from commits and PRs."""
        failures = []

        # Check commits for failure patterns
        for commit in commits:
            message_lower = commit["message"].lower()
            if any(pattern in message_lower for pattern in self.failure_patterns):
                failures.append(
                    {
                        "type": "commit",
                        "timestamp": self._normalize_timestamp_to_utc(commit["timestamp"]),
                        "identifier": commit["hash"],
                        "message": commit["message"],
                        "is_hotfix": "hotfix" in message_lower or "emergency" in message_lower,
                    }
                )

        # Check PRs for failure patterns
        for pr in prs:
            title_lower = pr.get("title", "").lower()
            labels_lower = [label.lower() for label in pr.get("labels", [])]

            is_failure = any(pattern in title_lower for pattern in self.failure_patterns) or any(
                any(pattern in label for pattern in self.failure_patterns) for label in labels_lower
            )

            if is_failure:
                raw_timestamp = pr.get("merged_at", pr.get("created_at"))
                failures.append(
                    {
                        "type": "pr",
                        "timestamp": self._normalize_timestamp_to_utc(raw_timestamp),
                        "identifier": f"PR#{pr.get('number', 'unknown')}",
                        "message": pr["title"],
                        "is_hotfix": "hotfix" in title_lower or "emergency" in title_lower,
                    }
                )

        # Filter out failures with None timestamps
        failures = [f for f in failures if f["timestamp"] is not None]

        return failures

    def _calculate_deployment_frequency(
        self, deployments: list[dict[str, Any]], start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Calculate deployment frequency metrics."""
        if not deployments:
            return {"daily_average": 0, "weekly_average": 0, "category": "Low"}

        # Normalize date range to timezone-aware UTC
        start_date_utc = self._normalize_timestamp_to_utc(start_date)
        end_date_utc = self._normalize_timestamp_to_utc(end_date)

        # Handle case where normalization failed
        if start_date_utc is None or end_date_utc is None:
            return {"daily_average": 0, "weekly_average": 0, "category": "Low"}

        # Filter deployments in date range (timestamps are already normalized to UTC)
        period_deployments = [
            d for d in deployments if start_date_utc <= d["timestamp"] <= end_date_utc
        ]

        days = (end_date_utc - start_date_utc).days
        weeks = days / 7

        daily_avg = len(period_deployments) / days if days > 0 else 0
        weekly_avg = len(period_deployments) / weeks if weeks > 0 else 0

        # Categorize based on DORA standards
        if daily_avg >= 1:
            category = "Elite"  # Multiple deploys per day
        elif weekly_avg >= 1:
            category = "High"  # Between once per day and once per week
        elif weekly_avg >= 0.25:
            category = "Medium"  # Between once per week and once per month
        else:
            category = "Low"  # Less than once per month

        return {"daily_average": daily_avg, "weekly_average": weekly_avg, "category": category}

    def _calculate_lead_time(
        self, prs: list[dict[str, Any]], deployments: list[dict[str, Any]]
    ) -> float:
        """Calculate lead time for changes in hours."""
        if not prs:
            return 0

        lead_times = []

        for pr in prs:
            if not pr.get("created_at") or not pr.get("merged_at"):
                continue

            # Calculate time from PR creation to merge
            # Normalize both timestamps to UTC
            created_at = self._normalize_timestamp_to_utc(pr["created_at"])
            merged_at = self._normalize_timestamp_to_utc(pr["merged_at"])

            # Skip if either timestamp is None after normalization
            if created_at is None or merged_at is None:
                continue

            lead_time = (merged_at - created_at).total_seconds() / 3600
            lead_times.append(lead_time)

        if not lead_times:
            return 0

        # Return median lead time
        return float(np.median(lead_times))

    def _calculate_change_failure_rate(
        self, deployments: list[dict[str, Any]], failures: list[dict[str, Any]]
    ) -> float:
        """Calculate the percentage of deployments causing failures."""
        if not deployments:
            return 0

        # Count failures that occurred within 24 hours of a deployment
        failure_causing_deployments = 0

        for deployment in deployments:
            deploy_time = deployment["timestamp"]  # Already normalized to UTC

            # Check if any failure occurred within 24 hours
            for failure in failures:
                failure_time = failure["timestamp"]  # Already normalized to UTC

                time_diff = abs((failure_time - deploy_time).total_seconds() / 3600)

                if time_diff <= 24:  # Within 24 hours
                    failure_causing_deployments += 1
                    break

        return (failure_causing_deployments / len(deployments)) * 100

    def _calculate_mttr(
        self, failures: list[dict[str, Any]], commits: list[dict[str, Any]]
    ) -> float:
        """Calculate mean time to recovery in hours."""
        if not failures:
            return 0

        recovery_times = []

        # For each failure, find the recovery time
        for _i, failure in enumerate(failures):
            failure_time = failure["timestamp"]  # Already normalized to UTC

            # Look for recovery indicators in subsequent commits
            recovery_time = None

            # Check subsequent commits for recovery patterns
            for commit in commits:
                commit_time = self._normalize_timestamp_to_utc(commit["timestamp"])

                if commit_time <= failure_time:
                    continue

                message_lower = commit["message"].lower()
                recovery_patterns = ["fixed", "resolved", "recovery", "restored"]

                if any(pattern in message_lower for pattern in recovery_patterns):
                    recovery_time = commit_time
                    break

            # If we found a recovery, calculate MTTR
            if recovery_time:
                mttr = (recovery_time - failure_time).total_seconds() / 3600
                recovery_times.append(mttr)
            # For hotfixes, assume quick recovery (2 hours)
            elif failure.get("is_hotfix"):
                recovery_times.append(2.0)

        if not recovery_times:
            # If no explicit recovery found, estimate based on failure type
            return 4.0  # Default 4 hours

        return float(np.mean(recovery_times))

    def _determine_performance_level(
        self,
        deployment_freq: dict[str, Any],
        lead_time_hours: float,
        change_failure_rate: float,
        mttr_hours: float,
    ) -> str:
        """Determine overall performance level based on DORA metrics."""
        scores = []

        # Deployment frequency score
        freq_category = deployment_freq["category"]
        freq_scores = {"Elite": 4, "High": 3, "Medium": 2, "Low": 1}
        scores.append(freq_scores.get(freq_category, 1))

        # Lead time score
        if lead_time_hours < 24:  # Less than one day
            scores.append(4)  # Elite
        elif lead_time_hours < 168:  # Less than one week
            scores.append(3)  # High
        elif lead_time_hours < 720:  # Less than one month
            scores.append(2)  # Medium
        else:
            scores.append(1)  # Low

        # Change failure rate score
        if change_failure_rate <= 15:
            scores.append(4)  # Elite (0-15%)
        elif change_failure_rate <= 20:
            scores.append(3)  # High
        elif change_failure_rate <= 30:
            scores.append(2)  # Medium
        else:
            scores.append(1)  # Low

        # MTTR score
        if mttr_hours < 1:  # Less than one hour
            scores.append(4)  # Elite
        elif mttr_hours < 24:  # Less than one day
            scores.append(3)  # High
        elif mttr_hours < 168:  # Less than one week
            scores.append(2)  # Medium
        else:
            scores.append(1)  # Low

        # Average score determines overall level
        avg_score = sum(scores) / len(scores)

        if avg_score >= 3.5:
            return "Elite"
        elif avg_score >= 2.5:
            return "High"
        elif avg_score >= 1.5:
            return "Medium"
        else:
            return "Low"

    def calculate_weekly_dora_metrics(
        self,
        commits: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict[str, Any]]:
        """Calculate DORA metrics broken down by week.

        WHY: Weekly breakdowns provide trend analysis and enable identification
        of performance patterns over time. This helps teams track improvements
        and identify periods of degraded performance.

        DESIGN DECISION: Uses Monday-Sunday week boundaries for consistency
        with other reporting functions. Includes rolling averages to smooth
        out weekly variations and provide clearer trend indicators.

        Args:
            commits: List of commit data dictionaries
            prs: List of pull request data dictionaries
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            List of weekly DORA metrics with trend analysis
        """
        # Normalize date range to timezone-aware UTC
        start_date_utc = self._normalize_timestamp_to_utc(start_date)
        end_date_utc = self._normalize_timestamp_to_utc(end_date)

        if start_date_utc is None or end_date_utc is None:
            return []

        # Identify deployments and failures for the entire period
        all_deployments = self._identify_deployments(commits, prs)
        all_failures = self._identify_failures(commits, prs)

        # Generate week boundaries
        weeks = self._generate_week_boundaries(start_date_utc, end_date_utc)

        weekly_metrics = []
        previous_weeks_data = []  # For rolling averages

        for week_start, week_end in weeks:
            # Filter data for this week
            week_deployments = [
                d for d in all_deployments if week_start <= d["timestamp"] <= week_end
            ]

            week_failures = [f for f in all_failures if week_start <= f["timestamp"] <= week_end]

            week_commits = [
                c
                for c in commits
                if week_start <= self._normalize_timestamp_to_utc(c["timestamp"]) <= week_end
            ]

            week_prs = [
                pr
                for pr in prs
                if pr.get("merged_at")
                and week_start <= self._normalize_timestamp_to_utc(pr["merged_at"]) <= week_end
            ]

            # Calculate weekly metrics
            deployment_frequency = len(week_deployments)

            # Calculate lead time for PRs merged this week
            lead_times = []
            for pr in week_prs:
                if pr.get("created_at") and pr.get("merged_at"):
                    created_at = self._normalize_timestamp_to_utc(pr["created_at"])
                    merged_at = self._normalize_timestamp_to_utc(pr["merged_at"])

                    if created_at and merged_at:
                        lead_time = (merged_at - created_at).total_seconds() / 3600
                        lead_times.append(lead_time)

            avg_lead_time = float(np.median(lead_times)) if lead_times else 0.0

            # Calculate change failure rate
            change_failure_rate = 0.0
            if week_deployments:
                failure_causing_deployments = 0
                for deployment in week_deployments:
                    deploy_time = deployment["timestamp"]

                    # Check if any failure occurred within 24 hours
                    for failure in week_failures:
                        failure_time = failure["timestamp"]
                        time_diff = abs((failure_time - deploy_time).total_seconds() / 3600)

                        if time_diff <= 24:  # Within 24 hours
                            failure_causing_deployments += 1
                            break

                change_failure_rate = (failure_causing_deployments / len(week_deployments)) * 100

            # Calculate MTTR for failures this week
            recovery_times = []
            for failure in week_failures:
                failure_time = failure["timestamp"]

                # Look for recovery in subsequent commits within reasonable time
                recovery_time = None
                for commit in week_commits:
                    commit_time = self._normalize_timestamp_to_utc(commit["timestamp"])

                    if commit_time <= failure_time:
                        continue

                    message_lower = commit["message"].lower()
                    recovery_patterns = ["fixed", "resolved", "recovery", "restored"]

                    if any(pattern in message_lower for pattern in recovery_patterns):
                        recovery_time = commit_time
                        break

                if recovery_time:
                    mttr = (recovery_time - failure_time).total_seconds() / 3600
                    recovery_times.append(mttr)
                elif failure.get("is_hotfix"):
                    recovery_times.append(2.0)  # Assume quick recovery for hotfixes

            avg_mttr = float(np.mean(recovery_times)) if recovery_times else 0.0

            # Store current week data
            week_data = {
                "week_start": week_start.strftime("%Y-%m-%d"),
                "week_end": week_end.strftime("%Y-%m-%d"),
                "deployment_frequency": deployment_frequency,
                "lead_time_hours": round(avg_lead_time, 2),
                "change_failure_rate": round(change_failure_rate, 2),
                "mttr_hours": round(avg_mttr, 2),
                "total_failures": len(week_failures),
                "total_commits": len(week_commits),
                "total_prs": len(week_prs),
            }

            # Calculate rolling averages (4-week window)
            previous_weeks_data.append(week_data.copy())
            if len(previous_weeks_data) > 4:
                previous_weeks_data.pop(0)

            # 4-week rolling averages
            if len(previous_weeks_data) >= 2:
                week_data["deployment_frequency_4w_avg"] = round(
                    np.mean([w["deployment_frequency"] for w in previous_weeks_data]), 1
                )

                lead_times_4w = [
                    w["lead_time_hours"] for w in previous_weeks_data if w["lead_time_hours"] > 0
                ]
                week_data["lead_time_4w_avg"] = round(
                    np.mean(lead_times_4w) if lead_times_4w else 0, 1
                )

                cfr_4w = [
                    w["change_failure_rate"]
                    for w in previous_weeks_data
                    if w["change_failure_rate"] > 0
                ]
                week_data["change_failure_rate_4w_avg"] = round(np.mean(cfr_4w) if cfr_4w else 0, 1)

                mttr_4w = [w["mttr_hours"] for w in previous_weeks_data if w["mttr_hours"] > 0]
                week_data["mttr_4w_avg"] = round(np.mean(mttr_4w) if mttr_4w else 0, 1)
            else:
                week_data["deployment_frequency_4w_avg"] = week_data["deployment_frequency"]
                week_data["lead_time_4w_avg"] = week_data["lead_time_hours"]
                week_data["change_failure_rate_4w_avg"] = week_data["change_failure_rate"]
                week_data["mttr_4w_avg"] = week_data["mttr_hours"]

            # Calculate week-over-week changes (if we have previous week)
            if len(weekly_metrics) > 0:
                prev_week = weekly_metrics[-1]

                # Deployment frequency change
                if prev_week["deployment_frequency"] > 0:
                    df_change = (
                        (week_data["deployment_frequency"] - prev_week["deployment_frequency"])
                        / prev_week["deployment_frequency"]
                        * 100
                    )
                    week_data["deployment_frequency_change_pct"] = round(df_change, 1)
                else:
                    week_data["deployment_frequency_change_pct"] = (
                        0.0 if week_data["deployment_frequency"] == 0 else 100.0
                    )

                # Lead time change
                if prev_week["lead_time_hours"] > 0:
                    lt_change = (
                        (week_data["lead_time_hours"] - prev_week["lead_time_hours"])
                        / prev_week["lead_time_hours"]
                        * 100
                    )
                    week_data["lead_time_change_pct"] = round(lt_change, 1)
                else:
                    week_data["lead_time_change_pct"] = (
                        0.0 if week_data["lead_time_hours"] == 0 else 100.0
                    )

                # Change failure rate change
                if prev_week["change_failure_rate"] > 0:
                    cfr_change = (
                        (week_data["change_failure_rate"] - prev_week["change_failure_rate"])
                        / prev_week["change_failure_rate"]
                        * 100
                    )
                    week_data["change_failure_rate_change_pct"] = round(cfr_change, 1)
                else:
                    week_data["change_failure_rate_change_pct"] = (
                        0.0 if week_data["change_failure_rate"] == 0 else 100.0
                    )

                # MTTR change
                if prev_week["mttr_hours"] > 0:
                    mttr_change = (
                        (week_data["mttr_hours"] - prev_week["mttr_hours"])
                        / prev_week["mttr_hours"]
                        * 100
                    )
                    week_data["mttr_change_pct"] = round(mttr_change, 1)
                else:
                    week_data["mttr_change_pct"] = 0.0 if week_data["mttr_hours"] == 0 else 100.0
            else:
                # First week - no changes to calculate
                week_data["deployment_frequency_change_pct"] = 0.0
                week_data["lead_time_change_pct"] = 0.0
                week_data["change_failure_rate_change_pct"] = 0.0
                week_data["mttr_change_pct"] = 0.0

            # Add trend indicators
            week_data["deployment_frequency_trend"] = self._get_trend_indicator(
                week_data["deployment_frequency_change_pct"], "higher_better"
            )
            week_data["lead_time_trend"] = self._get_trend_indicator(
                week_data["lead_time_change_pct"], "lower_better"
            )
            week_data["change_failure_rate_trend"] = self._get_trend_indicator(
                week_data["change_failure_rate_change_pct"], "lower_better"
            )
            week_data["mttr_trend"] = self._get_trend_indicator(
                week_data["mttr_change_pct"], "lower_better"
            )

            weekly_metrics.append(week_data)

        return weekly_metrics

    def _generate_week_boundaries(
        self, start_date: datetime, end_date: datetime
    ) -> list[tuple[datetime, datetime]]:
        """Generate Monday-Sunday week boundaries for the given date range.

        WHY: Consistent week boundaries ensure that weekly metrics align with
        other reporting functions and provide predictable time buckets for analysis.

        Args:
            start_date: Start of analysis period (timezone-aware UTC)
            end_date: End of analysis period (timezone-aware UTC)

        Returns:
            List of (week_start, week_end) tuples with Monday-Sunday boundaries
        """
        weeks = []

        # Find the Monday of the week containing start_date
        days_since_monday = start_date.weekday()
        current_week_start = start_date - timedelta(days=days_since_monday)
        current_week_start = current_week_start.replace(hour=0, minute=0, second=0, microsecond=0)

        while current_week_start <= end_date:
            week_end = current_week_start + timedelta(
                days=6, hours=23, minutes=59, seconds=59, microseconds=999999
            )

            # Only include weeks that overlap with our analysis period
            if week_end >= start_date:
                weeks.append((current_week_start, week_end))

            current_week_start += timedelta(days=7)

        return weeks

    def _get_trend_indicator(self, change_pct: float, direction: str) -> str:
        """Get trend indicator based on change percentage and desired direction.

        WHY: Provides intuitive trend indicators that account for whether
        increases or decreases are desirable for each metric.

        Args:
            change_pct: Percentage change from previous period
            direction: "higher_better" or "lower_better"

        Returns:
            Trend indicator: "improving", "declining", or "stable"
        """
        if abs(change_pct) < 5:  # Less than 5% change considered stable
            return "stable"

        if direction == "higher_better":
            return "improving" if change_pct > 0 else "declining"
        else:  # lower_better
            return "improving" if change_pct < 0 else "declining"
