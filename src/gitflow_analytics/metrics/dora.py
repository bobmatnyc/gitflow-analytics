"""DORA (DevOps Research and Assessment) metrics calculation."""

from datetime import datetime
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
