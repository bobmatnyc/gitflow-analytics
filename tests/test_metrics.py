"""
Tests for the metrics module.

These tests verify DORA metrics calculation and other performance indicators.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone, timedelta

from gitflow_analytics.metrics.dora import (
    DORAMetrics,
    DeploymentFrequency,
    LeadTime,
    ChangeFailureRate,
    RecoveryTime,
)


class TestDORAMetrics:
    """Test cases for DORA metrics calculation."""

    def test_init(self):
        """Test DORAMetrics initialization."""
        dora = DORAMetrics()

        assert isinstance(dora.deployment_frequency, DeploymentFrequency)
        assert isinstance(dora.lead_time, LeadTime)
        assert isinstance(dora.change_failure_rate, ChangeFailureRate)
        assert isinstance(dora.recovery_time, RecoveryTime)

    def test_calculate_all_metrics(self):
        """Test calculating all DORA metrics together."""
        dora = DORAMetrics()

        # Sample data
        commits = [
            Mock(date=datetime(2024, 1, 1, tzinfo=timezone.utc), hash="abc123"),
            Mock(date=datetime(2024, 1, 2, tzinfo=timezone.utc), hash="def456"),
            Mock(date=datetime(2024, 1, 5, tzinfo=timezone.utc), hash="ghi789"),
        ]

        deployments = [
            {
                "date": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "success": True,
                "commit_hash": "abc123",
            },
            {
                "date": datetime(2024, 1, 5, tzinfo=timezone.utc),
                "success": True,
                "commit_hash": "ghi789",
            },
            {
                "date": datetime(2024, 1, 10, tzinfo=timezone.utc),
                "success": False,
                "commit_hash": "jkl012",
            },
        ]

        incidents = [
            {
                "date": datetime(2024, 1, 10, tzinfo=timezone.utc),
                "resolved_date": datetime(2024, 1, 10, 2, tzinfo=timezone.utc),
                "severity": "high",
            }
        ]

        metrics = dora.calculate_all_metrics(commits, deployments, incidents)

        assert "deployment_frequency" in metrics
        assert "lead_time" in metrics
        assert "change_failure_rate" in metrics
        assert "recovery_time" in metrics

        # Verify metrics have expected structure
        assert isinstance(metrics["deployment_frequency"], dict)
        assert isinstance(metrics["lead_time"], dict)
        assert isinstance(metrics["change_failure_rate"], dict)
        assert isinstance(metrics["recovery_time"], dict)


class TestDeploymentFrequency:
    """Test cases for deployment frequency metric."""

    def test_init(self):
        """Test DeploymentFrequency initialization."""
        df = DeploymentFrequency()
        assert df is not None

    def test_calculate_daily_frequency(self):
        """Test calculating daily deployment frequency."""
        df = DeploymentFrequency()

        # Sample deployments over 7 days
        deployments = [
            {"date": datetime(2024, 1, 1, tzinfo=timezone.utc), "success": True},
            {"date": datetime(2024, 1, 2, tzinfo=timezone.utc), "success": True},
            {"date": datetime(2024, 1, 3, tzinfo=timezone.utc), "success": True},
            {"date": datetime(2024, 1, 5, tzinfo=timezone.utc), "success": True},
            {"date": datetime(2024, 1, 7, tzinfo=timezone.utc), "success": False},
        ]

        frequency = df.calculate(deployments, period_days=7)

        assert "deployments_per_day" in frequency
        assert "total_deployments" in frequency
        assert "successful_deployments" in frequency
        assert "period_days" in frequency

        assert frequency["total_deployments"] == 5
        assert frequency["successful_deployments"] == 4
        assert frequency["deployments_per_day"] == 5 / 7

    def test_calculate_weekly_frequency(self):
        """Test calculating weekly deployment frequency."""
        df = DeploymentFrequency()

        # Sample deployments over 4 weeks (28 days)
        deployments = []
        for week in range(4):
            for day in range(2):  # 2 deployments per week
                date = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=week * 7 + day)
                deployments.append({"date": date, "success": True})

        frequency = df.calculate(deployments, period_days=28)

        assert frequency["total_deployments"] == 8
        weekly_frequency = frequency["deployments_per_day"] * 7
        assert abs(weekly_frequency - 2.0) < 0.1  # Approximately 2 per week

    def test_calculate_empty_deployments(self):
        """Test calculating frequency with no deployments."""
        df = DeploymentFrequency()

        frequency = df.calculate([], period_days=7)

        assert frequency["deployments_per_day"] == 0
        assert frequency["total_deployments"] == 0
        assert frequency["successful_deployments"] == 0

    def test_classify_frequency(self):
        """Test deployment frequency classification."""
        df = DeploymentFrequency()

        # Test elite frequency (multiple per day)
        assert df.classify_frequency(5.0) == "Elite"  # 5 per day

        # Test high frequency (daily to weekly)
        assert df.classify_frequency(1.0) == "High"  # 1 per day
        assert df.classify_frequency(0.5) == "High"  # 3.5 per week

        # Test medium frequency (weekly to monthly)
        assert df.classify_frequency(0.1) == "Medium"  # 0.7 per week

        # Test low frequency (less than monthly)
        assert df.classify_frequency(0.01) == "Low"  # 0.07 per week


class TestLeadTime:
    """Test cases for lead time metric."""

    def test_init(self):
        """Test LeadTime initialization."""
        lt = LeadTime()
        assert lt is not None

    def test_calculate_commit_to_deploy(self):
        """Test calculating lead time from commit to deployment."""
        lt = LeadTime()

        commits = [
            Mock(hash="abc123", date=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)),
            Mock(hash="def456", date=datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc)),
            Mock(hash="ghi789", date=datetime(2024, 1, 3, 9, 0, tzinfo=timezone.utc)),
        ]

        deployments = [
            {
                "date": datetime(2024, 1, 1, 16, 0, tzinfo=timezone.utc),
                "commit_hash": "abc123",
                "success": True,
            },
            {
                "date": datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc),
                "commit_hash": "def456",
                "success": True,
            },
        ]

        lead_time = lt.calculate(commits, deployments)

        assert "average_hours" in lead_time
        assert "median_hours" in lead_time
        assert "samples" in lead_time

        assert lead_time["samples"] == 2
        # First commit: 6 hours lead time, Second commit: ~20 hours lead time
        assert 10 <= lead_time["average_hours"] <= 15

    def test_calculate_no_matching_deployments(self):
        """Test lead time calculation with no matching deployments."""
        lt = LeadTime()

        commits = [Mock(hash="abc123", date=datetime(2024, 1, 1, tzinfo=timezone.utc))]

        deployments = [
            {
                "date": datetime(2024, 1, 2, tzinfo=timezone.utc),
                "commit_hash": "xyz999",
                "success": True,
            }
        ]

        lead_time = lt.calculate(commits, deployments)

        assert lead_time["samples"] == 0
        assert lead_time["average_hours"] == 0
        assert lead_time["median_hours"] == 0

    def test_classify_lead_time(self):
        """Test lead time classification."""
        lt = LeadTime()

        # Test elite lead time (less than 1 hour)
        assert lt.classify_lead_time(0.5) == "Elite"

        # Test high lead time (1 hour to 1 day)
        assert lt.classify_lead_time(12) == "High"

        # Test medium lead time (1 day to 1 week)
        assert lt.classify_lead_time(72) == "Medium"  # 3 days

        # Test low lead time (more than 1 week)
        assert lt.classify_lead_time(200) == "Low"  # Over 8 days


class TestChangeFailureRate:
    """Test cases for change failure rate metric."""

    def test_init(self):
        """Test ChangeFailureRate initialization."""
        cfr = ChangeFailureRate()
        assert cfr is not None

    def test_calculate_from_deployments(self):
        """Test calculating change failure rate from deployment data."""
        cfr = ChangeFailureRate()

        deployments = [
            {"date": datetime(2024, 1, 1, tzinfo=timezone.utc), "success": True},
            {"date": datetime(2024, 1, 2, tzinfo=timezone.utc), "success": True},
            {"date": datetime(2024, 1, 3, tzinfo=timezone.utc), "success": False},
            {"date": datetime(2024, 1, 4, tzinfo=timezone.utc), "success": True},
            {"date": datetime(2024, 1, 5, tzinfo=timezone.utc), "success": False},
        ]

        failure_rate = cfr.calculate(deployments)

        assert "failure_rate" in failure_rate
        assert "total_deployments" in failure_rate
        assert "failed_deployments" in failure_rate

        assert failure_rate["total_deployments"] == 5
        assert failure_rate["failed_deployments"] == 2
        assert failure_rate["failure_rate"] == 0.4  # 40% failure rate

    def test_calculate_with_incidents(self):
        """Test calculating change failure rate including incidents."""
        cfr = ChangeFailureRate()

        deployments = [
            {"date": datetime(2024, 1, 1, tzinfo=timezone.utc), "success": True},
            {"date": datetime(2024, 1, 2, tzinfo=timezone.utc), "success": True},
        ]

        incidents = [
            {
                "date": datetime(2024, 1, 1, 2, tzinfo=timezone.utc),
                "severity": "high",
            }  # Caused by first deployment
        ]

        failure_rate = cfr.calculate(deployments, incidents)

        # Should include the incident as a failure
        assert failure_rate["failed_deployments"] >= 1

    def test_calculate_no_deployments(self):
        """Test change failure rate with no deployments."""
        cfr = ChangeFailureRate()

        failure_rate = cfr.calculate([])

        assert failure_rate["failure_rate"] == 0
        assert failure_rate["total_deployments"] == 0
        assert failure_rate["failed_deployments"] == 0

    def test_classify_failure_rate(self):
        """Test change failure rate classification."""
        cfr = ChangeFailureRate()

        # Test elite failure rate (0-15%)
        assert cfr.classify_failure_rate(0.10) == "Elite"

        # Test high failure rate (16-30%)
        assert cfr.classify_failure_rate(0.25) == "High"

        # Test medium failure rate (31-45%)
        assert cfr.classify_failure_rate(0.40) == "Medium"

        # Test low failure rate (>45%)
        assert cfr.classify_failure_rate(0.60) == "Low"


class TestRecoveryTime:
    """Test cases for recovery time metric."""

    def test_init(self):
        """Test RecoveryTime initialization."""
        rt = RecoveryTime()
        assert rt is not None

    def test_calculate_from_incidents(self):
        """Test calculating recovery time from incident data."""
        rt = RecoveryTime()

        incidents = [
            {
                "date": datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                "resolved_date": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
                "severity": "high",
            },
            {
                "date": datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc),
                "resolved_date": datetime(2024, 1, 2, 11, 30, tzinfo=timezone.utc),
                "severity": "medium",
            },
            {
                "date": datetime(2024, 1, 3, 14, 0, tzinfo=timezone.utc),
                "resolved_date": datetime(2024, 1, 3, 18, 0, tzinfo=timezone.utc),
                "severity": "high",
            },
        ]

        recovery_time = rt.calculate(incidents)

        assert "average_hours" in recovery_time
        assert "median_hours" in recovery_time
        assert "incidents_count" in recovery_time

        assert recovery_time["incidents_count"] == 3
        # Recovery times: 2 hours, 2.5 hours, 4 hours
        assert 2.5 <= recovery_time["average_hours"] <= 3.0

    def test_calculate_no_incidents(self):
        """Test recovery time with no incidents."""
        rt = RecoveryTime()

        recovery_time = rt.calculate([])

        assert recovery_time["average_hours"] == 0
        assert recovery_time["median_hours"] == 0
        assert recovery_time["incidents_count"] == 0

    def test_calculate_unresolved_incidents(self):
        """Test recovery time with unresolved incidents."""
        rt = RecoveryTime()

        incidents = [
            {
                "date": datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                "resolved_date": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
                "severity": "high",
            },
            {
                "date": datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc),
                "resolved_date": None,  # Unresolved
                "severity": "medium",
            },
        ]

        recovery_time = rt.calculate(incidents)

        # Should only count resolved incidents
        assert recovery_time["incidents_count"] == 1
        assert recovery_time["average_hours"] == 2.0

    def test_classify_recovery_time(self):
        """Test recovery time classification."""
        rt = RecoveryTime()

        # Test elite recovery time (less than 1 hour)
        assert rt.classify_recovery_time(0.5) == "Elite"

        # Test high recovery time (1 hour to 1 day)
        assert rt.classify_recovery_time(12) == "High"

        # Test medium recovery time (1 day to 1 week)
        assert rt.classify_recovery_time(72) == "Medium"  # 3 days

        # Test low recovery time (more than 1 week)
        assert rt.classify_recovery_time(200) == "Low"  # Over 8 days


class TestDORAIntegration:
    """Integration tests for DORA metrics."""

    def test_complete_dora_analysis(self):
        """Test complete DORA metrics analysis flow."""
        dora = DORAMetrics()

        # Create realistic sample data
        commits = []
        deployments = []
        incidents = []

        # Generate commits over 4 weeks
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for week in range(4):
            for day in range(5):  # Weekdays only
                commit_date = base_date + timedelta(days=week * 7 + day, hours=day * 2)
                commits.append(Mock(hash=f"commit_{week}_{day}", date=commit_date))

                # Deploy every few commits
                if day % 2 == 0:
                    deploy_date = commit_date + timedelta(hours=4)
                    deployments.append(
                        {
                            "date": deploy_date,
                            "commit_hash": f"commit_{week}_{day}",
                            "success": day != 4,  # Fail last commit of each week
                        }
                    )

        # Add some incidents
        incidents = [
            {
                "date": base_date + timedelta(days=3, hours=2),
                "resolved_date": base_date + timedelta(days=3, hours=6),
                "severity": "high",
            },
            {
                "date": base_date + timedelta(days=17, hours=3),
                "resolved_date": base_date + timedelta(days=17, hours=5),
                "severity": "medium",
            },
        ]

        # Calculate all metrics
        metrics = dora.calculate_all_metrics(commits, deployments, incidents)

        # Verify all metrics are present and reasonable
        assert metrics["deployment_frequency"]["total_deployments"] > 0
        assert metrics["lead_time"]["samples"] > 0
        assert 0 <= metrics["change_failure_rate"]["failure_rate"] <= 1
        assert metrics["recovery_time"]["incidents_count"] == 2

        # Verify classifications exist
        assert "classification" in metrics["deployment_frequency"]
        assert "classification" in metrics["lead_time"]
        assert "classification" in metrics["change_failure_rate"]
        assert "classification" in metrics["recovery_time"]
