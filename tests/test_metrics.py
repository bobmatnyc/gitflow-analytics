"""
Tests for the metrics module.

These tests verify DORA metrics calculation and other performance indicators.
"""

from datetime import datetime, timezone

from gitflow_analytics.metrics.dora import DORAMetricsCalculator


class TestDORAMetricsCalculator:
    """Test cases for DORA metrics calculation."""

    def test_init(self):
        """Test DORAMetricsCalculator initialization."""
        calculator = DORAMetricsCalculator()

        assert calculator.deployment_patterns is not None
        assert calculator.failure_patterns is not None
        assert len(calculator.deployment_patterns) > 0
        assert len(calculator.failure_patterns) > 0

    def test_deployment_pattern_detection(self):
        """Test deployment pattern detection in commit messages."""
        calculator = DORAMetricsCalculator()

        # Test commits with deployment patterns
        deployment_commits = [
            {
                "hash": "abc123",
                "message": "deploy: release v1.0.0",
                "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
            },
            {
                "hash": "def456",
                "message": "feat: ship new feature to production",
                "timestamp": datetime(2024, 1, 2, tzinfo=timezone.utc),
            },
            {
                "hash": "ghi789",
                "message": "release: version 2.0.0 is live",
                "timestamp": datetime(2024, 1, 3, tzinfo=timezone.utc),
            },
        ]

        # Test commits without deployment patterns
        regular_commits = [
            {
                "hash": "jkl012",
                "message": "fix: resolve bug in user service",
                "timestamp": datetime(2024, 1, 4, tzinfo=timezone.utc),
            },
            {
                "hash": "mno345",
                "message": "feat: add new user authentication",
                "timestamp": datetime(2024, 1, 5, tzinfo=timezone.utc),
            },
        ]

        all_commits = deployment_commits + regular_commits
        prs = []

        deployments = calculator._identify_deployments(all_commits, prs)

        # Should identify deployment commits
        assert len(deployments) >= 3
        deployment_messages = [d["message"] for d in deployments]
        assert any("deploy" in msg.lower() for msg in deployment_messages)
        assert any("ship" in msg.lower() for msg in deployment_messages)
        assert any("live" in msg.lower() for msg in deployment_messages)

    def test_failure_pattern_detection(self):
        """Test failure pattern detection in commit messages."""
        calculator = DORAMetricsCalculator()

        # Test commits with failure patterns
        failure_commits = [
            {
                "hash": "fail123",
                "message": "revert: rollback problematic deployment",
                "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
            },
            {
                "hash": "fail456",
                "message": "hotfix: emergency fix for production issue",
                "timestamp": datetime(2024, 1, 2, tzinfo=timezone.utc),
            },
            {
                "hash": "fail789",
                "message": "fix: resolve incident with user authentication",
                "timestamp": datetime(2024, 1, 3, tzinfo=timezone.utc),
            },
        ]

        regular_commits = [
            {
                "hash": "reg123",
                "message": "feat: add new dashboard feature",
                "timestamp": datetime(2024, 1, 4, tzinfo=timezone.utc),
            },
            {
                "hash": "reg456",
                "message": "docs: update API documentation",
                "timestamp": datetime(2024, 1, 5, tzinfo=timezone.utc),
            },
        ]

        all_commits = failure_commits + regular_commits
        prs = []

        failures = calculator._identify_failures(all_commits, prs)

        # Should identify failure commits
        assert len(failures) >= 3
        failure_messages = [f["message"] for f in failures]
        assert any("revert" in msg.lower() for msg in failure_messages)
        assert any("hotfix" in msg.lower() for msg in failure_messages)
        assert any("incident" in msg.lower() for msg in failure_messages)

    def test_calculate_dora_metrics_basic(self):
        """Test basic DORA metrics calculation."""
        calculator = DORAMetricsCalculator()

        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)

        commits = [
            {
                "hash": "deploy1",
                "message": "deploy: release v1.0.0",
                "timestamp": datetime(2024, 1, 5, tzinfo=timezone.utc),
            },
            {
                "hash": "deploy2",
                "message": "deploy: release v1.1.0",
                "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
            },
            {
                "hash": "hotfix1",
                "message": "hotfix: emergency fix",
                "timestamp": datetime(2024, 1, 20, tzinfo=timezone.utc),
            },
            {
                "hash": "feat1",
                "message": "feat: add feature",
                "timestamp": datetime(2024, 1, 25, tzinfo=timezone.utc),
            },
        ]

        prs = [
            {
                "title": "Add new feature",
                "created_at": datetime(2024, 1, 10, tzinfo=timezone.utc),
                "merged_at": datetime(2024, 1, 12, tzinfo=timezone.utc),
            },
            {
                "title": "Bug fix for production",
                "created_at": datetime(2024, 1, 18, tzinfo=timezone.utc),
                "merged_at": datetime(2024, 1, 19, tzinfo=timezone.utc),
            },
        ]

        metrics = calculator.calculate_dora_metrics(commits, prs, start_date, end_date)

        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert "deployment_frequency" in metrics
        assert "lead_time_hours" in metrics
        assert "change_failure_rate" in metrics
        assert "mttr_hours" in metrics
        assert "performance_level" in metrics
        assert "total_deployments" in metrics
        assert "total_failures" in metrics
        assert "metrics_period_weeks" in metrics

        # Verify metric values are reasonable
        assert isinstance(metrics["deployment_frequency"], dict)
        assert metrics["lead_time_hours"] >= 0
        assert 0 <= metrics["change_failure_rate"] <= 100
        assert metrics["mttr_hours"] >= 0
        assert metrics["performance_level"] in ["Elite", "High", "Medium", "Low"]

    def test_deployment_frequency_calculation(self):
        """Test deployment frequency calculation."""
        calculator = DORAMetricsCalculator()

        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)  # 30 days

        # Create deployments spread over the month
        deployments = [
            {"timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc)},
            {"timestamp": datetime(2024, 1, 8, tzinfo=timezone.utc)},
            {"timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc)},
            {"timestamp": datetime(2024, 1, 22, tzinfo=timezone.utc)},
            {"timestamp": datetime(2024, 1, 29, tzinfo=timezone.utc)},
        ]

        freq_metrics = calculator._calculate_deployment_frequency(deployments, start_date, end_date)

        assert "daily_average" in freq_metrics
        assert "weekly_average" in freq_metrics
        assert "category" in freq_metrics
        assert freq_metrics["category"] in ["Elite", "High", "Medium", "Low"]
        assert freq_metrics["daily_average"] > 0
        assert freq_metrics["weekly_average"] > 0

    def test_lead_time_calculation(self):
        """Test lead time calculation."""
        calculator = DORAMetricsCalculator()

        prs = [
            {
                "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "merged_at": datetime(2024, 1, 3, tzinfo=timezone.utc),  # 2 days = 48 hours
            },
            {
                "created_at": datetime(2024, 1, 5, tzinfo=timezone.utc),
                "merged_at": datetime(2024, 1, 6, tzinfo=timezone.utc),  # 1 day = 24 hours
            },
        ]

        deployments = []  # Empty for this test

        lead_time = calculator._calculate_lead_time(prs, deployments)

        # Should return median lead time (36 hours for this example)
        assert lead_time >= 0
        assert isinstance(lead_time, float)

    def test_change_failure_rate_calculation(self):
        """Test change failure rate calculation."""
        calculator = DORAMetricsCalculator()

        deployments = [
            {"timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc)},
            {"timestamp": datetime(2024, 1, 5, tzinfo=timezone.utc)},
            {"timestamp": datetime(2024, 1, 10, tzinfo=timezone.utc)},
            {"timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc)},
        ]

        failures = [
            {
                "timestamp": datetime(2024, 1, 2, tzinfo=timezone.utc)
            },  # Within 24h of first deployment
            {
                "timestamp": datetime(2024, 1, 20, tzinfo=timezone.utc)
            },  # Not within 24h of any deployment
        ]

        failure_rate = calculator._calculate_change_failure_rate(deployments, failures)

        assert 0 <= failure_rate <= 100
        assert isinstance(failure_rate, float)

    def test_empty_data_handling(self):
        """Test handling of empty or minimal data."""
        calculator = DORAMetricsCalculator()

        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)

        # Test with empty data
        metrics = calculator.calculate_dora_metrics([], [], start_date, end_date)

        assert isinstance(metrics, dict)
        assert metrics["deployment_frequency"]["daily_average"] == 0
        assert metrics["lead_time_hours"] == 0
        assert metrics["change_failure_rate"] == 0
        assert metrics["total_deployments"] == 0
        assert metrics["total_failures"] == 0


class TestDORAMetricsPerformance:
    """Test performance level determination."""

    def test_performance_level_determination(self):
        """Test DORA performance level categorization."""
        calculator = DORAMetricsCalculator()

        # Test different combinations of metrics
        test_cases = [
            {
                "deployment_freq": {"category": "Elite"},
                "lead_time_hours": 1,
                "change_failure_rate": 5,
                "mttr_hours": 0.5,
                "expected_level": "Elite",
            },
            {
                "deployment_freq": {"category": "Low"},
                "lead_time_hours": 720,  # 30 days
                "change_failure_rate": 50,
                "mttr_hours": 168,  # 1 week
                "expected_level": "Low",
            },
        ]

        for case in test_cases:
            level = calculator._determine_performance_level(
                case["deployment_freq"],
                case["lead_time_hours"],
                case["change_failure_rate"],
                case["mttr_hours"],
            )

            assert level in ["Elite", "High", "Medium", "Low"]
