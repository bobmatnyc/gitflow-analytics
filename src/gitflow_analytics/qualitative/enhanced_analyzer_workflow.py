"""Workflow analysis mixin for EnhancedQualitativeAnalyzer."""

"""Enhanced qualitative analyzer for GitFlow Analytics.

This module provides sophisticated qualitative analysis across four key dimensions:
1. Executive Summary Analysis - High-level team health and strategic insights
2. Project Analysis - Project-specific momentum and health assessment
3. Developer Analysis - Individual contribution patterns and career development
4. Workflow Analysis - Process effectiveness and Git-PM correlation analysis

WHY: Traditional quantitative metrics only tell part of the story. This enhanced analyzer
combines statistical analysis with pattern recognition to generate actionable insights
for different stakeholder levels - from executives to individual developers.

DESIGN DECISIONS:
- Confidence-based scoring: All insights include confidence scores for reliability
- Multi-dimensional analysis: Each section focuses on different aspects of team performance
- Natural language generation: Produces human-readable insights and recommendations
- Anomaly detection: Identifies unusual patterns that merit attention
- Risk assessment: Flags potential issues before they become critical

INTEGRATION: Works with existing qualitative pipeline and extends JSON export format
with structured analysis results that can be consumed by dashboards and reports.
"""

import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np

from .models.schemas import QualitativeCommitData
from .utils.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)




class WorkflowAnalysisMixin:
    """Mixin: git/PM correlation, bottlenecks, automation, compliance, team collab, cross insights, helpers."""

    def _assess_git_pm_correlation(
        self, commits: list[dict[str, Any]], pm_data: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess effectiveness of Git-PM platform correlation."""

        if not pm_data or not pm_data.get("correlations"):
            return {
                "effectiveness": "no_integration",
                "description": "No PM platform integration detected",
                "score": 0,
                "confidence": 0.9,
            }

        correlations = pm_data.get("correlations", [])
        total_correlations = len(correlations)

        # Analyze correlation quality
        high_confidence = sum(1 for c in correlations if c.get("confidence", 0) > 0.8)
        medium_confidence = sum(1 for c in correlations if 0.5 <= c.get("confidence", 0) <= 0.8)

        # Calculate effectiveness score
        if total_correlations > 0:
            quality_score = (
                (high_confidence * 1.0 + medium_confidence * 0.6) / total_correlations * 100
            )
        else:
            quality_score = 0

        # Determine effectiveness level
        if quality_score >= 80:
            effectiveness = "highly_effective"
        elif quality_score >= 60:
            effectiveness = "moderately_effective"
        elif quality_score >= 40:
            effectiveness = "partially_effective"
        else:
            effectiveness = "ineffective"

        return {
            "effectiveness": effectiveness,
            "description": f"{effectiveness.replace('_', ' ').title()} with {quality_score:.1f}% correlation quality",
            "score": round(quality_score, 1),
            "confidence": 0.8,
            "correlation_breakdown": {
                "total": total_correlations,
                "high_confidence": high_confidence,
                "medium_confidence": medium_confidence,
                "low_confidence": total_correlations - high_confidence - medium_confidence,
            },
        }

    def _identify_process_bottlenecks(
        self, commits: list[dict[str, Any]], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Identify potential process bottlenecks."""

        bottlenecks = []

        # Large commit bottleneck
        large_commits = sum(
            1
            for c in commits
            if (
                c.get("filtered_insertions", c.get("insertions", 0))
                + c.get("filtered_deletions", c.get("deletions", 0))
            )
            > self.thresholds["large_commit_lines"]
        )

        if large_commits > len(commits) * 0.25:  # More than 25% large commits
            bottlenecks.append(
                {
                    "type": "large_commit_pattern",
                    "severity": "medium",
                    "description": f"{large_commits} commits exceed {self.thresholds['large_commit_lines']} lines",
                    "impact": "Slower code reviews, increased merge conflicts",
                    "recommendation": "Implement commit size guidelines and encourage smaller, focused changes",
                }
            )

        # Irregular commit timing bottleneck
        daily_commits = defaultdict(int)
        for commit in commits:
            day_key = commit["timestamp"].strftime("%Y-%m-%d")
            daily_commits[day_key] += 1

        daily_values = list(daily_commits.values())
        if daily_values and len(daily_values) > 7:
            daily_std = statistics.pstdev(daily_values)
            daily_mean = statistics.mean(daily_values)

            if daily_std > daily_mean:  # High variability
                bottlenecks.append(
                    {
                        "type": "irregular_development_rhythm",
                        "severity": "low",
                        "description": "Highly variable daily commit patterns",
                        "impact": "Unpredictable integration and review workload",
                        "recommendation": "Encourage more consistent development and integration practices",
                    }
                )

        # Ticket linking bottleneck
        commits_with_tickets = sum(1 for c in commits if c.get("ticket_references"))
        ticket_coverage = (commits_with_tickets / len(commits)) * 100 if commits else 0

        if ticket_coverage < self.thresholds["ticket_coverage_poor"]:
            bottlenecks.append(
                {
                    "type": "poor_ticket_linking",
                    "severity": "medium",
                    "description": f"Only {ticket_coverage:.1f}% of commits reference tickets",
                    "impact": "Poor traceability and project management visibility",
                    "recommendation": "Implement mandatory ticket referencing and provide training",
                }
            )

        return bottlenecks

    def _identify_automation_opportunities(
        self, commits: list[dict[str, Any]], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Identify opportunities for process automation."""

        opportunities = []

        # Repetitive commit message patterns
        message_patterns = defaultdict(int)
        for commit in commits:
            # Simplified pattern detection - look for common prefixes
            message = commit.get("message", "").lower()
            words = message.split()
            if words:
                first_word = words[0]
                if first_word in ["fix", "update", "add", "remove", "refactor"]:
                    message_patterns[first_word] += 1

        total_commits = len(commits)
        for pattern, count in message_patterns.items():
            percentage = (count / total_commits) * 100
            if percentage > 30:  # More than 30% of commits follow this pattern
                opportunities.append(
                    {
                        "type": "commit_message_templates",
                        "description": f"{percentage:.1f}% of commits start with '{pattern}'",
                        "potential": "Implement commit message templates and validation",
                        "effort": "low",
                        "impact": "medium",
                    }
                )

        # Regular fix patterns suggesting test automation needs
        fix_commits = sum(
            1
            for c in commits
            if any(keyword in c.get("message", "").lower() for keyword in ["fix", "bug", "hotfix"])
        )
        fix_percentage = (fix_commits / total_commits) * 100 if total_commits else 0

        if fix_percentage > 25:
            opportunities.append(
                {
                    "type": "automated_testing",
                    "description": f"{fix_percentage:.1f}% of commits are fixes",
                    "potential": "Implement comprehensive automated testing to catch issues earlier",
                    "effort": "high",
                    "impact": "high",
                }
            )

        # Deployment frequency analysis
        deploy_keywords = ["deploy", "release", "version"]
        deploy_commits = sum(
            1
            for c in commits
            if any(keyword in c.get("message", "").lower() for keyword in deploy_keywords)
        )

        weeks_analyzed = context["weeks_analyzed"]
        deploy_frequency = deploy_commits / max(weeks_analyzed, 1)

        if deploy_frequency < 0.5:  # Less than 0.5 deploys per week
            opportunities.append(
                {
                    "type": "continuous_deployment",
                    "description": f"Low deployment frequency: {deploy_frequency:.1f} per week",
                    "potential": "Implement continuous deployment pipeline",
                    "effort": "high",
                    "impact": "high",
                }
            )

        return opportunities

    def _calculate_compliance_metrics(
        self,
        commits: list[dict[str, Any]],
        project_metrics: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate various compliance and process adherence metrics."""

        total_commits = len(commits)

        # Ticket coverage compliance
        commits_with_tickets = sum(1 for c in commits if c.get("ticket_references"))
        ticket_coverage = (commits_with_tickets / total_commits) * 100 if total_commits else 0

        # Commit message quality compliance
        descriptive_messages = sum(
            1 for c in commits if len(c.get("message", "").split()) >= 3
        )  # At least 3 words
        message_quality = (descriptive_messages / total_commits) * 100 if total_commits else 0

        # Size compliance (reasonable commit sizes)
        appropriate_size_commits = sum(
            1
            for c in commits
            if 10
            <= (
                c.get("filtered_insertions", c.get("insertions", 0))
                + c.get("filtered_deletions", c.get("deletions", 0))
            )
            <= 300
        )
        size_compliance = (appropriate_size_commits / total_commits) * 100 if total_commits else 0

        # PR approval compliance — derive from real PR metrics when available.
        # ``project_metrics`` may carry a ``pr_metrics`` sub-dict populated by
        # ``GitHubIntegration.calculate_pr_metrics()``.  Fall back to a neutral
        # default only when no PR data has been collected, and record whether the
        # value is real or estimated so callers can surface that to users.
        pr_metrics: dict[str, Any] = (
            project_metrics.get("pr_metrics", {}) if project_metrics else {}
        )
        real_approval_rate: float | None = pr_metrics.get("approval_rate")

        if real_approval_rate is not None and pr_metrics.get("total_prs", 0) > 0:
            pr_approval_rate = real_approval_rate
            pr_data_source = "measured"
        else:
            # No PR review data collected — exclude this factor from the mean
            # rather than biasing it with an arbitrary constant.
            pr_approval_rate = None
            pr_data_source = "unavailable"

        # Overall compliance score — only average over factors that have real data
        compliance_factors: list[float] = [ticket_coverage, message_quality, size_compliance]
        if pr_approval_rate is not None:
            compliance_factors.append(pr_approval_rate)
        overall_compliance = statistics.mean(compliance_factors)

        def _status(score: float) -> str:
            if score >= 80:
                return "excellent"
            if score >= 60:
                return "good"
            return "needs_improvement"

        pr_approval_entry: dict[str, Any]
        if pr_approval_rate is not None:
            pr_approval_entry = {
                "score": round(pr_approval_rate, 1),
                "status": _status(pr_approval_rate),
                "source": pr_data_source,
            }
        else:
            pr_approval_entry = {
                "score": None,
                "status": "no_data",
                "source": pr_data_source,
            }

        return {
            "overall_score": round(overall_compliance, 1),
            "ticket_coverage": {
                "score": round(ticket_coverage, 1),
                "status": _status(ticket_coverage),
            },
            "message_quality": {
                "score": round(message_quality, 1),
                "status": _status(message_quality),
            },
            "commit_size_compliance": {
                "score": round(size_compliance, 1),
                "status": _status(size_compliance),
            },
            "pr_approval_rate": pr_approval_entry,
        }

    def _analyze_team_collaboration_patterns(
        self, commits: list[dict[str, Any]], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze team collaboration patterns."""

        # Cross-project collaboration analysis
        developer_projects = defaultdict(set)
        for commit in commits:
            dev_id = commit.get("canonical_id", commit.get("author_email"))
            project_key = commit.get("project_key", "UNKNOWN")
            developer_projects[dev_id].add(project_key)

        # Count cross-project contributors
        cross_project_devs = sum(1 for projects in developer_projects.values() if len(projects) > 1)
        total_devs = len(developer_projects)
        cross_collaboration_rate = (cross_project_devs / total_devs) * 100 if total_devs else 0

        # Project contributor diversity
        project_contributors = defaultdict(set)
        for commit in commits:
            dev_id = commit.get("canonical_id", commit.get("author_email"))
            project_key = commit.get("project_key", "UNKNOWN")
            project_contributors[project_key].add(dev_id)

        avg_contributors_per_project = (
            statistics.mean([len(contributors) for contributors in project_contributors.values()])
            if project_contributors
            else 0
        )

        # Collaboration score calculation
        collaboration_factors = [
            min(100, cross_collaboration_rate * 2),  # Cross-project work
            min(100, avg_contributors_per_project * 25),  # Multi-contributor projects
        ]

        collaboration_score = statistics.mean(collaboration_factors)

        return {
            "collaboration_score": round(collaboration_score, 1),
            "cross_project_contributors": cross_project_devs,
            "cross_collaboration_rate": round(cross_collaboration_rate, 1),
            "avg_contributors_per_project": round(avg_contributors_per_project, 1),
            "collaboration_level": (
                "high"
                if collaboration_score >= 70
                else "medium"
                if collaboration_score >= 40
                else "low"
            ),
            "patterns": {
                "multi_project_engagement": cross_collaboration_rate >= 50,
                "team_project_distribution": avg_contributors_per_project >= 2,
                "siloed_development": cross_collaboration_rate < 20,
            },
        }

    def _generate_process_recommendations(
        self,
        git_pm_effectiveness: dict[str, Any],
        bottlenecks: list[dict[str, Any]],
        automation_opportunities: list[dict[str, Any]],
        compliance_metrics: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate process improvement recommendations."""

        recommendations = []

        # Git-PM integration recommendations
        effectiveness = git_pm_effectiveness.get("effectiveness", "")
        if effectiveness in ["ineffective", "partially_effective"]:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "integration",
                    "title": "Improve Git-PM Integration",
                    "action": "Enhance ticket referencing and correlation accuracy",
                    "timeline": "4-6 weeks",
                    "expected_impact": "Better project visibility and tracking",
                }
            )

        # Bottleneck recommendations
        high_severity_bottlenecks = [b for b in bottlenecks if b.get("severity") == "high"]
        if high_severity_bottlenecks:
            bottleneck = high_severity_bottlenecks[0]
            recommendations.append(
                {
                    "priority": "high",
                    "category": "process_optimization",
                    "title": f"Address {bottleneck['type'].replace('_', ' ').title()}",
                    "action": bottleneck.get("recommendation", "Address identified bottleneck"),
                    "timeline": "2-4 weeks",
                    "expected_impact": bottleneck.get("impact", "Improved process efficiency"),
                }
            )

        # Automation recommendations
        high_impact_automation = [a for a in automation_opportunities if a.get("impact") == "high"]
        if high_impact_automation:
            automation = high_impact_automation[0]
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "automation",
                    "title": f"Implement {automation['type'].replace('_', ' ').title()}",
                    "action": automation["potential"],
                    "timeline": "6-12 weeks" if automation.get("effort") == "high" else "2-4 weeks",
                    "expected_impact": "Reduced manual effort and improved consistency",
                }
            )

        # Compliance recommendations
        overall_compliance = compliance_metrics.get("overall_score", 0)
        if overall_compliance < 70:
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "compliance",
                    "title": "Improve Process Compliance",
                    "action": "Focus on ticket linking, commit message quality, and size guidelines",
                    "timeline": "4-8 weeks",
                    "expected_impact": "Better process adherence and project visibility",
                }
            )

        return recommendations[:4]  # Top 4 recommendations

    def _generate_workflow_narrative(
        self,
        git_pm_effectiveness: dict[str, Any],
        bottlenecks: list[dict[str, Any]],
        compliance_metrics: dict[str, Any],
    ) -> str:
        """Generate workflow analysis narrative."""

        narrative_parts = []

        # Git-PM effectiveness
        git_pm_effectiveness.get("effectiveness", "unknown")
        effectiveness_desc = git_pm_effectiveness.get("description", "integration status unclear")
        narrative_parts.append(f"Git-PM platform integration is {effectiveness_desc.lower()}.")

        # Process health
        compliance_score = compliance_metrics.get("overall_score", 0)
        if compliance_score >= 80:
            narrative_parts.append("Development processes show strong compliance and adherence.")
        elif compliance_score >= 60:
            narrative_parts.append(
                "Development processes are generally well-followed with room for improvement."
            )
        else:
            narrative_parts.append(
                "Development processes need attention to improve compliance and effectiveness."
            )

        # Bottleneck highlights
        high_severity_bottlenecks = [b for b in bottlenecks if b.get("severity") == "high"]
        if high_severity_bottlenecks:
            narrative_parts.append(
                f"Critical bottleneck identified: {high_severity_bottlenecks[0]['type'].replace('_', ' ')}."
            )
        elif bottlenecks:
            narrative_parts.append(
                f"Some process inefficiencies detected, particularly in {bottlenecks[0]['type'].replace('_', ' ')}."
            )

        return " ".join(narrative_parts)

    # Cross-Analysis Helper Methods

    def _generate_cross_insights(
        self,
        executive_analysis: dict[str, Any],
        project_analysis: dict[str, Any],
        developer_analysis: dict[str, Any],
        workflow_analysis: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate insights that span multiple analysis dimensions."""

        cross_insights = []

        # Executive-Project alignment insight
        exec_health = executive_analysis.get("health_assessment", "unknown")
        project_health_scores = []
        for project_data in project_analysis.values():
            health_indicators = project_data.get("health_indicators", {})
            overall_health = health_indicators.get("overall_health", {})
            score = overall_health.get("score", 0)
            project_health_scores.append(score)

        if project_health_scores:
            avg_project_health = statistics.mean(project_health_scores)
            if exec_health == "excellent" and avg_project_health < 60:
                cross_insights.append(
                    {
                        "type": "alignment_mismatch",
                        "title": "Executive-Project Health Disconnect",
                        "description": "Overall team health excellent but individual projects show concerns",
                        "priority": "medium",
                        "recommendation": "Investigate project-specific issues that may not be visible at team level",
                    }
                )

        # Developer-Workflow correlation insight
        high_burnout_devs = 0
        for dev_data in developer_analysis.values():
            burnout_indicators = dev_data.get("burnout_indicators", [])
            if len(burnout_indicators) >= 2:
                high_burnout_devs += 1

        workflow_bottlenecks = workflow_analysis.get("process_bottlenecks", [])
        if high_burnout_devs > 0 and len(workflow_bottlenecks) > 2:
            cross_insights.append(
                {
                    "type": "systemic_issue",
                    "title": "Process Issues Contributing to Developer Stress",
                    "description": f"{high_burnout_devs} developers show burnout indicators alongside {len(workflow_bottlenecks)} process bottlenecks",
                    "priority": "high",
                    "recommendation": "Address workflow inefficiencies to reduce developer burden",
                }
            )

        # Project-Developer resource allocation insight
        declining_projects = sum(
            1
            for p in project_analysis.values()
            if p.get("momentum", {}).get("classification") == "declining"
        )
        declining_developers = sum(
            1
            for d in developer_analysis.values()
            if d.get("growth_trajectory", {}).get("trajectory") == "declining"
        )

        if declining_projects > 0 and declining_developers > 0:
            cross_insights.append(
                {
                    "type": "resource_allocation",
                    "title": "Coordinated Decline Pattern",
                    "description": f"{declining_projects} projects and {declining_developers} developers showing decline",
                    "priority": "high",
                    "recommendation": "Review resource allocation and team capacity planning",
                }
            )

        return cross_insights

    # Utility Helper Methods

    def _get_weekly_commit_counts(self, commits: list[dict[str, Any]]) -> list[int]:
        """Get commit counts grouped by week."""

        if not commits:
            return []

        weekly_counts = defaultdict(int)
        for commit in commits:
            week_start = self._get_week_start(commit["timestamp"])
            week_key = week_start.strftime("%Y-%m-%d")
            weekly_counts[week_key] += 1

        # Return counts in chronological order
        sorted_weeks = sorted(weekly_counts.keys())
        return [weekly_counts[week] for week in sorted_weeks]

    def _get_week_start(self, date: datetime) -> datetime:
        """Get Monday of the week for a given date."""

        # Ensure timezone consistency
        if hasattr(date, "tzinfo") and date.tzinfo is not None:
            if date.tzinfo != timezone.utc:
                date = date.astimezone(timezone.utc)
        else:
            date = date.replace(tzinfo=timezone.utc)

        days_since_monday = date.weekday()
        monday = date - timedelta(days=days_since_monday)
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)

    def _calculate_gini_coefficient(self, values: list[float]) -> float:
        """Calculate Gini coefficient for measuring inequality."""

        if not values or len(values) == 1:
            return 0.0

        sorted_values = sorted([v for v in values if v > 0])  # Filter out zeros
        if not sorted_values:
            return 0.0

        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)

        # Use builtin sum instead of np.sum for generator expression (numpy deprecation)
        return (2 * sum((i + 1) * sorted_values[i] for i in range(n))) / (n * cumsum[-1]) - (
            n + 1
        ) / n
