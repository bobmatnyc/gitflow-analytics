#!/usr/bin/env python3
"""Example usage of the Enhanced Qualitative Analyzer.

This example demonstrates how to use the EnhancedQualitativeAnalyzer to generate
comprehensive qualitative insights across four key dimensions:
1. Executive Summary Analysis
2. Project Analysis
3. Developer Analysis
4. Workflow Analysis

The enhanced analyzer produces natural language insights, risk assessments,
and actionable recommendations for different stakeholder levels.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .enhanced_analyzer import EnhancedQualitativeAnalyzer


def create_sample_commits() -> list[dict[str, Any]]:
    """Create sample commit data for demonstration."""

    base_time = datetime.now(timezone.utc) - timedelta(weeks=8)

    sample_commits = [
        {
            "hash": "abc123",
            "message": "feat: implement user authentication system",
            "author_name": "Alice Developer",
            "author_email": "alice@company.com",
            "canonical_id": "alice@company.com",
            "timestamp": base_time + timedelta(days=1),
            "project_key": "AUTH_SERVICE",
            "insertions": 250,
            "deletions": 30,
            "filtered_insertions": 250,
            "filtered_deletions": 30,
            "files_changed": 8,
            "ticket_references": ["AUTH-123"],
            "story_points": 5,
        },
        {
            "hash": "def456",
            "message": "fix: resolve login timeout issue",
            "author_name": "Bob Engineer",
            "author_email": "bob@company.com",
            "canonical_id": "bob@company.com",
            "timestamp": base_time + timedelta(days=3),
            "project_key": "AUTH_SERVICE",
            "insertions": 45,
            "deletions": 12,
            "filtered_insertions": 45,
            "filtered_deletions": 12,
            "files_changed": 2,
            "ticket_references": ["AUTH-124"],
            "story_points": 2,
        },
        {
            "hash": "ghi789",
            "message": "refactor: improve database connection pooling",
            "author_name": "Alice Developer",
            "author_email": "alice@company.com",
            "canonical_id": "alice@company.com",
            "timestamp": base_time + timedelta(days=5),
            "project_key": "DATA_SERVICE",
            "insertions": 180,
            "deletions": 95,
            "filtered_insertions": 180,
            "filtered_deletions": 95,
            "files_changed": 5,
            "ticket_references": ["DATA-67"],
            "story_points": 8,
        },
        {
            "hash": "jkl012",
            "message": "docs: update API documentation",
            "author_name": "Carol Writer",
            "author_email": "carol@company.com",
            "canonical_id": "carol@company.com",
            "timestamp": base_time + timedelta(days=7),
            "project_key": "API_DOCS",
            "insertions": 120,
            "deletions": 20,
            "filtered_insertions": 120,
            "filtered_deletions": 20,
            "files_changed": 3,
            "ticket_references": ["DOC-45"],
            "story_points": 3,
        },
        # Add more commits for realistic analysis
        {
            "hash": "mno345",
            "message": "feat: add user preferences endpoint",
            "author_name": "Bob Engineer",
            "author_email": "bob@company.com",
            "canonical_id": "bob@company.com",
            "timestamp": base_time + timedelta(days=10),
            "project_key": "USER_SERVICE",
            "insertions": 320,
            "deletions": 15,
            "filtered_insertions": 320,
            "filtered_deletions": 15,
            "files_changed": 12,
            "ticket_references": ["USER-89"],
            "story_points": 13,
        },
    ]

    return sample_commits


def create_sample_developer_stats() -> list[dict[str, Any]]:
    """Create sample developer statistics."""

    base_time = datetime.now(timezone.utc)

    return [
        {
            "canonical_id": "alice@company.com",
            "primary_name": "Alice Developer",
            "primary_email": "alice@company.com",
            "total_commits": 45,
            "total_story_points": 89,
            "alias_count": 1,
            "first_seen": base_time - timedelta(weeks=12),
            "last_seen": base_time - timedelta(days=2),
            "github_username": "alice-dev",
        },
        {
            "canonical_id": "bob@company.com",
            "primary_name": "Bob Engineer",
            "primary_email": "bob@company.com",
            "total_commits": 38,
            "total_story_points": 72,
            "alias_count": 1,
            "first_seen": base_time - timedelta(weeks=10),
            "last_seen": base_time - timedelta(days=1),
            "github_username": "bob-eng",
        },
        {
            "canonical_id": "carol@company.com",
            "primary_name": "Carol Writer",
            "primary_email": "carol@company.com",
            "total_commits": 23,
            "total_story_points": 34,
            "alias_count": 1,
            "first_seen": base_time - timedelta(weeks=8),
            "last_seen": base_time - timedelta(days=7),
            "github_username": "carol-docs",
        },
    ]


def create_sample_project_metrics() -> dict[str, Any]:
    """Create sample project metrics."""

    return {
        "ticket_analysis": {
            "commit_coverage_pct": 87.5,
            "total_tickets_referenced": 156,
            "unique_tickets": 145,
        },
        "story_point_analysis": {"total_story_points": 195, "average_per_commit": 4.2},
    }


def create_sample_pm_data() -> dict[str, Any]:
    """Create sample PM platform integration data."""

    return {
        "correlations": [
            {
                "commit_hash": "abc123",
                "ticket_id": "AUTH-123",
                "confidence": 0.95,
                "correlation_method": "exact_match",
            },
            {
                "commit_hash": "def456",
                "ticket_id": "AUTH-124",
                "confidence": 0.89,
                "correlation_method": "semantic_similarity",
            },
        ],
        "metrics": {
            "total_pm_issues": 178,
            "platform_coverage": {
                "jira": {"total_issues": 156, "linked_issues": 142, "coverage_percentage": 91.0}
            },
            "story_point_analysis": {"story_point_coverage_pct": 78.5},
        },
    }


def demonstrate_enhanced_analysis():
    """Demonstrate the enhanced qualitative analyzer functionality."""

    print("🔍 Enhanced Qualitative Analysis Demonstration")
    print("=" * 60)

    # Create sample data
    commits = create_sample_commits()
    developer_stats = create_sample_developer_stats()
    project_metrics = create_sample_project_metrics()
    pm_data = create_sample_pm_data()

    print("📊 Sample data created:")
    print(f"  - {len(commits)} commits")
    print(f"  - {len(developer_stats)} developers")
    print(f"  - {len(project_metrics.get('ticket_analysis', {}))} ticket metrics")
    print(f"  - PM integration with {len(pm_data.get('correlations', []))} correlations")
    print()

    # Initialize the enhanced analyzer
    analyzer = EnhancedQualitativeAnalyzer()
    print("🚀 Enhanced Qualitative Analyzer initialized")
    print()

    # Perform comprehensive analysis
    print("🔄 Performing comprehensive analysis...")
    analysis_result = analyzer.analyze_comprehensive(
        commits=commits,
        qualitative_data=None,  # No detailed qualitative data in this example
        developer_stats=developer_stats,
        project_metrics=project_metrics,
        pm_data=pm_data,
        weeks_analyzed=8,
    )

    print("✅ Analysis completed!")
    print()

    # Display results by dimension
    print("📈 EXECUTIVE SUMMARY ANALYSIS")
    print("-" * 40)

    exec_analysis = analysis_result.get("executive_analysis", {})
    print(f"Team Health: {exec_analysis.get('health_assessment', 'Unknown')}")
    print(f"Health Confidence: {exec_analysis.get('health_confidence', 0):.2f}")

    velocity_trends = exec_analysis.get("velocity_trends", {})
    print(f"Velocity Trend: {velocity_trends.get('overall_trend', 'Unknown')}")
    print(f"Weekly Average: {velocity_trends.get('weekly_average', 0)} commits")

    achievements = exec_analysis.get("key_achievements", [])
    print(f"Key Achievements: {len(achievements)} identified")
    for achievement in achievements[:2]:
        print(f"  • {achievement.get('title', 'Unknown')}")

    concerns = exec_analysis.get("major_concerns", [])
    print(f"Major Concerns: {len(concerns)} identified")
    for concern in concerns[:2]:
        print(f"  ⚠️ {concern.get('title', 'Unknown')}")

    print(f"\\nExecutive Summary: {exec_analysis.get('executive_summary', 'No summary available')}")
    print()

    print("🏗️ PROJECT ANALYSIS")
    print("-" * 40)

    project_analysis = analysis_result.get("project_analysis", {})
    for project_key, project_data in list(project_analysis.items())[:3]:  # Show first 3 projects
        momentum = project_data.get("momentum", {})
        health = project_data.get("health_indicators", {}).get("overall_health", {})

        print(f"Project: {project_key}")
        print(
            f"  Momentum: {momentum.get('classification', 'Unknown')} ({momentum.get('confidence', 0):.2f} confidence)"
        )
        print(f"  Health Score: {health.get('score', 0)}/100 ({health.get('status', 'Unknown')})")

        recommendations = project_data.get("recommendations", [])
        if recommendations:
            print(f"  Top Recommendation: {recommendations[0].get('title', 'None')}")
        print()

    print("👨‍💻 DEVELOPER ANALYSIS")
    print("-" * 40)

    developer_analysis = analysis_result.get("developer_analysis", {})
    for dev_id, dev_data in list(developer_analysis.items())[:3]:  # Show first 3 developers
        contribution = dev_data.get("contribution_pattern", {})
        collaboration = dev_data.get("collaboration_score", {})
        growth = dev_data.get("growth_trajectory", {})

        # Use canonical_id for lookup if name not available
        dev_name = None
        for dev_stat in developer_stats:
            if dev_stat.get("canonical_id") == dev_id:
                dev_name = dev_stat.get("primary_name", dev_id)
                break

        print(f"Developer: {dev_name or dev_id}")
        print(
            f"  Pattern: {contribution.get('pattern', 'Unknown')} ({contribution.get('confidence', 0):.2f} confidence)"
        )
        print(
            f"  Collaboration: {collaboration.get('level', 'Unknown')} ({collaboration.get('score', 0):.1f}/100)"
        )
        print(f"  Growth: {growth.get('trajectory', 'Unknown')} trajectory")

        recommendations = dev_data.get("career_recommendations", [])
        if recommendations:
            print(f"  Career Recommendation: {recommendations[0].get('title', 'None')}")
        print()

    print("⚙️ WORKFLOW ANALYSIS")
    print("-" * 40)

    workflow_analysis = analysis_result.get("workflow_analysis", {})
    git_pm = workflow_analysis.get("git_pm_effectiveness", {})
    bottlenecks = workflow_analysis.get("process_bottlenecks", [])
    compliance = workflow_analysis.get("compliance_metrics", {})

    print(f"Git-PM Effectiveness: {git_pm.get('effectiveness', 'Unknown')}")
    print(f"Effectiveness Score: {git_pm.get('score', 0):.1f}%")
    print(f"Process Bottlenecks: {len(bottlenecks)} identified")
    print(f"Overall Compliance: {compliance.get('overall_score', 0):.1f}%")

    automation_opps = workflow_analysis.get("automation_opportunities", [])
    print(f"Automation Opportunities: {len(automation_opps)} identified")

    print(
        f"\\nWorkflow Summary: {workflow_analysis.get('workflow_narrative', 'No summary available')}"
    )
    print()

    print("🔗 CROSS-DIMENSIONAL INSIGHTS")
    print("-" * 40)

    cross_insights = analysis_result.get("cross_insights", [])
    print(f"Cross-insights identified: {len(cross_insights)}")
    for insight in cross_insights:
        print(
            f"  • {insight.get('title', 'Unknown')} ({insight.get('priority', 'unknown')} priority)"
        )
    print()

    # Show how to export to JSON format
    print("💾 JSON EXPORT EXAMPLE")
    print("-" * 40)

    # Create a sample output path
    output_path = Path("enhanced_analysis_example.json")

    # Export the results
    try:
        with open(output_path, "w") as f:
            json.dump(analysis_result, f, indent=2, default=str)

        print(f"✅ Enhanced analysis exported to: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

        # Show a sample of the JSON structure
        print("\\nJSON Structure Preview:")
        print(
            json.dumps(
                {
                    key: f"<{len(value) if isinstance(value, (list, dict)) else type(value).__name__}>"
                    for key, value in analysis_result.items()
                },
                indent=2,
            )
        )

    except Exception as e:
        print(f"❌ Export failed: {e}")

    print()
    print("🎯 INTEGRATION WITH EXISTING REPORTS")
    print("-" * 40)
    print("To integrate with existing GitFlow Analytics JSON exports:")
    print("1. Import EnhancedQualitativeAnalyzer in your analysis pipeline")
    print("2. Call analyzer.analyze_comprehensive() with your commit data")
    print("3. Pass the results to ComprehensiveJSONExporter.export_comprehensive_data()")
    print("4. The enhanced analysis will be included under 'enhanced_qualitative_analysis' key")
    print()
    print("Example integration code:")
    print(
        """
from gitflow_analytics.qualitative import EnhancedQualitativeAnalyzer
from gitflow_analytics.reports.json_exporter import ComprehensiveJSONExporter

# Initialize components
analyzer = EnhancedQualitativeAnalyzer()
exporter = ComprehensiveJSONExporter()

# Perform enhanced analysis
enhanced_analysis = analyzer.analyze_comprehensive(
    commits=commits,
    developer_stats=developer_stats,
    project_metrics=project_metrics,
    pm_data=pm_data,
    weeks_analyzed=weeks
)

# Export with enhanced analysis included
exporter.export_comprehensive_data(
    commits=commits,
    prs=prs,
    developer_stats=developer_stats,
    project_metrics=project_metrics,
    dora_metrics=dora_metrics,
    output_path=output_path,
    weeks=weeks,
    pm_data=pm_data,
    enhanced_qualitative_analysis=enhanced_analysis  # Include enhanced analysis
)
"""
    )
    print()
    print("🏁 Enhanced Qualitative Analysis demonstration completed!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_enhanced_analysis()
