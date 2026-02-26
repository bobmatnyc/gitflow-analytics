"""Narrative report mixin: recommendations, PM insights, and CI/CD health sections.

Extracted from narrative_writer.py to keep file sizes manageable.
"""

import logging
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)



class NarrativeRecommendationsMixin:
    """Mixin: recommendations, PM insights, CI/CD health, and commit classification sections."""

    def _write_recommendations(
        self,
        report: StringIO,
        insights: list[dict[str, Any]],
        ticket_analysis: dict[str, Any],
        focus_data: list[dict[str, Any]],
    ) -> None:
        """Write recommendations based on analysis."""
        recommendations = []

        # Ticket coverage recommendations
        coverage = ticket_analysis["commit_coverage_pct"]
        if coverage < 50:
            recommendations.append(
                "🎫 **Improve ticket tracking**: Current coverage is below 50%. "
                "Consider enforcing ticket references in commit messages or PR descriptions."
            )

        # Work distribution recommendations (handle missing insight field gracefully)
        for insight in insights:
            insight_text = insight.get("insight", insight.get("metric", ""))
            if insight_text == "Work distribution":
                insight_value = str(insight.get("value", ""))
                if "unbalanced" in insight_value.lower():
                    recommendations.append(
                        "⚖️ **Balance workload**: Work is concentrated among few developers. "
                        "Consider distributing tasks more evenly or adding team members."
                    )

        # Focus recommendations (handle missing focus_score field gracefully)
        if focus_data:
            low_focus = []
            for d in focus_data:
                focus_score = d.get("focus_score", d.get("focus_ratio", 0.5) * 100)
                if focus_score < 50:
                    low_focus.append(d)
            if len(low_focus) > len(focus_data) / 2:
                recommendations.append(
                    "🎯 **Reduce context switching**: Many developers work across multiple projects. "
                    "Consider more focused project assignments to improve efficiency."
                )

        # Branching strategy (handle missing insight field gracefully)
        for insight in insights:
            insight_text = insight.get("insight", insight.get("metric", ""))
            insight_value = str(insight.get("value", ""))
            if insight_text == "Branching strategy" and "Heavy" in insight_value:
                recommendations.append(
                    "🌿 **Review branching strategy**: High percentage of merge commits suggests "
                    "complex branching. Consider simplifying the Git workflow."
                )

        if recommendations:
            for rec in recommendations:
                report.write(f"{rec}\n\n")
        else:
            report.write("✅ The team shows healthy development patterns. ")
            report.write("Continue current practices while monitoring for changes.\n")

    def _write_commit_classification_analysis(
        self, report: StringIO, ticket_analysis: dict[str, Any]
    ) -> None:
        """Write commit classification analysis section.

        WHY: This section provides insights into automated commit categorization
        quality and distribution, helping teams understand their development patterns
        and the effectiveness of ML-based categorization.

        Args:
            report: StringIO buffer to write to
            ticket_analysis: Ticket analysis data containing ML classification results
        """
        ml_analysis = ticket_analysis.get("ml_analysis", {})
        if not ml_analysis.get("enabled", False):
            return

        report.write(
            "The team's commit patterns reveal the following automated classification insights:\n\n"
        )

        # Overall classification statistics
        total_ml_predictions = ml_analysis.get("total_ml_predictions", 0)
        total_rule_predictions = ml_analysis.get("total_rule_predictions", 0)
        total_cached_predictions = ml_analysis.get("total_cached_predictions", 0)
        total_predictions = total_ml_predictions + total_rule_predictions + total_cached_predictions

        if total_predictions > 0:
            report.write("### Classification Method Distribution\n\n")

            # Calculate percentages
            ml_pct = (total_ml_predictions / total_predictions) * 100
            rules_pct = (total_rule_predictions / total_predictions) * 100
            cached_pct = (total_cached_predictions / total_predictions) * 100

            report.write(
                f"- **ML-based Classifications**: {total_ml_predictions} commits ({ml_pct:.1f}%)\n"
            )
            report.write(
                f"- **Rule-based Classifications**: {total_rule_predictions} commits ({rules_pct:.1f}%)\n"
            )
            report.write(
                f"- **Cached Results**: {total_cached_predictions} commits ({cached_pct:.1f}%)\n\n"
            )

            # Classification confidence analysis
            avg_confidence = ml_analysis.get("avg_confidence", 0)
            confidence_dist = ml_analysis.get("confidence_distribution", {})

            if confidence_dist:
                report.write("### Classification Confidence\n\n")
                report.write(
                    f"- **Average Confidence**: {avg_confidence:.1%} across all classifications\n"
                )

                high_conf = confidence_dist.get("high", 0)
                medium_conf = confidence_dist.get("medium", 0)
                low_conf = confidence_dist.get("low", 0)
                total_conf_items = high_conf + medium_conf + low_conf

                if total_conf_items > 0:
                    high_pct = (high_conf / total_conf_items) * 100
                    medium_pct = (medium_conf / total_conf_items) * 100
                    low_pct = (low_conf / total_conf_items) * 100

                    report.write(
                        f"- **High Confidence** (≥80%): {high_conf} commits ({high_pct:.1f}%)\n"
                    )
                    report.write(
                        f"- **Medium Confidence** (60-79%): {medium_conf} commits ({medium_pct:.1f}%)\n"
                    )
                    report.write(
                        f"- **Low Confidence** (<60%): {low_conf} commits ({low_pct:.1f}%)\n\n"
                    )

            # Category confidence breakdown
            category_confidence = ml_analysis.get("category_confidence", {})
            if category_confidence:
                report.write("### Classification Categories\n\n")

                # Sort categories by count (descending)
                sorted_categories = sorted(
                    category_confidence.items(), key=lambda x: x[1].get("count", 0), reverse=True
                )

                # Calculate total commits for percentages
                total_categorized = sum(
                    data.get("count", 0) for data in category_confidence.values()
                )

                for category, data in sorted_categories:
                    count = data.get("count", 0)
                    avg_conf = data.get("avg", 0)

                    if count > 0:
                        category_pct = (count / total_categorized) * 100
                        category_display = category.replace("_", " ").title()
                        report.write(
                            f"- **{category_display}**: {count} commits ({category_pct:.1f}%, avg confidence: {avg_conf:.1%})\n"
                        )

                report.write("\n")

            # Performance metrics
            processing_stats = ml_analysis.get("processing_time_stats", {})
            if processing_stats.get("total_ms", 0) > 0:
                avg_ms = processing_stats.get("avg_ms", 0)
                total_ms = processing_stats.get("total_ms", 0)

                report.write("### Processing Performance\n\n")
                report.write(f"- **Average Processing Time**: {avg_ms:.1f}ms per commit\n")
                report.write(
                    f"- **Total Processing Time**: {total_ms:.0f}ms ({total_ms / 1000:.1f} seconds)\n\n"
                )

        else:
            report.write("No classification data available for analysis.\n\n")

    def _write_pm_insights(self, report: StringIO, pm_data: dict[str, Any]) -> None:
        """Write PM platform integration insights.

        WHY: PM platform integration provides valuable insights into work item
        tracking, story point accuracy, and development velocity that complement
        Git-based analytics. This section highlights the value of PM integration.
        """
        metrics = pm_data.get("metrics", {})

        # Platform overview
        platform_coverage = metrics.get("platform_coverage", {})
        total_issues = metrics.get("total_pm_issues", 0)
        correlations = len(pm_data.get("correlations", []))

        report.write(f"The team has integrated **{len(platform_coverage)} PM platforms** ")
        report.write(
            f"tracking **{total_issues:,} issues** with **{correlations} commit correlations**.\n\n"
        )

        # Story point analysis
        story_analysis = metrics.get("story_point_analysis", {})
        pm_story_points = story_analysis.get("pm_total_story_points", 0)
        git_story_points = story_analysis.get("git_total_story_points", 0)
        coverage_pct = story_analysis.get("story_point_coverage_pct", 0)

        if pm_story_points > 0:
            report.write("### Story Point Tracking\n\n")
            report.write(f"- **PM Platform Story Points**: {pm_story_points:,}\n")
            report.write(f"- **Git Extracted Story Points**: {git_story_points:,}\n")
            report.write(
                f"- **Story Point Coverage**: {coverage_pct:.1f}% of issues have story points\n"
            )

            if git_story_points > 0:
                accuracy = min(git_story_points / pm_story_points, 1.0) * 100
                report.write(
                    f"- **Extraction Accuracy**: {accuracy:.1f}% of PM story points found in Git\n"
                )
            report.write("\n")

        # Issue type distribution
        issue_types = metrics.get("issue_type_distribution", {})
        if issue_types:
            report.write("### Work Item Types\n\n")
            sorted_types = sorted(issue_types.items(), key=lambda x: x[1], reverse=True)
            total_typed_issues = sum(issue_types.values())

            for issue_type, count in sorted_types[:5]:  # Top 5 types
                pct = (count / total_typed_issues * 100) if total_typed_issues > 0 else 0
                report.write(f"- **{issue_type.title()}**: {count} issues ({pct:.1f}%)\n")
            report.write("\n")

        # Platform-specific insights
        if platform_coverage:
            report.write("### Platform Coverage\n\n")
            for platform, coverage_data in platform_coverage.items():
                platform_issues = coverage_data.get("total_issues", 0)
                linked_issues = coverage_data.get("linked_issues", 0)
                coverage_percentage = coverage_data.get("coverage_percentage", 0)

                report.write(f"**{platform.title()}**: ")
                report.write(f"{platform_issues} issues, {linked_issues} linked to commits ")
                report.write(f"({coverage_percentage:.1f}% coverage)\n")
            report.write("\n")

        # Correlation quality
        correlation_quality = metrics.get("correlation_quality", {})
        if correlation_quality.get("total_correlations", 0) > 0:
            avg_confidence = correlation_quality.get("average_confidence", 0)
            high_confidence = correlation_quality.get("high_confidence_correlations", 0)
            correlation_methods = correlation_quality.get("correlation_methods", {})

            report.write("### Correlation Quality\n\n")
            report.write(f"- **Average Confidence**: {avg_confidence:.2f} (0.0-1.0 scale)\n")
            report.write(f"- **High Confidence Matches**: {high_confidence} correlations\n")

            if correlation_methods:
                report.write("- **Methods Used**: ")
                method_list = [
                    f"{method.replace('_', ' ').title()} ({count})"
                    for method, count in correlation_methods.items()
                ]
                report.write(", ".join(method_list))
                report.write("\n")
            report.write("\n")

        # Key insights
        report.write("### Key Insights\n\n")

        if coverage_pct > 80:
            report.write(
                "✅ **Excellent story point coverage** - Most issues have effort estimates\n"
            )
        elif coverage_pct > 50:
            report.write(
                "⚠️ **Moderate story point coverage** - Consider improving estimation practices\n"
            )
        else:
            report.write(
                "❌ **Low story point coverage** - Story point tracking needs improvement\n"
            )

        if correlations > total_issues * 0.5:
            report.write(
                "✅ **Strong commit-issue correlation** - Good traceability between work items and code\n"
            )
        elif correlations > total_issues * 0.2:
            report.write(
                "⚠️ **Moderate commit-issue correlation** - Some work items lack code links\n"
            )
        else:
            report.write(
                "❌ **Weak commit-issue correlation** - Improve ticket referencing in commits\n"
            )

        if len(platform_coverage) > 1:
            report.write(
                "📊 **Multi-platform integration** - Comprehensive work item tracking across tools\n"
            )

        report.write("\n")

    def _write_cicd_health(self, report: StringIO, cicd_data: dict[str, Any]) -> None:
        """Write CI/CD pipeline health analysis.

        WHY: CI/CD pipeline metrics provide critical insights into build stability,
        deployment velocity, and infrastructure health. This section helps teams
        identify pipeline issues that may be blocking deployments or slowing velocity.

        CORRELATION: CI/CD success rates correlate with deployment frequency (DORA)
        and can explain variations in development velocity.
        """
        pipelines = cicd_data.get("pipelines", [])
        platform_metrics = cicd_data.get("metrics", {})

        if not pipelines:
            report.write("No CI/CD pipeline data available.\n\n")
            return

        # Overall statistics
        total_pipelines = len(pipelines)
        successful = len([p for p in pipelines if p.get("status") == "success"])
        failed = len([p for p in pipelines if p.get("status") in ["failure", "failed"]])
        overall_success_rate = (successful / total_pipelines * 100) if total_pipelines > 0 else 0

        # Duration statistics
        durations = [
            p.get("duration_seconds", 0) / 60.0 for p in pipelines if p.get("duration_seconds")
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Platform overview
        platforms = list(platform_metrics.keys())
        report.write(f"The team runs CI/CD pipelines across **{len(platforms)} platform(s)** ")
        report.write(f"with **{total_pipelines:,} total pipeline runs** in this period.\n\n")

        # Success rate analysis
        report.write("### Pipeline Success Rate\n\n")
        report.write(f"- **Total Pipelines**: {total_pipelines:,}\n")
        report.write(f"- **Successful**: {successful:,} ({overall_success_rate:.1f}%)\n")
        report.write(f"- **Failed**: {failed:,} ({(failed / total_pipelines * 100):.1f}%)\n")
        report.write(f"- **Average Build Time**: {avg_duration:.1f} minutes\n\n")

        # Success rate assessment
        if overall_success_rate >= 90:
            report.write(
                "✅ **Excellent pipeline health** - High success rate indicates stable builds\n\n"
            )
        elif overall_success_rate >= 75:
            report.write(
                "⚠️ **Moderate pipeline health** - Consider investigating frequent failure causes\n\n"
            )
        else:
            report.write(
                "❌ **Poor pipeline health** - High failure rate may be blocking deployments\n\n"
            )

        # Duration trends analysis
        report.write("### Build Duration\n\n")
        if avg_duration < 5:
            report.write(
                f"✅ **Fast builds** ({avg_duration:.1f} min average) enable quick feedback cycles\n"
            )
        elif avg_duration < 15:
            report.write(
                f"⚠️ **Moderate build times** ({avg_duration:.1f} min average) - Consider optimization\n"
            )
        else:
            report.write(
                f"❌ **Slow builds** ({avg_duration:.1f} min average) may impact development velocity\n"
            )
        report.write("\n")

        # Platform-specific breakdown
        if len(platform_metrics) > 1:
            report.write("### Platform Breakdown\n\n")
            for platform, metrics in platform_metrics.items():
                platform_pipelines = metrics.get("total_pipelines", 0)
                platform_success_rate = metrics.get("success_rate", 0)
                platform_duration = metrics.get("avg_duration_minutes", 0)

                report.write(f"**{platform}**:\n")
                report.write(f"- Pipelines: {platform_pipelines:,}\n")
                report.write(f"- Success Rate: {platform_success_rate:.1f}%\n")
                report.write(f"- Avg Duration: {platform_duration:.1f} min\n\n")

        # Correlation with DORA metrics
        report.write("### Impact on Delivery Velocity\n\n")
        if overall_success_rate >= 85:
            report.write(
                "High CI/CD success rates support frequent deployments and align with "
                "DORA elite performer benchmarks. Stable pipelines enable continuous delivery.\n"
            )
        else:
            report.write(
                "Pipeline failures may be limiting deployment frequency and blocking releases. "
                "Improving CI/CD stability should be a priority to increase delivery velocity.\n"
            )
        report.write("\n")

        # Recommendations
        report.write("### Recommendations\n\n")
        if overall_success_rate < 85:
            report.write(
                "- **Investigate failure patterns**: Analyze failed pipelines to identify common causes\n"
            )
            report.write("- **Add retry logic**: Implement automatic retries for flaky tests\n")

        if avg_duration > 10:
            report.write(
                "- **Optimize build times**: Consider parallelization and caching strategies\n"
            )
            report.write("- **Split test suites**: Run critical tests first for faster feedback\n")

        if failed / total_pipelines > 0.15:
            report.write(
                "- **Improve test reliability**: High failure rate suggests flaky or brittle tests\n"
            )
            report.write(
                "- **Monitor infrastructure**: Check if failures are due to resource constraints\n"
            )

        report.write("\n")
