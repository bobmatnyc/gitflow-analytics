"""HTML section generators and helper methods for HTMLReportGenerator.

Extracted from html_generator.py to keep file sizes manageable.
Contains workflow HTML, chart JS, interaction JS, qualitative analysis,
insight formatters, and badge color helpers.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HTMLSectionsMixin:
    """Mixin providing section generators and helpers for HTMLReportGenerator."""

    def _generate_workflow_html(self, json_data: Dict[str, Any]) -> str:
        """Generate workflow analysis HTML section."""
        workflow = json_data.get("workflow_analysis", {})

        if not workflow:
            return '<section id="workflow" class="mb-5"><h2>Workflow Analysis</h2><p class="text-muted">No workflow data available.</p></section>'

        branching = workflow.get("branching_strategy", {})
        commit_patterns = workflow.get("commit_patterns", {})
        process_health = workflow.get("process_health", {})

        html = f"""
        <section id="workflow" class="mb-5">
            <h2 class="mb-4">Workflow Analysis</h2>

            <div class="row mb-4">
                <!-- Branching Strategy -->
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">Branching Strategy</h6>
                        </div>
                        <div class="card-body">
                            <p class="h5 text-capitalize">{branching.get('strategy', 'Unknown')}</p>
                            <p class="text-muted">Merge Rate: {branching.get('merge_rate_percent', 0):.1f}%</p>
                            <span class="badge bg-{self._get_complexity_badge_color(branching.get('complexity_rating', 'medium'))}">
                                {branching.get('complexity_rating', 'Medium').title()}
                            </span>
                        </div>
                    </div>
                </div>

                <!-- Commit Timing -->
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">Commit Patterns</h6>
                        </div>
                        <div class="card-body">
                            <p><strong>Peak Hour:</strong> {commit_patterns.get('peak_hour', 'Unknown')}</p>
                            <p><strong>Peak Day:</strong> {commit_patterns.get('peak_day', 'Unknown')}</p>
                            <small class="text-muted">
                                Weekdays: {commit_patterns.get('weekday_pct', 0):.1f}%<br>
                                Weekends: {commit_patterns.get('weekend_pct', 0):.1f}%
                            </small>
                        </div>
                    </div>
                </div>

                <!-- Process Health -->
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">Process Health</h6>
                        </div>
                        <div class="card-body">
                            <p><strong>Ticket Linking:</strong> {process_health.get('ticket_linking_rate', 0):.1f}%</p>
                            <p><strong>Merge Commits:</strong> {process_health.get('merge_commit_rate', 0):.1f}%</p>
                            <span class="badge bg-{self._get_quality_badge_color(process_health.get('commit_message_quality', {}).get('overall_rating', 'fair'))}">
                                {process_health.get('commit_message_quality', {}).get('overall_rating', 'Fair').title()}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        """

        return html

    def _generate_charts_js(self, json_data: Dict[str, Any]) -> str:
        """Generate Chart.js initialization JavaScript."""

        # Get data for charts
        exec_summary = json_data.get("executive_summary", {})
        health_score = exec_summary.get("health_score", {})
        time_series = json_data.get("time_series", {})

        # Health score components
        health_components = health_score.get("components", {})
        health_labels = list(health_components.keys())
        health_data = list(health_components.values())

        # Time series data
        weekly_data = time_series.get("weekly", {})
        weekly_labels = weekly_data.get("labels", [])
        commits_data = weekly_data.get("datasets", {}).get("commits", {}).get("data", [])
        lines_data = weekly_data.get("datasets", {}).get("lines_changed", {}).get("data", [])

        js_code = f"""
        // Chart.js configuration and initialization
        document.addEventListener('DOMContentLoaded', function() {{
            // Health Score Radar Chart
            const healthCtx = document.getElementById('healthScoreChart');
            if (healthCtx) {{
                new Chart(healthCtx, {{
                    type: 'radar',
                    data: {{
                        labels: {json.dumps(health_labels)},
                        datasets: [{{
                            label: 'Health Score',
                            data: {json.dumps(health_data)},
                            backgroundColor: 'rgba(13, 110, 253, 0.2)',
                            borderColor: 'rgba(13, 110, 253, 1)',
                            borderWidth: 2
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            r: {{
                                beginAtZero: true,
                                max: 100
                            }}
                        }}
                    }}
                }});
            }}

            // Activity Trend Line Chart
            const activityCtx = document.getElementById('activityTrendChart');
            if (activityCtx) {{
                new Chart(activityCtx, {{
                    type: 'line',
                    data: {{
                        labels: {json.dumps(weekly_labels)},
                        datasets: [{{
                            label: 'Commits',
                            data: {json.dumps(commits_data)},
                            backgroundColor: 'rgba(13, 110, 253, 0.1)',
                            borderColor: 'rgba(13, 110, 253, 1)',
                            borderWidth: 2,
                            fill: true
                        }}, {{
                            label: 'Lines Changed',
                            data: {json.dumps(lines_data)},
                            backgroundColor: 'rgba(25, 135, 84, 0.1)',
                            borderColor: 'rgba(25, 135, 84, 1)',
                            borderWidth: 2,
                            fill: false,
                            yAxisID: 'y1'
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            y: {{
                                type: 'linear',
                                display: true,
                                position: 'left',
                            }},
                            y1: {{
                                type: 'linear',
                                display: true,
                                position: 'right',
                                grid: {{
                                    drawOnChartArea: false,
                                }},
                            }}
                        }}
                    }}
                }});
            }}
        }});
        """

        return js_code

    def _get_interaction_js(self) -> str:
        """Get JavaScript for page interactions."""
        return """
        // Smooth scrolling for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Update active navigation link on scroll
        window.addEventListener('scroll', function() {
            const sections = document.querySelectorAll('section[id]');
            const navLinks = document.querySelectorAll('.nav-link');

            let current = '';
            sections.forEach(section => {
                const sectionTop = section.offsetTop;
                const sectionHeight = section.clientHeight;
                if (scrollY >= (sectionTop - 200)) {
                    current = section.getAttribute('id');
                }
            });

            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === '#' + current) {
                    link.classList.add('active');
                }
            });
        });

        // Add hover effects and tooltips
        document.querySelectorAll('.metric-card').forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
            });
            card.addEventListener('mouseleave', function() {
                this.style.boxShadow = '';
            });
        });
        """

    def _generate_qualitative_analysis_section(self, json_data: Dict[str, Any]) -> str:
        """Generate enhanced qualitative analysis section with executive narrative."""

        # Get enhanced qualitative analysis if available
        enhanced_analysis = json_data.get("enhanced_qualitative_analysis", {})
        if not enhanced_analysis:
            return ""

        # Get executive analysis
        exec_analysis = enhanced_analysis.get("executive_analysis", {})
        if not exec_analysis:
            return ""

        # Build the qualitative analysis section
        html = f"""
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h6 class="mb-0">Qualitative Analysis</h6>
                    </div>
                    <div class="card-body">
                        <!-- Executive Summary Narrative -->
                        <div class="mb-4">
                            <h6 class="text-primary">Executive Summary</h6>
                            <p class="lead">{exec_analysis.get('executive_summary', 'No executive summary available.')}</p>
                        </div>

                        <!-- Health Assessment -->
                        <div class="mb-4">
                            <h6 class="text-primary">Team Health Assessment</h6>
                            <p>{exec_analysis.get('health_narrative', 'No health assessment available.')}</p>
                            <div class="d-flex align-items-center mb-2">
                                <strong class="me-2">Confidence:</strong>
                                <div class="progress flex-grow-1" style="height: 20px;">
                                    <div class="progress-bar" role="progressbar"
                                         style="width: {exec_analysis.get('health_confidence', 0) * 100}%"
                                         aria-valuenow="{exec_analysis.get('health_confidence', 0) * 100}"
                                         aria-valuemin="0" aria-valuemax="100">
                                        {exec_analysis.get('health_confidence', 0) * 100:.0f}%
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Velocity Trends -->
                        <div class="mb-4">
                            <h6 class="text-primary">Velocity Analysis</h6>
                            <p>{exec_analysis.get('velocity_trends', {}).get('narrative', 'No velocity analysis available.')}</p>
                        </div>

                        <!-- Key Achievements -->
                        {self._format_achievements_section(exec_analysis.get('key_achievements', []))}

                        <!-- Major Concerns with Recommendations -->
                        {self._format_concerns_section(exec_analysis.get('major_concerns', []))}

                        <!-- Cross-Dimensional Insights -->
                        {self._format_cross_insights_section(enhanced_analysis.get('cross_insights', []))}
                    </div>
                </div>
            </div>
        </div>
        """

        return html

    def _format_achievements_section(self, achievements: List[Dict[str, Any]]) -> str:
        """Format achievements section with details."""
        if not achievements:
            return ""

        items_html = []
        for achievement in achievements[:5]:  # Show top 5
            badge_color = {
                "exceptional": "success",
                "excellent": "primary",
                "good": "info",
                "notable": "secondary",
            }.get(achievement.get("impact", "notable"), "secondary")

            item = f"""
            <div class="achievement-item mb-2">
                <div class="d-flex align-items-start">
                    <span class="badge bg-{badge_color} me-2">{achievement.get('impact', 'notable').title()}</span>
                    <div>
                        <strong>{achievement.get('title', 'Achievement')}</strong>
                        <p class="mb-1 text-muted small">{achievement.get('description', '')}</p>
                        {f'<small class="text-success">{achievement.get("recommendation", "")}</small>' if achievement.get('recommendation') else ''}
                    </div>
                </div>
            </div>
            """
            items_html.append(item)

        return f"""
        <div class="mb-4">
            <h6 class="text-success">Key Achievements</h6>
            {''.join(items_html)}
        </div>
        """

    def _format_concerns_section(self, concerns: List[Dict[str, Any]]) -> str:
        """Format concerns section with recommendations."""
        if not concerns:
            return ""

        items_html = []
        for concern in concerns[:5]:  # Show top 5
            severity_color = {
                "critical": "danger",
                "high": "warning",
                "medium": "info",
                "low": "secondary",
            }.get(concern.get("severity", "medium"), "warning")

            item = f"""
            <div class="concern-item mb-2">
                <div class="d-flex align-items-start">
                    <span class="badge bg-{severity_color} me-2">{concern.get('severity', 'medium').title()}</span>
                    <div>
                        <strong>{concern.get('title', 'Concern')}</strong>
                        <p class="mb-1 text-muted small">{concern.get('description', '')}</p>
                        {f'<small class="text-primary"><strong>Recommendation:</strong> {concern.get("recommendation", "")}</small>' if concern.get('recommendation') else ''}
                    </div>
                </div>
            </div>
            """
            items_html.append(item)

        return f"""
        <div class="mb-4">
            <h6 class="text-warning">Areas Requiring Attention</h6>
            {''.join(items_html)}
        </div>
        """

    def _format_cross_insights_section(self, insights: List[Dict[str, Any]]) -> str:
        """Format cross-dimensional insights."""
        if not insights:
            return ""

        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_insights = sorted(
            insights, key=lambda x: priority_order.get(x.get("priority", "low"), 3)
        )

        items_html = []
        for insight in sorted_insights[:3]:  # Show top 3
            priority_color = {
                "critical": "danger",
                "high": "warning",
                "medium": "info",
                "low": "secondary",
            }.get(insight.get("priority", "medium"), "info")

            dimensions = insight.get("dimensions", [])
            dimensions_badges = " ".join(
                [f'<span class="badge bg-light text-dark me-1">{d}</span>' for d in dimensions]
            )

            item = f"""
            <div class="insight-item mb-3 p-3 border rounded">
                <div class="d-flex justify-content-between align-items-start mb-2">
                    <h6 class="mb-0">{insight.get('title', 'Insight')}</h6>
                    <span class="badge bg-{priority_color}">{insight.get('priority', 'medium').title()} Priority</span>
                </div>
                <p class="mb-2 text-muted">{insight.get('description', '')}</p>
                <div class="mb-2">{dimensions_badges}</div>
                {f'<div class="alert alert-info mb-0"><strong>Action:</strong> {insight.get("action_required", "")}</div>' if insight.get('action_required') else ''}
            </div>
            """
            items_html.append(item)

        return f"""
        <div class="mb-4">
            <h6 class="text-info">Strategic Insights</h6>
            <p class="text-muted small">Cross-dimensional patterns requiring leadership attention</p>
            {''.join(items_html)}
        </div>
        """

    def _format_insights_list(self, insights: List[Dict[str, Any]]) -> str:
        """Format a list of insights as HTML."""
        if not insights:
            return '<p class="text-muted">No insights available.</p>'

        html_items = []
        for insight in insights[:5]:  # Limit to top 5
            title = insight.get("title", "Insight")
            description = insight.get("description", "")
            impact = insight.get("impact", "medium")

            item_html = f"""
            <div class="mb-3">
                <div class="d-flex justify-content-between align-items-start">
                    <h6 class="mb-1">{title}</h6>
                    <span class="badge bg-{self._get_impact_badge_color(impact)}">{impact.title()}</span>
                </div>
                <p class="mb-0 text-muted small">{description}</p>
            </div>
            """
            html_items.append(item_html)

        return "".join(html_items)

    def _get_health_badge_color(self, rating: str) -> str:
        """Get Bootstrap badge color for health rating."""
        color_map = {
            "excellent": "success",
            "good": "info",
            "fair": "warning",
            "needs_improvement": "danger",
            "no_data": "secondary",
        }
        return color_map.get(rating, "secondary")

    def _get_complexity_badge_color(self, complexity: str) -> str:
        """Get Bootstrap badge color for complexity rating."""
        color_map = {"low": "success", "medium": "warning", "high": "danger"}
        return color_map.get(complexity, "secondary")

    def _get_quality_badge_color(self, quality: str) -> str:
        """Get Bootstrap badge color for quality rating."""
        color_map = {
            "excellent": "success",
            "good": "info",
            "fair": "warning",
            "needs_improvement": "danger",
            "poor": "danger",
        }
        return color_map.get(quality, "secondary")

    def _get_impact_badge_color(self, impact: str) -> str:
        """Get Bootstrap badge color for impact level."""
        color_map = {"high": "danger", "medium": "warning", "low": "info"}
        return color_map.get(impact, "secondary")

    # Maintain backward compatibility with the old method name
