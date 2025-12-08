"""ChatGPT-based qualitative analysis for GitFlow Analytics.

This module uses OpenAI's ChatGPT-4.1 to generate insightful executive summaries
based on comprehensive GitFlow Analytics data.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


class ChatGPTQualitativeAnalyzer:
    """Generate qualitative insights using ChatGPT-4.1."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the ChatGPT analyzer.

        Args:
            api_key: API key. If not provided, uses OPENROUTER_API_KEY or OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not provided. Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable."
            )

        # Check if this is an OpenRouter key
        self.use_openrouter = self.api_key.startswith("sk-or-")
        if self.use_openrouter:
            self.api_url = "https://openrouter.ai/api/v1/chat/completions"
            self.model = "openai/gpt-4-turbo-preview"  # OpenRouter model name
        else:
            # Fallback to OpenAI direct
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
            self.model = "gpt-4-turbo-preview"

    def generate_executive_summary(self, comprehensive_data: dict[str, Any]) -> str:
        """Generate a qualitative executive summary from comprehensive export data.

        Args:
            comprehensive_data: The comprehensive JSON export data

        Returns:
            A markdown-formatted executive summary with qualitative insights
        """
        # Extract key metrics for the prompt
        summary_data = self._extract_summary_data(comprehensive_data)

        # Create the prompt
        prompt = self._create_executive_summary_prompt(summary_data)

        try:
            # Call ChatGPT
            messages = [
                {
                    "role": "system",
                    "content": "You are a data-driven software development analyst providing objective insights on team productivity and project health. Use factual, analytical language. Strictly avoid: promotional language, subjective assessments, marketing terms, superlatives (best, worst, great, poor), praise words (impressive, commendable, excellent, strong, effective), emotional qualifiers (positive, negative, concerning), interpretive language (suggests, indicates potential, appears to show). Report only measurable patterns, quantifiable trends, and evidence-based observations. Use neutral technical terminology. Present findings as statistical facts without subjective interpretation.",
                },
                {"role": "user", "content": prompt},
            ]

            if self.use_openrouter:
                # Use OpenRouter API
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/EWTN-Global/gitflow-analytics",
                    "X-Title": "GitFlow Analytics",
                }

                data = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 1500,
                    "temperature": 0.7,
                }

                response = requests.post(self.api_url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
            else:
                # Use OpenAI directly
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.7, max_tokens=1500
                )
                content = response.choices[0].message.content

            return content

        except Exception as e:
            logger.error(f"Error generating ChatGPT summary: {e}")
            return self._generate_fallback_summary(summary_data)

    def _extract_summary_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract key data points for the executive summary."""

        exec_summary = data.get("executive_summary", {})
        metadata = data.get("metadata", {})
        projects = data.get("projects", {})
        developers = data.get("developers", {})

        # Get top contributors
        top_developers = []
        for _dev_id, dev_data in developers.items():
            identity = dev_data.get("identity", {})
            summary = dev_data.get("summary", {})
            top_developers.append(
                {
                    "name": identity.get("name", "Unknown"),
                    "commits": summary.get("total_commits", 0),
                    "story_points": summary.get("total_story_points", 0),
                    "projects": len(dev_data.get("projects", {})),
                }
            )
        top_developers.sort(key=lambda x: x["commits"], reverse=True)

        # Get project health
        project_health = []
        for proj_key, proj_data in projects.items():
            health = proj_data.get("health_score", {})
            summary = proj_data.get("summary", {})
            project_health.append(
                {
                    "name": proj_key,
                    "health_score": health.get("overall", 0),
                    "health_rating": health.get("rating", "unknown"),
                    "commits": summary.get("total_commits", 0),
                    "contributors": summary.get("total_contributors", 0),
                }
            )
        project_health.sort(key=lambda x: x["commits"], reverse=True)

        return {
            "period_weeks": metadata.get("analysis_weeks", 0),
            "total_commits": exec_summary.get("key_metrics", {}).get("commits", {}).get("total", 0),
            "total_developers": exec_summary.get("key_metrics", {})
            .get("developers", {})
            .get("total", 0),
            "lines_changed": exec_summary.get("key_metrics", {})
            .get("lines_changed", {})
            .get("total", 0),
            "story_points": exec_summary.get("key_metrics", {})
            .get("story_points", {})
            .get("total", 0),
            "ticket_coverage": exec_summary.get("key_metrics", {})
            .get("ticket_coverage", {})
            .get("percentage", 0),
            "team_health_score": exec_summary.get("health_score", {}).get("overall", 0),
            "team_health_rating": exec_summary.get("health_score", {}).get("rating", "unknown"),
            "top_developers": top_developers[:5],
            "project_health": project_health[:5],
            "velocity_trend": exec_summary.get("trends", {})
            .get("velocity", {})
            .get("direction", "stable"),
            "wins": exec_summary.get("wins", [])[:3],
            "concerns": exec_summary.get("concerns", [])[:3],
            "anomalies": data.get("anomaly_detection", {}).get("anomalies", [])[:3],
        }

    def _create_executive_summary_prompt(self, summary_data: dict[str, Any]) -> str:
        """Create the prompt for ChatGPT."""

        prompt = f"""Based on the following GitFlow Analytics data from the past {summary_data["period_weeks"]} weeks, provide a comprehensive executive summary with qualitative insights:

## Key Metrics:
- Total Commits: {summary_data["total_commits"]:,}
- Active Developers: {summary_data["total_developers"]}
- Lines Changed: {summary_data["lines_changed"]:,}
- Story Points Delivered: {summary_data["story_points"]}
- Ticket Coverage: {summary_data["ticket_coverage"]:.1f}%
- Team Health Score: {summary_data["team_health_score"]:.1f}/100 ({summary_data["team_health_rating"]})
- Velocity Trend: {summary_data["velocity_trend"]}

## Top Contributors:
"""

        for dev in summary_data["top_developers"]:
            prompt += f"- {dev['name']}: {dev['commits']} commits, {dev['story_points']} story points across {dev['projects']} projects\n"

        prompt += "\n## Project Health:\n"
        for proj in summary_data["project_health"]:
            prompt += f"- {proj['name']}: Health {proj['health_score']:.1f}/100 ({proj['health_rating']}), {proj['commits']} commits, {proj['contributors']} contributors\n"

        if summary_data["wins"]:
            prompt += "\n## Recent Wins:\n"
            for win in summary_data["wins"]:
                prompt += f"- {win.get('title', 'Achievement')}: {win.get('description', '')}\n"

        if summary_data["concerns"]:
            prompt += "\n## Areas of Concern:\n"
            for concern in summary_data["concerns"]:
                prompt += f"- {concern.get('title', 'Issue')}: {concern.get('description', '')}\n"

        if summary_data["anomalies"]:
            prompt += "\n## Detected Anomalies:\n"
            for anomaly in summary_data["anomalies"]:
                prompt += f"- {anomaly.get('type', 'anomaly')} in {anomaly.get('metric', 'unknown')}: {anomaly.get('description', '')}\n"

        prompt += """
Provide a technical analysis with:
1. A 2-3 paragraph data summary reporting team output patterns, work distribution, and process metrics
2. Quantitative observations about team dynamics derived from the metrics (commit frequency, ticket coverage ratios, health score calculations)
3. 3-5 specific, data-driven recommendations with measurable targets based on metric thresholds
4. Observable patterns in the data that deviate from baseline measurements (if applicable)

Report only statistical patterns, measurable trends, and process gaps. Use factual technical language. Base all statements on quantifiable metrics and their mathematical relationships. Avoid subjective interpretations or assessments.
"""

        return prompt

    def _generate_fallback_summary(self, summary_data: dict[str, Any]) -> str:
        """Generate a basic summary if ChatGPT fails."""

        return f"""## Executive Summary

Over the past {summary_data["period_weeks"]} weeks, the development team generated {summary_data["total_commits"]:,} commits across {summary_data["total_developers"]} active developers.

The team health score measured {summary_data["team_health_score"]:.1f}/100 ({summary_data["team_health_rating"]}). Ticket coverage reached {summary_data["ticket_coverage"]:.1f}% of total commits with trackable references.

### Measured Outputs:
- Code changes: {summary_data["lines_changed"]:,} lines modified
- Story points completed: {summary_data["story_points"]}
- Velocity trend: {summary_data["velocity_trend"]}

### Process Recommendations:
1. {"Maintain current output rate" if summary_data["velocity_trend"] == "increasing" else "Analyze velocity decline factors"}
2. {"Sustain current tracking rate" if summary_data["ticket_coverage"] > 60 else "Increase commit-ticket linking to reach 70% coverage target"}
3. Review projects with health scores below 60/100 for process gaps

*Note: This is a fallback summary. For detailed analysis, configure ChatGPT integration.*
"""


def generate_chatgpt_summary(json_file_path: Path, api_key: Optional[str] = None) -> str:
    """Convenience function to generate a ChatGPT summary from a JSON export file.

    Args:
        json_file_path: Path to the comprehensive JSON export
        api_key: Optional OpenAI API key

    Returns:
        Markdown-formatted executive summary
    """
    # Load the JSON data
    with open(json_file_path) as f:
        comprehensive_data = json.load(f)

    # Generate summary
    analyzer = ChatGPTQualitativeAnalyzer(api_key)
    return analyzer.generate_executive_summary(comprehensive_data)
