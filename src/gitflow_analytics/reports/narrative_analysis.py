"""Narrative report mixin: development patterns, PR analysis, ticket tracking, and untracked commits.

Extracted from narrative_writer.py to keep file sizes manageable.
"""

import logging
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)



class NarrativeAnalysisMixin:
    """Mixin: development patterns, PR analysis, ticket/untracked sections."""

    def _get_week_start(self, date: datetime) -> datetime:
        """Get Monday of the week for a given date."""
        # Ensure consistent timezone handling - keep timezone info
        if hasattr(date, "tzinfo") and date.tzinfo is not None:
            # Keep timezone-aware but ensure it's UTC
            if date.tzinfo != timezone.utc:
                date = date.astimezone(timezone.utc)
        else:
            # Convert naive datetime to UTC timezone-aware
            date = date.replace(tzinfo=timezone.utc)

        days_since_monday = date.weekday()
        monday = date - timedelta(days=days_since_monday)
        result = monday.replace(hour=0, minute=0, second=0, microsecond=0)

        return result

    def _write_development_patterns(
        self, report: StringIO, insights: list[dict[str, Any]], focus_data: list[dict[str, Any]]
    ) -> None:
        """Write development patterns analysis."""
        report.write("### Key Patterns Identified\n\n")

        # Group insights by category (handle missing category field gracefully)
        by_category: dict[str, list[dict[str, Any]]] = {}
        for insight in insights:
            category = insight.get("category", "General")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(insight)

        for category, category_insights in by_category.items():
            report.write(f"**{category}**:\n")
            for insight in category_insights:
                # Handle missing fields gracefully for test compatibility
                insight_text = insight.get("insight", insight.get("metric", "Unknown"))
                insight_value = insight.get("value", "N/A")
                insight_impact = insight.get("impact", "No impact specified")
                report.write(f"- {insight_text}: {insight_value} ")
                report.write(f"({insight_impact})\n")
            report.write("\n")

        # Add focus insights (handle missing focus_score field gracefully)
        if focus_data:
            # Use focus_ratio if focus_score is not available
            focus_scores = []
            for d in focus_data:
                if "focus_score" in d:
                    focus_scores.append(d["focus_score"])
                elif "focus_ratio" in d:
                    focus_scores.append(d["focus_ratio"] * 100)  # Convert ratio to percentage
                else:
                    focus_scores.append(50)  # Default value

            if focus_scores:
                avg_focus = sum(focus_scores) / len(focus_scores)
                report.write(f"**Developer Focus**: Average focus score of {avg_focus:.1f}% ")

                if avg_focus > 80:
                    report.write("indicates strong project concentration\n")
                elif avg_focus > 60:
                    report.write("shows moderate multi-project work\n")
                else:
                    report.write("suggests high context switching\n")

    def _write_pr_analysis(
        self, report: StringIO, pr_metrics: dict[str, Any], prs: list[dict[str, Any]]
    ) -> None:
        """Write pull request analysis.

        Sections emitted (always):
          - Overview: total PRs, average size, lifetime, story-point coverage

        Sections emitted only when review data is present (fetch_pr_reviews=true):
          - Review Metrics: approval rate, review coverage, avg reviews per PR
          - Comment Metrics: PR comments, inline review comments, averages
          - Time Metrics: avg and median time to first review
          - Revision Metrics: avg revisions per PR
          - Change Request Metrics: change request rate, avg per PR
        """
        total_prs = pr_metrics.get("total_prs", 0)

        # --- Overview ---
        report.write("### Overview\n\n")
        report.write(f"- **Total PRs Merged**: {total_prs}\n")
        report.write(f"- **Average PR Size**: {pr_metrics.get('avg_pr_size', 0):.0f} lines\n")

        if "avg_files_per_pr" in pr_metrics:
            report.write(f"- **Average Files per PR**: {pr_metrics['avg_files_per_pr']:.1f}\n")

        if "avg_pr_lifetime_hours" in pr_metrics:
            lifetime_h = pr_metrics["avg_pr_lifetime_hours"]
            if lifetime_h >= 24:
                lifetime_str = f"{lifetime_h / 24:.1f} days ({lifetime_h:.1f} hours)"
            else:
                lifetime_str = f"{lifetime_h:.1f} hours"
            report.write(f"- **Average PR Lifetime**: {lifetime_str}\n")

        if "story_point_coverage" in pr_metrics:
            prs_with_sp = pr_metrics.get("prs_with_story_points", 0)
            report.write(
                f"- **Story Point Coverage**: {pr_metrics['story_point_coverage']:.1f}%"
                f" ({prs_with_sp} of {total_prs} PRs)\n"
            )

        # --- Review Metrics (only when review data was collected) ---
        # Use the explicit ``review_data_collected`` flag set by
        # ``GitHubIntegration.calculate_pr_metrics()`` when fetch_pr_reviews
        # was enabled.  This avoids false-positive display when approval_rate
        # happens to be 0.0 for a no-data run.
        approval_rate = pr_metrics.get("approval_rate")
        review_coverage = pr_metrics.get("review_coverage")
        has_review_data = bool(pr_metrics.get("review_data_collected", False))

        if has_review_data:
            report.write("\n### Review Metrics\n\n")

            if approval_rate is not None:
                report.write(f"- **Approval Rate**: {approval_rate:.1f}%\n")

            if review_coverage is not None:
                report.write(f"- **Review Coverage**: {review_coverage:.1f}%\n")

            avg_approvals = pr_metrics.get("avg_approvals_per_pr")
            if avg_approvals is not None:
                report.write(f"- **Average Approvals per PR**: {avg_approvals:.2f}\n")

        # --- Comment Metrics ---
        total_inline = pr_metrics.get("total_review_comments", 0)
        total_pr_comments = pr_metrics.get("total_pr_comments", 0)
        has_comment_data = total_inline > 0 or total_pr_comments > 0

        if has_comment_data:
            report.write("\n### Comment Metrics\n\n")

            if total_inline > 0:
                avg_inline = pr_metrics.get(
                    "avg_review_comments_per_pr",
                    total_inline / total_prs if total_prs > 0 else 0,
                )
                report.write(
                    f"- **Total Inline Review Comments**: {total_inline}"
                    f" ({avg_inline:.1f} avg per PR)\n"
                )

            if total_pr_comments > 0:
                avg_pr_comments = pr_metrics.get("avg_pr_comments_per_pr", 0)
                report.write(
                    f"- **Total PR Comments**: {total_pr_comments}"
                    f" ({avg_pr_comments:.1f} avg per PR)\n"
                )

            # Combined total for quick scanning
            combined_total = total_inline + total_pr_comments
            if total_inline > 0 and total_pr_comments > 0:
                avg_combined = combined_total / total_prs if total_prs > 0 else 0
                report.write(
                    f"- **Combined Comments Total**: {combined_total}"
                    f" ({avg_combined:.1f} avg per PR)\n"
                )

        # --- Time-to-Review Metrics ---
        avg_ttfr = pr_metrics.get("avg_time_to_first_review_hours")
        median_ttfr = pr_metrics.get("median_time_to_first_review_hours")

        if avg_ttfr is not None:
            report.write("\n### Time-to-Review Metrics\n\n")

            def _fmt_hours(h: float) -> str:
                if h >= 24:
                    return f"{h / 24:.1f} days ({h:.1f} hours)"
                return f"{h:.1f} hours"

            report.write(f"- **Average Time to First Review**: {_fmt_hours(avg_ttfr)}\n")

            if median_ttfr is not None:
                report.write(f"- **Median Time to First Review**: {_fmt_hours(median_ttfr)}\n")

            # Qualitative interpretation to aid readers
            if avg_ttfr <= 4:
                interpretation = "fast review turnaround (under 4 hours)"
            elif avg_ttfr <= 24:
                interpretation = "same-day review turnaround"
            elif avg_ttfr <= 72:
                interpretation = "review within a few days"
            else:
                interpretation = "slow review cycle — consider review SLAs"
            report.write(f"- **Assessment**: {interpretation}\n")

        # --- Revision Metrics ---
        avg_revisions = pr_metrics.get("avg_revision_count")
        if avg_revisions is not None and avg_revisions > 0:
            report.write("\n### Revision Metrics\n\n")
            report.write(f"- **Average Revisions per PR**: {avg_revisions:.2f}\n")

            if avg_revisions < 1:
                pass
            elif avg_revisions < 2:
                interpretation = "typical single-revision cycle"
                report.write(f"- **Assessment**: {interpretation}\n")
            else:
                report.write(
                    "- **Assessment**: multiple revision cycles — consider smaller PRs"
                    " or clearer requirements\n"
                )

        # --- Change Request Metrics ---
        avg_cr = pr_metrics.get("avg_change_requests_per_pr")
        if avg_cr is not None and avg_cr > 0:
            report.write("\n### Change Request Metrics\n\n")
            report.write(f"- **Average Change Requests per PR**: {avg_cr:.2f}\n")

            # Derive change-request rate from prs list when possible
            if prs:
                prs_with_cr = sum(1 for pr in prs if (pr.get("change_requests_count") or 0) > 0)
                cr_rate = prs_with_cr / len(prs) * 100
                report.write(f"- **Change Request Rate**: {cr_rate:.1f}% of PRs\n")

    def _write_ticket_tracking(
        self,
        report: StringIO,
        ticket_analysis: dict[str, Any],
        developer_stats: list[dict[str, Any]],
    ) -> None:
        """Write ticket tracking analysis with simplified platform usage section."""
        # Simplified platform usage matching old report format
        ticket_summary = ticket_analysis.get("ticket_summary", {})
        total_tickets = sum(ticket_summary.values()) if ticket_summary else 0

        if total_tickets > 0:
            report.write("### Platform Usage\n\n")
            for platform, count in sorted(ticket_summary.items(), key=lambda x: x[1], reverse=True):
                pct = count / total_tickets * 100 if total_tickets > 0 else 0
                report.write(f"- **{platform.title()}**: {count} tickets ({pct:.1f}%)\n")

        report.write("\n### Coverage Analysis\n\n")

        # Handle missing fields gracefully
        commits_with_tickets = ticket_analysis.get("commits_with_tickets", 0)
        total_commits = ticket_analysis.get("total_commits", 0)
        coverage_pct = ticket_analysis.get("commit_coverage_pct", 0)

        # Debug logging for ticket coverage issues
        logger.debug(
            f"Ticket coverage analysis - commits_with_tickets: {commits_with_tickets}, total_commits: {total_commits}, coverage_pct: {coverage_pct}"
        )
        if commits_with_tickets == 0 and total_commits > 0:
            logger.warning(
                f"No commits found with ticket references out of {total_commits} total commits"
            )
            # Log sample of ticket_analysis structure for debugging
            if "ticket_summary" in ticket_analysis:
                logger.debug(f"Ticket summary: {ticket_analysis['ticket_summary']}")
            if "ticket_platforms" in ticket_analysis:
                logger.debug(f"Ticket platforms: {ticket_analysis['ticket_platforms']}")

        report.write(f"- **Commits with Tickets**: {commits_with_tickets} ")
        report.write(f"of {total_commits} ")
        report.write(f"({coverage_pct:.1f}%)\n")

        # Enhanced untracked commits reporting
        untracked_commits = ticket_analysis.get("untracked_commits", [])
        if untracked_commits:
            self._write_enhanced_untracked_analysis(
                report, untracked_commits, ticket_analysis, developer_stats
            )

    def _write_enhanced_untracked_analysis(
        self,
        report: StringIO,
        untracked_commits: list[dict[str, Any]],
        ticket_analysis: dict[str, Any],
        developer_stats: list[dict[str, Any]],
    ) -> None:
        """Write comprehensive untracked commits analysis.

        WHY: Enhanced untracked analysis provides actionable insights into what
        types of work are happening outside the tracked process, helping identify
        process improvements and training opportunities.
        """
        report.write("\n### Untracked Work Analysis\n\n")

        total_untracked = len(untracked_commits)
        total_commits = ticket_analysis.get("total_commits", 0)
        untracked_pct = (total_untracked / total_commits * 100) if total_commits > 0 else 0

        report.write(
            f"**Summary**: {total_untracked} commits ({untracked_pct:.1f}% of total) lack ticket references.\n\n"
        )

        # Analyze categories
        categories = {}
        contributors = {}
        projects = {}

        for commit in untracked_commits:
            # Category analysis
            category = commit.get("category", "other")
            if category not in categories:
                categories[category] = {"count": 0, "lines": 0, "examples": []}
            categories[category]["count"] += 1
            categories[category]["lines"] += commit.get("lines_changed", 0)
            if len(categories[category]["examples"]) < 2:
                categories[category]["examples"].append(
                    {
                        "hash": commit.get("hash", ""),
                        "message": commit.get("message", ""),
                        "author": commit.get("author", ""),
                    }
                )

            # Contributor analysis - use canonical_id as key for accurate lookup
            # Store display name separately so we can show the correct name in reports
            canonical_id = commit.get("canonical_id", "unknown")
            author = commit.get("author", "Unknown")
            if canonical_id not in contributors:
                contributors[canonical_id] = {"count": 0, "categories": set(), "name": author}
            contributors[canonical_id]["count"] += 1
            contributors[canonical_id]["categories"].add(category)

            # Project analysis
            project = commit.get("project_key", "UNKNOWN")
            if project not in projects:
                projects[project] = {"count": 0, "categories": set()}
            projects[project]["count"] += 1
            projects[project]["categories"].add(category)

        # Write category breakdown
        if categories:
            report.write("#### Work Categories\n\n")
            sorted_categories = sorted(
                categories.items(), key=lambda x: x[1]["count"], reverse=True
            )

            for category, data in sorted_categories[:8]:  # Show top 8 categories
                pct = (data["count"] / total_untracked) * 100
                avg_size = data["lines"] / data["count"] if data["count"] > 0 else 0

                # Categorize the impact
                if category in ["style", "documentation", "maintenance"]:
                    impact_note = " *(acceptable untracked)*"
                elif category in ["feature", "bug_fix"]:
                    impact_note = " *(should be tracked)*"
                else:
                    impact_note = ""

                report.write(f"- **{category.replace('_', ' ').title()}**: ")
                report.write(f"{data['count']} commits ({pct:.1f}%), ")
                report.write(f"avg {avg_size:.0f} lines{impact_note}\n")

                # Add examples
                if data["examples"]:
                    for example in data["examples"]:
                        report.write(f"  - `{example['hash']}`: {example['message'][:80]}...\n")
            report.write("\n")

        # Write top contributors to untracked work with enhanced percentage analysis
        if contributors:
            report.write("#### Top Contributors (Untracked Work)\n\n")

            # Create developer lookup for total commits
            dev_lookup = {}
            for dev in developer_stats:
                # Map canonical_id to developer data
                dev_lookup[dev["canonical_id"]] = dev
                # Also map primary name and primary email as fallbacks
                dev_lookup[dev["primary_name"]] = dev
                dev_lookup[dev["primary_email"]] = dev

            sorted_contributors = sorted(
                contributors.items(), key=lambda x: x[1]["count"], reverse=True
            )

            for canonical_id, data in sorted_contributors[:5]:  # Show top 5
                untracked_count = data["count"]
                pct_of_untracked = (untracked_count / total_untracked) * 100
                # Get the display name from the stored data
                author_name = data.get("name", "Unknown")

                # Find developer's total commits using canonical_id
                dev_data = dev_lookup.get(canonical_id)
                if dev_data:
                    total_dev_commits = dev_data["total_commits"]
                    pct_of_dev_work = (
                        (untracked_count / total_dev_commits) * 100 if total_dev_commits > 0 else 0
                    )
                    dev_context = f", {pct_of_dev_work:.1f}% of their work"
                else:
                    dev_context = ""

                categories_list = list(data["categories"])
                categories_str = ", ".join(categories_list[:3])  # Show up to 3 categories
                if len(categories_list) > 3:
                    categories_str += f" (+{len(categories_list) - 3} more)"

                report.write(f"- **{author_name}**: {untracked_count} commits ")
                report.write(f"({pct_of_untracked:.1f}% of untracked{dev_context}) - ")
                report.write(f"*{categories_str}*\n")
            report.write("\n")

        # Write project breakdown
        if len(projects) > 1:
            report.write("#### Projects with Untracked Work\n\n")
            sorted_projects = sorted(projects.items(), key=lambda x: x[1]["count"], reverse=True)

            for project, data in sorted_projects:
                pct = (data["count"] / total_untracked) * 100
                categories_list = list(data["categories"])
                report.write(f"- **{project}**: {data['count']} commits ({pct:.1f}%)\n")
            report.write("\n")

        # Write recent examples (configurable limit, default 15 for better visibility)
        if untracked_commits:
            report.write("#### Recent Untracked Commits\n\n")

            # Show configurable number of recent commits (increased from 10 to 15)
            max_recent_commits = 15

            # Safe timestamp sorting that handles mixed timezone types
            def safe_timestamp_key(commit):
                ts = commit.get("timestamp")
                if ts is None:
                    return datetime.min.replace(tzinfo=timezone.utc)
                # If it's a datetime object, handle timezone issues
                if hasattr(ts, "tzinfo"):
                    # Make timezone-naive datetime UTC-aware for consistent comparison
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    return ts
                # If it's a string or other type, try to parse or use as-is
                return ts

            recent_commits = sorted(untracked_commits, key=safe_timestamp_key, reverse=True)[
                :max_recent_commits
            ]

            if len(untracked_commits) > max_recent_commits:
                report.write(
                    f"*Showing {max_recent_commits} most recent of {len(untracked_commits)} untracked commits*\n\n"
                )

            for commit in recent_commits:
                # Format date
                timestamp = commit.get("timestamp")
                if timestamp and hasattr(timestamp, "strftime"):
                    date_str = timestamp.strftime("%Y-%m-%d")
                else:
                    date_str = "unknown date"

                report.write(f"- `{commit.get('hash', '')}` ({date_str}) ")
                report.write(f"**{commit.get('author', 'Unknown')}** ")
                report.write(f"[{commit.get('category', 'other')}]: ")
                report.write(f"{commit.get('message', '')[:100]}")
                if len(commit.get("message", "")) > 100:
                    report.write("...")
                report.write(f" *({commit.get('files_changed', 0)} files, ")
                report.write(f"{commit.get('lines_changed', 0)} lines)*\n")
            report.write("\n")

        # Add recommendations based on untracked analysis
        self._write_untracked_recommendations(
            report, categories, contributors, total_untracked, total_commits
        )

    def _write_untracked_recommendations(
        self,
        report: StringIO,
        categories: dict[str, Any],
        contributors: dict[str, Any],
        total_untracked: int,
        total_commits: int,
    ) -> None:
        """Write specific recommendations based on untracked commit analysis."""
        report.write("#### Recommendations for Untracked Work\n\n")

        recommendations = []

        # Category-based recommendations
        feature_count = categories.get("feature", {}).get("count", 0)
        bug_fix_count = categories.get("bug_fix", {}).get("count", 0)
        maintenance_count = categories.get("maintenance", {}).get("count", 0)
        docs_count = categories.get("documentation", {}).get("count", 0)
        style_count = categories.get("style", {}).get("count", 0)

        if feature_count > total_untracked * 0.2:
            recommendations.append(
                "🎫 **Require tickets for features**: Many feature developments lack ticket references. "
                "Consider enforcing ticket creation for new functionality."
            )

        if bug_fix_count > total_untracked * 0.15:
            recommendations.append(
                "🐛 **Track bug fixes**: Bug fixes should be linked to issue tickets for better "
                "visibility and follow-up."
            )

        # Positive recognition for appropriate untracked work
        acceptable_count = maintenance_count + docs_count + style_count
        if acceptable_count > total_untracked * 0.6:
            recommendations.append(
                "✅ **Good process balance**: Most untracked work consists of maintenance, "
                "documentation, and style improvements - this is acceptable and shows good "
                "development hygiene."
            )

        # Coverage recommendations
        untracked_pct = (total_untracked / total_commits * 100) if total_commits > 0 else 0
        if untracked_pct > 50:
            recommendations.append(
                "📈 **Improve overall tracking**: Over 50% of commits lack ticket references. "
                "Consider team training on linking commits to work items."
            )
        elif untracked_pct < 20:
            recommendations.append(
                "🎯 **Excellent tracking**: Less than 20% of commits are untracked - "
                "the team shows strong process adherence."
            )

        # Developer-specific recommendations
        if len(contributors) > 1:
            max_contributor_pct = max(
                (data["count"] / total_untracked * 100) for data in contributors.values()
            )
            if max_contributor_pct > 40:
                recommendations.append(
                    "👥 **Targeted training**: Some developers need additional guidance on "
                    "ticket referencing practices. Consider peer mentoring or process review."
                )

        if not recommendations:
            recommendations.append(
                "✅ **Balanced approach**: Untracked work appears well-balanced between "
                "necessary maintenance and tracked development work."
            )

        for rec in recommendations:
            report.write(f"{rec}\n\n")

