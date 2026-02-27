"""Developer activity, PR metrics, untracked commits, and categorization CSV reports.

Extracted from csv_writer.py to keep file sizes manageable.
Contains generate_developer_activity_summary, generate_pr_metrics_report,
generate_untracked_commits_report, and generate_weekly_categorization_report.
"""

import csv
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CSVDeveloperReportsMixin:
    """Mixin providing developer and PR CSV reports for CSVReportGenerator.

    Attributes expected from host class:
        activity_scorer, anonymize, exclude_authors, identity_resolver,
        _anonymization_map, _anonymous_counter
    """

    def generate_developer_activity_summary(
        self,
        commits: list[dict[str, Any]],
        developer_stats: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        output_path: Path,
        weeks: int = 12,
    ) -> Path:
        """Generate developer activity summary with curve-normalized scores.

        This report provides a high-level view of developer activity with
        curve-normalized scores that allow for fair comparison across the team.
        """
        # Apply exclusion filtering in Phase 2
        commits = self._filter_excluded_authors_list(commits)
        developer_stats = self._filter_excluded_authors_list(developer_stats)

        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)

        # Aggregate metrics by developer
        developer_metrics = defaultdict(
            lambda: {
                "commits": 0,
                "prs_involved": 0,
                "lines_added": 0,
                "lines_removed": 0,
                "files_changed": 0,
                "complexity_delta": 0.0,
                "story_points": 0,
                "unique_tickets": set(),
            }
        )

        # Process commits
        for commit in commits:
            timestamp = commit["timestamp"]
            if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            if timestamp < start_date or timestamp > end_date:
                continue

            dev_id = commit.get("canonical_id", commit.get("author_email", "unknown"))
            metrics = developer_metrics[dev_id]

            metrics["commits"] += 1
            metrics["lines_added"] += (
                commit.get("filtered_insertions", commit.get("insertions", 0)) or 0
            )
            metrics["lines_removed"] += (
                commit.get("filtered_deletions", commit.get("deletions", 0)) or 0
            )
            metrics["files_changed"] += (
                commit.get("filtered_files_changed", commit.get("files_changed", 0)) or 0
            )
            metrics["complexity_delta"] += commit.get("complexity_delta", 0.0) or 0.0
            metrics["story_points"] += commit.get("story_points", 0) or 0

            ticket_refs = commit.get("ticket_references", [])
            for ticket in ticket_refs:
                if isinstance(ticket, dict):
                    metrics["unique_tickets"].add(ticket.get("full_id", ""))
                else:
                    metrics["unique_tickets"].add(str(ticket))

        # Process PRs — basic count plus per-author review aggregation
        # Per-developer PR review stats (populated when fetch_pr_reviews=true)
        # Gap 3: also track prs_reviewed, prs_commented, prs_merged

        # Create lookup for identity resolution (Gap 1)
        dev_lookup = {dev["canonical_id"]: dev for dev in developer_stats}

        dev_pr_review: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "prs_authored": 0,
                "prs_merged": 0,  # Gap 3: subset of authored that were merged
                "prs_closed": 0,  # Gap 2: authored PRs closed without merge
                "prs_reviewed": 0,  # Gap 3: PRs where dev left an approval/change-request
                "prs_commented": 0,  # Gap 3: PRs where dev left comments (any review type)
                "total_approvals": 0,
                "total_change_requests": 0,
                "total_review_comments": 0,
                "total_pr_comments": 0,
                "total_revisions": 0,
                "ttfr_values": [],  # time-to-first-review samples
            }
        )

        for pr in prs:
            author_id = pr.get("canonical_id", pr.get("author", "unknown"))

            # Gap 1: bridge GitHub username → canonical identity using identity resolver
            if self.identity_resolver and author_id not in dev_lookup:
                # author_id may be a raw GitHub username — try to resolve it
                github_username = pr.get("author", "")
                if github_username:
                    resolved = self.identity_resolver.resolve_by_github_username(github_username)
                    if resolved:
                        author_id = resolved

            if author_id in developer_metrics:
                developer_metrics[author_id]["prs_involved"] += 1

            # Collect enhanced review stats keyed by author
            rev = dev_pr_review[author_id]
            rev["prs_authored"] += 1

            # Gap 3: track merged vs closed
            if pr.get("is_merged") or pr.get("pr_state") == "merged":
                rev["prs_merged"] += 1
            elif pr.get("pr_state") == "closed":
                rev["prs_closed"] += 1

            rev["total_approvals"] += pr.get("approvals_count", 0) or 0
            rev["total_change_requests"] += pr.get("change_requests_count", 0) or 0
            rev["total_review_comments"] += pr.get("review_comments", 0) or 0
            rev["total_pr_comments"] += pr.get("pr_comments_count", 0) or 0
            rev["total_revisions"] += pr.get("revision_count", 0) or 0
            ttfr = pr.get("time_to_first_review_hours")
            if ttfr is not None:
                rev["ttfr_values"].append(ttfr)

            # Gap 3: attribute reviewer/commenter counts to the reviewing developers
            # reviewers list contains GitHub logins of anyone who reviewed this PR
            for reviewer_login in pr.get("reviewers", []):
                reviewer_login_lower = reviewer_login.lower() if reviewer_login else ""
                if not reviewer_login_lower:
                    continue

                # Gap 1: resolve GitHub login → canonical identity
                reviewer_id: Optional[str] = None
                if self.identity_resolver:
                    reviewer_id = self.identity_resolver.resolve_by_github_username(
                        reviewer_login_lower
                    )
                # Fall back to using the login itself as the key
                if not reviewer_id:
                    reviewer_id = reviewer_login_lower

                dev_pr_review[reviewer_id]["prs_commented"] += 1

            # Gap 3: approved_by → increment prs_reviewed for each approver
            for approver_login in pr.get("approved_by", []):
                approver_login_lower = approver_login.lower() if approver_login else ""
                if not approver_login_lower:
                    continue

                approver_id: Optional[str] = None
                if self.identity_resolver:
                    approver_id = self.identity_resolver.resolve_by_github_username(
                        approver_login_lower
                    )
                if not approver_id:
                    approver_id = approver_login_lower

                dev_pr_review[approver_id]["prs_reviewed"] += 1

        # Calculate activity scores
        developer_scores = {}
        developer_results = {}

        for dev_id, metrics in developer_metrics.items():
            # Convert sets to counts
            metrics["unique_tickets"] = len(metrics["unique_tickets"])

            # Calculate activity score
            activity_result = self.activity_scorer.calculate_activity_score(metrics)
            developer_scores[dev_id] = activity_result["raw_score"]
            developer_results[dev_id] = activity_result

        # Apply curve normalization
        curve_normalized = self.activity_scorer.normalize_scores_on_curve(developer_scores)

        # Create developer lookup
        dev_lookup = {dev["canonical_id"]: dev for dev in developer_stats}

        # Build rows
        rows = []
        for dev_id, metrics in developer_metrics.items():
            developer = dev_lookup.get(dev_id, {})
            activity_result = developer_results[dev_id]
            curve_data = curve_normalized.get(dev_id, {})
            pr_rev = dev_pr_review.get(dev_id, {})

            # Per-developer review aggregation
            prs_authored = pr_rev.get("prs_authored", 0)
            ttfr_vals = pr_rev.get("ttfr_values", [])
            avg_ttfr = sum(ttfr_vals) / len(ttfr_vals) if ttfr_vals else None
            avg_approvals = (
                pr_rev.get("total_approvals", 0) / prs_authored if prs_authored else None
            )
            avg_cr = pr_rev.get("total_change_requests", 0) / prs_authored if prs_authored else None
            avg_revisions = (
                pr_rev.get("total_revisions", 0) / prs_authored if prs_authored else None
            )

            row = {
                "developer_id": self._anonymize_value(dev_id, "id"),
                "developer_name": self._anonymize_value(
                    self._get_canonical_display_name(
                        dev_id, developer.get("primary_name", "Unknown")
                    ),
                    "name",
                ),
                "commits": metrics["commits"],
                "prs": metrics["prs_involved"],
                "story_points": metrics["story_points"],
                "lines_added": metrics["lines_added"],
                "lines_removed": metrics["lines_removed"],
                "files_changed": metrics["files_changed"],
                "unique_tickets": metrics["unique_tickets"],
                # Gap 3: per-developer PR role breakdown
                "prs_authored": pr_rev.get("prs_authored", 0),
                "prs_merged": pr_rev.get("prs_merged", 0),
                "prs_closed_without_merge": pr_rev.get("prs_closed", 0),
                "prs_reviewed": pr_rev.get("prs_reviewed", 0),
                "prs_commented": pr_rev.get("prs_commented", 0),
                # PR review stats (empty string when review data not collected)
                "pr_review_comments": pr_rev.get("total_review_comments", "") or "",
                "pr_general_comments": pr_rev.get("total_pr_comments", "") or "",
                "pr_approvals_received": pr_rev.get("total_approvals", "") or "",
                "pr_change_requests_received": pr_rev.get("total_change_requests", "") or "",
                "avg_approvals_per_pr": (
                    round(avg_approvals, 2) if avg_approvals is not None else ""
                ),
                "avg_change_requests_per_pr": (round(avg_cr, 2) if avg_cr is not None else ""),
                "avg_revisions_per_pr": (
                    round(avg_revisions, 2) if avg_revisions is not None else ""
                ),
                "avg_time_to_first_review_hours": (
                    round(avg_ttfr, 2) if avg_ttfr is not None else ""
                ),
                # Raw activity scores
                "raw_activity_score": round(activity_result["raw_score"], 1),
                "normalized_activity_score": round(activity_result["normalized_score"], 1),
                "activity_level": activity_result["activity_level"],
                # Curve-normalized scores
                "curved_score": curve_data.get("curved_score", 0),
                "percentile": curve_data.get("percentile", 0),
                "quintile": curve_data.get("quintile", 0),
                "curved_activity_level": curve_data.get("activity_level", "unknown"),
                "level_description": curve_data.get("level_description", ""),
                # Component breakdown
                "commit_score": round(activity_result["components"]["commit_score"], 1),
                "pr_score": round(activity_result["components"]["pr_score"], 1),
                "code_impact_score": round(activity_result["components"]["code_impact_score"], 1),
                "complexity_score": round(activity_result["components"]["complexity_score"], 1),
            }
            rows.append(row)

        # Sort by curved score (highest first)
        rows.sort(key=lambda x: x["curved_score"], reverse=True)

        # Write CSV
        _fieldnames = [
            "developer_id",
            "developer_name",
            "commits",
            "prs",
            "story_points",
            "lines_added",
            "lines_removed",
            "files_changed",
            "unique_tickets",
            # Gap 3: per-developer PR role breakdown
            "prs_authored",
            "prs_merged",
            "prs_closed_without_merge",
            "prs_reviewed",
            "prs_commented",
            "pr_review_comments",
            "pr_general_comments",
            "pr_approvals_received",
            "pr_change_requests_received",
            "avg_approvals_per_pr",
            "avg_change_requests_per_pr",
            "avg_revisions_per_pr",
            "avg_time_to_first_review_hours",
            "raw_activity_score",
            "normalized_activity_score",
            "activity_level",
            "curved_score",
            "percentile",
            "quintile",
            "curved_activity_level",
            "level_description",
            "commit_score",
            "pr_score",
            "code_impact_score",
            "complexity_score",
        ]

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        else:
            # Write empty CSV with headers
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_fieldnames)
                writer.writeheader()

        return output_path

    def _anonymize_value(self, value: str, field_type: str) -> str:
        """Anonymize a value if anonymization is enabled."""
        if not self.anonymize or not value:
            return value

        if field_type == "email" and "@" in value:
            # Keep domain for email
            local, domain = value.split("@", 1)
            value = local  # Anonymize only local part
            suffix = f"@{domain}"
        else:
            suffix = ""

        if value not in self._anonymization_map:
            self._anonymous_counter += 1
            if field_type == "name":
                anonymous = f"Developer{self._anonymous_counter}"
            elif field_type == "email":
                anonymous = f"dev{self._anonymous_counter}"
            elif field_type == "id":
                anonymous = f"ID{self._anonymous_counter:04d}"
            else:
                anonymous = f"anon{self._anonymous_counter}"

            self._anonymization_map[value] = anonymous

        return self._anonymization_map[value] + suffix

    def generate_pr_metrics_report(
        self,
        prs: list[dict[str, Any]],
        output_path: Path,
    ) -> Path:
        """Generate a PR-level detailed CSV report with all available review metrics.

        Each row represents one pull request.  Columns for review-level fields
        (approvals_count, change_requests_count, time_to_first_review_hours,
        revision_count, pr_comments_count) are populated only when the GitHub
        integration was run with ``fetch_pr_reviews=true``; otherwise they are
        left empty so the report is still valid without review data.

        Args:
            prs: List of pull request data dictionaries as returned by
                ``GitHubIntegration.calculate_pr_metrics()`` or from cache.
            output_path: Destination CSV path.

        Returns:
            Path to the written CSV file.
        """
        rows = []
        for pr in prs:
            created_at = pr.get("created_at")
            merged_at = pr.get("merged_at")

            # Lifetime in hours — only when both timestamps are available
            lifetime_hours: str | float = ""
            if created_at and merged_at:
                try:
                    if hasattr(created_at, "total_seconds"):
                        # already a timedelta
                        lifetime_hours = round(created_at.total_seconds() / 3600, 2)
                    else:
                        delta = merged_at - created_at
                        lifetime_hours = round(delta.total_seconds() / 3600, 2)
                except Exception:
                    lifetime_hours = ""

            closed_at = pr.get("closed_at")
            is_merged_val = pr.get("is_merged")
            pr_state_val = pr.get("pr_state", "")

            row: dict[str, Any] = {
                "pr_number": pr.get("number", ""),
                "title": pr.get("title", ""),
                "author": self._anonymize_value(pr.get("author", ""), "name"),
                "created_at": (
                    self._safe_datetime_format(created_at, "%Y-%m-%d %H:%M:%S")
                    if created_at
                    else ""
                ),
                "merged_at": (
                    self._safe_datetime_format(merged_at, "%Y-%m-%d %H:%M:%S") if merged_at else ""
                ),
                # v4.0 state columns
                "pr_state": pr_state_val or "",
                "closed_at": (
                    self._safe_datetime_format(closed_at, "%Y-%m-%d %H:%M:%S") if closed_at else ""
                ),
                "is_merged": (
                    str(bool(is_merged_val)).lower() if is_merged_val is not None else ""
                ),
                "lifetime_hours": lifetime_hours,
                # Size
                "additions": pr.get("additions", 0) or 0,
                "deletions": pr.get("deletions", 0) or 0,
                "changed_files": pr.get("changed_files", 0) or 0,
                # Inline review comments (always present from GitHub base PR object)
                "review_comments": pr.get("review_comments", 0) or 0,
                # Story points
                "story_points": pr.get("story_points", 0) or 0,
                # Enhanced review fields (empty when fetch_pr_reviews was disabled)
                "approvals_count": pr.get("approvals_count", "")
                if pr.get("approvals_count") is not None
                else "",
                "change_requests_count": pr.get("change_requests_count", "")
                if pr.get("change_requests_count") is not None
                else "",
                "pr_comments_count": pr.get("pr_comments_count", "")
                if pr.get("pr_comments_count") is not None
                else "",
                "time_to_first_review_hours": (
                    round(pr["time_to_first_review_hours"], 2)
                    if pr.get("time_to_first_review_hours") is not None
                    else ""
                ),
                "revision_count": pr.get("revision_count", "")
                if pr.get("revision_count") is not None
                else "",
                "reviewers": ",".join(pr.get("reviewers") or []),
                "approved_by": ",".join(pr.get("approved_by") or []),
                # Labels
                "labels": ",".join(pr.get("labels") or []),
            }
            rows.append(row)

        # Sort by merged_at descending (most recent first), fallback to PR number
        rows.sort(
            key=lambda r: (r["merged_at"] or "", r["pr_number"] or 0),
            reverse=True,
        )

        _fieldnames = [
            "pr_number",
            "title",
            "author",
            "created_at",
            "merged_at",
            # v4.0 state columns
            "pr_state",
            "closed_at",
            "is_merged",
            "lifetime_hours",
            "additions",
            "deletions",
            "changed_files",
            "review_comments",
            "story_points",
            "approvals_count",
            "change_requests_count",
            "pr_comments_count",
            "time_to_first_review_hours",
            "revision_count",
            "reviewers",
            "approved_by",
            "labels",
        ]

        if rows:
            df = pd.DataFrame(rows)
            # Ensure consistent column ordering
            df = df.reindex(columns=_fieldnames)
            df.to_csv(output_path, index=False)
        else:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_fieldnames)
                writer.writeheader()

        return output_path

    def generate_untracked_commits_report(
        self, ticket_analysis: dict[str, Any], output_path: Path
    ) -> Path:
        """Generate detailed CSV report for commits without ticket references.

        WHY: Untracked commits represent work that may not be visible to project
        management tools. This report enables analysis of what types of work are
        being performed outside the tracked process, helping identify process
        improvements and training needs.

        Args:
            ticket_analysis: Ticket analysis results containing untracked commits
            output_path: Path where the CSV report should be written

        Returns:
            Path to the generated CSV file
        """
        untracked_commits = ticket_analysis.get("untracked_commits", [])

        if not untracked_commits:
            # Generate empty report with headers
            headers = [
                "commit_hash",
                "short_hash",
                "author",
                "author_email",
                "canonical_id",
                "date",
                "project",
                "message",
                "category",
                "files_changed",
                "lines_added",
                "lines_removed",
                "lines_changed",
                "is_merge",
            ]
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
            return output_path

        # Process untracked commits into CSV rows
        rows = []
        for commit in untracked_commits:
            # Handle datetime formatting
            timestamp = commit.get("timestamp")
            if timestamp:
                if hasattr(timestamp, "strftime"):
                    date_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    date_str = str(timestamp)
            else:
                date_str = ""

            row = {
                "commit_hash": commit.get("full_hash", commit.get("hash", "")),
                "short_hash": commit.get("hash", ""),
                "author": self._anonymize_value(commit.get("author", "Unknown"), "name"),
                "author_email": self._anonymize_value(commit.get("author_email", ""), "email"),
                "canonical_id": self._anonymize_value(commit.get("canonical_id", ""), "id"),
                "date": date_str,
                "project": commit.get("project_key", "UNKNOWN"),
                "message": commit.get("message", ""),
                "category": commit.get("category", "other"),
                "files_changed": commit.get("files_changed", 0),
                "lines_added": commit.get("lines_added", 0),
                "lines_removed": commit.get("lines_removed", 0),
                "lines_changed": commit.get("lines_changed", 0),
                "is_merge": commit.get("is_merge", False),
            }
            rows.append(row)

        # Write CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)

        return output_path

    def generate_weekly_categorization_report(
        self,
        all_commits: list[dict[str, Any]],
        ticket_extractor,  # TicketExtractor or MLTicketExtractor instance
        output_path: Path,
        weeks: int = 12,
    ) -> Path:
        """Generate weekly commit categorization metrics CSV report for ALL commits.

        WHY: Categorization trends provide insights into development patterns
        over time, helping identify process improvements and training needs.
        This enhanced version processes ALL commits (tracked and untracked) to provide
        complete visibility into work patterns across the entire development flow.

        DESIGN DECISION: Processes all commits using the same ML/rule-based categorization
        system used elsewhere in the application, ensuring consistent categorization
        across all reports and analysis.

        Args:
            all_commits: Complete list of commits to categorize
            ticket_extractor: TicketExtractor instance for commit categorization
            output_path: Path where the CSV report should be written
            weeks: Number of weeks to analyze

        Returns:
            Path to the generated CSV file
        """
        # Calculate week boundaries
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks)

        # Initialize weekly aggregation structures
        weekly_categories = defaultdict(lambda: defaultdict(int))
        weekly_metrics = defaultdict(
            lambda: {"lines_added": 0, "lines_removed": 0, "files_changed": 0, "developers": set()}
        )

        # Process ALL commits with classification
        processed_commits = 0
        for commit in all_commits:
            if not isinstance(commit, dict):
                continue

            # Get timestamp and validate date range
            timestamp = commit.get("timestamp")
            if not timestamp:
                continue

            # Ensure timezone consistency
            if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            elif hasattr(timestamp, "tzinfo") and timestamp.tzinfo != timezone.utc:
                timestamp = timestamp.astimezone(timezone.utc)

            if timestamp < start_date or timestamp > end_date:
                continue

            # Skip merge commits (consistent with untracked analysis)
            if commit.get("is_merge", False):
                continue

            # Categorize the commit using the same system as untracked analysis
            message = commit.get("message", "")
            files_changed_raw = commit.get("files_changed", [])

            # Handle both int and list types for files_changed
            if isinstance(files_changed_raw, int):
                files_changed_count = files_changed_raw
                files_changed_list = []  # Can't provide file names, only count
            elif isinstance(files_changed_raw, list):
                files_changed_count = len(files_changed_raw)
                files_changed_list = files_changed_raw
            else:
                files_changed_count = 0
                files_changed_list = []

            # Handle both TicketExtractor and MLTicketExtractor signatures
            try:
                # Try ML signature first (message, files_changed as list)
                category = ticket_extractor.categorize_commit(message, files_changed_list)
            except TypeError:
                # Fall back to base signature (message only)
                category = ticket_extractor.categorize_commit(message)

            # Get week boundary (Monday start)
            week_start = self._get_week_start(timestamp)

            # Aggregate by category
            weekly_categories[week_start][category] += 1

            # Aggregate metrics
            weekly_metrics[week_start]["lines_added"] += commit.get("insertions", 0)
            weekly_metrics[week_start]["lines_removed"] += commit.get("deletions", 0)
            weekly_metrics[week_start]["files_changed"] += files_changed_count

            # Track unique developers (use canonical_id or fallback to email)
            developer_id = commit.get("canonical_id") or commit.get("author_email", "Unknown")
            weekly_metrics[week_start]["developers"].add(developer_id)

            processed_commits += 1

        # Build CSV rows with comprehensive metrics
        rows = []
        all_categories = set()

        # Collect all categories across all weeks
        for week_data in weekly_categories.values():
            all_categories.update(week_data.keys())

        # Ensure standard categories are included even if not found
        standard_categories = [
            "bug_fix",
            "feature",
            "refactor",
            "documentation",
            "maintenance",
            "test",
            "style",
            "build",
            "integration",
            "other",
        ]
        all_categories.update(standard_categories)
        sorted_categories = sorted(all_categories)

        # Generate weekly rows
        for week_start in sorted(weekly_categories.keys()):
            week_data = weekly_categories[week_start]
            week_metrics = weekly_metrics[week_start]
            total_commits = sum(week_data.values())

            row = {
                "week_start": week_start.strftime("%Y-%m-%d"),
                "total_commits": total_commits,
                "lines_added": week_metrics["lines_added"],
                "lines_removed": week_metrics["lines_removed"],
                "files_changed": week_metrics["files_changed"],
                "developer_count": len(week_metrics["developers"]),
            }

            # Add each category count and percentage
            for category in sorted_categories:
                count = week_data.get(category, 0)
                pct = (count / total_commits * 100) if total_commits > 0 else 0

                row[f"{category}_count"] = count
                row[f"{category}_pct"] = round(pct, 1)

            rows.append(row)

        # Write CSV with comprehensive headers
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        else:
            # Write empty CSV with comprehensive headers
            headers = [
                "week_start",
                "total_commits",
                "lines_added",
                "lines_removed",
                "files_changed",
                "developer_count",
            ]

            for category in sorted_categories:
                headers.extend([f"{category}_count", f"{category}_pct"])

            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

        return output_path
