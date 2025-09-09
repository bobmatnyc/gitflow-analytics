"""Results screen for GitFlow Analytics TUI."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.table import Table
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, Rule, Static, TabbedContent, TabPane

from gitflow_analytics.config import Config

from ..widgets.data_table import EnhancedDataTable
from ..widgets.export_modal import ExportModal


class ResultsScreen(Screen):
    """
    Screen displaying comprehensive analysis results with interactive exploration.

    WHY: Analysis results are complex and multi-dimensional, requiring an
    interactive interface that allows users to explore different aspects of
    the data. Tabbed layout organizes information logically while providing
    powerful data exploration capabilities.

    DESIGN DECISION: Uses tabbed interface to separate different result categories
    while providing consistent export functionality across all views. Interactive
    tables allow users to sort, filter, and drill down into specific data points.
    """

    BINDINGS = [
        Binding("escape", "back", "Back to Main"),
        Binding("ctrl+s", "export", "Export Results"),
        Binding("ctrl+f", "filter", "Filter Data"),
        Binding("r", "refresh", "Refresh View"),
        Binding("ctrl+e", "export_current", "Export Current View"),
    ]

    def __init__(
        self,
        commits: list[dict],
        prs: list[dict],
        developers: list[dict],
        config: Config,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, id=id)
        self.commits = commits
        self.prs = prs
        self.developers = developers
        self.config = config
        self.current_tab = "summary"

    def compose(self):
        """Compose the results screen."""
        yield Header()

        with Container(id="results-container"):
            yield Label("GitFlow Analytics - Results", classes="screen-title")

            with TabbedContent(initial="summary"):
                # Summary Tab
                with TabPane("Summary", id="summary"):
                    yield self._create_summary_panel()

                # Developers Tab
                with TabPane("Developers", id="developers"):
                    yield self._create_developers_panel()

                # Commits Tab
                with TabPane("Commits", id="commits"):
                    yield self._create_commits_panel()

                # Pull Requests Tab (if available)
                if self.prs:
                    with TabPane("Pull Requests", id="pull-requests"):
                        yield self._create_prs_panel()

                # Qualitative Insights Tab (if available)
                if self._has_qualitative_data():
                    with TabPane("Qualitative Insights", id="qualitative"):
                        yield self._create_qualitative_panel()

                # Export Tab
                with TabPane("Export", id="export"):
                    yield self._create_export_panel()

        yield Footer()

    def _create_summary_panel(self) -> ScrollableContainer:
        """
        Create comprehensive summary statistics panel.

        WHY: Provides high-level overview of all analysis results in a single view,
        allowing users to quickly understand the overall scope and key metrics
        without diving into detailed data tables.
        """
        container = ScrollableContainer()

        # Key metrics section
        container.mount(Label("Analysis Summary", classes="section-title"))

        # Create summary table
        summary_table = Table(show_header=False, show_edge=False, pad_edge=False)
        summary_table.add_column("Metric", style="bold cyan", width=25)
        summary_table.add_column("Value", style="green", width=15)
        summary_table.add_column("Details", style="dim", width=40)

        # Calculate key metrics
        total_commits = len(self.commits)
        total_prs = len(self.prs)
        total_developers = len(self.developers)

        # Time range
        if self.commits:
            dates = [c.get("timestamp") for c in self.commits if c.get("timestamp")]
            if dates:
                min_date = min(dates).strftime("%Y-%m-%d")
                max_date = max(dates).strftime("%Y-%m-%d")
                date_range = f"{min_date} to {max_date}"
            else:
                date_range = "Unknown"
        else:
            date_range = "No data"

        # Story points
        total_story_points = sum(c.get("story_points", 0) or 0 for c in self.commits)

        # Ticket coverage
        commits_with_tickets = sum(1 for c in self.commits if c.get("ticket_references"))
        ticket_coverage = (commits_with_tickets / total_commits * 100) if total_commits > 0 else 0

        # Add metrics to table
        summary_table.add_row(
            "Total Commits", f"{total_commits:,}", "All commits in analysis period"
        )
        summary_table.add_row("Total Pull Requests", f"{total_prs:,}", "Detected pull requests")
        summary_table.add_row(
            "Active Developers", f"{total_developers:,}", "Unique developer identities"
        )
        summary_table.add_row("Analysis Period", date_range, "Date range of analyzed commits")
        summary_table.add_row(
            "Story Points", f"{total_story_points:,}", "Total story points completed"
        )
        summary_table.add_row(
            "Ticket Coverage", f"{ticket_coverage:.1f}%", "Commits with ticket references"
        )

        from rich.console import Console
        from rich.panel import Panel

        Console()

        container.mount(Static(Panel(summary_table, title="Key Metrics", border_style="blue")))

        # Top contributors section
        container.mount(Rule())
        container.mount(Label("Top Contributors", classes="section-title"))

        if self.developers:
            top_devs = sorted(
                self.developers, key=lambda d: d.get("total_commits", 0), reverse=True
            )[:10]

            contrib_table = Table(show_header=True, header_style="bold magenta")
            contrib_table.add_column("Developer", width=25)
            contrib_table.add_column("Commits", justify="right", width=10)
            contrib_table.add_column("Story Points", justify="right", width=12)
            contrib_table.add_column("Avg Points/Commit", justify="right", width=15)

            for dev in top_devs:
                commits = dev.get("total_commits", 0)
                points = dev.get("total_story_points", 0)
                avg_points = points / commits if commits > 0 else 0

                contrib_table.add_row(
                    dev.get("primary_name", "Unknown")[:23],
                    f"{commits:,}",
                    f"{points:,}",
                    f"{avg_points:.1f}",
                )

            container.mount(
                Static(Panel(contrib_table, title="Developer Activity", border_style="green"))
            )

        # Qualitative insights summary (if available)
        if self._has_qualitative_data():
            container.mount(Rule())
            container.mount(Label("Qualitative Analysis Summary", classes="section-title"))
            container.mount(Static(self._create_qualitative_summary()))

        return container

    def _create_developers_panel(self) -> Container:
        """Create interactive developers data panel."""
        container = Container()

        container.mount(Label("Developer Statistics", classes="section-title"))
        container.mount(
            Static(
                f"Showing {len(self.developers)} unique developers. Click column headers to sort.",
                classes="help-text",
            )
        )

        # Create enhanced data table
        developers_table = EnhancedDataTable(data=self.developers, id="developers-table")

        container.mount(developers_table)

        # Action buttons
        with container.mount(Horizontal(classes="action-bar")):
            yield Button("Export Developers", id="export-developers")
            yield Button("Show Identity Details", id="show-identities")

        return container

    def _create_commits_panel(self) -> Container:
        """Create interactive commits data panel."""
        container = Container()

        container.mount(Label("Commit Analysis", classes="section-title"))
        container.mount(
            Static(
                f"Showing {len(self.commits)} commits. Use filters to explore specific data.",
                classes="help-text",
            )
        )

        # Prepare commits data for table display
        commits_data = []
        for commit in self.commits[:1000]:  # Limit to 1000 for performance
            commit_row = {
                "date": (
                    commit.get("timestamp", "").strftime("%Y-%m-%d")
                    if commit.get("timestamp")
                    else ""
                ),
                "author": commit.get("author_name", ""),
                "message": (
                    commit.get("message", "")[:80] + "..."
                    if len(commit.get("message", "")) > 80
                    else commit.get("message", "")
                ),
                "files_changed": commit.get("files_changed_count", 0),
                "insertions": commit.get("insertions", 0),
                "deletions": commit.get("deletions", 0),
                "story_points": commit.get("story_points", 0),
                "project_key": commit.get("project_key", ""),
                "change_type": commit.get("change_type", "unknown"),
                "risk_level": commit.get("risk_level", "unknown"),
            }
            commits_data.append(commit_row)

        commits_table = EnhancedDataTable(data=commits_data, id="commits-table")

        container.mount(commits_table)

        # Action buttons
        with container.mount(Horizontal(classes="action-bar")):
            yield Button("Export Commits", id="export-commits")
            yield Button("Filter by Author", id="filter-author")
            yield Button("Filter by Project", id="filter-project")

        return container

    def _create_prs_panel(self) -> Container:
        """Create pull requests analysis panel."""
        container = Container()

        container.mount(Label("Pull Request Analysis", classes="section-title"))
        container.mount(
            Static(
                f"Showing {len(self.prs)} pull requests with metrics and timing data.",
                classes="help-text",
            )
        )

        # Prepare PR data for table
        prs_data = []
        for pr in self.prs:
            pr_row = {
                "title": (
                    pr.get("title", "")[:60] + "..."
                    if len(pr.get("title", "")) > 60
                    else pr.get("title", "")
                ),
                "author": pr.get("author", ""),
                "state": pr.get("state", ""),
                "created_date": (
                    pr.get("created_at", "").strftime("%Y-%m-%d") if pr.get("created_at") else ""
                ),
                "merged_date": (
                    pr.get("merged_at", "").strftime("%Y-%m-%d") if pr.get("merged_at") else ""
                ),
                "commits": pr.get("commits_count", 0),
                "changed_files": pr.get("changed_files", 0),
                "additions": pr.get("additions", 0),
                "deletions": pr.get("deletions", 0),
            }
            prs_data.append(pr_row)

        prs_table = EnhancedDataTable(data=prs_data, id="prs-table")

        container.mount(prs_table)

        # Action buttons
        with container.mount(Horizontal(classes="action-bar")):
            yield Button("Export PRs", id="export-prs")
            yield Button("Show PR Metrics", id="show-pr-metrics")

        return container

    def _create_qualitative_panel(self) -> ScrollableContainer:
        """Create qualitative insights panel."""
        container = ScrollableContainer()

        container.mount(Label("Qualitative Analysis Results", classes="section-title"))

        if not self._has_qualitative_data():
            container.mount(
                Static("No qualitative analysis data available.", classes="info-message")
            )
            container.mount(
                Static("Run analysis with qualitative processing enabled to see insights here.")
            )
            return container

        # Analyze qualitative data distributions
        change_types = {}
        risk_levels = {}
        domains = {}
        confidence_scores = []

        for commit in self.commits:
            if "change_type" in commit:
                change_type = commit.get("change_type", "unknown")
                change_types[change_type] = change_types.get(change_type, 0) + 1

                risk_level = commit.get("risk_level", "unknown")
                risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1

                domain = commit.get("business_domain", "unknown")
                domains[domain] = domains.get(domain, 0) + 1

                if "confidence_score" in commit:
                    confidence_scores.append(commit["confidence_score"])

        # Change types distribution
        container.mount(Label("Change Type Distribution", classes="subsection-title"))

        change_table = Table(show_header=True, header_style="bold cyan")
        change_table.add_column("Change Type", width=20)
        change_table.add_column("Count", justify="right", width=10)
        change_table.add_column("Percentage", justify="right", width=12)

        total_commits = len(self.commits)
        for change_type, count in sorted(change_types.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_commits) * 100
            change_table.add_row(change_type.title(), f"{count:,}", f"{pct:.1f}%")

        from rich.panel import Panel

        container.mount(Static(Panel(change_table, title="Change Types", border_style="cyan")))

        # Risk levels distribution
        container.mount(Rule())
        container.mount(Label("Risk Level Distribution", classes="subsection-title"))

        risk_table = Table(show_header=True, header_style="bold red")
        risk_table.add_column("Risk Level", width=20)
        risk_table.add_column("Count", justify="right", width=10)
        risk_table.add_column("Percentage", justify="right", width=12)

        for risk_level, count in sorted(risk_levels.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_commits) * 100
            risk_table.add_row(risk_level.title(), f"{count:,}", f"{pct:.1f}%")

        container.mount(Static(Panel(risk_table, title="Risk Levels", border_style="red")))

        # Business domains
        container.mount(Rule())
        container.mount(Label("Business Domain Activity", classes="subsection-title"))

        domain_table = Table(show_header=True, header_style="bold green")
        domain_table.add_column("Business Domain", width=25)
        domain_table.add_column("Count", justify="right", width=10)
        domain_table.add_column("Percentage", justify="right", width=12)

        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_commits) * 100
            domain_table.add_row(domain.title(), f"{count:,}", f"{pct:.1f}%")

        container.mount(Static(Panel(domain_table, title="Business Domains", border_style="green")))

        # Confidence score statistics
        if confidence_scores:
            container.mount(Rule())
            container.mount(Label("Analysis Confidence", classes="subsection-title"))

            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            min_confidence = min(confidence_scores)
            max_confidence = max(confidence_scores)

            confidence_text = f"""Average Confidence: {avg_confidence:.2f}
Minimum Confidence: {min_confidence:.2f}
Maximum Confidence: {max_confidence:.2f}
Total Analyzed: {len(confidence_scores):,} commits"""

            container.mount(
                Static(Panel(confidence_text, title="Confidence Statistics", border_style="yellow"))
            )

        return container

    def _create_export_panel(self) -> Container:
        """Create export options panel."""
        container = Container()

        container.mount(Label("Export Analysis Results", classes="section-title"))
        container.mount(
            Static(
                "Export your analysis results in various formats for further analysis or reporting.",
                classes="help-text",
            )
        )

        # Export options
        with container.mount(Vertical(id="export-options")):
            yield Button(
                "📄 Export Summary Report (CSV)", variant="primary", id="export-summary-csv"
            )
            yield Button("👥 Export Developer Statistics (CSV)", id="export-developers-csv")
            yield Button("📝 Export Commit Details (CSV)", id="export-commits-csv")

            if self.prs:
                yield Button("🔀 Export Pull Requests (CSV)", id="export-prs-csv")

            if self._has_qualitative_data():
                yield Button("🧠 Export Qualitative Insights (CSV)", id="export-qualitative-csv")

            yield Rule()
            yield Button("📊 Export Complete Dataset (JSON)", id="export-json")
            yield Button("📋 Generate Markdown Report", id="export-markdown")

        # Export status
        container.mount(Rule())
        container.mount(Static("", id="export-status"))

        return container

    def _has_qualitative_data(self) -> bool:
        """Check if qualitative analysis data is available."""
        return any("change_type" in commit for commit in self.commits)

    def _create_qualitative_summary(self) -> str:
        """Create a text summary of qualitative insights."""
        if not self._has_qualitative_data():
            return "No qualitative data available"

        # Count change types and risk levels
        change_types = {}
        risk_levels = {}

        for commit in self.commits:
            if "change_type" in commit:
                change_type = commit.get("change_type", "unknown")
                change_types[change_type] = change_types.get(change_type, 0) + 1

                risk_level = commit.get("risk_level", "unknown")
                risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1

        # Find most common values
        top_change_type = (
            max(change_types.items(), key=lambda x: x[1]) if change_types else ("unknown", 0)
        )
        top_risk_level = (
            max(risk_levels.items(), key=lambda x: x[1]) if risk_levels else ("unknown", 0)
        )

        total_analyzed = sum(change_types.values())

        return f"""Qualitative Analysis Summary:
• Total commits analyzed: {total_analyzed:,}
• Most common change type: {top_change_type[0]} ({top_change_type[1]} commits)
• Most common risk level: {top_risk_level[0]} ({top_risk_level[1]} commits)
• Coverage: {(total_analyzed/len(self.commits)*100):.1f}% of all commits"""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        export_actions = {
            "export-summary-csv": lambda: self._export_data("summary", "csv"),
            "export-developers-csv": lambda: self._export_data("developers", "csv"),
            "export-commits-csv": lambda: self._export_data("commits", "csv"),
            "export-prs-csv": lambda: self._export_data("prs", "csv"),
            "export-qualitative-csv": lambda: self._export_data("qualitative", "csv"),
            "export-json": lambda: self._export_data("complete", "json"),
            "export-markdown": lambda: self._export_data("report", "markdown"),
        }

        action = export_actions.get(event.button.id)
        if action:
            action()
        else:
            # Handle other button actions
            if event.button.id == "show-identities":
                self._show_identity_details()
            elif event.button.id == "show-pr-metrics":
                self._show_pr_metrics()

    def _export_data(self, data_type: str, format_type: str) -> None:
        """
        Export specific data type in specified format.

        WHY: Provides flexible export functionality that allows users to
        export exactly the data they need in their preferred format.
        """
        try:
            # Determine data and filename based on type
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if data_type == "summary":
                data = self._prepare_summary_data()
            elif data_type == "developers":
                data = self.developers
            elif data_type == "commits":
                data = self.commits
            elif data_type == "prs":
                data = self.prs
            elif data_type == "qualitative":
                data = self._prepare_qualitative_data()
            elif data_type == "complete":
                data = {
                    "commits": self.commits,
                    "prs": self.prs,
                    "developers": self.developers,
                    "config": self.config.__dict__ if hasattr(self.config, "__dict__") else {},
                }
            else:
                self.notify("Unknown export type", severity="error")
                return

            # Show export modal
            export_modal = ExportModal(
                available_formats=[format_type.upper()],
                default_path=Path("./reports"),
                data_info={
                    "type": data_type,
                    "row_count": len(data) if isinstance(data, list) else "N/A",
                    "timestamp": timestamp,
                },
            )

            def handle_export(config):
                if config:
                    self._perform_export(data, config, format_type)

            self.app.push_screen(export_modal, handle_export)

        except Exception as e:
            self.notify(f"Export preparation failed: {e}", severity="error")

    def _perform_export(self, data: Any, export_config: dict[str, Any], format_type: str) -> None:
        """Perform the actual export operation."""
        try:
            export_path = export_config["path"]

            if format_type == "csv":
                self._export_to_csv(data, export_path, export_config)
            elif format_type == "json":
                self._export_to_json(data, export_path, export_config)
            elif format_type == "markdown":
                self._export_to_markdown(data, export_path, export_config)

            self.notify(f"Successfully exported to {export_path}", severity="success")

        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")

    def _export_to_csv(self, data: list[dict], path: Path, config: dict[str, Any]) -> None:
        """Export data to CSV format."""
        import csv

        if not data:
            return

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = list(data[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if config.get("include_headers", True):
                writer.writeheader()

            for row in data:
                # Anonymize if requested
                if config.get("anonymize", False):
                    row = self._anonymize_row(row)
                writer.writerow(row)

    def _export_to_json(self, data: Any, path: Path, config: dict[str, Any]) -> None:
        """Export data to JSON format."""
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Anonymize if requested
        if config.get("anonymize", False):
            data = self._anonymize_data(data)

        with open(path, "w", encoding="utf-8") as jsonfile:
            json.dump(data, jsonfile, indent=2, default=str)

    def _export_to_markdown(self, data: Any, path: Path, config: dict[str, Any]) -> None:
        """Export data as markdown report."""
        self.notify("Markdown export not yet implemented", severity="info")
        # TODO: Implement markdown report generation

    def _prepare_summary_data(self) -> list[dict]:
        """Prepare summary statistics for export."""
        return [
            {"metric": "Total Commits", "value": len(self.commits)},
            {"metric": "Total PRs", "value": len(self.prs)},
            {"metric": "Active Developers", "value": len(self.developers)},
            {
                "metric": "Total Story Points",
                "value": sum(c.get("story_points", 0) or 0 for c in self.commits),
            },
        ]

    def _prepare_qualitative_data(self) -> list[dict]:
        """Prepare qualitative analysis data for export."""
        qualitative_commits = []
        for commit in self.commits:
            if "change_type" in commit:
                qual_commit = {
                    "commit_hash": commit.get("hash"),
                    "author": commit.get("author_name"),
                    "message": commit.get("message"),
                    "change_type": commit.get("change_type"),
                    "business_domain": commit.get("business_domain"),
                    "risk_level": commit.get("risk_level"),
                    "confidence_score": commit.get("confidence_score"),
                }
                qualitative_commits.append(qual_commit)
        return qualitative_commits

    def _anonymize_row(self, row: dict) -> dict:
        """Anonymize sensitive data in a row."""
        # Simple anonymization - replace names with hashed versions
        anonymized = row.copy()

        # Fields to anonymize
        sensitive_fields = ["author_name", "author_email", "primary_name", "primary_email"]

        for field in sensitive_fields:
            if field in anonymized and anonymized[field]:
                # Simple hash-based anonymization
                import hashlib

                hash_value = hashlib.md5(str(anonymized[field]).encode()).hexdigest()[:8]
                anonymized[field] = f"User_{hash_value}"

        return anonymized

    def _anonymize_data(self, data: Any) -> Any:
        """Anonymize data structure recursively."""
        if isinstance(data, list):
            return [self._anonymize_data(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._anonymize_data(value) for key, value in data.items()}
        else:
            return data

    def _show_identity_details(self) -> None:
        """Show detailed developer identity information."""
        self.notify("Identity details view not yet implemented", severity="info")

    def _show_pr_metrics(self) -> None:
        """Show detailed pull request metrics."""
        self.notify("PR metrics view not yet implemented", severity="info")

    def action_back(self) -> None:
        """Go back to main screen."""
        self.app.pop_screen()

    def action_export(self) -> None:
        """Show export options."""
        # Switch to export tab
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "export"

    def action_filter(self) -> None:
        """Show filter options for current tab."""
        self.notify("Filtering functionality not yet implemented", severity="info")

    def action_refresh(self) -> None:
        """Refresh current view."""
        self.refresh()

    def action_export_current(self) -> None:
        """Export data from currently active tab."""
        tabbed_content = self.query_one(TabbedContent)
        current_tab = tabbed_content.active

        if current_tab == "developers":
            self._export_data("developers", "csv")
        elif current_tab == "commits":
            self._export_data("commits", "csv")
        elif current_tab == "pull-requests":
            self._export_data("prs", "csv")
        elif current_tab == "qualitative":
            self._export_data("qualitative", "csv")
        else:
            self._export_data("summary", "csv")
