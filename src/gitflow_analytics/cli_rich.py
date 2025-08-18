"""Rich CLI components for GitFlow Analytics with beautiful terminal output."""

from pathlib import Path
from typing import Any, Optional

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.tree import Tree

from ._version import __version__


class RichProgressDisplay:
    """
    Rich terminal display for GitFlow Analytics progress and results.

    WHY: Provides a clean, structured interface that shows users exactly what's happening
    during analysis without the complexity of a full TUI. Uses Rich library for
    beautiful terminal output with progress bars, tables, and status indicators.

    DESIGN DECISION: Chose to use Rich's Live display for real-time updates because:
    - Allows multiple progress bars and status updates in a single view
    - Provides structured layout with panels and tables
    - Much simpler than TUI but still provides excellent user experience
    - Works in any terminal that supports ANSI colors
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        """Initialize the Rich progress display."""
        self.console = console or Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        self.live: Optional[Live] = None
        self._tasks: dict[str, TaskID] = {}

    def show_header(self) -> None:
        """Display the application header."""
        header = Panel(
            f"[bold blue]GitFlow Analytics v{__version__}[/bold blue]",
            box=box.DOUBLE,
            style="blue",
        )
        self.console.print(header)
        self.console.print()

    def show_configuration_status(
        self,
        config_path: Path,
        github_org: Optional[str] = None,
        github_token_valid: bool = False,
        jira_configured: bool = False,
        jira_valid: bool = False,
        analysis_weeks: int = 12,
    ) -> None:
        """
        Display configuration validation status.

        WHY: Users need immediate feedback on whether their configuration is valid
        before starting analysis. This prevents wasted time on invalid configs.
        """
        config_tree = Tree("[bold]Configuration Status[/bold]")

        # Config file status
        config_tree.add(f"[green]✓[/green] Config: {config_path}")

        # GitHub configuration
        if github_org:
            github_status = "[green]✓[/green]" if github_token_valid else "[red]✗[/red]"
            token_status = "Token: ✓" if github_token_valid else "Token: ✗"
            config_tree.add(f"{github_status} GitHub: {github_org} ({token_status})")

        # JIRA configuration
        if jira_configured:
            jira_status = "[green]✓[/green]" if jira_valid else "[red]✗[/red]"
            cred_status = "Credentials: ✓" if jira_valid else "Credentials: ✗"
            config_tree.add(f"{jira_status} JIRA: configured ({cred_status})")

        # Analysis period
        config_tree.add(f"Analysis Period: {analysis_weeks} weeks")

        self.console.print(config_tree)
        self.console.print()

    def start_live_display(self) -> None:
        """Start the live display for real-time updates."""
        self.live = Live(self.progress, console=self.console, refresh_per_second=10)
        self.live.start()

    def stop_live_display(self) -> None:
        """Stop the live display."""
        if self.live:
            self.live.stop()
            self.live = None

    def add_progress_task(self, name: str, description: str, total: int) -> None:
        """Add a new progress task."""
        task_id = self.progress.add_task(description, total=total)
        self._tasks[name] = task_id

    def update_progress_task(
        self, name: str, advance: int = 1, description: Optional[str] = None
    ) -> None:
        """Update progress for a specific task."""
        if name in self._tasks:
            kwargs = {"advance": advance}
            if description:
                kwargs["description"] = description
            self.progress.update(self._tasks[name], **kwargs)

    def complete_progress_task(self, name: str, description: Optional[str] = None) -> None:
        """Mark a progress task as complete."""
        if name in self._tasks:
            task = self.progress.tasks[self._tasks[name]]
            remaining = task.total - task.completed if task.total else 0
            if remaining > 0:
                self.progress.update(self._tasks[name], advance=remaining)
            if description:
                self.progress.update(self._tasks[name], description=description)

    def show_repository_discovery(self, repos: list[dict[str, Any]]) -> None:
        """
        Display repository discovery results.

        WHY: Users need to see which repositories were discovered and their status
        before analysis begins, especially with organization-based discovery.
        """
        if not repos:
            return

        self.console.print(f"[bold]Repository Discovery[/bold] - Found {len(repos)} repositories")

        repo_tree = Tree("")
        for repo in repos:
            status = "[green]✓[/green]" if repo.get("exists", True) else "[red]✗[/red]"
            name = repo.get("name", "unknown")
            github_repo = repo.get("github_repo", "")
            if github_repo:
                repo_tree.add(f"{status} {name} ({github_repo})")
            else:
                repo_tree.add(f"{status} {name}")

        self.console.print(repo_tree)
        self.console.print()

    def show_analysis_summary(
        self,
        total_commits: int,
        total_prs: int,
        active_developers: int,
        ticket_coverage: float,
        story_points: int,
        qualitative_analyzed: int = 0,
    ) -> None:
        """
        Display analysis results summary.

        WHY: Provides users with key metrics at a glance after analysis completes.
        Uses a structured table format for easy scanning of important numbers.
        """
        self.console.print()

        summary_table = Table(title="[bold]Analysis Summary[/bold]", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan", width=20)
        summary_table.add_column("Value", style="green", width=15)

        summary_table.add_row("Total Commits", f"{total_commits:,}")
        summary_table.add_row("Total PRs", f"{total_prs:,}")
        summary_table.add_row("Active Developers", f"{active_developers:,}")
        summary_table.add_row("Ticket Coverage", f"{ticket_coverage:.1f}%")
        summary_table.add_row("Story Points", f"{story_points:,}")

        if qualitative_analyzed > 0:
            summary_table.add_row("Qualitative Analysis", f"{qualitative_analyzed:,} commits")

        self.console.print(summary_table)

    def show_dora_metrics(self, dora_metrics: dict[str, Any]) -> None:
        """
        Display DORA metrics in a structured format.

        WHY: DORA metrics are key performance indicators that teams care about.
        Displaying them prominently helps users understand their team's performance level.
        """
        if not dora_metrics:
            return

        self.console.print()

        dora_table = Table(title="[bold]DORA Metrics[/bold]", box=box.ROUNDED)
        dora_table.add_column("Metric", style="cyan", width=25)
        dora_table.add_column("Value", style="yellow", width=20)

        # Deployment frequency
        df_category = dora_metrics.get("deployment_frequency", {}).get("category", "Unknown")
        dora_table.add_row("Deployment Frequency", df_category)

        # Lead time
        lead_time = dora_metrics.get("lead_time_hours", 0)
        dora_table.add_row("Lead Time", f"{lead_time:.1f} hours")

        # Change failure rate
        cfr = dora_metrics.get("change_failure_rate", 0)
        dora_table.add_row("Change Failure Rate", f"{cfr:.1f}%")

        # MTTR
        mttr = dora_metrics.get("mttr_hours", 0)
        dora_table.add_row("MTTR", f"{mttr:.1f} hours")

        # Performance level
        perf_level = dora_metrics.get("performance_level", "Unknown")
        dora_table.add_row("Performance Level", f"[bold]{perf_level}[/bold]")

        self.console.print(dora_table)

    def show_qualitative_stats(self, qual_stats: dict[str, Any]) -> None:
        """
        Display qualitative analysis statistics.

        WHY: Qualitative analysis can be expensive (time/cost), so users need
        visibility into processing efficiency and costs incurred.
        """
        if not qual_stats:
            return

        processing_summary = qual_stats.get("processing_summary", {})
        llm_stats = qual_stats.get("llm_statistics", {})

        self.console.print()

        qual_table = Table(title="[bold]Qualitative Analysis Stats[/bold]", box=box.ROUNDED)
        qual_table.add_column("Metric", style="cyan", width=25)
        qual_table.add_column("Value", style="magenta", width=20)

        # Processing speed
        commits_per_sec = processing_summary.get("commits_per_second", 0)
        qual_table.add_row("Processing Speed", f"{commits_per_sec:.1f} commits/sec")

        # Method breakdown
        method_breakdown = processing_summary.get("method_breakdown", {})
        cache_pct = method_breakdown.get("cache", 0)
        nlp_pct = method_breakdown.get("nlp", 0)
        llm_pct = method_breakdown.get("llm", 0)
        qual_table.add_row("Cache Usage", f"{cache_pct:.1f}%")
        qual_table.add_row("NLP Processing", f"{nlp_pct:.1f}%")
        qual_table.add_row("LLM Processing", f"{llm_pct:.1f}%")

        # LLM costs if available
        if llm_stats.get("model_usage") == "available":
            cost_tracking = llm_stats.get("cost_tracking", {})
            total_cost = cost_tracking.get("total_cost", 0)
            if total_cost > 0:
                qual_table.add_row("LLM Cost", f"${total_cost:.4f}")

        self.console.print(qual_table)

    def show_llm_cost_summary(
        self, cost_stats: dict[str, Any], identity_costs: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Display comprehensive LLM usage and cost summary.

        WHY: LLM usage can be expensive and users need visibility into:
        - Token consumption by component (identity analysis, qualitative analysis)
        - Cost breakdown and budget utilization
        - Optimization suggestions to reduce costs

        DESIGN DECISION: Separate method from qualitative stats because:
        - Cost tracking spans multiple components (identity + qualitative)
        - Users need this summary even when qualitative analysis is disabled
        - Provides actionable cost optimization suggestions

        Args:
            cost_stats: Cost statistics from qualitative analysis
            identity_costs: Optional cost statistics from identity analysis
        """
        # Check if we have any cost data to display
        has_qual_costs = cost_stats and cost_stats.get("total_cost", 0) > 0
        has_identity_costs = identity_costs and identity_costs.get("total_cost", 0) > 0

        if not (has_qual_costs or has_identity_costs):
            return

        self.console.print()

        # Create main cost summary table
        cost_table = Table(title="[bold]🤖 LLM Usage Summary[/bold]", box=box.ROUNDED)
        cost_table.add_column("Component", style="cyan", width=20)
        cost_table.add_column("Calls", style="yellow", width=8)
        cost_table.add_column("Tokens", style="green", width=12)
        cost_table.add_column("Cost", style="magenta", width=12)

        total_calls = 0
        total_tokens = 0
        total_cost = 0.0
        budget_info = {}

        # Add identity analysis costs if available
        if has_identity_costs:
            identity_calls = identity_costs.get("total_calls", 0)
            identity_tokens = identity_costs.get("total_tokens", 0)
            identity_cost = identity_costs.get("total_cost", 0)

            cost_table.add_row(
                "Identity Analysis",
                f"{identity_calls:,}",
                f"{identity_tokens:,}",
                f"${identity_cost:.4f}",
            )

            total_calls += identity_calls
            total_tokens += identity_tokens
            total_cost += identity_cost

        # Add qualitative analysis costs if available
        if has_qual_costs:
            qual_calls = cost_stats.get("total_calls", 0)
            qual_tokens = cost_stats.get("total_tokens", 0)
            qual_cost = cost_stats.get("total_cost", 0)

            cost_table.add_row(
                "Qualitative Analysis", f"{qual_calls:,}", f"{qual_tokens:,}", f"${qual_cost:.4f}"
            )

            total_calls += qual_calls
            total_tokens += qual_tokens
            total_cost += qual_cost

            # Extract budget information (assuming it's in qualitative cost stats)
            budget_info = {
                "daily_budget": cost_stats.get("daily_budget", 5.0),
                "daily_spend": cost_stats.get("daily_spend", total_cost),
                "remaining_budget": cost_stats.get("remaining_budget", 5.0 - total_cost),
            }

        # Add total row
        if total_calls > 0:
            cost_table.add_row(
                "[bold]Total[/bold]",
                f"[bold]{total_calls:,}[/bold]",
                f"[bold]{total_tokens:,}[/bold]",
                f"[bold]${total_cost:.4f}[/bold]",
            )

        self.console.print(cost_table)

        # Display budget information if available
        if budget_info:
            daily_budget = budget_info.get("daily_budget", 5.0)
            remaining = budget_info.get("remaining_budget", daily_budget - total_cost)
            utilization = (total_cost / daily_budget) * 100 if daily_budget > 0 else 0

            budget_text = f"Budget: ${daily_budget:.2f}, Remaining: ${remaining:.2f}, Utilization: {utilization:.1f}%"

            # Color code based on utilization
            if utilization >= 90:
                budget_color = "red"
            elif utilization >= 70:
                budget_color = "yellow"
            else:
                budget_color = "green"

            self.console.print(f"   [{budget_color}]💰 {budget_text}[/{budget_color}]")

        # Display cost optimization suggestions if we have detailed stats
        suggestions = []
        if has_qual_costs and "model_usage" in cost_stats:
            model_usage = cost_stats.get("model_usage", {})

            # Check for expensive model usage
            expensive_models = ["anthropic/claude-3-opus", "openai/gpt-4"]
            expensive_cost = sum(
                model_usage.get(model, {}).get("cost", 0) for model in expensive_models
            )

            if expensive_cost > total_cost * 0.3:
                suggestions.append(
                    "Consider using cheaper models (Claude Haiku, GPT-3.5) for routine tasks (save ~40%)"
                )

            # Check for free model opportunities
            free_usage = model_usage.get("meta-llama/llama-3.1-8b-instruct:free", {}).get(
                "cost", -1
            )
            if free_usage == 0 and total_cost > 0.01:  # If free models available but not used much
                suggestions.append(
                    "Increase usage of free Llama models for simple classification tasks"
                )

        # Budget-based suggestions
        if budget_info:
            utilization = (total_cost / budget_info.get("daily_budget", 5.0)) * 100
            if utilization > 80:
                suggestions.append(
                    "Approaching daily budget limit - consider increasing NLP confidence threshold"
                )

        # Display suggestions
        if suggestions:
            self.console.print()
            self.console.print("[bold]💡 Cost Optimization Suggestions:[/bold]")
            for suggestion in suggestions:
                self.console.print(f"   • {suggestion}")

        self.console.print()

    def show_reports_generated(self, output_dir: Path, report_files: list[str]) -> None:
        """
        Display generated reports with file paths.

        WHY: Users need to know where their reports were saved and what files
        were generated. This provides clear next steps after analysis completes.
        """
        if not report_files:
            return

        self.console.print()

        reports_panel = Panel(
            f"[bold green]✓[/bold green] Reports exported to: [cyan]{output_dir}[/cyan]",
            title="[bold]Generated Reports[/bold]",
            box=box.ROUNDED,
        )
        self.console.print(reports_panel)

        # List individual report files
        for report_file in report_files:
            self.console.print(f"  • {report_file}")

    def show_error(self, error_message: str, show_debug_hint: bool = True) -> None:
        """
        Display error messages in a prominent format.

        WHY: Errors need to be clearly visible and actionable. The panel format
        makes them stand out while providing helpful guidance.
        """
        error_panel = Panel(
            f"[red]{error_message}[/red]",
            title="[bold red]Error[/bold red]",
            box=box.HEAVY,
        )
        self.console.print(error_panel)

        if show_debug_hint:
            self.console.print("\n[dim]💡 Run with --debug for detailed error information[/dim]")

    def show_warning(self, warning_message: str) -> None:
        """Display warning messages."""
        warning_panel = Panel(
            f"[yellow]{warning_message}[/yellow]",
            title="[bold yellow]Warning[/bold yellow]",
            box=box.ROUNDED,
        )
        self.console.print(warning_panel)

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Delegate print calls to the console."""
        self.console.print(*args, **kwargs)

    def print_status(self, message: str, status: str = "info") -> None:
        """
        Print a status message with appropriate styling.

        WHY: Provides consistent status messaging throughout the analysis process.
        Different status types (info, success, warning, error) get appropriate styling.
        """
        if status == "success":
            self.console.print(f"[green]✓[/green] {message}")
        elif status == "warning":
            self.console.print(f"[yellow]⚠[/yellow] {message}")
        elif status == "error":
            self.console.print(f"[red]✗[/red] {message}")
        elif status == "info":
            self.console.print(f"[blue]ℹ[/blue] {message}")
        else:
            self.console.print(message)


def create_rich_display() -> RichProgressDisplay:
    """
    Factory function to create a Rich progress display.

    WHY: Centralizes the creation of the display component and ensures
    consistent configuration across the application.
    """
    return RichProgressDisplay()
