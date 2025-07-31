#!/usr/bin/env python3
"""
GitFlow Analytics Terminal UI Application
Built with Textual for modern, interactive terminal experience
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import json

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Button, Input, Label, Static, DataTable, 
    ProgressBar, Log, TabbedContent, TabPane, Select, Switch,
    Pretty, Rule, Markdown
)
from textual.screen import Screen, ModalScreen
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding
from textual import events
from textual.validation import Function, ValidationResult, Validator
from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

# Import GitFlow Analytics components
from gitflow_analytics.config import ConfigLoader, Config
from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.core.analyzer import GitAnalyzer
from gitflow_analytics.core.identity import DeveloperIdentityResolver
from gitflow_analytics.qualitative.core.processor import QualitativeProcessor
from gitflow_analytics.integrations.orchestrator import IntegrationOrchestrator


class ConfigurationScreen(ModalScreen[Optional[Config]]):
    """Modal screen for configuration management."""
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save Config"),
    ]
    
    def __init__(self, config_path: Optional[Path] = None, config: Optional[Config] = None):
        super().__init__()
        self.config_path = config_path
        self.config = config
        self.form_data = {}
        
    def compose(self) -> ComposeResult:
        with Container(id="config-modal"):
            yield Label("GitFlow Analytics Configuration", classes="modal-title")
            
            with TabbedContent():
                with TabPane("Basic Settings", id="basic"):
                    yield Label("Configuration File:")
                    yield Input(
                        value=str(self.config_path) if self.config_path else "",
                        placeholder="Path to config.yaml file",
                        id="config-path"
                    )
                    yield Button("Browse", id="browse-config")
                    
                    yield Rule()
                    yield Label("Analysis Settings:")
                    
                    with Horizontal():
                        yield Label("Weeks to analyze:")
                        yield Input(
                            value="12",
                            placeholder="12",
                            id="weeks"
                        )
                    
                    with Horizontal():
                        yield Label("Enable qualitative analysis:")
                        yield Switch(value=True, id="enable-qualitative")
                        
                with TabPane("API Keys", id="api-keys"):
                    yield Label("GitHub Configuration:")
                    yield Input(
                        placeholder="GitHub Personal Access Token",
                        password=True,
                        id="github-token"
                    )
                    yield Input(
                        placeholder="GitHub Organization (optional)",
                        id="github-org"
                    )
                    
                    yield Rule()
                    yield Label("OpenRouter Configuration:")
                    yield Input(
                        placeholder="OpenRouter API Key",
                        password=True,
                        id="openrouter-key"
                    )
                    
                    yield Rule()
                    yield Label("JIRA Configuration (Optional):")
                    yield Input(
                        placeholder="JIRA Base URL (e.g., https://company.atlassian.net)",
                        id="jira-url"
                    )
                    yield Input(
                        placeholder="JIRA Username/Email",
                        id="jira-user"
                    )
                    yield Input(
                        placeholder="JIRA API Token",
                        password=True,
                        id="jira-token"
                    )
                    
                with TabPane("Cache Settings", id="cache"):
                    yield Label("Cache Configuration:")
                    
                    with Horizontal():
                        yield Label("Cache directory:")
                        yield Input(
                            value=".gitflow-cache",
                            placeholder=".gitflow-cache",
                            id="cache-dir"
                        )
                    
                    with Horizontal():
                        yield Label("Cache TTL (hours):")
                        yield Input(
                            value="168",
                            placeholder="168",
                            id="cache-ttl"
                        )
                    
                    with Horizontal():
                        yield Label("Clear cache on startup:")
                        yield Switch(value=False, id="clear-cache")
            
            with Horizontal(classes="button-bar"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Save & Continue", variant="primary", id="save-config")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "save-config":
            self._save_configuration()
        elif event.button.id == "browse-config":
            self._browse_config_file()
    
    def _save_configuration(self) -> None:
        """Save configuration and dismiss modal."""
        try:
            # Collect form data
            form_data = {
                'config_path': self.query_one("#config-path").value,
                'weeks': int(self.query_one("#weeks").value or "12"),
                'enable_qualitative': self.query_one("#enable-qualitative").value,
                'github_token': self.query_one("#github-token").value,
                'github_org': self.query_one("#github-org").value,
                'openrouter_key': self.query_one("#openrouter-key").value,
                'jira_url': self.query_one("#jira-url").value,
                'jira_user': self.query_one("#jira-user").value,
                'jira_token': self.query_one("#jira-token").value,
                'cache_dir': self.query_one("#cache-dir").value,
                'cache_ttl': int(self.query_one("#cache-ttl").value or "168"),
                'clear_cache': self.query_one("#clear-cache").value,
            }
            
            # Update environment variables if provided
            self._update_env_vars(form_data)
            
            # Load or create config
            if form_data['config_path'] and Path(form_data['config_path']).exists():
                config = ConfigLoader.load(Path(form_data['config_path']))
            else:
                config = self._create_default_config(form_data)
            
            self.dismiss(config)
            
        except Exception as e:
            self.notify(f"Configuration error: {e}", severity="error")
    
    def _update_env_vars(self, form_data: Dict[str, Any]) -> None:
        """Update environment variables with provided values."""
        env_updates = {}
        
        if form_data['github_token']:
            env_updates['GITHUB_TOKEN'] = form_data['github_token']
        if form_data['github_org']:
            env_updates['GITHUB_ORG'] = form_data['github_org']
        if form_data['openrouter_key']:
            env_updates['OPENROUTER_API_KEY'] = form_data['openrouter_key']
        if form_data['jira_user']:
            env_updates['JIRA_ACCESS_USER'] = form_data['jira_user']
        if form_data['jira_token']:
            env_updates['JIRA_ACCESS_TOKEN'] = form_data['jira_token']
        
        # Update current environment
        os.environ.update(env_updates)
        
        # Save to .env file if we have a config path
        if form_data['config_path']:
            config_dir = Path(form_data['config_path']).parent
            env_file = config_dir / '.env'
            
            # Read existing .env
            existing_env = {}
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        if '=' in line and not line.strip().startswith('#'):
                            key, value = line.strip().split('=', 1)
                            existing_env[key] = value
            
            # Update with new values
            existing_env.update(env_updates)
            
            # Write back to .env
            with open(env_file, 'w') as f:
                for key, value in existing_env.items():
                    f.write(f"{key}={value}\n")


class AnalysisProgressScreen(Screen):
    """Screen showing real-time analysis progress."""
    
    BINDINGS = [
        Binding("ctrl+c", "cancel", "Cancel Analysis"),
        Binding("escape", "back", "Back to Main"),
    ]
    
    def __init__(self, config: Config, weeks: int = 12):
        super().__init__()
        self.config = config
        self.weeks = weeks
        self.progress_data = {}
        self.analysis_task: Optional[asyncio.Task] = None
        
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="progress-container"):
            yield Label("GitFlow Analytics - Analysis in Progress", classes="screen-title")
            
            with Vertical(id="progress-panels"):
                # Overall progress
                with Container(classes="progress-panel"):
                    yield Label("Overall Progress", classes="panel-title")
                    yield ProgressBar(total=100, id="overall-progress")
                    yield Label("Initializing...", id="overall-status")
                
                # Repository progress
                with Container(classes="progress-panel"):
                    yield Label("Repository Analysis", classes="panel-title")
                    yield ProgressBar(total=100, id="repo-progress")
                    yield Label("Waiting...", id="repo-status")
                
                # Qualitative analysis progress
                with Container(classes="progress-panel"):
                    yield Label("Qualitative Analysis", classes="panel-title")
                    yield ProgressBar(total=100, id="qual-progress")
                    yield Label("Waiting...", id="qual-status")
                
                # Live statistics
                with Container(classes="stats-panel"):
                    yield Label("Live Statistics", classes="panel-title")
                    yield Pretty({}, id="live-stats")
            
            # Log output
            with Container(classes="log-panel"):
                yield Label("Analysis Log", classes="panel-title")
                yield Log(auto_scroll=True, id="analysis-log")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Start analysis when screen mounts."""
        self.analysis_task = asyncio.create_task(self._run_analysis())
    
    async def _run_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        log = self.query_one("#analysis-log", Log)
        
        try:
            log.write_line("🚀 Starting GitFlow Analytics...")
            
            # Phase 1: Initialize components
            await self._update_progress("overall", 10, "Initializing components...")
            
            cache = GitAnalysisCache(
                self.config.cache.directory,
                ttl_hours=self.config.cache.ttl_hours
            )
            
            identity_resolver = DeveloperIdentityResolver(
                self.config.cache.directory / 'identities.db',
                similarity_threshold=self.config.analysis.similarity_threshold,
                manual_mappings=self.config.analysis.manual_identity_mappings
            )
            
            analyzer = GitAnalyzer(
                cache,
                branch_mapping_rules=self.config.analysis.branch_mapping_rules,
                allowed_ticket_platforms=getattr(self.config.analysis, 'ticket_platforms', None),
                exclude_paths=self.config.analysis.exclude_paths
            )
            
            orchestrator = IntegrationOrchestrator(self.config, cache)
            
            # Initialize qualitative processor if enabled
            qual_processor = None
            if hasattr(self.config, 'qualitative') and self.config.qualitative.enabled:
                qual_processor = QualitativeProcessor(self.config.qualitative)
            
            log.write_line("✅ Components initialized")
            
            # Phase 2: Repository discovery
            await self._update_progress("overall", 20, "Discovering repositories...")
            
            repositories = self.config.repositories
            if self.config.github.organization and not repositories:
                log.write_line(f"🔍 Discovering repositories from {self.config.github.organization}...")
                repositories = self.config.discover_organization_repositories()
                log.write_line(f"✅ Found {len(repositories)} repositories")
            
            # Phase 3: Analysis period setup
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=self.weeks)
            
            log.write_line(f"📅 Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Phase 4: Repository analysis
            await self._update_progress("overall", 30, "Analyzing repositories...")
            
            all_commits = []
            all_prs = []
            
            for i, repo_config in enumerate(repositories):
                repo_progress = (i / len(repositories)) * 100
                await self._update_progress("repo", repo_progress, f"Analyzing {repo_config.name}...")
                
                log.write_line(f"📁 Analyzing {repo_config.name}...")
                
                try:
                    # Clone if needed
                    if not repo_config.path.exists() and repo_config.github_repo:
                        log.write_line(f"   📥 Cloning {repo_config.github_repo}...")
                        await self._clone_repository(repo_config)
                    
                    # Analyze commits
                    commits = analyzer.analyze_repository(
                        repo_config.path,
                        start_date,
                        repo_config.branch
                    )
                    
                    # Resolve identities
                    for commit in commits:
                        commit['project_key'] = repo_config.project_key or commit.get('inferred_project', 'UNKNOWN')
                        commit['canonical_id'] = identity_resolver.resolve_developer(
                            commit['author_name'],
                            commit['author_email']
                        )
                    
                    all_commits.extend(commits)
                    log.write_line(f"   ✅ Found {len(commits)} commits")
                    
                    # Enrich with integration data
                    enrichment = orchestrator.enrich_repository_data(
                        repo_config, commits, start_date
                    )
                    
                    if enrichment['prs']:
                        all_prs.extend(enrichment['prs'])
                        log.write_line(f"   ✅ Found {len(enrichment['prs'])} pull requests")
                    
                    # Update live stats
                    await self._update_live_stats({
                        'repositories_analyzed': i + 1,
                        'total_repositories': len(repositories),
                        'total_commits': len(all_commits),
                        'total_prs': len(all_prs),
                        'current_repo': repo_config.name
                    })
                    
                except Exception as e:
                    log.write_line(f"   ❌ Error analyzing {repo_config.name}: {e}")
                    continue
            
            await self._update_progress("repo", 100, f"Completed {len(repositories)} repositories")
            
            # Phase 5: Qualitative analysis
            if qual_processor and all_commits:
                await self._update_progress("overall", 60, "Running qualitative analysis...")
                await self._update_progress("qual", 10, "Processing commit messages...")
                
                log.write_line("🧠 Starting qualitative analysis...")
                
                # Process in batches for progress updates
                batch_size = 1000
                qualitative_results = []
                
                for i in range(0, len(all_commits), batch_size):
                    batch = all_commits[i:i + batch_size]
                    batch_progress = (i / len(all_commits)) * 100
                    
                    await self._update_progress("qual", batch_progress, 
                                              f"Processing commits {i+1}-{min(i+batch_size, len(all_commits))}...")
                    
                    batch_results = qual_processor.process_commits(batch)
                    qualitative_results.extend(batch_results)
                    
                    # Update with qualitative data
                    for original, enhanced in zip(batch, batch_results):
                        original.update({
                            'change_type': enhanced.change_type,
                            'business_domain': enhanced.business_domain,
                            'risk_level': enhanced.risk_level,
                            'confidence_score': enhanced.confidence_score
                        })
                
                await self._update_progress("qual", 100, "Qualitative analysis complete")
                log.write_line("✅ Qualitative analysis complete")
            
            # Phase 6: Generate reports
            await self._update_progress("overall", 80, "Generating reports...")
            
            # Update developer statistics
            identity_resolver.update_commit_stats(all_commits)
            developer_stats = identity_resolver.get_developer_stats()
            
            log.write_line(f"👥 Identified {len(developer_stats)} unique developers")
            
            # Final progress
            await self._update_progress("overall", 100, "Analysis complete!")
            
            log.write_line("🎉 Analysis completed successfully!")
            log.write_line(f"   - Total commits: {len(all_commits)}")
            log.write_line(f"   - Total PRs: {len(all_prs)}")
            log.write_line(f"   - Active developers: {len(developer_stats)}")
            
            # Switch to results screen
            await asyncio.sleep(2)  # Brief pause to show completion
            self.app.push_screen(ResultsScreen(all_commits, all_prs, developer_stats))
            
        except Exception as e:
            log.write_line(f"❌ Analysis failed: {e}")
            await self._update_progress("overall", 0, f"Error: {e}")
    
    async def _update_progress(self, progress_id: str, value: float, status: str) -> None:
        """Update progress bar and status."""
        progress_bar = self.query_one(f"#{progress_id}-progress", ProgressBar)
        status_label = self.query_one(f"#{progress_id}-status", Label)
        
        progress_bar.update(progress=value)
        status_label.update(status)
    
    async def _update_live_stats(self, stats: Dict[str, Any]) -> None:
        """Update live statistics display."""
        stats_widget = self.query_one("#live-stats", Pretty)
        stats_widget.update(stats)
    
    async def _clone_repository(self, repo_config) -> None:
        """Clone repository if needed."""
        import git
        
        repo_config.path.parent.mkdir(parents=True, exist_ok=True)
        
        clone_url = f"https://github.com/{repo_config.github_repo}.git"
        if self.config.github.token:
            clone_url = f"https://{self.config.github.token}@github.com/{repo_config.github_repo}.git"
        
        git.Repo.clone_from(clone_url, repo_config.path, branch=repo_config.branch)
    
    def action_cancel(self) -> None:
        """Cancel the analysis."""
        if self.analysis_task:
            self.analysis_task.cancel()
        self.app.pop_screen()
    
    def action_back(self) -> None:
        """Go back to main screen."""
        if self.analysis_task:
            self.analysis_task.cancel()
        self.app.pop_screen()


class ResultsScreen(Screen):
    """Screen displaying analysis results."""
    
    BINDINGS = [
        Binding("escape", "back", "Back to Main"),
        Binding("ctrl+s", "save", "Save Results"),
        Binding("r", "refresh", "Refresh View"),
    ]
    
    def __init__(self, commits: List[Dict], prs: List[Dict], developers: List[Dict]):
        super().__init__()
        self.commits = commits
        self.prs = prs
        self.developers = developers
        
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="results-container"):
            yield Label("GitFlow Analytics - Results", classes="screen-title")
            
            with TabbedContent():
                with TabPane("Summary", id="summary"):
                    yield self._create_summary_panel()
                
                with TabPane("Developers", id="developers"):
                    yield self._create_developers_table()
                
                with TabPane("Commits", id="commits"):
                    yield self._create_commits_table()
                
                with TabPane("Qualitative Insights", id="qualitative"):
                    yield self._create_qualitative_panel()
                
                with TabPane("Export", id="export"):
                    yield self._create_export_panel()
        
        yield Footer()
    
    def _create_summary_panel(self) -> Container:
        """Create summary statistics panel."""
        # Calculate summary stats
        total_story_points = sum(c.get('story_points', 0) or 0 for c in self.commits)
        commits_with_tickets = sum(1 for c in self.commits if c.get('ticket_references'))
        ticket_coverage = (commits_with_tickets / len(self.commits)) * 100 if self.commits else 0
        
        # Create summary table
        table = Table(show_header=False, show_edge=False, pad_edge=False)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Commits", str(len(self.commits)))
        table.add_row("Total PRs", str(len(self.prs)))
        table.add_row("Active Developers", str(len(self.developers)))
        table.add_row("Story Points Completed", str(total_story_points))
        table.add_row("Ticket Coverage", f"{ticket_coverage:.1f}%")
        
        # Top contributors
        top_devs = sorted(self.developers, key=lambda d: d['total_commits'], reverse=True)[:5]
        top_contributors = "\n".join([f"• {dev['primary_name']}: {dev['total_commits']} commits" 
                                    for dev in top_devs])
        
        container = Container()
        container.mount(Pretty(table))
        container.mount(Rule())
        container.mount(Label("Top Contributors:", classes="section-title"))
        container.mount(Static(top_contributors))
        
        return container
    
    def _create_developers_table(self) -> DataTable:
        """Create developers data table."""
        table = DataTable()
        
        # Add columns
        table.add_column("Developer", width=30)
        table.add_column("Commits", width=10)
        table.add_column("Story Points", width=12)
        table.add_column("Avg Points/Commit", width=15)
        table.add_column("Last Active", width=20)
        
        # Add rows
        for dev in sorted(self.developers, key=lambda d: d['total_commits'], reverse=True):
            avg_points = dev['total_story_points'] / dev['total_commits'] if dev['total_commits'] > 0 else 0
            last_seen = dev['last_seen'].strftime('%Y-%m-%d') if dev['last_seen'] else 'Unknown'
            
            table.add_row(
                dev['primary_name'],
                str(dev['total_commits']),
                str(dev['total_story_points']),
                f"{avg_points:.1f}",
                last_seen
            )
        
        return table
    
    def _create_commits_table(self) -> DataTable:
        """Create commits data table."""
        table = DataTable()
        
        # Add columns
        table.add_column("Date", width=12)
        table.add_column("Author", width=20)
        table.add_column("Message", width=50)
        table.add_column("Type", width=10)
        table.add_column("Files", width=8)
        table.add_column("Risk", width=8)
        
        # Add recent commits (last 100)
        recent_commits = sorted(self.commits, key=lambda c: c['timestamp'], reverse=True)[:100]
        
        for commit in recent_commits:
            table.add_row(
                commit['timestamp'].strftime('%Y-%m-%d'),
                commit['author_name'][:18] + '...' if len(commit['author_name']) > 18 else commit['author_name'],
                commit['message'][:47] + '...' if len(commit['message']) > 47 else commit['message'],
                commit.get('change_type', 'unknown'),
                str(commit.get('files_changed', 0)),
                commit.get('risk_level', 'unknown')
            )
        
        return table
    
    def _create_qualitative_panel(self) -> ScrollableContainer:
        """Create qualitative insights panel."""
        container = ScrollableContainer()
        
        # Analyze qualitative data
        if any('change_type' in c for c in self.commits):
            # Change type distribution
            change_types = {}
            risk_levels = {}
            domains = {}
            
            for commit in self.commits:
                change_type = commit.get('change_type', 'unknown')
                change_types[change_type] = change_types.get(change_type, 0) + 1
                
                risk_level = commit.get('risk_level', 'unknown')
                risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
                
                domain = commit.get('business_domain', 'unknown')
                domains[domain] = domains.get(domain, 0) + 1
            
            # Create insights
            container.mount(Label("Qualitative Analysis Results", classes="section-title"))
            
            # Change types
            container.mount(Label("Change Type Distribution:", classes="subsection-title"))
            for change_type, count in sorted(change_types.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(self.commits)) * 100
                container.mount(Static(f"• {change_type.title()}: {count} commits ({pct:.1f}%)"))
            
            container.mount(Rule())
            
            # Risk levels
            container.mount(Label("Risk Level Distribution:", classes="subsection-title"))
            for risk_level, count in sorted(risk_levels.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(self.commits)) * 100
                container.mount(Static(f"• {risk_level.title()}: {count} commits ({pct:.1f}%)"))
            
            container.mount(Rule())
            
            # Business domains
            container.mount(Label("Business Domain Activity:", classes="subsection-title"))
            for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(self.commits)) * 100
                container.mount(Static(f"• {domain.title()}: {count} commits ({pct:.1f}%)"))
        else:
            container.mount(Label("No qualitative analysis data available.", classes="info-message"))
            container.mount(Static("Run analysis with qualitative processing enabled to see insights here."))
        
        return container
    
    def _create_export_panel(self) -> Container:
        """Create export options panel."""
        container = Container()
        
        container.mount(Label("Export Results", classes="section-title"))
        container.mount(Static("Choose export format and options:"))
        
        container.mount(Button("Export to CSV", id="export-csv"))
        container.mount(Button("Export to JSON", id="export-json"))
        container.mount(Button("Generate Markdown Report", id="export-markdown"))
        
        return container
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "export-csv":
            self._export_csv()
        elif event.button.id == "export-json":
            self._export_json()
        elif event.button.id == "export-markdown":
            self._export_markdown()
    
    def _export_csv(self) -> None:
        """Export results to CSV files."""
        # TODO: Implement CSV export
        self.notify("CSV export functionality coming soon!", severity="info")
    
    def _export_json(self) -> None:
        """Export results to JSON."""
        # TODO: Implement JSON export  
        self.notify("JSON export functionality coming soon!", severity="info")
    
    def _export_markdown(self) -> None:
        """Generate markdown report."""
        # TODO: Implement markdown report generation
        self.notify("Markdown report generation coming soon!", severity="info")
    
    def action_back(self) -> None:
        """Go back to main screen."""
        self.app.pop_screen()


class GitFlowAnalyticsApp(App):
    """Main GitFlow Analytics Terminal Application."""
    
    TITLE = "GitFlow Analytics"
    SUB_TITLE = "Developer Productivity Analysis"
    
    CSS = """
    .screen-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin: 1;
    }
    
    .modal-title {
        text-align: center;
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    .section-title {
        text-style: bold;
        color: $secondary;
        margin: 1 0;
    }
    
    .subsection-title {
        text-style: bold;
        color: $warning;
        margin: 1 0 0 0;
    }
    
    .info-message {
        color: $warning;
        text-style: italic;
        margin: 1 0;
    }
    
    .button-bar {
        dock: bottom;
        height: 3;
        align: center middle;
    }
    
    .progress-panel {
        height: 8;
        border: solid $primary;
        margin: 1;
        padding: 1;
    }
    
    .stats-panel {
        height: 12;
        border: solid $secondary;
        margin: 1;
        padding: 1;
    }
    
    .log-panel {
        border: solid $accent;
        margin: 1;
        padding: 1;
    }
    
    .panel-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    #config-modal {
        background: $surface;
        border: thick $primary;
        width: 80%;
        height: 80%;
        margin: 2;
        padding: 1;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+n", "new_analysis", "New Analysis"),
        Binding("ctrl+o", "open_config", "Open Config"),
        Binding("f1", "help", "Help"),
    ]
    
    def __init__(self):
        super().__init__()
        self.config: Optional[Config] = None
        self.config_path: Optional[Path] = None
        
    def compose(self) -> ComposeResult:
        """Compose the main UI."""
        yield Header()
        
        with Container(id="main-container"):
            yield Label("GitFlow Analytics", classes="screen-title")
            yield Static("Developer Productivity Analysis Tool", id="subtitle")
            
            with Vertical(id="main-menu"):
                if self.config:
                    yield Label(f"Configuration: {self.config_path}", id="config-status")
                    yield Button("Run Analysis", variant="primary", id="run-analysis")
                    yield Button("View Cache Status", id="cache-status")
                    yield Button("Manage Identities", id="manage-identities")
                else:
                    yield Label("No configuration loaded", id="config-status")
                
                yield Rule()
                yield Button("Load Configuration", id="load-config")
                yield Button("Create New Configuration", id="new-config")
                yield Button("Help & Documentation", id="help")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Handle app startup."""
        # Try to load default config
        default_config_paths = [
            Path("config.yaml"),
            Path("gitflow.yaml"),
            Path(".gitflow/config.yaml")
        ]
        
        for config_path in default_config_paths:
            if config_path.exists():
                try:
                    self.config = ConfigLoader.load(config_path)
                    self.config_path = config_path
                    self.refresh()
                    self.notify(f"Loaded configuration from {config_path}", severity="info")
                    break
                except Exception as e:
                    self.notify(f"Failed to load {config_path}: {e}", severity="warning")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "run-analysis":
            self._run_analysis()
        elif event.button.id == "load-config":
            self._load_configuration()
        elif event.button.id == "new-config":
            self._new_configuration()
        elif event.button.id == "cache-status":
            self._show_cache_status()
        elif event.button.id == "manage-identities":
            self._manage_identities()
        elif event.button.id == "help":
            self._show_help()
    
    def _run_analysis(self) -> None:
        """Start the analysis process."""
        if not self.config:
            self.notify("Please load or create a configuration first", severity="error")
            return
        
        # Show analysis progress screen
        self.push_screen(AnalysisProgressScreen(self.config))
    
    def _load_configuration(self) -> None:
        """Load configuration from file."""
        def config_loaded(config: Optional[Config]) -> None:
            if config:
                self.config = config
                self.refresh()
                self.notify("Configuration loaded successfully", severity="success")
        
        self.push_screen(ConfigurationScreen(), config_loaded)
    
    def _new_configuration(self) -> None:
        """Create new configuration."""
        def config_created(config: Optional[Config]) -> None:
            if config:
                self.config = config
                self.refresh()
                self.notify("Configuration created successfully", severity="success")
        
        self.push_screen(ConfigurationScreen(), config_created)
    
    def _show_cache_status(self) -> None:
        """Show cache status and statistics."""
        if not self.config:
            self.notify("No configuration loaded", severity="error")
            return
        
        try:
            cache = GitAnalysisCache(self.config.cache.directory)
            stats = cache.get_cache_stats()
            
            message = f"""Cache Statistics:
• Cached commits: {stats['cached_commits']}
• Cached PRs: {stats['cached_prs']}  
• Cached issues: {stats['cached_issues']}
• Stale entries: {stats['stale_commits']}
• Cache location: {self.config.cache.directory}"""
            
            self.notify(message, severity="info")
            
        except Exception as e:
            self.notify(f"Failed to get cache stats: {e}", severity="error")
    
    def _manage_identities(self) -> None:
        """Manage developer identities."""
        # TODO: Implement identity management screen
        self.notify("Identity management coming soon!", severity="info")
    
    def _show_help(self) -> None:
        """Show help and documentation."""
        help_text = """GitFlow Analytics Help

Key Bindings:
• Ctrl+Q: Quit application
• Ctrl+N: Start new analysis  
• Ctrl+O: Open configuration
• F1: Show this help
• Escape: Go back/cancel

Getting Started:
1. Load or create a configuration file
2. Set up API keys (GitHub, OpenRouter, JIRA)
3. Run analysis to process your repositories
4. View results and export reports

For more information, visit:
https://github.com/your-org/gitflow-analytics"""
        
        self.notify(help_text, severity="info")
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()
    
    def action_new_analysis(self) -> None:
        """Start new analysis."""
        self._run_analysis()
    
    def action_open_config(self) -> None:
        """Open configuration."""
        self._load_configuration()
    
    def action_help(self) -> None:
        """Show help."""
        self._show_help()


def main():
    """Main entry point."""
    app = GitFlowAnalyticsApp()
    app.run()


if __name__ == "__main__":
    main()
