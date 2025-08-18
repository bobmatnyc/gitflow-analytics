"""Analysis progress screen for GitFlow Analytics TUI."""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from rich.pretty import Pretty
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, Log

from gitflow_analytics.config import Config
from gitflow_analytics.core.analyzer import GitAnalyzer
from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.core.identity import DeveloperIdentityResolver
from gitflow_analytics.integrations.orchestrator import IntegrationOrchestrator

from ..widgets.progress_widget import AnalysisProgressWidget


class AnalysisProgressScreen(Screen):
    """
    Screen showing real-time analysis progress with detailed status updates.

    WHY: Long-running analysis operations require comprehensive progress feedback
    to keep users informed and allow them to monitor the process. This screen
    provides real-time updates on all phases of analysis.

    DESIGN DECISION: Uses multiple progress widgets to show different phases
    independently, allowing users to understand which part of the analysis is
    currently running and estimated completion times for each phase.
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel", "Cancel Analysis"),
        Binding("escape", "back", "Back to Main"),
        Binding("ctrl+l", "toggle_log", "Toggle Log"),
    ]

    def __init__(
        self,
        config: Config,
        weeks: int = 12,
        enable_qualitative: bool = True,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, id=id)
        self.config = config
        self.weeks = weeks
        self.enable_qualitative = enable_qualitative
        self.analysis_task: Optional[asyncio.Task] = None
        self.analysis_results = {}
        self.start_time = time.time()

    def compose(self):
        """Compose the analysis progress screen."""
        yield Header()

        with Container(id="progress-container"):
            yield Label("GitFlow Analytics - Analysis in Progress", classes="screen-title")

            # Progress panels for different phases
            with Vertical(id="progress-panels"):
                yield AnalysisProgressWidget("Overall Progress", total=100.0, id="overall-progress")

                yield AnalysisProgressWidget("Repository Analysis", total=100.0, id="repo-progress")

                yield AnalysisProgressWidget(
                    "Integration Data", total=100.0, id="integration-progress"
                )

                if self.enable_qualitative:
                    yield AnalysisProgressWidget(
                        "Qualitative Analysis", total=100.0, id="qual-progress"
                    )

                # Live statistics panel
                with Container(classes="stats-panel"):
                    yield Label("Live Statistics", classes="panel-title")
                    yield Pretty({}, id="live-stats")

            # Analysis log
            with Container(classes="log-panel"):
                yield Label("Analysis Log", classes="panel-title")
                yield Log(auto_scroll=True, id="analysis-log")

        yield Footer()

    def on_mount(self) -> None:
        """Start analysis when screen mounts."""
        self.analysis_task = asyncio.create_task(self._run_analysis())

    async def _run_analysis(self) -> None:
        """
        Run the complete analysis pipeline with progress updates.

        WHY: Implements the full analysis workflow with detailed progress tracking
        and error handling, ensuring users receive comprehensive feedback about
        the analysis process.
        """
        log = self.query_one("#analysis-log", Log)
        overall_progress = self.query_one("#overall-progress", AnalysisProgressWidget)

        try:
            log.write_line("🚀 Starting GitFlow Analytics...")

            # Phase 1: Initialize components (10%)
            overall_progress.update_progress(5, "Initializing components...")
            await self._initialize_components(log)
            overall_progress.update_progress(10, "Components initialized")

            # Phase 2: Repository discovery (20%)
            overall_progress.update_progress(10, "Discovering repositories...")
            repositories = await self._discover_repositories(log)
            overall_progress.update_progress(20, f"Found {len(repositories)} repositories")

            # Phase 3: Repository analysis (50%)
            overall_progress.update_progress(20, "Analyzing repositories...")
            commits, prs = await self._analyze_repositories(repositories, log)
            overall_progress.update_progress(50, f"Analyzed {len(commits)} commits")

            # Phase 4: Integration enrichment (70%)
            overall_progress.update_progress(50, "Enriching with integration data...")
            await self._enrich_with_integrations(repositories, commits, log)
            overall_progress.update_progress(70, "Integration data complete")

            # Phase 5: Identity resolution (80%)
            overall_progress.update_progress(70, "Resolving developer identities...")
            developer_stats = await self._resolve_identities(commits, log)
            overall_progress.update_progress(80, f"Identified {len(developer_stats)} developers")

            # Phase 6: Qualitative analysis (95%)
            if self.enable_qualitative:
                overall_progress.update_progress(80, "Running qualitative analysis...")
                await self._run_qualitative_analysis(commits, log)
                overall_progress.update_progress(95, "Qualitative analysis complete")

            # Phase 7: Finalization (100%)
            overall_progress.update_progress(95, "Finalizing results...")
            self.analysis_results = {
                "commits": commits,
                "prs": prs,
                "developers": developer_stats,
                "repositories": repositories,
            }

            overall_progress.complete("Analysis complete!")

            total_time = time.time() - self.start_time
            log.write_line(f"🎉 Analysis completed in {total_time:.1f} seconds!")
            log.write_line(f"   - Total commits: {len(commits):,}")
            log.write_line(f"   - Total PRs: {len(prs):,}")
            log.write_line(f"   - Active developers: {len(developer_stats):,}")

            # Switch to results screen after brief pause
            await asyncio.sleep(2)
            from .results_screen import ResultsScreen

            self.app.push_screen(
                ResultsScreen(
                    commits=commits, prs=prs, developers=developer_stats, config=self.config
                )
            )

        except asyncio.CancelledError:
            log.write_line("❌ Analysis cancelled by user")
            overall_progress.update_progress(0, "Cancelled")
        except Exception as e:
            log.write_line(f"❌ Analysis failed: {e}")
            overall_progress.update_progress(0, f"Error: {str(e)[:50]}...")
            self.notify(f"Analysis failed: {e}", severity="error")

    async def _initialize_components(self, log: Log) -> None:
        """Initialize analysis components."""
        log.write_line("📋 Initializing cache...")

        self.cache = GitAnalysisCache(
            self.config.cache.directory, ttl_hours=self.config.cache.ttl_hours
        )

        log.write_line("👥 Initializing identity resolver...")
        self.identity_resolver = DeveloperIdentityResolver(
            self.config.cache.directory / "identities.db",
            similarity_threshold=self.config.analysis.similarity_threshold,
            manual_mappings=self.config.analysis.manual_identity_mappings,
        )

        log.write_line("🔍 Initializing analyzer...")
        self.analyzer = GitAnalyzer(
            self.cache,
            branch_mapping_rules=self.config.analysis.branch_mapping_rules,
            allowed_ticket_platforms=getattr(self.config.analysis, "ticket_platforms", None),
            exclude_paths=self.config.analysis.exclude_paths,
            story_point_patterns=self.config.analysis.story_point_patterns,
        )

        log.write_line("🔗 Initializing integrations...")
        self.orchestrator = IntegrationOrchestrator(self.config, self.cache)

        # Check if we have pre-loaded NLP engine from startup
        if hasattr(self.app, "get_nlp_engine") and self.app.get_nlp_engine():
            log.write_line("✅ NLP engine already loaded from startup")
        elif self.enable_qualitative:
            log.write_line("⚠️ NLP engine will be loaded during qualitative analysis phase")

        # Small delay to show progress
        await asyncio.sleep(0.5)

    async def _discover_repositories(self, log: Log) -> list:
        """Discover repositories to analyze."""
        repositories = self.config.repositories

        if self.config.github.organization and not repositories:
            log.write_line(
                f"🔍 Discovering repositories from organization: {self.config.github.organization}"
            )

            try:
                # Use config directory for cloned repos
                config_dir = Path.cwd()  # TODO: Get actual config directory
                repos_dir = config_dir / "repos"

                discovered_repos = self.config.discover_organization_repositories(
                    clone_base_path=repos_dir
                )
                repositories = discovered_repos

                for repo in repositories:
                    log.write_line(f"   📁 {repo.name} ({repo.github_repo})")

            except Exception as e:
                log.write_line(f"   ❌ Repository discovery failed: {e}")
                raise

        await asyncio.sleep(0.5)  # Brief pause for UI updates
        return repositories

    async def _analyze_repositories(self, repositories: list, log: Log) -> tuple:
        """Analyze all repositories and return commits and PRs."""
        repo_progress = self.query_one("#repo-progress", AnalysisProgressWidget)

        all_commits = []
        all_prs = []

        # Analysis period (timezone-aware to match commit timestamps)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=self.weeks)

        for i, repo_config in enumerate(repositories):
            progress = (i / len(repositories)) * 100
            repo_progress.update_progress(progress, f"Analyzing {repo_config.name}...")

            log.write_line(f"📁 Analyzing {repo_config.name}...")

            try:
                # Clone repository if needed
                if not repo_config.path.exists() and repo_config.github_repo:
                    log.write_line(f"   📥 Cloning {repo_config.github_repo}...")
                    await self._clone_repository(repo_config, log)

                # Analyze commits
                commits = self.analyzer.analyze_repository(
                    repo_config.path, start_date, repo_config.branch
                )

                # Add project key and resolve identities
                for commit in commits:
                    commit["project_key"] = repo_config.project_key or commit.get(
                        "inferred_project", "UNKNOWN"
                    )
                    commit["canonical_id"] = self.identity_resolver.resolve_developer(
                        commit["author_name"], commit["author_email"]
                    )

                all_commits.extend(commits)
                log.write_line(f"   ✅ Found {len(commits)} commits")

                # Update live stats
                await self._update_live_stats(
                    {
                        "repositories_analyzed": i + 1,
                        "total_repositories": len(repositories),
                        "total_commits": len(all_commits),
                        "current_repo": repo_config.name,
                    }
                )

                # Small delay to allow UI updates
                await asyncio.sleep(0.1)

            except Exception as e:
                log.write_line(f"   ❌ Error analyzing {repo_config.name}: {e}")
                continue

        repo_progress.complete(f"Completed {len(repositories)} repositories")
        return all_commits, all_prs

    async def _enrich_with_integrations(self, repositories: list, commits: list, log: Log) -> None:
        """Enrich data with integration sources."""
        integration_progress = self.query_one("#integration-progress", AnalysisProgressWidget)

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=self.weeks)

        for i, repo_config in enumerate(repositories):
            progress = (i / len(repositories)) * 100
            integration_progress.update_progress(progress, f"Enriching {repo_config.name}...")

            try:
                # Get repository commits for this repo
                repo_commits = [c for c in commits if c.get("repository") == repo_config.name]

                enrichment = self.orchestrator.enrich_repository_data(
                    repo_config, repo_commits, start_date
                )

                if enrichment.get("prs"):
                    log.write_line(
                        f"   ✅ Found {len(enrichment['prs'])} pull requests for {repo_config.name}"
                    )

                await asyncio.sleep(0.1)

            except Exception as e:
                log.write_line(f"   ⚠️ Integration enrichment failed for {repo_config.name}: {e}")
                continue

        integration_progress.complete("Integration enrichment complete")

    async def _resolve_identities(self, commits: list, log: Log) -> list:
        """Resolve developer identities and return statistics."""
        log.write_line("👥 Updating developer statistics...")

        # Update commit statistics
        self.identity_resolver.update_commit_stats(commits)
        developer_stats = self.identity_resolver.get_developer_stats()

        log.write_line(f"   ✅ Resolved {len(developer_stats)} unique developer identities")

        # Show top contributors
        top_devs = sorted(developer_stats, key=lambda d: d["total_commits"], reverse=True)[:5]
        for dev in top_devs:
            log.write_line(f"   • {dev['primary_name']}: {dev['total_commits']} commits")

        await asyncio.sleep(0.5)
        return developer_stats

    async def _run_qualitative_analysis(self, commits: list, log: Log) -> None:
        """Run qualitative analysis if enabled."""
        if not self.enable_qualitative:
            return

        qual_progress = self.query_one("#qual-progress", AnalysisProgressWidget)

        try:
            log.write_line("🧠 Starting qualitative analysis...")

            # Check if NLP engine is pre-loaded from startup
            nlp_engine = None
            if hasattr(self.app, "get_nlp_engine"):
                nlp_engine = self.app.get_nlp_engine()

            if nlp_engine:
                log.write_line("   ✅ Using pre-loaded NLP engine")
                qual_processor = None  # We'll use the NLP engine directly
            else:
                log.write_line("   ⏳ Initializing qualitative processor...")
                # Import qualitative processor
                from gitflow_analytics.qualitative.core.processor import QualitativeProcessor

                qual_processor = QualitativeProcessor(self.config.qualitative)

                # Validate setup
                is_valid, issues = qual_processor.validate_setup()
                if not is_valid:
                    log.write_line("   ⚠️ Qualitative analysis setup issues:")
                    for issue in issues:
                        log.write_line(f"      - {issue}")
                    return

            # Process commits in batches
            batch_size = 100
            total_batches = (len(commits) + batch_size - 1) // batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(commits))
                batch = commits[start_idx:end_idx]

                progress = (batch_idx / total_batches) * 100
                qual_progress.update_progress(
                    progress, f"Processing batch {batch_idx + 1}/{total_batches}..."
                )

                # Convert to qualitative format
                qual_batch = []
                for commit in batch:
                    qual_commit = {
                        "hash": commit.get("hash"),
                        "message": commit.get("message"),
                        "author_name": commit.get("author_name"),
                        "author_email": commit.get("author_email"),
                        "timestamp": commit.get("timestamp"),
                        "files_changed": commit.get("files_changed", []),
                        "insertions": commit.get("insertions", 0),
                        "deletions": commit.get("deletions", 0),
                        "branch": commit.get("branch", "main"),
                    }
                    qual_batch.append(qual_commit)

                # Process batch using pre-loaded NLP engine or processor
                if nlp_engine:
                    # Use the pre-loaded NLP engine directly
                    results = nlp_engine.process_batch(qual_batch)
                else:
                    # Use the qualitative processor
                    results = qual_processor.process_commits(qual_batch, show_progress=False)

                # Update original commits with qualitative data
                for original, enhanced in zip(batch, results):
                    if hasattr(enhanced, "change_type"):
                        original["change_type"] = enhanced.change_type
                        original["business_domain"] = enhanced.business_domain
                        original["risk_level"] = enhanced.risk_level
                        original["confidence_score"] = enhanced.confidence_score

                await asyncio.sleep(0.1)  # Allow UI updates

            qual_progress.complete("Qualitative analysis complete")
            log.write_line("   ✅ Qualitative analysis completed")

        except ImportError:
            log.write_line("   ❌ Qualitative analysis dependencies not available")
            qual_progress.update_progress(0, "Dependencies missing")
        except Exception as e:
            log.write_line(f"   ❌ Qualitative analysis failed: {e}")
            qual_progress.update_progress(0, f"Error: {str(e)[:30]}...")

    async def _clone_repository(self, repo_config, log: Log) -> None:
        """Clone repository if needed."""
        try:
            import git

            repo_config.path.parent.mkdir(parents=True, exist_ok=True)

            clone_url = f"https://github.com/{repo_config.github_repo}.git"
            if self.config.github.token:
                clone_url = (
                    f"https://{self.config.github.token}@github.com/{repo_config.github_repo}.git"
                )

            # Try to clone with specified branch, fall back to default if it fails
            try:
                if repo_config.branch:
                    git.Repo.clone_from(clone_url, repo_config.path, branch=repo_config.branch)
                else:
                    git.Repo.clone_from(clone_url, repo_config.path)
            except git.GitCommandError as e:
                if repo_config.branch and "Remote branch" in str(e) and "not found" in str(e):
                    # Branch doesn't exist, try cloning without specifying branch
                    log.write_line(
                        f"   ⚠️  Branch '{repo_config.branch}' not found, using repository default"
                    )
                    git.Repo.clone_from(clone_url, repo_config.path)
                else:
                    raise
            log.write_line(f"   ✅ Successfully cloned {repo_config.github_repo}")

        except Exception as e:
            log.write_line(f"   ❌ Failed to clone {repo_config.github_repo}: {e}")
            raise

    async def _update_live_stats(self, stats: dict[str, Any]) -> None:
        """Update live statistics display."""
        stats_widget = self.query_one("#live-stats", Pretty)
        stats_widget.update(stats)

    def action_cancel(self) -> None:
        """Cancel the analysis."""
        if self.analysis_task and not self.analysis_task.done():
            self.analysis_task.cancel()
        self.app.pop_screen()

    def action_back(self) -> None:
        """Go back to main screen."""
        self.action_cancel()

    def action_toggle_log(self) -> None:
        """Toggle log panel visibility."""
        log_panel = self.query_one(".log-panel")
        log_panel.set_class(not log_panel.has_class("hidden"), "hidden")
