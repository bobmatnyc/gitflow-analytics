"""Analysis progress screen for GitFlow Analytics TUI."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, Log, Static

from gitflow_analytics.config import Config
from gitflow_analytics.core import progress as core_progress
from gitflow_analytics.core.analyzer import GitAnalyzer
from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.core.identity import DeveloperIdentityResolver
from gitflow_analytics.integrations.orchestrator import IntegrationOrchestrator

from ..progress_adapter import TUIProgressService
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
        self.progress_service = None  # Will be initialized on mount
        self.executor: Optional[ThreadPoolExecutor] = None  # Managed executor for cleanup

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
                    yield Static("No statistics yet...", id="live-stats")

            # Analysis log
            with Container(classes="log-panel"):
                yield Label("Analysis Log", classes="panel-title")
                yield Log(auto_scroll=True, id="analysis-log")

        yield Footer()

    def on_mount(self) -> None:
        """Start analysis when screen mounts."""
        # Initialize progress service for TUI
        self.progress_service = TUIProgressService(asyncio.get_event_loop())
        self.analysis_task = asyncio.create_task(self._run_analysis_wrapper())

    def on_unmount(self) -> None:
        """Cleanup when screen unmounts."""
        # Cancel the analysis task if it's still running
        if self.analysis_task and not self.analysis_task.done():
            self.analysis_task.cancel()
            # Don't wait for cancellation to complete to avoid blocking

        # Shutdown the executor to cleanup threads immediately
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None

    async def _run_analysis_wrapper(self) -> None:
        """Wrapper for analysis that handles cancellation gracefully."""
        try:
            await self._run_analysis()
        except asyncio.CancelledError:
            # Silently handle cancellation - this is expected during shutdown
            pass
        except Exception as e:
            # Log unexpected errors if the app is still running
            if self.app and self.app.is_running:
                try:
                    log = self.query_one("#analysis-log", Log)
                    log.write_line(f"❌ Unexpected error: {e}")
                except Exception:
                    pass

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
            # Check if the app is still running before updating UI
            if self.app and self.app.is_running:
                try:
                    log.write_line("❌ Analysis cancelled by user")
                    overall_progress.update_progress(0, "Cancelled")
                except Exception:
                    # Silently ignore if we can't update the UI
                    pass
            # Re-raise for the wrapper to handle
            raise
        except Exception as e:
            # Check if the app is still running before updating UI
            if self.app and self.app.is_running:
                try:
                    log.write_line(f"❌ Analysis failed: {e}")
                    overall_progress.update_progress(0, f"Error: {str(e)[:50]}...")
                    self.notify(f"Analysis failed: {e}", severity="error")
                except Exception:
                    # Silently ignore if we can't update the UI
                    pass

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

        # Enable branch analysis with progress logging for TUI
        branch_analysis_config = {
            "enable_progress_logging": True,
            "strategy": "all",
        }

        self.analyzer = GitAnalyzer(
            self.cache,
            branch_mapping_rules=self.config.analysis.branch_mapping_rules,
            allowed_ticket_platforms=getattr(self.config.analysis, "ticket_platforms", None),
            exclude_paths=self.config.analysis.exclude_paths,
            story_point_patterns=self.config.analysis.story_point_patterns,
            branch_analysis_config=branch_analysis_config,
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
        # Import progress module at the top of the function

        repo_progress = self.query_one("#repo-progress", AnalysisProgressWidget)
        overall_progress = self.query_one("#overall-progress", AnalysisProgressWidget)

        all_commits = []
        all_prs = []

        # Analysis period (timezone-aware to match commit timestamps)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=self.weeks)

        # Create progress adapter for repository analysis
        repo_adapter = self.progress_service.create_adapter("repo", repo_progress)

        # Set initial stats for the adapter
        repo_adapter.processing_stats["total"] = len(repositories)

        # Temporarily replace the global progress service with our adapter
        original_progress_service = core_progress._progress_service
        core_progress._progress_service = repo_adapter

        total_repos = len(repositories)

        # Clone repositories that don't exist locally first
        for repo_config in repositories:
            if not repo_config.path.exists() and repo_config.github_repo:
                log.write_line(f"   📥 Cloning {repo_config.github_repo}...")
                await self._clone_repository(repo_config, log)

        # Check if we should use async processing (for multiple repositories)
        # We use async processing for 2+ repositories to keep the UI responsive
        use_async = len(repositories) > 1

        if use_async:
            log.write_line(f"🚀 Starting async analysis of {len(repositories)} repositories...")

            # Import data fetcher for parallel processing
            from gitflow_analytics.core.data_fetcher import GitDataFetcher
            from gitflow_analytics.tui.progress_adapter import TUIProgressAdapter

            # Create and set up progress adapter for parallel processing
            tui_progress_adapter = TUIProgressAdapter(repo_progress)
            tui_progress_adapter.set_event_loop(asyncio.get_event_loop())

            # Replace the global progress service so parallel processing can use it
            # We'll restore the original one after processing
            core_progress._progress_service = tui_progress_adapter

            # Create data fetcher
            # Use skip_remote_fetch=True when analyzing already-cloned repositories
            # to avoid authentication issues with expired tokens
            data_fetcher = GitDataFetcher(cache=self.cache, skip_remote_fetch=True)

            # Prepare repository configurations for parallel processing
            repo_configs = []
            for repo_config in repositories:
                repo_configs.append(
                    {
                        "path": str(repo_config.path),
                        "project_key": repo_config.project_key or repo_config.name,
                        "branch_patterns": [repo_config.branch] if repo_config.branch else None,
                    }
                )

            # Run parallel processing in executor to avoid blocking
            loop = asyncio.get_event_loop()

            # Update overall progress
            overall_progress.update_progress(25, "Running parallel repository analysis...")

            try:
                # Process repositories asynchronously with yielding for UI updates
                parallel_results = await self._process_repositories_async(
                    data_fetcher,
                    repo_configs,
                    start_date,
                    end_date,
                    repo_progress,
                    overall_progress,
                    log,
                )

                # Process results
                for project_key, result in parallel_results["results"].items():
                    if result and "commits" in result:
                        commits_data = result["commits"]
                        # Add project key and resolve identities
                        for commit in commits_data:
                            commit["project_key"] = project_key
                            commit["canonical_id"] = self.identity_resolver.resolve_developer(
                                commit["author_name"], commit["author_email"]
                            )
                        all_commits.extend(commits_data)
                        log.write_line(f"   ✅ {project_key}: {len(commits_data)} commits")

                # Log final statistics
                stats = parallel_results.get("statistics", {})
                log.write_line("\n📊 Analysis Statistics:")
                log.write_line(f"   Total: {stats.get('total', 0)} repositories")
                log.write_line(f"   Success: {stats.get('success', 0)} (have commits)")
                log.write_line(
                    f"   No Commits: {stats.get('no_commits', 0)} (no activity in period)"
                )
                log.write_line(f"   Failed: {stats.get('failed', 0)} (processing errors)")
                log.write_line(f"   Timeout: {stats.get('timeout', 0)}")

            except Exception as e:
                log.write_line(f"   ❌ Async processing failed: {e}")
                log.write_line("   Falling back to sequential processing...")
                use_async = False
            finally:
                # Restore original progress service
                core_progress._progress_service = original_progress_service

        # Sequential processing fallback or for single repository
        if not use_async:
            # Ensure we have an executor for sequential processing
            if not self.executor:
                self.executor = ThreadPoolExecutor(max_workers=1)

            for i, repo_config in enumerate(repositories):
                # Update overall progress based on repository completion
                overall_pct = 20 + ((i / total_repos) * 30)  # 20-50% range for repo analysis
                overall_progress.update_progress(
                    overall_pct, f"Analyzing repositories ({i+1}/{total_repos})..."
                )

                repo_progress.update_progress(0, f"Analyzing {repo_config.name}...")

                log.write_line(f"📁 Analyzing {repo_config.name}...")

                try:
                    log.write_line(f"   ⏳ Starting analysis of {repo_config.name}...")

                    # Run repository analysis in a thread to avoid blocking
                    loop = asyncio.get_event_loop()
                    commits = await loop.run_in_executor(
                        (
                            self.executor if self.executor else None
                        ),  # Use managed executor if available
                        self.analyzer.analyze_repository,
                        repo_config.path,
                        start_date,
                        repo_config.branch,
                    )

                    log.write_line(f"   ✓ Analysis complete for {repo_config.name}")

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
                    await asyncio.sleep(0.05)  # Reduced delay for more responsive updates

                except Exception as e:
                    log.write_line(f"   ❌ Error analyzing {repo_config.name}: {e}")
                    continue

        # Restore original progress service
        core_progress._progress_service = original_progress_service

        repo_progress.complete(f"Completed {len(repositories)} repositories")
        overall_progress.update_progress(50, f"Analyzed {len(all_commits)} commits")
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

    async def _process_repositories_async(
        self,
        data_fetcher,
        repo_configs: list,
        start_date: datetime,
        end_date: datetime,
        repo_progress: AnalysisProgressWidget,
        overall_progress: AnalysisProgressWidget,
        log: Log,
    ) -> dict:
        """
        Process repositories asynchronously with proper yielding for UI updates.

        This method processes repositories one at a time but yields control back
        to the event loop between each repository to allow UI updates.
        """
        results = {
            "results": {},
            "statistics": {
                "total": len(repo_configs),
                "processed": 0,
                "success": 0,
                "no_commits": 0,
                "failed": 0,
                "timeout": 0,
            },
        }

        stats = results["statistics"]
        loop = asyncio.get_event_loop()

        # Create a managed executor for this analysis
        if not self.executor:
            self.executor = ThreadPoolExecutor(max_workers=1)

        for i, repo_config in enumerate(repo_configs):
            project_key = repo_config["project_key"]

            # Update progress before processing
            percentage = (i / stats["total"]) * 100
            repo_progress.update_progress(
                percentage, f"Processing {project_key} ({i+1}/{stats['total']})..."
            )

            # Update overall progress
            overall_percentage = 25 + ((i / stats["total"]) * 25)  # 25-50% range
            overall_progress.update_progress(
                overall_percentage, f"Analyzing repository {i+1}/{stats['total']}: {project_key}"
            )

            log.write_line(f"🔍 Processing {project_key} ({i+1}/{stats['total']})...")

            try:
                # Run the actual repository processing in a thread to avoid blocking
                # but await it properly so we can yield between repositories
                result = await loop.run_in_executor(
                    self.executor,  # Use managed executor instead of default
                    self._process_single_repository_sync,
                    data_fetcher,
                    repo_config,
                    self.weeks,
                    start_date,
                    end_date,
                )

                # Check for commits - data fetcher returns 'daily_commits' not 'commits'
                if result:
                    # Extract commits from daily_commits structure
                    daily_commits = result.get("daily_commits", {})
                    total_commits = result.get("stats", {}).get("total_commits", 0)

                    # Convert daily_commits to flat commits list
                    commits = []
                    for _date_str, day_commits in daily_commits.items():
                        commits.extend(day_commits)

                    # Add flattened commits to result for compatibility
                    result["commits"] = commits

                    if total_commits > 0 or commits:
                        results["results"][project_key] = result
                        stats["success"] += 1
                        log.write_line(f"   ✅ {project_key}: {total_commits} commits")
                    else:
                        stats["no_commits"] += 1
                        log.write_line(f"   ⏸️  {project_key}: No commits in analysis period")
                else:
                    stats["failed"] += 1
                    log.write_line(f"   ❌ {project_key}: Failed to process")

            except Exception as e:
                stats["failed"] += 1
                log.write_line(f"   ❌ {project_key}: Error - {str(e)[:50]}...")

            stats["processed"] += 1

            # Update progress after processing
            percentage = ((i + 1) / stats["total"]) * 100
            repo_progress.update_progress(
                percentage, f"Completed {project_key} ({i+1}/{stats['total']})"
            )

            # Yield control to event loop for UI updates
            # This is the key to keeping the UI responsive
            await asyncio.sleep(0.01)

            # Also update live stats
            await self._update_live_stats(
                {
                    "repositories_analyzed": stats["processed"],
                    "total_repositories": stats["total"],
                    "successful": stats["success"],
                    "no_commits": stats["no_commits"],
                    "failed": stats["failed"],
                    "current_repo": project_key if i < len(repo_configs) - 1 else "Complete",
                }
            )

        # Final progress update
        repo_progress.complete(f"Processed {stats['total']} repositories")

        # Cleanup executor after processing
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None

        return results

    def _process_single_repository_sync(
        self,
        data_fetcher,
        repo_config: dict,
        weeks_back: int,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[dict]:
        """
        Synchronous wrapper for processing a single repository.

        This runs in a thread executor to avoid blocking the event loop.
        """
        try:
            # Process the repository using data fetcher
            result = data_fetcher.fetch_repository_data(
                repo_path=Path(repo_config["path"]),
                project_key=repo_config["project_key"],
                weeks_back=weeks_back,
                branch_patterns=repo_config.get("branch_patterns"),
                jira_integration=None,
                progress_callback=None,
                start_date=start_date,
                end_date=end_date,
            )
            return result
        except Exception as e:
            import logging

            logging.getLogger(__name__).error(f"Error processing {repo_config['project_key']}: {e}")
            return None

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
        try:
            stats_widget = self.query_one("#live-stats", Static)

            # Format stats for display
            stats_text = "\n".join(
                [f"• {key.replace('_', ' ').title()}: {value}" for key, value in stats.items()]
            )
            stats_widget.update(stats_text)
        except Exception:
            # Silently ignore if widget doesn't exist (e.g., in testing)
            pass

    def action_cancel(self) -> None:
        """Cancel the analysis."""
        if self.analysis_task and not self.analysis_task.done():
            self.analysis_task.cancel()
            # Give the task a moment to cancel cleanly
            asyncio.create_task(self._delayed_pop_screen())
        else:
            self.app.pop_screen()

    async def _delayed_pop_screen(self) -> None:
        """Pop screen after a brief delay to allow cancellation to complete."""
        await asyncio.sleep(0.1)
        if self.app and self.app.is_running:
            self.app.pop_screen()

    def action_back(self) -> None:
        """Go back to main screen."""
        self.action_cancel()

    def action_toggle_log(self) -> None:
        """Toggle log panel visibility."""
        log_panel = self.query_one(".log-panel")
        log_panel.set_class(not log_panel.has_class("hidden"), "hidden")
