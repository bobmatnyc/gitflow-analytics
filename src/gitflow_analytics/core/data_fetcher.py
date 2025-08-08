"""Data fetcher for collecting raw git commits and ticket data without classification.

This module implements the first step of the two-step fetch/analyze process,
focusing purely on data collection from Git repositories and ticket systems
without performing any LLM-based classification.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

from ..extractors.story_points import StoryPointExtractor
from ..extractors.tickets import TicketExtractor
from ..integrations.jira_integration import JIRAIntegration
from ..models.database import (
    CachedCommit,
    CommitTicketCorrelation,
    DailyCommitBatch,
    Database,
    DetailedTicketData,
)
from .branch_mapper import BranchToProjectMapper
from .cache import GitAnalysisCache
from .identity import DeveloperIdentityResolver

logger = logging.getLogger(__name__)


class GitDataFetcher:
    """Fetches raw Git commit data and organizes it by day for efficient batch processing.
    
    WHY: This class implements the first step of the two-step process by collecting
    all raw data (commits, tickets, correlations) without performing classification.
    This separation enables:
    - Fast data collection without LLM costs
    - Repeatable analysis runs without re-fetching
    - Better batch organization for efficient LLM classification
    """

    def __init__(
        self,
        cache: GitAnalysisCache,
        branch_mapping_rules: Optional[Dict[str, List[str]]] = None,
        allowed_ticket_platforms: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
    ):
        """Initialize the data fetcher.
        
        Args:
            cache: Git analysis cache instance
            branch_mapping_rules: Rules for mapping branches to projects
            allowed_ticket_platforms: List of allowed ticket platforms
            exclude_paths: List of file paths to exclude from analysis
        """
        self.cache = cache
        self.database = Database(cache.cache_dir / "gitflow_cache.db")
        self.story_point_extractor = StoryPointExtractor()
        self.ticket_extractor = TicketExtractor(allowed_platforms=allowed_ticket_platforms)
        self.branch_mapper = BranchToProjectMapper(branch_mapping_rules)
        self.exclude_paths = exclude_paths or []
        
        # Initialize identity resolver
        identity_db_path = cache.cache_dir / "identities.db"
        self.identity_resolver = DeveloperIdentityResolver(identity_db_path)

    def fetch_repository_data(
        self,
        repo_path: Path,
        project_key: str,
        weeks_back: int = 4,
        branch_patterns: Optional[List[str]] = None,
        jira_integration: Optional[JIRAIntegration] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Fetch all data for a repository and organize by day.
        
        This method collects:
        1. All commits organized by day
        2. All referenced tickets with full metadata
        3. Commit-ticket correlations
        4. Developer identity mappings
        
        Args:
            repo_path: Path to the Git repository
            project_key: Project identifier
            weeks_back: Number of weeks to analyze
            branch_patterns: Branch patterns to include
            jira_integration: JIRA integration for ticket data
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing fetch results and statistics
        """
        logger.info(f"Starting data fetch for project {project_key} at {repo_path}")
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=weeks_back)
        
        # Step 1: Collect all commits organized by day
        logger.info("Fetching commits organized by day...")
        daily_commits = self._fetch_commits_by_day(
            repo_path, project_key, start_date, end_date, branch_patterns, progress_callback
        )
        
        # Step 2: Extract and fetch all referenced tickets
        logger.info("Extracting ticket references...")
        ticket_ids = self._extract_all_ticket_references(daily_commits)
        
        if jira_integration and ticket_ids:
            logger.info(f"Fetching {len(ticket_ids)} unique tickets from JIRA...")
            self._fetch_detailed_tickets(ticket_ids, jira_integration, project_key, progress_callback)
        
        # Step 3: Store commit-ticket correlations
        logger.info("Building commit-ticket correlations...")
        correlations_created = self._build_commit_ticket_correlations(daily_commits, repo_path)
        
        # Step 4: Store daily commit batches
        logger.info("Storing daily commit batches...")
        batches_created = self._store_daily_batches(daily_commits, repo_path, project_key)
        
        # Return summary statistics
        total_commits = sum(len(commits) for commits in daily_commits.values())
        
        results = {
            'project_key': project_key,
            'repo_path': str(repo_path),
            'date_range': {'start': start_date, 'end': end_date},
            'stats': {
                'total_commits': total_commits,
                'days_with_commits': len(daily_commits),
                'unique_tickets': len(ticket_ids),
                'correlations_created': correlations_created,
                'batches_created': batches_created,
            },
            'daily_commits': daily_commits,  # For immediate use if needed
        }
        
        logger.info(f"Data fetch completed for {project_key}: {total_commits} commits, {len(ticket_ids)} tickets")
        return results

    def _fetch_commits_by_day(
        self,
        repo_path: Path,
        project_key: str,
        start_date: datetime,
        end_date: datetime,
        branch_patterns: Optional[List[str]],
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all commits organized by day with full metadata.
        
        Returns:
            Dictionary mapping date strings (YYYY-MM-DD) to lists of commit data
        """
        from git import Repo
        
        try:
            repo = Repo(repo_path)
            # Update repository from remote before analysis
            self._update_repository(repo)
        except Exception as e:
            logger.error(f"Failed to open repository at {repo_path}: {e}")
            return {}
        
        # Collect commits from all relevant branches
        all_commits = []
        branches_to_analyze = self._get_branches_to_analyze(repo, branch_patterns)
        
        if not branches_to_analyze:
            logger.warning(f"No accessible branches found in repository {repo_path}")
            return {}
        
        logger.info(f"Analyzing branches: {branches_to_analyze}")
        
        for branch_name in branches_to_analyze:
            try:
                branch_commits = list(repo.iter_commits(
                    branch_name,
                    since=start_date,
                    until=end_date,
                    reverse=False
                ))
                
                logger.debug(f"Found {len(branch_commits)} commits in branch {branch_name} for date range")
                
                for commit in branch_commits:
                    # Include merge commits like the original analyzer
                    # The original analyzer marks merge commits with is_merge=True but doesn't skip them
                    
                    # Extract commit data with full metadata
                    commit_data = self._extract_commit_data(commit, branch_name, project_key, repo_path)
                    if commit_data:
                        all_commits.append(commit_data)
                        
            except Exception as e:
                logger.warning(f"Error processing branch {branch_name}: {e}")
                continue
        
        # Deduplicate commits (same commit may appear in multiple branches)
        seen_hashes = set()
        unique_commits = []
        for commit_data in all_commits:
            commit_hash = commit_data['commit_hash']
            if commit_hash not in seen_hashes:
                seen_hashes.add(commit_hash)
                unique_commits.append(commit_data)
        
        # Organize commits by day
        daily_commits = defaultdict(list)
        for commit_data in unique_commits:
            # Convert timestamp to date key
            commit_date = commit_data['timestamp'].date()
            date_key = commit_date.strftime('%Y-%m-%d')
            daily_commits[date_key].append(commit_data)
        
        # Sort commits within each day by timestamp
        for date_key in daily_commits:
            daily_commits[date_key].sort(key=lambda c: c['timestamp'])
        
        logger.info(f"Collected {len(unique_commits)} commits across {len(daily_commits)} days")
        return dict(daily_commits)

    def _extract_commit_data(
        self,
        commit: Any,
        branch_name: str,
        project_key: str,
        repo_path: Path
    ) -> Optional[Dict[str, Any]]:
        """Extract comprehensive data from a Git commit.
        
        Returns:
            Dictionary containing all commit metadata needed for classification
        """
        try:
            # Basic commit information
            commit_data = {
                'commit_hash': commit.hexsha,
                'commit_hash_short': commit.hexsha[:7],
                'message': commit.message.strip(),
                'author_name': commit.author.name,
                'author_email': commit.author.email,
                'timestamp': datetime.fromtimestamp(commit.committed_date, tz=timezone.utc),
                'branch': branch_name,
                'project_key': project_key,
                'repo_path': str(repo_path),
                'is_merge': len(commit.parents) > 1,  # Match the original analyzer behavior
            }
            
            # Calculate file changes
            try:
                if commit.parents:
                    # Compare with first parent
                    diff = commit.parents[0].diff(commit)
                else:
                    # Initial commit - compare with empty tree
                    diff = commit.diff(None)
                
                files_changed = []
                total_insertions = 0
                total_deletions = 0
                
                for diff_item in diff:
                    file_path = diff_item.a_path or diff_item.b_path
                    if file_path and not self._should_exclude_file(file_path):
                        files_changed.append(file_path)
                        
                        # Count line changes if available
                        if hasattr(diff_item, 'stats') and diff_item.stats:
                            total_insertions += diff_item.stats.insertions
                            total_deletions += diff_item.stats.deletions
                
                commit_data.update({
                    'files_changed': files_changed,
                    'files_changed_count': len(files_changed),
                    'lines_added': total_insertions,
                    'lines_deleted': total_deletions,
                })
                
            except Exception as e:
                logger.debug(f"Error calculating changes for commit {commit.hexsha}: {e}")
                commit_data.update({
                    'files_changed': [],
                    'files_changed_count': 0,
                    'lines_added': 0,
                    'lines_deleted': 0,
                })
            
            # Extract story points
            story_points = self.story_point_extractor.extract_from_text(commit_data['message'])
            commit_data['story_points'] = story_points
            
            # Extract ticket references
            ticket_refs_data = self.ticket_extractor.extract_from_text(commit_data['message'])
            # Convert to list of ticket IDs for compatibility
            # Fix: Use 'id' field instead of 'ticket_id' field from extractor output
            ticket_refs = [ref_data['id'] for ref_data in ticket_refs_data]
            commit_data['ticket_references'] = ticket_refs
            
            # Resolve developer identity
            canonical_id = self.identity_resolver.resolve_developer(
                commit_data['author_name'],
                commit_data['author_email']
            )
            commit_data['canonical_developer_id'] = canonical_id
            
            return commit_data
            
        except Exception as e:
            logger.error(f"Error extracting data for commit {commit.hexsha}: {e}")
            return None

    def _should_exclude_file(self, file_path: str) -> bool:
        """Check if a file should be excluded based on exclude patterns."""
        import fnmatch
        
        for pattern in self.exclude_paths:
            if fnmatch.fnmatch(file_path, pattern):
                return True
        return False

    def _get_branches_to_analyze(self, repo: Any, branch_patterns: Optional[List[str]]) -> List[str]:
        """Get list of branches to analyze based on patterns.
        
        WHY: Robust branch detection that handles missing remotes, missing default branches,
        and provides good fallback behavior. Based on the approach used in the existing analyzer.
        
        DESIGN DECISION: 
        - Try default branches first, fall back to all available branches
        - Handle missing remotes gracefully
        - Skip remote tracking branches to avoid duplicates
        - Use actual branch existence checking rather than assuming branches exist
        """
        if not branch_patterns:
            # Get all available branches (local branches preferred)
            available_branches = []
            
            # First, try local branches
            try:
                local_branches = [branch.name for branch in repo.branches]
                available_branches.extend(local_branches)
                logger.debug(f"Found local branches: {local_branches}")
            except Exception as e:
                logger.debug(f"Error getting local branches: {e}")
            
            # If we have remotes, also consider remote branches (but clean the names)
            try:
                if repo.remotes and hasattr(repo.remotes, 'origin'):
                    remote_branches = [
                        ref.name.replace('origin/', '') 
                        for ref in repo.remotes.origin.refs
                        if not ref.name.endswith('HEAD')  # Skip HEAD ref
                    ]
                    # Only add remote branches that aren't already in local branches
                    for branch in remote_branches:
                        if branch not in available_branches:
                            available_branches.append(branch)
                    logger.debug(f"Found remote branches: {remote_branches}")
            except Exception as e:
                logger.debug(f"Error getting remote branches: {e}")
            
            # If no branches found, fallback to trying common names directly
            if not available_branches:
                logger.warning("No branches found via normal detection, falling back to common names")
                available_branches = ['main', 'master', 'develop', 'dev']
            
            # Try default main branches first, in order of preference
            main_branches = ['main', 'master', 'develop', 'dev']
            for branch in main_branches:
                if branch in available_branches:
                    # Test that we can actually access this branch
                    try:
                        # Just try to get the commit object to verify branch exists and is accessible
                        next(iter(repo.iter_commits(branch, max_count=1)), None)
                        logger.info(f"Using main branch: {branch}")
                        return [branch]
                    except Exception as e:
                        logger.debug(f"Branch {branch} exists but not accessible: {e}")
                        continue
            
            # If no main branches work, try the first available branch that actually works
            for branch in available_branches:
                try:
                    next(iter(repo.iter_commits(branch, max_count=1)), None)
                    logger.info(f"Using fallback branch: {branch}")
                    return [branch]
                except Exception as e:
                    logger.debug(f"Branch {branch} not accessible: {e}")
                    continue
            
            # Last resort: return empty list (will be handled gracefully by caller)
            logger.warning("No accessible branches found")
            return []
        
        # Use specified patterns - match against all available branches
        import fnmatch
        available_branches = []
        
        # Collect all branches (local and remote)
        try:
            available_branches.extend([branch.name for branch in repo.branches])
        except Exception:
            pass
            
        try:
            if repo.remotes and hasattr(repo.remotes, 'origin'):
                remote_branches = [
                    ref.name.replace('origin/', '') 
                    for ref in repo.remotes.origin.refs
                    if not ref.name.endswith('HEAD')
                ]
                for branch in remote_branches:
                    if branch not in available_branches:
                        available_branches.append(branch)
        except Exception:
            pass
        
        # Match patterns against available branches
        matching_branches = []
        for pattern in branch_patterns:
            matching = [branch for branch in available_branches if fnmatch.fnmatch(branch, pattern)]
            matching_branches.extend(matching)
        
        # Test that matched branches are actually accessible
        accessible_branches = []
        for branch in list(set(matching_branches)):  # Remove duplicates
            try:
                next(iter(repo.iter_commits(branch, max_count=1)), None)
                accessible_branches.append(branch)
            except Exception as e:
                logger.debug(f"Matched branch {branch} not accessible: {e}")
        
        return accessible_branches

    def _update_repository(self, repo) -> bool:
        """Update repository from remote before analysis.
        
        WHY: This ensures we have the latest commits from the remote repository
        before performing analysis. Critical for getting accurate data especially
        when analyzing repositories that are actively being developed.
        
        DESIGN DECISION: Uses fetch() for all cases, then pull() only when on a
        tracking branch that's not in detached HEAD state. This approach:
        - Handles detached HEAD states gracefully (common in CI/CD)
        - Always gets latest refs from remote via fetch
        - Only attempts pull when it's safe to do so
        - Continues analysis even if update fails (logs warning)
        
        Args:
            repo: GitPython Repo object
            
        Returns:
            bool: True if update succeeded, False if failed (but analysis continues)
        """
        try:
            if repo.remotes:
                origin = repo.remotes.origin
                logger.info("Fetching latest changes from remote")
                origin.fetch()
                
                # Only try to pull if not in detached HEAD state
                if not repo.head.is_detached:
                    current_branch = repo.active_branch
                    tracking = current_branch.tracking_branch()
                    if tracking:
                        # Pull latest changes
                        origin.pull()
                        logger.debug(f"Pulled latest changes for {current_branch.name}")
                    else:
                        logger.debug(f"Branch {current_branch.name} has no tracking branch, skipping pull")
                else:
                    logger.debug("Repository in detached HEAD state, skipping pull")
                return True
            else:
                logger.debug("No remotes configured, skipping repository update")
                return True
        except Exception as e:
            logger.warning(f"Could not update repository: {e}")
            # Continue with analysis using local state
            return False

    def _extract_all_ticket_references(self, daily_commits: Dict[str, List[Dict[str, Any]]]) -> Set[str]:
        """Extract all unique ticket IDs from commits."""
        ticket_ids = set()
        
        for day_commits in daily_commits.values():
            for commit in day_commits:
                ticket_refs = commit.get('ticket_references', [])
                ticket_ids.update(ticket_refs)
        
        logger.info(f"Found {len(ticket_ids)} unique ticket references")
        return ticket_ids

    def _fetch_detailed_tickets(
        self,
        ticket_ids: Set[str],
        jira_integration: JIRAIntegration,
        project_key: str,
        progress_callback: Optional[callable] = None,
    ) -> None:
        """Fetch detailed ticket information and store in database."""
        session = self.database.get_session()
        
        try:
            # Check which tickets we already have
            existing_tickets = session.query(DetailedTicketData).filter(
                DetailedTicketData.ticket_id.in_(ticket_ids),
                DetailedTicketData.platform == 'jira'
            ).all()
            
            existing_ids = {ticket.ticket_id for ticket in existing_tickets}
            tickets_to_fetch = ticket_ids - existing_ids
            
            if not tickets_to_fetch:
                logger.info("All tickets already cached")
                return
            
            logger.info(f"Fetching {len(tickets_to_fetch)} new tickets")
            
            # Fetch tickets in batches
            batch_size = 50
            tickets_list = list(tickets_to_fetch)
            
            with tqdm(total=len(tickets_list), desc="Fetching tickets") as pbar:
                for i in range(0, len(tickets_list), batch_size):
                    batch = tickets_list[i:i + batch_size]
                    
                    for ticket_id in batch:
                        try:
                            # Fetch ticket from JIRA
                            issue_data = jira_integration.get_issue(ticket_id)
                            
                            if issue_data:
                                # Create detailed ticket record
                                detailed_ticket = self._create_detailed_ticket_record(
                                    issue_data, project_key, 'jira'
                                )
                                session.add(detailed_ticket)
                            
                        except Exception as e:
                            logger.warning(f"Failed to fetch ticket {ticket_id}: {e}")
                        
                        pbar.update(1)
                        
                        if progress_callback:
                            progress_callback(f"Fetched ticket {ticket_id}")
                    
                    # Commit batch to database
                    session.commit()
            
            logger.info(f"Successfully fetched {len(tickets_to_fetch)} tickets")
            
        except Exception as e:
            logger.error(f"Error fetching detailed tickets: {e}")
            session.rollback()
        finally:
            session.close()

    def _create_detailed_ticket_record(
        self,
        issue_data: Dict[str, Any],
        project_key: str,
        platform: str
    ) -> DetailedTicketData:
        """Create a detailed ticket record from JIRA issue data."""
        # Extract classification hints from issue type and labels
        classification_hints = []
        
        issue_type = issue_data.get('issue_type', '').lower()
        if 'bug' in issue_type or 'defect' in issue_type:
            classification_hints.append('bug_fix')
        elif 'story' in issue_type or 'feature' in issue_type:
            classification_hints.append('feature')
        elif 'task' in issue_type:
            classification_hints.append('maintenance')
        
        # Extract business domain from labels or summary
        business_domain = None
        labels = issue_data.get('labels', [])
        for label in labels:
            if any(keyword in label.lower() for keyword in ['frontend', 'backend', 'ui', 'api']):
                business_domain = label.lower()
                break
        
        # Create the record
        return DetailedTicketData(
            platform=platform,
            ticket_id=issue_data['key'],
            project_key=project_key,
            title=issue_data.get('summary', ''),
            description=issue_data.get('description', ''),
            summary=issue_data.get('summary', '')[:500],  # Truncated summary
            ticket_type=issue_data.get('issue_type', ''),
            status=issue_data.get('status', ''),
            priority=issue_data.get('priority', ''),
            labels=labels,
            assignee=issue_data.get('assignee', ''),
            reporter=issue_data.get('reporter', ''),
            created_at=issue_data.get('created'),
            updated_at=issue_data.get('updated'),
            resolved_at=issue_data.get('resolved'),
            story_points=issue_data.get('story_points'),
            classification_hints=classification_hints,
            business_domain=business_domain,
            platform_data=issue_data,  # Store full JIRA data
        )

    def _build_commit_ticket_correlations(
        self,
        daily_commits: Dict[str, List[Dict[str, Any]]],
        repo_path: Path
    ) -> int:
        """Build and store commit-ticket correlations."""
        session = self.database.get_session()
        correlations_created = 0
        
        try:
            for day_commits in daily_commits.values():
                for commit in day_commits:
                    commit_hash = commit['commit_hash']
                    ticket_refs = commit.get('ticket_references', [])
                    
                    for ticket_id in ticket_refs:
                        try:
                            # Create correlation record
                            correlation = CommitTicketCorrelation(
                                commit_hash=commit_hash,
                                repo_path=str(repo_path),
                                ticket_id=ticket_id,
                                platform='jira',  # Assuming JIRA for now
                                project_key=commit['project_key'],
                                correlation_type='direct',
                                confidence=1.0,
                                extracted_from='commit_message',
                                matching_pattern=None,  # Could add pattern detection
                            )
                            
                            # Check if correlation already exists
                            existing = session.query(CommitTicketCorrelation).filter(
                                CommitTicketCorrelation.commit_hash == commit_hash,
                                CommitTicketCorrelation.repo_path == str(repo_path),
                                CommitTicketCorrelation.ticket_id == ticket_id,
                                CommitTicketCorrelation.platform == 'jira'
                            ).first()
                            
                            if not existing:
                                session.add(correlation)
                                correlations_created += 1
                                
                        except Exception as e:
                            logger.warning(f"Failed to create correlation for {commit_hash}-{ticket_id}: {e}")
            
            session.commit()
            logger.info(f"Created {correlations_created} commit-ticket correlations")
            
        except Exception as e:
            logger.error(f"Error building correlations: {e}")
            session.rollback()
        finally:
            session.close()
        
        return correlations_created

    def _store_daily_batches(
        self,
        daily_commits: Dict[str, List[Dict[str, Any]]],
        repo_path: Path,
        project_key: str
    ) -> int:
        """Store daily commit batches for efficient retrieval."""
        session = self.database.get_session()
        batches_created = 0
        commits_stored = 0
        
        try:
            logger.info(f"Storing {sum(len(commits) for commits in daily_commits.values())} commits from {len(daily_commits)} days")
            for date_str, commits in daily_commits.items():
                if not commits:
                    continue
                
                logger.debug(f"Processing {len(commits)} commits for {date_str}")
                # Store individual commits in CachedCommit table
                for commit in commits:
                    try:
                        # Check if commit already exists
                        existing_commit = session.query(CachedCommit).filter(
                            CachedCommit.commit_hash == commit['commit_hash']
                        ).first()
                        
                        if not existing_commit:
                            # Create new cached commit
                            # Handle timestamp - it's already a datetime object from git
                            timestamp = commit['timestamp']
                            if isinstance(timestamp, str):
                                timestamp = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S%z')
                            
                            cached_commit = CachedCommit(
                                commit_hash=commit['commit_hash'],
                                repo_path=str(repo_path),
                                branch=commit.get('branch', 'main'),
                                author_name=commit.get('author_name', ''),
                                author_email=commit.get('author_email', ''),
                                timestamp=timestamp,
                                message=commit.get('message', ''),
                                files_changed=commit.get('files_changed_count', 0),
                                insertions=commit.get('lines_added', 0),
                                deletions=commit.get('lines_deleted', 0),
                                ticket_references=commit.get('ticket_references', []),
                                story_points=commit.get('story_points'),
                            )
                            session.add(cached_commit)
                            commits_stored += 1
                    except Exception as e:
                        logger.error(f"Failed to store commit {commit.get('commit_hash', 'unknown')[:7]}: {e}")
                
                # Calculate batch statistics
                total_files = sum(commit.get('files_changed_count', 0) for commit in commits)
                total_additions = sum(commit.get('lines_added', 0) for commit in commits)
                total_deletions = sum(commit.get('lines_deleted', 0) for commit in commits)
                
                # Get unique developers and tickets for this day
                active_devs = list(set(commit.get('canonical_developer_id', '') for commit in commits))
                unique_tickets = []
                for commit in commits:
                    unique_tickets.extend(commit.get('ticket_references', []))
                unique_tickets = list(set(unique_tickets))
                
                # Create context summary
                context_summary = f"{len(commits)} commits by {len(active_devs)} developers"
                if unique_tickets:
                    context_summary += f", {len(unique_tickets)} tickets referenced"
                
                # Check if batch already exists
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                existing_batch = session.query(DailyCommitBatch).filter(
                    DailyCommitBatch.date == date_obj,
                    DailyCommitBatch.project_key == project_key,
                    DailyCommitBatch.repo_path == str(repo_path)
                ).first()
                
                if existing_batch:
                    # Update existing batch
                    existing_batch.commit_count = len(commits)
                    existing_batch.total_files_changed = total_files
                    existing_batch.total_lines_added = total_additions
                    existing_batch.total_lines_deleted = total_deletions
                    existing_batch.active_developers = active_devs
                    existing_batch.unique_tickets = unique_tickets
                    existing_batch.context_summary = context_summary
                    existing_batch.fetched_at = datetime.utcnow()
                    existing_batch.classification_status = 'pending'
                else:
                    # Create new batch
                    batch = DailyCommitBatch(
                        date=date_obj,
                        project_key=project_key,
                        repo_path=str(repo_path),
                        commit_count=len(commits),
                        total_files_changed=total_files,
                        total_lines_added=total_additions,
                        total_lines_deleted=total_deletions,
                        active_developers=active_devs,
                        unique_tickets=unique_tickets,
                        context_summary=context_summary,
                        classification_status='pending',
                    )
                    session.add(batch)
                    batches_created += 1
            
            session.commit()
            logger.info(f"Created/updated {batches_created} daily commit batches, stored {commits_stored} commits")
            
        except Exception as e:
            logger.error(f"Error storing daily batches: {e}")
            session.rollback()
        finally:
            session.close()
        
        return batches_created

    def get_fetch_status(self, project_key: str, repo_path: Path) -> Dict[str, Any]:
        """Get status of data fetching for a project."""
        session = self.database.get_session()
        
        try:
            # Count daily batches
            batches = session.query(DailyCommitBatch).filter(
                DailyCommitBatch.project_key == project_key,
                DailyCommitBatch.repo_path == str(repo_path)
            ).all()
            
            # Count tickets
            tickets = session.query(DetailedTicketData).filter(
                DetailedTicketData.project_key == project_key
            ).count()
            
            # Count correlations
            correlations = session.query(CommitTicketCorrelation).filter(
                CommitTicketCorrelation.project_key == project_key,
                CommitTicketCorrelation.repo_path == str(repo_path)
            ).count()
            
            # Calculate statistics
            total_commits = sum(batch.commit_count for batch in batches)
            classified_batches = sum(1 for batch in batches if batch.classification_status == 'completed')
            
            return {
                'project_key': project_key,
                'repo_path': str(repo_path),
                'daily_batches': len(batches),
                'total_commits': total_commits,
                'unique_tickets': tickets,
                'commit_correlations': correlations,
                'classification_status': {
                    'completed_batches': classified_batches,
                    'pending_batches': len(batches) - classified_batches,
                    'completion_rate': classified_batches / len(batches) if batches else 0.0,
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting fetch status: {e}")
            return {}
        finally:
            session.close()