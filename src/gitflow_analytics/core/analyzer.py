"""Git repository analyzer with batch processing support."""
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Generator
from pathlib import Path
import git
from git import Repo
from tqdm import tqdm

from .cache import GitAnalysisCache
from ..extractors.story_points import StoryPointExtractor
from ..extractors.tickets import TicketExtractor
from .branch_mapper import BranchToProjectMapper


class GitAnalyzer:
    """Analyze Git repositories with caching and batch processing."""
    
    def __init__(self, cache: GitAnalysisCache, batch_size: int = 1000, 
                 branch_mapping_rules: Optional[Dict[str, List[str]]] = None):
        """Initialize analyzer with cache."""
        self.cache = cache
        self.batch_size = batch_size
        self.story_point_extractor = StoryPointExtractor()
        self.ticket_extractor = TicketExtractor()
        self.branch_mapper = BranchToProjectMapper(branch_mapping_rules)
    
    def analyze_repository(self, repo_path: Path, since: datetime, 
                         branch: Optional[str] = None) -> List[Dict[str, Any]]:
        """Analyze a Git repository with batch processing."""
        try:
            repo = Repo(repo_path)
        except Exception as e:
            raise ValueError(f"Failed to open repository at {repo_path}: {e}")
        
        # Get commits to analyze
        commits = self._get_commits(repo, since, branch)
        total_commits = len(commits)
        
        if total_commits == 0:
            return []
        
        analyzed_commits = []
        
        # Process in batches with progress bar
        with tqdm(total=total_commits, desc=f"Analyzing {repo_path.name}") as pbar:
            for batch in self._batch_commits(commits, self.batch_size):
                batch_results = self._process_batch(repo, repo_path, batch)
                analyzed_commits.extend(batch_results)
                
                # Cache the batch
                self.cache.cache_commits_batch(str(repo_path), batch_results)
                
                pbar.update(len(batch))
        
        return analyzed_commits
    
    def _get_commits(self, repo: Repo, since: datetime, 
                    branch: Optional[str] = None) -> List[git.Commit]:
        """Get commits from repository."""
        if branch:
            try:
                commits = list(repo.iter_commits(branch, since=since))
            except git.GitCommandError:
                # Branch doesn't exist
                return []
        else:
            # Get commits from all branches
            commits = []
            for ref in repo.refs:
                if ref.name.startswith('origin/'):
                    continue  # Skip remote branches
                try:
                    branch_commits = list(repo.iter_commits(ref, since=since))
                    commits.extend(branch_commits)
                except git.GitCommandError:
                    continue
            
            # Remove duplicates while preserving order
            seen = set()
            unique_commits = []
            for commit in commits:
                if commit.hexsha not in seen:
                    seen.add(commit.hexsha)
                    unique_commits.append(commit)
            
            commits = unique_commits
        
        # Sort by date
        return sorted(commits, key=lambda c: c.committed_datetime)
    
    def _batch_commits(self, commits: List[git.Commit], 
                      batch_size: int) -> Generator[List[git.Commit], None, None]:
        """Yield batches of commits."""
        for i in range(0, len(commits), batch_size):
            yield commits[i:i + batch_size]
    
    def _process_batch(self, repo: Repo, repo_path: Path, 
                      commits: List[git.Commit]) -> List[Dict[str, Any]]:
        """Process a batch of commits."""
        results = []
        
        for commit in commits:
            # Check cache first
            cached = self.cache.get_cached_commit(str(repo_path), commit.hexsha)
            if cached:
                results.append(cached)
                continue
            
            # Analyze commit
            commit_data = self._analyze_commit(repo, commit, repo_path)
            results.append(commit_data)
        
        return results
    
    def _analyze_commit(self, repo: Repo, commit: git.Commit, repo_path: Path) -> Dict[str, Any]:
        """Analyze a single commit."""
        # Basic commit data
        commit_data = {
            'hash': commit.hexsha,
            'author_name': commit.author.name,
            'author_email': commit.author.email,
            'message': commit.message,
            'timestamp': commit.committed_datetime,
            'is_merge': len(commit.parents) > 1
        }
        
        # Get branch name
        commit_data['branch'] = self._get_commit_branch(repo, commit)
        
        # Map branch to project
        commit_data['inferred_project'] = self.branch_mapper.map_branch_to_project(
            commit_data['branch'], repo_path
        )
        
        # Calculate metrics
        stats = commit.stats.total
        commit_data['files_changed'] = stats.get('files', 0)
        commit_data['insertions'] = stats.get('insertions', 0)
        commit_data['deletions'] = stats.get('deletions', 0)
        
        # Extract story points
        commit_data['story_points'] = self.story_point_extractor.extract_from_text(
            commit.message
        )
        
        # Extract ticket references
        commit_data['ticket_references'] = self.ticket_extractor.extract_from_text(
            commit.message
        )
        
        # Calculate complexity delta
        commit_data['complexity_delta'] = self._calculate_complexity_delta(commit)
        
        return commit_data
    
    def _get_commit_branch(self, repo: Repo, commit: git.Commit) -> str:
        """Get the branch name for a commit."""
        # This is a simplified approach - getting the first branch that contains the commit
        for branch in repo.branches:
            if commit in repo.iter_commits(branch):
                return branch.name
        return 'unknown'
    
    def _calculate_complexity_delta(self, commit: git.Commit) -> float:
        """Calculate complexity change for a commit."""
        total_delta = 0.0
        
        for diff in commit.diff(commit.parents[0] if commit.parents else None):
            if not self._is_code_file(diff.b_path or diff.a_path or ''):
                continue
            
            # Simple complexity estimation based on diff size
            # In a real implementation, you'd parse the code and calculate cyclomatic complexity
            if diff.new_file:
                total_delta += diff.b_blob.size / 100 if diff.b_blob else 0
            elif diff.deleted_file:
                total_delta -= diff.a_blob.size / 100 if diff.a_blob else 0
            else:
                # Modified file - estimate based on change size
                added = len(diff.diff.decode('utf-8', errors='ignore').split('\n+')) if diff.diff else 0
                removed = len(diff.diff.decode('utf-8', errors='ignore').split('\n-')) if diff.diff else 0
                total_delta += (added - removed) / 10
        
        return total_delta
    
    def _is_code_file(self, filepath: str) -> bool:
        """Check if file is a code file."""
        code_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
            '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
            '.cs', '.vb', '.r', '.m', '.mm', '.f90', '.f95', '.lua'
        }
        
        return any(filepath.endswith(ext) for ext in code_extensions)