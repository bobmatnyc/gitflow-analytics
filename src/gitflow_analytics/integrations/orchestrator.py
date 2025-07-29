"""Integration orchestrator for multiple platforms."""
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from ..core.cache import GitAnalysisCache
from .github_integration import GitHubIntegration


class IntegrationOrchestrator:
    """Orchestrate integrations with multiple platforms."""
    
    def __init__(self, config: Any, cache: GitAnalysisCache):
        """Initialize integration orchestrator."""
        self.config = config
        self.cache = cache
        self.integrations = {}
        
        # Initialize available integrations
        if config.github and config.github.token:
            self.integrations['github'] = GitHubIntegration(
                config.github.token,
                cache,
                config.github.max_retries,
                config.github.backoff_factor
            )
    
    def enrich_repository_data(self, repo_config: Any, commits: List[Dict[str, Any]], 
                             since: datetime) -> Dict[str, Any]:
        """Enrich repository data from all available integrations."""
        enrichment = {
            'prs': [],
            'issues': [],
            'pr_metrics': {}
        }
        
        # GitHub enrichment
        if 'github' in self.integrations and repo_config.github_repo:
            github = self.integrations['github']
            
            try:
                # Get PR data
                prs = github.enrich_repository_with_prs(
                    repo_config.github_repo, commits, since
                )
                enrichment['prs'] = prs
                
                # Calculate PR metrics
                if prs:
                    enrichment['pr_metrics'] = github.calculate_pr_metrics(prs)
                    
            except Exception as e:
                print(f"   ⚠️  GitHub enrichment failed: {e}")
        
        # Future: Add other platform integrations here
        # - ClickUp
        # - JIRA
        # - Linear
        
        return enrichment
    
    def get_platform_issues(self, project_key: str, since: datetime) -> List[Dict[str, Any]]:
        """Get issues from all configured platforms."""
        all_issues = []
        
        # Check cache first
        cached_issues = []
        for platform in ['github', 'jira', 'clickup', 'linear']:
            cached = self.cache.get_cached_issues(platform, project_key)
            cached_issues.extend(cached)
        
        if cached_issues:
            return cached_issues
        
        # Future: Fetch from APIs if not cached
        # This is where we'd add actual API calls to each platform
        
        return all_issues
    
    def export_to_json(self, 
                      commits: List[Dict[str, Any]],
                      prs: List[Dict[str, Any]],
                      developer_stats: List[Dict[str, Any]],
                      project_metrics: Dict[str, Any],
                      dora_metrics: Dict[str, Any],
                      output_path: str) -> str:
        """Export all data to JSON format for API consumption."""
        
        # Prepare data for JSON serialization
        def serialize_dates(obj):
            """Convert datetime objects to ISO format strings."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: serialize_dates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_dates(item) for item in obj]
            return obj
        
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '1.0',
                'total_commits': len(commits),
                'total_prs': len(prs),
                'total_developers': len(developer_stats)
            },
            'commits': serialize_dates(commits),
            'pull_requests': serialize_dates(prs),
            'developers': serialize_dates(developer_stats),
            'project_metrics': serialize_dates(project_metrics),
            'dora_metrics': serialize_dates(dora_metrics)
        }
        
        # Write JSON file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return output_path