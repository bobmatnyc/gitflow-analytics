#!/usr/bin/env python3
"""Train commit classification model using EWTN JIRA data."""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from gitflow_analytics.config import ConfigLoader
from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.core.analyzer import GitAnalyzer
from gitflow_analytics.integrations.jira_integration import JIRAIntegration
from gitflow_analytics.classification.classifier import CommitClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load EWTN config
    config_path = Path.home() / "Clients/EWTN/gfa/config.yaml"
    config = ConfigLoader.load(config_path)
    
    # Setup cache
    cache = GitAnalysisCache(config.cache.directory, ttl_hours=config.cache.ttl_hours)
    
    # Initialize JIRA integration
    jira = JIRAIntegration(
        base_url=config.jira.base_url,
        username=config.jira.access_user,
        api_token=config.jira.access_token,
        cache=cache
    )
    
    # Test JIRA connection
    if not jira.validate_connection():
        logger.error("Failed to connect to JIRA")
        return
    
    logger.info("✅ JIRA connection successful")
    
    # Initialize analyzer
    analyzer = GitAnalyzer(
        cache,
        batch_size=getattr(config.analysis, 'batch_size', 1000),
        allowed_ticket_platforms=getattr(config.analysis, 'allowed_ticket_platforms', None)
    )
    
    # Get repositories
    repositories = config.repositories
    if config.github.organization and not repositories:
        logger.info(f"Discovering repositories from organization: {config.github.organization}")
        config_dir = config_path.parent
        repos_dir = config_dir / "repos"
        repositories = config.discover_organization_repositories(clone_base_path=repos_dir)
    
    # Analyze commits from last 24 weeks (6 months)
    since = datetime.now(timezone.utc) - timedelta(weeks=24)
    all_commits = []
    
    for repo in repositories[:10]:  # Analyze first 10 repos
        if not repo.path.exists():
            logger.warning(f"Repository not found: {repo.path}")
            continue
            
        logger.info(f"Analyzing {repo.name}...")
        commits = analyzer.analyze_repository(
            repo.path,
            since=since,
            branch=repo.branch
        )
        all_commits.extend(commits)
    
    logger.info(f"Found {len(all_commits)} total commits")
    
    # Filter commits with JIRA references
    jira_commits = [c for c in all_commits if any(
        ref.get('platform') == 'jira' for ref in c.get('ticket_references', [])
    )]
    
    logger.info(f"Found {len(jira_commits)} commits with JIRA references")
    
    # Enrich commits with JIRA data
    logger.info("Fetching JIRA ticket data...")
    jira.enrich_commits_with_jira_data(jira_commits)
    
    # Extract unique JIRA tickets and fetch full data including issue types
    ticket_map = {}
    import requests
    import base64
    
    credentials = base64.b64encode(f"{config.jira.access_user}:{config.jira.access_token}".encode()).decode()
    headers = {
        "Authorization": f"Basic {credentials}",
        "Accept": "application/json"
    }
    
    for commit in jira_commits:
        for ref in commit.get('ticket_references', []):
            if ref.get('platform') == 'jira':
                ticket_id = ref['id']
                if ticket_id not in ticket_map:
                    # Fetch full ticket details including issue type
                    try:
                        response = requests.get(
                            f"{config.jira.base_url}/rest/api/3/issue/{ticket_id}",
                            headers=headers
                        )
                        if response.status_code == 200:
                            issue_data = response.json()
                            fields = issue_data.get('fields', {})
                            ticket_map[ticket_id] = {
                                'id': ticket_id,
                                'summary': fields.get('summary', ''),
                                'status': fields.get('status', {}).get('name', ''),
                                'issue_type': fields.get('issuetype', {}).get('name', 'Task'),
                                'description': fields.get('description', ''),
                                'created': fields.get('created', ''),
                                'updated': fields.get('updated', '')
                            }
                    except Exception as e:
                        logger.warning(f"Failed to fetch {ticket_id}: {e}")
    
    logger.info(f"Fetched data for {len(ticket_map)} unique JIRA tickets")
    
    # Show issue type distribution
    issue_types = {}
    for ticket_id, ticket_data in ticket_map.items():
        issue_type = ticket_data.get('issue_type', 'Unknown')
        issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
    
    logger.info("\nIssue type distribution:")
    for issue_type, count in issue_types.items():
        logger.info(f"  {issue_type}: {count}")
    
    # Debug: Show first ticket structure and get actual issue type
    if ticket_map:
        # Let's fetch full ticket data to see the issue type
        first_ticket_id = list(ticket_map.keys())[0]
        logger.info(f"\nFetching full data for {first_ticket_id} to discover fields...")
        
        # Use JIRA API directly to get issue type
        import requests
        import base64
        
        credentials = base64.b64encode(f"{config.jira.access_user}:{config.jira.access_token}".encode()).decode()
        headers = {
            "Authorization": f"Basic {credentials}",
            "Accept": "application/json"
        }
        
        try:
            response = requests.get(
                f"{config.jira.base_url}/rest/api/3/issue/{first_ticket_id}",
                headers=headers
            )
            if response.status_code == 200:
                issue_data = response.json()
                issue_type = issue_data.get('fields', {}).get('issuetype', {})
                logger.info(f"Issue type: {issue_type.get('name', 'Unknown')}")
                logger.info(f"Issue type details: {issue_type}")
        except Exception as e:
            logger.error(f"Failed to fetch full issue data: {e}")
    
    # Map ticket types to training labels
    type_mapping = {
        'Bug': 'bugfix',
        'Story': 'feature',
        'Task': 'chore',
        'Epic': 'feature',
        'Sub-task': 'chore',
        'Subtask': 'chore',  # EWTN uses this
        'Historia': 'feature'  # EWTN custom type (Spanish for Story)
    }
    
    # Create training data
    training_data = []
    for commit in jira_commits:
        ticket_refs = [ref for ref in commit.get('ticket_references', []) if ref.get('platform') == 'jira']
        if ticket_refs:
            # Get first ticket's type
            ticket_id = ticket_refs[0]['id']
            if ticket_id in ticket_map:
                # Extract issue type from ticket data
                ticket_type = ticket_map[ticket_id].get('issue_type', 'Task')
                logger.debug(f"Ticket {ticket_id} has type: {ticket_type}")
                
                # Ensure commit has files_changed as a list
                if isinstance(commit.get('files_changed'), int):
                    # GitAnalyzer stores number of files, not the list
                    # Create empty list for now
                    commit['files_changed'] = []
                
                label = type_mapping.get(ticket_type, 'chore')
                training_data.append((commit, label))
    
    logger.info(f"Created {len(training_data)} training examples")
    
    if len(training_data) < 20:
        logger.error("Insufficient training data (need at least 20 examples)")
        return
    
    # Initialize classifier
    classifier = CommitClassifier(cache_dir=cache.cache_dir)
    
    # Train model
    logger.info("Training classification model...")
    results = classifier.train_model(
        training_data,
        validation_split=0.2
    )
    
    logger.info(f"✅ Training complete! Accuracy: {results.get('accuracy', 0):.2%}")
    
    # Show per-class metrics
    if 'class_metrics' in results:
        logger.info("\nPer-category performance:")
        for category, metrics in results['class_metrics'].items():
            logger.info(f"  {category}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}")

if __name__ == "__main__":
    main()