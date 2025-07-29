"""Ticket reference extraction for multiple platforms."""
import re
from typing import List, Dict, Any, Set
from collections import defaultdict


class TicketExtractor:
    """Extract ticket references from various issue tracking systems."""
    
    def __init__(self):
        """Initialize with patterns for different platforms."""
        self.patterns = {
            'jira': [
                r'([A-Z]{2,10}-\d+)',  # Standard JIRA format: PROJ-123
            ],
            'github': [
                r'#(\d+)',              # GitHub issues: #123
                r'GH-(\d+)',            # Alternative format: GH-123
                r'(?:fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved)\s+#(\d+)',
            ],
            'clickup': [
                r'CU-([a-z0-9]+)',      # ClickUp: CU-abc123
                r'#([a-z0-9]{6,})',     # ClickUp short format
            ],
            'linear': [
                r'([A-Z]{2,5}-\d+)',    # Linear: ENG-123, similar to JIRA
                r'LIN-(\d+)',           # Alternative: LIN-123
            ]
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        for platform, patterns in self.patterns.items():
            self.compiled_patterns[platform] = [
                re.compile(pattern, re.IGNORECASE if platform != 'jira' else 0) 
                for pattern in patterns
            ]
    
    def extract_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract all ticket references from text."""
        if not text:
            return []
        
        tickets = []
        seen = set()  # Avoid duplicates
        
        for platform, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                for match in matches:
                    ticket_id = match if isinstance(match, str) else match[0]
                    
                    # Normalize ticket ID
                    if platform == 'jira' or platform == 'linear':
                        ticket_id = ticket_id.upper()
                    
                    # Create unique key
                    key = f"{platform}:{ticket_id}"
                    if key not in seen:
                        seen.add(key)
                        tickets.append({
                            'platform': platform,
                            'id': ticket_id,
                            'full_id': self._format_ticket_id(platform, ticket_id)
                        })
        
        return tickets
    
    def extract_by_platform(self, text: str) -> Dict[str, List[str]]:
        """Extract tickets grouped by platform."""
        tickets = self.extract_from_text(text)
        
        by_platform = defaultdict(list)
        for ticket in tickets:
            by_platform[ticket['platform']].append(ticket['id'])
        
        return dict(by_platform)
    
    def analyze_ticket_coverage(self, commits: List[Dict[str, Any]], 
                               prs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze ticket reference coverage across commits and PRs."""
        results = {
            'total_commits': len(commits),
            'total_prs': len(prs),
            'commits_with_tickets': 0,
            'prs_with_tickets': 0,
            'ticket_platforms': defaultdict(int),
            'untracked_commits': [],
            'ticket_summary': defaultdict(set)
        }
        
        # Analyze commits
        for commit in commits:
            ticket_refs = commit.get('ticket_references', [])
            if ticket_refs:
                results['commits_with_tickets'] += 1
                for ticket in ticket_refs:
                    if isinstance(ticket, dict):
                        platform = ticket.get('platform', 'unknown')
                        ticket_id = ticket.get('id', '')
                    else:
                        # Legacy format - assume JIRA
                        platform = 'jira'
                        ticket_id = ticket
                    
                    results['ticket_platforms'][platform] += 1
                    results['ticket_summary'][platform].add(ticket_id)
            else:
                # Track significant untracked commits
                if (not commit.get('is_merge') and 
                    commit.get('files_changed', 0) > 3):
                    results['untracked_commits'].append({
                        'hash': commit['hash'][:7],
                        'message': commit['message'].split('\n')[0][:60],
                        'files_changed': commit.get('files_changed', 0)
                    })
        
        # Analyze PRs
        for pr in prs:
            # Extract tickets from PR title and description
            pr_text = f"{pr.get('title', '')} {pr.get('description', '')}"
            tickets = self.extract_from_text(pr_text)
            
            if tickets:
                results['prs_with_tickets'] += 1
                for ticket in tickets:
                    platform = ticket['platform']
                    results['ticket_platforms'][platform] += 1
                    results['ticket_summary'][platform].add(ticket['id'])
        
        # Calculate coverage percentages
        results['commit_coverage_pct'] = (
            results['commits_with_tickets'] / results['total_commits'] * 100
            if results['total_commits'] > 0 else 0
        )
        
        results['pr_coverage_pct'] = (
            results['prs_with_tickets'] / results['total_prs'] * 100
            if results['total_prs'] > 0 else 0
        )
        
        # Convert sets to counts for summary
        results['ticket_summary'] = {
            platform: len(tickets) 
            for platform, tickets in results['ticket_summary'].items()
        }
        
        return results
    
    def _format_ticket_id(self, platform: str, ticket_id: str) -> str:
        """Format ticket ID for display."""
        if platform == 'github':
            return f"#{ticket_id}"
        elif platform == 'clickup':
            return f"CU-{ticket_id}" if not ticket_id.startswith('CU-') else ticket_id
        else:
            return ticket_id