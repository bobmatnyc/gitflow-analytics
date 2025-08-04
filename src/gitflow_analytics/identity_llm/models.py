"""Data models for LLM-based identity analysis."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime


@dataclass
class DeveloperAlias:
    """Represents a single developer alias."""
    name: str
    email: str
    commit_count: int
    first_seen: datetime
    last_seen: datetime
    repositories: Set[str]
    
    
@dataclass
class DeveloperCluster:
    """Represents a cluster of related developer identities."""
    canonical_name: str
    canonical_email: str
    aliases: List[DeveloperAlias]
    confidence: float  # 0.0 to 1.0
    reasoning: str
    total_commits: int
    total_story_points: int
    preferred_display_name: Optional[str] = None  # Optional preferred name for reports
    
    @property
    def all_emails(self) -> Set[str]:
        """Get all emails in this cluster."""
        emails = {self.canonical_email}
        emails.update(alias.email for alias in self.aliases)
        return emails
    
    @property
    def all_names(self) -> Set[str]:
        """Get all names in this cluster."""
        names = {self.canonical_name}
        names.update(alias.name for alias in self.aliases)
        return names


@dataclass
class IdentityAnalysisResult:
    """Result of LLM identity analysis."""
    clusters: List[DeveloperCluster]
    unresolved_identities: List[DeveloperAlias]
    analysis_metadata: Dict[str, any] = field(default_factory=dict)
    
    def get_manual_mappings(self) -> List[Dict[str, any]]:
        """Convert to manual mappings format for config."""
        mappings = []
        for cluster in self.clusters:
            if len(cluster.aliases) > 0:
                mapping = {}
                # Add name first if specified
                if cluster.preferred_display_name:
                    mapping["name"] = cluster.preferred_display_name
                mapping["primary_email"] = cluster.canonical_email
                mapping["aliases"] = [alias.email for alias in cluster.aliases]
                mappings.append(mapping)
        return mappings
    
    def get_cluster_by_email(self, email: str) -> Optional[DeveloperCluster]:
        """Find cluster containing the given email."""
        email_lower = email.lower()
        for cluster in self.clusters:
            if email_lower in [e.lower() for e in cluster.all_emails]:
                return cluster
        return None