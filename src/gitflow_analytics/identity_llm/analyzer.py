"""LLM-based developer identity analyzer."""

import difflib
import json
import logging
import re
from collections import defaultdict
from typing import Any, Optional

from ..core.identity import DeveloperIdentityResolver
from .models import DeveloperAlias, DeveloperCluster, IdentityAnalysisResult

logger = logging.getLogger(__name__)


class LLMIdentityAnalyzer:
    """Analyzes developer identities using LLM for intelligent aliasing."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
        confidence_threshold: float = 0.9,
    ):
        """Initialize the LLM identity analyzer.

        Args:
            api_key: OpenRouter API key for LLM-based analysis
            model: LLM model to use (default: openai/gpt-4o-mini)
            confidence_threshold: Minimum confidence for identity matches (default: 0.9 = 90%)
        """
        self.api_key = api_key
        self.model = model
        self.confidence_threshold = confidence_threshold
        self._has_openrouter = api_key is not None

    def analyze_identities(
        self,
        commits: list[dict[str, Any]],
        existing_resolver: Optional[DeveloperIdentityResolver] = None,
    ) -> IdentityAnalysisResult:
        """Analyze commits to identify developer aliases using LLM."""
        # Extract unique developer identities from commits
        identities = self._extract_identities(commits)

        # Pre-cluster using heuristics
        pre_clusters = self._pre_cluster_identities(identities)

        # Analyze with LLM if available
        if self._has_openrouter and self.api_key:
            clusters = self._analyze_with_llm(pre_clusters, identities)
        else:
            # Fall back to heuristic-only clustering
            clusters = self._finalize_heuristic_clusters(pre_clusters, identities)

        # Identify unresolved identities
        clustered_emails = set()
        for cluster in clusters:
            clustered_emails.update(cluster.all_emails)

        unresolved = [
            identity for identity in identities.values() if identity.email not in clustered_emails
        ]

        return IdentityAnalysisResult(
            clusters=clusters,
            unresolved_identities=unresolved,
            analysis_metadata={
                "total_identities": len(identities),
                "total_clusters": len(clusters),
                "analysis_method": "llm" if self._has_openrouter else "heuristic",
            },
        )

    def _extract_identities(self, commits: list[dict[str, Any]]) -> dict[str, DeveloperAlias]:
        """Extract unique developer identities from commits."""
        identities = {}

        for commit in commits:
            key = f"{commit['author_email'].lower()}:{commit['author_name']}"

            if key not in identities:
                identities[key] = DeveloperAlias(
                    name=commit["author_name"],
                    email=commit["author_email"].lower(),
                    commit_count=0,
                    first_seen=commit["timestamp"],
                    last_seen=commit["timestamp"],
                    repositories=set(),
                )

            identity = identities[key]
            identity.commit_count += 1
            identity.first_seen = min(identity.first_seen, commit["timestamp"])
            identity.last_seen = max(identity.last_seen, commit["timestamp"])

            # Track repository if available
            if "repository" in commit:
                identity.repositories.add(commit["repository"])

        return identities

    def _pre_cluster_identities(self, identities: dict[str, DeveloperAlias]) -> list[set[str]]:
        """Pre-cluster identities using heuristic rules."""
        clusters = []
        processed = set()

        identity_list = list(identities.values())

        for i, identity1 in enumerate(identity_list):
            if identity1.email in processed:
                continue

            cluster = {identity1.email}
            processed.add(identity1.email)

            for _j, identity2 in enumerate(identity_list[i + 1 :], i + 1):
                if identity2.email in processed:
                    continue

                # Check various similarity criteria
                if self._are_likely_same_developer(identity1, identity2):
                    cluster.add(identity2.email)
                    processed.add(identity2.email)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def _are_likely_same_developer(self, id1: DeveloperAlias, id2: DeveloperAlias) -> bool:
        """Check if two identities are likely the same developer."""
        # Same email domain with similar name
        domain1 = id1.email.split("@")[1] if "@" in id1.email else ""
        domain2 = id2.email.split("@")[1] if "@" in id2.email else ""

        name_similarity = difflib.SequenceMatcher(None, id1.name.lower(), id2.name.lower()).ratio()

        # GitHub noreply emails
        github_pattern = r"^\d+\+(.+)@users\.noreply\.github\.com$"
        match1 = re.match(github_pattern, id1.email)
        match2 = re.match(github_pattern, id2.email)

        # Check GitHub noreply patterns
        if match1 or match2:
            github_username1 = match1.group(1).lower() if match1 else None
            github_username2 = match2.group(1).lower() if match2 else None

            # Compare GitHub username with name/email
            if github_username1:
                # Check against other's name or email local part
                other_name = id2.name.lower().replace(" ", "").replace(".", "").replace("-", "")
                other_local = id2.email.split("@")[0].lower().replace(".", "").replace("-", "")

                if (
                    github_username1 in other_name
                    or other_name in github_username1
                    or github_username1 in other_local
                    or other_local in github_username1
                ):
                    return True

            if github_username2:
                # Check against other's name or email local part
                other_name = id1.name.lower().replace(" ", "").replace(".", "").replace("-", "")
                other_local = id1.email.split("@")[0].lower().replace(".", "").replace("-", "")

                if (
                    github_username2 in other_name
                    or other_name in github_username2
                    or github_username2 in other_local
                    or other_local in github_username2
                ):
                    return True

        # Check if one email's local part matches the other's name
        local1 = id1.email.split("@")[0].lower()
        local2 = id2.email.split("@")[0].lower()

        # Remove common suffixes/prefixes for comparison
        clean_local1 = local1
        clean_local2 = local2
        for suffix in ["-ewtn", "-zaelot", "dev", "developer", "zaelot"]:
            clean_local1 = clean_local1.replace(suffix, "")
            clean_local2 = clean_local2.replace(suffix, "")

        # Check if cleaned locals match names
        name1_clean = id1.name.lower().replace(" ", "").replace(".", "")
        name2_clean = id2.name.lower().replace(" ", "").replace(".", "")

        if (
            clean_local1 in name2_clean
            or name2_clean in clean_local1
            or clean_local2 in name1_clean
            or name1_clean in clean_local2
        ):
            return True

        # Strong indicators
        if name_similarity > 0.9 and domain1 == domain2:
            return True

        if name_similarity > 0.8 and (
            domain1 == domain2 or "github.com" in domain1 or "github.com" in domain2
        ):
            return True

        # Check if local part of email matches name parts
        name1_parts = set(id1.name.lower().split())
        name2_parts = set(id2.name.lower().split())

        if local1 in name2_parts or local2 in name1_parts:
            return True

        # Check first/last name combinations
        if len(name1_parts) >= 2 and len(name2_parts) >= 2:
            # Check if initials match
            initials1 = "".join(n[0] for n in sorted(name1_parts) if n)
            initials2 = "".join(n[0] for n in sorted(name2_parts) if n)

            if initials1 == initials2 and name_similarity > 0.6:
                return True

        # Check overlapping repositories with high name similarity
        return bool(id1.repositories & id2.repositories and name_similarity > 0.7)

    def _finalize_heuristic_clusters(
        self, pre_clusters: list[set[str]], identities: dict[str, DeveloperAlias]
    ) -> list[DeveloperCluster]:
        """Convert pre-clusters to final clusters without LLM."""
        clusters = []

        for email_set in pre_clusters:
            # Get all identities in this cluster
            cluster_identities = [
                identity for identity in identities.values() if identity.email in email_set
            ]

            if not cluster_identities:
                continue

            # Choose canonical identity (most commits)
            canonical = max(cluster_identities, key=lambda x: x.commit_count)
            aliases = [id for id in cluster_identities if id.email != canonical.email]

            # Calculate total stats
            total_commits = sum(id.commit_count for id in cluster_identities)

            cluster = DeveloperCluster(
                canonical_name=canonical.name,
                canonical_email=canonical.email,
                aliases=aliases,
                confidence=0.85,  # Heuristic confidence
                reasoning="Clustered based on name similarity and email patterns",
                total_commits=total_commits,
                total_story_points=0,  # Would need commit data to calculate
            )
            clusters.append(cluster)

        return clusters

    def _analyze_with_llm(
        self, pre_clusters: list[set[str]], identities: dict[str, DeveloperAlias]
    ) -> list[DeveloperCluster]:
        """Analyze pre-clusters with LLM for intelligent grouping."""
        try:
            import openai

            # Configure OpenAI client for OpenRouter
            client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)

            clusters = []

            # Analyze each pre-cluster
            for email_set in pre_clusters:
                cluster_identities = [
                    identity for identity in identities.values() if identity.email in email_set
                ]

                if len(cluster_identities) < 2:
                    continue

                # Prepare data for LLM
                identity_data = []
                for identity in cluster_identities:
                    identity_data.append(
                        {
                            "name": identity.name,
                            "email": identity.email,
                            "commit_count": identity.commit_count,
                            "repositories": list(identity.repositories),
                        }
                    )

                prompt = self._create_analysis_prompt(identity_data)

                # Call LLM
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at analyzing developer identities and determining if different email/name combinations belong to the same person.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                )

                # Parse LLM response
                cluster = self._parse_llm_response(
                    response.choices[0].message.content, cluster_identities, identities
                )
                if cluster:
                    clusters.append(cluster)

            # Also analyze unclustered identities
            clustered_emails = set()
            for cluster in clusters:
                clustered_emails.update(cluster.all_emails)

            unclustered = [
                identity
                for identity in identities.values()
                if identity.email not in clustered_emails
            ]

            # Group unclustered by name similarity for LLM analysis
            if unclustered:
                additional_clusters = self._analyze_unclustered_with_llm(unclustered, client)
                clusters.extend(additional_clusters)

            return clusters

        except Exception as e:
            logger.warning(f"LLM analysis failed, falling back to heuristics: {e}")
            return self._finalize_heuristic_clusters(pre_clusters, identities)

    def _create_analysis_prompt(self, identity_data: list[dict[str, Any]]) -> str:
        """Create prompt for LLM analysis."""
        return f"""Analyze these developer identities and determine if they belong to the same person:

{json.dumps(identity_data, indent=2)}

Consider:
1. Name variations (e.g., "John Doe" vs "John D" vs "jdoe")
2. Email patterns (company emails, personal emails, GitHub noreply)
3. Common repositories they work on
4. Email domain relationships

Respond with a JSON object:
{{
  "same_person": true/false,
  "confidence": 0.0-1.0,
  "canonical_identity": {{"name": "...", "email": "..."}},
  "reasoning": "explanation"
}}"""

    def _parse_llm_response(
        self,
        response: str,
        cluster_identities: list[DeveloperAlias],
        all_identities: dict[str, DeveloperAlias],
    ) -> Optional[DeveloperCluster]:
        """Parse LLM response into a cluster."""
        try:
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                return None

            data = json.loads(json_match.group())

            if not data.get("same_person", False):
                return None

            confidence = float(data.get("confidence", 0.8))
            if confidence < self.confidence_threshold:
                # Log why this cluster was rejected
                cluster_emails = [id.email for id in cluster_identities]
                logger.info(
                    f"Rejected identity cluster: {', '.join(cluster_emails)} "
                    f"(confidence {confidence:.1%} < threshold {self.confidence_threshold:.1%})"
                )
                return None

            # Find canonical identity
            canonical_data = data.get("canonical_identity", {})
            canonical_email = canonical_data.get("email", "").lower()

            # Find matching identity
            canonical = None
            for identity in cluster_identities:
                if identity.email == canonical_email:
                    canonical = identity
                    break

            if not canonical:
                # Use highest commit count as canonical
                canonical = max(cluster_identities, key=lambda x: x.commit_count)

            # Create aliases list
            aliases = [id for id in cluster_identities if id.email != canonical.email]

            # Calculate total stats
            total_commits = sum(id.commit_count for id in cluster_identities)

            return DeveloperCluster(
                canonical_name=canonical.name,
                canonical_email=canonical.email,
                aliases=aliases,
                confidence=confidence,
                reasoning=data.get("reasoning", "LLM analysis"),
                total_commits=total_commits,
                total_story_points=0,
            )

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return None

    def _analyze_unclustered_with_llm(
        self, unclustered: list[DeveloperAlias], client
    ) -> list[DeveloperCluster]:
        """Analyze unclustered identities with LLM."""
        clusters = []

        # Group by similar names for analysis
        name_groups = defaultdict(list)
        for identity in unclustered:
            # Get simplified name for grouping
            name_key = "".join(identity.name.lower().split())[:5]
            name_groups[name_key].append(identity)

        for group in name_groups.values():
            if len(group) < 2:
                continue

            # Prepare data for LLM
            identity_data = []
            for identity in group:
                identity_data.append(
                    {
                        "name": identity.name,
                        "email": identity.email,
                        "commit_count": identity.commit_count,
                        "repositories": list(identity.repositories),
                    }
                )

            prompt = self._create_analysis_prompt(identity_data)

            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at analyzing developer identities.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                )

                cluster = self._parse_llm_response(response.choices[0].message.content, group, {})
                if cluster:
                    clusters.append(cluster)

            except Exception as e:
                logger.warning(f"Failed to analyze group with LLM: {e}")
                continue

        return clusters
