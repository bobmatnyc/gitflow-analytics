"""Developer identity stats and mappings mixin."""

import logging
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

from ..models.database import DeveloperAlias, DeveloperIdentity

logger = logging.getLogger(__name__)


class IdentityStatsMixin:
    """Mixin: developer stats, commit stat updates, canonical names, manual mappings, fallback."""

    def get_developer_stats(
        self, ticket_coverage: Optional[dict[str, float]] = None
    ) -> list[dict[str, Any]]:
        """
        Get statistics for all developers.

        WHY: This method returns the authoritative developer information for reports,
        including display names that have been updated through manual mappings.
        It ensures that report generators get the correct canonical display names.

        DESIGN DECISION: Accepts optional ticket_coverage parameter to replace the
        previously hardcoded 0.0 ticket coverage values. This enables accurate
        per-developer ticket coverage reporting that matches overall metrics.

        Args:
            ticket_coverage: Optional dict mapping canonical_id to coverage percentage

        Returns:
            List of developer statistics with accurate ticket coverage data
        """
        stats = []

        if not self._database_available:
            # Handle in-memory fallback
            for canonical_id, identity_data in self._in_memory_identities.items():
                # Get actual ticket coverage if provided, otherwise default to 0.0
                coverage_pct = 0.0
                if ticket_coverage:
                    coverage_pct = ticket_coverage.get(canonical_id, 0.0)

                stats.append(
                    {
                        "canonical_id": canonical_id,
                        "primary_name": identity_data["primary_name"],
                        "primary_email": identity_data["primary_email"],
                        "github_username": identity_data.get("github_username"),
                        "total_commits": identity_data.get("total_commits", 0),
                        "total_story_points": identity_data.get("total_story_points", 0),
                        "alias_count": 0,  # Not tracked in memory
                        "first_seen": None,
                        "last_seen": None,
                        "ticket_coverage_pct": coverage_pct,
                    }
                )
            return sorted(stats, key=lambda x: x["total_commits"], reverse=True)

        with self.get_session() as session:
            identities = session.query(DeveloperIdentity).all()

            for identity in identities:
                # Count aliases
                alias_count = (
                    session.query(DeveloperAlias)
                    .filter(DeveloperAlias.canonical_id == identity.canonical_id)
                    .count()
                )

                # Get actual ticket coverage if provided, otherwise default to 0.0
                coverage_pct = 0.0
                if ticket_coverage:
                    coverage_pct = ticket_coverage.get(identity.canonical_id, 0.0)

                stats.append(
                    {
                        "canonical_id": identity.canonical_id,
                        "primary_name": identity.primary_name,
                        "primary_email": identity.primary_email,
                        "github_username": identity.github_username,
                        "total_commits": identity.total_commits,
                        "total_story_points": identity.total_story_points,
                        "alias_count": alias_count,
                        "first_seen": identity.first_seen,
                        "last_seen": identity.last_seen,
                        "ticket_coverage_pct": coverage_pct,
                    }
                )

        # Sort by total commits
        return sorted(stats, key=lambda x: x["total_commits"], reverse=True)

    def update_commit_stats(self, commits: list[dict[str, Any]]):
        """Update developer statistics based on commits.

        Gap 4: Co-author trailer attribution.
        WHY: When a commit carries ``Co-authored-by:`` trailers, every
        listed co-author should also receive credit for the commit.  The
        primary author is resolved as usual; each co-author is resolved
        through the same identity system and added to ``co_author_ids``
        on the commit dict so downstream reports can surface the full
        contributor list.
        """
        # Aggregate stats by canonical ID
        stats_by_dev = defaultdict(lambda: {"commits": 0, "story_points": 0})

        for commit in commits:
            # Debug: check if commit is actually a dictionary
            if not isinstance(commit, dict):
                print(f"Error: Expected commit to be dict, got {type(commit)}: {commit}")
                continue

            canonical_id = self.resolve_developer(commit["author_name"], commit["author_email"])
            # Update the commit with the resolved canonical_id for later use in reports
            commit["canonical_id"] = canonical_id
            # Also add the canonical display name so reports show the correct name
            commit["canonical_name"] = self.get_canonical_name(canonical_id)

            stats_by_dev[canonical_id]["commits"] += 1
            stats_by_dev[canonical_id]["story_points"] += commit.get("story_points", 0) or 0

            # Gap 4: Credit co-authors found in Co-authored-by trailers.
            co_author_ids: list[str] = []
            for co_author in commit.get("co_authors", []):
                co_name = co_author.get("name", "")
                co_email = co_author.get("email", "")
                if not co_email:
                    continue
                # Skip if this is the same person as the primary author (same email)
                if co_email.lower() == commit.get("author_email", "").lower():
                    continue
                co_id = self.resolve_developer(co_name, co_email)
                co_author_ids.append(co_id)
                # Co-author gets commit credit (not story points to avoid double-counting)
                stats_by_dev[co_id]["commits"] += 1
                logger.debug(
                    f"Attributed co-authored commit {commit.get('hash', '')[:8]} "
                    f"to co-author {co_name} <{co_email}>"
                )
            commit["co_author_ids"] = co_author_ids

        # Update database
        with self.get_session() as session:
            for canonical_id, stats in stats_by_dev.items():
                identity = (
                    session.query(DeveloperIdentity)
                    .filter(DeveloperIdentity.canonical_id == canonical_id)
                    .first()
                )

                if identity:
                    identity.total_commits += stats["commits"]
                    identity.total_story_points += stats["story_points"]
                    identity.last_seen = datetime.utcnow()

        # Apply manual mappings after all identities are created
        if self.manual_mappings:
            self.apply_manual_mappings()
            # Re-apply canonical names now that mappings have taken effect.
            # The canonical_name set above may be stale because apply_manual_mappings()
            # can rename identities (e.g. merge an alias into a preferred display name).
            for commit in commits:
                cid = commit.get("canonical_id")
                if cid:
                    commit["canonical_name"] = self.get_canonical_name(cid)

    def apply_manual_mappings(self):
        """Apply manual mappings - can be called explicitly after identities are created."""
        if self.manual_mappings:
            self._apply_manual_mappings(self.manual_mappings)

    def get_canonical_name(self, canonical_id: str) -> str:
        """
        Get the canonical display name for a given canonical ID.

        WHY: Reports need to show the proper display name from manual mappings
        instead of the original commit author name. This method provides the
        authoritative display name for any canonical ID.

        Args:
            canonical_id: The canonical ID to get the display name for

        Returns:
            The display name that should be used in reports, or "Unknown" if not found
        """
        if not self._database_available:
            # Check in-memory storage first
            if canonical_id in self._in_memory_identities:
                return self._in_memory_identities[canonical_id]["primary_name"]
            # Check cache
            if canonical_id in self._cache:
                cache_entry = self._cache[canonical_id]
                if isinstance(cache_entry, dict):
                    return cache_entry.get("primary_name", "Unknown")
            return "Unknown"

        with self.get_session() as session:
            identity = (
                session.query(DeveloperIdentity)
                .filter(DeveloperIdentity.canonical_id == canonical_id)
                .first()
            )

            if identity:
                return identity.primary_name

        return "Unknown"

    def _apply_manual_mappings_to_memory(self) -> None:
        """
        Apply manual mappings to in-memory storage when database is not available.

        WHY: When persistence fails, we still need to apply user-configured
        identity mappings for the current analysis session.
        """
        if not self.manual_mappings:
            return

        for mapping in self.manual_mappings:
            # Support both canonical_email and primary_email for backward compatibility
            canonical_email = (
                (mapping.get("primary_email", "") or mapping.get("canonical_email", ""))
                .lower()
                .strip()
            )
            aliases = mapping.get("aliases", [])
            preferred_name = mapping.get("name")  # Optional display name

            if not canonical_email or not aliases:
                continue

            # Create canonical identity in memory
            canonical_id = str(uuid.uuid4())
            self._in_memory_identities[canonical_id] = {
                "primary_name": preferred_name or canonical_email.split("@")[0],
                "primary_email": canonical_email,
                "github_username": None,
                "total_commits": 0,
                "total_story_points": 0,
            }

            # Add to cache
            self._cache[canonical_id] = self._in_memory_identities[canonical_id]

            # Process aliases
            for alias_email in aliases:
                alias_email = alias_email.lower().strip()
                alias_key = f"{alias_email}:{preferred_name or canonical_email.split('@')[0]}"
                self._in_memory_aliases[alias_key] = canonical_id
                self._cache[alias_key] = canonical_id

            logger.debug(
                f"Applied in-memory mapping: {preferred_name or canonical_email.split('@')[0]} "
                f"with {len(aliases)} aliases"
            )

    def _fallback_identity_resolution(self, name: str, email: str) -> str:
        """
        Fallback identity resolution when database is not available.

        WHY: Even without persistence, we need consistent identity resolution
        within a single analysis session to avoid duplicate developer entries.

        Args:
            name: Developer name
            email: Developer email

        Returns:
            Canonical ID for the developer
        """
        # Normalize inputs
        name = name.strip()
        email = email.lower().strip()
        cache_key = f"{email}:{name.lower()}"

        # Check if already resolved
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check in-memory aliases
        if cache_key in self._in_memory_aliases:
            canonical_id = self._in_memory_aliases[cache_key]
            self._cache[cache_key] = canonical_id
            return canonical_id

        # Check for email match in existing identities
        for canonical_id, identity in self._in_memory_identities.items():
            if identity["primary_email"] == email:
                # Add this name variant to cache
                self._cache[cache_key] = canonical_id
                return canonical_id

        # Create new identity
        canonical_id = str(uuid.uuid4())
        self._in_memory_identities[canonical_id] = {
            "primary_name": name,
            "primary_email": email,
            "github_username": None,
            "total_commits": 0,
            "total_story_points": 0,
        }

        # Add to cache
        self._cache[canonical_id] = self._in_memory_identities[canonical_id]
        self._cache[cache_key] = canonical_id

        logger.debug(f"Created in-memory identity for {name} <{email}>")
        return canonical_id
