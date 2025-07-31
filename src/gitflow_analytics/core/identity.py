"""Developer identity resolution with persistence."""

import difflib
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import and_

from ..models.database import Database, DeveloperAlias, DeveloperIdentity


class DeveloperIdentityResolver:
    """Resolve and normalize developer identities across repositories."""

    def __init__(
        self,
        db_path: str,
        similarity_threshold: float = 0.85,
        manual_mappings: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Initialize with database for persistence."""
        self.similarity_threshold = similarity_threshold
        self.db = Database(db_path)
        self._cache: dict[str, str] = {}  # In-memory cache for performance
        self._load_cache()

        # Store manual mappings to apply later
        self.manual_mappings = manual_mappings

    @contextmanager
    def get_session(self):
        """Get database session context manager."""
        session = self.db.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _load_cache(self) -> None:
        """Load identities into memory cache."""
        with self.get_session() as session:
            # Load all identities
            identities = session.query(DeveloperIdentity).all()
            for identity in identities:
                self._cache[identity.canonical_id] = {
                    "primary_name": identity.primary_name,
                    "primary_email": identity.primary_email,
                    "github_username": identity.github_username,
                }

            # Load all aliases
            aliases = session.query(DeveloperAlias).all()
            for alias in aliases:
                key = f"{alias.email.lower()}:{alias.name.lower()}"
                self._cache[key] = alias.canonical_id

    def _apply_manual_mappings(self, manual_mappings: list[dict[str, Any]]) -> None:
        """Apply manual identity mappings from configuration."""
        # Clear cache to ensure we get fresh data
        self._cache.clear()
        self._load_cache()

        with self.get_session() as session:
            for mapping in manual_mappings:
                canonical_email = mapping.get("canonical_email", "").lower().strip()
                aliases = mapping.get("aliases", [])

                if not canonical_email or not aliases:
                    continue

                # Find the canonical identity
                canonical_identity = (
                    session.query(DeveloperIdentity)
                    .filter(DeveloperIdentity.primary_email == canonical_email)
                    .first()
                )

                if not canonical_identity:
                    # Skip if canonical identity doesn't exist yet
                    print(f"Warning: Canonical identity not found for email: {canonical_email}")
                    continue

                # Process each alias
                for alias_email in aliases:
                    alias_email = alias_email.lower().strip()

                    # Check if alias identity exists as a primary identity
                    alias_identity = (
                        session.query(DeveloperIdentity)
                        .filter(DeveloperIdentity.primary_email == alias_email)
                        .first()
                    )

                    if alias_identity:
                        if alias_identity.canonical_id != canonical_identity.canonical_id:
                            # Merge the identities - commit before merge to avoid locks
                            session.commit()
                            print(
                                f"Merging identity: {alias_identity.primary_name} ({alias_email}) into {canonical_identity.primary_name} ({canonical_email})"
                            )
                            self.merge_identities(
                                canonical_identity.canonical_id, alias_identity.canonical_id
                            )
                            # Refresh session after merge
                            session.expire_all()
                    else:
                        # Just add as an alias if not a primary identity
                        existing_alias = (
                            session.query(DeveloperAlias)
                            .filter(
                                and_(
                                    DeveloperAlias.email == alias_email,
                                    DeveloperAlias.canonical_id == canonical_identity.canonical_id,
                                )
                            )
                            .first()
                        )

                        if not existing_alias:
                            # Get the name from any existing alias with this email
                            name_for_alias = None
                            any_alias = (
                                session.query(DeveloperAlias)
                                .filter(DeveloperAlias.email == alias_email)
                                .first()
                            )
                            if any_alias:
                                name_for_alias = any_alias.name
                            else:
                                name_for_alias = canonical_identity.primary_name

                            new_alias = DeveloperAlias(
                                canonical_id=canonical_identity.canonical_id,
                                name=name_for_alias,
                                email=alias_email,
                            )
                            session.add(new_alias)
                            print(
                                f"Added alias: {alias_email} for {canonical_identity.primary_name}"
                            )

        # Reload cache after all mappings
        self._cache.clear()
        self._load_cache()

    def resolve_developer(
        self, name: str, email: str, github_username: Optional[str] = None
    ) -> str:
        """Resolve developer identity and return canonical ID."""
        # Normalize inputs
        name = name.strip()
        email = email.lower().strip()

        # Check cache first
        cache_key = f"{email}:{name.lower()}"
        if cache_key in self._cache:
            canonical_id = self._cache[cache_key]
            # Update stats
            self._update_developer_stats(canonical_id)
            return canonical_id

        # Check exact email match in database
        with self.get_session() as session:
            # Check aliases
            alias = session.query(DeveloperAlias).filter(DeveloperAlias.email == email).first()

            if alias:
                self._cache[cache_key] = alias.canonical_id
                self._update_developer_stats(alias.canonical_id)
                return alias.canonical_id

            # Check primary identities
            identity = (
                session.query(DeveloperIdentity)
                .filter(DeveloperIdentity.primary_email == email)
                .first()
            )

            if identity:
                # Add as alias if name is different
                if identity.primary_name.lower() != name.lower():
                    self._add_alias(identity.canonical_id, name, email)
                self._cache[cache_key] = identity.canonical_id
                return identity.canonical_id

        # Find similar developer
        best_match = self._find_best_match(name, email)

        if best_match and best_match[1] >= self.similarity_threshold:
            canonical_id = best_match[0]
            self._add_alias(canonical_id, name, email)
            self._cache[cache_key] = canonical_id
            return canonical_id

        # Create new identity
        canonical_id = self._create_identity(name, email, github_username)
        self._cache[cache_key] = canonical_id
        return canonical_id

    def _find_best_match(self, name: str, email: str) -> Optional[tuple[str, float]]:
        """Find the best matching existing developer."""
        best_score = 0.0
        best_canonical_id = None

        name_lower = name.lower().strip()
        email_domain = email.split("@")[1] if "@" in email else ""

        with self.get_session() as session:
            # Get all identities for comparison
            identities = session.query(DeveloperIdentity).all()

            for identity in identities:
                score = 0.0

                # Name similarity (40% weight)
                name_sim = difflib.SequenceMatcher(
                    None, name_lower, identity.primary_name.lower()
                ).ratio()
                score += name_sim * 0.4

                # Email domain similarity (30% weight)
                identity_domain = (
                    identity.primary_email.split("@")[1] if "@" in identity.primary_email else ""
                )
                if email_domain and email_domain == identity_domain:
                    score += 0.3

                # Check aliases (30% weight)
                aliases = (
                    session.query(DeveloperAlias)
                    .filter(DeveloperAlias.canonical_id == identity.canonical_id)
                    .all()
                )

                best_alias_score = 0.0
                for alias in aliases:
                    alias_name_sim = difflib.SequenceMatcher(
                        None, name_lower, alias.name.lower()
                    ).ratio()

                    # Bonus for same email domain in aliases
                    alias_domain = alias.email.split("@")[1] if "@" in alias.email else ""
                    domain_bonus = 0.2 if alias_domain == email_domain else 0.0

                    alias_score = alias_name_sim + domain_bonus
                    best_alias_score = max(best_alias_score, alias_score)

                score += min(best_alias_score * 0.3, 0.3)

                if score > best_score:
                    best_score = score
                    best_canonical_id = identity.canonical_id

        return (best_canonical_id, best_score) if best_canonical_id else None

    def _create_identity(self, name: str, email: str, github_username: Optional[str] = None) -> str:
        """Create new developer identity."""
        canonical_id = str(uuid.uuid4())

        with self.get_session() as session:
            identity = DeveloperIdentity(
                canonical_id=canonical_id,
                primary_name=name,
                primary_email=email,
                github_username=github_username,
                total_commits=0,
                total_story_points=0,
            )
            session.add(identity)

        # Update cache
        self._cache[canonical_id] = {
            "primary_name": name,
            "primary_email": email,
            "github_username": github_username,
        }

        return canonical_id

    def _add_alias(self, canonical_id: str, name: str, email: str):
        """Add alias for existing developer."""
        with self.get_session() as session:
            # Check if alias already exists
            existing = (
                session.query(DeveloperAlias)
                .filter(
                    and_(
                        DeveloperAlias.canonical_id == canonical_id,
                        DeveloperAlias.email == email.lower(),
                    )
                )
                .first()
            )

            if not existing:
                alias = DeveloperAlias(canonical_id=canonical_id, name=name, email=email.lower())
                session.add(alias)

    def _update_developer_stats(self, canonical_id: str):
        """Update developer statistics."""
        with self.get_session() as session:
            identity = (
                session.query(DeveloperIdentity)
                .filter(DeveloperIdentity.canonical_id == canonical_id)
                .first()
            )

            if identity:
                identity.last_seen = datetime.utcnow()

    def merge_identities(self, canonical_id1: str, canonical_id2: str):
        """Merge two developer identities."""
        # First, add the alias outside of the main merge transaction
        with self.get_session() as session:
            identity2 = (
                session.query(DeveloperIdentity)
                .filter(DeveloperIdentity.canonical_id == canonical_id2)
                .first()
            )
            if identity2:
                identity2_name = identity2.primary_name
                identity2_email = identity2.primary_email

        # Add identity2's primary as alias to identity1 first
        self._add_alias(canonical_id1, identity2_name, identity2_email)

        # Now do the merge in a separate transaction
        with self.get_session() as session:
            # Get both identities fresh
            identity1 = (
                session.query(DeveloperIdentity)
                .filter(DeveloperIdentity.canonical_id == canonical_id1)
                .first()
            )
            identity2 = (
                session.query(DeveloperIdentity)
                .filter(DeveloperIdentity.canonical_id == canonical_id2)
                .first()
            )

            if not identity1 or not identity2:
                raise ValueError("One or both identities not found")

            # Keep identity1, merge identity2 into it
            identity1.total_commits += identity2.total_commits
            identity1.total_story_points += identity2.total_story_points
            identity1.first_seen = min(identity1.first_seen, identity2.first_seen)
            identity1.last_seen = max(identity1.last_seen, identity2.last_seen)

            # Move all aliases from identity2 to identity1
            aliases = (
                session.query(DeveloperAlias)
                .filter(DeveloperAlias.canonical_id == canonical_id2)
                .all()
            )

            for alias in aliases:
                alias.canonical_id = canonical_id1

            # Delete identity2
            session.delete(identity2)

        # Clear cache to force reload
        self._cache.clear()
        self._load_cache()

    def get_developer_stats(self) -> list[dict[str, Any]]:
        """Get statistics for all developers."""
        stats = []

        with self.get_session() as session:
            identities = session.query(DeveloperIdentity).all()

            for identity in identities:
                # Count aliases
                alias_count = (
                    session.query(DeveloperAlias)
                    .filter(DeveloperAlias.canonical_id == identity.canonical_id)
                    .count()
                )

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
                    }
                )

        # Sort by total commits
        return sorted(stats, key=lambda x: x["total_commits"], reverse=True)

    def update_commit_stats(self, commits: list[dict[str, Any]]):
        """Update developer statistics based on commits."""
        # Aggregate stats by canonical ID
        stats_by_dev = defaultdict(lambda: {"commits": 0, "story_points": 0})

        for commit in commits:
            canonical_id = self.resolve_developer(commit["author_name"], commit["author_email"])

            stats_by_dev[canonical_id]["commits"] += 1
            stats_by_dev[canonical_id]["story_points"] += commit.get("story_points", 0) or 0

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

    def apply_manual_mappings(self):
        """Apply manual mappings - can be called explicitly after identities are created."""
        if self.manual_mappings:
            self._apply_manual_mappings(self.manual_mappings)
