"""Developer identity resolution with persistence."""

import difflib
import logging
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import and_

from ..models.database import Database, DeveloperAlias, DeveloperIdentity

logger = logging.getLogger(__name__)


class DeveloperIdentityResolver:
    """Resolve and normalize developer identities across repositories."""

    def __init__(
        self,
        db_path: str,
        similarity_threshold: float = 0.85,
        manual_mappings: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize with database for persistence.

        WHY: This initializer handles database connection issues gracefully,
        allowing the system to continue functioning even when persistence fails.

        Args:
            db_path: Path to the SQLite database file
            similarity_threshold: Threshold for fuzzy matching (0.0-1.0)
            manual_mappings: Optional manual identity mappings from configuration
        """
        self.similarity_threshold = similarity_threshold
        self.db_path = Path(db_path)  # Convert string to Path
        self._cache: dict[str, str] = {}  # In-memory cache for performance

        # Initialize database with error handling
        try:
            self.db = Database(self.db_path)
            self._database_available = True

            # Warn user if using fallback database
            if self.db.is_readonly_fallback:
                logger.warning(
                    "Using temporary database for identity resolution. "
                    "Identity mappings will not persist between runs. "
                    f"Check permissions on: {db_path}"
                )

            # Load existing data from database
            self._load_cache()

        except Exception as e:
            logger.error(
                f"Failed to initialize identity database at {db_path}: {e}. "
                "Identity resolution will work but mappings won't persist."
            )
            self._database_available = False
            self.db = None

        # Store manual mappings to apply later
        self.manual_mappings = manual_mappings

        # When database is not available, we need in-memory fallback storage
        if not self._database_available:
            logger.info(
                "Database unavailable, using in-memory identity resolution. "
                "Identity mappings will not persist between runs."
            )
            self._in_memory_identities: dict[str, dict[str, Any]] = {}
            self._in_memory_aliases: dict[str, str] = {}

            # Apply manual mappings to in-memory storage if provided
            if self.manual_mappings:
                self._apply_manual_mappings_to_memory()
        else:
            # Apply manual mappings to database if provided
            if self.manual_mappings:
                self._apply_manual_mappings(self.manual_mappings)

    @contextmanager
    def get_session(self):
        """
        Get database session context manager with fallback handling.

        WHY: When database is not available, we need to provide a no-op
        context manager that allows the code to continue without failing.
        """
        if not self._database_available or not self.db:
            # No-op context manager when database is not available
            class NoOpSession:
                def query(self, *args, **kwargs):
                    return NoOpQuery()

                def add(self, *args, **kwargs):
                    pass

                def delete(self, *args, **kwargs):
                    pass

                def commit(self):
                    pass

                def rollback(self):
                    pass

                def expire_all(self):
                    pass

            class NoOpQuery:
                def filter(self, *args, **kwargs):
                    return self

                def first(self):
                    return None

                def all(self):
                    return []

                def count(self):
                    return 0

            yield NoOpSession()
            return

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
        """
        Load identities into memory cache.

        WHY: When database is not available, we start with an empty cache
        and rely on in-memory identity resolution for the current session.
        """
        if not self._database_available:
            logger.debug("Database not available, starting with empty identity cache")
            return

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
        # Handle database unavailable scenario
        if not self._database_available:
            self._apply_manual_mappings_to_memory()
            return

        # Clear cache to ensure we get fresh data
        self._cache.clear()
        self._load_cache()

        with self.get_session() as session:
            for mapping in manual_mappings:
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

                # Find or create the canonical identity
                canonical_identity = (
                    session.query(DeveloperIdentity)
                    .filter(DeveloperIdentity.primary_email == canonical_email)
                    .first()
                )

                if not canonical_identity:
                    # Create the canonical identity if it doesn't exist
                    canonical_id = str(uuid.uuid4())
                    canonical_identity = DeveloperIdentity(
                        canonical_id=canonical_id,
                        primary_name=preferred_name or canonical_email.split("@")[0],
                        primary_email=canonical_email,
                        first_seen=datetime.now(timezone.utc),
                        last_seen=datetime.now(timezone.utc),
                        total_commits=0,
                        total_story_points=0,
                    )
                    session.add(canonical_identity)
                    session.commit()
                    print(
                        f"Created canonical identity: {canonical_identity.primary_name} ({canonical_email})"
                    )

                # Update the preferred name if provided
                if preferred_name and preferred_name != canonical_identity.primary_name:
                    print(
                        f"Updating display name: {canonical_identity.primary_name} → {preferred_name}"
                    )
                    canonical_identity.primary_name = preferred_name

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
        """
        Resolve developer identity and return canonical ID.

        WHY: This method handles both database-backed and in-memory identity resolution,
        allowing the system to function even when persistence is not available.
        """
        # Use fallback resolution when database is not available
        if not self._database_available:
            return self._fallback_identity_resolution(name, email)

        # Normalize inputs
        name = name.strip()
        email = email.lower().strip()

        # Check cache first
        cache_key = f"{email}:{name.lower()}"
        if cache_key in self._cache:
            canonical_id = self._cache[cache_key]
            # Update stats
            self._update_developer_stats(canonical_id)
            logger.debug(f"Resolved {name} <{email}> from cache to {canonical_id}")
            return canonical_id

        # Check exact email match in database
        with self.get_session() as session:
            # Check aliases
            alias = session.query(DeveloperAlias).filter(DeveloperAlias.email == email).first()

            if alias:
                # Found an alias with this email - add this name variant to cache and DB
                self._cache[cache_key] = alias.canonical_id
                self._update_developer_stats(alias.canonical_id)
                logger.debug(f"Found alias for {email}, resolving {name} to {alias.canonical_id}")
                # Add this name variant as an alias if it's different
                if alias.name.lower() != name.lower():
                    logger.debug(f"Adding name variant '{name}' as alias for {email}")
                    self._add_alias(alias.canonical_id, name, email)
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
        logger.info(f"Creating new identity for {name} <{email}> - no matches found")
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
                # Update cache with the new alias
                cache_key = f"{email.lower()}:{name.lower()}"
                self._cache[cache_key] = canonical_id

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
        """Update developer statistics based on commits."""
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
