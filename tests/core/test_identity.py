"""
Tests for the developer identity resolution module.

These tests verify developer identity consolidation, email mapping,
and fuzzy matching functionality.
"""

from gitflow_analytics.core.identity import DeveloperIdentityResolver


class TestDeveloperIdentityResolver:
    """Test cases for the DeveloperIdentityResolver class."""

    def test_init(self, temp_dir):
        """Test DeveloperIdentityResolver initialization."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path, similarity_threshold=0.8)

        assert resolver.similarity_threshold == 0.8
        assert resolver.manual_mappings is None

    def test_init_with_manual_mappings(self, temp_dir):
        """Test DeveloperIdentityResolver initialization with manual mappings."""
        manual_mappings = [
            {
                "canonical_email": "john@personal.com",
                "aliases": ["john.doe@company.com", "jdoe@corp.com"],
            }
        ]

        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(
            db_path, similarity_threshold=0.85, manual_mappings=manual_mappings
        )

        assert resolver.similarity_threshold == 0.85
        assert resolver.manual_mappings == manual_mappings

    def test_resolve_developer_new(self, temp_dir):
        """Test resolving a new developer identity."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path)

        canonical_id = resolver.resolve_developer("John Doe", "john@example.com")

        assert canonical_id is not None
        assert len(canonical_id) > 0

        # Resolving the same developer should return the same ID
        canonical_id2 = resolver.resolve_developer("John Doe", "john@example.com")
        assert canonical_id == canonical_id2

    def test_resolve_developer_similar_names(self, temp_dir):
        """Test resolving developers with similar names."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path, similarity_threshold=0.8)

        # Add first developer
        canonical_id1 = resolver.resolve_developer("John Doe", "john@example.com")

        # Add similar developer (should potentially match based on similarity)
        canonical_id2 = resolver.resolve_developer("John S Doe", "john.doe@example.com")

        # The resolver should determine if these are the same person
        # (exact behavior depends on implementation logic)
        assert canonical_id1 is not None
        assert canonical_id2 is not None

    def test_fuzzy_matching_threshold(self, temp_dir):
        """Test fuzzy matching respects similarity threshold."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path, similarity_threshold=0.9)  # High threshold

        # Add first developer
        canonical_id1 = resolver.resolve_developer("John Smith", "john@example.com")

        # Add very different developer (should not match)
        canonical_id2 = resolver.resolve_developer("Jane Williams", "jane@example.com")

        # These should be different identities
        assert canonical_id1 != canonical_id2

    def test_manual_mappings_override(self, temp_dir):
        """Test that manual mappings override automatic matching."""
        manual_mappings = [
            {
                "canonical_email": "john@personal.com",
                "aliases": ["john.work@company.com", "john.doe@corp.com"],
            }
        ]

        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path, manual_mappings=manual_mappings)

        # Resolve using an alias email
        canonical_id1 = resolver.resolve_developer("John Doe", "john.work@company.com")

        # Resolve using canonical email
        canonical_id2 = resolver.resolve_developer("John Doe", "john@personal.com")

        # Should resolve to the same canonical identity
        assert canonical_id1 == canonical_id2

    def test_get_developer_stats(self, temp_dir):
        """Test getting developer statistics."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path)

        # Add multiple developers
        resolver.resolve_developer("John Doe", "john@example.com")
        resolver.resolve_developer("Jane Smith", "jane@example.com")

        # Update commit stats
        commits = [
            {"author_name": "John Doe", "author_email": "john@example.com"},
            {"author_name": "Jane Smith", "author_email": "jane@example.com"},
            {"author_name": "John Doe", "author_email": "john@example.com"},
        ]
        resolver.update_commit_stats(commits)

        stats = resolver.get_developer_stats()

        assert len(stats) == 2

        # Check structure
        for stat in stats:
            assert "canonical_id" in stat
            assert "primary_email" in stat
            assert "primary_name" in stat
            assert "total_commits" in stat
            assert stat["total_commits"] >= 0

    def test_merge_identities(self, temp_dir):
        """Test merging two developer identities."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path)

        # Create two separate identities
        canonical_id1 = resolver.resolve_developer("John Doe", "john@personal.com")
        canonical_id2 = resolver.resolve_developer("John D", "john@work.com")

        # Verify they are different initially
        assert canonical_id1 != canonical_id2

        # Merge the identities
        resolver.merge_identities(canonical_id1, canonical_id2)

        # After merging, resolving either should return the same canonical ID
        resolved_id1 = resolver.resolve_developer("John Doe", "john@personal.com")
        resolved_id2 = resolver.resolve_developer("John D", "john@work.com")

        assert resolved_id1 == resolved_id2


class TestDatabaseIntegration:
    """Test cases for database integration."""

    def test_persistence_across_sessions(self, temp_dir):
        """Test that identities persist across resolver sessions."""
        db_path = temp_dir / "identities.db"

        # Create first resolver session
        resolver1 = DeveloperIdentityResolver(db_path)
        canonical_id = resolver1.resolve_developer("John Doe", "john@example.com")

        # Create second resolver session
        resolver2 = DeveloperIdentityResolver(db_path)
        canonical_id2 = resolver2.resolve_developer("John Doe", "john@example.com")

        # Should resolve to the same identity
        assert canonical_id == canonical_id2

    def test_cache_functionality(self, temp_dir):
        """Test that caching improves performance."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path)

        # First resolution (populates cache)
        canonical_id1 = resolver.resolve_developer("John Doe", "john@example.com")

        # Second resolution (should use cache)
        canonical_id2 = resolver.resolve_developer("John Doe", "john@example.com")

        assert canonical_id1 == canonical_id2
        # Verify the result is cached
        cache_key = "john@example.com:john doe"
        assert cache_key in resolver._cache
        assert resolver._cache[cache_key] == canonical_id1
