"""
Unit tests for cache invalidation when exclude_merge_commits config changes.

Tests that the cache is properly invalidated when the exclude_merge_commits
setting changes, ensuring that analysis results are recalculated with the
new configuration.
"""

import pytest

from gitflow_analytics.core.cache import GitAnalysisCache


class TestCacheInvalidation:
    """Test cache invalidation for exclude_merge_commits configuration."""

    @pytest.fixture
    def cache(self, temp_dir):
        """Create a GitAnalysisCache instance."""
        cache_dir = temp_dir / ".gitflow-cache"
        return GitAnalysisCache(cache_dir)

    def test_cache_hash_includes_exclude_merge_commits(self, cache):
        """Config hash should include exclude_merge_commits setting."""
        # Generate hash with exclude_merge_commits = False
        hash_false = cache.generate_config_hash(
            branch_mapping_rules={},
            ticket_platforms=["github"],
            exclude_paths=[],
            ml_categorization_enabled=False,
            additional_config={"exclude_merge_commits": False},
        )

        # Generate hash with exclude_merge_commits = True
        hash_true = cache.generate_config_hash(
            branch_mapping_rules={},
            ticket_platforms=["github"],
            exclude_paths=[],
            ml_categorization_enabled=False,
            additional_config={"exclude_merge_commits": True},
        )

        # Hashes must be different when exclude_merge_commits changes
        assert hash_false != hash_true, (
            "Config hash should change when exclude_merge_commits changes"
        )

    def test_cache_invalidated_when_exclude_merge_commits_changes(self, cache):
        """Cache should be invalidated when exclude_merge_commits setting changes."""
        # Configuration with exclude_merge_commits = False
        config_false = {
            "branch_mapping_rules": {},
            "ticket_platforms": ["github", "jira"],
            "exclude_paths": ["node_modules/**"],
            "ml_categorization_enabled": False,
            "additional_config": {"exclude_merge_commits": False, "weeks": 8},
        }

        # Configuration with exclude_merge_commits = True (everything else same)
        config_true = {
            "branch_mapping_rules": {},
            "ticket_platforms": ["github", "jira"],
            "exclude_paths": ["node_modules/**"],
            "ml_categorization_enabled": False,
            "additional_config": {"exclude_merge_commits": True, "weeks": 8},
        }

        # Generate hashes
        hash1 = cache.generate_config_hash(**config_false)
        hash2 = cache.generate_config_hash(**config_true)

        # Verify they are different
        assert hash1 != hash2, "Cache should be invalidated when exclude_merge_commits changes"

    def test_cache_reused_when_exclude_merge_commits_unchanged(self, cache):
        """Cache should be reused when exclude_merge_commits setting remains the same."""
        # Same configuration called twice
        config = {
            "branch_mapping_rules": {"FRONTEND": ["frontend/*"]},
            "ticket_platforms": ["github"],
            "exclude_paths": ["*.md"],
            "ml_categorization_enabled": False,
            "additional_config": {"exclude_merge_commits": True, "weeks": 4},
        }

        # Generate hash twice with same config
        hash1 = cache.generate_config_hash(**config)
        hash2 = cache.generate_config_hash(**config)

        # Hashes should be identical (cache can be reused)
        assert hash1 == hash2, "Cache hash should be consistent for same configuration"

    def test_cache_hash_with_different_additional_config_values(self, cache):
        """Test that additional_config values are properly included in hash."""
        base_config = {
            "branch_mapping_rules": {},
            "ticket_platforms": ["github"],
            "exclude_paths": [],
            "ml_categorization_enabled": False,
        }

        # Different additional_config values
        hash1 = cache.generate_config_hash(
            **base_config,
            additional_config={"exclude_merge_commits": False, "weeks": 4},
        )

        hash2 = cache.generate_config_hash(
            **base_config,
            additional_config={"exclude_merge_commits": True, "weeks": 4},
        )

        hash3 = cache.generate_config_hash(
            **base_config,
            additional_config={"exclude_merge_commits": False, "weeks": 8},
        )

        # All hashes should be different
        assert hash1 != hash2, "exclude_merge_commits change should invalidate"
        assert hash1 != hash3, "weeks change should invalidate"
        assert hash2 != hash3, "Both parameters different should invalidate"

    def test_cache_hash_stability_with_none_values(self, cache):
        """Test cache hash handles None values in additional_config correctly."""
        # Config with exclude_merge_commits not specified (None/missing)
        hash_none = cache.generate_config_hash(
            branch_mapping_rules={},
            ticket_platforms=["github"],
            exclude_paths=[],
            ml_categorization_enabled=False,
            additional_config={},  # No exclude_merge_commits key
        )

        # Config with exclude_merge_commits = False (explicit)
        cache.generate_config_hash(
            branch_mapping_rules={},
            ticket_platforms=["github"],
            exclude_paths=[],
            ml_categorization_enabled=False,
            additional_config={"exclude_merge_commits": False},
        )

        # These might be the same or different depending on implementation
        # What's important is that each is stable and deterministic
        hash_none_2 = cache.generate_config_hash(
            branch_mapping_rules={},
            ticket_platforms=["github"],
            exclude_paths=[],
            ml_categorization_enabled=False,
            additional_config={},
        )

        # Same input should produce same hash
        assert hash_none == hash_none_2, "Hash should be stable for same input"

    def test_cache_hash_with_multiple_additional_config_keys(self, cache):
        """Test cache hash with multiple additional_config keys."""
        # Configuration with multiple additional settings
        config_full = {
            "branch_mapping_rules": {},
            "ticket_platforms": ["github"],
            "exclude_paths": [],
            "ml_categorization_enabled": False,
            "additional_config": {
                "exclude_merge_commits": True,
                "weeks": 8,
                "enable_qualitative": True,
                "enable_pm": False,
                "pm_platforms": ["jira"],
            },
        }

        # Same config but exclude_merge_commits = False
        config_different = config_full.copy()
        config_different["additional_config"] = config_full["additional_config"].copy()
        config_different["additional_config"]["exclude_merge_commits"] = False

        hash1 = cache.generate_config_hash(**config_full)
        hash2 = cache.generate_config_hash(**config_different)

        # Should be different due to exclude_merge_commits change
        assert hash1 != hash2, "Cache should invalidate when any additional_config value changes"

    def test_cache_hash_deterministic(self, cache):
        """Verify that cache hash generation is deterministic."""
        config = {
            "branch_mapping_rules": {"BACKEND": ["backend/*", "api/*"]},
            "ticket_platforms": ["jira", "github"],
            "exclude_paths": ["node_modules/**", "vendor/**"],
            "ml_categorization_enabled": True,
            "additional_config": {
                "exclude_merge_commits": True,
                "weeks": 12,
                "enable_qualitative": False,
            },
        }

        # Generate hash multiple times
        hashes = [cache.generate_config_hash(**config) for _ in range(5)]

        # All hashes should be identical
        assert len(set(hashes)) == 1, "Hash generation should be deterministic"

    def test_cache_hash_order_independence(self, cache):
        """Test that additional_config dict order doesn't affect hash."""
        # Two configs with same keys but different dict construction order
        config1 = {
            "branch_mapping_rules": {},
            "ticket_platforms": ["github"],
            "exclude_paths": [],
            "ml_categorization_enabled": False,
            "additional_config": {
                "exclude_merge_commits": True,
                "weeks": 8,
                "enable_qualitative": True,
            },
        }

        config2 = {
            "branch_mapping_rules": {},
            "ticket_platforms": ["github"],
            "exclude_paths": [],
            "ml_categorization_enabled": False,
            "additional_config": {
                "enable_qualitative": True,
                "weeks": 8,
                "exclude_merge_commits": True,
            },
        }

        hash1 = cache.generate_config_hash(**config1)
        hash2 = cache.generate_config_hash(**config2)

        # Should produce same hash regardless of dict key order
        assert hash1 == hash2, "Hash should be order-independent for dict keys"
