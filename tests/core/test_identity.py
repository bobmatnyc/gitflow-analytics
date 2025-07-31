"""
Tests for the developer identity resolution module.

These tests verify developer identity consolidation, email mapping,
and fuzzy matching functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from gitflow_analytics.core.identity import IdentityResolver, DeveloperIdentity
from gitflow_analytics.models.database import DeveloperIdentityModel


class TestIdentityResolver:
    """Test cases for the IdentityResolver class."""

    def test_init(self):
        """Test IdentityResolver initialization."""
        resolver = IdentityResolver(threshold=0.8)

        assert resolver.threshold == 0.8
        assert resolver.manual_mappings == {}
        assert resolver.identities == {}

    def test_init_with_manual_mappings(self):
        """Test IdentityResolver initialization with manual mappings."""
        manual_mappings = {
            "john.doe@company.com": "john@personal.com",
            "jane.smith@oldcompany.com": "jane@newcompany.com",
        }

        resolver = IdentityResolver(threshold=0.85, manual_mappings=manual_mappings)

        assert resolver.threshold == 0.85
        assert resolver.manual_mappings == manual_mappings

    def test_add_commit_new_developer(self):
        """Test adding a commit from a new developer."""
        resolver = IdentityResolver()

        resolver.add_commit("john@example.com", "John Doe", "abc123")

        assert "john@example.com" in resolver.identities
        identity = resolver.identities["john@example.com"]
        assert identity.primary_email == "john@example.com"
        assert "john@example.com" in identity.all_emails
        assert identity.name == "John Doe"
        assert identity.commit_count == 1
        assert "abc123" in identity.commit_hashes

    def test_add_commit_existing_developer(self):
        """Test adding a commit from an existing developer."""
        resolver = IdentityResolver()

        # Add first commit
        resolver.add_commit("john@example.com", "John Doe", "abc123")

        # Add second commit
        resolver.add_commit("john@example.com", "John Doe", "def456")

        identity = resolver.identities["john@example.com"]
        assert identity.commit_count == 2
        assert "abc123" in identity.commit_hashes
        assert "def456" in identity.commit_hashes

    def test_add_commit_with_manual_mapping(self):
        """Test adding a commit with manual email mapping."""
        manual_mappings = {"john.work@company.com": "john@personal.com"}
        resolver = IdentityResolver(manual_mappings=manual_mappings)

        # First establish the primary identity
        resolver.add_commit("john@personal.com", "John Doe", "abc123")

        # Then add commit with mapped email
        resolver.add_commit("john.work@company.com", "John Doe", "def456")

        # Should be consolidated under primary email
        assert "john@personal.com" in resolver.identities
        identity = resolver.identities["john@personal.com"]
        assert identity.commit_count == 2
        assert "john@personal.com" in identity.all_emails
        assert "john.work@company.com" in identity.all_emails

    def test_fuzzy_match_similar_names(self):
        """Test fuzzy matching of similar developer names."""
        resolver = IdentityResolver(threshold=0.8)

        # Add first developer
        resolver.add_commit("john@example.com", "John Doe", "abc123")

        # Add similar name that should match
        resolver.add_commit("john.doe@company.com", "John Doe", "def456")

        resolver._consolidate_identities()

        # Should be consolidated into one identity
        consolidated = resolver.get_consolidated_identities()
        assert len(consolidated) == 1

        identity = list(consolidated.values())[0]
        assert identity.commit_count == 2
        assert "john@example.com" in identity.all_emails
        assert "john.doe@company.com" in identity.all_emails

    def test_fuzzy_match_different_names(self):
        """Test that different names don't get matched."""
        resolver = IdentityResolver(threshold=0.8)

        # Add two clearly different developers
        resolver.add_commit("john@example.com", "John Doe", "abc123")
        resolver.add_commit("jane@example.com", "Jane Smith", "def456")

        resolver._consolidate_identities()

        # Should remain as separate identities
        consolidated = resolver.get_consolidated_identities()
        assert len(consolidated) == 2

    def test_calculate_name_similarity(self):
        """Test name similarity calculation."""
        resolver = IdentityResolver()

        # Test identical names
        similarity = resolver._calculate_name_similarity("John Doe", "John Doe")
        assert similarity == 1.0

        # Test similar names
        similarity = resolver._calculate_name_similarity("John Doe", "John D")
        assert similarity > 0.5

        # Test different names
        similarity = resolver._calculate_name_similarity("John Doe", "Jane Smith")
        assert similarity < 0.5

    def test_get_consolidated_identities(self):
        """Test getting consolidated developer identities."""
        resolver = IdentityResolver()

        # Add some commits
        resolver.add_commit("john@example.com", "John Doe", "abc123")
        resolver.add_commit("john@example.com", "John Doe", "def456")
        resolver.add_commit("jane@example.com", "Jane Smith", "ghi789")

        consolidated = resolver.get_consolidated_identities()

        assert len(consolidated) == 2
        assert "john@example.com" in consolidated
        assert "jane@example.com" in consolidated

        john_identity = consolidated["john@example.com"]
        assert john_identity.commit_count == 2

        jane_identity = consolidated["jane@example.com"]
        assert jane_identity.commit_count == 1

    @patch("gitflow_analytics.core.identity.session_scope")
    def test_save_to_database(self, mock_session_scope):
        """Test saving identities to database."""
        mock_session = Mock(spec=Session)
        mock_session_scope.return_value.__enter__.return_value = mock_session

        resolver = IdentityResolver()
        resolver.add_commit("john@example.com", "John Doe", "abc123")

        resolver.save_to_database("/tmp/test.db")

        mock_session.merge.assert_called()
        mock_session.commit.assert_called_once()

    @patch("gitflow_analytics.core.identity.session_scope")
    def test_load_from_database(self, mock_session_scope):
        """Test loading identities from database."""
        mock_session = Mock(spec=Session)
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Mock database record
        mock_identity = DeveloperIdentityModel(
            primary_email="john@example.com",
            all_emails="john@example.com,john.doe@company.com",
            name="John Doe",
            commit_count=5,
            commit_hashes="abc123,def456,ghi789",
        )
        mock_session.query.return_value.all.return_value = [mock_identity]

        resolver = IdentityResolver()
        resolver.load_from_database("/tmp/test.db")

        assert "john@example.com" in resolver.identities
        identity = resolver.identities["john@example.com"]
        assert identity.primary_email == "john@example.com"
        assert identity.commit_count == 5
        assert len(identity.all_emails) == 2


class TestDeveloperIdentity:
    """Test cases for the DeveloperIdentity class."""

    def test_init(self):
        """Test DeveloperIdentity initialization."""
        identity = DeveloperIdentity("john@example.com", "John Doe")

        assert identity.primary_email == "john@example.com"
        assert identity.name == "John Doe"
        assert identity.commit_count == 0
        assert identity.all_emails == {"john@example.com"}
        assert identity.commit_hashes == set()

    def test_add_email(self):
        """Test adding additional email addresses."""
        identity = DeveloperIdentity("john@example.com", "John Doe")

        identity.add_email("john.doe@company.com")

        assert "john.doe@company.com" in identity.all_emails
        assert len(identity.all_emails) == 2

    def test_add_commit(self):
        """Test adding commit to identity."""
        identity = DeveloperIdentity("john@example.com", "John Doe")

        identity.add_commit("abc123")

        assert identity.commit_count == 1
        assert "abc123" in identity.commit_hashes

        # Add same commit again - should not duplicate
        identity.add_commit("abc123")

        assert identity.commit_count == 1
        assert len(identity.commit_hashes) == 1

    def test_merge_identity(self):
        """Test merging two developer identities."""
        identity1 = DeveloperIdentity("john@example.com", "John Doe")
        identity1.add_commit("abc123")
        identity1.add_commit("def456")

        identity2 = DeveloperIdentity("john.doe@company.com", "John Doe")
        identity2.add_commit("ghi789")
        identity2.add_email("john.work@company.com")

        identity1.merge(identity2)

        assert identity1.commit_count == 3
        assert "abc123" in identity1.commit_hashes
        assert "def456" in identity1.commit_hashes
        assert "ghi789" in identity1.commit_hashes
        assert "john.doe@company.com" in identity1.all_emails
        assert "john.work@company.com" in identity1.all_emails

    def test_to_dict(self):
        """Test converting identity to dictionary."""
        identity = DeveloperIdentity("john@example.com", "John Doe")
        identity.add_email("john.doe@company.com")
        identity.add_commit("abc123")
        identity.add_commit("def456")

        data = identity.to_dict()

        assert data["primary_email"] == "john@example.com"
        assert data["name"] == "John Doe"
        assert data["commit_count"] == 2
        assert set(data["all_emails"]) == {"john@example.com", "john.doe@company.com"}
        assert set(data["commit_hashes"]) == {"abc123", "def456"}

    def test_from_dict(self):
        """Test creating identity from dictionary."""
        data = {
            "primary_email": "john@example.com",
            "name": "John Doe",
            "commit_count": 2,
            "all_emails": ["john@example.com", "john.doe@company.com"],
            "commit_hashes": ["abc123", "def456"],
        }

        identity = DeveloperIdentity.from_dict(data)

        assert identity.primary_email == "john@example.com"
        assert identity.name == "John Doe"
        assert identity.commit_count == 2
        assert "john@example.com" in identity.all_emails
        assert "john.doe@company.com" in identity.all_emails
        assert "abc123" in identity.commit_hashes
        assert "def456" in identity.commit_hashes
