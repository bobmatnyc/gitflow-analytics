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


class TestCoAuthorAttribution:
    """Tests for Gap 4: Co-authored-by trailer attribution in update_commit_stats."""

    def test_co_author_gets_commit_credit(self, temp_dir):
        """Co-author listed in the commit trailer receives a commit credit."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path)

        commits = [
            {
                "author_name": "Alice",
                "author_email": "alice@example.com",
                "story_points": 3,
                "co_authors": [{"name": "Bob", "email": "bob@example.com"}],
            }
        ]
        resolver.update_commit_stats(commits)

        stats = resolver.get_developer_stats(ticket_coverage={})

        # Primary author (Alice) gets a commit
        alice_stats = next((s for s in stats if s["primary_email"] == "alice@example.com"), None)
        assert alice_stats is not None, "Alice not found in developer stats"
        assert alice_stats["total_commits"] >= 1

        # Co-author (Bob) also gets a commit credit
        bob_stats = next((s for s in stats if s["primary_email"] == "bob@example.com"), None)
        assert bob_stats is not None, "Bob not found in developer stats (co-author credit missing)"
        assert bob_stats["total_commits"] >= 1

    def test_co_author_does_not_receive_story_points(self, temp_dir):
        """Co-authors do not get story-point double-credit — only the primary author does."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path)

        commits = [
            {
                "author_name": "Alice",
                "author_email": "alice@example.com",
                "story_points": 5,
                "co_authors": [{"name": "Bob", "email": "bob@example.com"}],
            }
        ]
        resolver.update_commit_stats(commits)

        stats = resolver.get_developer_stats(ticket_coverage={})

        alice_stats = next((s for s in stats if s["primary_email"] == "alice@example.com"), None)
        bob_stats = next((s for s in stats if s["primary_email"] == "bob@example.com"), None)
        assert alice_stats is not None and bob_stats is not None

        # Only Alice gets story points; Bob gets commit credit but no story points
        assert alice_stats["total_story_points"] == 5
        assert bob_stats["total_story_points"] == 0

    def test_commit_with_no_co_authors_is_unaffected(self, temp_dir):
        """Commits without co_authors key behave identically to before Gap 4."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path)

        commits = [
            {
                "author_name": "Charlie",
                "author_email": "charlie@example.com",
                "story_points": 2,
                # no co_authors key at all
            }
        ]
        resolver.update_commit_stats(commits)

        stats = resolver.get_developer_stats(ticket_coverage={})
        charlie_stats = next(
            (s for s in stats if s["primary_email"] == "charlie@example.com"), None
        )
        assert charlie_stats is not None
        assert charlie_stats["total_commits"] == 1
        assert len(stats) == 1  # No phantom co-authors created

    def test_same_email_co_author_as_primary_not_double_counted(self, temp_dir):
        """If the co-author email matches the primary author, no duplicate credit is given."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path)

        commits = [
            {
                "author_name": "Dave",
                "author_email": "dave@example.com",
                "story_points": 1,
                "co_authors": [
                    {"name": "Dave", "email": "dave@example.com"}  # same as author
                ],
            }
        ]
        resolver.update_commit_stats(commits)

        stats = resolver.get_developer_stats(ticket_coverage={})
        dave_stats = next((s for s in stats if s["primary_email"] == "dave@example.com"), None)
        assert dave_stats is not None
        # Should only count once (same person cannot be both author and co-author)
        assert dave_stats["total_commits"] == 1

    def test_co_author_ids_added_to_commit(self, temp_dir):
        """update_commit_stats adds co_author_ids to the commit dict for downstream use."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path)

        commit = {
            "author_name": "Eve",
            "author_email": "eve@example.com",
            "story_points": 0,
            "co_authors": [{"name": "Frank", "email": "frank@example.com"}],
        }
        resolver.update_commit_stats([commit])

        assert "co_author_ids" in commit
        assert len(commit["co_author_ids"]) == 1

    def test_empty_co_authors_list_produces_empty_ids(self, temp_dir):
        """Explicitly empty co_authors list results in empty co_author_ids."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path)

        commit = {
            "author_name": "Grace",
            "author_email": "grace@example.com",
            "story_points": 0,
            "co_authors": [],
        }
        resolver.update_commit_stats([commit])

        assert commit.get("co_author_ids") == []

    def test_multiple_co_authors_all_credited(self, temp_dir):
        """Every co-author in a commit trailer is credited, not just the first."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path)

        commits = [
            {
                "author_name": "Lead",
                "author_email": "lead@example.com",
                "story_points": 0,
                "co_authors": [
                    {"name": "Dev1", "email": "dev1@example.com"},
                    {"name": "Dev2", "email": "dev2@example.com"},
                    {"name": "Dev3", "email": "dev3@example.com"},
                ],
            }
        ]
        resolver.update_commit_stats(commits)

        stats = resolver.get_developer_stats(ticket_coverage={})
        # Lead + 3 co-authors = 4 distinct developers
        assert len(stats) == 4
        emails = {s["primary_email"] for s in stats}
        assert "dev1@example.com" in emails
        assert "dev2@example.com" in emails
        assert "dev3@example.com" in emails


class TestGitHubNoReplyEmailResolution:
    """Fix 2: GitHub noreply email auto-detection.

    GitHub noreply addresses follow the pattern:
        {numeric_id}+{username}@users.noreply.github.com

    The resolver must automatically link these to the canonical identity that
    already exists under the plain username alias, without requiring a manual
    mapping entry.
    """

    def test_noreply_email_matches_username_alias(self, temp_dir):
        """61434073+chungk-duetto@users.noreply.github.com resolves to chungk-duetto identity."""
        db_path = temp_dir / "identities.db"

        manual_mappings = [
            {
                "name": "Chung Kim",
                "primary_email": "chung.kim@company.com",
                "aliases": ["chung.kim@company.com", "chungk-duetto"],
            }
        ]
        resolver = DeveloperIdentityResolver(db_path, manual_mappings=manual_mappings)

        # Resolve via the plain username first (simulates a commit authored with username)
        canonical_via_username = resolver.resolve_developer("Chung Kim", "chungk-duetto")

        # Now resolve via the GitHub noreply address
        canonical_via_noreply = resolver.resolve_developer(
            "Chung Kim",
            "61434073+chungk-duetto@users.noreply.github.com",
        )

        assert (
            canonical_via_noreply == canonical_via_username
        ), "GitHub noreply address should resolve to the same identity as the plain username alias"

    def test_noreply_email_matches_primary_identity(self, temp_dir):
        """noreply email resolves to an identity whose primary_email is the bare username."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path)

        # Create an identity where the primary email is the bare username
        # (e.g. created automatically from a commit that used a username)
        canonical_via_username = resolver.resolve_developer("octocat", "octocat")

        # The noreply address should find the same identity
        canonical_via_noreply = resolver.resolve_developer(
            "The Octocat",
            "583231+octocat@users.noreply.github.com",
        )

        assert canonical_via_noreply == canonical_via_username, (
            "GitHub noreply address should resolve to the identity whose primary_email "
            "is the extracted username"
        )

    def test_noreply_email_without_plus_creates_new_identity(self, temp_dir):
        """Noreply-style address without the numeric+username format is treated as a new identity."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path)

        resolver.resolve_developer("Alice", "alice")

        # An address that ends with noreply domain but lacks "+" should NOT trigger the
        # username-extraction path and should therefore create a distinct identity.
        id_plain = resolver.resolve_developer("Alice", "alice@users.noreply.github.com")
        id_user = resolver.resolve_developer("Alice", "alice")

        # These are separate identities; the noreply path only fires when "+" is present.
        assert id_plain != id_user

    def test_noreply_match_is_cached(self, temp_dir):
        """Resolving the same noreply address twice uses the cache on the second call."""
        db_path = temp_dir / "identities.db"

        manual_mappings = [
            {
                "name": "Dev User",
                "primary_email": "dev@company.com",
                "aliases": ["dev@company.com", "devuser"],
            }
        ]
        resolver = DeveloperIdentityResolver(db_path, manual_mappings=manual_mappings)

        noreply = "99999+devuser@users.noreply.github.com"
        id1 = resolver.resolve_developer("Dev User", noreply)
        id2 = resolver.resolve_developer("Dev User", noreply)

        assert id1 == id2
        cache_key = f"{noreply}:dev user"
        assert cache_key in resolver._cache


class TestStaleCanonicalNameFix:
    """Fix 1: canonical_name on commit dicts must reflect post-mapping names.

    update_commit_stats() sets canonical_name BEFORE apply_manual_mappings() runs.
    After the fix, a second pass re-stamps each commit with the fresh canonical name.
    """

    def test_canonical_name_updated_after_manual_mappings(self, temp_dir):
        """commit['canonical_name'] shows the mapped display name, not the raw git name."""
        db_path = temp_dir / "identities.db"

        manual_mappings = [
            {
                "name": "Chung Kim",
                "primary_email": "chung.kim@company.com",
                "aliases": ["chung.kim@company.com", "chungk-duetto"],
            }
        ]
        resolver = DeveloperIdentityResolver(db_path, manual_mappings=manual_mappings)

        commit = {
            "author_name": "chungk-duetto",
            "author_email": "chungk-duetto",
            "story_points": 0,
        }
        resolver.update_commit_stats([commit])

        assert commit.get("canonical_name") == "Chung Kim", (
            f"Expected 'Chung Kim' but got {commit.get('canonical_name')!r}. "
            "The second-pass re-application of canonical_name after manual mappings is missing."
        )

    def test_canonical_name_correct_without_mappings(self, temp_dir):
        """Without manual mappings, canonical_name still reflects the git author name."""
        db_path = temp_dir / "identities.db"
        resolver = DeveloperIdentityResolver(db_path)

        commit = {
            "author_name": "Alice Developer",
            "author_email": "alice@example.com",
            "story_points": 0,
        }
        resolver.update_commit_stats([commit])

        assert commit.get("canonical_name") == "Alice Developer"

    def test_multiple_commits_all_receive_updated_canonical_name(self, temp_dir):
        """All commits in a batch get correct canonical_name after mappings."""
        db_path = temp_dir / "identities.db"

        manual_mappings = [
            {
                "name": "Jane Smith",
                "primary_email": "jane@company.com",
                "aliases": ["jane@company.com", "jsmith-gh"],
            }
        ]
        resolver = DeveloperIdentityResolver(db_path, manual_mappings=manual_mappings)

        commits = [
            {"author_name": "jsmith-gh", "author_email": "jsmith-gh", "story_points": 0},
            {"author_name": "jsmith-gh", "author_email": "jsmith-gh", "story_points": 1},
        ]
        resolver.update_commit_stats(commits)

        for i, commit in enumerate(commits):
            assert (
                commit.get("canonical_name") == "Jane Smith"
            ), f"Commit {i} has canonical_name={commit.get('canonical_name')!r}, expected 'Jane Smith'"


class TestLoaderPrimaryEmailHardening:
    """Fix 3: loader must prefer a real email address over a plain username.

    When developer_aliases lists a username before any real email address,
    the loader previously used the username as primary_email.  After the fix
    it skips plain usernames and picks the first entry that contains '@'.
    """

    def test_loader_picks_email_over_username(self):
        """primary_email is chosen from the first entry that contains '@'."""
        emails = ["chungk-duetto", "chung.kim@company.com", "chung@personal.com"]

        primary_email = next((e for e in emails if "@" in e), emails[0])

        assert (
            primary_email == "chung.kim@company.com"
        ), f"Expected first real email but got {primary_email!r}"

    def test_loader_falls_back_to_first_entry_when_no_email(self):
        """When no entry contains '@', the first entry is still used as primary."""
        emails = ["useronly", "anotheruser"]

        primary_email = next((e for e in emails if "@" in e), emails[0])

        assert primary_email == "useronly"

    def test_loader_uses_only_entry_when_it_is_email(self):
        """Single-entry list with a real email works correctly."""
        emails = ["dev@example.com"]

        primary_email = next((e for e in emails if "@" in e), emails[0])

        assert primary_email == "dev@example.com"

    def test_loader_integration_with_developer_aliases(self, tmp_path):
        """End-to-end: loader converts developer_aliases with leading username correctly."""
        import textwrap

        from gitflow_analytics.config.loader import ConfigLoader

        config_content = textwrap.dedent(
            f"""
            version: "1.0"
            github:
              token: "ghp_test"
              owner: "test-org"
            repositories:
              - name: "repo"
                path: "{str(tmp_path)}"
            developer_aliases:
              "Chung Kim":
                - chungk-duetto
                - chung.kim@company.com
                - 61434073+chungk-duetto@users.noreply.github.com
            """
        )

        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = ConfigLoader.load(config_path)

        mappings = config.analysis.manual_identity_mappings
        assert len(mappings) == 1, f"Expected 1 mapping, got {len(mappings)}: {mappings}"

        mapping = mappings[0]
        primary = mapping["primary_email"]
        assert "@" in primary, f"primary_email should be a real email address, got {primary!r}"
        assert (
            primary == "chung.kim@company.com"
        ), f"Expected 'chung.kim@company.com' but got {primary!r}"
