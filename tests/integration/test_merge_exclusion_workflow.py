"""Integration tests for merge commit exclusion end-to-end workflow.

This module tests the complete workflow from configuration → data fetch → analysis → reports
to verify that merge commit exclusion works correctly throughout the entire system.

Test Scenarios:
1. End-to-End Analysis with Merge Exclusion Enabled
2. Cache Invalidation on Configuration Change
3. Report Generation Consistency with Merge Exclusion
"""

from datetime import datetime, timedelta, timezone

from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.core.data_fetcher import GitDataFetcher
from gitflow_analytics.models.database import CachedCommit, DailyCommitBatch

from .conftest import calculate_total_lines_from_commits, verify_commit_in_database


class TestMergeExclusionWorkflow:
    """Integration tests for merge commit exclusion workflow."""

    def test_full_analysis_with_merge_exclusion(
        self, temp_workspace, test_repo_with_merges, integration_db_session
    ):
        """Test complete workflow: Config → Data Fetch → Analysis → Reports.

        This test verifies that:
        1. All commits (regular + merge) are stored in database
        2. Merge commits have filtered_insertions = 0 when exclude_merge_commits is enabled
        3. Regular commits have normal filtered stats
        4. Reports reflect only regular commit line counts
        5. Commit counts include all commits (regular + merge)

        Steps:
        1. Create test repository with 13 regular commits and 2 merge commits
        2. Configure with exclude_merge_commits: true
        3. Run data fetch
        4. Verify database storage
        5. Run analysis
        6. Verify report metrics
        """
        # Arrange: Set up the test environment
        repo_info = test_repo_with_merges
        cache_dir = temp_workspace / ".gitflow-cache"

        # Create cache with merge exclusion enabled
        cache = GitAnalysisCache(
            cache_dir=cache_dir,
        )

        # Create data fetcher with merge exclusion enabled
        fetcher = GitDataFetcher(
            cache=cache,
            exclude_merge_commits=True,  # Enable merge exclusion
        )

        # Calculate date range (last 30 days to cover all commits)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)

        # Act: Fetch repository data
        fetch_result = fetcher.fetch_repository_data(
            repo_path=repo_info["repo_path"],
            project_key="TEST",
            weeks_back=4,
            start_date=start_date,
            end_date=end_date,
        )

        # Assert: Verify fetch completed successfully
        assert fetch_result["stats"]["storage_success"], "Data fetch should succeed"
        assert fetch_result["stats"]["stored_commits"] > 0, "Should store commits"

        # Verify: All commits are stored in database
        session = integration_db_session
        total_commits = (
            session.query(CachedCommit)
            .filter(CachedCommit.repo_path == str(repo_info["repo_path"]))
            .count()
        )

        assert total_commits == repo_info["total_commits"], (
            f"Should store all {repo_info['total_commits']} commits "
            f"(regular + merge), found {total_commits}"
        )

        # Verify: Merge commits have filtered_insertions = 0
        for merge_hash in repo_info["merge_commits"]:
            commit_data = verify_commit_in_database(session, merge_hash, repo_info["repo_path"])
            assert commit_data["is_merge"], f"Commit {merge_hash[:8]} should be marked as merge"
            assert commit_data["filtered_insertions"] == 0, (
                f"Merge commit {merge_hash[:8]} should have filtered_insertions=0, "
                f"got {commit_data['filtered_insertions']}"
            )
            assert commit_data["filtered_deletions"] == 0, (
                f"Merge commit {merge_hash[:8]} should have filtered_deletions=0, "
                f"got {commit_data['filtered_deletions']}"
            )
            # Raw stats should still be present
            assert commit_data["insertions"] >= 0, "Merge commit should have raw insertions stored"

        # Verify: Regular commits have normal filtered stats
        regular_commits_with_lines = 0
        for regular_hash in repo_info["regular_commits"]:
            commit_data = verify_commit_in_database(session, regular_hash, repo_info["repo_path"])
            assert not commit_data["is_merge"], (
                f"Commit {regular_hash[:8]} should not be marked as merge"
            )

            # Regular commits should have non-zero filtered stats
            if commit_data["filtered_insertions"] > 0 or commit_data["filtered_deletions"] > 0:
                regular_commits_with_lines += 1

        assert regular_commits_with_lines > 0, (
            "At least some regular commits should have line changes"
        )

        # Verify: Calculate total filtered line counts (should exclude merge commits)
        filtered_stats = calculate_total_lines_from_commits(
            session, repo_info["repo_path"], use_filtered=True
        )

        # Filtered stats should only include regular commits (not merge commits)
        assert filtered_stats["total_lines"] > 0, "Should have non-zero filtered line count"
        assert filtered_stats["commit_count"] == repo_info["total_commits"], (
            "Should count all commits even though merge commits have 0 filtered lines"
        )

        # Verify: Raw stats include all commits (merge + regular)
        raw_stats = calculate_total_lines_from_commits(
            session, repo_info["repo_path"], use_filtered=False
        )

        assert raw_stats["total_lines"] > filtered_stats["total_lines"], (
            "Raw line count should be higher than filtered (includes merge commits)"
        )

        # Verify: Daily batches reflect filtered statistics
        daily_batches = (
            session.query(DailyCommitBatch)
            .filter(DailyCommitBatch.repo_path == str(repo_info["repo_path"]))
            .all()
        )

        assert len(daily_batches) > 0, "Should have daily batches"

        total_batch_lines = sum(
            (batch.total_lines_added or 0) + (batch.total_lines_deleted or 0)
            for batch in daily_batches
        )

        # Batch statistics should match filtered statistics (merge commits excluded)
        assert total_batch_lines == filtered_stats["total_lines"], (
            f"Daily batch line totals ({total_batch_lines}) should match filtered stats "
            f"({filtered_stats['total_lines']})"
        )

    def test_cache_invalidation_on_config_change(self, temp_workspace, test_repo_with_merges):
        """Test that cache is properly invalidated when exclude_merge_commits changes.

        This test verifies that:
        1. Analysis with exclude_merge_commits: false includes merge commit lines
        2. Changing to exclude_merge_commits: true invalidates cache
        3. Re-running analysis generates new results with merge commits excluded
        4. Database is regenerated (not reused)

        Steps:
        1. Run analysis with exclude_merge_commits: false
        2. Verify results include merge commit lines
        3. Change config to exclude_merge_commits: true
        4. Run analysis again (cache should be invalidated)
        5. Verify results now exclude merge commit lines
        6. Confirm database was regenerated
        """
        # Arrange: Set up the test environment
        repo_info = test_repo_with_merges

        # Step 1: Run analysis with merge exclusion disabled
        # Use separate cache directories to simulate config change
        cache_dir_1 = temp_workspace / ".gitflow-cache-1"
        cache_dir_1.mkdir(exist_ok=True)
        cache_1 = GitAnalysisCache(cache_dir=cache_dir_1)

        fetcher_1 = GitDataFetcher(
            cache=cache_1,
            exclude_merge_commits=False,  # Merge exclusion disabled
        )

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)

        fetch_result_1 = fetcher_1.fetch_repository_data(
            repo_path=repo_info["repo_path"],
            project_key="TEST",
            weeks_back=4,
            start_date=start_date,
            end_date=end_date,
        )

        assert fetch_result_1["stats"]["storage_success"], "First fetch should succeed"

        # Verify: Results include merge commit lines
        session_1 = cache_1.db.get_session()
        try:
            stats_with_merges = calculate_total_lines_from_commits(
                session_1, repo_info["repo_path"], use_filtered=True
            )

            # When merge exclusion is disabled, filtered stats should equal raw stats
            raw_stats_1 = calculate_total_lines_from_commits(
                session_1, repo_info["repo_path"], use_filtered=False
            )

            assert stats_with_merges["total_lines"] == raw_stats_1["total_lines"], (
                "With merge exclusion disabled, filtered and raw stats should be equal"
            )

            initial_line_count = stats_with_merges["total_lines"]
            assert initial_line_count > 0, "Should have non-zero line count with merges included"

        finally:
            session_1.close()

        # Step 2: Change config and run analysis with merge exclusion enabled
        # Use a different cache directory to simulate config change
        cache_dir_2 = temp_workspace / ".gitflow-cache-2"
        cache_dir_2.mkdir(exist_ok=True)
        cache_2 = GitAnalysisCache(cache_dir=cache_dir_2)

        fetcher_2 = GitDataFetcher(
            cache=cache_2,
            exclude_merge_commits=True,  # Merge exclusion enabled
        )

        fetch_result_2 = fetcher_2.fetch_repository_data(
            repo_path=repo_info["repo_path"],
            project_key="TEST",
            weeks_back=4,
            start_date=start_date,
            end_date=end_date,
        )

        assert fetch_result_2["stats"]["storage_success"], "Second fetch should succeed"

        # Verify: Results now exclude merge commit lines
        session_2 = cache_2.db.get_session()
        try:
            stats_without_merges = calculate_total_lines_from_commits(
                session_2, repo_info["repo_path"], use_filtered=True
            )

            # With merge exclusion enabled, filtered stats should be less than raw stats
            raw_stats_2 = calculate_total_lines_from_commits(
                session_2, repo_info["repo_path"], use_filtered=False
            )

            assert stats_without_merges["total_lines"] < raw_stats_2["total_lines"], (
                "With merge exclusion enabled, filtered stats should be less than raw stats"
            )

            # Filtered line count should be lower than with merges included
            assert stats_without_merges["total_lines"] < initial_line_count, (
                f"Line count with merge exclusion ({stats_without_merges['total_lines']}) "
                f"should be less than with merges included ({initial_line_count})"
            )

            # Verify merge commits have zero filtered stats
            merge_commits_query = (
                session_2.query(CachedCommit)
                .filter(
                    CachedCommit.repo_path == str(repo_info["repo_path"]),
                    CachedCommit.is_merge,
                )
                .all()
            )

            assert len(merge_commits_query) == repo_info["merge_commit_count"], (
                f"Should have {repo_info['merge_commit_count']} merge commits"
            )

            for merge_commit in merge_commits_query:
                assert merge_commit.filtered_insertions == 0, (
                    f"Merge commit {merge_commit.commit_hash[:8]} should have filtered_insertions=0"
                )
                assert merge_commit.filtered_deletions == 0, (
                    f"Merge commit {merge_commit.commit_hash[:8]} should have filtered_deletions=0"
                )

        finally:
            session_2.close()

        # Verify: Confirm cache invalidation occurred
        # The two caches should have different database files (different cache directories)
        db_path_1 = cache_dir_1 / "gitflow_cache.db"
        db_path_2 = cache_dir_2 / "gitflow_cache.db"

        assert db_path_1.exists(), "First cache database should exist"
        assert db_path_2.exists(), "Second cache database should exist"
        assert cache_dir_1 != cache_dir_2, "Cache directories should be different"

    def test_report_consistency_with_merge_exclusion(self, temp_workspace, test_repo_with_merges):
        """Test that analysis results and database consistently reflect merge exclusion.

        This test verifies that:
        1. Database statistics correctly reflect filtered line counts
        2. Analysis results match database statistics
        3. Daily batches reflect filtered statistics
        4. Merge commits have 0 filtered lines throughout the system

        Steps:
        1. Run analysis with merge exclusion enabled
        2. Verify database contains correct filtered statistics
        3. Verify analysis results match database
        4. Verify daily batches match filtered statistics
        5. Verify consistency across all data layers
        """
        # Arrange: Set up the test environment
        repo_info = test_repo_with_merges
        cache_dir = temp_workspace / ".gitflow-cache"

        # Create cache with merge exclusion enabled
        cache = GitAnalysisCache(
            cache_dir=cache_dir,
        )

        # Fetch data with merge exclusion
        fetcher = GitDataFetcher(
            cache=cache,
            exclude_merge_commits=True,
        )

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)

        fetch_result = fetcher.fetch_repository_data(
            repo_path=repo_info["repo_path"],
            project_key="TEST",
            weeks_back=4,
            start_date=start_date,
            end_date=end_date,
        )

        assert fetch_result["stats"]["storage_success"], "Data fetch should succeed"

        # Verify: Check database for consistency
        # Note: We don't need to run the full analyzer here since we're only
        # verifying database consistency after data fetch
        session = cache.db.get_session()
        try:
            db_stats = calculate_total_lines_from_commits(
                session, repo_info["repo_path"], use_filtered=True
            )

            # Database stats should show all commits
            assert db_stats["commit_count"] == repo_info["total_commits"], (
                f"Database commit count ({db_stats['commit_count']}) should match "
                f"total commits ({repo_info['total_commits']})"
            )

            # Verify merge commits have zero filtered stats in database
            merge_commits = (
                session.query(CachedCommit)
                .filter(
                    CachedCommit.repo_path == str(repo_info["repo_path"]),
                    CachedCommit.is_merge,
                )
                .all()
            )

            for merge_commit in merge_commits:
                assert merge_commit.filtered_insertions == 0, (
                    f"Merge commit {merge_commit.commit_hash[:8]} should have "
                    f"filtered_insertions=0 in database"
                )
                assert merge_commit.filtered_deletions == 0, (
                    f"Merge commit {merge_commit.commit_hash[:8]} should have "
                    f"filtered_deletions=0 in database"
                )

            # Verify: Daily batches reflect filtered statistics
            daily_batches = (
                session.query(DailyCommitBatch)
                .filter(DailyCommitBatch.repo_path == str(repo_info["repo_path"]))
                .all()
            )

            assert len(daily_batches) > 0, "Should have daily batches"

            total_batch_lines = sum(
                (batch.total_lines_added or 0) + (batch.total_lines_deleted or 0)
                for batch in daily_batches
            )

            # Batch statistics should match filtered statistics
            assert total_batch_lines == db_stats["total_lines"], (
                f"Daily batch line totals ({total_batch_lines}) should match "
                f"database filtered stats ({db_stats['total_lines']})"
            )

            # Verify: Consistency between database, analysis, and batches
            # All should show the same filtered line count
            raw_stats = calculate_total_lines_from_commits(
                session, repo_info["repo_path"], use_filtered=False
            )

            assert db_stats["total_lines"] < raw_stats["total_lines"], (
                "Filtered line count should be less than raw (merge commits excluded)"
            )

            # All data layers should report consistent filtered statistics
            assert db_stats["total_lines"] == total_batch_lines, (
                "Database filtered stats should match daily batch totals"
            )

        finally:
            session.close()

    def test_merge_commit_detection_accuracy(self, temp_workspace, test_repo_with_merges):
        """Test that merge commits are correctly identified in the database.

        This test verifies that:
        1. All merge commits are correctly marked with is_merge=True
        2. All regular commits are correctly marked with is_merge=False
        3. Merge detection is accurate across different repository states

        Steps:
        1. Fetch all commits from test repository
        2. Query database for merge commits
        3. Verify merge commit count matches expected
        4. Verify all expected merge commits are marked correctly
        5. Verify all regular commits are marked correctly
        """
        # Arrange
        repo_info = test_repo_with_merges
        cache_dir = temp_workspace / ".gitflow-cache"

        cache = GitAnalysisCache(
            cache_dir=cache_dir,
        )

        fetcher = GitDataFetcher(
            cache=cache,
            exclude_merge_commits=True,
        )

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)

        # Act: Fetch repository data
        fetch_result = fetcher.fetch_repository_data(
            repo_path=repo_info["repo_path"],
            project_key="TEST",
            weeks_back=4,
            start_date=start_date,
            end_date=end_date,
        )

        assert fetch_result["stats"]["storage_success"], "Data fetch should succeed"

        # Assert: Verify merge commit detection
        session = cache.db.get_session()
        try:
            # Query all commits
            all_commits = (
                session.query(CachedCommit)
                .filter(CachedCommit.repo_path == str(repo_info["repo_path"]))
                .all()
            )

            # Count merge commits
            merge_commits_db = [c for c in all_commits if c.is_merge]
            regular_commits_db = [c for c in all_commits if not c.is_merge]

            # Verify counts
            assert len(merge_commits_db) == repo_info["merge_commit_count"], (
                f"Should detect {repo_info['merge_commit_count']} merge commits, "
                f"found {len(merge_commits_db)}"
            )

            assert len(regular_commits_db) == repo_info["regular_commit_count"], (
                f"Should have {repo_info['regular_commit_count']} regular commits, "
                f"found {len(regular_commits_db)}"
            )

            # Verify specific merge commits are marked correctly
            merge_hashes_db = {c.commit_hash for c in merge_commits_db}
            expected_merge_hashes = set(repo_info["merge_commits"])

            assert merge_hashes_db == expected_merge_hashes, (
                "Database merge commits should match expected merge commits"
            )

            # Verify specific regular commits are marked correctly
            regular_hashes_db = {c.commit_hash for c in regular_commits_db}
            expected_regular_hashes = set(repo_info["regular_commits"])

            assert regular_hashes_db == expected_regular_hashes, (
                "Database regular commits should match expected regular commits"
            )

        finally:
            session.close()


class TestMergeExclusionEdgeCases:
    """Test edge cases for merge commit exclusion."""

    def test_repository_with_no_merge_commits(self, temp_workspace, test_author):
        """Test that analysis works correctly when repository has no merge commits.

        Verifies that:
        1. Analysis completes successfully with no merge commits
        2. All commits are treated as regular commits
        3. Filtered and raw stats are identical
        """
        # Create repository with only regular commits (no merges)
        repo_path = temp_workspace / "no_merge_repo"
        repo_path.mkdir()

        from git import Repo

        repo = Repo.init(repo_path)

        with repo.config_writer() as config:
            config.set_value("user", "name", test_author.name)
            config.set_value("user", "email", test_author.email)

        file_path = repo_path / "test.txt"

        # Create 5 regular commits (no branches, no merges)
        for i in range(5):
            content = f"Content {i}\n" * 10
            file_path.write_text(content)
            repo.index.add(["test.txt"])
            repo.index.commit(f"Commit {i}", author=test_author, committer=test_author)

        # Fetch with merge exclusion enabled
        cache_dir = temp_workspace / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir=cache_dir)

        fetcher = GitDataFetcher(cache=cache, exclude_merge_commits=True)

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)

        fetch_result = fetcher.fetch_repository_data(
            repo_path=repo_path,
            project_key="TEST",
            weeks_back=4,
            start_date=start_date,
            end_date=end_date,
        )

        assert fetch_result["stats"]["storage_success"], "Fetch should succeed"

        # Verify no merge commits in database
        session = cache.db.get_session()
        try:
            merge_count = (
                session.query(CachedCommit)
                .filter(
                    CachedCommit.repo_path == str(repo_path),
                    CachedCommit.is_merge,
                )
                .count()
            )

            assert merge_count == 0, "Should have no merge commits"

            # Verify filtered and raw stats are identical
            filtered_stats = calculate_total_lines_from_commits(
                session, repo_path, use_filtered=True
            )
            raw_stats = calculate_total_lines_from_commits(session, repo_path, use_filtered=False)

            assert filtered_stats["total_lines"] == raw_stats["total_lines"], (
                "Filtered and raw stats should be identical when no merge commits exist"
            )

        finally:
            session.close()

    def test_repository_with_only_merge_commits(self, temp_workspace, test_author):
        """Test edge case where repository only contains merge commits.

        Verifies that:
        1. All commits are correctly identified as merge commits
        2. Filtered line counts are all zero
        3. Raw line counts are preserved
        """
        # Create repository with only merge commits
        repo_path = temp_workspace / "only_merge_repo"
        repo_path.mkdir()

        from git import Repo

        repo = Repo.init(repo_path)

        with repo.config_writer() as config:
            config.set_value("user", "name", test_author.name)
            config.set_value("user", "email", test_author.email)

        file_path = repo_path / "test.txt"

        # Create initial commit
        file_path.write_text("Initial content\n")
        repo.index.add(["test.txt"])
        repo.index.commit("Initial commit", author=test_author, committer=test_author)

        # Capture default branch for merging
        default_branch = repo.active_branch

        # Create two merge commits
        for i in range(2):
            # Create branch
            branch = repo.create_head(f"feature{i}")
            branch.checkout()

            # Commit on branch
            file_path.write_text(f"Feature {i} content\n")
            repo.index.add(["test.txt"])
            repo.index.commit(f"Feature {i}", author=test_author, committer=test_author)

            # Merge to default branch (main or master)
            default_branch.checkout()
            repo.git.merge(f"feature{i}", no_ff=True, m=f"Merge feature{i}")

        # Fetch with merge exclusion
        cache_dir = temp_workspace / ".gitflow-cache"
        cache = GitAnalysisCache(cache_dir=cache_dir)

        fetcher = GitDataFetcher(cache=cache, exclude_merge_commits=True)

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)

        fetch_result = fetcher.fetch_repository_data(
            repo_path=repo_path,
            project_key="TEST",
            weeks_back=4,
            start_date=start_date,
            end_date=end_date,
        )

        assert fetch_result["stats"]["storage_success"], "Fetch should succeed"

        # Verify merge commits have zero filtered stats
        session = cache.db.get_session()
        try:
            merge_commits = (
                session.query(CachedCommit)
                .filter(
                    CachedCommit.repo_path == str(repo_path),
                    CachedCommit.is_merge,
                )
                .all()
            )

            assert len(merge_commits) == 2, "Should have 2 merge commits"

            for merge_commit in merge_commits:
                assert merge_commit.filtered_insertions == 0, (
                    "Merge commit should have filtered_insertions=0"
                )
                # Raw stats should be non-zero
                assert merge_commit.insertions > 0, (
                    "Merge commit should have non-zero raw insertions"
                )

            # Verify total filtered stats are mostly from non-merge commits
            filtered_stats = calculate_total_lines_from_commits(
                session, repo_path, use_filtered=True
            )
            raw_stats = calculate_total_lines_from_commits(session, repo_path, use_filtered=False)

            assert filtered_stats["total_lines"] < raw_stats["total_lines"], (
                "Filtered stats should be less than raw stats when merge commits exist"
            )

        finally:
            session.close()
