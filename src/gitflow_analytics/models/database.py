"""Database models for GitFlow Analytics using SQLAlchemy.

This module re-exports all models from sub-modules for backward compatibility.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    create_engine,
    text,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

from .database_base import Base, utcnow_tz_aware  # noqa: E402
from .database_commit_models import (  # noqa: E402
    CachedCommit,
    CommitClassificationBatch,
    CommitTicketCorrelation,
    DailyCommitBatch,
    DailyMetrics,
    DetailedTicketData,
    RepositoryAnalysisStatus,
    SchemaVersion,
    WeeklyFetchStatus,
    WeeklyTrends,
)
from .database_identity_models import (  # noqa: E402
    DeveloperAlias,
    DeveloperIdentity,
    PatternCache,
)
from .database_metrics_models import (  # noqa: E402
    CICDPipelineCache,
    ClassificationModel,
    ConfluencePageCache,
    IssueCache,
    LLMUsageStats,
    PullRequestCache,
    QualitativeCommitData,
    TicketingActivityCache,
    TrainingData,
    TrainingSession,
)

# Re-export everything for backward compatibility
__all__ = [
    "Base",
    "utcnow_tz_aware",
    "CachedCommit",
    "DeveloperIdentity",
    "DeveloperAlias",
    "PullRequestCache",
    "IssueCache",
    "QualitativeCommitData",
    "PatternCache",
    "LLMUsageStats",
    "TrainingData",
    "RepositoryAnalysisStatus",
    "TrainingSession",
    "ClassificationModel",
    "DailyCommitBatch",
    "DetailedTicketData",
    "CommitClassificationBatch",
    "CommitTicketCorrelation",
    "DailyMetrics",
    "WeeklyTrends",
    "CICDPipelineCache",
    "WeeklyFetchStatus",
    "SchemaVersion",
    "TicketingActivityCache",
    "ConfluencePageCache",
    "Database",
]


class Database:
    """Database connection manager with robust permission handling."""

    # Schema version constants
    CURRENT_SCHEMA_VERSION = "2.0"  # Timezone-aware timestamps
    LEGACY_SCHEMA_VERSION = "1.0"  # Timezone-naive timestamps

    def __init__(self, db_path: Path):
        """
        Initialize database connection with proper error handling.

        WHY: This method handles various permission scenarios that can occur
        in different deployment environments:
        - Readonly filesystems (Docker containers, CI/CD)
        - Permission denied on directory creation
        - Database file creation failures
        - Fallback to memory database when persistence isn't possible

        DESIGN DECISION: Uses fallback mechanisms rather than failing hard,
        allowing the application to continue running even in restricted environments.

        Args:
            db_path: Path to the SQLite database file

        Raises:
            RuntimeError: If database initialization fails completely
        """
        self.db_path = db_path
        self.is_readonly_fallback = False
        self.engine = None
        self.SessionLocal = None

        # Try to create database with proper error handling
        self._initialize_database()

    def _initialize_database(self) -> None:
        """
        Initialize database with comprehensive error handling.

        WHY: Database initialization can fail for multiple reasons:
        1. Directory doesn't exist and can't be created (permissions)
        2. Directory exists but database file can't be created (readonly filesystem)
        3. Database file exists but is readonly
        4. Filesystem is completely readonly (containers, CI)

        APPROACH: Try primary location first, then fallback strategies
        """
        # Strategy 1: Try primary database location
        if self._try_primary_database():
            return

        # Strategy 2: Try temp directory fallback
        if self._try_temp_database_fallback():
            return

        # Strategy 3: Use in-memory database as last resort
        self._use_memory_database_fallback()

    def _try_primary_database(self) -> bool:
        """
        Attempt to create database at the primary location.

        Returns:
            True if successful, False if fallback needed
        """
        try:
            # Check if we can create the directory
            if not self._ensure_directory_writable(self.db_path.parent):
                return False

            # Check if database file can be created/accessed
            if not self._ensure_database_writable(self.db_path):
                return False

            # Try to create the database
            self.engine = create_engine(
                f"sqlite:///{self.db_path}",
                # Add connection args to handle locked databases better
                connect_args={
                    "timeout": 30,  # 30 second timeout for database locks
                    "check_same_thread": False,  # Allow multi-threading
                },
            )

            # Check schema version BEFORE creating tables to detect legacy databases
            self.SessionLocal = sessionmaker(bind=self.engine)
            needs_migration = self._check_schema_version_before_create()

            # Create/update tables
            Base.metadata.create_all(self.engine)

            # Perform migration if needed (after tables are created/updated)
            if needs_migration:
                self._perform_schema_migration()
            else:
                # No migration needed - record current schema version if not already recorded
                self._ensure_schema_version_recorded()

            # Apply other migrations for existing databases
            self._apply_migrations()

            # Test that we can actually write to the database
            self._test_database_write()

            logger.info(f"Database initialized successfully at: {self.db_path}")
            return True

        except (OperationalError, OSError, PermissionError) as e:
            logger.warning(f"Failed to initialize primary database at {self.db_path}: {e}")
            return False

    def _try_temp_database_fallback(self) -> bool:
        """
        Try to create database in system temp directory as fallback.

        Returns:
            True if successful, False if fallback needed
        """
        try:
            # Create a temp file that will persist for the session
            temp_dir = Path(tempfile.gettempdir()) / "gitflow-analytics-cache"
            temp_dir.mkdir(exist_ok=True, parents=True)

            # Use the same filename but in temp directory
            temp_db_path = temp_dir / self.db_path.name

            self.engine = create_engine(
                f"sqlite:///{temp_db_path}",
                connect_args={
                    "timeout": 30,
                    "check_same_thread": False,
                },
            )

            # Check schema version BEFORE creating tables to detect legacy databases
            self.SessionLocal = sessionmaker(bind=self.engine)
            needs_migration = self._check_schema_version_before_create()

            # Create/update tables
            Base.metadata.create_all(self.engine)

            # Perform migration if needed (after tables are created/updated)
            if needs_migration:
                self._perform_schema_migration()
            else:
                # No migration needed - record current schema version if not already recorded
                self._ensure_schema_version_recorded()

            # Apply other migrations for existing databases
            self._apply_migrations()

            # Test write capability
            self._test_database_write()

            logger.warning(
                f"Primary database location not writable. Using temp fallback: {temp_db_path}"
            )
            self.db_path = temp_db_path  # Update path for reference
            return True

        except (OperationalError, OSError, PermissionError) as e:
            logger.warning(f"Temp database fallback failed: {e}")
            return False

    def _use_memory_database_fallback(self) -> None:
        """
        Use in-memory SQLite database as last resort.

        This allows the application to function even in completely readonly environments,
        but data will not persist between runs.
        """
        try:
            logger.warning(
                "All persistent database options failed. Using in-memory database. "
                "Data will not persist between runs."
            )

            self.engine = create_engine(
                "sqlite:///:memory:", connect_args={"check_same_thread": False}
            )

            # Check schema version BEFORE creating tables to detect legacy databases
            self.SessionLocal = sessionmaker(bind=self.engine)
            needs_migration = self._check_schema_version_before_create()

            # Create/update tables
            Base.metadata.create_all(self.engine)

            # Perform migration if needed (after tables are created/updated)
            if needs_migration:
                self._perform_schema_migration()
            else:
                # No migration needed - record current schema version if not already recorded
                self._ensure_schema_version_recorded()

            # Apply other migrations for existing databases
            self._apply_migrations()

            self.is_readonly_fallback = True

            # Test that memory database works
            self._test_database_write()

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize any database (including in-memory fallback): {e}. "
                "This may indicate a deeper system issue."
            ) from e

    def _ensure_directory_writable(self, directory: Path) -> bool:
        """
        Ensure directory exists and is writable.

        Args:
            directory: Directory to check/create

        Returns:
            True if directory is writable, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            directory.mkdir(parents=True, exist_ok=True)

            # Test write permissions by creating a temporary file
            test_file = directory / ".write_test"
            test_file.touch()
            test_file.unlink()  # Clean up

            return True

        except (PermissionError, OSError) as e:
            logger.debug(f"Directory {directory} is not writable: {e}")
            return False

    def _ensure_database_writable(self, db_path: Path) -> bool:
        """
        Check if database file can be created or is writable if it exists.

        Args:
            db_path: Path to the database file

        Returns:
            True if database file is writable, False otherwise
        """
        try:
            if db_path.exists():
                # Check if existing file is writable
                if not os.access(db_path, os.W_OK):
                    logger.debug(f"Database file {db_path} exists but is not writable")
                    return False
            else:
                # Test if we can create the file
                db_path.touch()
                db_path.unlink()  # Clean up test file

            return True

        except (PermissionError, OSError) as e:
            logger.debug(f"Cannot create/write database file {db_path}: {e}")
            return False

    def _test_database_write(self) -> None:
        """
        Test that we can actually write to the database.

        Raises:
            OperationalError: If database write test fails
        """
        try:
            # Try a simple write operation to verify database is writable
            session = self.get_session()
            try:
                # Just test that we can begin a transaction and rollback
                session.execute(text("SELECT 1"))
                session.rollback()
            finally:
                session.close()

        except Exception as e:
            raise OperationalError(f"Database write test failed: {e}", None, None) from e

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def init_db(self) -> None:
        """Initialize database tables and apply migrations."""
        needs_migration = self._check_schema_version_before_create()
        Base.metadata.create_all(self.engine)
        if needs_migration:
            self._perform_schema_migration()
        else:
            self._ensure_schema_version_recorded()
        self._apply_migrations()

    def _check_schema_version_before_create(self) -> bool:
        """Check if database needs migration BEFORE create_all is called.

        WHY: We need to check for legacy databases BEFORE creating new tables,
        otherwise we can't distinguish between a fresh database and a legacy one.

        Returns:
            True if migration is needed, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                # Check if schema_version table exists
                result = conn.execute(
                    text(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
                    )
                )
                schema_table_exists = result.fetchone() is not None

                if schema_table_exists:
                    # Check current version
                    result = conn.execute(
                        text("SELECT version FROM schema_version ORDER BY id DESC LIMIT 1")
                    )
                    row = result.fetchone()

                    if row and row[0] != self.CURRENT_SCHEMA_VERSION:
                        # Version mismatch - needs migration
                        logger.warning(
                            f"⚠️  Schema version mismatch: {row[0]} → {self.CURRENT_SCHEMA_VERSION}"
                        )
                        return True
                    # else: Already at current version or no version record yet
                    return False
                else:
                    # No schema_version table - check if this is legacy or new
                    result = conn.execute(
                        text(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name='cached_commits'"
                        )
                    )
                    has_cached_commits = result.fetchone() is not None

                    if has_cached_commits:
                        # Check if table has data
                        result = conn.execute(text("SELECT COUNT(*) FROM cached_commits"))
                        commit_count = result.fetchone()[0]

                        if commit_count > 0:
                            # Legacy database with data - needs migration
                            logger.warning("⚠️  Old cache schema detected (v1.0 → v2.0)")
                            logger.info("   This is a one-time operation due to timezone fix")
                            return True

                    # New database or empty legacy database - no migration needed
                    return False

        except Exception as e:
            # Don't fail initialization due to schema check issues
            logger.debug(f"Schema version check failed: {e}")
            return False

    def _perform_schema_migration(self) -> None:
        """Perform the actual schema migration after tables are created.

        WHY: Separating migration from detection allows us to update table schemas
        via create_all before clearing/migrating data.
        """
        try:
            with self.engine.connect() as conn:
                logger.info("🔄 Automatically upgrading cache database...")
                logger.info("   Clearing old cache data (timezone schema incompatible)...")

                # Clear cached data tables
                conn.execute(text("DELETE FROM cached_commits"))
                conn.execute(text("DELETE FROM pull_request_cache"))
                conn.execute(text("DELETE FROM issue_cache"))
                conn.execute(text("DELETE FROM repository_analysis_status"))

                # Also clear qualitative analysis data if it exists
                try:
                    conn.execute(text("DELETE FROM qualitative_commits"))
                    conn.execute(text("DELETE FROM pattern_cache"))
                except OperationalError as e:
                    # These tables might not exist in all database schema versions
                    logger.debug(
                        f"Non-critical: optional cache tables not present during migration: {e}"
                    )

                conn.commit()

                # Record the schema upgrade
                self._record_schema_version(
                    conn,
                    self.CURRENT_SCHEMA_VERSION,
                    self.LEGACY_SCHEMA_VERSION,
                    "Migrated to timezone-aware timestamps (v2.0)",
                )

                logger.info("   Migration complete - cache will be rebuilt on next analysis")
                logger.info("✅ Cache database upgraded successfully")

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            # Don't raise - let the system continue and rebuild cache from scratch

    def _ensure_schema_version_recorded(self) -> None:
        """Ensure schema version is recorded for databases that didn't need migration.

        WHY: Fresh databases and already-migrated databases need to have their
        schema version recorded for future migration detection.
        """
        try:
            with self.engine.connect() as conn:
                # Check if version is already recorded
                result = conn.execute(text("SELECT COUNT(*) FROM schema_version"))
                count = result.fetchone()[0]

                if count == 0:
                    # No version recorded - this is a fresh database
                    self._record_schema_version(
                        conn, self.CURRENT_SCHEMA_VERSION, None, "Initial schema creation"
                    )
                    logger.debug(f"Recorded initial schema version: {self.CURRENT_SCHEMA_VERSION}")

        except Exception as e:
            # Don't fail if we can't record version
            logger.debug(f"Could not ensure schema version recorded: {e}")

    def _record_schema_version(
        self, conn, version: str, previous_version: Optional[str], notes: Optional[str]
    ) -> None:
        """Record schema version in the database.

        Args:
            conn: Database connection
            version: New schema version
            previous_version: Previous schema version (None for initial)
            notes: Migration notes
        """
        try:
            from datetime import datetime, timezone

            # Insert new schema version record
            conn.execute(
                text(
                    """
                INSERT INTO schema_version (version, upgraded_at, previous_version, migration_notes)
                VALUES (:version, :upgraded_at, :previous_version, :notes)
            """
                ),
                {
                    "version": version,
                    "upgraded_at": datetime.now(timezone.utc),
                    "previous_version": previous_version,
                    "notes": notes,
                },
            )
            conn.commit()
        except Exception as e:
            logger.debug(f"Could not record schema version: {e}")

    def _apply_migrations(self) -> None:
        """Apply database migrations for backward compatibility.

        This method adds new columns to existing tables without losing data.
        """
        try:
            with self.engine.connect() as conn:
                # Check if filtered columns exist in cached_commits table
                result = conn.execute(text("PRAGMA table_info(cached_commits)"))
                columns = {row[1] for row in result}

                # Add filtered_insertions column if it doesn't exist
                if "filtered_insertions" not in columns:
                    logger.info("Adding filtered_insertions column to cached_commits table")
                    try:
                        conn.execute(
                            text(
                                "ALTER TABLE cached_commits ADD COLUMN filtered_insertions INTEGER DEFAULT 0"
                            )
                        )
                        conn.commit()
                    except Exception as e:
                        logger.debug(f"Column may already exist or database is readonly: {e}")

                # Add filtered_deletions column if it doesn't exist
                if "filtered_deletions" not in columns:
                    logger.info("Adding filtered_deletions column to cached_commits table")
                    try:
                        conn.execute(
                            text(
                                "ALTER TABLE cached_commits ADD COLUMN filtered_deletions INTEGER DEFAULT 0"
                            )
                        )
                        conn.commit()
                    except Exception as e:
                        logger.debug(f"Column may already exist or database is readonly: {e}")

                # Initialize filtered columns with existing values for backward compatibility
                if "filtered_insertions" not in columns or "filtered_deletions" not in columns:
                    logger.info("Initializing filtered columns with existing values")
                    try:
                        conn.execute(
                            text(
                                """
                            UPDATE cached_commits
                            SET filtered_insertions = COALESCE(filtered_insertions, insertions),
                                filtered_deletions = COALESCE(filtered_deletions, deletions)
                            WHERE filtered_insertions IS NULL OR filtered_deletions IS NULL
                        """
                            )
                        )
                        conn.commit()
                    except Exception as e:
                        logger.debug(f"Could not initialize filtered columns: {e}")

                # --- Enhanced PR tracking columns (v3.0 migration) ---
                # WHY: These columns are added incrementally to avoid destroying existing
                # cached PR data. New columns default to NULL/0 for existing rows.
                self._migrate_pull_request_cache_v3(conn)

                # --- PR state columns (v4.0 migration) ---
                # WHY: pr_state / closed_at / is_merged were not present in older databases.
                # They are added here without touching existing rows.
                self._migrate_pull_request_cache_v4(conn)

                # --- Complexity column (v5.0 migration) ---
                # WHY: complexity was added to qualitative_commits to capture engineering
                # sophistication ratings (1-5) from LLM classification.  Existing rows get
                # NULL, which is the correct default for rule-based classifications.
                self._migrate_qualitative_commits_v5(conn)

                # --- AI tool tracking columns (v6.0 migration) ---
                # WHY: Per-developer AI tool usage metrics were added to daily_metrics to
                # enable tracking of Claude Code, Copilot, and Cursor adoption over time.
                # Existing rows receive default values (0 / empty string).
                self._migrate_daily_metrics_ai_columns(conn)

                # --- 14-day churn rate column (v7.0 migration) ---
                # WHY: churn_rate_14d on weekly_trends provides a code quality proxy
                # metric showing how much of a week's added lines were deleted within
                # the following 14 days. Higher churn correlates with lower quality /
                # higher rework, and is especially relevant for AI-assisted code.
                self._migrate_weekly_trends_churn_column(conn)  # type: ignore[attr-defined]

                # --- Velocity columns (v8.0 migration) ---
                # WHY: PR cycle time, throughput, revision rate, and story points
                # delivered are aggregated per-week onto weekly_trends so that
                # the velocity report can be generated without re-querying
                # pull_request_cache at report time.
                self._migrate_weekly_trends_velocity_columns(conn)  # type: ignore[attr-defined]

                # --- AI confidence scoring columns (v9.0 migration) ---
                # WHY: NLP-based heuristic scoring of commit messages for AI-generation
                # probability is stored per-commit to power reporting and trending.
                # Existing rows receive NULL / empty string defaults.
                self._migrate_cached_commits_ai_columns(conn)

                # --- Ticketing activity tracking tables (v10.0 migration) ---
                # WHY: GitHub Issues + Confluence activity are tracked in new tables
                # (ticketing_activity_cache, confluence_page_cache).  The migration
                # is a no-op when the tables already exist with the expected columns
                # (Base.metadata.create_all handles table creation for new databases);
                # it only adds missing columns to existing tables.
                self._migrate_ticketing_activity_v10(conn)

                # --- developer_identities NULL back-fill (issue #39) ---
                # WHY: Older databases created before total_commits /
                # total_story_points had a server-side DEFAULT may contain NULL
                # values in these columns.  NULL values crash downstream
                # sorted() calls in get_developer_stats() with TypeError when
                # comparing None vs int.  This idempotent UPDATE sets any
                # legacy NULL rows to 0.
                self._migrate_developer_identities_null_stats(conn)

        except Exception as e:
            # Don't fail if migrations can't be applied (e.g., in-memory database)
            logger.debug(
                f"Could not apply migrations (may be normal for new/memory databases): {e}"
            )

    def _migrate_pull_request_cache_v3(self, conn) -> None:
        """Add enhanced PR tracking columns to pull_request_cache (v3.0 migration).

        WHY: These columns are new in v3.0 of the PR schema. Existing databases will
        not have them, so we use ALTER TABLE to add them without touching existing rows.
        SQLite does not support adding multiple columns in one statement, so each column
        is added individually inside its own try/except to be idempotent.

        Columns added:
            review_comments_count  - Inline review comment count
            pr_comments_count      - General PR/issue comment count
            approvals_count        - Number of approved reviews
            change_requests_count  - Number of change-request reviews
            reviewers              - JSON list of reviewer logins
            approved_by            - JSON list of approving reviewer logins
            time_to_first_review_hours - Hours from open to first review activity
            revision_count         - Commit pushes after PR opened
            changed_files          - Files changed (was discarded pre-v3.0)
            additions              - Lines added (was discarded pre-v3.0)
            deletions              - Lines removed (was discarded pre-v3.0)
        """
        try:
            result = conn.execute(text("PRAGMA table_info(pull_request_cache)"))
            existing_columns = {row[1] for row in result}
        except Exception as e:
            logger.debug(f"Could not read pull_request_cache schema: {e}")
            return

        # Map of column_name -> DDL fragment (type + default).
        # All nullable with sensible defaults so existing rows remain valid.
        new_columns: list[tuple[str, str]] = [
            ("review_comments_count", "INTEGER DEFAULT 0"),
            ("pr_comments_count", "INTEGER DEFAULT 0"),
            ("approvals_count", "INTEGER DEFAULT 0"),
            ("change_requests_count", "INTEGER DEFAULT 0"),
            ("reviewers", "TEXT"),  # JSON stored as TEXT in SQLite
            ("approved_by", "TEXT"),  # JSON stored as TEXT in SQLite
            ("time_to_first_review_hours", "REAL"),
            ("revision_count", "INTEGER DEFAULT 0"),
            ("changed_files", "INTEGER DEFAULT 0"),
            ("additions", "INTEGER DEFAULT 0"),
            ("deletions", "INTEGER DEFAULT 0"),
        ]

        added: list[str] = []
        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                try:
                    conn.execute(
                        text(f"ALTER TABLE pull_request_cache ADD COLUMN {col_name} {col_type}")
                    )
                    conn.commit()
                    added.append(col_name)
                except Exception as e:
                    # Column may already exist in a concurrent scenario or read-only DB.
                    logger.debug(
                        f"Could not add pull_request_cache.{col_name} "
                        f"(may already exist or DB is readonly): {e}"
                    )

        if added:
            logger.info(
                "Applied pull_request_cache v3.0 migration: added columns %s",
                ", ".join(added),
            )

    def _migrate_pull_request_cache_v4(self, conn) -> None:
        """Add PR state columns to pull_request_cache (v4.0 migration).

        WHY: pr_state, closed_at, and is_merged were added in v4.0 to enable
        rejection-rate reporting.  Existing databases will not have these columns,
        so we use ALTER TABLE to add them without touching existing rows.  SQLite
        does not support adding multiple columns in one statement, so each column
        is added individually inside its own try/except to be idempotent.

        Columns added:
            pr_state   - TEXT: "open", "closed" (rejected), or "merged"
            closed_at  - DATETIME: when the PR was closed (merge or rejection)
            is_merged  - INTEGER (BOOLEAN): 1 if merged, 0 if closed without merge
        """
        try:
            result = conn.execute(text("PRAGMA table_info(pull_request_cache)"))
            existing_columns = {row[1] for row in result}
        except Exception as e:
            logger.debug(f"Could not read pull_request_cache schema for v4 migration: {e}")
            return

        # Map of column_name -> DDL fragment (type + default).
        # All nullable so existing rows remain valid after the migration.
        new_columns: list[tuple[str, str]] = [
            ("pr_state", "TEXT"),
            ("closed_at", "DATETIME"),
            ("is_merged", "INTEGER"),  # SQLite stores booleans as INTEGER
        ]

        added: list[str] = []
        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                try:
                    conn.execute(
                        text(f"ALTER TABLE pull_request_cache ADD COLUMN {col_name} {col_type}")
                    )
                    conn.commit()
                    added.append(col_name)
                except Exception as e:
                    logger.debug(
                        f"Could not add pull_request_cache.{col_name} "
                        f"(may already exist or DB is readonly): {e}"
                    )

        if added:
            logger.info(
                "Applied pull_request_cache v4.0 migration: added columns %s",
                ", ".join(added),
            )

    def _migrate_qualitative_commits_v5(self, conn) -> None:
        """Add complexity column to qualitative_commits (v5.0 migration).

        WHY: The complexity field (1-5 sophistication rating) was introduced to
        capture engineering complexity as judged by LLM classification.  Existing
        rows and rule-based classifications use NULL, which means "not rated".
        SQLite does not support adding multiple columns in one statement, so each
        column is added individually inside its own try/except to be idempotent.

        Columns added:
            complexity  - INTEGER (1-5): engineering sophistication rating; NULL for
                          rule-based classifications or pre-existing rows.
        """
        try:
            result = conn.execute(text("PRAGMA table_info(qualitative_commits)"))
            existing_columns = {row[1] for row in result}
        except Exception as e:
            logger.debug(f"Could not read qualitative_commits schema for v5 migration: {e}")
            return

        if "complexity" not in existing_columns:
            try:
                conn.execute(text("ALTER TABLE qualitative_commits ADD COLUMN complexity INTEGER"))
                conn.commit()
                logger.info("Applied qualitative_commits v5.0 migration: added complexity column")
            except Exception as e:
                # Column may already exist in a concurrent scenario or read-only DB.
                logger.debug(
                    f"Could not add qualitative_commits.complexity "
                    f"(may already exist or DB is readonly): {e}"
                )

    def _migrate_daily_metrics_ai_columns(self, conn) -> None:
        """Add AI tool tracking columns to daily_metrics (v6.0 migration).

        WHY: These five columns were introduced to track per-developer AI tool
        usage (Claude Code, Copilot, Cursor) at commit level.  Existing databases
        will not have them, so ALTER TABLE is used to add them without touching
        existing rows.  Each column is added individually inside its own
        try/except to be idempotent.

        Columns added:
            ai_assisted_commits  - INTEGER: commits with any AI tool marker
            ai_generated_commits - INTEGER: commits that appear fully AI-generated
            ai_tool_primary      - VARCHAR: dominant tool for the day
            ai_assisted_lines    - INTEGER: lines added in AI-assisted commits
            ai_generated_lines   - INTEGER: lines added in AI-generated commits
        """
        try:
            result = conn.execute(text("PRAGMA table_info(daily_metrics)"))
            existing_columns = {row[1] for row in result}
        except Exception as e:
            logger.debug(f"Could not read daily_metrics schema for v6 migration: {e}")
            return

        new_columns: list[tuple[str, str]] = [
            ("ai_assisted_commits", "INTEGER DEFAULT 0"),
            ("ai_generated_commits", "INTEGER DEFAULT 0"),
            ("ai_tool_primary", "VARCHAR DEFAULT ''"),
            ("ai_assisted_lines", "INTEGER DEFAULT 0"),
            ("ai_generated_lines", "INTEGER DEFAULT 0"),
        ]

        added: list[str] = []
        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                try:
                    conn.execute(
                        text(f"ALTER TABLE daily_metrics ADD COLUMN {col_name} {col_type}")
                    )
                    conn.commit()
                    added.append(col_name)
                except Exception as e:
                    logger.debug(
                        f"Could not add daily_metrics.{col_name} "
                        f"(may already exist or DB is readonly): {e}"
                    )

        if added:
            logger.info(
                "Applied daily_metrics v6.0 migration: added columns %s",
                ", ".join(added),
            )

    def _migrate_weekly_trends_churn_column(self, conn) -> None:
        """Add 14-day churn rate column to weekly_trends (v7.0 migration).

        WHY: churn_rate_14d captures the fraction of a week's added lines that
        are subsequently deleted within 14 days — a proxy for code quality and
        AI-assisted rework. Added as a nullable REAL column so existing rows
        remain valid with the SQLite DEFAULT 0.0 sentinel.

        Columns added:
            churn_rate_14d  - REAL: fraction of week's lines deleted in next 14 days
        """
        try:
            result = conn.execute(text("PRAGMA table_info(weekly_trends)"))
            existing_columns = {row[1] for row in result}
        except Exception as e:
            logger.debug(f"Could not read weekly_trends schema for v7 migration: {e}")
            return

        if "churn_rate_14d" not in existing_columns:
            try:
                conn.execute(
                    text("ALTER TABLE weekly_trends ADD COLUMN churn_rate_14d REAL DEFAULT 0.0")
                )
                conn.commit()
                logger.info("Applied weekly_trends v7.0 migration: added churn_rate_14d column")
            except Exception as e:
                logger.debug(
                    f"Could not add weekly_trends.churn_rate_14d "
                    f"(may already exist or DB is readonly): {e}"
                )

    def _migrate_weekly_trends_velocity_columns(self, conn) -> None:
        """Add velocity columns to weekly_trends (v8.0 migration).

        WHY: PR cycle time, throughput, revision rate, and story points delivered
        are now pre-aggregated per-week on weekly_trends to support the native
        velocity report (Issue #25).  Added as separate columns so existing rows
        remain valid with their DEFAULT values.  Each ALTER TABLE is inside its
        own try/except to be idempotent.

        Columns added:
            prs_merged             - INTEGER: merged PRs in the week
            avg_cycle_time_hrs     - REAL: mean PR open→merge hours (excl. outliers)
            median_cycle_time_hrs  - REAL: median PR open→merge hours (excl. outliers)
            avg_revision_count     - REAL: mean commit pushes after PR open
            story_points_delivered - INTEGER: sum of story points on merged PRs
        """
        try:
            result = conn.execute(text("PRAGMA table_info(weekly_trends)"))
            existing_columns = {row[1] for row in result}
        except Exception as e:
            logger.debug(f"Could not read weekly_trends schema for v8 migration: {e}")
            return

        new_columns: list[tuple[str, str]] = [
            ("prs_merged", "INTEGER DEFAULT 0"),
            ("avg_cycle_time_hrs", "REAL DEFAULT 0.0"),
            ("median_cycle_time_hrs", "REAL DEFAULT 0.0"),
            ("avg_revision_count", "REAL DEFAULT 0.0"),
            ("story_points_delivered", "INTEGER DEFAULT 0"),
        ]

        added: list[str] = []
        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                try:
                    conn.execute(
                        text(f"ALTER TABLE weekly_trends ADD COLUMN {col_name} {col_type}")
                    )
                    conn.commit()
                    added.append(col_name)
                except Exception as e:
                    logger.debug(
                        f"Could not add weekly_trends.{col_name} "
                        f"(may already exist or DB is readonly): {e}"
                    )

        if added:
            logger.info(
                "Applied weekly_trends v8.0 migration: added columns %s",
                ", ".join(added),
            )

    def _migrate_cached_commits_ai_columns(self, conn) -> None:
        """Add ai_confidence_score and ai_detection_method to cached_commits (v9.0 migration).

        WHY: NLP-based heuristic scoring of commit messages for AI-generation probability
        is stored per-commit so that per-developer and per-project AI adoption trends can
        be computed without re-analysing raw Git data.  Existing rows receive NULL /
        empty-string defaults which are handled gracefully in reporting.

        Columns added:
            ai_confidence_score  - REAL: heuristic confidence 0.0-1.0 (NULL = not scored)
            ai_detection_method  - VARCHAR: 'pattern', 'nlp_heuristic', 'none', or ''
        """
        try:
            result = conn.execute(text("PRAGMA table_info(cached_commits)"))
            existing = {row[1] for row in result}
        except Exception as e:
            logger.debug(f"Could not read cached_commits schema for v9 migration: {e}")
            return

        for col, col_type in [
            ("ai_confidence_score", "REAL"),
            ("ai_detection_method", "VARCHAR DEFAULT ''"),
        ]:
            if col not in existing:
                try:
                    conn.execute(text(f"ALTER TABLE cached_commits ADD COLUMN {col} {col_type}"))
                    conn.commit()
                    logger.info(f"Applied cached_commits v9.0 migration: added {col}")
                except Exception as e:
                    logger.debug(f"Could not add cached_commits.{col}: {e}")

    def _migrate_ticketing_activity_v10(self, conn) -> None:
        """Idempotent v10 migration for ticketing_activity_cache and confluence_page_cache.

        WHY: These tables are created via Base.metadata.create_all for fresh databases.
        For existing databases that pre-date v10 the tables simply don't exist — which
        is fine because create_all already created them.  However, we defensively add
        missing columns via ALTER TABLE so that if an older partial schema somehow
        exists we can bring it up to current. Each ALTER TABLE is wrapped in its own
        try/except to remain idempotent.

        Args:
            conn: Active SQLAlchemy connection.
        """
        # ticketing_activity_cache columns — all nullable / defaulted for safe migration
        ticketing_columns: list[tuple[str, str]] = [
            ("platform", "TEXT"),
            ("item_id", "TEXT"),
            ("item_type", "TEXT"),
            ("repo_or_space", "TEXT"),
            ("actor", "TEXT"),
            ("actor_display_name", "TEXT"),
            ("actor_email", "TEXT"),
            ("action", "TEXT"),
            ("activity_at", "DATETIME"),
            ("item_title", "TEXT"),
            ("item_status", "TEXT"),
            ("item_url", "TEXT"),
            ("linked_ticket_id", "TEXT"),
            ("comment_count", "INTEGER DEFAULT 0"),
            ("reaction_count", "INTEGER DEFAULT 0"),
            ("platform_data", "JSON"),
            ("cached_at", "DATETIME"),
        ]

        try:
            result = conn.execute(text("PRAGMA table_info(ticketing_activity_cache)"))
            existing = {row[1] for row in result}
        except Exception as e:
            logger.debug(f"Could not read ticketing_activity_cache schema: {e}")
            existing = set()

        # Only attempt ALTER TABLE if the table exists with some columns already
        if existing:
            for col_name, col_type in ticketing_columns:
                if col_name not in existing:
                    try:
                        conn.execute(
                            text(
                                f"ALTER TABLE ticketing_activity_cache ADD COLUMN {col_name} {col_type}"
                            )
                        )
                        conn.commit()
                        logger.info(
                            f"Applied ticketing_activity_cache v10 migration: added {col_name}"
                        )
                    except Exception as e:
                        logger.debug(f"Could not add ticketing_activity_cache.{col_name}: {e}")

        # confluence_page_cache columns
        confluence_columns: list[tuple[str, str]] = [
            ("page_id", "TEXT"),
            ("space_key", "TEXT"),
            ("title", "TEXT"),
            ("version", "INTEGER"),
            ("author", "TEXT"),
            ("author_email", "TEXT"),
            ("last_editor", "TEXT"),
            ("last_editor_email", "TEXT"),
            ("created_at", "DATETIME"),
            ("updated_at", "DATETIME"),
            ("labels", "JSON"),
            ("ancestor_ids", "JSON"),
            ("page_url", "TEXT"),
            ("platform_data", "JSON"),
            ("cached_at", "DATETIME"),
        ]

        try:
            result = conn.execute(text("PRAGMA table_info(confluence_page_cache)"))
            existing_conf = {row[1] for row in result}
        except Exception as e:
            logger.debug(f"Could not read confluence_page_cache schema: {e}")
            existing_conf = set()

        if existing_conf:
            for col_name, col_type in confluence_columns:
                if col_name not in existing_conf:
                    try:
                        conn.execute(
                            text(
                                f"ALTER TABLE confluence_page_cache ADD COLUMN {col_name} {col_type}"
                            )
                        )
                        conn.commit()
                        logger.info(
                            f"Applied confluence_page_cache v10 migration: added {col_name}"
                        )
                    except Exception as e:
                        logger.debug(f"Could not add confluence_page_cache.{col_name}: {e}")

    def _migrate_developer_identities_null_stats(self, conn) -> None:
        """Back-fill NULL total_commits / total_story_points on developer_identities.

        WHY: Older databases created before these columns had a server-side
        DEFAULT may have NULL values in ``total_commits`` and
        ``total_story_points``.  Downstream ``sorted()`` calls in
        ``get_developer_stats()`` crash with ``TypeError`` when comparing
        ``None`` vs ``int``.  This idempotent UPDATE sets any legacy NULL rows
        to 0 so subsequent reads are always safe.  See issue #39.

        Args:
            conn: Active SQLAlchemy connection.
        """
        try:
            # Confirm the table exists before running UPDATEs.  Fresh databases
            # that were just created via ``Base.metadata.create_all`` will have
            # the table with defaults, so the UPDATE is a no-op.
            result = conn.execute(text("PRAGMA table_info(developer_identities)"))
            existing_columns = {row[1] for row in result}
        except Exception as e:
            logger.debug(f"Could not read developer_identities schema: {e}")
            return

        if not existing_columns:
            # Table does not exist yet — nothing to back-fill.
            return

        for column in ("total_commits", "total_story_points"):
            if column not in existing_columns:
                continue
            try:
                # Column name is hard-coded to the allow-list above, never
                # user-supplied, so f-string interpolation is safe here.
                sql = f"UPDATE developer_identities SET {column} = 0 WHERE {column} IS NULL"  # nosec B608
                res = conn.execute(text(sql))
                conn.commit()
                # SQLAlchemy's rowcount is -1 on some drivers; only log when > 0.
                rowcount = getattr(res, "rowcount", -1)
                if rowcount and rowcount > 0:
                    logger.info(
                        "Back-filled %d NULL %s rows in developer_identities (issue #39)",
                        rowcount,
                        column,
                    )
            except Exception as e:
                logger.debug(f"Could not back-fill developer_identities.{column}: {e}")
