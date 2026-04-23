"""Tests guarding against NULL total_commits / total_story_points in
developer_identities (issue #39).

WHY: Older databases — migrated via ``ALTER TABLE`` without a ``DEFAULT`` —
can contain NULL values in ``total_commits`` and ``total_story_points``.
Those NULLs historically caused ``TypeError: '<' not supported between
instances of 'NoneType' and 'int'`` inside ``get_developer_stats()`` because
of the ``sorted(..., key=lambda x: x["total_commits"])`` call, and similar
failures inside ``merge_identities()`` for the numeric/datetime fields.

These tests lock in the defensive behaviour so the regression cannot
silently return.
"""

from datetime import datetime, timezone

from sqlalchemy import text  # type: ignore[import-not-found]

from gitflow_analytics.core.identity import DeveloperIdentityResolver
from gitflow_analytics.models.database import DeveloperIdentity


def _recreate_developer_identities_legacy_schema(resolver: DeveloperIdentityResolver) -> None:
    """Rebuild ``developer_identities`` with the legacy (nullable) schema.

    WHY: The current model declares ``total_commits`` / ``total_story_points`` as
    ``NOT NULL`` with a server default.  To test the defensive behaviour we need
    to simulate the pre-fix state where those columns were nullable and no
    DEFAULT existed — exactly the state an ALTER TABLE-based migration would
    have left behind on older databases.
    """
    assert resolver.db is not None
    with resolver.db.engine.connect() as conn:  # type: ignore[union-attr]
        conn.execute(text("DROP TABLE IF EXISTS developer_identities"))
        conn.execute(
            text(
                """
                CREATE TABLE developer_identities (
                    id INTEGER PRIMARY KEY,
                    canonical_id TEXT UNIQUE NOT NULL,
                    primary_name TEXT NOT NULL,
                    primary_email TEXT NOT NULL,
                    github_username TEXT,
                    total_commits INTEGER,
                    total_story_points INTEGER,
                    first_seen DATETIME,
                    last_seen DATETIME,
                    created_at DATETIME,
                    updated_at DATETIME
                )
                """
            )
        )
        conn.commit()


class TestGetDeveloperStatsWithNullFields:
    """Verify get_developer_stats() tolerates NULL total_commits rows."""

    def test_get_developer_stats_with_null_total_commits(self, temp_dir):
        """Rows with NULL total_commits must not crash sorted()."""
        db_path = temp_dir / "identities_null.db"
        resolver = DeveloperIdentityResolver(db_path)

        # Recreate the table with the legacy (nullable, no DEFAULT) schema so
        # we can insert a row that mimics what an older ALTER TABLE migration
        # would have left behind.
        _recreate_developer_identities_legacy_schema(resolver)

        # Insert one "normal" row and one legacy NULL row to exercise the
        # sort comparator with a mix of integers and NULL values.
        assert resolver.db is not None
        with resolver.db.engine.connect() as conn:  # type: ignore[union-attr]
            conn.execute(
                text(
                    "INSERT INTO developer_identities "
                    "(canonical_id, primary_name, primary_email, "
                    " total_commits, total_story_points) "
                    "VALUES ('normal-id', 'Alice Normal', "
                    "'alice@example.com', 5, 3)"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO developer_identities "
                    "(canonical_id, primary_name, primary_email, "
                    " total_commits, total_story_points, first_seen, last_seen) "
                    "VALUES (:cid, :name, :email, NULL, NULL, NULL, NULL)"
                ),
                {
                    "cid": "legacy-null-id",
                    "name": "Legacy Null",
                    "email": "legacy@example.com",
                },
            )
            conn.commit()

        # Must not raise TypeError.
        stats = resolver.get_developer_stats()

        # The legacy row must be present and its NULL stats normalised to 0.
        legacy_stats = next((s for s in stats if s["canonical_id"] == "legacy-null-id"), None)
        assert legacy_stats is not None, "Legacy NULL row should be returned"
        assert legacy_stats["total_commits"] == 0
        assert legacy_stats["total_story_points"] == 0


class TestMergeIdentitiesWithNullFields:
    """Verify merge_identities() tolerates NULL numeric/datetime fields."""

    def test_merge_identities_with_null_fields(self, temp_dir):
        """Merging two identities with NULL stats/timestamps must not crash."""
        db_path = temp_dir / "identities_merge_null.db"
        resolver = DeveloperIdentityResolver(db_path)

        # Recreate the table with the legacy (nullable, no DEFAULT) schema so
        # we can insert rows mimicking the pre-fix state.
        _recreate_developer_identities_legacy_schema(resolver)

        # Insert two identities directly with NULL stats and NULL seen-dates.
        assert resolver.db is not None
        with resolver.db.engine.connect() as conn:  # type: ignore[union-attr]
            for cid, name, email in [
                ("cid-null-1", "Null One", "one@example.com"),
                ("cid-null-2", "Null Two", "two@example.com"),
            ]:
                conn.execute(
                    text(
                        "INSERT INTO developer_identities "
                        "(canonical_id, primary_name, primary_email, "
                        " total_commits, total_story_points, first_seen, last_seen) "
                        "VALUES (:cid, :name, :email, NULL, NULL, NULL, NULL)"
                    ),
                    {"cid": cid, "name": name, "email": email},
                )
            conn.commit()

        # Must not raise TypeError inside min()/max()/+ operations.
        resolver.merge_identities("cid-null-1", "cid-null-2")

        # Post-merge, the surviving identity should exist with sane values.
        with resolver.db.get_session() as session:
            survivor = (
                session.query(DeveloperIdentity)
                .filter(DeveloperIdentity.canonical_id == "cid-null-1")
                .first()
            )
            assert survivor is not None
            # NULL + NULL -> 0 + 0 == 0
            assert (survivor.total_commits or 0) == 0
            assert (survivor.total_story_points or 0) == 0
            # first_seen / last_seen should have been populated with a real
            # datetime (the ``now`` sentinel inside merge_identities()).
            # SQLite's default datetime adapter may strip tzinfo on readback,
            # so we only assert the value is a real datetime in the past.
            assert isinstance(survivor.first_seen, datetime)
            assert isinstance(survivor.last_seen, datetime)
            now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
            first_seen_naive = (
                survivor.first_seen.replace(tzinfo=None)
                if survivor.first_seen.tzinfo
                else survivor.first_seen
            )
            last_seen_naive = (
                survivor.last_seen.replace(tzinfo=None)
                if survivor.last_seen.tzinfo
                else survivor.last_seen
            )
            assert first_seen_naive <= now_naive
            assert last_seen_naive <= now_naive
