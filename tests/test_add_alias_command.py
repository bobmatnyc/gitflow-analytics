"""Tests for the add-alias CLI command."""

import json
import textwrap
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from gitflow_analytics.cli_identity_alias_ops import _load_alias_file, add_alias_command

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def minimal_config(tmp_path: Path) -> Path:
    """Config YAML with no manual_mappings section yet."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        textwrap.dedent("""\
            analysis:
              enabled: true
        """)
    )
    return cfg


@pytest.fixture()
def config_with_mapping(tmp_path: Path) -> Path:
    """Config YAML with an existing manual mapping for john@work.com."""
    cfg = tmp_path / "config.yaml"
    data = {
        "analysis": {
            "identity": {
                "manual_mappings": [
                    {
                        "primary_email": "john@work.com",
                        "aliases": ["john@gmail.com"],
                        "name": "John Doe",
                    }
                ]
            }
        }
    }
    cfg.write_text(yaml.dump(data, sort_keys=False))
    return cfg


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _read_mappings(config: Path) -> list[dict]:
    data = yaml.safe_load(config.read_text()) or {}
    return data.get("analysis", {}).get("identity", {}).get("manual_mappings", [])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_alias_adds_new_mapping(runner, minimal_config):
    """--canonical + --alias adds new mapping to config YAML."""
    result = runner.invoke(
        add_alias_command,
        [
            "--config",
            str(minimal_config),
            "--canonical",
            "jane@work.com",
            "--alias",
            "jane@gmail.com",
        ],
    )
    assert result.exit_code == 0, result.output
    mappings = _read_mappings(minimal_config)
    assert len(mappings) == 1
    assert mappings[0]["primary_email"] == "jane@work.com"
    assert "jane@gmail.com" in mappings[0]["aliases"]


def test_multiple_aliases_all_added(runner, minimal_config):
    """--alias with multiple values adds all aliases."""
    result = runner.invoke(
        add_alias_command,
        [
            "--config",
            str(minimal_config),
            "--canonical",
            "bob@work.com",
            "--alias",
            "bob@gmail.com",
            "--alias",
            "Bob Smith",
            "--alias",
            "bob@old-company.com",
        ],
    )
    assert result.exit_code == 0, result.output
    mappings = _read_mappings(minimal_config)
    assert len(mappings) == 1
    aliases = mappings[0]["aliases"]
    assert "bob@gmail.com" in aliases
    assert "Bob Smith" in aliases
    assert "bob@old-company.com" in aliases


def test_idempotent_duplicate_alias_not_added(runner, config_with_mapping):
    """Running twice with same alias doesn't duplicate."""
    # Run once to confirm baseline
    first = runner.invoke(
        add_alias_command,
        [
            "--config",
            str(config_with_mapping),
            "--canonical",
            "john@work.com",
            "--alias",
            "john@gmail.com",
        ],
    )
    assert first.exit_code == 0, first.output

    mappings = _read_mappings(config_with_mapping)
    # john@gmail.com was already present — should still be exactly one occurrence
    count = mappings[0]["aliases"].count("john@gmail.com")
    assert count == 1
    assert "already exists" in first.output.lower() or "skipped" in first.output.lower()


def test_dry_run_does_not_modify_file(runner, minimal_config):
    """--dry-run prints what would happen but doesn't modify the file."""
    original = minimal_config.read_text()
    result = runner.invoke(
        add_alias_command,
        [
            "--config",
            str(minimal_config),
            "--canonical",
            "dry@example.com",
            "--alias",
            "dry@other.com",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert minimal_config.read_text() == original
    assert "dry run" in result.output.lower() or "would add" in result.output.lower()


def test_from_file_gfa_format(runner, minimal_config, tmp_path):
    """--from-file with GFA aliases.yaml format loads correctly."""
    aliases_file = tmp_path / "aliases.yaml"
    aliases_file.write_text(
        textwrap.dedent("""\
            developer_aliases:
              - primary_email: alice@company.com
                aliases:
                  - alice@gmail.com
                  - alice@home.net
                name: Alice Wonder
        """)
    )
    result = runner.invoke(
        add_alias_command,
        ["--config", str(minimal_config), "--from-file", str(aliases_file)],
    )
    assert result.exit_code == 0, result.output
    mappings = _read_mappings(minimal_config)
    assert len(mappings) == 1
    assert mappings[0]["primary_email"] == "alice@company.com"
    assert "alice@gmail.com" in mappings[0]["aliases"]
    assert "alice@home.net" in mappings[0]["aliases"]


def test_from_file_flat_list_format(runner, minimal_config, tmp_path):
    """--from-file with flat list format."""
    aliases_file = tmp_path / "flat.yaml"
    aliases_file.write_text(
        textwrap.dedent("""\
            - canonical: eve@corp.com
              aliases:
                - eve@personal.io
              name: Eve Tester
        """)
    )
    result = runner.invoke(
        add_alias_command,
        ["--config", str(minimal_config), "--from-file", str(aliases_file)],
    )
    assert result.exit_code == 0, result.output
    mappings = _read_mappings(minimal_config)
    assert len(mappings) == 1
    assert mappings[0]["primary_email"] == "eve@corp.com"
    assert "eve@personal.io" in mappings[0]["aliases"]


def test_from_file_json_format(runner, minimal_config, tmp_path):
    """--from-file with JSON format."""
    aliases_file = tmp_path / "aliases.json"
    aliases_file.write_text(
        json.dumps(
            [
                {
                    "canonical": "frank@corp.com",
                    "aliases": ["frank@old.org"],
                    "name": "Frank Castle",
                }
            ]
        )
    )
    result = runner.invoke(
        add_alias_command,
        ["--config", str(minimal_config), "--from-file", str(aliases_file)],
    )
    assert result.exit_code == 0, result.output
    mappings = _read_mappings(minimal_config)
    assert len(mappings) == 1
    assert mappings[0]["primary_email"] == "frank@corp.com"
    assert "frank@old.org" in mappings[0]["aliases"]


def test_merges_new_aliases_into_existing_canonical(runner, config_with_mapping):
    """New aliases are merged into existing canonical mapping."""
    result = runner.invoke(
        add_alias_command,
        [
            "--config",
            str(config_with_mapping),
            "--canonical",
            "john@work.com",
            "--alias",
            "john@new-alias.com",
        ],
    )
    assert result.exit_code == 0, result.output
    mappings = _read_mappings(config_with_mapping)
    assert len(mappings) == 1  # no duplicate entry
    aliases = mappings[0]["aliases"]
    assert "john@gmail.com" in aliases  # original preserved
    assert "john@new-alias.com" in aliases  # new one added


def test_error_no_canonical_and_no_from_file(runner, minimal_config):
    """Error when neither --canonical nor --from-file is provided."""
    result = runner.invoke(
        add_alias_command,
        ["--config", str(minimal_config)],
    )
    assert result.exit_code != 0
    assert "required" in result.output.lower() or "either" in result.output.lower()


def test_error_canonical_without_alias(runner, minimal_config):
    """Error when --canonical is given without any --alias."""
    result = runner.invoke(
        add_alias_command,
        ["--config", str(minimal_config), "--canonical", "solo@work.com"],
    )
    assert result.exit_code != 0
    assert "alias" in result.output.lower()


def test_error_canonical_and_from_file_together(runner, minimal_config, tmp_path):
    """Error when --canonical and --from-file are both provided."""
    aliases_file = tmp_path / "a.yaml"
    aliases_file.write_text("developer_aliases: []\n")
    result = runner.invoke(
        add_alias_command,
        [
            "--config",
            str(minimal_config),
            "--canonical",
            "x@x.com",
            "--alias",
            "y@y.com",
            "--from-file",
            str(aliases_file),
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


def test_skipped_message_when_all_aliases_exist(runner, config_with_mapping):
    """Skipped message is shown when all requested aliases already exist."""
    result = runner.invoke(
        add_alias_command,
        [
            "--config",
            str(config_with_mapping),
            "--canonical",
            "john@work.com",
            "--alias",
            "john@gmail.com",  # already in mapping
        ],
    )
    assert result.exit_code == 0, result.output
    assert "skipped" in result.output.lower() or "already exists" in result.output.lower()


def test_creates_manual_mappings_when_key_absent(runner, tmp_path):
    """Works when analysis.identity.manual_mappings doesn't exist yet."""
    cfg = tmp_path / "bare.yaml"
    cfg.write_text("{}\n")  # completely empty config
    result = runner.invoke(
        add_alias_command,
        [
            "--config",
            str(cfg),
            "--canonical",
            "new@company.com",
            "--alias",
            "new@personal.com",
        ],
    )
    assert result.exit_code == 0, result.output
    mappings = _read_mappings(cfg)
    assert len(mappings) == 1
    assert mappings[0]["primary_email"] == "new@company.com"


def test_from_file_multiple_entries(runner, minimal_config, tmp_path):
    """--from-file processes multiple entries from a single file."""
    aliases_file = tmp_path / "multi.yaml"
    aliases_file.write_text(
        textwrap.dedent("""\
            developer_aliases:
              - primary_email: user1@corp.com
                aliases:
                  - user1@gmail.com
                name: User One
              - primary_email: user2@corp.com
                aliases:
                  - user2@hotmail.com
                name: User Two
        """)
    )
    result = runner.invoke(
        add_alias_command,
        ["--config", str(minimal_config), "--from-file", str(aliases_file)],
    )
    assert result.exit_code == 0, result.output
    mappings = _read_mappings(minimal_config)
    assert len(mappings) == 2
    emails = {m["primary_email"] for m in mappings}
    assert "user1@corp.com" in emails
    assert "user2@corp.com" in emails


def test_load_alias_file_unrecognised_format(tmp_path):
    """_load_alias_file raises ClickException for unrecognised YAML structure."""
    import click

    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("key: value\n")  # dict without developer_aliases
    with pytest.raises(click.ClickException):
        _load_alias_file(str(bad_file))


# ---------------------------------------------------------------------------
# Legacy developer_aliases handling (issue #27)
# ---------------------------------------------------------------------------


def test_legacy_developer_aliases_migrated_on_add(runner, tmp_path):
    """Top-level `developer_aliases` block is migrated to modern format on save."""
    cfg = tmp_path / "legacy.yaml"
    cfg.write_text(
        textwrap.dedent("""\
            developer_aliases:
              "Jane Dev":
                - jane@work.com
                - jane@home.com
        """)
    )

    result = runner.invoke(
        add_alias_command,
        [
            "--config",
            str(cfg),
            "--canonical",
            "new@corp.com",
            "--alias",
            "new@personal.com",
            "--no-apply",
        ],
    )
    assert result.exit_code == 0, result.output

    data = yaml.safe_load(cfg.read_text()) or {}
    assert "developer_aliases" not in data, "Legacy key should be removed after migration"
    mappings = data.get("analysis", {}).get("identity", {}).get("manual_mappings", [])
    # Migrated entry + new entry = 2 mappings
    assert len(mappings) == 2
    emails = {m["primary_email"] for m in mappings}
    assert "jane@work.com" in emails
    assert "new@corp.com" in emails
    # Migration message should be surfaced to the user
    assert "legacy" in result.output.lower() or "migrated" in result.output.lower()


def test_legacy_migrated_even_when_all_aliases_skipped(runner, tmp_path):
    """Legacy format is still cleaned up even if the user's new aliases are all duplicates."""
    cfg = tmp_path / "legacy_dup.yaml"
    cfg.write_text(
        textwrap.dedent("""\
            developer_aliases:
              "Jane Dev":
                - jane@work.com
                - jane@gmail.com
        """)
    )

    # First pass: migrates legacy, adds jane@home.com under jane@work.com
    r1 = runner.invoke(
        add_alias_command,
        [
            "--config",
            str(cfg),
            "--canonical",
            "jane@work.com",
            "--alias",
            "jane@home.com",
            "--no-apply",
        ],
    )
    assert r1.exit_code == 0, r1.output

    data = yaml.safe_load(cfg.read_text()) or {}
    assert "developer_aliases" not in data

    # Second pass with duplicate — should be idempotent no-op but file remains migrated
    r2 = runner.invoke(
        add_alias_command,
        [
            "--config",
            str(cfg),
            "--canonical",
            "jane@work.com",
            "--alias",
            "jane@home.com",  # now already present
            "--no-apply",
        ],
    )
    assert r2.exit_code == 0, r2.output
    assert "skipped" in r2.output.lower() or "already exists" in r2.output.lower()

    data = yaml.safe_load(cfg.read_text()) or {}
    assert "developer_aliases" not in data


# ---------------------------------------------------------------------------
# --name option
# ---------------------------------------------------------------------------


def test_name_option_sets_display_name(runner, minimal_config):
    """--name stores display name on the manual_mappings entry."""
    result = runner.invoke(
        add_alias_command,
        [
            "--config",
            str(minimal_config),
            "--canonical",
            "karen@work.com",
            "--alias",
            "karen@gmail.com",
            "--name",
            "Karen Doe",
            "--no-apply",
        ],
    )
    assert result.exit_code == 0, result.output
    mappings = _read_mappings(minimal_config)
    assert len(mappings) == 1
    assert mappings[0].get("name") == "Karen Doe"


# ---------------------------------------------------------------------------
# --no-apply / --apply behaviour (cache DB)
# ---------------------------------------------------------------------------


def test_no_apply_skips_cache_db_lookup(runner, tmp_path):
    """--no-apply prevents the command from touching ConfigLoader / cache DB."""
    cfg = tmp_path / "config.yaml"
    # Intentionally invalid/empty — ConfigLoader.load() would fail on this.
    # With --no-apply the command must not call ConfigLoader at all, so it
    # should still succeed at the YAML-write step.
    cfg.write_text("{}\n")

    result = runner.invoke(
        add_alias_command,
        [
            "--config",
            str(cfg),
            "--canonical",
            "p@x.com",
            "--alias",
            "p@y.com",
            "--no-apply",
        ],
    )
    assert result.exit_code == 0, result.output
    mappings = _read_mappings(cfg)
    assert len(mappings) == 1


def test_apply_updates_cache_db(runner, tmp_path):
    """--apply merges alias identities into the canonical identity in the DB."""
    # Build a real identity DB with two separate identities
    from gitflow_analytics.core.identity import DeveloperIdentityResolver

    cache_dir = tmp_path / ".gfa-cache"
    cache_dir.mkdir()
    db_path = cache_dir / "identities.db"

    resolver = DeveloperIdentityResolver(str(db_path))
    canonical_id = resolver.resolve_developer("Pat Canonical", "pat@work.com")
    alias_id = resolver.resolve_developer("Pat Personal", "pat@gmail.com")
    assert canonical_id != alias_id, "Pre-condition: identities should be distinct"

    # Build a config pointing at that cache dir
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.dump(
            {
                "version": "1.0",
                "repositories": [],
                "cache": {"directory": str(cache_dir)},
            },
            sort_keys=False,
        )
    )

    result = runner.invoke(
        add_alias_command,
        [
            "--config",
            str(cfg),
            "--canonical",
            "pat@work.com",
            "--alias",
            "pat@gmail.com",
            # --apply is the default, but be explicit for clarity
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "cache" in result.output.lower()

    # Re-open the resolver and confirm the alias identity has been merged.
    resolver2 = DeveloperIdentityResolver(str(db_path))
    from sqlalchemy import text

    with resolver2.get_session() as session:
        remaining = session.execute(
            text("SELECT COUNT(*) FROM developer_identities WHERE canonical_id = :cid"),
            {"cid": alias_id},
        ).scalar()
        assert remaining == 0, "Alias identity row should have been merged away"

        canonical_row = session.execute(
            text(
                "SELECT canonical_id FROM developer_identities "
                "WHERE LOWER(primary_email) = 'pat@work.com'"
            ),
        ).fetchone()
        assert canonical_row is not None
        assert canonical_row[0] == canonical_id


def test_apply_idempotent_second_run_noop(runner, tmp_path):
    """Running --apply twice is a no-op the second time (no errors)."""
    from gitflow_analytics.core.identity import DeveloperIdentityResolver

    cache_dir = tmp_path / ".gfa-cache"
    cache_dir.mkdir()
    db_path = cache_dir / "identities.db"

    resolver = DeveloperIdentityResolver(str(db_path))
    resolver.resolve_developer("Sam Canonical", "sam@work.com")
    resolver.resolve_developer("Sam Alt", "sam@gmail.com")

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.dump(
            {
                "version": "1.0",
                "repositories": [],
                "cache": {"directory": str(cache_dir)},
            },
            sort_keys=False,
        )
    )

    args = [
        "--config",
        str(cfg),
        "--canonical",
        "sam@work.com",
        "--alias",
        "sam@gmail.com",
    ]
    first = runner.invoke(add_alias_command, args)
    assert first.exit_code == 0, first.output

    second = runner.invoke(add_alias_command, args)
    assert second.exit_code == 0, second.output
    # Config-level mappings should remain a single entry with one alias
    mappings = _read_mappings(cfg)
    assert len(mappings) == 1
    assert mappings[0]["aliases"].count("sam@gmail.com") == 1


def test_apply_canonical_missing_in_cache_is_not_error(runner, tmp_path):
    """When the canonical email isn't in the cache DB yet, we report it and exit 0."""
    from gitflow_analytics.core.identity import DeveloperIdentityResolver

    cache_dir = tmp_path / ".gfa-cache"
    cache_dir.mkdir()
    db_path = cache_dir / "identities.db"

    # Empty DB — no identities pre-created
    DeveloperIdentityResolver(str(db_path))

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.dump(
            {
                "version": "1.0",
                "repositories": [],
                "cache": {"directory": str(cache_dir)},
            },
            sort_keys=False,
        )
    )

    result = runner.invoke(
        add_alias_command,
        [
            "--config",
            str(cfg),
            "--canonical",
            "ghost@nowhere.com",
            "--alias",
            "ghost@gmail.com",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "not in" in result.output.lower() or "skip" in result.output.lower()


def test_dry_run_with_apply_does_not_mutate_db(runner, tmp_path):
    """--dry-run + --apply previews cache changes without mutating DB."""
    from gitflow_analytics.core.identity import DeveloperIdentityResolver

    cache_dir = tmp_path / ".gfa-cache"
    cache_dir.mkdir()
    db_path = cache_dir / "identities.db"

    resolver = DeveloperIdentityResolver(str(db_path))
    canonical_id = resolver.resolve_developer("Dee Canonical", "dee@work.com")
    alias_id = resolver.resolve_developer("Dee Alt", "dee@gmail.com")
    assert canonical_id != alias_id

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.dump(
            {
                "version": "1.0",
                "repositories": [],
                "cache": {"directory": str(cache_dir)},
            },
            sort_keys=False,
        )
    )

    result = runner.invoke(
        add_alias_command,
        [
            "--config",
            str(cfg),
            "--canonical",
            "dee@work.com",
            "--alias",
            "dee@gmail.com",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output

    # Config unchanged
    data = yaml.safe_load(cfg.read_text()) or {}
    assert (
        not data.get("analysis", {}).get("identity", {}).get("manual_mappings")
    ), "Dry-run must not write mappings"

    # DB unchanged — alias identity still exists
    resolver2 = DeveloperIdentityResolver(str(db_path))
    from sqlalchemy import text

    with resolver2.get_session() as session:
        remaining = session.execute(
            text("SELECT COUNT(*) FROM developer_identities WHERE canonical_id = :cid"),
            {"cid": alias_id},
        ).scalar()
        assert remaining == 1, "Dry-run must not merge identity rows"
