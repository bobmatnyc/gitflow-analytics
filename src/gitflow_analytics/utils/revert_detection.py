"""Revert / rollback commit detection (issue #64).

WHY: Revert detection is needed at multiple layers — commit ingestion (to
populate ``cached_commits.is_revert``), metrics aggregation (to count
``reversion_commits`` in daily_metrics / weekly_trends), and the native
quality report.  Centralising the regex set here ensures all callers use the
same definition of "revert" and that adding a new pattern updates every
consumer simultaneously.

Patterns matched (case-insensitive, evaluated against the full commit
message including the body):

    * ``Revert "..."``                 — standard ``git revert``-generated subject.
    * ``revert: ...``                  — Conventional Commits style.
    * ``^revert ...`` / ``^reverts ...`` — free-form English subjects.
    * ``This reverts commit <sha>`` anywhere in the body — the canonical
      footer that ``git revert`` always emits, even when callers rewrote
      the subject.
"""

from __future__ import annotations

import re
from typing import Final

# Compiled patterns evaluated against the commit message.  Each pattern is
# applied to the lower-cased message; ``^`` anchors are evaluated with
# re.MULTILINE so the body's "this reverts commit ..." footer also matches
# even when prefixed with whitespace inside the message body.
_REVERT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r'^\s*revert\s+"', re.MULTILINE),
    re.compile(r"^\s*revert:\s", re.MULTILINE),
    re.compile(r"^\s*reverts?\s+", re.MULTILINE),
    re.compile(r"this reverts commit", re.IGNORECASE),
)


def is_revert_commit(message: object) -> bool:
    """Return True when *message* matches any known revert pattern.

    WHY: Used at commit ingestion to set ``cached_commits.is_revert`` and at
    backfill time to re-evaluate legacy rows.  The function is intentionally
    permissive about input types (str / bytes / None) so callers do not have
    to defensively coerce.

    Args:
        message: Commit message as ``str`` (preferred), ``bytes`` (decoded as
            UTF-8 with errors ignored), or any other type (treated as no match).

    Returns:
        ``True`` if any revert pattern matches the lowercased message.
    """
    if message is None:
        return False
    if isinstance(message, bytes):
        text = message.decode("utf-8", errors="ignore")
    elif isinstance(message, str):
        text = message
    else:
        return False

    if not text:
        return False

    lowered = text.lower()
    return any(p.search(lowered) for p in _REVERT_PATTERNS)


__all__ = ["is_revert_commit"]
