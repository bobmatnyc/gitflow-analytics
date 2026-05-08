"""Classification coverage measurement (issue #65).

When a repository has no ``jira_project_mappings`` configured and authors do
not follow conventional commit prefixes, every commit silently falls back to
the ``maintenance`` (KTLO) bucket.  Reports then look like the team is doing
100% maintenance work, with no warning that the underlying signal is missing.

This module computes the classification coverage percentage for a repository
in a date range:

    coverage_pct = (commits NOT classified as maintenance/ktlo/other/unknown)
                   / (total classified commits)
                   * 100

The pipeline calls :func:`compute_repo_coverage` after each repository is
classified so it can:

1. Emit a ``logging.warning()`` when coverage is below the configured
   threshold (default 20%).
2. Persist the value on ``RepositoryAnalysisStatus.classification_coverage_pct``
   so reports can render the metric.
3. Drive the optional ``--validate-coverage`` CLI flag, which exits with a
   non-zero code when any repository falls below the threshold (for CI use).

The helper is intentionally side-effect free: it just reads the cache.
Callers own logging, persistence, and exit-code decisions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import and_, func

from ..constants import CLASSIFICATION_FALLTHROUGH_CATEGORIES
from ..models.database import CachedCommit, QualitativeCommitData

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def compute_repo_coverage(
    session: Session,
    repo_path: str,
    start_date: datetime,
    end_date: datetime,
) -> float | None:
    """Compute classification coverage percentage for a single repository.

    Args:
        session: SQLAlchemy session attached to the analytics cache database.
        repo_path: Repository path used as the lookup key on
            :class:`CachedCommit.repo_path`.
        start_date: Inclusive lower bound on commit timestamps.
        end_date: Inclusive upper bound on commit timestamps.

    Returns:
        Coverage percentage in the range ``[0.0, 100.0]``, or ``None`` when
        the repository has no classified commits in the period (in which case
        coverage is undefined and the caller should skip the warning).
    """
    # Count classified commits.  We INNER JOIN qualitative_commits so commits
    # that have not yet been classified do not skew the denominator —
    # coverage describes the *quality* of classifications that exist, not
    # whether classification ran for every commit.
    base_query = (
        session.query(func.count(CachedCommit.id))
        .join(QualitativeCommitData, QualitativeCommitData.commit_id == CachedCommit.id)
        .filter(
            and_(
                CachedCommit.repo_path == repo_path,
                CachedCommit.timestamp >= start_date,
                CachedCommit.timestamp <= end_date,
            )
        )
    )

    total = base_query.scalar() or 0
    if total == 0:
        return None

    fallthrough = (
        base_query.filter(
            QualitativeCommitData.change_type.in_(tuple(CLASSIFICATION_FALLTHROUGH_CATEGORIES))
        ).scalar()
        or 0
    )

    meaningful = total - fallthrough
    return round((meaningful / total) * 100.0, 2)


def format_low_coverage_warning(
    repo_name: str,
    coverage_pct: float,
) -> str:
    """Build the human-readable low-coverage warning message.

    Centralised so the same wording appears in CLI output, log records, and
    the optional ``--validate-coverage`` summary block.
    """
    fallthrough_pct = round(100.0 - coverage_pct, 1)
    return (
        f"repo '{repo_name}' has only {coverage_pct:.1f}% classification "
        f"coverage ({fallthrough_pct:.1f}% of commits fell to "
        "maintenance/KTLO). Consider adding jira_project_mappings or "
        "conventional commit prefixes."
    )
