"""Native quality report — revert detection, risk profile, code review signals.

Produces ``quality_summary.json`` in the output directory containing:

- Org-level aggregate metrics
- Per-developer breakdown
- Composite quality score (0–1)

The report is intentionally self-contained: all inputs are plain ``dict``
lists so it can be driven from the pipeline without any ORM dependency.
"""

import contextlib
import json
import logging
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Regex patterns that identify revert / rollback commits.
# Compiled once at import time for efficiency.
_REVERT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^revert\b"),
    re.compile(r"^reverts?\s+"),
    re.compile(r"\brollback\b"),
    re.compile(r"\bundo\s+(commit|merge|pr)\b"),
]


def _is_revert(message: str) -> bool:
    """Return True when the commit message looks like a revert/rollback.

    Matching is case-insensitive and anchored to word boundaries so that
    words such as "reverse" or "irreversible" are not false-positives.

    Args:
        message: Raw commit message string (may be empty or None).

    Returns:
        True if any revert pattern matches the lowercased message.
    """
    msg = (message or "").lower().strip()
    return any(pattern.search(msg) for pattern in _REVERT_PATTERNS)


class QualityReportGenerator:
    """Compute and persist quality metrics for commits and pull requests.

    All public methods accept plain ``dict`` lists so no ORM session or
    SQLAlchemy models are required at call time — the pipeline is responsible
    for fetching data and passing it in.

    Args:
        config: ``QualityReportConfig`` instance (or ``None`` for defaults).
            Only ``enabled`` is currently read; the sub-flags
            (``revert_detection_patterns``, ``risk_profile``,
            ``code_review_signals``, ``quality_score``) are reserved for
            future selective computation.
    """

    def __init__(self, config: Any = None) -> None:
        self.cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        commits: list[dict[str, Any]],
        qualitative_data: list[dict[str, Any]],
        prs: list[dict[str, Any]],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Compute quality metrics and write ``quality_summary.json``.

        Args:
            commits: List of commit dicts (keys: ``commit_hash``, ``message``,
                ``author_email``, ``author_name``).
            qualitative_data: List of qualitative classification dicts (keys:
                ``commit_hash``, ``risk_level``, ``complexity``).  Rows are
                joined to *commits* via ``commit_hash``.  Pass ``[]`` when no
                qualitative data is available.
            prs: List of PR dicts (keys: ``author``, ``revision_count``,
                ``change_requests_count``, ``approvals_count``,
                ``is_merged``).
            output_dir: Directory where ``quality_summary.json`` will be written.

        Returns:
            The summary ``dict`` that was serialised to disk.
        """
        # Build commit lookup by hash for qualitative join.
        qual_by_hash: dict[str, dict[str, Any]] = {
            q["commit_hash"]: q for q in qualitative_data if q.get("commit_hash")
        }

        per_developer = self._by_developer(commits, qual_by_hash, prs)
        org_level = self._org_level(commits, qual_by_hash, prs)

        summary: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_commits_analyzed": len(commits),
            "total_prs_analyzed": len(prs),
            "org_level": org_level,
            "per_developer": per_developer,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "quality_summary.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("Wrote quality_summary.json (%d commits, %d PRs)", len(commits), len(prs))
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _commit_quality_metrics(
        self,
        commits: list[dict[str, Any]],
        qual_by_hash: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute commit-level quality metrics for a set of commits.

        Args:
            commits: Subset of commit dicts to analyse.
            qual_by_hash: Qualitative data keyed by ``commit_hash``.

        Returns:
            Dict with keys: ``total_commits``, ``revert_commits``,
            ``revert_rate``, ``high_risk_commits``, ``medium_risk_commits``,
            ``low_risk_commits``, ``high_risk_ratio``, ``avg_complexity``.
        """
        total = len(commits)
        if total == 0:
            return {
                "total_commits": 0,
                "revert_commits": 0,
                "revert_rate": 0.0,
                "high_risk_commits": 0,
                "medium_risk_commits": 0,
                "low_risk_commits": 0,
                "high_risk_ratio": 0.0,
                "avg_complexity": None,
            }

        revert_count = sum(1 for c in commits if _is_revert(c.get("message", "")))

        risk_counts: dict[str, int] = defaultdict(int)
        complexities: list[int] = []

        for c in commits:
            q = qual_by_hash.get(c.get("commit_hash", ""), {})

            risk = (q.get("risk_level") or "").lower()
            if risk in ("high", "medium", "low"):
                risk_counts[risk] += 1

            cplx = q.get("complexity")
            if cplx is not None:
                with contextlib.suppress(ValueError, TypeError):
                    complexities.append(int(cplx))

        return {
            "total_commits": total,
            "revert_commits": revert_count,
            "revert_rate": round(revert_count / total, 4),
            "high_risk_commits": risk_counts["high"],
            "medium_risk_commits": risk_counts["medium"],
            "low_risk_commits": risk_counts["low"],
            "high_risk_ratio": round(risk_counts["high"] / total, 4),
            "avg_complexity": (round(statistics.mean(complexities), 2) if complexities else None),
        }

    def _pr_quality_metrics(self, prs: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute PR-level quality metrics.

        Args:
            prs: Subset of PR dicts to analyse.

        Returns:
            Dict with keys: ``prs_total``, ``avg_revision_count``,
            ``change_request_rate``, ``approval_rate``.
        """
        if not prs:
            return {
                "prs_total": 0,
                "avg_revision_count": 0.0,
                "change_request_rate": 0.0,
                "approval_rate": 0.0,
            }

        revisions = [pr.get("revision_count") or 0 for pr in prs]
        # PRs that received at least one change-request review.
        change_req_count = sum(1 for pr in prs if (pr.get("change_requests_count") or 0) >= 1)
        # PRs approved on first pass (≥1 approval, zero change requests).
        approved_first_pass = sum(
            1
            for pr in prs
            if (pr.get("approvals_count") or 0) >= 1 and (pr.get("change_requests_count") or 0) == 0
        )

        return {
            "prs_total": len(prs),
            "avg_revision_count": round(statistics.mean(revisions), 2),
            "change_request_rate": round(change_req_count / len(prs), 4),
            "approval_rate": round(approved_first_pass / len(prs), 4),
        }

    def _quality_score(
        self,
        commit_m: dict[str, Any],
        pr_m: dict[str, Any],
    ) -> float | None:
        """Compute a composite quality score in the range [0, 1].

        Formula::

            score = (1 - revert_rate)
                  * (1 - high_risk_ratio)
                  * (1 - min(avg_revision_count / 5, 1))

        A score of 1.0 means no reverts, no high-risk commits, and an average
        of zero PR revisions.  Returns ``None`` when there are no commits to
        base the calculation on.

        Args:
            commit_m: Output of :meth:`_commit_quality_metrics`.
            pr_m: Output of :meth:`_pr_quality_metrics`.

        Returns:
            Float in [0, 1] or ``None``.
        """
        if commit_m.get("total_commits", 0) == 0:
            return None

        revert_rate: float = commit_m.get("revert_rate", 0.0)
        high_risk_ratio: float = commit_m.get("high_risk_ratio", 0.0)
        avg_rev: float = pr_m.get("avg_revision_count", 0.0)

        score = (1.0 - revert_rate) * (1.0 - high_risk_ratio) * (1.0 - min(avg_rev / 5.0, 1.0))
        return round(score, 4)

    def _by_developer(
        self,
        commits: list[dict[str, Any]],
        qual_by_hash: dict[str, dict[str, Any]],
        prs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build per-developer quality metrics.

        Developer identity is derived from ``author_email`` with a fallback
        to ``author_name`` and finally ``"unknown"``.  PR author identity uses
        the ``author`` field of each PR dict.

        Args:
            commits: All commit dicts.
            qual_by_hash: Qualitative data keyed by ``commit_hash``.
            prs: All PR dicts.

        Returns:
            Dict mapping developer identifier → merged metric dict.
        """
        commit_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for c in commits:
            dev = c.get("author_email") or c.get("author_name") or "unknown"
            commit_buckets[dev].append(c)

        pr_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pr in prs:
            author = pr.get("author") or "unknown"
            pr_buckets[author].append(pr)

        all_devs = set(commit_buckets) | set(pr_buckets)
        result: dict[str, Any] = {}
        for dev in sorted(all_devs):
            cm = self._commit_quality_metrics(commit_buckets.get(dev, []), qual_by_hash)
            pm = self._pr_quality_metrics(pr_buckets.get(dev, []))
            result[dev] = {**cm, **pm, "quality_score": self._quality_score(cm, pm)}
        return result

    def _org_level(
        self,
        commits: list[dict[str, Any]],
        qual_by_hash: dict[str, dict[str, Any]],
        prs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build organisation-wide quality metrics.

        Args:
            commits: All commit dicts.
            qual_by_hash: Qualitative data keyed by ``commit_hash``.
            prs: All PR dicts.

        Returns:
            Merged metric dict with a ``quality_score`` key.
        """
        cm = self._commit_quality_metrics(commits, qual_by_hash)
        pm = self._pr_quality_metrics(prs)
        return {**cm, **pm, "quality_score": self._quality_score(cm, pm)}
