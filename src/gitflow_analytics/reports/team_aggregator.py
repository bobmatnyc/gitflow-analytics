"""Team/pod aggregation for GFA report output.

Reads developer metrics and groups them by the team/pod membership
declared in the ``teams`` section of the YAML config.  Writes a
``weekly_summary.json`` file to the output directory.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Metric keys that should be summed as integers across team members.
_INT_KEYS = [
    "total_commits",
    "feature_commits",
    "bug_fix_commits",
    "refactor_commits",
    "documentation_commits",
    "maintenance_commits",
    "test_commits",
    "lines_added",
    "lines_deleted",
    "story_points",
    "tracked_commits",
    "untracked_commits",
    "ai_assisted_commits",
    "ai_generated_commits",
]

# Metric keys that should be averaged (mean) across team members.
_FLOAT_KEYS = [
    "churn_rate_14d",
]


class TeamAggregator:
    """Aggregate developer metrics by team/pod from config.

    Builds fast lookup tables from the ``TeamsConfig`` dataclass so that
    each developer dict can be resolved to a team (and optionally a pod)
    in O(1) time.

    Example usage::

        aggregator = TeamAggregator(cfg.teams)
        summary = aggregator.generate(daily_metrics_dicts, output_dir)
    """

    def __init__(self, teams_config: Any) -> None:
        """Initialise with a TeamsConfig instance (or None for no-op).

        Args:
            teams_config: A ``TeamsConfig`` dataclass instance, or ``None``.
        """
        self.teams_config = teams_config
        # Maps lowercased identifier → team name
        self._dev_to_team: dict[str, str] = {}
        # Maps lowercased identifier → pod name
        self._dev_to_pod: dict[str, str] = {}

        if teams_config and getattr(teams_config, "teams", None):
            for team in teams_config.teams:
                for member in team.members:
                    for key in (member.email, member.github, member.name):
                        if key:
                            self._dev_to_team[key.lower()] = team.name

                for pod in team.pods:
                    for member in pod.members:
                        for key in (member.email, member.github, member.name):
                            if key:
                                self._dev_to_pod[key.lower()] = pod.name

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def resolve_team(self, developer: dict[str, Any]) -> str | None:
        """Return the team name for a developer dict, or ``None`` if unmapped.

        Checks the following developer dict keys in order:
        ``developer_email``, ``developer_id``, ``developer_name``,
        ``author_email``, ``author``.

        Args:
            developer: A dict with developer metadata (e.g. a daily_metrics row).

        Returns:
            Team name string, or ``None`` if the developer is not in any team.
        """
        for key in (
            developer.get("developer_email"),
            developer.get("developer_id"),
            developer.get("developer_name"),
            developer.get("author_email"),
            developer.get("author"),
        ):
            if key and key.lower() in self._dev_to_team:
                return self._dev_to_team[key.lower()]
        return None

    def resolve_pod(self, developer: dict[str, Any]) -> str | None:
        """Return the pod name for a developer dict, or ``None`` if unmapped.

        Args:
            developer: A dict with developer metadata.

        Returns:
            Pod name string, or ``None``.
        """
        for key in (
            developer.get("developer_email"),
            developer.get("developer_id"),
            developer.get("developer_name"),
        ):
            if key and key.lower() in self._dev_to_pod:
                return self._dev_to_pod[key.lower()]
        return None

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_metrics(self, daily_metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Group ``daily_metrics`` rows by team and sum numeric counters.

        Rows belonging to developers not mapped to any team are silently
        excluded from the output.

        Args:
            daily_metrics: List of developer metric dicts (one row per
                developer per day, or any granularity).

        Returns:
            Mapping of team name → aggregated metrics dict.
        """
        team_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in daily_metrics:
            team = self.resolve_team(row)
            if team:
                team_buckets[team].append(row)

        return {
            team_name: self._sum_metrics(rows) for team_name, rows in sorted(team_buckets.items())
        }

    def _sum_metrics(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Reduce a list of metric rows into a single aggregated dict.

        Integer counters are summed; float keys are averaged; a
        ``developer_count`` of unique ``developer_id`` values is added;
        ``ai_adoption_pct`` is derived from the summed counters.

        Args:
            rows: Non-empty list of metric dicts for one team.

        Returns:
            Single dict of aggregated metrics.
        """
        totals: dict[str, int | float] = defaultdict(int)

        for row in rows:
            for k in _INT_KEYS:
                totals[k] += int(row.get(k) or 0)

        for k in _FLOAT_KEYS:
            vals = [float(row[k]) for row in rows if row.get(k) is not None and row.get(k) != ""]
            totals[k] = round(sum(vals) / len(vals), 4) if vals else 0.0

        totals["developer_count"] = len(
            {r.get("developer_id") for r in rows if r.get("developer_id")}
        )

        total_commits = int(totals.get("total_commits", 0))
        ai_assisted = int(totals.get("ai_assisted_commits", 0))
        totals["ai_adoption_pct"] = (
            round(ai_assisted / total_commits * 100, 1) if total_commits else 0.0
        )

        return dict(totals)

    # ------------------------------------------------------------------
    # File generation
    # ------------------------------------------------------------------

    def generate(
        self,
        daily_metrics: list[dict[str, Any]],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Aggregate metrics and write ``weekly_summary.json`` to *output_dir*.

        When no teams are configured (or the ``enabled`` flag is ``False``),
        this method logs a notice and returns an empty dict without writing
        any file.

        Args:
            daily_metrics: List of developer metric dicts.
            output_dir: Directory where ``weekly_summary.json`` will be written.

        Returns:
            The summary dict that was written, or ``{}`` when skipped.
        """
        if not self.teams_config or not getattr(self.teams_config, "teams", None):
            logger.info("No teams configured — skipping team aggregation")
            return {}

        if not getattr(self.teams_config, "enabled", True):
            logger.info("Teams aggregation disabled — skipping")
            return {}

        team_data = self.aggregate_metrics(daily_metrics)
        summary: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "team_count": len(team_data),
            "teams": team_data,
        }

        out_path = output_dir / "weekly_summary.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("Wrote weekly_summary.json (%d teams)", len(team_data))
        return summary
