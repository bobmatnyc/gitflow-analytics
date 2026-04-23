"""Developer activity scoring module using balanced metrics.

Based on research and best practices for measuring developer productivity in 2024,
this module implements a balanced scoring approach that considers:
- Commits (baseline activity)
- Pull Requests (collaboration and review)
- Lines of Code (impact, with diminishing returns)
- Code churn (deletions valued for refactoring)
"""

import math
from datetime import datetime
from typing import Any, Optional


class ActivityScorer:
    """Calculate balanced developer activity scores based on multiple metrics.

    WHY: Ticketing platforms (GitHub Issues, Confluence, JIRA) generate
    per-developer activity events that are now folded into the composite
    activity score.  When ``ticketing_weight > 0`` the scorer queries
    ``ticketing_activity_cache`` for per-developer event counts and blends
    them into ``raw_activity_score``.  ``ticketing_weight == 0`` preserves
    the historical code-only behaviour for backward compatibility.
    """

    # Default weights — adjusted from the historical (0.25/0.30/0.30/0.15)
    # split so that ticketing contributes 15% without re-normalising at
    # runtime.  See :class:`ActivityScoringConfig` for configuration.
    DEFAULT_WEIGHTS = {
        "commits": 0.22,
        "prs": 0.26,
        "code_impact": 0.26,
        "complexity": 0.11,
        "ticketing": 0.15,
    }

    # Scaling factors based on research
    COMMIT_BASE_SCORE = 10  # Each commit worth base 10 points
    PR_BASE_SCORE = 50  # Each PR worth base 50 points (5x commit)
    OPTIMAL_PR_SIZE = 200  # Research shows PRs under 200 lines are optimal

    # Per-event ticketing weights (same scale as TicketingActivityReport)
    TICKETING_EVENT_WEIGHTS = {
        "issues_opened": 1.0,
        "issues_closed": 1.0,
        "comments_posted": 0.5,
        "pages_created": 2.0,
        "pages_edited": 1.0,
        "jira_issues_opened": 1.5,
        "jira_issues_closed": 2.0,
        "jira_comments_posted": 0.5,
    }

    def __init__(
        self,
        config: Optional[Any] = None,
        cache: Optional[Any] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        identity_resolver: Optional[Any] = None,
    ) -> None:
        """Initialize scorer.

        Args:
            config: Optional :class:`ActivityScoringConfig`-shaped object
                providing ``commits_weight``, ``prs_weight``,
                ``code_impact_weight``, ``complexity_weight``,
                ``ticketing_weight`` attributes.  When None, default
                weights are used.
            cache: Optional :class:`GitAnalysisCache` for reading
                ``ticketing_activity_cache`` rows when
                ``ticketing_weight > 0``.
            since: Lower bound (inclusive) on ``activity_at`` for
                ticketing lookups.  Required when ticketing blending is
                active.
            until: Upper bound (inclusive) on ``activity_at`` for
                ticketing lookups.
            identity_resolver: Optional identity resolver used to bridge
                raw ticketing actor keys (usually GitHub logins) to
                canonical developer identity IDs.  When provided, the
                per-actor ticketing totals are additionally re-keyed by
                canonical_id so downstream lookups by ``developer_id``
                (e.g. from the developer-activity CSV) succeed.  Default
                ``None`` preserves legacy behaviour (actor-only keying).
        """
        if config is not None:
            self.WEIGHTS = {
                "commits": float(getattr(config, "commits_weight", 0.22)),
                "prs": float(getattr(config, "prs_weight", 0.26)),
                "code_impact": float(getattr(config, "code_impact_weight", 0.26)),
                "complexity": float(getattr(config, "complexity_weight", 0.11)),
                "ticketing": float(getattr(config, "ticketing_weight", 0.15)),
            }
        else:
            self.WEIGHTS = dict(self.DEFAULT_WEIGHTS)

        self._cache = cache
        self._since = since
        self._until = until
        self._identity_resolver = identity_resolver
        self._ticketing_cache: Optional[dict[str, float]] = None

    # ------------------------------------------------------------------
    # Ticketing integration
    # ------------------------------------------------------------------
    def _load_ticketing_scores(self) -> dict[str, float]:
        """Return per-developer weighted ticketing-event totals.

        Keys are lowercased actor identifiers.  Results are cached for
        the lifetime of this scorer instance to avoid repeated DB reads
        when scoring many developers.
        """
        if self._ticketing_cache is not None:
            return self._ticketing_cache
        if self._cache is None or self._since is None or self._until is None:
            self._ticketing_cache = {}
            return self._ticketing_cache

        try:
            from ..models.database import TicketingActivityCache
        except Exception:  # noqa: BLE001
            self._ticketing_cache = {}
            return self._ticketing_cache

        def _naive(dt: datetime) -> datetime:
            if dt.tzinfo is not None:
                from datetime import timezone

                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt

        totals: dict[str, float] = {}
        try:
            with self._cache.get_session() as session:
                rows = (
                    session.query(TicketingActivityCache)
                    .filter(
                        TicketingActivityCache.activity_at >= _naive(self._since),
                        TicketingActivityCache.activity_at <= _naive(self._until),
                    )
                    .all()
                )
                for row in rows:
                    actor = (getattr(row, "actor", None) or "").lower() or "unknown"
                    platform = getattr(row, "platform", None) or ""
                    item_type = getattr(row, "item_type", None) or ""
                    action = getattr(row, "action", None) or ""
                    key = self._ticketing_event_key(platform, item_type, action)
                    if key is None:
                        continue
                    totals[actor] = totals.get(actor, 0.0) + self.TICKETING_EVENT_WEIGHTS.get(
                        key, 0.0
                    )
        except Exception:  # noqa: BLE001
            totals = {}

        # WHY: ``TicketingActivityCache.actor`` stores raw ticketing-platform
        # identifiers — typically lowercase GitHub logins (e.g. ``bob-duetto``)
        # for GitHub Issues events, or email addresses for some JIRA flows.
        # Callers of :meth:`get_ticketing_score` (notably the developer-
        # activity CSV) look up scores by canonical_id UUID.  Without
        # translation the two key spaces are disjoint and every lookup
        # returns 0.0.  When an identity resolver is available we
        # additionally index each actor under its resolved canonical_id so
        # both forms of lookup succeed.  See GitHub issue #41.
        if self._identity_resolver is not None and totals:
            resolver = self._identity_resolver
            resolve_fn = getattr(resolver, "resolve_by_github_username", None)
            if callable(resolve_fn):
                for actor, score in list(totals.items()):
                    try:
                        canonical_id = resolve_fn(actor)
                    except Exception:  # noqa: BLE001
                        canonical_id = None
                    if canonical_id:
                        # Accumulate in case multiple actors map to the same
                        # canonical identity (aliases, renamed accounts).
                        cid = str(canonical_id)
                        totals[cid] = totals.get(cid, 0.0) + score

        self._ticketing_cache = totals
        return totals

    @staticmethod
    def _ticketing_event_key(platform: str, item_type: str, action: str) -> Optional[str]:
        """Map a ticketing-cache row to a canonical per-developer metric key."""
        if platform == "github_issues":
            if item_type == "issue" and action == "opened":
                return "issues_opened"
            if item_type == "issue" and action == "closed":
                return "issues_closed"
            if item_type == "comment":
                return "comments_posted"
        elif platform == "confluence":
            if item_type == "page_create":
                return "pages_created"
            if item_type == "page_edit":
                return "pages_edited"
        elif platform == "jira":
            if item_type == "issue_created":
                return "jira_issues_opened"
            if item_type == "issue_closed":
                return "jira_issues_closed"
            if item_type == "comment":
                return "jira_comments_posted"
        return None

    def get_ticketing_score(self, developer_id: Optional[str]) -> float:
        """Return the weighted ticketing-event total for ``developer_id``.

        Returns 0.0 when no ticketing data is available (missing cache,
        identity not present, or ``ticketing_weight == 0``).

        ``developer_id`` may be a canonical_id UUID (preferred — produced
        by the identity-resolution layer), a raw GitHub login, or an
        email address.  We first try a case-sensitive direct lookup so
        UUIDs and other case-sensitive keys match verbatim, then fall
        back to a lowercase lookup to handle display-name / login casing.
        """
        if self.WEIGHTS.get("ticketing", 0.0) <= 0:
            return 0.0
        if not developer_id:
            return 0.0
        scores = self._load_ticketing_scores()
        # Direct lookup (matches canonical_id UUIDs keyed in
        # ``_load_ticketing_scores`` when an identity resolver is wired).
        if developer_id in scores:
            return float(scores[developer_id])
        # Fallback: lowercase lookup for actor-keyed entries (GitHub logins,
        # emails) which are always stored lowercased.
        return float(scores.get(developer_id.lower(), 0.0))

    def calculate_activity_score(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Calculate balanced activity score for a developer.

        Args:
            metrics: Dictionary containing:
                - commits: Number of commits
                - prs_involved: Number of PRs
                - lines_added: Lines added
                - lines_removed: Lines removed
                - files_changed: Number of files changed
                - complexity_delta: Complexity change

        Returns:
            Dictionary with:
                - raw_score: Unscaled activity score
                - normalized_score: Score normalized to 0-100
                - components: Breakdown of score components
        """
        # Extract metrics with defaults
        commits = metrics.get("commits", 0)
        prs = metrics.get("prs_involved", 0)
        lines_added = metrics.get("lines_added", 0)
        lines_removed = metrics.get("lines_removed", 0)
        files_changed = metrics.get(
            "files_changed_count",
            (
                metrics.get("files_changed", 0)
                if isinstance(metrics.get("files_changed"), int)
                else len(metrics.get("files_changed", []))
            ),
        )
        complexity = metrics.get("complexity_delta", 0)

        # Calculate component scores
        commit_score = self._calculate_commit_score(commits)
        pr_score = self._calculate_pr_score(prs, lines_added + lines_removed)
        code_score = self._calculate_code_impact_score(lines_added, lines_removed)
        complexity_score = self._calculate_complexity_score(files_changed, complexity)

        # Ticketing score — computed only when configured weight > 0.
        ticketing_weight = float(self.WEIGHTS.get("ticketing", 0.0) or 0.0)
        ticketing_score = 0.0
        if ticketing_weight > 0:
            dev_id = (
                metrics.get("developer_id")
                or metrics.get("canonical_id")
                or metrics.get("primary_email")
            )
            # Allow callers to supply a pre-computed score via metrics
            # (useful for tests and non-DB-backed call sites).
            explicit = metrics.get("ticketing_score")
            if explicit is not None:
                try:
                    ticketing_score = float(explicit)
                except (TypeError, ValueError):
                    ticketing_score = 0.0
            elif dev_id:
                ticketing_score = self.get_ticketing_score(dev_id)

        # Weighted total
        components = {
            "commit_score": commit_score,
            "pr_score": pr_score,
            "code_impact_score": code_score,
            "complexity_score": complexity_score,
            "ticketing_score": ticketing_score,
        }

        code_raw_score = (
            commit_score * self.WEIGHTS["commits"]
            + pr_score * self.WEIGHTS["prs"]
            + code_score * self.WEIGHTS["code_impact"]
            + complexity_score * self.WEIGHTS["complexity"]
        )

        if ticketing_weight > 0:
            raw_score = code_raw_score + ticketing_score * ticketing_weight
        else:
            raw_score = code_raw_score

        # Determine if PR data is available for proper normalization
        has_pr_data = prs > 0

        return {
            "raw_score": raw_score,
            "normalized_score": self._normalize_score(raw_score, has_pr_data),
            "components": components,
            "activity_level": self._get_activity_level(raw_score, has_pr_data),
            "has_pr_data": has_pr_data,
        }

    def _calculate_commit_score(self, commits: int) -> float:
        """Calculate score from commit count with diminishing returns."""
        if commits == 0:
            return 0

        # Use logarithmic scaling for diminishing returns
        # First 10 commits worth full value, then diminishing
        if commits <= 10:
            return commits * self.COMMIT_BASE_SCORE
        else:
            base = 10 * self.COMMIT_BASE_SCORE
            extra = math.log10(commits - 9) * self.COMMIT_BASE_SCORE * 5
            return base + extra

    def _calculate_pr_score(self, prs: int, total_lines: int) -> float:
        """Calculate PR score considering optimal PR sizes."""
        if prs == 0:
            return 0

        base_score = prs * self.PR_BASE_SCORE

        # Bonus for maintaining optimal PR size
        avg_pr_size = total_lines / prs if prs > 0 else 0
        if avg_pr_size <= self.OPTIMAL_PR_SIZE:
            size_bonus = 1.2  # 20% bonus for optimal size
        else:
            # Penalty for oversized PRs
            size_bonus = max(0.7, 1 - (avg_pr_size - self.OPTIMAL_PR_SIZE) / 1000)

        return base_score * size_bonus

    def _calculate_code_impact_score(self, lines_added: int, lines_removed: int) -> float:
        """Calculate code impact score with balanced add/remove consideration and enhanced diminishing returns.

        WHY: Massive single commits can unfairly inflate scores. This implementation
        uses stronger diminishing returns to prevent score inflation from extremely
        large commits while still rewarding meaningful contributions.
        """
        # Research shows deletions are valuable (refactoring, cleanup)
        # Weight deletions at 70% of additions
        effective_lines = lines_added + (lines_removed * 0.7)

        if effective_lines == 0:
            return 0

        # Logarithmic scaling to prevent gaming with massive changes
        # First 500 lines worth full value
        if effective_lines <= 500:
            return effective_lines * 0.2
        else:
            base = 500 * 0.2
            # Enhanced diminishing returns for massive commits
            if effective_lines <= 2000:
                extra = math.log10(effective_lines - 499) * 15  # Reduced multiplier
            else:
                # Very large commits get even more aggressive diminishing returns
                medium_extra = math.log10(2000 - 499) * 15
                large_extra = math.log10(effective_lines - 1999) * 8  # Much smaller multiplier
                extra = medium_extra + large_extra
            return base + extra

    def _calculate_complexity_score(self, files_changed: int, complexity_delta: float) -> float:
        """Calculate score based on breadth and complexity of changes."""
        if files_changed == 0:
            return 0

        # Base score from files touched (breadth of impact)
        file_score = min(files_changed * 5, 50)  # Cap at 50 points

        # Complexity factor (can be negative for simplification)
        # Reward simplification (negative complexity delta)
        if complexity_delta < 0:
            complexity_bonus = abs(complexity_delta) * 0.5  # Reward simplification
        else:
            complexity_bonus = -min(
                complexity_delta * 0.2, 10
            )  # Small penalty for added complexity

        return max(0, file_score + complexity_bonus)

    def _normalize_score(self, raw_score: float, has_pr_data: bool = True) -> float:
        """Normalize score to 0-100 range.

        Args:
            raw_score: The calculated raw activity score
            has_pr_data: Whether PR data was available (affects normalization divisor)
        """
        # Based on research, a highly productive week might have:
        # - 15 commits (150 points after scaling)
        # - 3 PRs of optimal size (180 points)
        # - 1000 effective lines (120 points)
        # - 20 files changed (50 points)
        # Total: ~500 points = 100 normalized

        # When PR data is unavailable (e.g., weekly reports), adjust divisor
        # since PR component (30% weight) contributes 0
        # Effective max becomes 70% of 50 = 35
        divisor = 35 if not has_pr_data else 50

        normalized = (raw_score / divisor) * 100
        return min(100, normalized)  # Cap at 100

    def _get_activity_level(self, raw_score: float, has_pr_data: bool = True) -> str:
        """Categorize activity level based on score."""
        normalized = self._normalize_score(raw_score, has_pr_data)

        if normalized >= 80:
            return "exceptional"
        elif normalized >= 60:
            return "high"
        elif normalized >= 40:
            return "moderate"
        elif normalized >= 20:
            return "low"
        else:
            return "minimal"

    def calculate_team_relative_score(
        self, individual_score: float, team_scores: list[float]
    ) -> dict[str, Any]:
        """Calculate relative performance within team context.

        Args:
            individual_score: Individual's raw activity score
            team_scores: List of all team members' raw scores

        Returns:
            Dictionary with percentile and relative metrics
        """
        if not team_scores:
            return {"percentile": 50, "relative_score": 1.0, "team_position": "average"}

        # Calculate percentile
        scores_below = sum(1 for score in team_scores if score < individual_score)
        percentile = (scores_below / len(team_scores)) * 100

        # Calculate relative to team average
        team_avg = sum(team_scores) / len(team_scores)
        relative_score = individual_score / team_avg if team_avg > 0 else 1.0

        # Determine position
        if percentile >= 90:
            position = "top_performer"
        elif percentile >= 75:
            position = "above_average"
        elif percentile >= 25:
            position = "average"
        else:
            position = "below_average"

        return {
            "percentile": round(percentile, 1),
            "relative_score": round(relative_score, 2),
            "team_position": position,
            "team_average": round(team_avg, 1),
        }

    def normalize_scores_on_curve(
        self, developer_scores: dict[str, float], curve_mean: float = 50.0, curve_std: float = 15.0
    ) -> dict[str, dict[str, Any]]:
        """Normalize activity scores on a bell curve with quintile grouping.

        Args:
            developer_scores: Dictionary mapping developer IDs to raw scores
            curve_mean: Target mean for the normalized distribution (default: 50)
            curve_std: Target standard deviation for the distribution (default: 15)

        Returns:
            Dictionary with normalized scores and quintile groupings
        """
        if not developer_scores:
            return {}

        # Get all scores
        scores = list(developer_scores.values())

        # Calculate current statistics
        current_mean = sum(scores) / len(scores)
        variance = sum((x - current_mean) ** 2 for x in scores) / len(scores)
        current_std = math.sqrt(variance) if variance > 0 else 1.0

        # Normalize to bell curve
        normalized_scores = {}
        for dev_id, raw_score in developer_scores.items():
            # Z-score normalization
            z_score = (raw_score - current_mean) / current_std if current_std > 0 else 0

            # Transform to target distribution
            curved_score = curve_mean + (z_score * curve_std)

            # Ensure scores stay in reasonable range (0-100)
            curved_score = max(0, min(100, curved_score))

            normalized_scores[dev_id] = curved_score

        # Sort developers by normalized score for quintile assignment
        sorted_devs = sorted(normalized_scores.items(), key=lambda x: x[1])

        # Assign quintiles
        results = {}
        quintile_size = len(sorted_devs) / 5

        for idx, (dev_id, curved_score) in enumerate(sorted_devs):
            # Determine quintile (1-5)
            quintile = min(5, int(idx / quintile_size) + 1)

            # Determine activity level based on quintile
            if quintile == 5:
                activity_level = "exceptional"
                level_description = "Top 20%"
            elif quintile == 4:
                activity_level = "high"
                level_description = "60-80th percentile"
            elif quintile == 3:
                activity_level = "moderate"
                level_description = "40-60th percentile"
            elif quintile == 2:
                activity_level = "low"
                level_description = "20-40th percentile"
            else:  # quintile == 1
                activity_level = "minimal"
                level_description = "Bottom 20%"

            # Calculate exact percentile
            percentile = ((idx + 0.5) / len(sorted_devs)) * 100

            results[dev_id] = {
                "raw_score": developer_scores[dev_id],
                "curved_score": round(curved_score, 1),
                "quintile": quintile,
                "activity_level": activity_level,
                "level_description": level_description,
                "percentile": round(percentile, 0),
                "z_score": (
                    round((developer_scores[dev_id] - current_mean) / current_std, 2)
                    if current_std > 0
                    else 0
                ),
            }

        return results
