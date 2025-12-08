"""Ticket reference extraction for multiple platforms."""

import logging
import re
from collections import defaultdict
from datetime import timezone
from typing import Any, Optional, cast

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)


def filter_git_artifacts(message: str) -> str:
    """Filter out git artifacts from commit messages before classification.

    WHY: Git-generated content like Co-authored-by lines, Signed-off-by lines,
    and other metadata should not influence commit classification. This function
    removes such artifacts to provide cleaner input for categorization.

    Args:
        message: Raw commit message that may contain git artifacts

    Returns:
        Cleaned commit message with git artifacts removed
    """
    if not message or not message.strip():
        return ""

    # Remove Co-authored-by lines (including standalone ones)
    message = re.sub(r"^Co-authored-by:.*$", "", message, flags=re.MULTILINE | re.IGNORECASE)

    # Remove Signed-off-by lines
    message = re.sub(r"^Signed-off-by:.*$", "", message, flags=re.MULTILINE | re.IGNORECASE)

    # Remove Reviewed-by lines (common in some workflows)
    message = re.sub(r"^Reviewed-by:.*$", "", message, flags=re.MULTILINE | re.IGNORECASE)

    # Remove Tested-by lines
    message = re.sub(r"^Tested-by:.*$", "", message, flags=re.MULTILINE | re.IGNORECASE)

    # Remove merge artifact lines (dashes, stars, or other separator patterns)
    message = re.sub(r"^-+$", "", message, flags=re.MULTILINE)
    message = re.sub(r"^\*\s*$", "", message, flags=re.MULTILINE)
    message = re.sub(r"^#+$", "", message, flags=re.MULTILINE)

    # Remove GitHub Copilot co-authorship lines
    message = re.sub(
        r"^Co-authored-by:.*[Cc]opilot.*$", "", message, flags=re.MULTILINE | re.IGNORECASE
    )

    # Remove common merge commit artifacts
    message = re.sub(
        r"^\s*Merge\s+(branch|pull request).*$", "", message, flags=re.MULTILINE | re.IGNORECASE
    )
    message = re.sub(
        r"^\s*(into|from)\s+[a-zA-Z0-9/_-]+$", "", message, flags=re.MULTILINE | re.IGNORECASE
    )

    # Clean up whitespace while preserving meaningful blank lines
    lines = message.split("\n")
    cleaned_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped:  # Non-empty line
            cleaned_lines.append(stripped)
        elif (
            i > 0
            and i < len(lines) - 1
            and any(line.strip() for line in lines[:i])
            and any(line.strip() for line in lines[i + 1 :])
        ):  # Preserve blank lines in middle if there's content both before and after
            cleaned_lines.append("")

    cleaned = "\n".join(cleaned_lines)

    # Handle edge cases - empty or dots-only messages
    if not cleaned:
        return ""

    # Check if message is only dots (with any whitespace)
    dots_only = re.sub(r"[.\s\n]+", "", cleaned) == ""
    if dots_only and "..." in cleaned:
        return ""

    return cleaned.strip()


class TicketExtractor:
    """Extract ticket references from various issue tracking systems.

    Enhanced to support detailed untracked commit analysis including:
    - Commit categorization (maintenance, bug fix, refactor, docs, etc.)
    - Configurable file change thresholds
    - Extended untracked commit metadata collection
    """

    def __init__(
        self, allowed_platforms: Optional[list[str]] = None, untracked_file_threshold: int = 1
    ) -> None:
        """Initialize with patterns for different platforms.

        Args:
            allowed_platforms: List of platforms to extract tickets from.
                              If None, all platforms are allowed.
            untracked_file_threshold: Minimum number of files changed to consider
                                    a commit as 'significant' for untracked analysis.
                                    Default is 1 (all commits), previously was 3.
        """
        self.allowed_platforms = allowed_platforms
        self.untracked_file_threshold = untracked_file_threshold
        self.patterns = {
            "jira": [
                r"([A-Z]{2,10}-\d+)",  # Standard JIRA format: PROJ-123
            ],
            "github": [
                r"#(\d+)",  # GitHub issues: #123
                r"GH-(\d+)",  # Alternative format: GH-123
                r"(?:fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved)\s+#(\d+)",
            ],
            "clickup": [
                r"CU-([a-z0-9]+)",  # ClickUp: CU-abc123
                r"#([a-z0-9]{6,})",  # ClickUp short format
            ],
            "linear": [
                r"([A-Z]{2,5}-\d+)",  # Linear: ENG-123, similar to JIRA
                r"LIN-(\d+)",  # Alternative: LIN-123
            ],
        }

        # Compile patterns only for allowed platforms
        self.compiled_patterns = {}
        for platform, patterns in self.patterns.items():
            # Skip platforms not in allowed list
            if self.allowed_platforms and platform not in self.allowed_platforms:
                continue
            self.compiled_patterns[platform] = [
                re.compile(pattern, re.IGNORECASE if platform != "jira" else 0)
                for pattern in patterns
            ]

        # Commit categorization patterns
        self.category_patterns = {
            "bug_fix": [
                r"^fix(\([^)]*\))?:",  # Conventional commits: fix: or fix(scope):
                r"\b(fix|bug|error|issue|problem|crash|exception|failure)\b",
                r"\b(resolve|solve|repair|correct|corrected|address)\b",
                r"\b(hotfix|bugfix|patch|quickfix)\b",
                r"\b(broken|failing|failed|fault|defect)\b",
                r"\b(prevent|stop|avoid)\s+(error|bug|issue|crash)\b",
                r"\b(fixes|resolves|solves)\s+(bug|issue|error|problem)\b",
                r"\b(beacon|beacons)\b.*\b(fix|fixes|issue|problem)\b",
                r"\bmissing\s+(space|field|data|property)\b",
                r"\b(counting|allowing|episodes)\s+(was|not|issue)\b",
                r"^fixes\s+\b(beacon|beacons|combo|issue|problem)\b",
                r"\bfixing\b(?!\s+test)",  # "fixing" but not "fixing tests"
                r"\bfixed?\s+(issue|problem|bug|error)\b",
                r"\bresolve[ds]?\s+(issue|problem|bug)\b",
                r"\brepair\b",
            ],
            "feature": [
                r"^(feat|feature)(\([^)]*\))?:",  # Conventional commits: feat: or feat(scope):
                r"\b(add|new|feature|implement|create|build)\b",
                r"\b(introduce|enhance|extend|expand)\b",
                r"\b(functionality|capability|support|enable)\b",
                r"\b(initial|first)\s+(implementation|version)\b",
                r"\b(addition|initialize|prepare)\b",
                r"added?\s+(new|feature|functionality|capability)\b",
                r"added?\s+(column|field|property|thumbnail)\b",
                r"\b(homilists?|homily|homilies)\b",
                r"\b(sticky|column)\s+(feature|functionality)\b",
                r"adds?\s+(data|localization|beacon)\b",
                r"\b(episode|episodes|audio|video)\s+(feature|support|implementation)\b",
                r"\b(beacon)\s+(implementation|for|tracking)\b",
                r"\b(localization)\s+(data|structure)\b",
            ],
            "refactor": [
                r"^refactor(\([^)]*\))?:",  # Conventional commits: refactor: or refactor(scope):
                r"\b(refactor|restructure|reorganize|cleanup|clean up)\b",
                r"\b(optimize|improve|simplify|streamline)\b",
                r"\b(rename|move|extract|consolidate)\b",
                r"\b(modernize|redesign|rework|rewrite)\b",
                r"\b(code\s+quality|tech\s+debt|legacy)\b",
                r"\b(refine|ensure|replace)\b",
                r"improves?\s+(performance|efficiency|structure)\b",
                r"improves?\s+(combo|box|focus|behavior)\b",
                r"using\s+\w+\s+instead\s+of\s+\w+\b",  # "using X instead of Y" pattern
                r"\brenaming\b",
                r"\brenamed?\b",
                r"\breduce\s+code\b",
                r"\bsimplify\b",
                r"\bsimplified\b",
                r"\bboilerplate\b",
                r"\bcode\s+cleanup\b",
            ],
            "documentation": [
                r"\b(doc|docs|documentation|readme|comment|comments)\b",
                r"\b(javadoc|jsdoc|docstring|sphinx)\b",
                r"\b(manual|guide|tutorial|how-to|howto)\b",
                r"\b(explain|clarify|describe)\b",
                r"\b(changelog|notes|examples)\b",
                r"\bupdating\s+readme\b",
                r"\bdoc\s+update\b",
                r"\bdocumentation\s+fix\b",
            ],
            "deployment": [
                r"^deploy:",
                r"\b(deploy|deployment|publish|rollout)\b",
                r"\b(production|prod|staging|live)\b",
                r"\b(go\s+live|launch|ship)\b",
                r"\b(promote|migration|migrate)\b",
                r"\brelease\s+(v\d+\.\d+|\d+\.\d+\.\d+)?\s+(to|on)\s+(production|staging|live)\b",
            ],
            "configuration": [
                r"\b(config|configure|configuration|setup|settings)\b",
                r"\b(env|environment|parameter|option)\b",
                r"\b(property|properties|yaml|json|xml)\b",
                r"\b(database\s+config|db\s+config|connection)\b",
                r"\.env|\.config|\.yaml|\.json",
                r"\b(setup|configure)\s+(new|for)\b",
                r"\b(user|role|permission|access)\s+(change|update|configuration)\b",
                r"\b(api|service|system)\s+(config|configuration|setup)\b",
                r"\b(role|permission|access)\s+(update|change|management)\b",
                r"\b(schema|model)\s+(update|change|addition)\b",
                r"changing\s+(user|role|permission)\s+(roles?|settings?)\b",
                r"\b(schema)\b(?!.*\b(test|spec)\b)",  # Schema but not test schemas
                r"\bsanity\s+schema\b",
                r"changing\s+(some)?\s*(user|role)\s+(roles?|permissions?)\b",
            ],
            "content": [
                r"\b(content|copy|text|wording|messaging)\b",
                r"\b(translation|i18n|l10n|locale|localize)\b",
                r"\b(language|multilingual|international)\b",
                r"\b(strings|labels|captions|titles)\b",
                r"\b(typo|spelling|grammar|proofreading)\b",
                r"\b(typo|spelling)\s+(in|on|for)\b",
                r"\b(spanish|translations?)\b",
                r"\b(blast|banner|video|media)\s+(content|update)\b",
                r"added?\s+(spanish|translation|text|copy|label)\b",
                r"\b(label|message)\s+(change|update|fix)\b",
            ],
            "ui": [
                r"\b(ui|ux|design|layout|styling|visual)\b",
                r"\b(css|scss|sass|less|style)\b",
                r"\b(responsive|mobile|desktop|tablet)\b",
                r"\b(theme|color|font|icon|image)\b",
                r"\b(component|widget|element|button|form)\b",
                r"\b(frontend|front-end|client-side)\b",
                r"\b(sticky|column)\b(?!.*\b(database|table)\b)",  # UI sticky, not database
                r"\b(focus|behavior)\b.*\b(combo|box)\b",
            ],
            "infrastructure": [
                r"\b(infra|infrastructure|aws|azure|gcp|cloud)\b",
                r"\b(docker|k8s|kubernetes|container|pod)\b",
                r"\b(terraform|ansible|chef|puppet)\b",
                r"\b(server|hosting|network|load\s+balancer)\b",
                r"\b(monitoring|logging|alerting|metrics)\b",
            ],
            "security": [
                r"\b(security|vulnerability|cve|exploit)\b",
                r"\b(auth|authentication|authorization|permission)\b",
                r"\b(ssl|tls|https|certificate|cert)\b",
                r"\b(encrypt|decrypt|hash|token|oauth)\b",
                r"\b(access\s+control|rbac|cors|xss|csrf)\b",
                r"\b(secure|safety|protect|prevent)\b",
            ],
            "performance": [
                r"\b(perf|performance|optimize|speed|faster)\b",
                r"\b(cache|caching|memory|cpu|disk)\b",
                r"\b(slow|lag|delay|timeout|bottleneck)\b",
                r"\b(efficient|efficiency|throughput|latency)\b",
                r"\b(load\s+time|response\s+time|benchmark)\b",
                r"\b(improve|better)\s+(load|performance|speed)\b",
            ],
            "chore": [
                r"^chore:",
                r"\b(chore|cleanup|housekeeping|maintenance)\b",
                r"\b(routine|regular|scheduled)\b",
                r"\b(lint|linting|format|formatting|prettier)\b",
                r"\b(gitignore|ignore\s+file|artifacts)\b",
                r"\b(console|debug|log|logging)\s+(removal?|clean)\b",
                r"\b(sync|auto-sync)\b",
                r"\b(script\s+update|merge\s+main)\b",
                r"removes?\s+(console|debug|log)\b",
            ],
            "wip": [
                r"\b(wip|work\s+in\s+progress|temp|temporary|tmp)\b",
                r"\b(draft|unfinished|partial|incomplete)\b",
                r"\b(placeholder|todo|fixme)\b",
                r"^wip:",
                r"\b(experiment|experimental|poc|proof\s+of\s+concept)\b",
                r"\b(temporary|temp)\s+(fix|solution|workaround)\b",
            ],
            "version": [
                r"\b(version|bump|tag)\b",
                r"\b(v\d+\.\d+|version\s+\d+|\d+\.\d+\.\d+)\b",
                r"\b(major|minor|patch)\s+(version|release|bump)\b",
                r"^(version|bump):",
                r"\b(prepare\s+for\s+release|pre-release)\b",
            ],
            "maintenance": [
                r"^chore(\([^)]*\))?:",  # Conventional commits: chore: or chore(scope):
                r"\b(update|upgrade|bump|maintenance|maint)\b",
                r"\b(dependency|dependencies|package|packages)\b",
                r"\b(npm\s+update|pip\s+install|yarn\s+upgrade)\b",
                r"\b(deprecated|obsolete|outdated)\b",
                r"package\.json|requirements\.txt|pom\.xml|Gemfile",
                r"\b(combo|beacon)\s+(hacking|fixes?)\b",
                r"\b(temp|temporary|hack|hacking)\b",
                r"\b(test|testing)\s+(change|update|fix)\b",
                r"\b(more|only)\s+(combo|beacon)\s+(hacking|fires?)\b",
                r"adds?\s+(console|debug|log)\b",
            ],
            "test": [
                r"^test:",
                r"\b(test|testing|spec|unit\s+test|integration\s+test)\b",
                r"\b(junit|pytest|mocha|jest|cypress|selenium)\b",
                r"\b(mock|stub|fixture|factory)\b",
                r"\b(e2e|end-to-end|acceptance|smoke)\b",
                r"\b(coverage|assert|expect|should)\b",
                r"\bfixing\s+tests?\b",
                r"\btest.*broke\b",
                r"\bupdate.*test\b",
                r"\bbroke.*test\b",
                r"\btest\s+fix\b",
            ],
            "style": [
                r"^style:",
                r"\b(format|formatting|style|lint|linting)\b",
                r"\b(prettier|eslint|black|autopep8|rubocop)\b",
                r"\b(whitespace|indentation|spacing|tabs)\b",
                r"\b(code\s+style|consistent|standardize)\b",
            ],
            "build": [
                r"^build:",
                r"\b(build|compile|bundle|webpack|rollup)\b",
                r"\b(ci|cd|pipeline|workflow|github\s+actions)\b",
                r"\b(docker|dockerfile|makefile|npm\s+scripts)\b",
                r"\b(jenkins|travis|circleci|gitlab)\b",
                r"\b(artifact|binary|executable|jar|war)\b",
            ],
            "integration": [
                r"\b(integrate|integration)\s+(with|posthog|iubenda|auth0)\b",
                r"\b(posthog|iubenda|auth0|oauth|third-party|external)\b",
                r"\b(api|endpoint|service)\s+(integration|connection|setup)\b",
                r"\b(connect|linking|sync)\s+(with|to)\s+[a-z]+(hog|enda|auth)\b",
                r"implement\s+(posthog|iubenda|auth0|api)\b",
                r"adding\s+(posthog|auth|integration)\b",
                r"\b(third-party|external)\s+(service|integration|api)\b",
                r"\bniveles\s+de\s+acceso\s+a\s+la\s+api\b",  # Spanish: API access levels
                r"\b(implementation|removing)\s+(iubenda|posthog|auth0)\b",
            ],
        }

        # Compile categorization patterns
        self.compiled_category_patterns = {}
        for category, patterns in self.category_patterns.items():
            self.compiled_category_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def extract_from_text(self, text: str) -> list[dict[str, str]]:
        """Extract all ticket references from text."""
        if not text:
            return []

        tickets = []
        seen = set()  # Avoid duplicates

        for platform, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                for match in matches:
                    ticket_id = match if isinstance(match, str) else match[0]

                    # Normalize ticket ID
                    if platform == "jira" or platform == "linear":
                        ticket_id = ticket_id.upper()

                    # Create unique key
                    key = f"{platform}:{ticket_id}"
                    if key not in seen:
                        seen.add(key)
                        tickets.append(
                            {
                                "platform": platform,
                                "id": ticket_id,
                                "full_id": self._format_ticket_id(platform, ticket_id),
                            }
                        )

        return tickets

    def extract_by_platform(self, text: str) -> dict[str, list[str]]:
        """Extract tickets grouped by platform."""
        tickets = self.extract_from_text(text)

        by_platform = defaultdict(list)
        for ticket in tickets:
            by_platform[ticket["platform"]].append(ticket["id"])

        return dict(by_platform)

    def analyze_ticket_coverage(
        self, commits: list[dict[str, Any]], prs: list[dict[str, Any]], progress_display=None
    ) -> dict[str, Any]:
        """Analyze ticket reference coverage across commits and PRs.

        Args:
            commits: List of commit dictionaries to analyze
            prs: List of PR dictionaries to analyze
            progress_display: Optional progress display for showing analysis progress

        Note:
            This method re-extracts tickets from commit messages rather than using cached
            'ticket_references' to ensure the analysis respects the current allowed_platforms
            configuration. Cached data may contain tickets from all platforms from previous runs.
        """
        ticket_platforms: defaultdict[str, int] = defaultdict(int)
        untracked_commits: list[dict[str, Any]] = []
        ticket_summary: defaultdict[str, set[str]] = defaultdict(set)

        results = {
            "total_commits": len(commits),
            "total_prs": len(prs),
            "commits_with_tickets": 0,
            "prs_with_tickets": 0,
            "ticket_platforms": ticket_platforms,
            "untracked_commits": untracked_commits,
            "ticket_summary": ticket_summary,
        }

        # Analyze commits
        commits_analyzed = 0
        commits_with_ticket_refs = 0
        tickets_found = 0

        # Set up progress tracking for commits
        commit_iterator = commits
        if progress_display and hasattr(progress_display, "console"):
            # Rich progress display available
            commit_iterator = commits  # Rich will handle its own progress
        elif TQDM_AVAILABLE:
            # Fall back to tqdm for simple progress tracking
            commit_iterator = tqdm(
                commits, desc="🎫 Analyzing commits for tickets", unit="commits", leave=False
            )

        for commit in commit_iterator:
            # Debug: check if commit is actually a dictionary
            if not isinstance(commit, dict):
                logger.error(f"Expected commit to be dict, got {type(commit)}: {commit}")
                continue

            commits_analyzed += 1
            # IMPORTANT: Re-extract tickets using current allowed_platforms instead of cached values
            # This ensures the analysis respects the current configuration
            commit_message = commit.get("message", "")
            ticket_refs = self.extract_from_text(commit_message)

            # Debug logging for the first few commits
            if commits_analyzed <= 5:
                logger.debug(
                    f"Commit {commits_analyzed}: hash={commit.get('hash', 'N/A')[:8]}, "
                    f"re-extracted ticket_refs={ticket_refs} (allowed_platforms={self.allowed_platforms})"
                )

            if ticket_refs:
                commits_with_ticket_refs += 1
                commits_with_tickets = cast(int, results["commits_with_tickets"])
                results["commits_with_tickets"] = commits_with_tickets + 1
                for ticket in ticket_refs:
                    if isinstance(ticket, dict):
                        platform = ticket.get("platform", "unknown")
                        ticket_id = ticket.get("id", "")
                    else:
                        # Legacy format - assume JIRA
                        platform = "jira"
                        ticket_id = ticket

                    platform_count = ticket_platforms[platform]
                    ticket_platforms[platform] = platform_count + 1
                    ticket_summary[platform].add(ticket_id)
                    tickets_found += 1
            else:
                # Track untracked commits with configurable threshold and enhanced data
                files_changed = self._get_files_changed_count(commit)
                if not commit.get("is_merge") and files_changed >= self.untracked_file_threshold:
                    # Categorize the commit
                    category = self.categorize_commit(commit.get("message", ""))

                    # Extract enhanced commit data
                    commit_data = {
                        "hash": commit.get("hash", "")[:7],
                        "full_hash": commit.get("hash", ""),
                        "message": commit.get("message", "").split("\n")[0][
                            :100
                        ],  # Increased from 60 to 100
                        "full_message": commit.get("message", ""),
                        "author": commit.get(
                            "canonical_name", commit.get("author_name", "Unknown")
                        ),
                        "author_email": commit.get("author_email", ""),
                        "canonical_id": commit.get("canonical_id", commit.get("author_email", "")),
                        "timestamp": commit.get("timestamp"),
                        "project_key": commit.get("project_key", "UNKNOWN"),
                        "files_changed": files_changed,
                        "lines_added": commit.get("insertions", 0),
                        "lines_removed": commit.get("deletions", 0),
                        "lines_changed": (commit.get("insertions", 0) + commit.get("deletions", 0)),
                        "category": category,
                        "is_merge": commit.get("is_merge", False),
                    }

                    untracked_commits.append(commit_data)

            # Update progress if using tqdm
            if TQDM_AVAILABLE and hasattr(commit_iterator, "set_postfix"):
                commit_iterator.set_postfix(
                    {
                        "tickets": tickets_found,
                        "with_tickets": commits_with_ticket_refs,
                        "untracked": len(untracked_commits),
                    }
                )

        # Analyze PRs
        pr_tickets_found = 0

        # Set up progress tracking for PRs (only if there are PRs to analyze)
        pr_iterator = prs
        if (
            prs
            and TQDM_AVAILABLE
            and not (progress_display and hasattr(progress_display, "console"))
        ):
            # Only show PR progress if there are PRs and we're not using Rich
            pr_iterator = tqdm(prs, desc="🎫 Analyzing PRs for tickets", unit="PRs", leave=False)

        for pr in pr_iterator:
            # Extract tickets from PR title and description
            pr_text = f"{pr.get('title', '')} {pr.get('description', '')}"
            tickets = self.extract_from_text(pr_text)

            if tickets:
                prs_with_tickets = cast(int, results["prs_with_tickets"])
                results["prs_with_tickets"] = prs_with_tickets + 1
                for ticket in tickets:
                    platform = ticket["platform"]
                    platform_count = ticket_platforms[platform]
                    ticket_platforms[platform] = platform_count + 1
                    ticket_summary[platform].add(ticket["id"])
                    pr_tickets_found += 1

            # Update PR progress if using tqdm
            if TQDM_AVAILABLE and hasattr(pr_iterator, "set_postfix"):
                pr_iterator.set_postfix(
                    {"tickets": pr_tickets_found, "with_tickets": results["prs_with_tickets"]}
                )

        # Calculate coverage percentages
        total_commits = cast(int, results["total_commits"])
        commits_with_tickets_count = cast(int, results["commits_with_tickets"])
        results["commit_coverage_pct"] = (
            commits_with_tickets_count / total_commits * 100 if total_commits > 0 else 0
        )

        total_prs = cast(int, results["total_prs"])
        prs_with_tickets_count = cast(int, results["prs_with_tickets"])
        results["pr_coverage_pct"] = (
            prs_with_tickets_count / total_prs * 100 if total_prs > 0 else 0
        )

        # Convert sets to counts for summary
        results["ticket_summary"] = {
            platform: len(tickets) for platform, tickets in ticket_summary.items()
        }

        # Sort untracked commits by timestamp (most recent first)
        # Handle timezone-aware and timezone-naive datetimes
        def safe_timestamp_key(commit):
            ts = commit.get("timestamp")
            if ts is None:
                return ""
            # If it's a datetime object, handle timezone issues
            if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                # Make timezone-naive datetime UTC-aware for consistent comparison
                ts = ts.replace(tzinfo=timezone.utc)
            return ts

        untracked_commits.sort(key=safe_timestamp_key, reverse=True)

        # Debug logging for ticket coverage analysis
        final_commits_with_tickets = cast(int, results["commits_with_tickets"])
        logger.debug(
            f"Ticket coverage analysis complete: {commits_analyzed} commits analyzed, {commits_with_ticket_refs} had ticket_refs, {final_commits_with_tickets} counted as with tickets"
        )
        if commits_analyzed > 0 and final_commits_with_tickets == 0:
            logger.warning(
                f"Zero commits with tickets found out of {commits_analyzed} commits analyzed"
            )

        return results

    def calculate_developer_ticket_coverage(
        self, commits: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate ticket coverage percentage per developer.

        WHY: Individual developer ticket coverage was hardcoded to 0.0, causing
        reports to show contradictory information where total coverage was >0%
        but all individual developers showed 0%. This method provides the missing
        per-developer calculation.

        DESIGN DECISION: Uses canonical_id when available (post-identity resolution)
        or falls back to author_email for consistent developer identification.
        The coverage calculation only considers commits that meet the untracked
        file threshold to maintain consistency with the overall analysis.

        Args:
            commits: List of commit dictionaries with ticket_references and identity info

        Returns:
            Dictionary mapping canonical_id/author_email to coverage percentage
        """
        if not commits:
            return {}

        # Group commits by developer (canonical_id preferred, fallback to author_email)
        developer_commits = {}
        developer_with_tickets = {}

        for commit in commits:
            # Skip merge commits (consistent with main analysis)
            if commit.get("is_merge"):
                continue

            # Only count commits that meet the file threshold (consistent with untracked analysis)
            files_changed = self._get_files_changed_count(commit)
            if files_changed < self.untracked_file_threshold:
                continue

            # Determine developer identifier (canonical_id preferred)
            developer_id = commit.get("canonical_id") or commit.get("author_email", "unknown")

            # Initialize counters for this developer
            if developer_id not in developer_commits:
                developer_commits[developer_id] = 0
                developer_with_tickets[developer_id] = 0

            # Count total commits for this developer
            developer_commits[developer_id] += 1

            # Count commits with ticket references
            ticket_refs = commit.get("ticket_references", [])
            if ticket_refs:
                developer_with_tickets[developer_id] += 1

        # Calculate coverage percentages
        coverage_by_developer = {}
        for developer_id in developer_commits:
            total_commits = developer_commits[developer_id]
            commits_with_tickets = developer_with_tickets[developer_id]

            if total_commits > 0:
                coverage_pct = (commits_with_tickets / total_commits) * 100
                coverage_by_developer[developer_id] = round(coverage_pct, 1)
            else:
                coverage_by_developer[developer_id] = 0.0

        logger.debug(f"Calculated ticket coverage for {len(coverage_by_developer)} developers")
        return coverage_by_developer

    def _get_files_changed_count(self, commit: dict[str, Any]) -> int:
        """Extract the number of files changed from commit data.

        WHY: Commit data can have files_changed as either an integer count
        or a list of file paths. This method handles both cases correctly
        and provides a consistent integer count for analysis.

        DESIGN DECISION: Priority order is:
        1. files_changed_count (if present, use directly)
        2. files_changed as integer (use directly)
        3. files_changed as list (use length)
        4. Default to 0 if none available

        Args:
            commit: Commit data dictionary

        Returns:
            Integer count of files changed
        """
        # First priority: explicit count field
        if "files_changed_count" in commit:
            return commit["files_changed_count"]

        # Second priority: files_changed field
        files_changed = commit.get("files_changed")
        if files_changed is not None:
            if isinstance(files_changed, int):
                return files_changed
            elif isinstance(files_changed, list):
                return len(files_changed)

        # Default fallback
        return 0

    def categorize_commit(self, message: str) -> str:
        """Categorize a commit based on its message.

        WHY: Commit categorization helps identify patterns in untracked work,
        enabling better insights into what types of work are not being tracked
        through tickets. This supports improved process recommendations.

        DESIGN DECISION: Categories are checked in priority order to ensure
        more specific patterns match before general ones. For example,
        "security" patterns are checked before "feature" patterns to prevent
        "add authentication" from being classified as a feature instead of security.

        Args:
            message: The commit message to categorize

        Returns:
            String category (bug_fix, feature, refactor, documentation,
            maintenance, test, style, build, or other)
        """
        if not message:
            return "other"

        # Filter git artifacts before categorization
        cleaned_message = filter_git_artifacts(message)
        if not cleaned_message:
            return "other"

        # Remove ticket references to focus on content analysis
        # This helps classify commits with ticket references based on their actual content
        message_without_tickets = self._remove_ticket_references(cleaned_message)
        message_lower = message_without_tickets.lower()

        # Define priority order - conventional commits first, then specific patterns
        priority_order = [
            # Conventional commit formats (start with specific prefixes)
            "wip",  # ^wip: prefix
            "chore",  # ^chore: prefix
            "style",  # ^style: prefix
            "bug_fix",  # ^fix: prefix
            "feature",  # ^feat: prefix
            "test",  # ^test: prefix
            "build",  # ^build: prefix
            "deployment",  # ^deploy: prefix and specific deployment terms
            # Specific domain patterns (no conventional prefix conflicts)
            "version",  # Version-specific patterns
            "security",  # Security-specific terms
            "performance",  # Performance-specific terms
            "infrastructure",  # Infrastructure-specific terms
            "integration",  # Third-party integration terms
            "configuration",  # Configuration-specific terms
            "content",  # Content-specific terms
            "ui",  # UI-specific terms
            "documentation",  # Documentation terms
            "refactor",  # Refactoring terms
            "maintenance",  # General maintenance terms
        ]

        # First, check for conventional commit patterns (^prefix:) which have absolute priority
        conventional_patterns = {
            "chore": r"^chore:",
            "style": r"^style:",
            "bug_fix": r"^fix:",
            "feature": r"^(feat|feature):",
            "test": r"^test:",
            "build": r"^build:",
            "deployment": r"^deploy:",
            "wip": r"^wip:",
            "version": r"^(version|bump):",
        }

        for category, pattern in conventional_patterns.items():
            if re.match(pattern, message_lower):
                return category

        # Then check categories in priority order for non-conventional patterns
        for category in priority_order:
            if category in self.compiled_category_patterns:
                for pattern in self.compiled_category_patterns[category]:
                    if pattern.search(message_lower):
                        return category

        return "other"

    def _remove_ticket_references(self, message: str) -> str:
        """Remove ticket references from commit message to focus on content analysis.

        WHY: Ticket references like 'RMVP-941' or '[CNA-482]' don't indicate the type
        of work being done. We need to analyze the actual description to properly
        categorize commits with ticket references.

        Args:
            message: The commit message possibly containing ticket references

        Returns:
            Message with ticket references removed, focusing on the actual description
        """
        if not message:
            return ""

        # Remove common ticket patterns at the start of messages
        patterns_to_remove = [
            # JIRA-style patterns
            r"^[A-Z]{2,10}-\d+:?\s*",  # RMVP-941: or RMVP-941
            r"^\[[A-Z]{2,10}-\d+\]\s*",  # [CNA-482]
            # GitHub issue patterns
            r"^#\d+:?\s*",  # #123: or #123
            r"^GH-\d+:?\s*",  # GH-123:
            # ClickUp patterns
            r"^CU-[a-z0-9]+:?\s*",  # CU-abc123:
            # Linear patterns
            r"^[A-Z]{2,5}-\d+:?\s*",  # ENG-123:
            r"^LIN-\d+:?\s*",  # LIN-123:
            # GitHub PR patterns in messages
            r"\(#\d+\)$",  # (#115) at end
            r"\(#\d+\)\s*\(#\d+\)*\s*$",  # (#131) (#133) (#134) at end
            # Other ticket-like patterns
            r"^[A-Z]{2,10}\s+\d+\s*",  # NEWS 206
        ]

        cleaned_message = message
        for pattern in patterns_to_remove:
            cleaned_message = re.sub(pattern, "", cleaned_message, flags=re.IGNORECASE).strip()

        # If we removed everything, return the original message
        # This handles cases where the entire message was just a ticket reference
        if not cleaned_message.strip():
            return message

        return cleaned_message

    def analyze_untracked_patterns(self, untracked_commits: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze patterns in untracked commits for insights.

        WHY: Understanding patterns in untracked work helps identify:
        - Common types of work that bypass ticket tracking
        - Developers who need process guidance
        - Categories of work that should be tracked vs. allowed to be untracked

        Args:
            untracked_commits: List of untracked commit data

        Returns:
            Dictionary with pattern analysis results
        """
        if not untracked_commits:
            return {
                "total_untracked": 0,
                "categories": {},
                "top_contributors": [],
                "projects": {},
                "avg_commit_size": 0,
                "recommendations": [],
            }

        # Category analysis
        categories = {}
        for commit in untracked_commits:
            category = commit.get("category", "other")
            if category not in categories:
                categories[category] = {"count": 0, "lines_changed": 0, "examples": []}
            categories[category]["count"] += 1
            categories[category]["lines_changed"] += commit.get("lines_changed", 0)
            if len(categories[category]["examples"]) < 3:
                categories[category]["examples"].append(
                    {
                        "hash": commit.get("hash", ""),
                        "message": commit.get("message", ""),
                        "author": commit.get("author", ""),
                    }
                )

        # Contributor analysis
        contributors = {}
        for commit in untracked_commits:
            author = commit.get("canonical_id", commit.get("author_email", "Unknown"))
            if author not in contributors:
                contributors[author] = {"count": 0, "categories": set()}
            contributors[author]["count"] += 1
            contributors[author]["categories"].add(commit.get("category", "other"))

        # Convert sets to lists for JSON serialization
        for author_data in contributors.values():
            author_data["categories"] = list(author_data["categories"])

        # Top contributors
        top_contributors = sorted(
            [(author, data["count"]) for author, data in contributors.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        # Project analysis
        projects = {}
        for commit in untracked_commits:
            project = commit.get("project_key", "UNKNOWN")
            if project not in projects:
                projects[project] = {"count": 0, "categories": set()}
            projects[project]["count"] += 1
            projects[project]["categories"].add(commit.get("category", "other"))

        # Convert sets to lists for JSON serialization
        for project_data in projects.values():
            project_data["categories"] = list(project_data["categories"])

        # Calculate average commit size
        total_lines = sum(commit.get("lines_changed", 0) for commit in untracked_commits)
        avg_commit_size = total_lines / len(untracked_commits) if untracked_commits else 0

        # Generate recommendations
        recommendations = self._generate_untracked_recommendations(
            categories, contributors, projects, len(untracked_commits)
        )

        return {
            "total_untracked": len(untracked_commits),
            "categories": categories,
            "top_contributors": top_contributors,
            "projects": projects,
            "avg_commit_size": round(avg_commit_size, 1),
            "recommendations": recommendations,
        }

    def _generate_untracked_recommendations(
        self,
        categories: dict[str, Any],
        contributors: dict[str, Any],
        projects: dict[str, Any],
        total_untracked: int,
    ) -> list[dict[str, str]]:
        """Generate recommendations based on untracked commit patterns."""
        recommendations = []

        # Category-based recommendations
        if categories.get("feature", {}).get("count", 0) > total_untracked * 0.2:
            recommendations.append(
                {
                    "type": "process",
                    "title": "Track Feature Development",
                    "description": "Many feature commits lack ticket references. Consider requiring tickets for new features.",
                    "priority": "high",
                }
            )

        if categories.get("bug_fix", {}).get("count", 0) > total_untracked * 0.15:
            recommendations.append(
                {
                    "type": "process",
                    "title": "Improve Bug Tracking",
                    "description": "Bug fixes should be tracked through issue management systems.",
                    "priority": "high",
                }
            )

        # Allow certain categories to be untracked
        low_priority_categories = ["style", "documentation", "maintenance"]
        low_priority_count = sum(
            categories.get(cat, {}).get("count", 0) for cat in low_priority_categories
        )

        if low_priority_count > total_untracked * 0.6:
            recommendations.append(
                {
                    "type": "positive",
                    "title": "Appropriate Untracked Work",
                    "description": "Most untracked commits are maintenance/style/docs - this is acceptable.",
                    "priority": "low",
                }
            )

        # Contributor-based recommendations
        if len(contributors) > 1:
            max_contributor_count = max(data["count"] for data in contributors.values())
            if max_contributor_count > total_untracked * 0.5:
                recommendations.append(
                    {
                        "type": "team",
                        "title": "Provide Process Training",
                        "description": "Some developers need guidance on ticket referencing practices.",
                        "priority": "medium",
                    }
                )

        return recommendations

    def _format_ticket_id(self, platform: str, ticket_id: str) -> str:
        """Format ticket ID for display."""
        if platform == "github":
            return f"#{ticket_id}"
        elif platform == "clickup":
            return f"CU-{ticket_id}" if not ticket_id.startswith("CU-") else ticket_id
        else:
            return ticket_id
