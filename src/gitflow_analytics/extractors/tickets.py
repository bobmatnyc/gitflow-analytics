"""Ticket reference extraction for multiple platforms.

Core extraction and categorization live here.  Coverage analysis and untracked-
commit pattern analysis live in tickets_analysis.py (TicketAnalysisMixin).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config.schema import TicketDetectionConfig

logger = logging.getLogger(__name__)

from .tickets_analysis import TicketAnalysisMixin  # noqa: E402


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


class TicketExtractor(TicketAnalysisMixin):
    """Extract ticket references from various issue tracking systems.

    Enhanced to support detailed untracked commit analysis including:
    - Commit categorization (maintenance, bug fix, refactor, docs, etc.)
    - Configurable file change thresholds
    - Extended untracked commit metadata collection

    Coverage/untracked analysis methods are inherited from TicketAnalysisMixin.
    """

    def __init__(
        self,
        allowed_platforms: list[str] | None = None,
        untracked_file_threshold: int = 1,
        ticket_detection_config: TicketDetectionConfig | None = None,
    ) -> None:
        """Initialize with patterns for different platforms.

        Args:
            allowed_platforms: List of platforms to extract tickets from.
                              If None, all platforms are allowed.
            untracked_file_threshold: Minimum number of files changed to consider
                                    a commit as 'significant' for untracked analysis.
                                    Default is 1 (all commits), previously was 3.
            ticket_detection_config: Optional configurable detection settings.
                                    When None the extractor uses backward-compatible
                                    defaults (all commits, full-message scan, no excludes).
        """
        self.allowed_platforms = allowed_platforms
        self.untracked_file_threshold = untracked_file_threshold

        # Store detection config (may be None – all code below handles that gracefully)
        self._detection_config = ticket_detection_config

        # Build commit_filter / target_branches / position from config or defaults
        self._commit_filter: str = (
            ticket_detection_config.commit_filter if ticket_detection_config else "all"
        )
        self._target_branches: list[str] = (
            list(ticket_detection_config.target_branches)
            if ticket_detection_config
            else ["develop", "main", "master"]
        )
        self._position: str = (
            ticket_detection_config.position if ticket_detection_config else "anywhere"
        )

        # Compile exclude patterns (post-extraction filter)
        raw_excludes: list[str] = (
            list(ticket_detection_config.exclude_patterns) if ticket_detection_config else []
        )
        self._compiled_excludes: list[re.Pattern[str]] = [re.compile(p) for p in raw_excludes]

        # Repository names to exclude from ticket compliance metrics.
        self._exclude_repos: set[str] = (
            set(ticket_detection_config.exclude_repos) if ticket_detection_config else set()
        )

        # Author names to exclude from ticket compliance metrics.
        # Stored as lowercase for case-insensitive substring matching.
        self._exclude_authors: list[str] = (
            [a.lower() for a in ticket_detection_config.exclude_authors]
            if ticket_detection_config
            else []
        )

        # If config supplies custom per-platform patterns, use them; otherwise fall back
        # to the built-in defaults.  The config patterns dict maps platform -> single
        # pattern string; the built-in defaults use lists.
        config_patterns: dict[str, str] = (
            dict(ticket_detection_config.patterns)
            if ticket_detection_config and ticket_detection_config.patterns
            else {}
        )

        # Built-in defaults (always present as fallback)
        default_patterns: dict[str, list[str]] = {
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

        # Merge: config patterns override defaults for their platform
        self.patterns: dict[str, list[str]] = {}
        for platform, patterns in default_patterns.items():
            if platform in config_patterns:
                # Config supplies a single pattern string; wrap in a list
                self.patterns[platform] = [config_patterns[platform]]
            else:
                self.patterns[platform] = list(patterns)

        # Compile patterns only for allowed platforms
        self.compiled_patterns: dict[str, list[re.Pattern[str]]] = {}
        for platform, patterns in self.patterns.items():
            # Skip platforms not in allowed list
            if self.allowed_platforms and platform not in self.allowed_platforms:
                continue
            compiled: list[re.Pattern[str]] = []
            for pattern in patterns:
                anchored = f"^{pattern}" if self._position == "start" else pattern
                compiled.append(re.compile(anchored, re.IGNORECASE if platform != "jira" else 0))
            self.compiled_patterns[platform] = compiled

        # Commit categorization patterns
        self.category_patterns = {
            "bug_fix": [
                r"^fix(\([^)]+\))?:",  # Conventional commits: fix: or fix(scope):
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
                r"\b(incorrect|wrong|invalid)\b",
            ],
            "feature": [
                r"^(feat|feature)(\([^)]+\))?:",  # Conventional commits: feat: or feat(scope):
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
                r"\b(extract|harness|scaffold|bootstrap|wire|hook)\b",
                r"\b(disable|toggle|flag)\b",
            ],
            "refactor": [
                r"^refactor(\([^)]+\))?:",  # Conventional commits: refactor: or refactor(scope):
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
                r"\b(doc|docs|documentation|readme)\b",
                r"\b(javadoc|jsdoc|docstring|sphinx)\b",
                r"\b(manual|guide|tutorial|how-to|howto)\b",
                r"\b(explain|clarify|describe)\b",
                r"\b(changelog|notes|examples)\b",
                r"\bupdating\s+readme\b",
                r"\bdoc\s+update\b",
                r"\bdocumentation\s+fix\b",
                # "comment" only matches when paired with doc-context words to avoid
                # false positives on commits like "add comment to PR" or "remove comment"
                r"\bcomments?\b.*\b(doc|readme|changelog|docstring)\b",
                r"\b(doc|readme|changelog|docstring)\b.*\bcomments?\b",
            ],
            "deployment": [
                r"^deploy(\([^)]+\))?:",
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
                r"\b(npmrc|\.npmrc)\b",
                r"\b(ECR)\b",
                r"\b(GitHub\s+Actions|workflow)\b(?!.*\b(run|test)\b)",
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
                r"^chore(\([^)]+\))?:",
                r"\b(chore|cleanup|housekeeping|maintenance)\b",
                r"\b(routine|regular|scheduled)\b",
                r"\b(lint|linting|format|formatting|prettier)\b",
                r"\b(gitignore|ignore\s+file|artifacts)\b",
                r"\b(console|debug|log|logging)\s+(removal?|clean)\b",
                r"\b(sync|auto-sync)\b",
                r"\b(script\s+update|merge\s+main)\b",
                r"removes?\s+(console|debug|log)\b",
                # PR / code review response patterns
                r"^PR\s+comments?\s*$",
                r"(?:address|respond|applied?)\s+(?:PR|review|feedback|comments?)",
                r"suggestion\s+from\s+@?copilot",
                r"^apply\s+suggestion\b",
            ],
            "wip": [
                r"\b(wip|work\s+in\s+progress|temp|temporary|tmp)\b",
                r"\b(draft|unfinished|partial|incomplete)\b",
                r"\b(placeholder|todo|fixme)\b",
                r"^wip(\([^)]+\))?:",
                r"\b(experiment|experimental|poc|proof\s+of\s+concept)\b",
                r"\b(temporary|temp)\s+(fix|solution|workaround)\b",
            ],
            "version": [
                r"\b(version|bump|tag)\b",
                r"\b(v\d+\.\d+|version\s+\d+|\d+\.\d+\.\d+)\b",
                r"\b(major|minor|patch)\s+(version|release|bump)\b",
                r"^(version|bump)(\([^)]+\))?:",
                r"\b(prepare\s+for\s+release|pre-release)\b",
            ],
            "maintenance": [
                r"^chore(\([^)]+\))?:",  # Conventional commits: chore: or chore(scope):
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
                # Revert commits
                r"^revert\b",
                r"\breverting\b",
                r"\breverted?\s",
            ],
            "test": [
                r"^test(\([^)]+\))?:",
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
                r"^style(\([^)]+\))?:",
                r"\b(format|formatting|style|lint|linting)\b",
                r"\b(prettier|eslint|black|autopep8|rubocop)\b",
                r"\b(whitespace|indentation|spacing|tabs)\b",
                r"\b(code\s+style|consistent|standardize)\b",
            ],
            "build": [
                r"^build(\([^)]+\))?:",
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

    def should_skip_commit(self, commit: Any) -> bool:
        """Decide whether a commit should be skipped during ticket extraction.

        Uses the ``commit_filter`` setting from :class:`TicketDetectionConfig`.

        * ``"all"`` – never skip (backward-compatible default).
        * ``"squash_merges_only"`` – skip commits that are NOT squash merges.
          Heuristic: a squash merge has exactly one parent AND is on a target
          branch (or its branch metadata indicates so).
        * ``"merge_commits"`` – skip commits that do NOT have >1 parent.

        The *commit* argument is expected to be a dict with at least:
        ``{"parents": [...], "branch": str}`` or a ``git.Commit`` object with
        ``.parents`` and optionally a ``.branch`` attribute.  If the commit
        object does not expose branch information the filter degrades gracefully.

        Args:
            commit: A dict or object representing a single commit.

        Returns:
            True when the commit should be skipped, False when it should be
            processed normally.
        """
        if self._commit_filter == "all":
            return False

        # Normalise: support both dict and object access
        if isinstance(commit, dict):
            parents = commit.get("parents", [])
            branch = commit.get("branch", "")
        else:
            parents = getattr(commit, "parents", [])
            branch = getattr(commit, "branch", "")

        parent_count = len(parents)

        if self._commit_filter == "squash_merges_only":
            # A squash merge is a single-parent commit on an integration branch.
            on_target = any(
                branch == tb or (branch or "").endswith(f"/{tb}") for tb in self._target_branches
            )
            is_squash = parent_count == 1 and on_target
            return not is_squash

        if self._commit_filter == "merge_commits":
            return parent_count <= 1

        # Unknown filter value – default to not skipping
        return False

    def _is_excluded(self, ticket_id: str) -> bool:
        """Return True if *ticket_id* matches any configured exclude pattern.

        Args:
            ticket_id: The raw ticket identifier string (e.g. ``"CVE-2026-1234"``).

        Returns:
            True when the ticket should be discarded.
        """
        return any(pattern.fullmatch(ticket_id) for pattern in self._compiled_excludes)

    def extract_from_text(self, text: str) -> list[dict[str, str]]:
        """Extract all ticket references from text.

        Applies exclude-pattern filtering before returning results.
        """
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

                    # Post-extraction exclude-pattern filter
                    if self._is_excluded(ticket_id):
                        logger.debug("Discarding ticket %r: matches an exclude pattern", ticket_id)
                        continue

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

        # First, check for conventional commit patterns (^prefix: or ^prefix(scope):)
        # which have absolute priority.  The optional scope group handles
        # messages like "fix(searchlight): ..." or "feat(agents): ...".
        # Use a list of tuples to avoid duplicate key issues (build vs ci)
        conventional_patterns = [
            ("chore", r"^chore(\([^)]+\))?:"),
            ("style", r"^style(\([^)]+\))?:"),
            ("bug_fix", r"^fix(\([^)]+\))?:"),
            ("feature", r"^(feat|feature)(\([^)]+\))?:"),
            ("test", r"^test(\([^)]+\))?:"),
            ("build", r"^build(\([^)]+\))?:"),
            ("build", r"^ci(\([^)]+\))?:"),
            ("deployment", r"^deploy(\([^)]+\))?:"),
            ("wip", r"^wip(\([^)]+\))?:"),
            ("version", r"^(version|bump)(\([^)]+\))?:"),
            ("documentation", r"^docs(\([^)]+\))?:"),
            ("refactor", r"^refactor(\([^)]+\))?:"),
            ("performance", r"^perf(\([^)]+\))?:"),
        ]

        for category, pattern in conventional_patterns:
            if re.match(pattern, message_lower):
                return category

        # Check for revert commits early — before general body patterns, because
        # revert messages often contain words that match other categories
        # (e.g. "Revert incorrect ..." would match bug_fix's "incorrect" keyword).
        if re.match(r"^revert\b", message_lower):
            return "maintenance"

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

    def _format_ticket_id(self, platform: str, ticket_id: str) -> str:
        """Format ticket ID for display."""
        if platform == "github":
            return f"#{ticket_id}"
        elif platform == "clickup":
            return f"CU-{ticket_id}" if not ticket_id.startswith("CU-") else ticket_id
        else:
            return ticket_id
