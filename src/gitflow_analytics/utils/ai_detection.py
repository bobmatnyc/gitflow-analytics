"""AI tool detection from commit messages and co-author trailers.

Phase 1: regex pattern matching + heuristic NLP scoring (no external LLM calls).
Phase 2 (future): optional LLM confidence scoring.
"""

import re
from typing import Optional

AI_PATTERNS: dict[str, list[str]] = {
    "claude_code": [
        r"co-authored-by:.*noreply@anthropic\.com",
        r"🤖 generated with \[claude code\]",
        r"co-authored-by:.*claude.*<noreply@anthropic\.com>",
    ],
    "copilot": [
        r"co-authored-by:.*copilot\[bot\]",
        r"co-authored-by:.*175728472\+copilot@users\.noreply\.github\.com",
        r"co-authored-by:.*copilot@github\.com",
        r"suggestion from @?copilot",
    ],
    "cursor": [
        r"co-authored-by:.*cursor",
        r"generated with cursor",
        r"cursor ai",
    ],
}

# Patterns that indicate full generation rather than assistance
_GENERATED_PATTERNS: list[str] = [
    r"🤖 generated with",
    r"generated with \[claude code\]",
]


def detect_ai_tool(commit_message: str) -> Optional[str]:
    """Detect AI tool from commit message.

    Scans co-author trailers and commit message body for known AI tool
    signatures.  Returns the tool name, 'mixed' when multiple tools are
    detected, or None when no AI markers are found.

    Args:
        commit_message: Full commit message including trailers.

    Returns:
        One of 'claude_code', 'copilot', 'cursor', 'mixed', or None.
    """
    msg_lower = commit_message.lower()
    detected: list[str] = []
    for tool, patterns in AI_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, msg_lower):
                detected.append(tool)
                break  # Only count each tool once

    if not detected:
        return None
    if len(detected) == 1:
        return detected[0]
    return "mixed"


def is_ai_assisted(commit_message: str) -> bool:
    """Return True if commit contains any AI tool marker.

    Args:
        commit_message: Full commit message including trailers.

    Returns:
        True when at least one AI tool signature is detected.
    """
    return detect_ai_tool(commit_message) is not None


# ---------------------------------------------------------------------------
# NLP heuristic scoring (Phase 1)
# ---------------------------------------------------------------------------

# Phrases that AI assistants tend to overuse in commit messages.
_AI_PHRASE_PATTERNS: list[str] = [
    r"\bas requested\b",
    r"\bper your (request|instructions?)\b",
    r"\bbased on (your|the) (requirements?|feedback|instructions?)\b",
    r"\bimplemented? (as|per) (discussed|requested|specified)\b",
    r"\bensure[sd]? (that|proper|correct)\b",
    r"\bhandle[sd]? (edge cases?|error(s)?)\b",
    r"\badd(ed|s)? (proper|appropriate|necessary) (error handling|validation|logging)\b",
    r"\brefactored? (to|for) (improve|better|cleaner|clearer)\b",
    r"\bupdate[sd]? (to use|to support|to handle)\b",
    r"\bfix(es|ed)? (an? )?(issue|bug|problem) (where|when|with)\b",
]

# Formulaic opening structures common in AI-generated commit messages.
# These are intentionally specific to avoid false positives on short plain commits
# like "fix typo in README".
_FORMULAIC_PATTERNS: list[str] = [
    # "add X to improve/to support/to handle/for better/for cleaner" — AI verbosity
    r"^(add|implement|update|fix|refactor|improve)\w*\s+\w+(\s+\w+)*\s+(to improve|to support|to handle|for better|for improved|for cleaner)\b",
    r"^(this (commit|change|pr|update) (adds|implements|fixes|updates|improves))\b",
]


def score_ai_confidence(message: str) -> float:
    """Heuristic confidence score that a commit message was AI-generated.

    Returns float in [0.0, 1.0]:
    - 0.0: No AI markers detected
    - 0.3-0.6: Some AI-typical phrasing present
    - 0.7+: Strong AI markers (pattern match OR multiple phrase matches)
    - 1.0: Definitive AI marker (co-author trailer detected)

    This is Phase 1 (heuristic only).  Phase 2 would add optional LLM scoring.

    Args:
        message: Full commit message including trailers.

    Returns:
        Confidence score in [0.0, 1.0] rounded to 3 decimal places.
    """
    if not message:
        return 0.0

    msg_lower = message.lower().strip()

    # Definitive: existing regex pattern match -> 1.0
    if detect_ai_tool(message) is not None:
        return 1.0

    score = 0.0

    # AI phrase matches — each adds 0.15, capped at 0.6
    phrase_hits = sum(1 for p in _AI_PHRASE_PATTERNS if re.search(p, msg_lower))
    score += min(phrase_hits * 0.15, 0.6)

    # Formulaic opening — adds 0.2
    for p in _FORMULAIC_PATTERNS:
        if re.search(p, msg_lower):
            score += 0.2
            break

    # Very short commit messages are unlikely to be AI (AI tends to be verbose)
    words = len(msg_lower.split())
    if words < 3:
        score *= 0.5

    return min(round(score, 3), 1.0)


def detect_ai_detection_method(message: str) -> str:
    """Return detection method used for a commit message.

    Args:
        message: Full commit message including trailers.

    Returns:
        One of 'pattern', 'nlp_heuristic', or 'none'.
        - 'pattern': a known AI co-author trailer or marker was found
        - 'nlp_heuristic': no pattern match but NLP score is > 0
        - 'none': no AI signals detected
    """
    if detect_ai_tool(message) is not None:
        return "pattern"
    if score_ai_confidence(message) > 0.0:
        return "nlp_heuristic"
    return "none"


def is_ai_generated(commit_message: str) -> bool:
    """Return True if commit appears fully AI-generated (not just assisted).

    Heuristic: presence of 'generated with' or the Anthropic noreply trailer
    indicates the commit was produced wholesale by an AI tool rather than
    being an AI-assisted human commit.

    Args:
        commit_message: Full commit message including trailers.

    Returns:
        True when a generation (not just assistance) marker is detected.
    """
    msg_lower = commit_message.lower()
    return any(re.search(p, msg_lower) for p in _GENERATED_PATTERNS)
