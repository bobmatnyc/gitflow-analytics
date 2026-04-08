"""AI tool detection from commit messages and co-author trailers."""

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
