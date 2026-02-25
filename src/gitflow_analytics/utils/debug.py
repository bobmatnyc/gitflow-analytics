"""Debug mode utilities.

Centralises the GITFLOW_DEBUG environment variable check so that every
module uses the same authoritative helper instead of duplicating the
os.getenv call and truthy-value comparison.
"""

import os


def is_debug_mode() -> bool:
    """Return True when GITFLOW_DEBUG is set to a truthy value.

    Truthy values: "1", "true", "yes" (case-insensitive).
    All other values, including an unset variable, return False.

    Returns:
        True if debug mode is active, False otherwise.
    """
    return os.getenv("GITFLOW_DEBUG", "").lower() in ("1", "true", "yes")
