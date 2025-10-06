"""Application-wide constants and configuration values.

This module centralizes magic numbers and configuration defaults to improve
code maintainability and readability. Constants are organized by functional
area for easy navigation and updates.
"""


class Timeouts:
    """Timeout values in seconds for various git operations.

    These timeouts protect against hanging operations when repositories
    require authentication or have network issues.
    """

    # Git remote operations
    GIT_FETCH = 30  # Fetch from remote repository
    GIT_PULL = 30  # Pull latest changes

    # Git local operations
    GIT_BRANCH_ITERATION = 15  # Iterate commits for a branch/day
    GIT_DIFF = 10  # Calculate diff statistics
    GIT_CONFIG = 2  # Read git configuration
    GIT_REMOTE_LIST = 5  # List remote branches

    # Default timeout for generic git operations
    DEFAULT_GIT_OPERATION = 30

    # Process-level timeouts
    SUBPROCESS_DEFAULT = 5  # Default subprocess timeout
    THREAD_JOIN = 1  # Thread join timeout


class BatchSizes:
    """Batch processing sizes for efficient data handling.

    These sizes balance memory usage with performance gains from bulk operations.
    Tunable based on repository size and system capabilities.
    """

    COMMIT_STORAGE = 1000  # Commits per bulk insert operation
    TICKET_FETCH = 50  # Tickets fetched per JIRA batch
    CACHE_WARMUP = 100  # Commits per cache warmup batch

    # Estimation constants
    COMMITS_PER_WEEK_ESTIMATE = 50  # Estimated commits for progress tracking
    DEFAULT_PROGRESS_ESTIMATE = 100  # Default when estimation fails


class CacheTTL:
    """Cache time-to-live values.

    These values control how long cached data remains valid before
    requiring refresh. Measured in hours unless otherwise specified.
    """

    ONE_WEEK_HOURS = 168  # Standard cache TTL (7 days * 24 hours)
    IDENTITY_CACHE_DAYS = 7  # Developer identity analysis cache (in days)


class Thresholds:
    """Various threshold values for analysis and reporting."""

    # Cache performance
    CACHE_HIT_RATE_GOOD = 50  # Percentage threshold for good cache performance

    # Percentage calculations
    PERCENTAGE_MULTIPLIER = 100  # Standard percentage calculation multiplier


class Estimations:
    """Estimation constants for progress tracking and metrics."""

    COMMITS_PER_WEEK = 50  # Estimated commits per week for progress bars
    DEFAULT_ESTIMATE = 100  # Default estimate when actual count unavailable
