"""Shared CLI utility functions for GitFlow Analytics.

Contains infrastructure helpers used by multiple CLI commands that do not
belong to a specific domain (formatting, date math, or core analysis).
"""

import logging
import sys


def setup_logging(log: str, module_name: str = __name__) -> logging.Logger:
    """Configure logging for a CLI command based on the --log option value.

    This consolidates the duplicated logging-setup block that previously
    appeared verbatim in every CLI command (analyze, fetch, collect,
    classify, report, train).

    Args:
        log: Value of the --log CLI option.  One of "none", "INFO", "DEBUG"
             (case-insensitive).
        module_name: The ``__name__`` of the calling module, used to create
                     a properly-named logger.  Pass ``__name__`` from the
                     call site so the logger reflects the actual module.

    Returns:
        A configured :class:`logging.Logger` for the calling module.

    Example::

        # At the top of a CLI command function:
        logger = setup_logging(log, __name__)
    """
    if log.upper() != "NONE":
        # Configure structured logging with a detailed formatter
        log_level = getattr(logging, log.upper())
        logging.basicConfig(
            level=log_level,
            format="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            handlers=[logging.StreamHandler(sys.stderr)],
            force=True,  # Ensure reconfiguration of existing loggers
        )

        # Propagate the level to the root gitflow_analytics namespace so all
        # sub-module loggers pick it up without extra configuration.
        logging.getLogger("gitflow_analytics").setLevel(log_level)

        module_logger = logging.getLogger(module_name)
        module_logger.info("Logging enabled at %s level", log.upper())
        module_logger.debug("Logging configuration applied to all gitflow_analytics modules")
    else:
        # Silence all logging — keep the CLI output clean by default
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger("gitflow_analytics").setLevel(logging.CRITICAL)
        module_logger = logging.getLogger(module_name)

    return module_logger
