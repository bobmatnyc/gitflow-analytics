"""Git repository cloning utilities for GitFlow Analytics.

Provides retry-capable, timeout-aware git clone helpers used by the CLI when
it needs to clone GitHub repositories on-demand (e.g. during organisation
discovery or when a locally-configured path does not yet exist).
"""

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# Authentication-related error strings that should NOT be retried.
_AUTH_ERROR_TOKENS = ("authentication", "permission denied", "401", "403")


@dataclass
class CloneResult:
    """Outcome of a clone operation."""

    success: bool
    """True when the repository was cloned (or already existed)."""

    elapsed_seconds: float = 0.0
    """Wall-clock time taken for a successful clone."""

    error: Optional[str] = None
    """Human-readable error description when *success* is False."""

    auth_failure: bool = False
    """True when failure was caused by an authentication error (no retry)."""


def _build_clone_env() -> dict[str, str]:
    """Build a subprocess environment that disables interactive git prompts."""
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS"] = ""
    env["GCM_INTERACTIVE"] = "never"
    env["GIT_PROGRESS"] = "1"  # Force progress output on stderr
    return env


def _build_clone_url(github_repo: str, token: Optional[str]) -> str:
    """Construct a GitHub HTTPS clone URL, embedding a token when provided.

    Args:
        github_repo: GitHub ``owner/repo`` slug.
        token: Optional personal access token for authentication.

    Returns:
        A full HTTPS clone URL.
    """
    base = f"https://github.com/{github_repo}.git"
    if token:
        return f"https://{token}@github.com/{github_repo}.git"
    return base


def clone_repository(
    repo_path: Path,
    github_repo: str,
    token: Optional[str] = None,
    branch: Optional[str] = None,
    timeout_seconds: int = 300,
    max_retries: int = 2,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> CloneResult:
    """Clone a GitHub repository with retry and timeout handling.

    Implements the retry-with-fallback pattern previously duplicated in two
    places in ``cli.py``:

    1. Tries to clone the repository with the requested *branch* (if given).
    2. On ``TimeoutExpired``: cleans up the partial clone and retries up to
       *max_retries* times before giving up.
    3. On authentication failure: returns immediately without retrying.
    4. On ``GitCommandError`` for a missing branch: falls back to cloning
       without a branch specification.

    Args:
        repo_path: Destination path for the cloned repository.
        github_repo: GitHub ``owner/repo`` slug (e.g. ``"acme/myrepo"``).
        token: Optional GitHub personal access token for private repos.
        branch: Optional specific branch to clone.  Clones the default
                branch when *None* or when the specified branch is not found.
        timeout_seconds: Per-attempt timeout in seconds (default 300 s / 5 min).
        max_retries: Maximum number of retry attempts after a transient failure.
                     Authentication failures are never retried.
        progress_callback: Optional callable that receives human-readable
                           status strings.  Used by the CLI to relay messages
                           through the display layer.

    Returns:
        A :class:`CloneResult` describing whether the clone succeeded.
    """
    clone_url = _build_clone_url(github_repo, token)
    env = _build_clone_env()

    def _notify(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            logger.info(msg)

    retry_count = 0
    clone_success = False

    while retry_count <= max_retries and not clone_success:
        if retry_count > 0:
            _notify(f"Retry {retry_count}/{max_retries}: {github_repo}")
        else:
            _notify(f"Cloning {github_repo} from GitHub...")

        try:
            # Ensure the parent directory exists before cloning
            repo_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                "git",
                "clone",
                "--progress",
                "--config",
                "credential.helper=",
            ]
            if branch:
                cmd.extend(["-b", branch])
            cmd.extend([clone_url, str(repo_path)])

            start_time = time.time()
            result = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=None,  # Let stderr (git progress) flow to terminal
                text=True,
                timeout=timeout_seconds,
            )
            elapsed = time.time() - start_time

            if result.returncode != 0:
                # Distinguish auth failures from other errors
                stderr_out = (result.stderr or "").lower()
                if any(tok in stderr_out for tok in _AUTH_ERROR_TOKENS):
                    _notify(f"Authentication failed for {github_repo}")
                    return CloneResult(
                        success=False,
                        error=f"Authentication failed for {github_repo}",
                        auth_failure=True,
                    )
                # Raise so the outer except catches and retries
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )

            clone_success = True
            _notify(f"Cloned {github_repo} ({elapsed:.1f}s)")
            return CloneResult(success=True, elapsed_seconds=elapsed)

        except subprocess.TimeoutExpired:
            retry_count += 1
            _notify(f"Clone timeout ({timeout_seconds}s): {github_repo}")
            # Remove any partial clone directory before retrying
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
            if retry_count > max_retries:
                _notify(f"Skipping {github_repo} after {max_retries} timeouts")
                return CloneResult(
                    success=False,
                    error=f"Timeout after {max_retries} retries",
                )
            continue  # retry

        except Exception as exc:
            retry_count += 1
            err_str = str(exc)
            _notify(f"Clone error: {exc}")

            # Authentication failures: do not retry
            if any(tok in err_str.lower() for tok in _AUTH_ERROR_TOKENS):
                return CloneResult(
                    success=False,
                    error=f"Authentication error: {exc}",
                    auth_failure=True,
                )

            # Missing branch: retry without branch specification
            if branch and "Remote branch" in err_str and "not found" in err_str:
                _notify(f"Branch '{branch}' not found, using repository default")
                cmd_no_branch = [
                    "git",
                    "clone",
                    "--progress",
                    "--config",
                    "credential.helper=",
                    clone_url,
                    str(repo_path),
                ]
                try:
                    start_time = time.time()
                    result = subprocess.run(
                        cmd_no_branch,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=None,
                        text=True,
                        timeout=timeout_seconds,
                    )
                    elapsed = time.time() - start_time
                    if result.returncode == 0:
                        clone_success = True
                        _notify(f"Cloned {github_repo} (default branch, {elapsed:.1f}s)")
                        return CloneResult(success=True, elapsed_seconds=elapsed)
                    else:
                        return CloneResult(
                            success=False,
                            error=f"Clone failed (default branch): returncode={result.returncode}",
                        )
                except Exception as inner_exc:
                    return CloneResult(
                        success=False,
                        error=f"Clone failed without branch: {inner_exc}",
                    )

            if retry_count > max_retries:
                return CloneResult(success=False, error=f"Clone failed after retries: {exc}")
            continue  # retry

    # Exhausted retries without success
    return CloneResult(success=False, error=f"Clone failed for {github_repo}")
