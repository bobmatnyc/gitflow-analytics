"""Git operation wrapper with timeout protection.

This module provides timeout-protected git operations to prevent hanging
when repositories require authentication or have network issues.
"""

import logging
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional, TypeVar

from ..constants import Timeouts

logger = logging.getLogger(__name__)

T = TypeVar("T")


class GitOperationTimeout(Exception):
    """Raised when a git operation exceeds its timeout."""

    pass


class GitTimeoutWrapper:
    """Wrapper for git operations with timeout protection."""

    def __init__(self, default_timeout: int = Timeouts.DEFAULT_GIT_OPERATION):
        """Initialize the git timeout wrapper.

        Args:
            default_timeout: Default timeout in seconds for git operations
        """
        self.default_timeout = default_timeout
        self._operation_stack = []  # Track nested operations for debugging

    @contextmanager
    def operation_tracker(self, operation_name: str, repo_path: Optional[Path] = None):
        """Track the current git operation for debugging and heartbeat logging.

        Args:
            operation_name: Name of the operation being performed
            repo_path: Optional repository path for context
        """
        operation_info = {
            "name": operation_name,
            "repo_path": str(repo_path) if repo_path else None,
            "start_time": time.time(),
            "thread_id": threading.current_thread().ident,
        }

        self._operation_stack.append(operation_info)
        logger.info(
            f"🚀 Starting operation: {operation_name} {f'for {repo_path}' if repo_path else ''}"
        )

        try:
            yield operation_info
        finally:
            if self._operation_stack and self._operation_stack[-1] == operation_info:
                self._operation_stack.pop()
                elapsed = time.time() - operation_info["start_time"]
                logger.info(f"✅ Completed operation: {operation_name} in {elapsed:.1f}s")

    def get_current_operation(self) -> Optional[dict]:
        """Get the currently running operation for this thread."""
        thread_id = threading.current_thread().ident
        for op in reversed(self._operation_stack):
            if op.get("thread_id") == thread_id:
                return op
        return None

    def run_with_timeout(
        self,
        func: Callable[..., T],
        args: tuple = (),
        kwargs: dict = None,
        timeout: Optional[int] = None,
        operation_name: str = "git_operation",
    ) -> T:
        """Run a function with timeout protection using threading.

        Args:
            func: Function to run
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            timeout: Timeout in seconds (uses default if not specified)
            operation_name: Name of the operation for logging

        Returns:
            The result of the function

        Raises:
            GitOperationTimeout: If the operation times out
        """
        timeout = timeout or self.default_timeout
        kwargs = kwargs or {}
        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            # Thread is still running after timeout
            logger.error(f"⏱️ Operation '{operation_name}' timed out after {timeout}s")
            # Note: We can't actually kill the thread in Python, it will continue running
            # but we'll raise an exception to prevent waiting for it
            raise GitOperationTimeout(f"Operation '{operation_name}' timed out after {timeout}s")

        if exception[0]:
            raise exception[0]

        return result[0]

    def run_git_command(
        self,
        cmd: list[str],
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
        capture_output: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a git command with timeout protection.

        Args:
            cmd: Command to run as list of strings
            cwd: Working directory for the command
            timeout: Timeout in seconds (uses default if not specified)
            capture_output: Whether to capture stdout/stderr
            check: Whether to raise exception on non-zero return code

        Returns:
            CompletedProcess instance with the result

        Raises:
            GitOperationTimeout: If the command times out
            subprocess.CalledProcessError: If check=True and command fails
        """
        timeout = timeout or self.default_timeout

        # Set environment to prevent authentication prompts
        env = os.environ.copy()
        env.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_ASKPASS": "/bin/echo",
                "SSH_ASKPASS": "/bin/echo",
                "GCM_INTERACTIVE": "never",
                "GIT_SSH_COMMAND": "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o PasswordAuthentication=no",
                "DISPLAY": "",
                "GIT_CREDENTIAL_HELPER": "",
                "GCM_PROVIDER": "none",
            }
        )

        operation_name = " ".join(cmd[:2])  # e.g., "git fetch"

        with self.operation_tracker(operation_name, cwd):
            try:
                result = subprocess.run(
                    cmd,
                    cwd=cwd,
                    env=env,
                    capture_output=capture_output,
                    text=True,
                    timeout=timeout,
                    check=check,
                )
                return result

            except subprocess.TimeoutExpired as e:
                logger.error(f"⏱️ Git command timed out after {timeout}s: {' '.join(cmd)}")
                raise GitOperationTimeout(
                    f"Git command '{operation_name}' timed out after {timeout}s"
                ) from e

            except subprocess.CalledProcessError as e:
                # Check if it's an authentication error
                error_str = (e.stderr or "").lower()
                if any(
                    x in error_str
                    for x in ["authentication", "permission denied", "401", "403", "password"]
                ):
                    logger.error(f"🔐 Authentication failed for git command: {operation_name}")
                    logger.error(f"   Error details: {e.stderr}")
                raise

    def fetch_with_timeout(self, repo_path: Path, timeout: int = Timeouts.GIT_FETCH) -> bool:
        """Fetch from remote with timeout protection.

        Args:
            repo_path: Path to the repository
            timeout: Timeout in seconds

        Returns:
            True if fetch succeeded, False otherwise
        """
        try:
            self.run_git_command(
                ["git", "fetch", "--all"], cwd=repo_path, timeout=timeout, check=True
            )
            logger.info(f"✅ Fetch succeeded for {repo_path.name}")
            return True
        except (GitOperationTimeout, subprocess.CalledProcessError) as e:
            # Extract detailed error information
            error_detail = ""
            if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                error_detail = e.stderr.strip()
                # Check for authentication-specific errors
                if (
                    "could not read Username" in error_detail
                    or "could not read Password" in error_detail
                ):
                    logger.error(
                        f"🔐 Authentication required for {repo_path.name}. "
                        f"Repository uses HTTPS but no credentials configured. "
                        f"Consider: (1) Configure git credential helper, "
                        f"(2) Use SSH URLs instead, or (3) Set GITHUB_TOKEN in environment."
                    )
                elif "Authentication failed" in error_detail or "403" in error_detail:
                    logger.error(
                        f"🔐 Authentication failed for {repo_path.name}. "
                        f"Check that your GitHub token has proper permissions."
                    )
                else:
                    logger.warning(f"Git fetch failed for {repo_path.name}: {error_detail}")
            else:
                logger.warning(f"Git fetch failed for {repo_path.name}: {e}")
            return False

    def pull_with_timeout(self, repo_path: Path, timeout: int = Timeouts.GIT_PULL) -> bool:
        """Pull from remote with timeout protection.

        Args:
            repo_path: Path to the repository
            timeout: Timeout in seconds

        Returns:
            True if pull succeeded, False otherwise
        """
        try:
            self.run_git_command(["git", "pull"], cwd=repo_path, timeout=timeout, check=True)
            return True
        except (GitOperationTimeout, subprocess.CalledProcessError) as e:
            logger.warning(f"Git pull failed for {repo_path}: {e}")
            return False

    def clone_with_timeout(
        self, clone_url: str, target_path: Path, branch: Optional[str] = None, timeout: int = 60
    ) -> bool:
        """Clone a repository with timeout protection.

        Args:
            clone_url: URL of the repository to clone
            target_path: Target path for the cloned repository
            branch: Optional branch to checkout
            timeout: Timeout in seconds (default 60s for cloning)

        Returns:
            True if clone succeeded, False otherwise
        """
        cmd = ["git", "clone", "--config", "credential.helper="]
        if branch:
            cmd.extend(["-b", branch])
        cmd.extend([clone_url, str(target_path)])

        try:
            self.run_git_command(cmd, timeout=timeout, check=True)
            return True
        except (GitOperationTimeout, subprocess.CalledProcessError) as e:
            logger.warning(f"Git clone failed for {clone_url}: {e}")
            return False


class HeartbeatLogger:
    """Provides heartbeat logging for long-running operations."""

    def __init__(self, interval: int = 5):
        """Initialize heartbeat logger.

        Args:
            interval: Interval in seconds between heartbeat logs
        """
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self._wrapper = GitTimeoutWrapper()

    def start(self):
        """Start the heartbeat logging thread."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the heartbeat logging thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=Timeouts.THREAD_JOIN)

    def _heartbeat_loop(self):
        """Main heartbeat loop that logs current operations."""
        last_log_time = 0

        while not self._stop_event.is_set():
            current_time = time.time()

            if current_time - last_log_time >= self.interval:
                operation = self._wrapper.get_current_operation()
                if operation:
                    elapsed = current_time - operation["start_time"]
                    repo_info = f"for {operation['repo_path']} " if operation["repo_path"] else ""
                    logger.info(
                        f"💓 Heartbeat: Still running '{operation['name']}' "
                        f"{repo_info}"
                        f"(elapsed: {elapsed:.1f}s)"
                    )
                last_log_time = current_time

            # Sleep in small increments to be responsive to stop event
            self._stop_event.wait(0.5)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Global instance for convenience
git_wrapper = GitTimeoutWrapper()
