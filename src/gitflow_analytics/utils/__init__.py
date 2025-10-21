"""Utility modules for GitFlow Analytics."""

from .commit_utils import get_parent_count, is_initial_commit, is_merge_commit

__all__ = ["is_merge_commit", "get_parent_count", "is_initial_commit"]
