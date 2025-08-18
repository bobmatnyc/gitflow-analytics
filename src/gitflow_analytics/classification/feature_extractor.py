"""Feature extraction for commit classification.

This module extracts 68-dimensional feature vectors from git commits for machine learning
classification. Features include keyword analysis, file patterns, commit statistics,
temporal patterns, and author information.

The feature vector is designed to capture comprehensive information about commits
while maintaining computational efficiency and interpretability.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from .linguist_analyzer import LinguistAnalyzer

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts 68-dimensional feature vectors from git commits.

    The feature extraction process creates a comprehensive representation of each commit
    by analyzing multiple aspects:

    - Keyword features (20 dimensions): Semantic analysis of commit messages
    - File-based features (20 dimensions): Programming languages and activities
    - Commit statistics (15 dimensions): Size, complexity, and change metrics
    - Temporal features (8 dimensions): Time-based patterns and trends
    - Author features (5 dimensions): Developer behavior and collaboration patterns

    This design balances comprehensiveness with computational efficiency, allowing
    for accurate classification while maintaining fast processing speeds.
    """

    def __init__(self):
        """Initialize the feature extractor with analysis components."""
        self.linguist = LinguistAnalyzer()

        # Keyword categories for semantic analysis (20 dimensions)
        self.keyword_categories = {
            "feature_keywords": [
                "add",
                "implement",
                "create",
                "build",
                "introduce",
                "develop",
                "feature",
                "new",
                "functionality",
                "capability",
                "enhancement",
            ],
            "bugfix_keywords": [
                "fix",
                "bug",
                "issue",
                "resolve",
                "correct",
                "repair",
                "patch",
                "error",
                "problem",
                "defect",
                "broken",
                "wrong",
                "crash",
            ],
            "refactor_keywords": [
                "refactor",
                "restructure",
                "cleanup",
                "optimize",
                "improve",
                "simplify",
                "reorganize",
                "consolidate",
                "streamline",
            ],
            "docs_keywords": [
                "doc",
                "docs",
                "documentation",
                "readme",
                "comment",
                "explain",
                "guide",
                "tutorial",
                "example",
                "specification",
                "manual",
            ],
            "test_keywords": [
                "test",
                "testing",
                "spec",
                "unit",
                "integration",
                "e2e",
                "coverage",
                "mock",
                "stub",
                "fixture",
                "assert",
            ],
            "config_keywords": [
                "config",
                "configuration",
                "setting",
                "environment",
                "setup",
                "property",
                "parameter",
                "option",
                "flag",
                "variable",
            ],
            "security_keywords": [
                "security",
                "secure",
                "auth",
                "authentication",
                "authorization",
                "permission",
                "vulnerability",
                "exploit",
                "sanitize",
                "validate",
            ],
            "performance_keywords": [
                "performance",
                "optimize",
                "fast",
                "slow",
                "cache",
                "memory",
                "cpu",
                "speed",
                "efficient",
                "bottleneck",
                "profile",
            ],
            "ui_keywords": [
                "ui",
                "interface",
                "frontend",
                "design",
                "layout",
                "style",
                "component",
                "widget",
                "view",
                "screen",
                "page",
            ],
            "api_keywords": [
                "api",
                "endpoint",
                "service",
                "backend",
                "server",
                "client",
                "request",
                "response",
                "http",
                "rest",
                "graphql",
            ],
            "database_keywords": [
                "database",
                "db",
                "sql",
                "query",
                "table",
                "schema",
                "migration",
                "model",
                "data",
                "repository",
            ],
            "deployment_keywords": [
                "deploy",
                "deployment",
                "release",
                "build",
                "ci",
                "cd",
                "docker",
                "kubernetes",
                "infrastructure",
                "production",
            ],
            "dependency_keywords": [
                "dependency",
                "package",
                "library",
                "module",
                "import",
                "require",
                "install",
                "update",
                "upgrade",
                "version",
            ],
            "maintenance_keywords": [
                "maintenance",
                "cleanup",
                "housekeeping",
                "chore",
                "routine",
                "update",
                "bump",
                "remove",
                "delete",
                "deprecated",
            ],
            "hotfix_keywords": [
                "hotfix",
                "urgent",
                "critical",
                "emergency",
                "immediate",
                "asap",
                "production",
                "live",
                "quick",
                "temporary",
            ],
            "merge_keywords": [
                "merge",
                "cherry-pick",
                "rebase",
                "conflict",
                "branch",
                "pull",
                "request",
                "pr",
                "integration",
                "combine",
            ],
            "revert_keywords": [
                "revert",
                "rollback",
                "undo",
                "back",
                "restore",
                "reset",
                "previous",
                "original",
                "cancel",
                "abort",
            ],
            "wip_keywords": [
                "wip",
                "progress",
                "partial",
                "incomplete",
                "draft",
                "temporary",
                "placeholder",
                "todo",
                "fixme",
                "hack",
            ],
            "breaking_keywords": [
                "breaking",
                "break",
                "incompatible",
                "major",
                "change",
                "migration",
                "upgrade",
                "deprecated",
                "removed",
                "api",
            ],
            "experimental_keywords": [
                "experimental",
                "prototype",
                "poc",
                "spike",
                "trial",
                "test",
                "experiment",
                "explore",
                "research",
                "investigate",
            ],
        }

        # Compile regex patterns for efficiency
        self._compile_keyword_patterns()

    def _compile_keyword_patterns(self) -> None:
        """Compile keyword patterns for efficient matching."""
        self.compiled_keyword_patterns = {}
        for category, keywords in self.keyword_categories.items():
            # Create word boundary patterns for precise matching
            patterns = [rf"\b{re.escape(keyword)}\b" for keyword in keywords]
            combined_pattern = "|".join(patterns)
            self.compiled_keyword_patterns[category] = re.compile(combined_pattern, re.IGNORECASE)

    def extract_features(
        self, commit_data: dict[str, Any], author_stats: Optional[dict[str, Any]] = None
    ) -> np.ndarray:
        """Extract 68-dimensional feature vector from commit data.

        Args:
            commit_data: Dictionary containing commit information:
                - hash: Commit hash
                - message: Commit message
                - author_name: Author name
                - author_email: Author email
                - timestamp: Commit timestamp (datetime)
                - files_changed: List of changed file paths
                - insertions: Number of lines added
                - deletions: Number of lines deleted
            author_stats: Optional dictionary with author statistics:
                - total_commits: Total commits by this author
                - avg_commit_size: Average commit size for this author
                - languages_used: Set of languages this author typically uses

        Returns:
            68-dimensional numpy array with extracted features
        """
        features = np.zeros(68, dtype=np.float32)

        # Extract different feature categories
        keyword_features = self._extract_keyword_features(commit_data["message"])

        # Handle files_changed being either a list or an integer
        files_changed = commit_data.get("files_changed", [])
        if isinstance(files_changed, int):
            # If it's an integer, we can't extract file features, use empty list
            files_changed = []

        file_features = self._extract_file_features(files_changed)
        stats_features = self._extract_stats_features(commit_data)
        temporal_features = self._extract_temporal_features(commit_data["timestamp"])
        author_features = self._extract_author_features(commit_data, author_stats)

        # Combine all features into single vector
        idx = 0

        # Keyword features (20 dimensions)
        features[idx : idx + 20] = keyword_features
        idx += 20

        # File-based features (20 dimensions)
        features[idx : idx + 20] = file_features
        idx += 20

        # Commit statistics (15 dimensions)
        features[idx : idx + 15] = stats_features
        idx += 15

        # Temporal features (8 dimensions)
        features[idx : idx + 8] = temporal_features
        idx += 8

        # Author features (5 dimensions)
        features[idx : idx + 5] = author_features

        return features

    def _extract_keyword_features(self, message: str) -> np.ndarray:
        """Extract keyword-based features from commit message.

        Args:
            message: Commit message text

        Returns:
            20-dimensional array with keyword features
        """
        features = np.zeros(20, dtype=np.float32)

        if not message:
            return features

        # Normalize message for consistent analysis
        normalized_message = message.lower().strip()
        message_length = len(normalized_message.split())

        # Extract features for each keyword category
        for i, (_category, pattern) in enumerate(self.compiled_keyword_patterns.items()):
            matches = pattern.findall(normalized_message)
            match_count = len(matches)

            # Normalize by message length to handle varying message sizes
            if message_length > 0:
                features[i] = min(1.0, match_count / message_length)
            else:
                features[i] = 0.0

        return features

    def _extract_file_features(self, file_paths: list[str]) -> np.ndarray:
        """Extract file-based features using linguist analysis.

        Args:
            file_paths: List of changed file paths

        Returns:
            20-dimensional array with file-based features
        """
        features = np.zeros(20, dtype=np.float32)

        if not file_paths:
            return features

        # Get linguist analysis
        analysis = self.linguist.analyze_commit_files(file_paths)

        # Feature 0-4: Language distribution (top 5 languages)
        top_languages = analysis["languages"].most_common(5)
        for i, (_lang, count) in enumerate(top_languages):
            features[i] = count / analysis["file_count"]

        # Feature 5-9: Activity distribution (top 5 activities)
        top_activities = analysis["activities"].most_common(5)
        for i, (_activity, count) in enumerate(top_activities):
            features[5 + i] = count / len(file_paths)  # Activities can overlap

        # Feature 10: Language diversity (normalized)
        features[10] = min(1.0, analysis["language_diversity"] / 5.0)

        # Feature 11: Activity diversity (normalized)
        features[11] = min(1.0, analysis["activity_diversity"] / 5.0)

        # Feature 12: Generated file ratio
        features[12] = analysis["generated_ratio"]

        # Feature 13: Is multilingual
        features[13] = 1.0 if analysis["is_multilingual"] else 0.0

        # Feature 14: Is cross-functional
        features[14] = 1.0 if analysis["is_cross_functional"] else 0.0

        # Feature 15-19: File type patterns
        common_extensions = [".py", ".js", ".java", ".go", ".sql"]
        for i, ext in enumerate(common_extensions):
            if ext in analysis["file_types"]:
                features[15 + i] = analysis["file_types"][ext] / analysis["file_count"]

        return features

    def _extract_stats_features(self, commit_data: dict[str, Any]) -> np.ndarray:
        """Extract statistical features from commit data.

        Args:
            commit_data: Commit data dictionary

        Returns:
            15-dimensional array with statistical features
        """
        features = np.zeros(15, dtype=np.float32)

        files_changed = len(commit_data.get("files_changed", []))
        insertions = commit_data.get("insertions", 0)
        deletions = commit_data.get("deletions", 0)
        message = commit_data.get("message", "")

        # Feature 0: Number of files changed (log-scaled)
        features[0] = min(1.0, np.log1p(files_changed) / np.log1p(100))

        # Feature 1: Lines inserted (log-scaled)
        features[1] = min(1.0, np.log1p(insertions) / np.log1p(1000))

        # Feature 2: Lines deleted (log-scaled)
        features[2] = min(1.0, np.log1p(deletions) / np.log1p(1000))

        # Feature 3: Total lines changed (log-scaled)
        total_lines = insertions + deletions
        features[3] = min(1.0, np.log1p(total_lines) / np.log1p(2000))

        # Feature 4: Insert/delete ratio
        if total_lines > 0:
            features[4] = insertions / total_lines

        # Feature 5: Commit message length (normalized)
        features[5] = min(1.0, len(message) / 200.0)

        # Feature 6: Message word count (normalized)
        word_count = len(message.split())
        features[6] = min(1.0, word_count / 50.0)

        # Feature 7: Message lines count (normalized)
        line_count = len(message.split("\n"))
        features[7] = min(1.0, line_count / 10.0)

        # Feature 8: Average lines per file
        if files_changed > 0:
            features[8] = min(1.0, total_lines / files_changed / 100.0)

        # Feature 9: Has conventional commit format
        conventional_pattern = (
            r"^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\\(.+\\))?: .+"
        )
        features[9] = 1.0 if re.match(conventional_pattern, message.strip()) else 0.0

        # Feature 10: Contains ticket reference
        ticket_pattern = r"(#\\d+|[A-Z]+-\\d+|JIRA-\\d+|CU-\\d+)"
        features[10] = 1.0 if re.search(ticket_pattern, message) else 0.0

        # Feature 11: Is merge commit
        features[11] = 1.0 if message.lower().startswith("merge") else 0.0

        # Feature 12: Contains code in message (backticks or brackets)
        code_pattern = r"(`[^`]+`|\[[^\]]+\]|\{[^}]+\})"
        features[12] = 1.0 if re.search(code_pattern, message) else 0.0

        # Feature 13: Message complexity (punctuation diversity)
        punctuation = set(char for char in message if not char.isalnum() and not char.isspace())
        features[13] = min(1.0, len(punctuation) / 10.0)

        # Feature 14: Large commit indicator
        is_large = files_changed > 10 or total_lines > 500
        features[14] = 1.0 if is_large else 0.0

        return features

    def _extract_temporal_features(self, timestamp: datetime) -> np.ndarray:
        """Extract temporal features from commit timestamp.

        Args:
            timestamp: Commit timestamp

        Returns:
            8-dimensional array with temporal features
        """
        features = np.zeros(8, dtype=np.float32)

        if not timestamp:
            return features

        # Ensure timezone awareness
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Feature 0: Hour of day (normalized)
        features[0] = timestamp.hour / 24.0

        # Feature 1: Day of week (0=Monday, 6=Sunday, normalized)
        features[1] = timestamp.weekday() / 6.0

        # Feature 2: Day of month (normalized)
        features[2] = (timestamp.day - 1) / 30.0  # 0-based, max ~30 days

        # Feature 3: Month of year (normalized)
        features[3] = (timestamp.month - 1) / 11.0  # 0-based, 12 months

        # Feature 4: Is weekend
        features[4] = 1.0 if timestamp.weekday() >= 5 else 0.0

        # Feature 5: Is business hours (9 AM - 5 PM)
        features[5] = 1.0 if 9 <= timestamp.hour < 17 else 0.0

        # Feature 6: Is late night (10 PM - 6 AM)
        features[6] = 1.0 if timestamp.hour >= 22 or timestamp.hour < 6 else 0.0

        # Feature 7: Quarter of year (normalized)
        quarter = (timestamp.month - 1) // 3
        features[7] = quarter / 3.0

        return features

    def _extract_author_features(
        self, commit_data: dict[str, Any], author_stats: Optional[dict[str, Any]] = None
    ) -> np.ndarray:
        """Extract author-based features.

        Args:
            commit_data: Commit data dictionary
            author_stats: Optional author statistics

        Returns:
            5-dimensional array with author features
        """
        features = np.zeros(5, dtype=np.float32)

        author_name = commit_data.get("author_name", "")
        author_email = commit_data.get("author_email", "")

        # Feature 0: Author name length (normalized)
        features[0] = min(1.0, len(author_name) / 50.0)

        # Feature 1: Has corporate email
        corporate_domains = [".com", ".org", ".net", ".io", ".co"]
        has_corporate = any(domain in author_email.lower() for domain in corporate_domains)
        is_github_noreply = "noreply.github.com" in author_email.lower()
        features[1] = 1.0 if has_corporate and not is_github_noreply else 0.0

        # Feature 2: Is likely automated (bot/CI)
        automated_indicators = ["bot", "ci", "github-actions", "dependabot", "renovate"]
        is_automated = any(
            indicator in author_name.lower() or indicator in author_email.lower()
            for indicator in automated_indicators
        )
        features[2] = 1.0 if is_automated else 0.0

        # Features 3-4: Author statistics (if available)
        if author_stats:
            # Feature 3: Author experience (normalized commit count)
            total_commits = author_stats.get("total_commits", 1)
            features[3] = min(1.0, np.log1p(total_commits) / np.log1p(1000))

            # Feature 4: Typical commit size compared to this commit
            avg_size = author_stats.get("avg_commit_size", 0)
            current_size = commit_data.get("insertions", 0) + commit_data.get("deletions", 0)
            if avg_size > 0:
                features[4] = min(2.0, current_size / avg_size)  # Ratio, capped at 2x

        return features

    def get_feature_names(self) -> list[str]:
        """Get human-readable names for all 68 features.

        Returns:
            List of feature names corresponding to the feature vector indices
        """
        names = []

        # Keyword features (20)
        for category in self.keyword_categories:
            names.append(f"keyword_{category}")

        # File features (20)
        file_feature_names = [
            "lang_1st",
            "lang_2nd",
            "lang_3rd",
            "lang_4th",
            "lang_5th",
            "activity_1st",
            "activity_2nd",
            "activity_3rd",
            "activity_4th",
            "activity_5th",
            "lang_diversity",
            "activity_diversity",
            "generated_ratio",
            "is_multilingual",
            "is_cross_functional",
            "ext_py",
            "ext_js",
            "ext_java",
            "ext_go",
            "ext_sql",
        ]
        names.extend(file_feature_names)

        # Statistics features (15)
        stats_feature_names = [
            "files_changed",
            "insertions",
            "deletions",
            "total_lines",
            "insert_delete_ratio",
            "message_length",
            "word_count",
            "line_count",
            "avg_lines_per_file",
            "has_conventional_format",
            "has_ticket_ref",
            "is_merge",
            "has_code_in_msg",
            "message_complexity",
            "is_large_commit",
        ]
        names.extend(stats_feature_names)

        # Temporal features (8)
        temporal_feature_names = [
            "hour_of_day",
            "day_of_week",
            "day_of_month",
            "month_of_year",
            "is_weekend",
            "is_business_hours",
            "is_late_night",
            "quarter",
        ]
        names.extend(temporal_feature_names)

        # Author features (5)
        author_feature_names = [
            "author_name_length",
            "has_corporate_email",
            "is_automated",
            "author_experience",
            "commit_size_vs_typical",
        ]
        names.extend(author_feature_names)

        return names

    def extract_batch_features(
        self,
        commit_batch: list[dict[str, Any]],
        author_stats_batch: Optional[list[dict[str, Any]]] = None,
    ) -> np.ndarray:
        """Extract features for a batch of commits efficiently.

        Args:
            commit_batch: List of commit data dictionaries
            author_stats_batch: Optional list of author statistics

        Returns:
            2D numpy array of shape (n_commits, 68) with feature vectors
        """
        n_commits = len(commit_batch)
        features = np.zeros((n_commits, 68), dtype=np.float32)

        for i, commit_data in enumerate(commit_batch):
            author_stats = None
            if author_stats_batch and i < len(author_stats_batch):
                author_stats = author_stats_batch[i]

            features[i] = self.extract_features(commit_data, author_stats)

        return features
