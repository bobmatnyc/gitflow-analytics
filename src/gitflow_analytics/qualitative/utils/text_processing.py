"""Text processing utilities for qualitative analysis."""

import hashlib
import re


class TextProcessor:
    """Utility class for text preprocessing and feature extraction.

    This class provides common text processing operations needed across
    the qualitative analysis pipeline, including normalization, feature
    extraction, and similarity calculations.
    """

    def __init__(self) -> None:
        """Initialize text processor with common patterns."""
        # Common patterns for normalization
        self.url_pattern = re.compile(r"https?://[^\s]+")
        self.email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        self.hash_pattern = re.compile(r"\b[a-f0-9]{7,40}\b")  # Git hashes
        self.ticket_pattern = re.compile(r"\b(?:JIRA|TICKET|ISSUE|BUG|TASK)-?\d+\b", re.IGNORECASE)

        # Stop words for feature extraction
        self.stop_words: set[str] = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
        }

    def normalize_message(self, message: str) -> str:
        """Normalize commit message for consistent processing.

        This method standardizes commit messages by removing URLs, emails,
        hashes, and other variable content that doesn't contribute to
        semantic classification.

        Args:
            message: Raw commit message

        Returns:
            Normalized message suitable for classification
        """
        if not message:
            return ""

        # Convert to lowercase for consistency
        normalized = message.lower().strip()

        # Remove URLs, emails, and hashes
        normalized = self.url_pattern.sub("[URL]", normalized)
        normalized = self.email_pattern.sub("[EMAIL]", normalized)
        normalized = self.hash_pattern.sub("[HASH]", normalized)

        # Normalize ticket references
        normalized = self.ticket_pattern.sub("[TICKET]", normalized)

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized.strip()

    def extract_keywords(self, text: str, min_length: int = 3) -> list[str]:
        """Extract meaningful keywords from text.

        Extracts keywords by removing stop words, punctuation, and short words
        that are unlikely to be semantically meaningful.

        Args:
            text: Input text to extract keywords from
            min_length: Minimum length for keywords

        Returns:
            List of extracted keywords
        """
        if not text:
            return []

        # Split into words and clean
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

        # Filter stop words and short words
        keywords = [
            word for word in words if word not in self.stop_words and len(word) >= min_length
        ]

        return keywords

    def create_semantic_fingerprint(self, message: str, files: list[str]) -> str:
        """Create a semantic fingerprint for similarity matching.

        Creates a hash-based fingerprint that captures the semantic essence
        of a commit for pattern matching and caching.

        Args:
            message: Commit message
            files: List of changed files

        Returns:
            Hex-encoded fingerprint string
        """
        # Normalize message for consistent fingerprinting
        normalized_msg = self.normalize_message(message)
        keywords = self.extract_keywords(normalized_msg)

        # Extract file patterns (extensions, directories)
        file_patterns = []
        for file_path in files[:10]:  # Limit to prevent huge fingerprints
            # Get file extension
            if "." in file_path:
                ext = file_path.split(".")[-1].lower()
                file_patterns.append(f"ext:{ext}")

            # Get directory patterns
            parts = file_path.split("/")
            if len(parts) > 1:
                # First directory
                file_patterns.append(f"dir:{parts[0]}")
                # Last directory before file
                if len(parts) > 2:
                    file_patterns.append(f"dir:{parts[-2]}")

        # Combine keywords and file patterns
        semantic_elements = sorted(keywords[:10]) + sorted(set(file_patterns))

        # Create fingerprint
        fingerprint_text = "|".join(semantic_elements)
        return hashlib.md5(fingerprint_text.encode()).hexdigest()

    def calculate_message_similarity(self, msg1: str, msg2: str) -> float:
        """Calculate semantic similarity between two commit messages.

        Uses keyword overlap to estimate semantic similarity between
        commit messages for grouping similar commits.

        Args:
            msg1: First commit message
            msg2: Second commit message

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not msg1 or not msg2:
            return 0.0

        # Extract keywords from both messages
        keywords1 = set(self.extract_keywords(self.normalize_message(msg1)))
        keywords2 = set(self.extract_keywords(self.normalize_message(msg2)))

        if not keywords1 or not keywords2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))

        return intersection / union if union > 0 else 0.0

    def extract_file_patterns(self, files: list[str]) -> dict[str, int]:
        """Extract file patterns for domain classification.

        Analyzes file paths to extract patterns useful for determining
        the technical domain of changes.

        Args:
            files: List of file paths

        Returns:
            Dictionary mapping pattern types to counts
        """
        patterns = {
            "extensions": {},
            "directories": {},
            "special_files": {},
        }

        for file_path in files:
            # File extensions
            if "." in file_path:
                ext = file_path.split(".")[-1].lower()
                patterns["extensions"][ext] = patterns["extensions"].get(ext, 0) + 1

            # Directory patterns
            parts = file_path.split("/")
            for part in parts[:-1]:  # Exclude filename
                if part:  # Skip empty parts
                    patterns["directories"][part] = patterns["directories"].get(part, 0) + 1

            # Special files
            filename = parts[-1].lower()
            special_files = [
                "dockerfile",
                "makefile",
                "readme",
                "license",
                "changelog",
                "package.json",
                "requirements.txt",
                "setup.py",
                "pom.xml",
            ]
            for special in special_files:
                if special in filename:
                    patterns["special_files"][special] = (
                        patterns["special_files"].get(special, 0) + 1
                    )

        return patterns

    def calculate_commit_complexity(
        self, message: str, files: list[str], insertions: int, deletions: int
    ) -> dict[str, float]:
        """Calculate various complexity metrics for a commit.

        Estimates the complexity of a commit based on message content,
        file changes, and line changes to help with risk assessment.

        Args:
            message: Commit message
            files: List of changed files
            insertions: Number of lines inserted
            deletions: Number of lines deleted

        Returns:
            Dictionary of complexity metrics
        """
        metrics = {}

        # Message complexity (length, keywords)
        metrics["message_length"] = len(message)
        keywords = self.extract_keywords(message)
        metrics["keyword_count"] = len(keywords)
        metrics["message_complexity"] = min(1.0, len(keywords) / 10.0)

        # File complexity
        metrics["files_changed"] = len(files)
        metrics["file_complexity"] = min(1.0, len(files) / 20.0)

        # Line change complexity
        total_changes = insertions + deletions
        metrics["total_changes"] = total_changes
        metrics["change_complexity"] = min(1.0, total_changes / 500.0)

        # Overall complexity score (0.0 to 1.0)
        metrics["overall_complexity"] = (
            metrics["message_complexity"] * 0.2
            + metrics["file_complexity"] * 0.3
            + metrics["change_complexity"] * 0.5
        )

        return metrics
