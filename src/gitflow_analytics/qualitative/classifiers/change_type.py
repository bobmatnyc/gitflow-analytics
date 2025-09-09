"""Change type classifier using semantic analysis of commit messages."""

import importlib.util
import logging
import re
from typing import Any, Optional

from ..models.schemas import ChangeTypeConfig

# Check if spacy is available without importing it
SPACY_AVAILABLE = importlib.util.find_spec("spacy") is not None

if SPACY_AVAILABLE:
    from spacy.tokens import Doc
else:
    Doc = Any


class ChangeTypeClassifier:
    """Classify commits by change type using semantic analysis.

    This classifier determines the type of change represented by a commit
    (feature, bugfix, refactor, etc.) by analyzing the commit message semantics
    and file patterns.

    The classification uses a combination of:
    - Semantic keyword matching with action/object/context patterns
    - File pattern analysis for additional signals
    - Rule-based patterns for common commit message formats
    """

    def __init__(self, config: ChangeTypeConfig):
        """Initialize change type classifier.

        Args:
            config: Configuration for change type classification
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Define semantic patterns for each change type
        self.change_patterns = {
            "feature": {
                "action_words": {
                    "add",
                    "implement",
                    "create",
                    "build",
                    "introduce",
                    "develop",
                    "enable",
                    "support",
                    "allow",
                    "provide",
                    "include",
                    "addition",
                    "initialize",
                    "prepare",
                    "extend",
                },
                "object_words": {
                    "feature",
                    "functionality",
                    "capability",
                    "component",
                    "module",
                    "endpoint",
                    "api",
                    "service",
                    "interface",
                    "system",
                    "integration",
                    "column",
                    "field",
                    "property",
                },
                "context_words": {
                    "new",
                    "initial",
                    "first",
                    "user",
                    "client",
                    "support",
                    "enhancement",
                    "improvement",
                    "missing",
                    "space",
                    "sticky",
                },
            },
            "bugfix": {
                "action_words": {
                    "fix",
                    "resolve",
                    "correct",
                    "repair",
                    "patch",
                    "address",
                    "handle",
                    "solve",
                    "debug",
                    "prevent",
                    "corrected",
                },
                "object_words": {
                    "bug",
                    "issue",
                    "problem",
                    "error",
                    "defect",
                    "exception",
                    "crash",
                    "failure",
                    "leak",
                    "regression",
                    "beacon",
                    "beacons",
                },
                "context_words": {
                    "broken",
                    "failing",
                    "incorrect",
                    "wrong",
                    "invalid",
                    "missing",
                    "null",
                    "undefined",
                    "not",
                    "allowing",
                },
            },
            "refactor": {
                "action_words": {
                    "refactor",
                    "restructure",
                    "reorganize",
                    "cleanup",
                    "simplify",
                    "optimize",
                    "improve",
                    "enhance",
                    "streamline",
                    "consolidate",
                    "refine",
                    "ensure",
                    "replace",
                    "improves",
                },
                "object_words": {
                    "code",
                    "structure",
                    "architecture",
                    "design",
                    "logic",
                    "method",
                    "function",
                    "class",
                    "module",
                    "combo",
                    "behavior",
                    "focus",
                },
                "context_words": {
                    "better",
                    "cleaner",
                    "simpler",
                    "efficient",
                    "maintainable",
                    "readable",
                    "performance",
                    "box",
                    "hacking",
                },
            },
            "docs": {
                "action_words": {
                    "update",
                    "add",
                    "improve",
                    "write",
                    "document",
                    "clarify",
                    "explain",
                    "describe",
                    "detail",
                    "added",
                },
                "object_words": {
                    "documentation",
                    "readme",
                    "docs",
                    "comment",
                    "docstring",
                    "guide",
                    "tutorial",
                    "example",
                    "specification",
                    "translations",
                    "spanish",
                    "label",
                },
                "context_words": {
                    "explain",
                    "clarify",
                    "describe",
                    "instruction",
                    "help",
                    "change",
                    "dynamically",
                    "language",
                },
            },
            "test": {
                "action_words": {
                    "add",
                    "update",
                    "fix",
                    "improve",
                    "write",
                    "create",
                    "enhance",
                    "extend",
                },
                "object_words": {
                    "test",
                    "spec",
                    "coverage",
                    "unit",
                    "integration",
                    "e2e",
                    "testing",
                    "mock",
                    "stub",
                    "fixture",
                },
                "context_words": {
                    "testing",
                    "verify",
                    "validate",
                    "check",
                    "ensure",
                    "coverage",
                    "assertion",
                },
            },
            "chore": {
                "action_words": {
                    "update",
                    "bump",
                    "upgrade",
                    "configure",
                    "setup",
                    "install",
                    "remove",
                    "delete",
                    "clean",
                    "sync",
                    "merge",
                },
                "object_words": {
                    "dependency",
                    "package",
                    "config",
                    "configuration",
                    "build",
                    "version",
                    "tool",
                    "script",
                    "workflow",
                    "console",
                    "log",
                    "main",
                },
                "context_words": {
                    "maintenance",
                    "housekeeping",
                    "routine",
                    "automated",
                    "ci",
                    "cd",
                    "pipeline",
                    "auto",
                    "removal",
                },
            },
            "security": {
                "action_words": {
                    "fix",
                    "secure",
                    "protect",
                    "validate",
                    "sanitize",
                    "encrypt",
                    "authenticate",
                    "authorize",
                },
                "object_words": {
                    "security",
                    "vulnerability",
                    "exploit",
                    "xss",
                    "csrf",
                    "injection",
                    "authentication",
                    "authorization",
                    "permission",
                },
                "context_words": {
                    "secure",
                    "safe",
                    "protected",
                    "validated",
                    "sanitized",
                    "encrypted",
                    "threat",
                    "attack",
                },
            },
            "hotfix": {
                "action_words": {"hotfix", "fix", "patch", "urgent", "critical", "emergency"},
                "object_words": {
                    "production",
                    "critical",
                    "urgent",
                    "emergency",
                    "hotfix",
                    "issue",
                    "bug",
                    "problem",
                },
                "context_words": {
                    "urgent",
                    "critical",
                    "immediate",
                    "production",
                    "live",
                    "emergency",
                    "asap",
                },
            },
            "config": {
                "action_words": {
                    "configure",
                    "setup",
                    "adjust",
                    "modify",
                    "change",
                    "update",
                    "tweak",
                    "changing",
                },
                "object_words": {
                    "config",
                    "configuration",
                    "settings",
                    "environment",
                    "parameter",
                    "option",
                    "flag",
                    "variable",
                    "roles",
                    "user",
                    "schema",
                    "access",
                    "levels",
                },
                "context_words": {
                    "environment",
                    "production",
                    "development",
                    "staging",
                    "deployment",
                    "setup",
                    "roles",
                    "permission",
                    "api",
                },
            },
            "integration": {
                "action_words": {
                    "integrate",
                    "add",
                    "implement",
                    "connect",
                    "setup",
                    "remove",
                    "extend",
                    "removing",
                },
                "object_words": {
                    "integration",
                    "posthog",
                    "iubenda",
                    "auth0",
                    "oauth",
                    "api",
                    "service",
                    "third-party",
                    "external",
                    "mena",
                },
                "context_words": {
                    "collection",
                    "data",
                    "privacy",
                    "policy",
                    "implementation",
                    "access",
                    "redirect",
                },
            },
        }

        # File pattern signals for change types
        self.file_patterns = {
            "test": [
                r".*test.*\.py$",
                r".*spec.*\.js$",
                r".*test.*\.java$",
                r"test_.*\.py$",
                r".*_test\.go$",
                r".*\.test\.(js|ts)$",
                r"__tests__/.*",
                r"tests?/.*",
                r"spec/.*",
            ],
            "docs": [
                r".*\.md$",
                r".*\.rst$",
                r".*\.txt$",
                r"README.*",
                r"CHANGELOG.*",
                r"docs?/.*",
                r"documentation/.*",
            ],
            "config": [
                r".*\.ya?ml$",
                r".*\.json$",
                r".*\.toml$",
                r".*\.ini$",
                r".*\.env.*",
                r"Dockerfile.*",
                r".*config.*",
                r"\.github/.*",
            ],
            "chore": [
                r"package.*\.json$",
                r"requirements.*\.txt$",
                r"Pipfile.*",
                r"pom\.xml$",
                r"build\.gradle$",
                r".*\.lock$",
            ],
        }

        # Compile regex patterns for efficiency
        self._compile_file_patterns()

        # Common commit message prefixes
        self.prefix_patterns = {
            "feat": "feature",
            "feature": "feature",
            "fix": "bugfix",
            "bugfix": "bugfix",
            "refactor": "refactor",
            "docs": "docs",
            "test": "test",
            "chore": "chore",
            "security": "security",
            "hotfix": "hotfix",
            "config": "config",
            "integration": "integration",
            "integrate": "integration",
            "style": "chore",  # Style changes are usually chores
            "perf": "refactor",  # Performance improvements are refactoring
            "build": "chore",
            "ci": "chore",
        }

    def _compile_file_patterns(self) -> None:
        """Compile regex patterns for file matching."""
        self.compiled_file_patterns = {}
        for change_type, patterns in self.file_patterns.items():
            self.compiled_file_patterns[change_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def classify(self, message: str, doc: Doc, files: list[str]) -> tuple[str, float]:
        """Classify commit change type with confidence score.

        Args:
            message: Commit message
            doc: spaCy processed document
            files: List of changed files

        Returns:
            Tuple of (change_type, confidence_score)
        """
        if not message:
            return "unknown", 0.0

        # Step 1: Check for conventional commit prefixes
        prefix_result = self._check_conventional_prefix(message)
        if prefix_result:
            change_type, confidence = prefix_result
            if confidence >= self.config.min_confidence:
                return change_type, confidence

        # Step 2: Semantic analysis of message content
        semantic_scores = self._analyze_semantic_content(message, doc)

        # Step 3: File pattern analysis
        file_scores = self._analyze_file_patterns(files)

        # Step 4: Combine scores with weights
        combined_scores = self._combine_scores(semantic_scores, file_scores)

        # Step 5: Select best match
        if not combined_scores:
            return "unknown", 0.0

        best_type = max(combined_scores.keys(), key=lambda k: combined_scores[k])
        confidence = combined_scores[best_type]

        # Apply confidence threshold
        if confidence < self.config.min_confidence:
            return "unknown", confidence

        return best_type, confidence

    def _check_conventional_prefix(self, message: str) -> Optional[tuple[str, float]]:
        """Check for conventional commit message prefixes.

        Args:
            message: Commit message

        Returns:
            Tuple of (change_type, confidence) if found, None otherwise
        """
        # Look for conventional commit format: type(scope): description
        conventional_pattern = r"^(\w+)(?:\([^)]*\))?\s*:\s*(.+)"
        match = re.match(conventional_pattern, message.strip(), re.IGNORECASE)

        if match:
            prefix = match.group(1).lower()
            if prefix in self.prefix_patterns:
                return self.prefix_patterns[prefix], 0.9  # High confidence for explicit prefixes

        # Check for simple prefixes at start of message
        words = message.lower().split()
        if words:
            first_word = words[0].rstrip(":").rstrip("-")
            if first_word in self.prefix_patterns:
                return self.prefix_patterns[first_word], 0.8

        return None

    def _analyze_semantic_content(self, message: str, doc: Doc) -> dict[str, float]:
        """Analyze semantic content of commit message.

        Args:
            message: Commit message
            doc: spaCy processed document

        Returns:
            Dictionary of change_type -> confidence_score
        """
        if not SPACY_AVAILABLE or not doc:
            # Fallback to simple keyword matching
            return self._simple_keyword_analysis(message.lower())

        # Extract semantic features from spaCy doc
        features = self._extract_semantic_features(doc)

        # Calculate similarity to each change type
        scores = {}
        for change_type, patterns in self.change_patterns.items():
            similarity = self._calculate_semantic_similarity(features, patterns)
            if similarity > 0:
                scores[change_type] = similarity

        return scores

    def _extract_semantic_features(self, doc: Doc) -> dict[str, set[str]]:
        """Extract semantic features from spaCy document.

        Args:
            doc: spaCy processed document

        Returns:
            Dictionary of feature_type -> set_of_words
        """
        features = {
            "verbs": set(),
            "nouns": set(),
            "adjectives": set(),
            "entities": set(),
            "lemmas": set(),
        }

        for token in doc:
            if token.is_stop or token.is_punct or len(token.text) < 2:
                continue

            lemma = token.lemma_.lower()
            features["lemmas"].add(lemma)

            if token.pos_ == "VERB":
                features["verbs"].add(lemma)
            elif token.pos_ in ["NOUN", "PROPN"]:
                features["nouns"].add(lemma)
            elif token.pos_ == "ADJ":
                features["adjectives"].add(lemma)

        # Add named entities
        for ent in doc.ents:
            features["entities"].add(ent.text.lower())

        return features

    def _calculate_semantic_similarity(
        self, features: dict[str, set[str]], patterns: dict[str, set[str]]
    ) -> float:
        """Calculate semantic similarity between features and patterns.

        Args:
            features: Extracted semantic features
            patterns: Change type patterns

        Returns:
            Similarity score (0.0 to 1.0)
        """
        similarity_score = 0.0

        # Action words (verbs) - highest weight
        action_matches = len(features["verbs"].intersection(patterns["action_words"]))
        if action_matches > 0:
            similarity_score += action_matches * 0.5

        # Object words (nouns) - medium weight
        object_matches = len(features["nouns"].intersection(patterns["object_words"]))
        if object_matches > 0:
            similarity_score += object_matches * 0.3

        # Context words (any lemma) - lower weight
        all_lemmas = features["lemmas"]
        context_matches = len(all_lemmas.intersection(patterns["context_words"]))
        if context_matches > 0:
            similarity_score += context_matches * 0.2

        # Normalize by maximum possible score
        max_possible = (
            len(patterns["action_words"]) * 0.5
            + len(patterns["object_words"]) * 0.3
            + len(patterns["context_words"]) * 0.2
        )

        return min(1.0, similarity_score / max_possible) if max_possible > 0 else 0.0

    def _simple_keyword_analysis(self, message: str) -> dict[str, float]:
        """Simple keyword-based analysis fallback.

        Args:
            message: Lowercase commit message

        Returns:
            Dictionary of change_type -> confidence_score
        """
        scores = {}
        words = set(re.findall(r"\b\w+\b", message))

        for change_type, patterns in self.change_patterns.items():
            all_pattern_words = (
                patterns["action_words"] | patterns["object_words"] | patterns["context_words"]
            )
            matches = len(words.intersection(all_pattern_words))

            if matches > 0:
                # Simple scoring based on keyword matches
                scores[change_type] = min(1.0, matches / 5.0)  # Scale to 0-1

        return scores

    def _analyze_file_patterns(self, files: list[str]) -> dict[str, float]:
        """Analyze file patterns for change type signals.

        Args:
            files: List of changed file paths

        Returns:
            Dictionary of change_type -> confidence_score
        """
        if not files:
            return {}

        scores = {}

        for change_type, patterns in self.compiled_file_patterns.items():
            matching_files = 0

            for file_path in files:
                for pattern in patterns:
                    if pattern.search(file_path):
                        matching_files += 1
                        break  # Don't double-count same file

            if matching_files > 0:
                # File pattern confidence based on proportion of matching files
                confidence = min(1.0, matching_files / len(files))
                scores[change_type] = confidence

        return scores

    def _combine_scores(
        self, semantic_scores: dict[str, float], file_scores: dict[str, float]
    ) -> dict[str, float]:
        """Combine semantic and file pattern scores.

        Args:
            semantic_scores: Scores from semantic analysis
            file_scores: Scores from file pattern analysis

        Returns:
            Combined scores dictionary
        """
        combined = {}
        all_types = set(semantic_scores.keys()) | set(file_scores.keys())

        for change_type in all_types:
            semantic_score = semantic_scores.get(change_type, 0.0)
            file_score = file_scores.get(change_type, 0.0)

            # Weighted combination
            combined_score = (
                semantic_score * self.config.semantic_weight
                + file_score * self.config.file_pattern_weight
            )

            if combined_score > 0:
                combined[change_type] = combined_score

        return combined
