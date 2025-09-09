"""Domain classifier for identifying technical domains of commits."""

import importlib.util
import logging
import re
from collections import defaultdict
from typing import Any

from ..models.schemas import DomainConfig

# Check if spacy is available without importing it
SPACY_AVAILABLE = importlib.util.find_spec("spacy") is not None

if SPACY_AVAILABLE:
    from spacy.tokens import Doc
else:
    Doc = Any


class DomainClassifier:
    """Classify commits by technical domain (frontend, backend, etc.).

    This classifier determines the technical domain or business area
    affected by a commit by analyzing both the commit message content
    and the patterns of files that were changed.

    Domains identified:
    - frontend: UI/UX, client-side code
    - backend: Server-side logic, APIs
    - database: Data models, migrations, queries
    - infrastructure: Deployment, configuration, DevOps
    - mobile: Mobile app development
    - devops: CI/CD, build tools, automation
    """

    def __init__(self, config: DomainConfig):
        """Initialize domain classifier.

        Args:
            config: Configuration for domain classification
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Compile file patterns for efficient matching
        self._compile_file_patterns()

        # Keyword patterns for semantic analysis
        self.keyword_patterns = config.keyword_patterns

        # Directory patterns that strongly indicate domains
        self.directory_indicators = {
            "frontend": {
                "src/components",
                "src/pages",
                "src/views",
                "public",
                "assets",
                "static",
                "styles",
                "css",
                "scss",
                "ui",
                "components",
                "pages",
            },
            "backend": {
                "src/controllers",
                "src/services",
                "src/api",
                "api",
                "server",
                "controllers",
                "services",
                "handlers",
                "routes",
                "middleware",
            },
            "database": {
                "migrations",
                "models",
                "schemas",
                "seeds",
                "data",
                "sql",
                "database",
                "db",
                "repositories",
            },
            "infrastructure": {
                "terraform",
                "ansible",
                "k8s",
                "kubernetes",
                "helm",
                "charts",
                "infrastructure",
                "deploy",
                "deployment",
                "ops",
            },
            "mobile": {
                "android",
                "ios",
                "mobile",
                "app",
                "native",
                "react-native",
                "flutter",
                "swift",
                "kotlin",
            },
            "devops": {
                ".github",
                ".gitlab",
                "ci",
                "cd",
                "scripts",
                "build",
                "docker",
                "jenkins",
                "actions",
                "workflows",
            },
        }

        # Technology stack indicators
        self.tech_indicators = {
            "frontend": {
                "react",
                "vue",
                "angular",
                "svelte",
                "jquery",
                "bootstrap",
                "tailwind",
                "css",
                "html",
                "javascript",
                "typescript",
                "jsx",
                "tsx",
            },
            "backend": {
                "django",
                "flask",
                "fastapi",
                "express",
                "spring",
                "rails",
                "laravel",
                "api",
                "endpoint",
                "service",
                "controller",
            },
            "database": {
                "mysql",
                "postgresql",
                "mongodb",
                "redis",
                "elasticsearch",
                "migration",
                "schema",
                "query",
                "orm",
                "sql",
            },
            "infrastructure": {
                "aws",
                "gcp",
                "azure",
                "docker",
                "kubernetes",
                "terraform",
                "ansible",
                "helm",
                "nginx",
                "apache",
            },
            "mobile": {
                "android",
                "ios",
                "swift",
                "kotlin",
                "flutter",
                "react-native",
                "xamarin",
                "cordova",
                "ionic",
            },
            "devops": {
                "jenkins",
                "gitlab-ci",
                "github-actions",
                "circleci",
                "travis",
                "docker",
                "kubernetes",
                "helm",
                "terraform",
            },
        }

    def _compile_file_patterns(self) -> None:
        """Compile file extension patterns for efficient matching."""
        self.compiled_file_patterns = {}

        for domain, patterns in self.config.file_patterns.items():
            compiled_patterns = []
            for pattern in patterns:
                try:
                    # Convert glob patterns to regex
                    regex_pattern = self._glob_to_regex(pattern)
                    compiled_patterns.append(re.compile(regex_pattern, re.IGNORECASE))
                except re.error as e:
                    self.logger.warning(
                        f"Invalid file pattern '{pattern}' for domain {domain}: {e}"
                    )

            self.compiled_file_patterns[domain] = compiled_patterns

    def _glob_to_regex(self, pattern: str) -> str:
        """Convert glob pattern to regex.

        Args:
            pattern: Glob pattern (e.g., '*.js', '**/models/**')

        Returns:
            Equivalent regex pattern
        """
        # Simple glob to regex conversion
        pattern = pattern.replace(".", r"\.")
        pattern = pattern.replace("*", ".*")
        pattern = pattern.replace("?", ".")
        pattern = f"^{pattern}$"
        return pattern

    def classify(self, message: str, doc: Doc, files: list[str]) -> tuple[str, float]:
        """Classify commit domain with confidence score.

        Args:
            message: Commit message
            doc: spaCy processed document (may be None)
            files: List of changed files

        Returns:
            Tuple of (domain, confidence_score)
        """
        if not message and not files:
            return "unknown", 0.0

        # Analyze file patterns (primary signal)
        file_scores = self._analyze_file_patterns(files)

        # Analyze directory patterns
        dir_scores = self._analyze_directory_patterns(files)

        # Analyze message content
        message_scores = self._analyze_message_content(message, doc)

        # Combine all signals
        combined_scores = self._combine_domain_scores(file_scores, dir_scores, message_scores)

        if not combined_scores:
            return "unknown", 0.0

        # Select best domain
        best_domain = max(combined_scores.keys(), key=lambda k: combined_scores[k])
        confidence = combined_scores[best_domain]

        # Apply confidence threshold
        if confidence < self.config.min_confidence:
            return "unknown", confidence

        return best_domain, confidence

    def _analyze_file_patterns(self, files: list[str]) -> dict[str, float]:
        """Analyze file patterns to determine domain.

        Args:
            files: List of file paths

        Returns:
            Dictionary of domain -> confidence_score
        """
        if not files:
            return {}

        domain_matches = defaultdict(int)

        for file_path in files:
            for domain, patterns in self.compiled_file_patterns.items():
                for pattern in patterns:
                    if pattern.search(file_path):
                        domain_matches[domain] += 1
                        break  # Don't double-count same file for same domain

        # Convert to confidence scores
        scores = {}
        total_files = len(files)

        for domain, matches in domain_matches.items():
            # Confidence based on proportion of matching files
            confidence = matches / total_files
            scores[domain] = min(1.0, confidence * 2)  # Boost confidence for strong signals

        return scores

    def _analyze_directory_patterns(self, files: list[str]) -> dict[str, float]:
        """Analyze directory patterns for domain signals.

        Args:
            files: List of file paths

        Returns:
            Dictionary of domain -> confidence_score
        """
        if not files:
            return {}

        domain_scores = defaultdict(float)

        for file_path in files:
            # Normalize path separators and convert to lowercase
            normalized_path = file_path.replace("\\", "/").lower()
            path_parts = normalized_path.split("/")

            # Check each domain's directory indicators
            for domain, indicators in self.directory_indicators.items():
                for indicator in indicators:
                    # Check if indicator appears in any part of the path
                    if any(indicator in part for part in path_parts):
                        domain_scores[domain] += 1.0
                        break
                    # Also check full path contains indicator
                    elif indicator in normalized_path:
                        domain_scores[domain] += 0.5

        # Normalize scores
        scores = {}
        max_score = max(domain_scores.values()) if domain_scores else 0

        if max_score > 0:
            for domain, score in domain_scores.items():
                scores[domain] = min(1.0, score / max_score)

        return scores

    def _analyze_message_content(self, message: str, doc: Doc) -> dict[str, float]:
        """Analyze commit message content for domain keywords.

        Args:
            message: Commit message
            doc: spaCy processed document (may be None)

        Returns:
            Dictionary of domain -> confidence_score
        """
        if not message:
            return {}

        # Convert message to lowercase for analysis
        message_lower = message.lower()

        # Extract keywords from message
        if SPACY_AVAILABLE and doc:
            # Use spaCy for better keyword extraction
            keywords = self._extract_keywords_from_doc(doc)
        else:
            # Fallback to simple word extraction
            keywords = set(re.findall(r"\b\w+\b", message_lower))

        # Score domains based on keyword matches
        domain_scores = {}

        for domain, domain_keywords in self.keyword_patterns.items():
            keyword_matches = len(
                keywords.intersection(set(word.lower() for word in domain_keywords))
            )

            if keyword_matches > 0:
                # Base score from keyword matches
                base_score = min(1.0, keyword_matches / 3.0)  # Scale to 0-1

                # Boost score for technology indicators
                tech_keywords = self.tech_indicators.get(domain, set())
                tech_matches = len(keywords.intersection(tech_keywords))
                tech_boost = min(0.3, tech_matches * 0.1)

                domain_scores[domain] = min(1.0, base_score + tech_boost)

        return domain_scores

    def _extract_keywords_from_doc(self, doc: Doc) -> set[str]:
        """Extract meaningful keywords from spaCy document.

        Args:
            doc: spaCy processed document

        Returns:
            Set of extracted keywords
        """
        keywords = set()

        for token in doc:
            if (
                not token.is_stop
                and not token.is_punct
                and len(token.text) > 2
                and token.pos_ in ["NOUN", "PROPN", "ADJ", "VERB"]
            ):
                keywords.add(token.lemma_.lower())

        # Add named entities
        for ent in doc.ents:
            if len(ent.text) > 2:
                keywords.add(ent.text.lower())

        return keywords

    def _combine_domain_scores(
        self,
        file_scores: dict[str, float],
        dir_scores: dict[str, float],
        message_scores: dict[str, float],
    ) -> dict[str, float]:
        """Combine scores from different analysis methods.

        Args:
            file_scores: Scores from file pattern analysis
            dir_scores: Scores from directory pattern analysis
            message_scores: Scores from message content analysis

        Returns:
            Combined scores dictionary
        """
        all_domains = set(file_scores.keys()) | set(dir_scores.keys()) | set(message_scores.keys())
        combined_scores = {}

        # Weights for different signal types
        weights = {
            "file": 0.5,  # File patterns are strongest signal
            "directory": 0.3,  # Directory patterns are also strong
            "message": 0.2,  # Message content provides additional context
        }

        for domain in all_domains:
            file_score = file_scores.get(domain, 0.0)
            dir_score = dir_scores.get(domain, 0.0)
            message_score = message_scores.get(domain, 0.0)

            # Weighted combination
            combined_score = (
                file_score * weights["file"]
                + dir_score * weights["directory"]
                + message_score * weights["message"]
            )

            # Bonus for multiple signal types agreeing
            signal_count = sum(1 for score in [file_score, dir_score, message_score] if score > 0)
            if signal_count > 1:
                combined_score *= 1.0 + (signal_count - 1) * 0.1  # 10% bonus per additional signal

            if combined_score > 0:
                combined_scores[domain] = min(1.0, combined_score)

        return combined_scores

    def get_domain_statistics(self, files: list[str]) -> dict[str, Any]:
        """Get detailed domain analysis statistics for debugging.

        Args:
            files: List of file paths

        Returns:
            Dictionary with detailed analysis breakdown
        """
        stats = {
            "total_files": len(files),
            "file_analysis": self._analyze_file_patterns(files),
            "directory_analysis": self._analyze_directory_patterns(files),
            "file_extensions": {},
            "directory_breakdown": {},
        }

        # File extension breakdown
        extensions = defaultdict(int)
        directories = defaultdict(int)

        for file_path in files:
            # Extract extension
            if "." in file_path:
                ext = file_path.split(".")[-1].lower()
                extensions[ext] += 1

            # Extract directories
            path_parts = file_path.split("/")
            for part in path_parts[:-1]:  # Exclude filename
                if part:
                    directories[part] += 1

        stats["file_extensions"] = dict(extensions)
        stats["directory_breakdown"] = dict(directories)

        return stats
