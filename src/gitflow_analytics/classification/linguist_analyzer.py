"""File language and activity analysis inspired by GitHub Linguist.

This module provides capabilities to analyze file changes in commits to determine:
- Programming languages involved
- Development activities (UI, API, database, etc.)
- Generated/binary file detection
- Directory-based activity patterns

The analysis helps understand the technical context of commits for better classification.
"""

import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


class LinguistAnalyzer:
    """Analyzes files to determine programming languages and development activities.

    This class provides GitHub Linguist-inspired analysis of file changes,
    mapping file extensions to languages and directory patterns to activities.
    It's designed to work with commit file lists to provide context for ML classification.
    """

    def __init__(self):
        """Initialize the linguist analyzer with language and activity mappings."""
        # File extension to programming language mappings
        # Based on GitHub Linguist but simplified for common cases
        self.language_mappings = {
            # Web Frontend
            ".js": "JavaScript",
            ".jsx": "JavaScript",
            ".ts": "TypeScript",
            ".tsx": "TypeScript",
            ".vue": "Vue",
            ".html": "HTML",
            ".htm": "HTML",
            ".css": "CSS",
            ".scss": "SCSS",
            ".sass": "Sass",
            ".less": "Less",
            # Backend Languages
            ".py": "Python",
            ".java": "Java",
            ".kt": "Kotlin",
            ".scala": "Scala",
            ".go": "Go",
            ".rs": "Rust",
            ".rb": "Ruby",
            ".php": "PHP",
            ".cs": "C#",
            ".fs": "F#",
            ".vb": "Visual Basic",
            ".cpp": "C++",
            ".cc": "C++",
            ".cxx": "C++",
            ".c": "C",
            ".h": "C/C++",
            ".hpp": "C++",
            # Mobile
            ".swift": "Swift",
            ".m": "Objective-C",
            ".mm": "Objective-C++",
            ".dart": "Dart",
            # Data & Config
            ".sql": "SQL",
            ".json": "JSON",
            ".yaml": "YAML",
            ".yml": "YAML",
            ".xml": "XML",
            ".toml": "TOML",
            ".ini": "INI",
            ".env": "Environment",
            ".properties": "Properties",
            # Shell & Scripting
            ".sh": "Shell",
            ".bash": "Bash",
            ".zsh": "Zsh",
            ".fish": "Fish",
            ".ps1": "PowerShell",
            ".bat": "Batch",
            ".cmd": "Batch",
            # Documentation
            ".md": "Markdown",
            ".rst": "reStructuredText",
            ".txt": "Text",
            ".adoc": "AsciiDoc",
            # Build & CI
            ".dockerfile": "Dockerfile",
            ".gradle": "Gradle",
            ".maven": "Maven",
            ".cmake": "CMake",
            ".make": "Makefile",
            # Misc
            ".r": "R",
            ".jl": "Julia",
            ".ex": "Elixir",
            ".exs": "Elixir",
            ".erl": "Erlang",
            ".hrl": "Erlang",
            ".clj": "Clojure",
            ".cljs": "ClojureScript",
            ".hs": "Haskell",
            ".elm": "Elm",
            ".lua": "Lua",
            ".pl": "Perl",
            ".pm": "Perl",
        }

        # Directory patterns to activity type mappings
        self.directory_activity_patterns = {
            # Frontend/UI patterns
            "ui": [
                "ui/",
                "frontend/",
                "client/",
                "web/",
                "www/",
                "public/",
                "assets/",
                "static/",
                "components/",
                "views/",
                "pages/",
                "templates/",
                "layouts/",
                "styles/",
                "css/",
                "js/",
                "javascript/",
                "typescript/",
                "react/",
                "vue/",
                "angular/",
            ],
            # Backend/API patterns
            "api": [
                "api/",
                "backend/",
                "server/",
                "service/",
                "services/",
                "controllers/",
                "handlers/",
                "routes/",
                "endpoints/",
                "middleware/",
                "auth/",
                "authentication/",
                "authorization/",
                "business/",
                "domain/",
                "core/",
                "logic/",
            ],
            # Database patterns
            "database": [
                "database/",
                "db/",
                "data/",
                "models/",
                "entities/",
                "repositories/",
                "dao/",
                "migrations/",
                "schema/",
                "seeds/",
                "fixtures/",
                "sql/",
                "queries/",
            ],
            # Testing patterns
            "test": [
                "test/",
                "tests/",
                "testing/",
                "spec/",
                "specs/",
                "__tests__/",
                "e2e/",
                "integration/",
                "unit/",
                "fixtures/",
                "mocks/",
                "stubs/",
            ],
            # Documentation patterns
            "docs": [
                "docs/",
                "doc/",
                "documentation/",
                "readme/",
                "guides/",
                "tutorials/",
                "examples/",
                "samples/",
                "wiki/",
                "help/",
                "manual/",
            ],
            # Infrastructure/DevOps patterns
            "infrastructure": [
                "infrastructure/",
                "infra/",
                "ops/",
                "devops/",
                "deploy/",
                "deployment/",
                "k8s/",
                "kubernetes/",
                "docker/",
                "terraform/",
                "ansible/",
                "helm/",
                "ci/",
                "cd/",
                ".github/",
                ".gitlab/",
                "jenkins/",
                "scripts/",
                "tools/",
                "utilities/",
                "bin/",
            ],
            # Configuration patterns
            "config": [
                "config/",
                "configuration/",
                "settings/",
                "env/",
                "environment/",
                "properties/",
                "resources/",
                "assets/config/",
                "etc/",
            ],
            # Build patterns
            "build": [
                "build/",
                "dist/",
                "target/",
                "out/",
                "output/",
                "generated/",
                "artifacts/",
                "release/",
                "gradle/",
                "maven/",
                "npm/",
                "node_modules/",
            ],
            # Mobile patterns
            "mobile": [
                "mobile/",
                "app/",
                "android/",
                "ios/",
                "flutter/",
                "react-native/",
                "cordova/",
                "phonegap/",
                "ionic/",
            ],
        }

        # File patterns for generated/binary content detection
        self.generated_patterns = [
            # Compiled/Generated files
            r"\.min\.(js|css)$",
            r"\.bundle\.(js|css)$",
            r"\.generated\.",
            r"\.g\.(cs|java|py)$",
            r"_pb2\.py$",  # Protocol buffer generated files
            r"\.pb\.go$",
            # Build artifacts
            r"\.(class|o|obj|exe|dll|so|dylib)$",
            r"\.a$",  # Static libraries
            r"\.jar$",
            r"\.war$",
            r"\.ear$",
            # Package files
            r"package-lock\.json$",
            r"yarn\.lock$",
            r"Gemfile\.lock$",
            r"composer\.lock$",
            r"Pipfile\.lock$",
            # IDE/Editor files
            r"\.(idea|vscode|settings)/",
            r"\.swp$",
            r"\.swo$",
            r"~$",
            # OS files
            r"\.DS_Store$",
            r"Thumbs\.db$",
            r"desktop\.ini$",
            # Log files
            r"\.(log|logs)$",
            r"\.log\.",
        ]

        # Binary file extensions
        self.binary_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".ico",
            ".svg",
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".zip",
            ".tar",
            ".gz",
            ".bz2",
            ".7z",
            ".rar",
            ".mp3",
            ".mp4",
            ".avi",
            ".mov",
            ".wav",
            ".flv",
            ".ttf",
            ".otf",
            ".woff",
            ".woff2",
            ".eot",
            ".bin",
            ".dat",
            ".db",
            ".sqlite",
            ".sqlite3",
        }

        # Compile regex patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching."""
        self.compiled_generated_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.generated_patterns
        ]

    def analyze_commit_files(self, file_paths: list[str]) -> dict[str, any]:
        """Analyze a list of file paths from a commit.

        This method provides comprehensive analysis of files changed in a commit,
        including language detection, activity classification, and metadata extraction.

        Args:
            file_paths: List of file paths from a git commit

        Returns:
            Dictionary containing:
            - languages: Counter of programming languages
            - activities: Counter of development activities
            - primary_language: Most common language (or None)
            - primary_activity: Most common activity (or None)
            - file_count: Total number of files
            - generated_count: Number of generated/binary files
            - generated_ratio: Ratio of generated to total files
            - language_diversity: Number of unique languages
            - activity_diversity: Number of unique activities
            - file_types: Counter of file extensions
            - is_multilingual: Whether multiple languages are involved
            - is_cross_functional: Whether multiple activities are involved
        """
        if not file_paths:
            return self._empty_analysis_result()

        languages = Counter()
        activities = Counter()
        file_types = Counter()
        generated_count = 0

        for file_path in file_paths:
            # Analyze individual file
            file_analysis = self._analyze_single_file(file_path)

            # Aggregate language information
            if file_analysis["language"]:
                languages[file_analysis["language"]] += 1

            # Aggregate activity information
            for activity in file_analysis["activities"]:
                activities[activity] += 1

            # Track file extensions
            file_types[file_analysis["extension"]] += 1

            # Count generated/binary files
            if file_analysis["is_generated"] or file_analysis["is_binary"]:
                generated_count += 1

        # Calculate derived metrics
        total_files = len(file_paths)
        generated_ratio = generated_count / total_files if total_files > 0 else 0.0

        # Determine primary language and activity
        primary_language = languages.most_common(1)[0][0] if languages else None
        primary_activity = activities.most_common(1)[0][0] if activities else None

        # Calculate diversity metrics
        language_diversity = len(languages)
        activity_diversity = len(activities)

        return {
            "languages": languages,
            "activities": activities,
            "primary_language": primary_language,
            "primary_activity": primary_activity,
            "file_count": total_files,
            "generated_count": generated_count,
            "generated_ratio": generated_ratio,
            "language_diversity": language_diversity,
            "activity_diversity": activity_diversity,
            "file_types": file_types,
            "is_multilingual": language_diversity > 1,
            "is_cross_functional": activity_diversity > 1,
        }

    def _analyze_single_file(self, file_path: str) -> dict[str, any]:
        """Analyze a single file path.

        Args:
            file_path: Path to analyze

        Returns:
            Dictionary with file analysis results
        """
        path_obj = Path(file_path)
        extension = path_obj.suffix.lower()

        # Detect language from extension
        language = self.language_mappings.get(extension)

        # Handle special cases for files without extensions
        if not language and not extension:
            filename = path_obj.name.lower()
            if filename in ["dockerfile", "makefile", "rakefile", "gemfile"]:
                language = filename.title()
            elif filename.startswith("dockerfile"):
                language = "Dockerfile"

        # Detect activities from directory patterns
        activities = self._classify_directory_activities(file_path)

        # Check if file is generated or binary
        is_generated = any(
            pattern.search(file_path) for pattern in self.compiled_generated_patterns
        )
        is_binary = extension in self.binary_extensions

        return {
            "language": language,
            "activities": activities,
            "extension": extension,
            "is_generated": is_generated,
            "is_binary": is_binary,
            "filename": path_obj.name,
            "directory": str(path_obj.parent) if path_obj.parent != Path(".") else "",
        }

    def _classify_directory_activities(self, file_path: str) -> list[str]:
        """Classify development activities based on directory patterns.

        Args:
            file_path: File path to analyze

        Returns:
            List of activity types that match the file path
        """
        activities = []
        normalized_path = file_path.lower().replace("\\", "/")

        for activity, patterns in self.directory_activity_patterns.items():
            for pattern in patterns:
                if pattern in normalized_path:
                    activities.append(activity)
                    break  # Don't add the same activity multiple times

        # If no specific activity detected, classify as 'general'
        if not activities:
            activities = ["general"]

        return activities

    def _empty_analysis_result(self) -> dict[str, any]:
        """Return empty analysis result structure."""
        return {
            "languages": Counter(),
            "activities": Counter(),
            "primary_language": None,
            "primary_activity": None,
            "file_count": 0,
            "generated_count": 0,
            "generated_ratio": 0.0,
            "language_diversity": 0,
            "activity_diversity": 0,
            "file_types": Counter(),
            "is_multilingual": False,
            "is_cross_functional": False,
        }

    def get_language_category(self, language: str) -> str:
        """Get high-level category for a programming language.

        Args:
            language: Programming language name

        Returns:
            Language category (frontend, backend, mobile, data, etc.)
        """
        frontend_languages = {
            "JavaScript",
            "TypeScript",
            "HTML",
            "CSS",
            "SCSS",
            "Sass",
            "Less",
            "Vue",
        }
        backend_languages = {
            "Python",
            "Java",
            "Go",
            "Rust",
            "Ruby",
            "PHP",
            "C#",
            "C++",
            "C",
            "Scala",
            "Kotlin",
        }
        mobile_languages = {"Swift", "Objective-C", "Objective-C++", "Kotlin", "Dart"}
        data_languages = {"SQL", "R", "Julia", "Python"}  # Python can be both backend and data

        if language in frontend_languages:
            return "frontend"
        elif language in backend_languages:
            return "backend"
        elif language in mobile_languages:
            return "mobile"
        elif language in data_languages:
            return "data"
        else:
            return "other"

    def get_supported_languages(self) -> list[str]:
        """Get list of all supported programming languages.

        Returns:
            Sorted list of supported language names
        """
        return sorted(set(self.language_mappings.values()))

    def get_supported_activities(self) -> list[str]:
        """Get list of all supported activity types.

        Returns:
            Sorted list of supported activity types
        """
        return sorted(self.directory_activity_patterns.keys())
