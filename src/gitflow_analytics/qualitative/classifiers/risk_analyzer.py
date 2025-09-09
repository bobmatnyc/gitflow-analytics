"""Risk analyzer for assessing commit risk levels."""

import importlib.util
import logging
import re
from typing import Any

from ..models.schemas import RiskConfig

# Check if spacy is available without importing it
SPACY_AVAILABLE = importlib.util.find_spec("spacy") is not None

if SPACY_AVAILABLE:
    from spacy.tokens import Doc
else:
    Doc = Any


class RiskAnalyzer:
    """Analyze commits to assess risk level and identify risk factors.

    This analyzer evaluates multiple dimensions of risk:
    - Content risk: Security-sensitive keywords, critical system changes
    - Size risk: Large commits affecting many files/lines
    - Context risk: Production deployments, emergency fixes
    - Pattern risk: File patterns indicating high-risk areas

    Risk levels: low, medium, high, critical
    """

    def __init__(self, config: RiskConfig):
        """Initialize risk analyzer.

        Args:
            config: Configuration for risk analysis
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Compile file risk patterns for efficiency
        self._compile_file_patterns()

        # Additional risk patterns not in config
        self.critical_keywords = {
            "password",
            "secret",
            "key",
            "token",
            "credential",
            "auth",
            "admin",
            "root",
            "sudo",
            "permission",
            "access",
            "security",
        }

        self.production_keywords = {
            "production",
            "prod",
            "live",
            "release",
            "deploy",
            "deployment",
            "critical",
            "urgent",
            "emergency",
            "hotfix",
            "immediate",
        }

        self.database_keywords = {
            "database",
            "db",
            "migration",
            "schema",
            "table",
            "column",
            "index",
            "constraint",
            "trigger",
            "procedure",
        }

        # File extension risk mapping
        self.extension_risk = {
            # High risk extensions
            ".sql": "high",
            ".py": "medium",  # Could be config or critical logic
            ".js": "medium",
            ".php": "medium",
            ".java": "medium",
            ".cs": "medium",
            ".go": "medium",
            ".rb": "medium",
            # Configuration files
            ".yml": "medium",
            ".yaml": "medium",
            ".json": "medium",
            ".toml": "medium",
            ".ini": "medium",
            ".conf": "medium",
            ".config": "medium",
            # Low risk extensions
            ".md": "low",
            ".txt": "low",
            ".rst": "low",
            ".css": "low",
            ".scss": "low",
            ".less": "low",
        }

    def _compile_file_patterns(self) -> None:
        """Compile file risk patterns for efficient matching."""
        self.compiled_file_patterns = {}

        for pattern, risk_level in self.config.file_risk_patterns.items():
            try:
                # Convert glob pattern to regex
                regex_pattern = self._glob_to_regex(pattern)
                self.compiled_file_patterns[re.compile(regex_pattern, re.IGNORECASE)] = risk_level
            except re.error as e:
                self.logger.warning(f"Invalid risk pattern '{pattern}': {e}")

    def _glob_to_regex(self, pattern: str) -> str:
        """Convert glob pattern to regex."""
        pattern = pattern.replace(".", r"\.")
        pattern = pattern.replace("*", ".*")
        pattern = pattern.replace("?", ".")
        pattern = f"^{pattern}$"
        return pattern

    def assess(self, commit: dict[str, Any], doc: Doc) -> dict[str, Any]:
        """Assess risk level and identify risk factors for a commit.

        Args:
            commit: Commit dictionary with message, files, stats, etc.
            doc: spaCy processed document (may be None)

        Returns:
            Dictionary with 'level' and 'factors' keys
        """
        risk_factors = []
        risk_scores = []

        # Analyze message content for risk keywords
        message_risk = self._analyze_message_risk(commit.get("message", ""), doc)
        risk_factors.extend(message_risk["factors"])
        risk_scores.append(message_risk["score"])

        # Analyze file patterns for risk
        file_risk = self._analyze_file_risk(commit.get("files_changed", []))
        risk_factors.extend(file_risk["factors"])
        risk_scores.append(file_risk["score"])

        # Analyze commit size for risk
        size_risk = self._analyze_size_risk(commit)
        risk_factors.extend(size_risk["factors"])
        risk_scores.append(size_risk["score"])

        # Analyze timing and context
        context_risk = self._analyze_context_risk(commit)
        risk_factors.extend(context_risk["factors"])
        risk_scores.append(context_risk["score"])

        # Calculate overall risk level
        max_risk_score = max(risk_scores) if risk_scores else 0.0
        risk_level = self._score_to_level(max_risk_score)

        return {
            "level": risk_level,
            "factors": list(set(risk_factors)),  # Remove duplicates
            "score": max_risk_score,
            "breakdown": {
                "message_risk": message_risk["score"],
                "file_risk": file_risk["score"],
                "size_risk": size_risk["score"],
                "context_risk": context_risk["score"],
            },
        }

    def _analyze_message_risk(self, message: str, doc: Doc) -> dict[str, Any]:
        """Analyze commit message for risk indicators.

        Args:
            message: Commit message
            doc: spaCy processed document

        Returns:
            Dictionary with score and factors
        """
        if not message:
            return {"score": 0.0, "factors": []}

        message_lower = message.lower()
        factors = []
        risk_score = 0.0

        # Check for high-risk patterns
        for pattern in self.config.high_risk_patterns:
            if pattern.lower() in message_lower:
                factors.append(f"high_risk_keyword:{pattern}")
                risk_score = max(risk_score, 0.8)  # High risk

        # Check for medium-risk patterns
        for pattern in self.config.medium_risk_patterns:
            if pattern.lower() in message_lower:
                factors.append(f"medium_risk_keyword:{pattern}")
                risk_score = max(risk_score, 0.5)  # Medium risk

        # Check for critical security keywords
        for keyword in self.critical_keywords:
            if keyword in message_lower:
                factors.append(f"security_keyword:{keyword}")
                risk_score = max(risk_score, 0.9)  # Critical risk

        # Check for production-related keywords
        for keyword in self.production_keywords:
            if keyword in message_lower:
                factors.append(f"production_keyword:{keyword}")
                risk_score = max(risk_score, 0.7)  # High risk

        # Check for database-related keywords
        for keyword in self.database_keywords:
            if keyword in message_lower:
                factors.append(f"database_keyword:{keyword}")
                risk_score = max(risk_score, 0.6)  # Medium-high risk

        # Check for urgency indicators
        urgency_patterns = [
            r"\b(urgent|critical|emergency|asap|immediate)\b",
            r"\b(hotfix|quickfix|patch)\b",
            r"\b(breaking|major)\b",
        ]

        for pattern in urgency_patterns:
            if re.search(pattern, message_lower):
                factors.append(f"urgency_indicator:{pattern}")
                risk_score = max(risk_score, 0.6)

        return {"score": risk_score, "factors": factors}

    def _analyze_file_risk(self, files: list[str]) -> dict[str, Any]:
        """Analyze changed files for risk indicators.

        Args:
            files: List of file paths

        Returns:
            Dictionary with score and factors
        """
        if not files:
            return {"score": 0.0, "factors": []}

        factors = []
        risk_score = 0.0

        for file_path in files:
            file_lower = file_path.lower()

            # Check compiled file risk patterns
            for pattern, risk_level in self.compiled_file_patterns.items():
                if pattern.search(file_path):
                    factors.append(f"file_pattern:{risk_level}:{file_path}")
                    if risk_level == "critical":
                        risk_score = max(risk_score, 1.0)
                    elif risk_level == "high":
                        risk_score = max(risk_score, 0.8)
                    elif risk_level == "medium":
                        risk_score = max(risk_score, 0.5)

            # Check file extensions
            if "." in file_path:
                ext = "." + file_path.split(".")[-1].lower()
                if ext in self.extension_risk:
                    ext_risk = self.extension_risk[ext]
                    factors.append(f"file_extension:{ext_risk}:{ext}")
                    if ext_risk == "high":
                        risk_score = max(risk_score, 0.7)
                    elif ext_risk == "medium":
                        risk_score = max(risk_score, 0.4)

            # Check for sensitive file names
            sensitive_patterns = [
                r".*password.*",
                r".*secret.*",
                r".*key.*",
                r".*token.*",
                r".*config.*",
                r".*env.*",
                r".*credential.*",
            ]

            for pattern in sensitive_patterns:
                if re.search(pattern, file_lower):
                    factors.append(f"sensitive_filename:{file_path}")
                    risk_score = max(risk_score, 0.8)
                    break

        return {"score": risk_score, "factors": factors}

    def _analyze_size_risk(self, commit: dict[str, Any]) -> dict[str, Any]:
        """Analyze commit size for risk indicators.

        Args:
            commit: Commit dictionary

        Returns:
            Dictionary with score and factors
        """
        factors = []
        risk_score = 0.0

        files_changed = len(commit.get("files_changed", []))
        insertions = commit.get("insertions", 0)
        deletions = commit.get("deletions", 0)
        total_changes = insertions + deletions

        # Check file count thresholds
        if files_changed >= self.config.size_thresholds["large_commit_files"]:
            factors.append(f"large_file_count:{files_changed}")
            # Very large commits get higher risk score
            risk_score = max(risk_score, 0.8) if files_changed >= 50 else max(risk_score, 0.6)

        # Check line change thresholds
        if total_changes >= self.config.size_thresholds["massive_commit_lines"]:
            factors.append(f"massive_changes:{total_changes}")
            risk_score = max(risk_score, 0.9)
        elif total_changes >= self.config.size_thresholds["large_commit_lines"]:
            factors.append(f"large_changes:{total_changes}")
            risk_score = max(risk_score, 0.6)

        # Check deletion ratio (high deletion ratio can be risky)
        if total_changes > 0:
            deletion_ratio = deletions / total_changes
            if deletion_ratio > 0.7:  # More than 70% deletions
                factors.append(f"high_deletion_ratio:{deletion_ratio:.2f}")
                risk_score = max(risk_score, 0.5)

        return {"score": risk_score, "factors": factors}

    def _analyze_context_risk(self, commit: dict[str, Any]) -> dict[str, Any]:
        """Analyze commit context for risk indicators.

        Args:
            commit: Commit dictionary

        Returns:
            Dictionary with score and factors
        """
        factors = []
        risk_score = 0.0

        # Check branch context if available
        branch = commit.get("branch", "").lower()
        if branch:
            if any(term in branch for term in ["main", "master", "prod", "production"]):
                factors.append(f"main_branch:{branch}")
                risk_score = max(risk_score, 0.6)
            elif "hotfix" in branch:
                factors.append(f"hotfix_branch:{branch}")
                risk_score = max(risk_score, 0.8)

        # Check commit timing (if timestamp available)
        # Weekend/night commits might be higher risk
        timestamp = commit.get("timestamp")
        if timestamp:
            # This would require datetime analysis
            # For now, skip this check
            pass

        # Check for merge commits
        if commit.get("is_merge", False):
            factors.append("merge_commit")
            # Merges can be risky depending on what's being merged
            risk_score = max(risk_score, 0.3)

        return {"score": risk_score, "factors": factors}

    def _score_to_level(self, score: float) -> str:
        """Convert risk score to risk level.

        Args:
            score: Risk score (0.0 to 1.0)

        Returns:
            Risk level string
        """
        if score >= 0.9:
            return "critical"
        elif score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"

    def get_risk_statistics(self, commits: list[dict[str, Any]]) -> dict[str, Any]:
        """Get risk analysis statistics for a set of commits.

        Args:
            commits: List of commit dictionaries

        Returns:
            Dictionary with risk statistics
        """
        if not commits:
            return {"total_commits": 0}

        risk_levels = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        all_factors = []

        for commit in commits:
            # Quick risk assessment without full doc processing
            risk_result = self.assess(commit, None)
            risk_levels[risk_result["level"]] += 1
            all_factors.extend(risk_result["factors"])

        # Count factor frequencies
        factor_counts = {}
        for factor in all_factors:
            factor_type = factor.split(":")[0] if ":" in factor else factor
            factor_counts[factor_type] = factor_counts.get(factor_type, 0) + 1

        return {
            "total_commits": len(commits),
            "risk_distribution": risk_levels,
            "risk_percentages": {
                level: (count / len(commits)) * 100 for level, count in risk_levels.items()
            },
            "common_risk_factors": sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[
                :10
            ],
            "high_risk_commits": risk_levels["high"] + risk_levels["critical"],
            "high_risk_percentage": ((risk_levels["high"] + risk_levels["critical"]) / len(commits))
            * 100,
        }
