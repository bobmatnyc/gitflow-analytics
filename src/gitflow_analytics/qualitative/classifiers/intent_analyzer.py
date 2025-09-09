"""Intent analyzer for extracting developer intent and urgency from commits."""

import importlib.util
import logging
import re
from collections import defaultdict
from typing import Any

from ..models.schemas import IntentConfig

# Check if spacy is available without importing it
SPACY_AVAILABLE = importlib.util.find_spec("spacy") is not None

if SPACY_AVAILABLE:
    from spacy.tokens import Doc
else:
    Doc = Any


class IntentAnalyzer:
    """Analyze commit messages to extract developer intent and urgency signals.

    This analyzer identifies:
    - Urgency level (critical, important, routine)
    - Intent confidence (how clear the intent is)
    - Emotional tone (frustrated, confident, uncertain)
    - Planning signals (TODO, FIXME, temporary fixes)
    - Collaboration signals (pair programming, code review)
    """

    def __init__(self, config: IntentConfig):
        """Initialize intent analyzer.

        Args:
            config: Configuration for intent analysis
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Urgency keyword patterns from config
        self.urgency_keywords = config.urgency_keywords

        # Confidence indicators
        self.confidence_indicators = {
            "high_confidence": {
                "definitely",
                "clearly",
                "obviously",
                "certainly",
                "confirmed",
                "verified",
                "tested",
                "working",
                "complete",
                "finished",
                "implement",
                "solution",
                "resolve",
            },
            "low_confidence": {
                "maybe",
                "perhaps",
                "possibly",
                "might",
                "could",
                "should",
                "try",
                "attempt",
                "experiment",
                "test",
                "temporary",
                "quick",
                "hack",
                "workaround",
                "temp",
            },
            "uncertain": {
                "not sure",
                "unclear",
                "confusing",
                "weird",
                "strange",
                "unexpected",
                "unsure",
                "investigation",
                "debug",
                "investigate",
            },
        }

        # Emotional tone indicators
        self.tone_indicators = {
            "frustrated": {
                "annoying",
                "frustrating",
                "stupid",
                "broken",
                "terrible",
                "awful",
                "hate",
                "annoyed",
                "ugh",
                "argh",
                "damn",
                "wtf",
            },
            "confident": {
                "great",
                "excellent",
                "perfect",
                "awesome",
                "clean",
                "elegant",
                "nice",
                "good",
                "better",
                "improved",
                "optimized",
            },
            "cautious": {
                "careful",
                "cautious",
                "gentle",
                "safe",
                "conservative",
                "minimal",
                "small",
                "incremental",
                "gradual",
            },
        }

        # Planning and TODO indicators
        self.planning_indicators = {
            "todo": {
                "todo",
                "fixme",
                "hack",
                "temporary",
                "temp",
                "later",
                "placeholder",
                "stub",
                "incomplete",
                "wip",
            },
            "future_work": {
                "future",
                "later",
                "eventually",
                "someday",
                "next",
                "upcoming",
                "planned",
                "roadmap",
            },
            "immediate": {
                "now",
                "immediate",
                "urgent",
                "asap",
                "quickly",
                "fast",
                "emergency",
                "critical",
                "hotfix",
            },
        }

        # Collaboration indicators
        self.collaboration_indicators = {
            "pair_programming": {
                "pair",
                "pairing",
                "together",
                "with",
                "collaborative",
                "co-authored",
                "mob",
                "mobbing",
            },
            "code_review": {
                "review",
                "feedback",
                "suggestion",
                "requested",
                "comment",
                "pr",
                "pull request",
                "merge request",
            },
            "help_seeking": {
                "help",
                "assistance",
                "advice",
                "guidance",
                "input",
                "thoughts",
                "opinions",
                "feedback",
            },
        }

        # Technical complexity indicators
        self.complexity_indicators = {
            "simple": {
                "simple",
                "easy",
                "quick",
                "minor",
                "small",
                "tiny",
                "straightforward",
                "basic",
            },
            "complex": {
                "complex",
                "complicated",
                "difficult",
                "challenging",
                "major",
                "significant",
                "substantial",
                "extensive",
            },
            "refactoring": {
                "refactor",
                "restructure",
                "reorganize",
                "cleanup",
                "simplify",
                "optimize",
                "improve",
            },
        }

    def analyze(self, message: str, doc: Doc) -> dict[str, Any]:
        """Analyze commit message for intent signals.

        Args:
            message: Commit message
            doc: spaCy processed document (may be None)

        Returns:
            Dictionary with intent analysis results
        """
        if not message:
            return {
                "urgency": "routine",
                "confidence": 0.0,
                "tone": "neutral",
                "planning_stage": "implementation",
                "collaboration_signals": [],
                "complexity": "moderate",
                "signals": [],
            }

        message_lower = message.lower()

        # Extract all signals
        urgency = self._analyze_urgency(message_lower)
        confidence_info = self._analyze_confidence(message_lower, doc)
        tone = self._analyze_tone(message_lower)
        planning = self._analyze_planning_stage(message_lower)
        collaboration = self._analyze_collaboration(message_lower)
        complexity = self._analyze_complexity(message_lower)

        # Collect all detected signals
        all_signals = []
        all_signals.extend(urgency.get("signals", []))
        all_signals.extend(confidence_info.get("signals", []))
        all_signals.extend(tone.get("signals", []))
        all_signals.extend(planning.get("signals", []))
        all_signals.extend(collaboration.get("signals", []))
        all_signals.extend(complexity.get("signals", []))

        return {
            "urgency": urgency["level"],
            "confidence": confidence_info["score"],
            "tone": tone["dominant_tone"],
            "planning_stage": planning["stage"],
            "collaboration_signals": collaboration["types"],
            "complexity": complexity["level"],
            "signals": all_signals,
            "detailed_analysis": {
                "urgency_breakdown": urgency,
                "confidence_breakdown": confidence_info,
                "tone_breakdown": tone,
                "planning_breakdown": planning,
                "collaboration_breakdown": collaboration,
                "complexity_breakdown": complexity,
            },
        }

    def _analyze_urgency(self, message: str) -> dict[str, Any]:
        """Analyze urgency level from message content.

        Args:
            message: Lowercase commit message

        Returns:
            Dictionary with urgency analysis
        """
        signals = []
        urgency_scores = defaultdict(float)

        # Check configured urgency keywords
        for urgency_level, keywords in self.urgency_keywords.items():
            for keyword in keywords:
                if keyword.lower() in message:
                    signals.append(f"urgency:{urgency_level}:{keyword}")
                    urgency_scores[urgency_level] += 1.0

        # Additional urgency patterns
        urgent_patterns = [
            (r"\b(urgent|critical|emergency|asap|immediate)\b", "critical", 2.0),
            (r"\b(important|priority|needed|required)\b", "important", 1.5),
            (r"\b(hotfix|quickfix|patch)\b", "critical", 2.0),
            (r"\b(breaking|major)\b", "important", 1.5),
            (r"\b(minor|small|tiny)\b", "routine", 0.5),
        ]

        for pattern, level, weight in urgent_patterns:
            if re.search(pattern, message):
                signals.append(f"urgency_pattern:{level}:{pattern}")
                urgency_scores[level] += weight

        # Determine dominant urgency level
        if urgency_scores:
            dominant_urgency = max(urgency_scores.keys(), key=lambda k: urgency_scores[k])
        else:
            dominant_urgency = "routine"

        return {"level": dominant_urgency, "scores": dict(urgency_scores), "signals": signals}

    def _analyze_confidence(self, message: str, doc: Doc) -> dict[str, Any]:
        """Analyze confidence level in the commit.

        Args:
            message: Lowercase commit message
            doc: spaCy processed document

        Returns:
            Dictionary with confidence analysis
        """
        signals = []
        confidence_score = 0.5  # Start with neutral confidence

        # Check confidence indicators
        for confidence_type, keywords in self.confidence_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in message)
            if matches > 0:
                signals.append(f"confidence:{confidence_type}:{matches}")

                if confidence_type == "high_confidence":
                    confidence_score += matches * 0.2
                elif confidence_type == "low_confidence":
                    confidence_score -= matches * 0.15
                elif confidence_type == "uncertain":
                    confidence_score -= matches * 0.25

        # Check message structure and completeness
        if len(message.split()) >= 5:  # Detailed message
            confidence_score += 0.1
            signals.append("confidence:detailed_message")
        elif len(message.split()) <= 2:  # Very brief message
            confidence_score -= 0.1
            signals.append("confidence:brief_message")

        # Check for question marks (uncertainty)
        if "?" in message:
            confidence_score -= 0.2
            signals.append("confidence:contains_question")

        # Check for ellipsis or incomplete thoughts
        if "..." in message or message.endswith("."):
            confidence_score -= 0.1
            signals.append("confidence:incomplete_thought")

        # Normalize confidence score
        confidence_score = max(0.0, min(1.0, confidence_score))

        return {
            "score": confidence_score,
            "level": (
                "high" if confidence_score > 0.7 else "medium" if confidence_score > 0.4 else "low"
            ),
            "signals": signals,
        }

    def _analyze_tone(self, message: str) -> dict[str, Any]:
        """Analyze emotional tone of the commit message.

        Args:
            message: Lowercase commit message

        Returns:
            Dictionary with tone analysis
        """
        signals = []
        tone_scores = defaultdict(float)

        # Check tone indicators
        for tone_type, keywords in self.tone_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in message)
            if matches > 0:
                signals.append(f"tone:{tone_type}:{matches}")
                tone_scores[tone_type] += matches

        # Check punctuation for tone
        if "!" in message:
            tone_scores["confident"] += 0.5
            signals.append("tone:exclamation_mark")
        elif "..." in message:
            tone_scores["cautious"] += 0.5
            signals.append("tone:ellipsis")

        # Determine dominant tone
        if tone_scores:
            dominant_tone = max(tone_scores.keys(), key=lambda k: tone_scores[k])
        else:
            dominant_tone = "neutral"

        return {"dominant_tone": dominant_tone, "scores": dict(tone_scores), "signals": signals}

    def _analyze_planning_stage(self, message: str) -> dict[str, Any]:
        """Analyze what stage of planning/development this commit represents.

        Args:
            message: Lowercase commit message

        Returns:
            Dictionary with planning stage analysis
        """
        signals = []
        stage_scores = defaultdict(float)

        # Check planning indicators
        for stage_type, keywords in self.planning_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in message)
            if matches > 0:
                signals.append(f"planning:{stage_type}:{matches}")
                stage_scores[stage_type] += matches

        # Additional stage indicators
        if any(word in message for word in ["start", "initial", "begin", "setup"]):
            stage_scores["initial"] = stage_scores.get("initial", 0) + 1
            signals.append("planning:initial_stage")

        if any(word in message for word in ["complete", "finish", "done", "final"]):
            stage_scores["completion"] = stage_scores.get("completion", 0) + 1
            signals.append("planning:completion_stage")

        # Determine stage
        if stage_scores:
            if "immediate" in stage_scores:
                stage = "immediate"
            elif "todo" in stage_scores:
                stage = "planning"
            elif "future_work" in stage_scores:
                stage = "future_planning"
            elif "completion" in stage_scores:
                stage = "completion"
            elif "initial" in stage_scores:
                stage = "initiation"
            else:
                stage = "implementation"
        else:
            stage = "implementation"

        return {"stage": stage, "scores": dict(stage_scores), "signals": signals}

    def _analyze_collaboration(self, message: str) -> dict[str, Any]:
        """Analyze collaboration signals in the commit message.

        Args:
            message: Lowercase commit message

        Returns:
            Dictionary with collaboration analysis
        """
        signals = []
        collaboration_types = []

        # Check collaboration indicators
        for collab_type, keywords in self.collaboration_indicators.items():
            matches = [keyword for keyword in keywords if keyword in message]
            if matches:
                signals.extend([f"collaboration:{collab_type}:{match}" for match in matches])
                collaboration_types.append(collab_type)

        # Check for co-author patterns
        if "co-authored-by:" in message or "with @" in message:
            collaboration_types.append("co_authored")
            signals.append("collaboration:co_authored")

        return {
            "types": collaboration_types,
            "signals": signals,
            "is_collaborative": len(collaboration_types) > 0,
        }

    def _analyze_complexity(self, message: str) -> dict[str, Any]:
        """Analyze technical complexity signals in the commit message.

        Args:
            message: Lowercase commit message

        Returns:
            Dictionary with complexity analysis
        """
        signals = []
        complexity_scores = defaultdict(float)

        # Check complexity indicators
        for complexity_type, keywords in self.complexity_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in message)
            if matches > 0:
                signals.append(f"complexity:{complexity_type}:{matches}")
                complexity_scores[complexity_type] += matches

        # Determine complexity level
        if complexity_scores:
            if complexity_scores.get("complex", 0) > complexity_scores.get("simple", 0):
                level = "complex"
            elif complexity_scores.get("simple", 0) > 0:
                level = "simple"
            elif complexity_scores.get("refactoring", 0) > 0:
                level = "moderate"  # Refactoring is usually moderate complexity
            else:
                level = "moderate"
        else:
            level = "moderate"

        return {"level": level, "scores": dict(complexity_scores), "signals": signals}
