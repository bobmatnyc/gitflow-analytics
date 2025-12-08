"""Prompt templates and generation for LLM commit classification.

This module manages all prompt engineering for commit classification,
including templates, versioning, and context preparation.

WHY: Centralizing prompt management allows for easy experimentation,
A/B testing, and optimization without modifying classifier logic.

DESIGN DECISIONS:
- Version prompts for tracking and rollback capability
- Support template variables for dynamic content
- Separate system prompts from user prompts
- Include few-shot examples for better accuracy
- Make prompts provider-agnostic
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class PromptVersion(Enum):
    """Versions of prompt templates for A/B testing and evolution.

    WHY: Track prompt versions to measure performance improvements
    and enable rollback if newer versions perform worse.
    """

    V1_SIMPLE = "v1_simple"  # Original simple prompt
    V2_STRUCTURED = "v2_structured"  # More structured with examples
    V3_CONTEXTUAL = "v3_contextual"  # Enhanced with file context
    V4_FEWSHOT = "v4_fewshot"  # Few-shot learning with examples


@dataclass
class PromptTemplate:
    """Template for generating classification prompts.

    WHY: Structured templates ensure consistent prompt formatting
    and make it easy to swap different prompt strategies.
    """

    version: PromptVersion
    system_prompt: str
    user_prompt_template: str
    few_shot_examples: Optional[list[dict[str, str]]] = None

    def format(self, **kwargs) -> tuple[str, str]:
        """Format the prompt with provided variables.

        Args:
            **kwargs: Variables to substitute in the template

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        user_prompt = self.user_prompt_template.format(**kwargs)
        return self.system_prompt, user_prompt


class PromptGenerator:
    """Generates prompts for commit classification.

    WHY: Encapsulates all prompt engineering logic, making it easy
    to experiment with different prompt strategies and optimize
    classification accuracy.
    """

    # Streamlined categories optimized for enterprise workflows
    CATEGORIES = {
        "feature": "New functionality, capabilities, enhancements, additions",
        "bugfix": "Fixes, errors, issues, crashes, bugs, corrections",
        "maintenance": "Configuration, chores, dependencies, cleanup, refactoring, updates",
        "integration": "Third-party services, APIs, webhooks, external systems",
        "content": "Text, copy, documentation, README updates, comments",
        "media": "Video, audio, streaming, players, visual assets, images",
        "localization": "Translations, i18n, l10n, regional adaptations",
    }

    # Prompt templates for different versions
    TEMPLATES = {
        PromptVersion.V1_SIMPLE: PromptTemplate(
            version=PromptVersion.V1_SIMPLE,
            system_prompt="You are a commit classification expert.",
            user_prompt_template="""Classify this commit message into one of these 7 categories:

{categories_desc}

Commit message: "{message}"{context_info}

Respond with only: CATEGORY_NAME confidence_score reasoning
Example: feature 0.85 adds new user authentication system

Response:""",
        ),
        PromptVersion.V2_STRUCTURED: PromptTemplate(
            version=PromptVersion.V2_STRUCTURED,
            system_prompt="""You are an expert at classifying git commit messages.
Your task is to categorize commits accurately based on their content and context.
Be precise and consistent in your classifications.""",
            user_prompt_template="""Task: Classify the following git commit into exactly ONE category.

Available Categories:
{categories_desc}

Commit Information:
- Message: "{message}"
{context_info}

Output Format: CATEGORY confidence reasoning
- CATEGORY: One of the 7 categories above (lowercase)
- confidence: Float between 0.0 and 1.0
- reasoning: Brief explanation (max 10 words)

Response:""",
        ),
        PromptVersion.V3_CONTEXTUAL: PromptTemplate(
            version=PromptVersion.V3_CONTEXTUAL,
            system_prompt="""You are a specialized git commit classifier with deep understanding
of software development patterns. Consider both the commit message and file context
to make accurate classifications.""",
            user_prompt_template="""Analyze this commit and classify it into the most appropriate category.

Categories (choose ONE):
{categories_desc}

Commit Details:
Message: "{message}"
{context_info}

Classification Rules:
1. Focus on the PRIMARY purpose of the commit
2. Consider file types and patterns for additional context
3. If multiple categories apply, choose the most significant one
4. Be confident in clear cases, conservative when ambiguous

Format: CATEGORY confidence reasoning
Response:""",
        ),
        PromptVersion.V4_FEWSHOT: PromptTemplate(
            version=PromptVersion.V4_FEWSHOT,
            system_prompt="""You are an expert commit classifier. Classify commits based on 
the examples provided and return results in the exact format shown.""",
            user_prompt_template="""Learn from these examples, then classify the new commit.

Examples:
{examples}

Categories:
{categories_desc}

Now classify this commit:
Message: "{message}"
{context_info}

Response (format: CATEGORY confidence reasoning):""",
            few_shot_examples=[
                {
                    "message": "feat: add user authentication",
                    "response": "feature 0.95 adds authentication functionality",
                },
                {
                    "message": "fix: resolve login crash",
                    "response": "bugfix 0.90 fixes crash issue",
                },
                {
                    "message": "chore: update dependencies",
                    "response": "maintenance 0.85 dependency updates",
                },
                {"message": "docs: update README", "response": "content 0.95 documentation update"},
                {
                    "message": "feat: add Spanish translations",
                    "response": "localization 0.90 adds language support",
                },
            ],
        ),
    }

    def __init__(self, version: PromptVersion = PromptVersion.V3_CONTEXTUAL):
        """Initialize prompt generator with specified version.

        Args:
            version: Prompt template version to use
        """
        self.version = version
        self.template = self.TEMPLATES[version]
        self.domain_terms = self._get_default_domain_terms()

    def _get_default_domain_terms(self) -> dict[str, list[str]]:
        """Get default domain-specific terms for context enhancement.

        WHY: Domain-specific terms help the LLM understand the context
        better and make more accurate classifications.
        """
        return {
            "media": [
                "video",
                "audio",
                "streaming",
                "player",
                "media",
                "content",
                "broadcast",
                "live",
                "recording",
                "episode",
                "program",
            ],
            "localization": [
                "translation",
                "i18n",
                "l10n",
                "locale",
                "language",
                "spanish",
                "french",
                "german",
                "italian",
                "portuguese",
                "multilingual",
            ],
            "integration": [
                "api",
                "webhook",
                "third-party",
                "external",
                "service",
                "integration",
                "sync",
                "import",
                "export",
                "connector",
            ],
        }

    def prepare_context(
        self, message: str, files_changed: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Prepare context information from commit data.

        Args:
            message: Commit message
            files_changed: Optional list of changed files

        Returns:
            Context dictionary with relevant information
        """
        context = {"file_extensions": [], "file_patterns": [], "domain_indicators": []}

        if files_changed:
            # Extract file extensions
            extensions = set()
            for file_path in files_changed:
                ext = Path(file_path).suffix.lower()
                if ext:
                    extensions.add(ext)
            context["file_extensions"] = list(extensions)

            # Look for specific file patterns
            patterns = []
            for file_path in files_changed:
                file_lower = file_path.lower()
                if any(
                    term in file_lower for term in ["config", "settings", ".env", ".yaml", ".json"]
                ):
                    patterns.append("configuration")
                elif any(term in file_lower for term in ["test", "spec", "__test__"]):
                    patterns.append("test")
                elif any(term in file_lower for term in ["doc", "readme", "changelog"]):
                    patterns.append("documentation")
                elif any(
                    term in file_lower for term in ["video", "audio", "media", ".mp4", ".mp3"]
                ):
                    patterns.append("media")
            context["file_patterns"] = list(set(patterns))

        # Check for domain-specific terms in message
        message_lower = message.lower()
        for domain, terms in self.domain_terms.items():
            if any(term in message_lower for term in terms):
                context["domain_indicators"].append(domain)

        return context

    def generate_prompt(
        self, message: str, files_changed: Optional[list[str]] = None, include_examples: bool = True
    ) -> tuple[str, str]:
        """Generate classification prompt for the given commit.

        Args:
            message: Commit message to classify
            files_changed: Optional list of changed files
            include_examples: Whether to include few-shot examples

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Prepare context
        context = self.prepare_context(message, files_changed)

        # Format context information
        context_info = self._format_context(context)

        # Format categories description
        categories_desc = "\n".join([f"- {cat}: {desc}" for cat, desc in self.CATEGORIES.items()])

        # Prepare examples if needed
        examples = ""
        if include_examples and self.template.few_shot_examples:
            examples = self._format_examples(self.template.few_shot_examples)

        # Format the prompt
        return self.template.format(
            message=message,
            context_info=context_info,
            categories_desc=categories_desc,
            examples=examples,
        )

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context information for inclusion in prompt.

        Args:
            context: Context dictionary

        Returns:
            Formatted context string
        """
        parts = []

        if context.get("file_extensions"):
            parts.append(f"File types: {', '.join(context['file_extensions'])}")

        if context.get("file_patterns"):
            parts.append(f"File patterns: {', '.join(context['file_patterns'])}")

        if context.get("domain_indicators"):
            parts.append(f"Domain indicators: {', '.join(context['domain_indicators'])}")

        if parts:
            return "\n" + "\n".join(parts)
        return ""

    def _format_examples(self, examples: list[dict[str, str]]) -> str:
        """Format few-shot examples for inclusion in prompt.

        Args:
            examples: List of example classifications

        Returns:
            Formatted examples string
        """
        formatted = []
        for i, example in enumerate(examples, 1):
            formatted.append(f'{i}. Message: "{example["message"]}"')
            formatted.append(f"   Response: {example['response']}")
        return "\n".join(formatted)

    def get_version_info(self) -> dict[str, Any]:
        """Get information about the current prompt version.

        Returns:
            Dictionary with version information
        """
        return {
            "version": self.version.value,
            "has_few_shot": bool(self.template.few_shot_examples),
            "num_examples": (
                len(self.template.few_shot_examples) if self.template.few_shot_examples else 0
            ),
            "categories": list(self.CATEGORIES.keys()),
        }
