"""Data models and configuration schemas for qualitative analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class QualitativeCommitData:
    """Enhanced commit data with qualitative analysis results.

    This class extends basic commit information with semantic analysis results
    including change type, business domain, risk assessment, and processing metadata.
    """

    # Existing commit data from GitFlow Analytics
    hash: str
    message: str
    author_name: str
    author_email: str
    timestamp: datetime
    files_changed: list[str]
    insertions: int
    deletions: int

    # New qualitative analysis fields
    change_type: str  # feature|bugfix|refactor|docs|test|chore|security|hotfix|config
    change_type_confidence: float  # 0.0-1.0
    business_domain: str  # frontend|backend|database|infrastructure|mobile|devops|unknown
    domain_confidence: float  # 0.0-1.0
    risk_level: str  # low|medium|high|critical
    risk_factors: list[str]  # List of identified risk factors
    intent_signals: dict[str, Any]  # Intent analysis results
    collaboration_patterns: dict[str, Any]  # Team interaction patterns
    technical_context: dict[str, Any]  # Technical context information

    # Processing metadata
    processing_method: str  # 'nlp' or 'llm'
    processing_time_ms: float
    confidence_score: float  # Overall confidence in analysis

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "hash": self.hash,
            "message": self.message,
            "author_name": self.author_name,
            "author_email": self.author_email,
            "timestamp": self.timestamp.isoformat(),
            "files_changed": self.files_changed,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "change_type": self.change_type,
            "change_type_confidence": self.change_type_confidence,
            "business_domain": self.business_domain,
            "domain_confidence": self.domain_confidence,
            "risk_level": self.risk_level,
            "risk_factors": self.risk_factors,
            "intent_signals": self.intent_signals,
            "collaboration_patterns": self.collaboration_patterns,
            "technical_context": self.technical_context,
            "processing_method": self.processing_method,
            "processing_time_ms": self.processing_time_ms,
            "confidence_score": self.confidence_score,
        }


@dataclass
class ChangeTypeConfig:
    """Configuration for change type classification."""

    min_confidence: float = 0.7
    semantic_weight: float = 0.6  # Weight for semantic features
    file_pattern_weight: float = 0.4  # Weight for file pattern signals
    enable_custom_patterns: bool = True
    custom_patterns: dict[str, dict[str, list[str]]] = field(default_factory=dict)


@dataclass
class IntentConfig:
    """Configuration for intent analysis."""

    urgency_keywords: dict[str, list[str]] = field(
        default_factory=lambda: {
            "critical": ["critical", "urgent", "hotfix", "emergency", "immediate"],
            "important": ["important", "priority", "asap", "needed"],
            "routine": ["routine", "regular", "normal", "standard"],
        }
    )
    confidence_threshold: float = 0.6
    sentiment_analysis: bool = True


@dataclass
class DomainConfig:
    """Configuration for domain classification."""

    file_patterns: dict[str, list[str]] = field(
        default_factory=lambda: {
            "frontend": ["*.js", "*.jsx", "*.ts", "*.tsx", "*.vue", "*.html", "*.css", "*.scss"],
            "backend": ["*.py", "*.java", "*.go", "*.rb", "*.php", "*.cs", "*.cpp"],
            "database": ["*.sql", "migrations/*", "schema/*", "**/models/**"],
            "infrastructure": ["Dockerfile", "*.yaml", "*.yml", "terraform/*", "*.tf"],
            "mobile": ["*.swift", "*.kt", "*.java", "android/*", "ios/*"],
            "devops": ["*.yml", "*.yaml", "ci/*", ".github/*", "docker/*"],
        }
    )
    keyword_patterns: dict[str, list[str]] = field(
        default_factory=lambda: {
            "frontend": ["ui", "component", "styling", "interface", "layout"],
            "backend": ["api", "endpoint", "service", "server", "logic"],
            "database": ["query", "schema", "migration", "data", "model"],
            "infrastructure": ["deploy", "config", "environment", "setup"],
            "mobile": ["android", "ios", "mobile", "app"],
            "devops": ["build", "pipeline", "deploy", "ci", "docker"],
        }
    )
    min_confidence: float = 0.6


@dataclass
class RiskConfig:
    """Configuration for risk analysis."""

    high_risk_patterns: list[str] = field(
        default_factory=lambda: [
            # Security-related patterns
            "password",
            "secret",
            "key",
            "token",
            "auth",
            "security",
            # Critical system patterns
            "production",
            "prod",
            "critical",
            "emergency",
            # Infrastructure patterns
            "database",
            "migration",
            "schema",
            "deploy",
            # Large change patterns
            "refactor",
            "rewrite",
            "restructure",
        ]
    )
    medium_risk_patterns: list[str] = field(
        default_factory=lambda: [
            "config",
            "configuration",
            "settings",
            "environment",
            "api",
            "endpoint",
            "service",
            "integration",
        ]
    )
    file_risk_patterns: dict[str, str] = field(
        default_factory=lambda: {
            # High risk file patterns
            "**/*prod*": "high",
            "**/migrations/**": "high",
            "**/schema/**": "high",
            "Dockerfile": "medium",
            "*.yml": "medium",
            "*.yaml": "medium",
            "**/*config*": "medium",
        }
    )
    size_thresholds: dict[str, int] = field(
        default_factory=lambda: {
            "large_commit_files": 20,  # Files changed
            "large_commit_lines": 500,  # Lines changed
            "massive_commit_lines": 2000,  # Very large changes
        }
    )


@dataclass
class NLPConfig:
    """Configuration for NLP processing engine."""

    spacy_model: str = "en_core_web_sm"
    spacy_batch_size: int = 1000
    fast_mode: bool = True  # Disable parser/NER for speed

    # Component configurations
    change_type_config: ChangeTypeConfig = field(default_factory=ChangeTypeConfig)
    intent_config: IntentConfig = field(default_factory=IntentConfig)
    domain_config: DomainConfig = field(default_factory=DomainConfig)
    risk_config: RiskConfig = field(default_factory=RiskConfig)

    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4


@dataclass
class LLMConfig:
    """Configuration for LLM fallback processing via OpenRouter."""

    # OpenRouter API settings
    openrouter_api_key: str = "${OPENROUTER_API_KEY}"
    base_url: str = "https://openrouter.ai/api/v1"

    # Model selection strategy
    primary_model: str = "anthropic/claude-3-haiku"  # Fast, cheap classification
    fallback_model: str = "meta-llama/llama-3.1-8b-instruct:free"  # Free fallback
    complex_model: str = "anthropic/claude-3-sonnet"  # For complex cases

    # Model routing thresholds
    complexity_threshold: float = 0.5  # Route complex cases to better model
    cost_threshold_per_1k: float = 0.01  # Max cost per 1k commits

    # Processing settings
    max_tokens: int = 1000
    temperature: float = 0.1

    # Batching settings
    max_group_size: int = 10  # Process up to 10 commits per batch
    similarity_threshold: float = 0.8  # Group similar commits together

    # Rate limiting
    requests_per_minute: int = 200  # Higher limit with OpenRouter
    max_retries: int = 3

    # Cost control
    max_daily_cost: float = 5.0  # Max daily spend in USD
    enable_cost_tracking: bool = True


@dataclass
class CacheConfig:
    """Configuration for qualitative analysis caching."""

    cache_dir: str = ".qualitative_cache"
    semantic_cache_size: int = 10000  # Max cached patterns
    pattern_cache_ttl_hours: int = 168  # 1 week

    # Learning settings
    enable_pattern_learning: bool = True
    learning_threshold: int = 10  # Min examples to learn pattern
    confidence_boost_factor: float = 0.1  # Boost for learned patterns

    # Cache optimization
    enable_compression: bool = True
    max_cache_size_mb: int = 100


@dataclass
class QualitativeConfig:
    """Main configuration for qualitative analysis system.

    This configuration orchestrates the entire qualitative analysis pipeline,
    balancing performance, accuracy, and cost through intelligent NLP and
    strategic LLM usage.
    """

    # Processing settings
    enabled: bool = True
    batch_size: int = 1000  # Commits processed per batch
    max_llm_fallback_pct: float = 0.15  # Max 15% of commits use LLM
    confidence_threshold: float = 0.7  # Min confidence for NLP results

    # Component configurations
    nlp_config: NLPConfig = field(default_factory=NLPConfig)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)

    # Performance monitoring
    enable_performance_tracking: bool = True
    target_processing_time_ms: float = 2.0  # Target per-commit processing time

    # Quality settings
    min_overall_confidence: float = 0.6  # Min confidence for any result
    enable_quality_feedback: bool = True  # Learn from corrections

    def validate(self) -> list[str]:
        """Validate configuration and return any warnings.

        Returns:
            List of validation warning messages.
        """
        warnings = []

        if self.max_llm_fallback_pct > 0.3:
            warnings.append("LLM fallback percentage > 30% may result in high costs")

        if self.confidence_threshold > 0.9:
            warnings.append("Very high confidence threshold may route too many commits to LLM")

        if self.batch_size > 5000:
            warnings.append("Large batch size may cause memory issues")

        # Validate LLM config if API key is set
        if (
            self.llm_config.openrouter_api_key
            and self.llm_config.openrouter_api_key != "${OPENROUTER_API_KEY}"
        ) and self.llm_config.max_daily_cost < 1.0:
            warnings.append("Very low daily cost limit may restrict LLM usage")

        return warnings
