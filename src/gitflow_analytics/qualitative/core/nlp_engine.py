"""NLP processing engine using spaCy for fast commit analysis."""

import logging
import time
from datetime import datetime
from typing import Any

from ..classifiers.change_type import ChangeTypeClassifier
from ..classifiers.domain_classifier import DomainClassifier
from ..classifiers.intent_analyzer import IntentAnalyzer
from ..classifiers.risk_analyzer import RiskAnalyzer
from ..models.schemas import NLPConfig, QualitativeCommitData
from ..utils.metrics import PerformanceMetrics
from ..utils.text_processing import TextProcessor

try:
    import spacy
    from spacy.tokens import Doc

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    Doc = Any  # Type hint fallback


class NLPEngine:
    """Core NLP processing engine using spaCy for fast commit analysis.

    This engine provides the primary classification pipeline for commit analysis,
    handling 85-90% of commits through fast NLP processing without requiring
    expensive LLM calls.

    The engine orchestrates multiple specialized classifiers:
    - ChangeTypeClassifier: Determines commit type (feature, bugfix, etc.)
    - DomainClassifier: Identifies business domain (frontend, backend, etc.)
    - IntentAnalyzer: Extracts intent signals and urgency
    - RiskAnalyzer: Assesses commit risk level
    """

    def __init__(self, config: NLPConfig):
        """Initialize NLP engine with spaCy pipeline.

        Args:
            config: NLP configuration

        Raises:
            ImportError: If spaCy is not available
            OSError: If spaCy model is not installed
        """
        if not SPACY_AVAILABLE:
            # Create a temporary logger since self.logger doesn't exist yet
            temp_logger = logging.getLogger(__name__)
            temp_logger.warning(
                "spaCy is not available. NLP processing will be disabled. "
                "To enable ML features, install spaCy: pip install spacy"
            )
            raise ImportError(
                "spaCy is required for NLP processing. Install with: pip install spacy"
            )

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize spaCy pipeline
        self._init_spacy_pipeline()

        # Initialize text processor
        self.text_processor = TextProcessor()

        # Initialize classifiers
        self.change_classifier = ChangeTypeClassifier(config.change_type_config)
        self.domain_classifier = DomainClassifier(config.domain_config)
        self.intent_analyzer = IntentAnalyzer(config.intent_config)
        self.risk_analyzer = RiskAnalyzer(config.risk_config)

        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.processing_times = []

        self.logger.info(f"NLP engine initialized with model: {config.spacy_model}")

    def _init_spacy_pipeline(self) -> None:
        """Initialize spaCy NLP pipeline with optimizations."""
        try:
            self.nlp = spacy.load(self.config.spacy_model)

            # Optimize pipeline for speed if in fast mode
            if self.config.fast_mode:
                # Disable expensive components we don't need
                disabled_components = []
                if "parser" in self.nlp.pipe_names:
                    disabled_components.append("parser")
                if "ner" in self.nlp.pipe_names:
                    disabled_components.append("ner")

                if disabled_components:
                    self.nlp.disable_pipes(*disabled_components)
                    self.logger.info(f"Disabled spaCy components for speed: {disabled_components}")

        except OSError as e:
            self.logger.warning(
                f"spaCy model '{self.config.spacy_model}' not found. "
                f"ML features will be disabled. To enable, install with: python -m spacy download {self.config.spacy_model}"
            )
            # Raise the original error since the NLP engine requires spaCy
            raise OSError(
                f"spaCy model '{self.config.spacy_model}' not found. "
                f"Install with: python -m spacy download {self.config.spacy_model}"
            ) from e

    def process_batch(self, commits: list[dict[str, Any]]) -> list[QualitativeCommitData]:
        """Process a batch of commits efficiently using spaCy pipeline.

        This method leverages spaCy's batch processing capabilities to analyze
        multiple commit messages simultaneously for maximum efficiency.

        Args:
            commits: List of commit dictionaries with message, files_changed, etc.

        Returns:
            List of QualitativeCommitData with analysis results
        """
        if not commits:
            return []

        start_time = time.time()

        # Extract messages for batch processing
        messages = [commit.get("message", "") for commit in commits]

        # Process all messages through spaCy pipeline at once
        try:
            docs = list(
                self.nlp.pipe(
                    messages,
                    batch_size=self.config.spacy_batch_size,
                    disable=[] if not self.config.fast_mode else ["parser", "ner"],
                )
            )
        except Exception as e:
            self.logger.error(f"spaCy processing failed: {e}")
            # Fallback to individual processing
            docs = []
            for message in messages:
                try:
                    docs.append(self.nlp(message))
                except Exception:
                    # Create empty doc as fallback
                    docs.append(self.nlp(""))

        # Analyze each commit with its processed document
        results = []
        for commit, doc in zip(commits, docs):
            try:
                result = self._analyze_commit(commit, doc)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error analyzing commit {commit.get('hash', 'unknown')}: {e}")
                # Create fallback result
                results.append(self._create_fallback_result(commit))

        # Track performance
        processing_time = (time.time() - start_time) * 1000  # ms
        self.processing_times.append(processing_time)

        # Record metrics
        avg_confidence = sum(r.confidence_score for r in results) / len(results) if results else 0.0
        self.metrics.record_processing(
            operation="nlp_batch",
            processing_time_ms=processing_time,
            items_processed=len(commits),
            confidence_score=avg_confidence,
            method_used="nlp",
        )

        self.logger.debug(
            f"Processed {len(commits)} commits in {processing_time:.1f}ms "
            f"({len(commits) * 1000 / processing_time:.1f} commits/sec)"
        )

        return results

    def _analyze_commit(self, commit: dict[str, Any], doc: Doc) -> QualitativeCommitData:
        """Analyze a single commit with all classifiers.

        Args:
            commit: Commit dictionary with message, files, etc.
            doc: spaCy processed document

        Returns:
            QualitativeCommitData with analysis results
        """
        analysis_start = time.time()

        # Extract basic commit info
        message = commit.get("message", "")
        files_changed = commit.get("files_changed", [])

        # Run all classifiers
        change_type, change_confidence = self.change_classifier.classify(
            message, doc, files_changed
        )

        domain, domain_confidence = self.domain_classifier.classify(message, doc, files_changed)

        intent_signals = self.intent_analyzer.analyze(message, doc)

        risk_assessment = self.risk_analyzer.assess(commit, doc)

        # Calculate overall confidence score
        overall_confidence = self._calculate_overall_confidence(
            change_confidence, domain_confidence, intent_signals.get("confidence", 0.5)
        )

        # Extract technical context
        technical_context = {
            "file_patterns": self.text_processor.extract_file_patterns(files_changed),
            "complexity_metrics": self.text_processor.calculate_commit_complexity(
                message, files_changed, commit.get("insertions", 0), commit.get("deletions", 0)
            ),
            "semantic_fingerprint": self.text_processor.create_semantic_fingerprint(
                message, files_changed
            ),
        }

        processing_time = (time.time() - analysis_start) * 1000  # ms

        return QualitativeCommitData(
            # Copy existing commit fields
            hash=commit.get("hash", ""),
            message=message,
            author_name=commit.get("author_name", ""),
            author_email=commit.get("author_email", ""),
            timestamp=commit.get("timestamp", datetime.now()),
            files_changed=files_changed,
            insertions=commit.get("insertions", 0),
            deletions=commit.get("deletions", 0),
            # Qualitative analysis results
            change_type=change_type,
            change_type_confidence=change_confidence,
            business_domain=domain,
            domain_confidence=domain_confidence,
            risk_level=risk_assessment["level"],
            risk_factors=risk_assessment["factors"],
            intent_signals=intent_signals,
            collaboration_patterns={},  # TODO: Implement collaboration analysis
            technical_context=technical_context,
            # Processing metadata
            processing_method="nlp",
            processing_time_ms=processing_time,
            confidence_score=overall_confidence,
        )

    def _calculate_overall_confidence(
        self, change_confidence: float, domain_confidence: float, intent_confidence: float
    ) -> float:
        """Calculate weighted overall confidence score.

        Args:
            change_confidence: Change type classification confidence
            domain_confidence: Domain classification confidence
            intent_confidence: Intent analysis confidence

        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        # Weighted average with change_type being most important
        weights = {
            "change": 0.5,  # Change type is most critical
            "domain": 0.3,  # Domain is important for reporting
            "intent": 0.2,  # Intent is supplementary
        }

        overall = (
            change_confidence * weights["change"]
            + domain_confidence * weights["domain"]
            + intent_confidence * weights["intent"]
        )

        return min(1.0, max(0.0, overall))

    def _create_fallback_result(self, commit: dict[str, Any]) -> QualitativeCommitData:
        """Create a fallback result when analysis fails.

        Args:
            commit: Commit dictionary

        Returns:
            QualitativeCommitData with default values
        """
        return QualitativeCommitData(
            # Basic commit info
            hash=commit.get("hash", ""),
            message=commit.get("message", ""),
            author_name=commit.get("author_name", ""),
            author_email=commit.get("author_email", ""),
            timestamp=commit.get("timestamp", time.time()),
            files_changed=commit.get("files_changed", []),
            insertions=commit.get("insertions", 0),
            deletions=commit.get("deletions", 0),
            # Default classifications
            change_type="unknown",
            change_type_confidence=0.0,
            business_domain="unknown",
            domain_confidence=0.0,
            risk_level="medium",
            risk_factors=["analysis_failed"],
            intent_signals={"confidence": 0.0, "signals": []},
            collaboration_patterns={},
            technical_context={},
            # Processing metadata
            processing_method="nlp",
            processing_time_ms=0.0,
            confidence_score=0.0,
        )

    def get_performance_stats(self) -> dict[str, Any]:
        """Get NLP engine performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.processing_times:
            return {
                "total_batches": 0,
                "avg_processing_time_ms": 0.0,
                "min_processing_time_ms": 0.0,
                "max_processing_time_ms": 0.0,
                "total_processing_time_ms": 0.0,
            }

        return {
            "total_batches": len(self.processing_times),
            "avg_processing_time_ms": sum(self.processing_times) / len(self.processing_times),
            "min_processing_time_ms": min(self.processing_times),
            "max_processing_time_ms": max(self.processing_times),
            "total_processing_time_ms": sum(self.processing_times),
            "spacy_model": self.config.spacy_model,
            "fast_mode": self.config.fast_mode,
            "batch_size": self.config.spacy_batch_size,
        }

    def validate_setup(self) -> tuple[bool, list[str]]:
        """Validate NLP engine setup and dependencies.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check spaCy availability
        if not SPACY_AVAILABLE:
            issues.append("spaCy not installed")
            return False, issues

        # Check model availability
        try:
            test_nlp = spacy.load(self.config.spacy_model)
            # Test basic functionality
            test_doc = test_nlp("test commit message")
            if not test_doc:
                issues.append("spaCy model not functioning properly")
        except OSError:
            issues.append(f"spaCy model '{self.config.spacy_model}' not installed")
        except Exception as e:
            issues.append(f"spaCy model error: {e}")

        # Check classifier initialization
        for classifier_name, classifier in [
            ("change_type", self.change_classifier),
            ("domain", self.domain_classifier),
            ("intent", self.intent_analyzer),
            ("risk", self.risk_analyzer),
        ]:
            if (
                not hasattr(classifier, "classify")
                and not hasattr(classifier, "analyze")
                and not hasattr(classifier, "assess")
            ):
                issues.append(f"{classifier_name} classifier not properly initialized")

        return len(issues) == 0, issues
