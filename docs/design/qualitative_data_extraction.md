# Software Design Document: NLP-Based Qualitative Analysis for GitFlow Analytics

**Document Version:** 1.0  
**Date:** July 31, 2025  
**Author:** Product Team  
**Target Implementation:** Claude Code  

## 1. Executive Summary

### 1.1 Purpose
Enhance GitFlow Analytics with NLP-based qualitative data extraction to enable actionable narrative insights in Phase 2 reporting. This system will process commit messages, PR descriptions, and code changes to extract semantic meaning, intent, and business context at scale.

### 1.2 Success Criteria
- Process 10,000+ commits in under 60 seconds
- Achieve 85%+ accuracy in change type classification
- Reduce LLM API costs by 90% through intelligent NLP preprocessing
- Enable rich narrative reporting with business-relevant insights

### 1.3 Architecture Overview
Three-tier processing system:
1. **Fast NLP Layer**: spaCy-based processing for 85-90% of commits
2. **LLM Fallback Layer**: Strategic use of cheap LLMs for edge cases
3. **Pattern Learning Layer**: Continuous improvement through feedback loops

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Git Commits   │───▶│  NLP Processor   │───▶│  Qualitative    │
│   PR Data       │    │  (spaCy/NLTK)    │    │  Data Store     │
│   File Changes  │    └──────────────────┘    └─────────────────┘
└─────────────────┘             │                        ▲
                                 │                        │
                                 ▼                        │
                    ┌──────────────────┐                  │
                    │  LLM Fallback    │──────────────────┘
                    │  (Claude Haiku)  │
                    └──────────────────┘
                                 ▲
                                 │
                    ┌──────────────────┐
                    │  Pattern Cache   │
                    │  & Learning      │
                    └──────────────────┘
```

### 2.2 Component Architecture

```python
# Core module structure
src/gitflow_analytics/qualitative/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── processor.py           # Main processing orchestrator
│   ├── nlp_engine.py         # spaCy-based NLP processing
│   ├── llm_fallback.py       # LLM integration layer
│   └── pattern_cache.py      # Caching and pattern learning
├── classifiers/
│   ├── __init__.py
│   ├── change_type.py        # Commit change type classification
│   ├── intent_analyzer.py    # Intent and confidence detection
│   ├── domain_classifier.py  # Technical domain classification
│   ├── risk_analyzer.py      # Risk assessment
│   └── collaboration.py      # Team interaction patterns
├── models/
│   ├── __init__.py
│   ├── embeddings.py         # Semantic embeddings management
│   ├── schemas.py           # Data models for qualitative data
│   └── training_data.py     # Training data management
└── utils/
    ├── __init__.py
    ├── text_processing.py    # Text preprocessing utilities
    ├── batch_processor.py    # Batch processing optimization
    └── metrics.py           # Performance and accuracy metrics
```

## 3. Detailed Component Design

### 3.1 Core Processor (`core/processor.py`)

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

@dataclass
class QualitativeCommitData:
    """Enhanced commit data with qualitative signals."""
    # Existing commit data
    hash: str
    message: str
    author_name: str
    author_email: str
    timestamp: datetime
    files_changed: List[str]
    insertions: int
    deletions: int
    
    # New qualitative fields
    change_type: str
    change_type_confidence: float
    business_domain: str
    domain_confidence: float
    risk_level: str
    risk_factors: List[str]
    intent_signals: Dict[str, Any]
    collaboration_patterns: Dict[str, Any]
    technical_context: Dict[str, Any]
    
    # Processing metadata
    processing_method: str  # 'nlp' or 'llm'
    processing_time_ms: float
    confidence_score: float

class QualitativeProcessor:
    """Main orchestrator for qualitative analysis."""
    
    def __init__(self, config: QualitativeConfig):
        self.config = config
        self.nlp_engine = NLPEngine(config.nlp_config)
        self.llm_fallback = LLMFallback(config.llm_config)
        self.pattern_cache = PatternCache(config.cache_config)
        self.logger = logging.getLogger(__name__)
        
    def process_commits(self, commits: List[Dict[str, Any]]) -> List[QualitativeCommitData]:
        """Process commits with qualitative analysis."""
        start_time = time.time()
        
        # Step 1: Check cache for known patterns
        cached_results, uncached_commits = self._check_cache(commits)
        
        # Step 2: Process with NLP
        nlp_results = self._process_with_nlp(uncached_commits)
        
        # Step 3: Identify uncertain cases for LLM processing
        confident_results, uncertain_commits = self._separate_by_confidence(nlp_results)
        
        # Step 4: Process uncertain cases with LLM
        llm_results = []
        if uncertain_commits:
            llm_results = self._process_with_llm(uncertain_commits)
            
        # Step 5: Update cache with new patterns
        self._update_cache(nlp_results + llm_results)
        
        # Step 6: Combine and return results
        all_results = cached_results + confident_results + llm_results
        
        self.logger.info(f"Processed {len(commits)} commits in {time.time() - start_time:.2f}s")
        return all_results
        
    def _process_with_nlp(self, commits: List[Dict[str, Any]]) -> List[QualitativeCommitData]:
        """Process commits using NLP engine."""
        results = []
        
        # Batch process for efficiency
        for batch in self._create_batches(commits, self.config.batch_size):
            batch_results = self.nlp_engine.process_batch(batch)
            results.extend(batch_results)
            
        return results
        
    def _process_with_llm(self, commits: List[Dict[str, Any]]) -> List[QualitativeCommitData]:
        """Process uncertain commits with LLM fallback."""
        if not commits:
            return []
            
        # Group similar commits for batch processing
        grouped_commits = self.llm_fallback.group_similar_commits(commits)
        
        results = []
        for group in grouped_commits:
            group_results = self.llm_fallback.process_group(group)
            results.extend(group_results)
            
        return results
```

### 3.2 NLP Engine (`core/nlp_engine.py`)

```python
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List, Tuple, Any

class NLPEngine:
    """Core NLP processing engine using spaCy."""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        
        # Initialize spaCy with optimized pipeline
        self.nlp = spacy.load("en_core_web_sm")
        # Disable unnecessary components for speed
        self.nlp.disable_pipes(['parser', 'ner'] if config.fast_mode else [])
        
        # Initialize classifiers
        self.change_classifier = ChangeTypeClassifier(config.change_type_config)
        self.intent_analyzer = IntentAnalyzer(config.intent_config)
        self.domain_classifier = DomainClassifier(config.domain_config)
        self.risk_analyzer = RiskAnalyzer(config.risk_config)
        
        # Performance tracking
        self.processing_times = []
        
    def process_batch(self, commits: List[Dict[str, Any]]) -> List[QualitativeCommitData]:
        """Process a batch of commits efficiently."""
        start_time = time.time()
        
        # Extract messages for batch processing
        messages = [commit['message'] for commit in commits]
        
        # Process all messages through spaCy pipeline at once
        docs = list(self.nlp.pipe(messages, batch_size=self.config.spacy_batch_size))
        
        results = []
        for commit, doc in zip(commits, docs):
            result = self._analyze_commit(commit, doc)
            results.append(result)
            
        # Track performance
        processing_time = (time.time() - start_time) * 1000  # ms
        self.processing_times.append(processing_time)
        
        return results
        
    def _analyze_commit(self, commit: Dict[str, Any], doc: spacy.tokens.Doc) -> QualitativeCommitData:
        """Analyze a single commit with all classifiers."""
        
        # Change type classification
        change_type, change_confidence = self.change_classifier.classify(
            commit['message'], doc, commit.get('files_changed', [])
        )
        
        # Intent analysis
        intent_signals = self.intent_analyzer.analyze(commit['message'], doc)
        
        # Domain classification
        domain, domain_confidence = self.domain_classifier.classify(
            commit['message'], doc, commit.get('files_changed', [])
        )
        
        # Risk assessment
        risk_assessment = self.risk_analyzer.assess(commit, doc)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            change_confidence, domain_confidence, intent_signals.get('confidence', 0.5)
        )
        
        return QualitativeCommitData(
            # Copy existing fields
            hash=commit['hash'],
            message=commit['message'],
            author_name=commit['author_name'],
            author_email=commit['author_email'],
            timestamp=commit['timestamp'],
            files_changed=commit.get('files_changed', []),
            insertions=commit.get('insertions', 0),
            deletions=commit.get('deletions', 0),
            
            # New qualitative fields
            change_type=change_type,
            change_type_confidence=change_confidence,
            business_domain=domain,
            domain_confidence=domain_confidence,
            risk_level=risk_assessment['level'],
            risk_factors=risk_assessment['factors'],
            intent_signals=intent_signals,
            collaboration_patterns={},  # TODO: Implement
            technical_context={
                'file_patterns': self._analyze_file_patterns(commit.get('files_changed', [])),
                'size_category': self._categorize_commit_size(commit),
            },
            
            # Processing metadata
            processing_method='nlp',
            processing_time_ms=0,  # Set by caller
            confidence_score=overall_confidence
        )
```

### 3.3 Change Type Classifier (`classifiers/change_type.py`)

```python
class ChangeTypeClassifier:
    """Classify commits by change type using semantic analysis."""
    
    def __init__(self, config: ChangeTypeConfig):
        self.config = config
        
        # Semantic keyword clusters
        self.change_patterns = {
            'feature': {
                'action_words': ['add', 'implement', 'create', 'build', 'introduce'],
                'object_words': ['feature', 'functionality', 'capability', 'component'],
                'context_words': ['user', 'client', 'new', 'support']
            },
            'bugfix': {
                'action_words': ['fix', 'resolve', 'correct', 'repair', 'patch'],
                'object_words': ['bug', 'issue', 'problem', 'error', 'defect'],
                'context_words': ['crash', 'exception', 'fail', 'broken']
            },
            'refactor': {
                'action_words': ['refactor', 'restructure', 'reorganize', 'cleanup'],
                'object_words': ['code', 'structure', 'architecture', 'design'],
                'context_words': ['improve', 'optimize', 'simplify', 'maintain']
            },
            'docs': {
                'action_words': ['update', 'add', 'improve', 'write'],
                'object_words': ['documentation', 'readme', 'docs', 'comment'],
                'context_words': ['explain', 'clarify', 'describe']
            },
            'test': {
                'action_words': ['add', 'update', 'fix', 'improve'],
                'object_words': ['test', 'spec', 'coverage', 'unit', 'integration'],
                'context_words': ['testing', 'verify', 'validate']
            },
            'chore': {
                'action_words': ['update', 'bump', 'upgrade', 'configure'],
                'object_words': ['dependency', 'package', 'config', 'build'],
                'context_words': ['maintenance', 'housekeeping']
            }
        }
        
        # Build TF-IDF vectors for each category
        self._build_semantic_vectors()
        
    def classify(self, message: str, doc: spacy.tokens.Doc, 
                files: List[str]) -> Tuple[str, float]:
        """Classify commit change type with confidence score."""
        
        # Extract semantic features
        semantic_features = self._extract_semantic_features(doc)
        
        # Calculate similarity to each change type
        similarities = {}
        for change_type, patterns in self.change_patterns.items():
            similarity = self._calculate_semantic_similarity(
                semantic_features, patterns
            )
            similarities[change_type] = similarity
            
        # Consider file patterns as additional signal
        file_signals = self._analyze_file_patterns(files)
        for change_type, file_score in file_signals.items():
            similarities[change_type] = similarities.get(change_type, 0) + file_score * 0.3
            
        # Find best match
        best_type = max(similarities.keys(), key=lambda k: similarities[k])
        confidence = similarities[best_type]
        
        # Apply confidence threshold
        if confidence < self.config.min_confidence:
            return 'unknown', confidence
            
        return best_type, confidence
        
    def _extract_semantic_features(self, doc: spacy.tokens.Doc) -> Dict[str, List[str]]:
        """Extract semantic features from spaCy doc."""
        features = {
            'verbs': [],
            'nouns': [],
            'adjectives': [],
            'entities': []
        }
        
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
                
            if token.pos_ == 'VERB':
                features['verbs'].append(token.lemma_.lower())
            elif token.pos_ in ['NOUN', 'PROPN']:
                features['nouns'].append(token.lemma_.lower())
            elif token.pos_ == 'ADJ':
                features['adjectives'].append(token.lemma_.lower())
                
        # Add named entities
        for ent in doc.ents:
            features['entities'].append(ent.text.lower())
            
        return features
        
    def _calculate_semantic_similarity(self, features: Dict[str, List[str]], 
                                     patterns: Dict[str, List[str]]) -> float:
        """Calculate semantic similarity between features and patterns."""
        similarity_score = 0.0
        
        # Check action words (verbs) - highest weight
        action_matches = len(set(features['verbs']) & set(patterns['action_words']))
        similarity_score += action_matches * 0.5
        
        # Check object words (nouns) - medium weight
        object_matches = len(set(features['nouns']) & set(patterns['object_words']))
        similarity_score += object_matches * 0.3
        
        # Check context words - lower weight
        all_words = features['verbs'] + features['nouns'] + features['adjectives']
        context_matches = len(set(all_words) & set(patterns['context_words']))
        similarity_score += context_matches * 0.2
        
        # Normalize by total possible matches
        max_possible = len(patterns['action_words']) * 0.5 + \
                      len(patterns['object_words']) * 0.3 + \
                      len(patterns['context_words']) * 0.2
                      
        return similarity_score / max_possible if max_possible > 0 else 0.0
```

### 3.4 LLM Fallback System (`core/llm_fallback.py`)

```python
import asyncio
from typing import List, Dict, Any, Tuple
import hashlib
import json

class LLMFallback:
    """Strategic LLM usage for uncertain cases via OpenRouter."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._initialize_openrouter_client()
        self.batch_cache = {}
        self.cost_tracker = CostTracker()
        self.model_router = ModelRouter(config)
        
    def _initialize_openrouter_client(self):
        """Initialize OpenRouter client with API key."""
        import openai  # OpenRouter uses OpenAI-compatible API
        
        return openai.OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.openrouter_api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/your-org/gitflow-analytics",
                "X-Title": "GitFlow Analytics"
            }
        )
        
    def group_similar_commits(self, commits: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group similar commits for batch processing."""
        groups = []
        similarity_threshold = 0.8
        
        for commit in commits:
            # Find similar group or create new one
            placed = False
            for group in groups:
                if self._calculate_message_similarity(
                    commit['message'], group[0]['message']
                ) > similarity_threshold:
                    group.append(commit)
                    placed = True
                    break
                    
            if not placed:
                groups.append([commit])
                
        return groups
        
    def process_group(self, commits: List[Dict[str, Any]]) -> List[QualitativeCommitData]:
        """Process a group of similar commits with OpenRouter model selection."""
        if not commits:
            return []
            
        # Check cache first
        cache_key = self._generate_group_cache_key(commits)
        if cache_key in self.batch_cache:
            template_result = self.batch_cache[cache_key]
            return self._apply_template_to_group(template_result, commits)
            
        # Determine complexity and select appropriate model
        complexity_score = self._assess_complexity(commits)
        selected_model = self.model_router.select_model(complexity_score, len(commits))
        
        # Prepare optimized prompt for batch processing
        prompt = self._build_batch_classification_prompt(commits)
        
        # Make OpenRouter call with cost tracking
        try:
            start_time = time.time()
            response = self._call_openrouter(prompt, selected_model)
            processing_time = time.time() - start_time
            
            # Track costs and performance
            self.cost_tracker.record_call(
                model=selected_model,
                input_tokens=self._estimate_tokens(prompt),
                output_tokens=self._estimate_tokens(response),
                processing_time=processing_time
            )
            
            results = self._parse_llm_response(response, commits)
            
            # Cache successful result for similar patterns
            if results:
                self.batch_cache[cache_key] = self._create_template(results[0])
                
            return results
            
        except Exception as e:
            logging.error(f"OpenRouter processing failed: {e}")
            # Fallback to free model if primary fails
            if selected_model != self.config.fallback_model:
                return self._retry_with_fallback_model(commits, prompt)
            return self._create_fallback_results(commits)
            
    def _call_openrouter(self, prompt: str, model: str) -> str:
        """Make API call to OpenRouter with selected model."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a Git commit classifier. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"OpenRouter API call failed: {e}")
            raise
            
    def _build_batch_classification_prompt(self, commits: List[Dict[str, Any]]) -> str:
        """Build optimized prompt for OpenRouter batch processing."""
        
        # Create compact commit representations
        commit_data = []
        for i, commit in enumerate(commits[:10]):  # Process up to 10 at once
            # Include file context for better classification
            files_context = ""
            if commit.get('files_changed'):
                key_files = [f for f in commit['files_changed'][:5]]  # Top 5 files
                files_context = f" | Files: {', '.join(key_files)}"
            
            commit_data.append(
                f"{i+1}. {commit['message'][:120]}{'...' if len(commit['message']) > 120 else ''}{files_context}"
            )
            
        prompt = f"""Analyze these Git commits and classify each one. Consider the commit message and modified files.

Commits to classify:
{chr(10).join(commit_data)}

For each commit, determine:
- change_type: feature|bugfix|refactor|docs|test|chore|security|hotfix|config
- business_domain: frontend|backend|database|infrastructure|mobile|devops|unknown  
- risk_level: low|medium|high|critical
- confidence: 0.0-1.0 (how certain you are)
- urgency: routine|important|urgent|critical
- complexity: simple|moderate|complex

Respond with JSON array only, no other text:
[{{"id": 1, "change_type": "feature", "business_domain": "frontend", "risk_level": "low", "confidence": 0.9, "urgency": "routine", "complexity": "moderate"}}]"""

        return prompt
        
    def _parse_llm_response(self, response: str, commits: List[Dict[str, Any]]) -> List[QualitativeCommitData]:
        """Parse LLM response into QualitativeCommitData objects."""
        try:
            classifications = json.loads(response)
            results = []
            
            for i, (commit, classification) in enumerate(zip(commits, classifications)):
                result = QualitativeCommitData(
                    # Copy existing fields
                    hash=commit['hash'],
                    message=commit['message'],
                    author_name=commit['author_name'],
                    author_email=commit['author_email'],
                    timestamp=commit['timestamp'],
                    files_changed=commit.get('files_changed', []),
                    insertions=commit.get('insertions', 0),
                    deletions=commit.get('deletions', 0),
                    
                    # LLM-provided classifications
                    change_type=classification.get('change_type', 'unknown'),
                    change_type_confidence=classification.get('confidence', 0.5),
                    business_domain=classification.get('business_domain', 'unknown'),
                    domain_confidence=classification.get('confidence', 0.5),
                    risk_level=classification.get('risk_level', 'medium'),
                    risk_factors=classification.get('risk_factors', []),
                    intent_signals={
                        'signals': classification.get('intent_signals', []),
                        'confidence': classification.get('confidence', 0.5)
                    },
                    collaboration_patterns={},
                    technical_context={},
                    
                    # Processing metadata
                    processing_method='llm',
                    processing_time_ms=0,  # Set by caller
                    confidence_score=classification.get('confidence', 0.5)
                )
                results.append(result)
                
            return results
            
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to parse LLM response: {e}")
            return self._create_fallback_results(commits)
```

## 4. Data Models and Schemas

### 4.1 Configuration Schema (`models/schemas.py`)

```python
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class QualitativeConfig:
    """Main configuration for qualitative analysis."""
    
    # Processing settings
    batch_size: int = 1000
    max_llm_fallback_pct: float = 0.15  # Max 15% of commits use LLM
    confidence_threshold: float = 0.7
    
    # Component configs
    nlp_config: 'NLPConfig'
    llm_config: 'LLMConfig'
    cache_config: 'CacheConfig'

@dataclass
class NLPConfig:
    """NLP engine configuration."""
    
    spacy_model: str = "en_core_web_sm"
    spacy_batch_size: int = 1000
    fast_mode: bool = True  # Disable parser/NER for speed
    
    # Classifier configs
    change_type_config: 'ChangeTypeConfig'
    intent_config: 'IntentConfig'
    domain_config: 'DomainConfig'
    risk_config: 'RiskConfig'

@dataclass
class LLMConfig:
    """LLM fallback configuration using OpenRouter."""
    
    # OpenRouter settings
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
    max_group_size: int = 10  # Larger batches with cheaper models
    similarity_threshold: float = 0.8
    
    # Rate limiting
    requests_per_minute: int = 200  # Higher with OpenRouter
    max_retries: int = 3

@dataclass
class CacheConfig:
    """Caching configuration."""
    
    cache_dir: str = ".qualitative_cache"
    semantic_cache_size: int = 10000
    pattern_cache_ttl_hours: int = 168  # 1 week
    
    # Learning settings
    enable_pattern_learning: bool = True
    learning_threshold: int = 10  # Min examples to learn pattern
```

### 4.2 Database Schema Extensions

```python
# Add to existing models/database.py

class QualitativeCommitData(Base):
    """Extended commit data with qualitative analysis."""
    __tablename__ = 'qualitative_commits'
    
    # Link to existing commit
    commit_id = Column(Integer, ForeignKey('cached_commits.id'), primary_key=True)
    
    # Classification results
    change_type = Column(String, nullable=False)
    change_type_confidence = Column(Float, nullable=False)
    business_domain = Column(String, nullable=False)
    domain_confidence = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    risk_factors = Column(JSON)
    
    # Intent and context
    intent_signals = Column(JSON)
    collaboration_patterns = Column(JSON)
    technical_context = Column(JSON)
    
    # Processing metadata
    processing_method = Column(String, nullable=False)  # 'nlp' or 'llm'
    processing_time_ms = Column(Float)
    confidence_score = Column(Float, nullable=False)
    
    # Timestamps
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    analysis_version = Column(String, default="1.0")
    
    # Indexes
    __table_args__ = (
        Index('idx_change_type', 'change_type'),
        Index('idx_business_domain', 'business_domain'),
        Index('idx_risk_level', 'risk_level'),
        Index('idx_confidence', 'confidence_score'),
    )

class PatternCache(Base):
    """Cache for learned patterns and classifications."""
    __tablename__ = 'pattern_cache'
    
    id = Column(Integer, primary_key=True)
    
    # Pattern identification
    message_hash = Column(String, nullable=False, unique=True)
    semantic_fingerprint = Column(String, nullable=False)
    
    # Cached results
    classification_result = Column(JSON, nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Usage tracking
    hit_count = Column(Integer, default=1)
    last_used = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Source tracking
    source_method = Column(String, nullable=False)  # 'nlp' or 'llm'
    
    __table_args__ = (
        Index('idx_semantic_fingerprint', 'semantic_fingerprint'),
        Index('idx_confidence', 'confidence_score'),
    )
```

## 5. Performance Requirements

### 5.1 Processing Targets
- **Throughput**: 10,000 commits processed in < 60 seconds
- **NLP Processing**: < 2ms average per commit message
- **LLM Fallback**: < 15% of commits require LLM processing
- **Memory Usage**: < 2GB RAM for 100k commit analysis

### 5.2 Accuracy Targets
- **Change Type Classification**: 85%+ accuracy
- **Domain Classification**: 80%+ accuracy with file context
- **Risk Assessment**: 80%+ accuracy for high-risk identification
- **Overall Confidence**: 90%+ of predictions above 0.7 confidence

### 5.3 Cost Targets (Updated with OpenRouter)
- **LLM API Costs**: < $0.005 per 1000 commits processed (50% cost reduction)
- **Free Model Usage**: 60%+ of LLM calls use free Llama 3.1 model
- **Premium Model Usage**: < 5% of commits require Claude Sonnet/GPT-4
- **Total Processing Costs**: 95% reduction vs. all-premium-LLM approach

## 6. Implementation Plan

### 6.1 Phase 1: Core NLP Infrastructure (Week 1)
**Deliverables:**
- Basic NLP processing pipeline with spaCy
- Change type classifier with 80%+ accuracy
- Batch processing framework
- Initial performance benchmarking

**Tasks:**
- Set up spaCy pipeline with optimized components
- Implement ChangeTypeClassifier with semantic matching
- Create batch processing infrastructure
- Build performance monitoring

**Success Criteria:**
- Process 1000 commits in < 10 seconds
- Achieve 80%+ accuracy on manual test set
- Memory usage < 500MB for 10k commits

### 6.2 Phase 2: Domain and Risk Analysis (Week 2)
**Deliverables:**
- Domain classification with file context analysis
- Risk assessment combining multiple signals
- Intent analysis for confidence/urgency detection
- Confidence scoring framework

**Tasks:**
- Implement DomainClassifier with file pattern analysis
- Build RiskAnalyzer combining text and statistical signals
- Create IntentAnalyzer for confidence/urgency detection
- Develop overall confidence scoring

**Success Criteria:**
- Domain classification 80%+ accuracy
- Risk assessment identifies 90%+ of high-risk commits
- Intent analysis provides actionable confidence scores

### 6.3 Phase 3: LLM Integration (Week 3)
**Deliverables:**
- LLM fallback system for uncertain cases
- Batch LLM processing with grouping
- Pattern learning from LLM results
- Cost optimization through intelligent routing

**Tasks:**
- Implement LLMFallback with commit grouping
- Build pattern learning system
- Create cost tracking and optimization
- Add retry logic and error handling

**Success Criteria:**
- < 10% of commits require LLM processing (reduced from 15%)
- LLM processing cost < $0.005 per 1000 commits (OpenRouter efficiency)
- 60%+ of LLM calls use free models for routine commits
- Pattern learning improves NLP accuracy by 5% monthly

### 6.4 Phase 4: Integration and Optimization (Week 4)
**Deliverables:**
- Full integration with existing GitFlow Analytics
- Performance optimization and caching
- Documentation and testing
- Production deployment readiness

**Tasks:**
- Integrate with existing commit processing pipeline
- Implement caching and optimization strategies
- Write comprehensive tests and documentation
- Performance tuning and profiling

**Success Criteria:**
- Full integration with existing system
- Meet all performance and accuracy targets
- Production-ready with monitoring and alerting

## 7. Testing Strategy

### 7.1 Unit Testing
- Test each classifier independently
- Mock LLM calls for deterministic testing
- Validate caching and pattern learning logic
- Performance benchmarking for each component

### 7.2 Integration Testing
- End-to-end processing pipeline testing
- Real commit data validation
- LLM fallback integration testing
- Cache coherency and performance testing

### 7.3 Accuracy Validation
- Manual classification of 1000+ commits for ground truth
- Cross-validation with existing GitFlow Analytics users
- A/B testing against current metrics
- Continuous accuracy monitoring in production

### 7.4 Performance Testing
- Load testing with 100k+ commit datasets
- Memory usage profiling and optimization
- Concurrent processing validation
- Cost monitoring and optimization validation

## 8. Monitoring and Observability

### 8.1 Key Metrics
- **Processing Performance**: commits/second, average processing time
- **Accuracy Metrics**: classification accuracy by type, confidence distributions
- **Cost Metrics**: LLM API usage, cost per commit processed
- **System Health**: error rates, cache hit rates, memory usage

### 8.2 Alerting (Updated for OpenRouter)
- Processing time > 100ms per commit (99th percentile)
- LLM fallback usage > 15% of commits
- Premium model usage > 10% of LLM calls
- Daily OpenRouter costs > budget threshold
- Classification confidence < 70% for > 30% of commits
- Free model API rate limits approached

### 8.3 Dashboards
- Real-time processing metrics
- Classification accuracy trends
- Cost tracking and optimization opportunities
- Pattern learning effectiveness metrics

## 9. Deployment and Rollout

### 9.1 Deployment Strategy
- **Phase 1**: Deploy to development environment with synthetic data
- **Phase 2**: Limited beta with select users and real data
- **Phase 3**: Gradual rollout to all users with feature flags
- **Phase 4**: Full production deployment with monitoring

### 9.2 Rollback Plan
- Feature flags to disable qualitative analysis
- Fallback to existing quantitative-only processing
- Database rollback procedures for schema changes
- LLM processing can be disabled independently

### 9.3 Migration Strategy
- Backward compatibility with existing data structures
- Gradual backfill of historical data
- Optional qualitative analysis initially
- Performance comparison with existing system

## 10. Future Enhancements

### 10.1 Advanced NLP Features
- Custom domain-specific embeddings
- Multi-language support
- Advanced entity recognition for business concepts
- Sentiment analysis for team dynamics

### 10.2 Machine Learning Improvements
- Active learning for pattern improvement
- Custom model training for domain-specific classification
- Collaborative filtering for developer behavior patterns
- Predictive analytics for project health

### 10.3 Integration Expansions
- Additional VCS platforms (GitLab, Bitbucket)
- More ticket systems (Azure DevOps, etc.)
- CI/CD pipeline integration
- Code review quality analysis

This design document provides a comprehensive roadmap for implementing high-performance qualitative analysis in GitFlow Analytics, balancing accuracy, performance, and cost through intelligent NLP and strategic LLM usage.