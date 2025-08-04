"""ML-enhanced ticket reference extraction with sophisticated commit categorization.

This module extends the basic TicketExtractor with machine learning capabilities for
better commit categorization. It integrates with the existing qualitative analysis
infrastructure to provide hybrid rule-based + ML classification.

WHY: Traditional regex-based categorization has limitations in understanding context
and nuanced commit messages. This ML-enhanced version provides better accuracy while
maintaining backward compatibility and performance through intelligent caching.

DESIGN DECISIONS:
- Hybrid approach: Falls back to rule-based when ML confidence is low
- Confidence scoring: All classifications include confidence scores for reliability
- Caching strategy: ML predictions are cached to maintain performance
- Feature extraction: Uses both message content and file patterns for better accuracy
- Integration: Leverages existing ChangeTypeClassifier from qualitative analysis

PERFORMANCE: Designed to handle large repositories efficiently with:
- Batch processing for ML predictions
- Intelligent caching of ML results
- Fallback to fast rule-based classification when appropriate
"""

import logging
import pickle
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .tickets import TicketExtractor
from ..qualitative.classifiers.change_type import ChangeTypeClassifier
from ..qualitative.models.schemas import ChangeTypeConfig

try:
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    Doc = Any

logger = logging.getLogger(__name__)


class MLTicketExtractor(TicketExtractor):
    """ML-enhanced ticket extractor with sophisticated commit categorization.
    
    This extractor extends the basic TicketExtractor with machine learning capabilities
    while maintaining full backward compatibility. It uses a hybrid approach combining
    rule-based patterns with ML-based semantic analysis for improved accuracy.
    
    Key features:
    - Hybrid categorization (ML + rule-based fallback)
    - Confidence scoring for all predictions
    - Intelligent caching for performance
    - Feature extraction from commit message and file patterns
    - Integration with existing qualitative analysis infrastructure
    """
    
    def __init__(self, 
                 allowed_platforms: Optional[List[str]] = None,
                 untracked_file_threshold: int = 1,
                 ml_config: Optional[Dict[str, Any]] = None,
                 cache_dir: Optional[Path] = None,
                 enable_ml: bool = True) -> None:
        """Initialize ML-enhanced ticket extractor.
        
        Args:
            allowed_platforms: List of platforms to extract tickets from
            untracked_file_threshold: Minimum files changed for significant commits
            ml_config: Configuration for ML categorization (optional)
            cache_dir: Directory for caching ML predictions
            enable_ml: Whether to enable ML features (fallback to rule-based if False)
        """
        # Initialize parent class
        super().__init__(allowed_platforms, untracked_file_threshold)
        
        self.enable_ml = enable_ml and SPACY_AVAILABLE
        self.cache_dir = cache_dir or Path(".gitflow-cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # ML configuration with sensible defaults
        default_ml_config = {
            'min_confidence': 0.6,
            'semantic_weight': 0.7,
            'file_pattern_weight': 0.3,
            'hybrid_threshold': 0.5,  # Confidence threshold for using ML vs rule-based
            'cache_duration_days': 30,
            'batch_size': 100,
            'enable_caching': True
        }
        
        self.ml_config = {**default_ml_config, **(ml_config or {})}
        
        # Initialize ML components
        self.change_type_classifier = None
        self.nlp_model = None
        self.ml_cache = None
        
        if self.enable_ml:
            self._initialize_ml_components()
        
        logger.info(f"MLTicketExtractor initialized with ML {'enabled' if self.enable_ml else 'disabled'}")
    
    def _initialize_ml_components(self) -> None:
        """Initialize ML components (ChangeTypeClassifier and spaCy model).
        
        WHY: Separate initialization allows for graceful degradation if ML components
        fail to load. The extractor will fall back to rule-based classification.
        """
        try:
            # Initialize ChangeTypeClassifier
            change_type_config = ChangeTypeConfig(
                min_confidence=self.ml_config['min_confidence'],
                semantic_weight=self.ml_config['semantic_weight'],
                file_pattern_weight=self.ml_config['file_pattern_weight']
            )
            self.change_type_classifier = ChangeTypeClassifier(change_type_config)
            
            # Initialize spaCy model (try English first, then basic)
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("English spaCy model not found, trying basic model")
                try:
                    self.nlp_model = spacy.load("en_core_web_md")
                except OSError:
                    logger.warning("No spaCy models found, ML classification will use fallback")
                    self.nlp_model = None
            
            # Initialize ML cache
            if self.ml_config['enable_caching']:
                self._initialize_ml_cache()
                
            logger.info("ML components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML components: {e}")
            logger.info("Falling back to rule-based classification")
            self.enable_ml = False
    
    def _initialize_ml_cache(self) -> None:
        """Initialize SQLite cache for ML predictions.
        
        WHY: ML predictions can be expensive, so we cache results to improve performance
        on subsequent runs. The cache includes expiration and invalidation logic.
        """
        try:
            cache_path = self.cache_dir / "ml_predictions.db"
            self.ml_cache = MLPredictionCache(cache_path, self.ml_config['cache_duration_days'])
            logger.debug("ML prediction cache initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize ML cache: {e}")
            self.ml_cache = None
    
    def categorize_commit(self, message: str, files_changed: Optional[List[str]] = None) -> str:
        """Categorize a commit using hybrid ML + rule-based approach.
        
        This method extends the parent's categorize_commit with ML capabilities while
        maintaining backward compatibility. It returns the same category strings as
        the parent class.
        
        Args:
            message: The commit message to categorize
            files_changed: Optional list of changed files for additional context
            
        Returns:
            String category (bug_fix, feature, refactor, documentation, 
            maintenance, test, style, build, or other)
        """
        if not message:
            return "other"
        
        # Try ML categorization first if enabled
        if self.enable_ml:
            ml_result = self._ml_categorize_commit(message, files_changed or [])
            if ml_result and ml_result['confidence'] >= self.ml_config['hybrid_threshold']:
                # Map ML categories to parent class categories
                mapped_category = self._map_ml_to_parent_category(ml_result['category'])
                return mapped_category
        
        # Fall back to parent's rule-based categorization
        return super().categorize_commit(message)
    
    def categorize_commit_with_confidence(self, message: str, 
                                        files_changed: Optional[List[str]] = None) -> Dict[str, Any]:
        """Categorize commit with detailed confidence information.
        
        This is the main entry point for getting detailed categorization results
        including confidence scores, alternative predictions, and processing metadata.
        
        Args:
            message: The commit message to categorize
            files_changed: Optional list of changed files for additional context
            
        Returns:
            Dictionary with categorization results:
            {
                'category': str,
                'confidence': float,
                'method': str ('ml', 'rules', 'cached'),
                'alternatives': List[Dict],
                'features': Dict,
                'processing_time_ms': float
            }
        """
        start_time = time.time()
        
        if not message:
            return {
                'category': 'other',
                'confidence': 1.0,
                'method': 'default',
                'alternatives': [],
                'features': {},
                'processing_time_ms': 0.0
            }
        
        files_changed = files_changed or []
        
        # Check cache first
        if self.ml_cache and self.ml_config['enable_caching']:
            cached_result = self.ml_cache.get_prediction(message, files_changed)
            if cached_result:
                cached_result['processing_time_ms'] = (time.time() - start_time) * 1000
                return cached_result
        
        # Try ML categorization
        if self.enable_ml:
            ml_result = self._ml_categorize_commit_detailed(message, files_changed)
            if ml_result and ml_result['confidence'] >= self.ml_config['hybrid_threshold']:
                # Map to parent categories and cache result
                ml_result['category'] = self._map_ml_to_parent_category(ml_result['category'])
                ml_result['processing_time_ms'] = (time.time() - start_time) * 1000
                
                if self.ml_cache and self.ml_config['enable_caching']:
                    self.ml_cache.store_prediction(message, files_changed, ml_result)
                
                return ml_result
        
        # Fall back to rule-based categorization
        rule_category = super().categorize_commit(message)
        rule_result = {
            'category': rule_category,
            'confidence': 0.8 if rule_category != 'other' else 0.3,
            'method': 'rules',
            'alternatives': [],
            'features': {'rule_based': True},
            'processing_time_ms': (time.time() - start_time) * 1000
        }
        
        if self.ml_cache and self.ml_config['enable_caching']:
            self.ml_cache.store_prediction(message, files_changed, rule_result)
        
        return rule_result
    
    def _ml_categorize_commit(self, message: str, files_changed: List[str]) -> Optional[Dict[str, Any]]:
        """Internal ML categorization method (simplified version).
        
        Args:
            message: Commit message
            files_changed: List of changed files
            
        Returns:
            Dictionary with category and confidence, or None if ML unavailable
        """
        if not self.change_type_classifier or not message:
            return None
        
        try:
            # Process message with spaCy if available
            doc = None
            if self.nlp_model:
                doc = self.nlp_model(message)
            
            # Get ML classification
            ml_category, confidence = self.change_type_classifier.classify(message, doc, files_changed)
            
            if ml_category and ml_category != 'unknown':
                return {
                    'category': ml_category,
                    'confidence': confidence
                }
            
        except Exception as e:
            logger.warning(f"ML categorization failed: {e}")
        
        return None
    
    def _ml_categorize_commit_detailed(self, message: str, files_changed: List[str]) -> Optional[Dict[str, Any]]:
        """Detailed ML categorization with comprehensive metadata.
        
        Args:
            message: Commit message
            files_changed: List of changed files
            
        Returns:
            Detailed categorization result dictionary or None if ML unavailable
        """
        if not self.change_type_classifier or not message:
            return None
        
        try:
            # Process message with spaCy
            doc = None
            features = {}
            if self.nlp_model:
                doc = self.nlp_model(message)
                features = self._extract_features(message, doc, files_changed)
            
            # Get ML classification
            ml_category, confidence = self.change_type_classifier.classify(message, doc, files_changed)
            
            if ml_category and ml_category != 'unknown':
                return {
                    'category': ml_category,
                    'confidence': confidence,
                    'method': 'ml',
                    'alternatives': self._get_alternative_predictions(message, doc, files_changed),
                    'features': features
                }
            
        except Exception as e:
            logger.warning(f"Detailed ML categorization failed: {e}")
        
        return None
    
    def _extract_features(self, message: str, doc: Optional[Doc], files_changed: List[str]) -> Dict[str, Any]:
        """Extract features used for ML classification.
        
        Args:
            message: Commit message
            doc: spaCy processed document
            files_changed: List of changed files
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'message_length': len(message),
            'word_count': len(message.split()),
            'files_count': len(files_changed),
            'file_extensions': list(set(Path(f).suffix.lower() for f in files_changed if Path(f).suffix))
        }
        
        if doc:
            features.update({
                'has_verbs': any(token.pos_ == 'VERB' for token in doc),
                'has_entities': len(doc.ents) > 0,
                'sentiment_polarity': 0.0  # Placeholder - could add sentiment analysis
            })
        
        return features
    
    def _get_alternative_predictions(self, message: str, doc: Optional[Doc], 
                                   files_changed: List[str]) -> List[Dict[str, Any]]:
        """Get alternative predictions with lower confidence scores.
        
        This is a simplified version - in a full implementation, you would
        get all classification scores and return top N alternatives.
        
        Args:
            message: Commit message
            doc: spaCy processed document
            files_changed: List of changed files
            
        Returns:
            List of alternative predictions
        """
        # Placeholder implementation - could be enhanced to return actual alternatives
        alternatives = []
        
        # Add rule-based prediction as alternative
        rule_category = super().categorize_commit(message)
        if rule_category != 'other':
            alternatives.append({
                'category': rule_category,
                'confidence': 0.6,
                'method': 'rules'
            })
        
        return alternatives[:3]  # Top 3 alternatives
    
    def _map_ml_to_parent_category(self, ml_category: str) -> str:
        """Map ML categories to parent class categories.
        
        WHY: The ChangeTypeClassifier uses different category names than the parent
        TicketExtractor. This mapping ensures backward compatibility.
        
        Args:
            ml_category: Category from ML classifier
            
        Returns:
            Category compatible with parent class
        """
        mapping = {
            'feature': 'feature',
            'bugfix': 'bug_fix', 
            'refactor': 'refactor',
            'docs': 'documentation',
            'test': 'test',
            'chore': 'maintenance',
            'security': 'bug_fix',  # Security fixes are a type of bug fix
            'hotfix': 'bug_fix',    # Hotfixes are urgent bug fixes
            'config': 'maintenance' # Configuration changes are maintenance
        }
        
        return mapping.get(ml_category, 'other')
    
    def analyze_ticket_coverage(self, commits: List[Dict[str, Any]], 
                              prs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced ticket coverage analysis with ML categorization insights.
        
        This method extends the parent's analysis with ML-specific insights including
        confidence distributions, method breakdowns, and prediction quality metrics.
        
        Args:
            commits: List of commit data
            prs: List of PR data
            
        Returns:
            Enhanced analysis results with ML insights
        """
        # Get base analysis from parent
        base_analysis = super().analyze_ticket_coverage(commits, prs)
        
        if not self.enable_ml:
            # Add indicator that ML was not used
            base_analysis['ml_analysis'] = {
                'enabled': False,
                'reason': 'ML components not available or disabled'
            }
            return base_analysis
        
        # Enhance with ML-specific analysis
        ml_analysis = self._analyze_ml_categorization_quality(commits)
        base_analysis['ml_analysis'] = ml_analysis
        
        # Enhance untracked commits with confidence scores
        if 'untracked_commits' in base_analysis:
            self._enhance_untracked_commits(base_analysis['untracked_commits'])
        
        return base_analysis
    
    def _analyze_ml_categorization_quality(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the quality and distribution of ML categorizations.
        
        Args:
            commits: List of commit data
            
        Returns:
            ML analysis results including confidence distributions and method usage
        """
        ml_stats = {
            'enabled': True,
            'total_ml_predictions': 0,
            'total_rule_predictions': 0,
            'total_cached_predictions': 0,
            'avg_confidence': 0.0,
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'method_breakdown': defaultdict(int),
            'category_confidence': defaultdict(list),
            'processing_time_stats': {'total_ms': 0.0, 'avg_ms': 0.0}
        }
        
        total_confidence = 0.0
        total_processing_time = 0.0
        processed_commits = 0
        
        for commit in commits:
            if commit.get('is_merge') or commit.get('files_changed', 0) < self.untracked_file_threshold:
                continue
            
            # Get detailed categorization for analysis
            message = commit.get('message', '')
            files_changed = commit.get('files_changed_list', [])
            
            result = self.categorize_commit_with_confidence(message, files_changed)
            
            # Update statistics
            confidence = result['confidence']
            method = result['method']
            category = result['category']
            processing_time = result.get('processing_time_ms', 0.0)
            
            total_confidence += confidence
            total_processing_time += processing_time
            processed_commits += 1
            
            # Method breakdown
            ml_stats['method_breakdown'][method] += 1
            if method == 'ml':
                ml_stats['total_ml_predictions'] += 1
            elif method == 'rules':
                ml_stats['total_rule_predictions'] += 1
            elif method == 'cached':
                ml_stats['total_cached_predictions'] += 1
            
            # Confidence distribution
            if confidence >= 0.8:
                ml_stats['confidence_distribution']['high'] += 1
            elif confidence >= 0.6:
                ml_stats['confidence_distribution']['medium'] += 1
            else:
                ml_stats['confidence_distribution']['low'] += 1
            
            # Category confidence tracking
            ml_stats['category_confidence'][category].append(confidence)
        
        # Calculate averages
        if processed_commits > 0:
            ml_stats['avg_confidence'] = total_confidence / processed_commits
            ml_stats['processing_time_stats'] = {
                'total_ms': total_processing_time,
                'avg_ms': total_processing_time / processed_commits
            }
        
        # Convert defaultdicts to regular dicts for JSON serialization
        ml_stats['method_breakdown'] = dict(ml_stats['method_breakdown'])
        ml_stats['category_confidence'] = {
            cat: {'avg': sum(confidences) / len(confidences), 'count': len(confidences)}
            for cat, confidences in ml_stats['category_confidence'].items()
        }
        
        return ml_stats
    
    def _enhance_untracked_commits(self, untracked_commits: List[Dict[str, Any]]) -> None:
        """Enhance untracked commits with ML confidence scores and metadata.
        
        Args:
            untracked_commits: List of untracked commit data to enhance in-place
        """
        for commit in untracked_commits:
            message = commit.get('full_message', commit.get('message', ''))
            files_changed = []  # Would need to extract from commit data
            
            # Get detailed categorization
            result = self.categorize_commit_with_confidence(message, files_changed)
            
            # Add ML-specific fields
            commit['ml_confidence'] = result['confidence']
            commit['ml_method'] = result['method']
            commit['ml_alternatives'] = result.get('alternatives', [])
            commit['ml_processing_time_ms'] = result.get('processing_time_ms', 0.0)
    
    def get_ml_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ML usage and performance statistics.
        
        Returns:
            Dictionary with ML performance metrics and usage statistics
        """
        stats = {
            'ml_enabled': self.enable_ml,
            'spacy_available': SPACY_AVAILABLE,
            'components_loaded': {
                'change_type_classifier': self.change_type_classifier is not None,
                'nlp_model': self.nlp_model is not None,
                'ml_cache': self.ml_cache is not None
            },
            'configuration': self.ml_config.copy()
        }
        
        # Add cache statistics if available
        if self.ml_cache:
            stats['cache_statistics'] = self.ml_cache.get_statistics()
        
        return stats


class MLPredictionCache:
    """SQLite-based cache for ML predictions with expiration support.
    
    WHY: ML predictions can be expensive, especially for large repositories.
    This cache stores predictions with metadata to avoid re-processing identical
    commit messages and file patterns.
    
    DESIGN: Uses SQLite for persistence across runs with:
    - Expiration based on configurable time periods
    - Hash-based keys for efficient lookup
    - Metadata storage for cache invalidation
    """
    
    def __init__(self, cache_path: Path, expiration_days: int = 30):
        """Initialize ML prediction cache.
        
        Args:
            cache_path: Path to SQLite cache database
            expiration_days: Number of days to keep predictions
        """
        self.cache_path = cache_path
        self.expiration_days = expiration_days
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database with prediction cache table."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    key TEXT PRIMARY KEY,
                    message_hash TEXT NOT NULL,
                    files_hash TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    method TEXT NOT NULL,
                    features TEXT,  -- JSON encoded
                    alternatives TEXT,  -- JSON encoded
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            """)
            
            # Create index for efficient cleanup
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON ml_predictions(expires_at)
            """)
            
            conn.commit()
    
    def _generate_cache_key(self, message: str, files_changed: List[str]) -> Tuple[str, str, str]:
        """Generate cache key components.
        
        Args:
            message: Commit message
            files_changed: List of changed files
            
        Returns:
            Tuple of (cache_key, message_hash, files_hash)
        """
        import hashlib
        
        message_hash = hashlib.md5(message.encode('utf-8')).hexdigest()
        files_hash = hashlib.md5('|'.join(sorted(files_changed)).encode('utf-8')).hexdigest()
        cache_key = f"{message_hash}:{files_hash}"
        
        return cache_key, message_hash, files_hash
    
    def get_prediction(self, message: str, files_changed: List[str]) -> Optional[Dict[str, Any]]:
        """Get cached prediction if available and not expired.
        
        Args:
            message: Commit message
            files_changed: List of changed files
            
        Returns:
            Cached prediction dictionary or None if not found/expired
        """
        cache_key, _, _ = self._generate_cache_key(message, files_changed)
        
        try:
            with sqlite3.connect(self.cache_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT category, confidence, method, features, alternatives
                    FROM ml_predictions 
                    WHERE key = ? AND expires_at > datetime('now')
                """, (cache_key,))
                
                row = cursor.fetchone()
                if row:
                    import json
                    return {
                        'category': row['category'],
                        'confidence': row['confidence'],
                        'method': 'cached',  # Override method to indicate cached result
                        'features': json.loads(row['features']) if row['features'] else {},
                        'alternatives': json.loads(row['alternatives']) if row['alternatives'] else []
                    }
        
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
        
        return None
    
    def store_prediction(self, message: str, files_changed: List[str], result: Dict[str, Any]) -> None:
        """Store prediction in cache with expiration.
        
        Args:
            message: Commit message
            files_changed: List of changed files
            result: Prediction result to cache
        """
        cache_key, message_hash, files_hash = self._generate_cache_key(message, files_changed)
        
        try:
            import json
            from datetime import datetime, timedelta
            
            expires_at = datetime.now() + timedelta(days=self.expiration_days)
            
            with sqlite3.connect(self.cache_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO ml_predictions 
                    (key, message_hash, files_hash, category, confidence, method, 
                     features, alternatives, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_key, message_hash, files_hash,
                    result['category'], result['confidence'], result['method'],
                    json.dumps(result.get('features', {})),
                    json.dumps(result.get('alternatives', [])),
                    expires_at
                ))
                conn.commit()
        
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def cleanup_expired(self) -> int:
        """Remove expired predictions from cache.
        
        Returns:
            Number of expired entries removed
        """
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM ml_predictions WHERE expires_at <= datetime('now')
                """)
                conn.commit()
                return cursor.rowcount
        
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        COUNT(CASE WHEN expires_at > datetime('now') THEN 1 END) as active_entries,
                        COUNT(CASE WHEN expires_at <= datetime('now') THEN 1 END) as expired_entries,
                        COUNT(DISTINCT method) as unique_methods
                    FROM ml_predictions
                """)
                
                row = cursor.fetchone()
                if row:
                    return {
                        'total_entries': row[0],
                        'active_entries': row[1], 
                        'expired_entries': row[2],
                        'unique_methods': row[3],
                        'cache_file_size_mb': self.cache_path.stat().st_size / (1024 * 1024) if self.cache_path.exists() else 0
                    }
        
        except Exception as e:
            logger.warning(f"Cache statistics failed: {e}")
        
        return {
            'total_entries': 0,
            'active_entries': 0,
            'expired_entries': 0,
            'unique_methods': 0,
            'cache_file_size_mb': 0
        }