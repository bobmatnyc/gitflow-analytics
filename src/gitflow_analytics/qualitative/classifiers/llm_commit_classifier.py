"""LLM-based commit classification with streamlined categories.

This module provides commit classification using Large Language Models (LLMs)
via OpenRouter API. It focuses on 7 streamlined categories optimized for 
enterprise workflows and provides fast, affordable classification with caching.

WHY: Traditional rule-based classification can miss nuanced context that LLMs
excel at understanding. This implementation balances accuracy, speed, and cost
while maintaining reliability through comprehensive fallback mechanisms.

DESIGN DECISIONS:
- 7 streamlined categories for clear, actionable insights
- OpenRouter for model flexibility and cost optimization
- Aggressive caching to minimize API calls and costs
- Simple prompts for reliable, consistent results
- Graceful degradation when LLM services unavailable
"""

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import requests with fallback for graceful degradation
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM-based commit classification."""
    
    # OpenRouter API configuration
    api_key: Optional[str] = None
    api_base_url: str = "https://openrouter.ai/api/v1"
    model: str = "mistralai/mistral-7b-instruct"  # Fast, affordable model
    
    # Classification parameters
    confidence_threshold: float = 0.7  # Minimum confidence for LLM predictions
    max_tokens: int = 50  # Keep responses short
    temperature: float = 0.1  # Low temperature for consistent results
    timeout_seconds: float = 30.0  # API timeout
    
    # Caching configuration
    cache_duration_days: int = 90  # Long cache duration for cost optimization
    enable_caching: bool = True
    
    # Cost optimization
    batch_size: int = 1  # Process one at a time for simplicity
    max_daily_requests: int = 1000  # Rate limiting
    
    # Domain-specific terms for organization
    domain_terms: Dict[str, List[str]] = None
    
    def __post_init__(self):
        """Initialize default domain terms if not provided."""
        if self.domain_terms is None:
            self.domain_terms = {
                "media": [
                    "video", "audio", "streaming", "player", "media", "content",
                    "broadcast", "live", "recording", "episode", "program"
                ],
                "localization": [
                    "translation", "i18n", "l10n", "locale", "language", "spanish",
                    "french", "german", "italian", "portuguese", "multilingual"
                ],
                "integration": [
                    "api", "webhook", "third-party", "external", "service",
                    "integration", "sync", "import", "export", "connector"
                ]
            }


class LLMCommitClassifier:
    """LLM-based commit classifier with streamlined categories.
    
    This classifier uses LLMs via OpenRouter to categorize commits into 7 
    streamlined categories optimized for media and content organizations.
    """
    
    # Streamlined category definitions
    CATEGORIES = {
        "feature": "New functionality, capabilities, enhancements",
        "bugfix": "Fixes, errors, issues, crashes",
        "maintenance": "Configuration, chores, dependencies, cleanup, refactoring", 
        "integration": "Third-party services, APIs, webhooks, external systems",
        "content": "Text, copy, documentation, README updates",
        "media": "Video, audio, streaming, players, visual assets",
        "localization": "Translations, i18n, l10n, regional adaptations"
    }
    
    def __init__(self, config: LLMConfig, cache_dir: Optional[Path] = None):
        """Initialize LLM commit classifier.
        
        Args:
            config: LLM configuration
            cache_dir: Directory for caching predictions
        """
        self.config = config
        self.cache_dir = cache_dir or Path(".gitflow-cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Check dependencies
        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available, LLM classification disabled")
            raise ImportError("requests library required for LLM classification")
        
        # Initialize cache
        self.cache: Optional[LLMPredictionCache] = None
        if self.config.enable_caching:
            try:
                cache_path = self.cache_dir / "llm_predictions.db"
                self.cache = LLMPredictionCache(cache_path, self.config.cache_duration_days)
            except Exception as e:
                logger.warning(f"Failed to initialize LLM cache: {e}")
                self.cache = None
        
        # Request tracking for rate limiting
        self._daily_requests = 0
        self._last_reset_date = datetime.now().date()
        
        # Token and cost tracking
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.api_calls_made = 0
        
        logger.info(f"LLMCommitClassifier initialized with model: {self.config.model}")
    
    def classify_commit(self, message: str, files_changed: Optional[List[str]] = None) -> Dict[str, Any]:
        """Classify a commit message using LLM.
        
        Args:
            message: Cleaned commit message (git artifacts already filtered)
            files_changed: Optional list of changed files for additional context
            
        Returns:
            Classification result with category, confidence, and metadata
        """
        start_time = time.time()
        
        if not message or not message.strip():
            return self._create_result("maintenance", 0.3, "empty_message", start_time)
        
        # Check cache first
        if self.cache:
            cached_result = self.cache.get_prediction(message, files_changed or [])
            if cached_result:
                cached_result['processing_time_ms'] = (time.time() - start_time) * 1000
                cached_result['method'] = 'cached'
                return cached_result
        
        # Check rate limits
        if not self._check_rate_limits():
            logger.warning("Daily API request limit exceeded, skipping LLM classification")
            return self._create_result("maintenance", 0.1, "rate_limited", start_time)
        
        # Make LLM prediction
        try:
            llm_result = self._predict_with_llm(message, files_changed or [])
            llm_result['processing_time_ms'] = (time.time() - start_time) * 1000
            
            # Cache successful predictions
            if self.cache and llm_result['method'] == 'llm':
                self.cache.store_prediction(message, files_changed or [], llm_result)
            
            return llm_result
            
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return self._create_result("maintenance", 0.1, "llm_failed", start_time)
    
    def _predict_with_llm(self, message: str, files_changed: List[str]) -> Dict[str, Any]:
        """Make prediction using LLM API.
        
        Args:
            message: Commit message
            files_changed: List of changed files
            
        Returns:
            Prediction result dictionary
        """
        # Prepare context
        context = self._prepare_context(message, files_changed)
        
        # Create prompt
        prompt = self._create_classification_prompt(message, context)
        
        # Make API request
        response = self._make_api_request(prompt)
        
        # Parse response
        category, confidence, reasoning = self._parse_llm_response(response)
        
        # Validate category
        if category not in self.CATEGORIES:
            logger.warning(f"Invalid category from LLM: {category}, defaulting to maintenance")
            category = "maintenance"
            confidence = max(0.1, confidence - 0.3)
        
        self._daily_requests += 1
        
        return {
            'category': category,
            'confidence': confidence,
            'method': 'llm',
            'reasoning': reasoning,
            'model': self.config.model,
            'alternatives': []
        }
    
    def _prepare_context(self, message: str, files_changed: List[str]) -> Dict[str, Any]:
        """Prepare context information for the LLM.
        
        Args:
            message: Commit message
            files_changed: List of changed files
            
        Returns:
            Context dictionary with relevant information
        """
        context = {
            'file_extensions': [],
            'file_patterns': [],
            'domain_indicators': []
        }
        
        if files_changed:
            # Extract file extensions
            extensions = set()
            for file_path in files_changed:
                ext = Path(file_path).suffix.lower()
                if ext:
                    extensions.add(ext)
            context['file_extensions'] = list(extensions)
            
            # Look for specific file patterns
            patterns = []
            for file_path in files_changed:
                file_lower = file_path.lower()
                if any(term in file_lower for term in ['config', 'settings', '.env', '.yaml', '.json']):
                    patterns.append('configuration')
                elif any(term in file_lower for term in ['test', 'spec', '__test__']):
                    patterns.append('test')
                elif any(term in file_lower for term in ['doc', 'readme', 'changelog']):
                    patterns.append('documentation')
                elif any(term in file_lower for term in ['video', 'audio', 'media', '.mp4', '.mp3']):
                    patterns.append('media')
            context['file_patterns'] = patterns
        
        # Check for domain-specific terms
        message_lower = message.lower()
        for domain, terms in self.config.domain_terms.items():
            if any(term in message_lower for term in terms):
                context['domain_indicators'].append(domain)
        
        return context
    
    def _create_classification_prompt(self, message: str, context: Dict[str, Any]) -> str:
        """Create a focused classification prompt for the LLM.
        
        Args:
            message: Commit message
            context: Additional context information
            
        Returns:
            Formatted prompt string
        """
        categories_desc = "\n".join([
            f"- {cat}: {desc}" 
            for cat, desc in self.CATEGORIES.items()
        ])
        
        context_info = ""
        if context.get('file_extensions'):
            context_info += f"\nFile types: {', '.join(context['file_extensions'])}"
        if context.get('file_patterns'):
            context_info += f"\nFile patterns: {', '.join(context['file_patterns'])}"
        if context.get('domain_indicators'):
            context_info += f"\nDomain indicators: {', '.join(context['domain_indicators'])}"
        
        prompt = f"""Classify this commit message into one of these 7 categories:

{categories_desc}

Commit message: "{message}"{context_info}

Respond with only: CATEGORY_NAME confidence_score reasoning
Example: feature 0.85 adds new user authentication system

Response:"""
        
        return prompt
    
    def _make_api_request(self, prompt: str) -> str:
        """Make API request to OpenRouter.
        
        Args:
            prompt: The classification prompt
            
        Returns:
            Raw API response text
            
        Raises:
            Exception: If API request fails
        """
        if not self.config.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/gitflow-analytics",
            "X-Title": "GitFlow Analytics"
        }
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        url = f"{self.config.api_base_url}/chat/completions"
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.config.timeout_seconds
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
        
        data = response.json()
        
        if 'choices' not in data or not data['choices']:
            raise Exception("No response choices in API response")
        
        # Track token usage if available in response
        if 'usage' in data:
            usage = data['usage']
            tokens_used = usage.get('total_tokens', 0)
            self.total_tokens_used += tokens_used
            
            # Estimate cost based on model (rough estimates)
            # Mistral-7B on OpenRouter: ~$0.00025 per 1k tokens
            cost_per_1k = 0.00025
            if 'gpt' in self.config.model.lower():
                cost_per_1k = 0.001  # GPT-3.5 pricing
            
            cost = (tokens_used / 1000) * cost_per_1k
            self.total_cost += cost
            self.api_calls_made += 1
            
            logger.debug(f"API call used {tokens_used} tokens, cost: ${cost:.6f}")
        
        return data['choices'][0]['message']['content'].strip()
    
    def _parse_llm_response(self, response: str) -> Tuple[str, float, str]:
        """Parse LLM response to extract category, confidence, and reasoning.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Tuple of (category, confidence, reasoning)
        """
        try:
            # Expected format: "category confidence reasoning"
            parts = response.split(' ', 2)
            
            if len(parts) < 2:
                raise ValueError(f"Invalid response format: {response}")
            
            category = parts[0].lower().strip()
            confidence_str = parts[1].strip()
            reasoning = parts[2].strip() if len(parts) > 2 else "No reasoning provided"
            
            # Parse confidence
            try:
                confidence = float(confidence_str)
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            except ValueError:
                logger.warning(f"Invalid confidence score: {confidence_str}, using 0.5")
                confidence = 0.5
            
            return category, confidence, reasoning
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response '{response}': {e}")
            return "maintenance", 0.1, f"Parse error: {str(e)}"
    
    def _check_rate_limits(self) -> bool:
        """Check if we're within daily rate limits.
        
        Returns:
            True if request is allowed, False if rate limited
        """
        current_date = datetime.now().date()
        
        # Reset counter if it's a new day
        if current_date > self._last_reset_date:
            self._daily_requests = 0
            self._last_reset_date = current_date
        
        return self._daily_requests < self.config.max_daily_requests
    
    def _create_result(self, category: str, confidence: float, method: str, start_time: float) -> Dict[str, Any]:
        """Create a standardized result dictionary.
        
        Args:
            category: Classification category
            confidence: Confidence score
            method: Classification method used
            start_time: Processing start time
            
        Returns:
            Standardized result dictionary
        """
        return {
            'category': category,
            'confidence': confidence,
            'method': method,
            'reasoning': f"Classified using {method}",
            'model': self.config.model if method == 'llm' else 'rule-based',
            'alternatives': [],
            'processing_time_ms': (time.time() - start_time) * 1000
        }
    
    def classify_commits_batch(
        self,
        commits: List[Dict[str, Any]],
        batch_id: Optional[str] = None,
        include_confidence: bool = True
    ) -> List[Dict[str, Any]]:
        """Classify a batch of commits.
        
        Args:
            commits: List of commit dictionaries with enhanced context
            batch_id: Optional batch identifier for tracking
            include_confidence: Whether to include confidence scores
            
        Returns:
            List of classification results for each commit
        """
        results = []
        
        for commit in commits:
            # Extract commit message and files from enhanced commit
            message = commit.get('message', '')
            files_changed = []
            
            # Handle different formats of files_changed
            if 'files_changed' in commit:
                fc = commit['files_changed']
                if isinstance(fc, list):
                    files_changed = fc
                elif isinstance(fc, int):
                    # If it's just a count, we don't have file names
                    files_changed = []
            
            # Classify individual commit
            result = self.classify_commit(message, files_changed)
            
            # Add batch_id if provided
            if batch_id:
                result['batch_id'] = batch_id
                
            results.append(result)
            
        logger.info(f"Batch {batch_id}: Classified {len(results)} commits")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get classifier usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        stats = {
            'daily_requests': self._daily_requests,
            'max_daily_requests': self.config.max_daily_requests,
            'model': self.config.model,
            'cache_enabled': self.config.enable_caching,
            'api_configured': bool(self.config.api_key),
            'total_tokens_used': self.total_tokens_used,
            'total_cost': self.total_cost,
            'api_calls_made': self.api_calls_made,
            'average_tokens_per_call': self.total_tokens_used / self.api_calls_made if self.api_calls_made > 0 else 0
        }
        
        if self.cache:
            stats['cache_statistics'] = self.cache.get_statistics()
        
        return stats


class LLMPredictionCache:
    """SQLite-based cache for LLM predictions with long expiration.
    
    WHY: LLM API calls are expensive and relatively slow. Caching predictions
    for extended periods (90+ days) significantly reduces costs and improves
    performance while maintaining accuracy.
    """
    
    def __init__(self, cache_path: Path, expiration_days: int = 90):
        """Initialize LLM prediction cache.
        
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
                CREATE TABLE IF NOT EXISTS llm_predictions (
                    key TEXT PRIMARY KEY,
                    message_hash TEXT NOT NULL,
                    files_hash TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    method TEXT NOT NULL,
                    reasoning TEXT,
                    model TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            """)
            
            # Create indices for efficient lookup and cleanup
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON llm_predictions(expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_message_hash ON llm_predictions(message_hash)")
            
            conn.commit()
    
    def _generate_cache_key(self, message: str, files_changed: List[str]) -> Tuple[str, str, str]:
        """Generate cache key components.
        
        Args:
            message: Commit message
            files_changed: List of changed files
            
        Returns:
            Tuple of (cache_key, message_hash, files_hash)
        """
        message_hash = hashlib.md5(message.encode('utf-8')).hexdigest()
        files_hash = hashlib.md5('|'.join(sorted(files_changed)).encode('utf-8')).hexdigest()
        cache_key = f"llm:{message_hash}:{files_hash}"
        
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
                    SELECT category, confidence, reasoning, model
                    FROM llm_predictions 
                    WHERE key = ? AND expires_at > datetime('now')
                """, (cache_key,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'category': row['category'],
                        'confidence': row['confidence'],
                        'method': 'cached',
                        'reasoning': row['reasoning'] or 'Cached result',
                        'model': row['model'] or 'unknown',
                        'alternatives': []
                    }
        
        except Exception as e:
            logger.warning(f"LLM cache lookup failed: {e}")
        
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
            expires_at = datetime.now() + timedelta(days=self.expiration_days)
            
            with sqlite3.connect(self.cache_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO llm_predictions 
                    (key, message_hash, files_hash, category, confidence, 
                     method, reasoning, model, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_key, message_hash, files_hash,
                    result['category'], result['confidence'],
                    result.get('method', 'llm'),
                    result.get('reasoning', ''),
                    result.get('model', ''),
                    expires_at
                ))
                conn.commit()
        
        except Exception as e:
            logger.warning(f"LLM cache storage failed: {e}")
    
    def cleanup_expired(self) -> int:
        """Remove expired predictions from cache.
        
        Returns:
            Number of expired entries removed
        """
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM llm_predictions WHERE expires_at <= datetime('now')
                """)
                conn.commit()
                return cursor.rowcount
        
        except Exception as e:
            logger.warning(f"LLM cache cleanup failed: {e}")
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
                        COUNT(DISTINCT model) as unique_models
                    FROM llm_predictions
                """)
                
                row = cursor.fetchone()
                if row:
                    return {
                        'total_entries': row[0],
                        'active_entries': row[1], 
                        'expired_entries': row[2],
                        'unique_models': row[3],
                        'cache_file_size_mb': self.cache_path.stat().st_size / (1024 * 1024) if self.cache_path.exists() else 0
                    }
        
        except Exception as e:
            logger.warning(f"LLM cache statistics failed: {e}")
        
        return {
            'total_entries': 0,
            'active_entries': 0,
            'expired_entries': 0,
            'unique_models': 0,
            'cache_file_size_mb': 0
        }