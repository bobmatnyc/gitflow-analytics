"""LLM fallback system for uncertain commit classifications using OpenRouter."""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import os

from ..models.schemas import LLMConfig, QualitativeCommitData
from ..utils.cost_tracker import CostTracker
from ..utils.text_processing import TextProcessor

try:
    import openai
    import tiktoken
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Create mock objects for type hints when not available
    class MockOpenAI:
        class OpenAI:
            pass
    openai = MockOpenAI()
    tiktoken = None


class ModelRouter:
    """Smart model selection based on complexity and cost constraints."""
    
    def __init__(self, config: LLMConfig, cost_tracker: CostTracker):
        """Initialize model router.
        
        Args:
            config: LLM configuration
            cost_tracker: Cost tracking instance
        """
        self.config = config
        self.cost_tracker = cost_tracker
        self.logger = logging.getLogger(__name__)
        
    def select_model(self, complexity_score: float, batch_size: int) -> str:
        """Select appropriate model based on complexity and budget.
        
        Args:
            complexity_score: Complexity score (0.0 to 1.0)
            batch_size: Number of commits in batch
            
        Returns:
            Selected model name
        """
        # Check daily budget remaining
        remaining_budget = self.cost_tracker.check_budget_remaining()
        
        # If we're over budget, use free model
        if remaining_budget <= 0:
            self.logger.warning("Daily budget exceeded, using free model")
            return self.config.fallback_model
            
        # For simple cases or when budget is tight, use free model
        if complexity_score < 0.3 or remaining_budget < 0.50:
            return self.config.fallback_model
            
        # For complex cases with sufficient budget, use premium model
        if complexity_score > self.config.complexity_threshold and remaining_budget > 2.0:
            return self.config.complex_model
            
        # Default to primary model (Claude Haiku - fast and cheap)
        return self.config.primary_model


class LLMFallback:
    """Strategic LLM usage for uncertain cases via OpenRouter.
    
    This class provides intelligent fallback to LLM processing when NLP
    classification confidence is below the threshold. It uses OpenRouter
    to access multiple models cost-effectively.
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize LLM fallback system.
        
        Args:
            config: LLM configuration
            
        Raises:
            ImportError: If OpenAI library is not available
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library required for LLM fallback. Install with: pip install openai"
            )
            
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenRouter client
        self.client = self._initialize_openrouter_client()
        
        # Initialize utilities
        self.cost_tracker = CostTracker(daily_budget=config.max_daily_cost)
        self.model_router = ModelRouter(config, self.cost_tracker)
        self.text_processor = TextProcessor()
        
        # Batch processing cache
        self.batch_cache = {}
        
        # Token encoder for cost estimation
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        except Exception:
            self.encoding = None
            self.logger.warning("Could not load tiktoken encoder, token estimation may be inaccurate")
            
        self.logger.info("LLM fallback system initialized with OpenRouter")
        
    def _initialize_openrouter_client(self) -> openai.OpenAI:
        """Initialize OpenRouter client with API key.
        
        Returns:
            Configured OpenAI client for OpenRouter
            
        Raises:
            ValueError: If API key is not configured
        """
        api_key = self._resolve_api_key()
        if not api_key:
            raise ValueError(
                "OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable."
            )
            
        return openai.OpenAI(
            base_url=self.config.base_url,
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/bobmatnyc/gitflow-analytics",
                "X-Title": "GitFlow Analytics - Qualitative Analysis"
            }
        )
        
    def _resolve_api_key(self) -> Optional[str]:
        """Resolve OpenRouter API key from config or environment.
        
        Returns:
            API key string or None if not found
        """
        api_key = self.config.openrouter_api_key
        
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            return os.environ.get(env_var)
        else:
            return api_key
            
    def group_similar_commits(self, commits: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group similar commits for efficient batch processing.
        
        Args:
            commits: List of commit dictionaries
            
        Returns:
            List of commit groups
        """
        if not commits:
            return []
            
        groups = []
        similarity_threshold = self.config.similarity_threshold
        
        for commit in commits:
            # Find similar group or create new one
            placed = False
            
            for group in groups:
                if len(group) >= self.config.max_group_size:
                    continue  # Group is full
                    
                # Calculate similarity with first commit in group
                similarity = self.text_processor.calculate_message_similarity(
                    commit.get('message', ''), 
                    group[0].get('message', '')
                )
                
                if similarity > similarity_threshold:
                    group.append(commit)
                    placed = True
                    break
                    
            if not placed:
                groups.append([commit])
                
        self.logger.debug(f"Grouped {len(commits)} commits into {len(groups)} groups")
        return groups
        
    def process_group(self, commits: List[Dict[str, Any]]) -> List[QualitativeCommitData]:
        """Process a group of similar commits with OpenRouter.
        
        Args:
            commits: List of similar commit dictionaries
            
        Returns:
            List of QualitativeCommitData with LLM analysis
        """
        if not commits:
            return []
            
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_group_cache_key(commits)
        if cache_key in self.batch_cache:
            self.logger.debug(f"Using cached result for {len(commits)} commits")
            template_result = self.batch_cache[cache_key]
            return self._apply_template_to_group(template_result, commits)
            
        # Assess complexity and select model
        complexity_score = self._assess_complexity(commits)
        selected_model = self.model_router.select_model(complexity_score, len(commits))
        
        self.logger.debug(
            f"Processing {len(commits)} commits with {selected_model} "
            f"(complexity: {complexity_score:.2f})"
        )
        
        # Build optimized prompt
        prompt = self._build_batch_classification_prompt(commits)
        
        # Estimate tokens and cost
        estimated_input_tokens = self._estimate_tokens(prompt)
        if not self.cost_tracker.can_afford_call(selected_model, estimated_input_tokens * 2):
            self.logger.warning("Cannot afford LLM call, using fallback model")
            selected_model = self.config.fallback_model
            
        # Make OpenRouter API call
        try:
            response = self._call_openrouter(prompt, selected_model)
            processing_time = time.time() - start_time
            
            # Parse response
            results = self._parse_llm_response(response, commits)
            
            # Track costs and performance
            estimated_output_tokens = self._estimate_tokens(response)
            self.cost_tracker.record_call(
                model=selected_model,
                input_tokens=estimated_input_tokens,
                output_tokens=estimated_output_tokens,
                processing_time=processing_time,
                batch_size=len(commits),
                success=len(results) > 0
            )
            
            # Cache successful result
            if results:
                self.batch_cache[cache_key] = self._create_template_from_results(results)
                
            # Update processing time in results
            for result in results:
                result.processing_time_ms = (processing_time * 1000) / len(results)
                
            return results
            
        except Exception as e:
            self.logger.error(f"OpenRouter processing failed: {e}")
            
            # Record failed call
            self.cost_tracker.record_call(
                model=selected_model,
                input_tokens=estimated_input_tokens,
                output_tokens=0,
                processing_time=time.time() - start_time,
                batch_size=len(commits),
                success=False,
                error_message=str(e)
            )
            
            # Try fallback model if primary failed
            if selected_model != self.config.fallback_model:
                return self._retry_with_fallback_model(commits, prompt)
            else:
                return self._create_fallback_results(commits)
                
    def _call_openrouter(self, prompt: str, model: str) -> str:
        """Make API call to OpenRouter with selected model.
        
        Args:
            prompt: Classification prompt
            model: Model to use
            
        Returns:
            Response content
            
        Raises:
            Exception: If API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert Git commit classifier. Analyze commits and respond only with valid JSON. Be concise but accurate."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenRouter API call failed: {e}")
            raise
            
    def _build_batch_classification_prompt(self, commits: List[Dict[str, Any]]) -> str:
        """Build optimized prompt for OpenRouter batch processing.
        
        Args:
            commits: List of commit dictionaries
            
        Returns:
            Formatted prompt string
        """
        # Limit to max group size for token management
        commits_to_process = commits[:self.config.max_group_size]
        
        commit_data = []
        for i, commit in enumerate(commits_to_process, 1):
            message = commit.get('message', '')[:150]  # Truncate long messages
            files = commit.get('files_changed', [])
            
            # Include key file context
            files_context = ""
            if files:
                key_files = files[:5]  # Top 5 files
                files_context = f" | Modified: {', '.join(key_files)}"
                
            # Add size context
            insertions = commit.get('insertions', 0)
            deletions = commit.get('deletions', 0)
            size_context = f" | +{insertions}/-{deletions}"
            
            commit_data.append(f"{i}. {message}{files_context}{size_context}")
            
        prompt = f"""Analyze these Git commits and classify each one. Consider the commit message, modified files, and change size.

Commits to classify:
{chr(10).join(commit_data)}

For each commit, provide:
- change_type: feature|bugfix|refactor|docs|test|chore|security|hotfix|config
- business_domain: frontend|backend|database|infrastructure|mobile|devops|unknown
- risk_level: low|medium|high|critical  
- confidence: 0.0-1.0 (classification certainty)
- urgency: routine|important|urgent|critical
- complexity: simple|moderate|complex

Respond with JSON array only:
[{{"id": 1, "change_type": "feature", "business_domain": "frontend", "risk_level": "low", "confidence": 0.9, "urgency": "routine", "complexity": "moderate"}}]"""

        return prompt
        
    def _parse_llm_response(self, response: str, commits: List[Dict[str, Any]]) -> List[QualitativeCommitData]:
        """Parse LLM response into QualitativeCommitData objects.
        
        Args:
            response: JSON response from LLM
            commits: Original commit dictionaries
            
        Returns:
            List of QualitativeCommitData objects
        """
        try:
            # Clean response (remove any markdown formatting)
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            classifications = json.loads(cleaned_response)
            
            if not isinstance(classifications, list):
                raise ValueError("Response is not a JSON array")
                
            results = []
            
            for i, commit in enumerate(commits):
                if i < len(classifications):
                    classification = classifications[i]
                else:
                    # Fallback if fewer classifications than commits
                    classification = {
                        'change_type': 'unknown',
                        'business_domain': 'unknown',
                        'risk_level': 'medium',
                        'confidence': 0.5,
                        'urgency': 'routine',
                        'complexity': 'moderate'
                    }
                    
                result = QualitativeCommitData(
                    # Copy existing commit fields
                    hash=commit.get('hash', ''),
                    message=commit.get('message', ''),
                    author_name=commit.get('author_name', ''),
                    author_email=commit.get('author_email', ''),
                    timestamp=commit.get('timestamp', time.time()),
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
                    
                    # Intent signals from LLM analysis
                    intent_signals={
                        'urgency': classification.get('urgency', 'routine'),
                        'complexity': classification.get('complexity', 'moderate'),
                        'confidence': classification.get('confidence', 0.5),
                        'signals': [f"llm_classified:{classification.get('change_type', 'unknown')}"]
                    },
                    collaboration_patterns={},
                    technical_context={
                        'llm_model': 'openrouter',
                        'processing_method': 'batch'
                    },
                    
                    # Processing metadata
                    processing_method='llm',
                    processing_time_ms=0,  # Set by caller
                    confidence_score=classification.get('confidence', 0.5)
                )
                results.append(result)
                
            return results
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            self.logger.debug(f"Raw response: {response}")
            return self._create_fallback_results(commits)
            
    def _assess_complexity(self, commits: List[Dict[str, Any]]) -> float:
        """Assess complexity of commits for model selection.
        
        Args:
            commits: List of commit dictionaries
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        if not commits:
            return 0.0
            
        total_complexity = 0.0
        
        for commit in commits:
            # Message complexity
            message = commit.get('message', '')
            message_complexity = min(1.0, len(message.split()) / 20.0)
            
            # File change complexity
            files_changed = len(commit.get('files_changed', []))
            file_complexity = min(1.0, files_changed / 15.0)
            
            # Size complexity
            total_changes = commit.get('insertions', 0) + commit.get('deletions', 0)
            size_complexity = min(1.0, total_changes / 200.0)
            
            # Combine complexities
            commit_complexity = (message_complexity * 0.3 + 
                               file_complexity * 0.4 + 
                               size_complexity * 0.3)
            total_complexity += commit_complexity
            
        return total_complexity / len(commits)
        
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception:
                pass
                
        # Fallback estimation (roughly 4 characters per token)
        return len(text) // 4
        
    def _generate_group_cache_key(self, commits: List[Dict[str, Any]]) -> str:
        """Generate cache key for a group of commits.
        
        Args:
            commits: List of commit dictionaries
            
        Returns:
            Cache key string
        """
        # Create fingerprint from commit messages and file patterns
        fingerprints = []
        for commit in commits:
            message = commit.get('message', '')
            files = commit.get('files_changed', [])
            fingerprint = self.text_processor.create_semantic_fingerprint(message, files)
            fingerprints.append(fingerprint)
            
        combined_fingerprint = '|'.join(sorted(fingerprints))
        return hashlib.md5(combined_fingerprint.encode()).hexdigest()
        
    def _create_template_from_results(self, results: List[QualitativeCommitData]) -> Dict[str, Any]:
        """Create a template from successful results for caching.
        
        Args:
            results: List of analysis results
            
        Returns:
            Template dictionary
        """
        if not results:
            return {}
            
        # Use first result as template
        template = results[0]
        return {
            'change_type': template.change_type,
            'business_domain': template.business_domain,
            'risk_level': template.risk_level,
            'confidence_score': template.confidence_score
        }
        
    def _apply_template_to_group(self, template: Dict[str, Any], 
                                commits: List[Dict[str, Any]]) -> List[QualitativeCommitData]:
        """Apply cached template to a group of commits.
        
        Args:
            template: Cached analysis template
            commits: List of commit dictionaries
            
        Returns:
            List of QualitativeCommitData using template
        """
        results = []
        
        for commit in commits:
            result = QualitativeCommitData(
                # Copy existing commit fields
                hash=commit.get('hash', ''),
                message=commit.get('message', ''),
                author_name=commit.get('author_name', ''),
                author_email=commit.get('author_email', ''),
                timestamp=commit.get('timestamp', time.time()),
                files_changed=commit.get('files_changed', []),
                insertions=commit.get('insertions', 0),
                deletions=commit.get('deletions', 0),
                
                # Apply template values
                change_type=template.get('change_type', 'unknown'),
                change_type_confidence=template.get('confidence_score', 0.5),
                business_domain=template.get('business_domain', 'unknown'),
                domain_confidence=template.get('confidence_score', 0.5),
                risk_level=template.get('risk_level', 'medium'),
                risk_factors=[],
                
                intent_signals={'confidence': template.get('confidence_score', 0.5)},
                collaboration_patterns={},
                technical_context={'processing_method': 'cached_template'},
                
                # Processing metadata
                processing_method='llm',
                processing_time_ms=1.0,  # Very fast for cached results
                confidence_score=template.get('confidence_score', 0.5)
            )
            results.append(result)
            
        return results
        
    def _retry_with_fallback_model(self, commits: List[Dict[str, Any]], 
                                  prompt: str) -> List[QualitativeCommitData]:
        """Retry processing with fallback model.
        
        Args:
            commits: List of commit dictionaries
            prompt: Classification prompt
            
        Returns:
            List of QualitativeCommitData or fallback results
        """
        try:
            self.logger.info(f"Retrying with fallback model: {self.config.fallback_model}")
            response = self._call_openrouter(prompt, self.config.fallback_model)
            return self._parse_llm_response(response, commits)
        except Exception as e:
            self.logger.error(f"Fallback model also failed: {e}")
            return self._create_fallback_results(commits)
            
    def _create_fallback_results(self, commits: List[Dict[str, Any]]) -> List[QualitativeCommitData]:
        """Create fallback results when LLM processing fails.
        
        Args:
            commits: List of commit dictionaries
            
        Returns:
            List of QualitativeCommitData with default values
        """
        results = []
        
        for commit in commits:
            result = QualitativeCommitData(
                # Basic commit info
                hash=commit.get('hash', ''),
                message=commit.get('message', ''),
                author_name=commit.get('author_name', ''),
                author_email=commit.get('author_email', ''),
                timestamp=commit.get('timestamp', time.time()),
                files_changed=commit.get('files_changed', []),
                insertions=commit.get('insertions', 0),
                deletions=commit.get('deletions', 0),
                
                # Default classifications
                change_type='unknown',
                change_type_confidence=0.0,
                business_domain='unknown',
                domain_confidence=0.0,
                risk_level='medium',
                risk_factors=['llm_processing_failed'],
                intent_signals={'confidence': 0.0},
                collaboration_patterns={},
                technical_context={'processing_method': 'fallback'},
                
                # Processing metadata
                processing_method='llm',
                processing_time_ms=0.0,
                confidence_score=0.0
            )
            results.append(result)
            
        return results