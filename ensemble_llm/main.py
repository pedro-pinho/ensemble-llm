import asyncio
import aiohttp
import json
import numpy as np
import logging
import sys
import traceback
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from urllib.parse import quote
from pathlib import Path

from .web_search import WebSearcher
from .logger import setup_logger
from .config import (
    DEFAULT_MODELS, MODEL_CONFIGS, WEB_SEARCH_CONFIG, ENSEMBLE_CONFIG,
    SYSTEM_CONFIGS, PERFORMANCE_THRESHOLDS, MODEL_POOLS, QUERY_PATTERNS,
    LOGGING_CONFIG, TRACKING_CONFIG, OLLAMA_ENDPOINTS, GENERATION_OPTIONS,
    FEATURES, DISPLAY_CONFIG, ERROR_MESSAGES, SUCCESS_MESSAGES, SMART_LEARNING_CONFIG,
    SPEED_PROFILES, SPEED_OPTIMIZED_MODELS, FAST_MODEL_CONFIGS
)
from .performance_tracker import ModelPerformanceTracker, AdaptiveModelManager
from .learning_system import (
    SmartEnsembleOrchestrator, QueryCache, ModelOptimizer, 
    QueryPatternLearner, PrecomputeManager
)
from .fast_mode import FastModeOrchestrator, TurboMode, ModelWarmup
from .verbose_logger import VerboseFileLogger, ModelPerformanceLogger, LiveTailLogger


class EnsembleLLM:
    def __init__(self, models: List[str] = None, ollama_host: str = None, 
                 enable_web_search: bool = None, adaptive_mode: bool = None,
                 smart_learning: bool = True, speed_mode: str = 'balanced',
                 verbose_logging: bool = True):
        
        # Use config values with overrides
        self.host = ollama_host or OLLAMA_ENDPOINTS['default_host']
        self.models = models or DEFAULT_MODELS
        self.vectorizer = TfidfVectorizer()
        
        # Feature flags from config
        self.enable_web_search = enable_web_search if enable_web_search is not None else FEATURES['web_search']
        self.adaptive_mode = adaptive_mode if adaptive_mode is not None else FEATURES['adaptive_models']
        
        # Initialize web searcher if enabled
        self.web_searcher = WebSearcher() if self.enable_web_search else None
        
        # Setup logging with config
        self.logger = setup_logger('EnsembleLLM', LOGGING_CONFIG['default_level'])
        
        # Adaptive optimization from config
        if self.adaptive_mode and FEATURES['performance_tracking']:
            self.performance_tracker = ModelPerformanceTracker(data_dir=TRACKING_CONFIG['data_dir'])
            self.model_manager = AdaptiveModelManager(self.host)
            self.logger.info("Adaptive mode enabled - models will be optimized based on performance")
        else:
            self.performance_tracker = None
            self.model_manager = None
        
        # Query counter for periodic optimization
        self.query_count = 0
        self.optimization_interval = ENSEMBLE_CONFIG['optimization_interval']
        
        # Cache for model availability (using config)
        self.model_availability_cache = {}
        self.cache_timestamp = None
        self.cache_ttl = WEB_SEARCH_CONFIG['cache_ttl'] if FEATURES['caching'] else 0
        
        # Display settings from config
        self.display_config = DISPLAY_CONFIG
        
        # Success message from config
        self.logger.info(SUCCESS_MESSAGES['initialized'].format(count=len(self.models)))
        self.logger.info(f"Web search: {'Enabled' if self.enable_web_search else 'Disabled'}")
        self.logger.info(f"Adaptive optimization: {'Enabled' if self.adaptive_mode else 'Disabled'}")

        self.smart_learning = smart_learning
        if smart_learning:
            self.orchestrator = SmartEnsembleOrchestrator(
                data_dir=TRACKING_CONFIG.get('smart_data_dir', 'smart_data')
            )
            self.precompute_manager = PrecomputeManager(
                precompute_dir=str(Path(TRACKING_CONFIG.get('smart_data_dir', 'smart_data')) / 'precompute')
            )
            self.logger.info("Smart learning system enabled - queries will be optimized over time")
            
            # Load performance insights
            insights = self.orchestrator.get_performance_insights()
            self.logger.info(f"Loaded smart data: {insights['total_queries']} previous queries, "
                           f"{insights['cache_hit_rate']:.1f}% cache hit rate")
        else:
            self.orchestrator = None
            self.precompute_manager = None

        # Speed optimization
        self.speed_mode = speed_mode
        self.speed_profile = SPEED_PROFILES.get(speed_mode, SPEED_PROFILES['balanced'])
        
        # Initialize fast mode components
        self.fast_orchestrator = FastModeOrchestrator()
        self.turbo_mode = TurboMode(self.host)
        self.model_warmup = ModelWarmup(self.host)

        self.verbose_logging = verbose_logging
        if verbose_logging:
            self.verbose_logger = VerboseFileLogger()
            self.performance_logger = ModelPerformanceLogger()
            self.live_logger = LiveTailLogger()
            
            # Log initialization
            self.verbose_logger.log_query_start(
                query_id=0,
                prompt="SYSTEM INITIALIZATION",
                models=self.models,
                speed_mode=speed_mode,
                web_search=self.enable_web_search
            )

        # Optimize model selection based on speed mode
        if speed_mode in SPEED_OPTIMIZED_MODELS:
            available_models = SPEED_OPTIMIZED_MODELS[speed_mode]
            # Only use models that are available
            self.models = [m for m in available_models if m in (models or DEFAULT_MODELS)]
            if not self.models:
                self.models = models or DEFAULT_MODELS
            
            # Limit number of models based on speed profile
            self.models = self.models[:self.speed_profile['max_models']]
            
            self.logger.info(f"Speed mode '{speed_mode}' active: Using {len(self.models)} models")
    
    
    async def initialize(self):
        """Async initialization - preload models and resolve names"""
        # Resolve model names to actual tags
        self.models = await self.resolve_model_names(self.models)

        # Warmup models for faster first response
        if self.speed_mode in ['turbo', 'fast']:
            await self.model_warmup.parallel_warmup(
                self.models, 
                max_concurrent=2
            )
        
        if self.model_manager and FEATURES['staggered_starts']:
            # Preload models with staggered starts
            await self.model_manager.preload_models(self.models)
            
            # Initial optimization based on historical data
            if FEATURES['model_rotation']:
                self.models = await self.model_manager.optimize_models(self.models)
                self.logger.info(SUCCESS_MESSAGES['optimization_complete'].format(
                    models=', '.join(self.models)
                ))
        if self.smart_learning and self.orchestrator:
            # Get optimized model order based on learning
            self.models = self.orchestrator.get_optimized_models(
                self.models, 
                query_type=None  # General optimization
            )
            
            # Precompute common queries if needed
            common_queries = [
                "Hello",
                "What can you help me with?",
                "Explain quantum computing",
                "Write a Python function",
                "What is machine learning?"
            ]
            
            if self.precompute_manager:
                await self.precompute_manager.precompute_common_queries(
                    self.models[:3],  # Only precompute for top 3 models
                    common_queries
                )
    
    async def resolve_model_names(self, models: List[str]) -> List[str]:
        """Resolve model names to their actual tags in Ollama"""
        
        resolved_models = []
        
        try:
            # Get list of available models using config endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.host}{OLLAMA_ENDPOINTS['tags']}") as response:
                    if response.status == 200:
                        data = await response.json()
                        available_models = {m['name']: m['name'] for m in data.get('models', [])}
                        
                        # Create mapping for base names
                        base_mapping = {}
                        for model_name in available_models.keys():
                            base_name = model_name.split(':')[0]
                            if base_name not in base_mapping:
                                base_mapping[base_name] = model_name
                        
                        # Resolve each model
                        for model in models:
                            if model in available_models:
                                resolved_models.append(model)
                            elif model in base_mapping:
                                resolved = base_mapping[model]
                                self.logger.info(f"Resolved '{model}' to '{resolved}'")
                                resolved_models.append(resolved)
                            else:
                                # Try with :latest tag
                                latest_tag = f"{model}:latest"
                                if latest_tag in available_models:
                                    self.logger.info(f"Resolved '{model}' to '{latest_tag}'")
                                    resolved_models.append(latest_tag)
                                else:
                                    self.logger.warning(ERROR_MESSAGES['invalid_model'].format(model=model))
                                    resolved_models.append(model)
                    else:
                        return models
                        
        except Exception as e:
            if FEATURES['verbose_errors']:
                self.logger.error(f"Failed to resolve model names: {str(e)}")
            return models
        
        return resolved_models if resolved_models else models
    
    async def check_model_availability(self, force_refresh: bool = False) -> Dict[str, bool]:
        """Check which models are actually available with caching"""
        
        now = datetime.now()
        
        # Use cache if valid and caching is enabled
        if FEATURES['caching'] and not force_refresh and self.cache_timestamp and self.model_availability_cache:
            cache_age = (now - self.cache_timestamp).total_seconds()
            if cache_age < self.cache_ttl:
                return self.model_availability_cache
        
        # Refresh availability
        availability = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.host}{OLLAMA_ENDPOINTS['tags']}") as response:
                    if response.status == 200:
                        data = await response.json()
                        available_models = [m['name'] for m in data.get('models', [])]
                        
                        for model in self.models:
                            availability[model] = model in available_models
                        
                        if FEATURES['caching']:
                            self.model_availability_cache = availability
                            self.cache_timestamp = now
                        
        except Exception as e:
            if FEATURES['verbose_errors']:
                self.logger.error(f"Failed to check model availability: {str(e)}")
            # Assume all models are available if we can't check
            availability = {model: True for model in self.models}
        
        return availability
    
    async def detect_uncertainty(self, response: str) -> bool:
        """Detect if a model's response indicates uncertainty about current information"""
        
        # Use uncertainty threshold from config
        uncertainty_phrases = [
            "i don't have information",
            "i don't know",
            "i'm not sure",
            "i cannot provide current",
            "as of my knowledge cutoff",
            "i don't have access to real-time",
            "i don't have current data",
            "my training data",
            "i cannot access current",
            "i'm unable to provide recent",
            "my knowledge is limited to",
            "i don't have up-to-date"
        ]
        
        response_lower = response.lower()
        matches = sum(1 for phrase in uncertainty_phrases if phrase in response_lower)
        
        # Use threshold from config
        uncertainty_score = matches / len(uncertainty_phrases)
        return uncertainty_score >= WEB_SEARCH_CONFIG['uncertainty_threshold']
    
    async def detect_query_type(self, prompt: str) -> Optional[str]:
        """Detect the type of query for specialized model selection using config patterns"""
        
        if not FEATURES['specialized_selection']:
            return None
        
        prompt_lower = prompt.lower()
        
        # Check each query pattern from config
        for query_type, patterns in QUERY_PATTERNS.items():
            if any(pattern in prompt_lower for pattern in patterns):
                return query_type
        
        return None
    
    async def query_model_fast(self, model: str, prompt: str) -> Dict:
        """Fast query implementation"""
        
        if self.speed_mode == 'turbo':
            # Use turbo mode for ultra-fast responses
            return await self.turbo_mode.turbo_query(model, prompt)
        
        # Use speed-optimized parameters
        start_time = datetime.now()
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.speed_profile['temperature'],
                    "num_predict": self.speed_profile['num_predict'],
                    "num_ctx": 2048 if self.speed_mode != 'turbo' else 1024,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_thread": 4  # Limit CPU threads
                },
                "keep_alive": "5m"  # Keep model loaded
            }
            
            # Get timeout from fast model configs or speed profile
            timeout = FAST_MODEL_CONFIGS.get(model, {}).get('timeout', 
                                            self.speed_profile['timeout'])
            
            try:
                async with session.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(
                        total=timeout,
                        connect=2,  # Fast connect timeout
                        sock_read=timeout
                    )
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        return {
                            "model": model,
                            "response": result.get("response", ""),
                            "success": True,
                            "response_time": (datetime.now() - start_time).total_seconds(),
                            "speed_mode": self.speed_mode
                        }
                    else:
                        return {
                            "model": model,
                            "response": f"HTTP {response.status}",
                            "success": False,
                            "response_time": (datetime.now() - start_time).total_seconds()
                        }
                        
            except asyncio.TimeoutError:
                self.logger.warning(f"Fast timeout for {model} after {timeout}s")
                return {
                    "model": model,
                    "response": f"Timeout ({timeout}s)",
                    "success": False,
                    "response_time": (datetime.now() - start_time).total_seconds()
                }
            except Exception as e:
                return {
                    "model": model,
                    "response": str(e)[:100],
                    "success": False,
                    "response_time": (datetime.now() - start_time).total_seconds()
                }

    async def query_all_models_fast(self, prompt: str) -> List[Dict]:
        """Query models using fast strategies"""
        
        strategy = self.speed_profile['strategy']
        
        if strategy == 'race':
            # Race strategy: return as soon as we get 1-2 good responses
            min_responses = 1 if self.speed_mode == 'turbo' else 2
            
            results = await self.fast_orchestrator.race_strategy(
                self.query_model_fast,
                self.models,
                prompt,
                min_responses=min_responses
            )
            
        elif strategy == 'cascade':
            # Cascade strategy: start with fast model, add more if needed
            results = await self.fast_orchestrator.cascade_strategy(
                self.query_model_fast,
                self.models,
                prompt,
                cascade_delay=2.0
            )
            
        elif strategy == 'single':
            # Single best model
            results = await self.fast_orchestrator.single_best_strategy(
                self.query_model_fast,
                self.models[0],
                prompt
            )
            
        else:
            # Parallel strategy (default)
            tasks = []
            for model in self.models:
                tasks.append(self.query_model_fast(model, prompt))
            
            results = await asyncio.gather(*tasks)
        
        return results

    async def enhance_prompt_with_web_search(self, prompt: str) -> Tuple[str, bool]:
        """Enhance prompt with web search results if needed"""
        
        if not self.enable_web_search or not self.web_searcher:
            return prompt, False
        
        # Check if web search is needed
        needs_search = False
        
        # Keywords that indicate need for current information
        current_keywords = ['current', 'latest', 'today', 'now', 'recent', '2024', '2025', 
                          'news', 'update', 'happening', 'price', 'weather', 'stock', 
                          'yesterday', 'this week', 'this month']
        
        prompt_lower = prompt.lower()
        for keyword in current_keywords:
            if keyword in prompt_lower:
                needs_search = True
                break
        
        # Also check for questions about specific current events
        if '?' in prompt and any(word in prompt_lower for word in ['who is', 'what is', 'when is', 'where is']):
            if any(word in prompt_lower for word in ['president', 'ceo', 'champion', 'winner']):
                needs_search = True
        
        if not needs_search:
            return prompt, False
        
        # Perform web search
        self.logger.info("Enhancing prompt with web search results...")
        
        try:
            # Use max_results from config
            web_context = await self.web_searcher.search_with_fallback(prompt)
            
            if web_context and web_context != "No web search results found.":
                enhanced_prompt = f"""Context from web search:
                    {web_context}

                    Based on the above context and your knowledge, please answer the following question.
                    If the web search results don't contain relevant information, use your general knowledge.

                    Question: {prompt}"""
                return enhanced_prompt, True
                
        except Exception as e:
            if FEATURES['verbose_errors']:
                self.logger.error(f"Web search enhancement failed: {str(e)}")
        
        return prompt, False
    
    async def query_model_optimized(self, session, model: str, prompt: str, 
                                   stagger_delay: float = 0) -> Dict:
        """Query a model with optimizations using config settings"""

        start_time = datetime.now()
        
        if self.verbose_logging and hasattr(self, 'live_logger'):
            self.live_logger.log_live(f"Querying {model}...", "START")
        

        # Get optimal parameters from learning system
        if self.smart_learning and self.orchestrator:
            optimal_params = self.orchestrator.get_optimal_params_for_model(model)
            
            # Check for precomputed response
            if self.precompute_manager:
                precomputed = self.precompute_manager.get_precomputed_response(prompt, model)
                if precomputed:
                    self.logger.info(f"Using precomputed response for {model}")
                    return {
                        "model": model,
                        "response": precomputed,
                        "success": True,
                        "response_time": 0.01,  # Near instant
                        "precomputed": True
                    }
        else:
            optimal_params = GENERATION_OPTIONS.copy()
        
        # Add stagger delay to prevent resource competition
        if FEATURES['staggered_starts'] and stagger_delay > 0:
            await asyncio.sleep(stagger_delay)
        
        # Get optimized timeout for this model
        if self.model_manager:
            timeout = self.model_manager.get_optimized_timeout(model)
        else:
            # Use timeout from model config
            model_config = MODEL_CONFIGS.get(model, {})
            timeout = model_config.get('timeout', SYSTEM_CONFIGS['24GB']['timeout_base'])
        
        self.logger.debug(f"Querying {model} with {timeout}s timeout")
        
        # Enhanced prompt if web search is enabled
        enhanced_prompt = prompt
        used_web_search = False
        
        # Only enhance once per query session to save time
        if self.enable_web_search and not hasattr(self, '_current_web_context'):
            enhanced_prompt, used_web_search = await self.enhance_prompt_with_web_search(prompt)
            if used_web_search:
                self._current_web_context = enhanced_prompt
        elif hasattr(self, '_current_web_context'):
            enhanced_prompt = self._current_web_context
            used_web_search = True
        
        # Use generation options from config
        payload = {
            "model": model,
            "prompt": enhanced_prompt,
            "stream": False,
            "options": optimal_params
        }
        
        # Retry logic
        max_retries = ENSEMBLE_CONFIG['max_retries'] if FEATURES['auto_retry'] else 1
        
        for attempt in range(max_retries):
            try:
                async with session.post(
                    f"{self.host}{OLLAMA_ENDPOINTS['generate']}", 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        response_time = (datetime.now() - start_time).total_seconds()
                        response_text = result.get("response", "")
                        
                        # Check if response indicates uncertainty and we haven't searched yet
                        if (self.enable_web_search and not used_web_search and 
                            await self.detect_uncertainty(response_text)):
                            
                            self.logger.info(f"Model {model} seems uncertain, adding web search context...")
                            enhanced_prompt, used_web_search = await self.enhance_prompt_with_web_search(prompt)
                            
                            if used_web_search:
                                # Retry with enhanced prompt
                                payload["prompt"] = enhanced_prompt
                                continue
                        
                        response_dict = {
                            "model": model,
                            "response": response_text,
                            "success": True,
                            "used_web_search": used_web_search,
                            "response_time": response_time,
                            "metadata": {
                                "total_duration": result.get("total_duration", 0) / 1e9,
                                "eval_count": result.get("eval_count", 0)
                            }
                        }
                        
                        if self.verbose_logging and hasattr(self, 'verbose_logger'):
                            self.verbose_logger.log_model_response(
                                model=model,
                                response=response_dict,
                                query_time=(datetime.now() - start_time).total_seconds()
                            )
                            
                            # Log to performance logger
                            if hasattr(self, 'performance_logger'):
                                self.performance_logger.log_model_query(
                                    model=model,
                                    success=response_dict.get('success', False),
                                    response_time=response_dict.get('response_time', 0),
                                    query_type=None,  # Will be set later
                                    was_selected=False,  # Will be updated after voting
                                    quality_score=0,  # Will be updated after voting
                                    consensus_score=0  # Will be updated after voting
                                )
                        
                        return response_dict

                    else:
                        error_text = await response.text()
                        response_time = (datetime.now() - start_time).total_seconds()
                        
                        if attempt < max_retries - 1 and FEATURES['auto_retry']:
                            await asyncio.sleep(ENSEMBLE_CONFIG['retry_delay'] ** attempt)
                            continue
                        
                        # Try to parse error message
                        try:
                            error_data = json.loads(error_text)
                            error_msg = error_data.get('error', error_text)
                        except:
                            error_msg = error_text[:200]
                        
                        return {
                            "model": model,
                            "response": f"HTTP {response.status}: {error_msg}",
                            "success": False,
                            "response_time": response_time
                        }
                        
            except asyncio.TimeoutError:
                response_time = (datetime.now() - start_time).total_seconds()
                self.logger.error(ERROR_MESSAGES['timeout'].format(timeout=timeout))
                
                if attempt < max_retries - 1 and FEATURES['auto_retry']:
                    await asyncio.sleep(ENSEMBLE_CONFIG['retry_delay'] ** attempt)
                    continue
                
                return {
                    "model": model,
                    "response": ERROR_MESSAGES['timeout'].format(timeout=timeout),
                    "success": False,
                    "response_time": response_time
                }
                
            except aiohttp.ClientError as e:
                response_time = (datetime.now() - start_time).total_seconds()
                
                if FEATURES['verbose_errors']:
                    self.logger.error(f"Client error for model {model}: {type(e).__name__}: {str(e)}")
                
                if attempt < max_retries - 1 and FEATURES['auto_retry']:
                    await asyncio.sleep(ENSEMBLE_CONFIG['retry_delay'] ** attempt)
                    continue
                
                return {
                    "model": model,
                    "response": ERROR_MESSAGES['connection'].format(host=self.host),
                    "success": False,
                    "response_time": response_time
                }
                
            except Exception as e:
                response_time = (datetime.now() - start_time).total_seconds()
                
                if FEATURES['verbose_errors']:
                    self.logger.error(f"Unexpected error for model {model}: {type(e).__name__}: {str(e)}")
                
                return {
                    "model": model,
                    "response": f"Error: {type(e).__name__}",
                    "success": False,
                    "response_time": response_time
                }
        
        # Should not reach here, but just in case
        return {
            "model": model,
            "response": "All retry attempts failed",
            "success": False,
            "response_time": (datetime.now() - start_time).total_seconds()
        }
    
    async def query_all_models_optimized(self, prompt: str) -> List[Dict]:
        """Query all models with optimization using config settings"""
        
        # Check model availability
        availability = await self.check_model_availability()
        available_models = [m for m in self.models if availability.get(m, True)]
        
        if not available_models:
            self.logger.error(ERROR_MESSAGES['no_models'])
            return [{
                "model": "system",
                "response": ERROR_MESSAGES['no_models'],
                "success": False,
                "response_time": 0
            }]
        
        self.logger.info(f"Querying {len(available_models)} available models")
        
        # Clear web context for new query
        if hasattr(self, '_current_web_context'):
            delattr(self, '_current_web_context')
        
        async with aiohttp.ClientSession() as session:
            # Create tasks with staggered delays
            tasks = []
            
            # Get stagger delay from config based on system
            if self.model_manager:
                stagger_delay = self.model_manager.system_config.get('stagger_delay', 
                    ENSEMBLE_CONFIG['stagger_delays']['24GB'])
            else:
                stagger_delay = ENSEMBLE_CONFIG['stagger_delays'].get('24GB', 0.3)
            
            for i, model in enumerate(available_models):
                task = self.query_model_optimized(
                    session, 
                    model, 
                    prompt,
                    stagger_delay * i if FEATURES['staggered_starts'] else 0
                )
                tasks.append(task)
            
            # Execute all queries
            responses = await asyncio.gather(*tasks)
            
            # Add unavailable models to responses
            for model in self.models:
                if model not in available_models:
                    responses.append({
                        "model": model,
                        "response": "Model not available",
                        "success": False,
                        "response_time": 0
                    })
            
            # Track performance if enabled
            if self.performance_tracker and FEATURES['performance_tracking']:
                for response in responses:
                    if response['success'] or response.get('response_time', 0) > 0:
                        self.performance_tracker.record_query_result(
                            model=response['model'],
                            success=response['success'],
                            response_time=response.get('response_time', 0),
                            was_selected=False,  # Will update after voting
                            consensus_score=0,
                            quality_score=0
                        )
            
            return responses
    
    def calculate_similarity_matrix(self, responses: List[str]) -> np.ndarray:
        """Calculate pairwise similarity between responses"""
        
        if len(responses) < 2:
            return np.array([[1.0]])
        
        try:
            # Use TF-IDF for better similarity measurement
            tfidf_matrix = self.vectorizer.fit_transform(responses)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix
            
        except Exception as e:
            if FEATURES['verbose_errors']:
                self.logger.error(f"Similarity calculation failed: {str(e)}")
            # Fallback to simple length-based similarity
            return np.ones((len(responses), len(responses)))
    
    def weighted_voting(self, responses: List[Dict]) -> Tuple[str, Dict]:
        """
        Implement weighted voting based on config weights
        """
        
        valid_responses = [r for r in responses if r["success"] and len(r.get("response", "")) > 10]
        
        if not valid_responses:
            # If no valid responses, try to use partial responses
            partial_responses = [r for r in responses if r.get("response", "")]
            if partial_responses:
                return partial_responses[0]["response"], {
                    "selected_model": partial_responses[0]["model"],
                    "consensus_score": 0,
                    "quality_score": 0,
                    "all_failed": True
                }
            
            return ERROR_MESSAGES['all_failed'], {
                "error": "No valid responses",
                "all_failed": True
            }
        
        if len(valid_responses) == 1:
            return valid_responses[0]["response"], {
                "selected_model": valid_responses[0]["model"],
                "consensus_score": 1.0,
                "quality_score": 1.0,
                "used_web_search": valid_responses[0].get("used_web_search", False)
            }
        
        response_texts = [r["response"] for r in valid_responses]
        
        # Calculate similarity matrix
        similarity_matrix = self.calculate_similarity_matrix(response_texts)
        
        # Calculate consensus scores (how much each response agrees with others)
        consensus_scores = np.mean(similarity_matrix, axis=1)
        
        # Calculate quality scores
        quality_scores = []
        for i, response in enumerate(valid_responses):
            text = response["response"]
            
            # Length score (prefer moderate length)
            length = len(text)
            if length < 50:
                length_score = 0.3
            elif length < 100:
                length_score = 0.6
            elif length <= 400:
                length_score = 1.0
            elif length <= 600:
                length_score = 0.9
            else:
                length_score = 0.7
            
            # Structure score (paragraphs, sentences)
            structure_score = 1.0
            if '\n' in text:
                structure_score *= 1.1
            if text.count('.') > 2:
                structure_score *= 1.1
            structure_score = min(structure_score, 1.2)
            
            # Web search bonus from config
            web_bonus = ENSEMBLE_CONFIG['web_bonus'] if response.get("used_web_search", False) else 1.0
            
            # Response time penalty
            time_penalty = 1.0
            response_time = response.get("response_time", 0)
            max_time = PERFORMANCE_THRESHOLDS['max_avg_response_time']
            
            if response_time > 0:
                if response_time < max_time / 6:
                    time_penalty = 1.1
                elif response_time < max_time / 3:
                    time_penalty = 1.0
                elif response_time < max_time / 1.5:
                    time_penalty = 0.95
                else:
                    time_penalty = 0.9
            
            # Historical performance bonus (if available)
            historical_bonus = 1.0
            if self.performance_tracker and FEATURES['performance_tracking']:
                model_eval = self.performance_tracker.evaluate_model_performance(response["model"])
                if model_eval['status'] == 'healthy':
                    historical_bonus = 1.1
                elif model_eval['status'] in ['slow', 'underutilized']:
                    historical_bonus = 0.95
            
            quality_score = (length_score * structure_score * web_bonus * 
                           time_penalty * historical_bonus)
            quality_scores.append(quality_score)
        
        quality_scores = np.array(quality_scores)
        
        # Normalize scores
        if quality_scores.max() > 0:
            quality_scores = quality_scores / quality_scores.max()
        
        # Combined score using weights from config
        consensus_weight = ENSEMBLE_CONFIG['consensus_weight']
        quality_weight = ENSEMBLE_CONFIG['quality_weight']
        final_scores = (consensus_scores * consensus_weight + quality_scores * quality_weight)
        
        # Select best response
        best_idx = np.argmax(final_scores)
        best_response = valid_responses[best_idx]
        
        # Create detailed metadata
        metadata = {
            "selected_model": best_response["model"],
            "consensus_score": float(consensus_scores[best_idx]),
            "quality_score": float(quality_scores[best_idx]),
            "final_score": float(final_scores[best_idx]),
            "used_web_search": best_response.get("used_web_search", False),
            "response_time": best_response.get("response_time", 0),
            "all_scores": {
                valid_responses[i]["model"]: {
                    "consensus": float(consensus_scores[i]),
                    "quality": float(quality_scores[i]),
                    "final": float(final_scores[i]),
                    "response_time": valid_responses[i].get("response_time", 0),
                    "used_web": valid_responses[i].get("used_web_search", False)
                }
                for i in range(len(valid_responses))
            },
            "total_models": len(responses),
            "successful_models": len(valid_responses)
        }
        
        return best_response["response"], metadata
    
    async def ensemble_query(self, prompt: str, verbose: bool = False) -> Tuple[str, Dict]:
        """Main ensemble query method with optimization"""
        
        start_time = datetime.now()

        if self.verbose_logging and hasattr(self, 'verbose_logger'):
            self.verbose_logger.log_query_start(
                query_id=self.query_count + 1,
                prompt=prompt,
                models=self.models,
                speed_mode=self.speed_mode,
                web_search=self.enable_web_search
            )
        
        # Check cache first if smart learning is enabled
        if self.smart_learning and self.orchestrator:
            cached_result = await self.orchestrator.check_cache(prompt)
            
            if cached_result:
                response, metadata = cached_result
                
                # Double-check the cached response is valid
                if response and not metadata.get('all_failed'):
                    metadata['from_cache'] = True
                    metadata['total_ensemble_time'] = 0.01
                    
                    if verbose:
                        cache_similarity = metadata.get('cache_similarity', 1.0)
                        print(f"\n{'🎯' if DISPLAY_CONFIG['use_emojis'] else ''} "
                            f"Using cached result (similarity: {cache_similarity:.2f})")
                    
                    return response, metadata
                else:
                    # Invalid cache, proceed with normal query
                    self.logger.warning("Invalid cached response, proceeding with fresh query")
        
        
        # Detect query type if specialized selection is enabled
        query_type = None
        if FEATURES['specialized_selection']:
            query_type = await self.detect_query_type(prompt)
            if query_type:
                self.logger.info(f"Detected query type: {query_type}")
        
        # Periodic model optimization
        self.query_count += 1
        if (self.adaptive_mode and FEATURES['model_rotation'] and 
            self.query_count % self.optimization_interval == 0):
            
            self.logger.info("Running periodic model optimization...")
            self.models = await self.model_manager.optimize_models(self.models)
            # Clear availability cache after optimization
            if FEATURES['caching']:
                self.model_availability_cache = {}
        
        self.logger.info(f"Starting query in {self.speed_mode} mode with {len(self.models)} models")
        
        if self.speed_mode in ['turbo', 'fast']:
            responses = await self.query_all_models_fast(prompt)
        else:
            responses = await self.query_all_models_optimized(prompt)
        
        # For turbo mode, skip complex voting if we only have 1-2 responses
        if self.speed_mode == 'turbo' and len(responses) <= 2:
            # Just return the first successful response
            for response in responses:
                if response['success']:
                    metadata = {
                        "selected_model": response['model'],
                        "total_ensemble_time": (datetime.now() - start_time).total_seconds(),
                        "speed_mode": self.speed_mode,
                        "strategy": self.speed_profile['strategy']
                    }
                    
                    self.logger.info(f"Turbo mode completed in {metadata['total_ensemble_time']:.2f}s")
                    
                    return response['response'], metadata

        # Voting and selection
        best_response, metadata = self.weighted_voting(responses)
        
        # Update performance tracker with final selection
        if self.performance_tracker and FEATURES['performance_tracking'] and not metadata.get('all_failed', False):
            selected_model = metadata.get('selected_model')
            
            for response in responses:
                if response['success']:
                    is_selected = (response['model'] == selected_model)
                    
                    # Get scores from metadata
                    model_scores = metadata.get('all_scores', {}).get(response['model'], {})
                    
                    self.performance_tracker.record_query_result(
                        model=response['model'],
                        success=response['success'],
                        response_time=response.get('response_time', 0),
                        was_selected=is_selected,
                        consensus_score=model_scores.get('consensus', 0),
                        quality_score=model_scores.get('quality', 0)
                    )
        
        # Save performance data periodically (from config)
        if (self.performance_tracker and 
            self.query_count % TRACKING_CONFIG['save_interval'] == 0):
            self.performance_tracker.save_performance_data()
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        metadata["total_ensemble_time"] = total_time
        metadata["query_type"] = query_type
        
        if verbose:
            self.display_verbose_output(responses, metadata, total_time)
        
        self.logger.info(f"Query completed in {total_time:.2f}s - Selected: {metadata.get('selected_model', 'N/A')}")
        
        # Show performance summary periodically
        if (self.performance_tracker and FEATURES['performance_tracking'] and 
            self.query_count % 20 == 0):
            print("\n" + "="*60)
            print(self.performance_tracker.get_performance_summary())
            print("="*60 + "\n")

        # After getting the best response, update learning
        if self.smart_learning and self.orchestrator:
            # Prepare model performance data
            model_performances = []
            for response in responses:
                if response['success']:
                    model_scores = metadata.get('all_scores', {}).get(response['model'], {})
                    model_performances.append({
                        'model': response['model'],
                        'success': response['success'],
                        'response_time': response.get('response_time', 0),
                        'params': optimal_params if 'optimal_params' in locals() else {},
                        'quality_score': model_scores.get('final', 0.5)
                    })
            
            # Update all learning components
            self.orchestrator.update_learning(
                query=prompt,
                response=best_response,
                metadata=metadata,
                model_performances=model_performances
            )
        
        # After voting, log voting details
        if self.verbose_logging and hasattr(self, 'verbose_logger'):
            self.verbose_logger.log_voting_details(metadata)
            
            # Update performance logger with final selection
            if hasattr(self, 'performance_logger'):
                for response in responses:
                    if response['success']:
                        model_scores = metadata.get('all_scores', {}).get(response['model'], {})
                        
                        self.performance_logger.log_model_query(
                            model=response['model'],
                            success=response['success'],
                            response_time=response.get('response_time', 0),
                            query_type=metadata.get('query_type'),
                            was_selected=(response['model'] == metadata.get('selected_model')),
                            quality_score=model_scores.get('quality', 0),
                            consensus_score=model_scores.get('consensus', 0)
                        )
        
        # Log to live logger
        if self.verbose_logging and hasattr(self, 'live_logger'):
            self.live_logger.log_live(
                f"Query #{self.query_count} completed in {metadata.get('total_ensemble_time', 0):.2f}s - "
                f"Selected: {metadata.get('selected_model', 'N/A')}",
                "SUCCESS"
            )

        return best_response, metadata
    
    async def show_smart_insights(self):
        """Display smart learning insights"""
        
        if not self.smart_learning or not self.orchestrator:
            print("Smart learning is not enabled")
            return
        
        insights = self.orchestrator.get_performance_insights()
        
        use_emojis = DISPLAY_CONFIG['use_emojis']
        
        print(f"\n{'='*60}")
        print(f"{'🧠' if use_emojis else ''} Smart Learning Insights")
        print(f"{'='*60}")
        print(f"Total Queries Processed: {insights['total_queries']}")
        print(f"Cache Hit Rate: {insights['cache_hit_rate']:.1f}%")
        print(f"Average Response Time: {insights['avg_response_time']:.2f}s")
        
        print(f"\n{'📊' if use_emojis else ''} Model Confidence Scores:")
        sorted_confidences = sorted(insights['model_confidences'].items(), 
                                  key=lambda x: x[1], reverse=True)
        for model, confidence in sorted_confidences[:5]:
            bar = '█' * int(confidence * 20)
            print(f"  {model:30} {bar} {confidence:.2f}")
        
        if insights['top_models_by_type']:
            print(f"\n{'🎯' if use_emojis else ''} Best Models by Query Type:")
            for query_type, models in insights['top_models_by_type'].items():
                print(f"  {query_type:15} -> {', '.join(models[:3])}")
        
        # Show cache statistics
        cache_dir = Path(TRACKING_CONFIG.get('smart_data_dir', 'smart_data')) / 'cache'
        if cache_dir.exists():
            cache_size = sum(f.stat().st_size for f in cache_dir.glob('*.pkl'))
            cache_count = len(list(cache_dir.glob('*.pkl')))
            print(f"\n{'💾' if use_emojis else ''} Cache Statistics:")
            print(f"  Cached Queries: {cache_count}")
            print(f"  Cache Size: {cache_size / (1024*1024):.2f} MB")

    def display_verbose_output(self, responses: List[Dict], metadata: Dict, total_time: float):
        """Display detailed output in verbose mode using config settings"""
        
        use_emojis = DISPLAY_CONFIG['use_emojis']
        max_preview = DISPLAY_CONFIG['max_preview_length']
        show_timestamps = DISPLAY_CONFIG['show_timestamps']
        
        print("\n" + "="*60)
        print(f"{'📊 ' if use_emojis else ''}Individual Model Responses:")
        print("="*60)
        
        for r in responses:
            # Status indicators
            if use_emojis:
                status = "✅" if r["success"] else "❌"
                web = "🌐" if r.get("used_web_search", False) else ""
                
                rt = r.get("response_time", 0)
                if rt == 0:
                    time_indicator = "⏸️"
                elif rt < 5:
                    time_indicator = "🚀"
                elif rt < 15:
                    time_indicator = "⏱️"
                else:
                    time_indicator = "🐢"
            else:
                status = "[OK]" if r["success"] else "[FAIL]"
                web = "[WEB]" if r.get("used_web_search", False) else ""
                time_indicator = ""
            
            rt_str = f"({r.get('response_time', 0):.1f}s)" if show_timestamps else ""
            
            print(f"\n{status} [{r['model']}] {web} {time_indicator} {rt_str}")
            
            if r["success"]:
                # Show preview of response
                response_text = r["response"]
                if len(response_text) > max_preview:
                    preview = response_text[:max_preview] + "..."
                else:
                    preview = response_text
                
                # Indent the preview
                preview_lines = preview.split('\n')
                for line in preview_lines:
                    print(f"   {line}")
            else:
                print(f"   Error: {r['response']}")
        
        print("\n" + "="*60)
        print(f"{'🏆 ' if use_emojis else ''}Voting Results:")
        print("="*60)
        print(f"Selected Model: {metadata.get('selected_model', 'N/A')}")
        print(f"Consensus Score: {metadata.get('consensus_score', 0):.3f}")
        print(f"Quality Score: {metadata.get('quality_score', 0):.3f}")
        print(f"Final Score: {metadata.get('final_score', 0):.3f}")
        
        if metadata.get('used_web_search'):
            print(f"Web Search: Yes {'🌐' if use_emojis else ''}")
        
        if metadata.get('query_type'):
            print(f"Query Type: {metadata['query_type']}")
        
        if show_timestamps:
            print(f"Total Time: {total_time:.2f}s")
        
        print(f"Successful Models: {metadata.get('successful_models', 0)}/{metadata.get('total_models', 0)}")
        
        if 'all_scores' in metadata and len(metadata['all_scores']) > 1:
            print(f"\n{'📈 ' if use_emojis else ''}All Model Scores:")
            sorted_models = sorted(metadata['all_scores'].items(), 
                                 key=lambda x: x[1]['final'], reverse=True)
            
            for model, scores in sorted_models:
                if use_emojis:
                    web_indicator = "🌐" if scores.get('used_web', False) else "  "
                else:
                    web_indicator = "[W]" if scores.get('used_web', False) else "   "
                    
                time_str = f"{scores.get('response_time', 0):.1f}s" if show_timestamps else ""
                
                print(f"   {web_indicator} {model:30} | Final: {scores['final']:.3f} | "
                      f"Consensus: {scores['consensus']:.2f} | "
                      f"Quality: {scores['quality']:.2f}" + 
                      (f" | Time: {time_str:6}" if show_timestamps else ""))
        
        if self.performance_tracker and FEATURES['performance_tracking']:
            print(f"\n{'📊 ' if use_emojis else ''}Model Health Status:")
            for model in self.models[:5]:  # Show top 5 models
                eval_result = self.performance_tracker.evaluate_model_performance(model)
                
                if use_emojis:
                    status_emoji = {
                        'healthy': '✅',
                        'unhealthy': '⚠️',
                        'underutilized': '📉',
                        'slow': '🐢',
                        'failing': '❌',
                        'new': '🆕',
                        'insufficient_data': '📊'
                    }.get(eval_result['status'], '❓')
                else:
                    status_emoji = f"[{eval_result['status'][:4].upper()}]"
                
                print(f"   {status_emoji} {model:30} | Status: {eval_result['status']:15} | "
                      f"Score: {eval_result.get('score', 0):.2f}")
    
    async def cleanup(self):
        """Cleanup resources"""

        # Generate daily summary if verbose logging
        if self.verbose_logging and hasattr(self, 'performance_logger'):
            summary = self.performance_logger.generate_daily_summary()
            
            # Log session end
            if hasattr(self, 'verbose_logger'):
                stats = {
                    'total_queries': self.query_count,
                    'cache_hits': self.orchestrator.session_data['cache_hits'] if self.orchestrator else 0,
                    'avg_response_time': self.orchestrator.session_data['avg_response_time'] if self.orchestrator else 0,
                    'rotations': 0  # You can track this separately
                }
                
                self.verbose_logger.log_session_end(stats)

        if self.web_searcher:
            await self.web_searcher.close()
        
        if self.performance_tracker and FEATURES['performance_tracking']:
            self.performance_tracker.save_performance_data()
            self.logger.info("Performance data saved")

# Main function
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Optimized Ensemble LLM with adaptive model management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            ensemble-llm "What is quantum computing?"           # Simple query
            ensemble-llm -w "Latest news about AI?"             # With web search
            ensemble-llm -i                                     # Interactive mode
            ensemble-llm --performance-report                   # Show performance stats
            ensemble-llm -v "Complex question"                  # Verbose output
        """
    )
    
    parser.add_argument('prompt', nargs='?', help='Question to ask the ensemble')
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                       help='Models to use in ensemble')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show individual model responses and scores')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode')
    parser.add_argument('--web-search', '-w', action='store_true',
                       help='Enable web search for current information')
    parser.add_argument('--no-adaptive', action='store_true',
                       help='Disable adaptive model optimization')
    parser.add_argument('--performance-report', '-p', action='store_true',
                       help='Show performance report and exit')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default=LOGGING_CONFIG['default_level'], 
                       help='Set logging level')
    parser.add_argument('--no-emojis', action='store_true',
                       help='Disable emoji output')
    parser.add_argument('--host', default=None,
                       help=f'Ollama host (default: {OLLAMA_ENDPOINTS["default_host"]})')
    parser.add_argument('--no-smart', action='store_true',
                   help='Disable smart learning system')
    parser.add_argument('--clear-cache', action='store_true',
                    help='Clear smart cache and start fresh')
    parser.add_argument('--insights', action='store_true',
                    help='Show smart learning insights')
    parser.add_argument('--speed', choices=['turbo', 'fast', 'balanced', 'quality'],
                       default='balanced',
                       help='Speed optimization mode')
    parser.add_argument('--warmup', action='store_true',
                       help='Warmup models before querying')
    parser.add_argument('--clear-failed-cache', action='store_true',
                    help='Clear only failed cached responses')
    parser.add_argument('--cache-stats', action='store_true',
                    help='Show cache statistics')
    parser.add_argument('--tail-log', action='store_true',
                   help='Tail the live log file')
    parser.add_argument('--view-log', action='store_true',
                    help='View today\'s verbose log')
    parser.add_argument('--log-stats', action='store_true',
                    help='Show log statistics')
    
    args = parser.parse_args()
    
    # Override display config if requested
    if args.no_emojis:
        DISPLAY_CONFIG['use_emojis'] = False
    
    # Setup logging
    logger = setup_logger('EnsembleLLM', args.log_level)
    
    # Create data and log directories if they don't exist
    Path(TRACKING_CONFIG['data_dir']).mkdir(exist_ok=True)
    Path(LOGGING_CONFIG['log_dir']).mkdir(exist_ok=True)
    
    # Show performance report if requested
    if args.performance_report:
        tracker = ModelPerformanceTracker(data_dir=TRACKING_CONFIG['data_dir'])
        print("\n" + tracker.get_performance_summary())
        print(f"\nPerformance data location: {TRACKING_CONFIG['data_dir']}/{TRACKING_CONFIG['performance_file']}")
        return

    if args.clear_cache:
        import shutil
        smart_dir = Path(TRACKING_CONFIG.get('smart_data_dir', 'smart_data'))
        if smart_dir.exists():
            shutil.rmtree(smart_dir)
            print("✅ Smart cache cleared")

    if args.clear_failed_cache:
        from .learning_system import CacheManager
        CacheManager.clear_failed_cache()

    if args.cache_stats:
        from .learning_system import CacheManager
        CacheManager.show_cache_stats()

    if args.insights:
        ensemble = EnsembleLLM(
            models=args.models,
            smart_learning=True
        )
        await ensemble.show_smart_insights()
        return

    if args.tail_log:
        subprocess.run(['python', 'scripts/view_logs.py', 'tail'])
        return

    if args.view_log:
        subprocess.run(['python', 'scripts/view_logs.py', 'verbose'])
        return

    if args.log_stats:
        subprocess.run(['python', 'scripts/view_logs.py', 'performance'])
        return
    
    # Display startup banner
    use_emojis = DISPLAY_CONFIG['use_emojis']
    print(f"\n{'='*60}")
    print(f"{'🚀 ' if use_emojis else ''}Optimized Ensemble LLM System v2.1")
    print(f"{'='*60}")
    print(f"{'📦 ' if use_emojis else ''}Models: {', '.join(args.models[:3])}{'...' if len(args.models) > 3 else ''}")
    print(f"{'🌐 ' if use_emojis else ''}Web Search: {'Enabled' if args.web_search or FEATURES['web_search'] else 'Disabled'}")
    print(f"{'🧠 ' if use_emojis else ''}Adaptive Mode: {'Enabled' if not args.no_adaptive and FEATURES['adaptive_models'] else 'Disabled'}")
    print(f"{'📝 ' if use_emojis else ''}Log Level: {args.log_level}")
    print(f"{'📄 ' if use_emojis else ''}Log File: {LOGGING_CONFIG['log_dir']}/{LOGGING_CONFIG['log_file']}")
    print(f"{'⚡ ' if use_emojis else ''}Speed Mode: {args.speed}")
    if args.speed == 'turbo':
        print("   Ultra-fast mode: 2 models, 10s timeout, race strategy")
    elif args.speed == 'fast':
        print("   Fast mode: 3 models, 15s timeout, cascade strategy")
    
    print(f"{'='*60}\n")
    
    # Initialize ensemble
    ensemble = EnsembleLLM(
        models=args.models,
        ollama_host=args.host,
        enable_web_search=args.web_search or None,
        adaptive_mode=not args.no_adaptive if not args.no_adaptive else None,
        smart_learning=not args.no_smart and SMART_LEARNING_CONFIG['enabled'],
        speed_mode=args.speed
    )

    if args.warmup:
        print("🔥 Warming up models...")
        await ensemble.model_warmup.parallel_warmup(ensemble.models)
    
    # Initialize (preload models)
    print(f"{'🔄 ' if use_emojis else ''}Initializing models...")
    await ensemble.initialize()
    print(f"{'✅ ' if use_emojis else ''}Ready!\n")
    
    try:
        if args.interactive or not args.prompt:
            print(f"{'💡 ' if use_emojis else ''}Commands:")
            print("   'exit' or 'quit' - Exit the program")
            print("   'status' - Show model performance statistics")
            print("   'models' - List current active models")
            print("   'help' - Show this help message")
            print(f"\n{'💡 ' if use_emojis else ''}Enter your questions below:\n")
            
            while True:
                try:
                    prompt = input(f"\n{'💭 ' if use_emojis else ''}Q: ").strip()
                    
                    if not prompt:
                        continue
                    
                    if prompt.lower() in ['exit', 'quit', 'q']:
                        print(f"\n{'👋 ' if use_emojis else ''}Goodbye!")
                        break
                    
                    if prompt.lower() == 'help':
                        print(f"\n{'💡 ' if use_emojis else ''}Available commands:")
                        print("   'status' - Show model performance")
                        print("   'models' - List active models")
                        print("   'exit' - Quit the program")
                        continue
                    
                    if prompt.lower() == 'status':
                        if ensemble.performance_tracker:
                            print("\n" + ensemble.performance_tracker.get_performance_summary())
                        else:
                            print(f"\n{'⚠️ ' if use_emojis else ''}Performance tracking is disabled")
                        continue
                    
                    if prompt.lower() == 'models':
                        print(f"\n{'📦 ' if use_emojis else ''}Active models ({len(ensemble.models)}):")
                        for i, model in enumerate(ensemble.models, 1):
                            model_config = MODEL_CONFIGS.get(model, {})
                            desc = model_config.get('description', '')
                            print(f"   {i}. {model} - {desc}")
                        continue
                    
                    # Process the query
                    print("")  # Empty line for better readability
                    response, metadata = await ensemble.ensemble_query(prompt, args.verbose)
                    
                    # Display response
                    selected = metadata.get('selected_model', 'ensemble')
                    web_indicator = f"{'🌐 ' if use_emojis else '[WEB] '}" if metadata.get('used_web_search', False) else ""
                    
                    print(f"\n{web_indicator}{'💡 ' if use_emojis else ''}Answer (via {selected}):")
                    print("-" * 40)
                    print(response)
                    print("-" * 40)
                    
                    if DISPLAY_CONFIG['show_timestamps']:
                        print(f"{'⏱️ ' if use_emojis else ''}Response time: {metadata.get('total_ensemble_time', 0):.1f}s")
                    
                except KeyboardInterrupt:
                    print(f"\n\n{'⚠️ ' if use_emojis else ''}Interrupted! Press Ctrl+C again to exit or Enter to continue...")
                    try:
                        input()
                    except KeyboardInterrupt:
                        print(f"\n{'👋 ' if use_emojis else ''}Goodbye!")
                        break
                except Exception as e:
                    if FEATURES['verbose_errors']:
                        logger.error(f"Error in interactive mode: {str(e)}")
                        logger.debug(traceback.format_exc())
                    print(f"\n{'❌ ' if use_emojis else ''}Error: {str(e)}")
                    print("Please try again or type 'exit' to quit.")
                    
        else:
            # Single query mode
            response, metadata = await ensemble.ensemble_query(args.prompt, args.verbose)
            
            # Display response
            selected = metadata.get('selected_model', 'ensemble')
            web_indicator = f"{'🌐 ' if use_emojis else '[WEB] '}" if metadata.get('used_web_search', False) else ""
            
            print(f"\n{web_indicator}{'💡 ' if use_emojis else ''}Answer (via {selected}):")
            print("="*60)
            print(response)
            print("="*60)
            
            if DISPLAY_CONFIG['show_timestamps']:
                print(f"\n{'⏱️ ' if use_emojis else ''}Total time: {metadata.get('total_ensemble_time', 0):.1f}s")
            
            if not args.verbose and DISPLAY_CONFIG['progress_indicators']:
                print(f"{'📊 ' if use_emojis else ''}Score: {metadata.get('final_score', 0):.3f}")
                print(f"{'✅ ' if use_emojis else ''}Successful models: {metadata.get('successful_models', 0)}/{metadata.get('total_models', 0)}")
            
    except Exception as e:
        if FEATURES['verbose_errors']:
            logger.error(f"Fatal error: {str(e)}")
            logger.debug(traceback.format_exc())
        print(f"\n{'❌ ' if use_emojis else ''}Fatal error: {str(e)}")
        print(f"Check {LOGGING_CONFIG['log_dir']}/{LOGGING_CONFIG['log_file']} for details")
        
    finally:
        print(f"\n{'🔄 ' if use_emojis else ''}Shutting down...")
        await ensemble.cleanup()
        
        if ensemble.performance_tracker and FEATURES['performance_tracking']:
            print(f"{'💾 ' if use_emojis else ''}Performance data saved to {TRACKING_CONFIG['data_dir']}/{TRACKING_CONFIG['performance_file']}")
        
        print(f"{'✅ ' if use_emojis else ''}Shutdown complete")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())