"""Advanced learning system for continuous optimization"""

import json
import hashlib
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import asyncio
import aiohttp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import logging
from .config import ERROR_MESSAGES

class QueryCache:
    """Intelligent query caching with similarity matching"""
    
    def __init__(self, cache_dir: str = "cache", max_cache_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.logger = logging.getLogger('EnsembleLLM.QueryCache')
        
        # Load cache index
        self.cache_index = self.load_cache_index()
        
        # Build vectorizer from existing queries
        if self.cache_index['queries']:
            self.vectorizer.fit(self.cache_index['queries'])
    
    def load_cache_index(self) -> Dict:
        """Load the cache index"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'queries': [],
            'hashes': [],
            'timestamps': [],
            'hit_counts': [],
            'metadata': {}
        }
    
    def save_cache_index(self):
        """Save the cache index"""
        with open(self.cache_index_file, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def get_query_hash(self, query: str) -> str:
        """Generate a hash for a query"""
        return hashlib.sha256(query.encode()).hexdigest()[:16]
    
    def find_similar_query(self, query: str, threshold: float = 0.85) -> Optional[Tuple[str, float]]:
        """Find a similar cached query using cosine similarity"""
        
        if not self.cache_index['queries']:
            return None
        
        try:
            # Vectorize the new query
            query_vector = self.vectorizer.transform([query])
            
            # Vectorize all cached queries
            cached_vectors = self.vectorizer.transform(self.cache_index['queries'])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, cached_vectors)[0]
            
            # Find the most similar
            max_similarity_idx = np.argmax(similarities)
            max_similarity = similarities[max_similarity_idx]
            
            if max_similarity >= threshold:
                similar_hash = self.cache_index['hashes'][max_similarity_idx]
                
                # Update hit count
                self.cache_index['hit_counts'][max_similarity_idx] += 1
                
                self.logger.info(f"Found similar query with {max_similarity:.2f} similarity")
                return similar_hash, max_similarity
                
        except Exception as e:
            self.logger.error(f"Error finding similar query: {str(e)}")
        
        return None
    
    def get_cached_response(self, query_hash: str) -> Optional[Dict]:
        """Get a cached response"""
        cache_file = self.cache_dir / f"{query_hash}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.error(f"Error loading cache: {str(e)}")
        
        return None
    
    def cache_response(self, query: str, response: str, metadata: Dict):
        """Cache a response"""
        query_hash = self.get_query_hash(query)
        
        # Save response
        cache_file = self.cache_dir / f"{query_hash}.pkl"
        cache_data = {
            'query': query,
            'response': response,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Update index
            if query not in self.cache_index['queries']:
                self.cache_index['queries'].append(query)
                self.cache_index['hashes'].append(query_hash)
                self.cache_index['timestamps'].append(datetime.now().isoformat())
                self.cache_index['hit_counts'].append(0)
                
                # Refit vectorizer
                if len(self.cache_index['queries']) > 1:
                    self.vectorizer.fit(self.cache_index['queries'])
                
                # Maintain cache size
                if len(self.cache_index['queries']) > self.max_cache_size:
                    self.evict_oldest_cache()
                
                self.save_cache_index()
                
        except Exception as e:
            self.logger.error(f"Error caching response: {str(e)}")
    
    def evict_oldest_cache(self):
        """Evict the oldest cache entry with lowest hit count"""
        # Find entry with lowest hit count among oldest 20%
        cutoff = int(len(self.cache_index['queries']) * 0.2)
        oldest_indices = list(range(cutoff))
        
        # Find the one with lowest hit count
        min_hits = float('inf')
        min_idx = 0
        
        for idx in oldest_indices:
            if self.cache_index['hit_counts'][idx] < min_hits:
                min_hits = self.cache_index['hit_counts'][idx]
                min_idx = idx
        
        # Remove from index
        query_hash = self.cache_index['hashes'][min_idx]
        self.cache_index['queries'].pop(min_idx)
        self.cache_index['hashes'].pop(min_idx)
        self.cache_index['timestamps'].pop(min_idx)
        self.cache_index['hit_counts'].pop(min_idx)
        
        # Remove cache file
        cache_file = self.cache_dir / f"{query_hash}.pkl"
        if cache_file.exists():
            cache_file.unlink()

class ModelOptimizer:
    """Dynamic model parameter optimizer"""
    
    def __init__(self, optimizer_dir: str = "optimizers"):
        self.optimizer_dir = Path(optimizer_dir)
        self.optimizer_dir.mkdir(exist_ok=True)
        self.optimizer_file = self.optimizer_dir / "model_optimizations.json"
        self.logger = logging.getLogger('EnsembleLLM.ModelOptimizer')
        
        # Load existing optimizations
        self.optimizations = self.load_optimizations()
    
    def load_optimizations(self) -> Dict:
        """Load saved optimizations"""
        if self.optimizer_file.exists():
            try:
                with open(self.optimizer_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'models': {},
            'global_params': {
                'optimal_temperature': 0.7,
                'optimal_top_p': 0.9,
                'optimal_num_predict': 512,
                'optimal_timeout_multiplier': 1.0
            },
            'query_patterns': {}
        }
    
    def save_optimizations(self):
        """Save optimizations to disk"""
        with open(self.optimizer_file, 'w') as f:
            json.dump(self.optimizations, f, indent=2)
    
    def update_model_params(self, model: str, success: bool, response_time: float, 
                           params: Dict, response_quality: float):
        """Update optimal parameters for a model based on performance"""
        
        if model not in self.optimizations['models']:
            self.optimizations['models'][model] = {
                'temperature_history': deque(maxlen=50),
                'top_p_history': deque(maxlen=50),
                'timeout_history': deque(maxlen=50),
                'success_rate': deque(maxlen=100),
                'avg_response_time': deque(maxlen=100),
                'quality_scores': deque(maxlen=100),
                'optimal_params': params.copy()
            }
        
        model_opt = self.optimizations['models'][model]
        
        # Record performance
        model_opt['success_rate'].append(1 if success else 0)
        model_opt['avg_response_time'].append(response_time)
        model_opt['quality_scores'].append(response_quality)
        
        # Update optimal parameters if this was successful
        if success and response_quality > 0.7:
            # Weighted average with existing optimal params
            alpha = 0.1  # Learning rate
            
            for param_key in ['temperature', 'top_p']:
                if param_key in params:
                    current_val = model_opt['optimal_params'].get(param_key, params[param_key])
                    new_val = (1 - alpha) * current_val + alpha * params[param_key]
                    model_opt['optimal_params'][param_key] = new_val
            
            # Adjust timeout based on response time
            if response_time < 10:
                model_opt['optimal_params']['timeout_multiplier'] = 0.8
            elif response_time > 30:
                model_opt['optimal_params']['timeout_multiplier'] = 1.3
            else:
                model_opt['optimal_params']['timeout_multiplier'] = 1.0
        
        # Calculate optimal num_predict based on average successful response length
        if len(model_opt['quality_scores']) > 10:
            avg_quality = sum(model_opt['quality_scores']) / len(model_opt['quality_scores'])
            
            if avg_quality > 0.8:
                model_opt['optimal_params']['num_predict'] = min(768, params.get('num_predict', 512) + 50)
            elif avg_quality < 0.5:
                model_opt['optimal_params']['num_predict'] = max(256, params.get('num_predict', 512) - 50)
        
        # Save periodically
        if sum(1 for m in self.optimizations['models'] for _ in [1]) % 10 == 0:
            self.save_optimizations()
    
    def get_optimal_params(self, model: str) -> Dict:
        """Get optimal parameters for a model"""
        
        if model in self.optimizations['models']:
            return self.optimizations['models'][model]['optimal_params']
        
        # Return global optimal params
        return self.optimizations['global_params'].copy()
    
    def get_model_confidence(self, model: str) -> float:
        """Get confidence score for a model based on history"""
        
        if model not in self.optimizations['models']:
            return 0.5  # Neutral confidence for new models
        
        model_opt = self.optimizations['models'][model]
        
        if len(model_opt['success_rate']) < 5:
            return 0.5  # Not enough data
        
        # Calculate weighted confidence
        success_rate = sum(model_opt['success_rate']) / len(model_opt['success_rate'])
        avg_quality = sum(model_opt['quality_scores']) / len(model_opt['quality_scores']) if model_opt['quality_scores'] else 0.5
        
        # Recent performance matters more
        recent_success = sum(list(model_opt['success_rate'])[-10:]) / min(10, len(model_opt['success_rate']))
        
        confidence = (success_rate * 0.3 + avg_quality * 0.3 + recent_success * 0.4)
        
        return confidence

class QueryPatternLearner:
    """Learn query patterns and optimal model selection"""
    
    def __init__(self, pattern_dir: str = "patterns"):
        self.pattern_dir = Path(pattern_dir)
        self.pattern_dir.mkdir(exist_ok=True)
        self.pattern_file = self.pattern_dir / "query_patterns.json"
        self.logger = logging.getLogger('EnsembleLLM.PatternLearner')
        
        # Load patterns
        self.patterns = self.load_patterns()
        
        # Query clustering
        self.vectorizer = TfidfVectorizer(max_features=50)
        self.cluster_model = None
    
    def load_patterns(self) -> Dict:
        """Load saved patterns"""
        if self.pattern_file.exists():
            try:
                with open(self.pattern_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'query_clusters': {},
            'model_preferences': {},
            'response_patterns': {},
            'optimal_configs': {}
        }
    
    def save_patterns(self):
        """Save patterns to disk"""
        with open(self.pattern_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
    
    def learn_from_query(self, query: str, selected_model: str, 
                         response_quality: float, query_type: Optional[str]):
        """Learn from a query result"""
        
        # Create or update query type pattern
        if query_type:
            if query_type not in self.patterns['model_preferences']:
                self.patterns['model_preferences'][query_type] = {}
            
            if selected_model not in self.patterns['model_preferences'][query_type]:
                self.patterns['model_preferences'][query_type][selected_model] = {
                    'count': 0,
                    'avg_quality': 0
                }
            
            pref = self.patterns['model_preferences'][query_type][selected_model]
            pref['count'] += 1
            pref['avg_quality'] = (pref['avg_quality'] * (pref['count'] - 1) + response_quality) / pref['count']
    
    def get_recommended_models(self, query: str, query_type: Optional[str], 
                              available_models: List[str]) -> List[str]:
        """Get recommended models for a query based on patterns"""
        
        recommendations = []
        
        if query_type and query_type in self.patterns['model_preferences']:
            # Sort models by their performance for this query type
            prefs = self.patterns['model_preferences'][query_type]
            
            model_scores = {}
            for model in available_models:
                if model in prefs:
                    score = prefs[model]['avg_quality'] * np.log1p(prefs[model]['count'])
                    model_scores[model] = score
                else:
                    model_scores[model] = 0.5  # Default score
            
            # Sort by score
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = [m[0] for m in sorted_models]
        else:
            # Return models in original order
            recommendations = available_models
        
        return recommendations

class SmartEnsembleOrchestrator:
    """Main orchestrator for smart ensemble learning"""
    
    def __init__(self, data_dir: str = "smart_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.query_cache = QueryCache(cache_dir=str(self.data_dir / "cache"))
        self.model_optimizer = ModelOptimizer(optimizer_dir=str(self.data_dir / "optimizers"))
        self.pattern_learner = QueryPatternLearner(pattern_dir=str(self.data_dir / "patterns"))
        
        self.logger = logging.getLogger('EnsembleLLM.SmartOrchestrator')
        
        # Load session data
        self.session_file = self.data_dir / "session_data.json"
        self.session_data = self.load_session_data()
    
    def load_session_data(self) -> Dict:
        """Load session data"""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_response_time': 0,
            'last_optimization': datetime.now().isoformat(),
            'model_rankings': {},
            'fast_model_pool': [],
            'accurate_model_pool': []
        }
    
    def save_session_data(self):
        """Save session data"""
        with open(self.session_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)
    
    async def check_cache(self, query: str) -> Optional[Tuple[str, Dict]]:
        """Check if query result is cached - with validation"""
        
        # Check exact match first
        query_hash = self.query_cache.get_query_hash(query)
        cached = self.query_cache.get_cached_response(query_hash)
        
        if cached:
            # Validate cached response before returning
            response = cached.get('response', '')
            metadata = cached.get('metadata', {})
            
            # Check if this was a failed response
            if (not response or 
                'all_failed' in metadata or 
                metadata.get('error') or
                response == ERROR_MESSAGES.get('all_failed', 'All models failed')):
                
                # Remove bad cache entry
                self.logger.warning(f"Removing invalid cache entry for query: {query[:50]}...")
                cache_file = self.query_cache.cache_dir / f"{query_hash}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                
                # Remove from index
                if query in self.query_cache.cache_index['queries']:
                    idx = self.query_cache.cache_index['queries'].index(query)
                    self.query_cache.cache_index['queries'].pop(idx)
                    self.query_cache.cache_index['hashes'].pop(idx)
                    self.query_cache.cache_index['timestamps'].pop(idx)
                    self.query_cache.cache_index['hit_counts'].pop(idx)
                    self.query_cache.save_cache_index()
                
                return None  # Don't use invalid cache
            
            self.session_data['cache_hits'] += 1
            self.logger.info(f"Cache hit (exact match) - {self.session_data['cache_hits']} total hits")
            return response, metadata
        
        # Check for similar query
        similar_result = self.query_cache.find_similar_query(query)
        
        if similar_result:
            similar_hash, similarity = similar_result
            cached = self.query_cache.get_cached_response(similar_hash)
            
            if cached:
                response = cached.get('response', '')
                metadata = cached.get('metadata', {})
                
                # Validate similar cache too
                if (not response or 
                    'all_failed' in metadata or 
                    metadata.get('error')):
                    return None
                
                self.session_data['cache_hits'] += 1
                self.logger.info(f"Cache hit (similarity: {similarity:.2f}) - {self.session_data['cache_hits']} total hits")
                
                metadata['cache_similarity'] = similarity
                return response, metadata
        
        return None

    def update_learning(self, query: str, response: str, metadata: Dict, 
                       model_performances: List[Dict]):
        """Update all learning components"""
        
        if (response and 
            not metadata.get('all_failed') and 
            not metadata.get('error') and
            response != ERROR_MESSAGES.get('all_failed', 'All models failed')):
            
            # Cache the successful response
            self.query_cache.cache_response(query, response, metadata)
            self.logger.debug(f"Cached successful response for: {query[:50]}...")
        else:
            self.logger.debug(f"Not caching failed response for: {query[:50]}...")
        
        
        # Update model optimizations
        for perf in model_performances:
            if perf['success']:
                self.model_optimizer.update_model_params(
                    model=perf['model'],
                    success=perf['success'],
                    response_time=perf.get('response_time', 0),
                    params=perf.get('params', {}),
                    response_quality=perf.get('quality_score', 0.5)
                )
        
        # Learn query patterns
        self.pattern_learner.learn_from_query(
            query=query,
            selected_model=metadata.get('selected_model'),
            response_quality=metadata.get('final_score', 0.5),
            query_type=metadata.get('query_type')
        )
        
        # Update session data
        self.session_data['total_queries'] += 1
        
        # Update average response time
        current_avg = self.session_data['avg_response_time']
        new_time = metadata.get('total_ensemble_time', 0)
        self.session_data['avg_response_time'] = (
            (current_avg * (self.session_data['total_queries'] - 1) + new_time) / 
            self.session_data['total_queries']
        )
        
        # Save periodically
        if self.session_data['total_queries'] % 5 == 0:
            self.save_session_data()
            self.model_optimizer.save_optimizations()
            self.pattern_learner.save_patterns()
    
    def get_optimized_models(self, available_models: List[str], 
                            query_type: Optional[str]) -> List[str]:
        """Get optimized model list based on learning"""
        
        # Get pattern-based recommendations
        recommended = self.pattern_learner.get_recommended_models(
            query="",  # We don't have the query here yet
            query_type=query_type,
            available_models=available_models
        )
        
        # Sort by confidence
        model_confidences = {}
        for model in recommended:
            confidence = self.model_optimizer.get_model_confidence(model)
            model_confidences[model] = confidence
        
        # Keep top performers and add variety
        sorted_models = sorted(model_confidences.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 60% performers and randomly sample from bottom 40% for variety
        cutoff = int(len(sorted_models) * 0.6)
        optimized = [m[0] for m in sorted_models[:cutoff]]
        
        # Add some variety
        if len(sorted_models) > cutoff:
            import random
            variety_pool = [m[0] for m in sorted_models[cutoff:]]
            variety_count = min(len(variety_pool), max(1, len(optimized) // 3))
            optimized.extend(random.sample(variety_pool, variety_count))
        
        return optimized[:len(available_models)]  # Don't exceed original count
    
    def get_optimal_params_for_model(self, model: str) -> Dict:
        """Get optimal parameters for a model"""
        return self.model_optimizer.get_optimal_params(model)
    
    def get_performance_insights(self) -> Dict:
        """Get insights about system performance"""
        
        insights = {
            'total_queries': self.session_data['total_queries'],
            'cache_hit_rate': (self.session_data['cache_hits'] / max(1, self.session_data['total_queries'])) * 100,
            'avg_response_time': self.session_data['avg_response_time'],
            'model_confidences': {},
            'top_models_by_type': {}
        }
        
        # Add model confidences
        for model in self.model_optimizer.optimizations['models']:
            insights['model_confidences'][model] = self.model_optimizer.get_model_confidence(model)
        
        # Add top models by query type
        for query_type, prefs in self.pattern_learner.patterns['model_preferences'].items():
            sorted_prefs = sorted(prefs.items(), key=lambda x: x[1]['avg_quality'], reverse=True)
            insights['top_models_by_type'][query_type] = [m[0] for m in sorted_prefs[:3]]
        
        return insights

class PrecomputeManager:
    """Manage precomputed embeddings and quick lookups"""
    
    def __init__(self, precompute_dir: str = "precompute"):
        self.precompute_dir = Path(precompute_dir)
        self.precompute_dir.mkdir(exist_ok=True)
        self.embeddings_file = self.precompute_dir / "embeddings.pkl"
        self.logger = logging.getLogger('EnsembleLLM.PrecomputeManager')
        
        # Load precomputed embeddings
        self.embeddings = self.load_embeddings()
    
    def load_embeddings(self) -> Dict:
        """Load precomputed embeddings"""
        if self.embeddings_file.exists():
            try:
                with open(self.embeddings_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        return {
            'common_queries': {},
            'model_signatures': {},
            'quick_responses': {}
        }
    
    def save_embeddings(self):
        """Save embeddings"""
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
    
    async def precompute_common_queries(self, models: List[str], common_queries: List[str]):
        """Precompute responses for common queries"""
        
        self.logger.info(f"Precomputing {len(common_queries)} common queries...")
        
        for query in common_queries:
            if query not in self.embeddings['common_queries']:
                self.embeddings['common_queries'][query] = {}
            
            # This would normally query models, but we'll simulate
            for model in models:
                if model not in self.embeddings['common_queries'][query]:
                    # In real implementation, this would query the model
                    self.embeddings['common_queries'][query][model] = {
                        'response': f"Precomputed response for {query}",
                        'timestamp': datetime.now().isoformat()
                    }
        
        self.save_embeddings()
    
    def get_precomputed_response(self, query: str, model: str) -> Optional[str]:
        """Get a precomputed response if available"""
        
        if query in self.embeddings['common_queries']:
            if model in self.embeddings['common_queries'][query]:
                data = self.embeddings['common_queries'][query][model]
                
                # Check if not too old (24 hours)
                timestamp = datetime.fromisoformat(data['timestamp'])
                if (datetime.now() - timestamp) < timedelta(days=1):
                    return data['response']
        
        return None

class CacheManager:
    """Utilities for cache management"""
    
    @staticmethod
    def clear_all_cache(cache_dir: str = "smart_data/cache"):
        """Clear all cached responses"""
        import shutil
        from pathlib import Path
        
        cache_path = Path(cache_dir)
        if cache_path.exists():
            shutil.rmtree(cache_path)
            cache_path.mkdir(exist_ok=True)
            print(f"✅ Cleared all cache in {cache_dir}")
    
    @staticmethod
    def clear_failed_cache(cache_dir: str = "smart_data/cache"):
        """Clear only failed/invalid cached responses"""
        from pathlib import Path
        import pickle
        
        cache_path = Path(cache_dir)
        removed_count = 0
        
        for cache_file in cache_path.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    
                response = data.get('response', '')
                metadata = data.get('metadata', {})
                
                # Check if this is a failed response
                if (not response or 
                    'all_failed' in metadata or 
                    metadata.get('error') or
                    'All models failed' in response):
                    
                    cache_file.unlink()
                    removed_count += 1
                    print(f"❌ Removed failed cache: {data.get('query', 'unknown')[:50]}...")
                    
            except Exception as e:
                print(f"Error checking {cache_file}: {e}")
        
        print(f"✅ Removed {removed_count} failed cache entries")
        
    @staticmethod
    def show_cache_stats(cache_dir: str = "smart_data/cache"):
        """Show cache statistics"""
        from pathlib import Path
        import pickle
        
        cache_path = Path(cache_dir)
        total = 0
        failed = 0
        
        for cache_file in cache_path.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    total += 1
                    
                    response = data.get('response', '')
                    metadata = data.get('metadata', {})
                    
                    if (not response or 
                        'all_failed' in metadata or 
                        metadata.get('error')):
                        failed += 1
                        
            except:
                pass
        
        print(f"📊 Cache Statistics:")
        print(f"   Total entries: {total}")
        print(f"   Failed entries: {failed}")
        print(f"   Valid entries: {total - failed}")
        print(f"   Cache health: {((total-failed)/max(1,total))*100:.1f}%")