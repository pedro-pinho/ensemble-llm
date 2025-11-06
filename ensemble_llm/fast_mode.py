"""Fast mode optimizations for quick responses"""

import asyncio
import aiohttp
import time
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

class FastModeOrchestrator:
    """Orchestrate fast query execution strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger('EnsembleLLM.FastMode')
        
        # Fast execution strategies
        self.strategies = {
            'race': self.race_strategy,
            'cascade': self.cascade_strategy,
            'single_best': self.single_best_strategy,
            'quick_consensus': self.quick_consensus_strategy
        }
    
    async def race_strategy(self, query_func, models: List[str], prompt: str, 
                           min_responses: int = 1) -> List[Dict]:
        """Race models and return as soon as we have minimum responses"""
        
        results = []
        pending_tasks = []
        
        async def query_with_tracking(model, prompt):
            result = await query_func(model, prompt)
            return model, result
        
        # Start all queries
        for model in models:
            task = asyncio.create_task(query_with_tracking(model, prompt))
            pending_tasks.append(task)
        
        # Collect results as they complete
        while pending_tasks and len(results) < min_responses:
            done, pending = await asyncio.wait(
                pending_tasks, 
                return_when=asyncio.FIRST_COMPLETED,
                timeout=5  # Check every 5 seconds
            )
            
            for task in done:
                try:
                    model, result = await task
                    if result['success']:
                        results.append(result)
                        self.logger.info(f"Got response from {model} in race mode")
                except:
                    pass
            
            pending_tasks = list(pending)
        
        # Cancel remaining tasks
        for task in pending_tasks:
            task.cancel()
        
        return results
    
    async def cascade_strategy(self, query_func, models: List[str], prompt: str,
                              cascade_delay: float = 2.0) -> List[Dict]:
        """Start with fast model, add more if needed"""
        
        results = []
        
        # Start with fastest model
        fast_model = models[0]
        task1 = asyncio.create_task(query_func(fast_model, prompt))
        
        # Wait a bit, then start second model if first hasn't responded
        await asyncio.sleep(cascade_delay)
        
        if not task1.done():
            if len(models) > 1:
                task2 = asyncio.create_task(query_func(models[1], prompt))
                
                # Wait for either to complete
                done, pending = await asyncio.wait(
                    [task1, task2],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    result = await task
                    if result['success']:
                        results.append(result)
                        break
                
                # Cancel pending
                for task in pending:
                    task.cancel()
        else:
            result = await task1
            if result['success']:
                results.append(result)
        
        return results
    
    async def single_best_strategy(self, query_func, model: str, prompt: str) -> List[Dict]:
        """Use only the single best model"""
        result = await query_func(model, prompt)
        return [result] if result['success'] else []
    
    async def quick_consensus_strategy(self, query_func, models: List[str], 
                                      prompt: str) -> List[Dict]:
        """Get quick consensus from 2-3 fast models"""
        
        fast_models = models[:2]  # Use only 2 fastest
        tasks = [query_func(model, prompt) for model in fast_models]
        
        results = await asyncio.gather(*tasks)
        return [r for r in results if r['success']]

class TurboMode:
    """Ultra-fast query mode with aggressive optimizations"""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.logger = logging.getLogger('EnsembleLLM.TurboMode')
        
        # Turbo settings
        self.turbo_config = {
            'timeout': 10,  # Aggressive timeout
            'num_predict': 200,  # Shorter responses
            'temperature': 0.5,  # More deterministic
            'num_ctx': 1024,  # Smaller context
            'num_thread': 4,  # Limit threads
            'num_gpu': 1,  # GPU layers
            'repeat_penalty': 1.0
        }
    
    async def turbo_query(self, model: str, prompt: str, 
                         session: Optional[aiohttp.ClientSession] = None) -> Dict:
        """Execute turbo query with aggressive optimizations"""
        
        start_time = time.time()
        
        # Create session if not provided
        if not session:
            session = aiohttp.ClientSession()
            close_session = True
        else:
            close_session = False
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": self.turbo_config,
            "keep_alive": "5m"  # Keep model loaded
        }
        
        try:
            async with session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=self.turbo_config['timeout'],
                    connect=2,
                    sock_read=self.turbo_config['timeout']
                )
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    return {
                        "model": model,
                        "response": result.get("response", ""),
                        "success": True,
                        "response_time": time.time() - start_time,
                        "turbo": True
                    }
                else:
                    return {
                        "model": model,
                        "response": f"HTTP {response.status}",
                        "success": False,
                        "response_time": time.time() - start_time
                    }
                    
        except asyncio.TimeoutError:
            self.logger.warning(f"Turbo timeout for {model}")
            return {
                "model": model,
                "response": "Turbo timeout",
                "success": False,
                "response_time": time.time() - start_time
            }
        except Exception as e:
            return {
                "model": model,
                "response": str(e),
                "success": False,
                "response_time": time.time() - start_time
            }
        finally:
            if close_session:
                await session.close()

class ModelWarmup:
    """Warmup models for faster responses"""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.logger = logging.getLogger('EnsembleLLM.ModelWarmup')
    
    async def warmup_model(self, model: str) -> bool:
        """Warmup a single model"""
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {
                        "num_predict": 1,
                        "temperature": 0.1
                    },
                    "keep_alive": "10m"  # Keep loaded for 10 minutes
                }
                
                async with session.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    
                    if response.status == 200:
                        self.logger.info(f"Warmed up {model}")
                        return True
                        
        except Exception as e:
            self.logger.error(f"Failed to warmup {model}: {str(e)}")
        
        return False
    
    async def parallel_warmup(self, models: List[str], max_concurrent: int = 2):
        """Warmup multiple models with concurrency limit"""
        
        self.logger.info(f"Warming up {len(models)} models...")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def warmup_with_limit(model):
            async with semaphore:
                return await self.warmup_model(model)
        
        tasks = [warmup_with_limit(model) for model in models]
        results = await asyncio.gather(*tasks)
        
        successful = sum(1 for r in results if r)
        self.logger.info(f"Warmed up {successful}/{len(models)} models")