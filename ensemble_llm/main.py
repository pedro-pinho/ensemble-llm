#!/usr/bin/env python3

from .web_search import WebSearcher

import asyncio
import aiohttp
import json
import numpy as np
import logging
import sys
import traceback
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import Counter
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import quote
import re

# Configure logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output"""
    
    grey = "\x1b[38;21m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[34m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: green + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

# Set up logger
logger = logging.getLogger('EnsembleLLM')
logger.setLevel(logging.DEBUG)

# Console handler with colored output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ColoredFormatter())
logger.addHandler(console_handler)

# File handler for detailed logs
file_handler = logging.FileHandler('ensemble_llm.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

class EnsembleLLM:
    def __init__(self, models: List[str], ollama_host: str = "http://localhost:11434", 
                 enable_web_search: bool = False):
        self.models = models
        self.host = ollama_host
        self.vectorizer = TfidfVectorizer()
        self.enable_web_search = enable_web_search
        self.web_searcher = WebSearcher() if enable_web_search else None
        self.logger = logging.getLogger('EnsembleLLM')
        
        self.logger.info(f"Initialized with models: {', '.join(models)}")
        self.logger.info(f"Web search: {'Enabled' if enable_web_search else 'Disabled'}")
        
    async def check_model_health(self, model: str) -> bool:
        """Check if a model is available and responding"""
        try:
            async with aiohttp.ClientSession() as session:
                # First check if model is available
                async with session.get(f"{self.host}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        available_models = [m['name'] for m in data.get('models', [])]
                        if model not in available_models:
                            self.logger.warning(f"Model {model} not found. Available models: {', '.join(available_models)}")
                            return False
                
                # Try a simple test query
                test_payload = {
                    "model": model,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {"num_predict": 1}
                }
                
                async with session.post(
                    f"{self.host}/api/generate", 
                    json=test_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.error(f"Health check failed for {model}: {str(e)}")
            return False
    
    async def detect_uncertainty(self, response: str) -> bool:
        """Detect if a model's response indicates uncertainty"""
        uncertainty_phrases = [
            "i don't have information",
            "i don't know",
            "i'm not sure",
            "i cannot provide current",
            "as of my knowledge cutoff",
            "i don't have access to real-time",
            "i don't have current data",
            "my training data",
            "i cannot access current"
        ]
        
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in uncertainty_phrases)
    
    async def enhance_prompt_with_web_search(self, prompt: str) -> Tuple[str, bool]:
        """Enhance prompt with web search results if needed"""
        
        # Check if web search is needed
        needs_search = False
        
        # Keywords that indicate need for current information
        current_keywords = ['current', 'latest', 'today', 'now', 'recent', '2024', '2025', 
                          'news', 'update', 'happening', 'price', 'weather', 'stock']
        
        prompt_lower = prompt.lower()
        for keyword in current_keywords:
            if keyword in prompt_lower:
                needs_search = True
                break
        
        if not needs_search or not self.enable_web_search:
            return prompt, False
        
        # Perform web search
        self.logger.info("Enhancing prompt with web search results...")
        web_context = await self.web_searcher.search_with_fallback(prompt)
        
        if web_context and web_context != "No web search results found.":
            enhanced_prompt = f"""Context from web search:
{web_context}

Based on the above context and your knowledge, please answer: {prompt}"""
            return enhanced_prompt, True
        
        return prompt, False
        
    async def query_model(self, session, model: str, prompt: str, 
                         retry_count: int = 2) -> Dict:
        """Query a single model asynchronously with retry logic"""
        
        self.logger.debug(f"Querying model: {model}")
        
        # Check if we should enhance with web search
        enhanced_prompt = prompt
        used_web_search = False
        
        if self.enable_web_search:
            enhanced_prompt, used_web_search = await self.enhance_prompt_with_web_search(prompt)
        
        payload = {
            "model": model,
            "prompt": enhanced_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 1024,  # Limit response length
            }
        }
        
        for attempt in range(retry_count):
            try:
                self.logger.debug(f"Attempt {attempt + 1} for model {model}")
                
                async with session.post(
                    f"{self.host}/api/generate", 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Model {model} returned status {response.status}: {error_text}")
                        
                        # Try to parse error message
                        try:
                            error_data = json.loads(error_text)
                            error_msg = error_data.get('error', error_text)
                        except:
                            error_msg = error_text
                        
                        if attempt < retry_count - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        
                        return {
                            "model": model,
                            "response": f"HTTP {response.status}: {error_msg}",
                            "success": False,
                            "used_web_search": used_web_search
                        }
                    
                    result = await response.json()
                    response_text = result.get("response", "")
                    
                    # Check if response indicates uncertainty and we haven't searched yet
                    if (self.enable_web_search and not used_web_search and 
                        await self.detect_uncertainty(response_text)):
                        
                        self.logger.info(f"Model {model} seems uncertain, adding web search context...")
                        enhanced_prompt, used_web_search = await self.enhance_prompt_with_web_search(prompt)
                        payload["prompt"] = enhanced_prompt
                        
                        # Retry with enhanced prompt
                        continue
                    
                    self.logger.debug(f"Successfully got response from {model}")
                    return {
                        "model": model,
                        "response": response_text,
                        "success": True,
                        "used_web_search": used_web_search,
                        "metadata": {
                            "total_duration": result.get("total_duration", 0) / 1e9,  # Convert to seconds
                            "load_duration": result.get("load_duration", 0) / 1e9,
                            "eval_count": result.get("eval_count", 0)
                        }
                    }
                    
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout for model {model} on attempt {attempt + 1}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {
                    "model": model,
                    "response": "Error: Request timed out",
                    "success": False,
                    "used_web_search": used_web_search
                }
                
            except aiohttp.ClientError as e:
                self.logger.error(f"Client error for model {model}: {type(e).__name__}: {str(e)}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {
                    "model": model,
                    "response": f"Error: Connection failed - {str(e)}",
                    "success": False,
                    "used_web_search": used_web_search
                }
                
            except Exception as e:
                self.logger.error(f"Unexpected error for model {model}: {type(e).__name__}: {str(e)}")
                self.logger.debug(traceback.format_exc())
                
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                    
                return {
                    "model": model,
                    "response": f"Error: {type(e).__name__}: {str(e)}",
                    "success": False,
                    "used_web_search": used_web_search
                }
        
        return {
            "model": model,
            "response": "Error: All retry attempts failed",
            "success": False,
            "used_web_search": used_web_search
        }
    
    async def query_all_models(self, prompt: str) -> List[Dict]:
        """Query all models in parallel"""
        
        # First, check model health
        self.logger.info("Checking model availability...")
        model_health = {}
        
        for model in self.models:
            is_healthy = await self.check_model_health(model)
            model_health[model] = is_healthy
            if not is_healthy:
                self.logger.warning(f"Model {model} is not available or not responding")
        
        # Only query healthy models
        healthy_models = [m for m in self.models if model_health.get(m, False)]
        
        if not healthy_models:
            self.logger.error("No healthy models available!")
            return []
        
        self.logger.info(f"Querying {len(healthy_models)} healthy models: {', '.join(healthy_models)}")
        
        async with aiohttp.ClientSession() as session:
            tasks = [self.query_model(session, model, prompt) for model in healthy_models]
            responses = await asyncio.gather(*tasks)
            
            # Add unavailable models to responses
            for model in self.models:
                if not model_health.get(model, False):
                    responses.append({
                        "model": model,
                        "response": "Model not available or not installed",
                        "success": False,
                        "used_web_search": False
                    })
            
            return responses
    
    def calculate_similarity_matrix(self, responses: List[str]) -> np.ndarray:
        """Calculate pairwise similarity between responses"""
        if len(responses) < 2:
            return np.array([[1.0]])
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(responses)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {str(e)}")
            return np.ones((len(responses), len(responses)))
    
    def weighted_voting(self, responses: List[Dict]) -> Tuple[str, Dict]:
        """
        Implement weighted voting based on:
        1. Response similarity (consensus)
        2. Response length (detail)
        3. Model performance metrics
        4. Web search enhancement bonus
        """
        valid_responses = [r for r in responses if r["success"]]
        
        if not valid_responses:
            return "All models failed to respond", {"error": "No valid responses"}
        
        if len(valid_responses) == 1:
            return valid_responses[0]["response"], {
                "selected_model": valid_responses[0]["model"],
                "consensus_score": 1.0,
                "used_web_search": valid_responses[0].get("used_web_search", False)
            }
        
        response_texts = [r["response"] for r in valid_responses]
        
        # Calculate similarity matrix
        similarity_matrix = self.calculate_similarity_matrix(response_texts)
        
        # Calculate consensus scores
        consensus_scores = np.mean(similarity_matrix, axis=1)
        
        # Calculate quality scores
        quality_scores = []
        for i, response in enumerate(valid_responses):
            text = response["response"]
            
            # Length score (prefer moderate length)
            length_score = min(len(text) / 500, 2.0) / 2.0
            
            # Structure score
            has_structure = 1.0 if '\n' in text else 0.8
            
            # Web search bonus
            web_bonus = 1.2 if response.get("used_web_search", False) else 1.0
            
            # Response time penalty (faster is slightly better)
            metadata = response.get("metadata", {})
            time_penalty = 1.0
            if metadata.get("total_duration"):
                # Normalize: under 5s = 1.0, over 20s = 0.8
                time_penalty = max(0.8, 1.0 - (metadata["total_duration"] - 5) * 0.01)
            
            quality_score = length_score * has_structure * web_bonus * time_penalty
            quality_scores.append(quality_score)
        
        quality_scores = np.array(quality_scores)
        
        # Combined score
        final_scores = (consensus_scores * 0.6 + quality_scores * 0.4)
        
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
            "response_time": best_response.get("metadata", {}).get("total_duration", 0),
            "all_scores": {
                valid_responses[i]["model"]: {
                    "consensus": float(consensus_scores[i]),
                    "quality": float(quality_scores[i]),
                    "final": float(final_scores[i]),
                    "used_web": valid_responses[i].get("used_web_search", False)
                }
                for i in range(len(valid_responses))
            },
            "total_models": len(responses),
            "successful_models": len(valid_responses)
        }
        
        return best_response["response"], metadata
    
    async def ensemble_query(self, prompt: str, verbose: bool = False) -> Tuple[str, Dict]:
        """Main ensemble query method"""
        
        start_time = datetime.now()
        
        self.logger.info(f"Starting ensemble query with {len(self.models)} models")
        self.logger.info(f"Query: {prompt}")
        
        # Query all models
        responses = await self.query_all_models(prompt)
        
        if verbose:
            print("\n" + "="*60)
            print("📊 Individual Model Responses:")
            print("="*60)
            for r in responses:
                status = "✅" if r["success"] else "❌"
                web = "🌐" if r.get("used_web_search", False) else ""
                
                print(f"\n{status} [{r['model']}] {web}")
                
                if r["success"]:
                    # Show first 300 chars
                    preview = r["response"][:300] + "..." if len(r["response"]) > 300 else r["response"]
                    print(f"   {preview}")
                    
                    if r.get("metadata"):
                        meta = r["metadata"]
                        print(f"   ⏱️  Response time: {meta.get('total_duration', 0):.2f}s")
                else:
                    print(f"   Error: {r['response']}")
        
        # Perform weighted voting
        best_response, metadata = self.weighted_voting(responses)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        metadata["total_ensemble_time"] = total_time
        
        if verbose:
            print("\n" + "="*60)
            print("🏆 Voting Results:")
            print("="*60)
            print(f"Selected Model: {metadata.get('selected_model', 'N/A')}")
            print(f"Consensus Score: {metadata.get('consensus_score', 0):.2f}")
            print(f"Quality Score: {metadata.get('quality_score', 0):.2f}")
            print(f"Final Score: {metadata.get('final_score', 0):.2f}")
            print(f"Used Web Search: {metadata.get('used_web_search', False)}")
            print(f"Total Time: {total_time:.2f}s")
            
            if 'all_scores' in metadata:
                print("\n📈 All Model Scores:")
                sorted_models = sorted(metadata['all_scores'].items(), 
                                     key=lambda x: x[1]['final'], reverse=True)
                for model, scores in sorted_models:
                    web_indicator = "🌐" if scores.get('used_web', False) else "  "
                    print(f"   {web_indicator} {model}: {scores['final']:.3f} "
                          f"(consensus: {scores['consensus']:.2f}, quality: {scores['quality']:.2f})")
        
        self.logger.info(f"Ensemble query completed in {total_time:.2f}s")
        
        return best_response, metadata
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.web_searcher:
            await self.web_searcher.close()

# CLI Interface
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Ensemble LLM with voting and web search')
    parser.add_argument('prompt', nargs='?', help='Question to ask the ensemble')
    parser.add_argument('--models', nargs='+', 
                       default=['llama3.2:3b', 'phi3.5', 'qwen2.5:7b', 'mistral:7b-instruct-q4_K_M'],
                       help='Models to use in ensemble')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show individual model responses and scores')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode')
    parser.add_argument('--web-search', '-w', action='store_true',
                       help='Enable web search for current information')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set logging level')
    
    args = parser.parse_args()
    
    # Adjust logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    print(f"\n🚀 Ensemble LLM System v2.0")
    print(f"📦 Models: {', '.join(args.models)}")
    print(f"🌐 Web Search: {'Enabled' if args.web_search else 'Disabled'}")
    print(f"📝 Log Level: {args.log_level}")
    print(f"📄 Detailed logs: ensemble_llm.log\n")
    
    ensemble = EnsembleLLM(args.models, enable_web_search=args.web_search)
    
    try:
        if args.interactive or not args.prompt:
            print("💡 Type 'exit' to quit\n")
            
            while True:
                try:
                    prompt = input("\n💭 Your question: ").strip()
                    if prompt.lower() in ['exit', 'quit', 'q']:
                        break
                    
                    if prompt:
                        response, metadata = await ensemble.ensemble_query(prompt, args.verbose)
                        
                        web_indicator = "🌐 " if metadata.get('used_web_search', False) else ""
                        print(f"\n✨ {web_indicator}Best Answer (from {metadata.get('selected_model', 'ensemble')}):")
                        print(response)
                        
                        if not args.verbose:
                            print(f"\n📊 Score: {metadata.get('final_score', 0):.2f} | "
                                  f"Time: {metadata.get('total_ensemble_time', 0):.1f}s")
                        
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    logger.error(f"Error in interactive mode: {str(e)}")
                    print(f"\n❌ Error: {str(e)}")
        else:
            response, metadata = await ensemble.ensemble_query(args.prompt, args.verbose)
            
            web_indicator = "🌐 " if metadata.get('used_web_search', False) else ""
            print(f"\n✨ {web_indicator}Best Answer (from {metadata.get('selected_model', 'ensemble')}):")
            print(response)
            
            if not args.verbose:
                print(f"\n📊 Score: {metadata.get('final_score', 0):.2f} | "
                      f"Time: {metadata.get('total_ensemble_time', 0):.1f}s")
                      
    finally:
        await ensemble.cleanup()
        logger.info("Ensemble LLM shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())