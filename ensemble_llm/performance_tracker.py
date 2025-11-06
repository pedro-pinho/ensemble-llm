"""Performance tracking and model management for Ensemble LLM"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from pathlib import Path
import asyncio
import aiohttp
import logging

from .verbose_logger import VerboseFileLogger, ModelPerformanceLogger


class ModelPerformanceTracker:
    """Track model performance and manage model rotation"""

    def __init__(self, data_dir: str = "data", verbose_logging: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.performance_file = self.data_dir / "model_performance.json"
        self.config_file = self.data_dir / "adaptive_config.json"
        self.logger = logging.getLogger("EnsembleLLM.PerformanceTracker")

        # Load or initialize performance data
        self.performance_data = self.load_performance_data()

        # Model pool for rotation
        self.model_pool = {
            "primary": [
                "llama3.2:3b",
                "phi3.5:latest",
                "qwen2.5:7b",
                "mistral:7b-instruct-q4_K_M",
            ],
            "backup": [
                "gemma2:2b",
                "gemma2:9b",
                "tinyllama:1b",
                "orca-mini:3b",
                "neural-chat:7b",
                "openhermes:7b",
            ],
            "specialized": {
                "code": ["codellama:7b", "deepseek-coder:6.7b"],
                "math": ["wizard-math:7b"],
                "creative": ["zephyr:7b"],
            },
        }

        # Performance thresholds
        self.thresholds = {
            "min_success_rate": 0.7,  # Minimum 70% success rate
            "min_selection_rate": 0.05,  # Selected at least 5% of the time
            "max_avg_response_time": 30,  # Max 30 seconds average
            "evaluation_window": 50,  # Last 50 queries
            "retirement_threshold": 10,  # Retire after 10 consecutive failures
        }

        self.verbose_logging = verbose_logging
        if verbose_logging:
            self.verbose_logger = VerboseFileLogger()
            self.performance_logger = ModelPerformanceLogger()

    def load_performance_data(self) -> Dict:
        """Load performance data from disk"""
        if self.performance_file.exists():
            try:
                with open(self.performance_file, "r") as f:
                    return json.load(f)
            except:
                pass

        # Initialize with empty structure
        return {
            "models": {},
            "global_stats": {
                "total_queries": 0,
                "start_date": datetime.now().isoformat(),
            },
        }

    def save_performance_data(self):
        """Save performance data to disk"""
        with open(self.performance_file, "w") as f:
            json.dump(self.performance_data, f, indent=2)

    def record_query_result(
        self,
        model: str,
        success: bool,
        response_time: float,
        was_selected: bool,
        consensus_score: float = 0.0,
        quality_score: float = 0.0,
    ):
        """Record the result of a model query"""

        if model not in self.performance_data["models"]:
            self.performance_data["models"][model] = {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "times_selected": 0,
                "total_response_time": 0,
                "scores": [],
                "recent_performance": [],
                "consecutive_failures": 0,
                "added_date": datetime.now().isoformat(),
            }

        model_data = self.performance_data["models"][model]

        # Update counters
        model_data["total_queries"] += 1
        if success:
            model_data["successful_queries"] += 1
            model_data["consecutive_failures"] = 0
        else:
            model_data["failed_queries"] += 1
            model_data["consecutive_failures"] += 1

        if was_selected:
            model_data["times_selected"] += 1

        model_data["total_response_time"] += response_time

        # Track recent performance (sliding window)
        model_data["recent_performance"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "response_time": response_time,
                "selected": was_selected,
                "consensus_score": consensus_score,
                "quality_score": quality_score,
            }
        )

        # Keep only recent history
        if len(model_data["recent_performance"]) > self.thresholds["evaluation_window"]:
            model_data["recent_performance"] = model_data["recent_performance"][
                -self.thresholds["evaluation_window"] :
            ]

        # Update global stats
        self.performance_data["global_stats"]["total_queries"] += 1

        # Save periodically (every 10 queries)
        if self.performance_data["global_stats"]["total_queries"] % 10 == 0:
            self.save_performance_data()

    def evaluate_model_performance(self, model: str) -> Dict:
        """Evaluate a model's recent performance"""

        if model not in self.performance_data["models"]:
            return {"status": "new", "score": 0.5}

        model_data = self.performance_data["models"][model]
        recent = model_data["recent_performance"]

        if len(recent) < 5:  # Not enough data
            return {"status": "insufficient_data", "score": 0.5}

        # Calculate metrics
        recent_success_rate = sum(1 for r in recent if r["success"]) / len(recent)
        recent_selection_rate = sum(1 for r in recent if r["selected"]) / len(recent)
        avg_response_time = np.mean(
            [r["response_time"] for r in recent if r["success"]]
        )

        # Calculate composite score (0-1)
        success_score = recent_success_rate
        selection_score = min(
            recent_selection_rate / 0.2, 1.0
        )  # Normalize to expected 20% selection
        time_score = max(0, 1 - (avg_response_time / 30))  # Normalize to 30s max

        composite_score = success_score * 0.5 + selection_score * 0.3 + time_score * 0.2

        # Determine status
        status = "healthy"
        if recent_success_rate < self.thresholds["min_success_rate"]:
            status = "unhealthy"
        elif recent_selection_rate < self.thresholds["min_selection_rate"]:
            status = "underutilized"
        elif avg_response_time > self.thresholds["max_avg_response_time"]:
            status = "slow"
        elif (
            model_data["consecutive_failures"]
            >= self.thresholds["retirement_threshold"]
        ):
            status = "failing"

        # Log performance update if verbose
        if self.verbose_logging and hasattr(self, "verbose_logger"):
            self.verbose_logger.log_performance_update(
                model,
                {
                    "success_rate": recent_success_rate,
                    "avg_response_time": avg_response_time,
                    "quality_score": composite_score,
                    "status": status,
                },
            )

        return {
            "status": status,
            "score": composite_score,
            "success_rate": recent_success_rate,
            "selection_rate": recent_selection_rate,
            "avg_response_time": avg_response_time,
            "consecutive_failures": model_data["consecutive_failures"],
        }

    def get_model_recommendations(
        self, current_models: List[str], query_type: Optional[str] = None
    ) -> Dict:
        """Get recommendations for model rotation"""

        recommendations = {
            "keep": [],
            "remove": [],
            "add": [],
            "reasons": {},
            "performance_data": {},
        }

        # Evaluate current models
        model_scores = {}
        for model in current_models:
            evaluation = self.evaluate_model_performance(model)
            model_scores[model] = evaluation

            # Store performance data for logging
            recommendations["performance_data"][model] = evaluation

            if evaluation["status"] in ["healthy", "new", "insufficient_data"]:
                recommendations["keep"].append(model)
            else:
                recommendations["remove"].append(model)
                recommendations["reasons"][model] = evaluation["status"]

        # Find replacement models if needed
        if recommendations["remove"]:
            # Get available models from backup pool
            available_models = []

            # Check specialized models first if query type is known
            if query_type and query_type in self.model_pool["specialized"]:
                available_models.extend(self.model_pool["specialized"][query_type])

            # Add backup models
            available_models.extend(self.model_pool["backup"])

            # Filter out models already in use or recently failed
            for candidate in available_models:
                if candidate not in current_models:
                    candidate_eval = self.evaluate_model_performance(candidate)

                    # Only add if not recently failed
                    if candidate_eval["status"] != "failing":
                        recommendations["add"].append(candidate)

                        # Only add as many as we're removing
                        if len(recommendations["add"]) >= len(
                            recommendations["remove"]
                        ):
                            break

        if self.verbose_logging and (
            recommendations["remove"] or recommendations["add"]
        ):
            if hasattr(self, "verbose_logger"):
                self.verbose_logger.log_model_rotation(
                    removed_models=recommendations["remove"],
                    added_models=recommendations["add"],
                    reasons=recommendations["reasons"],
                    performance_data=recommendations["performance_data"],
                )

        return recommendations

    def get_performance_summary(self) -> str:
        """Get a human-readable performance summary"""

        summary = ["Model Performance Summary", "=" * 40]

        for model, data in self.performance_data["models"].items():
            evaluation = self.evaluate_model_performance(model)

            if data["total_queries"] > 0:
                success_rate = (
                    data["successful_queries"] / data["total_queries"]
                ) * 100
                selection_rate = (data["times_selected"] / data["total_queries"]) * 100
                avg_time = data["total_response_time"] / data["total_queries"]

                status_emoji = {
                    "healthy": "✅",
                    "unhealthy": "⚠️",
                    "underutilized": "📊",
                    "slow": "🐢",
                    "failing": "❌",
                    "new": "🆕",
                    "insufficient_data": "📈",
                }.get(evaluation["status"], "❓")

                summary.append(f"\n{status_emoji} {model}:")
                summary.append(f"   Status: {evaluation['status']}")
                summary.append(f"   Score: {evaluation['score']:.2f}")
                summary.append(f"   Success Rate: {success_rate:.1f}%")
                summary.append(f"   Selection Rate: {selection_rate:.1f}%")
                summary.append(f"   Avg Response Time: {avg_time:.1f}s")
                summary.append(f"   Total Queries: {data['total_queries']}")

        return "\n".join(summary)


class AdaptiveModelManager:
    """Manage dynamic model loading and optimization"""

    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.host = ollama_host
        self.tracker = ModelPerformanceTracker()
        self.logger = logging.getLogger("EnsembleLLM.AdaptiveManager")

        # Optimized settings for different RAM configurations
        self.ram_configs = {
            "16GB": {
                "max_models": 3,
                "max_total_memory_gb": 12,
                "timeout_base": 40,
                "stagger_delay": 0.5,
            },
            "24GB": {
                "max_models": 4,
                "max_total_memory_gb": 18,
                "timeout_base": 35,
                "stagger_delay": 0.3,
            },
            "32GB": {
                "max_models": 5,
                "max_total_memory_gb": 25,
                "timeout_base": 30,
                "stagger_delay": 0.2,
            },
        }

        # Detect RAM and set config
        self.system_config = self.detect_system_config()

    def detect_system_config(self) -> Dict:
        """Detect system RAM and return appropriate config"""
        try:
            import psutil

            total_ram_gb = psutil.virtual_memory().total / (1024**3)

            if total_ram_gb <= 16:
                config = self.ram_configs["16GB"]
            elif total_ram_gb <= 24:
                config = self.ram_configs["24GB"]
            else:
                config = self.ram_configs["32GB"]

            self.logger.info(
                f"Detected {total_ram_gb:.1f}GB RAM, using {config['max_models']} models max"
            )
            return config

        except ImportError:
            self.logger.warning("psutil not installed, using default config")
            return self.ram_configs["24GB"]

    async def optimize_models(self, current_models: List[str]) -> List[str]:
        """Optimize model selection based on performance"""

        # Get recommendations
        recommendations = self.tracker.get_model_recommendations(current_models)

        if recommendations["remove"]:
            self.logger.info(
                f"Removing underperforming models: {recommendations['remove']}"
            )
            self.logger.info(f"Reasons: {recommendations['reasons']}")

            # Unload poor performing models
            for model in recommendations["remove"]:
                await self.unload_model(model)
                current_models.remove(model)

        if recommendations["add"]:
            self.logger.info(f"Adding new models: {recommendations['add']}")

            # Load new models
            for model in recommendations["add"]:
                if await self.ensure_model_loaded(model):
                    current_models.append(model)

        # Ensure we don't exceed max models
        if len(current_models) > self.system_config["max_models"]:
            # Keep the best performing ones
            model_scores = {}
            for model in current_models:
                eval_result = self.tracker.evaluate_model_performance(model)
                model_scores[model] = eval_result["score"]

            # Sort by score and keep top N
            sorted_models = sorted(
                model_scores.items(), key=lambda x: x[1], reverse=True
            )
            current_models = [
                m[0] for m in sorted_models[: self.system_config["max_models"]]
            ]

            self.logger.info(
                f"Optimized to {len(current_models)} models: {current_models}"
            )

        return current_models

    async def ensure_model_loaded(self, model: str) -> bool:
        """Ensure a model is loaded and ready"""
        try:
            async with aiohttp.ClientSession() as session:
                # Check if model exists
                payload = {"name": model}

                async with session.post(
                    f"{self.host}/api/show", json=payload, timeout=10
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Model {model} is available")
                        return True
                    else:
                        # Try to pull the model
                        self.logger.info(f"Pulling model {model}...")
                        pull_payload = {"name": model}

                        async with session.post(
                            f"{self.host}/api/pull",
                            json=pull_payload,
                            timeout=300,  # 5 minutes for download
                        ) as pull_response:
                            if pull_response.status == 200:
                                self.logger.info(f"Successfully pulled {model}")
                                return True

        except Exception as e:
            self.logger.error(f"Failed to ensure model {model}: {str(e)}")

        return False

    async def unload_model(self, model: str):
        """Unload a model from memory"""
        try:
            # Ollama doesn't have a direct unload, but we can achieve this
            # by loading a tiny model which forces memory cleanup
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "tinyllama:1b",  # Use smallest model to trigger cleanup
                    "prompt": ".",
                    "options": {"num_predict": 1},
                }

                async with session.post(
                    f"{self.host}/api/generate", json=payload, timeout=10
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Memory cleared after unloading {model}")

        except Exception as e:
            self.logger.error(f"Failed to unload model {model}: {str(e)}")

    async def preload_models(self, models: List[str]):
        """Preload models with staggered starts to avoid resource competition"""

        self.logger.info(
            f"Preloading {len(models)} models with {self.system_config['stagger_delay']}s delay"
        )

        for i, model in enumerate(models):
            if i > 0:
                await asyncio.sleep(self.system_config["stagger_delay"])

            try:
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "model": model,
                        "prompt": "Hello",
                        "options": {"num_predict": 1},
                    }

                    async with session.post(
                        f"{self.host}/api/generate", json=payload, timeout=30
                    ) as response:
                        if response.status == 200:
                            self.logger.info(f"Preloaded {model}")
                        else:
                            self.logger.warning(f"Failed to preload {model}")

            except Exception as e:
                self.logger.error(f"Error preloading {model}: {str(e)}")

    def get_optimized_timeout(
        self, model: str, base_timeout: Optional[int] = None
    ) -> int:
        """Get optimized timeout for a model based on its performance history"""

        if base_timeout is None:
            base_timeout = self.system_config["timeout_base"]

        # Check model's historical performance
        evaluation = self.tracker.evaluate_model_performance(model)

        if evaluation["status"] == "slow":
            # Give slow models more time
            return int(base_timeout * 1.5)
        elif (
            evaluation["status"] == "new" or evaluation["status"] == "insufficient_data"
        ):
            # Give new models standard time
            return base_timeout
        else:
            # Well-performing models can use standard timeout
            return base_timeout
