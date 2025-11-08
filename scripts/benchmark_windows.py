#!/usr/bin/env python3
"""Benchmark script for Windows GPU testing"""

import asyncio
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble_llm.platform_utils import PlatformUtils, WindowsOptimizer


async def benchmark_gpu():
    """Benchmark GPU performance with multiple models"""

    # Get system info
    system_info = PlatformUtils.get_system_info()

    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Platform: {system_info['platform']}")
    print(f"CPU Cores: {system_info['cpu_cores']}")
    print(
        f"RAM: {system_info['ram_total_gb']:.1f}GB (Available: {system_info['ram_available_gb']:.1f}GB)"
    )

    gpu = system_info["gpu"]
    if gpu["available"]:
        print(f"GPU: {gpu['name']}")
        print(f"GPU Memory: {gpu['memory_total']}MB (Free: {gpu['memory_free']}MB)")
        print(f"CUDA Available: {gpu['cuda_available']}")
    else:
        print("GPU: Not detected")

    print("\n" + "=" * 60)
    print("GPU OPTIMIZATION")
    print("=" * 60)

    if system_info["platform"] == "windows":
        # Optimize for Windows GPU
        env_vars = WindowsOptimizer.optimize_ollama_for_gpu()
        print("Set environment variables:")
        for key, value in env_vars.items():
            print(f"  {key}={value}")

        # Set Ollama to high priority
        WindowsOptimizer.set_process_priority("ollama", "HIGH")
        print("Set Ollama process priority to HIGH")

    # Test different model configurations
    print("\n" + "=" * 60)
    print("TESTING MODEL CONFIGURATIONS")
    print("=" * 60)

    test_configs = [
        {"models": ["tinyllama:1b"], "name": "Single Tiny Model"},
        {"models": ["llama3.2:3b", "gemma2:2b"], "name": "2 Small Models"},
        {
            "models": ["llama3.2:3b", "phi3.5:latest", "gemma2:2b"],
            "name": "3 Small Models",
        },
        {"models": ["qwen2.5:7b", "mistral:7b"], "name": "2 Medium Models"},
        {
            "models": ["llama3.2:3b", "phi3.5:latest", "qwen2.5:7b", "mistral:7b"],
            "name": "4 Mixed Models",
        },
    ]

    if gpu["memory_total"] > 8000:  # If more than 8GB VRAM
        test_configs.append(
            {
                "models": ["llama3.1:13b", "mixtral:8x7b-instruct-q3_K_M"],
                "name": "2 Large Models (GPU Required)",
            }
        )

    for config in test_configs:
        print(f"\nTesting: {config['name']}")
        print(f"Models: {', '.join(config['models'])}")

        # Import and test
        from ensemble_llm.main import EnsembleLLM

        ensemble = EnsembleLLM(models=config["models"], speed_mode="fast")

        try:
            await ensemble.initialize()

            # Test query
            start_time = time.time()
            response, metadata = await ensemble.ensemble_query(
                "What is 2+2?", verbose=False
            )
            elapsed = time.time() - start_time

            print(f"  Success in {elapsed:.2f}s")
            print(f"  Selected: {metadata.get('selected_model')}")

        except Exception as e:
            print(f"  [X] Failed: {str(e)}")

        finally:
            await ensemble.cleanup()

        # Check memory usage
        memory = PlatformUtils.get_memory_usage("ollama")
        if memory:
            print(f"  Memory Usage: {memory:.0f}MB")

        await asyncio.sleep(2)  # Cool down between tests


if __name__ == "__main__":
    print("Starting GPU Benchmark for Windows...")
    asyncio.run(benchmark_gpu())
