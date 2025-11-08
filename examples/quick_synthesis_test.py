#!/usr/bin/env python3
"""
Quick Synthesis Test - See the difference immediately!

This script lets you test synthesis mode without editing config files.
Just run it and see council + synthesis in action.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def quick_test():
    """Quick demonstration - no config editing needed"""

    from ensemble_llm.main import EnsembleLLM
    from ensemble_llm import config

    # Enable council + synthesis programmatically
    config.COUNCIL_CONFIG["enabled"] = True
    config.COUNCIL_CONFIG["synthesis_mode"] = True

    print("=" * 70)
    print("QUICK SYNTHESIS TEST")
    print("=" * 70)
    print("\nCouncil Mode: ENABLED")
    print("Synthesis Mode: ENABLED")
    print("\nThis will:")
    print("1. Have models discuss as a council")
    print("2. Vote to select the best model")
    print("3. Ask winner to synthesize all responses")
    print("=" * 70)

    # Initialize
    print("\nInitializing ensemble...")
    ensemble = EnsembleLLM(
        models=["llama3.2:3b", "phi3.5:latest"],  # Using 2 models for speed
        enable_web_search=False,
        verbose_logging=False,
    )

    await ensemble.async_init()
    print("✓ Ensemble ready\n")

    # Test question
    question = "What are the main benefits of using Docker containers?"

    print(f"Question: {question}\n")
    print("-" * 70)

    # Query with verbose to see synthesis
    response, metadata = await ensemble.ensemble_query(question, verbose=True)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"\nVoting Winner: {metadata.get('selected_model', 'N/A')}")
    print(f"Spokesperson: {metadata.get('synthesis_model', 'N/A')}")
    print(f"Synthesized: {metadata.get('synthesized', False)}")
    print(
        f"Time: {metadata.get('total_ensemble_time', 0):.2f}s"
    )

    print("\n" + "-" * 70)
    print("FINAL SYNTHESIZED ANSWER:")
    print("-" * 70)
    print(response)
    print("-" * 70)

    await ensemble.cleanup()

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print("\nThe answer above was synthesized from multiple model perspectives.")
    print("Notice: No meta-discussion about councils or voting!")
    print("\nTo enable permanently, edit ensemble_llm/config.py:")
    print("  COUNCIL_CONFIG['enabled'] = True")
    print("  COUNCIL_CONFIG['synthesis_mode'] = True")
    print("=" * 70)


if __name__ == "__main__":
    print("\nStarting quick synthesis test...\n")

    try:
        asyncio.run(quick_test())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        print("\nMake sure:")
        print("1. Ollama is running: ollama serve")
        print("2. Models are pulled: ollama pull llama3.2:3b")
        import traceback

        traceback.print_exc()
