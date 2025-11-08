#!/usr/bin/env python3
"""
Demonstration of Council Mode - where models are aware they're part of an ensemble
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble_llm.main import EnsembleLLM


async def demo_simple_council():
    """Demonstrate simple council mode where models know they're part of a team"""

    print("=" * 70)
    print("COUNCIL MODE DEMO - Simple Mode")
    print("=" * 70)
    print("\nIn this mode, each model knows:")
    print("1. They are part of a council of AI models")
    print("2. Their specific role/specialty")
    print("3. The names of other council members")
    print("4. That the final answer will be selected by consensus\n")

    # To enable council mode, edit config.py and set COUNCIL_CONFIG["enabled"] = True
    # Or temporarily enable it programmatically:
    from ensemble_llm import config
    config.COUNCIL_CONFIG["enabled"] = True

    # Initialize ensemble
    ensemble = EnsembleLLM(
        models=["llama3.2:3b", "phi3.5:latest", "qwen2.5:7b"],
        enable_web_search=False,
        verbose_logging=True,
    )

    await ensemble.async_init()

    # Ask a question
    question = "What is the best way to learn Python programming?"

    print(f"Question: {question}\n")
    print("Querying council...\n")

    response, metadata = await ensemble.ensemble_query(question, verbose=True)

    print("\n" + "=" * 70)
    print("FINAL CONSENSUS ANSWER")
    print("=" * 70)
    print(f"\nSelected Model: {metadata['selected_model']}")
    print(f"Consensus Score: {metadata.get('consensus_score', 'N/A'):.3f}")
    print(f"\nAnswer:\n{response}")

    await ensemble.cleanup()


async def demo_comparison():
    """Compare responses with and without council awareness"""

    print("\n\n" + "=" * 70)
    print("COMPARISON: Council Mode vs Standard Mode")
    print("=" * 70)

    question = "Explain quantum computing in simple terms"

    # Test without council mode
    print("\n--- WITHOUT Council Awareness ---")
    from ensemble_llm import config
    config.COUNCIL_CONFIG["enabled"] = False

    ensemble1 = EnsembleLLM(
        models=["llama3.2:3b", "phi3.5:latest"],
        verbose_logging=False,
    )
    await ensemble1.async_init()
    response1, meta1 = await ensemble1.ensemble_query(question, verbose=False)
    print(f"\nStandard Response ({meta1['selected_model']}):")
    print(response1[:300] + "..." if len(response1) > 300 else response1)
    await ensemble1.cleanup()

    # Test with council mode
    print("\n--- WITH Council Awareness ---")
    config.COUNCIL_CONFIG["enabled"] = True

    ensemble2 = EnsembleLLM(
        models=["llama3.2:3b", "phi3.5:latest"],
        verbose_logging=False,
    )
    await ensemble2.async_init()
    response2, meta2 = await ensemble2.ensemble_query(question, verbose=False)
    print(f"\nCouncil Response ({meta2['selected_model']}):")
    print(response2[:300] + "..." if len(response2) > 300 else response2)
    await ensemble2.cleanup()

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\nEnsemble LLM - Council Mode Demonstration\n")

    # Run demos
    asyncio.run(demo_simple_council())
    asyncio.run(demo_comparison())

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("\nTo enable council mode permanently:")
    print("1. Edit ensemble_llm/config.py")
    print("2. Find COUNCIL_CONFIG")
    print('3. Set "enabled": True')
    print("\nYou can customize the system prompt template in the same section.")
    print("=" * 70 + "\n")
