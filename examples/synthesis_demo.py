#!/usr/bin/env python3
"""
Council + Synthesis Mode Demo

This demonstrates the complete flow:
1. Council Discussion: Models know they're in a council and provide perspectives
2. Voting: Best model/response is selected by consensus
3. Synthesis: Winning model acts as spokesperson, combining all insights into final answer

The user only sees the synthesized answer - clean and coherent, without meta-discussion.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble_llm.main import EnsembleLLM


async def demo_with_synthesis():
    """Full demonstration of council + synthesis mode"""

    print("=" * 70)
    print("COUNCIL + SYNTHESIS MODE DEMONSTRATION")
    print("=" * 70)
    print("\nFlow:")
    print("1. Council Discussion - Models provide their unique perspectives")
    print("2. Voting - Best model selected by consensus algorithm")
    print("3. Synthesis - Winner combines all insights into final answer")
    print("=" * 70)

    # Enable both council mode and synthesis
    from ensemble_llm import config

    config.COUNCIL_CONFIG["enabled"] = True
    config.COUNCIL_CONFIG["synthesis_mode"] = True

    # Initialize ensemble
    ensemble = EnsembleLLM(
        models=["llama3.2:3b", "phi3.5:latest", "qwen2.5:7b"],
        enable_web_search=False,
        verbose_logging=False,
    )

    await ensemble.async_init()

    # Test question
    question = "What are the key principles of good software architecture?"

    print(f"\nQuestion: {question}\n")
    print("=" * 70)
    print("PHASE 1: Council Discussion")
    print("=" * 70)
    print("Each model provides their perspective (knowing they're in a council)...\n")

    # Query with verbose to see the synthesis process
    response, metadata = await ensemble.ensemble_query(question, verbose=True)

    print("\n" + "=" * 70)
    print("FINAL ANSWER (After Synthesis)")
    print("=" * 70)
    print(f"\nSpokesperson: {metadata.get('synthesis_model', 'N/A')}")
    print(f"Synthesized: {metadata.get('synthesized', False)}")
    print(f"Total Time: {metadata.get('total_ensemble_time', 0):.2f}s")
    print("\n" + "-" * 70)
    print(response)
    print("-" * 70)

    print("\n" + "=" * 70)
    print("VOTING BREAKDOWN")
    print("=" * 70)

    all_scores = metadata.get("all_scores", {})
    for model, scores in all_scores.items():
        is_winner = model == metadata.get("selected_model")
        marker = "👑 WINNER" if is_winner else ""
        print(f"\n{model} {marker}")
        print(f"  Consensus: {scores.get('consensus', 0):.3f}")
        print(f"  Quality:   {scores.get('quality', 0):.3f}")
        print(f"  Final:     {scores.get('final', 0):.3f}")

    await ensemble.cleanup()


async def compare_with_and_without_synthesis():
    """Compare responses with and without synthesis"""

    print("\n\n" + "=" * 70)
    print("COMPARISON: With vs Without Synthesis")
    print("=" * 70)

    question = "Explain the concept of technical debt"

    # Test WITHOUT synthesis
    print("\n" + "-" * 70)
    print("WITHOUT Synthesis (original winning response)")
    print("-" * 70)

    from ensemble_llm import config

    config.COUNCIL_CONFIG["enabled"] = True
    config.COUNCIL_CONFIG["synthesis_mode"] = False  # Disable synthesis

    ensemble1 = EnsembleLLM(
        models=["llama3.2:3b", "phi3.5:latest"],
        verbose_logging=False,
    )
    await ensemble1.async_init()

    response1, meta1 = await ensemble1.ensemble_query(question, verbose=False)

    print(f"\nWinner: {meta1['selected_model']}")
    print(f"Synthesized: {meta1.get('synthesized', False)}")
    print(f"\nAnswer:\n{response1}\n")

    await ensemble1.cleanup()

    # Test WITH synthesis
    print("-" * 70)
    print("WITH Synthesis (combined perspectives)")
    print("-" * 70)

    config.COUNCIL_CONFIG["synthesis_mode"] = True  # Enable synthesis

    ensemble2 = EnsembleLLM(
        models=["llama3.2:3b", "phi3.5:latest"],
        verbose_logging=False,
    )
    await ensemble2.async_init()

    response2, meta2 = await ensemble2.ensemble_query(question, verbose=False)

    print(f"\nSpokesperson: {meta2.get('synthesis_model')}")
    print(f"Synthesized: {meta2.get('synthesized', False)}")
    print(f"\nSynthesized Answer:\n{response2}\n")

    await ensemble2.cleanup()

    print("=" * 70)
    print("Notice how the synthesized answer:")
    print("- Combines insights from multiple models")
    print("- Has no meta-discussion about councils/voting")
    print("- Provides a unified, coherent perspective")
    print("- Is often more comprehensive than single model response")
    print("=" * 70)


async def show_behind_the_scenes():
    """Show what happens behind the scenes with verbose output"""

    print("\n\n" + "=" * 70)
    print("BEHIND THE SCENES: Verbose Mode")
    print("=" * 70)
    print("\nThis shows the complete internal process...")
    print("=" * 70 + "\n")

    from ensemble_llm import config

    config.COUNCIL_CONFIG["enabled"] = True
    config.COUNCIL_CONFIG["synthesis_mode"] = True

    ensemble = EnsembleLLM(
        models=["llama3.2:3b", "phi3.5:latest", "qwen2.5:7b"],
        enable_web_search=False,
        verbose_logging=True,  # Enable verbose logging
    )

    await ensemble.async_init()

    question = "What makes a good API design?"

    print(f"Question: {question}\n")

    # Verbose mode shows all model responses and synthesis
    response, metadata = await ensemble.ensemble_query(question, verbose=True)

    print("\n" + "=" * 70)
    print("Process Summary:")
    print(f"  Models queried: {metadata.get('total_models', 0)}")
    print(f"  Successful: {metadata.get('successful_models', 0)}")
    print(f"  Voting winner: {metadata.get('selected_model', 'N/A')}")
    print(f"  Spokesperson: {metadata.get('synthesis_model', 'N/A')}")
    print(f"  Synthesized: {metadata.get('synthesized', False)}")
    print(f"  Total time: {metadata.get('total_ensemble_time', 0):.2f}s")
    print("=" * 70)

    await ensemble.cleanup()


async def test_different_questions():
    """Test synthesis with different types of questions"""

    from ensemble_llm import config

    config.COUNCIL_CONFIG["enabled"] = True
    config.COUNCIL_CONFIG["synthesis_mode"] = True

    questions = [
        "What is the difference between Docker and virtual machines?",
        "How do you prevent SQL injection attacks?",
        "Explain the CAP theorem in distributed systems",
    ]

    ensemble = EnsembleLLM(
        models=["llama3.2:3b", "phi3.5:latest", "qwen2.5:7b"],
        verbose_logging=False,
    )
    await ensemble.async_init()

    print("\n\n" + "=" * 70)
    print("TESTING MULTIPLE QUESTIONS")
    print("=" * 70)

    for i, question in enumerate(questions, 1):
        print(f"\n{'-'*70}")
        print(f"Question {i}: {question}")
        print(f"{'-'*70}")

        response, metadata = await ensemble.ensemble_query(question, verbose=False)

        print(f"\nSpokesperson: {metadata.get('synthesis_model', 'N/A')}")
        print(f"Consensus Score: {metadata.get('consensus_score', 0):.3f}")
        print(
            f"\nSynthesized Answer:\n{response[:300]}{'...' if len(response) > 300 else ''}\n"
        )

    await ensemble.cleanup()

    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ENSEMBLE LLM - COUNCIL + SYNTHESIS DEMONSTRATION")
    print("=" * 70)

    # Run different demonstrations
    print("\n[1] Full demonstration with synthesis")
    asyncio.run(demo_with_synthesis())

    print("\n[2] Comparison with/without synthesis")
    asyncio.run(compare_with_and_without_synthesis())

    print("\n[3] Behind the scenes (verbose)")
    asyncio.run(show_behind_the_scenes())

    print("\n[4] Multiple questions")
    asyncio.run(test_different_questions())

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nTo enable Council + Synthesis mode permanently:")
    print("1. Edit ensemble_llm/config.py")
    print("2. Find COUNCIL_CONFIG")
    print('3. Set "enabled": True')
    print('4. Set "synthesis_mode": True')
    print("\nBenefits:")
    print("✓ Models contribute unique perspectives")
    print("✓ Best model acts as spokesperson")
    print("✓ Final answer combines all insights")
    print("✓ Clean output without meta-discussion")
    print("✓ More comprehensive and balanced responses")
    print("=" * 70 + "\n")
