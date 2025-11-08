#!/usr/bin/env python3
"""
Improved Synthesis Demo - Shows fixes for AI meta-talk and clarity

This demonstrates the improvements:
1. Models clearly understand THEY are AIs, USER is human
2. AI meta-talk is filtered from final output
3. Synthesis produces clean, direct answers
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def demo_clear_roles():
    """Show that models now understand role distinctions"""

    from ensemble_llm.main import EnsembleLLM
    from ensemble_llm import config

    print("=" * 70)
    print("IMPROVED COUNCIL + SYNTHESIS")
    print("=" * 70)
    print("\nKey improvements:")
    print("1. ✓ Models know: THEY are AIs, USER is human")
    print("2. ✓ Internal council discussion vs external user answer")
    print("3. ✓ AI meta-talk automatically filtered from final output")
    print("4. ✓ Direct, authoritative answers without disclaimers")
    print("=" * 70)

    # Enable with improvements
    config.COUNCIL_CONFIG["enabled"] = True
    config.COUNCIL_CONFIG["synthesis_mode"] = True
    config.COUNCIL_CONFIG["filter_ai_meta_talk"] = True

    ensemble = EnsembleLLM(
        models=["llama3.2:3b", "phi3.5:latest"],
        verbose_logging=False,
    )

    await ensemble.async_init()

    # Test questions that might trigger AI meta-talk
    test_questions = [
        "What is machine learning?",
        "How do I learn Python programming?",
        "What are the benefits of cloud computing?",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'-'*70}")
        print(f"Test {i}: {question}")
        print(f"{'-'*70}")

        response, metadata = await ensemble.ensemble_query(question, verbose=False)

        print(f"\nSpokesperson: {metadata.get('synthesis_model', 'N/A')}")
        print(f"Filtered: {metadata.get('synthesized', False)}")

        print(f"\nFinal Answer:")
        print(response)

        # Check for unwanted patterns
        unwanted_patterns = [
            "as an ai",
            "i don't have access",
            "as a language model",
            "the council",
            "i cannot",
            "my training",
        ]

        found_issues = [p for p in unwanted_patterns if p.lower() in response.lower()]

        if found_issues:
            print(f"\n⚠️  WARNING: Found meta-talk: {', '.join(found_issues)}")
        else:
            print("\n✓ Clean response - no AI meta-talk detected")

    await ensemble.cleanup()

    print("\n" + "=" * 70)


async def show_prompt_structure():
    """Show what prompts look like with new structure"""

    from ensemble_llm.main import EnsembleLLM
    from ensemble_llm import config

    print("\n\n" + "=" * 70)
    print("PROMPT STRUCTURE DEMONSTRATION")
    print("=" * 70)

    config.COUNCIL_CONFIG["enabled"] = True
    config.COUNCIL_CONFIG["synthesis_mode"] = True

    ensemble = EnsembleLLM(
        models=["llama3.2:3b", "phi3.5:latest"],
        verbose_logging=False,
    )

    await ensemble.async_init()

    # Create a sample council prompt
    test_prompt = "What is Docker?"
    council_prompt = ensemble.create_council_aware_prompt("llama3.2:3b", test_prompt)

    print("\nCOUNCIL PROMPT (what models see internally):")
    print("-" * 70)
    print(council_prompt[:500] + "...")
    print("-" * 70)

    print("\nKey points in the prompt:")
    print("✓ 'INTERNAL SYSTEM MESSAGE' - clearly marked")
    print("✓ 'YOU are an AI model' - explicit identity")
    print("✓ 'The USER is a human' - clear distinction")
    print("✓ 'This message is ONLY for you and other AI models'")
    print("✓ 'The user does NOT see this council process'")

    print("\n" + "=" * 70)

    await ensemble.cleanup()


async def compare_before_after():
    """Compare old style vs new style responses"""

    from ensemble_llm.main import EnsembleLLM
    from ensemble_llm import config

    print("\n\n" + "=" * 70)
    print("BEFORE vs AFTER COMPARISON")
    print("=" * 70)

    question = "Explain what an API is"

    # Simulate "before" by disabling filter
    print("\n--- BEFORE (old style, no filter) ---")
    config.COUNCIL_CONFIG["enabled"] = True
    config.COUNCIL_CONFIG["synthesis_mode"] = True
    config.COUNCIL_CONFIG["filter_ai_meta_talk"] = False  # Disabled

    ensemble1 = EnsembleLLM(
        models=["llama3.2:3b", "phi3.5:latest"],
        verbose_logging=False,
    )
    await ensemble1.async_init()

    response1, _ = await ensemble1.ensemble_query(question, verbose=False)
    print(f"\n{response1[:400]}...")

    await ensemble1.cleanup()

    # Now with improvements
    print("\n--- AFTER (new style, with filter) ---")
    config.COUNCIL_CONFIG["filter_ai_meta_talk"] = True  # Enabled

    ensemble2 = EnsembleLLM(
        models=["llama3.2:3b", "phi3.5:latest"],
        verbose_logging=False,
    )
    await ensemble2.async_init()

    response2, _ = await ensemble2.ensemble_query(question, verbose=False)
    print(f"\n{response2[:400]}...")

    await ensemble2.cleanup()

    print("\n" + "=" * 70)


async def test_filter_patterns():
    """Test the AI meta-talk filter directly"""

    from ensemble_llm.main import EnsembleLLM
    from ensemble_llm import config

    print("\n\n" + "=" * 70)
    print("FILTER DEMONSTRATION")
    print("=" * 70)

    config.COUNCIL_CONFIG["enabled"] = True
    config.COUNCIL_CONFIG["filter_ai_meta_talk"] = True

    ensemble = EnsembleLLM(
        models=["llama3.2:3b"],
        verbose_logging=False,
    )

    # Test texts with AI meta-talk
    test_texts = [
        "As an AI language model, I can tell you that Python is a great language. It has many features.",
        "Docker is a containerization platform. I don't have access to real-time data about its usage. But it's very popular.",
        "The council discussed this topic and reached consensus. Based on my training, APIs are important.",
        "APIs are interfaces that allow software to communicate. As an AI, I cannot access current information, but they're widely used.",
    ]

    print("\nTesting filter on various AI meta-talk patterns:\n")

    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}:")
        print(f"Original: {text}")

        filtered = ensemble.filter_ai_meta_talk(text)
        print(f"Filtered: {filtered}")

        if text != filtered:
            print("✓ Filter applied")
        else:
            print("○ No changes needed")

        print()

    await ensemble.cleanup()

    print("=" * 70)


async def full_workflow_example():
    """Show complete workflow with verbose output"""

    from ensemble_llm.main import EnsembleLLM
    from ensemble_llm import config

    print("\n\n" + "=" * 70)
    print("COMPLETE WORKFLOW (Verbose Mode)")
    print("=" * 70)

    config.COUNCIL_CONFIG["enabled"] = True
    config.COUNCIL_CONFIG["synthesis_mode"] = True
    config.COUNCIL_CONFIG["filter_ai_meta_talk"] = True

    ensemble = EnsembleLLM(
        models=["llama3.2:3b", "phi3.5:latest", "qwen2.5:7b"],
        verbose_logging=False,
    )

    await ensemble.async_init()

    question = "What are the key differences between SQL and NoSQL databases?"

    print(f"\nQuestion: {question}\n")

    # Verbose shows the complete process
    response, metadata = await ensemble.ensemble_query(question, verbose=True)

    print("\n" + "=" * 70)
    print("METADATA")
    print("=" * 70)
    print(f"Voting winner: {metadata.get('selected_model')}")
    print(f"Spokesperson: {metadata.get('synthesis_model')}")
    print(f"Synthesized: {metadata.get('synthesized')}")
    print(f"Consensus score: {metadata.get('consensus_score', 0):.3f}")
    print(f"Total time: {metadata.get('total_ensemble_time', 0):.2f}s")

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(response)
    print("=" * 70)

    await ensemble.cleanup()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("IMPROVED SYNTHESIS MODE DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows the improvements made to address:")
    print("1. Models being confused about who is AI vs human")
    print("2. AI meta-talk appearing in final responses")
    print("=" * 70)

    try:
        # Run all demonstrations
        asyncio.run(demo_clear_roles())
        asyncio.run(show_prompt_structure())
        asyncio.run(test_filter_patterns())
        asyncio.run(compare_before_after())
        asyncio.run(full_workflow_example())

        print("\n" + "=" * 70)
        print("SUMMARY OF IMPROVEMENTS")
        print("=" * 70)
        print("\n1. ROLE CLARITY")
        print("   - Prompts explicitly state: 'YOU are an AI', 'USER is human'")
        print("   - 'INTERNAL SYSTEM MESSAGE' header for council context")
        print("   - Clear separation of internal vs external communication")
        print("\n2. META-TALK FILTERING")
        print("   - Automatic removal of 'as an AI', 'I don't have access', etc.")
        print("   - Sentence-level filtering (removes entire sentence)")
        print("   - Applied to both synthesized and fallback responses")
        print("\n3. SYNTHESIS INSTRUCTIONS")
        print("   - Explicit list of phrases to avoid (❌)")
        print("   - Examples of preferred style (✓)")
        print("   - Emphasis on direct, authoritative answers")
        print("\n4. RESULT")
        print("   - Clean, professional responses")
        print("   - No confusion about AI vs human roles")
        print("   - Direct answers without disclaimers")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
