#!/usr/bin/env python3
"""
Advanced Council Mode - Iterative debate where models see and respond to each other

This implements a multi-round discussion where:
1. Round 1: All models provide initial responses (knowing they're in a council)
2. Round 2: Models see other responses and can refine their answers
3. Final: Best response selected from refined answers
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble_llm.main import EnsembleLLM
from ensemble_llm.config import COUNCIL_CONFIG


class IterativeCouncil(EnsembleLLM):
    """Extended EnsembleLLM with iterative council discussion capabilities"""

    async def iterative_council_query(
        self, prompt: str, rounds: int = 2, verbose: bool = True
    ) -> Tuple[str, Dict]:
        """
        Conduct multi-round council discussion

        Args:
            prompt: User's question
            rounds: Number of discussion rounds (default 2)
            verbose: Show detailed output

        Returns:
            Final answer and metadata
        """

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"ITERATIVE COUNCIL DISCUSSION - {rounds} Rounds")
            print(f"{'=' * 70}")
            print(f"Council Members: {', '.join(self.models)}")
            print(f"{'=' * 70}\n")

        all_responses = {}

        # Round 1: Initial responses
        if verbose:
            print(f"\n--- ROUND 1: Initial Perspectives ---\n")

        round1_prompt = f"""You are {'{model}'}, part of a council of AI models working together.

Council members: {', '.join(self.models)}

Your role: Provide your best initial analysis of this question from your unique perspective.
Remember, other models will also contribute, and you may refine your answer in the next round.

Question: {prompt}

Your initial response:"""

        # Get responses from all models
        responses_r1 = await self.query_all_models(prompt)

        if verbose:
            for resp in responses_r1:
                if resp["success"]:
                    print(f"\n{resp['model']}:")
                    print(f"{resp['response'][:200]}...")
                    all_responses[resp["model"]] = [resp["response"]]

        # Round 2+: Iterative refinement
        for round_num in range(2, rounds + 1):
            if verbose:
                print(f"\n--- ROUND {round_num}: Council Discussion & Refinement ---\n")

            # Create summary of other models' responses
            other_responses_summary = self._create_response_summary(responses_r1)

            round_n_prompt = f"""You are {'{model}'}, continuing the council discussion.

Question: {prompt}

Other council members have provided these perspectives:

{other_responses_summary}

Now, considering what others have said:
1. What valuable points did they make that you agree with?
2. What perspective or insight can you add that wasn't fully covered?
3. Provide your refined, improved answer that builds on the collective wisdom.

Your refined response:"""

            # Query all models again with context
            responses_rn = []
            for model in self.models:
                model_prompt = round_n_prompt.replace("{model}", model)
                # Remove other models' responses from summary for this model
                filtered_summary = self._create_response_summary(
                    responses_r1, exclude_model=model
                )
                final_prompt = round_n_prompt.replace(
                    other_responses_summary, filtered_summary
                )
                final_prompt = final_prompt.replace("{model}", model)

                # Query this model
                async with self.session_manager.get_session() as session:
                    result = await self.query_model(session, model, final_prompt)
                    if result["success"]:
                        responses_rn.append(result)
                        if model in all_responses:
                            all_responses[model].append(result["response"])
                        else:
                            all_responses[model] = [result["response"]]

                        if verbose:
                            print(f"\n{model} (refined):")
                            print(f"{result['response'][:200]}...")

            responses_r1 = responses_rn  # Update for next round

        # Final selection using weighted voting
        if verbose:
            print(f"\n--- FINAL SELECTION ---\n")

        best_response, metadata = self.weighted_voting(responses_r1)

        metadata["rounds"] = rounds
        metadata["iterative_mode"] = True
        metadata["all_responses_history"] = all_responses

        if verbose:
            print(f"\nSelected: {metadata['selected_model']}")
            print(f"Consensus Score: {metadata.get('consensus_score', 'N/A'):.3f}")
            print(f"\n{'=' * 70}")

        return best_response, metadata

    def _create_response_summary(
        self, responses: List[Dict], exclude_model: str = None
    ) -> str:
        """Create a summary of responses from other models"""

        summary_parts = []

        for resp in responses:
            if not resp["success"]:
                continue

            model = resp["model"]

            # Skip excluded model
            if exclude_model and model == exclude_model:
                continue

            # Truncate long responses
            response_text = resp["response"]
            if len(response_text) > 300:
                response_text = response_text[:300] + "..."

            summary_parts.append(f"- {model}: {response_text}")

        return "\n\n".join(summary_parts) if summary_parts else "No responses yet."


async def demo_iterative_council():
    """Demonstrate iterative council mode"""

    # Enable council mode
    from ensemble_llm import config

    config.COUNCIL_CONFIG["enabled"] = True

    # Initialize
    council = IterativeCouncil(
        models=["llama3.2:3b", "phi3.5:latest", "qwen2.5:7b"],
        enable_web_search=False,
        verbose_logging=False,
    )

    await council.async_init()

    # Ask a complex question that benefits from multiple perspectives
    question = """Should schools teach programming to all students starting in elementary school?
    Consider educational, practical, and equity perspectives."""

    response, metadata = await council.iterative_council_query(
        question, rounds=2, verbose=True
    )

    print("\n" + "=" * 70)
    print("FINAL CONSENSUS ANSWER")
    print("=" * 70)
    print(f"\n{response}\n")

    # Show how responses evolved
    print("=" * 70)
    print("RESPONSE EVOLUTION")
    print("=" * 70)

    for model, responses in metadata["all_responses_history"].items():
        print(f"\n{model}:")
        for i, resp in enumerate(responses, 1):
            print(f"  Round {i}: {resp[:100]}...")

    await council.cleanup()


async def compare_single_vs_iterative():
    """Compare single-round vs multi-round council"""

    from ensemble_llm import config

    config.COUNCIL_CONFIG["enabled"] = True

    question = "What are the most important skills for the future workplace?"

    print("\n" + "=" * 70)
    print("COMPARISON: Single Round vs Iterative Council")
    print("=" * 70)

    # Single round
    print("\n--- Single Round Council ---")
    council1 = IterativeCouncil(
        models=["llama3.2:3b", "phi3.5:latest"], verbose_logging=False
    )
    await council1.async_init()
    resp1, meta1 = await council1.ensemble_query(question, verbose=False)
    print(f"\nAnswer: {resp1[:400]}...")
    await council1.cleanup()

    # Iterative (2 rounds)
    print("\n--- Iterative Council (2 rounds) ---")
    council2 = IterativeCouncil(
        models=["llama3.2:3b", "phi3.5:latest"], verbose_logging=False
    )
    await council2.async_init()
    resp2, meta2 = await council2.iterative_council_query(
        question, rounds=2, verbose=False
    )
    print(f"\nAnswer: {resp2[:400]}...")
    await council2.cleanup()

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\nIterative Council Mode - Advanced Demo\n")

    # Run demos
    asyncio.run(demo_iterative_council())
    # asyncio.run(compare_single_vs_iterative())

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey benefits of iterative mode:")
    print("- Models can build on each other's insights")
    print("- Responses become more comprehensive")
    print("- Weak points in initial responses get addressed")
    print("- More thorough exploration of complex topics")
    print("=" * 70 + "\n")
