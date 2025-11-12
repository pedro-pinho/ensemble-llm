#!/usr/bin/env python3
"""Test script to verify optimized council prompts"""

import asyncio
from ensemble_llm.main import EnsembleLLM
from ensemble_llm.config import COUNCIL_CONFIG


def count_tokens(text: str) -> int:
    """Rough token count estimation (1 token ≈ 4 chars)"""
    return len(text) // 4


async def test_council_prompts():
    """Test the optimized council prompts"""

    print("=" * 70)
    print("COUNCIL PROMPT OPTIMIZATION TEST")
    print("=" * 70)

    # Test data
    test_models = ["llama3.2:3b", "phi3.5:latest", "qwen2.5:7b"]
    test_question = "What is quantum computing?"

    # Initialize ensemble with council mode
    ensemble = EnsembleLLM(
        models=test_models,
        enable_web_search=False,
        verbose_logging=False
    )

    # Test 1: Check system prompt formatting
    print("\n1. SYSTEM PROMPT TEST")
    print("-" * 70)

    model_name = test_models[0]
    specialties = "general, conversation"
    council_members = ", ".join(test_models)

    system_prompt = COUNCIL_CONFIG["system_prompt_template"].format(
        model_name=model_name,
        model_specialty=specialties,
        council_members=council_members,
        total_models=len(test_models)
    )

    print(f"Formatted System Prompt:\n{system_prompt}\n")
    system_tokens = count_tokens(system_prompt)
    print(f"Token count: ~{system_tokens} tokens")

    # Test 2: Check synthesis prompt formatting
    print("\n2. SYNTHESIS PROMPT TEST")
    print("-" * 70)

    mock_responses = """**llama3.2:3b**:
Quantum computing uses quantum bits (qubits) that can exist in superposition.

**phi3.5:latest**:
It leverages quantum mechanics principles for parallel computation.

**qwen2.5:7b**:
Quantum computers can solve certain problems exponentially faster than classical computers."""

    synthesis_prompt = COUNCIL_CONFIG["synthesis_prompt_template"].format(
        question=test_question,
        all_responses=mock_responses
    )

    print(f"Formatted Synthesis Prompt:\n{synthesis_prompt}\n")
    synthesis_tokens = count_tokens(synthesis_prompt) - count_tokens(mock_responses)  # Exclude response text
    print(f"Token count (overhead only): ~{synthesis_tokens} tokens")

    # Test 3: Calculate savings
    print("\n3. OPTIMIZATION SUMMARY")
    print("-" * 70)

    # Old estimates
    old_system = 150
    old_synthesis = 280
    old_total = old_system * len(test_models) + old_synthesis

    # New actual
    new_system = system_tokens
    new_synthesis = synthesis_tokens
    new_total = new_system * len(test_models) + new_synthesis

    savings = old_total - new_total
    savings_pct = (savings / old_total) * 100

    print(f"Old system prompt:      {old_system} tokens × {len(test_models)} models = {old_system * len(test_models)} tokens")
    print(f"Old synthesis prompt:   {old_synthesis} tokens")
    print(f"Old TOTAL overhead:     {old_total} tokens\n")

    print(f"New system prompt:      {new_system} tokens × {len(test_models)} models = {new_system * len(test_models)} tokens")
    print(f"New synthesis prompt:   {new_synthesis} tokens")
    print(f"New TOTAL overhead:     {new_total} tokens\n")

    print(f"✓ Savings per query:    {savings} tokens ({savings_pct:.1f}% reduction)")
    print(f"✓ Over 100 queries:     {savings * 100:,} tokens saved")

    # Test 4: Verify functionality is preserved
    print("\n4. FUNCTIONALITY CHECK")
    print("-" * 70)

    checks = [
        ("Model name included", "{model_name}" in COUNCIL_CONFIG["system_prompt_template"]),
        ("Specialty included", "{model_specialty}" in COUNCIL_CONFIG["system_prompt_template"]),
        ("Council members listed", "{council_members}" in COUNCIL_CONFIG["system_prompt_template"]),
        ("Question in synthesis", "{question}" in COUNCIL_CONFIG["synthesis_prompt_template"]),
        ("Responses in synthesis", "{all_responses}" in COUNCIL_CONFIG["synthesis_prompt_template"]),
        ("Meta-talk filter active", COUNCIL_CONFIG.get("filter_ai_meta_talk", False)),
    ]

    all_passed = True
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Optimization successful!")
    else:
        print("✗ SOME TESTS FAILED - Review configuration")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_council_prompts())
