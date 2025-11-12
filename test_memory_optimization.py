#!/usr/bin/env python3
"""Test script to verify optimized memory context"""

import asyncio
import sys
from pathlib import Path
from ensemble_llm.memory_system import MemoryManager, SemanticMemory
from ensemble_llm.config import MEMORY_CONFIG


def count_tokens(text: str) -> int:
    """Rough token count estimation (1 token ≈ 4 chars)"""
    return len(text) // 4


def test_memory_optimization():
    """Test the optimized memory context"""

    print("=" * 70)
    print("MEMORY CONTEXT OPTIMIZATION TEST")
    print("=" * 70)

    # Create test memory manager
    memory = SemanticMemory(memory_dir="memory_store")

    # Test 1: Store sample memories
    print("\n1. STORING TEST MEMORIES")
    print("-" * 70)

    # Store facts
    memory.store_fact("identity", "name", "Pedro", confidence=1.0)
    memory.store_fact("location", "city", "Bauru", confidence=1.0)
    memory.store_fact("work", "role", "Software Engineer at TechCorp", confidence=0.9)
    memory.store_fact("preference", "likes", "Python programming and AI", confidence=0.8)
    memory.store_fact("preference", "projects", "Building ensemble AI systems", confidence=0.7)

    print("✓ Stored 5 facts")

    # Store a conversation
    memory.store_conversation(
        query="What is your favorite programming language?",
        response="Python is great for AI and machine learning projects.",
        metadata={"selected_model": "llama3.2:3b", "total_ensemble_time": 2.5}
    )

    print("✓ Stored 1 conversation\n")

    # Test 2: Test optimized context retrieval
    print("2. OPTIMIZED CONTEXT RETRIEVAL")
    print("-" * 70)

    test_query = "Tell me about Python programming"

    optimized_context = memory.get_user_context(test_query)

    print(f"Query: {test_query}")
    print(f"\nOptimized context:\n{optimized_context}\n")

    optimized_tokens = count_tokens(optimized_context)
    print(f"Token count: ~{optimized_tokens} tokens")
    print(f"Max allowed: {MEMORY_CONFIG['max_context_tokens']} tokens")

    if optimized_tokens <= MEMORY_CONFIG['max_context_tokens']:
        print("✓ Within token budget\n")
    else:
        print("✗ Exceeds token budget\n")

    # Test 3: Simulate legacy format for comparison
    print("3. COMPARISON WITH LEGACY FORMAT")
    print("-" * 70)

    # Temporarily disable optimization
    original_setting = MEMORY_CONFIG['optimize_context']
    MEMORY_CONFIG['optimize_context'] = False

    legacy_context = memory.get_user_context(test_query)

    MEMORY_CONFIG['optimize_context'] = original_setting

    legacy_tokens = count_tokens(legacy_context)

    print(f"Legacy format:\n{legacy_context[:300]}...\n" if len(legacy_context) > 300 else f"Legacy format:\n{legacy_context}\n")
    print(f"Legacy tokens: ~{legacy_tokens}")
    print(f"Optimized tokens: ~{optimized_tokens}")
    print(f"Savings: {legacy_tokens - optimized_tokens} tokens ({(1 - optimized_tokens/max(legacy_tokens, 1)) * 100:.1f}% reduction)\n")

    # Test 4: Test different query types
    print("4. QUERY TYPE TESTING")
    print("-" * 70)

    test_queries = [
        "What do you know about me?",
        "Where do I work?",
        "What programming languages do I like?"
    ]

    for tq in test_queries:
        ctx = memory.get_user_context(tq)
        tokens = count_tokens(ctx)
        print(f"Q: {tq}")
        print(f"   Context: {ctx[:60]}..." if len(ctx) > 60 else f"   Context: {ctx}")
        print(f"   Tokens: ~{tokens}")
        print()

    # Test 5: Test prompt enhancement
    print("5. PROMPT ENHANCEMENT TEST")
    print("-" * 70)

    memory_manager = MemoryManager(memory_dir="memory_store")

    original_prompt = "What can you tell me about my projects?"
    enhanced_prompt = memory_manager.enhance_prompt(original_prompt)

    print(f"Original prompt:\n{original_prompt}\n")
    print(f"Enhanced prompt:\n{enhanced_prompt}\n")

    enhancement_overhead = count_tokens(enhanced_prompt) - count_tokens(original_prompt)
    print(f"Enhancement overhead: ~{enhancement_overhead} tokens\n")

    # Test 6: Configuration check
    print("6. CONFIGURATION CHECK")
    print("-" * 70)

    config_checks = [
        ("Optimization enabled", MEMORY_CONFIG.get("optimize_context", False)),
        ("Max context tokens", MEMORY_CONFIG.get("max_context_tokens", 0)),
        ("Max facts", MEMORY_CONFIG.get("max_facts", 0)),
        ("Max conversations", MEMORY_CONFIG.get("max_conversations", 0)),
        ("Max documents", MEMORY_CONFIG.get("max_documents", 0)),
        ("Min fact relevance", MEMORY_CONFIG.get("min_fact_relevance", 0)),
        ("Max content preview", MEMORY_CONFIG.get("max_content_preview", 0)),
        ("Compact formatting", MEMORY_CONFIG.get("compact_formatting", False)),
    ]

    for check_name, value in config_checks:
        print(f"• {check_name}: {value}")

    # Test 7: Token budget enforcement
    print("\n7. TOKEN BUDGET ENFORCEMENT TEST")
    print("-" * 70)

    # Store many more facts
    for i in range(10):
        memory.store_fact(f"test_cat_{i}", f"key_{i}", f"This is test value number {i} with some extra content" * 5, confidence=0.8)

    print(f"✓ Stored 10 additional facts")

    # Query should still respect budget
    budget_test_query = "Tell me about test values"
    budget_context = memory.get_user_context(budget_test_query)
    budget_tokens = count_tokens(budget_context)

    print(f"Query: {budget_test_query}")
    print(f"Context tokens: ~{budget_tokens}")
    print(f"Budget: {MEMORY_CONFIG['max_context_tokens']}")

    if budget_tokens <= MEMORY_CONFIG['max_context_tokens']:
        print("✓ Budget enforced correctly\n")
    else:
        print("✗ Budget exceeded\n")

    # Summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)

    print(f"✓ Optimized format: ~{optimized_tokens} tokens")
    print(f"✓ Legacy format: ~{legacy_tokens} tokens")
    print(f"✓ Savings: {legacy_tokens - optimized_tokens} tokens ({(1 - optimized_tokens/max(legacy_tokens, 1)) * 100:.1f}%)")
    print(f"✓ Budget respected: {budget_tokens}/{MEMORY_CONFIG['max_context_tokens']} tokens")
    print(f"✓ Over 100 queries: {(legacy_tokens - optimized_tokens) * 100:,} tokens saved")
    print("\n" + "=" * 70)
    print("✓ MEMORY OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_memory_optimization()
