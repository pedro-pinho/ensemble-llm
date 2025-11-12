#!/usr/bin/env python3
"""Test script to verify optimized web search context"""

import asyncio
from ensemble_llm.web_search import WebSearcher
from ensemble_llm.config import WEB_SEARCH_CONFIG


def count_tokens(text: str) -> int:
    """Rough token count estimation (1 token ≈ 4 chars)"""
    return len(text) // 4


async def test_web_search_optimization():
    """Test the optimized web search context"""

    print("=" * 70)
    print("WEB SEARCH CONTEXT OPTIMIZATION TEST")
    print("=" * 70)

    # Initialize web searcher
    searcher = WebSearcher()

    # Test queries
    test_queries = [
        "Python asyncio tutorial",
        "latest AI news 2025",
        "what is quantum computing"
    ]

    # Test 1: Relevance scoring
    print("\n1. RELEVANCE SCORING TEST")
    print("-" * 70)

    test_snippet = "Python asyncio is a library to write concurrent code using async/await syntax. It is used for asynchronous programming."
    test_query = "Python asyncio tutorial"

    relevance = searcher.calculate_relevance(test_query, test_snippet)
    print(f"Query: {test_query}")
    print(f"Snippet: {test_snippet}")
    print(f"Relevance score: {relevance:.2f}\n")

    # Test 2: Sentence extraction
    print("2. SENTENCE EXTRACTION TEST")
    print("-" * 70)

    long_snippet = """
    Python asyncio is a library to write concurrent code using the async/await syntax.
    It is used as a foundation for multiple Python asynchronous frameworks.
    Asyncio provides high-level APIs for network connections and subprocesses.
    The library has been part of Python standard library since version 3.4.
    Many developers find it challenging to learn at first.
    """

    extracted = searcher.extract_relevant_sentences(long_snippet.strip(), test_query, max_sentences=2)
    print(f"Original snippet: {len(long_snippet)} chars (~{count_tokens(long_snippet)} tokens)")
    print(f"Extracted: {len(extracted)} chars (~{count_tokens(extracted)} tokens)")
    print(f"\nExtracted text:\n{extracted}\n")

    # Test 3: Mock search results formatting
    print("3. CONTEXT FORMATTING TEST")
    print("-" * 70)

    mock_results = [
        {
            "title": "Python Asyncio Tutorial - Real Python",
            "snippet": "Learn how to use Python's asyncio library for concurrent programming. This comprehensive guide covers async/await syntax, event loops, and coroutines. Perfect for beginners and advanced users.",
            "url": "https://realpython.com/python-asyncio-tutorial"
        },
        {
            "title": "Asyncio Documentation",
            "snippet": "Official Python documentation for asyncio module. Includes API reference and examples for asynchronous I/O operations.",
            "url": "https://docs.python.org/3/library/asyncio.html"
        },
        {
            "title": "What is Python?",
            "snippet": "Python is a high-level programming language. It supports multiple paradigms including object-oriented programming.",
            "url": "https://python.org/about"
        }
    ]

    # Format with old method (simulate)
    old_context = "Web search results:\n\n"
    for i, result in enumerate(mock_results, 1):
        old_context += f"{i}. {result['title']}\n"
        old_context += f"   {result['snippet']}\n"
        old_context += f"   Source: {result['url']}\n\n"

    old_wrapper = f"""Context from web search:
{old_context}

Based on the above context and your knowledge, please answer the following question.
If the web search results don't contain relevant information, use your general knowledge.

Question: {test_query}"""

    # Format with new method
    new_context = searcher._format_search_context(mock_results, test_query)
    new_wrapper = f"""[Context] {new_context}

Q: {test_query}"""

    print("OLD FORMAT:")
    print("-" * 70)
    print(old_wrapper[:500] + "..." if len(old_wrapper) > 500 else old_wrapper)
    print(f"\nTotal tokens: ~{count_tokens(old_wrapper)}\n")

    print("\nNEW FORMAT:")
    print("-" * 70)
    print(new_wrapper)
    print(f"\nTotal tokens: ~{count_tokens(new_wrapper)}\n")

    # Calculate savings
    old_tokens = count_tokens(old_wrapper)
    new_tokens = count_tokens(new_wrapper)
    savings = old_tokens - new_tokens
    savings_pct = (savings / old_tokens) * 100

    print("\n4. OPTIMIZATION SUMMARY")
    print("-" * 70)
    print(f"Old format: {old_tokens} tokens")
    print(f"New format: {new_tokens} tokens")
    print(f"✓ Savings: {savings} tokens ({savings_pct:.1f}% reduction)")
    print(f"✓ Over 100 web searches: {savings * 100:,} tokens saved\n")

    # Test 5: Configuration check
    print("5. CONFIGURATION CHECK")
    print("-" * 70)

    config_checks = [
        ("Optimization enabled", WEB_SEARCH_CONFIG.get("optimize_context", False)),
        ("Max sentences", WEB_SEARCH_CONFIG.get("max_snippet_sentences", 3)),
        ("Min relevance", WEB_SEARCH_CONFIG.get("min_relevance_score", 0.0)),
        ("URLs excluded", not WEB_SEARCH_CONFIG.get("include_urls", True)),
        ("Max context tokens", WEB_SEARCH_CONFIG.get("max_context_tokens", 0)),
    ]

    for check_name, value in config_checks:
        print(f"• {check_name}: {value}")

    print("\n" + "=" * 70)
    print("✓ WEB SEARCH OPTIMIZATION COMPLETE")
    print("=" * 70)

    # Cleanup
    await searcher.close()


if __name__ == "__main__":
    asyncio.run(test_web_search_optimization())
