"""Web search module for Ensemble LLM"""

import aiohttp
import re
import logging
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote
from .config import WEB_SEARCH_CONFIG


class WebSearcher:
    """Web search capability using DuckDuckGo (no API key required)"""

    def __init__(self):
        self.session = None
        self.logger = logging.getLogger("EnsembleLLM.WebSearch")

    def calculate_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score between query and text (0-1)"""
        query_lower = query.lower()
        text_lower = text.lower()

        # Extract query keywords (ignore common words)
        stop_words = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in", "with", "to", "for", "of", "as", "by"}
        query_words = [w for w in re.findall(r'\b\w+\b', query_lower) if w not in stop_words and len(w) > 2]

        if not query_words:
            return 0.5  # Default score if no keywords

        # Count keyword matches
        matches = sum(1 for word in query_words if word in text_lower)
        score = matches / len(query_words)

        # Bonus for exact phrase match
        if query_lower in text_lower:
            score = min(1.0, score + 0.3)

        return score

    def extract_relevant_sentences(self, snippet: str, query: str, max_sentences: int = 2) -> str:
        """Extract the most relevant sentences from a snippet"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', snippet)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]

        if not sentences:
            return snippet[:150]  # Fallback to truncation

        # Score each sentence by relevance
        scored_sentences = []
        for sentence in sentences:
            score = self.calculate_relevance(query, sentence)
            scored_sentences.append((score, sentence))

        # Sort by score and take top N
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s[1] for s in scored_sentences[:max_sentences]]

        return ". ".join(top_sentences) + "."

    async def search_duckduckgo(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search DuckDuckGo and return results"""
        self.logger.info(f"Searching web for: {query}")

        try:
            search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"

            if not self.session:
                self.session = aiohttp.ClientSession()

            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }

            async with self.session.get(
                search_url, headers=headers, timeout=10
            ) as response:
                html = await response.text()

                results = []
                result_pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)">([^<]+)</a>.*?<a class="result__snippet" href="[^"]+">([^<]+)</a>'
                matches = re.findall(result_pattern, html, re.DOTALL)

                for match in matches[:max_results]:
                    url, title, snippet = match
                    results.append(
                        {"url": url, "title": title.strip(), "snippet": snippet.strip()}
                    )

                self.logger.info(f"Found {len(results)} search results")
                return results

        except Exception as e:
            self.logger.error(f"Web search failed: {str(e)}")
            return []

    async def search_with_fallback(self, query: str) -> str:
        """Search with multiple fallback options"""

        results = await self.search_duckduckgo(query)

        if results:
            return self._format_search_context(results, query)

        # Try simplified query if no results
        if "current" in query.lower() or "latest" in query.lower():
            simplified_query = re.sub(
                r"\b(current|latest|today|2024|2025)\b", "", query, flags=re.IGNORECASE
            )
            self.logger.info(f"Retrying with simplified query: {simplified_query}")
            results = await self.search_duckduckgo(simplified_query)

            if results:
                return self._format_search_context(results, query)

        return "No web search results found."

    def _format_search_context(self, results: List[Dict], query: str) -> str:
        """Format search results into optimized context"""

        if not WEB_SEARCH_CONFIG.get("optimize_context", True):
            # Legacy format (verbose)
            context = "Web search results:\n\n"
            for i, result in enumerate(results, 1):
                context += f"{i}. {result['title']}\n"
                context += f"   {result['snippet']}\n"
                if WEB_SEARCH_CONFIG.get("include_urls", False):
                    context += f"   Source: {result['url']}\n"
                context += "\n"
            return context

        # Optimized format
        max_sentences = WEB_SEARCH_CONFIG.get("max_snippet_sentences", 2)
        min_relevance = WEB_SEARCH_CONFIG.get("min_relevance_score", 0.3)
        include_urls = WEB_SEARCH_CONFIG.get("include_urls", False)

        # Score and filter results
        scored_results = []
        for result in results:
            # Calculate relevance for title + snippet
            combined_text = f"{result['title']} {result['snippet']}"
            relevance = self.calculate_relevance(query, combined_text)

            if relevance >= min_relevance:
                # Extract most relevant sentences
                optimized_snippet = self.extract_relevant_sentences(
                    result['snippet'], query, max_sentences
                )
                scored_results.append({
                    "title": result['title'],
                    "snippet": optimized_snippet,
                    "url": result.get('url', ''),
                    "relevance": relevance
                })

        if not scored_results:
            # If filtering removed everything, use top result anyway
            if results:
                optimized_snippet = self.extract_relevant_sentences(
                    results[0]['snippet'], query, max_sentences
                )
                scored_results.append({
                    "title": results[0]['title'],
                    "snippet": optimized_snippet,
                    "url": results[0].get('url', ''),
                    "relevance": 0.5
                })

        # Sort by relevance
        scored_results.sort(key=lambda x: x['relevance'], reverse=True)

        # Build compact context
        context_parts = []
        for result in scored_results[:3]:  # Top 3 max
            # Compact format: "Title: snippet"
            if include_urls:
                context_parts.append(f"• {result['title']}: {result['snippet']} [{result['url']}]")
            else:
                context_parts.append(f"• {result['title']}: {result['snippet']}")

        context = "Web: " + " | ".join(context_parts)

        self.logger.info(f"Optimized context: {len(scored_results)} relevant results, ~{len(context)//4} tokens")

        return context

    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
