"""Web search module for Ensemble LLM"""

import aiohttp
import re
import logging
from typing import List, Dict, Optional
from urllib.parse import quote

class WebSearcher:
    """Web search capability using DuckDuckGo (no API key required)"""
    
    def __init__(self):
        self.session = None
        self.logger = logging.getLogger('EnsembleLLM.WebSearch')
        
    async def search_duckduckgo(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search DuckDuckGo and return results"""
        self.logger.info(f"Searching web for: {query}")
        
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            async with self.session.get(search_url, headers=headers, timeout=10) as response:
                html = await response.text()
                
                results = []
                result_pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)">([^<]+)</a>.*?<a class="result__snippet" href="[^"]+">([^<]+)</a>'
                matches = re.findall(result_pattern, html, re.DOTALL)
                
                for match in matches[:max_results]:
                    url, title, snippet = match
                    results.append({
                        'url': url,
                        'title': title.strip(),
                        'snippet': snippet.strip()
                    })
                
                self.logger.info(f"Found {len(results)} search results")
                return results
                
        except Exception as e:
            self.logger.error(f"Web search failed: {str(e)}")
            return []
    
    async def search_with_fallback(self, query: str) -> str:
        """Search with multiple fallback options"""
        
        results = await self.search_duckduckgo(query)
        
        if results:
            context = "Web search results:\n\n"
            for i, result in enumerate(results, 1):
                context += f"{i}. {result['title']}\n"
                context += f"   {result['snippet']}\n"
                context += f"   Source: {result['url']}\n\n"
            return context
        
        # Try simplified query if no results
        if "current" in query.lower() or "latest" in query.lower():
            simplified_query = re.sub(r'\b(current|latest|today|2024|2025)\b', '', query, flags=re.IGNORECASE)
            self.logger.info(f"Retrying with simplified query: {simplified_query}")
            results = await self.search_duckduckgo(simplified_query)
            
            if results:
                context = "Web search results (simplified query):\n\n"
                for i, result in enumerate(results, 1):
                    context += f"{i}. {result['title']}\n"
                    context += f"   {result['snippet']}\n\n"
                return context
        
        return "No web search results found."
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()