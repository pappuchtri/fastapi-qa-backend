import os
import aiohttp
import asyncio
from typing import List, Dict, Optional, Any
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSearchService:
    """Service for performing web searches using various APIs"""
    
    def __init__(self):
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cx = os.getenv("GOOGLE_CX")
        self.bing_api_key = os.getenv("BING_API_KEY")
        
        # Determine which search service to use
        if self.serpapi_key:
            self.search_provider = "serpapi"
            logger.info("🌐 Using SerpAPI for web search")
        elif self.google_api_key and self.google_cx:
            self.search_provider = "google"
            logger.info("🌐 Using Google Custom Search API")
        elif self.bing_api_key:
            self.search_provider = "bing"
            logger.info("🌐 Using Bing Search API")
        else:
            self.search_provider = "demo"
            logger.info("🎭 Using demo mode for web search")
    
    async def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Perform web search using the configured provider"""
        try:
            if self.search_provider == "serpapi":
                return await self._search_serpapi(query, num_results)
            elif self.search_provider == "google":
                return await self._search_google(query, num_results)
            elif self.search_provider == "bing":
                return await self._search_bing(query, num_results)
            else:
                return await self._search_demo(query, num_results)
        except Exception as e:
            logger.error(f"❌ Error in web search: {str(e)}")
            return await self._search_demo(query, num_results)
    
    async def _search_serpapi(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using SerpAPI"""
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self.serpapi_key,
            "engine": "google",
            "num": num_results,
            "format": "json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for i, result in enumerate(data.get("organic_results", [])[:num_results]):
                        results.append({
                            "title": result.get("title", ""),
                            "snippet": result.get("snippet", ""),
                            "url": result.get("link", ""),
                            "position": i + 1,
                            "source": "serpapi"
                        })
                    
                    return results
                else:
                    logger.error(f"SerpAPI error: {response.status}")
                    return await self._search_demo(query, num_results)
    
    async def _search_google(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API"""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": self.google_api_key,
            "cx": self.google_cx,
            "num": min(num_results, 10)  # Google API max is 10
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for i, item in enumerate(data.get("items", [])[:num_results]):
                        results.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "url": item.get("link", ""),
                            "position": i + 1,
                            "source": "google"
                        })
                    
                    return results
                else:
                    logger.error(f"Google API error: {response.status}")
                    return await self._search_demo(query, num_results)
    
    async def _search_bing(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using Bing Search API"""
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {
            "Ocp-Apim-Subscription-Key": self.bing_api_key
        }
        params = {
            "q": query,
            "count": num_results,
            "responseFilter": "Webpages"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for i, webpage in enumerate(data.get("webPages", {}).get("value", [])[:num_results]):
                        results.append({
                            "title": webpage.get("name", ""),
                            "snippet": webpage.get("snippet", ""),
                            "url": webpage.get("url", ""),
                            "position": i + 1,
                            "source": "bing"
                        })
                    
                    return results
                else:
                    logger.error(f"Bing API error: {response.status}")
                    return await self._search_demo(query, num_results)
    
    async def _search_demo(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Demo search results when no API is configured"""
        demo_results = [
            {
                "title": f"Demo Result 1 for '{query}'",
                "snippet": f"This is a demo search result for the query '{query}'. In a real implementation, this would contain actual web search results.",
                "url": "https://example.com/demo-result-1",
                "position": 1,
                "source": "demo"
            },
            {
                "title": f"Demo Result 2 for '{query}'",
                "snippet": f"Another demo search result showing how web search integration would work with the query '{query}'.",
                "url": "https://example.com/demo-result-2",
                "position": 2,
                "source": "demo"
            }
        ]
        
        return demo_results[:num_results]
    
    async def generate_answer_from_web_results(
        self, 
        question: str, 
        search_results: List[Dict[str, Any]], 
        rag_service
    ) -> Dict[str, Any]:
        """Generate an answer from web search results using the RAG service"""
        try:
            if not search_results:
                return {
                    "success": False,
                    "answer": "No web search results found.",
                    "sources": []
                }
            
            # Combine search results into context
            context_parts = []
            sources = []
            
            for result in search_results[:3]:  # Use top 3 results
                context_parts.append(f"Title: {result['title']}\nContent: {result['snippet']}")
                sources.append({
                    "title": result['title'],
                    "url": result['url'],
                    "snippet": result['snippet'][:100] + "..." if len(result['snippet']) > 100 else result['snippet']
                })
            
            context = "\n\n".join(context_parts)
            
            # Generate answer using RAG service
            prompt = f"""Based on the following web search results, please provide a comprehensive answer to the question: "{question}"

Web Search Results:
{context}

Please provide a clear, accurate answer based on the information found. If the search results don't contain enough information to answer the question completely, please indicate that."""
            
            answer = await rag_service.generate_answer(prompt)
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating answer from web results: {str(e)}")
            return {
                "success": False,
                "answer": f"Error processing web search results: {str(e)}",
                "sources": []
            }

# Global instance
web_search = WebSearchService()
