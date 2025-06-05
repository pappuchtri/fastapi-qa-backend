import os
import json
import logging
from typing import List, Dict, Any, Optional
import httpx
import asyncio
from urllib.parse import quote_plus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchService:
    """Service for performing web searches and extracting relevant information"""
    
    def __init__(self):
        """Initialize the web search service with API keys"""
        # Try to get API keys from environment variables
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cx = os.getenv("GOOGLE_CX")  # Custom Search Engine ID
        
        # Determine which search provider to use based on available keys
        if self.serpapi_key:
            self.search_provider = "serpapi"
            logger.info("✅ Using SerpAPI for web searches")
        elif self.google_api_key and self.google_cx:
            self.search_provider = "google_cse"
            logger.info("✅ Using Google Custom Search API for web searches")
        else:
            self.search_provider = "demo"
            logger.info("⚠️ No search API keys found. Running in demo mode with limited results.")
    
    async def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web for information related to the query
        Returns a list of search results with title, snippet, and URL
        """
        if self.search_provider == "serpapi":
            return await self._search_with_serpapi(query, num_results)
        elif self.search_provider == "google_cse":
            return await self._search_with_google_cse(query, num_results)
        else:
            # Demo mode with mock results
            return self._get_demo_results(query, num_results)
    
    async def _search_with_serpapi(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Perform web search using SerpAPI"""
        try:
            logger.info(f"🔍 Searching web with SerpAPI: {query[:50]}...")
            
            encoded_query = quote_plus(query)
            url = f"https://serpapi.com/search.json?q={encoded_query}&api_key={self.serpapi_key}&num={num_results}"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                
                if response.status_code != 200:
                    logger.error(f"❌ SerpAPI error: {response.status_code} - {response.text}")
                    return []
                
                data = response.json()
                
                # Extract organic search results
                results = []
                if "organic_results" in data:
                    for result in data["organic_results"][:num_results]:
                        results.append({
                            "title": result.get("title", ""),
                            "snippet": result.get("snippet", ""),
                            "url": result.get("link", ""),
                            "position": result.get("position", 0),
                            "source": "serpapi"
                        })
                
                logger.info(f"✅ Found {len(results)} web results via SerpAPI")
                return results
                
        except Exception as e:
            logger.error(f"❌ Error with SerpAPI search: {str(e)}")
            return []
    
    async def _search_with_google_cse(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Perform web search using Google Custom Search API"""
        try:
            logger.info(f"🔍 Searching web with Google CSE: {query[:50]}...")
            
            encoded_query = quote_plus(query)
            url = f"https://www.googleapis.com/customsearch/v1?q={encoded_query}&key={self.google_api_key}&cx={self.google_cx}&num={num_results}"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                
                if response.status_code != 200:
                    logger.error(f"❌ Google CSE error: {response.status_code} - {response.text}")
                    return []
                
                data = response.json()
                
                # Extract search results
                results = []
                if "items" in data:
                    for item in data["items"]:
                        results.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "url": item.get("link", ""),
                            "position": len(results) + 1,
                            "source": "google_cse"
                        })
                
                logger.info(f"✅ Found {len(results)} web results via Google CSE")
                return results
                
        except Exception as e:
            logger.error(f"❌ Error with Google CSE search: {str(e)}")
            return []
    
    def _get_demo_results(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Generate demo search results when no API keys are available"""
        logger.info(f"🎭 Generating demo web search results for: {query[:50]}...")
        
        # Create some generic results based on the query
        results = []
        
        # Add a Wikipedia-like result
        results.append({
            "title": f"{query} - Wikipedia",
            "snippet": f"Information about {query} from Wikipedia, the free encyclopedia. {query} refers to a concept or entity that has various interpretations and applications...",
            "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            "position": 1,
            "source": "demo"
        })
        
        # Add a few generic informational results
        topics = ["overview", "history", "examples", "applications", "research"]
        for i, topic in enumerate(topics[:num_results-1], 2):
            results.append({
                "title": f"{query} {topic} - Information Resource",
                "snippet": f"Comprehensive {topic} of {query} including detailed explanations, analysis, and references to primary sources...",
                "url": f"https://example.com/{query.replace(' ', '-').lower()}/{topic}",
                "position": i,
                "source": "demo"
            })
        
        logger.info(f"🎭 Generated {len(results)} demo web search results")
        return results
    
    async def extract_content_from_url(self, url: str) -> Optional[str]:
        """
        Extract main content from a URL
        This is a simplified version - in production, you'd want to use a more robust solution
        """
        try:
            logger.info(f"📄 Extracting content from URL: {url}")
            
            # In demo mode, return mock content
            if self.search_provider == "demo":
                return f"This is simulated content extracted from {url}. In a production environment, this would contain the actual text content from the webpage, processed to remove navigation elements, ads, and other non-content sections. The content would be relevant to the search query and provide valuable information to answer the user's question."
            
            # For real extraction, use httpx to get the page content
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0, follow_redirects=True)
                
                if response.status_code != 200:
                    logger.error(f"❌ Failed to fetch URL: {response.status_code}")
                    return None
                
                # This is a very simplified content extraction
                # In production, you'd want to use a library like newspaper3k, trafilatura, or readability
                # to extract the main content and remove boilerplate
                html_content = response.text
                
                # Very basic extraction - in production use a proper HTML parser
                import re
                # Remove script and style elements
                no_script = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL)
                no_style = re.sub(r'<style.*?</style>', '', no_script, flags=re.DOTALL)
                # Remove HTML tags
                text_only = re.sub(r'<[^>]+>', ' ', no_style)
                # Remove extra whitespace
                clean_text = re.sub(r'\s+', ' ', text_only).strip()
                
                # Truncate to a reasonable length
                max_length = 5000
                if len(clean_text) > max_length:
                    clean_text = clean_text[:max_length] + "..."
                
                logger.info(f"✅ Successfully extracted {len(clean_text)} characters from URL")
                return clean_text
                
        except Exception as e:
            logger.error(f"❌ Error extracting content from URL: {str(e)}")
            return None
    
    async def generate_answer_from_web_results(
        self, 
        query: str, 
        search_results: List[Dict[str, Any]],
        openai_service
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive answer based on web search results
        Uses the provided OpenAI service to generate the answer
        """
        try:
            if not search_results:
                logger.warning("⚠️ No web search results to generate answer from")
                return {
                    "answer": f"I couldn't find specific information about '{query}' from web sources. Please try rephrasing your question or ask something else.",
                    "sources": [],
                    "success": False
                }
            
            # Extract content from top results (limit to 3 to avoid rate limits)
            detailed_results = []
            for result in search_results[:3]:
                content = await self.extract_content_from_url(result["url"])
                if content:
                    detailed_results.append({
                        "title": result["title"],
                        "url": result["url"],
                        "content": content[:1000]  # Limit content length
                    })
            
            if not detailed_results:
                logger.warning("⚠️ Could not extract content from any web results")
                return {
                    "answer": f"I found some information about '{query}', but couldn't extract the content. Here are some sources you might want to check: " + 
                              ", ".join([f"{r['title']} ({r['url']})" for r in search_results[:3]]),
                    "sources": search_results[:3],
                    "success": False
                }
            
            # Prepare context for the AI
            context = "\n\n".join([
                f"Source: {r['title']} ({r['url']})\n{r['content']}" 
                for r in detailed_results
            ])
            
            # Generate answer using OpenAI
            if hasattr(openai_service, 'openai_configured') and openai_service.openai_configured:
                import openai
                
                prompt = f"""Based on the following web search results, provide a comprehensive answer to the question: "{query}"
                
                Search Results:
                {context}
                
                Instructions:
                1. Provide a clear, concise answer based only on the information in the search results
                2. Cite the sources using [Source: URL] format
                3. If the search results don't contain enough information to answer the question, clearly state that
                4. Focus on accuracy and relevance
                5. Format the answer in a readable way
                """
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on web search results."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.3
                )
                
                answer_text = response['choices'][0]['message']['content'].strip()
                logger.info(f"✅ Generated answer from web results using OpenAI")
            else:
                # Demo mode - create a synthesized answer
                sources_text = ", ".join([f"{r['title']}" for r in detailed_results])
                answer_text = f"""Based on web search results, I found the following information about "{query}":

According to the search results, {query} is a topic with several important aspects. The information indicates that it relates to {detailed_results[0]['content'][:100]}...

Multiple sources ({sources_text}) suggest that this is an important subject with various applications and interpretations.

[Source: {detailed_results[0]['url']}]
                """
                logger.info(f"🎭 Generated demo answer from web results")
            
            return {
                "answer": answer_text,
                "sources": [{"title": r["title"], "url": r["url"]} for r in detailed_results],
                "success": True
            }
                
        except Exception as e:
            logger.error(f"❌ Error generating answer from web results: {str(e)}")
            return {
                "answer": f"I found information about '{query}', but encountered an error processing it. You might want to check these sources directly: " + 
                          ", ".join([f"{r['title']} ({r['url']})" for r in search_results[:3]]),
                "sources": [{"title": r["title"], "url": r["url"]} for r in search_results[:3]],
                "success": False
            }

# Initialize the service
web_search = WebSearchService()
logger.info(f"🌐 Web Search Service initialized (provider: {web_search.search_provider})")
