import re
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCitationService:
    """Service for enhancing answers with proper citations"""
    
    def __init__(self):
        """Initialize the citation service"""
        logger.info("📚 Initializing Enhanced Citation Service")
    
    def enhance_answer_with_citations(self, answer_text: str, source_type: str, source_info: Dict[str, Any]) -> str:
        """
        Enhance an answer with proper citations
        
        Args:
            answer_text: The original answer text
            source_type: The type of source (pdf, web, etc.)
            source_info: Information about the sources
            
        Returns:
            Enhanced answer with citations
        """
        try:
            logger.info(f"📝 Enhancing answer with {source_type} citations")
            
            if source_type == "pdf":
                return self._enhance_with_pdf_citations(answer_text, source_info)
            elif source_type == "web":
                return self._enhance_with_web_citations(answer_text, source_info)
            else:
                logger.warning(f"⚠️ Unknown source type: {source_type}")
                return answer_text
                
        except Exception as e:
            logger.error(f"❌ Error enhancing answer with citations: {str(e)}")
            return answer_text
    
    def _enhance_with_pdf_citations(self, answer_text: str, source_info: Dict[str, Any]) -> str:
        """Enhance answer with PDF citations including page numbers"""
        try:
            documents = source_info.get("documents", [])
            chunks = source_info.get("chunks", [])
            
            if not documents or not chunks:
                return answer_text
            
            # Extract page numbers from chunks
            doc_pages = {}
            for chunk in chunks:
                filename = chunk.get("filename", "Unknown")
                page_num = chunk.get("page_num", chunk.get("page", None))
                
                if filename not in doc_pages:
                    doc_pages[filename] = set()
                
                if page_num is not None:
                    doc_pages[filename].add(page_num)
            
            # Format citations
            citations = []
            for doc in documents:
                if doc in doc_pages and doc_pages[doc]:
                    pages = sorted(doc_pages[doc])
                    page_str = ", ".join([str(p) for p in pages])
                    citations.append(f"{doc} (pages {page_str})")
                else:
                    citations.append(doc)
            
            # Add citations to the answer
            if citations:
                citation_text = "\n\nSources:\n" + "\n".join([f"- {c}" for c in citations])
                return answer_text + citation_text
            
            return answer_text
            
        except Exception as e:
            logger.error(f"❌ Error enhancing with PDF citations: {str(e)}")
            return answer_text
    
    def _enhance_with_web_citations(self, answer_text: str, source_info: Dict[str, Any]) -> str:
        """Enhance answer with web citations including URLs"""
        try:
            sources = source_info.get("sources", [])
            
            if not sources:
                return answer_text
            
            # Format citations
            citations = []
            for source in sources:
                title = source.get("title", "Untitled")
                url = source.get("url", "")
                
                if title and url:
                    citations.append(f"{title} - {url}")
                elif url:
                    citations.append(url)
            
            # Add citations to the answer
            if citations:
                citation_text = "\n\nSources:\n" + "\n".join([f"- {c}" for c in citations])
                return answer_text + citation_text
            
            return answer_text
            
        except Exception as e:
            logger.error(f"❌ Error enhancing with web citations: {str(e)}")
            return answer_text
    
    def format_pdf_citation(
        self, 
        document_name: str, 
        page_number: Optional[int] = None, 
        chunk_index: Optional[int] = None
    ) -> str:
        """Format a citation for a PDF document with page number if available"""
        if page_number:
            return f"[{document_name}, Page {page_number}]"
        elif chunk_index is not None:
            return f"[{document_name}, Section {chunk_index+1}]"
        else:
            return f"[{document_name}]"
    
    def format_web_citation(self, title: str, url: str) -> str:
        """Format a citation for a web source"""
        # Truncate URL if too long
        display_url = url
        if len(url) > 60:
            display_url = url[:57] + "..."
        
        return f"[{title}]({display_url})"
    
    def format_kb_citation(self, category: str, entry_id: int) -> str:
        """Format a citation for a knowledge base entry"""
        return f"[Knowledge Base: {category}, ID: {entry_id}]"
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def extract_page_numbers_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[int]:
        """Extract unique page numbers from document chunks"""
        if not chunks:
            return []
        
        page_numbers = []
        for chunk in chunks:
            page = chunk.get("page_number")
            if page and page not in page_numbers:
                page_numbers.append(page)
        
        return sorted(page_numbers)
    
    def extract_page_numbers_from_text(self, text: str) -> List[int]:
        """Extract page numbers from text content"""
        # Look for patterns like "page 5", "p. 10", "pg 15", etc.
        patterns = [
            r'page\s+(\d+)',
            r'p\.\s*(\d+)',
            r'pg\s+(\d+)',
            r'page\s*(\d+)'
        ]
        
        page_numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            page_numbers.extend([int(match) for match in matches])
        
        return sorted(list(set(page_numbers)))

# Create a singleton instance
citation_service = EnhancedCitationService()
