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
        logger.info("📚 Enhanced Citation Service initialized")
    
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
    
    def enhance_answer_with_citations(
        self, 
        answer: str, 
        source_type: str, 
        source_info: Dict[str, Any]
    ) -> str:
        """Enhance an answer with proper citations based on source type"""
        
        if source_type == "pdf":
            return self._add_pdf_citations(answer, source_info)
        elif source_type == "web":
            return self._add_web_citations(answer, source_info)
        elif source_type == "knowledge_base":
            return self._add_kb_citations(answer, source_info)
        else:
            return answer
    
    def _add_pdf_citations(self, answer: str, source_info: Dict[str, Any]) -> str:
        """Add PDF citations with document names and page numbers"""
        documents = source_info.get("documents", [])
        chunks = source_info.get("chunks", [])
        
        if not documents:
            return answer
        
        # Extract page information from chunks
        page_info = {}
        for chunk in chunks:
            filename = chunk.get('filename', 'Unknown')
            page_num = chunk.get('page_number', 'Unknown')
            if filename not in page_info:
                page_info[filename] = set()
            if page_num != 'Unknown':
                page_info[filename].add(str(page_num))
        
        # Build citation text
        citations = []
        for doc in documents:
            if doc in page_info and page_info[doc]:
                pages = sorted(list(page_info[doc]))
                if len(pages) == 1:
                    citations.append(f"{doc} (page {pages[0]})")
                else:
                    citations.append(f"{doc} (pages {', '.join(pages)})")
            else:
                citations.append(f"{doc}")
        
        citation_text = "Sources: " + "; ".join(citations)
        
        return f"{answer}\n\n📄 {citation_text}"
    
    def _add_web_citations(self, answer: str, source_info: Dict[str, Any]) -> str:
        """Add web citations with URLs and titles"""
        sources = source_info.get("sources", [])
        
        if not sources:
            return answer
        
        citations = []
        for i, source in enumerate(sources, 1):
            title = source.get("title", "Web Source")
            url = source.get("url", "")
            citations.append(f"[{i}] {title}: {url}")
        
        citation_text = "\n".join(citations)
        
        return f"{answer}\n\n🌐 **Sources:**\n{citation_text}"
    
    def _add_kb_citations(self, answer: str, source_info: Dict[str, Any]) -> str:
        """Add knowledge base citations"""
        category = source_info.get("category", "General")
        kb_id = source_info.get("id", "")
        
        citation_text = f"Source: Knowledge Base - {category}"
        if kb_id:
            citation_text += f" (ID: {kb_id})"
        
        return f"{answer}\n\n🧠 {citation_text}"
    
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

# Global instance
citation_service = EnhancedCitationService()
