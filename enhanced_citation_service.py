import re
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCitationService:
    """Service for formatting and enhancing citations from various sources"""
    
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
        answer_text: str, 
        source_type: str,
        source_info: Dict[str, Any]
    ) -> str:
        """
        Enhance an answer with properly formatted citations
        
        Parameters:
        - answer_text: The original answer text
        - source_type: Type of source ("pdf", "web", "kb")
        - source_info: Dictionary with source information
        
        Returns:
        - Enhanced answer text with citations
        """
        try:
            if source_type == "pdf":
                # Format PDF citations
                documents = source_info.get("documents", [])
                if not documents:
                    return answer_text
                
                # Check if answer already has citations
                if any(f"[{doc}]" in answer_text for doc in documents):
                    # Already has citations, just ensure they're properly formatted
                    return answer_text
                
                # Add citation footer
                citations = []
                for doc_info in source_info.get("chunks", []):
                    doc_name = doc_info.get("filename", "Unknown")
                    page_num = doc_info.get("page_number")
                    chunk_idx = doc_info.get("chunk_index")
                    
                    citation = self.format_pdf_citation(doc_name, page_num, chunk_idx)
                    if citation not in citations:
                        citations.append(citation)
                
                if citations:
                    citation_text = "\n\nSources:\n" + "\n".join(citations)
                    return answer_text + citation_text
                
                return answer_text
                
            elif source_type == "web":
                # Format web citations
                sources = source_info.get("sources", [])
                if not sources:
                    return answer_text
                
                # Check if answer already has citations
                has_citations = False
                for source in sources:
                    if source.get("url", "") in answer_text:
                        has_citations = True
                        break
                
                if has_citations:
                    # Already has citations, just ensure they're properly formatted
                    return answer_text
                
                # Add citation footer
                citations = []
                for source in sources:
                    title = source.get("title", "Unknown")
                    url = source.get("url", "")
                    
                    citation = self.format_web_citation(title, url)
                    if citation not in citations:
                        citations.append(citation)
                
                if citations:
                    citation_text = "\n\nSources:\n" + "\n".join(citations)
                    return answer_text + citation_text
                
                return answer_text
                
            elif source_type == "kb":
                # Knowledge base entries don't typically need citations
                # But we can add a subtle reference if desired
                category = source_info.get("category", "General")
                entry_id = source_info.get("id", 0)
                
                if entry_id > 0:
                    footer = f"\n\n(From Knowledge Base: {category})"
                    return answer_text + footer
                
                return answer_text
                
            else:
                # Unknown source type, return original
                return answer_text
                
        except Exception as e:
            logger.error(f"❌ Error enhancing answer with citations: {str(e)}")
            return answer_text
    
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

# Initialize the service
citation_service = EnhancedCitationService()
