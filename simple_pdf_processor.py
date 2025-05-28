import PyPDF2
import io
from typing import List, Dict
from sqlalchemy.orm import Session
from document_models import Document, DocumentChunk
from rag_service import RAGService
import asyncio

class SimplePDFProcessor:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.chunk_size = 1000  # Characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks
    
    async def process_pdf_content(self, db: Session, document_id: int, pdf_content: bytes) -> bool:
        """
        Process PDF content and store chunks with embeddings
        """
        try:
            print(f"📄 Processing PDF for document {document_id}")
            
            # Get document record
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                print(f"❌ Document {document_id} not found")
                return False
            
            # Update status
            document.processing_status = "processing"
            db.commit()
            
            # Extract text from PDF
            text_content = self.extract_text_from_pdf(pdf_content)
            
            if not text_content.strip():
                document.processing_status = "failed"
                document.error_message = "No text content found in PDF"
                db.commit()
                return False
            
            print(f"📝 Extracted {len(text_content)} characters from PDF")
            
            # Split into chunks
            chunks = self.split_text_into_chunks(text_content)
            print(f"🔪 Split into {len(chunks)} chunks")
            
            # Process each chunk
            for i, chunk_text in enumerate(chunks):
                try:
                    # Generate embedding for chunk
                    embedding = await self.rag_service.generate_embedding(chunk_text)
                    
                    # Create chunk record
                    chunk = DocumentChunk(
                        document_id=document_id,
                        chunk_index=i,
                        content=chunk_text,
                        chunk_embedding=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                        page_number=None  # Could be enhanced to track page numbers
                    )
                    
                    db.add(chunk)
                    
                    if i % 5 == 0:  # Commit every 5 chunks
                        db.commit()
                        print(f"💾 Processed chunk {i+1}/{len(chunks)}")
                
                except Exception as e:
                    print(f"⚠️ Error processing chunk {i}: {str(e)}")
                    continue
            
            # Final commit
            db.commit()
            
            # Update document status
            document.processing_status = "processed"
            document.processed = True
            document.total_chunks = len(chunks)
            document.error_message = None
            db.commit()
            
            print(f"✅ Successfully processed document {document_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing PDF: {str(e)}")
            
            # Update document with error
            try:
                document = db.query(Document).filter(Document.id == document_id).first()
                if document:
                    document.processing_status = "failed"
                    document.error_message = str(e)
                    db.commit()
            except:
                pass
            
            return False
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text content from PDF bytes
        """
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n[Page {page_num + 1}]\n{page_text}\n"
                except Exception as e:
                    print(f"⚠️ Error extracting text from page {page_num + 1}: {str(e)}")
                    continue
            
            return text_content
            
        except Exception as e:
            print(f"❌ Error reading PDF: {str(e)}")
            return ""
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for i in range(end, max(start + self.chunk_size - 100, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks

print("✅ Simple PDF Processor created")
