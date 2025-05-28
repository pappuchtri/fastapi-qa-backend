import os
import io
import hashlib
from typing import List, Tuple, Dict, Any
import PyPDF2
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session
import asyncio

class PDFService:
    def __init__(self, rag_service):
        self.rag_service = rag_service
        self.chunk_size = 1000  # Characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks
        
    async def process_pdf(self, db: Session, file_content: bytes, filename: str, original_filename: str) -> Dict[str, Any]:
        """Process uploaded PDF file and extract text chunks"""
        try:
            print(f"📄 Processing PDF: {original_filename}")
            
            # Create document record using raw SQL
            file_size = len(file_content)
            result = db.execute(text("""
                INSERT INTO documents 
                (filename, original_filename, file_size, content_type, processing_status)
                VALUES (:filename, :original_filename, :file_size, :content_type, :status)
                RETURNING id
            """), {
                "filename": filename,
                "original_filename": original_filename,
                "file_size": file_size,
                "content_type": "application/pdf",
                "status": "processing"
            })
            
            document_id = result.fetchone()[0]
            db.commit()
            
            # Extract text from PDF
            text_content, total_pages = self._extract_text_from_pdf(file_content)
            
            if not text_content.strip():
                # Update document with error
                db.execute(text("""
                    UPDATE documents 
                    SET processing_status = :status, error_message = :error, processed = false
                    WHERE id = :id
                """), {
                    "status": "failed",
                    "error": "No text content found in PDF",
                    "id": document_id
                })
                db.commit()
                
                return {
                    "document_id": document_id,
                    "status": "failed",
                    "error": "No text content found in PDF"
                }
            
            # Create text chunks
            chunks = self._create_text_chunks(text_content, total_pages)
            
            print(f"📝 Created {len(chunks)} text chunks")
            
            # Process chunks and generate embeddings
            chunk_count = 0
            for chunk_data in chunks:
                try:
                    # Generate embedding for chunk
                    embedding = await self.rag_service.generate_embedding(chunk_data['content'])
                    
                    # Create chunk record
                    db.execute(text("""
                        INSERT INTO document_chunks
                        (document_id, chunk_index, content, page_number, word_count, chunk_embedding)
                        VALUES (:doc_id, :index, :content, :page, :words, :embedding)
                    """), {
                        "doc_id": document_id,
                        "index": chunk_data['index'],
                        "content": chunk_data['content'],
                        "page": chunk_data['page_number'],
                        "words": len(chunk_data['content'].split()),
                        "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                    })
                    
                    chunk_count += 1
                    
                    # Commit every 5 chunks to avoid long transactions
                    if chunk_count % 5 == 0:
                        db.commit()
                    
                    # Add small delay to avoid rate limiting
                    if self.rag_service.openai_configured:
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    print(f"⚠️ Error processing chunk {chunk_data['index']}: {str(e)}")
                    continue
            
            # Final commit for any remaining chunks
            db.commit()
            
            # Update document status
            db.execute(text("""
                UPDATE documents
                SET processing_status = :status, processed = true, 
                    total_pages = :pages, total_chunks = :chunks
                WHERE id = :id
            """), {
                "status": "completed",
                "pages": total_pages,
                "chunks": chunk_count,
                "id": document_id
            })
            db.commit()
            
            print(f"✅ PDF processing completed: {chunk_count} chunks created")
            
            return {
                "document_id": document_id,
                "status": "completed",
                "total_pages": total_pages,
                "total_chunks": chunk_count
            }
            
        except Exception as e:
            print(f"❌ Error processing PDF: {str(e)}")
            # Update document with error status if document was created
            if 'document_id' in locals():
                db.execute(text("""
                    UPDATE documents
                    SET processing_status = :status, error_message = :error, processed = false
                    WHERE id = :id
                """), {
                    "status": "failed",
                    "error": str(e),
                    "id": document_id
                })
                db.commit()
                
                return {
                    "document_id": document_id,
                    "status": "failed",
                    "error": str(e)
                }
            
            raise e
    
    def _extract_text_from_pdf(self, file_content: bytes) -> Tuple[str, int]:
        """Extract text content from PDF bytes"""
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            total_pages = len(pdf_reader.pages)
            text_content = ""
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content += f"\n--- Page {page_num} ---\n{page_text}\n"
                except Exception as e:
                    print(f"⚠️ Error extracting text from page {page_num}: {str(e)}")
                    continue
            
            return text_content, total_pages
            
        except Exception as e:
            print(f"❌ Error reading PDF: {str(e)}")
            raise e
    
    def _create_text_chunks(self, text: str, total_pages: int) -> List[dict]:
        """Split text into overlapping chunks"""
        chunks = []
        chunk_index = 0
        
        # Split by pages first
        pages = text.split("--- Page ")
        
        for page_content in pages:
            if not page_content.strip():
                continue
                
            # Extract page number
            lines = page_content.split('\n')
            page_number = 1
            if lines[0].strip().endswith(" ---"):
                try:
                    page_number = int(lines[0].split()[0])
                    page_text = '\n'.join(lines[1:])
                except:
                    page_text = page_content
            else:
                page_text = page_content
            
            # Split page into chunks
            page_chunks = self._split_text_into_chunks(page_text)
            
            for chunk_text in page_chunks:
                if chunk_text.strip():
                    chunks.append({
                        'index': chunk_index,
                        'content': chunk_text.strip(),
                        'page_number': page_number
                    })
                    chunk_index += 1
        
        return chunks
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at sentence or word boundary
            chunk = text[start:end]
            
            # Look for sentence boundary
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            last_space = chunk.rfind(' ')
            
            break_point = max(last_period, last_newline, last_space)
            
            if break_point > start + self.chunk_size // 2:
                chunk = text[start:start + break_point + 1]
                start = start + break_point + 1 - self.chunk_overlap
            else:
                start = end - self.chunk_overlap
            
            chunks.append(chunk)
        
        return chunks

print("✅ PDF Service initialized:")
print("- Chunk size: 1000 characters")
print("- Chunk overlap: 200 characters")
print("- Using PyPDF2 for text extraction")
print("- Using direct SQL for database operations")
