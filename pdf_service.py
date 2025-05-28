import os
import io
import hashlib
from typing import List, Tuple, Optional
import PyPDF2
import numpy as np
from sqlalchemy.orm import Session
from document_models import Document, DocumentChunk
from document_schemas import DocumentCreate, DocumentChunkCreate
import document_crud
import asyncio

class PDFService:
    def __init__(self, rag_service):
        self.rag_service = rag_service
        self.chunk_size = 1000  # Characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks
        
    async def process_pdf(self, db: Session, file_content: bytes, filename: str, original_filename: str) -> Document:
        """Process uploaded PDF file and extract text chunks"""
        try:
            print(f"📄 Processing PDF: {original_filename}")
            
            # Create document record
            file_size = len(file_content)
            document_data = DocumentCreate(
                filename=filename,
                original_filename=original_filename,
                file_size=file_size,
                content_type="application/pdf"
            )
            
            document = document_crud.create_document(db, document_data)
            
            # Extract text from PDF
            text_content, total_pages = self._extract_text_from_pdf(file_content)
            
            if not text_content.strip():
                # Update document with error
                document_crud.update_document_status(
                    db, document.id, 
                    processing_status="failed",
                    error_message="No text content found in PDF"
                )
                return document
            
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
                    chunk_create = DocumentChunkCreate(
                        document_id=document.id,
                        chunk_index=chunk_data['index'],
                        content=chunk_data['content'],
                        page_number=chunk_data['page_number'],
                        word_count=len(chunk_data['content'].split())
                    )
                    
                    chunk = document_crud.create_document_chunk(db, chunk_create, embedding)
                    chunk_count += 1
                    
                    # Add small delay to avoid rate limiting
                    if self.rag_service.openai_configured:
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    print(f"⚠️ Error processing chunk {chunk_data['index']}: {str(e)}")
                    continue
            
            # Update document status
            document_crud.update_document_status(
                db, document.id,
                processing_status="completed",
                processed=True,
                total_pages=total_pages,
                total_chunks=chunk_count
            )
            
            print(f"✅ PDF processing completed: {chunk_count} chunks created")
            return document
            
        except Exception as e:
            print(f"❌ Error processing PDF: {str(e)}")
            # Update document with error status
            if 'document' in locals():
                document_crud.update_document_status(
                    db, document.id,
                    processing_status="failed",
                    error_message=str(e)
                )
                return document
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
print("- Supports PyPDF2 text extraction")
print("- Automatic embedding generation")
