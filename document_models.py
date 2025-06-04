from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import VECTOR, ARRAY, JSONB
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String(100), default='application/pdf')
    upload_date = Column(DateTime, default=datetime.utcnow, index=True)
    processed = Column(Boolean, default=False, index=True)
    processing_status = Column(String(50), default='pending')
    error_message = Column(Text, nullable=True)
    total_pages = Column(Integer, nullable=True)
    total_chunks = Column(Integer, default=0)
    metadata = Column(JSONB, default=lambda: {})
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True, index=True)
    chunk_embedding = Column(VECTOR(1536), nullable=True)  # OpenAI ada-002 embedding dimension
    word_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSONB, default=lambda: {})
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

print("✅ Document models created:")
print("- Document: stores PDF metadata and processing status")
print("- DocumentChunk: stores extracted text chunks with embeddings")
