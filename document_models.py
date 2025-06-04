from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

# Try to import VECTOR from pgvector, fallback to Text if not available
try:
    from pgvector.sqlalchemy import Vector
    VECTOR_TYPE = Vector(1536)  # OpenAI ada-002 embedding dimension
    print("✅ Using pgvector for embeddings")
except ImportError:
    print("⚠️  pgvector not available, using Text for embeddings")
    VECTOR_TYPE = Text  # Fallback to Text type

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
    doc_metadata = Column(JSONB, default=lambda: {})  # Renamed from 'metadata'
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True, index=True)
    chunk_embedding = Column(VECTOR_TYPE, nullable=True)  # Vector embeddings with fallback
    word_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    chunk_metadata = Column(JSONB, default=lambda: {})  # Renamed from 'metadata'
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

print("✅ Document models created:")
print("- Document: stores PDF metadata and processing status")
print("- DocumentChunk: stores extracted text chunks with embeddings")
