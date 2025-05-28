from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

# Custom VECTOR type for pgvector
class VECTOR:
    def __init__(self, dimensions):
        self.dimensions = dimensions

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String(100), default='application/pdf')
    upload_date = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    processing_status = Column(String(50), default='pending')
    error_message = Column(Text, nullable=True)
    total_pages = Column(Integer, nullable=True)
    total_chunks = Column(Integer, default=0)
    doc_metadata = Column(JSONB, default={})
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    # For now, we'll store embeddings as TEXT (JSON string) to avoid pgvector dependency
    chunk_embedding = Column(Text, nullable=True)  # Will store JSON string of the vector
    word_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    chunk_metadata = Column(JSONB, default={})
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

print("✅ Document models created:")
print("- Document: stores PDF metadata")
print("- DocumentChunk: stores extracted text chunks with embeddings")
