from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base
import sqlalchemy as sa

# Custom VECTOR type for pgvector
class VECTOR(sa.types.UserDefinedType):
    def __init__(self, dim=None):
        self.dim = dim

    def get_col_spec(self):
        if self.dim is not None:
            return f"VECTOR({self.dim})"
        return "VECTOR"

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String(100), default='application/pdf')
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    processed = Column(Boolean, default=False)
    processing_status = Column(String(50), default='pending')
    error_message = Column(Text, nullable=True)
    total_pages = Column(Integer, nullable=True)
    total_chunks = Column(Integer, default=0)
    metadata = Column(JSONB, default={})

    # Relationship to document chunks
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    chunk_embedding = Column(VECTOR(1536), nullable=True)  # OpenAI ada-002 embedding dimension
    word_count = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSONB, default={})

    # Relationship to document
    document = relationship("Document", back_populates="chunks")
