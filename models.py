from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Index
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String, index=True, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    keywords = Column(ARRAY(String), nullable=True)  # For keyword-based search
    priority = Column(Integer, default=1)  # Higher priority = shown first
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String, default="system")
    
    # Add indexes for better search performance
    __table_args__ = (
        Index('idx_kb_category_active', 'category', 'is_active'),
        Index('idx_kb_keywords', 'keywords', postgresql_using='gin'),
        Index('idx_kb_priority', 'priority'),
    )
