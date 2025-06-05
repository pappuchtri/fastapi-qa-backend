from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Index, Float, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from sqlalchemy import Enum

# Import Base from database instead of creating a new one
from database import Base

class Question(Base):
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    answers = relationship("Answer", back_populates="question", cascade="all, delete-orphan")
    embeddings = relationship("Embedding", back_populates="question", cascade="all, delete-orphan")
    # Add web_answers relationship but use string reference to avoid circular imports
    #web_answers = relationship("WebAnswer", back_populates="question")

class Answer(Base):
    __tablename__ = "answers"
    
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    text = Column(Text, nullable=False)
    confidence_score = Column(Float, default=0.9)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    question = relationship("Question", back_populates="answers")

class Embedding(Base):
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    vector = Column(ARRAY(Float), nullable=True)  # Store embedding as array of floats
    embedding = Column(ARRAY(Float), nullable=True)  # Alternative column name for compatibility
    model_name = Column(String, default="text-embedding-ada-002")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    question = relationship("Question", back_populates="embeddings")

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

# Enums
class OverrideStatus(enum.Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"

class ReviewStatus(enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    INSUFFICIENT = "insufficient"

class UserRole(enum.Enum):
    USER = "user"
    REVIEWER = "reviewer"
    ADMIN = "admin"

# User and related models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.USER)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class AnswerOverride(Base):
    __tablename__ = "answer_overrides"
    
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    original_answer_id = Column(Integer, ForeignKey("answers.id"), nullable=False)
    override_text = Column(Text, nullable=False)
    reason = Column(Text, nullable=False)
    status = Column(Enum(OverrideStatus), default=OverrideStatus.ACTIVE)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    approved_at = Column(DateTime, nullable=True)
    
    # Relationships using string references to avoid circular imports
    question = relationship("Question")
    original_answer = relationship("Answer")

class AnswerReview(Base):
    __tablename__ = "answer_reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    answer_id = Column(Integer, ForeignKey("answers.id"), nullable=False)
    review_status = Column(Enum(ReviewStatus), default=ReviewStatus.PENDING)
    review_notes = Column(Text, nullable=True)
    reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    reviewed_at = Column(DateTime, default=datetime.utcnow)
    
    # Quality flags
    is_insufficient = Column(Boolean, default=False)
    needs_more_context = Column(Boolean, default=False)
    factual_accuracy_concern = Column(Boolean, default=False)
    compliance_concern = Column(Boolean, default=False)
    
    # Relationships
    question = relationship("Question")
    answer = relationship("Answer")

# Web search models
class WebSearchResult(Base):
    __tablename__ = "web_search_results"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    title = Column(Text, nullable=True)
    snippet = Column(Text, nullable=True)
    url = Column(String(1024), nullable=True)
    position = Column(Integer, default=0)
    source = Column(String(50), default="ai_native")  # ai_native, serpapi, google, bing
    created_at = Column(DateTime, default=datetime.utcnow)

class WebAnswer(Base):
    __tablename__ = "web_answers"
    
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=True)
    search_result_id = Column(Integer, ForeignKey("web_search_results.id"), nullable=True)
    answer_text = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)  # Store sources as JSON
    confidence_score = Column(Float, default=0.7)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    question = relationship("Question")
    search_result = relationship("WebSearchResult")

class UnansweredQuestion(Base):
    """Model for tracking questions that couldn't be answered"""
    __tablename__ = "unanswered_questions"
    
    id = Column(Integer, primary_key=True, index=True)
    question_text = Column(Text, nullable=False)
    search_attempts = Column(JSON, nullable=True)  # Store search attempts as JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text, nullable=True)

# Additional tracking models
class PerformanceCache(Base):
    __tablename__ = "performance_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    question_hash = Column(String(64), unique=True, nullable=False)  # MD5 hash of question
    question_text = Column(Text, nullable=False)
    cached_answer = Column(Text, nullable=False)
    generation_time_ms = Column(Integer, nullable=False)  # Original generation time
    confidence_score = Column(Float, nullable=False)
    source_type = Column(String(50), nullable=False)  # document, historical, gpt
    cache_hits = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # Optional expiration

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    entity_type = Column(String(100), nullable=False)  # question, answer, override, etc.
    entity_id = Column(Integer, nullable=False)
    details = Column(Text, nullable=True)
    ip_address = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")

# Add ReviewQueue as an alias for AnswerReview for backward compatibility
ReviewQueue = AnswerReview
