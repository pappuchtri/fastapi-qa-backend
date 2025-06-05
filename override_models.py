from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Enum, Float
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime
import enum
from models import Question, Answer  # Import the models we need to reference

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

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.USER)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    overrides = relationship("AnswerOverride", back_populates="created_by_user")
    reviews = relationship("AnswerReview", back_populates="reviewer")

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
    
    # Relationships
    question = relationship("Question")
    original_answer = relationship("Answer")
    created_by_user = relationship("User", back_populates="overrides")
    reviews = relationship("AnswerReview", back_populates="override")

class AnswerReview(Base):
    __tablename__ = "answer_reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    answer_id = Column(Integer, ForeignKey("answers.id"), nullable=False)
    override_id = Column(Integer, ForeignKey("answer_overrides.id"), nullable=True)
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
    override = relationship("AnswerOverride", back_populates="reviews")
    reviewer = relationship("User", back_populates="reviews")

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

class ChunkUsageLog(Base):
    __tablename__ = "chunk_usage_log"
    
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id"), nullable=False)
    relevance_score = Column(Float, nullable=False)
    position_in_results = Column(Integer, nullable=False)  # 1-based position in results
    was_used_in_answer = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    question = relationship("Question")
    chunk = relationship("DocumentChunk")

class ReviewTag(Base):
    __tablename__ = "review_tags"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    tagged_reviews = relationship("ReviewTagAssociation", back_populates="tag")

class ReviewTagAssociation(Base):
    __tablename__ = "review_tag_associations"
    
    id = Column(Integer, primary_key=True, index=True)
    review_id = Column(Integer, ForeignKey("answer_reviews.id"), nullable=False)
    tag_id = Column(Integer, ForeignKey("review_tags.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    review = relationship("AnswerReview")
    tag = relationship("ReviewTag", back_populates="tagged_reviews")

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(255), nullable=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")

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
