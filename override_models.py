from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Enum
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime
import enum

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
    confidence_score = Column(Integer, nullable=False)
    source_type = Column(String(50), nullable=False)  # document, historical, gpt
    cache_hits = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # Optional expiration

print("✅ Override and Review models created:")
print("- User: manages user roles (user, reviewer, admin)")
print("- AnswerOverride: stores human-corrected answers")
print("- AnswerReview: tracks review workflow and quality flags")
print("- PerformanceCache: caches slow-generating answers")
