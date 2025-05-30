from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    USER = "user"
    REVIEWER = "reviewer"
    ADMIN = "admin"

class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    INSUFFICIENT = "insufficient"

class OverrideStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"

# User schemas
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    role: UserRole = UserRole.USER

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime
    
    model_config = {"from_attributes": True}

# Override schemas
class OverrideCreate(BaseModel):
    question_id: int
    original_answer_id: int
    override_text: str = Field(..., min_length=10)
    reason: str = Field(..., min_length=10)

class OverrideResponse(BaseModel):
    id: int
    question_id: int
    original_answer_id: int
    override_text: str
    reason: str
    status: OverrideStatus
    created_by: int
    created_at: datetime
    approved_at: Optional[datetime]
    
    model_config = {"from_attributes": True}

# Review schemas
class ReviewCreate(BaseModel):
    question_id: int
    answer_id: int
    override_id: Optional[int] = None
    review_status: ReviewStatus
    review_notes: Optional[str] = None
    is_insufficient: bool = False
    needs_more_context: bool = False
    factual_accuracy_concern: bool = False
    compliance_concern: bool = False

class ReviewResponse(BaseModel):
    id: int
    question_id: int
    answer_id: int
    override_id: Optional[int]
    review_status: ReviewStatus
    review_notes: Optional[str]
    reviewer_id: int
    reviewed_at: datetime
    is_insufficient: bool
    needs_more_context: bool
    factual_accuracy_concern: bool
    compliance_concern: bool
    
    model_config = {"from_attributes": True}

# Enhanced Q&A response with override support
class EnhancedQuestionAnswerResponse(BaseModel):
    answer: str
    question_id: int
    answer_id: int
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    is_cached: bool = Field(..., description="Whether the answer was retrieved from cache")
    source_documents: Optional[List[str]] = Field(None, description="Source documents used for the answer")
    
    # Enhanced fields
    low_confidence: bool = Field(False, description="Flag for low confidence answers (< 0.80)")
    answer_type: str = Field("document", description="Type of answer: cached, document, gpt")
    confidence_score: float = Field(0.95, ge=0.0, le=1.0, description="System confidence in the answer")
    
    # New override and review fields
    has_override: bool = Field(False, description="Whether this answer has been manually overridden")
    override_id: Optional[int] = Field(None, description="ID of the active override if any")
    needs_review: bool = Field(False, description="Whether this answer needs human review")
    review_flags: List[str] = Field(default_factory=list, description="Any review flags for this answer")
    
    # Performance fields
    generation_time_ms: Optional[int] = Field(None, description="Time taken to generate this answer")
    chunk_count: int = Field(0, description="Number of document chunks used")

# Review queue item
class ReviewQueueItem(BaseModel):
    question_id: int
    question_text: str
    answer_id: int
    answer_text: str
    confidence_score: float
    answer_type: str
    created_at: datetime
    has_override: bool
    review_count: int
    priority_score: float  # Based on confidence, flags, etc.

# Review statistics
class ReviewStats(BaseModel):
    total_pending_reviews: int
    total_insufficient_answers: int
    total_overrides: int
    avg_confidence_insufficient: float
    top_review_flags: List[dict]
    reviewer_activity: List[dict]
