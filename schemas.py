from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class QuestionBase(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="The question text")

class QuestionCreate(QuestionBase):
    pass

class QuestionResponse(QuestionBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class AnswerBase(BaseModel):
    text: str = Field(..., min_length=1, description="The answer text")
    confidence_score: Optional[float] = Field(0.95, ge=0.0, le=1.0, description="Confidence score")

class AnswerCreate(AnswerBase):
    question_id: int

class AnswerResponse(AnswerBase):
    id: int
    question_id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class QuestionAnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")

class QuestionAnswerResponse(BaseModel):
    answer: str
    question_id: int
    answer_id: int
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    is_cached: bool = Field(..., description="Whether the answer was retrieved from cache")
    source_documents: Optional[List[str]] = Field(None, description="Source documents used for the answer")
    
    # New enhanced fields
    low_confidence: bool = Field(False, description="Flag for low confidence answers (< 0.80)")
    answer_type: str = Field("document", description="Type of answer: cached, document, gpt")
    confidence_score: float = Field(0.95, ge=0.0, le=1.0, description="System confidence in the answer")

class HealthResponse(BaseModel):
    status: str = "healthy"
    message: str = "API is running"
    timestamp: datetime
    database_connected: bool
    openai_configured: bool

class AuthHealthResponse(BaseModel):
    status: str = "authenticated"
    message: str = "API key is valid"
    timestamp: datetime

# Feedback schemas
class FeedbackCreate(BaseModel):
    question_id: int
    answer_id: int
    is_helpful: bool
    feedback_text: Optional[str] = None
    user_session: Optional[str] = None

class FeedbackResponse(BaseModel):
    id: int
    question_id: int
    answer_id: int
    is_helpful: bool
    feedback_text: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

class FeedbackStats(BaseModel):
    total_feedback: int
    helpful_count: int
    not_helpful_count: int
    helpful_percentage: float
    
    # Breakdown by answer type
    document_answers_feedback: int
    cached_answers_feedback: int
    gpt_answers_feedback: int
    
    # Confidence correlation
    avg_confidence_helpful: float
    avg_confidence_not_helpful: float

# Finally, let's create a simple database migration to add the feedback table:
