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

# Knowledge Base schemas
class KnowledgeBaseCreate(BaseModel):
    category: str = Field(..., min_length=1, max_length=100)
    question: str = Field(..., min_length=1, max_length=1000)
    answer: str = Field(..., min_length=1)
    keywords: Optional[List[str]] = Field(None, description="Keywords for search optimization")
    priority: int = Field(1, ge=1, le=10, description="Priority level (1-10, higher = more important)")
    is_active: bool = Field(True)

class KnowledgeBaseResponse(BaseModel):
    id: int
    category: str
    question: str
    answer: str
    keywords: Optional[List[str]]
    priority: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    created_by: str
    
    class Config:
        from_attributes = True

class KnowledgeBaseUpdate(BaseModel):
    category: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    keywords: Optional[List[str]] = None
    priority: Optional[int] = Field(None, ge=1, le=10)
    is_active: Optional[bool] = None

class KnowledgeBaseSearch(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    category: Optional[str] = None
    limit: int = Field(5, ge=1, le=20)
