from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# Question schemas
class QuestionBase(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="The question text")

class QuestionCreate(QuestionBase):
    pass

class QuestionRequest(QuestionBase):
    pass

class QuestionResponse(QuestionBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Answer schemas
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

# Q&A Request/Response schemas
class QuestionAnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")

class QuestionAnswerResponse(BaseModel):
    answer: str
    question_id: int
    answer_id: int
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    is_cached: bool = Field(..., description="Whether the answer was retrieved from cache")
    source_documents: Optional[List[str]] = Field(None, description="Source documents used for the answer")

# Health check schemas
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

# Document schemas
class DocumentBase(BaseModel):
    filename: str
    original_filename: str
    file_size: int
    content_type: str = "application/pdf"

class DocumentCreate(DocumentBase):
    pass

class DocumentResponse(DocumentBase):
    id: int
    upload_date: datetime
    processed: bool
    processing_status: str
    error_message: Optional[str] = None
    total_pages: Optional[int] = None
    total_chunks: int = 0
    
    class Config:
        from_attributes = True

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    page: int
    per_page: int

class DocumentUploadResponse(BaseModel):
    message: str
    document_id: int
    filename: str
    file_size: int
    processing_status: str

class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(5, ge=1, le=20, description="Maximum number of results")
    document_ids: Optional[List[int]] = Field(None, description="Specific document IDs to search in")

class DocumentSearchResponse(BaseModel):
    results: List[dict]
    query: str
    total_results: int

# Legacy schemas for backward compatibility
class ApiKeyResponse(BaseModel):
    message: str
    valid: bool

class AuthErrorResponse(BaseModel):
    detail: str
    error_code: str = "AUTH_ERROR"
