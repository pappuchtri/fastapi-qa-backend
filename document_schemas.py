from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Base schemas for questions and answers
class QuestionBase(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="The question text")

class QuestionCreate(QuestionBase):
    pass

class QuestionResponse(QuestionBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class AnswerBase(BaseModel):
    text: str = Field(..., min_length=1, description="The answer text")
    confidence_score: Optional[float] = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score between 0 and 1")

class AnswerCreate(AnswerBase):
    question_id: int

class AnswerResponse(AnswerBase):
    id: int
    question_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Q&A Request/Response schemas
class QuestionAnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")

class QuestionAnswerResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    question_id: int = Field(..., description="ID of the stored question")
    answer_id: int = Field(..., description="ID of the stored answer")
    similarity_score: Optional[float] = Field(default=0.0, description="Similarity score with existing questions")
    is_cached: bool = Field(default=False, description="Whether the answer was retrieved from cache")
    source_documents: List[str] = Field(default_factory=list, description="List of source document filenames")

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
    processed: bool = False
    processing_status: str = "pending"
    error_message: Optional[str] = None
    total_pages: Optional[int] = None
    total_chunks: int = 0
    
    class Config:
        from_attributes = True

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    page: int = 1
    per_page: int = 10

class DocumentUploadResponse(BaseModel):
    message: str
    document_id: int
    filename: str
    file_size: int
    processing_status: str

class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of results")
    document_ids: Optional[List[int]] = Field(default=None, description="Optional list of document IDs to search within")

class DocumentSearchResult(BaseModel):
    chunk_id: int
    document_id: int
    document_filename: str
    content: str
    page_number: Optional[int]
    similarity_score: float

class DocumentSearchResponse(BaseModel):
    results: List[DocumentSearchResult]
    query: str
    total_results: int

# Health check schemas
class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime
    database_connected: bool
    openai_configured: bool

class AuthHealthResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime

# Legacy schemas for backward compatibility
class QuestionRequest(BaseModel):
    question: str

class ApiKeyResponse(BaseModel):
    message: str
    valid: bool

class AuthErrorResponse(BaseModel):
    detail: str
    error_code: str = "INVALID_API_KEY"
