from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

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
    metadata: Dict[str, Any] = {}
    
    class Config:
        from_attributes = True

class DocumentChunkBase(BaseModel):
    chunk_index: int
    content: str
    page_number: Optional[int] = None
    word_count: Optional[int] = None

class DocumentChunkCreate(DocumentChunkBase):
    document_id: int

class DocumentChunkResponse(DocumentChunkBase):
    id: int
    document_id: int
    created_at: datetime
    metadata: Dict[str, Any] = {}
    
    class Config:
        from_attributes = True

class DocumentUploadResponse(BaseModel):
    message: str
    document_id: int
    filename: str
    file_size: int
    processing_status: str

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    page: int
    per_page: int

class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    document_ids: Optional[List[int]] = None
    limit: int = Field(5, ge=1, le=20)

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
