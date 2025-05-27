from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What is the capital of France?"
            }
        }

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="The generated or retrieved answer")
    question_id: int = Field(..., description="ID of the question")
    answer_id: int = Field(..., description="ID of the answer")
    similarity_score: Optional[float] = Field(default=0.0, description="Similarity score with existing questions")
    is_cached: bool = Field(default=False, description="Whether the answer was retrieved from cache")
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "Paris is the capital of France.",
                "question_id": 1,
                "answer_id": 1,
                "similarity_score": 0.95,
                "is_cached": True
            }
        }

class AuthErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "Invalid API key"
            }
        }

class ApiKeyResponse(BaseModel):
    api_key: str = Field(..., description="The generated API key")
    message: str = Field(..., description="Success message")
    masked_key: str = Field(..., description="Masked version of the API key")
    
    class Config:
        schema_extra = {
            "example": {
                "api_key": "qa-abcd1234efgh5678ijkl9012mnop3456qrst7890uvwx",
                "message": "API key generated successfully",
                "masked_key": "qa-abcd1...3456"
            }
        }

class QuestionSchema(BaseModel):
    id: int
    text: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class AnswerSchema(BaseModel):
    id: int
    question_id: int
    text: str
    confidence_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class EmbeddingSchema(BaseModel):
    id: int
    question_id: int
    vector: List[float]
    model_name: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class RAGStatsResponse(BaseModel):
    total_embeddings: int
    total_questions: int
    total_answers: int
    embedding_coverage: float

print("Updated Pydantic schemas with authentication models:")
print("- Added AuthErrorResponse for authentication errors")
print("- Added ApiKeyResponse for API key management")
print("- Updated existing schemas with authentication context")
