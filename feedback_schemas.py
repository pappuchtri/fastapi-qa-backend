from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class FeedbackCreate(BaseModel):
    question_id: int
    answer_id: int
    is_helpful: bool = Field(..., description="True for helpful, False for not helpful")
    feedback_text: Optional[str] = Field(None, max_length=1000, description="Optional detailed feedback")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    answer_type: Optional[str] = Field(None, description="Type of answer: cached, document, gpt")
    user_session: Optional[str] = Field(None, max_length=100)

class FeedbackResponse(BaseModel):
    id: int
    question_id: int
    answer_id: int
    is_helpful: bool
    feedback_text: Optional[str]
    confidence_score: Optional[float]
    answer_type: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

class QuestionSuggestion(BaseModel):
    original_question: str
    suggested_questions: list[str] = Field(..., description="AI-generated question suggestions")
    reasoning: str = Field(..., description="Why these suggestions were made")

print("✅ Feedback schemas created:")
print("- Feedback creation and response models")
print("- Question suggestion schema")
print("- Validation and field constraints")
