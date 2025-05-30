from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class FeedbackCreate(BaseModel):
    question_id: int
    answer_id: int
    is_helpful: bool
    feedback_text: Optional[str] = None
    confidence_score: Optional[float] = None
    answer_type: Optional[str] = None
    user_session: Optional[str] = None

class FeedbackResponse(BaseModel):
    id: int
    question_id: int
    answer_id: int
    is_helpful: bool
    feedback_text: Optional[str] = None
    confidence_score: Optional[float] = None
    answer_type: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class QuestionSuggestion(BaseModel):
    original_question: str
    suggested_questions: List[str]
    reasoning: str

print("✅ Feedback schemas created:")
print("- Feedback creation and response models")
print("- Question suggestion schema")
print("- Validation and field constraints")
