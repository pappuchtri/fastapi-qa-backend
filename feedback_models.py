from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class AnswerFeedback(Base):
    __tablename__ = "answer_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    answer_id = Column(Integer, ForeignKey("answers.id"), nullable=False)
    is_helpful = Column(Boolean, nullable=False)
    feedback_text = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    answer_type = Column(String(50), nullable=True)
    user_session = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships - using string references to avoid circular imports
    question = relationship("Question")
    answer = relationship("Answer")
