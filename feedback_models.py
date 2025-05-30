from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class AnswerFeedback(Base):
    __tablename__ = "answer_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    answer_id = Column(Integer, ForeignKey("answers.id"), nullable=False)
    
    # Feedback data
    is_helpful = Column(Boolean, nullable=False)  # True for thumbs up, False for thumbs down
    feedback_text = Column(Text, nullable=True)  # Optional detailed feedback
    confidence_score = Column(Float, nullable=True)  # System confidence when answer was given
    answer_type = Column(String(50), nullable=True)  # "cached", "document", "gpt"
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    user_session = Column(String(100), nullable=True)  # Optional session tracking
    
    # Simple relationships without back_populates to avoid circular import issues
    # question = relationship("Question")
    # answer = relationship("Answer")

# Remove these lines that are causing the error:
# from models import Question, Answer
# Question.feedback = relationship("AnswerFeedback", back_populates="question")
# Answer.feedback = relationship("AnswerFeedback", back_populates="answer")

print("✅ Feedback models created:")
print("- Answer feedback tracking (thumbs up/down)")
print("- Detailed feedback text")
print("- Confidence score tracking")
print("- Answer type classification")
