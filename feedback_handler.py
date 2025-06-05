from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FeedbackHandler:
    """Handle user feedback and suggestions"""
    
    def collect_feedback(
        self, 
        db: Session, 
        question_id: Optional[int], 
        answer_id: Optional[int], 
        feedback_type: str, 
        rating: Optional[int], 
        comment: Optional[str]
    ) -> Dict[str, Any]:
        """Collect user feedback on answers"""
        try:
            from feedback_models import AnswerFeedback
            
            feedback = AnswerFeedback(
                question_id=question_id,
                answer_id=answer_id,
                feedback_type=feedback_type,
                rating=rating,
                comment=comment
            )
            
            db.add(feedback)
            db.commit()
            db.refresh(feedback)
            
            logger.info(f"✅ Feedback collected: {feedback_type} for Q{question_id}/A{answer_id}")
            
            return {
                "success": True,
                "feedback_id": feedback.id,
                "message": "Thank you for your feedback!"
            }
            
        except Exception as e:
            logger.error(f"❌ Error collecting feedback: {str(e)}")
            return {
                "success": False,
                "message": f"Error saving feedback: {str(e)}"
            }
    
    def suggest_improvements(
        self, 
        question: str, 
        answer: str, 
        feedback_type: str
    ) -> List[str]:
        """Suggest improvements based on feedback"""
        suggestions = []
        
        if feedback_type == "not_helpful":
            suggestions.extend([
                "Try rephrasing your question with more specific terms",
                "Include more context about what you're looking for",
                "Check if there are related documents you can upload"
            ])
        elif feedback_type == "incorrect":
            suggestions.extend([
                "Please provide the correct information so we can improve",
                "Consider uploading relevant documents to improve accuracy",
                "Try asking a more specific question"
            ])
        elif feedback_type == "incomplete":
            suggestions.extend([
                "Try asking follow-up questions for more details",
                "Upload additional documents that might contain more information",
                "Rephrase your question to be more specific"
            ])
        
        return suggestions

# Global instance
feedback_handler = FeedbackHandler()
