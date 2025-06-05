import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackHandler:
    """Handler for user feedback on answers"""
    
    def __init__(self):
        """Initialize the feedback handler"""
        logger.info("👍 Initializing Feedback Handler")
    
    def save_feedback(self, db: Session, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save user feedback on an answer
        
        Args:
            db: Database session
            feedback_data: Feedback data including rating, comments, etc.
            
        Returns:
            Dict with status and feedback ID
        """
        try:
            from feedback_models import AnswerFeedback
            
            question_id = feedback_data.get("question_id")
            answer_id = feedback_data.get("answer_id")
            rating = feedback_data.get("rating")
            comments = feedback_data.get("comments", "")
            
            if not question_id or not answer_id or rating is None:
                logger.warning("⚠️ Missing required feedback data")
                return {
                    "success": False,
                    "error": "Missing required feedback data"
                }
            
            feedback = AnswerFeedback(
                question_id=question_id,
                answer_id=answer_id,
                rating=rating,
                comments=comments,
                is_helpful=(rating >= 3)  # Ratings 3-5 are considered helpful
            )
            
            db.add(feedback)
            db.commit()
            db.refresh(feedback)
            
            logger.info(f"✅ Saved feedback for answer {answer_id} with rating {rating}")
            
            return {
                "success": True,
                "feedback_id": feedback.id,
                "message": "Feedback saved successfully"
            }
            
        except Exception as e:
            logger.error(f"❌ Error saving feedback: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_feedback_stats(self, db: Session) -> Dict[str, Any]:
        """
        Get feedback statistics
        
        Args:
            db: Database session
            
        Returns:
            Dict with feedback statistics
        """
        try:
            from sqlalchemy import func
            from feedback_models import AnswerFeedback
            
            total_count = db.query(func.count(AnswerFeedback.id)).scalar() or 0
            helpful_count = db.query(func.count(AnswerFeedback.id)).filter(AnswerFeedback.is_helpful == True).scalar() or 0
            
            if total_count > 0:
                helpful_percentage = (helpful_count / total_count) * 100
            else:
                helpful_percentage = 0
            
            return {
                "total_feedback": total_count,
                "helpful_count": helpful_count,
                "helpful_percentage": helpful_percentage,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting feedback stats: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "total_feedback": 0,
                "helpful_count": 0,
                "helpful_percentage": 0
            }

# Create a singleton instance
feedback_handler = FeedbackHandler()
