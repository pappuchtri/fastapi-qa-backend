from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackHandler:
    """Handler for processing and storing user feedback on answers"""
    
    def __init__(self):
        """Initialize the feedback handler"""
        logger.info("📝 Feedback Handler initialized")
    
    def store_feedback(
        self, 
        db: Session, 
        question_id: Optional[int],
        answer_id: Optional[int],
        feedback_type: str,
        is_helpful: bool,
        feedback_text: Optional[str] = None,
        source_type: Optional[str] = None,
        user_session: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store user feedback on an answer
        
        Parameters:
        - db: Database session
        - question_id: ID of the question (if available)
        - answer_id: ID of the answer (if available)
        - feedback_type: Type of feedback ("thumbs", "text", "report")
        - is_helpful: Whether the answer was helpful
        - feedback_text: Optional text feedback
        - source_type: Source of the answer (kb, pdf, web, chatgpt)
        - user_session: User session identifier
        
        Returns:
        - Dictionary with feedback information
        """
        try:
            # Import here to avoid circular imports
            from feedback_models import AnswerFeedback
            
            # Generate session ID if not provided
            if not user_session:
                user_session = str(uuid.uuid4())
            
            # Create feedback record
            feedback = AnswerFeedback(
                question_id=question_id,
                answer_id=answer_id,
                feedback_type=feedback_type,
                is_helpful=is_helpful,
                feedback_text=feedback_text,
                source_type=source_type,
                user_session=user_session,
                created_at=datetime.utcnow()
            )
            
            db.add(feedback)
            db.commit()
            db.refresh(feedback)
            
            logger.info(f"✅ Stored feedback (ID: {feedback.id}, helpful: {is_helpful})")
            
            return {
                "feedback_id": feedback.id,
                "status": "success",
                "message": "Feedback stored successfully",
                "timestamp": feedback.created_at.isoformat()
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error storing feedback: {str(e)}")
            
            return {
                "status": "error",
                "message": f"Error storing feedback: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_feedback_stats(self, db: Session) -> Dict[str, Any]:
        """Get statistics on collected feedback"""
        try:
            # Import here to avoid circular imports
            from feedback_models import AnswerFeedback
            from sqlalchemy import func
            
            # Get total counts
            total_count = db.query(func.count(AnswerFeedback.id)).scalar() or 0
            helpful_count = db.query(func.count(AnswerFeedback.id)).filter(
                AnswerFeedback.is_helpful == True
            ).scalar() or 0
            
            # Calculate helpful percentage
            helpful_percentage = 0
            if total_count > 0:
                helpful_percentage = (helpful_count / total_count) * 100
            
            # Get counts by source type
            source_counts = {}
            for source_type in ["kb", "pdf", "web", "chatgpt"]:
                count = db.query(func.count(AnswerFeedback.id)).filter(
                    AnswerFeedback.source_type == source_type
                ).scalar() or 0
                
                source_counts[source_type] = count
            
            return {
                "total_feedback": total_count,
                "helpful_count": helpful_count,
                "not_helpful_count": total_count - helpful_count,
                "helpful_percentage": helpful_percentage,
                "by_source": source_counts,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting feedback stats: {str(e)}")
            
            return {
                "status": "error",
                "message": f"Error getting feedback stats: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }

# Initialize the handler
feedback_handler = FeedbackHandler()
