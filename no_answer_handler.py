import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoAnswerHandler:
    """Handler for cases where no answer is found"""
    
    def __init__(self):
        """Initialize the no answer handler"""
        logger.info("🤔 Initializing No Answer Handler")
    
    def log_unanswered_question(self, db: Session, question_text: str, search_attempts: List[str]) -> None:
        """
        Log an unanswered question for later analysis
        
        Args:
            db: Database session
            question_text: The question that couldn't be answered
            search_attempts: List of search methods attempted
        """
        try:
            from override_models import UnansweredQuestion
            
            unanswered = UnansweredQuestion(
                question_text=question_text,
                search_attempts=search_attempts
            )
            
            db.add(unanswered)
            db.commit()
            
            logger.info(f"📝 Logged unanswered question: {question_text[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error logging unanswered question: {str(e)}")
    
    def suggest_related_questions(self, question_text: str, rag_service) -> List[str]:
        """
        Suggest related questions that might be answerable
        
        Args:
            question_text: The original question
            rag_service: The RAG service for generating suggestions
            
        Returns:
            List of suggested alternative questions
        """
        try:
            logger.info(f"🔄 Generating alternative questions for: {question_text[:50]}...")
            
            # In a real implementation, this would use the RAG service
            # For now, we'll return some generic suggestions
            suggestions = [
                f"What is the definition of {question_text.split()[-1]}?",
                f"Can you explain {question_text} in simpler terms?",
                f"What are the key concepts related to {question_text}?"
            ]
            
            return suggestions
            
        except Exception as e:
            logger.error(f"❌ Error suggesting related questions: {str(e)}")
            return []

# Create a singleton instance
no_answer_handler = NoAnswerHandler()
