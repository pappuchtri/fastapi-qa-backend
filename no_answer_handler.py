from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from sqlalchemy.orm import Session
import re  # Added for regex pattern matching

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoAnswerHandler:
    """Handler for gracefully managing cases where no answer is found"""
    
    def __init__(self):
        """Initialize the no answer handler"""
        logger.info("🤔 No Answer Handler initialized")
    
    def generate_no_answer_response(
        self, 
        query: str,
        search_attempts: List[str],
        suggested_actions: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response for when no answer is found
        
        Parameters:
        - query: The original query
        - search_attempts: List of places searched (e.g., ["knowledge base", "PDF documents", "web"])
        - suggested_actions: Optional list of suggested actions
        
        Returns:
        - Dictionary with response information
        """
        # Default suggested actions if none provided
        if not suggested_actions:
            suggested_actions = [
                {
                    "action": "rephrase",
                    "description": "Try rephrasing your question with more specific details"
                },
                {
                    "action": "upload",
                    "description": "Upload relevant documents that might contain the answer"
                },
                {
                    "action": "feedback",
                    "description": "Provide feedback to help improve the system"
                }
            ]
        
        # Create a helpful response
        searched_places = ", ".join(search_attempts)
        
        response = {
            "answer_type": "no_answer",
            "answer": f"I'm sorry, but I couldn't find a specific answer to your question about '{query}'. I searched {searched_places}, but didn't find relevant information.",
            "query": query,
            "search_attempts": search_attempts,
            "suggested_actions": suggested_actions,
            "show_feedback_form": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"⚠️ Generated no-answer response for query: {query[:50]}...")
        return response
    
    def suggest_related_questions(
        self, 
        original_question: str, 
        rag_service
    ) -> List[str]:
        """Suggest related questions that might help the user"""
        try:
            # Generate alternative question suggestions
            prompt = f"""Based on the question "{original_question}", suggest 3 related questions that might help the user find the information they're looking for. Make the suggestions more specific and actionable.

Format as a simple list:
1. [suggestion 1]
2. [suggestion 2] 
3. [suggestion 3]"""
            
            # This would use the RAG service to generate suggestions
            # For now, return some generic helpful suggestions
            suggestions = [
                f"What specific aspect of '{original_question.split()[0] if original_question.split() else 'this topic'}' are you most interested in?",
                f"Are you looking for general information or specific details about '{original_question.split()[-1] if original_question.split() else 'this topic'}'?",
                f"Would you like to know about related topics or similar concepts?"
            ]
            
            return suggestions
            
        except Exception as e:
            logger.error(f"❌ Error generating question suggestions: {str(e)}")
            return [
                "Try rephrasing your question with more specific terms",
                "Consider breaking down your question into smaller parts",
                "Upload relevant documents that might contain the information you need"
            ]
    
    def log_unanswered_question(
        self, 
        db: Session, 
        question: str, 
        search_attempts: List[str]
    ) -> None:
        """Log questions that couldn't be answered satisfactorily"""
        try:
            from override_models import UnansweredQuestion
            
            unanswered = UnansweredQuestion(
                question_text=question,
                search_attempts=search_attempts,
                timestamp=datetime.utcnow()
            )
            
            db.add(unanswered)
            db.commit()
            
            logger.info(f"📝 Logged unanswered question: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error logging unanswered question: {str(e)}")
    
    def get_feedback_form_data(self, question: str) -> Dict[str, Any]:
        """Get data for the feedback form when no answer is found"""
        return {
            "show_feedback_form": True,
            "feedback_message": "We couldn't find a satisfactory answer to your question. Your feedback helps us improve!",
            "feedback_options": [
                {"value": "too_specific", "label": "Question was too specific"},
                {"value": "too_broad", "label": "Question was too broad"},
                {"value": "missing_context", "label": "Missing important context"},
                {"value": "technical_issue", "label": "Technical issue with search"},
                {"value": "other", "label": "Other (please specify)"}
            ],
            "suggestion_prompt": "What information were you hoping to find?",
            "alternative_actions": [
                "Upload relevant documents",
                "Try a web search",
                "Contact support",
                "Browse knowledge base categories"
            ]
        }

# Global instance
no_answer_handler = NoAnswerHandler()
