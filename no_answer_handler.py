from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

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
    
    def suggest_related_questions(self, query: str, openai_service) -> List[str]:
        """
        Generate suggested alternative questions when no answer is found
        
        Parameters:
        - query: The original query
        - openai_service: Service for generating suggestions
        
        Returns:
        - List of suggested questions
        """
        try:
            if hasattr(openai_service, 'openai_configured') and openai_service.openai_configured:
                import openai
                
                prompt = f"""The user asked: "{query}"
                
                I couldn't find a good answer to this question. Please suggest 3 alternative questions that:
                1. Are related to the original question
                2. Might be more likely to have answers in a knowledge base
                3. Are more specific or clearer
                
                Format your response as a simple list of questions, one per line.
                """
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that suggests alternative questions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.7
                )
                
                suggestions_text = response['choices'][0]['message']['content'].strip()
                
                # Parse the suggestions (assuming one per line)
                suggestions = [line.strip() for line in suggestions_text.split('\n') if line.strip()]
                
                # Remove any numbering or bullet points
                suggestions = [re.sub(r'^[\d\-\*\•]+\.?\s*', '', q) for q in suggestions]
                
                # Remove quotes if present
                suggestions = [q.strip('"\'') for q in suggestions]
                
                logger.info(f"✅ Generated {len(suggestions)} alternative questions")
                return suggestions[:3]  # Limit to 3 suggestions
                
            else:
                # Demo mode - create generic suggestions
                base_suggestions = [
                    f"What are the key characteristics of {query}?",
                    f"Can you explain the history of {query}?",
                    f"What are some examples of {query} in practice?"
                ]
                
                logger.info(f"🎭 Generated demo alternative questions")
                return base_suggestions
                
        except Exception as e:
            logger.error(f"❌ Error generating question suggestions: {str(e)}")
            
            # Fallback suggestions
            return [
                f"What is {query}?",
                f"Can you provide more information about {query}?",
                f"What are the main aspects of {query}?"
            ]
    
    def log_unanswered_question(self, db, query: str, search_attempts: List[str]) -> None:
        """
        Log questions that couldn't be answered for later analysis
        
        Parameters:
        - db: Database session
        - query: The original query
        - search_attempts: List of places searched
        """
        try:
            # Import here to avoid circular imports
            from override_models import UnansweredQuestion
            
            # Create record
            unanswered = UnansweredQuestion(
                question_text=query,
                search_attempts=search_attempts,
                created_at=datetime.utcnow()
            )
            
            db.add(unanswered)
            db.commit()
            
            logger.info(f"📝 Logged unanswered question: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error logging unanswered question: {str(e)}")
            # Don't raise - this is a non-critical operation

# Initialize the handler
import re  # Added for regex pattern matching
no_answer_handler = NoAnswerHandler()
