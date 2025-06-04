from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional
import os
import uuid
from datetime import datetime
import uvicorn
from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager
import time

# Load environment variables
load_dotenv()

# Import our modules
from database import SessionLocal, engine, get_db, Base
from models import Question, Answer, Embedding, KnowledgeBase
from document_models import Document, DocumentChunk
from feedback_models import AnswerFeedback
from knowledge_base_service import knowledge_service
from schemas import (
    QuestionAnswerRequest, 
    QuestionAnswerResponse, 
    HealthResponse, 
    AuthHealthResponse,
    QuestionResponse,
    AnswerResponse,
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    KnowledgeBaseSearch
)
from feedback_schemas import FeedbackCreate, FeedbackResponse, QuestionSuggestion
from rag_service import RAGService
from simple_pdf_processor import SimplePDFProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG service
rag_service = RAGService()

# Create database tables
Base.metadata.create_all(bind=engine)

# Direct CRUD operations (avoiding import issues)
def create_question_direct(db: Session, question_text: str) -> Question:
    """Create a new question directly"""
    db_question = Question(text=question_text)
    db.add(db_question)
    db.commit()
    db.refresh(db_question)
    return db_question

def create_answer_direct(db: Session, question_id: int, answer_text: str, confidence_score: float = 0.9) -> Answer:
    """Create a new answer directly"""
    db_answer = Answer(
        question_id=question_id,
        text=answer_text,
        confidence_score=confidence_score
    )
    db.add(db_answer)
    db.commit()
    db.refresh(db_answer)
    return db_answer

def create_embedding_direct(db: Session, question_id: int, embedding_vector: List[float]) -> Embedding:
    """Create a new embedding directly"""
    db_embedding = Embedding(
        question_id=question_id,
        embedding=embedding_vector
    )
    db.add(db_embedding)
    db.commit()
    db.refresh(db_embedding)
    return db_embedding

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Enhanced PDF RAG Q&A API...")
    
    # Check DATABASE_URL
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        logger.info(f"📊 Database URL configured: {database_url[:30]}...")
        try:
            db = SessionLocal()
            result = db.execute(text("SELECT 1"))
            result.fetchone()
            db.close()
            logger.info("✅ Database connection successful!")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {str(e)}")
    else:
        logger.error("❌ DATABASE_URL not configured!")
    
    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        logger.info("🤖 OpenAI API Key configured - RAG enabled")
    else:
        logger.info("🎭 Running in demo mode - RAG with mock responses")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Enhanced PDF RAG Q&A API...")

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced PDF RAG Q&A API",
    description="A comprehensive FastAPI backend for PDF document processing and Q&A using RAG with user-controlled saving",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# API Keys for development/demo
VALID_API_KEYS = {
    "dev-api-key-123",
    "test-api-key-456", 
    "demo-key-789",
    "qa-development-key",
    "master-dev-key"
}

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key from Authorization header"""
    api_key = credentials.credentials
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced PDF RAG Q&A API with Knowledge-Base-First Logic",
        "version": "3.1.0",
        "status": "running",
        "logic": "Knowledge Base → PDF search → Q&A database → ChatGPT",
        "features": [
            "Built-in knowledge base (fastest responses)",
            "PDF document processing and search",
            "User-controlled answer saving",
            "Q&A database fallback",
            "ChatGPT-3.5 generation",
            "Answer feedback system",
            "Question rephrasing suggestions"
        ],
        "endpoints": {
            "health": "/health",
            "ask": "/ask",
            "save-answer": "/save-answer",
            "knowledge-base": "/knowledge-base",
            "knowledge-search": "/knowledge-base/search",
            "feedback": "/feedback",
            "suggest": "/suggest-questions",
            "documents": "/documents",
            "upload": "/documents/upload",
            "stats": "/stats",
            "qa-cache": "/qa-cache"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        db.execute(text("SELECT 1"))
        database_connected = True
    except Exception:
        database_connected = False
    
    return HealthResponse(
        status="healthy",
        message="Enhanced RAG API is running with PDF-first logic",
        timestamp=datetime.utcnow(),
        database_connected=database_connected,
        openai_configured=rag_service.openai_configured
    )

@app.post("/ask")
async def ask_question(
    request: QuestionAnswerRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Ask a question with updated logic prioritizing built-in knowledge base:
    1. FIRST: Check built-in knowledge base for common questions (fastest, most accurate)
    2. If not found: Search PDF documents using RAG
    3. If not found: Check Q&A database for similar questions  
    4. If not found: Use ChatGPT-3.5 as final fallback
    
    This ensures fastest response times and most accurate answers for known topics.
    """
    try:
        question_text = request.question.strip()
        start_time = time.time()
        
        if not question_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        print(f"🔍 Processing question with knowledge-base-first logic: {question_text[:50]}...")
        
        # STEP 1: FIRST, check built-in knowledge base (fastest, most accurate)
        print("🧠 Step 1: Checking built-in knowledge base first...")
        kb_match = knowledge_service.get_best_knowledge_match(db, question_text)
        
        if kb_match:
            print(f"✅ Found answer in knowledge base (score: {kb_match['relevance_score']:.3f})")
            generation_time = int((time.time() - start_time) * 1000)
            
            return {
                "answer": kb_match["answer"],
                "question_id": None,  # Knowledge base entries don't have question IDs
                "answer_id": None,
                "similarity_score": kb_match["relevance_score"],
                "is_cached": True,
                "source_documents": [],
                "answer_type": "knowledge_base",
                "confidence_score": 0.95,  # High confidence for knowledge base
                "generation_time_ms": generation_time,
                "found_in_knowledge_base": True,
                "knowledge_base_category": kb_match["category"],
                "knowledge_base_id": kb_match["id"],
                "show_save_prompt": False,  # Don't save knowledge base answers
                "cache_hit": True,
                "match_type": kb_match["match_type"]
            }
        
        # Generate embedding for remaining steps
        query_embedding = await rag_service.generate_embedding(question_text)
        
        # STEP 2: If not in knowledge base, search PDF documents
        print("📄 Step 2: Knowledge base miss, searching PDF documents...")
        relevant_chunks = await rag_service.search_document_chunks(db, query_embedding, limit=5)
        
        if relevant_chunks:
            print(f"✅ Found {len(relevant_chunks)} relevant chunks in PDF documents")
            
            # Generate answer from PDF content
            answer_text = await rag_service.generate_answer_from_chunks(question_text, relevant_chunks)
            source_docs = list(set([chunk.get('filename', 'Unknown') for chunk in relevant_chunks]))
            
            generation_time = int((time.time() - start_time) * 1000)
            
            return {
                "answer": answer_text,
                "question_id": None,  # Not saved yet
                "answer_id": None,    # Not saved yet
                "similarity_score": 0.0,
                "is_cached": False,
                "source_documents": source_docs,
                "answer_type": "pdf_document",
                "confidence_score": 0.9,
                "generation_time_ms": generation_time,
                "found_in_pdf": True,
                "show_save_prompt": True,
                "save_prompt_message": "Do you want to save this answer to the Q&A database?",
                "temp_question": question_text,
                "temp_answer": answer_text,
                "temp_sources": source_docs,
                "cache_hit": False
            }
        
        # STEP 3: If not in PDF, check Q&A database
        print("🗃️ Step 3: No PDF match, checking Q&A database...")
        similar_question, similarity = await rag_service.find_similar_question(db, query_embedding)
        
        if similar_question and similarity > 0.75:
            print(f"✅ Found similar question in database (similarity: {similarity:.3f})")
            
            latest_answer = db.query(Answer).filter(
                Answer.question_id == similar_question.id
            ).order_by(Answer.created_at.desc()).first()
            
            if latest_answer:
                generation_time = int((time.time() - start_time) * 1000)
                
                answer_type = "database_exact_match" if similarity > 0.95 else "database_adapted"
                
                return {
                    "answer": latest_answer.text,
                    "question_id": similar_question.id,
                    "answer_id": latest_answer.id,
                    "similarity_score": similarity,
                    "is_cached": True,
                    "source_documents": [],
                    "answer_type": answer_type,
                    "confidence_score": latest_answer.confidence_score,
                    "generation_time_ms": generation_time,
                    "found_in_knowledge_base": False,
                    "found_in_pdf": False,
                    "show_save_prompt": False,
                    "cache_hit": True,
                    "original_question": similar_question.text if similarity < 0.95 else None
                }
        
        # STEP 4: Final fallback to ChatGPT-3.5
        print("🤖 Step 4: Using ChatGPT-3.5 as final fallback...")
        answer_text = await rag_service.generate_answer(question_text)
        
        generation_time = int((time.time() - start_time) * 1000)
        
        return {
            "answer": answer_text,
            "question_id": None,
            "answer_id": None,
            "similarity_score": 0.0,
            "is_cached": False,
            "source_documents": [],
            "answer_type": "chatgpt_generated",
            "confidence_score": 0.7,
            "generation_time_ms": generation_time,
            "found_in_knowledge_base": False,
            "found_in_pdf": False,
            "show_save_prompt": True,
            "save_prompt_message": "Do you want to save this answer to the Q&A database?",
            "temp_question": question_text,
            "temp_answer": answer_text,
            "temp_sources": [],
            "cache_hit": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )

@app.post("/save-answer")
async def save_answer_to_knowledge_base(
    save_request: dict,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Save an answer ONLY to the knowledge base (NOT Q&A database)
    """
    try:
        print(f"🔥 SAVE-ANSWER ENDPOINT CALLED - KNOWLEDGE BASE ONLY")
        print(f"📥 Full save request: {save_request}")
        
        question_text = save_request.get("question")
        answer_text = save_request.get("answer")
        answer_type = save_request.get("answer_type", "user_saved")
        
        if not question_text or not answer_text:
            print("❌ Missing question or answer text")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question and answer are required"
            )
        
        print(f"🎯 SAVING TO KNOWLEDGE BASE ONLY (NO Q&A DATABASE)")
        print(f"📝 Question: {question_text[:100]}...")
        print(f"📝 Answer: {answer_text[:100]}...")
        
        # Determine category
        category = "User Generated"
        if answer_type == "pdf_document":
            category = "Document Based"
        elif answer_type == "chatgpt_generated":
            category = "AI Generated"
        
        # Extract keywords
        question_words = question_text.lower().split()
        stop_words = {"what", "how", "why", "when", "where", "is", "are", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word.strip(".,!?;:") for word in question_words if word.lower() not in stop_words and len(word) > 2]
        
        # Create knowledge base entry directly in database (bypass service for debugging)
        print("🚀 CREATING KNOWLEDGE BASE ENTRY DIRECTLY...")
        
        try:
            # Create the knowledge base entry directly
            kb_entry = KnowledgeBase(
                category=category,
                question=question_text,
                answer=answer_text,
                keywords=keywords[:10],
                priority=5,
                is_active=True,
                created_by="user_save"
            )
            
            db.add(kb_entry)
            db.commit()
            db.refresh(kb_entry)
            
            print(f"✅ SUCCESS! Knowledge base entry created with ID: {kb_entry.id}")
            print(f"✅ Category: {kb_entry.category}")
            print(f"✅ Keywords: {kb_entry.keywords}")
            
            # RETURN KNOWLEDGE BASE FORMAT (NOT Q&A FORMAT)
            response = {
                "message": "Answer saved successfully to knowledge base",
                "knowledge_base_id": kb_entry.id,  # This is the key field
                "category": kb_entry.category,
                "keywords": kb_entry.keywords,
                "priority": kb_entry.priority,
                "saved_at": datetime.utcnow().isoformat(),
                "success": True,
                "save_location": "knowledge_base"
            }
            
            print(f"🎉 RETURNING KNOWLEDGE BASE RESPONSE: {response}")
            return response
            
        except Exception as kb_error:
            print(f"💥 KNOWLEDGE BASE SAVE FAILED: {str(kb_error)}")
            import traceback
            print(f"💥 Full traceback: {traceback.format_exc()}")
            
            # Check if knowledge_base table exists
            try:
                table_exists = db.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'knowledge_base'
                """)).fetchone()
                
                if not table_exists:
                    print("💥 KNOWLEDGE_BASE TABLE DOES NOT EXIST!")
                    return {
                        "error": "knowledge_base table does not exist",
                        "message": "Please run database migration to create knowledge_base table",
                        "success": False
                    }
                else:
                    print("✅ knowledge_base table exists")
                    
            except Exception as table_check_error:
                print(f"💥 Error checking table: {table_check_error}")
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save to knowledge base: {str(kb_error)}"
            )
        
    except HTTPException as he:
        print(f"❌ HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        print(f"💥 UNEXPECTED ERROR: {str(e)}")
        import traceback
        print(f"💥 Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in save-answer endpoint: {str(e)}"
        )

@app.post("/debug/save-test")
async def debug_save_test(
    save_request: dict,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Debug endpoint to test knowledge base saving"""
    try:
        print(f"🧪 DEBUG SAVE TEST CALLED")
        print(f"📥 Request: {save_request}")
        
        # Test knowledge base creation directly
        test_entry = KnowledgeBaseCreate(
            category="Debug Test",
            question="Test question from debug endpoint",
            answer="Test answer from debug endpoint",
            keywords=["test", "debug"],
            priority=1,
            is_active=True
        )
        
        print("🧪 Testing knowledge_service.create_knowledge_entry...")
        new_entry = knowledge_service.create_knowledge_entry(db, test_entry, created_by="debug_test")
        print(f"✅ Debug test successful! Created KB entry {new_entry.id}")
        
        return {
            "debug_test": "success",
            "knowledge_base_id": new_entry.id,
            "message": "Knowledge base save test successful"
        }
        
    except Exception as e:
        print(f"💥 Debug test failed: {str(e)}")
        import traceback
        print(f"💥 Traceback: {traceback.format_exc()}")
        return {
            "debug_test": "failed",
            "error": str(e),
            "message": "Knowledge base save test failed"
        }

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Submit feedback for an answer"""
    try:
        # Verify question and answer exist
        question = db.query(Question).filter(Question.id == feedback.question_id).first()
        answer = db.query(Answer).filter(Answer.id == feedback.answer_id).first()
        
        if not question or not answer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question or answer not found"
            )
        
        # Create feedback record
        db_feedback = AnswerFeedback(
            question_id=feedback.question_id,
            answer_id=feedback.answer_id,
            is_helpful=feedback.is_helpful,
            feedback_text=feedback.feedback_text,
            confidence_score=feedback.confidence_score,
            answer_type=feedback.answer_type,
            user_session=feedback.user_session
        )
        
        db.add(db_feedback)
        db.commit()
        db.refresh(db_feedback)
        
        print(f"📝 Feedback recorded: {'👍' if feedback.is_helpful else '👎'} for Q{feedback.question_id}")
        
        return FeedbackResponse(
            id=db_feedback.id,
            question_id=db_feedback.question_id,
            answer_id=db_feedback.answer_id,
            is_helpful=db_feedback.is_helpful,
            feedback_text=db_feedback.feedback_text,
            confidence_score=db_feedback.confidence_score,
            answer_type=db_feedback.answer_type,
            created_at=db_feedback.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error submitting feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error submitting feedback: {str(e)}"
        )

@app.post("/reviews")
async def create_review(
    review_data: dict,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Create a review for an answer"""
    try:
        print(f"📝 Creating review: {review_data}")
        
        # For now, just log the review - you can enhance this later
        review_id = int(time.time())  # Simple ID generation
        
        return {
            "id": review_id,
            "message": "Review created successfully",
            "review_data": review_data,
            "created_at": datetime.utcnow()
        }
        
    except Exception as e:
        print(f"❌ Error creating review: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating review: {str(e)}"
        )

@app.get("/reviews/queue")
async def get_review_queue(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get answers that need review"""
    try:
        # Get low confidence answers that need review
        result = db.execute(text("""
            SELECT 
                q.id as question_id,
                q.text as question_text,
                a.id as answer_id,
                a.text as answer_text,
                a.confidence_score,
                a.created_at
            FROM questions q
            JOIN answers a ON q.id = a.question_id
            WHERE a.confidence_score < 0.8
            ORDER BY a.created_at DESC
            LIMIT 20
        """))
        
        queue_items = []
        for row in result:
            queue_items.append({
                "question_id": row.question_id,
                "question_text": row.question_text,
                "answer_id": row.answer_id,
                "answer_text": row.answer_text,
                "confidence_score": float(row.confidence_score),
                "created_at": row.created_at,
                "needs_review": True
            })
        
        return queue_items
        
    except Exception as e:
        print(f"❌ Error getting review queue: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting review queue: {str(e)}"
        )

@app.post("/overrides")
async def create_override(
    override_data: dict,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Create an override for an answer"""
    try:
        print(f"🔄 Creating override: {override_data}")
        
        # For now, just log the override - you can enhance this later
        override_id = int(time.time())  # Simple ID generation
        
        return {
            "id": override_id,
            "message": "Override created successfully",
            "override_data": override_data,
            "created_at": datetime.utcnow()
        }
        
    except Exception as e:
        print(f"❌ Error creating override: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating override: {str(e)}"
        )

@app.get("/overrides")
async def list_overrides(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """List all overrides"""
    try:
        # Return empty list for now - you can enhance this later
        return []
        
    except Exception as e:
        print(f"❌ Error listing overrides: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing overrides: {str(e)}"
        )

@app.get("/cache/stats")
async def get_cache_stats(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get cache statistics"""
    try:
        # Get basic stats
        question_count = db.execute(text("SELECT COUNT(*) FROM questions")).fetchone()[0]
        answer_count = db.execute(text("SELECT COUNT(*) FROM answers")).fetchone()[0]
        
        return {
            "total_questions": question_count,
            "total_answers": answer_count,
            "cache_hit_rate": 0.75,  # Mock data
            "avg_response_time": 250,  # Mock data
            "message": "Cache statistics retrieved"
        }
        
    except Exception as e:
        print(f"❌ Error getting cache stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting cache stats: {str(e)}"
        )

@app.post("/suggest-questions", response_model=QuestionSuggestion)
async def suggest_questions(
    request: QuestionAnswerRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get AI suggestions for rephrasing unclear questions"""
    try:
        original_question = request.question.strip()
        
        if not original_question:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        # Generate suggestions (simplified for demo)
        suggestions = [
            f"What is {original_question.split()[-1] if original_question.split() else 'the topic'}?",
            f"How does {original_question.split()[0] if original_question.split() else 'this'} work?",
            f"Can you explain {original_question.lower()}?"
        ]
        
        return QuestionSuggestion(
            original_question=original_question,
            suggested_questions=suggestions,
            reasoning="AI-generated suggestions to improve question clarity."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating question suggestions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating suggestions: {str(e)}"
        )

# Knowledge Base Management Endpoints
@app.post("/knowledge-base", response_model=KnowledgeBaseResponse)
async def create_knowledge_entry(
    entry: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Create a new knowledge base entry"""
    try:
        db_entry = knowledge_service.create_knowledge_entry(db, entry, created_by="api_user")
        return db_entry
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating knowledge base entry: {str(e)}"
        )

@app.get("/knowledge-base")
async def list_knowledge_entries(
    category: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """List knowledge base entries"""
    try:
        from models import KnowledgeBase
        
        query = db.query(KnowledgeBase).filter(KnowledgeBase.is_active == True)
        
        if category:
            query = query.filter(KnowledgeBase.category == category)
        
        entries = query.order_by(KnowledgeBase.priority.desc(), KnowledgeBase.created_at.desc()).offset(skip).limit(limit).all()
        
        return {
            "entries": entries,
            "total": len(entries),
            "category_filter": category
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing knowledge base entries: {str(e)}"
        )

@app.get("/knowledge-base/categories")
async def get_knowledge_categories(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get all knowledge base categories"""
    try:
        categories = knowledge_service.get_categories(db)
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting categories: {str(e)}"
        )

@app.post("/knowledge-base/search")
async def search_knowledge_base(
    search_request: KnowledgeBaseSearch,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Search knowledge base entries"""
    try:
        results = knowledge_service.search_knowledge_base(
            db, 
            search_request.query,
            search_request.category,
            search_request.limit
        )
        return {
            "results": results,
            "query": search_request.query,
            "total_found": len(results)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching knowledge base: {str(e)}"
        )

@app.put("/knowledge-base/{entry_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_entry(
    entry_id: int,
    update_data: KnowledgeBaseUpdate,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Update a knowledge base entry"""
    try:
        updated_entry = knowledge_service.update_knowledge_entry(db, entry_id, update_data)
        if not updated_entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base entry {entry_id} not found"
            )
        return updated_entry
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating knowledge base entry: {str(e)}"
        )

@app.delete("/knowledge-base/{entry_id}")
async def delete_knowledge_entry(
    entry_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Delete a knowledge base entry"""
    try:
        success = knowledge_service.delete_entry(db, entry_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base entry {entry_id} not found"
            )
        return {"message": f"Knowledge base entry {entry_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting knowledge base entry: {str(e)}"
        )

@app.get("/knowledge-base/stats")
async def get_knowledge_stats(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get knowledge base statistics"""
    try:
        stats = knowledge_service.get_stats(db)
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting knowledge base stats: {str(e)}"
        )

# Document Management Endpoints
@app.get("/documents")
async def list_documents(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """List all uploaded documents with enhanced debugging"""
    try:
        print("🔍 Fetching documents from database...")
        
        # First, check if documents table exists and has data
        from models import Document  # Import Document model here
        
        # Check if the documents table exists
        has_table = engine.dialect.has_table(engine.connect(), Document.__tablename__)
        if not has_table:
            print("⚠️ Documents table does not exist.")
            return {"message": "No documents found - table does not exist", "documents": [], "total": 0}
        
        # Get total count of documents
        total_documents = db.query(Document).count()
        
        if total_documents == 0:
            print("ℹ️ No documents found in database.")
            return {"message": "No documents found", "documents": [], "total": 0}
        
        # Fetch all documents
        documents = db.query(Document).all()
        
        # Convert documents to a list of dictionaries for easier serialization
        document_list = [{"id": doc.id, "filename": doc.filename, "upload_date": doc.upload_date} for doc in documents]
        
        print(f"✅ Found {len(documents)} documents.")
        
        return {
            "message": f"Found {len(documents)} documents",
            "documents": document_list,
            "total": total_documents
        }
        
    except Exception as e:
        print(f"❌ Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )

@app.get("/debug/knowledge-base-table")
async def debug_knowledge_base_table(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Debug endpoint to check if knowledge_base table exists and is accessible"""
    try:
        print("🔍 Checking knowledge_base table...")
        
        # Check if table exists
        table_exists = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'knowledge_base'
        """)).fetchone()
        
        if not table_exists:
            print("❌ knowledge_base table does not exist")
            return {
                "table_exists": False,
                "error": "knowledge_base table not found",
                "suggestion": "Run: python scripts/create_knowledge_base_table.py"
            }
        
        print("✅ knowledge_base table exists")
        
        # Check table structure
        columns = db.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'knowledge_base'
            ORDER BY ordinal_position
        """)).fetchall()
        
        # Try to count entries
        count = db.execute(text("SELECT COUNT(*) FROM knowledge_base")).fetchone()[0]
        
        # Try to insert a test entry
        test_entry = KnowledgeBase(
            category="Debug Test",
            question="Test question for debugging",
            answer="Test answer for debugging",
            keywords=["test", "debug"],
            priority=1,
            is_active=True,
            created_by="debug"
        )
        
        db.add(test_entry)
        db.commit()
        db.refresh(test_entry)
        
        print(f"✅ Test entry created with ID: {test_entry.id}")
        
        return {
            "table_exists": True,
            "columns": [{"name": col[0], "type": col[1]} for col in columns],
            "entry_count": count,
            "test_entry_id": test_entry.id,
            "message": "knowledge_base table is working correctly"
        }
        
    except Exception as e:
        print(f"❌ Error checking knowledge_base table: {str(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        return {
            "table_exists": False,
            "error": str(e),
            "message": "Error accessing knowledge_base table"
        }
