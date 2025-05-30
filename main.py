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
from models import Question, Answer, Embedding
from document_models import Document, DocumentChunk
from feedback_models import AnswerFeedback
from override_models import User, AnswerOverride, AnswerReview, PerformanceCache, UserRole, ReviewStatus
from schemas import (
    QuestionAnswerRequest, 
    QuestionAnswerResponse, 
    HealthResponse, 
    AuthHealthResponse,
    QuestionResponse,
    AnswerResponse
)
from override_schemas import (
    UserCreate, UserResponse, OverrideCreate, OverrideResponse, 
    ReviewCreate, ReviewResponse, EnhancedQuestionAnswerResponse,
    ReviewQueueItem, ReviewStats
)
from feedback_schemas import FeedbackCreate, FeedbackResponse, QuestionSuggestion
from enhanced_rag_service import EnhancedRAGService
import override_crud
import crud
from simple_pdf_processor import SimplePDFProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Enhanced RAG service
rag_service = EnhancedRAGService()

# Create database tables
Base.metadata.create_all(bind=engine)

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
    description="A comprehensive FastAPI backend for PDF document processing and Q&A using RAG with human override system, performance caching, and reviewer workflow",
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

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> User:
    """Get current user from API key or session"""
    api_key = credentials.credentials
    
    # For demo purposes, create a default admin user if none exists
    admin_user = override_crud.get_user_by_username(db, "admin")
    if not admin_user:
        admin_user = override_crud.create_user(
            db, 
            UserCreate(username="admin", email="admin@example.com", role=UserRole.ADMIN)
        )
    
    return admin_user

def require_role(required_role: UserRole):
    """Decorator to require specific user role"""
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role.value != required_role.value and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {required_role.value} role"
            )
        return current_user
    return role_checker

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced PDF RAG Q&A API",
        "version": "3.0.0",
        "status": "running",
        "features": [
            "PDF document processing with chunk limiting (3-5 chunks)",
            "Vector similarity search",
            "Historical Q&A integration",
            "Answer feedback system",
            "Human override system for compliance",
            "Reviewer workflow with insufficient answer tagging",
            "Performance caching for slow-generating answers",
            "Low confidence flagging",
            "Question rephrasing suggestions",
            "GPT fallback for unknown questions",
            "Role-based access control (user/reviewer/admin)"
        ],
        "endpoints": {
            "health": "/health",
            "ask": "/ask",
            "feedback": "/feedback",
            "suggest": "/suggest-questions",
            "documents": "/documents",
            "upload": "/documents/upload",
            "stats": "/stats",
            "qa-cache": "/qa-cache",
            "debug/chunks": "/debug/chunks",
            "users": "/users",
            "overrides": "/overrides",
            "reviews": "/reviews",
            "cache": "/cache"
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
        message="Enhanced RAG API is running",
        timestamp=datetime.utcnow(),
        database_connected=database_connected,
        openai_configured=rag_service.openai_configured
    )

@app.post("/ask", response_model=EnhancedQuestionAnswerResponse)
async def ask_question(
    request: QuestionAnswerRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Ask a question with enhanced features: chunk limiting, performance caching, override support"""
    try:
        question_text = request.question.strip()
        start_time = time.time()
        
        if not question_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        print(f"🔍 Processing question: {question_text[:50]}...")
        
        # STEP 1: Check performance cache first
        cached_result = await rag_service.check_performance_cache(db, question_text)
        if cached_result:
            print("⚡ Returning performance-cached answer")
            
            # Check for active override
            question_id = None
            override = None
            
            # Try to find existing question for override check
            query_embedding = await rag_service.generate_embedding(question_text)
            similar_question, similarity = await rag_service.find_similar_question(db, query_embedding)
            if similar_question and similarity > 0.9:  # Very high similarity
                question_id = similar_question.id
                override = override_crud.get_active_override(db, question_id)
            
            if override:
                return EnhancedQuestionAnswerResponse(
                    answer=override.override_text,
                    question_id=question_id,
                    answer_id=override.original_answer_id,
                    similarity_score=1.0,
                    is_cached=True,
                    source_documents=[],
                    low_confidence=False,
                    answer_type="override",
                    confidence_score=1.0,
                    has_override=True,
                    override_id=override.id,
                    needs_review=False,
                    review_flags=[],
                    generation_time_ms=0,
                    chunk_count=0
                )
            
            return EnhancedQuestionAnswerResponse(
                answer=cached_result["answer"],
                question_id=question_id or 0,
                answer_id=0,
                similarity_score=1.0,
                is_cached=True,
                source_documents=[],
                low_confidence=cached_result["confidence"] < 0.8,
                answer_type=cached_result["source_type"],
                confidence_score=cached_result["confidence"],
                has_override=False,
                override_id=None,
                needs_review=cached_result["confidence"] < 0.8,
                review_flags=["low_confidence"] if cached_result["confidence"] < 0.8 else [],
                generation_time_ms=0,
                chunk_count=0
            )
        
        # STEP 2: Generate embedding and analyze question
        query_embedding = await rag_service.generate_embedding(question_text)
        question_analysis = await rag_service.analyze_question_intent(question_text)
        
        # STEP 3: Process with enhanced RAG
        result = await rag_service.process_question(
            db, question_text, query_embedding, question_analysis
        )
        
        generation_time_ms = int((time.time() - start_time) * 1000)
        
        # STEP 4: Store or create question/answer records
        question_id = result.get("question_id")
        if not question_id:
            # Create new question
            new_question = crud.create_question(db, crud.QuestionCreate(text=question_text))
            question_id = new_question.id
            
            # Store embedding
            crud.create_embedding(db, question_id, query_embedding)
            
            # Store answer
            new_answer = crud.create_answer(
                db, 
                crud.AnswerCreate(
                    question_id=question_id,
                    text=result["answer"],
                    confidence_score=result["confidence"]
                )
            )
            answer_id = new_answer.id
        else:
            # Use existing question/answer
            answer_id = result.get("answer_id", 0)
        
        # STEP 5: Check for active override
        override = override_crud.get_active_override(db, question_id)
        if override:
            print(f"🔄 Using human override for question {question_id}")
            return EnhancedQuestionAnswerResponse(
                answer=override.override_text,
                question_id=question_id,
                answer_id=answer_id,
                similarity_score=result.get("similarity", 0.0),
                is_cached=False,
                source_documents=result.get("source_documents", []),
                low_confidence=False,
                answer_type="override",
                confidence_score=1.0,
                has_override=True,
                override_id=override.id,
                needs_review=False,
                review_flags=[],
                generation_time_ms=generation_time_ms,
                chunk_count=result.get("total_chunks_found", 0)
            )
        
        # STEP 6: Store in performance cache if slow
        await rag_service.store_performance_cache(
            db, question_text, result["answer"], generation_time_ms,
            result["confidence"], result["source_type"]
        )
        
        # STEP 7: Determine review flags
        review_flags = []
        needs_review = False
        
        if result["confidence"] < 0.8:
            review_flags.append("low_confidence")
            needs_review = True
        
        if result["source_type"] == "gpt":
            review_flags.append("gpt_fallback")
            needs_review = True
        
        if generation_time_ms > 5000:  # Very slow generation
            review_flags.append("slow_generation")
        
        return EnhancedQuestionAnswerResponse(
            answer=result["answer"],
            question_id=question_id,
            answer_id=answer_id,
            similarity_score=result.get("similarity", 0.0),
            is_cached=False,
            source_documents=result.get("source_documents", []),
            low_confidence=result["confidence"] < 0.8,
            answer_type=result["source_type"],
            confidence_score=result["confidence"],
            has_override=False,
            override_id=None,
            needs_review=needs_review,
            review_flags=review_flags,
            generation_time_ms=generation_time_ms,
            chunk_count=result.get("total_chunks_found", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Submit feedback for an answer (thumbs up/down)"""
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
        
        if not rag_service.openai_configured:
            # Demo mode suggestions
            return QuestionSuggestion(
                original_question=original_question,
                suggested_questions=[
                    f"What is {original_question.split()[-1] if original_question.split() else 'the topic'}?",
                    f"How does {original_question.split()[0] if original_question.split() else 'this'} work?",
                    f"Can you explain {original_question.lower()}?"
                ],
                reasoning="Demo mode: These are sample suggestions. With OpenAI configured, you'd get intelligent rephrasing suggestions."
            )
        
        # Use OpenAI to generate question suggestions
        import openai
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant that improves unclear questions. 
                    Given a user's question, suggest 3-5 clearer, more specific versions that would likely get better answers from a document search system.
                    Focus on making questions more specific, adding context, and clarifying intent.
                    Return your response as JSON with 'suggestions' array and 'reasoning' string."""
                },
                {
                    "role": "user",
                    "content": f"Original question: '{original_question}'\n\nPlease suggest better versions of this question and explain why these suggestions would work better."
                }
            ],
            max_tokens=400,
            temperature=0.7
        )
        
        ai_response = response['choices'][0]['message']['content'].strip()
        
        # Try to parse JSON response, fallback to simple parsing
        try:
            import json
            parsed = json.loads(ai_response)
            suggestions = parsed.get('suggestions', [])
            reasoning = parsed.get('reasoning', 'AI-generated suggestions for better question clarity.')
        except:
            # Fallback: extract suggestions from text
            lines = ai_response.split('\n')
            suggestions = [line.strip('- ').strip() for line in lines if line.strip() and not line.startswith('Reasoning')][:5]
            reasoning = "AI-generated suggestions to improve question clarity and get better search results."
        
        return QuestionSuggestion(
            original_question=original_question,
            suggested_questions=suggestions,
            reasoning=reasoning
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating question suggestions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating suggestions: {str(e)}"
        )

# Document Management Endpoints
@app.get("/documents")
async def list_documents(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """List all uploaded documents"""
    try:
        # Query documents table directly
        result = db.execute(text("""
            SELECT id, filename, original_filename, file_size, 
                   content_type, upload_date, processed, processing_status,
                   total_pages, total_chunks
            FROM documents 
            ORDER BY upload_date DESC
        """))
        
        documents = []
        for row in result:
            documents.append({
                "id": row[0],
                "filename": row[1],
                "original_filename": row[2],
                "file_size": row[3],
                "content_type": row[4],
                "upload_date": row[5],
                "processed": row[6],
                "processing_status": row[7],
                "total_pages": row[8],
                "total_chunks": row[9]
            })
        
        return {
            "documents": documents,
            "total": len(documents),
            "message": "Documents retrieved successfully"
        }
        
    except Exception as e:
        print(f"❌ Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )

@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Upload and process a PDF document"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        
        print(f"📤 Uploading document: {file.filename} ({len(file_content)} bytes)")
        
        # Insert document record using SQLAlchemy ORM
        document = Document(
            filename=unique_filename,
            original_filename=file.filename,
            file_size=len(file_content),
            content_type="application/pdf",
            processing_status="uploaded"
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Process PDF in background
        pdf_processor = SimplePDFProcessor(rag_service)
        background_tasks.add_task(
            pdf_processor.process_pdf_content,
            db,
            document.id,
            file_content
        )
        
        return {
            "message": "Document uploaded successfully and processing started",
            "document_id": document.id,
            "filename": file.filename,
            "file_size": len(file_content),
            "processing_status": "uploaded",
            "note": "PDF processing is running in the background. The system will extract content, generate embeddings, and enable search."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Delete a document and all its chunks"""
    try:
        # Check if document exists
        result = db.execute(text("SELECT id FROM documents WHERE id = :id"), {"id": document_id})
        if not result.fetchone():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Delete document (chunks will be deleted automatically due to CASCADE)
        db.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})
        db.commit()
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )

# Statistics and Debug Endpoints
@app.get("/stats")
async def get_stats(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get API statistics"""
    try:
        # Get counts from database
        question_count = db.execute(text("SELECT COUNT(*) FROM questions")).fetchone()[0]
        answer_count = db.execute(text("SELECT COUNT(*) FROM answers")).fetchone()[0]
        document_count = db.execute(text("SELECT COUNT(*) FROM documents")).fetchone()[0]
        chunk_count = db.execute(text("SELECT COUNT(*) FROM document_chunks")).fetchone()[0]
        override_count = db.execute(text("SELECT COUNT(*) FROM answer_overrides WHERE status = 'active'")).fetchone()[0]
        review_count = db.execute(text("SELECT COUNT(*) FROM answer_reviews")).fetchone()[0]
        cache_count = db.execute(text("SELECT COUNT(*) FROM performance_cache")).fetchone()[0]
        
        return {
            "questions": question_count,
            "answers": answer_count,
            "documents": document_count,
            "chunks": chunk_count,
            "active_overrides": override_count,
            "reviews": review_count,
            "cached_answers": cache_count,
            "openai_configured": rag_service.openai_configured,
            "version": "3.0.0 - Enhanced PDF RAG with Human Override System"
        }
        
    except Exception as e:
        print(f"❌ Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}"
        )

@app.get("/qa-cache")
async def get_qa_cache(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get cached Q&A pairs"""
    try:
        # Get questions with their latest answers
        result = db.execute(text("""
            SELECT q.id, q.text as question, q.created_at,
                   a.text as answer, a.confidence_score, a.created_at as answer_date,
                   EXISTS(SELECT 1 FROM answer_overrides ao WHERE ao.question_id = q.id AND ao.status = 'active') as has_override
            FROM questions q
            JOIN answers a ON q.id = a.question_id
            WHERE a.id = (
                SELECT id FROM answers a2 
                WHERE a2.question_id = q.id 
                ORDER BY a2.created_at DESC 
                LIMIT 1
            )
            ORDER BY q.created_at DESC
            LIMIT :limit OFFSET :skip
        """), {"limit": limit, "skip": skip})
        
        qa_pairs = []
        for row in result:
            qa_pairs.append({
                "question_id": row[0],
                "question": row[1],
                "question_date": row[2],
                "answer": row[3][:200] + "..." if len(row[3]) > 200 else row[3],
                "confidence": float(row[4]),
                "answer_date": row[5],
                "has_override": row[6]
            })
        
        return {
            "qa_pairs": qa_pairs,
            "total": len(qa_pairs),
            "message": "Q&A cache retrieved successfully"
        }
        
    except Exception as e:
        print(f"❌ Error getting Q&A cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting Q&A cache: {str(e)}"
        )

@app.get("/debug/chunks")
async def debug_chunks(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Debug endpoint to check document chunks"""
    try:
        # Get all chunks with document info
        result = db.execute(text("""
            SELECT dc.id, dc.content, dc.chunk_embedding IS NOT NULL as has_embedding,
                   d.original_filename, d.processed
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            ORDER BY dc.id
            LIMIT 10
        """))
        
        chunks = []
        for row in result:
            chunks.append({
                "chunk_id": row[0],
                "content_preview": row[1][:100] + "..." if len(row[1]) > 100 else row[1],
                "has_embedding": row[2],
                "filename": row[3],
                "document_processed": row[4]
            })
        
        return {
            "chunks": chunks,
            "total_chunks": len(chunks),
            "message": "Debug info for document chunks"
        }
        
    except Exception as e:
        print(f"❌ Error in debug endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in debug endpoint: {str(e)}"
        )

@app.get("/feedback/stats")
async def get_feedback_stats(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get feedback statistics"""
    try:
        # Get feedback counts
        total_feedback = db.query(AnswerFeedback).count()
        helpful_feedback = db.query(AnswerFeedback).filter(AnswerFeedback.is_helpful == True).count()
        unhelpful_feedback = db.query(AnswerFeedback).filter(AnswerFeedback.is_helpful == False).count()
        
        # Get feedback by answer type
        feedback_by_type = db.execute(text("""
            SELECT answer_type, 
                   COUNT(*) as total,
                   SUM(CASE WHEN is_helpful THEN 1 ELSE 0 END) as helpful
            FROM answer_feedback 
            WHERE answer_type IS NOT NULL
            GROUP BY answer_type
        """)).fetchall()
        
        type_stats = {}
        for row in feedback_by_type:
            type_stats[row[0]] = {
                "total": row[1],
                "helpful": row[2],
                "helpful_percentage": (row[2] / row[1] * 100) if row[1] > 0 else 0
            }
        
        return {
            "total_feedback": total_feedback,
            "helpful_feedback": helpful_feedback,
            "unhelpful_feedback": unhelpful_feedback,
            "helpful_percentage": (helpful_feedback / total_feedback * 100) if total_feedback > 0 else 0,
            "feedback_by_answer_type": type_stats
        }
        
    except Exception as e:
        print(f"❌ Error getting feedback stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting feedback stats: {str(e)}"
        )

# User Management Endpoints
@app.post("/users", response_model=UserResponse)
async def create_user(
    user: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Create a new user (Admin only)"""
    # Check if user already exists
    if override_crud.get_user_by_username(db, user.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    if override_crud.get_user_by_email(db, user.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists"
        )
    
    return override_crud.create_user(db, user)

@app.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """List all users (Admin only)"""
    return override_crud.get_users(db, skip, limit)

@app.get("/users/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information"""
    return current_user

# Override Management Endpoints
@app.post("/overrides", response_model=OverrideResponse)
async def create_override(
    override: OverrideCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.REVIEWER))
):
    """Create a human override for an answer (Reviewer/Admin only)"""
    # Verify question and answer exist
    question = db.query(Question).filter(Question.id == override.question_id).first()
    answer = db.query(Answer).filter(Answer.id == override.original_answer_id).first()
    
    if not question or not answer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question or answer not found"
        )
    
    return override_crud.create_override(db, override, current_user.id)

@app.get("/overrides", response_model=List[OverrideResponse])
async def list_overrides(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.REVIEWER))
):
    """List all overrides (Reviewer/Admin only)"""
    return override_crud.get_overrides(db, skip, limit)

@app.get("/overrides/question/{question_id}", response_model=OverrideResponse)
async def get_override_for_question(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get active override for a specific question"""
    override = override_crud.get_active_override(db, question_id)
    if not override:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active override found for this question"
        )
    return override

@app.delete("/overrides/{override_id}")
async def revoke_override(
    override_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.REVIEWER))
):
    """Revoke an override (Reviewer/Admin only)"""
    override = override_crud.revoke_override(db, override_id)
    if not override:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Override not found"
        )
    return {"message": "Override revoked successfully"}

# Review Management Endpoints
@app.post("/reviews", response_model=ReviewResponse)
async def create_review(
    review: ReviewCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.REVIEWER))
):
    """Create a review for an answer (Reviewer/Admin only)"""
    # Verify question and answer exist
    question = db.query(Question).filter(Question.id == review.question_id).first()
    answer = db.query(Answer).filter(Answer.id == review.answer_id).first()
    
    if not question or not answer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question or answer not found"
        )
    
    return override_crud.create_review(db, review, current_user.id)

@app.get("/reviews/queue", response_model=List[ReviewQueueItem])
async def get_review_queue(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.REVIEWER))
):
    """Get pending reviews queue (Reviewer/Admin only)"""
    pending_reviews = override_crud.get_pending_reviews(db, skip, limit)
    
    queue_items = []
    for review in pending_reviews:
        queue_items.append(ReviewQueueItem(
            question_id=review["question_id"],
            question_text=review["question_text"],
            answer_id=review["answer_id"],
            answer_text=review["answer_text"],
            confidence_score=review["confidence_score"],
            answer_type="system",  # Default type
            created_at=review["created_at"],
            has_override=review["has_override"],
            review_count=review["review_count"],
            priority_score=review["priority_score"]
        ))
    
    return queue_items

@app.get("/reviews/insufficient", response_model=List[dict])
async def get_insufficient_answers(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.REVIEWER))
):
    """Get answers marked as insufficient (Reviewer/Admin only)"""
    return override_crud.get_insufficient_answers(db, skip, limit)

@app.get("/reviews/stats", response_model=ReviewStats)
async def get_review_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.REVIEWER))
):
    """Get review statistics (Reviewer/Admin only)"""
    stats = override_crud.get_review_stats(db)
    
    return ReviewStats(
        total_pending_reviews=stats["total_pending_reviews"],
        total_insufficient_answers=stats["total_insufficient_answers"],
        total_overrides=stats["total_overrides"],
        avg_confidence_insufficient=stats["avg_confidence_insufficient"],
        top_review_flags=stats["top_review_flags"],
        reviewer_activity=stats["reviewer_activity"]
    )

# Performance Cache Endpoints
@app.get("/cache/stats")
async def get_cache_statistics(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get performance cache statistics"""
    return override_crud.get_cache_stats(db)

@app.delete("/cache/cleanup")
async def cleanup_cache(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Clean up expired cache entries (Admin only)"""
    expired_count = override_crud.cleanup_expired_cache(db)
    return {"message": f"Cleaned up {expired_count} expired cache entries"}

@app.delete("/cache/clear")
async def clear_all_cache(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Clear all cache entries (Admin only)"""
    try:
        result = db.execute(text("DELETE FROM performance_cache"))
        db.commit()
        return {"message": f"Cleared all cache entries"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing cache: {str(e)}"
        )

# Chunk Usage Analytics
@app.get("/analytics/chunks")
async def get_chunk_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.REVIEWER))
):
    """Get chunk usage analytics (Reviewer/Admin only)"""
    return override_crud.get_chunk_usage_stats(db)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
