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
from schemas import (
    QuestionAnswerRequest, 
    QuestionAnswerResponse, 
    HealthResponse, 
    AuthHealthResponse,
    QuestionResponse,
    AnswerResponse
)
from feedback_schemas import FeedbackCreate, FeedbackResponse, QuestionSuggestion
from rag_service import RAGService
import crud
from simple_pdf_processor import SimplePDFProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG service
rag_service = RAGService()

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
        "message": "Enhanced PDF RAG Q&A API with User-Controlled Saving",
        "version": "3.0.0",
        "status": "running",
        "logic": "PDF-first search → User-controlled saving → Database fallback → ChatGPT",
        "features": [
            "PDF document processing (search first)",
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
    Ask a question with improved logic:
    1. Check Q&A database for very similar questions first (similarity > 0.90)
    2. If not found, search PDF documents
    3. If found in PDF → return with save option
    4. If not in PDF, check Q&A database with lower threshold (similarity > 0.75)
    5. If still not found → use ChatGPT
    """
    try:
        question_text = request.question.strip()
        start_time = time.time()
        
        if not question_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        print(f"🔍 Processing question with improved logic: {question_text[:50]}...")
        
        # Generate embedding for the question
        query_embedding = await rag_service.generate_embedding(question_text)
        
        # STEP 1: Check Q&A database for VERY similar questions first (high threshold)
        print("🗃️ Step 1: Checking Q&A database for very similar questions...")
        similar_question, similarity = await rag_service.find_similar_question(db, query_embedding)
        
        if similar_question and similarity > 0.90:  # High threshold for exact matches
            print(f"✅ Found very similar question in database (similarity: {similarity:.3f})")
            
            # Get the latest answer for this question
            latest_answer = db.query(Answer).filter(
                Answer.question_id == similar_question.id
            ).order_by(Answer.created_at.desc()).first()
            
            if latest_answer:
                generation_time = int((time.time() - start_time) * 1000)
                
                return {
                    "answer": latest_answer.text,
                    "question_id": similar_question.id,
                    "answer_id": latest_answer.id,
                    "similarity_score": similarity,
                    "is_cached": True,
                    "source_documents": [],
                    "answer_type": "database_cached",
                    "confidence_score": latest_answer.confidence_score,
                    "generation_time_ms": generation_time,
                    "found_in_pdf": False,
                    "show_save_prompt": False,
                    "save_prompt_message": None,
                    "cache_hit": True
                }
        
        # STEP 2: If not found in Q&A database, search PDF documents
        print("📄 Step 2: Not found in Q&A database, searching PDF documents...")
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
                "save_prompt_message": "This answer was found in your PDF documents. Do you want to save it to the Q&A database for future quick access?",
                "temp_question": question_text,  # Store for potential saving
                "temp_answer": answer_text,      # Store for potential saving
                "temp_sources": source_docs,     # Store for potential saving
                "cache_hit": False
            }
        
        # STEP 3: If not found in PDF, check Q&A database with lower threshold
        print("🗃️ Step 3: Not found in PDF, checking Q&A database with lower threshold...")
        if similar_question and similarity > 0.75:  # Lower threshold for broader matches
            print(f"✅ Found similar question in database (similarity: {similarity:.3f})")
            
            # Get the latest answer for this question
            latest_answer = db.query(Answer).filter(
                Answer.question_id == similar_question.id
            ).order_by(Answer.created_at.desc()).first()
            
            if latest_answer:
                generation_time = int((time.time() - start_time) * 1000)
                
                return {
                    "answer": latest_answer.text,
                    "question_id": similar_question.id,
                    "answer_id": latest_answer.id,
                    "similarity_score": similarity,
                    "is_cached": True,
                    "source_documents": [],
                    "answer_type": "database_cached",
                    "confidence_score": latest_answer.confidence_score,
                    "generation_time_ms": generation_time,
                    "found_in_pdf": False,
                    "show_save_prompt": False,
                    "save_prompt_message": None,
                    "cache_hit": True
                }
        
        # STEP 4: If not found anywhere, use ChatGPT
        print("🤖 Step 4: Not found in PDF or database, using ChatGPT...")
        answer_text = await rag_service.generate_answer(question_text)
        
        generation_time = int((time.time() - start_time) * 1000)
        
        return {
            "answer": answer_text,
            "question_id": None,  # Not saved yet
            "answer_id": None,    # Not saved yet
            "similarity_score": 0.0,
            "is_cached": False,
            "source_documents": [],
            "answer_type": "chatgpt_generated",
            "confidence_score": 0.7,
            "generation_time_ms": generation_time,
            "found_in_pdf": False,
            "show_save_prompt": True,
            "save_prompt_message": "This answer was generated by ChatGPT. Do you want to save it to the Q&A database?",
            "temp_question": question_text,  # Store for potential saving
            "temp_answer": answer_text,      # Store for potential saving
            "temp_sources": [],              # No sources for ChatGPT
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
async def save_answer_to_database(
    save_request: dict,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Save an answer to the Q&A database when user clicks 'Yes'
    """
    try:
        question_text = save_request.get("question")
        answer_text = save_request.get("answer")
        answer_type = save_request.get("answer_type", "user_saved")
        confidence_score = save_request.get("confidence_score", 0.9)
        
        if not question_text or not answer_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question and answer are required"
            )
        
        print(f"💾 User chose to save answer for: {question_text[:50]}...")
        
        # Generate embedding for the question
        query_embedding = await rag_service.generate_embedding(question_text)
        
        # Create question record
        new_question = crud.create_question(db, crud.QuestionCreate(text=question_text))
        
        # Store embedding
        crud.create_embedding(db, new_question.id, query_embedding)
        
        # Create answer record
        new_answer = crud.create_answer(
            db, 
            crud.AnswerCreate(
                question_id=new_question.id,
                text=answer_text,
                confidence_score=confidence_score
            )
        )
        
        print(f"✅ Answer saved to database: Q{new_question.id}, A{new_answer.id}")
        
        return {
            "message": "Answer saved successfully to Q&A database",
            "question_id": new_question.id,
            "answer_id": new_answer.id,
            "saved_at": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error saving answer: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving answer: {str(e)}"
        )

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
        count_result = db.execute(text("SELECT COUNT(*) FROM documents"))
        total_count = count_result.fetchone()[0]
        print(f"📊 Total documents in database: {total_count}")
        
        if total_count == 0:
            print("⚠️ No documents found in database")
            return {
                "documents": [],
                "total": 0,
                "message": "No documents found in database",
                "debug_info": {
                    "table_exists": True,
                    "total_count": 0,
                    "query_executed": True
                }
            }
        
        # Fetch documents with detailed logging
        result = db.execute(text("""
            SELECT id, filename, original_filename, file_size, 
                   content_type, upload_date, processed, processing_status,
                   total_pages, total_chunks
            FROM documents 
            ORDER BY upload_date DESC
        """))
        
        documents = []
        for row in result:
            doc = {
                "id": row[0],
                "filename": row[1],
                "original_filename": row[2],
                "file_size": row[3],
                "content_type": row[4],
                "upload_date": row[5].isoformat() if row[5] else None,
                "processed": row[6],
                "processing_status": row[7],
                "total_pages": row[8],
                "total_chunks": row[9]
            }
            documents.append(doc)
            print(f"📄 Found document: {doc['original_filename']} (ID: {doc['id']})")
        
        print(f"✅ Successfully fetched {len(documents)} documents")
        
        return {
            "documents": documents,
            "total": len(documents),
            "message": f"Successfully retrieved {len(documents)} documents",
            "debug_info": {
                "table_exists": True,
                "total_count": total_count,
                "returned_count": len(documents),
                "query_executed": True
            }
        }
        
    except Exception as e:
        print(f"❌ Error listing documents: {str(e)}")
        
        # Try to provide more debugging info
        try:
            # Check if table exists
            table_check = db.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'documents'
            """))
            table_exists = table_check.fetchone() is not None
            
            return {
                "documents": [],
                "total": 0,
                "message": f"Error retrieving documents: {str(e)}",
                "debug_info": {
                    "table_exists": table_exists,
                    "error": str(e),
                    "query_executed": False
                }
            }
        except:
            pass
        
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
        
        # Insert document record
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
        
        print(f"✅ Document saved to database with ID: {document.id}")
        
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
            "processing_status": "uploaded"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )

@app.get("/stats")
async def get_stats(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get API statistics"""
    try:
        question_count = db.execute(text("SELECT COUNT(*) FROM questions")).fetchone()[0]
        answer_count = db.execute(text("SELECT COUNT(*) FROM answers")).fetchone()[0]
        document_count = db.execute(text("SELECT COUNT(*) FROM documents")).fetchone()[0]
        chunk_count = db.execute(text("SELECT COUNT(*) FROM document_chunks")).fetchone()[0]
        
        return {
            "questions": question_count,
            "answers": answer_count,
            "documents": document_count,
            "chunks": chunk_count,
            "openai_configured": rag_service.openai_configured,
            "version": "3.0.0 - PDF-First Logic with User-Controlled Saving"
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
        result = db.execute(text("""
            SELECT q.id, q.text as question, q.created_at,
                   a.text as answer, a.confidence_score, a.created_at as answer_date
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
                "answer_date": row[5]
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

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
