from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import text, func, desc, and_, or_
from contextlib import asynccontextmanager
import os
import time
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import hashlib
import traceback
import uuid
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database imports
from database import SessionLocal, engine, Base, get_db

# Model imports - now all from models.py
from models import (
    Question, Answer, Embedding, KnowledgeBase, User, AnswerOverride, 
    AnswerReview, WebSearchResult, WebAnswer, UnansweredQuestion,
    PerformanceCache, AuditLog, ReviewQueue
)

# Document models
from document_models import Document, DocumentChunk

# Schema imports - fix to only import existing schemas
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
    KnowledgeBaseSearch,
    QuestionCreate,
    DocumentResponse
)

# Service imports - only import existing services
from rag_service import RAGService
from knowledge_base_service import knowledge_service  # This is already an instance
from feedback_models import AnswerFeedback
from feedback_schemas import FeedbackCreate, FeedbackResponse, QuestionSuggestion
from simple_pdf_processor import SimplePDFProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
rag_service = RAGService()
# knowledge_service is already imported as an instance, don't call it
kb_service = knowledge_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Enhanced PDF RAG Q&A API...")
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created/verified")
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {str(e)}")
    
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

# Create FastAPI app with lifespan
app = FastAPI(
    title="Enhanced PDF RAG Q&A API",
    description="A comprehensive FastAPI backend for PDF document processing and Q&A using RAG",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
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

# Direct CRUD operations
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

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced PDF RAG Q&A API",
        "version": "4.0.0",
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
        message="Enhanced RAG API is running",
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
    Ask a question with comprehensive search logic:
    1. FIRST: Check built-in knowledge base (fastest, most accurate)
    2. If not found: Search PDF documents using RAG
    3. If not found: Check Q&A database for similar questions  
    4. If all fail: Use ChatGPT-3.5 as final fallback
    """
    try:
        question_text = request.question.strip()
        start_time = time.time()
        search_attempts = []
        
        if not question_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        print(f"🔍 Processing question: {question_text[:50]}...")
        
        # STEP 1: Check built-in knowledge base
        print("🧠 Step 1: Checking built-in knowledge base...")
        search_attempts.append("knowledge base")
        
        try:
            kb_match = kb_service.get_best_knowledge_match(db, question_text)
            
            if kb_match:
                print(f"✅ Found answer in knowledge base (score: {kb_match['relevance_score']:.3f})")
                generation_time = int((time.time() - start_time) * 1000)
                
                return {
                    "answer": kb_match["answer"],
                    "question_id": None,
                    "answer_id": None,
                    "similarity_score": kb_match["relevance_score"],
                    "is_cached": True,
                    "source_documents": [],
                    "answer_type": "knowledge_base",
                    "confidence_score": 0.95,
                    "generation_time_ms": generation_time,
                    "found_in_knowledge_base": True,
                    "knowledge_base_category": kb_match["category"],
                    "knowledge_base_id": kb_match["id"],
                    "show_save_prompt": False,
                    "cache_hit": True,
                    "match_type": kb_match["match_type"]
                }
        except Exception as e:
            print(f"⚠️ Knowledge base search failed: {str(e)}")
        
        # Generate embedding for remaining steps
        query_embedding = await rag_service.generate_embedding(question_text)
        
        # STEP 2: Search PDF documents
        print("📄 Step 2: Searching PDF documents...")
        search_attempts.append("PDF documents")
        
        try:
            relevant_chunks = await rag_service.search_document_chunks(db, query_embedding, limit=5)
            
            if relevant_chunks:
                print(f"✅ Found {len(relevant_chunks)} relevant chunks in PDF documents")
                
                # Generate answer from PDF content
                answer_text = await rag_service.generate_answer_from_chunks(question_text, relevant_chunks)
                source_docs = list(set([chunk.get('filename', 'Unknown') for chunk in relevant_chunks]))
                
                generation_time = int((time.time() - start_time) * 1000)
                
                return {
                    "answer": answer_text,
                    "question_id": None,
                    "answer_id": None,
                    "similarity_score": 0.0,
                    "is_cached": False,
                    "source_documents": source_docs,
                    "answer_type": "pdf_document",
                    "confidence_score": 0.9,
                    "generation_time_ms": generation_time,
                    "found_in_knowledge_base": False,
                    "found_in_pdf": True,
                    "show_save_prompt": True,
                    "save_prompt_message": "Do you want to save this answer to the Q&A database?",
                    "temp_question": question_text,
                    "temp_answer": answer_text,
                    "temp_sources": source_docs,
                    "cache_hit": False
                }
        except Exception as e:
            print(f"⚠️ PDF search failed: {str(e)}")
        
        # STEP 3: Check Q&A database
        print("🗃️ Step 3: Checking Q&A database...")
        search_attempts.append("Q&A database")
        
        try:
            similar_question, similarity = await rag_service.find_similar_question(db, query_embedding)
            
            if similar_question and similarity > 0.75:
                print(f"✅ Found similar question in database (similarity: {similarity:.3f})")
                
                latest_answer = db.query(Answer).filter(
                    Answer.question_id == similar_question.id
                ).order_by(Answer.created_at.desc()).first()
                
                if latest_answer:
                    generation_time = int((time.time() - start_time) * 1000)
                    answer_type = "database_cached"  # Changed from database_exact_match/database_adapted for frontend compatibility
                    
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
        except Exception as e:
            print(f"⚠️ Database search failed: {str(e)}")
        
        # STEP 4: Fallback to ChatGPT
        print("🤖 Step 4: Using ChatGPT as fallback...")
        search_attempts.append("AI generation")
        
        try:
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
                "cache_hit": False,
                "search_attempts": search_attempts
            }
        except Exception as e:
            print(f"⚠️ ChatGPT generation failed: {str(e)}")
            
            # Final fallback - return a helpful message
            generation_time = int((time.time() - start_time) * 1000)
            
            return {
                "answer": "I'm sorry, I couldn't find an answer to your question. Please try rephrasing your question or check if there are any typos.",
                "question_id": None,
                "answer_id": None,
                "similarity_score": 0.0,
                "is_cached": False,
                "source_documents": [],
                "answer_type": "no_answer",
                "confidence_score": 0.0,
                "generation_time_ms": generation_time,
                "found_in_knowledge_base": False,
                "found_in_pdf": False,
                "show_save_prompt": False,
                "cache_hit": False,
                "search_attempts": search_attempts,
                "error": "All search methods failed"
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
    """Save an answer to the Q&A database when user clicks 'Yes'"""
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
        
        # Generate embedding for the question
        try:
            query_embedding = await rag_service.generate_embedding(question_text)
        except Exception as e:
            print(f"❌ Error generating embedding: {str(e)}")
            query_embedding = []
        
        # Create question record
        new_question = create_question_direct(db, question_text)
        
        # Store embedding if we have one
        if query_embedding:
            try:
                create_embedding_direct(db, new_question.id, query_embedding)
            except Exception as e:
                print(f"⚠️ Warning: Could not store embedding: {str(e)}")
        
        # Create answer record
        new_answer = create_answer_direct(
            db, 
            new_question.id,
            answer_text,
            confidence_score
        )
        
        return {
            "message": "Answer saved successfully to Q&A database",
            "question_id": new_question.id,
            "answer_id": new_answer.id,
            "saved_at": datetime.utcnow().isoformat(),
            "success": True
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Unexpected error saving answer: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving answer: {str(e)}"
        )

@app.get("/documents")
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """List all uploaded documents"""
    try:
        # Ensure tables exist
        Base.metadata.create_all(bind=engine)
        
        # Get total count safely
        try:
            count_result = db.execute(text("SELECT COUNT(*) FROM documents"))
            total_count = count_result.fetchone()[0]
        except Exception:
            return {
                "documents": [],
                "total": 0,
                "message": "Documents table not found or empty"
            }
        
        if total_count == 0:
            return {
                "documents": [],
                "total": 0,
                "message": "No documents found in database"
            }
        
        # Fetch documents
        result = db.execute(text("""
            SELECT id, filename, original_filename, file_size, 
                   content_type, upload_date, processed, processing_status,
                   total_pages, total_chunks
            FROM documents 
            ORDER BY upload_date DESC
            LIMIT :limit OFFSET :skip
        """), {"limit": limit, "skip": skip})
        
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
        
        return {
            "documents": documents,
            "total": total_count,
            "message": f"Successfully retrieved {len(documents)} documents"
        }
        
    except Exception as e:
        print(f"❌ Error listing documents: {str(e)}")
        return {
            "documents": [],
            "total": 0,
            "message": f"Error retrieving documents: {str(e)}"
        }

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
        
        # Ensure tables exist
        Base.metadata.create_all(bind=engine)
        
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

@app.get("/qa-cache")
async def get_qa_cache(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get cached Q&A pairs"""
    try:
        # Ensure tables exist
        Base.metadata.create_all(bind=engine)
        
        # Get total count safely
        try:
            count_result = db.execute(text("SELECT COUNT(*) FROM questions"))
            total_count = count_result.fetchone()[0]
        except Exception:
            return {
                "qa_pairs": [],
                "total": 0,
                "message": "Questions table not found or empty"
            }
        
        if total_count == 0:
            return {
                "qa_pairs": [],
                "total": 0,
                "message": "No Q&A pairs found in database"
            }
        
        # Fetch Q&A pairs
        result = db.execute(text("""
            SELECT q.id, q.text, q.created_at, a.text, a.confidence_score, a.created_at
            FROM questions q
            JOIN answers a ON q.id = a.question_id
            ORDER BY q.created_at DESC
            LIMIT :limit OFFSET :skip
        """), {"limit": limit, "skip": skip})
        
        qa_pairs = []
        for row in result:
            qa_pair = {
                "question_id": row[0],
                "question": row[1],
                "question_date": row[2].isoformat() if row[2] else None,
                "answer": row[3],
                "confidence": row[4],
                "answer_date": row[5].isoformat() if row[5] else None
            }
            qa_pairs.append(qa_pair)
        
        return {
            "qa_pairs": qa_pairs,
            "total": total_count,
            "message": f"Successfully retrieved {len(qa_pairs)} Q&A pairs"
        }
        
    except Exception as e:
        print(f"❌ Error getting Q&A cache: {str(e)}")
        return {
            "qa_pairs": [],
            "total": 0,
            "message": f"Error retrieving Q&A cache: {str(e)}"
        }

@app.post("/suggest-questions")
async def suggest_questions(
    request: dict,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Suggest related questions based on user input"""
    try:
        question_text = request.get("question", "").strip()
        
        if not question_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        # Generate suggestions using ChatGPT
        suggestions = await rag_service.generate_question_suggestions(question_text)
        
        return {
            "original_question": question_text,
            "suggested_questions": suggestions.get("questions", []),
            "reasoning": suggestions.get("reasoning", "Here are some related questions you might find helpful.")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating question suggestions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating question suggestions: {str(e)}"
        )

@app.get("/stats")
async def get_stats(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get API statistics"""
    try:
        # Use safe queries with error handling
        try:
            question_count = db.execute(text("SELECT COUNT(*) FROM questions")).fetchone()[0]
        except:
            question_count = 0
            
        try:
            answer_count = db.execute(text("SELECT COUNT(*) FROM answers")).fetchone()[0]
        except:
            answer_count = 0
            
        try:
            document_count = db.execute(text("SELECT COUNT(*) FROM documents")).fetchone()[0]
        except:
            document_count = 0
            
        try:
            chunk_count = db.execute(text("SELECT COUNT(*) FROM document_chunks")).fetchone()[0]
        except:
            chunk_count = 0
        
        # Get knowledge base stats safely
        try:
            kb_stats = kb_service.get_stats(db)
        except Exception as e:
            kb_stats = {"total_entries": 0, "categories": 0}
        
        return {
            "questions": question_count,
            "answers": answer_count,
            "documents": document_count,
            "chunks": chunk_count,
            "knowledge_base": kb_stats,
            "openai_configured": rag_service.openai_configured,
            "version": "4.0.0 - Enhanced RAG System",
            "response_priority": [
                "1. Built-in Knowledge Base (fastest)",
                "2. PDF Document Search", 
                "3. Q&A Database Cache",
                "4. ChatGPT Generation"
            ]
        }
        
    except Exception as e:
        print(f"❌ Error getting stats: {str(e)}")
        return {
            "questions": 0,
            "answers": 0,
            "documents": 0,
            "chunks": 0,
            "knowledge_base": {"total_entries": 0, "categories": 0},
            "openai_configured": rag_service.openai_configured,
            "version": "4.0.0 - Enhanced RAG System",
            "error": f"Error getting detailed stats: {str(e)}"
        }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
