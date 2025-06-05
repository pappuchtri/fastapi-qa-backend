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
from models import Question, Answer, Embedding, AnswerOverride, ReviewQueue
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
                "found_in_knowledge_base": False,
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
async def save_answer_to_database(
    save_request: dict,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Save an answer to the Q&A database when user clicks 'Yes'
    """
    try:
        print(f"📥 Received save request: {save_request}")
        
        question_text = save_request.get("question")
        answer_text = save_request.get("answer")
        answer_type = save_request.get("answer_type", "user_saved")
        confidence_score = save_request.get("confidence_score", 0.9)
        
        print(f"📝 Extracted data - Question: {question_text[:50] if question_text else 'None'}...")
        print(f"📝 Answer length: {len(answer_text) if answer_text else 0}")
        print(f"📝 Answer type: {answer_type}")
        print(f"📝 Confidence: {confidence_score}")
        
        if not question_text or not answer_text:
            print("❌ Missing question or answer text")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question and answer are required"
            )
        
        print(f"💾 User chose to save answer for: {question_text[:50]}...")
        
        # Generate embedding for the question
        print("🔄 Generating embedding...")
        try:
            query_embedding = await rag_service.generate_embedding(question_text)
            print(f"✅ Generated embedding with length: {len(query_embedding) if query_embedding else 0}")
        except Exception as e:
            print(f"❌ Error generating embedding: {str(e)}")
            # Continue without embedding for now
            query_embedding = []
        
        # Create question record using direct method
        print("🔄 Creating question record...")
        try:
            new_question = create_question_direct(db, question_text)
            print(f"✅ Created question with ID: {new_question.id}")
        except Exception as e:
            print(f"❌ Error creating question: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating question: {str(e)}"
            )
        
        # Store embedding if we have one
        if query_embedding:
            print("🔄 Storing embedding...")
            try:
                create_embedding_direct(db, new_question.id, query_embedding)
                print("✅ Embedding stored successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not store embedding: {str(e)}")
                # Continue without embedding
        
        # Create answer record using direct method
        print("🔄 Creating answer record...")
        try:
            new_answer = create_answer_direct(
                db, 
                new_question.id,
                answer_text,
                confidence_score
            )
            print(f"✅ Created answer with ID: {new_answer.id}")
        except Exception as e:
            print(f"❌ Error creating answer: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating answer: {str(e)}"
            )
        
        print(f"✅ Answer saved to database: Q{new_question.id}, A{new_answer.id}")
        
        return {
            "message": "Answer saved successfully to Q&A database",
            "question_id": new_question.id,
            "answer_id": new_answer.id,
            "saved_at": datetime.utcnow().isoformat(),
            "success": True
        }
        
    except HTTPException as he:
        print(f"❌ HTTP Exception in save-answer: {he.detail}")
        raise he
    except Exception as e:
        print(f"❌ Unexpected error saving answer: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
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
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """List all uploaded documents with enhanced error handling"""
    try:
        print("🔍 Fetching documents from database...")
        
        # First, check if documents table exists
        try:
            table_check = db.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'documents'
                )
            """))
            table_exists = table_check.fetchone()[0]
            print(f"📊 Documents table exists: {table_exists}")
            
            if not table_exists:
                print("⚠️ Documents table does not exist, creating it...")
                # Create the table
                Base.metadata.create_all(bind=engine)
                print("✅ Documents table created")
                
                return {
                    "documents": [],
                    "total": 0,
                    "message": "Documents table created, no documents found",
                    "debug_info": {
                        "table_exists": True,
                        "total_count": 0,
                        "query_executed": True,
                        "table_created": True
                    }
                }
        except Exception as table_error:
            print(f"❌ Error checking table existence: {str(table_error)}")
            # Continue anyway, maybe the table exists but we can't check
        
        # Get total count
        try:
            count_result = db.execute(text("SELECT COUNT(*) FROM documents"))
            total_count = count_result.fetchone()[0]
            print(f"📊 Total documents in database: {total_count}")
        except Exception as count_error:
            print(f"❌ Error getting document count: {str(count_error)}")
            total_count = 0
        
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
        try:
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
                print(f"📄 Found document: {doc['original_filename']} (ID: {doc['id']})")
            
            print(f"✅ Successfully fetched {len(documents)} documents")
            
            return {
                "documents": documents,
                "total": total_count,
                "message": f"Successfully retrieved {len(documents)} documents",
                "debug_info": {
                    "table_exists": True,
                    "total_count": total_count,
                    "returned_count": len(documents),
                    "query_executed": True,
                    "skip": skip,
                    "limit": limit
                }
            }
            
        except Exception as query_error:
            print(f"❌ Error executing documents query: {str(query_error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error querying documents: {str(query_error)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error listing documents: {str(e)}")
        
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

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Delete a document and all its chunks"""
    try:
        print(f"🗑️ Deleting document ID: {document_id}")
        
        # Check if document exists
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found"
            )
        
        original_filename = document.original_filename
        
        # Delete all chunks for this document (CASCADE should handle this, but let's be explicit)
        chunks_deleted = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        print(f"🗑️ Deleted {chunks_deleted} chunks for document {document_id}")
        
        # Delete the document
        db.delete(document)
        db.commit()
        
        print(f"✅ Successfully deleted document: {original_filename} (ID: {document_id})")
        
        return {
            "message": f"Document '{original_filename}' deleted successfully",
            "document_id": document_id,
            "chunks_deleted": chunks_deleted,
            "deleted_at": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting document {document_id}: {str(e)}")
        db.rollback()  # Rollback in case of error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )

@app.get("/stats")
async def get_stats(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get API statistics including knowledge base"""
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
            kb_stats = knowledge_service.get_stats(db)
        except Exception as e:
            print(f"⚠️ Warning: Could not get knowledge base stats: {str(e)}")
            kb_stats = {"total_entries": 0, "categories": 0}
        
        return {
            "questions": question_count,
            "answers": answer_count,
            "documents": document_count,
            "chunks": chunk_count,
            "knowledge_base": kb_stats,
            "openai_configured": rag_service.openai_configured,
            "version": "3.1.0 - Knowledge-Base-First Logic",
            "response_priority": [
                "1. Built-in Knowledge Base (fastest)",
                "2. PDF Document Search", 
                "3. Q&A Database Cache",
                "4. ChatGPT Generation"
            ]
        }
        
    except Exception as e:
        print(f"❌ Error getting stats: {str(e)}")
        # Return basic stats even if there's an error
        return {
            "questions": 0,
            "answers": 0,
            "documents": 0,
            "chunks": 0,
            "knowledge_base": {"total_entries": 0, "categories": 0},
            "openai_configured": rag_service.openai_configured,
            "version": "3.1.0 - Knowledge-Base-First Logic",
            "error": f"Error getting detailed stats: {str(e)}"
        }

@app.get("/qa-cache")
async def get_qa_cache(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get cached Q&A pairs with error handling"""
    try:
        # Check if tables exist first
        try:
            table_check = db.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'questions'
                ) AND EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'answers'
                )
            """))
            tables_exist = table_check.fetchone()[0]
            
            if not tables_exist:
                print("⚠️ Q&A tables do not exist, creating them...")
                Base.metadata.create_all(bind=engine)
                return {
                    "qa_pairs": [],
                    "total": 0,
                    "message": "Q&A tables created, no cached pairs found"
                }
        except Exception as table_error:
            print(f"❌ Error checking Q&A tables: {str(table_error)}")
            # Continue anyway
        
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
        return {
            "qa_pairs": [],
            "total": 0,
            "message": f"Error retrieving Q&A cache: {str(e)}",
            "error": str(e)
        }

@app.get("/debug/tables")
async def debug_tables(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Debug endpoint to check which tables exist"""
    try:
        result = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """))
        
        tables = [row[0] for row in result]
        
        return {
            "existing_tables": tables,
            "total_tables": len(tables),
            "message": f"Found {len(tables)} tables in database"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Error checking tables: {str(e)}"
        }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
