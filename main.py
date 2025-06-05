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
from typing import Dict

# Load environment variables
load_dotenv()

# Import our modules - fix the import error
from database import SessionLocal, engine, get_db, Base
from models import Question, Answer, Embedding, AnswerOverride, AnswerReview
from document_models import Document, DocumentChunk
from feedback_models import AnswerFeedback
from override_models import WebSearchResult, WebAnswer, KnowledgeBase, UnansweredQuestion
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
from ai_web_search_service import AIWebSearchService, ai_web_search
from enhanced_citation_service import EnhancedCitationService, citation_service
from feedback_handler import FeedbackHandler, feedback_handler
from no_answer_handler import NoAnswerHandler, no_answer_handler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG service
rag_service = RAGService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Enhanced PDF RAG Q&A API with Web Search...")
    
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
    
    # Check Web Search API keys
    if os.getenv("SERPAPI_KEY"):
        logger.info("🌐 SerpAPI Key configured - Web search enabled")
    elif os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CX"):
        logger.info("🌐 Google Custom Search API configured - Web search enabled")
    else:
        logger.info("🎭 No search API keys - Web search in demo mode")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Enhanced PDF RAG Q&A API...")

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced PDF RAG Q&A API with Web Search",
    description="A comprehensive FastAPI backend for PDF document processing, web search, and Q&A using RAG with user-controlled saving",
    version="4.1.0",
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

def create_web_search_result(
    db: Session, 
    query: str, 
    title: str, 
    snippet: str, 
    url: str, 
    position: int, 
    source: str
) -> WebSearchResult:
    """Create a new web search result"""
    result = WebSearchResult(
        query=query,
        title=title,
        snippet=snippet,
        url=url,
        position=position,
        source=source
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result

def create_web_answer(
    db: Session, 
    question_id: Optional[int], 
    search_result_id: Optional[int], 
    answer_text: str, 
    sources: List[Dict[str, str]], 
    confidence_score: float = 0.7
) -> WebAnswer:
    """Create a new web answer"""
    answer = WebAnswer(
        question_id=question_id,
        search_result_id=search_result_id,
        answer_text=answer_text,
        sources=sources,
        confidence_score=confidence_score
    )
    db.add(answer)
    db.commit()
    db.refresh(answer)
    return answer

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced PDF RAG Q&A API with Web Search",
        "version": "4.1.0",
        "status": "running",
        "logic": "Knowledge Base → PDF search → Q&A database → AI Web search → ChatGPT",
        "features": [
            "Built-in knowledge base (fastest responses)",
            "PDF document processing and search",
            "AI-native web search integration",
            "User-controlled answer saving",
            "Q&A database fallback",
            "ChatGPT-3.5 generation",
            "Enhanced citations with page numbers",
            "Answer feedback system",
            "Question rephrasing suggestions",
            "No-answer handling with suggestions"
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
            "ai-web-search": "/ai-web-search",
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
        message="Enhanced RAG API is running with Web Search",
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
    4. If not found: Search the web for information
    5. If all fail: Use ChatGPT-3.5 as final fallback
    
    This ensures fastest response times and most accurate answers with proper source citations.
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
        
        print(f"🔍 Processing question with comprehensive search logic: {question_text[:50]}...")
        
        # STEP 1: FIRST, check built-in knowledge base (fastest, most accurate)
        print("🧠 Step 1: Checking built-in knowledge base first...")
        search_attempts.append("knowledge base")
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
        search_attempts.append("PDF documents")
        relevant_chunks = await rag_service.search_document_chunks(db, query_embedding, limit=5)
        
        if relevant_chunks:
            print(f"✅ Found {len(relevant_chunks)} relevant chunks in PDF documents")
            
            # Generate answer from PDF content
            answer_text = await rag_service.generate_answer_from_chunks(question_text, relevant_chunks)
            
            # Extract source documents and page numbers
            source_docs = list(set([chunk.get('filename', 'Unknown') for chunk in relevant_chunks]))
            
            # Enhance answer with proper citations
            source_info = {
                "documents": source_docs,
                "chunks": relevant_chunks
            }
            enhanced_answer = citation_service.enhance_answer_with_citations(
                answer_text, "pdf", source_info
            )
            
            generation_time = int((time.time() - start_time) * 1000)
            
            return {
                "answer": enhanced_answer,
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
                "temp_answer": enhanced_answer,
                "temp_sources": source_docs,
                "cache_hit": False
            }
        
        # STEP 3: If not in PDF, check Q&A database
        print("🗃️ Step 3: No PDF match, checking Q&A database...")
        search_attempts.append("Q&A database")
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
        
        # STEP 4: NEW - AI-native web search if no match found in KB, PDFs, or Q&A database
        print("🤖 Step 4: No database match, performing AI-native web search...")
        search_attempts.append("AI web search")

        # Prepare context for AI web search
        search_context = f"User is asking about: {question_text}"
        if relevant_chunks:
            search_context += f" (Some PDF context was found but not sufficient)"

        web_search_result = await ai_web_search.search_web_with_ai(question_text, search_context)

        if web_search_result.get("success", False):
            print(f"✅ AI web search completed successfully")
            
            # Generate enhanced answer with web context
            enhanced_result = await ai_web_search.generate_answer_with_web_context(
                question_text, web_search_result
            )
            
            if enhanced_result.get("success", False):
                # Store web search results in database
                stored_results = []
                sources = web_search_result.get("sources", [])
                
                for i, source in enumerate(sources[:3]):  # Store top 3 sources
                    try:
                        stored_result = create_web_search_result(
                            db,
                            question_text,
                            source.get("title", ""),
                            source.get("snippet", ""),
                            source.get("url", ""),
                            i + 1,
                            "ai_native"
                        )
                        stored_results.append(stored_result)
                    except Exception as e:
                        print(f"⚠️ Error storing web result: {str(e)}")
                
                # Create question record
                new_question = create_question_direct(db, question_text)
                
                # Store embedding
                if query_embedding is not None:
                    try:
                        create_embedding_direct(db, new_question.id, query_embedding)
                    except Exception as e:
                        print(f"⚠️ Error storing embedding: {str(e)}")
                
                # Store web answer
                try:
                    web_answer_record = create_web_answer(
                        db,
                        new_question.id,
                        stored_results[0].id if stored_results else None,
                        enhanced_result["answer"],
                        [{"title": s.get("title", ""), "url": s.get("url", "")} for s in sources],
                        enhanced_result.get("confidence", 0.8)
                    )
                except Exception as e:
                    print(f"⚠️ Error storing web answer: {str(e)}")
                    web_answer_record = None
                
                generation_time = int((time.time() - start_time) * 1000)
                
                return {
                    "answer": enhanced_result["answer"],
                    "question_id": new_question.id if new_question else None,
                    "answer_id": web_answer_record.id if web_answer_record else None,
                    "similarity_score": 0.0,
                    "is_cached": False,
                    "source_documents": [],
                    "source_urls": [source.get("url", "") for source in sources],
                    "answer_type": "ai_web_search",
                    "confidence_score": enhanced_result.get("confidence", 0.8),
                    "generation_time_ms": generation_time,
                    "found_in_knowledge_base": False,
                    "found_in_pdf": False,
                    "found_on_web": True,
                    "web_search_type": "ai_native",
                    "show_save_prompt": True,
                    "save_prompt_message": "Do you want to save this AI web search answer to the Q&A database?",
                    "temp_question": question_text,
                    "temp_answer": enhanced_result["answer"],
                    "temp_sources": [source.get("url", "") for source in sources],
                    "cache_hit": False,
                    "search_summary": web_search_result.get("search_summary", "")
                }
        
        # STEP 5: Final fallback to ChatGPT-3.5
        print("🤖 Step 5: Using ChatGPT-3.5 as final fallback...")
        search_attempts.append("AI generation")
        answer_text = await rag_service.generate_answer(question_text)
        
        generation_time = int((time.time() - start_time) * 1000)
        
        # If we got this far, we couldn't find a good answer anywhere
        # Log the unanswered question for later analysis
        try:
            no_answer_handler.log_unanswered_question(db, question_text, search_attempts)
        except Exception as e:
            print(f"⚠️ Error logging unanswered question: {str(e)}")
        
        # Generate alternative question suggestions
        alternative_questions = no_answer_handler.suggest_related_questions(question_text, rag_service)
        
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
            "found_on_web": False,
            "show_save_prompt": True,
            "save_prompt_message": "Do you want to save this answer to the Q&A database?",
            "temp_question": question_text,
            "temp_answer": answer_text,
            "temp_sources": [],
            "cache_hit": False,
            "search_attempts": search_attempts,
            "alternative_questions": alternative_questions,
            "show_feedback_form": True
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
        source_urls = save_request.get("source_urls", [])
        
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
        
        # Store source URLs if available (for web answers)
        if source_urls and answer_type == "web_search":
            try:
                # Store as WebAnswer for better tracking
                web_answer = WebAnswer(
                    question_id=new_question.id,
                    answer_text=answer_text,
                    sources=[{"url": url} for url in source_urls],
                    confidence_score=confidence_score
                )
                db.add(web_answer)
                db.commit()
                print(f"✅ Stored web answer sources: {source_urls}")
            except Exception as e:
                print(f"⚠️ Warning: Could not store web sources: {str(e)}")
        
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

# Document Management Endpoints - Fixed to prevent 404
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
        
        # Ensure tables exist
        try:
            Base.metadata.create_all(bind=engine)
            print("✅ Database tables verified/created")
        except Exception as table_error:
            print(f"⚠️ Warning creating tables: {str(table_error)}")
        
        # Get total count safely
        try:
            count_result = db.execute(text("SELECT COUNT(*) FROM documents"))
            total_count = count_result.fetchone()[0]
            print(f"📊 Total documents in database: {total_count}")
        except Exception as count_error:
            print(f"❌ Error getting document count: {str(count_error)}")
            # Table might not exist, return empty result
            return {
                "documents": [],
                "total": 0,
                "message": "Documents table not found or empty",
                "debug_info": {
                    "table_exists": False,
                    "total_count": 0,
                    "error": str(count_error)
                }
            }
        
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
            return {
                "documents": [],
                "total": 0,
                "message": f"Error querying documents: {str(query_error)}",
                "debug_info": {
                    "table_exists": True,
                    "query_executed": False,
                    "error": str(query_error)
                }
            }
        
    except Exception as e:
        print(f"❌ Unexpected error listing documents: {str(e)}")
        return {
            "documents": [],
            "total": 0,
            "message": f"Error retrieving documents: {str(e)}",
            "debug_info": {
                "table_exists": False,
                "error": str(e),
                "query_executed": False
            }
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
        
        print(f"📤 Uploading document: {file.filename} ({len(file_content)} bytes)")
        
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
            "version": "4.1.0 - AI-Native Web Search",
            "response_priority": [
                "1. Built-in Knowledge Base (fastest)",
                "2. PDF Document Search", 
                "3. Q&A Database Cache",
                "4. AI-Native Web Search",
                "5. ChatGPT Generation"
            ],
            "ai_web_search": ai_web_search.get_search_capabilities_info(),
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
            "version": "4.1.0 - AI-Native Web Search",
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
        # Ensure tables exist
        Base.metadata.create_all(bind=engine)
        
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

@app.post("/ai-web-search")
async def test_ai_web_search(
    request: dict,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Test endpoint for AI-native web search functionality"""
    try:
        query = request.get("query", "").strip()
        
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query is required"
            )
        
        print(f"🧪 Testing AI web search for: {query}")
        
        # Perform AI web search
        search_result = await ai_web_search.search_web_with_ai(query)
        
        # Generate enhanced answer
        enhanced_result = await ai_web_search.generate_answer_with_web_context(
            query, search_result
        )
        
        return {
            "query": query,
            "search_result": search_result,
            "enhanced_answer": enhanced_result,
            "capabilities": ai_web_search.get_search_capabilities_info(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in AI web search test: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error testing AI web search: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
