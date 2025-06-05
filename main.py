from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# Schema imports
from schemas import QuestionCreate, AnswerResponse, DocumentResponse
from document_schemas import DocumentCreate, DocumentUpdate

# Service imports
from rag_service import RAGService
from knowledge_base_service import KnowledgeBaseService
from document_crud import DocumentCRUD
from ai_web_search_service import AIWebSearchService
from enhanced_citation_service import EnhancedCitationService
from no_answer_handler import NoAnswerHandler
from feedback_handler import FeedbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
rag_service = RAGService()
kb_service = KnowledgeBaseService()
doc_crud = DocumentCRUD()
web_search_service = AIWebSearchService()
citation_service = EnhancedCitationService()
no_answer_handler = NoAnswerHandler()
feedback_handler = FeedbackHandler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    try:
        logger.info("🚀 Starting FastAPI application...")
        
        # Create all database tables
        logger.info("📊 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        
        # Initialize knowledge base
        logger.info("🧠 Initializing knowledge base...")
        await kb_service.initialize_knowledge_base()
        logger.info("✅ Knowledge base initialized")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("🛑 Shutting down FastAPI application...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Enhanced RAG System",
    description="A comprehensive RAG system with Knowledge Base, PDF search, and AI web search",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with system status"""
    try:
        # Test database connection
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            db_status = "✅ Connected"
        except Exception as e:
            db_status = f"❌ Error: {str(e)}"
        finally:
            db.close()
        
        # Check knowledge base
        kb_count = await kb_service.get_knowledge_base_count()
        
        # Check documents
        doc_count = await doc_crud.get_document_count()
        
        return {
            "message": "Enhanced RAG System API",
            "version": "2.0.0",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "database": db_status,
            "knowledge_base_entries": kb_count,
            "documents": doc_count,
            "features": {
                "knowledge_base_search": True,
                "pdf_document_search": True,
                "ai_web_search": True,
                "save_to_knowledge_base": True,
                "enhanced_citations": True,
                "feedback_system": True
            },
            "endpoints": {
                "ask_question": "/ask",
                "upload_document": "/upload",
                "documents": "/documents",
                "knowledge_base": "/knowledge-base",
                "ai_web_search": "/ai-web-search",
                "feedback": "/feedback"
            }
        }
    except Exception as e:
        logger.error(f"Error in root endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "details": str(e)}
        )

@app.post("/ask")
async def ask_question(
    question_data: QuestionCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Enhanced question answering with multiple search strategies"""
    start_time = time.time()
    
    try:
        question_text = question_data.text.strip()
        if not question_text:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"🔍 Processing question: {question_text}")
        
        # Create question record
        question = Question(text=question_text)
        db.add(question)
        db.commit()
        db.refresh(question)
        
        # Step 1: Search Knowledge Base
        logger.info("📚 Step 1: Searching Knowledge Base...")
        kb_result = await kb_service.search_knowledge_base(question_text, db)
        
        if kb_result and kb_result.get("found"):
            processing_time = int((time.time() - start_time) * 1000)
            logger.info(f"✅ Found answer in Knowledge Base (took {processing_time}ms)")
            
            # Create answer record
            answer = Answer(
                question_id=question.id,
                text=kb_result["answer"],
                confidence_score=kb_result.get("confidence", 0.95)
            )
            db.add(answer)
            db.commit()
            
            return AnswerResponse(
                answer=kb_result["answer"],
                source="knowledge_base",
                confidence_score=kb_result.get("confidence", 0.95),
                generation_time_ms=processing_time,
                source_documents=[{
                    "title": "Knowledge Base",
                    "content": kb_result.get("category", "General"),
                    "page": 1,
                    "relevance_score": kb_result.get("confidence", 0.95)
                }],
                can_save_to_kb=False,
                question_id=question.id
            )
        
        # Step 2: Search PDF Documents
        logger.info("📄 Step 2: Searching PDF Documents...")
        try:
            pdf_result = await rag_service.get_answer(question_text, db)
            
            if pdf_result and pdf_result.get("answer") and pdf_result.get("source_documents"):
                processing_time = int((time.time() - start_time) * 1000)
                logger.info(f"✅ Found answer in PDF documents (took {processing_time}ms)")
                
                # Create answer record
                answer = Answer(
                    question_id=question.id,
                    text=pdf_result["answer"],
                    confidence_score=pdf_result.get("confidence_score", 0.8)
                )
                db.add(answer)
                db.commit()
                
                # Enhanced citations for PDF sources
                enhanced_sources = citation_service.enhance_pdf_citations(
                    pdf_result.get("source_documents", [])
                )
                
                return AnswerResponse(
                    answer=pdf_result["answer"],
                    source="document",
                    confidence_score=pdf_result.get("confidence_score", 0.8),
                    generation_time_ms=processing_time,
                    source_documents=enhanced_sources,
                    can_save_to_kb=True,
                    question_id=question.id
                )
        except Exception as e:
            logger.warning(f"PDF search failed: {e}")
        
        # Step 3: Search Q&A Database
        logger.info("🗃️ Step 3: Searching Q&A Database...")
        try:
            similar_questions = db.query(Question).join(Answer).filter(
                Question.text.ilike(f"%{question_text}%")
            ).limit(3).all()
            
            if similar_questions:
                best_match = similar_questions[0]
                best_answer = best_match.answers[0] if best_match.answers else None
                
                if best_answer:
                    processing_time = int((time.time() - start_time) * 1000)
                    logger.info(f"✅ Found similar answer in Q&A database (took {processing_time}ms)")
                    
                    return AnswerResponse(
                        answer=best_answer.text,
                        source="database",
                        confidence_score=0.7,
                        generation_time_ms=processing_time,
                        source_documents=[{
                            "title": "Q&A Database",
                            "content": f"Similar question: {best_match.text}",
                            "page": 1,
                            "relevance_score": 0.7
                        }],
                        can_save_to_kb=True,
                        question_id=question.id
                    )
        except Exception as e:
            logger.warning(f"Database search failed: {e}")
        
        # Step 4: AI Web Search
        logger.info("🌐 Step 4: Performing AI Web Search...")
        try:
            web_result = await web_search_service.search_web(question_text, db)
            
            if web_result and web_result.get("answer"):
                processing_time = int((time.time() - start_time) * 1000)
                logger.info(f"✅ Found answer via AI web search (took {processing_time}ms)")
                
                # Create answer record
                answer = Answer(
                    question_id=question.id,
                    text=web_result["answer"],
                    confidence_score=web_result.get("confidence_score", 0.75)
                )
                db.add(answer)
                db.commit()
                
                # Enhanced citations for web sources
                enhanced_sources = citation_service.enhance_web_citations(
                    web_result.get("sources", [])
                )
                
                return AnswerResponse(
                    answer=web_result["answer"],
                    source="ai_web_search",
                    confidence_score=web_result.get("confidence_score", 0.75),
                    generation_time_ms=processing_time,
                    source_documents=enhanced_sources,
                    web_sources=web_result.get("sources", []),
                    can_save_to_kb=True,
                    question_id=question.id
                )
        except Exception as e:
            logger.warning(f"AI web search failed: {e}")
        
        # Step 5: Fallback to ChatGPT
        logger.info("🤖 Step 5: Fallback to ChatGPT...")
        try:
            gpt_result = await rag_service.get_chatgpt_fallback(question_text)
            processing_time = int((time.time() - start_time) * 1000)
            
            # Create answer record
            answer = Answer(
                question_id=question.id,
                text=gpt_result,
                confidence_score=0.6
            )
            db.add(answer)
            db.commit()
            
            logger.info(f"✅ Generated ChatGPT response (took {processing_time}ms)")
            
            return AnswerResponse(
                answer=gpt_result,
                source="chatgpt",
                confidence_score=0.6,
                generation_time_ms=processing_time,
                source_documents=[],
                can_save_to_kb=True,
                question_id=question.id
            )
        except Exception as e:
            logger.error(f"ChatGPT fallback failed: {e}")
        
        # Step 6: No answer found
        logger.warning("❌ No answer found from any source")
        processing_time = int((time.time() - start_time) * 1000)
        
        # Log unanswered question
        await no_answer_handler.log_unanswered_question(question_text, db)
        
        # Get suggestions
        suggestions = await no_answer_handler.get_suggestions(question_text, db)
        
        return AnswerResponse(
            answer="I couldn't find a specific answer to your question. You might try rephrasing your question or checking if there are any typos.",
            source="no_answer",
            confidence_score=0.0,
            generation_time_ms=processing_time,
            source_documents=[],
            can_save_to_kb=False,
            question_id=question.id,
            suggestions=suggestions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/save-to-kb")
async def save_to_knowledge_base(
    data: dict,
    db: Session = Depends(get_db)
):
    """Save an answer to the knowledge base"""
    try:
        question_text = data.get("question")
        answer_text = data.get("answer")
        category = data.get("category", "General")
        
        if not question_text or not answer_text:
            raise HTTPException(status_code=400, detail="Question and answer are required")
        
        # Save to knowledge base
        result = await kb_service.add_to_knowledge_base(
            question=question_text,
            answer=answer_text,
            category=category,
            db=db
        )
        
        return {"success": True, "message": "Answer saved to knowledge base", "id": result.id}
        
    except Exception as e:
        logger.error(f"Error saving to knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(""),
    db: Session = Depends(get_db)
):
    """Upload and process a document"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read file content
        content = await file.read()
        
        # Create document record
        document_data = DocumentCreate(
            title=title,
            description=description,
            filename=file.filename,
            content_type=file.content_type or "application/pdf"
        )
        
        # Process document
        result = await doc_crud.create_document(db, document_data, content)
        
        return {
            "success": True,
            "message": "Document uploaded and processed successfully",
            "document_id": result.id,
            "chunks_created": result.chunk_count
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents(db: Session = Depends(get_db)):
    """Get all documents"""
    try:
        documents = await doc_crud.get_documents(db)
        return {
            "documents": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "description": doc.description,
                    "filename": doc.filename,
                    "upload_date": doc.upload_date.isoformat(),
                    "chunk_count": doc.chunk_count
                }
                for doc in documents
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        return {"documents": []}

@app.get("/knowledge-base")
async def get_knowledge_base(db: Session = Depends(get_db)):
    """Get knowledge base entries"""
    try:
        entries = await kb_service.get_all_entries(db)
        return {
            "entries": [
                {
                    "id": entry.id,
                    "category": entry.category,
                    "question": entry.question,
                    "answer": entry.answer,
                    "priority": entry.priority,
                    "created_at": entry.created_at.isoformat()
                }
                for entry in entries
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching knowledge base: {e}")
        return {"entries": []}

@app.post("/ai-web-search")
async def ai_web_search(data: dict, db: Session = Depends(get_db)):
    """Test AI web search functionality"""
    try:
        query = data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        result = await web_search_service.search_web(query, db)
        return result
        
    except Exception as e:
        logger.error(f"Error in AI web search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(data: dict, db: Session = Depends(get_db)):
    """Submit user feedback"""
    try:
        result = await feedback_handler.submit_feedback(data, db)
        return {"success": True, "message": "Feedback submitted successfully", "id": result.id}
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
