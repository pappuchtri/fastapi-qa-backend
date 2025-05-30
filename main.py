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
    description="A comprehensive FastAPI backend for PDF document processing and Q&A using RAG with feedback system",
    version="2.0.0",
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
        "message": "Enhanced PDF RAG Q&A API",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "PDF document processing",
            "Vector similarity search",
            "Historical Q&A integration",
            "Answer feedback system",
            "Low confidence flagging",
            "Question rephrasing suggestions",
            "GPT fallback for unknown questions"
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
            "debug/chunks": "/debug/chunks"
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

@app.post("/ask", response_model=QuestionAnswerResponse)
async def ask_question(
    request: QuestionAnswerRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Ask a question with enhanced features: caching, confidence flagging, source attribution"""
    try:
        question_text = request.question.strip()
        
        if not question_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        print(f"🔍 Processing question: {question_text[:50]}...")
        
        # Step 1: Generate embedding for semantic search
        query_embedding = await rag_service.generate_embedding(question_text)
        
        # Step 2: Check Q&A cache first
        print("🔍 STEP 1: Checking Q&A cache for similar questions...")
        similar_question, similarity_score = await rag_service.find_similar_question(db, query_embedding)
        
        answer_type = "gpt"  # Default
        confidence = 0.7
        source_documents = []
        low_confidence_flag = False
        
        if similar_question and similarity_score > rag_service.similarity_threshold:
            print(f"✅ Found cached answer (similarity: {similarity_score:.3f})")
            # Get the most recent answer for this cached question
            cached_answer = db.query(Answer).filter(
                Answer.question_id == similar_question.id
            ).order_by(Answer.created_at.desc()).first()
            
            if cached_answer:
                answer_text = cached_answer.text
                confidence = float(cached_answer.confidence_score)
                answer_type = "cached"
                
                # Check for low confidence
                if confidence < 0.80:
                    low_confidence_flag = True
                    print(f"⚠️ Low confidence cached answer: {confidence:.2f}")
                
                return QuestionAnswerResponse(
                    answer=answer_text,
                    question_id=similar_question.id,
                    answer_id=cached_answer.id,
                    similarity_score=similarity_score,
                    is_cached=True,
                    source_documents=[],
                    low_confidence=low_confidence_flag,
                    answer_type=answer_type,
                    confidence_score=confidence
                )
        
        print("⚠️ No cached answer found, searching documents...")
        
        # Step 3: Search document chunks
        print("🔍 STEP 2: Searching PDF documents...")
        document_chunks = await rag_service.search_document_chunks(db, query_embedding, limit=5)
        
        # Step 4: Generate answer based on available context
        if document_chunks:
            print(f"✅ Found {len(document_chunks)} relevant document chunks")
            answer_type = "document"
            confidence = 0.95  # High confidence for document-based answers
            
            # Create detailed context with source attribution
            context_parts = []
            source_documents = list(set([chunk['filename'] for chunk in document_chunks]))
            
            for chunk in document_chunks:
                source_info = f"[Source: {chunk['filename']}, Page {chunk.get('page_number', 'N/A')}]"
                context_parts.append(f"{source_info}\n{chunk['content'][:500]}...")
            
            context_text = "\n\n".join(context_parts)
            
            if rag_service.openai_configured:
                # Use OpenAI to generate answer from document context
                import openai
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions based on the provided document context. Always cite the source documents and page numbers when possible. Be comprehensive and accurate."
                        },
                        {
                            "role": "user",
                            "content": f"Question: {question_text}\n\nDocument Context:\n{context_text}\n\nPlease answer the question based on the document context provided. Include specific references to the source documents and page numbers."
                        }
                    ],
                    max_tokens=600,
                    temperature=0.3
                )
                answer_text = response['choices'][0]['message']['content'].strip()
            else:
                # Demo mode - show document context
                answer_text = f"""📄 **Answer based on uploaded documents:**

{context_text}

**Sources:** {', '.join(source_documents)}

*Note: This is demo mode. With OpenAI configured, this would be a comprehensive answer based on the document content above.*"""
            
            print(f"📄 Generated answer from {len(source_documents)} documents")
            
        else:
            print("⚠️ No relevant documents found, using GPT fallback")
            # Fallback to GPT
            answer_text = await rag_service.generate_answer(question_text)
            confidence = 0.7
            answer_type = "gpt"
            source_documents = []
        
        # Check for low confidence
        if confidence < 0.80:
            low_confidence_flag = True
            print(f"⚠️ Low confidence answer: {confidence:.2f}")
        
        # Step 5: Store the NEW question and answer in Q&A cache
        print("💾 Storing question and answer in Q&A cache...")
        new_question = crud.create_question(db, crud.QuestionCreate(text=question_text))
        question_id = new_question.id
        
        # Store embedding for future similarity searches
        crud.create_embedding(db, question_id, query_embedding)
        
        # Store answer
        new_answer = crud.create_answer(
            db, 
            crud.AnswerCreate(
                question_id=question_id,
                text=answer_text,
                confidence_score=confidence
            )
        )
        answer_id = new_answer.id
        
        print(f"✅ Cached Q&A pair (ID: {question_id}) for future use")
        
        # Prepare final response with enhanced metadata
        return QuestionAnswerResponse(
            answer=answer_text,
            question_id=question_id,
            answer_id=answer_id,
            similarity_score=0.0,  # New question, no similarity
            is_cached=False,  # This is a new answer
            source_documents=source_documents,
            low_confidence=low_confidence_flag,
            answer_type=answer_type,
            confidence_score=confidence
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
            suggestions = [line.strip('- ').strip() for line in lines if line.strip() and not line.startswith('Reasoning'))][:5]
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
        
        return {
            "questions": question_count,
            "answers": answer_count,
            "documents": document_count,
            "chunks": chunk_count,
            "openai_configured": rag_service.openai_configured,
            "version": "2.0.0 - Enhanced PDF RAG with Document Search"
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

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
