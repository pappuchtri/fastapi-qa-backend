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
from schemas import (
    QuestionAnswerRequest, 
    QuestionAnswerResponse, 
    HealthResponse, 
    AuthHealthResponse,
    QuestionResponse,
    AnswerResponse
)
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
    logger.info("🚀 Starting PDF RAG Q&A API...")
    
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
        logger.info("🤖 OpenAI API Key configured")
    else:
        logger.info("🎭 Running in demo mode (no OpenAI key)")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down PDF RAG Q&A API...")

# Initialize FastAPI app
app = FastAPI(
    title="PDF RAG Q&A API",
    description="A FastAPI backend for PDF document processing and Q&A using RAG",
    version="1.0.0",
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
        "message": "PDF RAG Q&A API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "auth_health": "/auth/health",
            "ask": "/ask",
            "questions": "/questions",
            "documents": "/documents",
            "upload": "/documents/upload"
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
        message="API is running",
        timestamp=datetime.utcnow(),
        database_connected=database_connected,
        openai_configured=rag_service.openai_configured
    )

@app.get("/auth/health", response_model=AuthHealthResponse)
async def auth_health_check(api_key: str = Depends(verify_api_key)):
    """Authenticated health check endpoint"""
    return AuthHealthResponse(
        status="authenticated",
        message="API key is valid",
        timestamp=datetime.utcnow()
    )

@app.post("/ask", response_model=QuestionAnswerResponse)
async def ask_question(
    request: QuestionAnswerRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Ask a question and get an AI-generated answer using RAG"""
    try:
        question_text = request.question.strip()
        
        if not question_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        print(f"🔍 Processing question: {question_text[:50]}...")
        
        # Generate embedding for the question
        query_embedding = await rag_service.generate_embedding(question_text)
        
        # Search for similar questions
        similar_question, similarity_score = await rag_service.find_similar_question(
            db, query_embedding
        )
        
        # Determine if we should use cached answer or generate new one
        is_cached = False
        answer_text = ""
        
        if similar_question and similarity_score >= rag_service.similarity_threshold:
            # Use cached answer from similar question
            print(f"✅ Found similar question (similarity: {similarity_score:.3f})")
            latest_answer = crud.get_answers_for_question(db, similar_question.id)
            if latest_answer:
                answer_text = latest_answer[0].text
                is_cached = True
                question_id = similar_question.id
                answer_id = latest_answer[0].id
                print("📋 Using cached answer")
        
        if not is_cached:
            # Generate new answer
            print("🧠 Generating new answer...")
            answer_text = await rag_service.generate_answer(question_text)
            
            # Store new question and answer
            new_question = crud.create_question(db, crud.QuestionCreate(text=question_text))
            question_id = new_question.id
            
            # Store embedding
            crud.create_embedding(db, question_id, query_embedding)
            
            # Store answer
            new_answer = crud.create_answer(
                db, 
                crud.AnswerCreate(
                    question_id=question_id,
                    text=answer_text,
                    confidence_score=0.95
                )
            )
            answer_id = new_answer.id
            print("💾 Stored new question and answer")
        
        # Search for relevant document chunks using better text search
        relevant_chunks = []
        source_documents = []

        try:
            # First try vector similarity search if embeddings are available
            if hasattr(rag_service, 'search_document_chunks'):
                relevant_chunks = await rag_service.search_document_chunks(db, query_embedding, limit=5)
            
            # Fallback to text-based search if no vector search results
            if not relevant_chunks:
                # Use PostgreSQL full-text search for better results
                result = db.execute(text("""
                    SELECT dc.id, dc.document_id, dc.content, dc.page_number, 
                           d.original_filename, d.id as doc_id
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.processed = true 
                    AND (
                        dc.content ILIKE :query1 OR
                        dc.content ILIKE :query2 OR
                        to_tsvector('english', dc.content) @@ plainto_tsquery('english', :query3)
                    )
                    ORDER BY 
                        CASE 
                            WHEN dc.content ILIKE :query1 THEN 1
                            WHEN dc.content ILIKE :query2 THEN 2
                            ELSE 3
                        END,
                        ts_rank(to_tsvector('english', dc.content), plainto_tsquery('english', :query3)) DESC
                    LIMIT 5
                """), {
                    "query1": f"%{question_text}%",
                    "query2": f"%{' '.join(question_text.split()[:3])}%",  # First 3 words
                    "query3": question_text
                })
                
                for row in result:
                    chunk_id, doc_id, content, page_number, filename, document_id = row
                    relevant_chunks.append({
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                        "content": content[:500],  # Limit content length
                        "page_number": page_number,
                        "filename": filename
                    })
                    if filename not in source_documents:
                        source_documents.append(filename)
            
            print(f"📄 Found {len(relevant_chunks)} relevant document chunks from {len(source_documents)} documents")
            
            # If still no results, try broader search
            if not relevant_chunks:
                # Search for any documents that might be relevant
                keywords = question_text.lower().split()
                keyword_queries = []
                for keyword in keywords[:5]:  # Use first 5 keywords
                    if len(keyword) > 3:  # Skip short words
                        keyword_queries.append(f"dc.content ILIKE '%{keyword}%'")
                
                if keyword_queries:
                    broad_query = f"""
                        SELECT dc.id, dc.document_id, dc.content, dc.page_number, 
                               d.original_filename
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE d.processed = true AND ({' OR '.join(keyword_queries)})
                        LIMIT 3
                    """
                    
                    result = db.execute(text(broad_query))
                    for row in result:
                        chunk_id, doc_id, content, page_number, filename = row
                        relevant_chunks.append({
                            "chunk_id": chunk_id,
                            "document_id": doc_id,
                            "content": content[:500],
                            "page_number": page_number,
                            "filename": filename
                        })
                        if filename not in source_documents:
                            source_documents.append(filename)
                    
                    print(f"📄 Broad search found {len(relevant_chunks)} additional chunks")

        except Exception as e:
            print(f"⚠️ Error searching document chunks: {str(e)}")

        # Enhanced context preparation
        context = ""
        if relevant_chunks:
            print(f"🔍 Preparing context from {len(relevant_chunks)} chunks")
            context_parts = []
            for i, chunk in enumerate(relevant_chunks):
                chunk_preview = chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content']
                context_parts.append(
                    f"[Document {i+1}: {chunk['filename']} - Page {chunk['page_number'] or 'N/A'}]\n{chunk_preview}"
                )
            
            context = "\n\n".join(context_parts)
            print(f"📝 Context prepared: {len(context)} characters")

        # Generate answer with enhanced prompting
        if not is_cached:
            print("🧠 Generating new answer with document context...")
            
            if context:
                enhanced_question = f"""You are an AI assistant helping to answer questions based on uploaded documents. 

CONTEXT FROM UPLOADED DOCUMENTS:
{context}

USER QUESTION: {question_text}

INSTRUCTIONS:
1. First, carefully analyze the provided context from the uploaded documents
2. If the context contains relevant information to answer the question, use it as your primary source
3. Clearly indicate which document(s) you're referencing in your answer
4. If the context doesn't contain sufficient information, say so and provide a general answer
5. Be specific about page numbers when referencing document content
6. Keep your answer concise but comprehensive

Please provide a helpful answer based on the above context and question."""
            else:
                # Check if there are any documents at all
                doc_count = db.query(Document).filter(Document.processed == True).count()
                if doc_count > 0:
                    enhanced_question = f"""You are an AI assistant. The user has uploaded {doc_count} document(s) to the system, but none of them appear to contain information directly relevant to this question: "{question_text}"

Please provide a general answer to this question, and suggest that the user might want to upload more specific documents if they're looking for document-based information."""
                else:
                    enhanced_question = f"""You are an AI assistant. The user is asking: "{question_text}"

No documents have been uploaded to the system yet. Please provide a general answer to this question and suggest that they can upload relevant documents for more specific, document-based answers."""
            
            answer_text = await rag_service.generate_answer(enhanced_question)
        
        return QuestionAnswerResponse(
            answer=answer_text,
            question_id=question_id,
            answer_id=answer_id,
            similarity_score=similarity_score,
            is_cached=is_cached,
            source_documents=source_documents  # Will be populated when document processing is added
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )

@app.get("/questions", response_model=List[QuestionResponse])
async def get_questions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get a list of questions"""
    questions = crud.get_questions(db, skip=skip, limit=limit)
    return questions

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
            "note": "PDF processing is running in the background. Check back in a few moments."
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
            "similarity_threshold": rag_service.similarity_threshold
        }
        
    except Exception as e:
        print(f"❌ Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
