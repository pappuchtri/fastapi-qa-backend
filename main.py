from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Header, BackgroundTasks
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
import asyncio
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
from document_schemas import (
    DocumentResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    DocumentSearchRequest,
    DocumentSearchResponse
)
from rag_service import RAGService
from pdf_service import PDFService
import crud
import document_crud

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
rag_service = RAGService()
pdf_service = PDFService(rag_service)

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
            "upload": "/documents/upload",
            "documents": "/documents",
            "search": "/documents/search"
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

@app.post("/ask", response_model=QuestionAnswerResponse)
async def ask_question(
    request: QuestionAnswerRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Ask a question and get an AI-generated answer using RAG with document context"""
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
        
        # Search for similar questions first
        similar_question, similarity_score = await rag_service.find_similar_question(
            db, query_embedding
        )
        
        # Search for relevant document chunks
        relevant_chunks = document_crud.search_similar_chunks(
            db, query_embedding, limit=3
        )
        
        # Determine if we should use cached answer or generate new one
        is_cached = False
        answer_text = ""
        source_documents = []
        
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
            # Generate new answer with document context
            print("🧠 Generating new answer with document context...")
            
            # Prepare context from relevant chunks
            context = ""
            if relevant_chunks:
                print(f"📄 Found {len(relevant_chunks)} relevant document chunks")
                context_parts = []
                for chunk in relevant_chunks:
                    doc = document_crud.get_document(db, chunk.document_id)
                    if doc:
                        context_parts.append(f"From {doc.original_filename} (Page {chunk.page_number or 'N/A'}):\n{chunk.content}")
                        source_documents.append(doc.original_filename)
                
                context = "\n\n".join(context_parts)
            
            # Generate answer with context
            if context:
                enhanced_question = f"""Context from uploaded documents:
{context}

Question: {question_text}

Please answer the question based on the provided context. If the context doesn't contain relevant information, please say so and provide a general answer."""
            else:
                enhanced_question = question_text
            
            answer_text = await rag_service.generate_answer(enhanced_question)
            
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
        
        return QuestionAnswerResponse(
            answer=answer_text,
            question_id=question_id,
            answer_id=answer_id,
            similarity_score=similarity_score,
            is_cached=is_cached,
            source_documents=list(set(source_documents))  # Remove duplicates
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )

# Document Management Endpoints

@app.post("/documents/upload", response_model=DocumentUploadResponse)
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
        
        # Process PDF in background
        document = await pdf_service.process_pdf(
            db, file_content, unique_filename, file.filename
        )
        
        return DocumentUploadResponse(
            message="Document uploaded and processed successfully",
            document_id=document.id,
            filename=document.original_filename,
            file_size=document.file_size,
            processing_status=document.processing_status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )

@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = 1,
    per_page: int = 10,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """List all uploaded documents"""
    skip = (page - 1) * per_page
    documents = document_crud.get_documents(db, skip=skip, limit=per_page)
    total = document_crud.get_document_count(db)
    
    return DocumentListResponse(
        documents=documents,
        total=total,
        page=page,
        per_page=per_page
    )

@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get document details"""
    document = document_crud.get_document(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return document

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Delete a document and all its chunks"""
    success = document_crud.delete_document(db, document_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return {"message": "Document deleted successfully"}

@app.post("/documents/search", response_model=DocumentSearchResponse)
async def search_documents(
    request: DocumentSearchRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Search for relevant document chunks"""
    try:
        # Generate embedding for search query
        query_embedding = await rag_service.generate_embedding(request.query)
        
        # Search for similar chunks
        chunks = document_crud.search_similar_chunks(
            db, query_embedding, 
            limit=request.limit,
            document_ids=request.document_ids
        )
        
        # Format results
        results = []
        for chunk in chunks:
            document = document_crud.get_document(db, chunk.document_id)
            if document:
                # Calculate similarity score
                chunk_vector = np.array(chunk.chunk_embedding) if chunk.chunk_embedding else None
                similarity_score = 0.0
                if chunk_vector is not None:
                    similarity_score = float(np.dot(query_embedding, chunk_vector) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_vector)
                    ))
                
                results.append({
                    "chunk_id": chunk.id,
                    "document_id": document.id,
                    "document_filename": document.original_filename,
                    "content": chunk.content,
                    "page_number": chunk.page_number,
                    "similarity_score": similarity_score
                })
        
        return DocumentSearchResponse(
            results=results,
            query=request.query,
            total_results=len(results)
        )
        
    except Exception as e:
        print(f"❌ Error searching documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching documents: {str(e)}"
        )

@app.get("/stats")
async def get_stats(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get API statistics"""
    question_count = crud.get_question_count(db)
    answer_count = crud.get_answer_count(db)
    document_count = document_crud.get_document_count(db)
    chunk_count = document_crud.get_chunk_count(db)
    
    return {
        "questions": question_count,
        "answers": answer_count,
        "documents": document_count,
        "chunks": chunk_count,
        "embeddings": len(crud.get_all_embeddings(db)),
        "openai_configured": rag_service.openai_configured,
        "similarity_threshold": rag_service.similarity_threshold
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
