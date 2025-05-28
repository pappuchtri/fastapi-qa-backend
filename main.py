from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional, Dict, Any
import os
import uuid
from datetime import datetime
import numpy as np
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
from schemas import (
    QuestionAnswerRequest, 
    QuestionAnswerResponse, 
    HealthResponse, 
    AuthHealthResponse,
    QuestionResponse,
    AnswerResponse
)
from rag_service import RAGService
from pdf_service import PDFService
import crud

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
            "documents": "/documents",
            "upload": "/documents/upload",
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
        relevant_chunks = []
        source_documents = []
        
        try:
            # Convert embedding to list for SQL query
            embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            # Search for similar chunks using cosine similarity
            result = db.execute(text("""
                SELECT dc.id, dc.document_id, dc.content, dc.page_number, 
                       d.original_filename,
                       1 - (dc.chunk_embedding <=> :embedding) as similarity
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE dc.chunk_embedding IS NOT NULL
                ORDER BY dc.chunk_embedding <=> :embedding
                LIMIT 3
            """), {"embedding": embedding_list})
            
            for row in result:
                chunk_id, doc_id, content, page_number, filename, similarity = row
                relevant_chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "content": content,
                    "page_number": page_number,
                    "filename": filename,
                    "similarity": similarity
                })
                source_documents.append(filename)
                
            print(f"📄 Found {len(relevant_chunks)} relevant document chunks")
        except Exception as e:
            print(f"⚠️ Error searching document chunks: {str(e)}")
        
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
            # Generate new answer with document context
            print("🧠 Generating new answer with document context...")
            
            # Prepare context from relevant chunks
            context = ""
            if relevant_chunks:
                context_parts = []
                for chunk in relevant_chunks:
                    context_parts.append(f"From {chunk['filename']} (Page {chunk['page_number'] or 'N/A'}):\n{chunk['content']}")
                
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
    page: int = 1,
    per_page: int = 10,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """List all uploaded documents"""
    try:
        # Calculate offset
        offset = (page - 1) * per_page
        
        # Get total count
        total_result = db.execute(text("SELECT COUNT(*) FROM documents"))
        total = total_result.fetchone()[0]
        
        # Query documents with pagination
        result = db.execute(text("""
            SELECT id, filename, original_filename, file_size, 
                   content_type, upload_date, processed, processing_status,
                   total_pages, total_chunks
            FROM documents 
            ORDER BY upload_date DESC
            LIMIT :limit OFFSET :offset
        """), {"limit": per_page, "offset": offset})
        
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
            "total": total,
            "page": page,
            "per_page": per_page
        }
        
    except Exception as e:
        print(f"❌ Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )

@app.post("/documents/upload")
async def upload_document(
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
        
        # Process PDF
        result = await pdf_service.process_pdf(
            db, file_content, unique_filename, file.filename
        )
        
        return {
            "message": "Document uploaded and processing started",
            "document_id": result["document_id"],
            "filename": file.filename,
            "file_size": len(file_content),
            "processing_status": result.get("status", "processing")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )

@app.get("/documents/{document_id}")
async def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get document details"""
    try:
        # Get document
        result = db.execute(text("""
            SELECT id, filename, original_filename, file_size, 
                   content_type, upload_date, processed, processing_status,
                   error_message, total_pages, total_chunks
            FROM documents 
            WHERE id = :id
        """), {"id": document_id})
        
        document = result.fetchone()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Get chunk count
        chunk_result = db.execute(text("""
            SELECT COUNT(*) FROM document_chunks WHERE document_id = :id
        """), {"id": document_id})
        
        chunk_count = chunk_result.fetchone()[0]
        
        return {
            "id": document[0],
            "filename": document[1],
            "original_filename": document[2],
            "file_size": document[3],
            "content_type": document[4],
            "upload_date": document[5],
            "processed": document[6],
            "processing_status": document[7],
            "error_message": document[8],
            "total_pages": document[9],
            "total_chunks": document[10],
            "actual_chunk_count": chunk_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting document: {str(e)}"
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

@app.post("/documents/search")
async def search_documents(
    query: str,
    limit: int = 5,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Search for relevant document chunks"""
    try:
        if not query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query cannot be empty"
            )
        
        # Generate embedding for search query
        query_embedding = await rag_service.generate_embedding(query)
        
        # Convert embedding to list for SQL query
        embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Search for similar chunks using cosine similarity
        result = db.execute(text("""
            SELECT dc.id, dc.document_id, dc.content, dc.page_number, 
                   d.original_filename,
                   1 - (dc.chunk_embedding <=> :embedding) as similarity
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.chunk_embedding IS NOT NULL
            ORDER BY dc.chunk_embedding <=> :embedding
            LIMIT :limit
        """), {"embedding": embedding_list, "limit": limit})
        
        results = []
        for row in result:
            chunk_id, doc_id, content, page_number, filename, similarity = row
            results.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "document_filename": filename,
                "content": content,
                "page_number": page_number,
                "similarity_score": similarity
            })
        
        return {
            "results": results,
            "query": query,
            "total_results": len(results)
        }
        
    except HTTPException:
        raise
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
    try:
        # Get counts from database
        question_count = db.execute(text("SELECT COUNT(*) FROM questions")).fetchone()[0]
        answer_count = db.execute(text("SELECT COUNT(*) FROM answers")).fetchone()[0]
        document_count = db.execute(text("SELECT COUNT(*) FROM documents")).fetchone()[0]
        chunk_count = db.execute(text("SELECT COUNT(*) FROM document_chunks")).fetchone()[0]
        embedding_count = db.execute(text("SELECT COUNT(*) FROM embeddings")).fetchone()[0]
        
        # Get processing status counts
        processing_counts = {}
        status_result = db.execute(text("""
            SELECT processing_status, COUNT(*) 
            FROM documents 
            GROUP BY processing_status
        """))
        
        for row in status_result:
            processing_counts[row[0]] = row[1]
        
        return {
            "questions": question_count,
            "answers": answer_count,
            "documents": document_count,
            "chunks": chunk_count,
            "embeddings": embedding_count,
            "document_status": processing_counts,
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
