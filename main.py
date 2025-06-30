from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text, func
import os
import logging
from datetime import datetime
import traceback
import json

# Import your modules
from database import get_db, engine
from models import Document, Question, Answer, DocumentChunk
from schemas import QuestionRequest, AnswerResponse, DocumentResponse, StatsResponse
from document_crud import create_document, get_documents, delete_document
from enhanced_rag_service import EnhancedRAGService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API with PDF processing",
    version="1.0.0"
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Security
security = HTTPBearer()
API_KEY = "dev-api-key-123"

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return credentials.credentials

# Initialize RAG service
rag_service = EnhancedRAGService()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG API is running",
        "version": "1.0.0",
        "status": "online",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/test")
async def test_endpoint():
    """Test endpoint for connectivity checks"""
    try:
        return {
            "status": "success",
            "message": "Backend is online and responding",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "api_key_required": False,
            "endpoints": {
                "health": "/health",
                "stats": "/stats (requires auth)",
                "documents": "/documents (requires auth)",
                "ask": "/ask (requires auth)",
                "upload": "/documents/upload (requires auth)"
            }
        }
    except Exception as e:
        logger.error(f"Test endpoint error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Test endpoint failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            db_connected = result.fetchone() is not None
        
        return {
            "status": "healthy" if db_connected else "unhealthy",
            "database_connected": db_connected,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "message": "Health check completed successfully"
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "database_connected": False,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "message": "Health check failed"
            }
        )

@app.get("/stats", dependencies=[Depends(verify_api_key)])
async def get_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    try:
        questions_count = db.query(Question).count()
        answers_count = db.query(Answer).count()
        documents_count = db.query(Document).count()
        chunks_count = db.query(DocumentChunk).count()
        
        return StatsResponse(
            questions=questions_count,
            answers=answers_count,
            documents=documents_count,
            chunks=chunks_count,
            openai_configured=bool(os.getenv("OPENAI_API_KEY"))
        )
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.get("/documents", dependencies=[Depends(verify_api_key)])
async def list_documents(db: Session = Depends(get_db)):
    """List all documents"""
    try:
        documents = get_documents(db)
        return {"documents": documents}
    except Exception as e:
        logger.error(f"List documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.post("/documents/upload", dependencies=[Depends(verify_api_key)])
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process a PDF document"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save uploaded file
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create document record
        document = create_document(
            db=db,
            filename=file.filename,
            file_path=file_path,
            file_size=len(content)
        )
        
        # Process document with RAG service
        try:
            await rag_service.process_document(file_path, document.id, db)
            logger.info(f"Document {file.filename} processed successfully")
        except Exception as process_error:
            logger.error(f"Document processing error: {str(process_error)}")
            # Don't fail the upload, just log the error
        
        return {"message": "Document uploaded successfully", "document_id": document.id}
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.delete("/documents/{document_id}", dependencies=[Depends(verify_api_key)])
async def remove_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document"""
    try:
        success = delete_document(db, document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Delete document error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.post("/ask", dependencies=[Depends(verify_api_key)])
async def ask_question(request: QuestionRequest, db: Session = Depends(get_db)):
    """Ask a question and get an AI-powered answer"""
    try:
        result = await rag_service.get_answer(request.question, db)
        return result
    except Exception as e:
        logger.error(f"Ask question error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get answer: {str(e)}")

@app.get("/qa-cache", dependencies=[Depends(verify_api_key)])
async def get_qa_cache(db: Session = Depends(get_db)):
    """Get cached Q&A pairs"""
    try:
        qa_pairs = db.query(Question, Answer).join(Answer, Question.id == Answer.question_id).all()
        
        result = []
        for question, answer in qa_pairs:
            result.append({
                "question_id": question.id,
                "question": question.question_text,
                "question_date": question.created_at.isoformat(),
                "answer": answer.answer_text,
                "confidence": answer.confidence_score,
                "answer_date": answer.created_at.isoformat()
            })
        
        return {"qa_pairs": result}
    except Exception as e:
        logger.error(f"QA cache error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get QA cache: {str(e)}")

@app.post("/save-answer", dependencies=[Depends(verify_api_key)])
async def save_answer(request: dict, db: Session = Depends(get_db)):
    """Save an answer to the knowledge base"""
    try:
        # Create question record
        question = Question(
            question_text=request["question"],
            created_at=datetime.now()
        )
        db.add(question)
        db.flush()
        
        # Create answer record
        answer = Answer(
            question_id=question.id,
            answer_text=request["answer"],
            answer_type=request.get("answer_type", "saved"),
            confidence_score=request.get("confidence_score", 1.0),
            created_at=datetime.now()
        )
        db.add(answer)
        db.commit()
        
        return {"message": "Answer saved successfully", "question_id": question.id}
    except Exception as e:
        logger.error(f"Save answer error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save answer: {str(e)}")

@app.post("/suggest-questions", dependencies=[Depends(verify_api_key)])
async def suggest_questions(request: QuestionRequest, db: Session = Depends(get_db)):
    """Get question suggestions based on input"""
    try:
        suggestions = await rag_service.suggest_questions(request.question, db)
        return suggestions
    except Exception as e:
        logger.error(f"Suggest questions error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")

# Handle OPTIONS requests for CORS
@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle CORS preflight requests"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
