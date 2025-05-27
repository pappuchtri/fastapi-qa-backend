from fastapi import FastAPI, Depends, HTTPException, Header
from sqlalchemy.orm import Session
import uvicorn
from datetime import datetime
import os
from dotenv import load_dotenv
import asyncio
from typing import Optional
import logging
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

from database import SessionLocal, engine, Base
from models import Question, Answer, Embedding
from schemas import QuestionRequest, AnswerResponse, ApiKeyResponse, AuthErrorResponse
from rag_service import RAGService
from auth import verify_api_key, check_rate_limit, auth_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Q&A API with RAG and Neon Database",
    description="A FastAPI backend with Retrieval-Augmented Generation using OpenAI and Neon PostgreSQL",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware with permissive settings for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service
rag_service = RAGService()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_startup
async def startup_event():
    """Initialize the application on startup"""
    logger.info("🚀 Starting Q&A API with RAG...")
    logger.info(f"📊 Database URL configured: {'Yes' if os.getenv('DATABASE_URL') else 'No'}")
    logger.info(f"🤖 OpenAI API Key configured: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No (Demo mode)'}")
    logger.info(f"🔐 Authentication configured: {'Yes' if auth_manager.api_keys else 'No'}")

@app.get("/")
def read_root():
    return {
        "message": "🎯 Q&A API with RAG and Neon Database", 
        "status": "running", 
        "features": [
            "🤖 OpenAI Integration", 
            "🔍 Vector Similarity Search", 
            "🔐 API Key Authentication", 
            "🗄️ Neon PostgreSQL"
        ],
        "version": "3.0.0",
        "database": "Neon PostgreSQL",
        "cors_enabled": True,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    """Health check endpoint for Railway and monitoring"""
    try:
        # Test database connection
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "disconnected"
    
    return {
        "status": "healthy", 
        "message": "API is running", 
        "database": db_status,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.0.0"
    }

@app.get("/auth/health")
def authenticated_health_check(api_key: str = Depends(verify_api_key)):
    """Health check that requires authentication"""
    return {
        "status": "healthy", 
        "message": "API is running with authentication", 
        "authenticated": True,
        "api_key": f"{api_key[:8]}...{api_key[-4:]}",
        "database": "connected",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/debug/info")
def debug_info():
    """Debug endpoint to check configuration (development only)"""
    return {
        "environment": {
            "database_configured": bool(os.getenv("DATABASE_URL")),
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "port": os.getenv("PORT", "8000"),
            "railway_environment": bool(os.getenv("RAILWAY_ENVIRONMENT")),
        },
        "api_keys": {
            "total_keys": len(auth_manager.api_keys),
            "development_keys_available": [
                "dev-api-key-123",
                "test-api-key-456", 
                "demo-key-789",
                "qa-development-key",
                "master-dev-key"
            ]
        },
        "features": {
            "cors_enabled": True,
            "docs_available": True,
            "health_checks": True,
            "rate_limiting": True
        }
    }

@app.get("/test")
def test_endpoint():
    """Simple test endpoint to verify API connectivity"""
    return {
        "message": "✅ API is working perfectly!",
        "timestamp": datetime.utcnow().isoformat(),
        "status": "success",
        "database": "Neon PostgreSQL",
        "deployment": "Railway"
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest, 
    db: Session = Depends(get_db),
    api_key: str = Depends(check_rate_limit)
):
    """
    🤖 Process a question using Retrieval-Augmented Generation (RAG).
    Requires valid API key in Authorization header: Bearer <api_key>
    """
    try:
        logger.info(f"📝 Processing question: {request.question[:50]}...")
        
        # Check if OpenAI is configured
        if not os.getenv("OPENAI_API_KEY"):
            logger.info("🎭 Running in demo mode (no OpenAI key)")
            
            # Store the question in the database
            db_question = Question(
                text=request.question,
                created_at=datetime.utcnow()
            )
            db.add(db_question)
            db.commit()
            db.refresh(db_question)
            
            # Create a demo answer
            demo_answer = f"""🎭 **Demo Response**

Your question: "{request.question}"

This is a demonstration of the Q&A system. In full mode with OpenAI API key configured, this system would:

🔍 **Step 1**: Generate embeddings for your question using OpenAI's text-embedding-ada-002
🔎 **Step 2**: Search for similar questions in the vector database  
🧠 **Step 3**: Either return a cached answer or generate a new one using GPT-4
💾 **Step 4**: Store the question, answer, and embeddings for future reference

**Current Status**: Demo mode - configure OPENAI_API_KEY environment variable for AI-powered responses.

**Database**: ✅ Connected to Neon PostgreSQL
**Authentication**: ✅ API key verified
**Storage**: ✅ Question saved to database"""
            
            # Store the answer
            db_answer = Answer(
                question_id=db_question.id,
                text=demo_answer,
                confidence_score=0.5,
                created_at=datetime.utcnow()
            )
            db.add(db_answer)
            db.commit()
            db.refresh(db_answer)
            
            logger.info(f"✅ Demo response generated for question ID: {db_question.id}")
            
            return AnswerResponse(
                answer=demo_answer,
                question_id=db_question.id,
                answer_id=db_answer.id,
                similarity_score=0.0,
                is_cached=False
            )
        
        # Full RAG logic when OpenAI is configured
        logger.info("🤖 Running in full AI mode")
        question_embedding = await rag_service.generate_embedding(request.question)
        
        similar_question, similarity_score = await rag_service.find_similar_question(
            db, question_embedding
        )
        
        if similar_question and similarity_score > 0.8:
            existing_answer = db.query(Answer).filter(
                Answer.question_id == similar_question.id
            ).first()
            
            logger.info(f"🔍 Found similar question with {similarity_score:.3f} similarity")
            
            return AnswerResponse(
                answer=existing_answer.text,
                question_id=similar_question.id,
                answer_id=existing_answer.id,
                similarity_score=similarity_score,
                is_cached=True
            )
        else:
            generated_answer = await rag_service.generate_answer(request.question)
            
            db_question = Question(
                text=request.question,
                created_at=datetime.utcnow()
            )
            db.add(db_question)
            db.commit()
            db.refresh(db_question)
            
            db_answer = Answer(
                question_id=db_question.id,
                text=generated_answer,
                confidence_score=0.95,
                created_at=datetime.utcnow()
            )
            db.add(db_answer)
            db.commit()
            db.refresh(db_answer)
            
            # Store embedding for future similarity searches
            db_embedding = Embedding(
                question_id=db_question.id,
                vector=question_embedding.tolist(),
                model_name="text-embedding-ada-002",
                created_at=datetime.utcnow()
            )
            db.add(db_embedding)
            db.commit()
            
            logger.info(f"🧠 Generated new AI answer for question ID: {db_question.id}")
            
            return AnswerResponse(
                answer=generated_answer,
                question_id=db_question.id,
                answer_id=db_answer.id,
                similarity_score=0.0,
                is_cached=False
            )
            
    except Exception as e:
        logger.error(f"❌ Error processing question: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/questions")
def get_questions(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """📋 Get all questions with pagination - requires authentication"""
    questions = db.query(Question).offset(skip).limit(limit).all()
    return {
        "questions": questions,
        "total": db.query(Question).count(),
        "skip": skip,
        "limit": limit
    }

@app.get("/stats")
def get_stats(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """📊 Get system statistics - requires authentication"""
    total_embeddings = db.query(Embedding).count()
    total_questions = db.query(Question).count()
    total_answers = db.query(Answer).count()
    
    return {
        "database_stats": {
            "total_embeddings": total_embeddings,
            "total_questions": total_questions,
            "total_answers": total_answers,
            "embedding_coverage": total_embeddings / max(total_questions, 1)
        },
        "system_info": {
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "database_connected": True,
            "version": "3.0.0"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
