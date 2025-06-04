from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os
import uuid
import shutil
import json
import time
from sqlalchemy.orm import Session
import numpy as np

# Import database and models
from database import get_db
from document_models import Document, DocumentChunk
import document_crud as crud

# Import RAG service
from enhanced_rag_service import EnhancedRAGService
from models import Question, Answer, Embedding

app = FastAPI(title="Document RAG API", version="1.0.0")

# Initialize RAG service
rag_service = EnhancedRAGService()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class StatsResponse(BaseModel):
    questions: int
    answers: int
    documents: int
    chunks: int
    openai_configured: bool

class QACacheItem(BaseModel):
    question_id: int
    question: str
    question_date: str
    answer: str
    confidence: float
    answer_date: str

class QACacheResponse(BaseModel):
    qa_pairs: List[QACacheItem]

class QuestionRequest(BaseModel):
    question: str

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Document RAG API is running", "status": "healthy"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": str(datetime.utcnow())}

# Stats endpoint
@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    try:
        # Get actual document and chunk counts from database
        document_count = crud.get_document_count(db)
        chunk_count = crud.get_chunk_count(db)
        
        # Get question and answer counts
        question_count = db.query(Question).count() if Question.__table__.exists(db.bind) else 0
        answer_count = db.query(Answer).count() if Answer.__table__.exists(db.bind) else 0
        
        stats = {
            "questions": question_count,
            "answers": answer_count,
            "documents": document_count,
            "chunks": chunk_count,
            "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
        }
        return stats
    except Exception as e:
        print(f"Error fetching stats: {str(e)}")
        # Return default stats if there's an error
        return {
            "questions": 0,
            "answers": 0,
            "documents": 0,
            "chunks": 0,
            "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
        }

# QA Cache endpoint
@app.get("/qa-cache")
async def get_qa_cache(db: Session = Depends(get_db)):
    try:
        # Get recent question-answer pairs
        qa_pairs = []
        
        # Check if tables exist
        if Question.__table__.exists(db.bind) and Answer.__table__.exists(db.bind):
            # Get recent questions with answers
            recent_questions = db.query(Question).order_by(Question.created_at.desc()).limit(10).all()
            
            for question in recent_questions:
                # Get the most recent answer for this question
                answer = db.query(Answer).filter(Answer.question_id == question.id).order_by(Answer.created_at.desc()).first()
                
                if answer:
                    qa_pairs.append({
                        "question_id": question.id,
                        "question": question.text,
                        "question_date": question.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "answer": answer.text,
                        "confidence": float(answer.confidence_score) / 100 if answer.confidence_score else 0.9,
                        "answer_date": answer.created_at.strftime("%Y-%m-%d %H:%M:%S")
                    })
        
        return {"qa_pairs": qa_pairs}
    except Exception as e:
        print(f"Error fetching QA cache: {str(e)}")
        return {"qa_pairs": []}

# Documents endpoints
@app.get("/documents")
async def list_documents(
    skip: int = 0, 
    limit: int = 100,
    db: Session = Depends(get_db)
):
    try:
        # Get documents from database
        documents = crud.get_documents(db, skip=skip, limit=limit)
        total = crud.get_document_count(db)
        
        # Format response
        response = {
            "documents": documents,
            "total": total,
            "page": skip // limit + 1 if limit > 0 else 1,
            "per_page": limit
        }
        return response
    except Exception as e:
        print(f"Error listing documents: {str(e)}")
        return {"documents": [], "total": 0, "page": 1, "per_page": limit}

@app.get("/documents/{document_id}")
async def get_document(document_id: int, db: Session = Depends(get_db)):
    try:
        # Get document from database
        document = crud.get_document(db, document_id=document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching document: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    try:
        # Delete document from database
        success = crud.delete_document(db, document_id=document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": f"Document {document_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

# Document upload endpoint
@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Generate a unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.pdf"
        
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Save the file
        file_path = f"uploads/{filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Create document in database
        document_data = {
            "filename": filename,
            "original_filename": file.filename,
            "file_size": file_size,
            "content_type": "application/pdf",
            "processed": False,
            "processing_status": "uploaded",
            "error_message": None,
            "total_pages": None,
            "total_chunks": 0,
            "doc_metadata": {}
        }
        
        db_document = Document(**document_data)
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        # Return document data
        return {
            "message": "Document uploaded successfully",
            "document_id": db_document.id,
            "filename": db_document.filename,
            "file_size": db_document.file_size,
            "processing_status": db_document.processing_status
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

# Document chunks endpoint
@app.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: int, db: Session = Depends(get_db)):
    try:
        # Get chunks from database
        chunks = crud.get_document_chunks(db, document_id=document_id)
        return {"chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching document chunks: {str(e)}")

# Ask endpoint
@app.post("/ask")
async def ask_question(request: Request, db: Session = Depends(get_db)):
    try:
        # Parse request body
        body = await request.json()
        question = body.get("question", "")
        
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        print(f"Processing question: {question}")
        start_time = time.time()
        
        # Check performance cache first
        cached_response = await rag_service.check_performance_cache(db, question)
        if cached_response:
            print("Found in performance cache")
            return {
                "answer": cached_response["answer"],
                "question_id": cached_response.get("question_id", 0),
                "answer_id": cached_response.get("answer_id", 0),
                "similarity_score": 1.0,
                "is_cached": True,
                "source_documents": [],
                "answer_type": "cached",
                "confidence_score": cached_response.get("confidence", 0.95),
                "generation_time_ms": 0,
                "found_in_pdf": False,
                "show_save_prompt": False
            }
        
        # Generate embedding for the question
        query_embedding = await rag_service.generate_embedding(question)
        
        # Analyze question intent
        question_analysis = await rag_service.analyze_question_intent(question)
        
        # Process question with RAG
        rag_response = await rag_service.process_question(db, question, query_embedding, question_analysis)
        
        # Store question in database
        db_question = Question(text=question)
        db.add(db_question)
        db.commit()
        db.refresh(db_question)
        
        # Store embedding
        db_embedding = Embedding(
            question_id=db_question.id,
            embedding=json.dumps(query_embedding)
        )
        db.add(db_embedding)
        
        # Store answer
        confidence = rag_response.get("confidence", 0.9)
        db_answer = Answer(
            question_id=db_question.id,
            text=rag_response["answer"],
            confidence_score=int(confidence * 100)
        )
        db.add(db_answer)
        db.commit()
        db.refresh(db_answer)
        
        # Calculate generation time
        generation_time_ms = int((time.time() - start_time) * 1000)
        
        # Store in performance cache if generation was slow
        if generation_time_ms > 1000:  # If generation took more than 1 second
            await rag_service.store_performance_cache(
                db, 
                question, 
                rag_response["answer"], 
                generation_time_ms,
                confidence,
                rag_response["source_type"]
            )
        
        # Format response
        response = {
            "answer": rag_response["answer"],
            "question_id": db_question.id,
            "answer_id": db_answer.id,
            "similarity_score": 0.0,  # No similar question found
            "is_cached": False,
            "source_documents": rag_response.get("source_documents", []),
            "answer_type": rag_response["source_type"],
            "confidence_score": confidence,
            "generation_time_ms": generation_time_ms,
            "found_in_pdf": rag_response["source_type"] == "document",
            "show_save_prompt": rag_response["source_type"] == "gpt"
        }
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        # Fallback response
        return {
            "answer": f"I apologize, but I encountered an error while processing your question. Error: {str(e)}",
            "question_id": 0,
            "answer_id": 0,
            "similarity_score": 0.0,
            "is_cached": False,
            "source_documents": [],
            "answer_type": "error",
            "confidence_score": 0.0,
            "generation_time_ms": 0,
            "found_in_pdf": False,
            "show_save_prompt": False
        }

# Save answer endpoint
@app.post("/save-answer")
async def save_answer(request: Request, db: Session = Depends(get_db)):
    try:
        # Parse request body
        body = await request.json()
        question = body.get("question")
        answer = body.get("answer")
        
        if not question or not answer:
            raise HTTPException(status_code=400, detail="Question and answer are required")
        
        # Store in database
        db_question = Question(text=question)
        db.add(db_question)
        db.commit()
        db.refresh(db_question)
        
        db_answer = Answer(
            question_id=db_question.id,
            text=answer,
            confidence_score=95  # High confidence for manually saved answers
        )
        db.add(db_answer)
        db.commit()
        
        # Generate and store embedding
        query_embedding = await rag_service.generate_embedding(question)
        db_embedding = Embedding(
            question_id=db_question.id,
            embedding=json.dumps(query_embedding)
        )
        db.add(db_embedding)
        db.commit()
        
        return {"success": True, "knowledge_base_id": db_question.id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving answer: {str(e)}")

# Suggest questions endpoint
@app.post("/suggest-questions")
async def suggest_questions(request: Request):
    try:
        # Parse request body
        body = await request.json()
        question = body.get("question", "")
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        # Use OpenAI to generate suggested questions
        if os.getenv("OPENAI_API_KEY"):
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful assistant that suggests related questions. 
                        Given an original question, suggest 3 related follow-up questions that would help explore the topic further.
                        Return ONLY the questions as a JSON array, nothing else."""
                    },
                    {
                        "role": "user",
                        "content": f"Original question: {question}"
                    }
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            suggestions_text = response['choices'][0]['message']['content'].strip()
            
            # Try to parse as JSON
            try:
                import json
                suggestions = json.loads(suggestions_text)
                if isinstance(suggestions, list):
                    suggested_questions = suggestions
                else:
                    suggested_questions = suggestions.get("questions", [])
            except:
                # If JSON parsing fails, extract questions using simple heuristics
                suggested_questions = [q.strip() for q in suggestions_text.split('\n') if q.strip() and '?' in q]
                if not suggested_questions:
                    suggested_questions = [
                        f"Can you explain more about {question}?",
                        f"What are the key aspects of {question}?",
                        f"How does {question} relate to other topics?"
                    ]
        else:
            # Default suggestions if OpenAI is not configured
            suggested_questions = [
                f"Can you explain more about {question}?",
                f"What are the key aspects of {question}?",
                f"How does {question} relate to other topics?"
            ]
        
        response = {
            "original_question": question,
            "suggested_questions": suggested_questions[:3],  # Limit to 3 questions
            "reasoning": f"These questions help explore different aspects of '{question}'"
        }
        return response
    except HTTPException:
        raise
    except Exception as e:
        # Fallback suggestions
        return {
            "original_question": question,
            "suggested_questions": [
                f"Can you explain more about {question}?",
                f"What are the key aspects of {question}?",
                f"How does {question} relate to other topics?"
            ],
            "reasoning": "Default suggestions due to error"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
