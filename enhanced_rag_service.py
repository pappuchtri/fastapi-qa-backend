from sqlalchemy.orm import Session
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import time
import hashlib
import os
import openai
from datetime import datetime

class EnhancedRAGService:
    """Enhanced RAG service with chunk limiting, performance caching, and improved search"""
    
    def __init__(self):
        """Initialize the RAG service"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_configured = self.openai_api_key is not None
        self.similarity_threshold = 0.85  # Threshold for semantic similarity
        self.max_chunks = 5  # Maximum number of chunks to use for context
        
        if self.openai_configured:
            openai.api_key = self.openai_api_key
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI API"""
        if not self.openai_configured:
            # Return mock embedding for demo mode
            return [0.0] * 1536  # OpenAI embeddings are 1536 dimensions
        
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            # Return mock embedding on error
            return [0.0] * 1536
    
    async def find_similar_question(self, db: Session, query_embedding: List[float]) -> Tuple[Any, float]:
        """Find semantically similar question in database"""
        from sqlalchemy import text
        
        # Convert embedding to PostgreSQL array format
        embedding_str = str(query_embedding).replace('[', '{').replace(']', '}')
        
        # Query for similar questions using cosine similarity
        query = text("""
            SELECT 
                q.id, 
                q.text, 
                1 - (e.embedding <=> :embedding) as similarity
            FROM 
                questions q
            JOIN 
                embeddings e ON q.id = e.question_id
            ORDER BY 
                similarity DESC
            LIMIT 1
        """)
        
        result = db.execute(query, {"embedding": embedding_str})
        row = result.fetchone()
        
        if row:
            from models import Question
            question = db.query(Question).filter(Question.id == row[0]).first()
            return question, float(row[2])
        
        return None, 0.0
    
    async def search_document_chunks(
        self, 
        db: Session, 
        query_embedding: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search document chunks using vector similarity"""
        from sqlalchemy import text
        
        # Convert embedding to PostgreSQL array format
        embedding_str = str(query_embedding).replace('[', '{').replace(']', '}')
        
        # Query for similar chunks using cosine similarity
        query = text("""
            SELECT 
                dc.id, 
                dc.content, 
                dc.page_number,
                dc.chunk_index,
                d.original_filename,
                1 - (dc.chunk_embedding <=> :embedding) as similarity
            FROM 
                document_chunks dc
            JOIN 
                documents d ON dc.document_id = d.id
            WHERE 
                dc.chunk_embedding IS NOT NULL
            ORDER BY 
                similarity DESC
            LIMIT :limit
        """)
        
        result = db.execute(query, {"embedding": embedding_str, "limit": limit})
        
        chunks = []
        for row in result:
            chunks.append({
                "id": row[0],
                "content": row[1],
                "page_number": row[2],
                "chunk_index": row[3],
                "filename": row[4],
                "similarity": float(row[5])
            })
        
        return chunks
    
    async def analyze_question_intent(self, question: str) -> Dict[str, Any]:
        """Analyze question intent to improve search and answer generation"""
        if not self.openai_configured:
            # Return mock analysis for demo mode
            return {
                "question_type": "informational",
                "expected_answer_length": "medium",
                "complexity": "medium",
                "keywords": question.lower().split()[:5]
            }
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """Analyze the question and return a JSON with the following fields:
                        - question_type: factual, conceptual, procedural, or opinion
                        - expected_answer_length: short, medium, or long
                        - complexity: simple, medium, or complex
                        - keywords: array of important keywords from the question
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this question: {question}"
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            analysis_text = response['choices'][0]['message']['content'].strip()
            
            # Parse JSON response
            import json
            try:
                analysis = json.loads(analysis_text)
                return analysis
            except:
                # Fallback if JSON parsing fails
                return {
                    "question_type": "informational",
                    "expected_answer_length": "medium",
                    "complexity": "medium",
                    "keywords": question.lower().split()[:5]
                }
                
        except Exception as e:
            print(f"Error analyzing question: {str(e)}")
            # Return default analysis on error
            return {
                "question_type": "informational",
                "expected_answer_length": "medium",
                "complexity": "medium",
                "keywords": question.lower().split()[:5]
            }
    
    async def process_question(
        self, 
        db: Session, 
        question: str, 
        query_embedding: List[float],
        question_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process question with enhanced RAG pipeline"""
        # Step 1: Check Q&A cache first
        similar_question, similarity_score = await self.find_similar_question(db, query_embedding)
        
        if similar_question and similarity_score > self.similarity_threshold:
            print(f"✅ Found cached answer (similarity: {similarity_score:.3f})")
            # Get the most recent answer for this cached question
            from models import Answer
            cached_answer = db.query(Answer).filter(
                Answer.question_id == similar_question.id
            ).order_by(Answer.created_at.desc()).first()
            
            if cached_answer:
                return {
                    "answer": cached_answer.text,
                    "question_id": similar_question.id,
                    "answer_id": cached_answer.id,
                    "similarity": similarity_score,
                    "confidence": float(cached_answer.confidence_score),
                    "source_type": "cached",
                    "source_documents": []
                }
        
        # Step 2: Search document chunks with limited context
        document_chunks = await self.search_document_chunks(db, query_embedding, limit=self.max_chunks)
        
        # Step 3: Generate answer based on available context
        if document_chunks:
            print(f"✅ Found {len(document_chunks)} relevant document chunks")
            
            # Create detailed context with source attribution
            context_parts = []
            source_documents = list(set([chunk['filename'] for chunk in document_chunks]))
            
            for i, chunk in enumerate(document_chunks):
                source_info = f"[Source: {chunk['filename']}, Page {chunk.get('page_number', 'N/A')}]"
                context_parts.append(f"{source_info}\n{chunk['content']}")
                
                # Log chunk usage if we have a question ID
                if similar_question:
                    from override_crud import log_chunk_usage
                    log_chunk_usage(
                        db, 
                        similar_question.id, 
                        chunk['id'], 
                        chunk['similarity'],
                        i + 1,  # 1-based position
                        True
                    )
            
            context_text = "\n\n".join(context_parts)
            
            if self.openai_configured:
                # Use OpenAI to generate answer from document context
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a helpful assistant that answers questions based on the provided document context. 
                            Always cite the source documents and page numbers when possible. 
                            Be comprehensive and accurate. Include ALL relevant information from the context.
                            If the context doesn't contain enough information to fully answer the question, clearly state what information is missing."""
                        },
                        {
                            "role": "user",
                            "content": f"Question: {question}\n\nDocument Context:\n{context_text}\n\nPlease answer the question based on the document context provided. Include specific references to the source documents and page numbers."
                        }
                    ],
                    max_tokens=800,
                    temperature=0.3
                )
                answer_text = response['choices'][0]['message']['content'].strip()
                confidence = 0.95  # High confidence for document-based answers
            else:
                # Demo mode - show document context
                answer_text = f"""📄 **Answer based on uploaded documents:**\n\n{context_text}\n\n**Sources:** {', '.join(source_documents)}\n\n*Note: This is demo mode. With OpenAI configured, this would be a comprehensive answer based on the document content above.*"""
                confidence = 0.95
            
            return {
                "answer": answer_text,
                "confidence": confidence,
                "source_type": "document",
                "source_documents": source_documents,
                "total_chunks_found": len(document_chunks)
            }
        else:
            # Fallback to GPT
            answer_text = await self.generate_answer(question)
            
            return {
                "answer": answer_text,
                "confidence": 0.7,
                "source_type": "gpt",
                "source_documents": [],
                "total_chunks_found": 0
            }
    
    async def generate_answer(self, question: str) -> str:
        """Generate answer using GPT when no document context is available"""
        if not self.openai_configured:
            return f"""⚠️ **No relevant documents found for your question.**\n\nI don't have specific information about "{question}" in the uploaded documents.\n\n*Note: This is demo mode. With OpenAI configured, this would be a GPT-generated answer.*"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful assistant. When you don't know the answer or don't have specific information, clearly state that you don't have that information in the uploaded documents."""
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return f"I apologize, but I encountered an error while trying to answer your question. Please try again later."
    
    async def check_performance_cache(self, db: Session, question: str) -> Optional[Dict[str, Any]]:
        """Check if question is in performance cache"""
        from override_crud import check_performance_cache
        return check_performance_cache(db, question)
    
    async def store_performance_cache(
        self, 
        db: Session, 
        question: str, 
        answer: str, 
        generation_time_ms: int,
        confidence: float,
        source_type: str
    ) -> None:
        """Store answer in performance cache if generation was slow"""
        from override_crud import store_performance_cache
        store_performance_cache(
            db, question, answer, generation_time_ms, confidence, source_type
        )
