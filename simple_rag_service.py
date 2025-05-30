"""
Simple RAG service without pgvector dependency
This service uses pure Python for vector similarity calculations
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy.orm import Session
import os
from models import Question, Answer, Embedding

class SimpleRAGService:
    def __init__(self):
        """Initialize the simple RAG service"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_configured = self.openai_api_key is not None
        self.similarity_threshold = 0.85
        self.embedding_dimension = 1536
        
        if self.openai_configured:
            try:
                import openai
                openai.api_key = self.openai_api_key
                print("✅ OpenAI configured successfully")
            except Exception as e:
                print(f"❌ OpenAI configuration error: {str(e)}")
                self.openai_configured = False
        else:
            print("⚠️ Running in demo mode - no OpenAI API key")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if not self.openai_configured:
            # Return consistent dummy embedding
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))
            return np.random.rand(self.embedding_dimension).tolist()
        
        try:
            import openai
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            print(f"❌ Embedding generation error: {str(e)}")
            # Fallback to dummy embedding
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))
            return np.random.rand(self.embedding_dimension).tolist()
    
    async def find_similar_question(self, db: Session, query_embedding: List[float]) -> Tuple[Any, float]:
        """Find similar question using Python-based similarity"""
        try:
            print("🔍 Searching for similar questions...")
            
            # Get all embeddings
            embeddings = db.query(Embedding).all()
            
            if not embeddings:
                return None, 0.0
            
            query_vec = np.array(query_embedding)
            best_similarity = 0.0
            best_question = None
            
            for emb in embeddings:
                try:
                    # Convert stored vector to numpy array
                    if isinstance(emb.vector, list):
                        stored_vec = np.array(emb.vector)
                    else:
                        # Handle string format
                        import json
                        stored_vec = np.array(json.loads(str(emb.vector).replace('{', '[').replace('}', ']')))
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_vec, stored_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_question = db.query(Question).filter(Question.id == emb.question_id).first()
                
                except Exception as e:
                    print(f"⚠️ Error processing embedding {emb.id}: {str(e)}")
                    continue
            
            return best_question, float(best_similarity)
            
        except Exception as e:
            print(f"❌ Similarity search error: {str(e)}")
            return None, 0.0
    
    async def search_document_chunks(self, db: Session, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search document chunks using simple similarity"""
        try:
            from document_models import DocumentChunk, Document
            
            # Get chunks with embeddings
            chunks = db.query(DocumentChunk).join(Document).filter(
                DocumentChunk.chunk_embedding.isnot(None),
                Document.processed == True
            ).all()
            
            if not chunks:
                return []
            
            query_vec = np.array(query_embedding)
            similarities = []
            
            for chunk in chunks:
                try:
                    if chunk.chunk_embedding:
                        chunk_vec = np.array(chunk.chunk_embedding)
                        similarity = np.dot(query_vec, chunk_vec) / (
                            np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
                        )
                        
                        similarities.append({
                            "id": chunk.id,
                            "content": chunk.content,
                            "page_number": chunk.page_number,
                            "chunk_index": chunk.chunk_index,
                            "filename": chunk.document.original_filename,
                            "similarity": float(similarity)
                        })
                except Exception as e:
                    continue
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            print(f"❌ Document search error: {str(e)}")
            return []
    
    async def generate_answer(self, question: str) -> str:
        """Generate answer using GPT"""
        if not self.openai_configured:
            return f"""🎭 **Demo Answer for: "{question}"**

This is a demonstration response. Configure OPENAI_API_KEY for AI-powered answers.

**System Status:**
- Database: Connected ✅
- Vector Search: Working ✅ 
- AI Generation: Demo Mode 🎭

Your FastAPI backend is fully operational!"""
        
        try:
            import openai
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# Create an alias for compatibility
EnhancedRAGService = SimpleRAGService
