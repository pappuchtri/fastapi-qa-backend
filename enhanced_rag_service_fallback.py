import numpy as np
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session

class EnhancedRAGServiceFallback:
    """
    Enhanced RAG service with fallback to Python-based similarity calculation.
    This class provides methods for finding similar questions and searching document chunks
    using embeddings and cosine similarity. It falls back to Python-based calculations
    when pgvector is not available.
    """

    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Ensure vectors have the same length
            if len(vec1) != len(vec2):
                print(f"⚠️ Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
                return 0.0
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            
            if norm_vec1 == 0 or norm_vec2 == 0:
                return 0.0
            
            similarity = dot_product / (norm_vec1 * norm_vec2)
            return float(similarity)
            
        except Exception as e:
            print(f"❌ Error calculating similarity: {str(e)}")
            return 0.0

    async def find_similar_question(self, db: Session, query_embedding: List[float]) -> Tuple[Any, float]:
        """Find semantically similar question in database using standard PostgreSQL"""
        try:
            from sqlalchemy import text
            from models import Question, Embedding
            
            # Get all embeddings and calculate similarity in Python
            embeddings = db.query(Embedding).all()
            
            if not embeddings:
                return None, 0.0
            
            best_similarity = 0.0
            best_question = None
            
            query_vector = np.array(query_embedding)
            
            for embedding in embeddings:
                try:
                    # Convert stored embedding to numpy array
                    if isinstance(embedding.vector, list):
                        stored_vector = np.array(embedding.vector)
                    else:
                        # Handle string format if needed
                        import json
                        stored_vector = np.array(json.loads(str(embedding.vector)))
                    
                    # Calculate cosine similarity
                    similarity = self.calculate_cosine_similarity(query_vector, stored_vector)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        question = db.query(Question).filter(Question.id == embedding.question_id).first()
                        best_question = question
                        
                except Exception as e:
                    print(f"⚠️ Error processing embedding {embedding.id}: {str(e)}")
                    continue
            
            return best_question, best_similarity
            
        except Exception as e:
            print(f"❌ Error in similarity search: {str(e)}")
            return None, 0.0

    async def search_document_chunks(
        self, 
        db: Session, 
        query_embedding: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search document chunks using Python-based similarity calculation"""
        try:
            from document_models import DocumentChunk, Document
            
            # Get all document chunks with embeddings
            chunks = db.query(DocumentChunk).join(Document).filter(
                DocumentChunk.chunk_embedding.isnot(None),
                Document.processed == True
            ).all()
            
            if not chunks:
                print("📭 No document chunks with embeddings found")
                return []
            
            print(f"📊 Found {len(chunks)} document chunks to search")
            
            query_vector = np.array(query_embedding)
            similarities = []
            
            for chunk in chunks:
                try:
                    if chunk.chunk_embedding:
                        # Convert stored embedding to numpy array
                        if isinstance(chunk.chunk_embedding, list):
                            chunk_vector = np.array(chunk.chunk_embedding)
                        else:
                            import json
                            chunk_vector = np.array(json.loads(str(chunk.chunk_embedding)))
                    
                    # Calculate cosine similarity
                    similarity = self.calculate_cosine_similarity(query_vector, chunk_vector)
                    
                    similarities.append({
                        "chunk": chunk,
                        "similarity": float(similarity),
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "page_number": chunk.page_number,
                        "filename": chunk.document.original_filename if chunk.document else "Unknown"
                    })
            except Exception as e:
                print(f"⚠️ Error processing chunk {chunk.id}: {str(e)}")
                continue
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Filter by minimum similarity threshold
            min_similarity = 0.5
            relevant_chunks = [
                {
                    "id": item["chunk_id"],
                    "content": item["content"],
                    "page_number": item["page_number"],
                    "chunk_index": getattr(item["chunk"], "chunk_index", 0),
                    "filename": item["filename"],
                    "similarity": item["similarity"]
                }
                for item in similarities 
                if item["similarity"] >= min_similarity
            ][:limit]
            
            print(f"🎯 Found {len(relevant_chunks)} relevant chunks (similarity >= {min_similarity})")
            return relevant_chunks
            
        except Exception as e:
            print(f"❌ Error in document chunk search: {str(e)}")
            return []
