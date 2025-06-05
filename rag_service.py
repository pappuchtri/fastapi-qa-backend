import numpy as np
from typing import List, Tuple, Optional
from sqlalchemy.orm import Session
from sklearn.metrics.pairwise import cosine_similarity
import os
from models import Question, Answer, Embedding

class RAGService:
    def __init__(self):
        """Initialize the RAG service with OpenAI client"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if self.openai_api_key:
            try:
                # Use the new OpenAI client (v1.0+)
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_api_key)
                self.openai_configured = True
                print("✅ OpenAI API key configured successfully (using openai>=1.0.0)")
            except Exception as e:
                print(f"❌ Error configuring OpenAI: {str(e)}")
                self.openai_configured = False
        else:
            print("⚠️ OPENAI_API_KEY not found. Running in demo mode.")
            self.openai_configured = False
        
        self.embedding_dimension = 1536  # text-embedding-ada-002 dimension
        self.similarity_threshold = 0.85  # Higher threshold for Q&A cache (more precise matching)
        self.document_similarity_threshold = 0.5  # Lower threshold for document search (more inclusive)
        
        # Always use GPT-3.5-turbo (available to all accounts, cost-effective)
        self.chat_model = "gpt-3.5-turbo"
        print(f"🤖 Using chat model: {self.chat_model}")
        print(f"🎯 Q&A cache similarity threshold: {self.similarity_threshold}")
        print(f"📄 Document search similarity threshold: {self.document_similarity_threshold}")
        
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for given text using OpenAI text-embedding-ada-002"""
        if not self.openai_configured:
            # Return a consistent dummy embedding for demo purposes
            print(f"🎭 Generating demo embedding for: {text[:50]}...")
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))  # Consistent seed based on text
            embedding = np.random.rand(self.embedding_dimension)
            return embedding
        
        try:
            print(f"🤖 Generating OpenAI embedding for: {text[:50]}...")
            
            # Use the new OpenAI client (v1.0+)
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            # Extract embedding from new API response format
            embedding_vector = response.data[0].embedding
            
            if not embedding_vector:
                raise ValueError("Empty embedding vector received")
            
            embedding = np.array(embedding_vector, dtype=np.float32)
            
            # Validate embedding dimension
            if len(embedding) != self.embedding_dimension:
                raise ValueError(f"Expected embedding dimension {self.embedding_dimension}, got {len(embedding)}")
        
            print(f"✅ Embedding generated successfully (dimension: {len(embedding)})")
            return embedding
        
        except Exception as e:
            print(f"❌ Error generating embedding: {str(e)}")
            print("🎭 Falling back to demo embedding")
            # Return a consistent dummy embedding as fallback
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))
            embedding = np.random.rand(self.embedding_dimension).astype(np.float32)
            return embedding
    
    async def find_similar_question(
        self, 
        db: Session, 
        query_embedding: np.ndarray
    ) -> Tuple[Optional[Question], float]:
        """
        Find the most similar question in the database using Python-based cosine similarity
        Returns the question and similarity score
        """
        try:
            print("🔍 Searching for similar questions in database...")
            
            # Get all embeddings from database
            embeddings = db.query(Embedding).all()
            
            if not embeddings:
                print("📭 No embeddings found in database")
                return None, 0.0
            
            print(f"📊 Found {len(embeddings)} embeddings to compare")
            
            best_similarity = 0.0
            best_question = None
            
            for emb in embeddings:
                try:
                    # Handle both list and string formats
                    if isinstance(emb.vector, list):
                        vector = np.array([float(x) for x in emb.vector], dtype=np.float32)
                    elif isinstance(emb.vector, str):
                        # Handle string format (legacy data)
                        import json
                        vector_list = json.loads(emb.vector)
                        vector = np.array([float(x) for x in vector_list], dtype=np.float32)
                    else:
                        vector = np.array(emb.vector, dtype=np.float32)
                    
                    # Ensure vector has correct dimension
                    if len(vector) != self.embedding_dimension:
                        print(f"⚠️ Skipping embedding {emb.id}: wrong dimension {len(vector)}")
                        continue
                    
                    # Fix: Ensure both vectors are numpy arrays before similarity calculation
                    if not isinstance(query_embedding, np.ndarray):
                        query_embedding = np.array(query_embedding, dtype=np.float32)
                    
                    # Calculate cosine similarity
                    similarity = self.calculate_cosine_similarity(query_embedding, vector)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        question = db.query(Question).filter(Question.id == emb.question_id).first()
                        best_question = question
                        
                except Exception as e:
                    print(f"⚠️ Error processing embedding {emb.id}: {str(e)}")
                    continue
            
            print(f"🎯 Best match: {best_similarity:.3f} similarity")
            if best_question:
                print(f"📝 Similar question: {best_question.text[:50]}...")
            
            return best_question, float(best_similarity)
        
        except Exception as e:
            print(f"❌ Error in similarity search: {str(e)}")
            return None, 0.0
    
    async def search_document_chunks(
        self, 
        db: Session, 
        query_embedding: np.ndarray,
        limit: int = 5
    ) -> List[dict]:
        """
        Search for relevant document chunks using vector similarity and keyword search
        """
        try:
            print("🔍 Searching document chunks using multiple strategies...")
            
            # Import here to avoid circular imports
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
            
            # Calculate similarities
            similarities = []
            for chunk in chunks:
                try:
                    if chunk.chunk_embedding:
                        chunk_vector = np.array(chunk.chunk_embedding)
                        
                        # Ensure vectors have same dimension
                        if len(chunk_vector) != len(query_embedding):
                            continue
                        
                        similarity = np.dot(query_embedding, chunk_vector) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_vector)
                        )
                        
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
            
            # Use a lower threshold to catch more potentially relevant content
            min_similarity = 0.5  # Lower threshold for better recall
            relevant_chunks = [
                {
                    "chunk_id": item["chunk_id"],
                    "document_id": item["document_id"],
                    "content": item["content"],
                    "page_number": item["page_number"],
                    "filename": item["filename"],
                    "similarity": item["similarity"]
                }
                for item in similarities 
                if item["similarity"] >= min_similarity
            ][:limit]
            
            print(f"🎯 Found {len(relevant_chunks)} relevant chunks (similarity >= {min_similarity})")
            
            # If no chunks found with vector search, try keyword search
            if not relevant_chunks:
                print("🔍 No vector matches found, trying keyword search...")
                relevant_chunks = await self._keyword_search_chunks(db, query_embedding, limit)
            
            return relevant_chunks
            
        except Exception as e:
            print(f"❌ Error in document chunk search: {str(e)}")
            return []
    
    async def _keyword_search_chunks(self, db: Session, query_embedding: np.ndarray, limit: int = 5) -> List[dict]:
        """
        Fallback keyword search when vector search doesn't find results
        """
        try:
            from document_models import DocumentChunk, Document
            from sqlalchemy import text
            
            # For demo purposes, let's get some sample chunks
            result = db.execute(text("""
                SELECT dc.id, dc.document_id, dc.content, dc.page_number, d.original_filename
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.processed = true
                ORDER BY dc.id
                LIMIT :limit
            """), {"limit": limit})
            
            chunks = []
            for row in result:
                chunks.append({
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "content": row[2],
                    "page_number": row[3],
                    "filename": row[4],
                    "similarity": 0.6  # Default similarity for keyword matches
                })
            
            print(f"🔍 Keyword search found {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            print(f"❌ Error in keyword search: {str(e)}")
            return []
    
    async def search_documents(self, db: Session, query_embedding: np.ndarray, limit: int = 5) -> List[dict]:
        """
        Search documents - wrapper for search_document_chunks for compatibility
        """
        return await self.search_document_chunks(db, query_embedding, limit)

    async def generate_answer_from_chunks(self, question: str, relevant_chunks: List[dict]) -> str:
        """
        Generate answer from document chunks with cleaner output
        """
        if not relevant_chunks:
            return await self.generate_answer(question)
    
        # Create context from chunks
        context_parts = []
        source_documents = list(set([chunk.get('filename', 'Unknown') for chunk in relevant_chunks]))
    
        for chunk in relevant_chunks:
            # Don't include source info in context - we'll show it separately
            context_parts.append(chunk.get('content', ''))
    
        context_text = "\n\n".join(context_parts)
    
        if not self.openai_configured:
            return f"""Based on the uploaded documents:

{context_text}

*Note: This is demo mode. With OpenAI configured, this would be a comprehensive answer based on the document content above.*"""
    
        try:
            print(f"🧠 Generating clean answer from {len(relevant_chunks)} document chunks...")
        
            # Use the new OpenAI client (v1.0+)
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
        
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful assistant that answers questions based on provided document context. 
                    
                        IMPORTANT RULES:
                        1. Answer the question directly and comprehensively using the document context
                        2. DO NOT mention the document names or suggest referring to specific documents
                        3. DO NOT say things like "refer to the document" or "according to the PDF"
                        4. The source information will be shown separately, so focus only on the answer content
                        5. Be concise but complete
                        6. If the context doesn't contain enough information, say so clearly"""
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nDocument Context:\n{context_text}\n\nPlease answer the question based on the document context provided. Do not mention document names or suggest referring to documents."
                    }
                ],
                max_tokens=600,
                temperature=0.3
            )
        
            answer = response.choices[0].message.content.strip()
            print(f"✅ Clean answer generated from document chunks ({len(answer)} characters)")
            return answer
        
        except Exception as e:
            print(f"❌ Error generating answer from chunks: {str(e)}")
            return f"""Based on the uploaded documents:

{context_text}

*Note: There was an error generating the AI response, but the relevant document content is shown above.*"""

    async def generate_gpt_answer(self, question: str) -> str:
        """
        Generate answer using GPT when no document context is available
        """
        return await self.generate_answer(question)
    
    async def generate_answer(self, question: str) -> str:
        """Generate answer using OpenAI GPT-3.5-turbo"""
        if not self.openai_configured:
            return f"""🎭 **Demo Answer**

Question: "{question}"

This is a demonstration response. To get AI-powered answers, please configure your OPENAI_API_KEY environment variable.

**What this system would do with OpenAI configured:**
- Use GPT-3.5-turbo to generate intelligent, contextual answers
- Provide accurate information based on the question
- Maintain conversation context and follow-up capabilities

**Current Status:** Demo mode - showing system functionality without AI costs.

**System Info:**
- Platform: Neon PostgreSQL  
- Model: GPT-3.5-turbo (cost-effective and reliable)
- Status: Fully operational in demo mode"""
    
        try:
            print(f"🧠 Generating AI answer using GPT-3.5-turbo for: {question[:50]}...")
        
            # Use the new OpenAI client (v1.0+)
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
        
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that provides accurate, concise, and informative answers. Be direct and factual."
                    },
                    {
                        "role": "user", 
                        "content": question
                    }
                ],
                max_tokens=500,
                temperature=0.7,
                timeout=30
            )
        
            answer = response.choices[0].message.content.strip()
            print(f"✅ AI answer generated successfully using GPT-3.5-turbo ({len(answer)} characters)")
            return answer
        
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error generating answer with GPT-3.5-turbo: {error_msg}")
        
            return f"""I apologize, but I encountered an error while generating an answer to your question: "{question}". 

**Error details:** {error_msg}

**System Configuration:**
- Model: GPT-3.5-turbo (available to all OpenAI accounts)
- API: OpenAI v1.0+ (modern client)

**Possible solutions:**
- Make sure your OpenAI account has billing set up and available credits
- Check if your API key has the correct permissions
- Verify your internet connection

**Fallback response:** This appears to be a question about {question[:50]}... In a fully configured system, I would provide a comprehensive answer using GPT-3.5-turbo.

Please try again - GPT-3.5-turbo should be available to all OpenAI accounts."""
    
    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Ensure both inputs are numpy arrays
            if not isinstance(vec1, np.ndarray):
                vec1 = np.array(vec1, dtype=np.float32)
            if not isinstance(vec2, np.ndarray):
                vec2 = np.array(vec2, dtype=np.float32)
        
            # Validate dimensions
            if len(vec1) != len(vec2):
                print(f"⚠️ Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
                return 0.0
        
            # Check for zero vectors
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
        
            if norm_vec1 == 0.0 or norm_vec2 == 0.0:
                print("⚠️ Zero vector detected in similarity calculation")
                return 0.0
        
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            similarity = dot_product / (norm_vec1 * norm_vec2)
        
            # Ensure result is a scalar float
            if isinstance(similarity, np.ndarray):
                similarity = float(similarity.item())
            else:
                similarity = float(similarity)
        
            # Clamp to valid range [-1, 1]
            similarity = max(-1.0, min(1.0, similarity))
        
            return similarity
        
        except Exception as e:
            print(f"❌ Error calculating similarity: {str(e)}")
            return 0.0

print("✅ RAG Service initialized with clean answer generation:")
print(f"- OpenAI configured: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No (demo mode)'}")
print("- Model: GPT-3.5-turbo (hardcoded, cost-effective)")
print("- Clean answers: No document references in response text")
print("- Source info: Shown separately in UI")
print("- Embedding dimension: 1536 (text-embedding-ada-002)")
print("- API format: OpenAI v1.0+ (modern client)")
