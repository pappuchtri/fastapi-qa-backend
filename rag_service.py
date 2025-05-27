import openai
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
                # Initialize OpenAI client with proper configuration
                self.openai_client = openai.OpenAI(
                    api_key=self.openai_api_key,
                    timeout=30.0,  # 30 second timeout
                    max_retries=3   # Retry failed requests 3 times
                )
                print("✅ OpenAI client initialized successfully")
            except Exception as e:
                print(f"❌ Error initializing OpenAI client: {str(e)}")
                self.openai_client = None
        else:
            self.openai_client = None
            print("⚠️ OPENAI_API_KEY not found. RAG functionality will run in demo mode.")
        
        self.embedding_dimension = 1536  # text-embedding-ada-002 dimension
        self.similarity_threshold = 0.8
        
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for given text using OpenAI text-embedding-ada-002"""
        if not self.openai_client:
            # Return a dummy embedding for demo purposes
            print(f"🎭 Generating demo embedding for: {text[:50]}...")
            # Create a consistent dummy embedding based on text hash for demo
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))  # Consistent seed based on text
            embedding = np.random.rand(self.embedding_dimension)
            return embedding
        
        try:
            print(f"🤖 Generating OpenAI embedding for: {text[:50]}...")
            
            # Use the correct method for the current OpenAI library
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
                encoding_format="float"
            )
            
            embedding = np.array(response.data[0].embedding)
            print(f"✅ Embedding generated successfully (dimension: {len(embedding)})")
            return embedding
            
        except Exception as e:
            print(f"❌ Error generating embedding: {str(e)}")
            print("🎭 Falling back to demo embedding")
            # Return a consistent dummy embedding as fallback
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))
            embedding = np.random.rand(self.embedding_dimension)
            return embedding
    
    async def find_similar_question(
        self, 
        db: Session, 
        query_embedding: np.ndarray
    ) -> Tuple[Optional[Question], float]:
        """
        Find the most similar question in the database using cosine similarity
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
            
            # Convert stored embeddings to numpy array
            stored_embeddings = []
            questions = []
            
            for emb in embeddings:
                try:
                    # Convert stored vector to numpy array
                    if isinstance(emb.vector, list):
                        vector = np.array([float(x) for x in emb.vector])
                    else:
                        vector = np.array(emb.vector)
                    
                    # Ensure vector has correct dimension
                    if len(vector) != self.embedding_dimension:
                        print(f"⚠️ Skipping embedding {emb.id}: wrong dimension {len(vector)}")
                        continue
                    
                    stored_embeddings.append(vector)
                    question = db.query(Question).filter(Question.id == emb.question_id).first()
                    questions.append(question)
                except Exception as e:
                    print(f"⚠️ Error processing embedding {emb.id}: {str(e)}")
                    continue
            
            if not stored_embeddings:
                print("📭 No valid embeddings found")
                return None, 0.0
            
            # Calculate cosine similarities
            stored_embeddings_matrix = np.vstack(stored_embeddings)
            query_embedding_reshaped = query_embedding.reshape(1, -1)
            
            similarities = cosine_similarity(query_embedding_reshaped, stored_embeddings_matrix)[0]
            
            # Find the most similar question
            max_similarity_idx = np.argmax(similarities)
            max_similarity = similarities[max_similarity_idx]
            most_similar_question = questions[max_similarity_idx]
            
            print(f"🎯 Best match: {max_similarity:.3f} similarity")
            if most_similar_question:
                print(f"📝 Similar question: {most_similar_question.text[:50]}...")
            
            return most_similar_question, float(max_similarity)
            
        except Exception as e:
            print(f"❌ Error in similarity search: {str(e)}")
            return None, 0.0
    
    async def generate_answer(self, question: str) -> str:
        """Generate answer using OpenAI GPT-4"""
        if not self.openai_client:
            return f"""🎭 **Demo Answer**

Question: "{question}"

This is a demonstration response. To get AI-powered answers, please configure your OPENAI_API_KEY environment variable.

**What this system would do with OpenAI configured:**
- Use GPT-4 to generate intelligent, contextual answers
- Provide accurate information based on the question
- Maintain conversation context and follow-up capabilities

**Current Status:** Demo mode - showing system functionality without AI costs.

**System Info:**
- Platform: Render.com
- Database: Neon PostgreSQL  
- Status: Fully operational in demo mode"""
        
        try:
            print(f"🧠 Generating AI answer for: {question[:50]}...")
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that provides accurate, concise, and informative answers. Always be factual and cite sources when possible."
                    },
                    {
                        "role": "user", 
                        "content": question
                    }
                ],
                max_tokens=500,
                temperature=0.7,
                timeout=30.0
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"✅ AI answer generated successfully ({len(answer)} characters)")
            return answer
            
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
            return f"""I apologize, but I encountered an error while generating an answer to your question: "{question}". 

**Error details:** {str(e)}

**Fallback response:** This appears to be a question about {question[:50]}... In a fully configured system, I would provide a comprehensive answer using GPT-4.

Please try again later or contact support if the issue persists."""
    
    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
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

print("✅ RAG Service initialized:")
print(f"- OpenAI client configured: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No (demo mode)'}")
print("- Using scikit-learn for cosine similarity")
print("- Similarity threshold: 0.8")
print("- Embedding dimension: 1536 (text-embedding-ada-002)")
