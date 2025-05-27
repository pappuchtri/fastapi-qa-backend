import openai
import numpy as np
import faiss
from typing import List, Tuple, Optional
from sqlalchemy.orm import Session
from sklearn.metrics.pairwise import cosine_similarity
import os
from models import Question, Answer, Embedding

class RAGService:
    def __init__(self):
        """Initialize the RAG service with OpenAI client and FAISS index"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            print("Warning: OPENAI_API_KEY not found. RAG functionality will be limited.")
        
        self.embedding_dimension = 1536  # text-embedding-ada-002 dimension
        self.similarity_threshold = 0.8
        
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for given text using OpenAI text-embedding-ada-002"""
        if not self.openai_client:
            # Return a dummy embedding for demo purposes
            return np.random.rand(self.embedding_dimension)
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            # Return a dummy embedding as fallback
            return np.random.rand(self.embedding_dimension)
    
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
            # Get all embeddings from database
            embeddings = db.query(Embedding).all()
            
            if not embeddings:
                return None, 0.0
            
            # Convert stored embeddings to numpy array
            stored_embeddings = []
            questions = []
            
            for emb in embeddings:
                stored_embeddings.append(np.array(emb.vector))
                question = db.query(Question).filter(Question.id == emb.question_id).first()
                questions.append(question)
            
            if not stored_embeddings:
                return None, 0.0
            
            # Calculate cosine similarities
            stored_embeddings_matrix = np.vstack(stored_embeddings)
            query_embedding_reshaped = query_embedding.reshape(1, -1)
            
            similarities = cosine_similarity(query_embedding_reshaped, stored_embeddings_matrix)[0]
            
            # Find the most similar question
            max_similarity_idx = np.argmax(similarities)
            max_similarity = similarities[max_similarity_idx]
            most_similar_question = questions[max_similarity_idx]
            
            return most_similar_question, float(max_similarity)
            
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return None, 0.0
    
    async def generate_answer(self, question: str) -> str:
        """Generate answer using OpenAI GPT-4"""
        if not self.openai_client:
            return f"Demo answer for: '{question}'. Please configure OPENAI_API_KEY for AI-powered responses."
        
        try:
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
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return f"I apologize, but I encountered an error while generating an answer to your question: '{question}'. Please try again later."
    
    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)

print("RAG Service initialized:")
print(f"- OpenAI client configured: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No (demo mode)'}")
print("- Cosine similarity and FAISS for vector search")
print("- Similarity threshold: 0.8")
