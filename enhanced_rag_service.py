import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy.orm import Session
from sklearn.metrics.pairwise import cosine_similarity
import os
from models import Question, Answer, Embedding
from document_models import Document, DocumentChunk
import json
import re

class EnhancedRAGService:
    def __init__(self):
        """Initialize the enhanced RAG service with OpenAI client"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if self.openai_api_key:
            try:
                import openai
                openai.api_key = self.openai_api_key
                self.openai_configured = True
                print("✅ OpenAI API key configured successfully")
            except Exception as e:
                print(f"❌ Error configuring OpenAI: {str(e)}")
                self.openai_configured = False
        else:
            print("⚠️ OPENAI_API_KEY not found. Running in demo mode.")
            self.openai_configured = False
        
        self.embedding_dimension = 1536  # text-embedding-ada-002 dimension
        self.document_similarity_threshold = 0.75
        self.qa_similarity_threshold = 0.8
        self.chat_model = "gpt-3.5-turbo"
        
        print(f"🤖 Enhanced RAG Service initialized with {self.chat_model}")
        
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for given text using OpenAI text-embedding-ada-002"""
        if not self.openai_configured:
            # Return a consistent dummy embedding for demo purposes
            print(f"🎭 Generating demo embedding for: {text[:50]}...")
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))
            embedding = np.random.rand(self.embedding_dimension)
            return embedding
        
        try:
            print(f"🤖 Generating OpenAI embedding for: {text[:50]}...")
            
            import openai
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            embedding = np.array(response['data'][0]['embedding'])
            print(f"✅ Embedding generated successfully (dimension: {len(embedding)})")
            return embedding
            
        except Exception as e:
            print(f"❌ Error generating embedding: {str(e)}")
            print("🎭 Falling back to demo embedding")
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))
            embedding = np.random.rand(self.embedding_dimension)
            return embedding
    
    async def analyze_question_intent(self, question: str) -> Dict[str, Any]:
        """
        Analyze the user's question to understand intent and extract key concepts
        """
        try:
            if not self.openai_configured:
                # Simple keyword extraction for demo mode
                keywords = [word.lower() for word in re.findall(r'\b\w+\b', question) if len(word) > 3]
                return {
                    "intent": "general_inquiry",
                    "keywords": keywords[:10],
                    "question_type": "factual",
                    "requires_documents": len(keywords) > 2,
                    "complexity": "medium"
                }
            
            import openai
            analysis_prompt = f"""Analyze this user question and provide a JSON response with the following structure:
{{
    "intent": "one of: factual_inquiry, how_to, definition, comparison, analysis, troubleshooting, general_inquiry",
    "keywords": ["list", "of", "key", "concepts", "and", "terms"],
    "question_type": "one of: factual, procedural, conceptual, analytical",
    "requires_documents": true/false,
    "complexity": "one of: simple, medium, complex",
    "domain": "subject area if identifiable",
    "specific_entities": ["any", "specific", "names", "or", "entities"]
}}

Question: "{question}"

Provide only the JSON response, no additional text."""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            analysis_text = response['choices'][0]['message']['content'].strip()
            
            # Parse JSON response
            try:
                analysis = json.loads(analysis_text)
                print(f"🧠 Question analysis: {analysis['intent']} - {analysis['complexity']}")
                return analysis
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                keywords = [word.lower() for word in re.findall(r'\b\w+\b', question) if len(word) > 3]
                return {
                    "intent": "general_inquiry",
                    "keywords": keywords[:10],
                    "question_type": "factual",
                    "requires_documents": True,
                    "complexity": "medium"
                }
                
        except Exception as e:
            print(f"⚠️ Error in question analysis: {str(e)}")
            # Fallback analysis
            keywords = [word.lower() for word in re.findall(r'\b\w+\b', question) if len(word) > 3]
            return {
                "intent": "general_inquiry",
                "keywords": keywords[:10],
                "question_type": "factual",
                "requires_documents": True,
                "complexity": "medium"
            }
    
    async def search_relevant_documents(
        self, 
        db: Session, 
        query_embedding: np.ndarray,
        question_analysis: Dict[str, Any],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks using multiple strategies
        """
        try:
            print("🔍 Searching for relevant document chunks...")
            
            relevant_chunks = []
            
            # Strategy 1: Vector similarity search
            vector_chunks = await self._vector_similarity_search(db, query_embedding, limit)
            relevant_chunks.extend(vector_chunks)
            
            # Strategy 2: Keyword-based search
            keyword_chunks = await self._keyword_search(db, question_analysis.get("keywords", []), limit)
            relevant_chunks.extend(keyword_chunks)
            
            # Strategy 3: Entity-based search
            if question_analysis.get("specific_entities"):
                entity_chunks = await self._entity_search(db, question_analysis["specific_entities"], limit)
                relevant_chunks.extend(entity_chunks)
            
            # Remove duplicates and rank by relevance
            unique_chunks = self._deduplicate_and_rank_chunks(relevant_chunks, query_embedding)
            
            print(f"📄 Found {len(unique_chunks)} relevant document chunks")
            return unique_chunks[:limit]
            
        except Exception as e:
            print(f"❌ Error in document search: {str(e)}")
            return []
    
    async def _vector_similarity_search(
        self, 
        db: Session, 
        query_embedding: np.ndarray, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Vector-based similarity search"""
        try:
            chunks = db.query(DocumentChunk).join(Document).filter(
                DocumentChunk.chunk_embedding.isnot(None),
                Document.processed == True
            ).all()
            
            if not chunks:
                return []
            
            similarities = []
            for chunk in chunks:
                try:
                    if chunk.chunk_embedding:
                        chunk_vector = np.array(chunk.chunk_embedding)
                        if len(chunk_vector) == len(query_embedding):
                            similarity = np.dot(query_embedding, chunk_vector) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_vector)
                            )
                            
                            if similarity >= self.document_similarity_threshold:
                                similarities.append({
                                    "chunk_id": chunk.id,
                                    "document_id": chunk.document_id,
                                    "content": chunk.content,
                                    "page_number": chunk.page_number,
                                    "filename": chunk.document.original_filename,
                                    "similarity": float(similarity),
                                    "search_method": "vector"
                                })
                except Exception as e:
                    continue
            
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            print(f"⚠️ Error in vector search: {str(e)}")
            return []
    
    async def _keyword_search(
        self, 
        db: Session, 
        keywords: List[str], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Keyword-based search with TF-IDF-like scoring"""
        try:
            if not keywords:
                return []
            
            # Build dynamic query for keyword search
            keyword_conditions = []
            params = {}
            
            for i, keyword in enumerate(keywords[:10]):  # Limit to 10 keywords
                if len(keyword) > 2:  # Skip very short words
                    param_name = f"keyword_{i}"
                    keyword_conditions.append(f"dc.content ILIKE :{param_name}")
                    params[param_name] = f"%{keyword}%"
            
            if not keyword_conditions:
                return []
            
            query = f"""
                SELECT dc.id, dc.document_id, dc.content, dc.page_number, 
                       d.original_filename,
                       ({' + '.join([f"CASE WHEN dc.content ILIKE :{param} THEN 1 ELSE 0 END" for param in params.keys()])}) as keyword_score
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.processed = true AND ({' OR '.join(keyword_conditions)})
                ORDER BY keyword_score DESC, dc.id
                LIMIT :limit
            """
            
            params["limit"] = limit
            result = db.execute(text(query), params)
            
            chunks = []
            for row in result:
                chunks.append({
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "content": row[2],
                    "page_number": row[3],
                    "filename": row[4],
                    "similarity": float(row[5]) / len(keywords),  # Normalize score
                    "search_method": "keyword"
                })
            
            return chunks
            
        except Exception as e:
            print(f"⚠️ Error in keyword search: {str(e)}")
            return []
    
    async def _entity_search(
        self, 
        db: Session, 
        entities: List[str], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search for specific entities or proper nouns"""
        try:
            entity_conditions = []
            params = {}
            
            for i, entity in enumerate(entities[:5]):  # Limit to 5 entities
                param_name = f"entity_{i}"
                # Use word boundaries for more precise matching
                entity_conditions.append(f"dc.content ~* :{param_name}")
                params[param_name] = f"\\b{re.escape(entity)}\\b"
            
            if not entity_conditions:
                return []
            
            query = f"""
                SELECT dc.id, dc.document_id, dc.content, dc.page_number, 
                       d.original_filename
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.processed = true AND ({' OR '.join(entity_conditions)})
                LIMIT :limit
            """
            
            params["limit"] = limit
            result = db.execute(text(query), params)
            
            chunks = []
            for row in result:
                chunks.append({
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "content": row[2],
                    "page_number": row[3],
                    "filename": row[4],
                    "similarity": 0.9,  # High relevance for entity matches
                    "search_method": "entity"
                })
            
            return chunks
            
        except Exception as e:
            print(f"⚠️ Error in entity search: {str(e)}")
            return []
    
    def _deduplicate_and_rank_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        query_embedding: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Remove duplicates and rank chunks by relevance"""
        seen_chunks = set()
        unique_chunks = []
        
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_chunks.append(chunk)
        
        # Sort by similarity score, then by search method priority
        method_priority = {"vector": 3, "entity": 2, "keyword": 1}
        
        unique_chunks.sort(
            key=lambda x: (
                x["similarity"], 
                method_priority.get(x["search_method"], 0)
            ), 
            reverse=True
        )
        
        return unique_chunks
    
    async def search_historical_qa(
        self, 
        db: Session, 
        query_embedding: np.ndarray,
        question_analysis: Dict[str, Any],
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant historical Q&A pairs using semantic similarity
        """
        try:
            print("🔍 Searching historical Q&A pairs...")
            
            # Get all question embeddings
            embeddings = db.query(Embedding).all()
            
            if not embeddings:
                return []
            
            relevant_qa = []
            
            for emb in embeddings:
                try:
                    if isinstance(emb.vector, list):
                        vector = np.array([float(x) for x in emb.vector])
                    elif isinstance(emb.vector, str):
                        import json
                        vector_list = json.loads(emb.vector)
                        vector = np.array([float(x) for x in vector_list])
                    else:
                        vector = np.array(emb.vector)
                    
                    if len(vector) != self.embedding_dimension:
                        continue
                    
                    similarity = np.dot(query_embedding, vector) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(vector)
                    )
                    
                    if similarity >= self.qa_similarity_threshold:
                        question = db.query(Question).filter(Question.id == emb.question_id).first()
                        if question:
                            answers = db.query(Answer).filter(Answer.question_id == question.id).order_by(Answer.created_at.desc()).first()
                            if answers:
                                relevant_qa.append({
                                    "question_id": question.id,
                                    "question_text": question.text,
                                    "answer_text": answers.text,
                                    "similarity": float(similarity),
                                    "confidence": answers.confidence_score
                                })
                
                except Exception as e:
                    continue
            
            # Sort by similarity
            relevant_qa.sort(key=lambda x: x["similarity"], reverse=True)
            
            print(f"📚 Found {len(relevant_qa)} relevant historical Q&A pairs")
            return relevant_qa[:limit]
            
        except Exception as e:
            print(f"❌ Error searching historical Q&A: {str(e)}")
            return []
    
    async def generate_contextual_answer(
        self, 
        question: str,
        question_analysis: Dict[str, Any],
        document_context: List[Dict[str, Any]],
        historical_qa: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a comprehensive answer using all available context
        """
        try:
            if not self.openai_configured:
                return self._generate_demo_answer(question, question_analysis, document_context, historical_qa)
            
            # Prepare context sections
            doc_context_text = self._format_document_context(document_context)
            qa_context_text = self._format_qa_context(historical_qa)
            
            # Build comprehensive prompt
            system_prompt = """You are an intelligent AI assistant with access to uploaded documents and historical Q&A data. Your goal is to provide accurate, comprehensive, and well-sourced answers.

INSTRUCTIONS:
1. Analyze the user's question carefully
2. Use document context as your primary source when available
3. Reference historical Q&A for additional insights
4. Clearly cite your sources (document names, page numbers)
5. If information is insufficient, acknowledge limitations
6. Provide structured, easy-to-read responses
7. Be concise but thorough"""

            user_prompt = f"""QUESTION: {question}

QUESTION ANALYSIS:
- Intent: {question_analysis.get('intent', 'general_inquiry')}
- Type: {question_analysis.get('question_type', 'factual')}
- Complexity: {question_analysis.get('complexity', 'medium')}
- Key concepts: {', '.join(question_analysis.get('keywords', [])[:5])}

DOCUMENT CONTEXT:
{doc_context_text}

HISTORICAL Q&A CONTEXT:
{qa_context_text}

Please provide a comprehensive answer based on the available context. If the context doesn't fully address the question, supplement with your general knowledge while clearly distinguishing between sourced and general information."""

            import openai
            response = openai.ChatCompletion.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.7,
                timeout=30
            )
            
            answer = response['choices'][0]['message']['content'].strip()
            print(f"✅ Contextual answer generated ({len(answer)} characters)")
            return answer
            
        except Exception as e:
            print(f"❌ Error generating contextual answer: {str(e)}")
            return self._generate_fallback_answer(question, document_context, historical_qa)
    
    def _format_document_context(self, document_context: List[Dict[str, Any]]) -> str:
        """Format document context for the prompt"""
        if not document_context:
            return "No relevant documents found."
        
        formatted_context = []
        for i, doc in enumerate(document_context, 1):
            content_preview = doc['content'][:400] + "..." if len(doc['content']) > 400 else doc['content']
            formatted_context.append(
                f"[Source {i}: {doc['filename']} - Page {doc.get('page_number', 'N/A')} - Relevance: {doc['similarity']:.2f}]\n{content_preview}"
            )
        
        return "\n\n".join(formatted_context)
    
    def _format_qa_context(self, historical_qa: List[Dict[str, Any]]) -> str:
        """Format historical Q&A context for the prompt"""
        if not historical_qa:
            return "No relevant historical Q&A found."
        
        formatted_qa = []
        for i, qa in enumerate(historical_qa, 1):
            formatted_qa.append(
                f"[Previous Q&A {i} - Similarity: {qa['similarity']:.2f}]\nQ: {qa['question_text']}\nA: {qa['answer_text'][:300]}..."
            )
        
        return "\n\n".join(formatted_qa)
    
    def _generate_demo_answer(
        self, 
        question: str, 
        question_analysis: Dict[str, Any],
        document_context: List[Dict[str, Any]], 
        historical_qa: List[Dict[str, Any]]
    ) -> str:
        """Generate a demo answer when OpenAI is not configured"""
        context_summary = []
        
        if document_context:
            context_summary.append(f"📄 Found {len(document_context)} relevant document sections")
            for doc in document_context[:2]:
                context_summary.append(f"  • {doc['filename']} (Page {doc.get('page_number', 'N/A')})")
        
        if historical_qa:
            context_summary.append(f"📚 Found {len(historical_qa)} similar previous questions")
        
        return f"""🎭 **Demo Response** (OpenAI not configured)

**Question:** {question}

**Analysis:**
- Intent: {question_analysis.get('intent', 'general_inquiry')}
- Complexity: {question_analysis.get('complexity', 'medium')}
- Key concepts: {', '.join(question_analysis.get('keywords', [])[:5])}

**Available Context:**
{chr(10).join(context_summary) if context_summary else "No specific context found"}

**What the full system would provide:**
With OpenAI configured, this system would analyze your question contextually and provide a comprehensive answer by:
1. Understanding the intent and complexity of your question
2. Searching through uploaded documents using semantic similarity
3. Finding relevant historical Q&A pairs
4. Generating a well-sourced, contextual response

**Current Status:** Demo mode - showing enhanced RAG capabilities without AI costs."""
    
    def _generate_fallback_answer(
        self, 
        question: str, 
        document_context: List[Dict[str, Any]], 
        historical_qa: List[Dict[str, Any]]
    ) -> str:
        """Generate a fallback answer when AI generation fails"""
        context_info = []
        
        if document_context:
            context_info.append(f"Found {len(document_context)} relevant document sections:")
            for doc in document_context[:3]:
                context_info.append(f"• {doc['filename']} (Page {doc.get('page_number', 'N/A')}): {doc['content'][:200]}...")
        
        if historical_qa:
            context_info.append(f"\nSimilar previous questions:")
            for qa in historical_qa[:2]:
                context_info.append(f"• Q: {qa['question_text'][:100]}...")
                context_info.append(f"  A: {qa['answer_text'][:200]}...")
        
        return f"""I found relevant context for your question: "{question}"

{chr(10).join(context_info)}

However, I encountered an error generating a comprehensive response. The system has identified relevant information from your uploaded documents and previous Q&A history that would help answer your question."""

print("✅ Enhanced RAG Service initialized with contextual understanding:")
print("- Semantic question analysis")
print("- Multi-strategy document search (vector + keyword + entity)")
print("- Historical Q&A semantic matching")
print("- Contextual answer generation")
print("- Intelligent source attribution")
