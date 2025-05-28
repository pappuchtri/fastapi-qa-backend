import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
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
                print("✅ OpenAI API Key configured successfully")
            except Exception as e:
                print(f"❌ Error configuring OpenAI: {str(e)}")
                self.openai_configured = False
        else:
            print("⚠️ OPENAI_API_KEY not found. Running in demo mode.")
            self.openai_configured = False
        
        self.embedding_dimension = 1536  # text-embedding-ada-002 dimension
        self.document_similarity_threshold = 0.65  # Lowered to be more inclusive for document content
        self.qa_similarity_threshold = 0.85  # Higher threshold for historical Q&A to ensure relevance
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
            print("🔍 Searching for relevant document chunks (PRIORITY 1)...")
            
            # First check if there are any documents at all
            doc_count = db.query(Document).filter(Document.processed == True).count()
            if doc_count == 0:
                print("📚 No processed documents found in the database")
                return []
            
            relevant_chunks = []
            
            # Strategy 1: Vector similarity search (most accurate)
            vector_chunks = await self._vector_similarity_search(db, query_embedding, limit)
            if vector_chunks:
                print(f"✅ Found {len(vector_chunks)} relevant chunks via vector search")
                relevant_chunks.extend(vector_chunks)
            
            # Strategy 2: Entity-based search (more precise than keywords)
            if question_analysis.get("specific_entities") and len(relevant_chunks) < limit:
                entity_chunks = await self._entity_search(db, question_analysis["specific_entities"], limit)
                if entity_chunks:
                    print(f"✅ Found {len(entity_chunks)} relevant chunks via entity search")
                    relevant_chunks.extend(entity_chunks)
            
            # Strategy 3: Keyword-based search (fallback)
            if len(relevant_chunks) < limit:
                keyword_chunks = await self._keyword_search(db, question_analysis.get("keywords", []), limit)
                if keyword_chunks:
                    print(f"✅ Found {len(keyword_chunks)} relevant chunks via keyword search")
                    relevant_chunks.extend(keyword_chunks)
            
            # Strategy 4: Broad search (last resort)
            if len(relevant_chunks) == 0:
                print("⚠️ No relevant chunks found, trying broad search...")
                broad_chunks = await self._broad_search(db, question_analysis.get("keywords", []), limit)
                if broad_chunks:
                    print(f"✅ Found {len(broad_chunks)} chunks via broad search")
                    relevant_chunks.extend(broad_chunks)
            
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
    
    async def _broad_search(
        self, 
        db: Session, 
        keywords: List[str], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Broad search as a last resort when other methods fail"""
        try:
            # Get a sample of chunks from each document
            query = """
                WITH ranked_chunks AS (
                    SELECT 
                        dc.id, 
                        dc.document_id, 
                        dc.content, 
                        dc.page_number,
                        d.original_filename,
                        ROW_NUMBER() OVER (PARTITION BY dc.document_id ORDER BY dc.id) as rn
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.processed = true
                )
                SELECT id, document_id, content, page_number, original_filename
                FROM ranked_chunks
                WHERE rn <= 2  -- Get first 2 chunks from each document
                LIMIT :limit
            """
            
            result = db.execute(text(query), {"limit": limit})
            
            chunks = []
            for row in result:
                chunks.append({
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "content": row[2],
                    "page_number": row[3],
                    "filename": row[4],
                    "similarity": 0.5,  # Low confidence for broad search
                    "search_method": "broad"
                })
            
            return chunks
            
        except Exception as e:
            print(f"⚠️ Error in broad search: {str(e)}")
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
        method_priority = {"vector": 4, "entity": 3, "keyword": 2, "broad": 1}
        
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
            print("🔍 Searching historical Q&A pairs (PRIORITY 2)...")
            
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
    
    async def process_question(
        self,
        db: Session,
        question: str,
        query_embedding: np.ndarray,
        question_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a question following the strict priority order:
        1. Check uploaded PDFs
        2. Check historical Q&A
        3. Fallback to GPT
        """
        print("🔄 Processing question with strict priority order...")
        
        # STEP 1: Check if there are any documents in the database
        doc_count = db.query(Document).filter(Document.processed == True).count()
        has_documents = doc_count > 0
        
        if has_documents:
            print(f"📚 Found {doc_count} processed documents in database")
            
            # Search for relevant document chunks
            document_chunks = await self.search_relevant_documents(
                db, query_embedding, question_analysis, limit=5
            )
            
            # If we found relevant document chunks, use them to generate an answer
            if document_chunks:
                print("✅ PRIORITY 1: Using document-based answer generation")
                document_answer = await self._generate_document_based_answer(
                    question, question_analysis, document_chunks, []
                )
                document_answer["source_type"] = "document"
                return document_answer
            else:
                print("⚠️ No relevant document chunks found despite having documents")
        else:
            print("⚠️ No processed documents found in database")
        
        # STEP 2: Check historical Q&A
        historical_qa = await self.search_historical_qa(
            db, query_embedding, question_analysis, limit=3
        )
        
        if historical_qa:
            best_match = historical_qa[0]
            if best_match["similarity"] >= self.qa_similarity_threshold:
                print(f"✅ PRIORITY 2: Using historical Q&A (similarity: {best_match['similarity']:.3f})")
                return {
                    "answer": best_match["answer_text"],
                    "confidence": best_match.get("confidence", 0.9),
                    "primary_source": "historical_qa",
                    "source_documents": [],
                    "source_type": "historical"
                }
        
        # STEP 3: Fallback to GPT
        print("✅ PRIORITY 3: Falling back to GPT-3.5 Turbo")
        gpt_answer = await self._generate_gpt_fallback_answer(
            question, question_analysis, [], historical_qa
        )
        gpt_answer["source_type"] = "gpt"
        return gpt_answer
    
    async def generate_contextual_answer(
        self, 
        question: str,
        question_analysis: Dict[str, Any],
        document_context: List[Dict[str, Any]],
        historical_qa: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive answer using all available context with intelligent fallback
        Returns dict with answer, confidence, and source information
        """
        try:
            if not self.openai_configured:
                return self._generate_demo_answer_with_fallback(question, question_analysis, document_context, historical_qa)
            
            # Determine the appropriate response strategy based on available context
            if document_context:
                return await self._generate_document_based_answer(question, question_analysis, document_context, historical_qa)
            elif historical_qa:
                return await self._generate_historical_based_answer(question, question_analysis, document_context, historical_qa)
            else:
                return await self._generate_gpt_fallback_answer(question, question_analysis, document_context, historical_qa)
            
        except Exception as e:
            print(f"❌ Error generating contextual answer: {str(e)}")
            return await self._generate_gpt_fallback_answer(question, question_analysis, document_context, historical_qa)

    async def _generate_document_based_answer(
        self, 
        question: str,
        question_analysis: Dict[str, Any],
        document_context: List[Dict[str, Any]],
        historical_qa: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate answer primarily based on document context"""
        doc_context_text = self._format_document_context(document_context)
    
        system_prompt = """You are an AI assistant specializing in document analysis. Your primary task is to answer questions based on the provided document context. Be precise, cite your sources, and stay focused on the document content."""

        user_prompt = f"""QUESTION: {question}

DOCUMENT CONTEXT:
{doc_context_text}

Please provide a comprehensive answer based primarily on the document context. Include specific references to the source documents and page numbers. If the documents don't fully address the question, clearly state what information is missing."""

        import openai
        response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=600,
            temperature=0.3,  # Lower temperature for more focused responses
            timeout=30
        )
    
        answer = response['choices'][0]['message']['content'].strip()
    
        return {
            'answer': answer,
            'confidence': 0.9,
            'primary_source': 'documents',
            'source_documents': [doc['filename'] for doc in document_context]
        }

    async def _generate_historical_based_answer(
        self, 
        question: str,
        question_analysis: Dict[str, Any],
        document_context: List[Dict[str, Any]],
        historical_qa: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate answer primarily based on historical Q&A"""
        qa_context_text = self._format_qa_context(historical_qa)
    
        system_prompt = """You are an AI assistant that builds upon previous Q&A interactions. Use the historical context to provide consistent and comprehensive answers while adapting to the current question."""

        user_prompt = f"""QUESTION: {question}

HISTORICAL Q&A CONTEXT:
{qa_context_text}

Based on the similar previous questions and answers, provide a comprehensive response to the current question. Maintain consistency with previous answers while addressing any new aspects of the current question."""

        import openai
        response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=600,
            temperature=0.4,
            timeout=30
        )
    
        answer = response['choices'][0]['message']['content'].strip()
    
        return {
            'answer': answer,
            'confidence': 0.85,
            'primary_source': 'historical_qa',
            'source_documents': []
        }

    async def _generate_hybrid_answer(
        self, 
        question: str,
        question_analysis: Dict[str, Any],
        document_context: List[Dict[str, Any]],
        historical_qa: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate answer using both document and historical context"""
        doc_context_text = self._format_document_context(document_context)
        qa_context_text = self._format_qa_context(historical_qa)
    
        system_prompt = """You are an intelligent AI assistant with access to both document content and historical Q&A data. Synthesize information from both sources to provide comprehensive, well-sourced answers."""

        user_prompt = f"""QUESTION: {question}

DOCUMENT CONTEXT:
{doc_context_text}

HISTORICAL Q&A CONTEXT:
{qa_context_text}

Please provide a comprehensive answer that synthesizes information from both the document context and historical Q&A. Clearly distinguish between information from documents versus previous Q&A sessions."""

        import openai
        response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=700,
            temperature=0.5,
            timeout=30
        )
    
        answer = response['choices'][0]['message']['content'].strip()
    
        return {
            'answer': answer,
            'confidence': 0.8,
            'primary_source': 'hybrid',
            'source_documents': [doc['filename'] for doc in document_context]
        }

    async def _generate_gpt_fallback_answer(
        self, 
        question: str,
        question_analysis: Dict[str, Any],
        document_context: List[Dict[str, Any]],
        historical_qa: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate answer using GPT-3.5 Turbo when document/historical context is insufficient
        """
        print("🤖 Falling back to GPT-3.5 Turbo for general knowledge response")
    
        # Prepare context summary for transparency
        context_summary = []
        if document_context:
            context_summary.append(f"Found {len(document_context)} potentially relevant document sections, but with low confidence.")
        if historical_qa:
            context_summary.append(f"Found {len(historical_qa)} similar previous questions, but not closely related.")
    
        context_note = " ".join(context_summary) if context_summary else "No relevant context found in uploaded documents or previous Q&A."
    
        system_prompt = f"""You are a knowledgeable AI assistant. The user has asked a question, but the available document context and historical Q&A data don't provide sufficient information to answer confidently.

Context Status: {context_note}

Provide a helpful, accurate answer based on your general knowledge. Be clear that this response is generated from general AI knowledge rather than specific uploaded documents. If the question would benefit from specific documentation, suggest what type of documents might be helpful."""

        user_prompt = f"""QUESTION: {question}

QUESTION ANALYSIS:
- Intent: {question_analysis.get('intent', 'general_inquiry')}
- Type: {question_analysis.get('question_type', 'factual')}
- Complexity: {question_analysis.get('complexity', 'medium')}

Please provide a comprehensive answer based on general knowledge. Since no specific document context is available, focus on providing accurate, helpful information while noting that more specific information might be available with relevant documentation."""

        import openai
        try:
            response = openai.ChatCompletion.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.7,  # Higher temperature for more creative general responses
                timeout=30
            )
        
            answer = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return await self._handle_gpt_error_fallback(question, str(e))
    
        # Add a note about the source
        answer_with_note = f"{answer}\n\n---\n*Note: This response is generated from general AI knowledge as no specific relevant information was found in your uploaded documents or previous Q&A history. For more specific information, consider uploading relevant documents.*"
    
        return {
            'answer': answer_with_note,
            'confidence': 0.7,
            'primary_source': 'gpt_fallback',
            'source_documents': []
        }

    def _generate_demo_answer_with_fallback(
        self, 
        question: str, 
        question_analysis: Dict[str, Any],
        document_context: List[Dict[str, Any]], 
        historical_qa: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a demo answer with fallback logic demonstration"""
        context_summary = []
    
        if document_context:
            context_summary.append(f"📄 Found {len(document_context)} document sections")
            avg_similarity = sum(doc.get('similarity', 0) for doc in document_context) / len(document_context)
            context_summary.append(f"   Average relevance: {avg_similarity:.2f}")
    
        if historical_qa:
            context_summary.append(f"📚 Found {len(historical_qa)} similar Q&A pairs")
            best_similarity = max(qa.get('similarity', 0) for qa in historical_qa)
            context_summary.append(f"   Best match: {best_similarity:.2f}")
    
        # Simulate fallback decision
        has_good_context = (
            (document_context and any(doc.get('similarity', 0) >= 0.75 for doc in document_context)) or
            (historical_qa and any(qa.get('similarity', 0) >= 0.85 for qa in historical_qa))
        )
    
        if has_good_context:
            approach = "Document/Historical Context"
            confidence = 0.9
        else:
            approach = "GPT-3.5 Turbo Fallback"
            confidence = 0.7
    
        demo_answer = f"""🎭 **Demo Response** (OpenAI not configured)

**Question:** {question}

**Context Analysis:**
{chr(10).join(context_summary) if context_summary else "No relevant context found"}

**Selected Approach:** {approach}
**Confidence:** {confidence}

**What the full system would do:**
{
        "Use the high-quality document/historical context to generate a well-sourced answer with specific citations." 
        if has_good_context 
        else "Fall back to GPT-3.5 Turbo to provide a general knowledge response, clearly noting the lack of specific context."
    }

**Enhanced RAG Features:**
✅ Intelligent context assessment
✅ Multi-strategy source evaluation  
✅ Automatic GPT fallback when needed
✅ Transparent source attribution
✅ Confidence scoring"""

        return {
            'answer': demo_answer,
            'confidence': confidence,
            'primary_source': 'demo',
            'source_documents': [doc['filename'] for doc in document_chunks] if document_context else []
        }

    async def _handle_gpt_error_fallback(self, question: str, error: str) -> Dict[str, Any]:
        """Handle errors in GPT generation with a graceful fallback"""
        fallback_answer = f"""I apologize, but I encountered an error while generating a response to your question: "{question}"

Error details: {error}

This appears to be a temporary issue. Please try asking your question again, or consider:
1. Rephrasing your question
2. Uploading relevant documents for more specific context
3. Checking if the question can be broken down into smaller parts

If the issue persists, please contact support."""

        return {
            'answer': fallback_answer,
            'confidence': 0.3,
            'primary_source': 'error_fallback',
            'source_documents': []
        }
    
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

print("✅ Enhanced RAG Service initialized with strict prioritization:")
print("- PRIORITY 1: Document search (PDF content)")
print("- PRIORITY 2: Historical Q&A (previous answers)")
print("- PRIORITY 3: GPT fallback (general knowledge)")
