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
        # Much lower thresholds to be more inclusive of PDF content
        self.document_similarity_threshold = 0.3  # Very low to catch any potentially relevant content
        self.qa_similarity_threshold = 0.85  # High threshold for historical Q&A
        self.chat_model = "gpt-3.5-turbo"
        
        print(f"🤖 Enhanced RAG Service initialized with {self.chat_model}")
        print(f"📄 Document similarity threshold: {self.document_similarity_threshold} (very inclusive)")
        
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
                keywords = [word.lower() for word in re.findall(r'\b\w+\b', question) if len(word) > 2]
                return {
                    "intent": "general_inquiry",
                    "keywords": keywords[:15],  # More keywords for better search
                    "question_type": "factual",
                    "requires_documents": True,
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
                keywords = [word.lower() for word in re.findall(r'\b\w+\b', question) if len(word) > 2]
                return {
                    "intent": "general_inquiry",
                    "keywords": keywords[:15],
                    "question_type": "factual",
                    "requires_documents": True,
                    "complexity": "medium"
                }
                
        except Exception as e:
            print(f"⚠️ Error in question analysis: {str(e)}")
            # Fallback analysis
            keywords = [word.lower() for word in re.findall(r'\b\w+\b', question) if len(word) > 2]
            return {
                "intent": "general_inquiry",
                "keywords": keywords[:15],
                "question_type": "factual",
                "requires_documents": True,
                "complexity": "medium"
            }
    
    async def exhaustive_pdf_search(
        self, 
        db: Session, 
        question: str,
        query_embedding: np.ndarray,
        question_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Perform exhaustive search through ALL PDF documents using multiple strategies
        """
        print("🔍 STEP 1: EXHAUSTIVE PDF DOCUMENT SEARCH")
        print("=" * 60)
        
        # Check if there are any documents at all
        doc_count = db.query(Document).filter(Document.processed == True).count()
        if doc_count == 0:
            print("❌ No processed documents found in the database")
            return []
        
        print(f"📚 Found {doc_count} processed documents in database")
        
        # Get all document information
        documents = db.query(Document).filter(Document.processed == True).all()
        for doc in documents:
            print(f"   📄 {doc.original_filename} ({doc.total_chunks} chunks)")
        
        all_relevant_chunks = []
        
        # Strategy 1: Vector similarity search (with very low threshold)
        print("\n🔍 Strategy 1: Vector Similarity Search")
        vector_chunks = await self._comprehensive_vector_search(db, query_embedding)
        if vector_chunks:
            print(f"✅ Found {len(vector_chunks)} chunks via vector search")
            all_relevant_chunks.extend(vector_chunks)
        else:
            print("⚠️ No chunks found via vector search")
        
        # Strategy 2: Comprehensive keyword search
        print("\n🔍 Strategy 2: Comprehensive Keyword Search")
        keyword_chunks = await self._comprehensive_keyword_search(db, question_analysis.get("keywords", []))
        if keyword_chunks:
            print(f"✅ Found {len(keyword_chunks)} chunks via keyword search")
            all_relevant_chunks.extend(keyword_chunks)
        else:
            print("⚠️ No chunks found via keyword search")
        
        # Strategy 3: Entity and phrase search
        print("\n🔍 Strategy 3: Entity and Phrase Search")
        entity_chunks = await self._comprehensive_entity_search(db, question, question_analysis)
        if entity_chunks:
            print(f"✅ Found {len(entity_chunks)} chunks via entity search")
            all_relevant_chunks.extend(entity_chunks)
        else:
            print("⚠️ No chunks found via entity search")
        
        # Strategy 4: Fuzzy text search
        print("\n🔍 Strategy 4: Fuzzy Text Search")
        fuzzy_chunks = await self._fuzzy_text_search(db, question)
        if fuzzy_chunks:
            print(f"✅ Found {len(fuzzy_chunks)} chunks via fuzzy search")
            all_relevant_chunks.extend(fuzzy_chunks)
        else:
            print("⚠️ No chunks found via fuzzy search")
        
        # Strategy 5: Broad content sampling (last resort)
        if not all_relevant_chunks:
            print("\n🔍 Strategy 5: Broad Content Sampling (Last Resort)")
            broad_chunks = await self._sample_all_documents(db)
            if broad_chunks:
                print(f"✅ Sampled {len(broad_chunks)} chunks from all documents")
                all_relevant_chunks.extend(broad_chunks)
        
        # Remove duplicates and rank by relevance
        unique_chunks = self._deduplicate_and_rank_chunks(all_relevant_chunks, query_embedding)
        
        print(f"\n📊 TOTAL RELEVANT CHUNKS FOUND: {len(unique_chunks)}")
        if unique_chunks:
            print("📄 Top matches:")
            for i, chunk in enumerate(unique_chunks[:3], 1):
                print(f"   {i}. {chunk['filename']} (Page {chunk.get('page_number', 'N/A')}) - Similarity: {chunk['similarity']:.3f}")
        
        return unique_chunks
    
    async def _comprehensive_vector_search(
        self, 
        db: Session, 
        query_embedding: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Comprehensive vector-based similarity search with very low threshold"""
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
                            
                            # Very low threshold to catch any potentially relevant content
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
            return similarities
            
        except Exception as e:
            print(f"⚠️ Error in vector search: {str(e)}")
            return []
    
    async def _comprehensive_keyword_search(
        self, 
        db: Session, 
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """Comprehensive keyword search with multiple variations"""
        try:
            if not keywords:
                return []
            
            all_chunks = []
            
            # Search for individual keywords
            for keyword in keywords:
                if len(keyword) > 2:  # Skip very short words
                    query = """
                        SELECT dc.id, dc.document_id, dc.content, dc.page_number, 
                               d.original_filename
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE d.processed = true AND dc.content ILIKE :keyword
                        ORDER BY dc.id
                    """
                    
                    result = db.execute(text(query), {"keyword": f"%{keyword}%"})
                    
                    for row in result:
                        all_chunks.append({
                            "chunk_id": row[0],
                            "document_id": row[1],
                            "content": row[2],
                            "page_number": row[3],
                            "filename": row[4],
                            "similarity": 0.7,  # Fixed similarity for keyword matches
                            "search_method": "keyword",
                            "matched_keyword": keyword
                        })
            
            # Search for keyword combinations
            if len(keywords) > 1:
                for i in range(len(keywords) - 1):
                    for j in range(i + 1, len(keywords)):
                        keyword1, keyword2 = keywords[i], keywords[j]
                        if len(keyword1) > 2 and len(keyword2) > 2:
                            query = """
                                SELECT dc.id, dc.document_id, dc.content, dc.page_number, 
                                       d.original_filename
                                FROM document_chunks dc
                                JOIN documents d ON dc.document_id = d.id
                                WHERE d.processed = true 
                                AND dc.content ILIKE :keyword1 
                                AND dc.content ILIKE :keyword2
                                ORDER BY dc.id
                            """
                            
                            result = db.execute(text(query), {
                                "keyword1": f"%{keyword1}%",
                                "keyword2": f"%{keyword2}%"
                            })
                            
                            for row in result:
                                all_chunks.append({
                                    "chunk_id": row[0],
                                    "document_id": row[1],
                                    "content": row[2],
                                    "page_number": row[3],
                                    "filename": row[4],
                                    "similarity": 0.8,  # Higher similarity for multiple keyword matches
                                    "search_method": "keyword_combo",
                                    "matched_keywords": [keyword1, keyword2]
                                })
            
            return all_chunks
            
        except Exception as e:
            print(f"⚠️ Error in keyword search: {str(e)}")
            return []
    
    async def _comprehensive_entity_search(
        self, 
        db: Session, 
        question: str,
        question_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search for entities, proper nouns, and important phrases"""
        try:
            all_chunks = []
            
            # Extract potential entities from the question
            entities = question_analysis.get("specific_entities", [])
            
            # Also extract capitalized words as potential entities
            capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', question)
            entities.extend(capitalized_words)
            
            # Extract quoted phrases
            quoted_phrases = re.findall(r'"([^"]*)"', question)
            entities.extend(quoted_phrases)
            
            # Remove duplicates
            entities = list(set(entities))
            
            for entity in entities:
                if len(entity) > 2:
                    # Search for exact entity matches
                    query = """
                        SELECT dc.id, dc.document_id, dc.content, dc.page_number, 
                               d.original_filename
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE d.processed = true AND dc.content ~* :entity
                        ORDER BY dc.id
                    """
                    
                    result = db.execute(text(query), {"entity": f"\\b{re.escape(entity)}\\b"})
                    
                    for row in result:
                        all_chunks.append({
                            "chunk_id": row[0],
                            "document_id": row[1],
                            "content": row[2],
                            "page_number": row[3],
                            "filename": row[4],
                            "similarity": 0.9,  # High relevance for entity matches
                            "search_method": "entity",
                            "matched_entity": entity
                        })
            
            return all_chunks
            
        except Exception as e:
            print(f"⚠️ Error in entity search: {str(e)}")
            return []
    
    async def _fuzzy_text_search(
        self, 
        db: Session, 
        question: str
    ) -> List[Dict[str, Any]]:
        """Fuzzy text search for partial matches"""
        try:
            # Extract meaningful phrases from the question
            words = question.lower().split()
            meaningful_words = [word for word in words if len(word) > 3 and word not in ['what', 'where', 'when', 'how', 'why', 'which', 'that', 'this', 'with', 'from', 'they', 'have', 'been', 'were', 'said', 'each', 'their']]
            
            all_chunks = []
            
            for word in meaningful_words:
                # Use PostgreSQL's similarity function if available, otherwise use ILIKE
                query = """
                    SELECT dc.id, dc.document_id, dc.content, dc.page_number, 
                           d.original_filename
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.processed = true AND dc.content ILIKE :word
                    ORDER BY dc.id
                """
                
                result = db.execute(text(query), {"word": f"%{word}%"})
                
                for row in result:
                    all_chunks.append({
                        "chunk_id": row[0],
                        "document_id": row[1],
                        "content": row[2],
                        "page_number": row[3],
                        "filename": row[4],
                        "similarity": 0.6,  # Medium relevance for fuzzy matches
                        "search_method": "fuzzy",
                        "matched_word": word
                    })
            
            return all_chunks
            
        except Exception as e:
            print(f"⚠️ Error in fuzzy search: {str(e)}")
            return []
    
    async def _sample_all_documents(
        self, 
        db: Session
    ) -> List[Dict[str, Any]]:
        """Sample content from all documents as a last resort"""
        try:
            # Get a representative sample from each document
            query = """
                WITH ranked_chunks AS (
                    SELECT 
                        dc.id, 
                        dc.document_id, 
                        dc.content, 
                        dc.page_number,
                        d.original_filename,
                        ROW_NUMBER() OVER (PARTITION BY dc.document_id ORDER BY dc.chunk_index) as rn
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.processed = true
                )
                SELECT id, document_id, content, page_number, original_filename
                FROM ranked_chunks
                WHERE rn <= 3  -- Get first 3 chunks from each document
                ORDER BY document_id, rn
            """
            
            result = db.execute(text(query))
            
            chunks = []
            for row in result:
                chunks.append({
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "content": row[2],
                    "page_number": row[3],
                    "filename": row[4],
                    "similarity": 0.4,  # Low confidence for broad sampling
                    "search_method": "sample"
                })
            
            return chunks
            
        except Exception as e:
            print(f"⚠️ Error in document sampling: {str(e)}")
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
        method_priority = {"vector": 5, "entity": 4, "keyword_combo": 3, "keyword": 2, "fuzzy": 1, "sample": 0}
        
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
            print("\n🔍 STEP 2: CHECKING HISTORICAL Q&A DATABASE")
            print("=" * 60)
            
            # Get all question embeddings
            embeddings = db.query(Embedding).all()
            
            if not embeddings:
                print("📚 No historical Q&A pairs found")
                return []
            
            print(f"📚 Found {len(embeddings)} historical Q&A pairs to check")
            
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
            
            if relevant_qa:
                print(f"✅ Found {len(relevant_qa)} relevant historical Q&A pairs")
                for i, qa in enumerate(relevant_qa[:3], 1):
                    print(f"   {i}. Similarity: {qa['similarity']:.3f} - Q: {qa['question_text'][:60]}...")
            else:
                print("⚠️ No relevant historical Q&A pairs found")
            
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
        Process a question with exhaustive PDF search first, then Q&A, then GPT
        """
        print("🔄 PROCESSING QUESTION WITH EXHAUSTIVE PDF PRIORITY")
        print("=" * 80)
        
        # STEP 1: EXHAUSTIVE PDF DOCUMENT SEARCH
        document_chunks = await self.exhaustive_pdf_search(
            db, question, query_embedding, question_analysis
        )
        
        # If we found ANY relevant document chunks, use them to generate an answer
        if document_chunks:
            print(f"\n✅ SUCCESS: Found {len(document_chunks)} relevant chunks in PDF documents")
            print("🎯 GENERATING ANSWER FROM PDF CONTENT")
            
            document_answer = await self._generate_document_based_answer(
                question, question_analysis, document_chunks, []
            )
            document_answer["source_type"] = "document"
            document_answer["total_chunks_found"] = len(document_chunks)
            return document_answer
        
        print("\n⚠️ NO RELEVANT CONTENT FOUND IN PDF DOCUMENTS")
        
        # STEP 2: Check historical Q&A only if no PDF content found
        historical_qa = await self.search_historical_qa(
            db, query_embedding, question_analysis, limit=3
        )
        
        if historical_qa:
            best_match = historical_qa[0]
            if best_match["similarity"] >= self.qa_similarity_threshold:
                print(f"\n✅ SUCCESS: Using historical Q&A (similarity: {best_match['similarity']:.3f})")
                return {
                    "answer": best_match["answer_text"],
                    "confidence": best_match.get("confidence", 0.9),
                    "primary_source": "historical_qa",
                    "source_documents": [],
                    "source_type": "historical",
                    "question_id": best_match["question_id"],
                    "similarity": best_match["similarity"]
                }
        
        print("\n⚠️ NO RELEVANT HISTORICAL Q&A FOUND")
        
        # STEP 3: Fallback to GPT only as last resort
        print("\n🤖 STEP 3: FALLING BACK TO GPT-3.5 TURBO")
        print("=" * 60)
        gpt_answer = await self._generate_gpt_fallback_answer(
            question, question_analysis, [], historical_qa
        )
        gpt_answer["source_type"] = "gpt"
        return gpt_answer
    
    async def _generate_document_based_answer(
        self, 
        question: str,
        question_analysis: Dict[str, Any],
        document_context: List[Dict[str, Any]],
        historical_qa: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate answer primarily based on document context"""
        doc_context_text = self._format_document_context(document_context)
    
        if not self.openai_configured:
            return self._generate_demo_document_answer(question, document_context)
    
        system_prompt = """You are an AI assistant specializing in document analysis. Your primary task is to answer questions based EXCLUSIVELY on the provided document context. 

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the information provided in the document context
2. Include specific references to source documents and page numbers
3. If the documents don't contain enough information to fully answer the question, clearly state what information is missing
4. Do not add information from your general knowledge that isn't in the documents
5. Be comprehensive and detailed in your response"""

        user_prompt = f"""QUESTION: {question}

DOCUMENT CONTEXT FROM UPLOADED PDFs:
{doc_context_text}

Please provide a comprehensive answer based EXCLUSIVELY on the document context above. Include specific references to the source documents and page numbers. If the documents don't fully address the question, clearly state what information is available and what is missing."""

        try:
            import openai
            response = openai.ChatCompletion.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.2,  # Low temperature for factual responses
                timeout=30
            )
        
            answer = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"❌ Error generating document-based answer: {str(e)}")
            return self._generate_demo_document_answer(question, document_context)
    
        return {
            'answer': answer,
            'confidence': 0.95,  # High confidence for document-based answers
            'primary_source': 'documents',
            'source_documents': list(set([doc['filename'] for doc in document_context]))
        }
    
    def _generate_demo_document_answer(
        self, 
        question: str, 
        document_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a demo answer showing document-based response"""
        
        source_info = []
        for doc in document_context[:5]:  # Show top 5 sources
            source_info.append(f"📄 {doc['filename']} (Page {doc.get('page_number', 'N/A')}) - Relevance: {doc['similarity']:.3f}")
        
        content_preview = []
        for i, doc in enumerate(document_context[:3], 1):
            preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
            content_preview.append(f"Source {i}: {preview}")
        
        demo_answer = f"""🎭 **Demo Response** (OpenAI not configured)

**Question:** {question}

**ANSWER GENERATED FROM PDF DOCUMENTS:**

Based on the uploaded PDF documents, I found the following relevant information:

{chr(10).join(content_preview)}

**Sources Found:**
{chr(10).join(source_info)}

**What the full system would provide:**
With OpenAI configured, this system would analyze the {len(document_context)} relevant document chunks found and generate a comprehensive, well-sourced answer based exclusively on the PDF content.

**PDF Search Results:**
✅ {len(document_context)} relevant chunks found across {len(set([doc['filename'] for doc in document_context]))} documents
✅ Multiple search strategies used (vector, keyword, entity, fuzzy)
✅ Content prioritized from uploaded PDFs"""

        return {
            'answer': demo_answer,
            'confidence': 0.95,
            'primary_source': 'documents',
            'source_documents': list(set([doc['filename'] for doc in document_context]))
        }

    async def _generate_gpt_fallback_answer(
        self, 
        question: str,
        question_analysis: Dict[str, Any],
        document_context: List[Dict[str, Any]],
        historical_qa: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate answer using GPT-3.5 Turbo when no relevant content is found
        """
        print("🤖 Generating GPT fallback response")
    
        if not self.openai_configured:
            return self._generate_demo_gpt_fallback(question)
    
        system_prompt = """You are a knowledgeable AI assistant. The user has asked a question, but no relevant information was found in their uploaded PDF documents or historical Q&A database.

Provide a helpful, accurate answer based on your general knowledge. Be clear that this response is generated from general AI knowledge rather than specific uploaded documents. If the question would benefit from specific documentation, suggest what type of documents might be helpful."""

        user_prompt = f"""QUESTION: {question}

CONTEXT: No relevant information was found in the user's uploaded PDF documents or previous Q&A history.

Please provide a comprehensive answer based on general knowledge. Since no specific document context is available, focus on providing accurate, helpful information while noting that more specific information might be available with relevant documentation."""

        try:
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
        except Exception as e:
            return self._generate_demo_gpt_fallback(question)
    
        # Add a note about the source
        answer_with_note = f"{answer}\n\n---\n*Note: This response is generated from general AI knowledge as no relevant information was found in your uploaded PDF documents or previous Q&A history. For more specific information, consider uploading relevant documents.*"
    
        return {
            'answer': answer_with_note,
            'confidence': 0.7,
            'primary_source': 'gpt_fallback',
            'source_documents': []
        }
    
    def _generate_demo_gpt_fallback(self, question: str) -> Dict[str, Any]:
        """Generate demo GPT fallback response"""
        demo_answer = f"""🎭 **Demo Response** (OpenAI not configured)

**Question:** {question}

**GPT-3.5 Turbo Fallback Response:**

No relevant information was found in your uploaded PDF documents or historical Q&A database after exhaustive search.

**Search Summary:**
❌ PDF Documents: No relevant content found despite comprehensive search
❌ Historical Q&A: No similar questions found
✅ GPT Fallback: Activated

**What the full system would provide:**
With OpenAI configured, GPT-3.5 Turbo would provide a comprehensive answer based on general knowledge, while clearly noting that no specific information was found in your uploaded documents.

**Recommendation:** Consider uploading documents that contain information relevant to your question for more specific answers."""

        return {
            'answer': demo_answer,
            'confidence': 0.7,
            'primary_source': 'gpt_fallback',
            'source_documents': []
        }
    
    def _format_document_context(self, document_context: List[Dict[str, Any]]) -> str:
        """Format document context for the prompt"""
        if not document_context:
            return "No relevant documents found."
        
        formatted_context = []
        for i, doc in enumerate(document_context, 1):
            content_preview = doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content']
            formatted_context.append(
                f"[Source {i}: {doc['filename']} - Page {doc.get('page_number', 'N/A')} - Relevance: {doc['similarity']:.3f} - Method: {doc.get('search_method', 'unknown')}]\n{content_preview}"
            )
        
        return "\n\n".join(formatted_context)

print("✅ Enhanced RAG Service initialized with EXHAUSTIVE PDF PRIORITIZATION:")
print("- PRIORITY 1: Exhaustive PDF document search (5 strategies)")
print("- PRIORITY 2: Historical Q&A database")
print("- PRIORITY 3: GPT-3.5 Turbo fallback")
print("- Document similarity threshold: 0.3 (very inclusive)")
print("- Multiple search strategies: vector, keyword, entity, fuzzy, sampling")
