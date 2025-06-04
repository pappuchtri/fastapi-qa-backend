from sqlalchemy.orm import Session
from sqlalchemy import text, or_, and_
from typing import List, Dict, Any, Optional, Tuple
import re
from models import KnowledgeBase
from schemas import KnowledgeBaseCreate, KnowledgeBaseUpdate
import logging

logger = logging.getLogger(__name__)

class KnowledgeBaseService:
    """Service for managing and searching the built-in knowledge base"""
    
    def __init__(self):
        self.similarity_threshold = 0.7  # Threshold for fuzzy matching
        
    def create_knowledge_entry(
        self, 
        db: Session, 
        entry: KnowledgeBaseCreate,
        created_by: str = "system"
    ) -> KnowledgeBase:
        """Create a new knowledge base entry"""
        try:
            db_entry = KnowledgeBase(
                category=entry.category,
                question=entry.question,
                answer=entry.answer,
                keywords=entry.keywords or [],
                priority=entry.priority,
                is_active=entry.is_active,
                created_by=created_by
            )
            
            db.add(db_entry)
            db.commit()
            db.refresh(db_entry)
            
            logger.info(f"✅ Created knowledge base entry: {entry.question[:50]}...")
            return db_entry
            
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error creating knowledge base entry: {str(e)}")
            raise
    
    def update_knowledge_entry(
        self,
        db: Session,
        entry_id: int,
        update_data: KnowledgeBaseUpdate
    ) -> Optional[KnowledgeBase]:
        """Update an existing knowledge base entry"""
        try:
            entry = db.query(KnowledgeBase).filter(KnowledgeBase.id == entry_id).first()
            if not entry:
                return None
            
            update_dict = update_data.dict(exclude_unset=True)
            for field, value in update_dict.items():
                setattr(entry, field, value)
            
            db.commit()
            db.refresh(entry)
            
            logger.info(f"✅ Updated knowledge base entry {entry_id}")
            return entry
            
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error updating knowledge base entry: {str(e)}")
            raise
    
    def search_knowledge_base(
        self, 
        db: Session, 
        query: str,
        category: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base using multiple strategies:
        1. Exact question match
        2. Keyword matching
        3. Fuzzy text matching
        """
        try:
            logger.info(f"🔍 Searching knowledge base for: {query[:50]}...")
            
            # Normalize query for better matching
            normalized_query = self._normalize_text(query)
            query_words = set(normalized_query.split())
            
            # Build base query
            base_query = db.query(KnowledgeBase).filter(
                KnowledgeBase.is_active == True
            )
            
            if category:
                base_query = base_query.filter(KnowledgeBase.category == category)
            
            # Get all active entries
            all_entries = base_query.all()
            
            if not all_entries:
                logger.info("📭 No active knowledge base entries found")
                return []
            
            logger.info(f"📊 Searching through {len(all_entries)} knowledge base entries")
            
            # Score each entry
            scored_entries = []
            for entry in all_entries:
                score = self._calculate_relevance_score(query, normalized_query, query_words, entry)
                if score > 0:
                    scored_entries.append({
                        "entry": entry,
                        "score": score,
                        "match_type": self._get_match_type(score)
                    })
            
            # Sort by score (highest first) and priority
            scored_entries.sort(key=lambda x: (x["score"], x["entry"].priority), reverse=True)
            
            # Return top results
            results = []
            for item in scored_entries[:limit]:
                entry = item["entry"]
                results.append({
                    "id": entry.id,
                    "category": entry.category,
                    "question": entry.question,
                    "answer": entry.answer,
                    "keywords": entry.keywords or [],
                    "priority": entry.priority,
                    "relevance_score": item["score"],
                    "match_type": item["match_type"],
                    "source": "knowledge_base"
                })
            
            logger.info(f"🎯 Found {len(results)} relevant knowledge base entries")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching knowledge base: {str(e)}")
            return []
    
    def get_best_knowledge_match(
        self, 
        db: Session, 
        query: str,
        category: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get the single best match from knowledge base"""
        results = self.search_knowledge_base(db, query, category, limit=1)
        
        if results and results[0]["relevance_score"] >= self.similarity_threshold:
            logger.info(f"✅ Found high-confidence knowledge base match (score: {results[0]['relevance_score']:.3f})")
            return results[0]
        
        logger.info("📭 No high-confidence knowledge base match found")
        return None
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Convert to lowercase and remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        # Remove common punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        return text
    
    def _calculate_relevance_score(
        self, 
        original_query: str,
        normalized_query: str, 
        query_words: set, 
        entry: KnowledgeBase
    ) -> float:
        """Calculate relevance score for a knowledge base entry"""
        score = 0.0
        
        # Normalize entry question and answer
        normalized_question = self._normalize_text(entry.question)
        normalized_answer = self._normalize_text(entry.answer)
        
        # 1. Exact question match (highest score)
        if normalized_query == normalized_question:
            score += 10.0
        
        # 2. Query contained in question
        elif normalized_query in normalized_question:
            score += 8.0
        
        # 3. Question contained in query
        elif normalized_question in normalized_query:
            score += 7.0
        
        # 4. Keyword matching
        if entry.keywords:
            keyword_matches = 0
            for keyword in entry.keywords:
                normalized_keyword = self._normalize_text(keyword)
                if normalized_keyword in normalized_query:
                    keyword_matches += 1
                    score += 2.0
            
            # Bonus for multiple keyword matches
            if keyword_matches > 1:
                score += keyword_matches * 0.5
        
        # 5. Word overlap scoring
        question_words = set(normalized_question.split())
        answer_words = set(normalized_answer.split())
        
        # Question word overlap
        question_overlap = len(query_words.intersection(question_words))
        if question_overlap > 0:
            score += question_overlap * 1.0
        
        # Answer word overlap (lower weight)
        answer_overlap = len(query_words.intersection(answer_words))
        if answer_overlap > 0:
            score += answer_overlap * 0.5
        
        # 6. Priority bonus
        score += entry.priority * 0.1
        
        # 7. Length penalty for very long entries (prefer concise matches)
        if len(entry.question) > 200:
            score *= 0.9
        
        return score
    
    def _get_match_type(self, score: float) -> str:
        """Determine match type based on score"""
        if score >= 10.0:
            return "exact_match"
        elif score >= 8.0:
            return "high_confidence"
        elif score >= 5.0:
            return "good_match"
        elif score >= 2.0:
            return "partial_match"
        else:
            return "low_confidence"
    
    def get_categories(self, db: Session) -> List[str]:
        """Get all available categories"""
        try:
            result = db.query(KnowledgeBase.category).filter(
                KnowledgeBase.is_active == True
            ).distinct().all()
            
            categories = [row[0] for row in result]
            logger.info(f"📂 Found {len(categories)} knowledge base categories")
            return categories
            
        except Exception as e:
            logger.error(f"❌ Error getting categories: {str(e)}")
            return []
    
    def get_entry_by_id(self, db: Session, entry_id: int) -> Optional[KnowledgeBase]:
        """Get knowledge base entry by ID"""
        try:
            return db.query(KnowledgeBase).filter(KnowledgeBase.id == entry_id).first()
        except Exception as e:
            logger.error(f"❌ Error getting knowledge base entry: {str(e)}")
            return None
    
    def delete_entry(self, db: Session, entry_id: int) -> bool:
        """Delete knowledge base entry"""
        try:
            entry = db.query(KnowledgeBase).filter(KnowledgeBase.id == entry_id).first()
            if not entry:
                return False
            
            db.delete(entry)
            db.commit()
            
            logger.info(f"🗑️ Deleted knowledge base entry {entry_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error deleting knowledge base entry: {str(e)}")
            return False
    
    def get_stats(self, db: Session) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            total_entries = db.query(KnowledgeBase).count()
            active_entries = db.query(KnowledgeBase).filter(KnowledgeBase.is_active == True).count()
            categories = self.get_categories(db)
            
            # Get entries by priority
            priority_stats = {}
            for priority in range(1, 11):
                count = db.query(KnowledgeBase).filter(
                    KnowledgeBase.priority == priority,
                    KnowledgeBase.is_active == True
                ).count()
                if count > 0:
                    priority_stats[f"priority_{priority}"] = count
            
            return {
                "total_entries": total_entries,
                "active_entries": active_entries,
                "inactive_entries": total_entries - active_entries,
                "categories": categories,
                "category_count": len(categories),
                "priority_distribution": priority_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting knowledge base stats: {str(e)}")
            return {}

# Initialize service
knowledge_service = KnowledgeBaseService()
