from sqlalchemy.orm import Session
from sqlalchemy import func, text, desc, and_, or_
from datetime import datetime, timedelta
import hashlib
from typing import List, Dict, Any, Optional

from override_models import (
    User, UserRole, AnswerOverride, OverrideStatus, 
    AnswerReview, ReviewStatus, PerformanceCache,
    ChunkUsageLog, ReviewTag, ReviewTagAssociation
)
from override_schemas import UserCreate, OverrideCreate, ReviewCreate

# User CRUD operations
def get_user(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
    return db.query(User).offset(skip).limit(limit).all()

def create_user(db: Session, user: UserCreate) -> User:
    db_user = User(
        username=user.username,
        email=user.email,
        role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user_role(db: Session, user_id: int, role: UserRole) -> Optional[User]:
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        db_user.role = role
        db.commit()
        db.refresh(db_user)
    return db_user

# Override CRUD operations
def create_override(db: Session, override: OverrideCreate, created_by: int) -> AnswerOverride:
    # First, supersede any existing active overrides
    existing_overrides = db.query(AnswerOverride).filter(
        AnswerOverride.question_id == override.question_id,
        AnswerOverride.status == OverrideStatus.ACTIVE
    ).all()
    
    for existing in existing_overrides:
        existing.status = OverrideStatus.SUPERSEDED
    
    # Create new override
    db_override = AnswerOverride(
        question_id=override.question_id,
        original_answer_id=override.original_answer_id,
        override_text=override.override_text,
        reason=override.reason,
        status=OverrideStatus.ACTIVE,
        created_by=created_by
    )
    
    db.add(db_override)
    db.commit()
    db.refresh(db_override)
    return db_override

def get_override(db: Session, override_id: int) -> Optional[AnswerOverride]:
    return db.query(AnswerOverride).filter(AnswerOverride.id == override_id).first()

def get_active_override(db: Session, question_id: int) -> Optional[AnswerOverride]:
    return db.query(AnswerOverride).filter(
        AnswerOverride.question_id == question_id,
        AnswerOverride.status == OverrideStatus.ACTIVE
    ).first()

def get_overrides(db: Session, skip: int = 0, limit: int = 100) -> List[AnswerOverride]:
    return db.query(AnswerOverride).order_by(desc(AnswerOverride.created_at)).offset(skip).limit(limit).all()

def revoke_override(db: Session, override_id: int) -> Optional[AnswerOverride]:
    db_override = db.query(AnswerOverride).filter(AnswerOverride.id == override_id).first()
    if db_override:
        db_override.status = OverrideStatus.REVOKED
        db.commit()
        db.refresh(db_override)
    return db_override

# Review CRUD operations
def create_review(db: Session, review: ReviewCreate, reviewer_id: int) -> AnswerReview:
    db_review = AnswerReview(
        question_id=review.question_id,
        answer_id=review.answer_id,
        override_id=review.override_id,
        review_status=review.review_status,
        review_notes=review.review_notes,
        reviewer_id=reviewer_id,
        is_insufficient=review.is_insufficient,
        needs_more_context=review.needs_more_context,
        factual_accuracy_concern=review.factual_accuracy_concern,
        compliance_concern=review.compliance_concern
    )
    
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    return db_review

def get_review(db: Session, review_id: int) -> Optional[AnswerReview]:
    return db.query(AnswerReview).filter(AnswerReview.id == review_id).first()

def get_reviews_for_answer(db: Session, answer_id: int) -> List[AnswerReview]:
    return db.query(AnswerReview).filter(AnswerReview.answer_id == answer_id).all()

def get_pending_reviews(db: Session, skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
    # Get questions with low confidence or other flags that need review
    query = text("""
        SELECT 
            q.id as question_id, 
            q.text as question_text,
            a.id as answer_id, 
            a.text as answer_text,
            a.confidence_score,
            a.created_at,
            (SELECT COUNT(*) FROM answer_reviews ar WHERE ar.answer_id = a.id) as review_count,
            EXISTS(SELECT 1 FROM answer_overrides ao WHERE ao.question_id = q.id AND ao.status = 'active') as has_override,
            -- Priority score calculation: lower confidence = higher priority
            (1 - a.confidence_score) * 10 + 
            (CASE WHEN a.confidence_score < 0.7 THEN 5 ELSE 0 END) +
            (CASE WHEN review_count = 0 THEN 3 ELSE 0 END) as priority_score
        FROM 
            questions q
        JOIN 
            answers a ON q.id = a.question_id
        WHERE
            a.id = (SELECT id FROM answers WHERE question_id = q.id ORDER BY created_at DESC LIMIT 1)
            AND a.confidence_score < 0.9
            AND NOT EXISTS (
                SELECT 1 FROM answer_reviews ar 
                WHERE ar.answer_id = a.id 
                AND ar.review_status IN ('approved', 'rejected')
            )
        ORDER BY 
            priority_score DESC, a.created_at DESC
        LIMIT :limit OFFSET :skip
    """)
    
    result = db.execute(query, {"limit": limit, "skip": skip})
    
    reviews = []
    for row in result:
        reviews.append({
            "question_id": row.question_id,
            "question_text": row.question_text,
            "answer_id": row.answer_id,
            "answer_text": row.answer_text,
            "confidence_score": float(row.confidence_score),
            "created_at": row.created_at,
            "review_count": row.review_count,
            "has_override": row.has_override,
            "priority_score": float(row.priority_score)
        })
    
    return reviews

def get_insufficient_answers(db: Session, skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
    query = text("""
        SELECT 
            q.id as question_id, 
            q.text as question_text,
            a.id as answer_id, 
            a.text as answer_text,
            a.confidence_score,
            ar.review_notes,
            ar.reviewed_at,
            u.username as reviewer_name
        FROM 
            answer_reviews ar
        JOIN 
            answers a ON ar.answer_id = a.id
        JOIN 
            questions q ON ar.question_id = q.id
        JOIN
            users u ON ar.reviewer_id = u.id
        WHERE
            ar.is_insufficient = TRUE
        ORDER BY 
            ar.reviewed_at DESC
        LIMIT :limit OFFSET :skip
    """)
    
    result = db.execute(query, {"limit": limit, "skip": skip})
    
    insufficient = []
    for row in result:
        insufficient.append({
            "question_id": row.question_id,
            "question_text": row.question_text,
            "answer_id": row.answer_id,
            "answer_text": row.answer_text,
            "confidence_score": float(row.confidence_score),
            "review_notes": row.review_notes,
            "reviewed_at": row.reviewed_at,
            "reviewer_name": row.reviewer_name
        })
    
    return insufficient

def get_review_stats(db: Session) -> Dict[str, Any]:
    # Get total pending reviews
    pending_reviews = db.query(func.count(AnswerReview.id)).filter(
        AnswerReview.review_status == ReviewStatus.PENDING
    ).scalar()
    
    # Get total insufficient answers
    insufficient_answers = db.query(func.count(AnswerReview.id)).filter(
        AnswerReview.is_insufficient == True
    ).scalar()
    
    # Get total overrides
    total_overrides = db.query(func.count(AnswerOverride.id)).scalar()
    
    # Get average confidence of insufficient answers
    avg_confidence = db.query(func.avg(AnswerReview.confidence_score)).filter(
        AnswerReview.is_insufficient == True
    ).scalar() or 0.0
    
    # Get top review flags
    flags_query = text("""
        SELECT 
            CASE 
                WHEN is_insufficient THEN 'insufficient'
                WHEN needs_more_context THEN 'needs_context'
                WHEN factual_accuracy_concern THEN 'factual_concern'
                WHEN compliance_concern THEN 'compliance_concern'
                ELSE 'other'
            END as flag_type,
            COUNT(*) as count
        FROM 
            answer_reviews
        WHERE
            is_insufficient OR needs_more_context OR 
            factual_accuracy_concern OR compliance_concern
        GROUP BY 
            flag_type
        ORDER BY 
            count DESC
        LIMIT 5
    """)
    
    flags_result = db.execute(flags_query)
    top_flags = [{"flag": row.flag_type, "count": row.count} for row in flags_result]
    
    # Get reviewer activity
    activity_query = text("""
        SELECT 
            u.username,
            COUNT(*) as review_count,
            MAX(ar.reviewed_at) as last_activity
        FROM 
            answer_reviews ar
        JOIN
            users u ON ar.reviewer_id = u.id
        GROUP BY 
            u.username
        ORDER BY 
            review_count DESC
        LIMIT 5
    """)
    
    activity_result = db.execute(activity_query)
    reviewer_activity = [
        {
            "reviewer": row.username, 
            "review_count": row.review_count,
            "last_activity": row.last_activity
        } 
        for row in activity_result
    ]
    
    return {
        "total_pending_reviews": pending_reviews or 0,
        "total_insufficient_answers": insufficient_answers or 0,
        "total_overrides": total_overrides or 0,
        "avg_confidence_insufficient": float(avg_confidence),
        "top_review_flags": top_flags,
        "reviewer_activity": reviewer_activity
    }

# Performance Cache CRUD operations
def check_performance_cache(db: Session, question_text: str) -> Optional[Dict[str, Any]]:
    # Create hash of question for lookup
    question_hash = hashlib.md5(question_text.encode()).hexdigest()
    
    # Look up in cache
    cache_entry = db.query(PerformanceCache).filter(
        PerformanceCache.question_hash == question_hash,
        or_(
            PerformanceCache.expires_at.is_(None),
            PerformanceCache.expires_at > datetime.utcnow()
        )
    ).first()
    
    if cache_entry:
        # Update access time and hit count
        cache_entry.last_accessed = datetime.utcnow()
        cache_entry.cache_hits += 1
        db.commit()
        
        return {
            "answer": cache_entry.cached_answer,
            "confidence": float(cache_entry.confidence_score),
            "source_type": cache_entry.source_type,
            "generation_time_ms": cache_entry.generation_time_ms,
            "cache_hits": cache_entry.cache_hits
        }
    
    return None

def store_performance_cache(
    db: Session, 
    question_text: str, 
    answer_text: str, 
    generation_time_ms: int,
    confidence_score: float,
    source_type: str,
    expire_hours: int = 24
) -> PerformanceCache:
    # Only cache if generation was slow (> 3 seconds)
    if generation_time_ms < 3000:
        return None
        
    # Create hash of question
    question_hash = hashlib.md5(question_text.encode()).hexdigest()
    
    # Check if already exists
    existing = db.query(PerformanceCache).filter(
        PerformanceCache.question_hash == question_hash
    ).first()
    
    if existing:
        # Update existing entry
        existing.cached_answer = answer_text
        existing.generation_time_ms = generation_time_ms
        existing.confidence_score = confidence_score
        existing.source_type = source_type
        existing.last_accessed = datetime.utcnow()
        existing.expires_at = datetime.utcnow() + timedelta(hours=expire_hours)
        db.commit()
        return existing
    
    # Create new cache entry
    cache_entry = PerformanceCache(
        question_hash=question_hash,
        question_text=question_text,
        cached_answer=answer_text,
        generation_time_ms=generation_time_ms,
        confidence_score=confidence_score,
        source_type=source_type,
        expires_at=datetime.utcnow() + timedelta(hours=expire_hours)
    )
    
    db.add(cache_entry)
    db.commit()
    db.refresh(cache_entry)
    return cache_entry

def get_cache_stats(db: Session) -> Dict[str, Any]:
    # Get total cache entries
    total_entries = db.query(func.count(PerformanceCache.id)).scalar() or 0
    
    # Get active cache entries
    active_entries = db.query(func.count(PerformanceCache.id)).filter(
        or_(
            PerformanceCache.expires_at.is_(None),
            PerformanceCache.expires_at > datetime.utcnow()
        )
    ).scalar() or 0
    
    # Get expired cache entries
    expired_entries = db.query(func.count(PerformanceCache.id)).filter(
        PerformanceCache.expires_at <= datetime.utcnow()
    ).scalar() or 0
    
    # Get total cache hits
    total_hits = db.query(func.sum(PerformanceCache.cache_hits)).scalar() or 0
    
    # Get average generation time
    avg_generation_time = db.query(func.avg(PerformanceCache.generation_time_ms)).scalar() or 0
    
    # Get cache by source type
    source_query = text("""
        SELECT 
            source_type,
            COUNT(*) as count,
            AVG(generation_time_ms) as avg_time
        FROM 
            performance_cache
        GROUP BY 
            source_type
    """)
    
    source_result = db.execute(source_query)
    source_stats = [
        {
            "source_type": row.source_type, 
            "count": row.count,
            "avg_generation_time_ms": float(row.avg_time)
        } 
        for row in source_result
    ]
    
    # Get most frequently accessed cache entries
    top_query = text("""
        SELECT 
            question_text,
            cache_hits,
            generation_time_ms,
            created_at
        FROM 
            performance_cache
        ORDER BY 
            cache_hits DESC
        LIMIT 5
    """)
    
    top_result = db.execute(top_query)
    top_entries = [
        {
            "question": row.question_text[:50] + "..." if len(row.question_text) > 50 else row.question_text,
            "hits": row.cache_hits,
            "generation_time_ms": row.generation_time_ms,
            "created_at": row.created_at
        } 
        for row in top_result
    ]
    
    return {
        "total_entries": total_entries,
        "active_entries": active_entries,
        "expired_entries": expired_entries,
        "total_hits": total_hits,
        "avg_generation_time_ms": float(avg_generation_time),
        "source_stats": source_stats,
        "top_entries": top_entries
    }

def cleanup_expired_cache(db: Session) -> int:
    # Delete expired cache entries
    result = db.query(PerformanceCache).filter(
        PerformanceCache.expires_at <= datetime.utcnow()
    ).delete()
    
    db.commit()
    return result

# Chunk usage logging
def log_chunk_usage(
    db: Session, 
    question_id: int, 
    chunk_id: int, 
    relevance_score: float,
    position: int,
    was_used: bool = True
) -> ChunkUsageLog:
    log_entry = ChunkUsageLog(
        question_id=question_id,
        chunk_id=chunk_id,
        relevance_score=relevance_score,
        position_in_results=position,
        was_used_in_answer=was_used
    )
    
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return log_entry

def get_chunk_usage_stats(db: Session) -> Dict[str, Any]:
    # Get total chunks used
    total_chunks = db.query(func.count(ChunkUsageLog.id)).scalar() or 0
    
    # Get average chunks per question
    avg_chunks_query = text("""
        SELECT 
            AVG(chunk_count) as avg_chunks
        FROM (
            SELECT 
                question_id, 
                COUNT(*) as chunk_count
            FROM 
                chunk_usage_log
            WHERE 
                was_used_in_answer = TRUE
            GROUP BY 
                question_id
        ) as subquery
    """)
    
    avg_chunks = db.execute(avg_chunks_query).scalar() or 0
    
    # Get most frequently used chunks
    top_chunks_query = text("""
        SELECT 
            dc.id as chunk_id,
            d.original_filename,
            COUNT(*) as usage_count,
            AVG(cul.relevance_score) as avg_relevance
        FROM 
            chunk_usage_log cul
        JOIN
            document_chunks dc ON cul.chunk_id = dc.id
        JOIN
            documents d ON dc.document_id = d.id
        WHERE
            cul.was_used_in_answer = TRUE
        GROUP BY 
            dc.id, d.original_filename
        ORDER BY 
            usage_count DESC
        LIMIT 5
    """)
    
    top_chunks_result = db.execute(top_chunks_query)
    top_chunks = [
        {
            "chunk_id": row.chunk_id,
            "document": row.original_filename,
            "usage_count": row.usage_count,
            "avg_relevance": float(row.avg_relevance)
        } 
        for row in top_chunks_result
    ]
    
    return {
        "total_chunks_used": total_chunks,
        "avg_chunks_per_question": float(avg_chunks),
        "top_chunks": top_chunks
    }
