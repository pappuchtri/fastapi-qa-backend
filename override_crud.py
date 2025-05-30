from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_
from typing import List, Optional
from override_models import User, AnswerOverride, AnswerReview, PerformanceCache, UserRole, ReviewStatus, OverrideStatus
from override_schemas import UserCreate, OverrideCreate, ReviewCreate
from models import Question, Answer
from datetime import datetime, timedelta
import hashlib

# User management
def create_user(db: Session, user: UserCreate) -> User:
    """Create a new user"""
    db_user = User(
        username=user.username,
        email=user.email,
        role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
    """Get list of users"""
    return db.query(User).offset(skip).limit(limit).all()

# Override management
def create_override(db: Session, override: OverrideCreate, created_by: int) -> AnswerOverride:
    """Create a new answer override"""
    # Mark any existing overrides for this question as superseded
    existing_overrides = db.query(AnswerOverride).filter(
        and_(
            AnswerOverride.question_id == override.question_id,
            AnswerOverride.status == OverrideStatus.ACTIVE
        )
    ).all()
    
    for existing in existing_overrides:
        existing.status = OverrideStatus.SUPERSEDED
    
    # Create new override
    db_override = AnswerOverride(
        question_id=override.question_id,
        original_answer_id=override.original_answer_id,
        override_text=override.override_text,
        reason=override.reason,
        created_by=created_by,
        status=OverrideStatus.ACTIVE
    )
    
    db.add(db_override)
    db.commit()
    db.refresh(db_override)
    return db_override

def get_active_override(db: Session, question_id: int) -> Optional[AnswerOverride]:
    """Get active override for a question"""
    return db.query(AnswerOverride).filter(
        and_(
            AnswerOverride.question_id == question_id,
            AnswerOverride.status == OverrideStatus.ACTIVE
        )
    ).first()

def get_overrides(db: Session, skip: int = 0, limit: int = 100) -> List[AnswerOverride]:
    """Get list of overrides"""
    return db.query(AnswerOverride).order_by(desc(AnswerOverride.created_at)).offset(skip).limit(limit).all()

# Review management
def create_review(db: Session, review: ReviewCreate, reviewer_id: int) -> AnswerReview:
    """Create a new answer review"""
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

def get_pending_reviews(db: Session, skip: int = 0, limit: int = 50) -> List[dict]:
    """Get pending reviews with question and answer details"""
    from sqlalchemy import text
    
    query = text("""
        SELECT 
            q.id as question_id,
            q.text as question_text,
            a.id as answer_id,
            a.text as answer_text,
            a.confidence_score,
            a.created_at,
            COUNT(ar.id) as review_count,
            CASE 
                WHEN a.confidence_score < 0.8 THEN 1.0
                WHEN a.confidence_score < 0.9 THEN 0.7
                ELSE 0.3
            END as priority_score,
            EXISTS(SELECT 1 FROM answer_overrides ao WHERE ao.question_id = q.id AND ao.status = 'active') as has_override
        FROM questions q
        JOIN answers a ON q.id = a.question_id
        LEFT JOIN answer_reviews ar ON a.id = ar.answer_id
        WHERE a.id = (
            SELECT id FROM answers a2 
            WHERE a2.question_id = q.id 
            ORDER BY a2.created_at DESC 
            LIMIT 1
        )
        AND (
            a.confidence_score < 0.9 
            OR EXISTS(SELECT 1 FROM answer_reviews ar2 WHERE ar2.answer_id = a.id AND ar2.is_insufficient = true)
        )
        GROUP BY q.id, q.text, a.id, a.text, a.confidence_score, a.created_at
        ORDER BY priority_score DESC, a.created_at DESC
        LIMIT :limit OFFSET :skip
    """)
    
    result = db.execute(query, {"limit": limit, "skip": skip})
    
    reviews = []
    for row in result:
        reviews.append({
            "question_id": row[0],
            "question_text": row[1],
            "answer_id": row[2],
            "answer_text": row[3][:200] + "..." if len(row[3]) > 200 else row[3],
            "confidence_score": float(row[4]),
            "created_at": row[5],
            "review_count": row[6],
            "priority_score": float(row[7]),
            "has_override": row[8]
        })
    
    return reviews

def get_insufficient_answers(db: Session, skip: int = 0, limit: int = 50) -> List[dict]:
    """Get answers marked as insufficient"""
    from sqlalchemy import text
    
    query = text("""
        SELECT DISTINCT
            q.id as question_id,
            q.text as question_text,
            a.id as answer_id,
            a.text as answer_text,
            a.confidence_score,
            ar.review_notes,
            ar.reviewed_at,
            u.username as reviewer_name
        FROM questions q
        JOIN answers a ON q.id = a.question_id
        JOIN answer_reviews ar ON a.id = ar.answer_id
        JOIN users u ON ar.reviewer_id = u.id
        WHERE ar.is_insufficient = true
        ORDER BY ar.reviewed_at DESC
        LIMIT :limit OFFSET :skip
    """)
    
    result = db.execute(query, {"limit": limit, "skip": skip})
    
    insufficient = []
    for row in result:
        insufficient.append({
            "question_id": row[0],
            "question_text": row[1],
            "answer_id": row[2],
            "answer_text": row[3][:200] + "..." if len(row[3]) > 200 else row[3],
            "confidence_score": float(row[4]),
            "review_notes": row[5],
            "reviewed_at": row[6],
            "reviewer_name": row[7]
        })
    
    return insufficient

def get_review_stats(db: Session) -> dict:
    """Get review statistics"""
    from sqlalchemy import text, func
    
    # Get basic counts
    total_pending = db.query(Answer).filter(Answer.confidence_score < 0.9).count()
    total_insufficient = db.query(AnswerReview).filter(AnswerReview.is_insufficient == True).count()
    total_overrides = db.query(AnswerOverride).filter(AnswerOverride.status == OverrideStatus.ACTIVE).count()
    
    # Get average confidence for insufficient answers
    avg_confidence_result = db.query(func.avg(Answer.confidence_score)).join(AnswerReview).filter(
        AnswerReview.is_insufficient == True
    ).scalar()
    avg_confidence_insufficient = float(avg_confidence_result) if avg_confidence_result else 0.0
    
    # Get top review flags
    flag_counts = db.execute(text("""
        SELECT 
            'insufficient' as flag_type, COUNT(*) as count FROM answer_reviews WHERE is_insufficient = true
        UNION ALL
        SELECT 
            'needs_more_context' as flag_type, COUNT(*) as count FROM answer_reviews WHERE needs_more_context = true
        UNION ALL
        SELECT 
            'factual_accuracy_concern' as flag_type, COUNT(*) as count FROM answer_reviews WHERE factual_accuracy_concern = true
        UNION ALL
        SELECT 
            'compliance_concern' as flag_type, COUNT(*) as count FROM answer_reviews WHERE compliance_concern = true
        ORDER BY count DESC
    """)).fetchall()
    
    top_flags = [{"flag": row[0], "count": row[1]} for row in flag_counts]
    
    # Get reviewer activity
    reviewer_activity = db.execute(text("""
        SELECT 
            u.username,
            COUNT(ar.id) as total_reviews,
            COUNT(CASE WHEN ar.review_status = 'approved' THEN 1 END) as approved,
            COUNT(CASE WHEN ar.is_insufficient THEN 1 END) as insufficient_flagged
        FROM users u
        LEFT JOIN answer_reviews ar ON u.id = ar.reviewer_id
        WHERE u.role IN ('reviewer', 'admin')
        GROUP BY u.id, u.username
        ORDER BY total_reviews DESC
        LIMIT 10
    """)).fetchall()
    
    reviewer_stats = [
        {
            "username": row[0],
            "total_reviews": row[1],
            "approved": row[2],
            "insufficient_flagged": row[3]
        }
        for row in reviewer_activity
    ]
    
    return {
        "total_pending_reviews": total_pending,
        "total_insufficient_answers": total_insufficient,
        "total_overrides": total_overrides,
        "avg_confidence_insufficient": avg_confidence_insufficient,
        "top_review_flags": top_flags,
        "reviewer_activity": reviewer_stats
    }

# Performance cache management
def get_cache_stats(db: Session) -> dict:
    """Get performance cache statistics"""
    from sqlalchemy import func
    
    total_cached = db.query(PerformanceCache).count()
    total_hits = db.query(func.sum(PerformanceCache.cache_hits)).scalar() or 0
    avg_generation_time = db.query(func.avg(PerformanceCache.generation_time_ms)).scalar() or 0
    
    # Get cache hit rate by source type
    cache_by_source = db.execute(text("""
        SELECT 
            source_type,
            COUNT(*) as cached_count,
            SUM(cache_hits) as total_hits,
            AVG(generation_time_ms) as avg_generation_time
        FROM performance_cache
        GROUP BY source_type
        ORDER BY cached_count DESC
    """)).fetchall()
    
    source_stats = [
        {
            "source_type": row[0],
            "cached_count": row[1],
            "total_hits": row[2],
            "avg_generation_time": row[3]
        }
        for row in cache_by_source
    ]
    
    return {
        "total_cached_answers": total_cached,
        "total_cache_hits": total_hits,
        "avg_generation_time_ms": avg_generation_time,
        "cache_by_source": source_stats
    }

def cleanup_expired_cache(db: Session) -> int:
    """Clean up expired cache entries"""
    expired_count = db.query(PerformanceCache).filter(
        PerformanceCache.expires_at < datetime.utcnow()
    ).count()
    
    db.query(PerformanceCache).filter(
        PerformanceCache.expires_at < datetime.utcnow()
    ).delete()
    
    db.commit()
    return expired_count
