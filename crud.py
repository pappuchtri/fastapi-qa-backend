from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from models import Question, Answer, Embedding
from schemas import QuestionCreate, AnswerCreate
import numpy as np

def create_question(db: Session, question: QuestionCreate) -> Question:
    """Create a new question in the database"""
    db_question = Question(text=question.text)
    db.add(db_question)
    db.commit()
    db.refresh(db_question)
    return db_question

def get_question(db: Session, question_id: int) -> Optional[Question]:
    """Get a question by ID"""
    return db.query(Question).filter(Question.id == question_id).first()

def get_questions(db: Session, skip: int = 0, limit: int = 100) -> List[Question]:
    """Get a list of questions with pagination"""
    return db.query(Question).order_by(desc(Question.created_at)).offset(skip).limit(limit).all()

def create_answer(db: Session, answer: AnswerCreate) -> Answer:
    """Create a new answer in the database"""
    db_answer = Answer(
        question_id=answer.question_id,
        text=answer.text,
        confidence_score=answer.confidence_score
    )
    db.add(db_answer)
    db.commit()
    db.refresh(db_answer)
    return db_answer

def get_answer(db: Session, answer_id: int) -> Optional[Answer]:
    """Get an answer by ID"""
    return db.query(Answer).filter(Answer.id == answer_id).first()

def get_answers_for_question(db: Session, question_id: int) -> List[Answer]:
    """Get all answers for a specific question"""
    return db.query(Answer).filter(Answer.question_id == question_id).order_by(desc(Answer.created_at)).all()

def create_embedding(db: Session, question_id: int, vector: np.ndarray, model_name: str = "text-embedding-ada-002") -> Embedding:
    """Create a new embedding in the database"""
    # Convert numpy array to list for PostgreSQL ARRAY storage
    vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
    
    db_embedding = Embedding(
        question_id=question_id,
        vector=vector_list,
        model_name=model_name
    )
    db.add(db_embedding)
    db.commit()
    db.refresh(db_embedding)
    return db_embedding

def get_embedding_by_question(db: Session, question_id: int) -> Optional[Embedding]:
    """Get embedding for a specific question"""
    return db.query(Embedding).filter(Embedding.question_id == question_id).first()

def get_all_embeddings(db: Session) -> List[Embedding]:
    """Get all embeddings from the database"""
    return db.query(Embedding).all()

def delete_question(db: Session, question_id: int) -> bool:
    """Delete a question and all related data"""
    question = db.query(Question).filter(Question.id == question_id).first()
    if question:
        db.delete(question)
        db.commit()
        return True
    return False

def get_question_count(db: Session) -> int:
    """Get total number of questions"""
    return db.query(Question).count()

def get_answer_count(db: Session) -> int:
    """Get total number of answers"""
    return db.query(Answer).count()
