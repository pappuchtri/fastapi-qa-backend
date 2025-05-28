from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from typing import List, Optional
import numpy as np
from document_models import Document, DocumentChunk
from document_schemas import DocumentCreate, DocumentChunkCreate

def create_document(db: Session, document: DocumentCreate) -> Document:
    """Create a new document record"""
    db_document = Document(**document.dict())
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def get_document(db: Session, document_id: int) -> Optional[Document]:
    """Get document by ID"""
    return db.query(Document).filter(Document.id == document_id).first()

def get_documents(db: Session, skip: int = 0, limit: int = 100) -> List[Document]:
    """Get list of documents with pagination"""
    return db.query(Document).order_by(desc(Document.upload_date)).offset(skip).limit(limit).all()

def get_document_count(db: Session) -> int:
    """Get total number of documents"""
    return db.query(Document).count()

def delete_document(db: Session, document_id: int) -> bool:
    """Delete document and all its chunks"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if document:
        db.delete(document)
        db.commit()
        return True
    return False

def update_document_status(
    db: Session, 
    document_id: int, 
    processing_status: str,
    processed: bool = None,
    error_message: str = None,
    total_pages: int = None,
    total_chunks: int = None
) -> Optional[Document]:
    """Update document processing status"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if document:
        document.processing_status = processing_status
        if processed is not None:
            document.processed = processed
        if error_message is not None:
            document.error_message = error_message
        if total_pages is not None:
            document.total_pages = total_pages
        if total_chunks is not None:
            document.total_chunks = total_chunks
        
        db.commit()
        db.refresh(document)
        return document
    return None

def create_document_chunk(db: Session, chunk: DocumentChunkCreate, embedding: np.ndarray = None) -> DocumentChunk:
    """Create a new document chunk with optional embedding"""
    chunk_data = chunk.dict()
    
    if embedding is not None:
        # Convert numpy array to list for PostgreSQL VECTOR storage
        chunk_data['chunk_embedding'] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
    
    db_chunk = DocumentChunk(**chunk_data)
    db.add(db_chunk)
    db.commit()
    db.refresh(db_chunk)
    return db_chunk

def get_document_chunks(db: Session, document_id: int) -> List[DocumentChunk]:
    """Get all chunks for a document"""
    return db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).order_by(DocumentChunk.chunk_index).all()

def search_similar_chunks(db: Session, query_embedding: np.ndarray, limit: int = 5, document_ids: List[int] = None) -> List[DocumentChunk]:
    """Search for similar document chunks using vector similarity"""
    try:
        query = db.query(DocumentChunk).filter(DocumentChunk.chunk_embedding.isnot(None))
        
        if document_ids:
            query = query.filter(DocumentChunk.document_id.in_(document_ids))
        
        chunks = query.all()
        
        if not chunks:
            return []
        
        # Calculate similarities
        similarities = []
        for chunk in chunks:
            try:
                if chunk.chunk_embedding:
                    chunk_vector = np.array(chunk.chunk_embedding)
                    similarity = np.dot(query_embedding, chunk_vector) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_vector)
                    )
                    similarities.append((chunk, float(similarity)))
            except Exception as e:
                print(f"Error calculating similarity for chunk {chunk.id}: {str(e)}")
                continue
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in similarities[:limit]]
        
    except Exception as e:
        print(f"Error in similarity search: {str(e)}")
        return []

def get_chunk_count(db: Session) -> int:
    """Get total number of chunks"""
    return db.query(DocumentChunk).count()

print("✅ Document CRUD operations created:")
print("- Document management (create, read, update, delete)")
print("- Chunk management with embeddings")
print("- Vector similarity search")
print("- Status tracking and error handling")
