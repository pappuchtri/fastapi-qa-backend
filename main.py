from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from document_models import Document, DocumentChunk

app = FastAPI()


class DocumentCreate(BaseModel):
    title: str
    content: str


class DocumentChunkCreate(BaseModel):
    document_id: int
    content: str
    order: int


@app.post("/documents/", response_model=Document)
def create_document(document: DocumentCreate):
    """
    Creates a new document.
    """
    # In a real application, you would save this to a database.
    # For this example, we'll just return the document.
    new_document = Document(id=1, title=document.title, content=document.content)
    return new_document


@app.get("/documents/", response_model=List[Document])
def list_documents():
    """
    Lists all documents.
    """
    try:
        # In a real application, you would fetch this from a database.
        # For this example, we'll just return a hardcoded list.
        documents = [
            Document(id=1, title="Document 1", content="This is the first document."),
            Document(id=2, title="Document 2", content="This is the second document."),
        ]
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}", response_model=Document)
def read_document(document_id: int):
    """
    Reads a specific document by ID.
    """
    # In a real application, you would fetch this from a database.
    # For this example, we'll just return a hardcoded document.
    if document_id == 1:
        return Document(id=1, title="Document 1", content="This is the first document.")
    else:
        raise HTTPException(status_code=404, detail="Document not found")


@app.post("/document_chunks/", response_model=DocumentChunk)
def create_document_chunk(document_chunk: DocumentChunkCreate):
    """
    Creates a new document chunk.
    """
    # In a real application, you would save this to a database.
    # For this example, we'll just return the document chunk.
    new_document_chunk = DocumentChunk(
        id=1,
        document_id=document_chunk.document_id,
        content=document_chunk.content,
        order=document_chunk.order,
    )
    return new_document_chunk


@app.get("/document_chunks/{document_id}", response_model=List[DocumentChunk])
def list_document_chunks(document_id: int):
    """
    Lists all document chunks for a specific document.
    """
    # In a real application, you would fetch this from a database.
    # For this example, we'll just return a hardcoded list.
    document_chunks = [
        DocumentChunk(id=1, document_id=document_id, content="Chunk 1", order=1),
        DocumentChunk(id=2, document_id=document_id, content="Chunk 2", order=2),
    ]
    return document_chunks
