import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional

from app.models.schemas import UploadResponse, AskRequest, AskResponse
from app.services.ingestion import IngestionService
from app.services.rag import RAGService
from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

def get_ingestion_service() -> IngestionService:
    return IngestionService(vector_store_path=settings.vector_store_path)

def get_rag_service() -> RAGService:
    return RAGService(vector_store_path=settings.vector_store_path)

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    ingestor: IngestionService = Depends(get_ingestion_service)
):
    """Upload and process a document for the RAG system."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    
    logger.info(f"Processing uploaded file: {file.filename}")
    
    try:
        doc_id = await ingestor.ingest_file(file)
        return {
            "document_id": doc_id,
            "filename": file.filename,
            "status": "ingested",
            "message": "Document processed successfully"
        }
    except Exception as e:
        logger.error(f"Error processing document {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    rag: RAGService = Depends(get_rag_service)
):
    """Ask a question to the RAG system."""
    try:
        logger.info(f"Processing question: {request.question}")
        
        answer, sources = await rag.answer_query(
            query=request.question,
            top_k=request.top_k
        )
        
        # Extract source information
        source_texts = [s["source"] for s in sources if s.get("source")]
        
        return AskResponse(
            answer=answer,
            sources=source_texts,
            relevant_docs=None  # Can be populated if tracking document IDs
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing your question: {str(e)}"
        )
