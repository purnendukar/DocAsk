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

@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload Document",
    description="""
    Upload a document to be processed and indexed by the RAG system.
    
    The document will be chunked, embedded, and stored in the vector database for future queries.
    Supported formats include: PDF, TXT, DOCX, and other common document formats.
    """,
    response_description="Document processing status and metadata",
    responses={
        200: {"description": "Document processed successfully"},
        400: {"description": "Invalid or missing file"},
        500: {"description": "Error processing document"}
    }
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload and process"),
    ingestor: IngestionService = Depends(get_ingestion_service)
):
    """
    Upload and process a document for the RAG system.
    
    - **file**: The document file to be processed
    - **returns**: Processing status and document metadata
    
    The document will be processed asynchronously and made available for querying once complete.
    """
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

@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a Question",
    description="""
    Ask a natural language question and get an answer based on the indexed documents.
    
    The system will retrieve relevant document chunks and generate a contextual answer.
    """,
    response_description="Answer to the question with source references",
    responses={
        200: {"description": "Successfully generated answer"},
        500: {"description": "Error processing question"}
    }
)
async def ask(
    request: AskRequest,
    rag: RAGService = Depends(get_rag_service)
):
    """
    Query the RAG system with a question to get an informed answer.
    
    - **question**: The natural language question to ask
    - **top_k**: Number of relevant document chunks to consider (default: 3)
    - **returns**: Generated answer with source references
    
    The response includes the answer along with the document sources used to generate it.
    """
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
