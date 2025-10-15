from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import UploadResponse, AskRequest, AskResponse
from app.services.ingestion import IngestionService
from app.services.rag import RAGService

router = APIRouter()
ingestor = IngestionService()
rag = RAGService()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    try:
        doc_id = await ingestor.ingest_file(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"document_id": doc_id, "status":"ingested"}

@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        answer, sources = rag.answer_query(req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"answer": answer, "sources": sources}
