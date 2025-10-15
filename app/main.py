from fastapi import FastAPI
from app.api.v1.endpoints import router as v1_router
from app.core.config import settings

app = FastAPI(title="RAG-QA", version="0.1.0")
app.include_router(v1_router, prefix="/api/v1")
