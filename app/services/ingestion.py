import uuid
from typing import List
from app.utils.text import chunk_text
from app.services.embeddings import OpenAIEmbeddingProvider
from app.services.vector_store import FaissVectorStore

class IngestionService:
    def __init__(self, embedding_provider=None, vector_store=None):
        self.embedding_provider = embedding_provider or OpenAIEmbeddingProvider(...)
        self.vector_store = vector_store or FaissVectorStore(dim=1536)

    async def ingest_file(self, upload_file) -> str:
        # save file, parse text (pdf, docx, txt) - use pdfplumber or pypandoc
        raw_text = await self._read_file(upload_file)
        chunks = chunk_text(raw_text, chunk_size=1000, overlap=200)
        vectors = self.embedding_provider.embed_texts(chunks)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadata = [{"source": upload_file.filename, "chunk_index": i} for i in range(len(chunks))]
        self.vector_store.add_vectors(ids, vectors, metadata)
        # record document metadata in DB (omitted)
        return "doc:" + str(uuid.uuid4())

    async def _read_file(self, file):
        contents = await file.read()
        # detect mime & extract text using appropriate lib
        return contents.decode('utf-8', errors='ignore')
