import uuid
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from app.utils.text import chunk_text
from app.services.embeddings import HFEmbeddingProvider, EmbeddingProvider
from app.services.vector_store import FaissVectorStore, VectorStore
from app.core.config import settings

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(
        self, 
        embedding_provider: Optional[EmbeddingProvider] = None, 
        vector_store: Optional[VectorStore] = None,
        vector_store_path: Optional[Path] = None
    ):
        """Initialize the ingestion service.
        
        Args:
            embedding_provider: Provider for generating embeddings
            vector_store: Vector store for storing document chunks
            vector_store_path: Path to store the vector store (defaults to settings.VECTOR_STORE_PATH)
        """
        self.embedding_provider = embedding_provider or HFEmbeddingProvider()
        
        # Initialize vector store with persistence
        if vector_store is None:
            vs_path = vector_store_path or settings.VECTOR_STORE_PATH
            vs_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing vector store at {vs_path}")
            self.vector_store = FaissVectorStore(dim=384, persist_path=str(vs_path))  # all-MiniLM-L6-v2 uses 384 dim
        else:
            self.vector_store = vector_store

    async def ingest_file(self, upload_file) -> str:
        """Process and ingest an uploaded file.
        
        Args:
            upload_file: The uploaded file to process
            
        Returns:
            str: Document ID of the ingested file
            
        Raises:
            ValueError: If the file is empty or invalid
            HTTPException: For any processing errors
        """
        from fastapi import HTTPException
        from pathlib import Path
        import tempfile
        from app.utils.file_utils import extract_text_from_file, save_uploaded_file
        
        if not upload_file.filename:
            raise ValueError("No filename provided")
            
        try:
            # Save the uploaded file to a temporary location
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / upload_file.filename
                
                # Save the uploaded file
                with open(temp_path, 'wb') as f:
                    content = await upload_file.read()
                    if not content:
                        raise ValueError("File is empty")
                    f.write(content)
                
                # Extract text based on file type
                text, error = extract_text_from_file(str(temp_path))
                if error or not text.strip():
                    error_msg = error or "No text content found in file"
                    logger.error(f"Error extracting text from {upload_file.filename}: {error_msg}")
                    raise ValueError(f"Could not extract text from file: {error_msg}")
                
                logger.info(f"Extracted {len(text)} characters from {upload_file.filename}")
            
            # Generate a unique ID for this document
            doc_id = str(uuid.uuid4())
            
            # Chunk the text
            chunks = chunk_text(text, chunk_size=1000, overlap=200)
            if not chunks:
                raise ValueError("No text content found in file")
            
            # Prepare documents for embedding
            docs = []
            for i, chunk in enumerate(chunks):
                docs.append({
                    'id': f"{doc_id}_{i}",
                    'text': chunk,
                    'metadata': {
                        'source': upload_file.filename,
                        'chunk_index': i,
                        'doc_id': doc_id,
                        'text': chunk  # Store the text in metadata for retrieval
                    }
                })
            
            logger.info(f"Generated {len(chunks)} chunks from {upload_file.filename}")
            
            # Get embeddings for all chunks
            texts = [doc['text'] for doc in docs]
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = await self.embedding_provider.embed_texts(texts)
            
            # Add to vector store
            ids = [doc['id'] for doc in docs]
            texts = [doc['text'] for doc in docs]
            metadatas = [doc['metadata'] for doc in docs]
            
            logger.info(f"Adding {len(ids)} vectors to the vector store...")
            # Pass both the text and metadata to ensure proper storage
            self.vector_store.add_vectors(ids, embeddings, metadatas, texts=texts)
            
            # Verify the vectors were added
            if hasattr(self.vector_store, 'index'):
                logger.info(f"Vector store now contains {self.vector_store.index.ntotal} vectors")
            
            logger.info(f"Successfully ingested document {upload_file.filename} with {len(chunks)} chunks")
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Error ingesting file {upload_file.filename}: {str(e)}", exc_info=True)
            from fastapi import HTTPException
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file: {str(e)}"
            )
