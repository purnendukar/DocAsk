import logging
import asyncio
from typing import List, Optional, Any

import traceback

# Try to import required libraries with better error messages
try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        "NumPy is required but not installed. Please install it with: "
        "pip install numpy"
    ) from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers is required but not installed. "
        "Please install it with: pip install sentence-transformers"
    ) from e

logger = logging.getLogger(__name__)

class EmbeddingProvider:
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


# -------------------------------
# HuggingFace Implementation
# -------------------------------
class HFEmbeddingProvider(EmbeddingProvider):
    """
    HuggingFace Sentence-Transformers embeddings provider.
    Fully local and free.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        """
        Args:
            model_name: pre-trained SentenceTransformer model
            batch_size: number of texts per batch
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Optional[SentenceTransformer] = None
        
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model to avoid loading it during import."""
        if self._model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {str(e)}")
                raise RuntimeError(
                    f"Failed to load model {self.model_name}. "
                    "Make sure it's a valid model name from the sentence-transformers library."
                ) from e
        return self._model

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using the SentenceTransformer model.

        Args:
            texts: list of text strings

        Returns:
            List of embedding vectors (lists of floats)
            
        Raises:
            RuntimeError: If there's an error during embedding
        """
        if not texts:
            return []
            
        try:
            # Run the synchronous embedding in a thread pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,  # Use default thread pool
                self._embed_texts_sync,
                texts
            )
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e

    # -------------------------------
    # Internal synchronous embedding
    # -------------------------------
    def _embed_texts_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous method to embed texts in batches.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If there's an error during embedding
        """
        if not texts:
            return []
            
        try:
            embeddings: List[List[float]] = []
            total_texts = len(texts)
            
            logger.info(f"Starting to embed {total_texts} texts in batches of {self.batch_size}")
            
            # Process in batches
            for i in range(0, total_texts, self.batch_size):
                batch = texts[i: i + self.batch_size]
                logger.debug(f"Processing batch {i//self.batch_size + 1}/{(total_texts + self.batch_size - 1)//self.batch_size}")
                
                # Generate embeddings
                batch_embeddings = self.model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                    normalize_embeddings=True
                )
                
                # Convert to list of lists
                embeddings.extend(batch_embeddings.tolist())
            
            logger.info(f"Successfully embedded {len(embeddings)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in _embed_texts_sync: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate text embeddings: {str(e)}") from e
