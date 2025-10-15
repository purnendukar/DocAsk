from typing import List

from sentence_transformers import SentenceTransformer

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
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using the SentenceTransformer model.

        Args:
            texts: list of text strings

        Returns:
            List of embedding vectors (lists of floats)
        """
        # SentenceTransformer is sync, so wrap in asyncio thread
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_texts_sync, texts)

    # -------------------------------
    # Internal synchronous embedding
    # -------------------------------
    def _embed_texts_sync(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            embeddings.extend(batch_embeddings.tolist())

        return embeddings
