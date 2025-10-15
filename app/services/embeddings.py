from typing import List

class EmbeddingProvider:
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

# Example implementation using OpenAI (wrap so keys used from config)
class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, client):
        self.client = client

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # call client with batching, retry/backoff, return vectors
        # keep small batch size to avoid rate limits
        ...
