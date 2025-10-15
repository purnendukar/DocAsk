from typing import Tuple, List
from app.services.embeddings import OpenAIEmbeddingProvider
from app.services.vector_store import FaissVectorStore

class RAGService:
    def __init__(self, emb_provider=None, vector_store=None, llm_client=None):
        self.emb = emb_provider or OpenAIEmbeddingProvider(...)
        self.vs = vector_store or FaissVectorStore(dim=1536)
        self.llm_client = llm_client  # wrapper to LLM

    def answer_query(self, query: str, top_k: int = 5) -> Tuple[str, List[dict]]:
        q_vec = self.emb.embed_texts([query])[0]
        hits = self.vs.query(q_vec, top_k)
        contexts = [self._format_hit(h) for h in hits]
        prompt = self._build_prompt(query, contexts)
        answer = self.llm_client.generate(prompt)
        sources = [{"source": h[2]["source"], "chunk_index": h[2]["chunk_index"], "score": h[1]} for h in hits]
        return answer, sources

    def _format_hit(self, hit):
        return hit[2].get("text", "")

    def _build_prompt(self, query, contexts: List[str]) -> str:
        context_block = "\n\n".join(contexts)
        prompt = f"Use the following context to answer the question.\n\nContext:\n{context_block}\n\nQuestion: {query}\nAnswer:"
        return prompt
